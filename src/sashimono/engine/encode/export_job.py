"""書き出しを別のプロセスで走らせる入口（#453 #443 の段階 2）

:func:`~sashimono.engine.worker.run_in_process` が子の中で呼ぶ 子の中で GL のコンテキストを
作るので、子で GL を作れない機械では :class:`ExportGLUnavailableError` の文で親へ返す
親はそれを見てスレッドの書き出しへ戻す（:func:`is_gl_unavailable`）

書き出し側からの知らせ（次の候補のエンコーダで書き直した #365）は、知らせの関数を管で
渡せないので、終わったときに結果と一緒に返す
互換性の記録（AviUtl スクリプトの未対応 API や失敗）も同じで、子の ``global_report`` に
溜まった物は子が終わると消える 失敗・取り消しのときも含めて写しを返し、親が自分の記録へ
足し込む
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sashimono.compat.aviutl.report import NO_RECORDS, ReportRecords, global_report
from sashimono.core.model import Project
from sashimono.engine.encode.exporter import ExportError, ExportSettings, export_project
from sashimono.engine.worker import (
    WorkerCrashedError,
    WorkerFailedError,
    WorkerUnavailableError,
    run_in_process,
)

__all__ = [
    "GL_UNAVAILABLE_MARK",
    "MEASURED_EXPORT_SPEED",
    "ExportOutcome",
    "export_anywhere",
    "export_job",
    "is_gl_unavailable",
]

_log = logging.getLogger(__name__)

#: 先読みを回しながら画面を操作している所で書き出した速さ（倍速）
#: アプリの中のスレッド / 別のプロセス
#: ``tools/bench_contention.py --onscreen --prefetch``（2026-11-11 1080p60 60 秒 RTX 5060 Ti
#: 先読みは 3 枚をそれぞれ別の素材として読む） 単体ではスレッド 2.5 倍速・別のプロセス 2.7 倍速
#: Python を回し続けるスレッドが重なると、
#: スレッドの書き出しは 0.1 倍速まで落ちた 子の起動（部品を読む分を含めて 1 秒前後）は
#: 書き出し 1 本に 1 回なので、長さで分けずにいつも寄せる 設定の画面がここから文言を作る
MEASURED_EXPORT_SPEED = (1.1, 2.2)

#: 子の失敗の文の頭に付く例外の名前（:mod:`sashimono.engine.worker` が ``型: 文`` で渡す）
#: 書き出しの失敗は利用者にそのまま見せる文なので、型の名前は外す
_EXPORT_ERROR_PREFIX = f"{ExportError.__name__}: "

#: 子で GL を作れなかったときの文の頭 親はこれで、スレッドへ戻してよい失敗かを見分ける
#: 例外の型は管を越えない（子の例外は文で渡す :mod:`sashimono.engine.worker`）ので、文で印を付ける
GL_UNAVAILABLE_MARK = "子のプロセスで GL を作れない"


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """子の書き出しの結果"""

    #: 書き上げたファイル（文字）
    path: str
    #: 書き出しの間に出た知らせ 親で画面へ出す
    notes: tuple[str, ...]
    #: 子で溜まった互換性の記録の写し 親の ``global_report`` へ足し込む
    records: ReportRecords = NO_RECORDS
    #: 書き出しの失敗の文 失敗を例外で投げると管の向こうで記録の写しを返せないので、
    #: 結果として返す（例外のときの ``型: 文`` と同じ形 :class:`ExportError` だけは文のまま）
    error: str | None = None
    #: 取り消した
    cancelled: bool = False


def export_job(
    *,
    progress: Callable[[float], None],
    should_cancel: Callable[[], bool],
    project: Project,
    settings: ExportSettings,
) -> ExportOutcome | None:
    """子のプロセスで書き出す

    失敗と取り消しも例外にせず結果で返す（:attr:`ExportOutcome.error` と
    :attr:`ExportOutcome.cancelled`） 例外で返すと互換性の記録の写しを一緒に渡せない
    書き出しの失敗（:class:`ExportError`）は文のまま親へ渡り、親が同じ文で利用者に出す
    やめたときも書き出しは :class:`ExportError` を投げる（書きかけを消してから）ので、
    ここで取り消しへ直す 直さないと、取り消しが失敗として画面に出る
    """
    # GL を作れるかを先に確かめる 書き出しの途中で作れないと分かると、どこまで書いたかの
    # 後片付けまで子の中で済ませてから親が頭からやり直すことになり、時間が倍かかる
    from sashimono.engine.gpu import GLContextError, OffscreenGLContext

    try:
        OffscreenGLContext().release()
    except GLContextError as exc:
        raise RuntimeError(f"{GL_UNAVAILABLE_MARK}: {exc}") from exc
    notes: list[str] = []
    try:
        path = export_project(
            project,
            settings,
            progress=progress,
            should_cancel=should_cancel,
            notice=notes.append,
        )
    except Exception as exc:
        records = global_report.records()
        if isinstance(exc, ExportError):
            if should_cancel():
                return ExportOutcome(path="", notes=tuple(notes), records=records, cancelled=True)
            reason = str(exc)
        else:
            reason = f"{type(exc).__name__}: {exc}"
        return ExportOutcome(path="", notes=tuple(notes), records=records, error=reason)
    return ExportOutcome(path=str(path), notes=tuple(notes), records=global_report.records())


def is_gl_unavailable(reason: str) -> bool:
    """子の失敗の文が「子で GL を作れない」か"""
    return GL_UNAVAILABLE_MARK in reason


def _discard(path: Path) -> None:
    """子の仮のファイルを消す 消せなくても元の失敗を隠さない"""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _scratch_for(target: Path) -> Path:
    """子が書く仮のファイル 書き出し先と同じフォルダに、同じ拡張子で置く

    子は書き出し先へ直には書かない 子が落ちた・取り消しの猶予を過ぎて止めたときは子の中の
    後片付けが走らないので、直に書くと途中で切れた動画が利用者の選んだ場所に残る 親が
    書き出し先を消せば、まだ書き始めてもいないのに前からあったファイルまで消える 仮の
    ファイルに書かせて、終わったときだけ置き換える
    同じフォルダに置くのは、置き換えをドライブを跨がない名前の付け替えで済ませるため
    拡張子を残すのは、入れ物の形とチャプターを埋めるかを拡張子で決めているため
    """
    return target.with_name(f".{target.stem}.{uuid.uuid4().hex[:8]}.writing{target.suffix}")


def export_anywhere(
    project: Project,
    settings: ExportSettings,
    *,
    separate_process: bool,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    notice: Callable[[str], None] | None = None,
) -> Path:
    """書き出す ``separate_process`` なら別のプロセスで（:func:`export_project` と同じ約束）

    子を起こせない・子で GL を作れない機械では、このプロセス（呼んだスレッド）で書き出し直す
    子が書き出しの途中で落ちたときは戻さない（素材のせいで FFmpeg が落ちたのなら、今度は
    アプリごと落ちる）
    """
    if separate_process:
        target = settings.path
        scratch = _scratch_for(target)
        try:
            outcome = run_in_process(
                export_job,
                {"project": project, "settings": dataclasses.replace(settings, path=scratch)},
                progress=progress,
                should_cancel=should_cancel,
                name="sashimono-export",
            )
        except WorkerUnavailableError as exc:
            _discard(scratch)
            _log.warning("書き出しを別のプロセスで走らせられないので、スレッドで書く: %s", exc)
        except WorkerFailedError as exc:
            _discard(scratch)
            reason = str(exc)
            if not is_gl_unavailable(reason):
                raise ExportError(reason.removeprefix(_EXPORT_ERROR_PREFIX)) from exc
            _log.warning("子のプロセスで GL を作れないので、スレッドで書き出す: %s", reason)
        except WorkerCrashedError as exc:
            _discard(scratch)
            raise ExportError(f"書き出しの途中で別のプロセスが落ちた: {exc}") from exc
        except BaseException:
            _discard(scratch)
            raise
        else:
            if outcome is not None:
                # 成否より先に足し込む 失敗した書き出しこそ、どの効果で躓いたかを見たい
                global_report.merge(outcome.records)
            if outcome is None or outcome.cancelled:
                _discard(scratch)
                # 取り消した 同じプロセスで書いたときと同じ文で返す（呼ぶ側は中止の印で見分ける）
                # None は猶予を過ぎて子を止めたとき（記録の写しは返ってこない）
                raise ExportError("書き出しを中止した")
            if outcome.error is not None:
                _discard(scratch)
                raise ExportError(outcome.error)
            try:
                # run_in_process は子が終わるまで待ってから戻るので、子はもうファイルを掴んでいない
                scratch.replace(target)
            except OSError as exc:
                _discard(scratch)
                raise ExportError(
                    f"書き出したファイルを置き換えられない: {target} ({exc})"
                ) from exc
            if notice is not None:
                for note in outcome.notes:
                    notice(note)
            return target
    return export_project(
        project, settings, progress=progress, should_cancel=should_cancel, notice=notice
    )
