"""書き出しを別のプロセスで走らせる入口（#453 #443 の段階 2）

:func:`~sashimono.engine.worker.run_in_process` が子の中で呼ぶ 子の中で GL のコンテキストを
作るので、子で GL を作れない機械では :class:`ExportGLUnavailableError` の文で親へ返す
親はそれを見てスレッドの書き出しへ戻す（:func:`is_gl_unavailable`）

書き出し側からの知らせ（次の候補のエンコーダで書き直した #365）は、知らせの関数を管で
渡せないので、終わったときに結果と一緒に返す
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
#: ``tools/bench_contention.py --onscreen --prefetch``（2026-11-09 1080p60 60 秒 RTX 5060 Ti）
#: 単体ではスレッド 1.7 倍速・別のプロセス 2.3 倍速 Python を回し続けるスレッドが重なると、
#: スレッドの書き出しは 0.1 倍速まで落ちた 子の起動（部品を読む分を含めて 1 秒前後）は
#: 書き出し 1 本に 1 回なので、長さで分けずにいつも寄せる 設定の画面がここから文言を作る
MEASURED_EXPORT_SPEED = (1.0, 1.8)

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


def export_job(
    *,
    progress: Callable[[float], None],
    should_cancel: Callable[[], bool],
    project: Project,
    settings: ExportSettings,
) -> ExportOutcome | None:
    """子のプロセスで書き出す やめたときは ``None``

    書き出しの失敗（:class:`ExportError`）は文のまま親へ渡り、親が同じ文で利用者に出す
    やめたときも書き出しは :class:`ExportError` を投げる（書きかけを消してから）ので、
    ここで ``None`` へ直す 直さないと、取り消しが失敗として画面に出る
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
    except ExportError:
        if should_cancel():
            return None
        raise
    return ExportOutcome(path=str(path), notes=tuple(notes))


def is_gl_unavailable(reason: str) -> bool:
    """子の失敗の文が「子で GL を作れない」か"""
    return GL_UNAVAILABLE_MARK in reason


def _discard(path: Path) -> None:
    """子が後片付けをせずに終わったときの書きかけを消す

    スレッドで書く道（:func:`export_project`）は失敗すると書き出し先を消す 落ちた子・猶予を
    過ぎて止めた子はそれをしないので、親が代わりに消す 消さないと、窓の書き出しは利用者が
    選んだ場所へ直に書くので、途中で切れた動画がそこに残る ``run_in_process`` は子が終わる
    まで待ってから戻るので、ここではもう子がファイルを掴んでいない 消せなくても元の失敗を
    隠さない
    """
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


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
        try:
            outcome = run_in_process(
                export_job,
                {"project": project, "settings": settings},
                progress=progress,
                should_cancel=should_cancel,
                name="sashimono-export",
            )
        except WorkerUnavailableError as exc:
            _log.warning("書き出しを別のプロセスで走らせられないので、スレッドで書く: %s", exc)
        except WorkerFailedError as exc:
            reason = str(exc)
            if not is_gl_unavailable(reason):
                raise ExportError(reason.removeprefix(_EXPORT_ERROR_PREFIX)) from exc
            _log.warning("子のプロセスで GL を作れないので、スレッドで書き出す: %s", reason)
        except WorkerCrashedError as exc:
            _discard(settings.path)
            raise ExportError(f"書き出しの途中で別のプロセスが落ちた: {exc}") from exc
        else:
            if outcome is None:
                # 取り消しの猶予を過ぎて子を止めたときは、子の中の後片付けが走っていない
                _discard(settings.path)
                # 取り消した 同じプロセスで書いたときと同じ文で返す（呼ぶ側は中止の印で見分ける）
                raise ExportError("書き出しを中止した")
            if notice is not None:
                for note in outcome.notes:
                    notice(note)
            return Path(outcome.path)
    return export_project(
        project, settings, progress=progress, should_cancel=should_cancel, notice=notice
    )
