"""落ちた・拾われなかったエラーの記録（#516）

利用者の機械でだけ起きる不具合は、起きた時の様子が残っていないと調べられない
診断情報（:mod:`sashimono.diagnostic_report`）に載せるため、次の 2 つを本人の置き場へ書き残す

- 拾われなかった Python のエラー（``sys.excepthook`` と ``threading.excepthook``）
  画面のボタンから呼んだ所で上がったエラーは Qt が握りつぶして画面は動き続けるので、
  何が起きたかが本人にも分からない ``errors.log`` に時刻と呼び出しの並びを足していく
- 落ちた起動（``faulthandler``） 起動ごとに ``sessions/<起動の印>.log`` を開いたまま持ち、
  落ちたときの Python の並びをそこへ書かせる 正しく終わった起動は自分の記録を消す
  残っていて持ち主のいない記録が「落ちた起動」 Windows の ``faulthandler`` は落ちない
  例外（駆動の中で拾われる物）でも書くことがあるので、正しく終わった起動の記録は捨てる

Qt を使わない 書けない置き場（容量が尽きた・読み取り専用）でも起動と編集は止めない
診断のために本人の作業を落とすのは本末転倒（:mod:`sashimono.stall_watch` と同じ考え方）
送らない 書くのは本人の機械の中だけ 送るかどうかは診断情報の窓で本人が決める
"""

from __future__ import annotations

import atexit
import contextlib
import faulthandler
import os
import re
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO

from sashimono import __version__
from sashimono.core.io.locks import HeldLock, is_held, try_hold
from sashimono.core.userdirs import state_root

__all__ = [
    "ERRORS_NAME",
    "MAX_LOG_BYTES",
    "FaultLog",
    "LostSession",
    "diagnostics_folder",
    "install",
    "installed",
]

#: 拾われなかったエラーの記録の名前
ERRORS_NAME = "errors.log"
#: 起動ごとの記録を置くフォルダの名前
SESSIONS = "sessions"
#: 記録がこの大きさを超えたら 1 世代前へ回す 入れたまま何年使っても置き場を食い続けない
MAX_LOG_BYTES = 1024 * 1024
#: 残しておく落ちた起動の数 古い物から消す 毎回落ちる機械でも溜まり続けない
KEEP_LOST = 5
#: エラーの記録の 1 件の頭 ``=== 2026-11-24T12:00:00`` の前で切る
#: 起動ごとの記録の頭の行（:meth:`FaultLog.install` が書く形）
_SESSION_HEAD = re.compile(
    r"起動 (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) 版 (\d+\.\d+[0-9A-Za-z.+\-]*)"
)
_ENTRY_HEAD = re.compile(r"^(?==== \d{4}-\d{2}-\d{2}T)", re.MULTILINE)


def diagnostics_folder() -> Path:
    """診断の記録の置き場 画面の止まりの記録（``stall_watch``）と同じ所"""
    return state_root() / "diagnostics"


@dataclass(frozen=True, slots=True)
class LostSession:
    """正しく終わらなかった起動 1 つ"""

    #: 起動した時刻（記録の頭の行から読む 読めなければファイルの時刻）
    started: str
    #: 起動した版
    version: str
    #: ``faulthandler`` が書いた中身 空なら Python の外で終わった（強制終了・電源断など）
    text: str


def _rotate(path: Path) -> None:
    """大きくなった記録を 1 世代前へ回す 回せなくても上げない

    2 つ目の窓が同じ記録を開いている間は、Windows では名前を変えられない 上げると、
    回せなかっただけで記録を書くこと（:meth:`FaultLog.record_error`）や記録の係を掛けること
    （:meth:`FaultLog.install`）まで止まる 回すのは次の機会でよい
    """
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))


class FaultLog:
    """記録の係 :meth:`install` で掛け、:meth:`uninstall`（終わるとき）で外す

    ``folder`` は試験で差し替える ``catch_faults`` が偽なら ``faulthandler`` を掛けない
    （試験の中で掛けると、pytest が自分で掛けた物を外してしまう）
    """

    def __init__(self, folder: Path | None = None, *, catch_faults: bool = True) -> None:
        self.folder = folder if folder is not None else diagnostics_folder()
        self._catch_faults = catch_faults
        self.errors_path = self.folder / ERRORS_NAME
        self.sessions = self.folder / SESSIONS
        #: この起動の記録 掛けていなければ ``None``
        self.session_path: Path | None = None
        self._stream: IO[bytes] | None = None
        self._lock: HeldLock | None = None
        self._previous_hook: Callable[..., object] | None = None
        self._previous_thread_hook: Callable[[threading.ExceptHookArgs], object] | None = None
        self._write_lock = threading.Lock()

    # --- 掛ける・外す ---

    def install(self) -> bool:
        """記録を掛ける 置き場に書けなければ何も掛けずに偽"""
        started = datetime.now()
        mark = f"{started:%Y%m%d-%H%M%S}-{os.getpid()}"
        path = self.sessions / f"{mark}.log"
        stream: IO[bytes] | None = None
        try:
            self.sessions.mkdir(parents=True, exist_ok=True)
            _rotate(self.errors_path)
            # 古い記録を片付けられなくても掛ける 片付けは次の起動でもできる
            with contextlib.suppress(OSError):
                self._forget_old()
            # 持ち主がいることを錠で示す 中身（プロセス番号）では見分けない 番号は使い回される
            self._lock = try_hold(self.sessions / f"{mark}.lock")
            # 新しく作る（"x"） 同じ名前が先にあれば作らずに偽を返す 下の片付けで、自分が
            # 作っていない記録を消さないため
            stream = path.open("xb")
            stream.write(
                f"起動 {started.isoformat(timespec='seconds')} 版 {__version__}\n".encode()
            )
            stream.flush()
        except OSError:
            # 作りかけの記録を片付けてから錠を手放す 残すと、正しく終わったこの起動が次の診断情報で
            # 落ちた起動に数えられる（PR #550 の指摘） 錠を先に手放すと、ほかの窓がその間に
            # 持ち主のいない記録と見て読む
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
                with contextlib.suppress(OSError):
                    path.unlink()
            self._release_lock()
            return False
        self._stream = stream
        self.session_path = path
        # 全部のスレッドの並びを書く 落ちたのが裏の係（先読み・書き出し）のことが多い
        # 掛けられなくても（記録の先が使えない など）エラーの記録は掛ける
        if self._catch_faults:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                faulthandler.enable(file=stream, all_threads=True)
        self._previous_hook = sys.excepthook
        sys.excepthook = self._on_error
        self._previous_thread_hook = threading.excepthook
        threading.excepthook = self._on_thread_error
        atexit.register(self.uninstall)
        return True

    def uninstall(self) -> None:
        """外す 正しく終わったので、この起動の記録を消す"""
        if self._stream is None:
            return
        if sys.excepthook == self._on_error and self._previous_hook is not None:
            sys.excepthook = self._previous_hook
        if threading.excepthook == self._on_thread_error and self._previous_thread_hook:
            threading.excepthook = self._previous_thread_hook
        if self._catch_faults:
            with contextlib.suppress(RuntimeError):
                faulthandler.disable()
        stream, self._stream = self._stream, None
        with contextlib.suppress(OSError):
            stream.close()
        if self.session_path is not None:
            with contextlib.suppress(OSError):
                self.session_path.unlink()
        self._release_lock()
        with contextlib.suppress(ValueError):
            atexit.unregister(self.uninstall)

    def _release_lock(self) -> None:
        lock, self._lock = self._lock, None
        # 手放せなくても上げない 終わるとき（atexit）に上げると、終わり際にエラーの並びが出る
        # 錠はプロセスが終われば外れる
        if lock is not None:
            with contextlib.suppress(OSError):
                lock.release()

    def _forget_old(self) -> None:
        """落ちた起動の記録を、新しい :data:`KEEP_LOST` 件だけ残す"""
        lost = sorted(
            (path for path in self.sessions.glob("*.log") if not self._alive(path)),
            key=lambda path: path.name,
        )
        for path in lost[:-KEEP_LOST] if len(lost) > KEEP_LOST else []:
            with contextlib.suppress(OSError):
                path.unlink()

    def _alive(self, log: Path) -> bool:
        """その記録の起動がまだ動いているか（同じ機械で 2 つ目の窓を開いている間など）"""
        return is_held(log.with_suffix(".lock"))

    # --- 拾われなかったエラー ---

    def _on_error(
        self,
        kind: type[BaseException],
        error: BaseException,
        trace: TracebackType | None,
    ) -> None:
        # 書くことに何があっても、前の受け手へは必ず渡す（PR #550 の指摘） 渡さないと、
        # 開発の環境でコマンドから起こした人にエラーが見えなくなる
        try:
            self.record_error(kind, error, trace)
        finally:
            if self._previous_hook is not None:
                self._previous_hook(kind, error, trace)

    def _on_thread_error(self, args: threading.ExceptHookArgs) -> None:
        try:
            if args.exc_value is not None:
                name = args.thread.name if args.thread is not None else "?"
                self.record_error(args.exc_type, args.exc_value, args.exc_traceback, thread=name)
        finally:
            if self._previous_thread_hook is not None:
                self._previous_thread_hook(args)

    def record_error(
        self,
        kind: type[BaseException],
        error: BaseException,
        trace: TracebackType | None,
        *,
        thread: str = "",
    ) -> None:
        """エラーを 1 件足す

        ここで上がったエラーは握る（置き場に書けない・文字にできない・例外の文を作れない など）
        エラーを書く所で落ちると元のエラーが消える 対になっていない文字（サロゲート）は
        ``\\udcxx`` の形で書く UTF-8 にできずに書くこと自体が落ちる（PR #550 の指摘）
        """
        try:
            head = f"=== {datetime.now().isoformat(timespec='seconds')} 版 {__version__}"
            if thread:
                head += f" スレッド {thread}"
            body = "".join(traceback.format_exception(kind, error, trace))
            with self._write_lock:
                self.folder.mkdir(parents=True, exist_ok=True)
                _rotate(self.errors_path)
                with self.errors_path.open(
                    "a", encoding="utf-8", errors="backslashreplace"
                ) as stream:
                    stream.write(f"{head}\n{body}\n")
        except Exception:  # 記録の失敗で元のエラーを消さない どの失敗でも握る
            return

    # --- 読む（診断情報から） ---

    def recent_errors(self, count: int = 5, *, unread: list[Path] | None = None) -> list[str]:
        """新しいエラーから ``count`` 件（古い順に並べ直して返す）

        無いファイルは黙る（1 世代前は無いのが普通） 在るのに読めなかった物は ``unread`` へ足す
        黙ると、診断情報に「なし」と出て、エラーが無かったのか読めなかったのかが分からない
        """
        texts: list[str] = []
        for path in (self.errors_path.with_suffix(".log.1"), self.errors_path):
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except FileNotFoundError:
                continue
            except OSError:
                if unread is not None:
                    unread.append(path)
        # 頭の行（行の頭の ``=== 時刻``）で切る 並びの中の文に ``===`` が混ざっても切らない
        entries = [entry.strip() for entry in _ENTRY_HEAD.split("\n".join(texts)) if entry.strip()]
        return entries[-count:] if count > 0 else []

    def lost_sessions(self, *, unread: list[Path] | None = None) -> list[LostSession]:
        """正しく終わらなかった起動 古い順 動いている起動（この起動を含む）は外す

        読めなかった記録は ``unread`` へ足す（:meth:`recent_errors` と同じ訳）
        """
        if not self.sessions.is_dir():
            return []
        found: list[LostSession] = []
        for path in sorted(self.sessions.glob("*.log"), key=lambda path: path.name):
            if path == self.session_path:
                continue
            try:
                # 動いているかを確かめられない記録（錠を見られない）は読めなかった物に数える
                # 上げると、ほかの落ちた起動の記録まで集められない
                if self._alive(path):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            except OSError:
                if unread is not None:
                    unread.append(path)
                continue
            head, _, rest = text.partition("\n")
            # 頭の行は決まった形のときだけ読む 形が違えば（壊れた・手で書き換えた）中身を写さない
            shape = _SESSION_HEAD.fullmatch(head.strip())
            started = shape.group(1) if shape else path.stem
            version = shape.group(2) if shape else "不明"
            found.append(LostSession(started=started, version=version, text=rest.strip()))
        return found


#: いま掛けている係 起動のときに 1 つだけ作る（診断情報の窓が同じ物を読む）
_installed: FaultLog | None = None


def installed() -> FaultLog:
    """起動で掛けた係 掛けていなければ（試験・自己診断）同じ置き場を読むだけの係"""
    return _installed if _installed is not None else FaultLog()


def install(folder: Path | None = None) -> FaultLog:
    """記録を掛けて係を返す 起動の頭（:func:`sashimono.app._start_editor`）が 1 度呼ぶ"""
    global _installed
    log = FaultLog(folder)
    if log.install():
        _installed = log
    return log
