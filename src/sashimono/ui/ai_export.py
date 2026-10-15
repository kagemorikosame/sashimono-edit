"""AI アシスタントが始めた書き出しを、裏で走らせて見守る（B1 #327）

書き出しの処理そのものは画面の書き出しの窓と同じ（:func:`~sashimono.engine.encode.export_project`
を同じ作り方の設定で呼ぶ） 違うのは書く場所の守りだけ

- 書き出しはまずアプリ自身の置き場（キャッシュ）の仮のファイルへ書く 書き出し先のフォルダで
  直に書くと、書いている数分の間にフォルダや名前を差し替えられたとき、確かめていない所へ
  書くか、元の素材へのハードリンクの中身を上書きする
- 書き終えたら、書き出し先のフォルダを開いたまま持ち（:func:`~sashimono.ai.safety.pinned`）、
  書ける場所かを確かめ直してから、新しく作った仮のファイルへ写して名前を付け替える
  （:func:`~sashimono.core.io.atomic.copy_atomic`） 付け替えた後に、その名前の先が自分の書いた
  物かを照らす

画面を固めない 書き出しは別のスレッドで回し、進み具合はステータスバーと AI の両方へ出す
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from sashimono.ai.host import EXPORT_RUNNING, ToolError
from sashimono.ai.safety import pinned
from sashimono.core.io.atomic import Expected, copy_atomic
from sashimono.core.model import Project
from sashimono.core.userdirs import cache_root
from sashimono.engine.encode import ExportError, ExportSettings
from sashimono.engine.encode import export_project as run_export

__all__ = ["AssistantExport", "scratch_folder"]


def scratch_folder() -> Path:
    """書き出しの仮のファイルを置く所 アプリ自身のキャッシュの中

    書き出し先のフォルダではなくここへ書く理由はモジュールの説明のとおり キャッシュの
    置き場なので、落ちて残った物は「キャッシュを消す」で一緒に片付く
    """
    return cache_root() / "assistant-export"


class _Worker(QObject):
    """別のスレッドで、仮のファイルへ書き出し、書き出し先へ置く"""

    progressed = Signal(float)
    placing = Signal()
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        project: Project,
        settings: ExportSettings,
        check: Callable[[], Path],
        expect: Expected | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._settings = settings
        self._check = check
        self._expect = expect
        self._cancel = threading.Event()
        #: 書き出し先へ置き始めたか 中止の印との確かめと立てるのを、同じ錠の中で行う
        #: 画面のスレッドの様子（placing）は知らせが届いてから変わるので、それで決めると
        #: 置き始めた後の中止を受けてしまい、止めると答えたのに前のファイルが置き換わる
        self._placing = False
        self._lock = threading.Lock()

    def cancel(self) -> bool:
        """止めるよう頼む 置き始めた後なら断って偽を返す（置き換えは途中で止めない）"""
        with self._lock:
            if self._placing:
                return False
            self._cancel.set()
            return True

    def _begin_placing(self) -> bool:
        """置き始める 先に中止を頼まれていれば偽（置かない）"""
        with self._lock:
            if self._cancel.is_set():
                return False
            self._placing = True
            return True

    def run(self) -> None:
        target = self._settings.path
        folder = scratch_folder()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            scratch = folder / f"{uuid.uuid4().hex}{target.suffix.lower()}"
            # 自分で新しく作ってから書き出しに渡す 名前が先にあれば、その中身へは書かない
            scratch.open("xb").close()
        except OSError as exc:
            self.failed.emit(f"書き出しの仮のファイルを作れませんでした: {exc}")
            return
        try:
            self._export(scratch)
        finally:
            with contextlib.suppress(OSError):
                scratch.unlink(missing_ok=True)

    def _export(self, scratch: Path) -> None:
        try:
            run_export(
                self._project,
                replace(self._settings, path=scratch),
                progress=self.progressed.emit,
                should_cancel=self._cancel.is_set,
            )
        except ExportError as exc:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"書き出しに失敗しました: {exc}")
            return
        if not self._begin_placing():
            self.cancelled.emit()
            return
        self.placing.emit()
        try:
            placed = self._place(scratch)
        except (ToolError, OSError) as exc:
            self.failed.emit(f"書き出し先へ置けませんでした（書き出し先は前のままです）: {exc}")
            return
        note = ""
        if self._expect is not None and self._expect.kept:
            # 置き換える前の書き出しは消さずに残した（:mod:`sashimono.core.io.atomic`） 場所を伝える
            note = f"（置き換える前のファイルは {self._expect.kept[-1]} として残しました）"
        self.finished.emit(f"{placed}{note}")

    def _place(self, scratch: Path) -> Path:
        """書き出し先のフォルダを持ったまま確かめ直し、写して名前を付け替える"""
        expected = self._settings.path
        with pinned(self._check) as path:
            if os.path.normcase(str(path)) != os.path.normcase(str(expected)):
                raise ToolError(f"書き出し先が確かめた時と変わりました: {expected} → {path}")
            # 許したときの書き出し先のままかを、名前を付ける所で照らす 描く数分の間に名前が
            # 現れたり中身が変わったりしていれば、置かずに断る（前のファイルは残る）
            written = copy_atomic(scratch, path, durable=True, expect=self._expect)
            # 付け替えた名前の先が、自分の書いた物かを照らす 違えば、付け替えの直前に
            # 差し替えられている 書いたと伝えると、AI も人も別の物を完成品だと思う
            now = path.stat(follow_symlinks=False)
            if (now.st_dev, now.st_ino) != (written.st_dev, written.st_ino):
                raise ToolError(f"置いた後に書き出し先が別の物に変わっていました: {path}")
        return path


class AssistantExport(QObject):
    """AI が始めた書き出し 1 本ずつ 様子は :meth:`status` で読む"""

    #: 様子が変わった（進んだ・終わった） 編集画面のスレッドで出る
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._state = "idle"
        self._progress = 0.0
        self._path: Path | None = None
        self._message = ""
        self._started = 0.0
        self._ended = 0.0
        #: 走っていない間は立っている AI のスレッドが終わりを待つのに使う（:meth:`wait`）
        #: 様子（``_state``）そのものは画面のスレッドだけが読み書きする
        self._idle = threading.Event()
        self._idle.set()

    @property
    def running(self) -> bool:
        return self._state in EXPORT_RUNNING

    def wait(self, seconds: float) -> bool:
        """終わるまで最長 ``seconds`` 秒待つ 走っていなければ真 どのスレッドから呼んでもよい"""
        return self._idle.wait(seconds)

    def start(
        self,
        project: Project,
        settings: ExportSettings,
        check: Callable[[], Path],
        expect: Expected | None = None,
    ) -> None:
        """書き出しを始める ``check`` は書き出し先を確かめ直す関数（終わったときに呼ぶ）"""
        if self.running:
            raise ToolError("書き出しがもう走っています")
        self._state = "rendering"
        self._idle.clear()
        self._progress = 0.0
        self._path = settings.path
        self._message = ""
        self._started = time.monotonic()
        self._ended = 0.0
        self._worker = _Worker(project, settings, check, expect)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._on_progress)
        self._worker.placing.connect(self._on_placing)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.start()
        self.changed.emit()

    def cancel(self) -> bool:
        """止めるよう頼む 書き出し先へ置き始めた後は止めない（置き終わるのを待つ）

        置き始めたかはワーカーに聞く 画面の様子（``_state``）は知らせが届くまで rendering の
        ままなので、それだけで決めると、置き換えが進んでいるのに「止めています」と答える
        """
        if self._worker is None or self._state != "rendering":
            return False
        if not self._worker.cancel():
            return False
        self._message = "止めています"
        self.changed.emit()
        return True

    def shutdown(self) -> None:
        """窓を閉じるとき 止めるよう頼み、スレッドが抜けるまで待つ

        待たずに窓を壊すと、走っているスレッドを持ったまま Qt が落ちる
        """
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        # 終わりの知らせはもう届かない 待っている AI を上限まで待たせない
        self._idle.set()

    def status(self) -> dict[str, object]:
        """AI へ返す様子"""
        ended = self._ended or time.monotonic()
        return {
            "state": self._state,
            "progress": round(self._progress, 3),
            "path": str(self._path) if self._path is not None else None,
            "message": self._message,
            "elapsed_seconds": round(ended - self._started, 1) if self._started else 0.0,
        }

    def _on_progress(self, value: float) -> None:
        self._progress = value
        self.changed.emit()

    def _on_placing(self) -> None:
        self._state = "placing"
        self._progress = 1.0
        self._message = "書き出し先へ置いています"
        self.changed.emit()

    def _on_finished(self, path: str) -> None:
        self._finish("done", f"書き出しました: {path}")

    def _on_failed(self, message: str) -> None:
        self._finish("failed", message)

    def _on_cancelled(self) -> None:
        self._finish("cancelled", "書き出しを止めました 書き出し先は前のままです")

    def _finish(self, state: str, message: str) -> None:
        self._state = state
        self._message = message
        self._ended = time.monotonic()
        if self._thread is not None:
            self._thread.quit()
            # 知らせはワーカーが書き終えて出した後 スレッドはイベントループを抜けるだけ
            self._thread.wait()
            self._thread = None
        self._worker = None
        # 様子を書き終えてから立てる 先に立てると、待っていた AI が終わる前の様子を読む
        self._idle.set()
        self.changed.emit()
