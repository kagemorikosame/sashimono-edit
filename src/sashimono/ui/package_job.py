"""持ち出し用パッケージの書き出しと展開を、裏で走らせて見守る（#423）

素材ごとまとめるので数十 GB になりうる 画面を固めないよう別のスレッドで回し、進み具合は
ステータスバーと AI の両方へ出す 画面の窓と AI の道具が同じ 1 本を使う（同時に 2 本は
走らせない ディスクを取り合って両方が遅くなり、空きの見積もりも外れる）
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from sashimono.ai.host import ToolError
from sashimono.engine.package.archive import Cancelled, PackageError

__all__ = ["PACKAGE_RUNNING", "PackageJob", "Work"]

#: 走っている様子
PACKAGE_RUNNING = "working"

#: 裏で走らせる仕事 進み具合を知らせる関数と、止めるよう頼まれたかを返す関数を受け取る
Work = Callable[[Callable[[float, str], None], Callable[[], bool]], object]


class _Worker(QObject):
    progressed = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, work: Work) -> None:
        super().__init__()
        self._work = work
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = self._work(self.progressed.emit, self._cancel.is_set)
        except Cancelled:
            self.cancelled.emit()
            return
        except (PackageError, ToolError, OSError) as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            # 素材の中の思わぬ形で落ちても、窓ごと落とさず人と AI へ伝える
            self.failed.emit(f"パッケージの処理に失敗しました: {exc}")
            return
        self.finished.emit(result)


class PackageJob(QObject):
    """書き出しか展開の 1 本 様子は :meth:`status` で読む"""

    #: 様子が変わった（進んだ・終わった） 編集画面のスレッドで出る
    changed = Signal()
    #: 終わった（``kind``・結果 失敗と中止は ``None``） 編集画面のスレッドで 1 度だけ出る
    ended = Signal(str, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._state = "idle"
        self._kind = ""
        self._by_assistant = False
        self._progress = 0.0
        self._text = ""
        self._path: Path | None = None
        self._message = ""
        self._started = 0.0
        self._ended = 0.0
        self.result: object = None
        #: 走っていない間は立っている AI のスレッドが終わりを待つのに使う
        self._idle = threading.Event()
        self._idle.set()

    @property
    def running(self) -> bool:
        return self._state == PACKAGE_RUNNING

    @property
    def kind(self) -> str:
        """``export``（書き出し）か ``open``（展開） 走らせる前は空"""
        return self._kind

    @property
    def by_assistant(self) -> bool:
        """AI の道具が始めたか 終わりの知らせを窓で出すか、AI へ返すかを分ける"""
        return self._by_assistant

    def wait(self, seconds: float) -> bool:
        """終わるまで最長 ``seconds`` 秒待つ 走っていなければ真 どのスレッドから呼んでもよい"""
        return self._idle.wait(seconds)

    def start(self, kind: str, path: Path, work: Work, *, by_assistant: bool = False) -> None:
        if self.running:
            raise ToolError("持ち出し用パッケージの書き出しか展開がもう走っています")
        self._state = PACKAGE_RUNNING
        self._idle.clear()
        self._kind = kind
        self._by_assistant = by_assistant
        self._progress = 0.0
        self._text = ""
        self._path = path
        self._message = ""
        self._started = time.monotonic()
        self._ended = 0.0
        self.result = None
        self._worker = _Worker(work)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.start()
        self.changed.emit()

    def cancel(self) -> bool:
        """止めるよう頼む 書いていた物は消え、前のファイルはそのまま残る"""
        if self._worker is None or not self.running:
            return False
        self._worker.cancel()
        self._message = "止めています"
        self.changed.emit()
        return True

    def shutdown(self) -> None:
        """窓を閉じるとき 止めるよう頼み、スレッドが抜けるまで待つ"""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._idle.set()

    def progress_text(self) -> tuple[str, float]:
        verb = "持ち出し用に書き出し中" if self._kind == "export" else "パッケージを展開中"
        return f"{verb}: {self._text}" if self._text else verb, self._progress

    def remembered_names(self) -> tuple[str, ...]:
        """書き出し先・開いたパッケージの場所（#342 伏せ字の表に入れる）"""
        return (str(self._path),) if self._path is not None else ()

    def status(self) -> dict[str, object]:
        """AI へ返す様子"""
        ended = self._ended or time.monotonic()
        return {
            "kind": self._kind or None,
            "state": self._state,
            "progress": round(self._progress, 3),
            "path": str(self._path) if self._path is not None else None,
            "message": self._message,
            "elapsed_seconds": round(ended - self._started, 1) if self._started else 0.0,
        }

    def note(self, message: str) -> None:
        """終わった後の知らせを差し替える（開いた後に足りない書体を添えるなど）"""
        self._message = message
        self.changed.emit()

    def _on_progress(self, value: float, text: str) -> None:
        self._progress = value
        self._text = text
        self.changed.emit()

    def _on_finished(self, result: object) -> None:
        self.result = result
        self._finish("done", "終わりました", result)

    def _on_failed(self, message: str) -> None:
        self._finish("failed", message, None)

    def _on_cancelled(self) -> None:
        self._finish("cancelled", "止めました 書き出し先と開いている作品は前のままです", None)

    def _finish(self, state: str, message: str, result: object) -> None:
        self._state = state
        self._message = message
        self._ended = time.monotonic()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        # 終わりの扱い（開いた作品へ差し替えるなど）を済ませてから待っている AI を起こす
        # 先に起こすと、AI が差し替える前の作品を読む
        self.ended.emit(self._kind, result)
        self._idle.set()
        self.changed.emit()
