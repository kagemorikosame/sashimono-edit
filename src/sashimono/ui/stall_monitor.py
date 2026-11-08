"""画面のスレッドから止まりの見張り（:class:`~sashimono.stall_watch.StallWatch`）を打つ

打つのは 10ms ごとの精密なタイマー Qt の粗いタイマーは 5% までずらして束ねるので、
30ms の線に対して揺れが大きすぎる 打つ間隔そのものは止まりに数えない（見張りは
最後に打たれた時刻からの長さで見る）
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer

from sashimono.stall_watch import DEFAULT_THRESHOLD_MS, StallWatch

__all__ = ["BEAT_MS", "StallMonitor"]

#: 何ミリ秒ごとに打つか 線（既定 30ms）より十分に短くする
BEAT_MS = 10


class StallMonitor(QObject):
    """入れている間だけ見張りを動かす 切ればタイマーも見張りのスレッドも止める"""

    def __init__(self, parent: QObject | None = None, *, path: Path | None = None) -> None:
        super().__init__(parent)
        self._path = path
        self._watch: StallWatch | None = None
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(BEAT_MS)
        self._timer.timeout.connect(self._beat)

    @property
    def watch(self) -> StallWatch | None:
        return self._watch

    def set_threshold(self, threshold_ms: float | None) -> None:
        """``None`` で止める 数なら、その線で見張る（線が変われば立て直す）"""
        if threshold_ms is None:
            self.stop()
            return
        if self._watch is not None and self._watch.threshold_ms == threshold_ms:
            return
        self.stop()
        self._watch = StallWatch(self._path, threshold_ms=threshold_ms)
        self._watch.start()
        self._watch.beat()
        self._timer.start()

    def start(self) -> None:
        self.set_threshold(DEFAULT_THRESHOLD_MS)

    def stop(self) -> None:
        self._timer.stop()
        watch, self._watch = self._watch, None
        if watch is not None:
            watch.stop()

    def _beat(self) -> None:
        if self._watch is not None:
            self._watch.beat()
