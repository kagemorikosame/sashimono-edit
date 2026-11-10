"""画面のスレッドから止まりの見張り（:class:`~sashimono.stall_watch.StallWatch`）を打つ

打つのは 10ms ごとの精密なタイマー Qt の粗いタイマーは 5% までずらして束ねるので、
30ms の線に対して揺れが大きすぎる 打つ間隔そのものは止まりに数えない（見張りは
最後に打たれた時刻からの長さで見る）
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer

from sashimono.stall_watch import DEFAULT_THRESHOLD_MS, StallWatch

__all__ = ["BEAT_MS", "MIN_THRESHOLD_MS", "StallMonitor"]

#: 何ミリ秒ごとに打つか 線（既定 30ms）より十分に短くする
BEAT_MS = 10

#: 線の下限 打つ間隔の 2 倍より短い線では、止まっていない普段の打ちの揺れまで
#: 止まりとして記録され、並びの写しと書き込みが本物の止まりを埋もれさせる
MIN_THRESHOLD_MS = BEAT_MS * 2


class _Held:
    """見張りを持つ所 部品が壊されたときにも止められるよう、部品の外に置く

    ``destroyed`` に部品自身のメソッドをつなぐと、知らせが来る頃には Python の側が
    もう使えないことがある Qt のタイマーには触らない（部品と一緒に消えている）
    """

    def __init__(self) -> None:
        self.watch: StallWatch | None = None

    def stop(self, *_args: object) -> None:
        watch, self.watch = self.watch, None
        if watch is not None:
            watch.stop()


class StallMonitor(QObject):
    """入れている間だけ見張りを動かす 切ればタイマーも見張りのスレッドも止める"""

    def __init__(self, parent: QObject | None = None, *, path: Path | None = None) -> None:
        super().__init__(parent)
        self._path = path
        self._held = _Held()
        # 閉じるのを断った窓が deleteLater で壊されると、stop を通らない 見張りのスレッドと
        # ごみ集めの知らせが残り、壊れた部品の分まで記録を書き続ける
        self.destroyed.connect(self._held.stop)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(BEAT_MS)
        self._timer.timeout.connect(self._beat)

    @property
    def watch(self) -> StallWatch | None:
        return self._held.watch

    def set_threshold(self, threshold_ms: float | None) -> None:
        """``None`` で止める 数なら、その線で見張る（線が変われば立て直す）"""
        if threshold_ms is None:
            self.stop()
            return
        threshold_ms = max(threshold_ms, MIN_THRESHOLD_MS)
        current = self._held.watch
        if current is not None and current.threshold_ms == threshold_ms:
            return
        self.stop()
        watch = self._held.watch = StallWatch(self._path, threshold_ms=threshold_ms)
        watch.start()
        watch.beat()
        self._timer.start()

    def start(self) -> None:
        self.set_threshold(DEFAULT_THRESHOLD_MS)

    def stop(self) -> None:
        self._timer.stop()
        self._held.stop()

    def _beat(self) -> None:
        watch = self._held.watch
        if watch is not None:
            watch.beat()
