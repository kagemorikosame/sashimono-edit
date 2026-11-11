"""画面のスレッドから止まりの見張りを打つ部品（#442 の 0 番）"""

from __future__ import annotations

import gc
from pathlib import Path

import shiboken6
from PySide6.QtWidgets import QApplication

from sashimono.ui.stall_monitor import MIN_THRESHOLD_MS, StallMonitor


def test_a_destroyed_monitor_stops_the_watch(qt_application: QApplication, tmp_path: Path) -> None:
    """壊れると、閉じるのを断った窓が壊されたとき、見張りのスレッドとごみ集めの知らせが残る"""
    del qt_application
    monitor = StallMonitor(path=tmp_path / "stalls.jsonl")
    monitor.set_threshold(30.0)
    watch = monitor.watch
    assert watch is not None and watch.running
    shiboken6.delete(monitor)
    assert not watch.running
    assert watch._on_gc not in gc.callbacks


def test_the_threshold_stays_above_the_beat(qt_application: QApplication, tmp_path: Path) -> None:
    """壊れると、短い線（環境変数で 5 など）で、普段の 10ms ごとの打ちまで止まりとして
    記録され続ける
    """
    del qt_application
    monitor = StallMonitor(path=tmp_path / "stalls.jsonl")
    try:
        monitor.set_threshold(5.0)
        assert monitor.watch is not None
        assert monitor.watch.threshold_ms == MIN_THRESHOLD_MS
    finally:
        monitor.stop()
