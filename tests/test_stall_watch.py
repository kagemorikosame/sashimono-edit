"""画面のスレッドの止まりを見張る（#442 の 0 番）

時計とスレッドの並びを差し替えて、本当に止まらずに確かめる
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import FrameType

from sashimono.stall_watch import StallWatch, threshold_from_env


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _frames() -> dict[int, FrameType]:
    frame = sys._getframe()
    return {threading.get_ident() + 1: frame}


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _watch(tmp_path: Path, clock: Clock) -> StallWatch:
    return StallWatch(tmp_path / "stalls.jsonl", threshold_ms=30.0, clock=clock, frames=_frames)


def test_a_stall_is_written_with_the_stacks_taken_during_it(tmp_path: Path) -> None:
    """壊れると、カクついた瞬間に何が画面を止めていたかが記録に残らない"""
    clock = Clock()
    watch = _watch(tmp_path, clock)
    watch.beat()
    clock.now = 0.040
    watch.check()
    clock.now = 0.100
    watch.beat()
    watch.flush()
    [record] = _records(watch.path)
    assert record["stall_ms"] == 100.0
    samples = record["samples"]
    assert isinstance(samples, list) and len(samples) == 1
    stacks = samples[0]["threads"]
    assert any("_frames" in line for lines in stacks.values() for line in lines)


def test_short_gaps_are_not_written(tmp_path: Path) -> None:
    """壊れると、ふつうの 1 コマごとの揺れまで書いて記録が埋もれる"""
    clock = Clock()
    watch = _watch(tmp_path, clock)
    for step in range(10):
        clock.now = step * 0.010
        watch.beat()
        watch.check()
    watch.flush()
    assert not watch.path.exists()
    assert watch.recorded == 0


def test_a_stall_noticed_only_when_resuming_is_still_written(tmp_path: Path) -> None:
    """見張りが写す前に動き出した止まりも、長さだけは残す"""
    clock = Clock()
    watch = _watch(tmp_path, clock)
    watch.beat()
    clock.now = 0.050
    watch.beat()
    watch.flush()
    [record] = _records(watch.path)
    assert record["stall_ms"] == 50.0
    assert record["samples"] == []


def test_samples_are_limited_while_stalled(tmp_path: Path) -> None:
    """壊れると、長い止まりの間ずっと写し続け、見張りが止まりを伸ばす"""
    clock = Clock()
    watch = _watch(tmp_path, clock)
    watch.beat()
    for step in range(1, 40):
        clock.now = step * 0.010
        watch.check()
    watch.beat()
    watch.flush()
    [record] = _records(watch.path)
    samples = record["samples"]
    assert isinstance(samples, list) and len(samples) == 3


def test_garbage_collection_during_the_stall_is_recorded(tmp_path: Path) -> None:
    """壊れると、ごみ集めが止めていたのかどうかを記録から見分けられない"""
    clock = Clock()
    watch = _watch(tmp_path, clock)
    watch.beat()
    clock.now = 0.010
    watch._on_gc("start", {"generation": 2})
    clock.now = 0.046
    watch._on_gc("stop", {"generation": 2})
    clock.now = 0.060
    watch.beat()
    watch.flush()
    [record] = _records(watch.path)
    assert record["gc"] == [{"generation": 2, "ms": 36.0}]


def test_the_thread_starts_and_stops(tmp_path: Path) -> None:
    """壊れると、設定で切っても見張りのスレッドとごみ集めの知らせが残る"""
    import gc

    watch = StallWatch(tmp_path / "stalls.jsonl")
    watch.start()
    assert watch.running and watch._on_gc in gc.callbacks
    watch.stop()
    assert not watch.running and watch._on_gc not in gc.callbacks


def test_the_environment_variable() -> None:
    """``1`` は既定の線 数はその線 空と 0 と読めない値は入っていない"""
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "1"}) == 30.0
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "50"}) == 50.0
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "0"}) is None
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "x"}) is None
    assert threshold_from_env({}) is None
