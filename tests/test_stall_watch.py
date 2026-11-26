"""画面のスレッドの止まりを見張る（#442 の 0 番）

時計とスレッドの並びを差し替えて、本当に止まらずに確かめる
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

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
    thread = watch._thread
    try:
        assert watch.running and watch._on_gc in gc.callbacks
        assert thread is not None and thread.is_alive()
    finally:
        watch.stop()
    assert not watch.running and watch._on_gc not in gc.callbacks
    assert not thread.is_alive(), "止めた後も見張りのスレッドが回っている"


def test_a_stall_cut_short_by_stopping_is_still_written(tmp_path: Path) -> None:
    """壊れると、止まっている最中に見張りを止めた（閉じる処理が重かった）分の並びが捨てられる"""
    clock = Clock()
    watch = _watch(tmp_path, clock)
    watch.start()
    watch.beat()
    clock.now = 0.100
    watch.check()
    watch.stop()
    [record] = _records(watch.path)
    assert record["stall_ms"] == 100.0


def test_stopping_during_a_slow_write_keeps_the_order(tmp_path: Path) -> None:
    """壊れると、前の記録の書き込みが 1 秒を超えたとき、止めた側が後の記録を先に書き、
    記録の順が入れ替わる
    """
    clock = Clock()
    watch = _watch(tmp_path, clock)
    writing = threading.Event()
    release = threading.Event()
    written: list[tuple[float, str]] = []
    real_write = watch._write

    def slow_write(stall: Any, ended: float) -> None:
        writing.set()
        release.wait(30.0)
        written.append((ended, threading.current_thread().name))
        real_write(stall, ended)

    watch._write = slow_write  # type: ignore[method-assign]
    watch.beat()
    clock.now = 0.100
    watch.check()
    watch.beat()
    watch.start()
    thread = watch._thread
    assert thread is not None
    assert writing.wait(30.0)
    clock.now = 0.300
    watch.check()
    clock.now = 0.400
    watch.stop()
    release.set()
    thread.join(30.0)
    assert [ended for ended, _name in written] == [0.100, 0.400]
    assert {name for _ended, name in written} == {"sashimono-stall-watch"}


def test_threads_with_the_same_name_are_kept_apart(tmp_path: Path) -> None:
    """壊れると、同じ名前の係が 2 本あるとき片方の並びが上書きされ、止めた方を見落とす"""
    release = threading.Event()
    twins = [threading.Thread(target=release.wait, name="twin", daemon=True) for _ in range(2)]
    for thread in twins:
        thread.start()
    try:
        stacks = StallWatch(tmp_path / "stalls.jsonl")._stacks()
    finally:
        release.set()
        for thread in twins:
            thread.join()
    assert len([name for name in stacks if name.startswith("twin")]) == 2
    # 1 本目は名前のまま（MainThread を名前で探す読み方が変わらない）
    assert "twin" in stacks


def test_a_collection_while_copying_the_history_does_not_stop_the_watch(
    tmp_path: Path,
) -> None:
    """壊れると、記録を写す途中でごみ集めが終わったとき見張りのスレッドが落ち、以後何も記録しない"""

    class Shaky(deque[tuple[float, int, float]]):
        def __init__(self) -> None:
            super().__init__()
            self.left = 1

        def __iter__(self) -> Iterator[tuple[float, int, float]]:
            if self.left:
                self.left -= 1
                raise RuntimeError("deque mutated during iteration")
            return super().__iter__()

    clock = Clock()
    watch = _watch(tmp_path, clock)
    watch._gc = Shaky()
    watch.beat()
    clock.now = 0.100
    watch.beat()
    watch.flush()
    assert watch.recorded == 1


def test_the_environment_variable() -> None:
    """``1`` は既定の線 数はその線 空と 0 と読めない値は入っていない"""
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "1"}) == 30.0
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "50"}) == 50.0
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "0"}) is None
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "x"}) is None
    # 限りの無い値も入っていない扱い 通すと見張りのスレッドが待つ長さを求めて落ちる
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "inf"}) is None
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "1e309"}) is None
    assert threshold_from_env({"SASHIMONO_STALL_WATCH": "nan"}) is None
    assert threshold_from_env({}) is None


def test_our_files_are_written_with_their_folder() -> None:
    """壊れると、設定パネルの panel.py と会話の panel.py が同じ名前になり、原因を取り違える"""
    import sashimono
    from sashimono.stall_watch import _where

    package = Path(sashimono.__file__).resolve().parent
    assert (
        _where(str(package / "ui" / "inspector" / "panel.py")) == "sashimono/ui/inspector/panel.py"
    )
    assert _where(str(package / "ui" / "chat" / "panel.py")) == "sashimono/ui/chat/panel.py"
    assert _where(str(package / "app.py")) == "sashimono/app.py"
    # 配布版は本体の場所を相対で書く 利用者のフォルダを含まないのでそのまま残す
    assert _where(str(Path("sashimono/ui/chat/panel.py"))) == "sashimono/ui/chat/panel.py"
    # 外の部品は名前だけ 置き場まで書くと、本人の名前の入ったパスが記録に残る
    assert _where(str(Path("C:/Users/someone/python/threading.py"))) == "threading.py"
    # 本人のフォルダが sashimono という名前でも、本体の置き場の外なら名前だけ（PR #550 の指摘）
    outside = Path("C:/Users/sashimono/顧客/site-packages/numpy/core.py")
    assert _where(str(outside)) == "core.py"


def test_memory_is_written_with_each_stall(tmp_path: Path) -> None:
    """壊れると、ごみ集めを手で回したときにメモリが増え続けているかを記録から読めない"""
    from sashimono.memory_usage import MemoryUsage

    clock = Clock()
    watch = StallWatch(
        tmp_path / "stalls.jsonl",
        threshold_ms=30.0,
        clock=clock,
        frames=_frames,
        memory=lambda: MemoryUsage(resident_mb=512.04, private_mb=700.06),
    )
    watch.beat()
    clock.now = 0.100
    watch.beat()
    watch.flush()
    [record] = _records(watch.path)
    assert record["memory_mb"] == {"resident": 512.0, "private": 700.1}


def test_the_memory_of_this_process_can_be_read() -> None:
    """壊れると、記録のメモリの欄がいつも空になる（Windows の呼び方を間違えると黙って None）"""
    from sashimono.memory_usage import process_memory

    usage = process_memory()
    assert usage is not None
    assert usage.resident_mb > 1
    if sys.platform == "win32":
        assert usage.private_mb is not None and usage.private_mb > 1


def test_the_stack_reads_like_the_traceback_module(tmp_path: Path) -> None:
    """壊れると、写し方を速くしたときに並びの向きや形が変わり、``stall_report.py`` が読み違える"""
    import traceback

    from sashimono.stall_watch import STACK_DEPTH, _frame_lines, _where

    def deep(level: int) -> FrameType:
        return deep(level - 1) if level else sys._getframe()

    frame = deep(STACK_DEPTH + 5)
    expected = [
        f"{_where(entry.filename)}:{entry.lineno} {entry.name}"
        for entry in traceback.extract_stack(frame, limit=STACK_DEPTH)
    ]
    assert _frame_lines(frame) == expected
    assert len(expected) == STACK_DEPTH


def test_a_python_bound_stall_is_sampled_while_it_lasts(tmp_path: Path) -> None:
    """壊れると、画面のスレッドが Python を回し続けた止まりで並びを写せず、記録が並び無しになる

    本物の時計とスレッドで確かめる 写すのに GIL を何度も手放す書き方（ソースの行を読む）だと、
    400 ms の止まりでも 1 回も写せなかった（#455）
    """
    watch = StallWatch(tmp_path / "stalls.jsonl", threshold_ms=30.0)
    watch.start()
    try:
        watch.beat()
        began = time.perf_counter()

        def sampled() -> int:
            stall = watch._stall
            return len(stall.samples) if stall is not None else 0

        # 400 ms は回し続ける 混んだ機械で見張りの係がなかなか回ってこないときは、2 回写すまで
        # 回し続ける 400 ms で切ると、その間に係が 1 度も動かなかっただけで落ちる 上限は 10 秒
        # 写せない書き方なら上限まで回って並び無しのまま記録され、下の確かめで落ちる
        while time.perf_counter() - began < 0.4 or (
            sampled() < 2 and time.perf_counter() - began < 10
        ):
            sum(range(1000))
        watch.beat()
        deadline = time.perf_counter() + 5
        while not watch.recorded and time.perf_counter() < deadline:
            time.sleep(0.01)
    finally:
        watch.stop()
    # 後片付けの間に試験の側が遅れると、別の短い止まりも書かれうる 狙った 1 回目だけを見る
    record = _records(tmp_path / "stalls.jsonl")[0]
    assert isinstance(record["samples"], list)
    assert len(record["samples"]) >= 2
