"""ごみ集めの全体回を手が止まったときに回す（#455）

本物の全体回は試験の中の物まで集めて遅いので、回す関数と時計を差し替える
gc の世代ごとの数（しきい値）は本物を書き換えるので、試験の後に必ず戻す
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sashimono.gc_policy import MAX_DEFER_S, NEVER, FullCollectPolicy

SRC = Path(__file__).resolve().parent.parent / "src"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def keep_threshold() -> Iterator[None]:
    saved = gc.get_threshold()
    yield
    gc.set_threshold(*saved)


def _policy(clock: Clock, calls: list[int]) -> FullCollectPolicy:
    def collect(generation: int) -> int:
        calls.append(generation)
        return 0

    return FullCollectPolicy(collect=collect, clock=clock)


def _age_young_generations() -> None:
    """世代 1 の回を 1 度起こす 起こさないと「集める物が無い」と見て回さない"""
    gc.collect(1)


def test_manual_mode_stops_only_the_automatic_full_collection() -> None:
    """壊れると、世代 0 と 1 まで止まってメモリが増え続けるか、全体回が自動のまま止まりが減らない"""
    first, second, _third = gc.get_threshold()
    policy = _policy(Clock(), [])
    policy.set_manual(True)
    assert gc.get_threshold() == (first, second, NEVER)
    policy.set_manual(False)
    assert gc.get_threshold()[2] < NEVER


#: 循環した物を作り続け、そのあいだに自動で走った全体回の数を出す 手で回す形なら先に入れる
#: まっさらなプロセスで数える ほかの試験が gc を止めたり積もらせたりした後だと、自動の側でも
#: 全体回が走らず、比べられない
_PROBE = """
import gc, sys
from sashimono.gc_policy import FullCollectPolicy
gc.set_threshold(700, 10, 10)
policy = FullCollectPolicy()
policy.set_manual(sys.argv[1] == "manual")
started = []
gc.callbacks.append(lambda phase, info: phase == "start" and info["generation"] == 2
                    and started.append(1))
kept = []
for index in range(400_000):
    loop = [index]
    loop.append(loop)
    if index % 3 == 0:
        kept.append({"at": index})
print(len(started))
"""


def _automatic_full_collections(mode: str) -> int:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    # 固定の文字列を、今の Python へ list で渡すだけ（shell は通さない）
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, mode], env=env, capture_output=True, text=True, check=True
    )
    return int(done.stdout.strip())


def test_manual_mode_really_stops_the_automatic_full_collection() -> None:
    """壊れると、設定を入れても操作の最中に全体回が走り続ける

    世代 2 の数を 0 にした前の作りがそうだった
    """
    assert _automatic_full_collections("auto") > 0
    assert _automatic_full_collections("manual") == 0


def test_turning_manual_off_restores_the_previous_threshold() -> None:
    """壊れると、設定を切っても自動の全体回の間隔が既定と違う値のまま残る"""
    gc.set_threshold(700, 10, 7)
    policy = _policy(Clock(), [])
    policy.set_manual(True)
    policy.close()
    assert gc.get_threshold() == (700, 10, 7)


def test_a_full_collection_runs_once_the_hands_stop() -> None:
    """壊れると、手で回す設定にしたのに全体回が 1 度も走らず、循環した物が溜まり続ける"""
    clock, calls = Clock(), list[int]()
    policy = _policy(clock, calls)
    policy.set_manual(True)
    _age_young_generations()
    record = policy.collect_if_due(idle=True)
    assert record is not None and record.idle
    assert calls == [2]


def test_nothing_runs_while_working_before_the_limit() -> None:
    """壊れると、操作の最中に全体回が走り、手で回す意味が無くなる"""
    clock, calls = Clock(), list[int]()
    policy = _policy(clock, calls)
    policy.set_manual(True)
    _age_young_generations()
    clock.now = MAX_DEFER_S / 2
    assert policy.collect_if_due(idle=False) is None
    assert calls == []


def test_the_limit_forces_a_collection_even_while_working() -> None:
    """壊れると、ずっと操作し続ける人のメモリが、全体回の無いまま増え続ける"""
    clock, calls = Clock(), list[int]()
    policy = _policy(clock, calls)
    policy.set_manual(True)
    clock.now = MAX_DEFER_S + 1
    record = policy.collect_if_due(idle=False)
    assert record is not None and not record.idle
    assert calls == [2]


def test_automatic_mode_never_collects_by_hand() -> None:
    """壊れると、設定を切っている人にも手が止まるたびに全体回が走り、自動の回と 2 重になる"""
    clock, calls = Clock(), list[int]()
    policy = _policy(clock, calls)
    clock.now = MAX_DEFER_S * 3
    assert policy.collect_if_due(idle=True) is None
    assert calls == []
