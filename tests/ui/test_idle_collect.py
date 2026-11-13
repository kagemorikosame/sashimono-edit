"""手が止まったらごみ集めの全体回を回す画面の側（#455）

タイマーを待たずに、時間切れの関数を直に呼んで確かめる
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest

from sashimono.gc_policy import NEVER, FullCollectPolicy
from sashimono.ui.idle_collect import IdleCollect


@pytest.fixture(autouse=True)
def keep_threshold() -> Iterator[None]:
    saved = gc.get_threshold()
    yield
    gc.set_threshold(*saved)


def _collector(playing: list[bool], calls: list[int]) -> IdleCollect:
    def collect(generation: int) -> int:
        calls.append(generation)
        return 0

    return IdleCollect(lambda: playing[0], policy=FullCollectPolicy(collect=collect))


def test_nothing_runs_while_playing(qt_application: object) -> None:
    """壊れると、再生中に手を離しただけで全体回が走り、出すコマが遅れて再生がつかえる"""
    playing, calls = [True], list[int]()
    collector = _collector(playing, calls)
    try:
        collector.set_manual(True)
        gc.collect(1)
        collector._on_idle()
        assert calls == []
        assert collector._idle.isActive(), "再生が止まった後に見直す約束が無い"
        playing[0] = False
        collector._on_idle()
        assert calls == [2]
    finally:
        collector.close()


def test_input_restarts_the_wait_only_in_manual_mode(qt_application: object) -> None:
    """壊れると、設定を切っている人の操作のたびにタイマーが回り、手で回さないのに画面が働く"""
    collector = _collector([False], list[int]())
    try:
        collector.poke()
        assert not collector._idle.isActive()
        collector.set_manual(True)
        collector._idle.stop()
        collector.poke()
        assert collector._idle.isActive()
    finally:
        collector.close()


def test_closing_returns_to_automatic(qt_application: object) -> None:
    """壊れると、窓を閉じた後も自動の全体回が止まったまま、同じプロセスのほかの窓のメモリが増える"""
    collector = _collector([False], list[int]())
    collector.set_manual(True)
    assert gc.get_threshold()[2] == NEVER
    collector.close()
    assert gc.get_threshold()[2] < NEVER
