"""検査の終わりを待つ関数（:meth:`CheckJob.waiter`）が、渡した秒より長く止まらないか（#405）

AI のスレッドはこの関数を 0.2 秒ずつ呼び、中断と待つ上限を見る（ブリッジの ``_wait``）
1 回の呼び出しが秒を守らないと、中断も上限も効かないまま止まり続ける 走り係が鍵を
持ったまま重い仕事（字幕のはみ出しの測り）をしている間がそれに当たる

かかった秒では見ない 遅い機械では、戻るのが速くても試験の側が止まって秒が伸びる
別のスレッドが鍵を持ち続けている間に戻ったか（順番）で見る
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import NamedTuple

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.engine.checks import CheckJob, CheckLimits, CheckRequest
from sashimono.engine.checks.job import _Progress
from tests.waiting import GUARD_SECONDS


class _Held(NamedTuple):
    job: CheckJob
    #: 鍵を持っているスレッド 試験が放すまで（片付けまで）持ち続ける
    holder: threading.Thread


@pytest.fixture
def held(qt_application: QApplication) -> Iterator[_Held]:
    """走っている検査の入れ物を、別のスレッドが鍵を持ったままにした係"""
    del qt_application
    job = CheckJob()
    progress = _Progress(CheckRequest(CheckLimits()), lambda: None)
    # 走り係を立てずに入れ物だけを置く 鍵を放す時を試験で決めるため
    job._progress = progress
    taken = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with progress.lock:
            taken.set()
            release.wait(GUARD_SECONDS)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert taken.wait(GUARD_SECONDS)
    try:
        yield _Held(job, holder)
    finally:
        release.set()
        holder.join(GUARD_SECONDS)


def _returned(call: Callable[[], bool]) -> bool:
    """別のスレッドで呼び、戻った値を返す 戻らなければ落とす（鍵が放されるのは片付けの時）"""
    box: list[bool] = []
    waiting = threading.Thread(target=lambda: box.append(call()), daemon=True)
    waiting.start()
    waiting.join(GUARD_SECONDS)
    assert not waiting.is_alive(), "鍵を持たれている間、待つ関数が戻らなかった"
    return box[0]


def test_the_wait_keeps_its_seconds_while_the_worker_holds_the_lock(held: _Held) -> None:
    """壊れると、走り係が鍵を持つ間は AI のスレッドが鍵を待って止まり、待つ上限（最長 2 分）も
    中断のボタンも効かない（#405 で道具が 600 秒返らなかった道の 1 つ）
    """
    wait = held.job.waiter()
    assert _returned(lambda: wait(0.2)) is False
    # 戻ったのは鍵が放される前（鍵を持つスレッドはまだ持っている）
    assert held.holder.is_alive()


def test_waiting_zero_seconds_does_not_block(held: _Held) -> None:
    """壊れると、待たずに様子だけを見る呼び出し（0 秒）でも、鍵が空くまで止まる"""
    wait = held.job.waiter()
    assert _returned(lambda: wait(0.0)) is False
    assert held.holder.is_alive()


def test_a_finished_check_is_still_seen(qt_application: QApplication) -> None:
    """壊れると、鍵を短く区切ったせいで、終わった検査を「まだ」と読み、待つ上限まで待つ

    0 秒で呼んでも終わったと読めること 待つ秒が無くても、鍵が空いていれば様子は見られる
    """
    del qt_application
    job = CheckJob()
    progress = _Progress(CheckRequest(CheckLimits()), lambda: None)
    job._progress = progress
    wait = job.waiter()
    with progress.lock:
        progress.running = False
        progress.lock.notify_all()
    assert wait(0.0) is True
