"""検査の終わりを待つ関数（:meth:`CheckJob.waiter`）が、渡した秒より長く止まらないか（#405）

AI のスレッドはこの関数を 0.2 秒ずつ呼び、中断と待つ上限を見る（ブリッジの ``_wait``）
1 回の呼び出しが秒を守らないと、中断も上限も効かないまま止まり続ける 走り係が鍵を
持ったまま重い仕事（字幕のはみ出しの測り）をしている間がそれに当たる
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.engine.checks import CheckJob, CheckLimits, CheckRequest
from sashimono.engine.checks.job import _Progress

#: 走り係が鍵を持ち続ける長さ（秒） 待つ側の秒（0.2）よりずっと長くする
_HELD = 30.0


@pytest.fixture
def held(qt_application: QApplication) -> Iterator[tuple[CheckJob, _Progress]]:
    """走っている検査の入れ物を、別のスレッドが鍵を持ったままにした係"""
    del qt_application
    job = CheckJob()
    progress = _Progress(CheckRequest(CheckLimits()), lambda: None)
    # 走り係を立てずに入れ物だけを置く 鍵を持つ長さを試験で決めるため
    job._progress = progress
    taken = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with progress.lock:
            taken.set()
            release.wait(_HELD)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert taken.wait(10.0)
    try:
        yield job, progress
    finally:
        release.set()
        holder.join(10.0)


def test_the_wait_keeps_its_seconds_while_the_worker_holds_the_lock(
    held: tuple[CheckJob, _Progress],
) -> None:
    """壊れると、走り係が鍵を持つ間は AI のスレッドが鍵を待って止まり、待つ上限（最長 2 分）も
    中断のボタンも効かない（#405 で道具が 600 秒返らなかった道の 1 つ）
    """
    job, _progress = held
    wait = job.waiter()
    started = time.monotonic()
    assert wait(0.2) is False
    # 鍵が放されるまで（30 秒）待たずに戻ったこと 上限は遅い機械でも揺れない広さにする
    assert time.monotonic() - started < _HELD / 3


def test_waiting_zero_seconds_does_not_block(held: tuple[CheckJob, _Progress]) -> None:
    """壊れると、待たずに様子だけを見る呼び出し（0 秒）でも、鍵が空くまで止まる"""
    job, _progress = held
    started = time.monotonic()
    assert job.waiter()(0.0) is False
    assert time.monotonic() - started < _HELD / 3


def test_a_finished_check_is_still_seen(qt_application: QApplication) -> None:
    """壊れると、鍵を短く区切ったせいで、終わった検査を「まだ」と読み、待つ上限まで待つ"""
    del qt_application
    job = CheckJob()
    progress = _Progress(CheckRequest(CheckLimits()), lambda: None)
    job._progress = progress
    wait = job.waiter()
    with progress.lock:
        progress.running = False
        progress.lock.notify_all()
    started = time.monotonic()
    assert wait(30.0) is True
    assert time.monotonic() - started < 10.0
