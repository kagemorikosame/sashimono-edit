"""本物の窓へ、AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道）

試験のファイルごとに同じ形を持っていたのを 1 か所にまとめた 待ちの上限を 1 つ直せば
全部に効くように 待ち方の決まりは tests/waiting.py
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import allow_sending
from tests.waiting import GUARD_SECONDS, wait_until

__all__ = [
    "TOOL_TIMEOUT",
    "call_after_check",
    "call_tool",
    "wait_for_check",
    "wait_for_export",
]

#: 道具が返るのを待つ上限（秒） 返ればすぐ戻るので、ふつうの速さは変わらない
#: 本物の上限（``bridge.CALL_TIMEOUT`` 60 秒）より長くとる CI で pytest を 4〜6 本並べると、
#: 画面のスレッドで GL のコンテキストを作る・手放す所が、ほかの試験の描画と GPU を取り合って
#: 数十秒止まることがあり、本物の上限のままだと検査の結果に関係なく「編集画面が応答しません」
#: で落ちた（#386 の verify で tests/ui/test_check_output.py の 1 本） 試験が見るのは結果で、
#: 速さではない 止まったまま返らない作りの崩れは、この長い上限でも落ちて分かる
TOOL_TIMEOUT = GUARD_SECONDS


def call_tool(
    window: MainWindow,
    name: str,
    arguments: dict[str, Any],
    *,
    bridge: EditorBridge | None = None,
    allow_irreversible: bool = False,
) -> Any:
    """道具を呼び、結果を返す 変更は自動で承認し、外へ送る物は尋ねずに許す

    送る物の確かめ（#342）は tests/ui/test_send_consent.py が見る 画面のスレッドは、
    道具が返るまで仕事を拾い続ける ``bridge`` を渡すと、そのブリッジで呼ぶ（指示の段を
    続けて見る試験） ``allow_irreversible`` は取り消せない操作の確かめに許すで答える
    """
    operation = find_operation(name)
    assert operation is not None
    if bridge is None:
        bridge = allow_sending(EditorBridge(window, auto_approve=True))
    bridge.call_timeout = TOOL_TIMEOUT
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    def step() -> None:
        bridge.pump()
        if allow_irreversible:
            approval = bridge.take_approval()
            if approval is not None:
                approval.allow()
        QApplication.processEvents()
        thread.join(0.01)

    try:
        wait_until(lambda: not thread.is_alive(), "道具", step=step, pause=0.0)
    except pytest.fail.Exception:
        # 畳まないと、落ちた後も AI のスレッドが残りの待ちを続け、次の試験の窓へ手を出す
        bridge.cancel()
        thread.join(5.0)
        raise
    if "error" in box:
        raise box["error"]
    return box["result"]


def call_after_check(
    window: MainWindow,
    name: str,
    arguments: dict[str, Any],
    finished: Callable[[Any], bool] | None,
    *,
    bridge: EditorBridge | None = None,
) -> Any:
    """検査で測る道具（check_output・normalize_loudness・analyze_audio など）を、
    窓の検査が終わった後の答えで返す ``finished`` は答えが終わった物かを見る

    道具に秒を渡して道具の中で待たせない 渡した秒で測り終えない遅い機械では、道具が
    途中を返して落ちる（#405） 1 回目は待たずに測り始めさせ、検査の終わりの知らせを
    待ってから同じ引数で呼び直す 測り終えた頼みは使い回されるので、2 回目は待たずに
    答えが出る 1 回目のうちに測り終えていれば、それが答え（呼び直すと、そろえた後の
    作品を測り直す）
    直しの回の数え（#339）は、使えなかった検査（途中の答え）では動かないので、呼び直しても
    数えは変わらない（:func:`sashimono.ai.fixes.start_check`）
    ``finished`` を省くのは、読むだけで答えから終わったかを見分けにくい道具（絵を返す
    audio_graph） いつも終わりを待って呼び直す 読むだけなので呼び直しても作品は変わらない
    """
    quick = {**arguments, "wait_seconds": 0}
    try:
        first = call_tool(window, name, quick, bridge=bridge)
    except ToolError:
        # 音の道具は、読み始めたばかりで答えが無いと途中の答えの代わりに断る（もう 1 度
        # 呼ぶように言う） 終わりを待って呼び直す 本当の断り（場所が違うなど）は、
        # 呼び直しても同じ断りになって、そのまま投げられる
        first = None
    if first is not None and finished is not None and finished(first):
        return first
    wait_for_check(window)
    again = call_tool(window, name, quick, bridge=bridge)
    if finished is not None:
        assert finished(again), again
    return again


def wait_for_check(window: MainWindow, then: Callable[[], bool] | None = None) -> None:
    """窓の裏の検査（音量の測りや仕上がりの検査）が終わるまで、画面の仕事を拾いながら待つ

    秒で区切らず、検査の終わりの知らせ（:meth:`MainWindow.check_waiter`）を待つ
    速い機械では終わればすぐ戻る

    ``then`` は、測り終えた後に画面のスレッドが済ませる仕事（右クリックからの音量の
    掛け算）が済んだか 走り係は終わったと記した後に知らせ（``changed``）を出すので、
    終わったと分かった時にはまだ知らせが画面へ届いていないことがある 済むまで拾い続ける
    """
    _wait_for(window.check_waiter(), "検査", then)


def wait_for_export(window: MainWindow) -> None:
    """AI の書き出しが終わるまで、画面の仕事を拾いながら待つ

    書き出しは取り消せない操作なので、途中を返した道具を呼び直せない（呼び直すと
    もう 1 度確かめて書き出す） 終わりの知らせ（:meth:`MainWindow.export_waiter`）を待ち、
    結果は :meth:`MainWindow.export_status`（道具が待った後に返すのと同じ物）で読む
    """
    _wait_for(window.export_waiter(), "書き出し", None)


def _wait_for(waiter: Callable[[float], bool], what: str, then: Callable[[], bool] | None) -> None:
    # 待つのは 10 ms ずつ 待つ間も画面の仕事を拾わないと、進みの知らせが溜まり続け、
    # 書き出しのように画面のスレッドで終わりを記す仕事は終わらない
    wait_until(lambda: waiter(0.01), what, step=QApplication.processEvents, pause=0.0)
    QApplication.processEvents()
    if then is not None:
        wait_until(then, f"{what}の後の画面の仕事", step=QApplication.processEvents)
