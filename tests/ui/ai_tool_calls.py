"""本物の窓へ、AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道）

試験のファイルごとに同じ形を持っていたのを 1 か所にまとめた 待ちの上限を 1 つ直せば
全部に効くように
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import allow_sending

__all__ = ["TOOL_TIMEOUT", "call_tool", "dump_threads", "wait_for_check"]

#: 道具が返るのを待つ上限（秒） 返ればすぐ戻るので、ふつうの速さは変わらない
#: 本物の上限（``bridge.CALL_TIMEOUT`` 60 秒）より長くとる CI で pytest を 4〜6 本並べると、
#: 画面のスレッドで GL のコンテキストを作る・手放す所が、ほかの試験の描画と GPU を取り合って
#: 数十秒止まることがあり、本物の上限のままだと検査の結果に関係なく「編集画面が応答しません」
#: で落ちた（#386 の verify で tests/ui/test_check_output.py の 1 本） 試験が見るのは結果で、
#: 速さではない 止まったまま返らない作りの崩れは、この長い上限でも落ちて分かる
TOOL_TIMEOUT = 600.0


def call_tool(window: MainWindow, name: str, arguments: dict[str, Any]) -> Any:
    """道具を呼び、結果を返す 変更は自動で承認し、外へ送る物は尋ねずに許す

    送る物の確かめ（#342）は tests/ui/test_send_consent.py が見る 画面のスレッドは、
    道具が返るまで仕事を拾い続ける
    """
    operation = find_operation(name)
    assert operation is not None
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
    deadline = time.monotonic() + TOOL_TIMEOUT
    while thread.is_alive():
        bridge.pump()
        QApplication.processEvents()
        thread.join(0.01)
        if time.monotonic() > deadline:
            # 畳む前に全部のスレッドの居場所を残す 畳んだ後だと、どこで止まっていたかが消える
            # #405 は 600 秒止まったのに、記録からはどのスレッドが何を待っていたか分からなかった
            dump_threads()
            bridge.cancel()
            thread.join(5.0)
            pytest.fail("道具が返らなかった")
    if "error" in box:
        raise box["error"]
    return box["result"]


def wait_for_check(window: MainWindow, then: Callable[[], bool] | None = None) -> None:
    """窓の裏の検査（音量の測りや仕上がりの検査）が終わるまで、画面の仕事を拾いながら待つ

    道具に ``wait_seconds`` を渡して待たせると、その秒で測り終えない遅い機械では、
    道具が「測っている途中」を返して試験が落ちる（#405） ここは秒で区切らず、検査の
    終わりの知らせ（:meth:`MainWindow.check_waiter`）を待つ 速い機械では終わればすぐ戻る
    :data:`TOOL_TIMEOUT` は止まったまま終わらない作りの崩れを見つけるためだけの上限

    ``then`` は、測り終えた後に画面のスレッドが済ませる仕事（右クリックからの音量の
    掛け算）が済んだか 走り係は終わったと記した後に知らせ（``changed``）を出すので、
    終わったと分かった時にはまだ知らせが画面へ届いていないことがある 済むまで拾い続ける
    """
    waiter = window.check_waiter()
    deadline = time.monotonic() + TOOL_TIMEOUT
    # 待つのは 10 ms ずつ 待つ間も画面の仕事を拾わないと、検査の進みの知らせが溜まり続ける
    while not waiter(0.01):
        QApplication.processEvents()
        if time.monotonic() > deadline:
            dump_threads()
            pytest.fail("検査が終わらなかった")
    QApplication.processEvents()
    while then is not None and not then():
        QApplication.processEvents()
        time.sleep(0.001)
        if time.monotonic() > deadline:
            dump_threads()
            pytest.fail("測り終えた後の画面の仕事が済まなかった")


def dump_threads() -> None:
    """全部のスレッドの今の呼び出しの並びを標準エラーへ書く（pytest が落ちた試験に添える）

    ``faulthandler`` は使わない pytest の取り込み方によっては標準エラーがファイルの
    番号を持たず、書けずに例外で落ちる Qt のスレッド（検査の走り係）も、Python を
    動かしている間はここに出る
    """
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        print(f"--- スレッド {names.get(ident, ident)}", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
