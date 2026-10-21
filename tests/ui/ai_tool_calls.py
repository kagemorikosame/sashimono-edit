"""本物の窓へ、AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道）

試験のファイルごとに同じ形を持っていたのを 1 か所にまとめた 待ちの上限を 1 つ直せば
全部に効くように
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import allow_sending

__all__ = ["TOOL_TIMEOUT", "call_tool"]

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
            bridge.cancel()
            thread.join(5.0)
            pytest.fail("道具が返らなかった")
    if "error" in box:
        raise box["error"]
    return box["result"]
