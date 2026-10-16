"""AI が外（Anthropic）へ送る前に、送ってよいかを尋ねる窓（#342 D4）

何がどこへ送られるかを、送る前に言葉で出す 答えは 3 つ

- 送る 次からは尋ねない（設定の「AI へ送る物」に覚える）
- アプリを閉じるまで送る（設定には書かない 次に起動したらまた尋ねる）
- 送らない（アプリを閉じるまで尋ねない 設定で変えればまた尋ねる）

既定のボタンは「送らない」にする Enter を押しただけで絵が外へ出ると、読まずに
押した人が取り返せない
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from sashimono.ai.outbound import SendAnswer, SendRequest

__all__ = ["SEND_ALWAYS_TEXT", "SEND_DENY_TEXT", "SEND_ONCE_TEXT", "ask_send"]

SEND_ALWAYS_TEXT = "送る（次からは尋ねない）"
SEND_ONCE_TEXT = "アプリを閉じるまで送る"
SEND_DENY_TEXT = "送らない"


def ask_send(parent: QWidget | None, request: SendRequest) -> SendAnswer | None:
    """窓を出して答えを返す 閉じるボタン（✕・Esc）で閉じたら ``None``（送らない 覚えない）"""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("AI へ送る物の確かめ")
    box.setText("AI アシスタントへ、次の物を送ってよいですか？")
    box.setInformativeText(request.text())
    always = box.addButton(SEND_ALWAYS_TEXT, QMessageBox.ButtonRole.AcceptRole)
    once = box.addButton(SEND_ONCE_TEXT, QMessageBox.ButtonRole.YesRole)
    deny = box.addButton(SEND_DENY_TEXT, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(deny)
    box.setEscapeButton(deny)
    box.exec()
    clicked = box.clickedButton()
    box.deleteLater()
    if clicked is always:
        return SendAnswer.ALWAYS
    if clicked is once:
        return SendAnswer.UNTIL_CLOSE
    if clicked is deny:
        return SendAnswer.DENY
    return None
