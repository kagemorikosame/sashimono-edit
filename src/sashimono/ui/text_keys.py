"""文字の入力欄に打っている間は、欄が使うキーを窓のショートカットに取らせない

窓のショートカット（メニューの項目）は、入力欄にフォーカスがあっても動く Qt の入力欄は
自分が使うキー（文字・Ctrl+V などの標準のキー・Delete・矢印）だけを ShortcutOverride で
受け取り、そのキーでは窓のショートカットが動かない 2026-10-07 に確かめた結果、
Ctrl+V・Ctrl+C・Delete・S・スペースはこれで守られていた

Ctrl+Shift+V は Qt の標準のキーに無いので守られず、文字を打っている途中に押すと
挿入貼り付け（Issue #257）が動いてタイムラインへクリップが貼られた ほかのソフトでは
書式なしの貼り付けに使うキーで、欄の中での貼り付けを期待して押す人がいる
ここでは同じ ShortcutOverride の仕組みで、入力欄がこのキーを受け取るようにし、
欄の中に書式なしで貼る

Ctrl+T（テキストを追加）のように、欄が使わないキーの窓のショートカットはそのまま動かす
欄の中では何も起きないキーなので、取り上げても打ち手の得にならない
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

from sashimono.ui import system_clipboard

__all__ = [
    "PLAIN_PASTE",
    "TextFieldKeys",
    "install_text_field_keys",
    "is_text_field",
    "paste_plain",
]

#: 書式なしの貼り付けのキー 入力欄の中ではこのキーを窓のショートカットより先に欄へ渡す
PLAIN_PASTE = QKeySequence("Ctrl+Shift+V")

#: 入れた見張りを QApplication に覚えさせる名前 窓を何枚開いても 1 つだけ入れる
_INSTALLED = "_sashimono_text_field_keys"


def _line_edit(widget: QWidget) -> QLineEdit | None:
    """数の欄（スピンボックス）の中の文字の欄 フォーカスは数の欄そのものに来る"""
    if isinstance(widget, QLineEdit):
        return widget
    if isinstance(widget, QAbstractSpinBox):
        return widget.findChild(QLineEdit)
    return None


def is_text_field(widget: QObject | None) -> bool:
    """文字を打つ欄か 読むだけの欄は打てないので入れない（窓のショートカットを止めない）"""
    if isinstance(widget, (QPlainTextEdit, QTextEdit)):
        return not widget.isReadOnly()
    if isinstance(widget, QLineEdit):
        return not widget.isReadOnly()
    if isinstance(widget, QAbstractSpinBox):
        return not widget.isReadOnly()
    return False


def paste_plain(widget: QWidget) -> None:
    """欄の中へ書式なしで貼る 文字だけを取り出して、打ったのと同じように入れる

    クリップボードは :mod:`~sashimono.ui.system_clipboard` から読む 欄の ``paste`` は
    OS のクリップボードを直に読み、試験で差し替えられない
    """
    text = system_clipboard.clipboard().text()
    if not text:
        return
    if isinstance(widget, (QTextEdit, QPlainTextEdit)):
        widget.insertPlainText(text)
        return
    line = _line_edit(widget)
    if line is not None:
        line.insert(text)


class TextFieldKeys(QObject):
    """入力欄に来た ShortcutOverride を見て、欄が使うキーなら欄に取らせる見張り"""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        if not isinstance(event, QKeyEvent) or not is_text_field(watched):
            return False
        if QKeySequence(event.keyCombination()) != PLAIN_PASTE:
            return False
        if kind is QEvent.Type.ShortcutOverride:
            # 受け取ったと印を付けると、Qt は窓のショートカットを探さずにキーを欄へ送る
            event.accept()
            return True
        assert isinstance(watched, QWidget)
        paste_plain(watched)
        return True


def install_text_field_keys(app: QApplication) -> TextFieldKeys:
    """アプリ全体へ見張りを入れる 何度呼んでも 1 つだけ（窓を開くたびに呼ばれる）

    窓ではなくアプリへ入れるのは、設定画面のような別の窓の欄も同じに扱うため
    """
    existing = app.property(_INSTALLED)
    if isinstance(existing, TextFieldKeys):
        return existing
    keys = TextFieldKeys(app)
    app.installEventFilter(keys)
    app.setProperty(_INSTALLED, keys)
    return keys
