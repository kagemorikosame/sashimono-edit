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

挿入貼り付けをショートカットの設定で別のキーへ変えた人は、そのキーでも同じことが起きる
窓が今の割り当てを :meth:`TextFieldKeys.hold` で渡し、入力欄ではそのキーも欄が受け取る
欄に貼るのは Ctrl+Shift+V だけで、変えた先のキーは欄の既定の動きに任せる（多くは何も
しない） 変えた先は人によって何でもあり得て、そこへ貼り付けを勝手に足すと、欄で別の
意味を持つキーの動きまで変わる

Ctrl+T（テキストを追加）のように、欄が使わないキーの窓のショートカットはそのまま動かす
クリップを足すだけで、打っていた文字を壊さない 押せば何が起きるかが欄の外と同じ方が分かりやすい
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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: 窓ごとの、入力欄では欄に取らせるキー（挿入貼り付けの今の割り当て）
        #: 鍵は窓の ``id`` 窓を何枚開いても見張りは 1 つなので、窓ごとに分けて持つ
        self._held: dict[int, tuple[QKeySequence, ...]] = {}

    def hold(self, owner: QObject, keys: list[QKeySequence]) -> None:
        """``owner`` の窓で、入力欄に打っている間は ``keys`` も欄が受け取るようにする

        割り当てが変わるたびに呼び直す（前の分は捨てる） 窓が消えれば外す
        """
        key = id(owner)
        if key not in self._held:
            owner.destroyed.connect(lambda _owner=None, key=key: self._held.pop(key, None))
        self._held[key] = tuple(sequence for sequence in keys if not sequence.isEmpty())

    def holds(self, pressed: QKeySequence) -> bool:
        """入力欄ではこのキーを欄が受け取るか"""
        if pressed == PLAIN_PASTE:
            return True
        return any(pressed == held for keys in self._held.values() for held in keys)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        if not isinstance(event, QKeyEvent) or not is_text_field(watched):
            return False
        pressed = QKeySequence(event.keyCombination())
        if kind is QEvent.Type.ShortcutOverride:
            if not self.holds(pressed):
                return False
            # 受け取ったと印を付けると、Qt は窓のショートカットを探さずにキーを欄へ送る
            event.accept()
            return True
        if pressed != PLAIN_PASTE:
            # 変えた先のキーは欄の既定の動きに任せる
            return False
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
