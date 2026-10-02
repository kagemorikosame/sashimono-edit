"""狭いパネルの置き方 幅が足りなければ次の行へ折り返す・選択肢の長さで幅を決めない

パネルのボタンを横 1 列に並べると、ボタンの幅の和がそのパネルの最小の幅になり、
窓の最小の幅はパネルの最小の幅の和になる 字幕と AI のパネルのボタンの列だけで
1366 の画面に窓が収まらなかった 折り返せば、最小の幅はいちばん広いボタン 1 つ分で済む
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QComboBox, QLabel, QLayout, QLayoutItem, QSizePolicy, QWidget

__all__ = ["ElidedLabel", "FlowLayout", "flow_of", "narrow_combo"]


class ElidedLabel(QLabel):
    """入り切らない分を「…」にする 1 行の名前 長い名前でパネルの最小の幅を押し広げない

    設定パネルのエフェクトの見出しに使う 配布スクリプトの名前は長い物があり、そのまま
    出すと名前の幅が設定パネルの最小の幅になり、窓が画面からはみ出す 省いたときは
    指せば全部を読める
    """

    #: 縮めても残す文字の数 これより狭くはしない（「…」だけでは何の組か分からない）
    KEEP = 4

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = text
        # 名前は本人や配布物が付けた文字 ``<b>`` などを装飾として読ませない
        self.setTextFormat(Qt.TextFormat.PlainText)
        super().setText(text)

    def full_text(self) -> str:
        return self._full

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt の命名規約
        # 広い所では全部を出したい 今出ている（省いた）文字で測ると、広げても戻らない
        hint = super().sizeHint()
        margins = self.contentsMargins()
        width = self.fontMetrics().horizontalAdvance(self._full) + margins.left() + margins.right()
        return QSize(width + 2 * self.margin(), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt の命名規約
        keep = self.fontMetrics().horizontalAdvance(self._full[: self.KEEP] + "…")
        return QSize(min(keep, self.sizeHint().width()), super().minimumSizeHint().height())

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        shown = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, self.contentsRect().width()
        )
        if shown != self.text():
            super().setText(shown)
        self.setToolTip(self._full if shown != self._full else "")


def narrow_combo(box: QComboBox, characters: int, *, name_tip: bool = True) -> None:
    """選択肢の中でいちばん長い物ではなく、``characters`` 文字ぶんを最小の幅にする

    Qt の既定は、いちばん長い選択肢が入る幅を最小にする 素材やフォントの名前の長さで
    パネルの最小の幅が決まり、窓が画面に収まらなくなる 置き方が伸ばす所（横に伸びる欄）
    では今までどおり広がる 開いた一覧には全部の名前が出る

    ``name_tip`` が真なら、選んでいる名前を補足に出す 縮めた欄では名前の後ろが見えない
    欄に別の補足（説明）があるときは偽にする 上書きすると説明が消える
    """
    box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(characters)
    if name_tip:
        box.currentTextChanged.connect(box.setToolTip)
        box.setToolTip(box.currentText())


class FlowLayout(QLayout):
    """左から並べ、入らなければ折り返す 行の高さは、その行でいちばん高い物に合わせる

    高さは幅で決まる（:meth:`heightForWidth`） 縦の箱の中に置けば、箱が折り返した分の
    高さを取ってくれる 取らないと、折り返した 2 行目が下の部品に重なる
    """

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt の命名規約
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt の命名規約
        # 範囲の外で None を返すのは Qt の約束 投げると Qt が並べ直す途中で落ちる
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt の命名規約
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def spacing(self) -> int:
        return self._spacing

    def setSpacing(self, spacing: int) -> None:  # noqa: N802 - Qt の命名規約
        self._spacing = spacing
        self.invalidate()

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt の命名規約
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt の命名規約
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt の命名規約
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt の命名規約
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt の命名規約
        # 広い所では 1 行に並べたい 和の幅を望みの幅として返す
        width = 0
        height = 0
        for item in self._visible():
            hint = item.sizeHint()
            width += hint.width() + (self._spacing if width else 0)
            height = max(height, hint.height())
        return self._with_margins(QSize(width, height))

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt の命名規約
        # 1 つずつ縦に並べても入る幅 いちばん広い物 1 つ分
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        return self._with_margins(size)

    def _with_margins(self, size: QSize) -> QSize:
        margins: QMargins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _visible(self) -> list[QLayoutItem]:
        # 隠した部品の分まで場所を取ると、並びに穴が空く
        return [item for item in self._items if not item.isEmpty()]

    def _arrange(self, rect: QRect, *, move: bool) -> int:
        """並べる ``move`` が偽なら置かずに高さだけ数える 並べた高さを返す"""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = area.x()
        y = area.y()
        line = 0
        for item in self._visible():
            hint = item.sizeHint()
            # 1 つで幅を超える物は、その幅まで縮めて置く（最小の幅までは縮む）
            width = max(min(hint.width(), area.width()), item.minimumSize().width())
            if x > area.x() and x + width > area.right() + 1:
                x = area.x()
                y += line + self._spacing
                line = 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, hint.height())))
            x += width + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y() + margins.bottom()


def flow_of(*widgets: QWidget, parent: QWidget | None = None) -> FlowLayout:
    """部品を並べた折り返しの置き方 縦に伸ばさない（行の高さは中身で決まる）"""
    layout = FlowLayout(parent)
    for widget in widgets:
        policy = widget.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        widget.setSizePolicy(policy)
        layout.addWidget(widget)
    return layout
