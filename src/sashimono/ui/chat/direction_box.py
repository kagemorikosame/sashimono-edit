"""AI のパネルの演出の計画の欄（#470 段 4 #479）

AI が出した演出の計画（:mod:`sashimono.ai.direction_plan`）を項目ごとの印で見せる 人は
要らない項目の印を外し、〔付ける〕で残りを 1 回で付ける 取り消し 1 回で全部戻る

欄の中身は計画の写しから作る パネルの見張りの時計から呼ばれるので、写しの版と動いているかが
変わったときだけ書き換える 印だけが変わったとき（人が外したとき）は項目を作り直さない
作り直すと、外した直後に巻物が頭へ戻り、続けて外したい項目を探し直すことになる
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai.direction_plan import DirectionItem, ProposalState, ProposalView
from sashimono.ui.flow_layout import FlowLayout
from sashimono.ui.theme import Colors, themed_style

__all__ = ["APPLY_TEXT", "CANCEL_TEXT", "DirectionPlanBox", "item_text"]

APPLY_TEXT = "付ける"
CANCEL_TEXT = "やめる"


def item_text(item: DirectionItem) -> str:
    """項目の行 時刻・何を・どれだけ"""
    text = f"{item.at} {item.action_name}"
    if item.amount is not None:
        unit = " %" if item.action == "zoom" else " px"
        text += f" {item.amount}{unit}"
    if item.sound_name:
        # どの音を付けるかを〔付ける〕の前に見せる 登録したフォルダから AI が選んだ音（#522）は、
        # 人がまだ聞いていないことがある
        text += f"（{item.sound_name}）"
    return text


class DirectionPlanBox(QScrollArea):
    """演出の計画の欄 計画が無い間は隠す"""

    #: 項目の番号と、外したか（真で外す）
    toggled = Signal(int, bool)
    apply_requested = Signal()
    cancel_requested = Signal()
    close_requested = Signal()

    #: 欄の高さの上限（行） 項目が多いと会話の欄の高さを奪う
    MAX_LINES = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        lines = self.fontMetrics().lineSpacing()
        frame = 2 * self.frameWidth()
        self.setMinimumHeight(3 * lines + frame)
        self.setMaximumHeight(self.MAX_LINES * (lines + 6) + frame)
        self.setVisible(False)
        #: 最後に描いた版と、AI が応えている最中か
        self._drawn: tuple[int, bool] | None = None
        #: 印を作った項目 同じ項目なら印の向きだけを直す
        self._items: tuple[DirectionItem, ...] = ()
        self._boxes: dict[int, QCheckBox] = {}

        inner = QWidget(self)
        self._title = QLabel(inner)
        bold = self._title.font()
        bold.setBold(True)
        self._title.setFont(bold)
        self._title.setTextFormat(Qt.TextFormat.PlainText)
        self._title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._close = QToolButton(inner)
        self._close.setText("×")
        self._close.setToolTip("演出の計画を閉じる")
        self._close.setAccessibleName("演出の計画を閉じる")
        self._close.setAutoRaise(True)
        self._close.clicked.connect(self.close_requested)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._title, 1)
        header.addWidget(self._close)

        self._list = QVBoxLayout()
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(2)

        self._status = QLabel(inner)
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        themed_style(self._status, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._apply = QPushButton(APPLY_TEXT, inner)
        self._apply.setToolTip("印の付いた演出を付ける 取り消し 1 回で全部戻る")
        self._apply.clicked.connect(self.apply_requested)
        self._cancel = QPushButton(CANCEL_TEXT, inner)
        self._cancel.setToolTip("何も付けずに計画を閉じる")
        self._cancel.clicked.connect(self.cancel_requested)
        buttons = QWidget(inner)
        buttons_layout = FlowLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self._apply)
        buttons_layout.addWidget(self._cancel)

        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addLayout(self._list)
        layout.addWidget(self._status)
        layout.addWidget(buttons)
        self.setWidget(inner)

    @property
    def apply_button(self) -> QPushButton:
        return self._apply

    @property
    def cancel_button(self) -> QPushButton:
        return self._cancel

    @property
    def close_button(self) -> QToolButton:
        return self._close

    def title(self) -> str:
        return self._title.text()

    def status_text(self) -> str:
        return self._status.text()

    def check_box(self, number: int) -> QCheckBox:
        return self._boxes[number]

    def show_proposal(self, view: ProposalView | None, *, working: bool) -> bool:
        """計画を見せる 描き直したら真 ``working`` は AI が応えている最中か

        応えている間は〔付ける〕を押せなくする AI の 1 回の指示は取り消し 1 回にまとまって
        いるので、その最中に付けると演出が AI の段に混ざり、計画だけを取り消せなくなる
        """
        if view is None:
            changed = self._drawn is not None
            self._drawn = None
            self._set_items(())
            self.setVisible(False)
            return changed
        key = (view.version, working)
        if key == self._drawn:
            return False
        self._drawn = key
        if view.items != self._items:
            self._set_items(view.items)
        proposed = view.state is ProposalState.PROPOSED
        for number, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(number not in view.excluded)
            box.blockSignals(False)
            box.setEnabled(proposed)
        chosen = len(view.chosen)
        self._title.setText(
            f"演出の計画（{view.strength_name} {chosen}/{len(view.items)} 個を付ける）"
        )
        self._status.setText(_status(view))
        self._apply.setVisible(proposed)
        self._apply.setEnabled(proposed and chosen > 0 and not working)
        self._cancel.setVisible(proposed)
        self._close.setEnabled(not proposed)
        self.setVisible(True)
        return True

    def _set_items(self, items: tuple[DirectionItem, ...]) -> None:
        for box in self._boxes.values():
            self._list.removeWidget(box)
            box.deleteLater()
        self._boxes = {}
        self._items = items
        parent = self.widget()
        for item in items:
            box = QCheckBox(item_text(item), parent)
            box.setToolTip(item.reason)
            box.setChecked(True)
            number = item.number
            # 印を外したら外す（真） 付けたら戻す
            box.toggled.connect(lambda on, n=number: self.toggled.emit(n, not on))
            self._list.addWidget(box)
            self._boxes[number] = box


def _status(view: ProposalView) -> str:
    """欄の下の一言 外した物の数と、付けた後の結果"""
    lines: list[str] = []
    if view.state is ProposalState.PROPOSED:
        lines.append("要らない項目の印を外して〔付ける〕 取り消し 1 回で全部戻ります")
    elif view.state is ProposalState.CANCELLED:
        lines.append("やめました 何も付けていません")
    elif view.result is not None:
        lines.append(f"{len(view.result.applied)} 個を付けました 取り消し 1 回で全部戻ります")
        if view.result.skipped:
            lines.append(f"付けられなかった物 {len(view.result.skipped)} 個")
    if not view.items:
        lines.append("付けられる演出がありません")
    if view.dropped:
        lines.append(f"計画に入れなかった物 {len(view.dropped)} 個（AI に訳を尋ねられます）")
    return "\n".join(lines)
