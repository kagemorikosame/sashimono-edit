"""タイムラインの右クリックの〔強調の動きを付ける…〕の窓（#331）

AI の道具 ``emphasize`` と同じ所（:mod:`sashimono.effects.emphasis`）を通す 窓は何を付けるかを
尋ねるだけ 既定は本人の設定（`表示 → 設定…` の強調の動き）で、窓で変えた値はその 1 回だけに
効く 毎回の既定を変えたい人は設定で変える（窓で変えた値で設定を書き換えると、試しに変えた
だけの値が次から既定になる）
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from sashimono.core.emphasis import (
    EMPHASIS_EASING_NAMES,
    EMPHASIS_EASINGS,
    EMPHASIS_KIND_NAMES,
    EMPHASIS_KINDS,
    EMPHASIS_PAN_PX_RANGE,
    EMPHASIS_POP_PERCENT_RANGE,
    EMPHASIS_RAMP_MS_RANGE,
    EMPHASIS_SHAKE_PX_RANGE,
    EMPHASIS_ZOOM_PERCENT_RANGE,
    PAN,
    PAN_DIRECTION_NAMES,
    PAN_DIRECTIONS,
    POP,
    SHAKE,
    ZOOM,
    EmphasisStyle,
)

__all__ = ["EmphasisChoice", "EmphasisDialog", "ask_emphasis"]


@dataclass(frozen=True, slots=True)
class EmphasisChoice:
    """〔強調の動きを付ける…〕で選んだ物 ``strength`` は種類ごとの強さ

    zoom と pop は拡大率の %、shake は揺れの幅の画素、pan はずらす画素（向きは ``direction``）
    """

    kind: str
    strength: int
    ramp_ms: int
    easing: str
    direction: str = "up"


def _strength_of(style: EmphasisStyle, kind: str) -> tuple[int, tuple[int, int], str]:
    """種類ごとの強さの既定・範囲・単位"""
    if kind == ZOOM:
        return style.zoom_percent, EMPHASIS_ZOOM_PERCENT_RANGE, " %"
    if kind == POP:
        return style.pop_percent, EMPHASIS_POP_PERCENT_RANGE, " %"
    if kind == SHAKE:
        return style.shake_px, EMPHASIS_SHAKE_PX_RANGE, " px"
    return style.pan_px, EMPHASIS_PAN_PX_RANGE, " px"


class EmphasisDialog(QDialog):
    """付ける動きを選ぶ 既定は本人の設定"""

    def __init__(self, style: EmphasisStyle, span_note: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._style = style
        self.setWindowTitle("強調の動きを付ける")
        form = QFormLayout(self)
        note = QLabel(
            f"選んだクリップの {span_note} に動きを付けます 配置の値（揺らすときはランダム移動の"
            "エフェクト）にキーフレームを打つので、後から設定パネルとグラフエディタで直せます"
            " 範囲の外の動きは変えません 取り消し 1 回で戻ります",
            self,
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.kind_box = QComboBox(self)
        for kind in EMPHASIS_KINDS:
            self.kind_box.addItem(EMPHASIS_KIND_NAMES[kind], kind)
        form.addRow("動き", self.kind_box)
        self.strength_box = QSpinBox(self)
        form.addRow("強さ", self.strength_box)
        self.direction_box = QComboBox(self)
        for direction in PAN_DIRECTIONS:
            self.direction_box.addItem(PAN_DIRECTION_NAMES[direction], direction)
        form.addRow("ずらす向き", self.direction_box)
        self.ramp_box = QSpinBox(self)
        self.ramp_box.setRange(*EMPHASIS_RAMP_MS_RANGE)
        self.ramp_box.setSingleStep(50)
        self.ramp_box.setSuffix(" ms")
        self.ramp_box.setValue(style.ramp_ms)
        form.addRow("動き切る長さ", self.ramp_box)
        self.easing_box = QComboBox(self)
        for easing in EMPHASIS_EASINGS:
            self.easing_box.addItem(EMPHASIS_EASING_NAMES[easing], easing)
        self.easing_box.setCurrentIndex(max(0, self.easing_box.findData(style.easing)))
        form.addRow("動き方", self.easing_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.kind_box.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()

    def _kind_changed(self) -> None:
        # 種類を替えたら強さを本人の既定へ戻す 寄る量の 120 % のまま揺れへ替えると、120 px で揺れる
        value, bounds, suffix = _strength_of(self._style, self.kind())
        self.strength_box.setRange(*bounds)
        self.strength_box.setSuffix(suffix)
        self.strength_box.setValue(value)
        self.direction_box.setEnabled(self.kind() == PAN)

    def kind(self) -> str:
        return str(self.kind_box.currentData())

    def choice(self) -> EmphasisChoice:
        return EmphasisChoice(
            kind=self.kind(),
            strength=self.strength_box.value(),
            ramp_ms=self.ramp_box.value(),
            easing=str(self.easing_box.currentData()),
            direction=str(self.direction_box.currentData()),
        )


def ask_emphasis(parent: QWidget, style: EmphasisStyle, span_note: str) -> EmphasisChoice | None:
    """窓を出して選ばせる やめたら ``None``"""
    dialog = EmphasisDialog(style, span_note, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()
