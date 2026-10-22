"""〔縦の作品に作り替える…〕の窓（#334）

AI の道具 ``reframe`` と同じ所（:func:`sashimono.engine.reframe.plan_reframe`）を通す 窓は
大きさ・枠の動かし方・追うときのならし方を尋ねるだけ 既定は本人の設定（`表示 → 設定…` の
縦動画への作り替え）で、窓で変えた値はその 1 回だけに効く（窓で変えた値で設定を書き換えると、
試しに変えただけの値が次から既定になる）
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from sashimono.core.reframe import (
    REFRAME_FOLLOW_NAMES,
    REFRAME_FOLLOWS,
    REFRAME_SIZE_NAMES,
    REFRAME_SIZES,
    REFRAME_SMOOTH_MS_RANGE,
    ReframeStyle,
)

__all__ = ["ReframeDialog", "ask_reframe"]


class ReframeDialog(QDialog):
    """作り替えの大きさと枠の動かし方を選ぶ 既定は本人の設定"""

    def __init__(self, style: ReframeStyle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("縦の作品に作り替える")
        form = QFormLayout(self)
        note = QLabel(
            "今の作品を縦の大きさに作り替えます 背景の動画は前の画面に置いた縦の枠の中を映し、"
            "枠は場面ごとに被写体（動きと肌色から見積もる 顔は見分けません）へ合わせます"
            " 字幕や文字は縦の画面の同じ割合の所へ置き直して折り返します メインとすべての"
            "シーンが変わり、取り消し 1 回で戻ります 横の作品も残すなら、先に"
            "〔名前を付けて保存…〕で別の名前にしてください",
            self,
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.size_box = QComboBox(self)
        for size in REFRAME_SIZES:
            self.size_box.addItem(REFRAME_SIZE_NAMES[size], size)
        self.size_box.setCurrentIndex(max(0, self.size_box.findData(style.size)))
        form.addRow("大きさ", self.size_box)
        self.follow_box = QComboBox(self)
        for follow in REFRAME_FOLLOWS:
            self.follow_box.addItem(REFRAME_FOLLOW_NAMES[follow], follow)
        self.follow_box.setCurrentIndex(max(0, self.follow_box.findData(style.follow)))
        form.addRow("枠の動かし方", self.follow_box)
        self.smooth_box = QSpinBox(self)
        self.smooth_box.setRange(*REFRAME_SMOOTH_MS_RANGE)
        self.smooth_box.setSingleStep(100)
        self.smooth_box.setSuffix(" ms")
        self.smooth_box.setValue(style.smooth_ms)
        self.smooth_box.setToolTip("長いほどゆっくり動いて落ち着き、短いほど被写体にぴったり付く")
        form.addRow("追うときのならし", self.smooth_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def choice(self) -> ReframeStyle:
        return ReframeStyle(
            size=str(self.size_box.currentData()),
            follow=str(self.follow_box.currentData()),
            smooth_ms=self.smooth_box.value(),
        )


def ask_reframe(parent: QWidget, style: ReframeStyle) -> ReframeStyle | None:
    """窓を出して選ばせる やめたら ``None``"""
    dialog = ReframeDialog(style, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()
