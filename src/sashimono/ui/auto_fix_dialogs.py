"""タイムラインの右クリックの〔色を自動で整える…〕〔ノイズを自動で減らす…〕の窓（#336）

AI の道具 ``auto_color`` ``reduce_noise`` と同じ計算（:mod:`sashimono.ai.auto_fix`）を通す
窓は直し方を尋ねるだけ 強さと下げる量の既定は本人の設定（`表示 → 設定…`）で、窓で変えた値は
その 1 回だけに効く（窓で変えた値で設定を書き換えると、試しに変えただけの値が次から既定になる）
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from sashimono.ai.auto_fix import ColorRequest, NoiseRequest
from sashimono.core.model import ClipId
from sashimono.effects.denoise import (
    AUTO_REDUCTION_DB_RANGE,
    DEFAULT_SENSITIVITY_DB,
    SENSITIVITY_DB_RANGE,
)
from sashimono.engine.auto_color import AUTO_COLOR_STRENGTH_RANGE

__all__ = ["ColorDialog", "NoiseDialog", "ask_color", "ask_noise"]


class ColorDialog(QDialog):
    """〔色を自動で整える…〕 強さ・直す物・合わせる相手を選ぶ 既定は本人の設定"""

    def __init__(
        self,
        strength: int,
        partners: list[tuple[ClipId, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("色を自動で整える")
        form = QFormLayout(self)
        label = QLabel(
            "選んだクリップの素材の絵を測り、暗い端を黒へ・明るい端を白へ広げ、白や灰色の物の"
            "色かぶりを取ります 合わせる相手を選ぶと、その明るさと色合いに寄せます 値はクリップの"
            "頭の「拡張色調補正」に入り、設定パネルで直せます 取り消し 1 回で戻ります",
            self,
        )
        label.setWordWrap(True)
        form.addRow(label)
        self.strength = QSpinBox(self)
        self.strength.setRange(*AUTO_COLOR_STRENGTH_RANGE)
        self.strength.setSingleStep(10)
        self.strength.setSuffix(" %")
        self.strength.setValue(strength)
        form.addRow("強さ", self.strength)
        self.exposure = QCheckBox("明るさ（露出とコントラスト）を直す", self)
        self.exposure.setChecked(True)
        form.addRow(self.exposure)
        self.white_balance = QCheckBox("白のつり合い（色かぶり）を直す", self)
        self.white_balance.setChecked(True)
        form.addRow(self.white_balance)
        self.partner = QComboBox(self)
        self.partner.addItem("合わせない（素材ごとに整える）", "")
        for clip_id, name in partners:
            self.partner.addItem(name, str(clip_id))
        form.addRow("合わせる相手", self.partner)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        # 両方を外すと何も直さない OK を押せなくして、押したのに何も起きないことを防ぐ
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        for box in (self.exposure, self.white_balance):
            box.toggled.connect(
                lambda _on: ok.setEnabled(
                    self.exposure.isChecked() or self.white_balance.isChecked()
                )
            )

    def choice(self) -> ColorRequest:
        partner = str(self.partner.currentData() or "")
        return ColorRequest(
            strength=float(self.strength.value()),
            exposure=self.exposure.isChecked(),
            white_balance=self.white_balance.isChecked(),
            match=ClipId(partner) if partner else None,
        )


def ask_color(
    parent: QWidget, strength: int, partners: list[tuple[ClipId, str]]
) -> ColorRequest | None:
    """窓を出して選ばせる やめたら ``None``"""
    dialog = ColorDialog(strength, partners, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()


class NoiseDialog(QDialog):
    """〔ノイズを自動で減らす…〕 下げる量と、測って掛けないと決めたクリップにも掛けるかを選ぶ"""

    def __init__(self, reduction: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ノイズを自動で減らす")
        form = QFormLayout(self)
        label = QLabel(
            "選んだクリップの波形から雑音の大きさを測り、空調やマイクのサーのような一定の雑音が"
            "聞こえるクリップにだけ、音のエフェクト「ノイズ除去」を足します 音楽や雑音に埋もれた"
            "声は削れるので掛けません 咳や物音のような突然の音は消えません 取り消し 1 回で戻ります",
            self,
        )
        label.setWordWrap(True)
        form.addRow(label)
        self.reduction = QSpinBox(self)
        self.reduction.setRange(*AUTO_REDUCTION_DB_RANGE)
        self.reduction.setSuffix(" dB")
        self.reduction.setValue(reduction)
        self.reduction.setToolTip("6 dB で雑音がおよそ半分、12 dB で 4 分の 1 大きいほど声も削れる")
        form.addRow("下げる量", self.reduction)
        self.sensitivity = QDoubleSpinBox(self)
        self.sensitivity.setRange(*SENSITIVITY_DB_RANGE)
        self.sensitivity.setSingleStep(0.5)
        self.sensitivity.setDecimals(1)
        self.sensitivity.setSuffix(" dB")
        self.sensitivity.setValue(DEFAULT_SENSITIVITY_DB)
        self.sensitivity.setToolTip("大きいほど残る雑音が減り、小さな声も下がる")
        form.addRow("雑音とみなす幅", self.sensitivity)
        self.force = QCheckBox("雑音が小さい・静かな所が無いクリップにも掛ける", self)
        form.addRow(self.force)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def choice(self) -> NoiseRequest:
        return NoiseRequest(
            reduction=float(self.reduction.value()),
            sensitivity=float(self.sensitivity.value()),
            force=self.force.isChecked(),
        )


def ask_noise(parent: QWidget, reduction: int) -> NoiseRequest | None:
    """窓を出して選ばせる 最初の量は本人の設定 やめたら ``None``"""
    dialog = NoiseDialog(reduction, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()
