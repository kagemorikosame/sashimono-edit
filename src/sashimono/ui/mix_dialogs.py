"""タイムラインの右クリックの〔BGM を声に合わせて下げる…〕〔音量を基準にそろえる…〕の窓（#329）

AI の道具（``duck_audio`` ``normalize_loudness``）と同じ所（:mod:`sashimono.core.mixing`）を
通す 窓は何を使うかを尋ねるだけ 既定は本人の設定（`表示 → 設定…` のダッキングの値と、
仕上がりを調べる音量の基準）で、窓で変えた値はその 1 回だけに効く 毎回の既定を変えたい人は
設定で変える（窓で変えた値で設定を書き換えると、試しに変えただけの値が次から既定になる）
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from sashimono.core.mixing import (
    DEFAULT_PEAK_CEILING_DB,
    DUCK_ATTACK_MS_RANGE,
    DUCK_DEPTH_DB_RANGE,
    DUCK_RELEASE_MS_RANGE,
    DuckShape,
)
from sashimono.core.model import Track, TrackId
from sashimono.engine.checks import LOUDNESS_TARGETS

__all__ = [
    "DEFAULT_PEAK_CEILING_DB",
    "DuckChoice",
    "DuckDialog",
    "LoudnessChoice",
    "LoudnessDialog",
    "ask_duck",
    "ask_loudness",
]


@dataclass(frozen=True, slots=True)
class DuckChoice:
    """〔BGM を声に合わせて下げる…〕で選んだ物"""

    voice_track: TrackId
    shape: DuckShape


@dataclass(frozen=True, slots=True)
class LoudnessChoice:
    """〔音量を基準にそろえる…〕で選んだ物"""

    target: float
    ceiling: float = DEFAULT_PEAK_CEILING_DB


class DuckDialog(QDialog):
    """声のトラックと下げ方を選ぶ 下げ方の既定は本人の設定"""

    def __init__(
        self, voices: Sequence[Track], shape: DuckShape, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("BGM を声に合わせて下げる")
        form = QFormLayout(self)
        note = QLabel(
            "選んだクリップの音量を、声のトラックが話している間だけ下げます"
            " 話している所は字幕から（無ければ波形から）拾います"
            " 音量の線に点を打つので、後から線で直せます 取り消し 1 回で戻ります",
            self,
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.voice_box = QComboBox(self)
        for track in voices:
            self.voice_box.addItem(track.name or "トラック", str(track.id))
        form.addRow("声のトラック", self.voice_box)
        self.depth_box = QSpinBox(self)
        self.depth_box.setRange(*DUCK_DEPTH_DB_RANGE)
        self.depth_box.setSuffix(" dB")
        self.depth_box.setValue(round(shape.depth_db))
        form.addRow("下げる量", self.depth_box)
        self.attack_box = QSpinBox(self)
        self.attack_box.setRange(*DUCK_ATTACK_MS_RANGE)
        self.attack_box.setSingleStep(50)
        self.attack_box.setSuffix(" ms")
        self.attack_box.setValue(shape.attack_ms)
        form.addRow("下げる速さ", self.attack_box)
        self.release_box = QSpinBox(self)
        self.release_box.setRange(*DUCK_RELEASE_MS_RANGE)
        self.release_box.setSingleStep(100)
        self.release_box.setSuffix(" ms")
        self.release_box.setValue(shape.release_ms)
        form.addRow("戻る速さ", self.release_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def choice(self) -> DuckChoice:
        return DuckChoice(
            TrackId(str(self.voice_box.currentData())),
            DuckShape(
                depth_db=float(self.depth_box.value()),
                attack_ms=self.attack_box.value(),
                release_ms=self.release_box.value(),
            ),
        )


class LoudnessDialog(QDialog):
    """そろえる基準と音の山の上限を選ぶ 基準の既定は本人の設定"""

    def __init__(self, target: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("音量を基準にそろえる")
        form = QFormLayout(self)
        note = QLabel(
            "選んだクリップを混ぜた音を測り、どのクリップにも同じだけ掛けて基準の音量へそろえます"
            "（声と BGM の釣り合いは変えません） 測るあいだは裏で走り、ステータスバーの"
            "〔止める〕で止められます 取り消し 1 回で戻ります",
            self,
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.target_box = QComboBox(self)
        for value, name in LOUDNESS_TARGETS:
            self.target_box.addItem(name, value)
        self.target_box.setCurrentIndex(max(0, self.target_box.findData(round(target))))
        form.addRow("基準", self.target_box)
        self.ceiling_box = QDoubleSpinBox(self)
        self.ceiling_box.setRange(-20.0, 0.0)
        self.ceiling_box.setDecimals(1)
        self.ceiling_box.setSingleStep(0.5)
        self.ceiling_box.setSuffix(" dBFS")
        self.ceiling_box.setValue(DEFAULT_PEAK_CEILING_DB)
        self.ceiling_box.setToolTip("上げたときに音の山がこれを超えるなら、基準の手前で止めます")
        form.addRow("音の山の上限", self.ceiling_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def choice(self) -> LoudnessChoice:
        return LoudnessChoice(float(self.target_box.currentData()), float(self.ceiling_box.value()))


def ask_duck(parent: QWidget, voices: Sequence[Track], shape: DuckShape) -> DuckChoice | None:
    """窓を出して選ばせる やめたら ``None``"""
    dialog = DuckDialog(voices, shape, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()


def ask_loudness(parent: QWidget, target: float) -> LoudnessChoice | None:
    """窓を出して選ばせる やめたら ``None``"""
    dialog = LoudnessDialog(target, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()
