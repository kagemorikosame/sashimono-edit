"""タイムラインの右クリックの〔速度を変える…〕〔つなぎ目を目立たなくする…〕と、無音カットの窓の
「つなぎ目も処理する」の欄（#332）

AI の道具 ``set_speed`` ``smooth_cuts`` と同じ命令
（:class:`~sashimono.core.commands.speed.ChangeSpeed` :func:`~sashimono.core.seams.plan_smoothing`）
を通す 窓は何をするかを尋ねるだけ つなぎ目の処理の既定は本人の設定（`表示 → 設定…` の
つなぎ目）で、窓で変えた値はその 1 回だけに効く 毎回の既定を変えたい人は設定で変える
（窓で変えた値で設定を書き換えると、試しに変えただけの値が次から既定になる）
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from sashimono.core.commands.speed import SPEED_PERCENT_RANGE
from sashimono.core.seams import (
    SEAM_AUDIO_FADE_MS_RANGE,
    SEAM_CROSSFADE,
    SEAM_CROSSFADE_MS_RANGE,
    SEAM_VIDEO_NAMES,
    SEAM_VIDEOS,
    SEAM_ZOOM,
    SEAM_ZOOM_PERCENT_RANGE,
    SeamStyle,
)

__all__ = [
    "SeamDialog",
    "SeamFields",
    "SpeedChoice",
    "SpeedDialog",
    "ask_seams",
    "ask_speed",
]


class SeamFields:
    """つなぎ目の処理の欄 右クリックの窓と無音カットの窓で同じ欄を出す"""

    def __init__(self, parent: QWidget, form: QFormLayout, style: SeamStyle) -> None:
        self.video = QComboBox(parent)
        for video in SEAM_VIDEOS:
            self.video.addItem(SEAM_VIDEO_NAMES[video], video)
        self.video.setCurrentIndex(max(0, self.video.findData(style.video)))
        form.addRow("映像の処理", self.video)
        self.crossfade = _spin(parent, SEAM_CROSSFADE_MS_RANGE, style.crossfade_ms, " ms", 10)
        form.addRow("クロスフェードの長さ", self.crossfade)
        self.zoom = _spin(parent, SEAM_ZOOM_PERCENT_RANGE, style.zoom_percent, " %", 1)
        form.addRow("寄る量", self.zoom)
        self.audio = _spin(parent, SEAM_AUDIO_FADE_MS_RANGE, style.audio_fade_ms, " ms", 5)
        self.audio.setToolTip("つなぎ目の前後の音に入れる短いフェード 0 で音は触らない")
        form.addRow("音のフェード", self.audio)
        self.video.currentIndexChanged.connect(self._video_changed)
        self._video_changed()

    def _video_changed(self) -> None:
        # 使わない欄は灰色にする 寄りを選んだのにクロスフェードの長さを変えて、効いたと思わせない
        video = self.video.currentData()
        self.crossfade.setEnabled(video == SEAM_CROSSFADE)
        self.zoom.setEnabled(video == SEAM_ZOOM)

    def set_enabled(self, enabled: bool) -> None:
        self.video.setEnabled(enabled)
        self.audio.setEnabled(enabled)
        if enabled:
            self._video_changed()
        else:
            self.crossfade.setEnabled(False)
            self.zoom.setEnabled(False)

    def style(self) -> SeamStyle:
        return SeamStyle(
            video=str(self.video.currentData()),
            crossfade_ms=self.crossfade.value(),
            zoom_percent=self.zoom.value(),
            audio_fade_ms=self.audio.value(),
        )


def _spin(parent: QWidget, bounds: tuple[int, int], value: int, suffix: str, step: int) -> QSpinBox:
    box = QSpinBox(parent)
    box.setRange(*bounds)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


class SeamDialog(QDialog):
    """〔つなぎ目を目立たなくする…〕 付ける物を選ぶ 既定は本人の設定"""

    def __init__(self, style: SeamStyle, note: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("つなぎ目を目立たなくする")
        form = QFormLayout(self)
        label = QLabel(
            f"{note} 場面切り替え（クロスフェード）を重ねるか、後ろのクリップを 1 本おきに少し"
            "寄せ、音の継ぎ目に短いフェードを入れます 後から設定パネルとタイムラインで直せます"
            " 取り消し 1 回で戻ります",
            self,
        )
        label.setWordWrap(True)
        form.addRow(label)
        self.fields = SeamFields(self, form, style)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


def ask_seams(parent: QWidget, style: SeamStyle, note: str) -> SeamStyle | None:
    """窓を出して選ばせる やめたら ``None``"""
    dialog = SeamDialog(style, note, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.fields.style()
    finally:
        dialog.deleteLater()


#: 速度の変え方 AI の set_speed の keep と同じ名前
KEEPS = (("content", "中身を保って長さを合わせる（2 倍速で半分の長さ）"), ("length", "長さを保つ"))


@dataclass(frozen=True, slots=True)
class SpeedChoice:
    """〔速度を変える…〕で選んだ物"""

    speed: Fraction
    keep_length: bool


class SpeedDialog(QDialog):
    """〔速度を変える…〕 速度と変え方を選ぶ 既定は中身を保つ（AI の set_speed と同じ）"""

    def __init__(self, percent: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("速度を変える")
        form = QFormLayout(self)
        label = QLabel(
            "選んだクリップとリンクした絵と音を同じ速度にします 音は高さも変わります"
            " 中身を保つと、キーフレームも中身の同じ時刻へ動きます 取り消し 1 回で戻ります",
            self,
        )
        label.setWordWrap(True)
        form.addRow(label)
        self.speed_box = _spin(self, SPEED_PERCENT_RANGE, percent, " %", 10)
        form.addRow("速度", self.speed_box)
        self.keep_box = QComboBox(self)
        for value, name in KEEPS:
            self.keep_box.addItem(name, value)
        form.addRow("変え方", self.keep_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def choice(self) -> SpeedChoice:
        return SpeedChoice(
            speed=Fraction(self.speed_box.value(), 100),
            keep_length=self.keep_box.currentData() == "length",
        )


def ask_speed(parent: QWidget, percent: int) -> SpeedChoice | None:
    """窓を出して選ばせる 最初の速度は ``percent``（選んだクリップの今の速度） やめたら ``None``"""
    dialog = SpeedDialog(percent, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()
    finally:
        dialog.deleteLater()
