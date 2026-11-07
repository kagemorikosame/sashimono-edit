"""字幕の時刻を声に合わせる欄と窓（#426）

ずらす欄は設定画面・起こしの窓・字幕パネルの〔時刻を合わせる…〕の 3 か所で同じ形にする
所ごとに幅や刻みが違うと、設定で選んだ値を窓で選び直せない
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.subtitle_timing import OFFSET_MS_RANGE, SubtitleTiming
from sashimono.ui.theme import Colors, themed_style

__all__ = ["FIT_VOICE_TEXT", "RetimeDialog", "offset_box"]

#: 声に寄せる印の文言 設定画面と窓でそろえる
FIT_VOICE_TEXT = "行の頭と終わりを声に合わせる"


def offset_box(parent: QWidget | None, value: int) -> QSpinBox:
    """字幕の時刻をずらすミリ秒の欄 負は早く、正は遅く"""
    box = QSpinBox(parent)
    box.setRange(*OFFSET_MS_RANGE)
    box.setSingleStep(50)
    box.setSuffix(" ms")
    box.setValue(value)
    return box


class RetimeDialog(QDialog):
    """起こした後の字幕の時刻を合わせ直す（字幕パネルの〔時刻を合わせる…〕）

    前の版で起こした字幕は声に寄せていない 起こし直すと人が直した文言が消えるので、文言を
    残したまま時刻だけを直せるようにする ずらす量は今の時刻から数える（2 回当てると 2 回ずれる）
    """

    def __init__(
        self, timing: SubtitleTiming, *, selected: int, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("字幕の時刻を合わせる")

        self._fit = QCheckBox(FIT_VOICE_TEXT, self)
        self._fit.setChecked(timing.fit_to_voice)
        self._fit.setToolTip("近く（前後 0.5 秒）の声の出だしと終わりへ寄せる 波形の解析が要る")
        self._start = offset_box(self, timing.start_offset_ms)
        self._end = offset_box(self, timing.end_offset_ms)

        self._all = QRadioButton("全部の行", self)
        self._selected = QRadioButton(f"選んだ行（{selected} 行）", self)
        self._all.setChecked(True)
        # 選んでいなければ選べない 選べても当てる行が無い
        self._selected.setEnabled(selected > 0)

        note = QLabel(
            "ずらす量は今の時刻から数えます 声に合わせた後にずらします 次の行の頭は越えません"
            " 取り消しで戻せます",
            self,
        )
        note.setWordWrap(True)
        themed_style(note, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        form = QFormLayout()
        form.addRow("", self._fit)
        form.addRow("出す時刻をずらす（負は早く）", self._start)
        form.addRow("消す時刻をずらす（正は遅く）", self._end)
        form.addRow("合わせる行", self._all)
        form.addRow("", self._selected)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setText("合わせる")
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel is not None:
            # 既定の文言は環境の言語に従うので、ここで日本語に固定する
            cancel.setText("やめる")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def timing(self) -> SubtitleTiming:
        return SubtitleTiming(
            fit_to_voice=self._fit.isChecked(),
            start_offset_ms=self._start.value(),
            end_offset_ms=self._end.value(),
        )

    @property
    def only_selected(self) -> bool:
        return self._selected.isEnabled() and self._selected.isChecked()
