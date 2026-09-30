"""整形とジェットカットの設定ダイアログ

どちらも「実行する前に、何がどれだけ変わるかを見せる」ことを重視している
起こし結果の一括整形も無音カットも、当たれば数十か所を一度に変える操作で、
やってみて違ったから戻す、では確認の手間が大きい
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.asr import DEFAULT_FILLERS, EXTRA_FILLERS, CleanupOptions, clean_transcript
from sashimono.core.model import Transcript
from sashimono.engine.audio.silence import SilenceOptions
from sashimono.ui.theme import Colors

__all__ = ["BurnDialog", "CleanupDialog", "Estimator", "JetCutDialog"]

#: 削る量の見積もり 条件と「発話を守るか」を受け取り、(か所, 秒) を返す
type Estimator = Callable[[SilenceOptions, bool], tuple[int, float]]


def _confirm_buttons(parent: QDialog, accept: str) -> QDialogButtonBox:
    """OK / Cancel の文言を日本語にした確定ボタン

    Qt の標準ボタンは環境の言語に従うので、そのままだと日本語の画面に英語が
    混じる 押すと何が起きるかを名前にしておく
    """
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent
    )
    for standard, label in (
        (QDialogButtonBox.StandardButton.Ok, accept),
        (QDialogButtonBox.StandardButton.Cancel, "やめる"),
    ):
        button = buttons.button(standard)
        if button is not None:
            button.setText(label)
    buttons.accepted.connect(parent.accept)
    buttons.rejected.connect(parent.reject)
    return buttons


#: 句読点の扱いの選択肢
PUNCTUATION = (("keep", "そのまま"), ("space", "空白にする"), ("strip", "落とす"))


class CleanupDialog(QDialog):
    """フィラー語の除去と改行の設定

    条件を変えるたびに、結果の 1 例と変化する枚数をその場で出す 字幕は数十枚
    あるので、全部を目で追ってから決めるわけにいかない
    """

    def __init__(self, transcript: Transcript, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("字幕を整形")
        self.resize(480, 360)
        self._transcript = transcript

        self._fillers = QLineEdit("、".join(DEFAULT_FILLERS), self)
        self._fillers.setToolTip("読点で区切って並べる")
        self._extra = QCheckBox("「" + "」「".join(EXTRA_FILLERS[:3]) + "」なども落とす", self)
        self._repeats = QCheckBox("言い直しをまとめる", self)
        self._repeats.setChecked(True)

        self._punctuation = QComboBox(self)
        for value, label in PUNCTUATION:
            self._punctuation.addItem(label, value)

        self._line_chars = QSpinBox(self)
        self._line_chars.setRange(0, 80)
        self._line_chars.setValue(20)
        self._line_chars.setSuffix(" 文字")
        self._line_chars.setSpecialValueText("折り返さない")

        self._lines = QSpinBox(self)
        self._lines.setRange(1, 4)
        self._lines.setValue(2)
        self._lines.setSuffix(" 行")

        self._skip_edited = QCheckBox("手で直した字幕には触らない", self)
        self._skip_edited.setChecked(True)

        self._preview = QLabel(self)
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()};")

        form = QFormLayout()
        form.addRow("落とす語", self._fillers)
        form.addRow("", self._extra)
        form.addRow("", self._repeats)
        form.addRow("句読点", self._punctuation)
        form.addRow("1 行の長さ", self._line_chars)
        form.addRow("行数", self._lines)
        form.addRow("", self._skip_edited)

        buttons = _confirm_buttons(self, "整形する")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._preview, 1)
        layout.addWidget(buttons)

        self._fillers.textChanged.connect(self._update_preview)
        for check in (self._extra, self._repeats, self._skip_edited):
            check.toggled.connect(self._update_preview)
        self._punctuation.currentIndexChanged.connect(self._update_preview)
        self._line_chars.valueChanged.connect(self._update_preview)
        self._lines.valueChanged.connect(self._update_preview)
        self._update_preview()

    def options(self) -> CleanupOptions:
        words = tuple(part.strip() for part in self._fillers.text().split("、") if part.strip())
        if self._extra.isChecked():
            words = words + EXTRA_FILLERS
        return CleanupOptions(
            remove_fillers=bool(words),
            fillers=words,
            collapse_repeats=self._repeats.isChecked(),
            punctuation=str(self._punctuation.currentData()),
            max_line_chars=self._line_chars.value(),
            max_lines=self._lines.value(),
            skip_edited=self._skip_edited.isChecked(),
        )

    def result_transcript(self) -> Transcript:
        return clean_transcript(self._transcript, self.options())

    def _update_preview(self) -> None:
        cleaned = self.result_transcript()
        before = {segment.id: segment.text for segment in self._transcript.segments}
        changed = sum(1 for s in cleaned.segments if before.get(s.id) != s.text)
        dropped = len(self._transcript) - len(cleaned)

        sample = next(
            (
                (before[s.id], s.text)
                for s in cleaned.segments
                if s.id in before and before[s.id] != s.text
            ),
            None,
        )
        lines = [f"{changed} 枚が変わり、{dropped} 枚が消えます"]
        if sample is not None:
            lines.append(f"例: 「{sample[0]}」 → 「{sample[1]}」")
        self._preview.setText("\n".join(lines))


class JetCutDialog(QDialog):
    """無音カットの設定

    切る前に「何か所・合計何秒を削るか」を出す 無音カットは 1 回で数十か所を
    変えるので、実行してから確認するには変化が大きすぎる
    """

    def __init__(
        self,
        *,
        has_transcript: bool,
        estimate: Estimator | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("無音カット")
        self.resize(420, 260)

        self._threshold = QDoubleSpinBox(self)
        self._threshold.setRange(-80.0, 0.0)
        self._threshold.setValue(-40.0)
        self._threshold.setSuffix(" dB")

        self._min_silence = QDoubleSpinBox(self)
        self._min_silence.setRange(0.05, 10.0)
        self._min_silence.setValue(0.5)
        self._min_silence.setSingleStep(0.05)
        self._min_silence.setSuffix(" 秒")

        self._padding = QDoubleSpinBox(self)
        self._padding.setRange(0.0, 2.0)
        self._padding.setValue(0.1)
        self._padding.setSingleStep(0.01)
        self._padding.setSuffix(" 秒")

        self._keep_speech = QCheckBox("起こし結果のある区間は切らない", self)
        self._keep_speech.setChecked(has_transcript)
        self._keep_speech.setEnabled(has_transcript)
        if not has_transcript:
            self._keep_speech.setToolTip("先に字幕を起こすと使えます")

        self._summary = QLabel(self)
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()};")

        form = QFormLayout()
        form.addRow("しきい値", self._threshold)
        form.addRow("最短の無音", self._min_silence)
        form.addRow("前後に残す余白", self._padding)
        form.addRow("", self._keep_speech)

        buttons = _confirm_buttons(self, "カットする")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._summary, 1)
        layout.addWidget(buttons)

        self._estimate = estimate
        for spin in (self._threshold, self._min_silence, self._padding):
            spin.valueChanged.connect(self._update_summary)
        self._keep_speech.toggled.connect(self._update_summary)
        self._update_summary()

    @property
    def keep_speech(self) -> bool:
        return self._keep_speech.isChecked()

    def options(self) -> SilenceOptions:
        return SilenceOptions(
            threshold_db=self._threshold.value(),
            min_silence=_to_fraction(self._min_silence.value()),
            padding=_to_fraction(self._padding.value()),
        )

    def _update_summary(self) -> None:
        """削る量を出す 見積もりの手当てが無ければ何も出さない"""
        estimate = self._estimate
        if estimate is None:
            self._summary.setText("")
            return
        count, seconds = estimate(self.options(), self.keep_speech)
        if count == 0:
            self._summary.setText("切る場所が見つかりません しきい値を上げてみてください")
            return
        self._summary.setText(f"{count} か所、合計 {seconds:.1f} 秒を削ります")


def _to_fraction(value: float) -> Fraction:
    """秒（float）を有理数へ 10ms 単位で十分"""
    return Fraction(round(value * 100), 100)


class BurnDialog(QDialog):
    """字幕の焼き込み 焼き込む話し手（素材と音声）を選び、見た目のひな形を示す

    話し手ごとに別のレイヤーへ入れる 既定は全部 ひな形はタイムラインで選んでいる
    テキストがあればそれ（フォント・色・位置・縁取り・エフェクトを写し、本文だけ差し替える）
    無ければ既定の見た目 どちらで置くかを窓に書く（置いてから違うと気付くと戻す手間になる）
    """

    def __init__(
        self,
        voices: list[tuple[object, str]],
        template_note: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("字幕を焼き込む")
        self._voices: list[tuple[object, QCheckBox]] = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("焼き込む字幕（話し手ごとに別のレイヤーへ入れます）", self))
        for voice, label in voices:
            check = QCheckBox(label, self)
            check.setChecked(True)
            layout.addWidget(check)
            self._voices.append((voice, check))
        note = QLabel(template_note, self)
        note.setObjectName("burn_template")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()};")
        layout.addWidget(note)
        layout.addWidget(_confirm_buttons(self, "焼き込む"))

    def chosen(self) -> list[object]:
        """印の付いた話し手"""
        return [voice for voice, check in self._voices if check.isChecked()]
