"""整形とジェットカットの設定ダイアログ

どちらも「実行する前に、何がどれだけ変わるかを見せる」ことを重視している
起こし結果の一括整形も無音カットも、当たれば数十か所を一度に変える操作で、
やってみて違ったから戻す、では確認の手間が大きい
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.asr import DEFAULT_FILLERS, EXTRA_FILLERS, CleanupOptions, clean_transcript
from sashimono.compat.subtitle_looks import LookChoice, default_choice
from sashimono.core.commands import Voice
from sashimono.core.model import Transcript
from sashimono.engine.audio.silence import SilenceOptions
from sashimono.ui.flow_layout import narrow_combo
from sashimono.ui.library_thumbnails import LookThumbnails
from sashimono.ui.library_view import LibraryOptions
from sashimono.ui.subtitle.look_picker import SAME, LookPicker, LookSources
from sashimono.ui.theme import Colors, themed_style

__all__ = [
    "BurnAnswer",
    "BurnDialog",
    "BurnRequest",
    "CleanupDialog",
    "Estimator",
    "JetCutDialog",
]

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
        themed_style(self._preview, lambda: f"color: {Colors.TEXT_MUTED.name()};")

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
        themed_style(self._summary, lambda: f"color: {Colors.TEXT_MUTED.name()};")

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


@dataclass(frozen=True, slots=True)
class BurnRequest:
    """焼き込みの窓に渡す物 ``initial`` は最初に選んでおく見た目の鍵"""

    voices: list[tuple[Voice, str]]
    sources: LookSources
    initial: str
    #: 選んだ見た目の説明（既定の見た目の大きさ・選んでいるテキストの名前など）を作る手
    describe: Callable[[LookChoice], str]
    title: str = "字幕を焼き込む"
    accept: str = "焼き込む"
    #: 話し手を選ばせるか 字幕パネルの〔選んだ行を置く〕は見ている 1 本だけなので選ばせない
    choose_voices: bool = True


@dataclass(frozen=True, slots=True)
class BurnAnswer:
    """焼き込みの窓の答え ``looks`` は全員と違う見た目を選んだ話し手だけ"""

    voices: list[Voice]
    look: LookChoice
    looks: dict[Voice, LookChoice] = field(default_factory=dict)


class BurnDialog(QDialog):
    """字幕の焼き込み 焼き込む話し手（素材と音声）と、見た目を選ぶ（#308）

    話し手ごとに別のレイヤーへ入れる 既定は全部 見た目は見本の絵つきの一覧から選ぶ
    （プリセットとエイリアスの管理の窓と同じ係と枠） 既定は「全員」へ 1 つ 話し手を
    選ぶ先に切り替えると、その話し手だけ別の見た目にできる（既定は「全員と同じ」）
    何で置くかを窓に書く（置いてから違うと気付くと戻す手間になる）
    """

    def __init__(
        self,
        request: BurnRequest,
        *,
        thumbnails: LookThumbnails | None = None,
        options: LibraryOptions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(request.title)
        self.resize(720, 560)
        self._request = request
        self._voices: list[tuple[Voice, QCheckBox, QLabel]] = []
        #: 話し手ごとの見た目 選んでいない話し手は「全員と同じ」
        self._own: dict[Voice, LookChoice] = {}
        self._all: LookChoice | None = None

        layout = QVBoxLayout(self)
        if request.choose_voices:
            layout.addWidget(QLabel("焼き込む字幕（話し手ごとに別のレイヤーへ入れます）", self))
            rows = QFormLayout()
            for voice, label in request.voices:
                check = QCheckBox(label, self)
                check.setChecked(True)
                look = QLabel(self)
                themed_style(look, lambda: f"color: {Colors.TEXT_MUTED.name()};")
                rows.addRow(check, look)
                self._voices.append((voice, check, look))
            layout.addLayout(rows)
        else:
            # 話し手を選ばせない窓でも、印と見た目の名前の組は持つ（答えの作り方を 1 つにする）
            # 窓には出さない
            for voice, _label in request.voices:
                check = QCheckBox(self)
                check.setChecked(True)
                check.hide()
                look = QLabel(self)
                look.hide()
                self._voices.append((voice, check, look))

        target_row = QWidget(self)
        self._target = QComboBox(target_row)
        self._target.setObjectName("burn_look_target")
        # 選ぶ先の項目は 0 が全員、1 から先が話し手の並び 話し手そのもの（組）を Qt に持たせると
        # 並び（list）になって戻り、比べても探しても一致しない
        self._target.addItem("全員")
        for _voice, label in request.voices:
            self._target.addItem(label)
        narrow_combo(self._target, 8)
        row = QHBoxLayout(target_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("見た目を選ぶ先", target_row))
        row.addWidget(self._target, 1)
        layout.addWidget(target_row)
        # 話し手が 1 人なら選ぶ先は要らない
        target_row.setVisible(request.choose_voices and len(request.voices) > 1)

        self._picker = LookPicker(
            request.sources, thumbnails=thumbnails, options=options, parent=self
        )
        layout.addWidget(self._picker, 1)

        self._note = QLabel(self)
        self._note.setObjectName("burn_template")
        self._note.setWordWrap(True)
        themed_style(self._note, lambda: f"color: {Colors.TEXT_MUTED.name()};")
        layout.addWidget(self._note)
        self._buttons = _confirm_buttons(self, request.accept)
        layout.addWidget(self._buttons)

        if not self._picker.choose(request.initial):
            # 覚えた見た目が消えていた（名前を変えた・消した） 既定の見た目から選び直してもらう
            self._picker.choose(default_choice().key)
        self._all = self._picker.current()
        self._picker.changed.connect(self._on_picked)
        self._target.currentIndexChanged.connect(self._on_target)
        for _voice, check, _look in self._voices:
            check.toggled.connect(lambda _on: self._update())
        self._update()

    # --- 外から ---

    def chosen(self) -> list[Voice]:
        """印の付いた話し手"""
        return [voice for voice, check, _look in self._voices if check.isChecked()]

    def answer(self) -> BurnAnswer | None:
        """窓の答え 見た目が選べない物なら ``None``"""
        everyone = self._all
        if everyone is None or everyone.refusal:
            return None
        chosen = self.chosen()
        looks = {voice: look for voice, look in self._own.items() if voice in chosen}
        return BurnAnswer(chosen, everyone, looks)

    def choose(self, key: str, voice: Voice | None = None) -> bool:
        """``voice``（省けば全員）の見た目を鍵で選ぶ 試験と、覚えた選択を戻すのに使う"""
        self.choose_target(voice)
        return self._picker.choose(key)

    def choose_target(self, voice: Voice | None) -> None:
        voices = [known for known, _label in self._request.voices]
        index = voices.index(voice) + 1 if voice in voices else 0
        self._target.setCurrentIndex(index)

    @property
    def picker(self) -> LookPicker:
        return self._picker

    def done(self, result: int) -> None:
        self._picker.release()
        super().done(result)

    # --- 中の手順 ---

    def _voice(self) -> Voice | None:
        index = self._target.currentIndex()
        if index < 1 or index > len(self._request.voices):
            return None
        return self._request.voices[index - 1][0]

    def _on_target(self) -> None:
        voice = self._voice()
        self._picker.set_allow_same(voice is not None)
        if voice is None:
            key = self._all.key if self._all is not None else default_choice().key
        else:
            own = self._own.get(voice)
            key = own.key if own is not None else SAME
        self._picker.choose(key)
        self._update()

    def _on_picked(self) -> None:
        picked = self._picker.current()
        if picked is None:
            return
        voice = self._voice()
        if voice is None:
            self._all = picked
        elif picked.kind == SAME:
            self._own.pop(voice, None)
        else:
            self._own[voice] = picked
        self._update()

    def _update(self) -> None:
        everyone = self._all
        for voice, _check, look in self._voices:
            own = self._own.get(voice)
            look.setText(f"見た目: {own.label}" if own is not None else "見た目: 全員と同じ")
        lines: list[str] = []
        if everyone is not None:
            lines.append(self._request.describe(everyone))
        refused = [c for c in (everyone, *self._own.values()) if c is not None and c.refusal]
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(everyone is not None and not refused and bool(self.chosen()))
        for choice in refused:
            lines.append(f"「{choice.label}」は使えない: {choice.refusal}")
        self._note.setText("\n".join(lines))
