"""〔字幕〕→〔ナレーションを作る…〕の窓（#335）

台本と声（エンジンの一覧から）を選んで、AI の generate_narration と同じ処理で声を作って置く
（2026-10-20 利用者の決定 人の入口） 窓の既定は本人の設定で、窓で変えた値はその 1 回だけ
声を作る・書く・置くのは :mod:`sashimono.ai.operations` の同じ関数を通す 窓は選ぶだけ
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.engine.audio.voicevox import (
    SPEED_PERCENT_RANGE,
    TEXT_LIMIT,
    VOLUME_PERCENT_RANGE,
    Catalog,
    EngineError,
    NarrationSettings,
)
from sashimono.ui.voice_fetch import BUSY_TEXT, Fetched, VoiceFetch, fetch_voices, terms_text

__all__ = ["NarrationChoice", "NarrationDialog", "ask_narration"]


@dataclass(frozen=True, slots=True)
class NarrationChoice:
    """窓で選んだ物 generate_narration の引数と同じ名前で渡す"""

    text: str
    voice_id: int
    speed_percent: int
    volume_percent: int
    subtitles: bool

    def arguments(self, at_frame: int) -> dict[str, object]:
        return {
            "text": self.text,
            "voice_id": self.voice_id,
            "speed_percent": self.speed_percent,
            "volume_percent": self.volume_percent,
            "subtitles": self.subtitles,
            "at_frame": at_frame,
        }


class NarrationDialog(QDialog):
    """台本と声を選ぶ 声の一覧はエンジンから読む（読めるまで〔作る〕は押せない）"""

    def __init__(self, settings: NarrationSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ナレーションを作る")
        self._settings = settings
        #: 読んだ声の一覧 読めていなければ ``None``
        self.catalog: Catalog | None = None
        #: エンジンへの問い合わせを裏で進める係
        self._fetch = VoiceFetch(self)

        self._text = QPlainTextEdit(self)
        self._text.setPlaceholderText(
            f"読ませる台本（{TEXT_LIMIT} 文字まで 長い台本は場面ごとに分けて作る）"
        )
        self._text.setMinimumHeight(120)
        self._count = QLabel(self)
        self._voice = QComboBox(self)
        self._voice.setToolTip(
            "声はエンジンの一覧から選ぶ 声ごとに規約が違うので〔規約を見る〕で確かめる"
        )
        read = QPushButton("声の一覧を読む", self)
        read.clicked.connect(self.read_voices)
        terms = QPushButton("規約を見る", self)
        terms.clicked.connect(self._show_terms)
        voice_row = QHBoxLayout()
        voice_row.addWidget(self._voice, 1)
        voice_row.addWidget(read)
        voice_row.addWidget(terms)
        self._status = QLabel(self)
        self._status.setWordWrap(True)

        self._speed = self._spin(SPEED_PERCENT_RANGE, settings.speed_percent)
        self._volume = self._spin(VOLUME_PERCENT_RANGE, settings.volume_percent)
        self._subtitles = QCheckBox("台本から字幕を付ける（文ごとに、声の長さに合わせる）", self)
        self._subtitles.setChecked(settings.subtitles)

        form = QFormLayout()
        form.addRow("台本", self._text)
        form.addRow("", self._count)
        form.addRow("声", voice_row)
        form.addRow("", self._status)
        form.addRow("話す速さ", self._speed)
        form.addRow("声の大きさ", self._volume)
        form.addRow("", self._subtitles)
        note = QLabel(
            "再生ヘッドの位置から、空いた音声トラックへ置く WAV は作品のフォルダの narration へ"
            "書き、取り消しても残る 使った声の表記は書き出しのときに知らせる"
            " 台本は自分の機械の中のエンジンへだけ渡す",
            self,
        )
        note.setWordWrap(True)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("作る")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self._buttons)
        self._text.textChanged.connect(self._update)
        self._update()

    def _spin(self, bounds: tuple[int, int], value: int) -> QSpinBox:
        box = QSpinBox(self)
        box.setRange(*bounds)
        box.setSingleStep(5)
        box.setSuffix(" %")
        box.setValue(value)
        return box

    def set_text(self, text: str) -> None:
        self._text.setPlainText(text)

    def read_voices(self, _checked: bool = False) -> None:
        """エンジンから声の一覧を読む 読めなければ理由を窓の中に書く（別の窓は出さない）

        読むのは裏のスレッド エンジンが返事を止めても窓は固まらず、〔やめる〕を押せる
        """
        self._ask_engine(terms=False)

    def _show_terms(self, _checked: bool = False) -> None:
        """選んだ声の規約とエンジンの規約を、エンジンが返した文のまま見せる

        アプリは中身を判断しない 一覧をまだ読んでいなければ、一緒に読む
        """
        self._ask_engine(terms=True)

    @property
    def reading(self) -> bool:
        """エンジンに問い合わせている途中か"""
        return self._fetch.running

    def _ask_engine(self, *, terms: bool) -> None:
        data = self._voice.currentData()
        voice = int(data) if data is not None else self._settings.voice
        try:
            work = fetch_voices(self._settings.url, self.catalog, voice if terms else None)
        except EngineError as exc:
            self._status.setText(str(exc))
            self._update()
            return
        asked = voice if terms else None
        if self._fetch.start(work, lambda fetched: self._fetched(asked, fetched)):
            self._status.setText("エンジンに問い合わせています…")
        else:
            # 設定の窓と同じく、前の問い合わせの途中で押されたことを黙って捨てない
            self._status.setText(BUSY_TEXT)

    def _fetched(self, asked: int | None, fetched: Fetched) -> None:
        catalog = fetched.catalog
        if catalog is None:
            self._status.setText(fetched.error)
            self._update()
            return
        if catalog is not self.catalog:
            self._fill_voices(catalog)
        if asked is not None:
            # 規約の文は問い合わせた声の物 待つ間に欄を変えても、名前と規約を食い違わせない
            found = catalog.find(asked)
            if found is not None:
                QMessageBox.information(
                    self, "声の規約", terms_text(catalog, *found, fetched.policy or "")
                )

    def _fill_voices(self, catalog: Catalog) -> None:
        self.catalog = catalog
        wanted = self._voice.currentData()
        if wanted is None:
            default = catalog.find(self._settings.voice)
            wanted = default[1].id if default is not None else None
        self._voice.clear()
        for speaker in catalog.speakers:
            for style in speaker.styles:
                self._voice.addItem(f"{speaker.name}（{style.name}）", style.id)
        index = self._voice.findData(wanted)
        self._voice.setCurrentIndex(max(0, index))
        self._status.setText(
            f"{catalog.engine_name or catalog.brand} の声 表記は「{catalog.brand}:声の名前」の形"
        )
        self._update()

    def _update(self) -> None:
        """台本が空か長すぎるとき・声が読めていないときは〔作る〕を押せなくする"""
        length = len(self._text.toPlainText().strip())
        self._count.setText(f"{length} / {TEXT_LIMIT} 文字")
        ready = self.catalog is not None and self._voice.count() > 0
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            ready and 0 < length <= TEXT_LIMIT
        )

    def choice(self) -> NarrationChoice:
        return NarrationChoice(
            text=self._text.toPlainText().strip(),
            voice_id=int(self._voice.currentData()),
            speed_percent=self._speed.value(),
            volume_percent=self._volume.value(),
            subtitles=self._subtitles.isChecked(),
        )


def ask_narration(
    parent: QWidget, settings: NarrationSettings
) -> tuple[NarrationChoice, Catalog] | None:
    """窓を出して台本と声を選んでもらう やめたら ``None`` 試験はここを差し替えて答える"""
    dialog = NarrationDialog(settings, parent)
    try:
        dialog.read_voices()
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.catalog is None:
            return None
        return dialog.choice(), dialog.catalog
    finally:
        dialog.deleteLater()
