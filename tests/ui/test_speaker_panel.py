"""字幕パネルの話者の分離（#323）

- 〔話者を分ける…〕で分けた結果が、見ている素材と音声の字幕へ付き、取り消し 1 回で外れる
- 話者の列は話者の付いた字幕があるときだけ出て、書き換えるとその行の話者が変わる
- 右クリックの〔話者の名前を変える…〕は、同じ名前の行をすべて付け替える
- AI からの依頼は起こしと同じ列で走り、終わった時の字幕へ付く
- 起こし終えたら続けて分けるのは、本人が設定で入れたときだけ（既定は切）

窓が開いたらその場で落とす 部品（sherpa-onnx）は使わず、代わりの分離係で通す
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox

from sashimono.asr.backend import Progress, ShouldCancel
from sashimono.asr.service import TranscriptionService
from sashimono.asr.speakers import SPEAKER_PACK, SpeakerOptions
from sashimono.core.commands import Command
from sashimono.core.model import MediaItem, Project, Transcript, TranscriptSegment
from sashimono.core.speakers import SpeakerTurn
from sashimono.engine.cache import MediaAnalyzer
from sashimono.runtime import PackageStatus, PackStatus
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.subtitle import panel as panel_module
from sashimono.ui.subtitle.panel import SPEAKER_COLUMN, SubtitlePanel
from sashimono.ui.subtitle.speaker_dialog import SpeakerDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.asr.test_service import FakeBackend

#: 2 人の対談 0〜3.5 秒が 1 人目、3.5〜10 秒が 2 人目
TURNS = (
    SpeakerTurn(Fraction(0), Fraction(7, 2), 4),
    SpeakerTurn(Fraction(7, 2), Fraction(10), 9),
)


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 開く窓は試験の中で差し替える"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QInputDialog, "getText", popped)


class _Speakers:
    """代わりの分離係 頼まれた条件を覚え、決まった区間を返す"""

    def __init__(self) -> None:
        self.asked: list[SpeakerOptions] = []
        # 返す区間 空にすると、無音や見分けられなかった回の部品の答えになる
        self.result: tuple[SpeakerTurn, ...] = TURNS
        self.gate = threading.Event()
        self.gate.set()

    def is_available(self) -> bool:
        return True

    def separate(
        self,
        path: Path,
        options: SpeakerOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> tuple[SpeakerTurn, ...] | None:
        del path, progress, should_cancel
        self.gate.wait(5.0)
        self.asked.append(options)
        return self.result


class _Host:
    """字幕パネルが頼んだ命令を当て、取り消しのために前の形を積む"""

    def __init__(self, panel: SubtitlePanel, project: Project) -> None:
        self.project = project
        self.panel = panel
        self.history: list[Project] = []
        self.labels: list[str] = []
        panel.commands_requested.connect(self._apply)

    def _apply(self, commands: list[Command], label: str) -> None:
        self.history.append(self.project)
        self.labels.append(label)
        for command in commands:
            self.project = command.apply(self.project)
        self.panel.set_project(self.project)

    def undo(self) -> None:
        self.project = self.history.pop()
        self.panel.set_project(self.project)


def _media() -> MediaItem:
    from sashimono.core.model import AudioStreamInfo

    return MediaItem(
        path=Path("C:/素材/対談.wav"),
        duration=Fraction(10),
        audio_streams=(AudioStreamInfo(0, 48000, 1, Fraction(1, 48000), "pcm_s16le"),),
    )


TRANSCRIPT = Transcript(
    (
        TranscriptSegment(Fraction(1), Fraction(3), "こんにちは"),
        TranscriptSegment(Fraction(4), Fraction(6), "よろしく"),
        TranscriptSegment(Fraction(7), Fraction(9), "はじめます"),
    )
)


@pytest.fixture
def setup(
    qt_application: QApplication,
) -> Iterator[tuple[SubtitlePanel, _Host, MediaItem, _Speakers]]:
    del qt_application
    media = _media().with_transcript(TRANSCRIPT)
    project = Project.create(media=(media,))
    analyzer = MediaAnalyzer(sample_rate=48000, channels=1)
    panel = SubtitlePanel(project, analyzer)
    speakers = _Speakers()
    panel._service = TranscriptionService(FakeBackend(), speakers)
    panel.speakers_ready = lambda: True
    host = _Host(panel, project)
    yield panel, host, media, speakers
    panel.close()
    analyzer.close()


def _speakers_of(host: _Host, media: MediaItem) -> list[str | None]:
    transcript = host.project.require_media(media.id).transcript
    return [] if transcript is None else [s.speaker for s in transcript]


def _settle(panel: SubtitlePanel) -> str:
    deadline = time.monotonic() + 10.0
    status = panel.poll_transcription()
    while panel._jobs and time.monotonic() < deadline:
        time.sleep(0.02)
        status = panel.poll_transcription()
    return status


class TestTheButton:
    def test_separated_speakers_land_on_the_rows_and_undo_once(
        self,
        setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """壊れると、窓で分けても字幕に話者が付かないか、取り消し 1 回で外れない"""
        panel, host, media, _speakers = setup
        shown: list[str] = []
        panel.status_message.connect(shown.append)

        class Done:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                self.turns = TURNS
                self.chosen_stream = media.transcript_stream(None)

            def exec(self) -> int:
                return 1

            def deleteLater(self) -> None:  # noqa: N802 - Qt の名前に合わせる
                return None

        monkeypatch.setattr(panel_module, "SpeakerDialog", Done)
        assert panel._speakers_button.isEnabled()
        panel.separate_speakers()
        assert _speakers_of(host, media) == ["話者 1", "話者 2", "話者 2"]
        assert host.labels[-1] == f"話者を分ける: {media.name}"
        assert shown[-1].startswith("2 人に分けました")
        assert not panel._table.isColumnHidden(SPEAKER_COLUMN)
        host.undo()
        assert _speakers_of(host, media) == [None, None, None]
        assert panel._table.isColumnHidden(SPEAKER_COLUMN)

    def test_without_subtitles_the_button_is_off(self, qt_application: QApplication) -> None:
        """壊れると、字幕の無い素材で窓が開き、数分分けた末に付ける字幕が無い"""
        del qt_application
        project = Project.create(media=(_media(),))
        analyzer = MediaAnalyzer(sample_rate=48000, channels=1)
        panel = SubtitlePanel(project, analyzer)
        try:
            assert not panel._speakers_button.isEnabled()
        finally:
            panel.close()
            analyzer.close()


class TestFixingByHand:
    def test_editing_the_column_changes_one_row(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、分離が取り違えた 1 行を表で直せず、全員の名前を変えるしかない"""
        panel, host, media, _speakers = setup
        panel._apply_speakers(media.id, None, TURNS)
        item = panel._table.item(0, SPEAKER_COLUMN)
        assert item is not None and item.text() == "話者 1"
        item.setText("話者 2")
        assert _speakers_of(host, media) == ["話者 2", "話者 2", "話者 2"]
        # 本文の列は今までどおり本文を直す（話者の列を足して列の番号がずれていない）
        text = panel._table.item(1, 1)
        assert text is not None
        text.setText("よろしくね")
        transcript = host.project.require_media(media.id).transcript
        assert transcript is not None and transcript.segments[1].text == "よろしくね"

    def test_renaming_from_the_menu_renames_every_row(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、「話者 2」を人の名前に替えても 1 行しか変わらず、焼き込みのレイヤーが割れる"""
        panel, host, media, _speakers = setup
        panel._apply_speakers(media.id, None, TURNS)
        asked: list[str] = []

        def answer(old: str) -> str:
            asked.append(old)
            return "  佐藤  "

        panel.ask_speaker_name = answer
        panel.rename_speaker("話者 2")
        assert asked == ["話者 2"]
        assert _speakers_of(host, media) == ["話者 1", "佐藤", "佐藤"]
        # やめたときと同じ名前のときは何もしない（履歴に空の段を積まない）
        before = len(host.labels)
        panel.ask_speaker_name = lambda _old: None
        panel.rename_speaker("佐藤")
        panel.ask_speaker_name = lambda old: old
        panel.rename_speaker("佐藤")
        assert len(host.labels) == before


class TestFromTheAssistant:
    def test_a_request_runs_in_the_queue_and_lands(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、AI が分けるよう頼んでも結果が字幕へ入らず、様子にも出ない"""
        panel, host, media, speakers = setup
        message = panel.start_speaker_separation(media.id, speakers=2)
        assert "話者の分離を始めました" in message
        status = _settle(panel)
        assert speakers.asked[0].speakers == 2
        assert _speakers_of(host, media) == ["話者 1", "話者 2", "話者 2"]
        assert "2 人に分けました" in status

    def test_refusals_say_what_to_do(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、部品の無い人や字幕の無い素材で、AI が理由も分からず待ち続ける"""
        panel, _host, media, speakers = setup
        speakers.gate.clear()
        panel.start_speaker_separation(media.id)
        with pytest.raises(RuntimeError, match="もう分けています"):
            panel.start_speaker_separation(media.id)
        speakers.gate.set()
        _settle(panel)
        panel.speakers_ready = lambda: False
        with pytest.raises(RuntimeError, match="部品"):
            panel.start_speaker_separation(media.id)

    def test_results_go_onto_the_rows_as_they_are_when_done(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、分けている間に人が直した本文が、頼んだ時の字幕で上書きされて消える"""
        panel, host, media, speakers = setup
        speakers.gate.clear()
        panel.start_speaker_separation(media.id)
        transcript = host.project.require_media(media.id).transcript
        assert transcript is not None
        from sashimono.core.commands import SetSegmentText

        panel.commands_requested.emit(
            [SetSegmentText(media.id, transcript.segments[0].id, "こんばんは")], "字幕を編集"
        )
        speakers.gate.set()
        _settle(panel)
        after = host.project.require_media(media.id).transcript
        assert after is not None
        assert after.segments[0].text == "こんばんは" and after.segments[0].speaker == "話者 1"

    def test_a_run_that_finds_nobody_keeps_the_names_given_before(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、AI が分け直して 1 人も見分けられなかった回に、前の回の話者と人が
        付け替えた名前が全部消え、知らせは「付けていません」と言うので消えたことにも気付けない
        """
        panel, host, media, speakers = setup
        panel._apply_speakers(media.id, None, TURNS)
        panel.ask_speaker_name = lambda _old: "佐藤"
        panel.rename_speaker("話者 2")
        before = len(host.labels)
        speakers.result = ()
        panel.start_speaker_separation(media.id)
        status = _settle(panel)
        assert _speakers_of(host, media) == ["話者 1", "佐藤", "佐藤"]
        assert len(host.labels) == before
        assert "字幕は変えていません" in status


class TestFindingNobody:
    def test_the_window_result_with_nobody_leaves_the_rows_alone(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、窓で分け直して見分けられなかった回に、前の回の話者が全部外れ、
        取り消しの履歴にも中身の無い段が積まれる
        """
        panel, host, media, _speakers = setup
        panel._apply_speakers(media.id, None, TURNS)
        before = len(host.labels)
        message = panel._apply_speakers(media.id, None, ())
        assert message == "話している人を見分けられませんでした（字幕は変えていません）"
        assert _speakers_of(host, media) == ["話者 1", "話者 2", "話者 2"]
        assert len(host.labels) == before


class TestContinuingAfterTranscription:
    def _transcribe(self, panel: SubtitlePanel, media: MediaItem) -> None:
        panel._service = TranscriptionService(
            FakeBackend(result=TRANSCRIPT),
            panel._service.speaker_backend if panel._service else None,
        )
        panel.start_transcription(media.id, "tiny")

    def test_off_by_default_no_separation_runs(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、部品を入れていない人・1 人で話す人まで起こしのたびに分離を待たされる"""
        panel, host, media, speakers = setup
        assert Preferences().speakers_after_transcribe is False
        self._transcribe(panel, media)
        _settle(panel)
        assert speakers.asked == []
        assert _speakers_of(host, media) == [None, None, None]

    def test_when_turned_on_speakers_follow_the_transcription(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、設定で入れても起こした字幕に話者が付かず、毎回ボタンを押すことになる"""
        panel, host, media, speakers = setup
        panel.speakers_after_transcribe = True
        self._transcribe(panel, media)
        _settle(panel)
        _settle(panel)
        assert len(speakers.asked) == 1
        assert _speakers_of(host, media) == ["話者 1", "話者 2", "話者 2"]

    def test_turned_on_without_the_part_says_so(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、設定を入れたのに部品が無いことに気付けず、話者が付かない理由が分からない"""
        panel, _host, media, speakers = setup
        shown: list[str] = []
        panel.status_message.connect(shown.append)
        panel.speakers_after_transcribe = True
        panel.speakers_ready = lambda: False
        self._transcribe(panel, media)
        _settle(panel)
        assert speakers.asked == []
        assert any("続けて話者を分けられませんでした" in line for line in shown)

    def test_the_setting_is_on_the_settings_screen(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """壊れると、続けて分けるかを本人が切り替えられず、決め打ちの動きになる"""
        del qt_application
        chosen = Preferences(speakers_after_transcribe=True)
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences().speakers_after_transcribe is True
        finally:
            dialog.deleteLater()
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(chosen)
        assert store.load().speakers_after_transcribe is True
        # 壊れた値で起動を止めない 既定（切）へ戻す
        store.path.write_text(json.dumps({"speakers_after_transcribe": "壊れた"}), "utf-8")
        assert store.load().speakers_after_transcribe is False


def _status(installed: bool) -> PackStatus:
    version = "1.13.8" if installed else None
    return PackStatus(
        pack=SPEAKER_PACK,
        packages=tuple(PackageStatus(name, version) for name in SPEAKER_PACK.required),
    )


class TestTheWindow:
    def test_without_the_part_it_offers_to_install_first(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、部品の無い人に〔分ける〕が押せて落ちるか、何をどれだけ落とすか分からない"""
        panel, _host, media, _speakers = setup
        assert panel._service is not None
        dialog = SpeakerDialog(media, panel._service, status=lambda: _status(False))
        try:
            assert not dialog._run_button.isEnabled()
            assert dialog._install_button.text() == "部品を入れる"
            assert "sherpa-onnx" in dialog._status.text() and "MB" in dialog._status.text()
        finally:
            dialog.deleteLater()

    def test_the_chosen_count_reaches_the_queue_and_the_turns_come_back(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, _Speakers]
    ) -> None:
        """壊れると、窓で 3 人を選んでも人数が渡らないか、分け終えても結果が呼ぶ側へ届かない"""
        panel, _host, media, speakers = setup
        assert panel._service is not None
        dialog = SpeakerDialog(media, panel._service, status=lambda: _status(True))
        accepted: list[bool] = []
        dialog.accepted.connect(lambda: accepted.append(True))
        try:
            assert dialog._run_button.isEnabled()
            dialog._count.setCurrentIndex(dialog._count.findData(3))
            dialog._start()
            deadline = time.monotonic() + 10.0
            while dialog.turns is None and time.monotonic() < deadline:
                time.sleep(0.02)
                dialog._poll()
            assert speakers.asked[0].speakers == 3
            assert dialog.turns == TURNS and accepted
        finally:
            dialog.deleteLater()
