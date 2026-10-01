"""字幕パネルと起こしで、字幕を素材と音声ごとに扱う（利用者の要望）

- 音声 1 を起こした後に音声 2 を起こしても、音声 1 の字幕が残る
- AI から 2 本続けて頼むと順番待ちになり、どちらの結果もその素材と音声へ入る
- 窓から起こし直すときは、その音にもう字幕があれば置き換えるかを尋ねる
- 字幕パネルは見る音を選べ、表の中身は選んだ音の字幕
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.asr.backend import Progress, ShouldCancel, TranscribeOptions
from sashimono.asr.service import TranscriptionService
from sashimono.core.commands import Command
from sashimono.core.model import (
    AudioStreamInfo,
    MediaItem,
    Project,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.subtitle.panel import SubtitlePanel
from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog


def _movie(name: str = "録画", voices: int = 2) -> MediaItem:
    return MediaItem(
        path=Path(f"C:/素材/{name}.mp4"),
        duration=Fraction(10),
        video_streams=(VideoStreamInfo(0, 1920, 1080, FrameRate(30), Fraction(1, 15360), "h264"),),
        audio_streams=tuple(
            AudioStreamInfo(index, 48000, 2, Fraction(1, 48000), "aac")
            for index in range(1, voices + 1)
        ),
    )


class _Speaker:
    """起こした音の番号を本文にする代わりの起こし係 ``gate`` が開くまで待つ"""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.order: list[int | None] = []

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def transcribe(
        self,
        path: Path,
        options: TranscribeOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> Transcript | None:
        del progress, should_cancel
        self.gate.wait(5.0)
        self.order.append(options.audio_stream)
        text = f"{path.stem} の音声 {options.audio_stream}"
        return Transcript((TranscriptSegment(Fraction(1), Fraction(2), text),))


class _Host:
    """字幕パネルが頼んだ命令を当てて、プロジェクトを進める"""

    def __init__(self, panel: SubtitlePanel, project: Project) -> None:
        self.project = project
        self.panel = panel
        panel.commands_requested.connect(self._apply)

    def _apply(self, commands: list[Command], _label: str) -> None:
        for command in commands:
            self.project = command.apply(self.project)
        self.panel.set_project(self.project)


@pytest.fixture
def speaker() -> _Speaker:
    return _Speaker()


@pytest.fixture
def setup(
    qt_application: QApplication, speaker: _Speaker
) -> Iterator[tuple[SubtitlePanel, _Host, MediaItem, MediaItem]]:
    del qt_application
    first, second = _movie("録画"), _movie("別の録画")
    project = Project.create(media=(first, second))
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    panel = SubtitlePanel(project, analyzer)
    panel._service = TranscriptionService(speaker)
    host = _Host(panel, project)
    yield panel, host, first, second
    panel.close()
    analyzer.close()


def _texts(project: Project, media: MediaItem, stream: int) -> list[str]:
    transcript = project.require_media(media.id).transcript_for(stream)
    return [] if transcript is None else [s.text for s in transcript.segments]


def _settle(panel: SubtitlePanel) -> str:
    deadline = time.monotonic() + 10.0
    status = panel.poll_transcription()
    while panel._jobs and time.monotonic() < deadline:
        time.sleep(0.02)
        status = panel.poll_transcription()
    return status


class TestFromTheAssistant:
    def test_two_requests_both_land(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 前は起こしを 1 つしか覚えず、2 本目を頼むと 1 本目の結果が捨てられた
        panel, host, first, second = setup
        speaker.gate.clear()
        panel.start_transcription(first.id, "tiny", audio_stream=1)
        message = panel.start_transcription(first.id, "tiny", audio_stream=2)
        panel.start_transcription(second.id, "tiny", audio_stream=1)
        assert "順番待ち" in message
        service = panel._service
        assert service is not None
        deadline = time.monotonic() + 5.0
        while service.jobs()[0].waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        status = panel.poll_transcription()
        assert "順番待ち" in status and "起こしている" in status
        speaker.gate.set()
        _settle(panel)
        assert _texts(host.project, first, 1) == ["録画 の音声 1"]
        assert _texts(host.project, first, 2) == ["録画 の音声 2"]
        assert _texts(host.project, second, 1) == ["別の録画 の音声 1"]
        # 1 本ずつ頼んだ順に走った
        assert speaker.order == [1, 2, 1]

    def test_the_same_voice_twice_is_refused(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        panel, _host, first, _second = setup
        speaker.gate.clear()
        panel.start_transcription(first.id, "tiny", audio_stream=2)
        with pytest.raises(RuntimeError, match="もう起こしています"):
            panel.start_transcription(first.id, "tiny", audio_stream=2)
        speaker.gate.set()
        _settle(panel)

    def test_the_first_voice_is_the_same_however_it_is_written(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 1 本目は「省く（None）」と「番号（1）」の 2 通りで届く 書き方で比べると、同じ音の
        # 起こしが 2 回走り、同じ字幕を 2 回取り込んだ（PR #231 の指摘）
        panel, _host, first, _second = setup
        speaker.gate.clear()
        panel.start_transcription(first.id, "tiny")
        with pytest.raises(RuntimeError, match="もう起こしています"):
            panel.start_transcription(first.id, "tiny", audio_stream=first.audio_streams[0].index)
        # 窓から頼んでも同じ列で見分ける
        dialog = TranscribeDialog(first, panel._service or TranscriptionService(_Speaker()))
        try:
            dialog._stream.setCurrentIndex(0)
            dialog._start_transcribe()
            assert dialog._job is None
            assert "もう起こしています" in dialog._status.text()
        finally:
            dialog.deleteLater()
        speaker.gate.set()
        _settle(panel)


class TestTheWindow:
    def test_redoing_a_voice_asks_before_replacing(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem]
    ) -> None:
        panel, host, first, _second = setup
        panel.start_transcription(first.id, "tiny", audio_stream=1)
        _settle(panel)
        media = host.project.require_media(first.id)
        asked: list[str] = []
        dialog = TranscribeDialog(media, panel._service or TranscriptionService(_Speaker()))
        try:
            dialog._stream.setCurrentIndex(0)

            def refuse(name: str) -> bool:
                asked.append(name)
                return False

            dialog.confirm_replace = refuse
            dialog._start_transcribe()
            # 断ったので起こさない
            assert asked and dialog._job is None
            # まだ字幕の無い音声 2 は尋ねずに起こす
            asked.clear()
            dialog._stream.setCurrentIndex(1)
            dialog._start_transcribe()
            assert asked == [] and dialog._job is not None
            assert dialog.chosen_stream == 2
            assert dialog._job.wait(5.0)
        finally:
            dialog.deleteLater()


class TestCancellingWhileWaiting:
    def test_the_window_closes_at_once(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 前は順番待ちのまま〔中断〕を押すと、前の起こしが終わる（数分）まで窓が閉じなかった
        panel, _host, first, _second = setup
        speaker.gate.clear()
        panel.start_transcription(first.id, "tiny", audio_stream=1)
        service = panel._service
        assert service is not None
        dialog = TranscribeDialog(first, service)
        try:
            dialog._stream.setCurrentIndex(1)
            dialog._start_transcribe()
            assert dialog._job is not None and dialog._job.waiting
            dialog.reject()
            assert dialog.result() == TranscribeDialog.DialogCode.Rejected
            assert dialog._job is None
        finally:
            dialog.deleteLater()
        speaker.gate.set()
        _settle(panel)
        deadline = time.monotonic() + 5.0
        while service.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        # 止めた依頼は走らない（音声 2 は起こされない）
        assert speaker.order == [1]


class TestThePanel:
    def test_the_panel_shows_the_chosen_voice(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem]
    ) -> None:
        panel, _host, first, _second = setup
        panel.start_transcription(first.id, "tiny", audio_stream=1)
        panel.start_transcription(first.id, "tiny", audio_stream=2)
        _settle(panel)
        panel.select_media(first.id)
        assert not panel._stream_box.isHidden()
        panel.select_stream(1)
        assert [s.text for s in panel._segments()] == ["録画 の音声 1"]
        panel.select_stream(2)
        assert [s.text for s in panel._segments()] == ["録画 の音声 2"]

    def test_a_single_voice_shows_no_choice(self, qt_application: QApplication) -> None:
        del qt_application
        single = _movie("一本", voices=1)
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        panel = SubtitlePanel(Project.create(media=(single,)), analyzer)
        try:
            assert panel._stream_box.isHidden()
        finally:
            panel.close()
            analyzer.close()
