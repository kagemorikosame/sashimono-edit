"""字幕パネルと起こしで、字幕を素材と音声ごとに扱う（利用者の要望）

- 音声 1 を起こした後に音声 2 を起こしても、音声 1 の字幕が残る
- AI から 2 本続けて頼むと順番待ちになり、どちらの結果もその素材と音声へ入る
- 窓から起こし直すときは、その音にもう字幕があれば置き換えるかを尋ねる
- 字幕パネルは見る音を選べ、表の中身は選んだ音の字幕
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.asr.backend import Progress, ShouldCancel, TranscribeOptions
from sashimono.asr.service import TranscriptionService
from sashimono.core.commands import Command, SetTranscript
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
    Word,
)
from sashimono.core.subtitle_timing import SubtitleTiming
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.subtitle.panel import SubtitlePanel
from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.waiting import wait_until


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
        #: 頼まれた語の時刻 順に
        self.words: list[bool] = []
        #: 語の時刻の付いた結果を返す
        self.with_words = False
        #: 頼まれた字幕の時刻の合わせ方 順に（#426）
        self.timings: list[SubtitleTiming] = []

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
        self.words.append(options.word_timestamps)
        self.timings.append(options.timing)
        text = f"{path.stem} の音声 {options.audio_stream}"
        words = (Word(Fraction(1), Fraction(2), "語"),) if self.with_words else ()
        return Transcript((TranscriptSegment(Fraction(1), Fraction(2), text, words),))


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
    """字幕起こしの依頼が全部片付くまで見張りを回し、最後の様子を返す

    片付く時間は機械の速さで変わる 秒で区切らず、片付くのを待つ（#405 tests/waiting.py）
    """
    status = [panel.poll_transcription()]

    def poll() -> None:
        status[0] = panel.poll_transcription()

    wait_until(lambda: not panel._jobs, "字幕起こし", step=poll, pause=0.02)
    return status[0]


class TestWordTimes:
    """語の時刻（#420）を本人の設定どおりに頼み、字幕の文言を守る"""

    def test_the_preference_reaches_the_request(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        """切っても語の時刻を取ると、設定がある方が質が悪い"""
        panel, _host, first, second = setup
        panel.start_transcription(first.id, "tiny", word_timestamps=True)
        panel.start_transcription(second.id, "tiny", word_timestamps=False)
        _settle(panel)
        assert speaker.words == [True, False]

    def test_words_only_asks_for_words_even_when_turned_off(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 語の時刻を足す頼みで語の時刻を取らないと、字幕の語を全部消す
        panel, _host, first, _second = setup
        panel.start_transcription(first.id, "tiny", word_timestamps=False, words_only=True)
        _settle(panel)
        assert speaker.words == [True]

    def test_words_only_keeps_the_subtitles(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        """壊れると、語の時刻のためだけに人が直した文言が消える"""
        panel, host, first, _second = setup
        speaker.with_words = True
        line = TranscriptSegment(Fraction(1), Fraction(2), "人が直した文")
        host._apply([SetTranscript(first.id, Transcript((line,)), stream=1)], "字幕")
        panel.start_transcription(first.id, "tiny", audio_stream=1, words_only=True)
        _settle(panel)
        assert _texts(host.project, first, 1) == ["人が直した文"]
        transcript = host.project.require_media(first.id).transcript_for(1)
        assert transcript is not None
        assert [word.text for word in transcript.segments[0].words] == ["語"]

    def test_words_only_without_subtitles_makes_them(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 壊れると、字幕の無い素材に頼んだ起こしの結果を捨てる
        panel, host, first, _second = setup
        speaker.with_words = True
        panel.start_transcription(first.id, "tiny", audio_stream=1, words_only=True)
        _settle(panel)
        assert _texts(host.project, first, 1) == ["録画 の音声 1"]

    def test_the_setting_is_on_the_settings_screen(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """壊れると、起こす時間と語の時刻の釣り合いを本人が選べず、決め打ちの動きになる"""
        del qt_application
        assert Preferences().word_timestamps is True
        chosen = Preferences(word_timestamps=False)
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences().word_timestamps is False
        finally:
            dialog.deleteLater()
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(chosen)
        assert store.load().word_timestamps is False
        # 壊れた値で起動を止めない 既定（入り）へ戻す
        store.path.write_text(json.dumps({"word_timestamps": "壊れた"}), "utf-8")
        assert store.load().word_timestamps is True

    def test_the_dialog_starts_from_the_setting(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 壊れると、窓から起こす人に設定が効かない
        _panel, _host, first, _second = setup
        service = TranscriptionService(speaker)
        for wanted in (True, False):
            dialog = TranscribeDialog(first, service, words=wanted)
            assert dialog._words.isChecked() is wanted
            dialog.deleteLater()


class TestSubtitleTiming:
    """起こした字幕の時刻の合わせ方（#426）を本人の設定どおりに頼む"""

    def test_the_preference_reaches_the_request(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        """壊れると、設定で選んだ「少し早く出す」が AI から頼んだ起こしに効かない"""
        panel, _host, first, second = setup
        chosen = SubtitleTiming(fit_to_voice=False, start_offset_ms=-200, end_offset_ms=300)
        panel.timing = chosen
        panel.start_transcription(first.id, "tiny")
        panel.start_transcription(second.id, "tiny", timing=SubtitleTiming())
        _settle(panel)
        assert speaker.timings == [chosen, SubtitleTiming()]

    def test_adding_words_leaves_the_times_alone(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 語の時刻だけを足す頼みで行を寄せると、人が合わせた行の時刻まで動いたように見える
        panel, _host, first, _second = setup
        panel.start_transcription(first.id, "tiny", words_only=True, timing=SubtitleTiming())
        _settle(panel)
        assert speaker.timings == [SubtitleTiming(fit_to_voice=False)]

    def test_the_setting_is_on_the_settings_screen(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """壊れると、寄せ方が合わない素材の人と、字幕を早めに出したい人が選べない"""
        del qt_application
        assert Preferences().subtitle_timing == SubtitleTiming()
        chosen = Preferences(
            subtitle_fit_voice=False, subtitle_start_offset_ms=-250, subtitle_end_offset_ms=400
        )
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences().subtitle_timing == chosen.subtitle_timing
        finally:
            dialog.deleteLater()
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(chosen)
        assert store.load().subtitle_timing == chosen.subtitle_timing
        # 壊れた値と範囲の外で起動を止めない 項目ごとに既定へ戻す
        store.path.write_text(
            json.dumps(
                {
                    "subtitle_fit_voice": "壊れた",
                    "subtitle_start_offset_ms": -99999,
                    "subtitle_end_offset_ms": 1.5,
                }
            ),
            "utf-8",
        )
        assert store.load().subtitle_timing == SubtitleTiming()

    def test_the_dialog_starts_from_the_setting(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 壊れると、窓から起こす人に設定が効かず、窓で変えた値も起こしへ届かない
        _panel, _host, first, _second = setup
        service = TranscriptionService(speaker)
        for wanted in (SubtitleTiming(), SubtitleTiming(False, -100, 600)):
            dialog = TranscribeDialog(first, service, timing=wanted)
            assert dialog.timing() == wanted
            dialog.deleteLater()


class TestFromTheAssistant:
    def test_the_waiter_opens_when_the_queue_is_empty(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 壊れると、AI は起こしの終わりを待てず、別の道具を時計代わりに呼び続ける（#413）
        panel, host, first, _second = setup
        waiter = panel.transcription_waiter()
        assert waiter(0.0)
        speaker.gate.clear()
        panel.start_transcription(first.id, "tiny", audio_stream=1)
        assert panel.transcribing
        assert not waiter(0.0)
        speaker.gate.set()
        _settle(panel)
        assert waiter(0.0)
        # 知らせが立った時には結果が作品へ入っている
        assert _texts(host.project, first, 1) == ["録画 の音声 1"]

    def test_releasing_lets_the_waiter_go(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem], speaker: _Speaker
    ) -> None:
        # 壊れると、放しても待ちが続く 窓を閉じる道は tests/ui/test_ripple_playhead.py が見る
        panel, _host, first, _second = setup
        speaker.gate.clear()
        panel.start_transcription(first.id, "tiny", audio_stream=1)
        waiter = panel.transcription_waiter()
        assert not waiter(0.0)
        panel.release_transcription_waiters()
        assert waiter(0.0)
        speaker.gate.set()
        _settle(panel)

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
        wait_until(lambda: not service.jobs()[0].waiting, "1 本目の起こし始め", pause=0.01)
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
        # 同じ音声の起こしを 2 回受け付けると、同じ字幕が 2 回取り込まれ、待つ時間も倍になる
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
        # 尋ねずに起こし直すと、手で直した字幕が新しい起こしで黙って上書きされて消える
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
        wait_until(lambda: not service.busy, "起こしの係が空くの", pause=0.02)
        # 止めた依頼は走らない（音声 2 は起こされない）
        assert speaker.order == [1]


class TestThePanel:
    def test_the_panel_shows_the_chosen_voice(
        self, setup: tuple[SubtitlePanel, _Host, MediaItem, MediaItem]
    ) -> None:
        # 選んだ音声の字幕が出ないと、2 本目の音声の字幕を直そうとしても 1 本目の字幕しか見えない
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

    @pytest.mark.parametrize("kind", [TrackKind.AUDIO, TrackKind.MIXED])
    def test_changing_the_clips_voice_redoes_the_list(
        self, qt_application: QApplication, kind: TrackKind
    ) -> None:
        # 前は一覧の印に鳴らす音の番号が入っておらず、クリップの音を替えても作り直さず、
        # 時刻が前の音の物（ここでは出ていない「—」）のまま残った（PR #231 の指摘）
        del qt_application
        said = Transcript((TranscriptSegment(Fraction(1), Fraction(2), "声"),))
        media = _movie().with_transcript(said, 1).with_transcript(said, 2)

        def placed(voice: int) -> Project:
            if kind is TrackKind.AUDIO:
                clip = Clip(timeline_start=0, duration=300, media_id=media.id, stream_index=voice)
            else:
                clip = Clip(timeline_start=0, duration=300, media_id=media.id, audio_stream=voice)
            base = Project.create(media=(media,))
            track = Track(kind, "音", (clip,))
            return base.with_timeline(replace(base.timeline, tracks=(track,)))

        first = placed(1)
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        panel = SubtitlePanel(first, analyzer)
        try:
            panel.select_media(media.id)
            panel.select_stream(2)
            assert [start for _, start, _ in panel._rows] == [-1]
            clip = first.timeline.tracks[0].clips[0]
            changed = (
                replace(clip, stream_index=2)
                if kind is TrackKind.AUDIO
                else replace(clip, audio_stream=2)
            )
            track = replace(first.timeline.tracks[0], clips=(changed,))
            panel.set_project(first.with_timeline(replace(first.timeline, tracks=(track,))))
            assert [start for _, start, _ in panel._rows] == [30]
        finally:
            panel.close()
            analyzer.close()

    def test_moving_the_clip_to_a_picture_track_redoes_the_list(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、音声 2 の音のクリップを映像トラックへ移して音が 1 本目の扱いになっても、
        # 一覧の印にトラックの種類が入っておらず、音声 2 の字幕の時刻が残った（PR #231）
        del qt_application
        said = Transcript((TranscriptSegment(Fraction(1), Fraction(2), "声"),))
        media = _movie().with_transcript(said, 1).with_transcript(said, 2)
        clip = Clip(timeline_start=0, duration=300, media_id=media.id, stream_index=2)
        base = Project.create(media=(media,))
        sound = Track(TrackKind.AUDIO, "A1", (clip,))
        first = base.with_timeline(replace(base.timeline, tracks=(sound,)))
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        panel = SubtitlePanel(first, analyzer)
        try:
            panel.select_media(media.id)
            panel.select_stream(2)
            assert [start for _, start, _ in panel._rows] == [30]
            moved = replace(sound, kind=TrackKind.VIDEO)
            panel.set_project(first.with_timeline(replace(first.timeline, tracks=(moved,))))
            assert [start for _, start, _ in panel._rows] == [-1]
        finally:
            panel.close()
            analyzer.close()

    def test_splitting_takes_the_time_from_the_voice_on_show(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、リンクを外して音声 2 だけを 2 秒切り詰めたとき、再生位置で最初に当たる
        # 音声 1 のクリップの時刻（1 秒）で音声 2 の字幕を割った 正しくは 3 秒（PR #231）
        del qt_application
        said = Transcript((TranscriptSegment(Fraction(0), Fraction(5), "長い声"),))
        media = _movie().with_transcript(said, 1).with_transcript(said, 2)
        voice1 = Clip(timeline_start=0, duration=150, media_id=media.id, stream_index=1)
        voice2 = Clip(
            timeline_start=0,
            duration=90,
            media_id=media.id,
            stream_index=2,
            source_in=Fraction(2),
        )
        base = Project.create(media=(media,))
        tracks = (
            Track(TrackKind.AUDIO, "A1", (voice1,)),
            Track(TrackKind.AUDIO, "A2", (voice2,)),
        )
        project = base.with_timeline(replace(base.timeline, tracks=tracks))
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        panel = SubtitlePanel(project, analyzer)
        try:
            panel.select_media(media.id)
            panel.select_stream(2)
            assert panel._source_time(media, 30) == Fraction(3)
            panel.select_stream(1)
            assert panel._source_time(media, 30) == Fraction(1)
        finally:
            panel.close()
            analyzer.close()

    def test_a_single_voice_shows_no_choice(self, qt_application: QApplication) -> None:
        # 音声が 1 本の素材に選ぶ欄を出すと、選ぶ物の無い欄が場所を取り、何を選ぶのか迷わせる
        del qt_application
        single = _movie("一本", voices=1)
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        panel = SubtitlePanel(Project.create(media=(single,)), analyzer)
        try:
            assert panel._stream_box.isHidden()
        finally:
            panel.close()
            analyzer.close()
