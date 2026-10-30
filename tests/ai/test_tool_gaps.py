"""通しの確かめで分かった AI の道具の問題（#413）

実際の素材で AI に編集を通して分かった物 手数が増える・誤った判断へ誘う所を道具の側で直す
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sashimono.ai.fixes import ASK, AUTO, fix_for
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Pending, find_operation
from sashimono.core.commands import AddClip, AddTrack, RemoveClip, RippleCut, SetTrackState
from sashimono.core.commands.insert import insert_generated, insert_media
from sashimono.core.mixing import db_to_ratio
from sashimono.core.model import (
    Clip,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    SegmentId,
    SubtitleOrigin,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.effects.sources import TEXT
from sashimono.engine.audio.waveform import PeakLevel, Waveform
from sashimono.engine.checks import CheckRequest, CheckState
from sashimono.engine.checks.profile import SoundProfile, SoundProfiler
from sashimono.engine.checks.report import HIGH, MEDIUM, assemble
from sashimono.engine.checks.scan import LOUDNESS
from tests.ai.conftest import RATE_30, FakeHost, make_loaded
from tests.ai.test_audio_tools import _voiced
from tests.ai.test_mix_tools import _measured, _Work
from tests.check_media import RATE
from tests.conftest import make_clip
from tests.ui.test_subtitle_panel import make_waveform


def _run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None
    return operation(host, arguments)


@pytest.fixture
def work(transcript: Transcript, video_media: MediaItem) -> _Work:
    return _Work(transcript, video_media)


class TestNormalizeTellsTheTruth:
    def test_a_ceiling_at_the_peak_is_not_called_already_there(self, work: _Work) -> None:
        """通しの記録の再現 -20 LUFS で山が天井にあると、1 dB も上げられない

        壊れると「すでに基準に合っています」と返り、基準に届いていないのに AI は直ったと読む
        """
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, -20.3, -1.0)
        result = _run(work.host, "normalize_loudness", wait_seconds=0)
        assert result["gain_db"] == 0.0
        assert result["limited_by"] == ["peak"]
        assert result["reached"] is False
        assert result["short_of_target_lu"] == pytest.approx(6.3)
        assert "すでに" not in result["note"]

    def test_a_small_rise_also_says_how_far_it_stopped(self, work: _Work) -> None:
        """壊れると、途中で止まったときの足りない量が AI に見えず、また同じ道具を呼ぶ"""
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, -31.3, -12.0)
        result = _run(work.host, "normalize_loudness", wait_seconds=0)
        assert result["limited_by"] == ["peak"]
        assert result["reached"] is False
        assert result["short_of_target_lu"] == pytest.approx(6.3)

    def test_a_loudness_on_target_is_still_on_target(self, work: _Work) -> None:
        """壊れると、基準に合っている作品にも止まった訳を返す"""
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, -14.0, -1.0)
        result = _run(work.host, "normalize_loudness", wait_seconds=0)
        assert result["reached"] is True
        assert "limited_by" not in result
        assert "すでに基準に合っています" in result["note"]


class TestCheckStopsRecommendingTheSameTool:
    @pytest.mark.parametrize("severity", [HIGH, MEDIUM])
    def test_a_quiet_work_at_the_ceiling_goes_to_the_person(self, severity: str) -> None:
        """壊れると、上げられない音量を AI が同じ道具で直そうとして回数を使い切る"""
        measure = {"lufs": -20.3, "target_lufs": -14.0, "difference_lu": -6.3}
        fix = fix_for(LOUDNESS, severity, measure, peak_db=-1.0)
        assert fix.action == ASK
        assert "set_volume" in fix.tools

    def test_room_left_stays_automatic(self) -> None:
        """壊れると、まだ上げられる作品まで人に尋ねる"""
        measure = {"lufs": -20.3, "target_lufs": -14.0, "difference_lu": -6.3}
        assert fix_for(LOUDNESS, HIGH, measure, peak_db=-6.0).action == AUTO

    def test_a_loud_work_stays_automatic(self) -> None:
        """壊れると、下げれば直る大きすぎる作品まで人に尋ねる"""
        measure = {"lufs": -8.0, "target_lufs": -14.0, "difference_lu": 6.0}
        assert fix_for(LOUDNESS, HIGH, measure, peak_db=-1.0).action == AUTO

    def test_the_check_result_uses_the_measured_peak(self, work: _Work) -> None:
        """壊れると、表だけ直って check_output は auto を返し続ける"""
        _run(work.host, "check_output", wait_seconds=0)
        request: CheckRequest = work.host.check_requests[-1]
        end = request.end or 300
        report = assemble(
            [],
            start=request.start,
            end=end,
            analyzed=end,
            fps=30.0,
            complete=True,
            limits=request.limits,
            index=None,
            loudness=-20.3,
            peak=db_to_ratio(-1.0),
            length_seconds=None,
        )
        work.host.check = CheckState(request=request, report=report)
        result = _run(work.host, "check_output", wait_seconds=0)
        rows = [row for row in result["problems"] if row["kind"] == LOUDNESS]
        assert rows[0]["fix"]["action"] == ASK
        assert result["summary"]["fixes"]["auto"] == 0


def _empty(media: MediaItem) -> FakeHost:
    project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(media,))
    return FakeHost(project)


class TestPlaceParts:
    def test_parts_of_a_long_media_line_up(self, video_media: MediaItem) -> None:
        """壊れると、長い素材から数か所を取るのに分割を何十回も呼ぶ"""
        host = _empty(video_media)
        result = _run(
            host,
            "place_media",
            media_id=str(video_media.id),
            ranges=[
                {"start_seconds": 1, "end_seconds": 2},
                {"start_seconds": 5, "end_seconds": 6.5},
            ],
        )
        tracks = host.project.timeline.tracks
        clips = sorted(
            (clip for track in tracks for clip in track.clips),
            key=lambda clip: (clip.timeline_start, str(clip.stream_index)),
        )
        assert sorted({(c.timeline_start, c.duration, c.source_in) for c in clips}) == [
            (0, 30, Fraction(1)),
            (30, 45, Fraction(5)),
        ]
        assert [part["start"] for part in result["clips"]] == [0, 30]
        assert result["end_frame"] == 75

    def test_one_undo_takes_them_all_back(self, video_media: MediaItem) -> None:
        """壊れると、置いた数か所を 1 本ずつ戻すことになる"""
        host = _empty(video_media)
        before = host.project
        _run(
            host,
            "place_media",
            media_id=str(video_media.id),
            ranges=[
                {"start_seconds": 1, "end_seconds": 2},
                {"start_seconds": 5, "end_seconds": 6},
            ],
        )
        host.undo()
        assert host.project == before

    def test_the_picture_and_sound_stay_linked(self, video_media: MediaItem) -> None:
        """壊れると、取り出した所の映像と音声が別々に動く"""
        host = _empty(video_media)
        _run(
            host,
            "place_media",
            media_id=str(video_media.id),
            ranges=[{"start_seconds": 1, "end_seconds": 2}],
        )
        clips = [clip for track in host.project.timeline.tracks for clip in track.clips]
        assert len(clips) == 2
        assert len({clip.link_group for clip in clips}) == 1
        assert clips[0].link_group is not None

    def test_the_ranges_start_at_at_frame(self, video_media: MediaItem) -> None:
        """壊れると、頼んだ位置ではなく末尾へ置く"""
        host = _empty(video_media)
        _run(
            host,
            "place_media",
            media_id=str(video_media.id),
            at_frame=60,
            ranges=[{"start_seconds": 1, "end_seconds": 2}],
        )
        assert host.project.timeline.tracks[0].clips[0].timeline_start == 60

    @pytest.mark.parametrize(
        "ranges",
        [
            [],
            None,
            "1-2",
            [{"start_seconds": 2, "end_seconds": 1}],
            [{"start_seconds": 2, "end_seconds": 2}],
            # 丸めると長さが無くなる区間と、1 フレームに届かない区間
            [{"start_seconds": 2, "end_seconds": 2.0001}],
            [{"start_seconds": 2, "end_seconds": 2.01}],
            [{"start_seconds": -1, "end_seconds": 1}],
            [{"start_seconds": 9, "end_seconds": 11}],
            [{"start_seconds": 1}],
            [{"start_seconds": "1", "end_seconds": 2}],
            [[1, 2]],
        ],
    )
    def test_a_range_that_does_not_fit_is_refused(
        self, video_media: MediaItem, ranges: object
    ) -> None:
        """壊れると、素材の外や逆向きの区間を黙って丸め、頼んだのと違う所が置かれる"""
        host = _empty(video_media)
        before = host.project
        with pytest.raises(ToolError):
            _run(host, "place_media", media_id=str(video_media.id), ranges=ranges)
        assert host.project == before

    def test_a_still_has_no_part_to_take(self, video_media: MediaItem) -> None:
        """壊れると、静止画の秒を指して何も映らないクリップを置く"""
        still = replace(video_media, duration=Fraction(0), audio_streams=())
        host = _empty(still)
        with pytest.raises(ToolError, match="静止画"):
            _run(
                host,
                "place_media",
                media_id=str(still.id),
                ranges=[{"start_seconds": 0, "end_seconds": 1}],
            )

    def test_without_ranges_the_whole_is_placed(self, video_media: MediaItem) -> None:
        """壊れると、今までの呼び方で置いた素材の長さが変わる"""
        host = _empty(video_media)
        result = _run(host, "place_media", media_id=str(video_media.id))
        assert result == {"placed": video_media.name}
        assert host.project.timeline.tracks[0].clips[0].duration == 300


class TestWaitForTranscription:
    def test_the_status_can_wait_for_the_end(self, host: FakeHost) -> None:
        """壊れると、AI が別の道具を時計代わりに呼び続ける"""
        host.transcribing_now = True
        result = _run(host, "transcription_status", wait_seconds=30)
        assert isinstance(result, Pending)
        assert result.seconds == 30
        host.transcribing_now = False
        host.transcription = "起こしも話者の分離も走っていません"
        assert result.then() == {"status": host.transcription, "transcribing": False}

    def test_without_waiting_the_status_comes_back_at_once(self, host: FakeHost) -> None:
        """壊れると、今までの呼び方が待つようになる"""
        result = _run(host, "transcription_status")
        assert result == {"status": host.transcription, "transcribing": False}

    def test_nothing_running_needs_no_wait(self, host: FakeHost) -> None:
        """壊れると、走っていないのに待ち、応えが遅れる"""
        result = _run(host, "transcription_status", wait_seconds=30)
        assert not isinstance(result, Pending)

    def test_transcribe_can_wait_too(self, host: FakeHost, video_media: MediaItem) -> None:
        """壊れると、起こしを頼んだ後にもう 1 回待つ呼び出しが要る"""
        host.transcribing_now = True
        result = _run(
            host, "transcribe", media_id=str(video_media.id), replace=True, wait_seconds=60
        )
        assert isinstance(result, Pending)
        assert result.seconds == 60
        done = result.then()
        assert isinstance(done, dict)
        assert "started" in done
        assert done["transcribing"] is True

    @pytest.mark.parametrize("value", ["10", True, float("nan")])
    def test_a_bad_wait_is_refused_before_starting(
        self, host: FakeHost, video_media: MediaItem, value: object
    ) -> None:
        """壊れると、始めてから断り、AI が呼び直すと「もう起こしている」で断られる"""
        before = host.transcription
        with pytest.raises(ToolError):
            _run(host, "transcribe", media_id=str(video_media.id), replace=True, wait_seconds=value)
        assert host.transcription == before


class TestSubtitlesWhileTranscribing:
    def test_an_empty_list_says_it_is_on_the_way(self, video_media: MediaItem) -> None:
        """壊れると、起こしの途中の空を「字幕が無い」と読み、AI が起こし直す"""
        host = _empty(video_media)
        host.transcribing_now = True
        result = _run(host, "get_subtitles")
        assert result["subtitles"] == []
        assert result["transcribing"] is True
        assert "起こしの途中" in result["note"]

    def test_finished_lists_do_not_mention_it(self, host: FakeHost) -> None:
        """壊れると、走っていないのに待ち続ける"""
        assert "transcribing" not in _run(host, "get_subtitles")


class TestOneRowPerLine:
    def test_a_linked_pair_gives_one_row(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """通しの記録の再現 リンクした映像と音声の組で、同じ行が 2 回ずつ返っていた

        壊れると、字幕の読み直しの手数と返事の長さが倍になる
        """
        media = video_media.with_transcript(transcript)
        project = Project.create(ProjectSettings(frame_rate=RATE_30, layer_mode=LayerMode.MIXED))
        for command in insert_media(project, media, at_frame=0, split_audio=True):
            project = command.apply(project)
        result = _run(FakeHost(project), "get_subtitles")
        assert [row["text"] for row in result["subtitles"]] == [
            segment.text for segment in transcript.segments
        ]

    def test_the_same_media_placed_twice_keeps_both(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、2 回置いた素材の 2 回目の字幕が消える"""
        project = make_loaded(video_media, transcript)
        track = project.timeline.tracks[0]
        media = project.media[0]
        project = AddClip(track.id, make_clip(300, 300, media)).apply(project)
        result = _run(FakeHost(project), "get_subtitles")
        assert len(result["subtitles"]) == 2 * len(transcript.segments)


def _mixed() -> SoundProfile:
    """タイムラインの混ぜた音の推移 勧める値は読まないので、中身は何でもよい"""
    profiler = SoundProfiler(RATE, 2)
    profiler.add(np.full((RATE, 2), 0.1, dtype=np.float32))
    return profiler.snapshot()


def _alternating(floor_db: float) -> Waveform:
    """話（-12 dBFS）と間（``floor_db``）が 1 秒ずつ交互の素材の波形 1 秒は 100 ピーク"""
    levels = np.repeat([db_to_ratio(-12.0), db_to_ratio(floor_db)], 100)
    amplitude = np.tile(levels, 10).astype(np.float32)
    peaks = np.stack([-amplitude, amplitude], axis=-1)[:, np.newaxis, :]
    return Waveform(
        sample_rate=48000,
        channels=1,
        total_samples=20 * 48000,
        levels=(PeakLevel(480, peaks),),
    )


class TestSuggestedThreshold:
    def test_the_value_sits_above_the_pauses(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、AI が threshold_db を当てずっぽうで振り、切っては戻すを繰り返す"""
        host = _voiced(video_media, transcript)
        host.heard = _mixed()
        host.stub_waveform = _alternating(-46.0)
        suggestion = _run(host, "analyze_audio", points=0)["jet_cut"]
        assert -46.0 < suggestion["threshold_db"] < -12.0
        assert suggestion["media_id"] == str(host.project.media[0].id)
        # 間の 10 か所のうち、終わりの間の 1 秒も含めて全部を切れる
        assert suggestion["cuts"] == 10
        assert suggestion["seconds"] == pytest.approx(8.0, abs=1.0)
        assert [row["threshold_db"] for row in suggestion["nearby"]] == sorted(
            row["threshold_db"] for row in suggestion["nearby"]
        )

    def test_steady_music_has_no_value_to_suggest(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、切る所の無い音にも値を勧め、AI が曲の途中を切る"""
        host = _voiced(video_media, transcript)
        host.heard = _mixed()
        host.stub_waveform = _alternating(-12.0)
        result = _run(host, "analyze_audio", points=0)
        assert result["jet_cut"]["threshold_db"] is None

    def test_a_clip_turned_down_does_not_change_the_value(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、混ぜた音で決めた値を jet_cut が素材の音へ当て、別の所を切る"""
        host = _voiced(video_media, transcript)
        host.heard = _mixed()
        host.stub_waveform = _alternating(-30.0)
        suggestion = _run(host, "analyze_audio", points=0)["jet_cut"]
        assert -30.0 < suggestion["threshold_db"] < -12.0

    def test_a_new_waveform_changes_the_answer(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、素材を解析し直した後も、前の波形で決めた値を AI に返す"""
        host = _voiced(video_media, transcript)
        host.heard = _mixed()
        host.stub_waveform = _alternating(-46.0)
        before = _run(host, "analyze_audio", points=0)["jet_cut"]
        assert _run(host, "analyze_audio", points=0)["jet_cut"] == before
        host.stub_waveform = _alternating(-30.0)
        after = _run(host, "analyze_audio", points=0)["jet_cut"]
        assert after["threshold_db"] != before["threshold_db"]

    def test_a_waveform_still_being_read_is_said_so(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、解析の途中を「切れる無音が無い」と読む"""
        host = _voiced(video_media, transcript)
        host.heard = _mixed()
        result = _run(host, "analyze_audio", points=0)
        assert result["jet_cut"]["threshold_db"] is None
        assert "波形" in result["jet_cut"]["note"]


class TestSameValueOnManyClips:
    @staticmethod
    def _texts(count: int, media: MediaItem, transcript: Transcript) -> tuple[FakeHost, list[Clip]]:
        project = make_loaded(media, transcript)
        clips: list[Clip] = []
        for index in range(count):
            commands = insert_generated(
                project, TEXT.create(text="札"), at_frame=index * 30, duration=30
            )
            for command in commands:
                project = command.apply(project)
            clips.append(next(c.clip for c in commands if isinstance(c, AddClip)))
        return FakeHost(project), clips

    def test_one_call_sets_them_all(self, video_media: MediaItem, transcript: Transcript) -> None:
        """通しの記録の再現 位置を揃えるのに set_param を 6 回呼んでいた"""
        host, clips = self._texts(3, video_media, transcript)
        result = _run(
            host, "set_param", clip_ids=[str(c.id) for c in clips], name="pos_x", value=-300
        )
        assert result["clip_ids"] == [str(c.id) for c in clips]
        for clip in clips:
            located = host.project.timeline.locate_clip(clip.id)
            assert located is not None
            source = located[1].source
            assert source is not None
            assert source.params["pos_x"].static == -300  # type: ignore[union-attr]

    def test_one_undo_takes_them_all_back(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、揃えた値を戻すのに本数ぶん取り消す"""
        host, clips = self._texts(3, video_media, transcript)
        before = host.project
        _run(host, "set_param", clip_ids=[str(c.id) for c in clips], name="pos_x", value=-300)
        host.undo()
        assert host.project == before

    def test_a_clip_that_cannot_take_it_stops_them_all(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、途中まで変わり、どれが変わったのかを AI も人も追えない"""
        host, clips = self._texts(2, video_media, transcript)
        movie = host.project.timeline.tracks[0].clips[0]
        before = host.project
        with pytest.raises(ToolError):
            _run(
                host,
                "set_param",
                clip_ids=[str(clips[0].id), str(movie.id)],
                name="pos_x",
                value=-300,
            )
        assert host.project == before

    def test_clip_id_and_clip_ids_together_are_refused(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、どちらかが黙って捨てられ、別のクリップが変わる"""
        host, clips = self._texts(2, video_media, transcript)
        with pytest.raises(ToolError):
            _run(
                host,
                "set_param",
                clip_id=str(clips[0].id),
                clip_ids=[str(clips[1].id)],
                name="pos_x",
                value=1,
            )

    @pytest.mark.parametrize("value", [[], None])
    def test_an_empty_list_does_not_fall_back_to_the_selection(
        self, video_media: MediaItem, transcript: Transcript, value: object
    ) -> None:
        """壊れると、指したつもりの無い選択中のクリップが変わる"""
        host, clips = self._texts(1, video_media, transcript)
        host.selection = (clips[0].id,)
        before = host.project
        with pytest.raises(ToolError):
            _run(host, "set_param", clip_ids=value, name="pos_x", value=1)
        assert host.project == before


class TestCutRange:
    def test_every_layer_loses_the_frames(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、頭を切るのに分割と削除を組み合わせ、層ごとに切り直すことになる"""
        project = make_loaded(video_media, transcript)
        commands = insert_generated(project, TEXT.create(text="札"), at_frame=100, duration=30)
        for command in commands:
            project = command.apply(project)
        text = next(c.clip for c in commands if isinstance(c, AddClip))
        host = FakeHost(project)
        result = _run(host, "cut_range", start_frame=0, end_frame=30)
        assert result["removed_frames"] == 30
        assert host.project.timeline.duration == 270
        located = host.project.timeline.locate_clip(text.id)
        assert located is not None
        assert located[1].timeline_start == 70

    @pytest.mark.parametrize(
        ("start", "end"), [(30, 30), (40, 30), (-1, 30), (0, 301), ("0", 30), (0.5, 30)]
    )
    def test_a_range_outside_is_refused(self, host: FakeHost, start: object, end: object) -> None:
        """壊れると、範囲の外を頼んで何も切れないのに切ったと答える"""
        before = host.project
        with pytest.raises(ToolError):
            _run(host, "cut_range", start_frame=start, end_frame=end)
        assert host.project == before

    def test_one_undo_restores(self, host: FakeHost) -> None:
        """壊れると、切った範囲が取り消し 1 回で戻らない"""
        before = host.project
        _run(host, "cut_range", start_frame=0, end_frame=30)
        host.undo()
        assert host.project == before


class TestWhatTheAnswerSays:
    @staticmethod
    def _work(
        video_media: MediaItem, audio_media: MediaItem
    ) -> tuple[FakeHost, Clip, Track, Track]:
        """映像 2 本と、後ろの字幕のトラックと、全体をまたぐ BGM のトラックの作品"""
        project = Project.create(
            ProjectSettings(frame_rate=RATE_30), media=(video_media, audio_media)
        )
        first = make_clip(0, 100, video_media)
        second = make_clip(100, 100, video_media)
        title = Clip(timeline_start=150, duration=10, source=TEXT.create(text="札"))
        picture = Track(kind=TrackKind.VIDEO, name="V1", clips=(first, second))
        words = Track(kind=TrackKind.VIDEO, name="字幕", clips=(title,))
        music = Track(kind=TrackKind.AUDIO, name="BGM", clips=(make_clip(0, 200, audio_media),))
        project = project.with_timeline(replace(project.timeline, tracks=(picture, words, music)))
        return FakeHost(project), first, words, music

    def test_a_sound_lost_in_the_gap_is_listed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、頼んでいない効果音が黙って消え、AI も人も気付かない"""
        host, first, _words, _music = self._work(video_media, audio_media)
        effect = make_clip(50, 10, audio_media)
        extra = Track(kind=TrackKind.AUDIO, name="効果音")
        host.apply_commands([AddTrack(extra), AddClip(extra.id, effect)], "効果音")
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["clips_removed"] == 1
        assert result["removed_clips"] == [
            {
                "clip_id": str(effect.id),
                "track_id": str(extra.id),
                "start_frame": 50,
                "end_frame": 60,
            }
        ]
        assert "取り消し" in result["note"]

    def test_nothing_lost_says_nothing(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、何も消えていないのに消えた欄が並び、AI が確かめに行く"""
        host, first, _words, _music = self._work(video_media, audio_media)
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert "clips_removed" not in result

    def test_many_lost_are_cut_short_but_counted(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、消えた行で答えが長くなり、残りの案内が読めない"""
        host, first, _words, _music = self._work(video_media, audio_media)
        extra = Track(kind=TrackKind.AUDIO, name="効果音")
        clips = [make_clip(start, 2, audio_media) for start in range(0, 90, 3)]
        host.apply_commands(
            [AddTrack(extra), *(AddClip(extra.id, clip) for clip in clips)], "効果音"
        )
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["clips_removed"] == 30
        assert len(result["removed_clips"]) == 20

    def test_cut_range_lists_what_it_took_from_other_tracks(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、範囲の中の字幕が消えたことが cut_range の答えから読めない"""
        host, _first, words, _music = self._work(video_media, audio_media)
        result = _run(host, "cut_range", start_frame=150, end_frame=170)
        assert result["clips_removed"] == 1
        assert result["removed_clips"][0]["track_id"] == str(words.id)

    def test_a_title_lost_in_the_gap_is_not_called_followed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、字幕が消えただけのトラックを、付いてきたと読む"""
        host, first, words, _music = self._work(video_media, audio_media)
        host.apply_commands([RemoveClip(host.project.timeline.tracks[1].clips[0].id)], "空に")
        title = Clip(timeline_start=50, duration=10, source=TEXT.create(text="札"))
        host.apply_commands([AddClip(words.id, title)], "字幕")
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["clips_removed"] == 1
        assert result["layers_followed"] is None

    def test_a_title_cut_at_its_tail_is_not_called_moved(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、消す所で終わる字幕が短くなっただけなのに、動いたと読む"""
        host, _first, words, _music = self._work(video_media, audio_media)
        host.apply_commands([RemoveClip(host.project.timeline.tracks[1].clips[0].id)], "空に")
        title = Clip(timeline_start=50, duration=100, source=TEXT.create(text="札"))
        host.apply_commands([AddClip(words.id, title)], "字幕")
        second = host.project.timeline.tracks[0].clips[1]
        result = _run(host, "delete_clip", clip_id=str(second.id), ripple=True)
        assert str(words.id) not in result["tracks_moved"]
        assert str(words.id) not in result["tracks_left"]
        assert result["layers_followed"] is None

    def test_a_long_title_cut_in_two_counts_as_moved(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、消す所をまたいだ字幕の後ろ半分が詰まったことを読めない"""
        host, _first, words, _music = self._work(video_media, audio_media)
        host.apply_commands([RemoveClip(host.project.timeline.tracks[1].clips[0].id)], "空に")
        title = Clip(timeline_start=50, duration=200, source=TEXT.create(text="札"))
        host.apply_commands([AddClip(words.id, title)], "字幕")
        second = host.project.timeline.tracks[0].clips[1]
        result = _run(host, "delete_clip", clip_id=str(second.id), ripple=True)
        assert str(words.id) in result["tracks_moved"]
        assert result["layers_followed"] is True

    def test_one_clip_alone_has_nothing_to_follow(self, video_media: MediaItem) -> None:
        """壊れると、付いてくる物が無い作品の答えを、付いてこなかったと読む"""
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media,))
        clip = make_clip(0, 100, video_media)
        track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
        host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(track,))))
        result = _run(host, "delete_clip", clip_id=str(clip.id), ripple=True)
        assert result["layers_followed"] is None
        assert result["tracks_moved"] == []
        assert result["tracks_left"] == []
        assert "note" not in result

    def test_deleting_a_title_alone_keeps_the_picture(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、字幕 1 枚を詰めて消しただけで絵まで動いたと答える"""
        host, _first, words, _music = self._work(video_media, audio_media)
        title = host.project.timeline.tracks[1].clips[0]
        result = _run(host, "delete_clip", clip_id=str(title.id), ripple=True)
        assert result["layers_followed"] is None
        assert str(words.id) not in result["tracks_moved"]

    def test_a_title_on_the_same_layer_counts(self, video_media: MediaItem) -> None:
        """壊れると、消したクリップと同じレイヤーの字幕が詰まっても、付いてきたと答えない"""
        project = Project.create(
            ProjectSettings(frame_rate=RATE_30, layer_mode=LayerMode.MIXED), media=(video_media,)
        )
        clip = make_clip(0, 100, video_media)
        title = Clip(timeline_start=150, duration=10, source=TEXT.create(text="札"))
        track = Track(kind=TrackKind.MIXED, name="L1", clips=(clip, title))
        host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(track,))))
        result = _run(host, "delete_clip", clip_id=str(clip.id), ripple=True)
        assert result["layers_followed"] is True

    def test_a_title_before_the_gap_is_not_called_followed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、動いていない前の字幕を理由に、消えた字幕のトラックを付いてきたと読む"""
        host, _first, words, _music = self._work(video_media, audio_media)
        title = Clip(timeline_start=50, duration=10, source=TEXT.create(text="札"))
        host.apply_commands([AddClip(words.id, title)], "字幕")
        second = host.project.timeline.tracks[0].clips[1]
        result = _run(host, "delete_clip", clip_id=str(second.id), ripple=True)
        assert str(words.id) not in result["tracks_moved"]
        assert result["layers_followed"] is None

    def test_a_title_before_the_gap_on_a_stayed_track_does_not_count_as_left(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、置いていかれていない前の字幕のせいで、付いてきた字幕まで偽と答える"""
        host, _first, words, _music = self._work(video_media, audio_media)
        later = Clip(timeline_start=250, duration=10, source=TEXT.create(text="札"))
        host.apply_commands([AddClip(words.id, later)], "後ろの字幕")
        # BGM をまたがせて動かないトラックにする 字幕は消す所より前にあり、置いていかれていない
        track = Track(
            kind=TrackKind.MIXED,
            name="重ね",
            clips=(
                Clip(timeline_start=5, duration=3, source=TEXT.create(text="札")),
                make_clip(50, 150, audio_media),
            ),
        )
        host.apply_commands([AddTrack(track)], "重ね")
        second = host.project.timeline.tracks[0].clips[1]
        result = _run(host, "delete_clip", clip_id=str(second.id), ripple=True)
        assert str(track.id) in result["tracks_left"]
        assert result["layers_followed"] is True

    def test_only_music_cut_is_not_called_followed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、字幕の無い作品で BGM が切れただけなのに字幕が付いてきたと読む"""
        host, first, words, _music = self._work(video_media, audio_media)
        host.apply_commands([RemoveClip(host.project.timeline.tracks[1].clips[0].id)], "空に")
        tune = make_clip(50, 100, audio_media)
        extra = Track(kind=TrackKind.AUDIO, name="曲")
        host.apply_commands([AddTrack(extra), AddClip(extra.id, tune)], "曲")
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["layers_followed"] is None
        assert str(extra.id) not in result["tracks_left"]
        assert str(words.id) not in result["tracks_moved"]

    def test_one_subtitle_track_left_is_not_called_followed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、2 本の字幕のうち 1 本がずれたままなのに、付いてきたと読む"""
        host, first, words, _music = self._work(video_media, audio_media)
        second = Track(kind=TrackKind.VIDEO, name="字幕 2")
        title = Clip(timeline_start=150, duration=10, source=TEXT.create(text="札"))
        host.apply_commands([AddTrack(second), AddClip(second.id, title)], "字幕")
        host.apply_commands([SetTrackState(second.id, locked=True)], "ロック")
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["layers_followed"] is False
        assert result["tracks_moved"] == [str(words.id)]
        assert str(second.id) in result["tracks_left"]

    def test_overlapping_clips_answer_as_one_cut(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、まとめて消せば付いてくる字幕を、付いてこないと答える"""
        host, first, words, _music = self._work(video_media, audio_media)
        over = make_clip(20, 30, video_media)
        extra = Track(kind=TrackKind.VIDEO, name="V2")
        host.apply_commands([AddTrack(extra), AddClip(extra.id, over)], "重ね")
        result = _run(host, "delete_clips", clip_ids=[str(first.id), str(over.id)], ripple=True)
        assert result["layers_followed"] is True
        assert str(words.id) in result["tracks_moved"]

    def test_a_track_left_behind_is_named(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、BGM が動かないことも字幕が動いたことも AI に見えない"""
        host, first, words, music = self._work(video_media, audio_media)
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["layers_followed"] is True
        assert result["tracks_moved"] == [str(words.id)]
        assert result["tracks_left"] == [str(music.id)]

    def test_a_locked_subtitle_track_is_not_called_followed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、ロックして動かなかった字幕を動いたと答え、AI がずれたまま進む"""
        host, first, words, music = self._work(video_media, audio_media)
        host.apply_commands([SetTrackState(words.id, locked=True)], "ロック")
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["layers_followed"] is False
        assert set(result["tracks_left"]) == {str(words.id), str(music.id)}
        assert "cut_range" in result["note"]

    def test_cut_range_names_the_locked_track(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、ロックして切れなかったトラックを切ったと読み、そこだけずれる"""
        host, _first, words, _music = self._work(video_media, audio_media)
        host.apply_commands([SetTrackState(words.id, locked=True)], "ロック")
        result = _run(host, "cut_range", start_frame=0, end_frame=30)
        assert result["tracks_left"] == [str(words.id)]


class TestJetCutSaysWhatWentWithIt:
    @staticmethod
    def _work(
        video_media: MediaItem, audio_media: MediaItem, *lost: Clip
    ) -> tuple[FakeHost, MediaItem, Track]:
        """1.6 秒鳴って 1.1 秒黙り、また鳴る素材を置いた作品 ``lost`` は無音の所に重ねる物"""
        project = Project.create(
            ProjectSettings(frame_rate=RATE_30), media=(video_media, audio_media)
        )
        main = Track(kind=TrackKind.VIDEO, name="V1", clips=(make_clip(0, 300, video_media),))
        extra = Track(kind=TrackKind.AUDIO, name="重ね", clips=lost)
        host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(main, extra))))
        host.stub_waveform = make_waveform([(0.5, 300), (0.0, 200), (0.5, 5500)])
        return host, video_media, extra

    def test_a_sound_in_the_silence_is_listed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、無音の所に置いた効果音が黙って消え、AI も人も気付かない"""
        effect = make_clip(60, 5, audio_media)
        host, media, extra = self._work(video_media, audio_media, effect)
        result = _run(host, "jet_cut", media_id=str(media.id), keep_speech=False)
        assert result["clips_removed"] == 1
        assert result["removed_clips"] == [
            {
                "clip_id": str(effect.id),
                "track_id": str(extra.id),
                "start_frame": 60,
                "end_frame": 65,
            }
        ]
        assert "subtitles_removed" not in result

    def test_a_burned_subtitle_is_called_out(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、字幕の焼き込みが消えたことを効果音と見分けられず、字幕の抜けを見落とす"""
        burned = Clip(
            timeline_start=60,
            duration=5,
            source=TEXT.create(text="札"),
            subtitle_origin=SubtitleOrigin(video_media.id, 1, SegmentId("s1")),
        )
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media,))
        main = Track(kind=TrackKind.VIDEO, name="V1", clips=(make_clip(0, 300, video_media),))
        words = Track(kind=TrackKind.VIDEO, name="字幕", clips=(burned,))
        host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(main, words))))
        host.stub_waveform = make_waveform([(0.5, 300), (0.0, 200), (0.5, 5500)])
        result = _run(host, "jet_cut", media_id=str(video_media.id), keep_speech=False)
        assert result["subtitles_removed"] == 1
        assert result["removed_clips"][0]["subtitle"] is True
        assert "字幕" in result["note"]

    def test_another_voice_of_the_same_media_is_listed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、見ていない音声のクリップが消えても、切った素材の欠片と同じに扱って黙る"""
        first = video_media.audio_streams[0]
        media = replace(video_media, audio_streams=(first, replace(first, index=2)))
        voice = Track(kind=TrackKind.AUDIO, name="A1", clips=(make_clip(0, 300, media),))
        other = replace(make_clip(60, 5, media), stream_index=2)
        second = Track(kind=TrackKind.AUDIO, name="A2", clips=(other,))
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(media,))
        host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(voice, second))))
        host.stub_waveform = make_waveform([(0.5, 300), (0.0, 200), (0.5, 5500)])
        result = _run(host, "jet_cut", media_id=str(media.id), audio=1, keep_speech=False)
        assert [row["clip_id"] for row in result["removed_clips"]] == [str(other.id)]

    def test_a_placement_with_no_silence_of_its_own_is_listed(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、同じ素材の話している所のクリップが別の所の範囲で消えても黙る"""
        placed = make_clip(60, 5, video_media)
        host, media, _extra = self._work(video_media, audio_media)
        words = Track(kind=TrackKind.VIDEO, name="V2", clips=(placed,))
        host.apply_commands([AddTrack(words)], "重ね")
        result = _run(host, "jet_cut", media_id=str(media.id), keep_speech=False)
        assert [row["clip_id"] for row in result["removed_clips"]] == [str(placed.id)]

    def test_the_cut_media_is_not_counted(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、切った素材の欠片まで「一緒に消えた」と毎回言う"""
        host, media, _extra = self._work(video_media, audio_media)
        result = _run(host, "jet_cut", media_id=str(media.id), keep_speech=False)
        assert "clips_removed" not in result


class TestDeleteFollowsTheLayers:
    @staticmethod
    def _work(video_media: MediaItem, transcript: Transcript) -> tuple[FakeHost, Clip, Clip]:
        project = make_loaded(video_media, transcript)
        project = RippleCut(((100, 110),)).apply(project)
        first = project.timeline.tracks[0].clips[0]
        commands = insert_generated(project, TEXT.create(text="札"), at_frame=200, duration=30)
        for command in commands:
            project = command.apply(project)
        text = next(c.clip for c in commands if isinstance(c, AddClip))
        return FakeHost(project), first, text

    def test_the_setting_decides(self, video_media: MediaItem, transcript: Transcript) -> None:
        """壊れると、設定を切った人の AI も重ねの層を動かす"""
        host, first, text = self._work(video_media, transcript)
        host.ripple_layers = False
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["layers_followed"] is False
        located = host.project.timeline.locate_clip(text.id)
        assert located is not None
        assert located[1].timeline_start == 200

    def test_the_layers_follow_by_default(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、AI が詰めるたびに字幕と重ねの層がずれ、全部を戻すことになる"""
        host, first, text = self._work(video_media, transcript)
        result = _run(host, "delete_clip", clip_id=str(first.id), ripple=True)
        assert result["layers_followed"] is True
        located = host.project.timeline.locate_clip(text.id)
        assert located is not None
        assert located[1].timeline_start == 100

    def test_delete_clips_takes_the_same_road(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、まとめて消す道具だけが重ねの層を置いていく"""
        host, first, text = self._work(video_media, transcript)
        result = _run(host, "delete_clips", clip_ids=[str(first.id)], ripple=True)
        assert result["layers_followed"] is True
        located = host.project.timeline.locate_clip(text.id)
        assert located is not None
        assert located[1].timeline_start == 100


def test_set_param_keeps_its_one_clip_form(host: FakeHost) -> None:
    """壊れると、前からの呼び方の答えの形が変わる"""
    commands = insert_generated(host.project, TEXT.create(text="札"), at_frame=0, duration=30)
    host.apply_commands(commands, "置く")
    clip = next(c.clip for c in commands if isinstance(c, AddClip))
    result = _run(host, "set_param", clip_id=str(clip.id), name="pos_x", value=5)
    assert result == {"name": "pos_x", "value": "5"}
