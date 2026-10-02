"""字幕の編集コマンドと焼き込み"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sashimono.core.commands import (
    AddClip,
    AddSegment,
    MergeWithNext,
    RemoveSegment,
    RetimeSegment,
    SetSegmentText,
    SetTranscript,
    SplitSegment,
    burn_defaults,
    burn_subtitles,
)
from sashimono.core.model import (
    AnimatedValue,
    GeneratedSource,
    MediaItem,
    Project,
    Transcript,
    TranscriptSegment,
    Word,
)
from sashimono.core.projection import project_timeline
from tests.conftest import make_clip


@pytest.fixture
def placed(project: Project, video_media: MediaItem, transcript: Transcript) -> Project:
    with_transcript = SetTranscript(video_media.id, transcript).apply(project)
    track = with_transcript.timeline.tracks[0]
    return AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)


def _segments(project: Project, media: MediaItem) -> list[tuple[str, Fraction, Fraction]]:
    item = project.require_media(media.id)
    assert item.transcript is not None
    return [(s.text, s.start, s.end) for s in item.transcript.segments]


class TestSetSegmentText:
    def test_editing_changes_every_place_the_media_is_used(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 同じ素材を 2 回置く 字幕は素材に属するので、直すのは 1 回で済む
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        track = with_transcript.timeline.tracks[0].id
        placed = AddClip(track, make_clip(0, 300, video_media)).apply(with_transcript)
        placed = AddClip(track, make_clip(300, 300, video_media)).apply(placed)

        first = transcript.segments[0]
        edited = SetSegmentText(video_media.id, first.id, "こんにちは").apply(placed)

        texts = [p.segment.text for p in project_timeline(edited)]
        assert texts.count("こんにちは") == 2

    def test_edited_flag_is_set(self, placed: Project, video_media: MediaItem) -> None:
        item = placed.require_media(video_media.id)
        assert item.transcript is not None
        target = item.transcript.segments[0]
        edited = SetSegmentText(video_media.id, target.id, "直した").apply(placed)

        updated = edited.require_media(video_media.id).transcript
        assert updated is not None
        assert updated.segments[0].edited is True

    def test_unknown_segment_fails(self, placed: Project, video_media: MediaItem) -> None:
        from sashimono.core.model import SegmentId

        with pytest.raises(KeyError):
            SetSegmentText(video_media.id, SegmentId("なし"), "x").apply(placed)


class TestRetimeSegment:
    def test_moving_a_segment_keeps_the_order_valid(
        self, placed: Project, video_media: MediaItem
    ) -> None:
        item = placed.require_media(video_media.id)
        assert item.transcript is not None
        first = item.transcript.segments[0]

        # 1 枚目を最後尾へ動かす 順序の不変条件はコマンド側で整える
        moved = RetimeSegment(video_media.id, first.id, Fraction(9, 2), Fraction(11, 2)).apply(
            placed
        )
        assert [text for text, _, _ in _segments(moved, video_media)] == [
            "編集ソフトを",
            "今日は",
            "作ります",
        ]

    def test_inverted_range_fails(self, placed: Project, video_media: MediaItem) -> None:
        item = placed.require_media(video_media.id)
        assert item.transcript is not None
        first = item.transcript.segments[0]
        with pytest.raises(ValueError, match="終了が開始以前"):
            RetimeSegment(video_media.id, first.id, Fraction(5), Fraction(5)).apply(placed)


class TestRemoveAndAdd:
    def test_removing_drops_it_from_the_timeline(
        self, placed: Project, video_media: MediaItem
    ) -> None:
        item = placed.require_media(video_media.id)
        assert item.transcript is not None
        removed = RemoveSegment(video_media.id, item.transcript.segments[1].id).apply(placed)
        assert [p.segment.text for p in project_timeline(removed)] == ["今日は", "作ります"]

    def test_adding_to_media_without_transcript(
        self, project: Project, video_media: MediaItem
    ) -> None:
        added = AddSegment(video_media.id, Fraction(1), Fraction(2), "手で入れた").apply(project)
        assert _segments(added, video_media) == [("手で入れた", Fraction(1), Fraction(2))]


class TestSplitSegment:
    def test_word_timestamps_decide_the_boundary(
        self, project: Project, video_media: MediaItem
    ) -> None:
        segment = TranscriptSegment(
            start=Fraction(0),
            end=Fraction(4),
            text="今日は編集します",
            words=(
                Word(Fraction(0), Fraction(1), "今日は"),
                Word(Fraction(2), Fraction(4), "編集します"),
            ),
        )
        placed = SetTranscript(video_media.id, Transcript((segment,))).apply(project)
        split = SplitSegment(video_media.id, segment.id, Fraction(2)).apply(placed)

        assert _segments(split, video_media) == [
            ("今日は", Fraction(0), Fraction(2)),
            ("編集します", Fraction(2), Fraction(4)),
        ]

    def test_without_words_the_text_is_split_by_time_ratio(
        self, project: Project, video_media: MediaItem
    ) -> None:
        segment = TranscriptSegment(start=Fraction(0), end=Fraction(4), text="ABCDEFGH")
        placed = SetTranscript(video_media.id, Transcript((segment,))).apply(project)
        split = SplitSegment(video_media.id, segment.id, Fraction(1)).apply(placed)
        assert [text for text, _, _ in _segments(split, video_media)] == ["AB", "CDEFGH"]

    def test_split_outside_the_segment_fails(
        self, project: Project, video_media: MediaItem
    ) -> None:
        segment = TranscriptSegment(start=Fraction(0), end=Fraction(4), text="ABC")
        placed = SetTranscript(video_media.id, Transcript((segment,))).apply(project)
        with pytest.raises(ValueError, match="内側にない"):
            SplitSegment(video_media.id, segment.id, Fraction(9)).apply(placed)


class TestMergeWithNext:
    def test_two_segments_become_one(self, placed: Project, video_media: MediaItem) -> None:
        item = placed.require_media(video_media.id)
        assert item.transcript is not None
        merged = MergeWithNext(video_media.id, item.transcript.segments[0].id).apply(placed)
        assert _segments(merged, video_media) == [
            ("今日は 編集ソフトを", Fraction(1), Fraction(6)),
            ("作ります", Fraction(7), Fraction(9)),
        ]

    def test_merging_the_last_one_fails(self, placed: Project, video_media: MediaItem) -> None:
        item = placed.require_media(video_media.id)
        assert item.transcript is not None
        with pytest.raises(ValueError, match="次の字幕が無い"):
            MergeWithNext(video_media.id, item.transcript.segments[-1].id).apply(placed)


class TestBurnSubtitles:
    def test_projected_subtitles_become_text_clips(self, placed: Project) -> None:
        template = GeneratedSource(kind="text", params={"text": "", "size": AnimatedValue(48.0)})
        burned = placed
        for command in burn_subtitles(placed, template):
            burned = command.apply(burned)

        track = burned.timeline.tracks[-1]
        assert track.name == "字幕"
        assert [(c.timeline_start, c.duration) for c in track.clips] == [
            (30, 60),
            (120, 60),
            (210, 60),
        ]
        # 本文が入り、テンプレートの他のパラメータは残る
        first = track.clips[0].source
        assert first is not None
        assert first.params["text"] == "今日は"
        assert first.params["size"] == AnimatedValue(48.0)

    def test_the_default_look_is_unchanged_at_1080p(self) -> None:
        # 1080p では前の版と同じ見た目 変わると、前の版で焼いた字幕と並べたときに揃わない
        assert burn_defaults(1080) == {"size": 48.0, "pos_y": -380.0, "border_width": 4.0}

    @pytest.mark.parametrize("height", [480, 720, 1080, 2160])
    def test_the_default_look_stays_inside_the_frame(self, height: int) -> None:
        """既定の焼き込みが、どの高さの作品でも画面の中に出ること

        画素のまま -380 に置いていたときは、720p（下端が -360）で字幕が丸ごと画面の
        外に出て、焼き込んだのにプレビューにも書き出しにも何も出なかった
        """
        look = burn_defaults(height)
        bottom = look["pos_y"] - look["size"] / 2 - look["border_width"]
        assert bottom > -height / 2
        # 下寄せのまま 真ん中より下に出る
        assert look["pos_y"] < -height / 4

    def test_nothing_to_burn_returns_no_commands(self, project: Project) -> None:
        assert burn_subtitles(project, GeneratedSource(kind="text")) == []

    def test_overlapping_subtitles_are_trimmed_to_fit_one_track(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 同じ素材を重ねて置くと、字幕も重なる 1 本のトラックには重ねられない
        overlapping = Transcript(
            (TranscriptSegment(start=Fraction(0), end=Fraction(4), text="重なる"),)
        )
        from sashimono.core.commands import AddTrack
        from sashimono.core.model import Track, TrackKind

        placed = SetTranscript(video_media.id, overlapping).apply(project)
        second = Track(kind=TrackKind.VIDEO, name="V2")
        placed = AddTrack(second).apply(placed)
        placed = AddClip(placed.timeline.tracks[0].id, make_clip(0, 60, video_media)).apply(placed)
        placed = AddClip(second.id, make_clip(30, 60, video_media)).apply(placed)

        burned = placed
        for command in burn_subtitles(placed, GeneratedSource(kind="text")):
            burned = command.apply(burned)

        clips = burned.timeline.tracks[-1].clips
        assert [(c.timeline_start, c.duration) for c in clips] == [(0, 30), (30, 60)]
