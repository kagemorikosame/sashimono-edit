"""ジェットカット 無音区間の切り出しと、切ったあとに字幕が追従すること

字幕がずれないことを追従処理で保証しているのではなく、字幕が位置を持たない設計から
自動的に出てくる カットのテストで字幕まで確認するのは、その性質が実際に成り立って
いることをここでも押さえておくため
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sashimono.core.commands import (
    AddClip,
    RippleCut,
    SetTranscript,
    insert_media,
)
from sashimono.core.jetcut import merge_ranges, plan_cuts
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    Transcript,
)
from sashimono.core.projection import project_timeline
from tests.conftest import make_clip

#: 素材の無音区間（ソース秒） 発話は 1-3 / 4-6 / 7-9 秒にある
SILENCES = (
    (Fraction(0), Fraction(1)),
    (Fraction(3), Fraction(4)),
    (Fraction(6), Fraction(7)),
    (Fraction(9), Fraction(10)),
)


@pytest.fixture
def placed(project: Project, video_media: MediaItem, transcript: Transcript) -> Project:
    """10 秒の素材を丸ごと 1 本置き、字幕を付けた状態"""
    with_transcript = SetTranscript(video_media.id, transcript).apply(project)
    track = with_transcript.timeline.tracks[0]
    return AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)


def _clips(project: Project) -> list[tuple[int, int, Fraction]]:
    return [
        (clip.timeline_start, clip.duration, clip.source_in)
        for track in project.timeline.tracks
        for clip in track.clips
    ]


def _subtitles(project: Project) -> list[tuple[str, int, int]]:
    return [(p.segment.text, p.start_frame, p.end_frame) for p in project_timeline(project)]


class TestPlanCuts:
    def test_source_silences_become_timeline_ranges(
        self, placed: Project, video_media: MediaItem
    ) -> None:
        # 30fps でクリップが素材の先頭から始まっているので、秒がそのまま 30 倍になる
        assert plan_cuts(placed, video_media.id, SILENCES) == (
            (0, 30),
            (90, 120),
            (180, 210),
            (270, 300),
        )

    def test_silence_outside_the_used_range_is_ignored(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 素材の 2..5 秒だけを使っているクリップ
        clip = Clip(timeline_start=0, duration=90, media_id=video_media.id, source_in=Fraction(2))
        placed = AddClip(project.timeline.tracks[0].id, clip).apply(project)
        assert plan_cuts(placed, video_media.id, SILENCES) == ((30, 60),)

    def test_speed_change_compresses_the_range(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 2 倍速なので、素材の 3..4 秒（1 秒）はタイムライン上 15 フレームになる
        clip = Clip(timeline_start=0, duration=150, media_id=video_media.id, speed=Fraction(2))
        placed = AddClip(project.timeline.tracks[0].id, clip).apply(project)
        assert (45, 60) in plan_cuts(placed, video_media.id, SILENCES)

    def test_rounding_never_eats_into_speech(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 端数のある無音 内側へ丸めるので、両端は必ず無音の内部に収まる
        silences = ((Fraction(1, 3), Fraction(5, 3)),)
        placed = AddClip(project.timeline.tracks[0].id, make_clip(0, 300, video_media)).apply(
            project
        )
        ((start, end),) = plan_cuts(placed, video_media.id, silences)
        assert start >= 10 and end <= 50

    def test_the_same_media_placed_twice_is_cut_in_both_places(
        self, project: Project, video_media: MediaItem
    ) -> None:
        track = project.timeline.tracks[0].id
        placed = AddClip(track, make_clip(0, 300, video_media)).apply(project)
        placed = AddClip(track, make_clip(300, 300, video_media)).apply(placed)
        ranges = plan_cuts(placed, video_media.id, SILENCES)
        assert (90, 120) in ranges
        assert (390, 420) in ranges

    def test_too_short_ranges_are_dropped(self, placed: Project, video_media: MediaItem) -> None:
        tiny = ((Fraction(3), Fraction(3) + Fraction(1, 60)),)
        assert plan_cuts(placed, video_media.id, tiny) == ()


class TestMergeRanges:
    def test_overlapping_ranges_become_one(self) -> None:
        assert merge_ranges([(0, 10), (5, 20), (30, 40)]) == ((0, 20), (30, 40))

    def test_touching_ranges_become_one(self) -> None:
        assert merge_ranges([(10, 20), (20, 30)]) == ((10, 30),)

    def test_empty_ranges_are_dropped(self) -> None:
        assert merge_ranges([(10, 10), (5, 4)]) == ()


class TestRippleCut:
    def test_gaps_are_closed_and_source_positions_follow(self, placed: Project) -> None:
        cut = RippleCut(((0, 30), (90, 120), (180, 210), (270, 300))).apply(placed)
        # 無音 4 か所を抜いて 3 本になり、隙間なく並ぶ 素材のどこから始まるかは
        # 発話の頭（1 秒 / 4 秒 / 7 秒）に一致する
        assert _clips(cut) == [
            (0, 60, Fraction(1)),
            (60, 60, Fraction(4)),
            (120, 60, Fraction(7)),
        ]

    def test_subtitles_follow_without_any_sync_step(self, placed: Project) -> None:
        before = _subtitles(placed)
        assert before == [("今日は", 30, 90), ("編集ソフトを", 120, 180), ("作ります", 210, 270)]

        cut = RippleCut(((0, 30), (90, 120), (180, 210), (270, 300))).apply(placed)
        # 無音を抜いた分だけ前へ詰まり、本文も対応も変わらない
        assert _subtitles(cut) == [
            ("今日は", 0, 60),
            ("編集ソフトを", 60, 120),
            ("作ります", 120, 180),
        ]

    def test_a_clip_fully_inside_the_range_disappears(
        self, project: Project, video_media: MediaItem
    ) -> None:
        track = project.timeline.tracks[0].id
        placed = AddClip(track, make_clip(0, 30, video_media)).apply(project)
        placed = AddClip(track, make_clip(60, 30, video_media)).apply(placed)
        cut = RippleCut(((0, 60),)).apply(placed)
        assert _clips(cut) == [(0, 30, Fraction(0))]

    def test_linked_video_and_audio_stay_aligned(
        self, project: Project, video_media: MediaItem
    ) -> None:
        placed = project
        for command in insert_media(project, video_media):
            placed = command.apply(placed)
        cut = RippleCut(((90, 120),)).apply(placed)

        video = cut.timeline.tracks[0].clips
        audio = cut.timeline.tracks[1].clips
        assert len(video) == len(audio) == 2
        for left, right in zip(video, audio, strict=True):
            assert (left.timeline_start, left.duration) == (right.timeline_start, right.duration)

    def test_split_halves_get_a_shared_new_link_group(
        self, project: Project, video_media: MediaItem
    ) -> None:
        placed = project
        for command in insert_media(project, video_media):
            placed = command.apply(placed)
        cut = RippleCut(((90, 120),)).apply(placed)

        video = cut.timeline.tracks[0].clips
        audio = cut.timeline.tracks[1].clips
        # 右側どうしは同じグループ 左側とは別のグループ 片方を消したときに
        # もう片方まで消えるのを避けつつ、映像と音声の連動は保つ
        assert video[1].link_group == audio[1].link_group
        assert video[1].link_group != video[0].link_group

    def test_locked_tracks_are_left_alone(self, project: Project, video_media: MediaItem) -> None:
        from dataclasses import replace

        placed = AddClip(project.timeline.tracks[0].id, make_clip(0, 300, video_media)).apply(
            project
        )
        locked = placed.timeline.tracks[0]
        placed = placed.with_timeline(placed.timeline.replace_track(replace(locked, locked=True)))

        cut = RippleCut(((90, 120),)).apply(placed)
        assert _clips(cut) == [(0, 300, Fraction(0))]

    def test_overlapping_input_ranges_are_merged(self, placed: Project) -> None:
        # 重なった指定を二重に適用すると、指定より多く消える まとめてから切る
        cut = RippleCut(((0, 60), (30, 90))).apply(placed)
        assert _clips(cut) == [(0, 210, Fraction(3))]

    def test_markers_inside_the_cut_are_removed_and_the_rest_shift(self, placed: Project) -> None:
        from dataclasses import replace

        from sashimono.core.model import Marker

        marked = placed.with_timeline(
            replace(
                placed.timeline,
                markers=(Marker(frame=10), Marker(frame=100), Marker(frame=200)),
            )
        )
        cut = RippleCut(((90, 120),)).apply(marked)
        assert [m.frame for m in cut.timeline.markers] == [10, 170]


def test_the_silence_of_voice_two_is_cut_where_voice_two_plays() -> None:
    # 壊れると、リンクを外して音声 2 だけを 2 秒切り詰めたとき、音声 2 の無音を音声 1 の
    # クリップの位置で切り、話している所を削る（PR #231 の見直し）
    from dataclasses import replace
    from pathlib import Path

    from sashimono.core.model import AudioStreamInfo, Track, TrackKind

    media = MediaItem(
        path=Path("C:/素材/録画.mp4"),
        duration=Fraction(10),
        audio_streams=(
            AudioStreamInfo(1, 48000, 2, Fraction(1, 48000), "aac"),
            AudioStreamInfo(2, 48000, 2, Fraction(1, 48000), "aac"),
        ),
    )
    voice1 = Clip(timeline_start=0, duration=300, media_id=media.id, stream_index=1)
    voice2 = Clip(
        timeline_start=0, duration=240, media_id=media.id, stream_index=2, source_in=Fraction(2)
    )
    base = Project.create(media=(media,))
    tracks = (
        Track(TrackKind.AUDIO, "A1", (voice1,)),
        Track(TrackKind.AUDIO, "A2", (voice2,)),
    )
    project = base.with_timeline(replace(base.timeline, tracks=tracks))
    quiet = ((Fraction(5, 2), Fraction(3)),)
    # 音声 2 のクリップでは素材の 2.5 秒がタイムラインの 0.5 秒（15 コマ目）
    assert plan_cuts(project, media.id, quiet, stream=2) == ((15, 30),)
    # 音を渡さなければ前と同じく、素材のクリップすべての位置で切る
    assert plan_cuts(project, media.id, quiet) == ((15, 30), (75, 90))
