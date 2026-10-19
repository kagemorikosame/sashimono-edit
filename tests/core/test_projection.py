"""字幕の投影

設計の要なので厚めに固める 「カットしても分割しても速度を変えても字幕がずれない」
ことは、追従処理を書いて保証するのではなく、位置を持たせない設計から自動的に
出てくる そのことをここで確認する
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddTrack,
    MoveClip,
    RemoveClip,
    SetTranscript,
    SplitClip,
    TrimClip,
)
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.projection import project_timeline
from tests.conftest import make_clip


@pytest.fixture
def project_with_subtitles(
    project: Project, video_media: MediaItem, transcript: Transcript
) -> Project:
    """10 秒の素材を丸ごと 1 本置き、字幕を付けた状態

    字幕は 1..3 秒「今日は」、4..6 秒「編集ソフトを」、7..9 秒「作ります」
    30fps なので 300 フレームのクリップになる
    """
    with_transcript = SetTranscript(video_media.id, transcript).apply(project)
    track = with_transcript.timeline.tracks[0]
    return AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)


def _texts_with_bounds(project: Project) -> list[tuple[str, int, int]]:
    return [(p.segment.text, p.start_frame, p.end_frame) for p in project_timeline(project)]


class TestBasicProjection:
    def test_untouched_clip_maps_source_seconds_to_frames(
        self, project_with_subtitles: Project
    ) -> None:
        # クリップが素材の先頭から始まっているので、1 秒 = 30 フレームがそのまま出る
        assert _texts_with_bounds(project_with_subtitles) == [
            ("今日は", 30, 90),
            ("編集ソフトを", 120, 180),
            ("作ります", 210, 270),
        ]

    def test_media_without_transcript_yields_nothing(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 文字起こし前の素材でつまずくと、字幕をまだ作っていない動画を置いただけで
        # 字幕の書き出しと焼き込み（字幕パネルと AI の道具）が例外で止まる
        track = project.timeline.tracks[0]
        placed = AddClip(track.id, make_clip(0, 300, video_media)).apply(project)
        assert list(project_timeline(placed)) == []

    def test_generated_clip_without_media_is_skipped(self, project: Project) -> None:
        # 素材を持たないテロップや図形のクリップでつまずくと、テキストを 1 枚置いただけで
        # 字幕の書き出しと焼き込み（字幕パネルと AI の道具）が例外で止まる
        track = project.timeline.tracks[0]
        text_clip = Clip(timeline_start=0, duration=60, media_id=None)
        placed = AddClip(track.id, text_clip).apply(project)
        assert list(project_timeline(placed)) == []


class TestFollowsEditing:
    def test_moving_the_clip_moves_the_subtitles(self, project_with_subtitles: Project) -> None:
        # 字幕がクリップの位置を追わないと、動画を後ろへずらしても字幕だけ元の時刻に残り、
        # 話す前に字幕が出る
        clip = project_with_subtitles.timeline.tracks[0].clips[0]
        moved = MoveClip(clip.id, 600).apply(project_with_subtitles)
        assert _texts_with_bounds(moved) == [
            ("今日は", 630, 690),
            ("編集ソフトを", 720, 780),
            ("作ります", 810, 870),
        ]

    def test_splitting_divides_the_subtitles(self, project_with_subtitles: Project) -> None:
        # 3.5 秒（105 フレーム）で割る 1 枚目は左、2・3 枚目は右に入る
        clip = project_with_subtitles.timeline.tracks[0].clips[0]
        split = SplitClip(clip.id, 105).apply(project_with_subtitles)

        left, right = split.timeline.tracks[0].clips
        projected = list(project_timeline(split))
        by_clip = {p.segment.text: p.clip_id for p in projected}
        assert by_clip["今日は"] == left.id
        assert by_clip["編集ソフトを"] == right.id
        assert by_clip["作ります"] == right.id

    def test_splitting_does_not_move_subtitles_on_the_timeline(
        self, project_with_subtitles: Project
    ) -> None:
        # 分割は見た目を変えない操作 字幕の位置も変わってはいけない
        before = _texts_with_bounds(project_with_subtitles)
        clip = project_with_subtitles.timeline.tracks[0].clips[0]
        split = SplitClip(clip.id, 105).apply(project_with_subtitles)
        assert _texts_with_bounds(split) == before

    def test_ripple_delete_pulls_subtitles_along(self, project_with_subtitles: Project) -> None:
        # 前半 2 秒を切り取って詰めると、残った字幕が 2 秒分前へ来る
        clip = project_with_subtitles.timeline.tracks[0].clips[0]
        split = SplitClip(clip.id, 60).apply(project_with_subtitles)
        head = split.timeline.tracks[0].clips[0]
        rippled = RemoveClip(head.id, ripple=True).apply(split)

        assert _texts_with_bounds(rippled) == [
            ("今日は", 0, 30),
            ("編集ソフトを", 60, 120),
            ("作ります", 150, 210),
        ]

    def test_trimming_the_head_drops_hidden_subtitles(
        self, project_with_subtitles: Project
    ) -> None:
        # 先頭 3.5 秒を捨てると「今日は」は完全に範囲外になり、出なくなる
        clip = project_with_subtitles.timeline.tracks[0].clips[0]
        trimmed = TrimClip(clip.id, head_delta=105).apply(project_with_subtitles)
        assert [text for text, _, _ in _texts_with_bounds(trimmed)] == [
            "編集ソフトを",
            "作ります",
        ]

    def test_subtitle_straddling_the_cut_is_clipped(self, project_with_subtitles: Project) -> None:
        # 2 秒（60 フレーム）で頭を切ると「今日は」（1..3 秒）は途中から始まる
        clip = project_with_subtitles.timeline.tracks[0].clips[0]
        trimmed = TrimClip(clip.id, head_delta=60).apply(project_with_subtitles)

        first = next(iter(project_timeline(trimmed)))
        assert first.segment.text == "今日は"
        assert first.clipped_head
        assert not first.clipped_tail
        # 元の 1..3 秒のうち 2..3 秒だけが残る クリップは 60 フレームから始まる
        assert (first.start_frame, first.end_frame) == (60, 90)


class TestSpeed:
    def test_double_speed_halves_the_spacing(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 倍速で字幕の間隔が詰まらないと、早回しした所で字幕が声よりどんどん遅れて出る
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        track = with_transcript.timeline.tracks[0]
        fast = Clip(timeline_start=0, duration=150, media_id=video_media.id, speed=Fraction(2))
        placed = AddClip(track.id, fast).apply(with_transcript)

        assert _texts_with_bounds(placed) == [
            ("今日は", 15, 45),
            ("編集ソフトを", 60, 90),
            ("作ります", 105, 135),
        ]

    def test_half_speed_doubles_the_spacing(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # スローで字幕の間隔が伸びないと、字幕が声より先に進み、後ろの字幕ほど話す前に出る
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        track = with_transcript.timeline.tracks[0]
        # 半速では 10 秒の素材が 20 秒（600 フレーム）になる
        slow = Clip(timeline_start=0, duration=600, media_id=video_media.id, speed=Fraction(1, 2))
        placed = AddClip(track.id, slow).apply(with_transcript)

        assert _texts_with_bounds(placed) == [
            ("今日は", 60, 180),
            ("編集ソフトを", 240, 360),
            ("作ります", 420, 540),
        ]


class TestReuse:
    def test_same_media_placed_twice_shows_subtitles_twice(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 素材を使い回したとき、字幕が片方にしか出ない方が驚きが大きい
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        track = with_transcript.timeline.tracks[0]
        placed = AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)
        placed = AddClip(track.id, make_clip(300, 300, video_media)).apply(placed)

        texts = [text for text, _, _ in _texts_with_bounds(placed)]
        assert texts == ["今日は", "編集ソフトを", "作ります"] * 2

    def test_editing_the_transcript_updates_every_occurrence(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 使い回した片方だけが直ると、誤字を直したのに 2 回目の場面では古い文字が焼き込まれる
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        track = with_transcript.timeline.tracks[0]
        placed = AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)
        placed = AddClip(track.id, make_clip(300, 300, video_media)).apply(placed)

        edited = transcript.replace_segment(transcript.segments[0].with_text("こんばんは"))
        updated = SetTranscript(video_media.id, edited).apply(placed)

        texts = [text for text, _, _ in _texts_with_bounds(updated)]
        assert texts.count("こんばんは") == 2
        assert "今日は" not in texts

    def test_projection_is_sorted_across_tracks(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # トラックの順のまま返すと、AI に渡す字幕の一覧が時刻順にならず、AI が場面と違う
        # 順で字幕を読んで、話の流れを取り違える
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        second = Track(kind=TrackKind.VIDEO, name="V2")
        with_track = AddTrack(second).apply(with_transcript)

        first_track = with_track.timeline.tracks[0]
        placed = AddClip(first_track.id, make_clip(300, 300, video_media)).apply(with_track)
        placed = AddClip(second.id, make_clip(0, 300, video_media)).apply(placed)

        starts = [p.start_frame for p in project_timeline(placed)]
        assert starts == sorted(starts)


class TestEdgeCases:
    def test_subtitle_shorter_than_a_frame_still_shows(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 丸めの結果 0 フレームになると画面に出ない 最低 1 フレームは確保する
        tiny = Transcript(
            segments=(TranscriptSegment(start=Fraction(1, 1000), end=Fraction(2, 1000), text="あ"),)
        )
        with_transcript = SetTranscript(video_media.id, tiny).apply(project)
        track = with_transcript.timeline.tracks[0]
        placed = AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)

        projected = list(project_timeline(placed))
        assert len(projected) == 1
        assert projected[0].duration >= 1

    def test_media_not_in_pool_is_skipped(
        self, project: Project, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 参照だけ残った状態でも投影は落ちない オフライン素材の扱いに関わる
        with_media = AddMedia(audio_media).apply(project)
        track = with_media.timeline.tracks[0]
        placed = AddClip(track.id, make_clip(0, 300, video_media)).apply(with_media)
        broken = placed.with_media(tuple(m for m in placed.media if m.id != video_media.id))
        assert list(project_timeline(broken)) == []
