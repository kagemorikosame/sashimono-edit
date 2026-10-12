"""クリップの頭を前へ伸ばすトリム（#309）

素材の先頭の判定は、描くときに ``source_in`` から時刻を読むクリップ（動画・音声の素材・
入れ子のシーン・音声波形）だけに掛ける テキスト・図形・フィルタ・場面切り替え・静止画は
素材の時間を持たないので、タイムラインの先頭と同じトラックの前のクリップでだけ止まる

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import (
    AddClip,
    AddScene,
    Document,
    HeadRoom,
    TrimClip,
    TrimClips,
    head_room,
    new_scene,
)
from sashimono.core.commands.edit import (
    HEAD_STOP_PREVIOUS,
    HEAD_STOP_SCENE,
    HEAD_STOP_SOURCE,
    HEAD_STOP_TIMELINE,
)
from sashimono.core.io import load_project, save_project
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
    new_group_id,
)
from sashimono.effects.sources import FILTER, SHAPE, TEXT, TRANSITION
from tests.conftest import RATE_30, make_clip


def _with(project: Project, *clips: Clip) -> Project:
    """映像トラック 1 本へ ``clips`` を置く 置くのも命令を通す"""
    track = project.timeline.tracks[0]
    for clip in clips:
        project = AddClip(track.id, clip).apply(project)
    return project


def _with_track(project: Project, track: Track) -> Project:
    return project.with_timeline(
        replace(project.timeline, tracks=(*project.timeline.tracks, track))
    )


def _located(project: Project, clip: Clip) -> Clip:
    found = project.timeline.locate_clip(clip.id)
    assert found is not None
    return found[1]


def _generated(source: GeneratedSource, start: int = 100, duration: int = 60) -> Clip:
    return Clip(timeline_start=start, duration=duration, source=source)


def _still() -> MediaItem:
    """静止画 長さ 0 で音を持たない（:attr:`MediaItem.is_still`）"""
    return MediaItem(
        path=Path("C:/素材/絵.png"),
        duration=Fraction(0),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=640,
                height=360,
                frame_rate=RATE_30,
                time_base=Fraction(1, 30),
                codec="png",
                pixel_format="rgba",
            ),
        ),
    )


_SOURCES = {
    "テキスト": TEXT,
    "図形": SHAPE,
    "フィルタ": FILTER,
    "場面切り替え": TRANSITION,
}


class TestObjectsWithoutSourceTime:
    @pytest.mark.parametrize("name", list(_SOURCES))
    def test_the_head_extends_forward(self, project: Project, name: str) -> None:
        # 置いたテキストの頭を前へ伸ばすと「素材の先頭より前」と断られ、見せる時間を前へ延ばせない
        clip = _generated(_SOURCES[name].create())
        placed = _with(project, clip)
        result = _located(TrimClip(clip.id, head_delta=-10).apply(placed), clip)
        assert (result.timeline_start, result.duration) == (90, 70)
        assert result.source_in == 0

    def test_it_stops_at_the_start_of_the_timeline(self, project: Project) -> None:
        # 0 より前へ伸びると、クリップが負の時刻に置かれて描く所が無くなる
        clip = _generated(TEXT.create(), start=100)
        placed = _with(project, clip)
        assert head_room(placed, (clip.id,)) == HeadRoom(100, HEAD_STOP_TIMELINE)
        reached = _located(TrimClip(clip.id, head_delta=-100).apply(placed), clip)
        assert (reached.timeline_start, reached.duration) == (0, 160)
        with pytest.raises(ValueError, match=r"タイムラインの先頭.*100 フレーム"):
            TrimClip(clip.id, head_delta=-101).apply(placed)

    def test_cutting_the_head_leaves_source_in_alone(self, project: Project) -> None:
        # 削るたびに source_in が溜まり、意味の無い値が保存したファイルに残り続ける
        clip = _generated(TEXT.create())
        placed = _with(project, clip)
        cut = TrimClip(clip.id, head_delta=10).apply(placed)
        assert _located(cut, clip).source_in == 0
        back = _located(TrimClip(clip.id, head_delta=-20).apply(cut), clip)
        assert (back.timeline_start, back.duration, back.source_in) == (90, 70, Fraction(0))

    def test_a_still_picture_extends_forward(self, project: Project) -> None:
        # 静止画はいつも同じ 1 枚を描くのに、置いたばかりだと前へ伸ばせない
        still = _still()
        project = replace(project, media=(*project.media, still))
        clip = make_clip(100, 60, still)
        placed = _with(project, clip)
        result = _located(TrimClip(clip.id, head_delta=-30).apply(placed), clip)
        assert (result.timeline_start, result.source_in) == (70, Fraction(0))

    def test_keyframes_move_with_the_head(self, project: Project) -> None:
        # 頭に付いて動かないと、「登場」の動きが伸ばした頭から始まらない（利用者の決め）
        keyed = AnimatedValue(1.0, keyframes=(Keyframe(0, 0.0), Keyframe(15, 1.0)))
        clip = replace(_generated(TEXT.create()), opacity=keyed)
        placed = _with(project, clip)
        result = _located(TrimClip(clip.id, head_delta=-10).apply(placed), clip)
        assert result.opacity == keyed


class TestObjectsWithSourceTime:
    def test_video_stops_at_the_start_of_the_media(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 素材の前に無い時刻を読みに行くと、頭に何も映らない所ができる
        clip = make_clip(100, 60, video_media, source_in=1)
        placed = _with(project, clip)
        assert head_room(placed, (clip.id,)) == HeadRoom(30, HEAD_STOP_SOURCE)
        reached = _located(TrimClip(clip.id, head_delta=-30).apply(placed), clip)
        assert (reached.timeline_start, reached.source_in) == (70, Fraction(0))
        with pytest.raises(ValueError, match=r"素材の先頭より前へは伸ばせない.*あと 30 フレーム"):
            TrimClip(clip.id, head_delta=-31).apply(placed)

    def test_the_room_counts_speed(self, video_media: MediaItem, project: Project) -> None:
        # 2 倍速で 1 秒の余りは 15 フレーム 速さを見ないと、伸ばせると言った量で失敗する
        clip = replace(make_clip(100, 60, video_media, source_in=1), speed=Fraction(2))
        placed = _with(project, clip)
        assert head_room(placed, (clip.id,)).frames == 15
        TrimClip(clip.id, head_delta=-15).apply(placed)

    def test_audio_stops_at_the_start_of_the_media(self, audio_media: MediaItem) -> None:
        # 音も同じ 素材の前を読みに行くと失敗し、断る理由も伝わらない
        base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(audio_media,))
        base = _with_track(base, Track(kind=TrackKind.AUDIO, name="A1"))
        clip = make_clip(100, 60, audio_media)
        placed = _with(base, clip)
        assert head_room(placed, (clip.id,)) == HeadRoom(0, HEAD_STOP_SOURCE)
        with pytest.raises(ValueError, match=r"素材の先頭.*あと 0 フレーム"):
            TrimClip(clip.id, head_delta=-1).apply(placed)

    def test_a_scene_stops_at_its_start(self, project: Project) -> None:
        # シーンの先頭より前は無い 判定を外すと、伸ばした所でシーンの中の負の時刻を読む
        scene = new_scene(project, "中")
        project = AddScene(scene).apply(project)
        clip = Clip(timeline_start=100, duration=60, scene_id=scene.id, source_in=Fraction(1, 3))
        placed = _with(project, clip)
        assert head_room(placed, (clip.id,)) == HeadRoom(10, HEAD_STOP_SCENE)
        reached = _located(TrimClip(clip.id, head_delta=-10).apply(placed), clip)
        assert reached.source_in == 0
        with pytest.raises(ValueError, match="シーンの先頭"):
            TrimClip(clip.id, head_delta=-11).apply(placed)

    def test_a_waveform_keeps_its_reading_position(self, project: Project) -> None:
        # 音声波形は source_in から音を読む 動かさないと、頭を削っても波形が頭から描き直される
        clip = _generated(SHAPE.create(shape="waveform"))
        placed = _with(project, clip)
        cut = _located(TrimClip(clip.id, head_delta=30).apply(placed), clip)
        assert cut.source_in == Fraction(1)
        assert head_room(placed, (clip.id,)) == HeadRoom(0, HEAD_STOP_SOURCE)


class TestTheClipBefore:
    def test_it_stops_at_the_end_of_the_clip_before(self, project: Project) -> None:
        # 重なる所まで伸ばすとトラックが重なりを断り、何も変わらないまま失敗する（利用者の決め）
        before = _generated(TEXT.create(), start=0, duration=60)
        clip = _generated(TEXT.create(), start=100)
        placed = _with(project, before, clip)
        assert head_room(placed, (clip.id,)) == HeadRoom(40, HEAD_STOP_PREVIOUS)
        reached = _located(TrimClip(clip.id, head_delta=-40).apply(placed), clip)
        assert reached.timeline_start == 60
        with pytest.raises(ValueError, match=r"前のクリップの終わり.*あと 40 フレーム"):
            TrimClip(clip.id, head_delta=-41).apply(placed)

    def test_a_clip_after_does_not_count(self, project: Project) -> None:
        # 後ろのクリップを前のクリップと数えると、離れているのに伸ばせなくなる
        clip = _generated(TEXT.create(), start=100)
        after = _generated(TEXT.create(), start=200)
        placed = _with(project, clip, after)
        assert head_room(placed, (clip.id,)) == HeadRoom(100, HEAD_STOP_TIMELINE)

    def test_touching_clips_cannot_extend(self, project: Project) -> None:
        # 隙間の無い所で 1 フレームでも伸ばせると言うと、離したときに重なりで失敗する
        before = _generated(TEXT.create(), start=40, duration=60)
        clip = _generated(TEXT.create(), start=100)
        placed = _with(project, before, clip)
        assert head_room(placed, (clip.id,)).frames == 0


class TestLinkedAndSelected:
    def test_a_linked_partner_limits_the_text(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # リンクした音は一緒に伸びる（_linked_group の決まり） 音の余りを見ないと、ずれるか失敗する
        audio_track = Track(kind=TrackKind.AUDIO, name="A1")
        project = _with_track(project, audio_track)
        link = new_group_id()
        text = replace(_generated(TEXT.create()), link_group=link)
        sound = replace(make_clip(100, 60, video_media), link_group=link)
        placed = AddClip(audio_track.id, sound).apply(_with(project, text))
        assert head_room(placed, (text.id,)) == HeadRoom(0, HEAD_STOP_SOURCE)

    def test_a_locked_partner_does_not_limit(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # ロックした相手は伸びずに残る（TrimClip の決まり） それで止めると、伸ばせるのに伸ばせない
        audio_track = Track(kind=TrackKind.AUDIO, name="A1")
        project = _with_track(project, audio_track)
        link = new_group_id()
        text = replace(_generated(TEXT.create()), link_group=link)
        sound = replace(make_clip(100, 60, video_media), link_group=link)
        placed = AddClip(audio_track.id, sound).apply(_with(project, text))
        locked = replace(placed.timeline.tracks[1], locked=True)
        placed = placed.with_timeline(placed.timeline.replace_track(locked))
        assert head_room(placed, (text.id,)) == HeadRoom(100, HEAD_STOP_TIMELINE)
        result = TrimClip(text.id, head_delta=-20).apply(placed)
        assert _located(result, text).timeline_start == 80
        assert _located(result, sound).timeline_start == 100

    def test_several_clips_take_the_tightest(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # まとめて伸ばすと全員に同じ量を当てる 緩い方で数えると、きつい方で失敗して何も変わらない
        layer = Track(kind=TrackKind.VIDEO, name="V2")
        project = _with_track(project, layer)
        text = _generated(TEXT.create(), start=100)
        movie = replace(make_clip(100, 60, video_media), source_in=Fraction(1, 3))
        placed = AddClip(layer.id, movie).apply(_with(project, text))
        assert head_room(placed, (text.id, movie.id)) == HeadRoom(10, HEAD_STOP_SOURCE)
        result = TrimClips((text.id, movie.id), head_delta=-10).apply(placed)
        assert _located(result, text).timeline_start == 90
        assert _located(result, movie).timeline_start == 90


class TestHistoryAndFiles:
    def test_one_undo_brings_it_back(self, project: Project) -> None:
        # 取り消しが 2 回要ると、伸ばす前に戻したつもりで別の手まで消す
        clip = _generated(TEXT.create())
        document = Document(_with(project, clip))
        before = document.project
        document.execute(TrimClip(clip.id, head_delta=-30))
        assert _located(document.project, clip).timeline_start == 70
        document.undo()
        assert document.project == before

    def test_it_survives_saving_and_opening(self, project: Project, tmp_path: Path) -> None:
        # 開き直すと位置や source_in が変わると、保存した作品のテロップが出る時刻がずれる
        clip = _generated(TEXT.create())
        extended = TrimClip(clip.id, head_delta=-30).apply(_with(project, clip))
        path = tmp_path / "伸ばした.sme"
        save_project(extended, path)
        reopened = load_project(path)
        assert _located(reopened, clip) == _located(extended, clip)
        # 開き直した後も同じ決まりで伸ばせる
        again = _located(TrimClip(clip.id, head_delta=-70).apply(reopened), clip)
        assert (again.timeline_start, again.source_in) == (0, Fraction(0))
