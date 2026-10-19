"""最初の動画にプロジェクトを合わせる（フレームレート・解像度）

フレームレートはクリップの長さを数える物差し 置いた後に変えると長さが全部ずれるので、
空のときだけ変えられることを確かめる
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import (
    AddScene,
    Document,
    SetFrameRate,
    SetResolution,
    format_to_match,
    match_commands,
    new_scene,
)
from sashimono.core.commands.project_format import rate_for_project, video_format
from sashimono.core.model import (
    Clip,
    Marker,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT

_SIXTY = FrameRate(60)


def _video(
    width: int = 1280, height: int = 720, rate: FrameRate = _SIXTY, rotation: int = 0
) -> MediaItem:
    return MediaItem(
        path=Path("C:/素材/合成.mp4"),
        duration=Fraction(5),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=width,
                height=height,
                frame_rate=rate,
                time_base=Fraction(1, 90000),
                codec="h264",
                rotation=rotation,
            ),
        ),
    )


def _with_clip(project: Project) -> Project:
    clip = Clip(timeline_start=0, duration=30, source=TEXT.create())
    return project.with_timeline(
        replace(project.timeline, tracks=(Track(TrackKind.VIDEO, "V1", (clip,)),))
    )


class TestSetFrameRate:
    def test_an_empty_project_takes_the_new_rate(self) -> None:
        # 空のプロジェクトでも選んだレートが本体と時間軸の両方に入らないと、
        # 60fps の素材を置いても 30fps のまま書き出されて動きが粗くなる
        project = SetFrameRate(FrameRate(60)).apply(Project.create())
        assert project.rate == FrameRate(60)
        assert project.timeline.rate == FrameRate(60)

    def test_a_fractional_rate_stays_a_fraction(self) -> None:
        # 29.97 を小数で持つと 1 時間で 3 フレーム以上ずれる
        project = SetFrameRate(FrameRate(30000, 1001)).apply(Project.create())
        assert project.rate.fps == Fraction(30000, 1001)

    def test_refused_once_a_clip_is_placed(self) -> None:
        # 置いた後に変えると、フレームで数えた長さが別の秒数になる
        with pytest.raises(ValueError, match="クリップ"):
            SetFrameRate(FrameRate(60)).apply(_with_clip(Project.create()))

    def test_a_clip_inside_a_scene_also_refuses(self) -> None:
        # メインが空でもシーンの中のクリップを見落とすと、レートを変えた途端に
        # シーンに置いたクリップの長さが別の秒数になる
        base = Project.create()
        scene = new_scene(base, "中")
        project = AddScene(replace(scene, timeline=_with_clip(base).timeline)).apply(base)
        with pytest.raises(ValueError, match="クリップ"):
            SetFrameRate(FrameRate(60)).apply(project)

    def test_scenes_follow_the_project(self) -> None:
        # シーンだけ古いレートに残ると、プロジェクトの検査が開くのを断る
        base = Project.create()
        project = AddScene(new_scene(base, "中")).apply(base)
        changed = SetFrameRate(FrameRate(24)).apply(project)
        assert changed.scenes[0].timeline.rate == FrameRate(24)

    def test_markers_and_the_work_area_keep_their_time(self) -> None:
        # フレームの数のまま持ち越すと、30fps で 1 秒の所に置いた目印が 60fps では 0.5 秒になる
        base = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
        timeline = replace(base.timeline, markers=(Marker(30),), work_area=(15, 45))
        changed = SetFrameRate(FrameRate(60)).apply(base.with_timeline(timeline))
        assert changed.timeline.markers[0].frame == 60
        assert changed.timeline.work_area == (30, 90)

    def test_it_can_be_undone(self) -> None:
        # 取り消しても 50fps のまま残ると、Ctrl+Z で戻したつもりの作品が別のレートで書き出される
        document = Document(Project.create())
        document.execute(SetFrameRate(FrameRate(50)))
        document.undo()
        assert document.project.rate == FrameRate(30)


class TestRateForProject:
    def test_ntsc_stays_ntsc(self) -> None:
        # 29.97 を 30 に寄せると、1 時間の素材で 3 秒以上ずれる
        assert rate_for_project(FrameRate(30000, 1001)) == FrameRate(30000, 1001)

    def test_a_long_denominator_of_ntsc_is_recognised(self) -> None:
        # コンテナによっては 29.97 が 1798200/60001 のような形で書かれている
        assert rate_for_project(FrameRate(1798200, 60001)) == FrameRate(30000, 1001)

    def test_a_variable_rate_phone_video_goes_to_the_nearest_standard(self) -> None:
        # 可変フレームレートの平均値のままにすると、書き出しまで 29.87 fps になる
        assert rate_for_project(FrameRate(2987, 100)) == FrameRate(30)

    def test_an_unusual_rate_is_kept(self) -> None:
        # 決まったレートから遠い（15fps の画面録画など）なら、そのまま使う
        assert rate_for_project(FrameRate(15)) == FrameRate(15)


class TestFormatToMatch:
    def test_the_first_video_of_an_empty_project(self) -> None:
        # 最初の動画の形を拾えないと、1280x720 60fps の素材が 1920x1080 30fps の作品に入り、
        # 引き伸ばされたうえにコマが間引かれる
        found = format_to_match(Project.create(), [_video()])
        assert found is not None
        assert (found.width, found.height, found.rate) == (1280, 720, FrameRate(60))

    def test_nothing_when_they_already_match(self) -> None:
        # 同じ形なのに尋ねると、毎回「はい」を押すだけの窓になる
        assert format_to_match(Project.create(), [_video(1920, 1080, FrameRate(30))]) is None

    def test_nothing_once_a_clip_is_placed(self) -> None:
        # 2 本目の素材でも尋ねると、そこで合わせた途端に置いてあるクリップの長さが全部ずれる
        assert format_to_match(_with_clip(Project.create()), [_video()]) is None

    def test_a_portrait_phone_video_uses_the_turned_size(self) -> None:
        # 回す前の大きさに合わせると、縦の動画が横長のプロジェクトで小さく映る
        found = video_format(_video(1920, 1080, rotation=90))
        assert found is not None
        assert (found.width, found.height) == (1080, 1920)

    def test_an_odd_size_goes_to_the_next_even(self) -> None:
        # 奇数の大きさは書き出しのエンコーダが断る
        found = video_format(_video(853, 481))
        assert found is not None
        assert (found.width, found.height) == (854, 482)

    def test_a_still_image_is_not_a_video(self) -> None:
        # 静止画を動画と取り違えると、画像を最初に置いただけで画像の大きさへ合わせるか尋ねられる
        still = replace(_video(), duration=Fraction(0))
        assert format_to_match(Project.create(), [still]) is None

    def test_the_commands_change_only_what_differs(self) -> None:
        # 違わない方まで入れると、レートだけ違う 1080p の動画でも「解像度を変更」の手順が走り、
        # 合わせる段に要らない変更が混ざる
        project = Project.create()
        wanted = format_to_match(project, [_video(1920, 1080, FrameRate(60))])
        assert wanted is not None
        commands = match_commands(project, wanted)
        assert [type(c) for c in commands] == [SetFrameRate]
        wanted = format_to_match(project, [_video(1280, 720, FrameRate(30))])
        assert wanted is not None
        assert [type(c) for c in match_commands(project, wanted)] == [SetResolution]
