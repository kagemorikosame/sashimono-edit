"""画素で置いて小さく入った動画を知らせ、画面に合わせる（#494）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands.fit import fit_commands, placed_small, small_share
from sashimono.core.commands.insert import insert_media
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Interpolation,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate


def _video(width: int, height: int) -> MediaItem:
    stream = VideoStreamInfo(
        index=0,
        width=width,
        height=height,
        frame_rate=FrameRate(30),
        codec="h264",
        pixel_format="yuv420p",
        time_base=Fraction(1, 15360),
    )
    return MediaItem(path=Path("clip.mp4"), duration=Fraction(10), video_streams=(stream,))


def _placed(width: int, height: int) -> tuple[Project, Clip]:
    project = Project.create(ProjectSettings(width=1920, height=1080))
    for command in insert_media(project, _video(width, height), at_frame=0):
        project = command.apply(project)
    clip = next(c for t in project.timeline.tracks for c in t.clips if c.show_picture)
    return project, clip


def _scaled(clip: Clip, value: AnimatedValue) -> Clip:
    effects = tuple(
        effect.with_param("scale", value) if effect.kind == "transform" else effect
        for effect in clip.effects
    )
    return replace(clip, effects=effects)


class TestSmall:
    def test_a_video_enlarged_by_hand_is_not_told(self) -> None:
        """数えると、ワイプとして 150% に決めた動画にも知らせが出続ける"""
        project, clip = _placed(640, 360)
        assert placed_small(project, _scaled(clip, AnimatedValue(static=150.0))) is None

    def test_a_360p_video_in_1080p(self) -> None:
        """壊れると、1/3 の大きさで入ったことに書き出すまで気付かない"""
        project, clip = _placed(640, 360)
        assert placed_small(project, clip) == pytest.approx(1 / 3)

    @pytest.mark.parametrize(("width", "height"), [(1920, 1080), (1920, 800), (3840, 2160)])
    def test_a_video_that_fills_the_screen_is_quiet(self, width: int, height: int) -> None:
        """足りている素材で知らせると、毎回出て読まれなくなる"""
        assert small_share(ProjectSettings(width=1920, height=1080), _video(width, height)) is None

    def test_a_scaled_up_clip_is_quiet(self) -> None:
        """手で拡大した物に知らせ続けると、直したつもりの人が迷う"""
        project, clip = _placed(640, 360)
        assert placed_small(project, _scaled(clip, AnimatedValue(static=300.0))) is None

    def test_a_zooming_clip_is_quiet(self) -> None:
        """ズームの演出を合わせると、付けた動きの大きさが変わる"""
        project, clip = _placed(640, 360)
        zoom = AnimatedValue(keyframes=(Keyframe(0, 100.0, Interpolation.LINEAR),))
        assert placed_small(project, _scaled(clip, zoom)) is None


class TestFit:
    def test_the_default_still_places_by_pixels(self) -> None:
        """既定を変えると、YMM4・AviUtl から写した拡大率の意味が変わる"""
        _project, clip = _placed(640, 360)
        assert clip.native_size

    def test_fitting_turns_off_pixels_and_resets_the_scale(self) -> None:
        """拡大率を残して収めると、画面の 3 倍に拡大される"""
        project, clip = _placed(640, 360)
        bigger = _scaled(clip, AnimatedValue(static=150.0))
        tracks = tuple(
            replace(t, clips=tuple(bigger if c.id == clip.id else c for c in t.clips))
            for t in project.timeline.tracks
        )
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        for command in fit_commands(bigger):
            project = command.apply(project)
        fitted = project.timeline.tracks[0].clips[0]
        assert not fitted.native_size
        assert placed_small(project, fitted) is None
        transform = next(e for e in fitted.effects if e.kind == "transform")
        assert transform.params["scale"] == AnimatedValue(static=100.0)
