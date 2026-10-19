"""縦動画への再構図（#334）を描いた絵で確かめる

命令の試験（tests/engine/test_reframe.py）は配置の値を見る ここでは、その場で焼いた動画を
本物の場面の解析に読ませ、作り替えた作品を本物の描画に通して、右にいた顔が縦の画面の
真ん中に描かれること・画面の端に黒い帯が出ないことを見る 値が合っていても、配置の
掛かり方や素材の置き方と食い違えば、顔は真ん中に来ない
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import AddClip, Document
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import AnimatedValue, Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.reframe import AUTO
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import forget_probe, probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.reframe import ReframeRequest, plan_reframe
from sashimono.engine.render import FrameRenderer
from sashimono.engine.scenes import analyze_scenes
from tests.subject_video import FPS, HEIGHT, WIDTH, write_subject_video


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_subject_video(tmp_path_factory.mktemp("reframe") / "subject.mp4")


def _reframed(video: Path) -> Project:
    forget_probe(video)
    media = probe_media(video)
    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(FPS))
    base = Project.create(settings, media=(media,))
    track = Track(TrackKind.VIDEO, "V1")
    base = base.with_timeline(replace(base.timeline, tracks=(track,)))
    clip = with_fixed_items(
        Clip(timeline_start=0, duration=FPS * 4, media_id=media.id, native_size=True),
        picture=True,
    )
    project = AddClip(track.id, clip).apply(base)
    signal = analyze_scenes(video)
    assert signal is not None
    plan = plan_reframe(project, ReframeRequest(HEIGHT, WIDTH, AUTO, 1000, 90), {media.id: signal})
    document = Document(project)
    with document.checkpoint("作り替え"):
        for command in plan.commands:
            document.execute(command)
    return document.project


def _skin_columns(project: Project, frame: int, context: OffscreenGLContext) -> np.ndarray:
    renderer = FrameRenderer(project, context=context)
    try:
        image = renderer.render(frame)
    finally:
        renderer.close()
    assert image.shape[:2] == (WIDTH, HEIGHT)  # 縦長（高さ 320・幅 180）
    red, green, blue = (image[:, :, channel].astype(int) for channel in range(3))
    skin = (red > 180) & (green > 130) & (green < 210) & (blue > 100) & (blue < 180)
    _rows, columns = np.nonzero(skin)
    assert columns.size, "顔が描かれていない（枠から外れた）"
    return columns


def test_a_face_on_the_right_comes_to_the_middle(
    video: Path, gl_context: OffscreenGLContext
) -> None:
    # 壊れると、右に座った人の縦の作品で、顔が画面の外に切れて誰もいない所が映る
    project = _reframed(video)
    columns = _skin_columns(project, FPS, gl_context)
    assert abs(float(columns.mean()) - HEIGHT / 2) < HEIGHT * 0.15


def test_the_face_moving_left_is_followed(video: Path, gl_context: OffscreenGLContext) -> None:
    # 前の位置で止めたままだと、左へ移った人が縦の画面の外へ出る
    project = _reframed(video)
    columns = _skin_columns(project, FPS * 3, gl_context)
    assert abs(float(columns.mean()) - HEIGHT / 2) < HEIGHT * 0.15


def test_a_large_picture_shrunk_is_not_cut_at_the_screen(
    video: Path, gl_context: OffscreenGLContext
) -> None:
    # 画面の大きさで切ってから縮めると、画素で置いた 4K を 1080p の作品で 50% にしたとき、
    # 真ん中の 1/4 だけが小さく映り、周りが黒くなる（YMM4 は絵の全体を縮めて映す）
    forget_probe(video)
    media = probe_media(video)
    settings = ProjectSettings(width=WIDTH // 2, height=HEIGHT // 2, frame_rate=FrameRate(FPS))
    base = Project.create(settings, media=(media,))
    track = Track(TrackKind.VIDEO, "V1")
    base = base.with_timeline(replace(base.timeline, tracks=(track,)))
    clip = with_fixed_items(
        Clip(timeline_start=0, duration=FPS, media_id=media.id, native_size=True), picture=True
    )
    clip = replace(
        clip,
        effects=tuple(
            e.with_param("scale", AnimatedValue(50.0)) if e.kind == "transform" else e
            for e in clip.effects
        ),
    )
    project = AddClip(track.id, clip).apply(base)
    renderer = FrameRenderer(project, context=gl_context)
    try:
        image = renderer.render(10)
    finally:
        renderer.close()
    for corner in (image[:3, :3], image[:3, -3:], image[-3:, :3], image[-3:, -3:]):
        assert float(corner[:, :, 2].mean()) > 40


def test_no_black_band_at_the_edges(video: Path, gl_context: OffscreenGLContext) -> None:
    # 枠が絵の外へ出ると、縦の画面の端に黒い帯が出る（地は暗い青で、真っ黒にはならない）
    project = _reframed(video)
    renderer = FrameRenderer(project, context=gl_context)
    try:
        image = renderer.render(FPS)
    finally:
        renderer.close()
    for edge in (image[:, :2], image[:, -2:], image[:2, :], image[-2:, :]):
        assert float(edge[:, :, 2].mean()) > 40
