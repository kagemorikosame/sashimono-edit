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

import av
import numpy as np
import pytest

from sashimono.core.commands import AddClip, Document
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
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
    # 枠は 1 秒に枠の幅までしか動かないので、移ってから追い付いた所で見る
    columns = _skin_columns(project, FPS * 4 - 5, gl_context)
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


def _wide_image_project(path: Path, width: int) -> Project:
    """横 ``width`` の画像を画素で置き、50% に縮めた小さい作品（はみ出す大きさの素材）"""
    rows = np.zeros((64, width, 3), np.uint8)
    rows[:, :, 2] = 200
    with av.open(str(path), "w") as container:
        stream = container.add_stream("png", rate=1)
        stream.width, stream.height, stream.pix_fmt = width, 64, "rgb24"
        for packet in stream.encode(av.VideoFrame.from_ndarray(rows, format="rgb24")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    forget_probe(path)
    media = probe_media(path)
    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(FPS))
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
    return AddClip(track.id, clip).apply(base)


def test_a_huge_picture_falls_back_to_the_screen_buffer(
    tmp_path: Path, gl_context: OffscreenGLContext
) -> None:
    # 上限を設けないと、とても大きな画像を画素で置いたとき、はみ出し用のバッファが RGBA16F で
    # 何枚も GPU に作られてメモリが尽きるか、作れずにフレームごと描けなくなる 上限を超えたら
    # 前の道（画面の大きさで切ってから掛ける）で描き、落ちない
    project = _wide_image_project(tmp_path / "wide.png", 7000)
    renderer = FrameRenderer(project, context=gl_context)
    try:
        image = renderer.render(0)
        assert renderer._spill is None
    finally:
        renderer.close()
    # 縮めた帯（高さ 32）が画面の真ん中に描かれている
    assert int(image[HEIGHT // 2, WIDTH // 2, 2]) > 100


def test_a_lowered_limit_is_kept(tmp_path: Path, gl_context: OffscreenGLContext) -> None:
    # 上限を GPU から読み直して上書きすると、GPU の上限の小さい機械で作れない大きさを作る
    project = _wide_image_project(tmp_path / "small.png", 400)
    renderer = FrameRenderer(project, context=gl_context)
    renderer.spill_limit = 256
    try:
        renderer.render(0)
        assert renderer._spill is None
    finally:
        renderer.close()


def test_a_big_pan_shows_no_black_band(video: Path, gl_context: OffscreenGLContext) -> None:
    # 大きくパンする背景で枠を頭の時刻の所に残すと、後の時刻に枠が絵の外へ出て黒い帯が出る
    forget_probe(video)
    media = probe_media(video)
    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(FPS))
    base = Project.create(settings, media=(media,))
    track = Track(TrackKind.VIDEO, "V1")
    base = base.with_timeline(replace(base.timeline, tracks=(track,)))
    duration = FPS * 4
    pan = AnimatedValue(0.0, (Keyframe(0, -250.0), Keyframe(duration - 1, 250.0)))
    clip = with_fixed_items(
        Clip(timeline_start=0, duration=duration, media_id=media.id, native_size=True),
        picture=True,
    )
    clip = replace(
        clip,
        effects=tuple(
            e.with_param("pos_x", pan).with_param("scale", AnimatedValue(300.0))
            if e.kind == "transform"
            else e
            for e in clip.effects
        ),
    )
    project = AddClip(track.id, clip).apply(base)
    signal = analyze_scenes(video)
    assert signal is not None
    plan = plan_reframe(project, ReframeRequest(HEIGHT, WIDTH, AUTO, 1000, 90), {media.id: signal})
    document = Document(project)
    with document.checkpoint("作り替え"):
        for command in plan.commands:
            document.execute(command)
    renderer = FrameRenderer(document.project, context=gl_context)
    try:
        for frame in (0, duration // 2, duration - 1):
            image = renderer.render(frame)
            for edge in (image[:, :2], image[:, -2:]):
                assert float(edge[:, :, 2].mean()) > 40, frame
    finally:
        renderer.close()


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
