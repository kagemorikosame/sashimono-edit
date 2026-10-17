"""強調の動き（#331）を描いた絵で確かめる

偽のホストの試験（tests/ai/test_emphasis_tools.py）は線の値を見る ここでは、その線を
本物の描画に通して、寄る先の点が止まったまま絵が大きくなること、「上へ」のパンで絵が
画面の上へ動くこと（Y は上が正）を見る 線の値が合っていても、配置の掛かり方と食い違えば
寄る先がずれる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, fixed_effect
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from tests.ai.conftest import FakeHost

SETTINGS = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30))


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _host(pos: tuple[float, float]) -> tuple[FakeHost, Clip]:
    """赤い 8 画素の四角を、配置の位置 ``pos``（Y は上が正）に 60 フレーム置いた作品"""
    base = Project.create(SETTINGS)
    track = Track(TrackKind.VIDEO, "V1")
    project = base.with_timeline(replace(base.timeline, tracks=(track,)))
    transform = (
        fixed_effect(TRANSFORM_EFFECT_KIND)
        .with_param("pos_x", AnimatedValue(pos[0]))
        .with_param("pos_y", AnimatedValue(pos[1]))
    )
    square = Clip(
        timeline_start=0,
        duration=60,
        source=GeneratedSource(
            kind="shape",
            params={
                "shape": "rect",
                "color": (1.0, 0.0, 0.0, 1.0),
                "width": AnimatedValue(8.0),
                "height": AnimatedValue(8.0),
            },
        ),
        effects=(transform,),
    )
    return FakeHost(AddClip(track.id, square).apply(project)), square


def _emphasize(host: FakeHost, **arguments: Any) -> None:
    operation = find_operation("emphasize")
    assert operation is not None
    operation(host, arguments)


def _red_box(project: Project, frame: int, context: OffscreenGLContext) -> tuple[float, float, int]:
    """赤い所の中心（列, 行 行は上から）と幅"""
    renderer = FrameRenderer(project, context=context)
    try:
        image = renderer.render(frame)
    finally:
        renderer.close()
    red = (image[:, :, 0] > 128) & (image[:, :, 2] < 64)
    rows, columns = np.nonzero(red)
    assert rows.size, "赤い四角が描かれていない"
    return float(columns.mean()), float(rows.mean()), int(columns.max() - columns.min() + 1)


def test_zooming_keeps_the_focus_point_still(gl_context: OffscreenGLContext) -> None:
    # 壊れると、画面の中央に止めて寄ったつもりの絵がずれ、話者へ寄る頼みで顔が画面の外へ流れる
    host, square = _host((10.0, 10.0))
    _emphasize(
        host,
        clip_id=str(square.id),
        kind="zoom",
        start_frame=0,
        end_frame=40,
        scale_percent=200,
        focus_x=0,
        focus_y=0,
        ramp_ms=100,
    )
    column, row, width = _red_box(host.document.project, 20, gl_context)
    # 中央（32, 32）を止めて 2 倍 中央から右上へ 10 画素の四角は、右上へ 20 画素の所で 16 画素
    assert (column, row) == pytest.approx((32 + 20 - 0.5, 32 - 20 - 0.5), abs=1.0)
    assert width == pytest.approx(16, abs=1)
    # 範囲の外は元の位置と大きさ
    column, row, width = _red_box(host.document.project, 50, gl_context)
    assert (column, row) == pytest.approx((32 + 10 - 0.5, 32 - 10 - 0.5), abs=1.0)
    assert width == pytest.approx(8, abs=1)


def test_panning_up_moves_the_picture_up(gl_context: OffscreenGLContext) -> None:
    # 壊れると、「上へずらして」と頼んだ絵が画面の下へ動く
    host, square = _host((0.0, 0.0))
    _emphasize(
        host,
        clip_id=str(square.id),
        kind="pan",
        start_frame=0,
        end_frame=40,
        direction="up",
        distance_px=12,
        ramp_ms=100,
    )
    _column, row, _width = _red_box(host.document.project, 20, gl_context)
    assert row == pytest.approx(32 - 12 - 0.5, abs=1.0)
