"""エフェクトの「グラデーション」「模様で塗る」を色のバーの止まりで塗る（#364）

止まりの並び（``gradient``）が空の間は、前からの色の項目（開始色と終了色、色 1・位置 1 …）で
塗る 前の版の作品と AviUtl2・YMM4 の読み込みはどれもこの形で、絵は前と 1 画素も変わらない
（手元の実物 61 本を前後で描き比べた docs/development.md）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import GradientSpec, ParamInput, registry
from sashimono.effects.sources import SHAPE
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.render.renderer import RenderQuality

WIDTH, HEIGHT = 200, 200
RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def flat(*stops: tuple[float, tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(number for position, colour in stops for number in (position, *colour))


THREE = flat((0.0, RED), (0.5, GREEN), (1.0, BLUE))


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def square(size: int = 120, x: float = 0.0) -> GeneratedSource:
    return SHAPE.create(shape="rect", width=size, height=size, color=(1.0, 1.0, 1.0, 1.0), pos_x=x)


def render(
    context: OffscreenGLContext,
    *layers: tuple[GeneratedSource, tuple[Effect, ...]],
    quality: int = 1,
) -> np.ndarray:
    base = Project.create(ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30)))
    tracks = tuple(
        Track(
            TrackKind.VIDEO,
            f"V{index}",
            (Clip(timeline_start=0, duration=30, source=source, effects=effects),),
        )
        for index, (source, effects) in enumerate(layers)
    )
    project = base.with_timeline(replace(base.timeline, tracks=tracks))
    renderer = FrameRenderer(project, context=context, quality=RenderQuality(quality))
    try:
        return renderer.render(0).astype(np.int16)
    finally:
        renderer.close()


def gradient(**params: ParamInput) -> Effect:
    base: dict[str, ParamInput] = {"start_color": RED, "end_color": BLUE, "angle": 0, "span": 120}
    base.update(params)
    return registry.require("gradient").create(**base)


def brush(**params: ParamInput) -> Effect:
    base: dict[str, ParamInput] = {
        "pattern": "linear",
        "size": 120,
        "stops": 2,
        "color0": RED,
        "offset0": 0.0,
        "color1": BLUE,
        "offset1": 1.0,
    }
    base.update(params)
    return registry.require("brush_fill").create(**base)


def row_at(share: float, size: int = 120) -> int:
    """正方形の上端から ``share`` の所の行（角度 0 で上が位置 0）"""
    return round(HEIGHT / 2 - size / 2 + share * size)


def column_at(share: float, size: int = 120) -> int:
    """正方形の左端から ``share`` の所の列（模様で塗るの線形は左が位置 0）"""
    return round(WIDTH / 2 - size / 2 + share * size)


def near(found: np.ndarray, expected: tuple[float, ...], tolerance: int = 8) -> bool:
    return all(abs(int(a) - b) <= tolerance for a, b in zip(found[:3], expected, strict=True))


class TestTheGradientEffect:
    def test_an_empty_list_draws_the_two_colours_as_before(self, gl: OffscreenGLContext) -> None:
        # 空の並びを白から黒の既定で塗ると、前の版の作品と AviUtl2 の読み込みの色が全部変わる
        legacy = render(gl, (square(), (gradient(),)))
        definition = registry.require("gradient")
        spec = definition.spec("gradient")
        assert isinstance(spec, GradientSpec)
        assert gradient().params["gradient"] == ()
        # バーに見せる並び（開始色と終了色）でそのまま塗っても同じ絵になる 違うと、バーを
        # 触っただけで（色を変えなくても）前からのグラデーションの色が変わる
        shown = spec.shown(gradient().params)
        assert shown == flat((0.0, RED), (1.0, BLUE))
        explicit = render(gl, (square(), (gradient(gradient=shown),)))
        assert np.abs(legacy - explicit).max() <= 1

    def test_the_stops_are_kept_in_their_places(self, gl: OffscreenGLContext) -> None:
        # 止まりの色が置いた所に出ないと、バーで真ん中に置いた緑が画面では外れた所に出る
        image = render(gl, (square(), (gradient(gradient=THREE),)))
        x = WIDTH // 2
        assert near(image[row_at(0.02), x], (255, 0, 0), 20)
        assert near(image[row_at(0.5), x], (0, 255, 0), 12)
        assert near(image[row_at(0.98), x], (0, 0, 255), 20)

    def test_each_span_between_stops_is_an_s_curve(self, gl: OffscreenGLContext) -> None:
        # AviUtl2 の 2 色の S 字を止まりの間ごとに当てる 直線で混ぜると 2 色のときの絵と
        # 合わなくなり、前からの 2 色の作品にバーで 1 色足しただけで両端の色の出方が変わる
        image = render(gl, (square(), (gradient(gradient=THREE),)))
        found = image[row_at(0.125), WIDTH // 2]
        # 赤から緑の 1/4 の所 S 字なら赤が 255 x (1 - 0.156) = 215、直線なら 191
        assert abs(int(found[0]) - 215) <= 8, found
        assert abs(int(found[1]) - 40) <= 8, found

    def test_a_clip_without_stops_does_not_take_the_previous_ones(
        self, gl: OffscreenGLContext
    ) -> None:
        # 止まりの数を毎回渡さないと、並びを持たない（前からの 2 色の）クリップが、前に描いた
        # 別のクリップの止まり（真ん中の緑）で塗られる
        image = render(
            gl,
            (square(60, -50), (gradient(gradient=THREE, span=60),)),
            (square(60, 50), (gradient(span=60),)),
        )
        right = image[HEIGHT // 2, WIDTH // 2 + 50]
        assert int(right[1]) < 30, right
        assert int(right[0]) > 90 and int(right[2]) > 90, right

    def test_half_quality_puts_the_same_colours_in_the_same_places(
        self, gl: OffscreenGLContext
    ) -> None:
        # 画質 1/2 のプレビューで止まりの位置がずれると、書き出しと違う所に色が出る
        full = render(gl, (square(), (gradient(gradient=THREE),)))
        half = render(gl, (square(), (gradient(gradient=THREE),)), quality=2)
        for share in (0.1, 0.3, 0.5, 0.7, 0.9):
            expected = full[row_at(share), WIDTH // 2]
            found = half[row_at(share) // 2, WIDTH // 4]
            assert np.abs(expected[:3] - found[:3]).max() <= 24, (share, expected, found)


class TestPaintWithAPattern:
    def test_an_empty_list_draws_the_numbered_fields_as_before(
        self, gl: OffscreenGLContext
    ) -> None:
        # 空の並びを既定で塗ると、YMM4 から読み込んだ塗り（色 1・位置 1 … で持つ）が全部変わる
        legacy = render(gl, (square(), (brush(),)))
        spec = registry.require("brush_fill").spec("gradient")
        assert isinstance(spec, GradientSpec)
        shown = spec.shown(brush().params)
        assert shown == flat((0.0, RED), (1.0, BLUE))
        explicit = render(gl, (square(), (brush(gradient=shown),)))
        assert np.abs(legacy - explicit).max() <= 1

    def test_the_bar_wins_over_the_numbered_fields(self, gl: OffscreenGLContext) -> None:
        # バーで並べた止まりが効かないと、バーを動かしても絵が変わらない
        image = render(gl, (square(), (brush(gradient=THREE),)))
        y = HEIGHT // 2
        assert near(image[y, column_at(0.02)], (255, 0, 0), 20)
        assert near(image[y, column_at(0.5)], (0, 255, 0), 12)
        assert near(image[y, column_at(0.98)], (0, 0, 255), 20)
        # 止まりの間は直線で混ぜる（YMM4 と同じ「模様で塗る」の決まりのまま）
        quarter = image[y, column_at(0.25)]
        assert abs(int(quarter[0]) - 128) <= 10 and abs(int(quarter[1]) - 128) <= 10, quarter

    def test_a_solid_pattern_keeps_the_first_colour(self, gl: OffscreenGLContext) -> None:
        # 単色の模様は「色 1」の欄だけを出す バーの並びで塗ると、出ている色の欄を直しても
        # 絵が変わらない
        solid = brush(pattern="solid", stops=1, color0=GREEN, gradient=THREE)
        image = render(gl, (square(), (solid,)))
        assert near(image[HEIGHT // 2, WIDTH // 2], (0, 255, 0), 4)

    def test_moving_numbered_offsets_still_move(self, gl: OffscreenGLContext) -> None:
        # 並びが空なら前からの位置のキーフレームで描く（読み込んだ動く模様を止めない）
        moving = AnimatedValue(0.0, (Keyframe(0, 0.9), Keyframe(29, 0.0)))
        image = render(gl, (square(), (brush(offset0=moving),)))
        assert near(image[HEIGHT // 2, column_at(0.5)], (255, 0, 0), 30)
