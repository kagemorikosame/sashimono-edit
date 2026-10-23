"""クリップの色を測って拡張色調補正の値を決める（#336）

決めた値は拡張色調補正のシェーダが掛ける 値を決める所は、シェーダと同じ式を numpy で辿って
掛けた後の色を見積もる（:func:`graded`） 式がずれると、測った後の様子を AI に違う数で
伝えるので、GPU で描いた絵とも比べる
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from sashimono.core.model import (
    Clip,
    Effect,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE
from sashimono.engine.auto_color import (
    GRADE_KIND,
    graded,
    measure,
    pixels_of,
    plan_auto,
    plan_match,
)
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SIZE = 120


def _scene(
    *, dark: float, light: float, tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """暗い所から明るい所まで灰色の階調と、色の付いた所を並べた絵（RGBA uint8）

    ``tint`` を掛けると色かぶりした絵になる 灰色の階調が白のつり合いの手がかり
    """
    ramp = np.linspace(dark, light, 96)
    grey = np.repeat(ramp[None, :, None], 3, axis=2).repeat(40, axis=0)
    rng = np.random.default_rng(0)
    colourful = dark + (light - dark) * rng.uniform(0.2, 0.9, (24, 96, 3))
    image = np.concatenate([grey, colourful], axis=0) * np.array(tint)
    rgba = np.concatenate([image, np.ones((*image.shape[:2], 1))], axis=2)
    return np.asarray(np.clip(rgba, 0.0, 1.0) * 255, dtype=np.uint8)


class TestAutomatic:
    def test_a_dull_picture_is_spread_from_black_to_white(self) -> None:
        # 壊れると、眠い（黒が浮いて白がくすんだ）絵を整えても幅が広がらない
        pixels = pixels_of([_scene(dark=0.25, light=0.7)])
        plan = plan_auto(pixels, strength=100)
        assert plan.after.black < plan.before.black - 0.1
        assert plan.after.white > plan.before.white + 0.1

    def test_a_blue_cast_is_taken_out(self) -> None:
        # 壊れると、青くかぶった絵の灰色が灰色に戻らない
        pixels = pixels_of([_scene(dark=0.05, light=0.95, tint=(0.85, 0.95, 1.1))])
        plan = plan_auto(pixels, strength=100, exposure=False)
        red, _green, blue = plan.after.neutral
        before_red, _, before_blue = plan.before.neutral
        assert abs(blue - red) < abs(before_blue - before_red) * 0.3
        assert plan.params["blue_gain"] < 0 < plan.params["red_gain"]

    def test_a_good_picture_is_left_alone(self) -> None:
        # 壊れると、整っている絵まで毎回いじり、掛け直すたびに色が動く
        pixels = pixels_of([_scene(dark=0.0, light=1.0)])
        assert not plan_auto(pixels, strength=100).changes

    def test_a_night_scene_is_not_turned_into_day(self) -> None:
        # 壊れると、わざと暗く撮った夜の絵を昼のように持ち上げる 広げる倍率に上限を置く
        pixels = pixels_of([_scene(dark=0.0, light=0.3)])
        plan = plan_auto(pixels, strength=100, white_balance=False)
        assert plan.after.white < 0.75
        assert any("止めました" in note for note in plan.notes)

    def test_strength_scales_the_values(self) -> None:
        # 壊れると、設定画面で強さを下げても同じだけ直る
        pixels = pixels_of([_scene(dark=0.25, light=0.7, tint=(1.1, 1.0, 0.9))])
        full = plan_auto(pixels, strength=100).params
        half = plan_auto(pixels, strength=50).params
        for name, value in full.items():
            assert half[name] == pytest.approx(value / 2, abs=0.1)

    def test_one_flat_colour_is_not_stretched(self) -> None:
        # 壊れると、黒一色の画面（暗転）の雑音を何倍にも広げてざらつかせる
        black = np.zeros((20, 20, 4), dtype=np.uint8)
        black[..., 3] = 255
        assert not plan_auto(pixels_of([black]), strength=100, white_balance=False).changes

    def test_a_colourful_picture_keeps_its_colours(self) -> None:
        # 壊れると、灰色の物がほとんど映らない絵（色の付いた物で埋まった録画）の偏りを色かぶりと
        # 見て、映っている物の色を消す（本人の手元の実素材の画面の録画で起きた）
        rng = np.random.default_rng(1)
        green = np.clip(rng.normal((0.3, 0.6, 0.2), 0.05, (60, 96, 3)), 0.0, 1.0)
        image = np.concatenate([green, np.ones((60, 96, 1))], axis=2)
        pixels = pixels_of([np.asarray(image * 255, dtype=np.uint8)])
        plan = plan_auto(pixels, strength=100, exposure=False)
        assert not plan.changes
        assert any("見分けられず" in note for note in plan.notes)

    def test_a_picture_that_already_reaches_black_and_white_is_not_stretched(self) -> None:
        # 壊れると、白が 95% ほどで止まる整った絵（画面の録画）を毎回少しずつ広げて変える
        pixels = pixels_of([_scene(dark=0.02, light=0.94)])
        plan = plan_auto(pixels, strength=100, white_balance=False)
        assert "luma_gain" not in plan.params

    def test_see_through_pixels_are_not_measured(self) -> None:
        # 壊れると、透けた所（黒の 0）まで暗い所として数え、黒が足りていると見誤る
        image = _scene(dark=0.3, light=0.7)
        image[:20, :, :] = 0
        assert measure(pixels_of([image])).black > 0.25


class TestMatching:
    def test_a_clip_moves_toward_the_reference(self) -> None:
        # 壊れると、同じ場面の 2 台のカメラの色を合わせても、明るさと色合いが寄らない
        mine = pixels_of([_scene(dark=0.2, light=0.6, tint=(0.9, 1.0, 1.1))])
        model = pixels_of([_scene(dark=0.05, light=0.95, tint=(1.05, 1.0, 0.95))])
        plan = plan_match(mine, model, strength=100)
        target = measure(model)
        assert abs(plan.after.white - target.white) < abs(plan.before.white - target.white) / 3
        assert abs(plan.after.black - target.black) < abs(plan.before.black - target.black) / 3
        before = np.array(plan.before.average) / np.mean(plan.before.average)
        after = np.array(plan.after.average) / np.mean(plan.after.average)
        aim = np.array(target.average) / np.mean(target.average)
        assert np.abs(after - aim).max() < np.abs(before - aim).max() / 2


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture
def paint(gl: OffscreenGLContext) -> Callable[..., np.ndarray]:
    """一色で塗った四角へエフェクトを掛け、真ん中の色（sRGB 0..255）を返す"""

    def run(colour: tuple[float, float, float], *effects: Effect) -> np.ndarray:
        source: GeneratedSource = SHAPE.create(
            shape="rect", width=100, height=100, color=(*colour, 1.0)
        )
        project = Project.create(ProjectSettings(width=SIZE, height=SIZE, frame_rate=FrameRate(30)))
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(Clip(timeline_start=0, duration=30, source=source, effects=effects),),
        )
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            image = renderer.render(0)
        finally:
            renderer.close()
        return np.asarray(image[SIZE // 2, SIZE // 2, :3], dtype=np.float64)

    return run


@pytest.mark.parametrize(
    "params",
    [
        {"luma_gain": 40.0, "luma_offset": -12.0, "luma_gamma": 25.0},
        {"red_gain": 12.0, "green_gain": -3.0, "blue_gain": -9.0},
        {"sat_gain": 30.0, "luma_gain": -20.0, "blue_gain": 10.0},
    ],
)
def test_the_estimate_matches_the_shader(
    paint: Callable[..., np.ndarray], params: dict[str, float]
) -> None:
    # 壊れると、AI へ返す「直した後」の明るさと色が、実際に描かれる絵と食い違う
    # （シェーダの式を変えて numpy の側を直し忘れたときに落ちる）
    effect = registry.require(GRADE_KIND).create(**params)
    for colour in ((0.2, 0.4, 0.6), (0.7, 0.5, 0.3), (0.45, 0.45, 0.45)):
        before = paint(colour) / 255.0
        drawn = paint(colour, effect)
        estimated = graded(before[None, :], params)[0] * 255.0
        assert np.abs(drawn - estimated).max() <= 3.0, (colour, drawn, estimated)
