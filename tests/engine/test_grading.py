"""色を作り替えるエフェクトの効き方（AviUtl の色まわり 4 種）

値の意味は AviUtl2 に 1 つずつ動かした見本を描かせて測った ここに並ぶ数は
その実測から取ったもので、落ちたときに疑うのはこちらの実装の側
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from sashimono.core.model import (
    Blending,
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
from sashimono.effects.spec import ParamInput
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

WIDTH, HEIGHT = 120, 120


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
    """一色で塗った四角へエフェクトを掛け、真ん中の色を返す"""

    def run(colour: tuple[float, float, float], *effects: Effect) -> np.ndarray:
        source: GeneratedSource = SHAPE.create(
            shape="rect", width=100, height=100, color=(*colour, 1.0)
        )
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
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
        return np.asarray(image[HEIGHT // 2, WIDTH // 2, :3], dtype=np.int16)

    return run


def _grade(**params: float | bool | str) -> Effect:
    return registry.require("color_grade").create(**params)


class TestColorGrade:
    """拡張色調補正 4 つのつまみは AviUtl2 の実測に合わせてある"""

    def test_gain_multiplies(self, paint: Callable[..., np.ndarray]) -> None:
        # ゲイン 50 は 1.5 倍 足し算として扱うと、暗い所ばかりが持ち上がる
        grey = (0.25, 0.25, 0.25)
        before = int(paint(grey)[0])
        after = int(paint(grey, _grade(luma_gain=50.0))[0])
        assert after == pytest.approx(before * 1.5, abs=6)

    def test_offset_adds(self, paint: Callable[..., np.ndarray]) -> None:
        # オフセット 50 は +127/255 倍率として扱うと、黒がいつまでも黒いまま
        assert int(paint((0.0, 0.0, 0.0), _grade(luma_offset=50.0))[0]) == pytest.approx(127, abs=8)

    def test_lift_raises_the_black_and_keeps_the_white(
        self, paint: Callable[..., np.ndarray]
    ) -> None:
        # リフトは黒を持ち上げて白を残す オフセットと同じにすると白が飛ぶ
        assert int(paint((0.0, 0.0, 0.0), _grade(luma_lift=50.0))[0]) == pytest.approx(127, abs=8)
        assert int(paint((1.0, 1.0, 1.0), _grade(luma_lift=50.0))[0]) == pytest.approx(255, abs=4)

    def test_gamma_bends_the_middle_and_leaves_the_ends(
        self, paint: Callable[..., np.ndarray]
    ) -> None:
        # ガンマ 50 は 0.72 乗 中ほどだけが持ち上がる
        grey = (0.5, 0.5, 0.5)
        assert int(paint(grey, _grade(luma_gamma=50.0))[0]) > int(paint(grey)[0]) + 8
        # 端が動くなら、それはガンマではなく明るさの足し算になっている
        assert int(paint((0.0, 0.0, 0.0), _grade(luma_gamma=50.0))[0]) <= 3

    def test_the_hue_turns(self, paint: Callable[..., np.ndarray]) -> None:
        # 色相を 120 度ずらすと赤が緑へ 度ではなく割合で渡すと少ししか回らない
        after = paint((1.0, 0.0, 0.0), _grade(hue_offset=120.0))
        assert after[1] > after[0] and after[1] > after[2]

    def test_defaults_change_nothing(self, paint: Callable[..., np.ndarray]) -> None:
        # つまみが全部 0 のときに絵が変わるなら、既定値の意味を取り違えている
        colour = (0.6, 0.3, 0.2)
        assert np.abs(paint(colour, _grade()) - paint(colour)).max() <= 2


class TestGradientMap:
    def _effect(self, **params: ParamInput) -> Effect:
        base: dict[str, ParamInput] = {
            "strength": 100.0,
            "dark_color": (0.0, 0.0, 1.0, 1.0),
            "light_color": (1.0, 1.0, 0.0, 1.0),
        }
        base.update(params)
        return registry.require("gradient_map").create(**base)

    def test_dark_and_light_go_to_the_two_colours(self, paint: Callable[..., np.ndarray]) -> None:
        # 暗い色と明るい色を逆に当てると、影が黄色く光る所が青く沈む反転したような絵になる
        dark = paint((0.0, 0.0, 0.0), self._effect())
        light = paint((1.0, 1.0, 1.0), self._effect())
        assert dark[2] > 200 and dark[0] < 40
        assert light[0] > 200 and light[2] < 40

    def test_it_measures_brightness_with_rec601(self, paint: Callable[..., np.ndarray]) -> None:
        """赤は 30% の位置の色になる Rec.709 の重みだと 21% で別の色が出る

        AviUtl2 に赤を通して測った結果（30% の位置）に合わせてある
        """
        effect = self._effect(dark_color=(0.0, 0.0, 0.0, 1.0), light_color=(1.0, 1.0, 1.0, 1.0))
        assert int(paint((1.0, 0.0, 0.0), effect)[0]) == pytest.approx(0.299 * 255, abs=10)

    def test_strength_zero_leaves_the_picture(self, paint: Callable[..., np.ndarray]) -> None:
        # 強さを 0 にしても色が置き換わると、キーフレームで 0 から効かせ始めても
        # 頭から色が変わっている
        after = paint((1.0, 0.0, 0.0), self._effect(strength=0.0))
        assert after[0] > 200 and after[2] < 40


class TestColorRangeShift:
    def _effect(self) -> Effect:
        return registry.require("color_range_shift").create(
            key_color=(1.0, 0.0, 0.0, 1.0),
            to_color=(0.0, 1.0, 0.0, 1.0),
            hue_range=22.0,
            saturation_range=38.0,
            feather=2.0,
        )

    def test_the_named_colour_is_replaced(self, paint: Callable[..., np.ndarray]) -> None:
        # 指定した赤そのものが塗り替わらないと、色の置き換えを掛けても絵が何も変わらない
        after = paint((1.0, 0.0, 0.0), self._effect())
        assert after[1] > 200 and after[0] < 60

    def test_a_far_colour_is_left_alone(self, paint: Callable[..., np.ndarray]) -> None:
        # 色相の範囲を広く取りすぎると、関係のない色まで塗り替わる
        after = paint((0.0, 0.0, 1.0), self._effect())
        assert after[2] > 200 and after[1] < 60

    def test_the_same_hue_but_pale_is_left_alone(self, paint: Callable[..., np.ndarray]) -> None:
        """色相が同じでも、彩度が範囲から外れていれば塗り替えない

        彩度の範囲を 0..1 でなく 0..100 のまま比べると、しきい値が必ず上回って
        彩度の縛りが消える そのときは、この薄い赤まで緑へ変わってしまう
        """
        # 彩度 0.25 の赤 変換前の色（彩度 1.0）との差 0.75 は範囲 38% の外
        after = paint((1.0, 0.75, 0.75), self._effect())
        assert after[0] > 200, "薄い赤が塗り替わっている"
        assert after[1] < 220

    def test_the_key_colour_is_fully_replaced_even_with_a_wide_edge(
        self, paint: Callable[..., np.ndarray]
    ) -> None:
        """境界補正が範囲より大きくても、変換前の色そのものは残さない

        境界を範囲の前後へ散らすと下限が負になり、差 0 の画素まで
        塗り替えが中途半端になる（下限を 0 で止めていないと落ちる）
        """
        effect = registry.require("color_range_shift").create(
            key_color=(1.0, 0.0, 0.0, 1.0),
            to_color=(0.0, 1.0, 0.0, 1.0),
            hue_range=16.0,
            saturation_range=38.0,
            feather=40.0,
        )
        after = paint((1.0, 0.0, 0.0), effect)
        assert after[1] > 200 and after[0] < 60


class TestFlash:
    def test_the_light_spills_outside_the_shape(self, gl: OffscreenGLContext) -> None:
        """光が図形の外へ広がる

        1 パス目の出力を元の絵と取り違えると画面が灰色の靄になり、
        不透明度を上げ忘れると図形の外に何も出ない どちらも一度やった
        """
        effect = registry.require("flash").create(strength=100.0, light_color=(0.0, 0.8, 1.0, 1.0))
        source: GeneratedSource = SHAPE.create(
            shape="ellipse", width=30, height=30, color=(1.0, 1.0, 1.0, 1.0)
        )
        # 広がり方を見る試験で、重ね合わせの方法とは関係ない しきい値はリニアで混ぜたときの
        # 明るさで決めてある（sRGB で混ぜると、薄い光の裾は同じ量でも暗く出る）
        project = Project.create(
            ProjectSettings(
                width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30), blending=Blending.LINEAR
            )
        )
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(Clip(timeline_start=0, duration=30, source=source, effects=(effect,)),),
        )
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            image = renderer.render(0)
        finally:
            renderer.close()
        outside = image[HEIGHT // 2, WIDTH // 2 + 30, :3]
        assert int(outside.max()) > 10, "図形の外に光が出ていない"
        assert int(image[HEIGHT // 2, WIDTH // 2, :3].min()) > 200, "図形の中が暗い"


class TestTheGradientRamp:
    """グラデーションの帯の並び 新しい効果を測る下地にもなっている"""

    def test_the_ramp_is_eased_at_both_ends(self, gl: OffscreenGLContext) -> None:
        """位置に対して直線ではなく S 字

        AviUtl2 に黒から白の階調を描かせて位置ごとの明るさを測ると、
        1/4 の位置が 255 中の 28 ほど（直線なら 64）だった
        直線で混ぜると、帯の両端が濃くなりすぎて中ほどが薄い別の絵になる
        """
        effect = registry.require("gradient").create(
            strength=100.0,
            start_color=(0.0, 0.0, 0.0, 1.0),
            end_color=(1.0, 1.0, 1.0, 1.0),
            shape="linear",
            angle=0.0,
            span=float(HEIGHT),
        )
        source: GeneratedSource = SHAPE.create(
            shape="rect", width=WIDTH, height=HEIGHT, color=(1.0, 1.0, 1.0, 1.0)
        )
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(Clip(timeline_start=0, duration=30, source=source, effects=(effect,)),),
        )
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            image = renderer.render(0)
        finally:
            renderer.close()
        quarter = int(image[HEIGHT // 4, WIDTH // 2, :3].mean())
        middle = int(image[HEIGHT // 2, WIDTH // 2, :3].mean())
        assert middle == pytest.approx(127, abs=12), "真ん中が中点になっていない"
        assert quarter < 50, f"1/4 の位置が濃すぎる（直線で混ぜている） {quarter}"
