"""エフェクトと生成オブジェクト

判定を確実にするため、映像素材ではなく図形やテキストを使う testsrc2 のような
模様の上では「変わった／変わらない」しか言えず、どう変わるべきかを書けない
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

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
from sashimono.effects import ColorSpec, ParamInput, TrackSpec, registry
from sashimono.effects.definition import EffectDefinition
from sashimono.effects.sources import SHAPE, TEXT, source_registry
from sashimono.engine.gpu import BlendMode, GLContextError, OffscreenGLContext, srgb_to_linear
from sashimono.engine.render import FrameRenderer
from sashimono.engine.sources import render_source


def builtin_effects() -> tuple[EffectDefinition, ...]:
    """GPU で動く自前のエフェクトだけ

    AviUtl スクリプトは Lua なので除く 音のエフェクトも GPU を通さないので除く
    （音は :attr:`EffectDefinition.audio_process` に関数を持ち、シェーダを持たない）
    """
    return tuple(
        d for d in registry.all() if not d.kind.startswith("aviutl:") and d.audio_process is None
    )


WIDTH, HEIGHT = 200, 200


@pytest.fixture(scope="session")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture
def draw(gl: OffscreenGLContext) -> Callable[..., np.ndarray]:
    """1 クリップだけのプロジェクトを描いて画像を返す"""

    def render(
        source: GeneratedSource,
        effects: tuple[Effect, ...] = (),
        *,
        frame: int = 0,
        blend: str = BlendMode.NORMAL,
    ) -> np.ndarray:
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(
                Clip(
                    timeline_start=0,
                    duration=60,
                    source=source,
                    effects=effects,
                    blend_mode=blend,
                ),
            ),
        )
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            return renderer.render(frame)
        finally:
            renderer.close()

    return render


def white_square(size: int = 100) -> GeneratedSource:
    """中央に置いた白い正方形 位置と大きさが分かっているので判定しやすい"""
    return SHAPE.create(shape="rect", width=size, height=size, color=(1.0, 1.0, 1.0, 1.0))


def centre(image: np.ndarray) -> list[int]:
    return [int(v) for v in image[HEIGHT // 2, WIDTH // 2, :3]]


def lit_pixels(image: np.ndarray, threshold: int = 20) -> int:
    return int((image[..., :3].max(axis=2) > threshold).sum())


class TestParameterSpecs:
    def test_track_clamps_out_of_range(self) -> None:
        spec = TrackSpec("radius", "範囲", 0, 100, 10)
        assert spec.coerce(500).static == 100
        assert spec.coerce(-5).static == 0

    def test_track_keeps_animation(self) -> None:
        # キーフレームの付いた値は、範囲で切らずにそのまま通す
        spec = TrackSpec("radius", "範囲", 0, 100, 10)
        animated = AnimatedValue(keyframes=(Keyframe(frame=0, value=0.0),))
        assert spec.coerce(animated) is animated

    def test_track_rejects_bad_definition(self) -> None:
        with pytest.raises(ValueError, match="既定値が範囲外"):
            TrackSpec("x", "X", 0, 10, 50)

    def test_color_pads_missing_alpha(self) -> None:
        spec = ColorSpec("color", "色")
        assert spec.coerce((1.0, 0.5, 0.0)) == (1.0, 0.5, 0.0, 1.0)

    def test_garbage_falls_back_to_the_default(self) -> None:
        # 配布エイリアスから読んだ値は型が信用できない 既定値へ寄せる
        spec = TrackSpec("radius", "範囲", 0, 100, 10)
        assert spec.coerce("でたらめ").static == 10


class TestRegistry:
    def test_standard_effects_are_registered(self) -> None:
        # P2 の完了条件が標準エフェクト 10 種
        assert len(registry) >= 10

    def test_every_effect_has_a_shader_and_parameters(self) -> None:
        # AviUtl スクリプトは Lua で動くのでシェーダを持たない ここでは
        # 自前の（GPU で動く）エフェクトだけを見る
        # 色の反転は、そもそも調整する値を持たない（反転するかしないかだけ）
        parameterless = {"invert"}
        for definition in builtin_effects():
            assert definition.fragment_shader, f"{definition.kind}: シェーダが無い"
            if definition.kind not in parameterless:
                assert definition.parameters, f"{definition.kind}: パラメータが無い"

    def test_defaults_round_trip_through_normalize(self) -> None:
        for definition in registry.all():
            params = definition.default_params()
            assert definition.normalize(params) == params

    def test_unknown_effect_is_not_an_error(self) -> None:
        assert registry.get("存在しない") is None

    def test_duplicate_registration_is_refused(self) -> None:
        with pytest.raises(ValueError, match="すでに登録"):
            registry.register(registry.require("blur"))


class TestShaders:
    def test_every_effect_compiles(self, gl: OffscreenGLContext) -> None:
        # コンパイルできないエフェクトは黙って素通しになる仕様なので、
        # 「絵が出た」だけでは検出できない 1 つずつ通して確かめる
        from sashimono.engine.gpu.effects import EffectProcessor
        from sashimono.engine.gpu.glutil import ScreenQuad

        with gl:
            quad = ScreenQuad()
            processor = EffectProcessor(16, 16, quad)
            try:
                for definition in builtin_effects():
                    # has_work では見ない 既定のままの変形は何もしない値なので、組めても偽になる
                    assert processor._compile(definition.create()) is not None, (
                        f"{definition.kind}: シェーダをコンパイルできない"
                    )
            finally:
                processor.release()
                quad.release()


class TestColorEffect:
    def test_brightness(self, draw: Callable[..., np.ndarray]) -> None:
        grey = SHAPE.create(shape="rect", width=180, height=180, color=(0.5, 0.5, 0.5, 1.0))
        plain = draw(grey)
        brighter = draw(grey, (registry.require("color").create(brightness=50),))
        assert centre(brighter)[0] > centre(plain)[0]

    def test_saturation_to_zero_makes_grey(self, draw: Callable[..., np.ndarray]) -> None:
        red = SHAPE.create(shape="rect", width=180, height=180, color=(1.0, 0.2, 0.2, 1.0))
        desaturated = draw(red, (registry.require("color").create(saturation=-100),))
        r, g, b = centre(desaturated)
        assert abs(r - g) <= 2
        assert abs(g - b) <= 2

    def test_defaults_change_nothing(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square()
        assert np.array_equal(draw(square), draw(square, (registry.require("color").create(),)))

    def test_gain_multiplies_the_encoded_value(self, draw: Callable[..., np.ndarray]) -> None:
        """輝度（``gain``）は sRGB の値に掛ける YMM4 の色調補正の「輝度」と同じ形

        リニアのまま掛けると 150% でも sRGB で 1.2 倍ほどにしかならず、明るく飛ばす
        場面切り替え（ペイントトランジション）の真ん中が YMM4 より 30 ほど暗く出た
        """
        grey = SHAPE.create(shape="rect", width=180, height=180, color=(0.5, 0.5, 0.5, 1.0))
        plain = centre(draw(grey))
        brighter = centre(draw(grey, (registry.require("color").create(gain=150),)))
        assert brighter[0] == pytest.approx(plain[0] * 1.5, abs=3)
        # 白は白のまま頭打ち
        white = centre(draw(white_square(180), (registry.require("color").create(gain=150),)))
        assert white[0] >= 254


class TestGeometryEffects:
    def test_transform_moves_the_image(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(60)
        moved = draw(square, (registry.require("transform").create(pos_x=60),))
        # 中央は空き、右へずれた位置に現れる
        assert moved[HEIGHT // 2, WIDTH // 2, 0] < 20
        assert moved[HEIGHT // 2, WIDTH // 2 + 60, 0] > 200

    def test_transform_moves_up_for_a_positive_y(self, draw: Callable[..., np.ndarray]) -> None:
        # Y は正が上 テキストの位置・影のずれ・マスクの中心と同じ向きでないと、
        # 同じ「Y」という表示なのに項目ごとに上下が入れ替わる
        square = white_square(40)
        moved = draw(square, (registry.require("transform").create(pos_y=60),))
        assert moved[HEIGHT // 2 - 60, WIDTH // 2, 0] > 200, "上へ動いていない"
        assert moved[HEIGHT // 2 + 60, WIDTH // 2, 0] < 20

    def test_transform_rotates_clockwise(self, draw: Callable[..., np.ndarray]) -> None:
        # AviUtl の「回転」も時計回り 読み込んだ角度をそのまま渡せる
        bar = SHAPE.create(shape="rect", width=20, height=140, color=(1.0, 1.0, 1.0, 1.0))
        turned = draw(bar, (registry.require("transform").create(rotation=45),))
        rows, columns = (turned[:, :, 0] > 100).nonzero()
        upper = columns[rows < HEIGHT // 2].mean()
        lower = columns[rows > HEIGHT // 2].mean()
        assert upper > lower, "時計回りになっていない"

    def test_a_tilt_is_seen_from_the_origin_not_from_a_far_pivot(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """支点を絵から遠く離して傾けても、遠近は絵の原点から見た形になる

        支点の正面にカメラを置くと、1000 画素下の支点から見上げる形になり、Y 軸で
        傾けただけの四角が上下に歪む YMM4 は原点から見た遠近のまま傾けた
        （ページめくり風その2 は支点が 1700 画素下にあり、差が最大 13 から 0.3 まで減った）
        """
        square = white_square(60)
        tilted = draw(
            square,
            (
                registry.require("transform").create(
                    rotation_y=40, pivot_h="origin", pivot_v="origin", anchor_y=-1000
                ),
            ),
        )
        lit = tilted[:, :, 0] > 100
        rows = lit.any(axis=1).nonzero()[0]
        # 原点から見れば、Y 軸の傾きは上下に対して対称
        assert rows.min() + rows.max() + 1 == pytest.approx(HEIGHT, abs=2)
        assert np.abs(lit.astype(int) - lit[::-1].astype(int)).sum() < 40

    def test_transform_scale(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(60)
        plain = lit_pixels(draw(square))
        doubled = lit_pixels(draw(square, (registry.require("transform").create(scale=200),)))
        assert doubled == pytest.approx(plain * 4, rel=0.15)

    def test_crop_cuts_the_edges(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(180)
        cropped = draw(square, (registry.require("crop").create(top=80, bottom=80),))
        assert cropped[HEIGHT // 2, WIDTH // 2, 0] > 200
        assert cropped[10, WIDTH // 2, 0] < 20

    def test_crop_measures_from_the_edges_of_the_picture(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """切る量は絵の端から数える 画面の端から数えると、画面より小さい絵は切れない

        SFっぽい吹き出しは 604 の高さの図形の上を 280 切る 画面の上から数えると
        図形の上端（238）より手前で終わり、1 画素も切れずに板が上へはみ出していた
        """
        square = white_square(100)
        cropped = draw(
            square,
            (registry.require("crop").create(top=30, bottom=10, left=20, right=40),),
        )
        lit = cropped[:, :, 0] > 100
        rows = lit.any(axis=1).nonzero()[0]
        columns = lit.any(axis=0).nonzero()[0]
        # 100 の四角は 50〜149 に置かれる
        assert (int(rows.min()), int(rows.max())) == (80, 139)
        assert (int(columns.min()), int(columns.max())) == (70, 109)

    def test_mask_hides_the_outside(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(180)
        masked = draw(
            square,
            (registry.require("mask").create(shape="ellipse", mask_width=60, mask_height=60),),
        )
        assert masked[HEIGHT // 2, WIDTH // 2, 0] > 200
        assert masked[HEIGHT // 2, 20, 0] < 20

    def test_mask_centre_moves_up_for_a_positive_y(self, draw: Callable[..., np.ndarray]) -> None:
        big = white_square(180)
        masked = draw(
            big,
            (
                registry.require("mask").create(
                    shape="rect", center_y=60, mask_width=200, mask_height=60, feather=0
                ),
            ),
        )
        assert masked[HEIGHT // 2 - 60, WIDTH // 2, 0] > 200, "上が残っていない"
        assert masked[HEIGHT // 2 + 60, WIDTH // 2, 0] < 20

    def test_mask_invert(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(180)
        masked = draw(
            square,
            (
                registry.require("mask").create(
                    shape="ellipse", mask_width=60, mask_height=60, invert=True
                ),
            ),
        )
        assert masked[HEIGHT // 2, WIDTH // 2, 0] < 20
        assert masked[HEIGHT // 2, 20, 0] > 200


class TestBlurEffects:
    def test_blur_softens_the_edge(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(80)
        sharp = draw(square)
        blurred = draw(square, (registry.require("blur").create(radius=12),))

        # 縁の外側に色がにじみ出る
        edge_x = WIDTH // 2 + 44
        assert sharp[HEIGHT // 2, edge_x, 0] < 20
        assert blurred[HEIGHT // 2, edge_x, 0] > 20

    def test_blur_does_not_darken_the_edge(self, draw: Callable[..., np.ndarray]) -> None:
        # ストレートアルファのまま畳むと、透明画素の黒が混ざって縁が黒ずむ
        # 事前乗算で畳んでいれば、白い四角の縁は白いままにじむ
        square = white_square(80)
        blurred = draw(square, (registry.require("blur").create(radius=10),))
        row = blurred[HEIGHT // 2, WIDTH // 2 : WIDTH // 2 + 50, :3]
        lit = row[row.max(axis=1) > 30]
        assert lit.size > 0
        # にじんだ部分も無彩色（白）のままであること
        assert int(np.abs(lit[:, 0].astype(int) - lit[:, 2].astype(int)).max()) <= 3

    def test_glow_brightens_around_the_shape(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(60)
        plain = draw(square)
        glowing = draw(
            square,
            (registry.require("glow").create(threshold=0.2, intensity=200, radius=20),),
        )
        outside = (HEIGHT // 2, WIDTH // 2 + 40)
        assert plain[outside][0] < 20
        assert glowing[outside][0] > plain[outside][0]

    def test_sharpen_keeps_flat_areas_flat(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(120)
        sharpened = draw(square, (registry.require("sharpen").create(strength=200),))
        assert centre(sharpened)[0] > 200

    def test_mosaic_makes_uniform_blocks(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(101)
        blocky = draw(square, (registry.require("mosaic").create(size=40),))
        # 縁がブロックの境界に揃うので、中間の値がほとんど無くなる
        values = blocky[..., 0]
        midtones = int(((values > 40) & (values < 210)).sum())
        assert midtones < 400


class TestDecorationEffects:
    def test_border_surrounds_the_shape(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(60)
        bordered = draw(
            square,
            (registry.require("border").create(width=8, color=(1.0, 0.0, 0.0, 1.0)),),
        )
        just_outside = bordered[HEIGHT // 2, WIDTH // 2 + 34]
        assert just_outside[0] > 100, "縁が描かれていない"
        assert just_outside[1] < 80, "縁が赤くない"
        assert centre(bordered)[1] > 200, "中身が塗り潰されている"

    def test_border_outline_only_drops_the_inside(self, draw: Callable[..., np.ndarray]) -> None:
        """縁だけにすると、縁の輪が残って元の絵が消える

        YMM4 の縁取りの「縁だけ」（``IsOutlineOnly``）はこの絵になる 中身が残ると
        SFっぽい吹き出しが縁の線ではなく塗りの四角のまま出る（#175）
        """
        square = white_square(60)
        effect = registry.require("border").create(
            width=8, color=(1.0, 0.0, 0.0, 1.0), outline_only=True
        )
        ring = draw(square, (effect,))
        just_outside = ring[HEIGHT // 2, WIDTH // 2 + 34]
        assert just_outside[0] > 100, "縁が描かれていない"
        assert just_outside[1] < 80, "縁が赤くない"
        assert max(centre(ring)) < 20, "縁だけなのに中身が残っている"

    def test_border_opacity_thins_the_edge(self, draw: Callable[..., np.ndarray]) -> None:
        """縁取りの不透明度は縁の濃さだけを変え、中身は変えない"""
        square = white_square(60)
        border = registry.require("border")
        half = draw(square, (border.create(width=8, color=(1.0, 0.0, 0.0, 1.0), opacity=50),))
        just_outside = half[HEIGHT // 2, WIDTH // 2 + 34]
        assert 90 < just_outside[0] < 170, "縁の濃さが半分になっていない"
        assert centre(half)[1] > 200, "中身まで薄くなった"

    def test_border_blur_softens_the_outer_edge(self, draw: Callable[..., np.ndarray]) -> None:
        """縁のぼかしは、縁の外側をなだらかに薄くする ぼかし 0 は今までと同じくっきりした縁

        YMM4 の縁取りの ``Blur`` 読まずにいると、ぼかして薄く光らせる縁（SFっぽい
        吹き出し(右) の 2.5）がくっきり太く出て、格子の隙間を埋める（#179）
        """
        square = white_square(60)
        border = registry.require("border")
        sharp = draw(square, (border.create(width=8, color=(1.0, 0.0, 0.0, 1.0)),))
        soft = draw(square, (border.create(width=8, color=(1.0, 0.0, 0.0, 1.0), blur=8),))
        plain = draw(square, (border.create(width=8, color=(1.0, 0.0, 0.0, 1.0), blur=0),))
        assert np.array_equal(sharp, plain), "ぼかし 0 で絵が変わった"
        # 四角の右端は 130 縁はその外 8 画素 ぼかすと、縁の外まで薄く広がり、縁の外寄りは薄まる
        row = HEIGHT // 2
        assert sharp[row, 130 + 11][0] < 20
        assert soft[row, 130 + 11][0] > 20, "縁の外へ広がっていない"
        assert soft[row, 130 + 7][0] < sharp[row, 130 + 7][0] - 20, "縁の外寄りが薄まっていない"
        assert soft[row, 130 + 1][0] > 200, "縁の内側まで薄まった"

    def test_a_crop_of_nothing_keeps_the_border_outside(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """切る量が 0 の辺では、前の縁取りが絵の範囲の外に描いた縁を消さない"""
        square = white_square(60)
        border = registry.require("border").create(width=8, color=(1.0, 0.0, 0.0, 1.0))
        crop = registry.require("crop").create(top=10)
        image = draw(square, (border, crop))
        just_outside = image[HEIGHT // 2, WIDTH // 2 + 34]
        assert just_outside[0] > 100, "切っていない右の辺の縁が消えた"
        assert image[HEIGHT // 2 - 28, WIDTH // 2][0] < 20, "切った上の辺が残った"

    def test_a_transparent_pattern_colour_does_not_bleed(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """模様の透明な色の RGB は、縮めたときに隣の色へ混ざらない

        YMM4 の格子のブラシは背景が透明な白（``#00FFFFFF``）のことが多い 透明な所に
        白を残すと、縮めたり動かしたりして隣の画素と混ぜたときに白が浮き出し、
        暗い格子が白っぽい板になる（SFっぽい吹き出し(右) #175）
        """
        stripes = registry.require("brush_fill").create(
            pattern="stripe",
            pattern_only=True,
            stops=2,
            color0=(1.0, 0.0, 0.0, 1.0),
            color1=(1.0, 1.0, 1.0, 0.0),
            width_a=1,
            width_b=1,
            angle=0,
        )
        shrink = registry.require("transform").create(scale=50)
        image = draw(white_square(100), (stripes, shrink))
        red, green, blue = centre(image)
        assert red > 40, "縞が描かれていない"
        assert max(green, blue) < 20, "透明な所の白が混ざった"

    def test_a_thin_slanted_grid_is_evenly_dense(self, draw: Callable[..., np.ndarray]) -> None:
        """斜めの細い格子は、どこを切り取っても同じ濃さに見える

        線を画素ごとに「乗る・乗らない」で描くと、太さ 1.3 の線が場所によって 1 画素にも
        2 画素にもなり、菱形の大きな濃淡（モアレ）が浮く YMM4 の格子は線の縁をなめらかに
        描くので出ない（SFっぽい吹き出し(右) の 45 度・12 画素の格子 #179）
        """
        grid = registry.require("brush_fill").create(
            pattern="grid",
            pattern_only=True,
            stops=2,
            color0=(1.0, 1.0, 1.0, 1.0),
            color1=(1.0, 1.0, 1.0, 0.0),
            thickness=1.3,
            cell_width=12,
            cell_height=12,
            angle=-45,
        )
        image = draw(white_square(180), (grid,)).astype(float)[..., 0]
        # 格子の 2 周期ほどの升目ごとの濃さ 升目の大きさは周期の倍数から外し、区切り方で
        # 濃さが揃って見えるのを避ける
        blocks = [
            image[y : y + 34, x : x + 34].mean()
            for y in range(15, 15 + 34 * 5, 34)
            for x in range(15, 15 + 34 * 5, 34)
        ]
        assert min(blocks) > 20, "格子が描かれていない"
        assert max(blocks) - min(blocks) < 10, f"升目ごとの濃さが揺れる: {np.round(blocks)}"

    def test_shadow_falls_in_the_requested_direction(self, draw: Callable[..., np.ndarray]) -> None:
        # Y は正が上 変形の pos_y と揃っていないと、同じ「Y」の表示で
        # 上下が反対に動くことになる
        square = white_square(60)
        effect = registry.require("shadow")
        down = draw(square, (effect.create(offset_y=-20, blur=0, opacity=100),))
        up = draw(square, (effect.create(offset_y=20, blur=0, opacity=100),))

        below = (HEIGHT // 2 + 45, WIDTH // 2)
        above = (HEIGHT // 2 - 45, WIDTH // 2)
        assert down[below][3] == 255
        assert down[below][0] < 20, "下に影が出ていない"
        assert up[above][0] < 20, "上に影が出ていない"

    def test_gradient_runs_from_one_colour_to_the_other(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 角度 0 が上から下、90 が右から左 AviUtl2 に両方を描かせて読み取った
        # 金色のテキストなどは 0 の向きで作られている
        #
        # 逆にすると、画面の端に寄せた配布物の文字が丸ごと終了色（多くは黒）になる
        square = white_square(120)

        def paint(angle: int) -> np.ndarray:
            return draw(
                square,
                (
                    registry.require("gradient").create(
                        start_color=(1.0, 0.0, 0.0, 1.0),
                        end_color=(0.0, 0.0, 1.0, 1.0),
                        angle=angle,
                        span=120,
                        strength=100,
                    ),
                ),
            )

        down = paint(0)
        assert down[HEIGHT // 2 - 50, WIDTH // 2][0] > down[HEIGHT // 2 - 50, WIDTH // 2][2]
        assert down[HEIGHT // 2 + 50, WIDTH // 2][2] > down[HEIGHT // 2 + 50, WIDTH // 2][0]

        sideways = paint(90)
        assert sideways[HEIGHT // 2, WIDTH // 2 + 50][0] > sideways[HEIGHT // 2, WIDTH // 2 + 50][2]
        assert sideways[HEIGHT // 2, WIDTH // 2 - 50][2] > sideways[HEIGHT // 2, WIDTH // 2 - 50][0]

    def test_the_middle_of_a_gradient_is_not_too_bright(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """赤から青の真ん中が、符号化した値の中点に来る

        リニアのまま混ぜると真ん中が 186 ほどの派手なマゼンタになる
        AviUtl2 の実物は 117 だった（配布物の中間色が全部違ってくる）
        """
        painted = draw(
            white_square(120),
            (
                registry.require("gradient").create(
                    start_color=(1.0, 0.0, 0.0, 1.0),
                    end_color=(0.0, 0.0, 1.0, 1.0),
                    angle=0,
                    span=120,
                    strength=100,
                ),
            ),
        )
        middle = painted[HEIGHT // 2, WIDTH // 2]
        assert 100 <= int(middle[0]) <= 150
        assert 100 <= int(middle[2]) <= 150

    def test_gradient_keeps_the_alpha(self, draw: Callable[..., np.ndarray]) -> None:
        # 色だけを塗り替える 不透明度まで触ると、図形の外まで色が付く
        square = white_square(60)
        painted = draw(
            square,
            (
                registry.require("gradient").create(
                    start_color=(1.0, 0.0, 0.0, 1.0), end_color=(1.0, 0.0, 0.0, 1.0)
                ),
            ),
        )
        assert painted[4, 4][:3].max() < 8, "図形の外まで塗られている"
        assert painted[HEIGHT // 2, WIDTH // 2][0] > 200

    def test_gradient_strength_blends_with_the_original(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        square = white_square(120)
        none = draw(square, (registry.require("gradient").create(strength=0),))
        assert centre(none)[0] > 240, "強さ 0 で色が変わっている"

    def test_noise_varies_over_time(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(120)
        effect = registry.require("noise").create(strength=60, animate=True)
        assert not np.array_equal(
            draw(square, (effect,), frame=0), draw(square, (effect,), frame=7)
        )

    def test_static_noise_is_stable(self, draw: Callable[..., np.ndarray]) -> None:
        square = white_square(120)
        effect = registry.require("noise").create(strength=60, animate=False)
        assert np.array_equal(draw(square, (effect,), frame=0), draw(square, (effect,), frame=7))


def _emboss(**values: ParamInput) -> Effect:
    return registry.require("emboss").create(**values)


def diagonal_slope() -> tuple[GeneratedSource, Effect]:
    """画面いっぱいの、左下が暗く右上へ向かって明るくなる坂

    グラデーションの角度 45 度は、始めの色が右上・終わりの色が左下に来る（試験の中で前提を確かめる）
    """
    plate = SHAPE.create(shape="rect", width=WIDTH, height=HEIGHT, color=(1.0, 1.0, 1.0, 1.0))
    ramp = registry.require("gradient").create(
        start_color=(1.0, 1.0, 1.0, 1.0),
        end_color=(0.0, 0.0, 0.0, 1.0),
        angle=45,
        span=400,
    )
    return plate, ramp


class TestEmboss:
    # 白い四角は 50..149 の画素に載る 縁から 2 画素内側を見る
    LEFT, RIGHT, TOP, BOTTOM = 52, 147, 52, 147
    MIDDLE = HEIGHT // 2

    def test_flat_picture_is_mid_grey_from_every_direction(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 平らな所に凹凸は無い 光の向きで明るさが変わると、ただの色の付け替えになる
        grey = SHAPE.create(shape="rect", width=WIDTH, height=HEIGHT, color=(0.3, 0.3, 0.3, 1.0))
        values = {
            angle: centre(draw(grey, (_emboss(angle=angle),)))[0] for angle in (0, 90, 135, -45)
        }
        assert len(set(values.values())) == 1, values
        assert 120 <= values[0] <= 136, "平らな所が中間の灰色にならない"

    def test_the_side_facing_the_light_is_bright(self, draw: Callable[..., np.ndarray]) -> None:
        # 浮き上がった四角に左から光を当てると、左の縁が照り、右の縁が陰になる
        # 逆だと、浮き彫りではなくへこみに見える
        lit = draw(white_square(), (_emboss(angle=180),))
        assert lit[self.MIDDLE, self.LEFT, 0] > 200
        assert lit[self.MIDDLE, self.RIGHT, 0] < 60
        assert 120 <= lit[self.MIDDLE, self.MIDDLE, 0] <= 136, "四角の中の平らな所は灰色のまま"

    def test_the_angle_turns_the_light(self, draw: Callable[..., np.ndarray]) -> None:
        # Y は上が正 90 度は上から光が来る 向きが逆だと、上下の陰影が入れ替わる
        square = white_square()
        from_right = draw(square, (_emboss(angle=0),))
        assert from_right[self.MIDDLE, self.RIGHT, 0] > 200
        assert from_right[self.MIDDLE, self.LEFT, 0] < 60
        from_above = draw(square, (_emboss(angle=90),))
        assert from_above[self.TOP, self.MIDDLE, 0] > 200
        assert from_above[self.BOTTOM, self.MIDDLE, 0] < 60

    def test_a_slope_is_bright_or_dark_by_the_light(self, draw: Callable[..., np.ndarray]) -> None:
        # 右上へ上る坂は、右上を向いた光から見ると陰になり、左下からの光で照る
        # 坂を横から照らすと（坂の向きと直交）、凹凸が出ず中間の灰色になる
        plate, ramp = diagonal_slope()
        plain = draw(plate, (ramp,))
        assert plain[60, 140, 0] > plain[140, 60, 0], "前提の坂が右上へ上っていない"
        strong = {"height": 500, "reach": 8}
        towards_top = centre(draw(plate, (ramp, _emboss(angle=45, **strong))))[0]
        towards_bottom = centre(draw(plate, (ramp, _emboss(angle=225, **strong))))[0]
        across = centre(draw(plate, (ramp, _emboss(angle=135, **strong))))[0]
        assert towards_bottom > 140
        assert towards_top < 116
        assert abs(across - 128) <= 6

    def test_zero_amount_keeps_the_picture(self, draw: Callable[..., np.ndarray]) -> None:
        # 量 0 は元の絵 リニアのまま混ぜるので、色の往復で 1 段ずれることもない
        plate, ramp = diagonal_slope()
        plain = draw(plate, (ramp,))
        assert np.array_equal(draw(plate, (ramp, _emboss(amount=0))), plain)

    def test_zero_height_keeps_the_colour(self, draw: Callable[..., np.ndarray]) -> None:
        # 元の色を残すときは、高さ 0 で凹凸が無くなり元の絵に戻る
        plate, ramp = diagonal_slope()
        plain = draw(plate, (ramp,)).astype(int)
        flat = draw(plate, (ramp, _emboss(height=0, keep_color=True))).astype(int)
        assert np.abs(flat - plain).max() <= 1

    def test_zero_height_keeps_a_colour_brighter_than_white(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 途中の絵は 1 を超える明るさを持てる 露出で白を 2 倍にしてから掛け、半分に戻して見る
        # 元の色を sRGB へ直して置き換えると 2 が 1 に詰められ、戻した後は白でなく灰色になる
        square = white_square(120)
        brighter = registry.require("exposure").create(amount=200)
        back = registry.require("exposure").create(amount=50)
        plain = draw(square, (brighter, back))
        kept = draw(square, (brighter, _emboss(height=0, keep_color=True), back))
        assert centre(plain)[0] > 250
        assert np.abs(kept.astype(int) - plain.astype(int)).max() <= 1

    def test_keep_colour_adds_the_relief_to_the_picture(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 元の色を残すと、平らな所は元の色のまま、照る縁だけ明るくなる
        orange = SHAPE.create(shape="rect", width=100, height=100, color=(0.9, 0.3, 0.1, 1.0))
        plain = draw(orange)
        kept = draw(orange, (_emboss(angle=180, keep_color=True),))
        middle = (self.MIDDLE, self.MIDDLE)
        assert np.abs(kept[middle][:3].astype(int) - plain[middle][:3].astype(int)).max() <= 1
        assert kept[self.MIDDLE, self.LEFT, 1] > plain[self.MIDDLE, self.LEFT, 1] + 40

    def test_alpha_basis_ignores_the_colour_inside(self, draw: Callable[..., np.ndarray]) -> None:
        # 不透明度を高さにすると、中の明暗の坂は平らに見え、外形の縁だけが浮く
        plate, ramp = diagonal_slope()
        by_alpha = draw(plate, (ramp, _emboss(basis="alpha", height=500, reach=8)))
        assert abs(centre(by_alpha)[0] - 128) <= 1

    def test_transparent_area_stays_transparent(self, draw: Callable[..., np.ndarray]) -> None:
        # 透明な所に灰色の面を敷くと、下の絵が隠れる 形は元の絵のまま残す
        # 書き出しの地は黒なので、外が黒のままなら何も置いていない
        embossed = draw(white_square(), (_emboss(),))
        assert embossed[10, 10, :3].max() == 0
        assert embossed[self.MIDDLE, self.MIDDLE, 0] > 100


class TestChromaKey:
    def test_removes_the_key_colour(self, draw: Callable[..., np.ndarray]) -> None:
        green = SHAPE.create(shape="rect", width=180, height=180, color=(0.0, 1.0, 0.0, 1.0))
        keyed = draw(
            green,
            (
                registry.require("chroma_key").create(
                    key_color=(0.0, 1.0, 0.0, 1.0), similarity=30, smoothness=10
                ),
            ),
        )
        assert centre(keyed)[1] < 30, "緑が抜けていない"

    def test_keeps_other_colours(self, draw: Callable[..., np.ndarray]) -> None:
        red = SHAPE.create(shape="rect", width=180, height=180, color=(1.0, 0.0, 0.0, 1.0))
        keyed = draw(
            red,
            (registry.require("chroma_key").create(key_color=(0.0, 1.0, 0.0, 1.0)),),
        )
        assert centre(keyed)[0] > 200


class TestKeyframes:
    def test_effect_parameters_animate(self, draw: Callable[..., np.ndarray]) -> None:
        # P2 の完了条件のもう半分 エフェクトの値が時間で変わること
        square = white_square(60)
        moving = Effect(
            kind="transform",
            params={
                **registry.require("transform").default_params(),
                "pos_x": AnimatedValue(
                    keyframes=(
                        Keyframe(frame=0, value=0.0),
                        Keyframe(frame=30, value=60.0),
                    )
                ),
            },
        )
        at_start = draw(square, (moving,), frame=0)
        at_end = draw(square, (moving,), frame=30)

        assert at_start[HEIGHT // 2, WIDTH // 2, 0] > 200
        assert at_end[HEIGHT // 2, WIDTH // 2, 0] < 20
        assert at_end[HEIGHT // 2, WIDTH // 2 + 60, 0] > 200

    def test_source_parameters_animate(self, draw: Callable[..., np.ndarray]) -> None:
        growing = GeneratedSource(
            kind="shape",
            params={
                **SHAPE.default_params(),
                "width": AnimatedValue(
                    keyframes=(
                        Keyframe(frame=0, value=20.0),
                        Keyframe(frame=30, value=180.0),
                    )
                ),
            },
        )
        assert lit_pixels(draw(growing, frame=30)) > lit_pixels(draw(growing, frame=0)) * 4


class TestSources:
    def test_text_draws_something(self) -> None:
        image = render_source(TEXT.create(text="あア亜A", size=48), WIDTH, HEIGHT)
        assert image is not None
        assert int((image[..., 3] > 0).sum()) > 0

    def test_empty_text_draws_nothing(self) -> None:
        image = render_source(TEXT.create(text=""), WIDTH, HEIGHT)
        assert image is not None
        assert int((image[..., 3] > 0).sum()) == 0

    def test_text_border_widens_the_glyphs(self) -> None:
        plain = render_source(TEXT.create(text="あ", size=64, border_width=0), WIDTH, HEIGHT)
        outlined = render_source(TEXT.create(text="あ", size=64, border_width=6), WIDTH, HEIGHT)
        assert plain is not None
        assert outlined is not None
        assert int((outlined[..., 3] > 0).sum()) > int((plain[..., 3] > 0).sum())

    @pytest.mark.parametrize("shape", ["rect", "rounded", "ellipse", "triangle", "star"])
    def test_every_shape_draws(self, shape: str) -> None:
        image = render_source(SHAPE.create(shape=shape, width=120, height=120), WIDTH, HEIGHT)
        assert image is not None
        assert int((image[..., 3] > 0).sum()) > 0

    def test_shape_size_is_respected(self) -> None:
        small = render_source(SHAPE.create(shape="rect", width=40, height=40), WIDTH, HEIGHT)
        large = render_source(SHAPE.create(shape="rect", width=80, height=80), WIDTH, HEIGHT)
        assert small is not None
        assert large is not None
        assert int((large[..., 3] > 0).sum()) == pytest.approx(
            int((small[..., 3] > 0).sum()) * 4, rel=0.1
        )

    def test_unknown_source(self) -> None:
        assert render_source(GeneratedSource(kind="なにか"), WIDTH, HEIGHT) is None

    def test_image_has_no_row_padding(self) -> None:
        # QImage は行ごとに詰め物を入れることがある 幅だけで整形すると絵が斜めにずれる
        image = render_source(SHAPE.create(shape="rect", width=10, height=10), 101, 51)
        assert image is not None
        assert image.shape == (51, 101, 4)

    def test_registry(self) -> None:
        assert "text" in source_registry
        assert "shape" in source_registry
        assert source_registry.get("なにか") is None


class TestColourConversion:
    def test_srgb_to_linear(self) -> None:
        assert srgb_to_linear(0.0) == 0.0
        assert srgb_to_linear(1.0) == pytest.approx(1.0)
        # sRGB の中間 (0.5) はリニアでは 0.21 前後 ここを取り違えると、
        # 色パラメータを指定した縁取りや影の色が明るく出る
        assert srgb_to_linear(0.5) == pytest.approx(0.2140, abs=0.001)


class TestBlendModes:
    def test_add_is_brighter_than_normal(self, gl: OffscreenGLContext) -> None:
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
        grey = SHAPE.create(shape="rect", width=180, height=180, color=(0.5, 0.5, 0.5, 1.0))

        def stack(blend: str) -> np.ndarray:
            lower = Track(
                kind=TrackKind.VIDEO,
                clips=(Clip(timeline_start=0, duration=30, source=grey),),
            )
            upper = Track(
                kind=TrackKind.VIDEO,
                clips=(Clip(timeline_start=0, duration=30, source=grey, blend_mode=blend),),
            )
            stacked = project.with_timeline(
                project.timeline.__class__(rate=project.rate, tracks=(lower, upper))
            )
            renderer = FrameRenderer(stacked, context=gl)
            try:
                return renderer.render(0)
            finally:
                renderer.close()

        assert centre(stack(BlendMode.ADD))[0] > centre(stack(BlendMode.NORMAL))[0]
