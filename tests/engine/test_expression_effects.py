"""動きと加工のエフェクト（YMM4 の配布物に合わせて足したもの）

シェーダは書き間違えても例外にならず、コンパイルに失敗したエフェクトは黙って
素通しになる（再生を止めないため） だから「全部コンパイルできる」ことを
まず押さえ、効き方の要所は画素で見る
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from sashimono.core.model import Effect
from sashimono.effects.definition import registry
from sashimono.effects.motion import register_motion_effects
from sashimono.effects.spec import GridSpec
from sashimono.effects.stylize import register_stylize_effects
from sashimono.engine.gpu import (
    Compositor,
    EffectProcessor,
    GLContextError,
    OffscreenGLContext,
    Texture,
)
from sashimono.engine.gpu.glutil import Framebuffer

SIZE = 64

#: このフェーズで足した種類 1 つでも登録が漏れると、YMM4 のテンプレートを
#: 読んでも写す先が無く、記録に残るだけになる
NEW_KINDS = (
    "random_move",
    "random_zoom",
    "random_rotate",
    "repeat_move",
    "repeat_rotate",
    "repeat_opacity",
    "inout_move",
    "inout_zoom",
    "inout_jump",
    "inout_getup",
    "skew",
    "spiral",
    "wave",
    "mesh_deform",
    "circular_duplicate",
    "crash",
    "noise_displacement",
    "morphology",
    "crop_angle",
    "crop_slant",
    "round_corner",
    "edge_trim",
    "exposure",
    "halftone_border",
    "stripe_glitch",
    "long_shadow",
    "highlights_shadows",
    "color_shift",
    "radial_blur",
    "circular_blur",
    "invert",
    "tint",
    "edge_detect",
)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def processor(gl_context: OffscreenGLContext) -> Iterator[EffectProcessor]:
    with gl_context:
        compositor = Compositor(SIZE, SIZE)
        created = EffectProcessor(SIZE, SIZE, compositor.quad)
        yield created
        created.release()
        compositor.release()


def _square(value: int = 200, inner: int = 32) -> np.ndarray:
    """透明な地に、真ん中 ``inner`` 四方の灰色の四角"""
    image = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    start = (SIZE - inner) // 2
    image[start : start + inner, start : start + inner] = (value, value, value, 255)
    return image


def _run(
    gl_context: OffscreenGLContext,
    processor: EffectProcessor,
    effect: Effect,
    *,
    image: np.ndarray | None = None,
    frame: int = 0,
    duration: int = 60,
) -> np.ndarray:
    """エフェクトを掛けた結果を ``(高さ, 幅, 4)`` の float（リニア、GL の向き）で返す"""
    picture = image if image is not None else _square()
    columns = np.flatnonzero(picture[..., 3].any(axis=0))
    rows = np.flatnonzero(picture[..., 3].any(axis=1))
    # レンダラと同じく、色の付いた範囲を絵の大きさとして渡す GL の向きへ直す分は
    # エフェクトの側でやるので、ここは画像の向き（上が 0）のまま
    bounds = (
        float(columns[0]),
        float(SIZE - rows[-1] - 1),
        float(columns[-1] + 1),
        float(SIZE - rows[0]),
    )
    with gl_context:
        texture = Texture.from_array(picture)
        result: Framebuffer = processor.apply(
            texture,
            (effect,),
            frame=frame,
            fps=30.0,
            duration=duration,
            flip_source=False,
            bounds=bounds,
        )
        from OpenGL import GL

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, result.handle)
        raw = GL.glReadPixels(0, 0, SIZE, SIZE, GL.GL_RGBA, GL.GL_FLOAT)
        texture.release()
    return np.frombuffer(raw, dtype=np.float32).reshape(SIZE, SIZE, 4).copy()


def _make(kind: str, **params: object) -> Effect:
    definition = registry.require(kind)
    return definition.create(**params)  # type: ignore[arg-type]


class TestRegistration:
    def test_every_new_kind_is_registered(self) -> None:
        # 登録が漏れた種類は、YMM4 のテンプレートを読んでも写す先が無く、
        # その動きや加工が黙って抜ける
        missing = [kind for kind in NEW_KINDS if kind not in registry]
        assert missing == []

    def test_registering_twice_is_harmless(self) -> None:
        # 読み込みのたびに呼ばれても「すでに登録されている」で落ちないこと
        register_motion_effects()
        register_stylize_effects()

    @pytest.mark.parametrize("kind", NEW_KINDS)
    def test_every_shader_compiles(self, processor: EffectProcessor, kind: str) -> None:
        # コンパイルに失敗すると黙って素通しになり、効かない理由が見えない
        assert processor.has_work((_make(kind),)), f"{kind} のシェーダがコンパイルできない"

    def test_the_transform_shader_still_compiles(self, processor: EffectProcessor) -> None:
        # 立体の回転と中心の選び方を足した変形が壊れると、既存の位置指定が全部効かなくなる
        assert processor.has_work((_make("transform", rotation_y=30, pivot_h="left"),))


def _alpha_sum(pixels: np.ndarray) -> float:
    return float(pixels[..., 3].sum())


class TestPixels:
    def test_exposure_scales_the_light(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 100% が元のまま 200% で倍 取り違えると配布物の「光る登場」が暗転になる
        base = _run(gl_context, processor, _make("exposure", amount=100))
        doubled = _run(gl_context, processor, _make("exposure", amount=200))
        assert doubled[32, 32, 0] == pytest.approx(base[32, 32, 0] * 2, rel=0.02)

    def test_invert_turns_black_to_white(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 反転しないと、ネガのような演出の配布テンプレートが元の色のまま出る
        dark = _run(gl_context, processor, _make("invert"), image=_square(value=0))
        assert dark[32, 32, 0] == pytest.approx(1.0, abs=0.01)

    def test_the_slant_clip_drops_the_lower_side_at_zero(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        """斜めクリッピング は角度 0 で中心より下を落とす（#170）

        向きは sigma の 単純図形σ の 菱形 が 4 回の切り落としで角を落とす並びから読んだ
        取り違えると、角ではなく真ん中を落として菱形が消える（結果は GL の向きで、行 0 が下）
        """
        result = _run(gl_context, processor, _make("crop_slant", angle=0, blur=0))
        assert result[40, 32, 3] > 0.9
        assert result[24, 32, 3] < 0.1

    def test_the_slant_clip_turns_clockwise(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 90 度で落とす側が下から左へ回る 反時計回りだと右が落ちる
        result = _run(gl_context, processor, _make("crop_slant", angle=90, blur=0))
        assert result[32, 24, 3] < 0.1
        assert result[32, 40, 3] > 0.9

    def test_the_slant_clip_moves_with_its_centre(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 中心 Y は上が正 -8 で線が下へ 8 動き、真ん中より少し下の行も残る側に入る
        result = _run(gl_context, processor, _make("crop_slant", center_y=-8, blur=0))
        assert result[28, 32, 3] > 0.9
        assert result[20, 32, 3] < 0.1

    def test_a_fill_can_keep_the_brightness(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        """単色化 の 輝度を保持する 色は付くが、明るさは元の絵のまま（#170）

        保たないと、下の絵を透かすアクリル矩形が塗った色一色の板になる
        """
        grey = _run(gl_context, processor, _make("fill", amount=0))
        kept = _run(
            gl_context,
            processor,
            _make("fill", color=(1.0, 0.0, 0.0, 1.0), amount=100, keep_luma=True),
        )
        luma = (0.2126, 0.7152, 0.0722)

        def brightness(pixel: np.ndarray) -> float:
            return float(sum(weight * value for weight, value in zip(luma, pixel[:3], strict=True)))

        assert brightness(kept[32, 32]) == pytest.approx(brightness(grey[32, 32]), abs=0.02)
        assert kept[32, 32, 0] > kept[32, 32, 1]

    def test_the_colour_offset_adds_to_the_encoded_value(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        """色調補正 の 明るさを足す は sRGB の値へ足す（AviUtl の 明るさ の写し先 #170）

        輝度 30% と 明るさ +35 で、灰色 200 は 200/255 x 0.3 + 0.35 = 0.585 になる
        倍率で代わりにすると 0.35 前後まで暗く沈む
        """
        result = _run(gl_context, processor, _make("color", gain=30, offset=35))
        encoded = 0.585
        linear = ((encoded + 0.055) / 1.055) ** 2.4
        assert result[32, 32, 0] == pytest.approx(linear, abs=0.02)

    def test_round_corner_cuts_the_corners(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 角が残ると、角丸の字幕帯が四角いまま出る
        result = _run(gl_context, processor, _make("round_corner", radius=12, blur=0))
        assert result[16, 16, 3] < 0.1
        assert result[32, 32, 3] > 0.9

    def test_inout_move_starts_off_screen(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 始まりで画面の中に残っていると、「外から入ってくる」ように見えない
        effect = _make("inout_move", direction="top", effect_time=1.0)
        assert _alpha_sum(_run(gl_context, processor, effect, frame=0)) == 0.0
        settled = _run(gl_context, processor, effect, frame=45)
        assert _alpha_sum(settled) == pytest.approx(32 * 32, rel=0.05)

    def test_inout_zoom_can_squash_one_axis(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 軸の割合は隠れたときのその向きの大きさ X=0 Y=100 は横だけ潰れた所から広がる
        # （YMM4 に描かせて確かめた 「広がって登場」の配布物がこの値）
        effect = _make("inout_zoom", zoom=100, zoom_x=0, zoom_y=100, effect_time=1.0)
        halfway = _run(gl_context, processor, effect, frame=15)
        rows = np.nonzero(halfway[..., 3].max(axis=1) > 0.5)[0]
        columns = np.nonzero(halfway[..., 3].max(axis=0) > 0.5)[0]
        assert len(rows) == pytest.approx(32, abs=2)
        assert 0 < len(columns) < 32

    def test_the_exit_counts_back_from_the_end(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 長さを知らないと、退場が始まらずに最後まで表示されたままになる
        effect = _make("inout_zoom", effect_in=False, effect_out=True, effect_time=0.5)
        assert _alpha_sum(_run(gl_context, processor, effect, frame=10, duration=60)) > 0
        assert _alpha_sum(_run(gl_context, processor, effect, frame=59, duration=60)) < 50

    def test_circular_duplicate_makes_copies(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 複製が 1 つに重なると、円形に並べる演出が 1 つの絵にしか見えない
        small = _square(inner=6)
        effect = _make("circular_duplicate", count=4, radius=20, synced=False)
        result = _run(gl_context, processor, effect, image=small)
        assert _alpha_sum(result) == pytest.approx(4 * 36, rel=0.2)

    def test_mesh_without_offsets_changes_nothing(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 動かしていない四隅で絵がずれると、配布物の「変形なし」の設定で絵が崩れる
        result = _run(gl_context, processor, _make("mesh_deform"))
        assert _alpha_sum(result) == pytest.approx(32 * 32, rel=0.05)
        assert result[32, 32, 3] > 0.9

    def test_skewing_both_ways_keeps_the_picture(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 行列式の符号を取り違えると、縦横とも 45 度で割る数が 0 になり絵が消える
        effect = _make("skew", angle_x=45, angle_y=45)
        assert _alpha_sum(_run(gl_context, processor, effect)) > 100

    def test_a_collapsed_mesh_draws_nothing_broken(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 四隅を 1 本の線へ潰すと割る数が 0 になる NaN を描くと、重ねた先の色まで壊れる
        effect = _make("mesh_deform", point1_x=-32, point2_x=-32)
        result = _run(gl_context, processor, effect)
        assert np.all(np.isfinite(result))

    def test_long_shadow_extends_behind(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 影が後ろへ伸びないか本体を塗り潰すと、長い影の字幕が読めなくなる
        effect = _make("long_shadow", angle=0, length_=12, color1=(0.0, 0.0, 0.0, 1.0))
        result = _run(gl_context, processor, effect)
        assert result[32, 52, 3] > 0.9
        assert result[32, 32, 0] > 0.5

    def test_repeat_opacity_swings(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 揺れないと、点滅させる配布テンプレートが表示されたまま止まる
        effect = _make("repeat_opacity", opacity=0, interval=1.0)
        start = _run(gl_context, processor, effect, frame=0)
        middle = _run(gl_context, processor, effect, frame=15)
        assert start[32, 32, 3] > 0.9
        assert middle[32, 32, 3] < 0.1

    def test_a_tilted_transform_narrows(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # YMM4 の RotateEffect の X / Y は板を傾ける 平らなままだと立体の動きが消える
        result = _run(gl_context, processor, _make("transform", rotation_y=60))
        columns = np.nonzero(result[..., 3].max(axis=0) > 0.5)[0]
        assert len(columns) == pytest.approx(16, abs=3)

    def test_the_pivot_can_sit_on_the_object_edge(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 左端を中心に半分へ縮めると、左端は動かない 画面の中央を中心にすると左端が動く
        result = _run(gl_context, processor, _make("transform", scale=50, pivot_h="left"))
        columns = np.nonzero(result[..., 3].max(axis=0) > 0.5)[0]
        assert int(columns.min()) == 16


def _grid(columns: int, rows: int, moved: dict[int, tuple[float, float]]) -> tuple[float, ...]:
    """格子の値を組む 番号は左上から行ごと 動かさない点は 0"""
    offsets = [0.0] * (columns * rows * 2)
    for index, (x, y) in moved.items():
        offsets[index * 2] = x
        offsets[index * 2 + 1] = y
    return (float(columns), float(rows), *offsets)


def _ramp(inner: int = 32) -> np.ndarray:
    """透明な地に、上から下へ明るくなる ``inner`` 四方の四角

    一様な灰色だと、絵がどれだけ縦にずれても画素の値が変わらず、
    格子の効き方を画素で確かめられない
    """
    image = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    start = (SIZE - inner) // 2
    for row in range(inner):
        level = 8 + row * 7
        image[start + row, start : start + inner] = (level, level, level, 255)
    return image


class TestMeshGrid:
    """2x2 より細かい格子（Issue #31）"""

    def test_a_grid_that_does_not_match_its_counts_is_dropped(self) -> None:
        # 点数と点の数が食い違う値を渡すと、シェーダが配列の外を読む
        spec = registry.require("mesh_deform").spec("grid")
        assert isinstance(spec, GridSpec)
        assert spec.coerce((3.0, 3.0, 0.0, 0.0)) == ()
        assert spec.coerce((1.0, 1.0)) == ()
        assert spec.coerce((99.0, 99.0, *([0.0] * (99 * 99 * 2)))) == ()
        assert spec.coerce((2.0, 2.0, *([float("nan")] * 8))) == ()
        assert spec.size(spec.coerce(None)) == (0, 0)

    def test_a_grid_size_that_is_not_a_whole_number_is_dropped(self) -> None:
        """点数を先に丸めると、2.9 が 2 扱いで通って別の格子になる

        点の数の検査をすり抜けるので、絵が畳まれたまま出る
        """
        spec = registry.require("mesh_deform").spec("grid")
        assert isinstance(spec, GridSpec)
        assert spec.coerce((2.9, 2.0, *([0.0] * 8))) == ()
        assert spec.coerce((3.0, 2.5, *([0.0] * 12))) == ()

    def test_a_grid_size_that_is_not_a_number_is_dropped(self) -> None:
        """整える所から例外が出ると、プロジェクトを開く所で落ちる

        `float("x")` も `int(float("nan"))` も例外になる 辻褄が合わない値は
        「格子なし」へ戻すのが約束なので、例外にしてはいけない
        """
        spec = registry.require("mesh_deform").spec("grid")
        assert isinstance(spec, GridSpec)
        # 型の付かない値が来る道（保存ファイル・プリセット・AI の書き換え）を真似る
        letters: Any = ("x",) * 18
        empties: Any = (None,) * 8
        assert spec.coerce((3.0, 3.0, *letters)) == ()
        assert spec.coerce((float("nan"), 2.0, *([0.0] * 8))) == ()
        assert spec.coerce((float("inf"), 2.0, *([0.0] * 8))) == ()
        assert spec.coerce((2.0, 2.0, *empties)) == ()
        # Python の整数には桁の上限が無い float へ直す所で落ちると、
        # 設定画面も描画も開けなくなる
        huge: Any = (10**1000,) * 8
        assert spec.coerce((2.0, 2.0, *huge)) == ()

    def test_a_flat_grid_leaves_the_picture_alone(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 動かしていない格子で絵がずれると、YMM4 の「変形なし」の 5x5 が崩れて出る
        effect = _make("mesh_deform", grid=_grid(5, 5, {}))
        result = _run(gl_context, processor, effect)
        assert _alpha_sum(result) == pytest.approx(32 * 32, rel=0.05)
        assert result[32, 32, 3] > 0.9

    def test_moving_the_middle_point_leaves_the_corners_alone(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # セルを 1 つしか見ない作りだと、真ん中の点が効かないか、端まで一緒に歪む
        picture = _ramp()
        flat = _run(
            gl_context, processor, _make("mesh_deform", grid=_grid(3, 3, {})), image=picture
        )
        # 3x3 の真ん中（番号 4）だけを上へ 16 px
        bent = _run(
            gl_context,
            processor,
            _make("mesh_deform", grid=_grid(3, 3, {4: (0.0, 16.0)})),
            image=picture,
        )
        middle = abs(float(bent[32, 32, 0]) - float(flat[32, 32, 0]))
        corner = abs(float(bent[18, 18, 0]) - float(flat[18, 18, 0]))
        assert middle > 0.02, "真ん中の点が効いていない"
        assert corner < middle / 5.0, "四隅の近くまで一緒に歪んでいる"

    def test_a_collapsed_finer_mesh_draws_nothing_broken(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 列を 1 本の線へ潰し、1 つのセルを裏返す 割る数が 0 になったり、
        # 根号の中が負になったりする NaN を描くと、重ねた先の色まで壊れる
        collapsed = _grid(
            3, 3, {index: (16.0 if index % 3 == 0 else -16.0, 0.0) for index in range(9)}
        )
        result = _run(gl_context, processor, _make("mesh_deform", grid=collapsed))
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize(
        ("slider", "moved", "still"),
        [
            # 返ってくる配列では行 45 が下側 列 45 が右（測って確かめた位置）
            ("point2_y", (45, 45), (45, 18)),
            ("point3_y", (45, 18), (45, 45)),
        ],
    )
    def test_the_old_corner_sliders_move_their_own_corner(
        self,
        gl_context: OffscreenGLContext,
        processor: EffectProcessor,
        slider: str,
        moved: tuple[int, int],
        still: tuple[int, int],
    ) -> None:
        """スライダは一周の順（右下が 2・左下が 3）、格子は行の順（左下が 2・右下が 3）

        並べ替えを取り違えると、既存のプロジェクトで右下を動かしたはずが
        左下が動く 左上だけを動かす試験では、2 と 3 の入れ替わりを見つけられない
        """
        picture = _ramp()
        flat = _run(gl_context, processor, _make("mesh_deform"), image=picture)
        bent = _run(gl_context, processor, _make("mesh_deform", **{slider: 8.0}), image=picture)
        near = abs(float(bent[moved][0]) - float(flat[moved][0]))
        far = abs(float(bent[still][0]) - float(flat[still][0]))
        assert near > 0.02, f"{slider} が効いていない"
        assert far < near / 5.0, f"{slider} で反対側の角が動いている"


class TestAxes:
    """Y は上が正 例外は表示名に「下が正」と書いてあるものだけ"""

    def test_only_the_two_exceptions_say_down_is_positive(self) -> None:
        # 例外は YMM4 の値をそのまま持つ 2 つだけ（CLAUDE.md） ほかに「下が正」と書くと、
        # シェーダは上を正として読むのに、表示を信じた人が逆向きの値を入れる
        # 図形で切り抜く の Y がそうなっていた（#173）
        import sashimono.effects  # noqa: F401 - 全部のエフェクトを登録させる

        # 等しいかで比べる 含まれるかだけだと、例外の 2 つから書き添えが消えても気付けない
        labelled = {
            (definition.kind, spec.name)
            for definition in registry.all()
            for spec in definition.parameters
            if "下が正" in spec.label
        }
        assert labelled == {("brush_fill", "center_y"), ("particles", "emitter_y")}

    def test_the_mask_centre_moves_up_with_a_positive_y(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 逆向きだと、YMM4 から写した切り抜きが上下反対の場所に出る
        picture = np.full((SIZE, SIZE, 4), 200, dtype=np.uint8)
        effect = _make("shape_mask", shape="ellipse", width=16.0, height=16.0, center_y=24.0)
        result = _run(gl_context, processor, effect, image=picture)
        # GL の向きなので、行番号が大きいほど画面の上
        rows = np.flatnonzero(result[..., 3].max(axis=1) > 0.5)
        assert rows.mean() > SIZE / 2

    def test_a_negative_emitter_y_puts_particles_above_the_centre(
        self, gl_context: OffscreenGLContext, processor: EffectProcessor
    ) -> None:
        # 放つ位置だけは YMM4 と同じ下向き正（表示名にもそう書いてある）
        effect = _make(
            "particles",
            rate=60.0,
            lifetime=1.0,
            size=100.0,
            emitter_y=-20.0,
            speed=0.0,
            gravity=0.0,
            randomness=0.0,
        )
        result = _run(gl_context, processor, effect)
        rows = np.flatnonzero(result[..., 3].max(axis=1) > 0.5)
        assert len(rows), "粒が 1 つも出ていない"
        # 下向き正なので -20 は画面の上（GL の向きでは行番号が大きい側）
        assert rows.mean() > SIZE / 2


class TestFrameBuffer:
    def test_it_reads_what_was_drawn_below(self, gl_context: OffscreenGLContext) -> None:
        # 下の絵を読めないと、YMM4 の「背景だけぼかす帯」が黒い板になる
        from sashimono.core.commands import AddClip, AddEffect, AddTrack
        from sashimono.core.model import (
            Clip,
            GeneratedSource,
            Project,
            ProjectSettings,
            Track,
            TrackKind,
        )
        from sashimono.core.timebase import FrameRate
        from sashimono.engine.render import FrameRenderer

        project = Project.create(ProjectSettings(width=64, height=64, frame_rate=FrameRate(30)))
        below, above = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
        project = AddTrack(below).apply(project)
        project = AddTrack(above).apply(project)
        white = GeneratedSource(
            kind="shape",
            params={"shape": "background", "color": (1.0, 1.0, 1.0, 1.0)},
        )
        project = AddClip(below.id, Clip(timeline_start=0, duration=10, source=white)).apply(
            project
        )
        grab = Clip(timeline_start=0, duration=10, source=GeneratedSource(kind="framebuffer"))
        project = AddClip(above.id, grab).apply(project)
        project = AddEffect(grab.id, _make("invert")).apply(project)

        renderer = FrameRenderer(project, context=gl_context)
        try:
            image = renderer.render(0)
        finally:
            renderer.close()
        assert int(image[32, 32, 0]) < 8
