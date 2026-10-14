"""テキストの塗り・縁取りの層・影のグラデーションを描いた絵で見る（#312）

字は「■」を並べて、塗りの中の色を読みやすくする 基準は字の枠（字の輪郭を囲む四角）なので、
枠の上端の色が位置 0 の止まり、下端の色が位置 1 の止まりになる（角度 0）
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.core.model import AnimatedValue, GeneratedSource, Stroke
from sashimono.effects.sources import TEXT
from sashimono.effects.spec import ParamInput
from sashimono.effects.strokes import STROKE
from sashimono.engine.sources import render_source_framed, source_canvas

SCREEN = (640, 360)
RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)


def flat(*stops: tuple[float, tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(number for position, colour in stops for number in (position, *colour))


RED_TO_BLUE = flat((0.0, RED), (1.0, BLUE))


def text(**params: ParamInput) -> GeneratedSource:
    base: dict[str, ParamInput] = {"text": "■■", "size": 160.0, "color": WHITE}
    base.update(params)
    return TEXT.create(**base)


def drawn(source: GeneratedSource, scale: float = 1.0) -> np.ndarray:
    """描く道と同じく、:func:`source_canvas` の大きさの絵に描く"""
    width, height = round(SCREEN[0] * scale), round(SCREEN[1] * scale)
    canvas = source_canvas(source, width, height, scale=(scale, scale))
    image = render_source_framed(source, *canvas, scale=(scale, scale), screen=(width, height))[0]
    assert image is not None
    return image


def solid(image: np.ndarray) -> tuple[int, int, int, int]:
    """不透明に塗れた所を囲む四角（左・上・右・下 右と下は含まない）"""
    ys, xs = np.nonzero(image[..., 3] == 255)
    assert len(xs), "何も描かれていない"
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def rgb(image: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    red, green, blue = (int(v) for v in image[y, x, :3])
    return red, green, blue


def near(found: tuple[int, ...], expected: tuple[float, ...], tolerance: int = 8) -> bool:
    return all(
        abs(a - round(b * 255)) <= tolerance for a, b in zip(found, expected[:3], strict=False)
    )


def first_glyph_x(image: np.ndarray) -> int:
    """1 文字目の真ん中あたりの列（字と字の間のすき間を避ける）"""
    left, _top, right, _bottom = solid(image)
    return left + (right - left) // 8


def column_colours(image: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """塗りの中の真ん中の列の、一番上と一番下の色"""
    x = first_glyph_x(image)
    _left, top, _right, bottom = solid(image)
    column = [y for y in range(top, bottom) if image[y, x, 3] == 255]
    return rgb(image, x, column[1]), rgb(image, x, column[-2])


class TestSolidStaysAsItWas:
    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"border_width": 6.0, "border_color": BLUE},
            {"shadow_x": 8.0, "shadow_y": -8.0, "shadow_blur": 3.0, "shadow_color": RED},
        ],
    )
    def test_an_old_text_draws_pixel_for_pixel_the_same(self, extra: dict[str, ParamInput]) -> None:
        # 単色の字が 1 画素でも変わると、前の版の作品を開いただけで字幕の縁や色がずれる
        old = text(**extra)
        old = GeneratedSource(
            kind="text",
            params={
                name: value
                for name, value in old.params.items()
                if not name.startswith(("fill_", "shadow_fill_"))
            },
        )
        # 種類が単色なら、グラデーションの値が入っていても読まない
        solid_with_values = text(
            **extra,
            fill_type="solid",
            fill_gradient=RED_TO_BLUE,
            shadow_fill_type="solid",
            shadow_fill_gradient=RED_TO_BLUE,
        )
        assert np.array_equal(drawn(old), drawn(solid_with_values))

    def test_a_layered_text_draws_the_same(self) -> None:
        # 縁取りの層を持つ前の版の字も、層の塗りの項目が無いまま同じに描けないと縁が変わる
        layer = Stroke(params={"width": AnimatedValue(5.0), "color": BLUE})
        with_values = Stroke(
            params={**layer.params, "fill_type": "solid", "fill_gradient": RED_TO_BLUE}
        )
        old = drawn(text().with_strokes((layer,)))
        assert np.array_equal(old, drawn(text().with_strokes((with_values,))))


class TestLinear:
    def test_angle_zero_runs_from_the_top_to_the_bottom(self) -> None:
        # 向きが逆だと、上を明るくしたテロップが下から明るくなる
        image = drawn(text(fill_type="linear", fill_gradient=RED_TO_BLUE))
        top, bottom = column_colours(image)
        assert near(top, RED, 16) and near(bottom, BLUE, 16), (top, bottom)

    def test_angle_ninety_starts_on_the_right(self) -> None:
        # 「グラデーション」エフェクト（AviUtl2 で測った向き）と同じでないと、同じ角度で向きが違う
        image = drawn(text(fill_type="linear", fill_gradient=RED_TO_BLUE, fill_angle=90.0))
        left, top, right, bottom = solid(image)
        y = (top + bottom) // 2
        row = [x for x in range(left, right) if image[y, x, 3] == 255]
        assert near(rgb(image, row[-2], y), RED, 16)
        assert near(rgb(image, row[1], y), BLUE, 16)

    def test_stops_in_between_and_their_order(self) -> None:
        # 真ん中の止まりが効かないと、3 色以上のグラデーション（金属風の文字など）が作れない
        three = flat((0.0, RED), (0.5, GREEN), (1.0, BLUE))
        image = drawn(text(fill_type="linear", fill_gradient=three))
        _left, top, _right, bottom = solid(image)
        x, middle = first_glyph_x(image), (top + bottom) // 2
        assert near(rgb(image, x, middle), GREEN, 24)
        # 止まりを動かすと、真ん中の色も動く（緑を下へ寄せると真ん中は赤と緑の間）
        moved = flat((0.0, RED), (0.9, GREEN), (1.0, BLUE))
        shifted = drawn(text(fill_type="linear", fill_gradient=moved))
        red, green, blue = rgb(shifted, x, middle)
        assert red > 80 and green > 80 and blue < 30, (red, green, blue)
        # 真ん中の止まりを消すと、赤と青の間の紫になる
        removed = drawn(text(fill_type="linear", fill_gradient=RED_TO_BLUE))
        red, green, blue = rgb(removed, x, middle)
        assert green < 20 and abs(red - blue) < 40, (red, green, blue)

    def test_a_see_through_stop(self) -> None:
        # 不透明度つきの止まりが効かないと、下から透けていくテロップが作れない
        # 色と不透明度は別々に混ぜる（YMM4 と同じ） 事前乗算で混ぜると真ん中が赤のまま薄くなる
        fading = flat((0.0, RED), (1.0, (0.0, 0.0, 1.0, 0.0)))
        image = drawn(text(fill_type="linear", fill_gradient=fading))
        ys, xs = np.nonzero(image[..., 3] > 0)
        x = int(xs.min()) + (int(xs.max()) - int(xs.min())) // 8
        column = [y for y in range(int(ys.min()), int(ys.max()) + 1) if image[y, x, 3] > 0]
        alphas = [int(image[y, x, 3]) for y in column]
        assert alphas[1] > 240 and alphas[-2] < 15, alphas
        middle = column[len(column) // 2]
        assert 100 < int(image[middle, x, 3]) < 160
        red, green, blue = rgb(image, x, middle)
        assert green < 10 and abs(red - blue) < 50, (red, green, blue)

    def test_the_length_and_shift(self) -> None:
        # 長さが効かないと、端の色をはっきり出す（帯を短くする）ことができない
        short = drawn(text(fill_type="linear", fill_gradient=RED_TO_BLUE, fill_length=10.0))
        _left, top, _right, bottom = solid(short)
        x = first_glyph_x(short)
        # 長さ 10% なら、真ん中の少し上はもう赤、少し下はもう青
        assert near(rgb(short, x, (top + bottom) // 2 - 30), RED, 10)
        assert near(rgb(short, x, (top + bottom) // 2 + 30), BLUE, 10)
        # ずらし 50% で、真ん中の色が上の端の色になる（グラデーションが下へ動く）
        shifted = drawn(
            text(fill_type="linear", fill_gradient=RED_TO_BLUE, fill_length=10.0, fill_shift=40.0)
        )
        assert near(rgb(shifted, x, (top + bottom) // 2 + 30), RED, 10)

    def test_the_width_basis(self) -> None:
        # YMM4 の割合の指定は、角度に関わらず絵の横の幅が 100 枠の向きの長さで測ると
        # 横に長い字幕で縦のグラデーションの長さが変わり、読み込んだ字の色が合わない
        wide = drawn(
            text(
                text="■■■■",
                fill_type="linear",
                fill_gradient=RED_TO_BLUE,
                fill_length=50.0,
                fill_span="width",
            )
        )
        left, top, right, bottom = solid(wide)
        # 幅の半分（約 320 画素）は字の高さ（約 160 画素）より長いので、上と下は中ほどの色
        top_colour, bottom_colour = column_colours(wide)
        assert not near(top_colour, RED, 40) and not near(bottom_colour, BLUE, 40)
        assert right - left > 2 * (bottom - top)


class TestRadial:
    def test_the_centre_is_the_first_stop(self) -> None:
        # 円形の中心が字の枠の真ん中に来ないと、中心から光る字が片寄る
        image = drawn(text(text="■", fill_type="radial", fill_gradient=RED_TO_BLUE))
        left, top, right, bottom = solid(image)
        x, y = (left + right) // 2, (top + bottom) // 2
        assert near(rgb(image, x, y), RED, 12)
        assert near(rgb(image, left + 1, top + 1), BLUE, 30)
        # 円の中心が画素の真ん中に重なると、Qt はその 1 画素を塗らずに透明な点を残す
        assert (image[y - 2 : y + 3, x - 2 : x + 3, 3] == 255).all()

    def test_the_centre_moves_up_with_a_positive_y(self) -> None:
        # Y は上が正（テキストの位置と同じ） 逆だと、上へ寄せたつもりの光が下へ行く
        image = drawn(
            text(text="■", fill_type="radial", fill_gradient=RED_TO_BLUE, fill_center_y=40.0)
        )
        left, top, right, bottom = solid(image)
        x = (left + right) // 2
        assert rgb(image, x, top + 3)[0] > rgb(image, x, bottom - 3)[0]


class TestWhatItPaints:
    def test_the_border_and_shadow_stay_solid(self) -> None:
        # 塗りのグラデーションが縁や影に掛かると、エフェクトの「グラデーション」と同じ絵になる
        image = drawn(
            text(
                fill_type="linear",
                fill_gradient=RED_TO_BLUE,
                border_width=8.0,
                border_color=GREEN,
                shadow_x=30.0,
                shadow_y=-30.0,
                shadow_color=(1.0, 1.0, 0.0, 1.0),
            )
        )
        ys, xs = np.nonzero(image[..., 3] == 255)
        colours = {tuple(int(v) for v in image[y, x, :3]) for y, x in zip(ys, xs, strict=True)}
        # 縁の緑と影の黄色がそのまま残る
        assert (0, 255, 0) in colours
        assert (255, 255, 0) in colours

    def test_a_layer_takes_its_own_gradient(self) -> None:
        # 層ごとのグラデーションが効かないと、縁だけ金属風にする飾りが作れない
        layer = Stroke(
            params={
                **STROKE.default_params(),
                "width": AnimatedValue(12.0),
                "fill_type": "linear",
                "fill_gradient": flat((0.0, GREEN), (1.0, (1.0, 1.0, 0.0, 1.0))),
            }
        )
        image = drawn(text(text="■", color=WHITE).with_strokes((layer,)))
        left, top, right, bottom = solid(image)
        x = (left + right) // 2
        # 縁の上端は緑、下端は黄色、字の塗りは白のまま
        assert near(rgb(image, x, top + 2), GREEN, 40)
        assert near(rgb(image, x, bottom - 3), (1.0, 1.0, 0.0), 40)
        assert rgb(image, x, (top + bottom) // 2) == (255, 255, 255)

    def test_a_layer_opacity_thins_its_gradient(self) -> None:
        # 層の不透明度がグラデーションに掛からないと、薄くした縁が濃いまま残る
        layer = Stroke(
            params={
                **STROKE.default_params(),
                "width": AnimatedValue(12.0),
                "opacity": AnimatedValue(50.0),
                "fill_type": "linear",
                "fill_gradient": flat((0.0, GREEN), (1.0, GREEN)),
            }
        )
        image = drawn(text(text="■").with_strokes((layer,)))
        ys, xs = np.nonzero(image[..., 3] > 0)
        top = int(ys.min())
        x = (int(xs.min()) + int(xs.max())) // 2
        assert 110 < int(image[top + 3, x, 3]) < 145

    def test_a_shadow_gradient(self) -> None:
        # 影のグラデーションが効かないと、影だけ色を変えていく飾りが作れない
        image = drawn(
            text(
                text="■",
                color=(1.0, 1.0, 1.0, 1.0),
                shadow_x=0.0,
                shadow_y=-200.0,
                shadow_fill_type="linear",
                shadow_fill_gradient=RED_TO_BLUE,
            )
        )
        ys, xs = np.nonzero(image[..., 3] == 255)
        # 影は字の下 200 画素に、字と同じ枠をずらした所で赤から青になる
        shadow = [
            (y, x)
            for y, x in zip(ys, xs, strict=True)
            if image[y, x, 0] != 255 or image[y, x, 2] != 255
        ]
        top = min(y for y, _ in shadow)
        bottom = max(y for y, _ in shadow)
        x = shadow[0][1]
        assert near(rgb(image, x, top + 2), RED, 30)
        assert near(rgb(image, x, bottom - 2), BLUE, 30)

    def test_a_blurred_light_shadow_keeps_its_colour(self) -> None:
        # ぼかした影の外へ広がった所の色が 0 のままだと、明るい影の縁が黒ずむ
        white = flat((0.0, WHITE), (1.0, WHITE))
        image = drawn(
            text(
                text="■",
                color=BLACK,
                shadow_x=60.0,
                shadow_y=0.0,
                shadow_blur=12.0,
                shadow_fill_type="linear",
                shadow_fill_gradient=white,
            )
        )
        # 黒い字の右の、影だけの所を見る（字の縁の半透明の黒を数えない）
        _ys, xs = np.nonzero((image[..., 3] == 255) & (image[..., 0] == 0))
        beyond = np.zeros(image.shape[:2], dtype=bool)
        beyond[:, int(xs.max()) + 4 :] = True
        halo = beyond & (image[..., 3] > 10) & (image[..., 3] < 200)
        assert halo.any()
        assert image[..., :3][halo].min() > 240


class TestTheBasis:
    def test_wrapped_lines_share_one_gradient(self) -> None:
        # 行ごとに掛け直すと、折り返した字幕の 1 行目と 2 行目が同じ色の帯を繰り返す
        image = drawn(
            text(text="■■■■■■", wrap_width=360.0, fill_type="linear", fill_gradient=RED_TO_BLUE)
        )
        top, bottom = column_colours(image)
        assert near(top, RED, 16) and near(bottom, BLUE, 16), (top, bottom)

    def test_the_preview_matches_the_export(self) -> None:
        # 画質を落としたプレビューだけ枠がずれると、見て決めた色の位置が書き出しで変わる
        source = text(fill_type="linear", fill_gradient=flat((0.0, RED), (0.5, GREEN), (1.0, BLUE)))
        full = drawn(source)
        half = drawn(source, scale=0.5)
        _left, top, _right, bottom = solid(full)
        _half_left, half_top, _half_right, half_bottom = solid(half)
        for share in (0.2, 0.5, 0.8):
            y = round(top + (bottom - top) * share)
            half_y = round(half_top + (half_bottom - half_top) * share)
            expected = rgb(full, first_glyph_x(full), y)
            found = rgb(half, first_glyph_x(half), half_y)
            assert all(abs(a - b) <= 24 for a, b in zip(found, expected, strict=True)), (
                share,
                found,
                expected,
            )

    def test_a_text_wider_than_the_screen(self) -> None:
        # 広げた絵（#256）の大きさを基準にすると、長い字幕だけ端の色が出ない
        source = text(
            text="■" * 8,
            size=200.0,
            fill_type="linear",
            fill_gradient=RED_TO_BLUE,
            fill_angle=-90.0,
        )
        image = drawn(source)
        assert image.shape[1] > SCREEN[0]
        left, top, right, bottom = solid(image)
        y = (top + bottom) // 2
        assert near(rgb(image, left + 1, y), RED, 16)
        assert near(rgb(image, right - 2, y), BLUE, 16)

    def test_revealing_letters_keeps_their_colours(self) -> None:
        # 出ている字の枠で塗ると、1 文字出るたびにグラデーションが伸び縮みして色が変わっていく
        source = text(text="■■■■", fill_type="linear", fill_gradient=RED_TO_BLUE, fill_angle=-90.0)
        whole = drawn(source)
        part = drawn(
            GeneratedSource(kind="text", params={**source.params, "reveal": AnimatedValue(50.0)})
        )
        _left, top, right, bottom = solid(part)
        y = (top + bottom) // 2
        assert rgb(part, right - 3, y) == rgb(whole, right - 3, y)

    def test_a_colour_tag_wins_over_the_gradient(self) -> None:
        # 字ごとに書いた色（<#ff0000>）が塗りに上書きされると、強調した字の色が消える
        image = drawn(
            text(
                text="■<#00ff00>■",
                layout="aviutl",
                fill_type="linear",
                fill_gradient=RED_TO_BLUE,
            )
        )
        left, top, right, bottom = solid(image)
        y = (top + bottom) // 2
        assert rgb(image, right - 10, y) == (0, 255, 0)
        assert rgb(image, left + 10, y) != (0, 255, 0)
