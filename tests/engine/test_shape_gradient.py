"""図形の塗りと縁を、テキストと同じ塗りの種類とグラデーションで塗る（#364）

基準は図形の枠（回す前の幅 x 高さ） 角度 0 で枠の上端が位置 0、下端が位置 1 の色になる
"""

from __future__ import annotations

import numpy as np

from sashimono.core.model import GeneratedSource
from sashimono.effects.sources import SHAPE
from sashimono.effects.spec import ParamInput
from sashimono.engine.sources import render_source_framed, source_canvas

SCREEN = (640, 360)
RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
RED_TO_BLUE = (0.0, *RED, 1.0, *BLUE)


def shape(**params: ParamInput) -> GeneratedSource:
    base: dict[str, ParamInput] = {"shape": "rect", "width": 200, "height": 100, "color": WHITE}
    base.update(params)
    return SHAPE.create(**base)


def gradient_shape(**params: ParamInput) -> GeneratedSource:
    return shape(fill_type="linear", fill_gradient=RED_TO_BLUE, **params)


def drawn(source: GeneratedSource, scale: float = 1.0) -> np.ndarray:
    width, height = round(SCREEN[0] * scale), round(SCREEN[1] * scale)
    canvas = source_canvas(source, width, height, scale=(scale, scale))
    assert canvas == (width, height)
    image = render_source_framed(source, *canvas, scale=(scale, scale))[0]
    assert image is not None
    return image.astype(np.int16)


def at(image: np.ndarray, x: float, y: float) -> tuple[int, int, int]:
    """画面の真ん中から右へ ``x``・下へ ``y`` の画素（画面の画素で数える）"""
    height, width = image.shape[:2]
    scale = width / SCREEN[0]
    px = round(width / 2 + x * scale)
    py = round(height / 2 + y * scale)
    red, green, blue = (int(v) for v in image[py, px, :3])
    return red, green, blue


class TestSolid:
    def test_a_shape_from_an_older_version_draws_the_same(self) -> None:
        # 項目の無い前の版の図形を単色の道で描かないと、開いただけで図形の縁の画素が変わる
        current = shape(line_width=6, rotation=17)
        older = GeneratedSource(
            kind="shape",
            params={k: v for k, v in current.params.items() if not k.startswith("fill_")},
        )
        assert np.array_equal(drawn(current), drawn(older))

    def test_choosing_solid_ignores_the_gradient(self) -> None:
        # 単色に戻してもグラデーションで塗ると、種類を戻した意味が無い
        plain = shape()
        kept = shape(fill_type="solid", fill_gradient=RED_TO_BLUE)
        assert np.array_equal(drawn(plain), drawn(kept))


class TestGradient:
    def test_the_fill_runs_over_the_shape_box(self) -> None:
        # 枠に合わせて置かないと、上の端から下の端まで同じ色になるか、帯が枠の外で終わる
        image = drawn(gradient_shape())
        top, middle, bottom = at(image, 0, -48), at(image, 0, 0), at(image, 0, 48)
        assert top[0] > 230 and top[2] < 25, top
        assert bottom[2] > 230 and bottom[0] < 25, bottom
        assert 100 < middle[0] < 160 and 100 < middle[2] < 160, middle
        # 外は塗らない
        assert int(image[10, 10, 3]) == 0

    def test_the_gradient_turns_with_the_shape(self) -> None:
        # 画面の向きのまま置くと、回した図形の上で色の帯が回らずに残る
        image = drawn(gradient_shape(rotation=90))
        left, right = at(image, -45, 0), at(image, 45, 0)
        assert abs(left[0] - right[0]) > 150, (left, right)
        top, bottom = at(image, 0, -80), at(image, 0, 80)
        assert abs(top[0] - bottom[0]) < 20, (top, bottom)

    def test_the_outline_takes_the_gradient(self) -> None:
        # 線だけの図形（縁）を単色の色で引くと、塗りの種類を選んでも縁の色が変わらない
        image = drawn(gradient_shape(outline_only=True, line_width=8))
        top, bottom = at(image, 0, -47), at(image, 0, 47)
        assert top[0] > 200 and top[2] < 60, top
        assert bottom[2] > 200 and bottom[0] < 60, bottom
        assert int(image[SCREEN[1] // 2, SCREEN[0] // 2, 3]) == 0, "線だけなのに中を塗っている"

    def test_a_centred_outline_takes_the_gradient(self) -> None:
        # 輪郭の中央に引く線だけの図形はペンで引く ペンに単色を渡すと縁が白いまま残る
        image = drawn(gradient_shape(outline_only=True, line_width=8, line_align="center"))
        top, bottom = at(image, 0, -50), at(image, 0, 50)
        assert top[0] > 200 and top[2] < 60, top
        assert bottom[2] > 200 and bottom[0] < 60, bottom

    def test_a_centred_line_takes_the_same_brush(self) -> None:
        # 塗りと縁で違う物を使うと、輪郭の中央に引いた縁だけ白く浮く
        image = drawn(gradient_shape(line_width=12, line_align="center"))
        outside_top = at(image, 0, -53)
        assert outside_top[0] > 200 and outside_top[1] < 40, outside_top

    def test_half_quality_puts_the_same_colours_in_the_same_places(self) -> None:
        # 画質 1/2 のプレビューで筆の置き場がずれると、書き出しと違う所に色が出る
        full = drawn(gradient_shape(rotation=30))
        half = drawn(gradient_shape(rotation=30), scale=0.5)
        for x, y in ((0, -40), (0, 0), (0, 40), (-60, 10), (60, -10)):
            expected, found = at(full, x, y), at(half, x, y)
            assert all(abs(a - b) <= 24 for a, b in zip(expected, found, strict=True)), (
                (x, y),
                expected,
                found,
            )

    def test_a_shape_drawn_its_own_way_keeps_its_colour(self) -> None:
        # 線や集中線は塗りの欄を出さない 描く所が筆を読むと、見えない設定で色が変わる
        plain = shape(shape="polyline", points="-100,0;100,0", line_width=10)
        painted = gradient_shape(shape="polyline", points="-100,0;100,0", line_width=10)
        assert np.array_equal(drawn(plain), drawn(painted))
