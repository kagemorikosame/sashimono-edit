"""グラデーションを 1 行ごと・1 文字ごとに掛け直す（#312）

標準・AviUtl2 と同じ・縦書きの 3 通りの組み方で、色が行ごと・字ごとに繰り返すことを見る
字は「■」 字と字・行と行の間のすき間で塗れた所を分け、1 つずつの端の色を読む
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.model import AnimatedValue, Stroke
from sashimono.effects.spec import ParamInput
from sashimono.effects.strokes import STROKE
from tests.engine.test_text_gradient import BLUE, GREEN, RED, RED_TO_BLUE, drawn, near, rgb, text

Box = tuple[int, int, int, int]


def runs(occupied: np.ndarray) -> list[tuple[int, int]]:
    """真が続く所の（頭, 尻の次）の並び"""
    found: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(occupied.tolist()):
        if value and start is None:
            start = index
        if not value and start is not None:
            found.append((start, index))
            start = None
    if start is not None:
        found.append((start, len(occupied)))
    return found


def rows_of(image: np.ndarray) -> list[list[Box]]:
    """塗れた所を、横の帯（行）ごとに字へ分けた四角（左・上・右・下）"""
    mask = image[..., 3] == 255
    found = []
    for top, bottom in runs(mask.any(axis=1)):
        band = mask[top:bottom]
        found.append([(left, top, right, bottom) for left, right in runs(band.any(axis=0))])
    return found


def columns_of(image: np.ndarray) -> list[list[Box]]:
    """縦書き 塗れた所を、縦の帯（列）ごとに字へ分けた四角"""
    mask = image[..., 3] == 255
    found = []
    for left, right in runs(mask.any(axis=0)):
        band = mask[:, left:right]
        found.append([(left, top, right, bottom) for top, bottom in runs(band.any(axis=1))])
    return found


def ends_down(image: np.ndarray, box: Box) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """四角の上の端と下の端の色"""
    left, top, right, bottom = box
    x = (left + right) // 2
    return rgb(image, x, top + 2), rgb(image, x, bottom - 3)


def ends_across(image: np.ndarray, box: Box) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """四角の左の端と右の端の色"""
    left, top, right, bottom = box
    y = (top + bottom) // 2
    return rgb(image, left + 2, y), rgb(image, right - 3, y)


#: 3 通りの組み方 縦書きは列が行になる
LAYOUTS: dict[str, dict[str, ParamInput]] = {
    "標準": {},
    "AviUtl2 と同じ": {"layout": "aviutl"},
    "縦書き": {"vertical": True},
}


@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
def test_each_line_repeats_the_gradient(layout: str) -> None:
    # 行ごとに掛け直せないと、2 行の字幕の 2 行目が 1 行目の続きの色（下半分の色）だけになる
    vertical = layout == "縦書き"
    # 1 行の中で、行を重ねる向きに掛ける（横書きは上から下、縦書きは左から右）
    source = text(
        text="■■\n■■",
        size=120.0,
        fill_type="linear",
        fill_gradient=RED_TO_BLUE,
        fill_angle=-90.0 if vertical else 0.0,
        fill_basis="line",
        **LAYOUTS[layout],
    )
    image = drawn(source)
    lines = columns_of(image) if vertical else rows_of(image)
    assert len(lines) == 2
    for line in lines:
        first, last = ends_across(image, line[0]) if vertical else ends_down(image, line[0])
        assert near(first, RED, 24) and near(last, BLUE, 24), (layout, first, last)
    # 字の枠全体なら、行の 1 つは頭が赤ではない（行ごとに掛け直したことの確かめ）
    whole = drawn(replace(source, params={**source.params, "fill_basis": "box"}))
    lines = columns_of(whole) if vertical else rows_of(whole)
    heads = [
        (ends_across(whole, line[0]) if vertical else ends_down(whole, line[0]))[0]
        for line in lines
    ]
    assert not all(near(head, RED, 24) for head in heads), (layout, heads)


@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
def test_each_letter_repeats_the_gradient(layout: str) -> None:
    # 字ごとに掛け直せないと、1 文字ずつ同じ光り方をさせる飾りが作れない
    vertical = layout == "縦書き"
    # 字の並ぶ向きに掛ける（縦書きは上から下、横書きは左から右）
    source = text(
        text="■■■",
        size=120.0,
        fill_type="linear",
        fill_gradient=RED_TO_BLUE,
        fill_angle=0.0 if vertical else -90.0,
        fill_basis="char",
        **LAYOUTS[layout],
    )
    image = drawn(source)
    (line,) = columns_of(image) if vertical else rows_of(image)
    assert len(line) == 3
    for box in line:
        first, last = ends_down(image, box) if vertical else ends_across(image, box)
        assert near(first, RED, 24) and near(last, BLUE, 24), (layout, first, last)


def test_a_layer_follows_the_letters() -> None:
    # 縁取りの層だけ字の枠全体で塗ると、字ごとに掛けた塗りと縁の色の並びがずれる
    layer = Stroke(
        params={
            **STROKE.default_params(),
            "width": AnimatedValue(10.0),
            "fill_type": "linear",
            "fill_gradient": RED_TO_BLUE,
            "fill_angle": AnimatedValue(-90.0),
            "fill_basis": "char",
        }
    )
    image = drawn(text(text="■　■", size=120.0, color=GREEN).with_strokes((layer,)))
    (line,) = rows_of(image)
    assert len(line) == 2
    for box in line:
        first, last = ends_across(image, box)
        assert near(first, RED, 40) and near(last, BLUE, 40), (first, last)


def test_a_shadow_follows_the_letters() -> None:
    # 影だけ字の枠全体で塗ると、字ごとの影の色が字の並びと合わない
    image = drawn(
        text(
            text="■　■",
            size=120.0,
            color=GREEN,
            # 横にもずらす 字の並ぶ向きにずらさないと、基準を影と一緒に動かさなくても
            # 色が合ってしまう
            shadow_x=60.0,
            shadow_y=-200.0,
            shadow_fill_type="linear",
            shadow_fill_gradient=RED_TO_BLUE,
            shadow_fill_angle=-90.0,
            shadow_fill_basis="char",
        )
    )
    lines = rows_of(image)
    assert len(lines) == 2
    for box in lines[1]:
        first, last = ends_across(image, box)
        assert near(first, RED, 40) and near(last, BLUE, 40), (first, last)
