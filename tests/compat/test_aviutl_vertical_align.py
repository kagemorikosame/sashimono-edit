"""AviUtl2 の縦書きの文字揃え（``縦書 上寄[右]`` など 9 通り）を読んで描く（#312 の続き）

2026-10-15 に AviUtl2 v2.1.6a に 18 通り（横書き 9・縦書き 9）を描かせて比べた 横書きは前から
合っていた（差 0.13〜0.37） 縦書きは名前を知らずに横書きの中央揃え[下] で描いていた（差 2.7〜3.0）
測った決まり（Yu Gothic UI 80 ``Hg字あ`` と ``長い二行目の字`` の 2 列）

- 上寄・中央・下寄は塊の上端・真ん中・下端を位置に合わせ、短い列も同じ向きに寄せる
- [右]・[中]・[左] は塊の右端・真ん中・左端を位置に合わせる 列は右から左へ並ぶ
- 字の送りも列の幅も字の大きさ（80） 字は升目の縦の真ん中に字の輪郭の真ん中を合わせる
  （H は升目の上から 12 画素 基準線で置くと g の下がりの分だけ下へずれる）

ここでは同じ形のエイリアスをコードで作って見る（実物の絵はリポジトリに入れない）
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.aviutl.exo import load_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.model import GeneratedSource
from sashimono.core.timebase import FrameRate
from sashimono.engine.sources import render_source

ALIAS = """[Object]
frame=0,59
[Object.0]
effect.name=テキスト
サイズ=80.00
字間=0.00
行間=0.00
表示速度=0.00
フォント=Yu Gothic UI
文字色=ffffff
影・縁色=000000
文字装飾=標準文字
文字揃え={align}
B=0
I=0
テキスト={text}
文字毎に個別オブジェクト=0
自動スクロール=0
移動座標上に表示=0
オブジェクトの長さを自動調節=0
"""


def read(
    tmp_path: Path, align: str, text: str = "■■\\n■"
) -> tuple[GeneratedSource, CompatibilityReport]:
    path = tmp_path / "縦書き.object"
    path.write_text(ALIAS.format(align=align, text=text), encoding="utf-8")
    (obj,) = load_exo(path).objects
    report = CompatibilityReport()
    found = map_object(obj, FrameRate(60), report=report)
    assert found is not None and found.clip.source is not None
    return found.clip.source, report


def boxes(source: GeneratedSource) -> list[tuple[int, int, int, int]]:
    """描いた字の四角（左・上・右・下） 列ごとに右から"""
    image = render_source(source, 1920, 1080)
    assert image is not None
    lit = image[..., 3] > 128
    found: list[tuple[int, int, int, int]] = []
    columns = np.flatnonzero(np.diff(np.concatenate([[0], lit.any(axis=0).astype(int), [0]])))
    for left, right in zip(columns[::2], columns[1::2], strict=True):
        rows = np.flatnonzero(
            np.diff(np.concatenate([[0], lit[:, left:right].any(axis=1).astype(int), [0]]))
        )
        found.extend(
            (int(left), int(top), int(right), int(bottom))
            for top, bottom in zip(rows[::2], rows[1::2], strict=True)
        )
    return sorted(found, key=lambda box: (-box[0], box[1]))


VERTICAL = {
    "縦書 上寄[右]": ("top", "right"),
    "縦書 中央[右]": ("middle", "right"),
    "縦書 下寄[右]": ("bottom", "right"),
    "縦書 上寄[中]": ("top", "center"),
    "縦書 中央[中]": ("middle", "center"),
    "縦書 下寄[中]": ("bottom", "center"),
    "縦書 上寄[左]": ("top", "left"),
    "縦書 中央[左]": ("middle", "left"),
    "縦書 下寄[左]": ("bottom", "left"),
}


@pytest.mark.parametrize("align", list(VERTICAL), ids=list(VERTICAL))
def test_the_vertical_names_are_read(tmp_path: Path, align: str) -> None:
    # 名前を知らないと、縦書きの字幕が横書きの中央揃え[下] で出る（前の版の差 2.7〜3.0）
    source, report = read(tmp_path, align)
    valign, anchor = VERTICAL[align]
    assert source.params["vertical"] is True
    assert source.params["valign"] == valign
    assert source.params["anchor"] == anchor
    assert not any("文字揃え" in line for line in report.lines())


def test_an_unknown_name_is_counted(tmp_path: Path) -> None:
    # 知らない名前を黙って中央揃え[下] で読むと、置き場所のずれが実物を数えても上がってこない
    _source, report = read(tmp_path, "左揃え[上]")
    assert any("左揃え[上]" in line for line in report.lines())


# 1 字 80 画素の升目 字の真ん中の 1920x1080 の画面の真ん中（960, 540）が位置
@pytest.mark.parametrize(
    ("align", "right", "top"),
    [
        # 塊（2 列 x 2 字 = 160 x 160）の右上が位置
        ("縦書 上寄[右]", 960, 540),
        # 塊の真ん中が位置
        ("縦書 中央[中]", 1040, 460),
        # 塊の左下が位置
        ("縦書 下寄[左]", 1120, 380),
    ],
)
def test_the_block_sits_on_the_position(tmp_path: Path, align: str, right: int, top: int) -> None:
    # 塊の基準が AviUtl2 と違うと、縦書きの字幕が画面の別の所に出る
    source, _report = read(tmp_path, align)
    first = boxes(source)[0]
    # 1 列目の 1 字目（■）は升目 80 の真ん中に、輪郭の真ん中を合わせて置く
    assert abs((first[0] + first[2]) / 2 - (right - 40)) <= 2, first
    assert abs((first[1] + first[3]) / 2 - (top + 40)) <= 2, first


@pytest.mark.parametrize(
    ("align", "offset"),
    [("縦書 上寄[中]", 0), ("縦書 中央[中]", 40), ("縦書 下寄[中]", 80)],
    ids=["上寄", "中央", "下寄"],
)
def test_a_short_column_follows_the_alignment(tmp_path: Path, align: str, offset: int) -> None:
    # 短い列を上で揃えたままだと、中央・下寄の 2 行の字幕の短い行が AviUtl2 より上へ出る
    source, _report = read(tmp_path, align)
    found = boxes(source)
    long_top = min(box[1] for box in found if box[0] > 960)
    (short,) = [box for box in found if box[0] < 960]
    assert abs(short[1] - long_top - offset) <= 2, (found, offset)


def test_the_pitch_is_the_font_size(tmp_path: Path) -> None:
    # 行の高さ（106）で送ると、縦書きの字の間が AviUtl2 より 3 割広く開く
    source, _report = read(tmp_path, "縦書 上寄[右]", text="■■■")
    tops = [box[1] for box in boxes(source)]
    assert [b - a for a, b in pairwise(tops)] == pytest.approx([80, 80], abs=1)


def test_the_standard_layout_keeps_its_vertical_text(tmp_path: Path) -> None:
    # 標準の組み方まで変えると、前の版で作った縦書きの字幕の字の間と短い列の位置が動く
    source, _report = read(tmp_path, "縦書 下寄[中]", text="■■■")
    native = GeneratedSource(kind="text", params={**source.params, "layout": "native"})
    tops = [box[1] for box in boxes(native)]
    assert tops[1] - tops[0] > 95
