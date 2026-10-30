"""字の輪郭の塗りの決まりと、輪郭が重なる所に穴が開かないこと（#415）

可変フォントの系統（Noto Sans JP・源ノ角ゴシックなど）は 1 つの字の中で輪郭が重なっている
輪郭を集めるパスが OddEvenFill のままだと、重なった所が偶数として抜け、字の白に市松の穴が
開いて下の黒い縁が透ける 太さ 40 の字幕でも、縁がガビガビに荒れて見えた

書体が CI にある保証が無いので、塗りの決まりは組み方ごとに返す形を見て確かめ、穴の画素は
字間や行間を詰めて字どうしの輪郭を重ねて数える どちらも書体に依らない
"""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPolygonF

from sashimono.core.model import AnimatedValue, GeneratedSource, ParamValue, Stroke
from sashimono.effects.sources import source_registry
from sashimono.engine import sources
from sashimono.engine.sources import render_source_framed

SIZE = (640, 360)
WHITE = (1.0, 1.0, 1.0, 1.0)
CLEAR = (1.0, 1.0, 1.0, 0.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0, 1.0)
#: 字の外形そのものの四角 中の空きが無いので、輪郭どうしを重ねた所の穴だけが数に出る
BLOCK = "■"


def text(body: str, *strokes: Stroke, **params: ParamValue) -> GeneratedSource:
    base: dict[str, ParamValue] = {
        "text": body,
        "size": AnimatedValue(100.0),
        "color": WHITE,
    }
    base.update(params)
    return GeneratedSource(kind="text", params=base, strokes=strokes)


def layers_of(source: GeneratedSource) -> list[QPainterPath]:
    definition = source_registry.get("text")
    assert definition is not None
    values = sources._resolve(definition, source.params, 0)
    sources._resolve_strokes(values, source, 0)
    laid = sources._text_layers(values, *SIZE)
    assert laid is not None
    return [path for path, _look, _clip in laid[0]]


def alpha_of(source: GeneratedSource) -> np.ndarray:
    image = render_source_framed(source, *SIZE)[0]
    assert image is not None
    return image[..., 3].copy()


def gap_pixels(ink: np.ndarray) -> int:
    """行と列の中で、最初の墨と最後の墨の間に空いた画素の数

    四角を縦横に重ねただけの形は、どの行と列でも墨が 1 続きになる 空きがあれば重なりの穴
    """
    gaps = 0
    for lines in (ink, ink.T):
        for line in lines:
            found = np.flatnonzero(line)
            if found.size:
                gaps += int(found[-1] - found[0] + 1 - found.size)
    return gaps


def has_block() -> bool:
    """■ が字の外形いっぱいに塗られる書体か 字が無いと中の空いた豆腐の箱になり、数が読めない"""
    return gap_pixels(alpha_of(text(BLOCK)) > 128) == 0 and bool(alpha_of(text(BLOCK)).any())


#: AviUtl の組み方の太字は入れない 太字は ``QPainterPath.united`` が輪郭を解いて重なりの無い形に
#: するので OddEvenFill のままで正しい 穴が無いことは :data:`OVERLAPS` で絵を見て確かめる
LAYOUTS: dict[str, GeneratedSource] = {
    "標準 1 行": text("字あ"),
    "標準 2 行": text("字\nあ"),
    "標準 太字": text("字あ", bold=1),
    "標準 字間": text("字あ", letter_spacing=AnimatedValue(4.0)),
    "縦書き": text("字あ", vertical=True),
    "AviUtl": text("字あ", layout="aviutl"),
    "AviUtl 2 行": text("字\nあ", layout="aviutl"),
    "AviUtl 変形": text("<th2>字あ", layout="aviutl"),
    "AviUtl 変形と素の字": text("字<tw2>あ", layout="aviutl"),
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_every_layout_fills_glyphs_with_winding(name: str) -> None:
    """どの組み方でも字の輪郭は WindingFill で塗る OddEvenFill だと重なった輪郭が抜ける"""
    rules = [path.fillRule() for path in layers_of(LAYOUTS[name])]

    assert rules
    assert set(rules) == {Qt.FillRule.WindingFill}


OVERLAPS: dict[str, GeneratedSource] = {
    "標準 字間で重ねる": text(BLOCK * 2, letter_spacing=AnimatedValue(-50.0)),
    "標準 行間で重ねる": text(f"{BLOCK}\n{BLOCK}", line_spacing=AnimatedValue(-60.0)),
    "標準 太字": text(BLOCK * 2, bold=1, letter_spacing=AnimatedValue(-50.0)),
    "AviUtl 変形": text(f"<tw3>{BLOCK * 3}", layout="aviutl"),
    "AviUtl 変形の太字": text(f"<tw3>{BLOCK * 3}", layout="aviutl", bold=1),
    # 影だけを見るため字は透明にする 字を塗ると穴が隠れる
    "影": text(
        BLOCK * 2,
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
        shadow_x=AnimatedValue(6.0),
        shadow_y=AnimatedValue(-6.0),
        shadow_color=BLACK,
    ),
    # 字いっぱいの太さで引く 内側の縁の絵を字の形で切ったときに、重なった所が抜けないか
    "内側の縁": text(
        BLOCK * 2,
        Stroke(
            params={
                "width": AnimatedValue(100.0),
                "color": RED,
                "position": "inside",
            }
        ),
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
    ),
}


@pytest.mark.parametrize("name", list(OVERLAPS))
def test_overlapping_outlines_leave_no_hole(name: str) -> None:
    """輪郭が重なった所を抜かない 抜けると字の白に穴が開いて、下の黒い縁や映像が透ける"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")

    ink = alpha_of(OVERLAPS[name]) > 128
    rows, columns = np.nonzero(ink)
    assert rows.size

    # 外周の数画素は数えない 輪郭の辺がぴったり重なると、太字の QPainterPath.united が辺に
    # 1 画素の段を残す それは外形の段で、数えたいのは中の穴
    edge = 3
    inner = ink[
        int(rows.min()) + edge : int(rows.max()) + 1 - edge,
        int(columns.min()) + edge : int(columns.max()) + 1 - edge,
    ]

    assert gap_pixels(inner) == 0


RING = 3.0
SEE_THROUGH: dict[str, GeneratedSource] = {
    "内側の層": text(
        BLOCK * 2,
        Stroke(params={"width": AnimatedValue(RING), "color": RED, "position": "inside"}),
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
    ),
    "中央の層": text(
        BLOCK * 2,
        Stroke(params={"width": AnimatedValue(RING), "color": RED, "position": "center"}),
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
    ),
    "透明の塗りと外側の層": text(
        BLOCK * 2,
        Stroke(params={"width": AnimatedValue(RING), "color": RED, "position": "outside"}),
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
    ),
    "透明の塗りと前からの縁": text(
        BLOCK * 2,
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
        border_width=AnimatedValue(RING),
        border_color=RED,
    ),
}


@pytest.mark.parametrize("name", list(SEE_THROUGH))
def test_ring_follows_outer_shape_only(name: str) -> None:
    """重なった輪郭に線を引かない 字の外形の中を線が横切ると、字に無い線が見える"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")

    ink = alpha_of(SEE_THROUGH[name]) > 64
    rows, columns = np.nonzero(ink)
    assert rows.size

    # 外形の線が届かない中ほどに線があれば、それは字の中の輪郭に引いた線
    reach = int(RING) * 2 + 2
    middle = ink[
        int(rows.min()) + reach : int(rows.max()) + 1 - reach,
        int(columns.min()) + reach : int(columns.max()) + 1 - reach,
    ]

    assert middle.size
    assert not middle.any()


def test_outer_outline_keeps_a_path_without_overlap() -> None:
    """重なりが無ければ元の形のまま 解くと曲線が折れ線になり、縁の絵が前の版から動く"""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    path.addEllipse(QRectF(0.0, 0.0, 100.0, 100.0))
    path.addEllipse(QRectF(200.0, 0.0, 100.0, 100.0))

    assert sources._outer_outline(path) == path


def test_outer_outline_merges_separate_contours() -> None:
    """重なった別々の輪郭は 1 つの外形にする 残すと内側の縁の線が字の中を横切る"""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    path.addRect(QRectF(0.0, 0.0, 100.0, 100.0))
    path.addRect(QRectF(50.0, 0.0, 100.0, 100.0))

    assert len(sources._outer_outline(path).toSubpathPolygons()) == 1


def test_outer_outline_merges_a_contour_crossing_itself() -> None:
    """1 つの輪郭が自分と重なる字は輪郭の数が変わらないので、長さの縮みで見つける"""
    star = [QPointF(sin(pi * 0.8 * n), -cos(pi * 0.8 * n)) * 100.0 for n in range(5)]
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    path.addPolygon(QPolygonF(star))
    path.closeSubpath()

    merged = sources._outer_outline(path)

    assert len(merged.toSubpathPolygons()) == 1
    assert merged.length() < path.length() * 0.9


@pytest.mark.parametrize("bold", [0, 1])
@pytest.mark.parametrize("glyph", ["O", "口"])
def test_counter_of_a_glyph_stays_open(glyph: str, bold: int) -> None:
    """字の中の本来の穴は埋まらない WindingFill でも内側の輪郭は向きが逆なので抜ける"""
    ink = alpha_of(text(glyph, bold=bold, size=AnimatedValue(200.0))) > 128
    rows, columns = np.nonzero(ink)

    centre_row = (int(rows.min()) + int(rows.max())) // 2
    centre_column = (int(columns.min()) + int(columns.max())) // 2

    assert not ink[centre_row, centre_column]


def test_font_with_overlapping_contours_has_no_hole() -> None:
    """輪郭が重なる実物の書体 描いた絵を WindingFill で塗り直した形と比べて、穴を数える"""
    if not QFont("Noto Sans JP").exactMatch():
        pytest.skip("Noto Sans JP が入っていない")
    source = text("もキ調理部門", font="Noto Sans JP", bold=1, size=AnimatedValue(80.0))

    mask = QImage(*SIZE, QImage.Format.Format_RGBA8888)
    mask.fill(Qt.GlobalColor.transparent)
    painter = QPainter(mask)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for path in layers_of(source):
        solid = QPainterPath(path)
        solid.setFillRule(Qt.FillRule.WindingFill)
        painter.fillPath(solid, QColor(255, 255, 255, 255))
    painter.end()
    expected = np.frombuffer(mask.constBits(), np.uint8).reshape(SIZE[1], SIZE[0], 4)[..., 3]

    drawn = alpha_of(source)

    assert expected.any()
    assert int(((expected > 200) & (drawn < 128)).sum()) == 0
