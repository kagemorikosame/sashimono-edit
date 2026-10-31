"""字の輪郭の塗りの決まりと、輪郭が重なる所に穴が開かないこと（#415）

可変フォントの系統（Noto Sans JP・源ノ角ゴシックなど）は 1 つの字の中で輪郭が重なっている
輪郭を集めるパスが OddEvenFill のままだと、重なった所が偶数として抜け、字の白に市松の穴が
開いて下の黒い縁が透ける 太さ 40 の字幕でも、縁がガビガビに荒れて見えた

書体が CI にある保証が無いので、塗りの決まりは組み方ごとに返す形を見て確かめ、穴の画素は
字間や行間を詰めて字どうしの輪郭を重ねて数える どちらも書体に依らない
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from math import cos, pi, sin

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPolygonF

from sashimono.compat.aviutl.text_tags import parse_tags
from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    ParamValue,
    Project,
    ProjectSettings,
    Stroke,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import source_registry
from sashimono.engine import sources
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, RenderQuality
from sashimono.engine.sources import render_source_framed

SIZE = (640, 360)
WHITE = (1.0, 1.0, 1.0, 1.0)
CLEAR = (1.0, 1.0, 1.0, 0.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0, 1.0)
#: 字の外形そのものの四角 中の空きが無いので、輪郭どうしを重ねた所の穴だけが数に出る
BLOCK = "■"


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


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


def render_alpha(source: GeneratedSource, width: int, height: int, scale: float) -> np.ndarray:
    image = render_source_framed(source, width, height, scale=(scale, scale))[0]
    assert image is not None
    return image[..., 3] > 128


def assert_no_gap(ink: np.ndarray) -> None:
    """外周の 1 画素を除いて、墨の行と列に空きが無いことを確かめる 縮めた絵は端が滑らかになる"""
    rows, columns = np.nonzero(ink)
    assert rows.size
    inner = ink[
        int(rows.min()) + 1 : int(rows.max()),
        int(columns.min()) + 1 : int(columns.max()),
    ]
    assert gap_pixels(inner) == 0


def has_block() -> bool:
    """■ が字の外形いっぱいに塗られる書体か 字が無いと中の空いた豆腐の箱になり、数が読めない"""
    ink = alpha_of(text(BLOCK)) > 128
    return bool(ink.any()) and gap_pixels(ink) == 0


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


def test_outer_outline_finds_one_crossing_contour_in_a_long_text() -> None:
    """長い文の中の 1 字の重なりも見つける 全体の長さだけで見ると埋もれて、線が字の中に残る"""
    star = [QPointF(sin(pi * 0.8 * n), -cos(pi * 0.8 * n)) * 10.0 for n in range(5)]
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    path.addPolygon(QPolygonF(star))
    path.closeSubpath()
    for index in range(100):
        path.addEllipse(QRectF(100.0 + index * 120.0, 0.0, 100.0, 100.0))

    merged = sources._outer_outline(path)

    assert merged != path
    assert len(merged.toSubpathPolygons()) == 101


def test_outer_outline_finds_contours_that_make_a_ring() -> None:
    """2 つの輪郭が重なって外形と穴になると、輪郭の数は変わらないので長さで見つける"""
    left = [(0, 0), (60, 0), (60, 20), (20, 20), (20, 80), (60, 80), (60, 100), (0, 100)]
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    for corners in (left, [(100 - x, y) for x, y in left]):
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in corners]))
        path.closeSubpath()

    merged = sources._outer_outline(path)

    assert len(merged.toSubpathPolygons()) == 2
    assert merged.length() < path.length()


def test_translucent_shadow_shows_no_line_of_inner_contours() -> None:
    """半透明の影の濃さが字の中で揃う 重なった輪郭の線があると、そこだけ濃い筋が出る"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    source = text(
        BLOCK * 2,
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
        border_width=AnimatedValue(RING),
        border_color=BLACK,
        shadow_x=AnimatedValue(200.0),
        shadow_color=(0.0, 0.0, 0.0, 0.5),
    )

    # 影を字から離して右側だけを見る 字の縁は不透明なので、重なると濃さが比べられない
    alpha = alpha_of(source)[:, SIZE[0] * 2 // 3 :]
    rows, columns = np.nonzero(alpha > 64)
    assert rows.size
    # 外形の線が届かない中ほどの濃さを見る
    reach = int(RING) * 2 + 2
    middle = alpha[
        int(rows.min()) + reach : int(rows.max()) + 1 - reach,
        int(columns.min()) + reach : int(columns.max()) + 1 - reach,
    ]

    assert middle.size
    assert int(middle.max()) - int(middle.min()) <= 2


def test_layers_work_out_the_outline_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """層と影がいくつあっても輪郭を解くのは 1 回 層の数だけ解くと長い字幕で描くのが遅くなる"""
    calls: list[int] = []
    work_out = sources._outer_outline

    def counted(path: QPainterPath) -> QPainterPath:
        calls.append(1)
        return work_out(path)

    monkeypatch.setattr(sources, "_outer_outline", counted)
    rings = [
        Stroke(params={"width": AnimatedValue(width), "color": RED, "position": position})
        for width, position in ((2.0, "inside"), (4.0, "inside"), (2.0, "center"))
    ]

    alpha_of(
        text(
            "字",
            *rings,
            shadow_x=AnimatedValue(6.0),
            shadow_y=AnimatedValue(-6.0),
            shadow_color=(0.0, 0.0, 0.0, 0.5),
        )
    )

    assert calls == [1]


def inner_ring(width: float = RING) -> Stroke:
    return Stroke(params={"width": AnimatedValue(width), "color": RED, "position": "inside"})


#: 字どうしが別の層で重なる形 素の字と変形した字、色の違う字
ACROSS_LAYERS: dict[str, GeneratedSource] = {
    "素の字と変形した字": text(f"{BLOCK}<tw3>{BLOCK}", inner_ring(), layout="aviutl", color=CLEAR),
    "色の違う字": text(
        f"{BLOCK}<#00ff00>{BLOCK}",
        inner_ring(),
        layout="aviutl",
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
    ),
}


@pytest.mark.parametrize("name", list(ACROSS_LAYERS))
def test_ring_ignores_the_border_between_layers(name: str) -> None:
    """別の層の字が重なっても境に線を引かない 層ごとに引くと字の中に線が残る"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")

    ink = alpha_of(ACROSS_LAYERS[name]) > 64
    rows, columns = np.nonzero(ink)
    assert rows.size

    reach = int(RING) * 2 + 2
    middle = ink[
        int(rows.min()) + reach : int(rows.max()) + 1 - reach,
        int(columns.min()) + reach : int(columns.max()) + 1 - reach,
    ]

    assert middle.size
    assert not middle.any()


def test_ring_across_layers_keeps_the_cut_at_the_frame() -> None:
    """素の字と重なっても、変形した字は文字の枠で切る 外へ伸びた線を出さない（#184）"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    body = f"{BLOCK}<tw3>{BLOCK}"
    source = text(body, inner_ring(), layout="aviutl", color=CLEAR)
    plain = text(body, layout="aviutl")

    # 塗りは枠で切られるので、線の右端は塗りの右端を越えない 左端は素の字の線の分だけ出る
    ink = np.nonzero((alpha_of(source) > 64).any(axis=0))[0]
    filled = np.nonzero((alpha_of(plain) > 64).any(axis=0))[0]

    assert int(ink.max()) <= int(filled.max())
    assert int(ink.min()) >= int(filled.min()) - 2


def outside_frame(source: GeneratedSource) -> np.ndarray:
    """文字の枠の外の墨 ``False`` にした枠の中は比べない"""
    image, framed = render_source_framed(source, *SIZE)
    assert image is not None and framed is not None
    ink = image[..., 3] > 64
    left, top = math.floor(framed[0]) - 1, math.floor(framed[1]) - 1
    right, bottom = math.ceil(framed[2]) + 1, math.ceil(framed[3]) + 1
    ink[top:bottom, left:right] = False
    return ink


def test_transformed_glyph_stays_inside_the_frame_near_a_plain_glyph() -> None:
    """素の字の近くでも、変形した字の線は枠の外へ出さない 出ると #184 で切った字の縁が戻る"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    centre = Stroke(params={"width": AnimatedValue(60.0), "color": RED, "position": "center"})
    joined = text(f"{BLOCK}<tw3><th3>{BLOCK}", centre, layout="aviutl", color=CLEAR)
    # 変形した字を点まで小さくすると、枠の外に残るのは素の字の線だけになる
    alone = text(f"{BLOCK}<tw0.001><th0.001>{BLOCK}", centre, layout="aviutl", color=CLEAR)

    expected = outside_frame(alone)

    assert expected.any()
    assert int((outside_frame(joined) != expected).sum()) <= 8


def ring_entry(
    box: QRectF, *, reverse: bool = False, color: ParamValue = WHITE
) -> tuple[QPainterPath, sources._Ring, QPainterPath | None, dict[str, object]]:
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    path.addRect(box)
    shape = path.toReversed() if reverse else path
    shape.setFillRule(Qt.FillRule.WindingFill)
    return shape, sources._Ring(width=3.0, colour=RED), None, {"color": color}


def joined_rings(
    *entries: tuple[QPainterPath, sources._Ring, QPainterPath | None, dict[str, object]],
) -> list[tuple[QPainterPath, dict[str, object], QPainterPath | None]]:
    layers = [(path, look, clip) for path, _ring, clip, look in entries]
    return sources._painted_once(layers, sources._ring_key(entries, baked=False))


def test_opaque_glyph_joins_a_translucent_partner() -> None:
    """不透明の字の線も、重なった相手が透けるならまとめる 残すと相手の塗りを透けて線が見える"""
    opaque = ring_entry(QRectF(0.0, 0.0, 100.0, 100.0))
    translucent = ring_entry(QRectF(50.0, 0.0, 100.0, 100.0), color=(1.0, 1.0, 1.0, 0.5))

    assert len(joined_rings(opaque, translucent)) == 1


def test_opaque_glyphs_keep_their_own_rings() -> None:
    """線が全部塗りに隠れる字はまとめない 字幕の縁はほとんどこれで、絵も手間も前のままにする"""
    first = ring_entry(QRectF(0.0, 0.0, 100.0, 100.0))
    second = ring_entry(QRectF(50.0, 0.0, 100.0, 100.0))

    assert len(joined_rings(first, second)) == 2


def test_joining_keeps_overlap_of_opposite_contours() -> None:
    """輪郭の向きが逆の書体が重なっても、重なった所を残す 足すだけだと巻き数が消えて穴になる"""
    first = ring_entry(QRectF(0.0, 0.0, 100.0, 100.0), color=CLEAR)
    second = ring_entry(QRectF(50.0, 0.0, 100.0, 100.0), reverse=True, color=CLEAR)

    assert joined_rings(first, second)[0][0].contains(QPointF(75.0, 50.0))


def fill_layer(
    box: QRectF, color: ParamValue
) -> tuple[QPainterPath, dict[str, object], QPainterPath | None]:
    path = QPainterPath()
    path.addRect(box)
    return path, {"color": color}, None


def test_layers_between_keep_the_drawing_order() -> None:
    """間の層が組と重なるならまとめない まとめると後の字が間の層の下へ回り、色が変わる"""
    red, blue = (1.0, 0.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)
    layers = [
        fill_layer(QRectF(0.0, 0.0, 100.0, 100.0), red),
        fill_layer(QRectF(40.0, 0.0, 100.0, 100.0), blue),
        fill_layer(QRectF(80.0, 0.0, 100.0, 100.0), red),
    ]

    assert sources._painted_once(layers, sources._fill_key) is layers


def test_layers_between_that_stay_clear_do_not_block_joining() -> None:
    """間の層が離れていればまとめる 上下は入れ替わっても重ならないので色は変わらない"""
    red, blue = (1.0, 0.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)
    layers = [
        fill_layer(QRectF(0.0, 0.0, 100.0, 100.0), red),
        fill_layer(QRectF(0.0, 500.0, 100.0, 100.0), blue),
        fill_layer(QRectF(80.0, 0.0, 100.0, 100.0), red),
    ]

    assert len(sources._painted_once(layers, sources._fill_key)) == 2


def test_a_lone_layer_is_passed_through_untouched() -> None:
    """層が 1 つの字幕は何も調べない 字幕はほとんどこれで、毎コマの手間を増やさない"""
    look: dict[str, object] = {"color": CLEAR}
    only: list[tuple[QPainterPath, dict[str, object], QPainterPath | None]] = [
        (QPainterPath(), look, None)
    ]

    def must_not_run(_look: dict[str, object]) -> tuple[float, str]:
        raise AssertionError

    assert sources._painted_once(only, must_not_run) is only


def test_a_font_tag_makes_its_own_layer() -> None:
    """書体の違う字は別の層にする 輪郭の向きが逆の書体を 1 つのパスへ足すと、重なりが穴になる"""
    assert len(layers_of(text(f"{BLOCK}{BLOCK}", layout="aviutl"))) == 1
    if not QFont("Courier New").exactMatch():
        pytest.skip("Courier New が入っていない")
    assert len(layers_of(text(f"{BLOCK}<@Courier New>{BLOCK}", layout="aviutl"))) == 2


def test_only_boxes_reaching_sideways_are_compared() -> None:
    """横に届かない四角は比べない 字ごとに色を変えると、全部の組では字の数の 2 乗で増える"""
    apart = [QRectF(index * 20.0, 0.0, 10.0, 10.0) for index in range(50)]
    assert list(sources._near_pairs(apart)) == []

    rows = [QRectF(0.0, 0.0, 10.0, 10.0), QRectF(5.0, 50.0, 10.0, 10.0)]
    assert list(sources._near_pairs(rows)) == [(0, 1)]


def test_groups_hold_only_shapes_that_touch() -> None:
    """重なった字だけを組にする 離れた字まで形を合わせると、描くたびに無駄な計算が増える"""
    paths = [QPainterPath() for _ in range(4)]
    for path, left in zip(paths, (0.0, 50.0, 500.0, 1000.0), strict=True):
        path.addRect(QRectF(left, 0.0, 100.0, 100.0))

    assert sources._touching(paths, 0.0) == [[0, 1]]


def test_bands_that_touch_form_a_group() -> None:
    """字が離れていても、線の帯が届くなら組にする 半透明の帯が別々に重なると濃くなる"""
    paths = [QPainterPath() for _ in range(2)]
    for path, left in zip(paths, (0.0, 110.0), strict=True):
        path.addRect(QRectF(left, 0.0, 100.0, 100.0))

    assert sources._touching(paths, 0.0) == []
    assert sources._touching(paths, 6.0) == [[0, 1]]


HALF_RED = (1.0, 0.0, 0.0, 0.5)
#: 重なった所が 2 重に塗られると、不透明度 0.5 のはずの所が 0.75 になる
ONCE: dict[str, GeneratedSource] = {
    "書体の違う半透明の字の塗り": text(
        f"{BLOCK}<@Courier New>{BLOCK}",
        layout="aviutl",
        color=HALF_RED,
        letter_spacing=AnimatedValue(-50.0),
    ),
    "枠の外で重なる素の字の線": text(
        f"{BLOCK}<#00ff00>{BLOCK}<tw3>{BLOCK}",
        Stroke(params={"width": AnimatedValue(60.0), "color": HALF_RED, "position": "outside"}),
        layout="aviutl",
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
    ),
    "グラデーションの影": text(
        f"{BLOCK}<#00ff00>{BLOCK}",
        layout="aviutl",
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
        shadow_x=AnimatedValue(200.0),
        shadow_fill_type="linear",
        shadow_fill_gradient=(0.0, *HALF_RED, 1.0, *HALF_RED),
    ),
    "隣の字に届く半透明の縁": text(
        f"{BLOCK}<#00ff00>{BLOCK}",
        layout="aviutl",
        color=CLEAR,
        border_width=AnimatedValue(20.0),
        border_color=HALF_RED,
    ),
}


@pytest.mark.parametrize("name", list(ONCE))
def test_translucent_paint_lands_only_once(name: str) -> None:
    """半透明の物は重なった所でも 1 回だけ塗る 2 回塗ると、そこだけ濃い筋が出る"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    if "@Courier New" in str(ONCE[name].params["text"]) and not QFont("Courier New").exactMatch():
        pytest.skip("Courier New が入っていない")

    alpha = alpha_of(ONCE[name])

    assert alpha.any()
    assert int(alpha.max()) <= 130


def test_translucent_shadow_of_layers_is_even_where_they_overlap() -> None:
    """別の層の字の影が重なっても濃さが揃う 層ごとに重ねると、重なった所だけ濃くなる"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    source = text(
        f"{BLOCK}<#00ff00>{BLOCK}",
        layout="aviutl",
        color=CLEAR,
        letter_spacing=AnimatedValue(-50.0),
        shadow_x=AnimatedValue(200.0),
        shadow_color=(0.0, 0.0, 0.0, 0.5),
    )

    alpha = alpha_of(source)

    assert alpha.any()
    # 影の色の不透明度 0.5 を超える所があれば、影が 2 重に重なっている
    assert int(alpha.max()) <= 130


def test_a_negative_shape_tag_stays_text() -> None:
    """負の倍率の字は作らない 裏返した字は輪郭の向きが逆で、重なると WindingFill で穴になる

    裏返しを読めるようにするときは、輪郭の向きをそろえてから集める
    """
    lines = parse_tags("<tw-1>字<th-1>字", 100.0)

    assert lines[0].text == "<tw-1>字<th-1>字"


#: 絵の大きさと、プレビューの縮め方（絵の画素 1 つが画面の画素いくつ分かの逆数）
CANVASES = [(320, 180, 1.0), (1280, 720, 1.0), (1920, 1080, 1.0), (3840, 2160, 1.0)]


@pytest.mark.parametrize("glyph_size", [24.0, 40.0, 100.0])
@pytest.mark.parametrize(("width", "height", "scale"), CANVASES)
def test_no_hole_at_any_canvas_or_glyph_size(
    width: int, height: int, scale: float, glyph_size: float
) -> None:
    """絵の大きさや字の大きさが変わっても穴が開かない 塗りの決まりは大きさに依らないはず"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    source = text(
        BLOCK * 2, size=AnimatedValue(glyph_size), letter_spacing=AnimatedValue(-glyph_size / 2.0)
    )

    assert_no_gap(render_alpha(source, width, height, scale))


@pytest.mark.parametrize("scale", [0.5, 0.25])
def test_no_hole_in_a_shrunk_preview(scale: float) -> None:
    """画質を落としたプレビューでも穴が開かない 描く座標ごと縮めるので同じ形のはず"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    source = text(BLOCK * 2, letter_spacing=AnimatedValue(-50.0))

    assert_no_gap(render_alpha(source, round(1280 * scale), round(720 * scale), scale))


@pytest.mark.parametrize("divisor", [1, 2, 4])
def test_no_hole_through_the_renderer(gl_context: OffscreenGLContext, divisor: int) -> None:
    """プレビューと書き出しが通るレンダラでも穴が開かない 画質の 1/2 と 1/4 も同じ"""
    if not has_block():
        pytest.skip("■ を塗り切る書体が無い")
    settings = ProjectSettings(width=640, height=360, frame_rate=FrameRate(30))
    project = Project.create(settings)
    track = Track(kind=TrackKind.VIDEO, name="V")
    source = text(BLOCK * 2, letter_spacing=AnimatedValue(-50.0))
    for command in (AddTrack(track), AddClip(track.id, Clip(0, 10, source=source))):
        project = command.apply(project)

    renderer = FrameRenderer(project, context=gl_context, quality=RenderQuality(divisor))
    try:
        image = renderer.render(0)
    finally:
        renderer.close()

    assert_no_gap(image[..., :3].mean(axis=2) > 128)


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
