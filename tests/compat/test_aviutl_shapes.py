"""AviUtl の図形まわり

AviUtl1 と AviUtl2 で書き方が違う AviUtl2 に置かせたエイリアスから読み取った
（``tests/fixtures/aviutl/probes/kumiki_shape1.object`` など）
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.mapped import MappedObject
from sashimono.core.model import AnimatedValue, GeneratedSource, ParamValue
from sashimono.core.timebase import FrameRate
from sashimono.engine.sources import format_time, render_source, source_canvas, timer_text

RATE = FrameRate(60)


def _object(body: str) -> str:
    """中身 1 つと標準描画だけのオブジェクト ``body`` は ``名前\n項目=値``"""
    name, _, rest = body.partition("\n")
    lines = ["[Object]", "frame=0,59", "[Object.0]", f"effect.name={name}"]
    lines.extend(line for line in rest.splitlines() if line)
    lines += ["[Object.1]", "effect.name=標準描画", "X=0.00", "合成モード=通常"]
    return "\n".join(lines) + "\n"


def _mapped(body: str) -> tuple[MappedObject, CompatibilityReport]:
    report = CompatibilityReport()
    item = map_object(parse_exo(_object(body)).objects[0], RATE, report=report)
    assert item is not None
    return item, report


def _source(body: str) -> GeneratedSource:
    item, _ = _mapped(body)
    assert item.clip.source is not None
    return item.clip.source


def _value(source: GeneratedSource, name: str) -> float:
    value = source.params[name]
    assert isinstance(value, AnimatedValue)
    return value.static


class TestTheSecondGeneration:
    """AviUtl2 は種類を名前で、色を ``色`` で書く"""

    @pytest.mark.parametrize(
        ("named", "expected"),
        [
            ("背景", "background"),
            ("円", "ellipse"),
            ("四角形", "rect"),
            ("三角形", "inscribed_triangle"),
            ("五角形", "pentagon"),
            ("六角形", "hexagon"),
            ("星型", "star"),
        ],
    )
    def test_the_kind_comes_from_the_name(self, named: str, expected: str) -> None:
        # 番号（AviUtl1 の書き方）だけを見ていたので、AviUtl2 の図形は全部 円 になっていた
        source = _source(f"図形\n図形の種類={named}\nサイズ=100\n色=ff0000\nライン幅=4000")
        assert source.params["shape"] == expected

    def test_the_colour_comes_from_the_japanese_key(self) -> None:
        # ``color`` だけを見ていたので、AviUtl2 の図形は必ず白かった
        source = _source("図形\n図形の種類=円\nサイズ=100\n色=ff0000\nライン幅=4000")
        assert source.params["color"] == (1.0, 0.0, 0.0, 1.0)

    def test_a_thick_line_means_filled(self) -> None:
        # ライン幅 4000 は塗りつぶし そのまま線の太さにすると画面を覆う輪郭になる
        source = _source("図形\n図形の種類=円\nサイズ=100\n色=ffffff\nライン幅=4000")
        assert _value(source, "line_width") == 0.0
        assert source.params["outline_only"] is False

    def test_a_thin_line_stays_an_outline(self) -> None:
        # 細い線を塗りつぶしと見なすと、AviUtl で輪郭だけだった円が塗られた円として読み込まれる
        source = _source("図形\n図形の種類=円\nサイズ=100\n色=ffffff\nライン幅=8")
        assert _value(source, "line_width") == 8.0
        assert source.params["outline_only"] is True

    def test_rounded_corners_change_the_kind(self) -> None:
        # 角を丸くする を見落とすと、角丸の四角形が角の立った四角形として読み込まれる
        source = _source("図形\n図形の種類=四角形\nサイズ=100\n色=ffffff\n角を丸くする=1")
        assert source.params["shape"] == "rounded"

    def test_an_unknown_kind_is_recorded(self) -> None:
        # ハート に当たる形はこちらに無い 黙って矩形にすると別の絵が出たまま気付けない
        item, report = _mapped("図形\n図形の種類=ハート\nサイズ=100\n色=ffffff")
        assert item.clip.source is not None
        assert item.clip.source.params["shape"] == "rect"
        assert any("図形の種類: ハート" in line for line in report.lines())


def _bounds(body: str) -> tuple[int, int, int, int, int]:
    """1920x1080 に描いた図形の塗られた範囲 画面の中心から測る

    返すのは 上端・下端（下が正の画素）・下端の行の左端・右端・塗られた画素の数
    """
    image = render_source(_source(body), 1920, 1080)
    assert image is not None
    mask = image[:, :, 3] > 128
    rows, _ = np.nonzero(mask)
    bottom = int(rows.max())
    base = np.nonzero(mask[bottom])[0]
    return (
        int(rows.min()) - 540,
        bottom - 540,
        int(base.min()) - 960,
        int(base.max()) - 960,
        int(mask.sum()),
    )


class TestTheTriangle:
    """AviUtl2 の三角形は円に内接する形

    ``kumiki_p8_tri_*`` を AviUtl2 v2.1.6a に 1920x1080 で書き出させて、
    白く塗られた範囲を画素で測った値と比べる
    """

    @pytest.mark.parametrize(
        ("size", "ratio", "expected"),
        [
            # 頂点が中心の 200 上・底辺が 100 下・底辺の幅 346（半径 200 の円に内接）
            (400, "0.00", (-199, 99, -173, 172, 51955)),
            (200, "0.00", (-99, 49, -86, 85, 12981)),
            # 縦横比は同じ形を横か縦に縮めるだけ 頂点と中心の位置は変わらない
            (400, "50.00", (-198, 99, -86, 85, 25972)),
            (400, "-50.00", (-100, 49, -173, 172, 25985)),
        ],
    )
    def test_it_matches_what_aviutl2_draws(
        self, size: int, ratio: str, expected: tuple[int, int, int, int, int]
    ) -> None:
        # 四角に合わせた三角形で写すと、サイズ 400 で底辺が 100 下へはみ出し、
        # 幅も 54 広がって面積が 1.5 倍になる
        top, bottom, left, right, area = _bounds(
            f"図形\n図形の種類=三角形\nサイズ={size}\n縦横比={ratio}\n色=ffffff\nライン幅=4000"
        )
        assert abs(top - expected[0]) <= 2
        assert abs(bottom - expected[1]) <= 2
        assert abs(left - expected[2]) <= 2
        assert abs(right - expected[3]) <= 2
        # 面積は圧縮で縁がにじむ分だけずれる 1% に収まれば同じ形
        assert abs(area - expected[4]) <= expected[4] * 0.01


def _spans(body: str, row: int = 0) -> list[tuple[int, int]]:
    """1920x1080 に描いた図形の、中心から ``row`` 画素下の行で塗られた帯

    返すのは ``(左端, 右端)`` の並び どちらも画面の中心から測る
    """
    image = render_source(_source(body), 1920, 1080)
    assert image is not None
    mask = image[540 + row, :, 3] > 128
    columns = np.nonzero(mask)[0]
    found: list[tuple[int, int]] = []
    start = previous = -10
    for column in columns:
        if column != previous + 1:
            if start >= 0:
                found.append((start - 960, previous - 960))
            start = int(column)
        previous = int(column)
    if start >= 0:
        found.append((start - 960, previous - 960))
    return found


class TestTheOutline:
    """AviUtl2 の図形のライン幅は**図形の内側**に引かれる

    ``kumiki_p8_tri_s400_line20`` を AviUtl2 v2.1.6a に 1920x1080 で書き出させて、
    白い画素の位置を測った 外形は塗りつぶしたときと同じで、中心の行を横に切ると
    左右それぞれ 24 画素（= 20 / sin 60 度）の帯だった
    """

    _TRIANGLE = "図形\n図形の種類=三角形\nサイズ=400\n縦横比=0.00\n色=ffffff\nライン幅=20"

    def test_the_outline_does_not_grow_the_shape(self) -> None:
        # 輪郭の中央に引くと、外形が太さの半分（20 なら約 10 画素）外へ広がり、
        # 塗りつぶした同じ図形より一回り大きく見える
        top, bottom, left, right, _ = _bounds(self._TRIANGLE)
        assert abs(top - -199) <= 2
        assert abs(bottom - 99) <= 2
        assert abs(left - -173) <= 2
        assert abs(right - 172) <= 2

    def test_the_band_lies_inside_the_contour(self) -> None:
        # 中央に引くと帯が外へずれ、左の帯が -126..-104 まで出る
        bands = _spans(self._TRIANGLE)
        assert len(bands) == 2
        (left_start, left_end), (right_start, right_end) = bands
        assert abs(left_start - -116) <= 2
        assert abs(right_end - 115) <= 2
        assert abs((left_end - left_start + 1) - 24) <= 2
        assert abs((right_end - right_start + 1) - 24) <= 2

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            # 四角形 サイズ 400 ライン幅 20 外形は ±200 のまま、帯は内側の 20 画素
            ("図形\n図形の種類=四角形\nサイズ=400\n色=ffffff\nライン幅=20", (-200, -181, 180, 199)),
            ("図形\n図形の種類=四角形\nサイズ=400\n色=ffffff\nライン幅=60", (-200, -141, 140, 199)),
            # 円も同じ 半径 200 の円の内側に帯が入る
            ("図形\n図形の種類=円\nサイズ=400\n色=ffffff\nライン幅=20", (-200, -180, 179, 199)),
            ("図形\n図形の種類=円\nサイズ=400\n色=ffffff\nライン幅=60", (-200, -140, 139, 199)),
        ],
    )
    def test_every_shape_draws_the_line_inside(
        self, body: str, expected: tuple[int, int, int, int]
    ) -> None:
        # AviUtl2 v2.1.6a に 1920x1080 で書き出させた kumiki_p9_* を測った値
        # 中央に引くと外形が太さの半分ぶん広がり、どの図形も一回り大きく見える
        bands = _spans(body)
        assert len(bands) == 2
        measured = (bands[0][0], bands[0][1], bands[1][0], bands[1][1])
        assert all(abs(a - b) <= 2 for a, b in zip(measured, expected, strict=True)), measured

    def test_the_fan_ring_sits_inside_too(self) -> None:
        # 扇型（カスタムオブジェクト）の輪も内側 円のときと同じ位置に出る
        # 中央に引いたままだと見本との差が 6.7 残る（内側にすると 2.0）
        # 残りの差は輪郭の引き方とは別の所（中心の角の丸め方と 中心角 の基準）
        bands = _spans("扇型\n中心角=360.0\nサイズ=400.0\nライン幅=40.0\n色=ffffff")
        assert abs(bands[0][0] - -200) <= 2
        assert abs(bands[0][1] - -160) <= 2
        assert abs(bands[-1][0] - 159) <= 2
        assert abs(bands[-1][1] - 199) <= 2

    def test_the_polygon_keeps_its_centred_line(self) -> None:
        # 多角形（カスタムオブジェクト）だけは輪郭の中央に引かれる 図形と同じにすると
        # ±150 の四角形の外形が ±170 から ±150 へ縮む（kumiki_p9_poly_sq_l40 で確かめた）
        bands = _spans(
            "多角形\n色=ffffff\nライン幅=40\n頂点数=4\n"
            "座標=-150,-150,150,-150,150,150,-150,150\n簡易塗り潰し=0"
        )
        assert len(bands) == 2
        assert abs(bands[0][0] - -170) <= 2
        assert abs(bands[1][1] - 169) <= 2

    def test_the_line_align_comes_from_the_mapping(self) -> None:
        # ここが落ちると、読み込んだ図形が今までどおり輪郭の中央に線を引く
        assert _source(self._TRIANGLE).params["line_align"] == "inside"

    @pytest.mark.parametrize("name", ["図形", "扇型"])
    def test_the_first_generation_keeps_the_centred_line(self, name: str) -> None:
        # 内側に引くと確かめられたのは AviUtl2 だけ AviUtl1 のファイルにも当てると、
        # 今まで読めていた絵が測らないまま一回り小さくなる
        body = f"[0]\nstart=1\nend=30\nlayer=1\n[0.0]\n_name={name}\nサイズ=400\nライン幅=20\n"
        report = CompatibilityReport()
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None
        assert item.clip.source is not None
        assert item.clip.source.params["line_align"] == "center"

    def _native(self, align: str | None) -> int:
        """Sashimono で作った幅 400 の輪郭だけの四角形の、右端の画素"""
        params: dict[str, ParamValue] = {
            "shape": "rect",
            "width": AnimatedValue(400.0),
            "height": AnimatedValue(400.0),
            "color": (1.0, 1.0, 1.0, 1.0),
            "line_width": AnimatedValue(20.0),
            "outline_only": True,
        }
        if align is not None:
            params["line_align"] = align
        image = render_source(GeneratedSource(kind="shape", params=params), 1920, 1080)
        assert image is not None
        return int(np.nonzero(image[540, :, 3] > 128)[0].max()) - 960

    def test_a_native_shape_draws_the_line_inside_by_default(self) -> None:
        # 既定を内側にしたので、幅 400 の図形の外形が 400（±200）になる
        # 中央のままだと 420（±210）に広がり、指定した幅と見た目が食い違う
        assert abs(self._native(None) - 199) <= 2

    def test_the_canvas_does_not_grow_for_an_inside_line(self) -> None:
        # 内側の線は外形を超えないので、絵を太さぶん広げる必要が無い
        # 広げたままだと、線の太い大きな図形で毎フレーム余分に大きな絵を作る
        def params(align: str) -> dict[str, ParamValue]:
            return {
                "shape": "rect",
                "width": AnimatedValue(2000.0),
                "height": AnimatedValue(2000.0),
                "line_width": AnimatedValue(200.0),
                "outline_only": True,
                "line_align": align,
            }

        inside = source_canvas(GeneratedSource(kind="shape", params=params("inside")), 1920, 1080)
        centred = source_canvas(GeneratedSource(kind="shape", params=params("center")), 1920, 1080)
        assert inside[0] < centred[0]
        # 外形は 2000x2000 の対角線 回しても収まる大きさで足りる
        assert inside[0] == 2830

    def test_the_centred_line_is_still_available(self) -> None:
        # 前の版の見た目に戻せること ここが落ちると、古い作品を直す手立てが無くなる
        assert abs(self._native("center") - 209) <= 2


class TestTheFirstGeneration:
    def test_the_number_still_works(self) -> None:
        # AviUtl1 の書き方（番号と color）も読めること
        source = _source("図形\ntype=2\nサイズ=100\ncolor=00ff00\nライン幅=4000")
        assert source.params["shape"] == "triangle"
        assert source.params["color"] == (0.0, 1.0, 0.0, 1.0)


class TestConcentration:
    def test_the_custom_object_becomes_a_shape(self) -> None:
        # 集中線を図形の集中線として写せないと、読み込んだ集中線が別の絵で描かれる
        # 速さは既定の 25 と違う値にする 同じ値だと速さを読み落としても通る
        source = _source("集中線\n濃さ=40.0\n速さ=60.0\n中心幅=0\n色=ffffff")
        assert source.kind == "shape"
        assert source.params["shape"] == "concentration"
        assert _value(source, "flicker") == 60.0
        # AviUtl の集中線は中心幅が 0 でも画面いっぱい 空きの有無で描き方を
        # 分けると、空きを 0 にしたものだけが小さな円に縮む
        assert source.params["fill_frame"] is True

    def test_the_density_drives_both_the_count_and_the_width(self) -> None:
        # AviUtl2 に濃さ 40・80・160 を描かせて測ると、線の占める角度が
        # 18%・約 65%・100% と 2 乗で増える 本数だけに渡すと、
        # 濃くしたはずの画面が埋まらず、線が増えただけの絵になる
        source = _source("集中線\n濃さ=40.0\n速さ=25.0\n中心幅=0\n色=ffffff")
        assert _value(source, "density") == 64.0
        assert _value(source, "line_thickness") == 33.0
        thick = _source("集中線\n濃さ=80.0\n速さ=25.0\n中心幅=0\n色=ffffff")
        assert _value(thick, "line_thickness") == 132.0

    def test_the_centre_gap_is_the_radius_of_the_hole(self) -> None:
        # AviUtl2 の実物で 中心幅 300 の絵は半径 300 まで真っ暗だった
        # 直径として扱うと、空きが半分になって線が中心まで伸びる
        source = _source("集中線\n濃さ=40.0\n速さ=25.0\n中心幅=300.0\n色=ffffff")
        assert _value(source, "center_gap") == 300.0


class TestTheThingsReviewFound:
    def test_a_missing_line_width_means_filled(self) -> None:
        # AviUtl1 の図形に ライン幅 は無い 輪郭だけにすると、
        # 塗ってあった図形が中抜きになる
        source = _source("図形\ntype=1\nサイズ=100\ncolor=ffffff")
        assert source.params["outline_only"] is False
        assert _value(source, "line_width") == 0.0

    def test_a_zero_line_width_does_not_make_the_shape_hollow(self) -> None:
        # 0 を輪郭として扱うと、塗ってあった図形が中抜きになる
        source = _source("図形\n図形の種類=円\nサイズ=100\n色=ffffff\nライン幅=0")
        assert source.params["outline_only"] is False

    def test_the_concentration_keeps_its_motion(self) -> None:
        # 動きを落とすと、濃さが変わっていく集中線が最初の濃さで止まる
        source = _source("集中線\n濃さ=10,80,直線移動,0\n速さ=25\n中心幅=0\n色=ffffff")
        density = source.params["density"]
        assert isinstance(density, AnimatedValue)
        assert density.is_animated

    def test_a_moving_centre_gap_keeps_moving(self) -> None:
        # 動きを落とすと、広がっていく空きが最初の大きさで止まる
        source = _source("集中線\n濃さ=40\n速さ=25\n中心幅=0,300,直線移動,0\n色=ffffff")
        gap = source.params["center_gap"]
        assert isinstance(gap, AnimatedValue)
        assert gap.is_animated

    def test_a_moving_size_is_recorded(self) -> None:
        # 大きさは サイズ と 縦横比 から計算してから渡すので、動きを残せない
        _, report = _mapped("図形\n図形の種類=円\nサイズ=100,400,直線移動,0\n色=ffffff")
        assert any("図形の動くサイズ" in line for line in report.lines())

    def test_a_still_expression_on_a_size_is_recorded(self) -> None:
        # 値が同じでも式なら時間で変わる 記録しないと、大きさが止まったことに気付けない
        _, report = _mapped("図形\n図形の種類=円\nサイズ=100,100,瞬間移動,8|100+time\n色=ffffff")
        assert any("図形の動くサイズ" in line for line in report.lines())


class TestCustomObjects:
    """AviUtl2 の「カスタムオブジェクト」 効果ではなく中身として置かれる

    どれも AviUtl2 に置かせたエイリアスから項目名を読み取った
    """

    def test_a_fan_keeps_its_angle(self) -> None:
        # 中心角を落とすと、扇が必ず全円（360 度）になる
        source = _source("扇型\n中心角=270.0\nサイズ=100.0\nライン幅=2000.0\n色=ffffff")
        assert source.params["shape"] == "fan"
        assert _value(source, "span") == 270.0
        # ライン幅 2000 は図形より大きい 塗りつぶしとして扱う
        assert source.params["outline_only"] is False

    def test_a_polygon_becomes_a_line_through_its_points(self) -> None:
        # 座標は x と y を並べて書く（Y は下が正） 反転を忘れると上下逆の形になる
        source = _source("多角形\n色=ffffff\nライン幅=20\n頂点数=3\n座標=0,-150,130,75,-130,75")
        assert source.params["shape"] == "polyline"
        assert source.params["points"] == "0.0,150.0;130.0,-75.0;-130.0,-75.0"
        # 閉じないと、最後の頂点から最初へ戻る線が引かれず開いた折れ線になる
        assert source.params["closed"] is True

    def test_a_polygon_is_hollow_unless_told_otherwise(self) -> None:
        # 中抜きの指定を落とすと、線だけのはずの多角形が塗りつぶされる
        source = _source(
            "多角形\n色=ffffff\nライン幅=20\n座標=0,-150,130,75,-130,75\n簡易塗り潰し=0"
        )
        assert source.params["fill_color"] == (0.0, 0.0, 0.0, 0.0)

    def test_a_filled_polygon_uses_its_colour(self) -> None:
        # 塗り潰しの指定を落とすと、塗ってあるはずの中が透明になる
        source = _source(
            "多角形\n色=ff0000\nライン幅=20\n座標=0,-150,130,75,-130,75\n簡易塗り潰し=1"
        )
        assert source.params["fill_color"] == (1.0, 0.0, 0.0, 1.0)

    def test_a_broken_polygon_is_recorded(self) -> None:
        # 座標が読めないと線が 1 本も出ない 黙って空の図形を置くと気付けない
        _, report = _mapped("多角形\n色=ffffff\nライン幅=20\n座標=")
        assert any("多角形の座標" in line for line in report.lines())

    def test_a_counter_becomes_a_timer(self) -> None:
        # カウンターが数の進む文字として読み込まれず、初期値・速さ・フォントが
        # AviUtl と違う表示になる
        source = _source(
            "カウンター\n初期値=5.0\n速度=2.0\nサイズ=34.0"
            + "\n表示形式=標準\nフォント名=MS UI Gothic\n文字色=ffffff"
        )
        assert source.kind == "text"
        assert _value(source, "timer_start") == 5.0
        # 速度は 1 秒あたりの進み方 こちらは百分率なので 100 倍する
        assert _value(source, "timer_rate") == 200.0
        assert source.params["font"] == "MS UI Gothic"
        # テキストと同じく AviUtl2 の入れ物で組む 字の形だと効果の基準がずれる（#64）
        assert source.params["layout"] == "aviutl"

    def test_an_unknown_counter_format_is_recorded(self) -> None:
        # 書式を落とすと、時計のつもりの表示がただの秒数になる
        _, report = _mapped(
            "カウンター\n初期値=0\n速度=1\n表示形式=時分秒\nフォント名=MS UI Gothic"
        )
        assert any("表示形式" in line for line in report.lines())

    def test_the_motion_trail_is_not_a_polyline(self) -> None:
        # ライン(移動軌跡) は折れ線ではなく、動いた跡を描く別物
        # 折れ線として写すと、まったく違う絵が出たまま気付けない
        # 描き方は tests/compat/test_aviutl_time_objects.py で確かめる
        source = _source("ライン(移動軌跡)\nライン幅=16.0\n先端=48.0\n色=ffffff")
        assert source.params["shape"] == "motion_trail"


class TestWhatTheSecondReviewFound:
    def test_the_counter_does_not_wrap_at_a_minute(self) -> None:
        # 書式 s は 60 で分へ繰り上がる カウンターは数を数えるだけなので、
        # そのまま使うと 60 で 0 に戻り、100 のつもりが 40 と出る
        source = _source("カウンター\n初期値=0\n速度=1\nフォント名=MS UI Gothic")
        assert source.params["timer_format"] == "n"
        assert format_time(100.0, "n") == "100"

    def test_the_counter_keeps_its_decoration(self) -> None:
        # 装飾タイプ を読まないと、縁取りのカウンターが素の文字になる
        source = _source(
            "カウンター\n初期値=0\n速度=1\n装飾タイプ=縁取り文字"
            + "\n影・縁色=000000\nフォント名=MS UI Gothic"
        )
        assert "border_width" in source.params or "shadow_x" in source.params

    def test_a_broken_coordinate_drops_the_whole_polygon(self) -> None:
        # 読めない値だけを捨てると、その後ろの x と y が入れ替わって別の形になる
        _, report = _mapped("多角形\n色=ffffff\nライン幅=20\n座標=0,だめ,130,75")
        assert any("多角形の座標" in line for line in report.lines())

    def test_an_odd_number_of_coordinates_is_refused(self) -> None:
        # x と y の組にできない並び 1 つ足りないまま組むと全部ずれる
        _, report = _mapped("多角形\n色=ffffff\nライン幅=20\n座標=0,-150,130")
        assert any("多角形の座標" in line for line in report.lines())

    def test_extra_coordinates_are_trimmed_to_the_corner_count(self) -> None:
        # 頂点数 より座標が多いファイル 余分を描くと形が変わる
        source = _source("多角形\n色=ffffff\nライン幅=20\n頂点数=2\n座標=0,0,10,0,20,0")
        assert source.params["points"] == "0.0,-0.0;10.0,-0.0"

    def test_a_moving_fan_angle_is_recorded(self) -> None:
        # 動きを落とすと、開いていく扇が開いたまま止まる
        _, report = _mapped("扇型\n中心角=0,270,直線移動,0\nサイズ=100\n色=ffffff")
        assert any("扇型の動く中心角" in line for line in report.lines())

    def test_a_moving_counter_speed_is_recorded(self) -> None:
        # 速度の動きは先頭の値で止めて写す 記録に残らないと、だんだん速く数えるはずの
        # カウンターが一定の速さになったことに互換性レポートで気付けない
        _, report = _mapped("カウンター\n初期値=0\n速度=1,5,直線移動,0\nフォント名=MS UI Gothic")
        assert any("カウンターの動く速度" in line for line in report.lines())

    def test_a_counter_can_go_below_zero(self) -> None:
        # 0 で止めると、下がっていくはずの数字が途中から動かなくなる
        assert format_time(-5.0, "n") == "-5"

    def test_a_coordinate_that_is_not_finite_is_refused(self) -> None:
        # NaN や無限大は描画の側で落ちる 受け取ると黙って別の形になる
        _, report = _mapped("多角形\n色=ffffff\nライン幅=20\n座標=0,nan,130,75")
        assert any("多角形の座標" in line for line in report.lines())

    def test_too_few_corners_are_recorded_without_trimming(self) -> None:
        # 頂点数 が座標より多いファイル 切り詰めないが、食い違いは残す
        _, report = _mapped("多角形\n色=ffffff\nライン幅=20\n頂点数=5\n座標=0,0,10,0")
        assert any("頂点数と座標の数が合わない" in line for line in report.lines())

    def test_a_clock_format_still_stops_at_zero(self) -> None:
        # 時計が負になることはない 通算の n だけ負の値を出す
        assert timer_text({"_seconds": 0.0, "timer_start": -5.0, "timer_format": "s"}) == "0"
        assert timer_text({"_seconds": 0.0, "timer_start": -5.0, "timer_format": "n"}) == "-5"

    def test_a_value_that_is_not_a_number_does_not_stop_the_drawing(self) -> None:
        # NaN を int() へ渡すと例外になり、そのフレームの描画ごと止まる
        assert format_time(float("nan"), "n") == "0"

    def test_a_corner_count_mismatch_is_recorded_even_with_no_points(self) -> None:
        # 座標を全部捨てたときも、頂点数の食い違いは残す
        _, report = _mapped("多角形\n色=ffffff\n頂点数=3\n座標=0,だめ")
        assert any("頂点数と座標の数が合わない" in line for line in report.lines())

    def test_an_escaped_n_is_not_the_total_format(self) -> None:
        # 逃がした（文字としての）n まで拾うと、時計の書式が 0 で止まらなくなる
        values = {"_seconds": 0.0, "timer_start": -5.0, "timer_format": "s\\n"}
        assert timer_text(values) == "0n"
