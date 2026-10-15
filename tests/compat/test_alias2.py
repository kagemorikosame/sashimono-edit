"""AviUtl2 世代のエイリアス（``.object``）

AviUtl1 の ``.exa`` とは節の名前もパラメータ名もテキストの入り方も違う
ここに並んでいるのは全部、``%PROGRAMDATA%\\aviutl2\\Alias`` に実際に置かれていた
配布エイリアスを読ませて見つかったもの
"""

from __future__ import annotations

import pytest

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.model import AnimatedValue
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)

#: 実際の配布エイリアスと同じ書き方をした最小の 1 本
SUBTITLE = (
    "[Object]"
    + chr(10)
    + "frame=0,179"
    + chr(10)
    + "[Object.0]"
    + chr(10)
    + "effect.name=テキスト"
    + chr(10)
    + "サイズ=60.00"
    + chr(10)
    + "字間=1.00"
    + chr(10)
    + "行間=8.00"
    + chr(10)
    + "フォント=Noto Sans JP"
    + chr(10)
    + "文字色=ffee00"
    + chr(10)
    + "影・縁色=102030"
    + chr(10)
    + "文字装飾=縁取り文字（太）"
    + chr(10)
    + "文字揃え=中央揃え[下]"
    + chr(10)
    + "B=1"
    + chr(10)
    + "I=0"
    + chr(10)
    + "テキスト=一行目"
    + chr(92)
    + "n二行目"
    + chr(10)
    + "[Object.1]"
    + chr(10)
    + "effect.name=標準描画"
    + chr(10)
    + "X=0.00"
    + chr(10)
    + "Y=400.00"
    + chr(10)
    + "拡大率=100.000"
    + chr(10)
    + "透明度=25.00"
    + chr(10)
    + "合成モード=加算"
    + chr(10)
)


def mapped(text: str) -> object:
    return map_object(parse_exo(text).objects[0], RATE, report=CompatibilityReport())


class TestSections:
    def test_the_object_section_is_recognised(self) -> None:
        # ``[Object]`` ``[Object.0]`` AviUtl1 の ``[0]`` ``[0.0]`` とは違う
        exo = parse_exo(SUBTITLE)
        assert len(exo.objects) == 1
        assert exo.generation == 2

    def test_the_effect_name_key(self) -> None:
        # 要素の名前は ``_name`` ではなく ``effect.name`` に入る
        entry = parse_exo(SUBTITLE).objects[0].content
        assert entry is not None
        assert entry.name == "テキスト"
        assert "effect.name" not in entry.params

    def test_the_frame_line_gives_the_span(self) -> None:
        # ``frame=0,179`` は 0 始まりで終端を含む 180 フレーム
        obj = parse_exo(SUBTITLE).objects[0]
        assert (obj.start, obj.end, obj.duration) == (0, 179, 180)

    def test_an_alias_without_a_span_says_so(self) -> None:
        # 長さを書かないエイリアスがある 1 フレームのクリップを置くのではなく、
        # 置く側が既定の長さを決められるようにしておく
        exo = parse_exo("[Object]" + chr(10) + "[Object.0]" + chr(10) + "effect.name=テキスト")
        assert exo.objects[0].span_given is False


class TestTextField:
    def test_the_text_is_plain(self) -> None:
        # AviUtl1 は UTF-16LE の 16 進 AviUtl2 は素のまま
        entry = parse_exo(SUBTITLE).objects[0].content
        assert entry is not None
        assert entry.text() == "一行目" + chr(10) + "二行目"

    def test_the_old_generation_still_decodes_hex(self) -> None:
        old = (
            "[0]"
            + chr(10)
            + "start=1"
            + chr(10)
            + "end=30"
            + chr(10)
            + "[0.0]"
            + chr(10)
            + "_name=テキスト"
            + chr(10)
            + "text=4200"
        )
        entry = parse_exo(old).objects[0].content
        assert entry is not None
        assert entry.text() == "B"


class TestDecoration:
    def test_the_outline_thickness_follows_the_font_size(self) -> None:
        # 装飾は文字サイズに対する割合で持つ 固定の画素数にすると、
        # サイズを変えたときだけ縁の太さが合わなくなる
        item = mapped(SUBTITLE)
        source = item.clip.source  # type: ignore[attr-defined]
        assert source is not None
        width = source.params["border_width"]
        assert isinstance(width, AnimatedValue)
        assert width.at(0) == pytest.approx(60.0 * 0.12)

    def test_the_outline_takes_the_edge_colour(self) -> None:
        item = mapped(SUBTITLE)
        source = item.clip.source  # type: ignore[attr-defined]
        assert source is not None
        assert source.params["border_color"] == pytest.approx(
            (0x10 / 255, 0x20 / 255, 0x30 / 255, 1.0)
        )

    def test_a_shadow_decoration_offsets_down_and_right(self) -> None:
        text = SUBTITLE.replace("縁取り文字（太）", "影付き文字")
        source = mapped(text).clip.source  # type: ignore[attr-defined]
        assert source is not None
        assert source.params["shadow_x"].at(0) > 0
        # 画面では下へ落ちる 設定の Y は上向きなので負
        assert source.params["shadow_y"].at(0) < 0
        assert "border_width" not in source.params

    def test_the_thin_outline_is_thinner_than_the_thick_one(self) -> None:
        thin = mapped(SUBTITLE.replace("（太）", "（細）")).clip.source  # type: ignore[attr-defined]
        thick = mapped(SUBTITLE).clip.source  # type: ignore[attr-defined]
        assert thin.params["border_width"].at(0) < thick.params["border_width"].at(0)

    def test_a_plain_decoration_adds_nothing(self) -> None:
        source = mapped(SUBTITLE.replace("縁取り文字（太）", "標準文字")).clip.source  # type: ignore[attr-defined]
        assert "border_width" not in source.params
        assert "shadow_x" not in source.params

    def test_an_unknown_decoration_is_recorded(self) -> None:
        report = CompatibilityReport()
        obj = parse_exo(SUBTITLE.replace("縁取り文字（太）", "未知の飾り")).objects[0]
        map_object(obj, RATE, report=report)
        assert any("未知の飾り" in line for line in report.lines())

    def test_half_width_brackets_mean_the_same(self) -> None:
        # 手で書き換える人がいる 綴りの違いだけで装飾が消えるのは困る
        source = mapped(SUBTITLE.replace("（太）", "(太)")).clip.source  # type: ignore[attr-defined]
        assert source.params["border_width"].at(0) == pytest.approx(60.0 * 0.12)


class TestLayout:
    def test_aviutl_text_is_laid_out_like_aviutl2(self) -> None:
        # 標準の組み方のままだと入れ物が字の形になり、画像合成や万華鏡の基準が
        # AviUtl2 より 22〜24 画素内側になる 太字も右へ 4〜5 画素寄る（#64）
        source = mapped(SUBTITLE).clip.source  # type: ignore[attr-defined]
        assert source.params["layout"] == "aviutl"


class TestAlignment:
    def test_the_combined_alignment_splits_into_two(self) -> None:
        # ``中央揃え[下]`` は横が中央、縦が下
        source = mapped(SUBTITLE).clip.source  # type: ignore[attr-defined]
        assert (source.params["align"], source.params["valign"]) == ("center", "bottom")

    def test_the_middle_variant(self) -> None:
        source = mapped(SUBTITLE.replace("中央揃え[下]", "左寄せ[中]")).clip.source  # type: ignore[attr-defined]
        assert (source.params["align"], source.params["valign"]) == ("left", "middle")


class TestDrawSettings:
    def test_the_blend_mode_arrives_as_a_name(self) -> None:
        # AviUtl1 は番号、AviUtl2 は表示名
        assert mapped(SUBTITLE).clip.blend_mode == "add"  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("name", "mode"),
        [
            ("オーバーレイ", "overlay"),
            ("比較(明)", "lighten"),
            ("比較(暗)", "darken"),
            ("減算", "subtract"),
        ],
    )
    def test_every_aviutl_blend_is_carried_over(self, name: str, mode: str) -> None:
        # 合成器に揃ったので、通常へ落とさずにそのまま写す 落とすと配布物の光や影が消える
        report = CompatibilityReport()
        obj = parse_exo(SUBTITLE.replace("合成モード=加算", f"合成モード={name}")).objects[0]
        item = map_object(obj, RATE, report=report)
        assert item is not None
        assert item.clip.blend_mode == mode
        assert not any(name in line for line in report.lines())

    def test_an_unknown_blend_falls_back_and_is_recorded(self) -> None:
        # 知らない名前を黙って通常にすると、何が足りないのか分からない
        report = CompatibilityReport()
        obj = parse_exo(SUBTITLE.replace("合成モード=加算", "合成モード=差の絶対値")).objects[0]
        item = map_object(obj, RATE, report=report)
        assert item is not None
        assert item.clip.blend_mode == "normal"
        assert any("差の絶対値" in line for line in report.lines())

    def test_opacity_comes_from_the_transparency(self) -> None:
        assert mapped(SUBTITLE).clip.opacity.at(0) == pytest.approx(0.75)  # type: ignore[attr-defined]

    def test_the_position_becomes_a_transform(self) -> None:
        effects = mapped(SUBTITLE).clip.effects  # type: ignore[attr-defined]
        assert effects[0].kind == "transform"
        # 画面の Y は下向き、AviUtl の Y も下向き こちらは上向きなので反転する
        assert effects[0].params["pos_y"].at(0) == pytest.approx(-400.0)


class TestGradient:
    # 図形に掛けたグラデーション テキストのすぐ後ろの物は塗りへ写る（#312
    # tests/compat/test_aviutl_text_fill.py）ので、エフェクトとしての読み方は図形で見る
    GRADIENT = (
        "[Object]"
        + chr(10)
        + "frame=0,59"
        + chr(10)
        + "[Object.0]"
        + chr(10)
        + "effect.name=図形"
        + chr(10)
        + "サイズ=100"
        + chr(10)
        + "[Object.1]"
        + chr(10)
        + "effect.name=グラデーション"
        + chr(10)
        + "強さ=80.0"
        + chr(10)
        + "角度=90.00"
        + chr(10)
        + "幅=120"
        + chr(10)
        + "形状=線形"
        + chr(10)
        + "開始色=00fb7b"
        + chr(10)
        + "終了色=beb804"
        + chr(10)
    )

    def test_the_gradient_is_mapped(self) -> None:
        effects = mapped(self.GRADIENT).clip.effects  # type: ignore[attr-defined]
        assert [e.kind for e in effects] == ["gradient"]

    def test_its_numbers_come_across(self) -> None:
        effect = mapped(self.GRADIENT).clip.effects[0]  # type: ignore[attr-defined]
        assert effect.params["strength"].at(0) == pytest.approx(80.0)
        assert effect.params["angle"].at(0) == pytest.approx(90.0)
        assert effect.params["span"].at(0) == pytest.approx(120.0)

    def test_its_colours_are_read_as_colours(self) -> None:
        # 数値として読むと 0 になり、白から黒のグラデーションになってしまう
        effect = mapped(self.GRADIENT).clip.effects[0]  # type: ignore[attr-defined]
        assert effect.params["start_color"] == pytest.approx((0.0, 0xFB / 255, 0x7B / 255, 1.0))

    def test_the_shape_is_read_as_a_choice(self) -> None:
        effect = mapped(self.GRADIENT).clip.effects[0]  # type: ignore[attr-defined]
        assert effect.params["shape"] == "linear"
        radial = mapped(self.GRADIENT.replace("形状=線形", "形状=円形"))
        assert radial.clip.effects[0].params["shape"] == "radial"  # type: ignore[attr-defined]


class TestScriptFilter:
    def test_a_missing_script_is_recorded_by_name(self) -> None:
        # AviUtl2 は ``表示名@ファイル名`` でスクリプトを直接書く
        report = CompatibilityReport()
        text = (
            "[Object]"
            + chr(10)
            + "frame=0,59"
            + chr(10)
            + "[Object.0]"
            + chr(10)
            + "effect.name=テキスト"
            + chr(10)
            + "テキスト=あ"
            + chr(10)
            + "[Object.1]"
            + chr(10)
            + "effect.name=存在しない効果@どこにも無い"
            + chr(10)
        )
        map_object(parse_exo(text).objects[0], RATE, report=report)
        assert any("存在しない効果@どこにも無い" in line for line in report.lines())


class TestEmbeddedLua:
    def test_text_containing_a_lua_block_is_kept_for_rendering(self) -> None:
        # 読み込む時点で走らせたり消したりすると、時刻で変わる字幕が止まる
        # 本文のまま残し、未対応としても記録しない（描くときに走る）
        report = CompatibilityReport()
        text = (
            "[Object]"
            + chr(10)
            + "[Object.0]"
            + chr(10)
            + "effect.name=テキスト"
            + chr(10)
            + "テキスト=<?obj.mes('x')?>"
            + chr(10)
        )
        mapped = map_object(parse_exo(text).objects[0], RATE, report=report)
        assert not any("<?" in line for line in report.lines())
        assert mapped is not None and mapped.clip.source is not None
        assert mapped.clip.source.params["text"] == "<?obj.mes('x')?>"
