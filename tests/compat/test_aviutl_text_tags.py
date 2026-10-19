"""AviUtl2 のテキストの制御文字を読む（``compat/aviutl/text_tags.py``）

振る舞いは AviUtl2 v2.1.6a に描かせて測った物（Issue #108 ``tools/aviutl_tag_probes.py``）
読み違えると、合成フォントで組み替えた字幕がタグの文字のまま出るか、書体や大きさが
AviUtl2 と違う所で切り替わる
"""

from __future__ import annotations

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.text_tags import DROPPED_STRIKE, TextStyle, parse_tags


def runs(text: str, size: float = 100.0) -> list[list[tuple[str, TextStyle]]]:
    return [[(run.text, run.style) for run in line.runs] for line in parse_tags(text, size)]


class TestFont:
    def test_the_composite_font_output_becomes_runs_without_the_tags(self) -> None:
        # comfont.aux2 が実際に返した形 読まないと <@游明朝> がそのまま字として出る
        text = "<@游明朝>ここに<@><@メイリオ>テキスト<@><@游明朝>を<@><@ＭＳ ゴシック>入力<@>"
        (line,) = runs(text)
        assert [(part, style.font) for part, style in line] == [
            ("ここに", "游明朝"),
            ("テキスト", "メイリオ"),
            ("を", "游明朝"),
            ("入力", "ＭＳ ゴシック"),
        ]

    def test_reset_returns_to_the_object_font_not_the_previous_one(self) -> None:
        # AviUtl2 の見本 tag05 では、入れ子の後の <@> で 4 文字目が設定欄の Arial に戻った
        # 1 つ前へ戻す作りにすると 4 文字目が前の書体のまま残る
        (line,) = runs("H<@A>H<@B>H<@>H<@>H")
        assert [style.font for _part, style in line] == [None, "A", "B", None, None]

    def test_a_font_carries_over_a_line_break(self) -> None:
        # 見本 tag06 改行の後の 2 文字もメイリオの送り幅で描かれた
        first, second = runs("H<@メイリオ>H\nHH<@>H")
        assert [style.font for _part, style in first] == [None, "メイリオ"]
        assert [(part, style.font) for part, style in second] == [("HH", "メイリオ"), ("H", None)]

    def test_the_decoration_number_is_not_taken_for_the_font_name(self) -> None:
        # 書体名に ,3 まで含めると、見つからない書体として既定の書体で描いてしまう
        (line,) = runs("<@メイリオ,3>H")
        assert line[0][1].font == "メイリオ"

    def test_style_changes_add_and_remove_bold_and_italic(self) -> None:
        # 見本 tag28 2 文字目が太字、3 文字目で戻り、4 文字目が斜体、<@> の後は素の字だった
        # 読まないと <@+B> の字がそのまま画面に出る
        (line,) = runs("H<@+B>H<@-B>H<@+I>H<@>H")
        assert [(style.bold, style.italic) for _part, style in line] == [
            (None, None),
            (True, None),
            (False, None),
            (False, True),
            (None, None),
        ]

    def test_the_decoration_number_and_styles_after_it(self) -> None:
        # 見本 tag42 と tag51 番号は 0〜6 の文字装飾 後ろの B は太字
        (line,) = runs("H<@Arial,5>H<@Arial,3B>H<@>H")
        assert [(style.decoration, style.bold) for _part, style in line] == [
            (None, None),
            (5, None),
            (3, True),
            (None, None),
        ]

    def test_written_styles_replace_the_earlier_ones(self) -> None:
        # スタイルを書いた書体の字は書いた物だけになる 前の <@+B> を残すと、斜体だけを
        # 指定した字が太字にもなる 番号だけなら前のスタイルのまま（#183 のレビュー）
        (line,) = runs("<@+B>H<@Arial,3I>H<@+B><@Arial,3>H<@+I><@Arial,3B>H")
        assert [(style.bold, style.italic) for _part, style in line] == [
            (True, None),
            (False, True),
            (True, True),
            (True, False),
        ]

    def test_a_decoration_without_a_font_name_changes_nothing(self) -> None:
        # 見本 tag26 tag27 tag29 <@,3> も <@,0> も <@,3B> も AviUtl2 では何も変えなかった
        # 装飾として読むと、AviUtl2 では素の字の所に縁や影が付く
        (line,) = runs("H<@,3B>H")
        assert "".join(part for part, _style in line) == "HH"
        assert {style for _part, style in line} == {TextStyle()}

    def test_a_decoration_number_out_of_range_stays_as_text(self) -> None:
        # 7 以上の番号は aviutl2.txt に無い 推測で近い装飾を当てない
        (line,) = runs("H<@Arial,9>H")
        assert [part for part, _style in line] == ["H<@Arial,9>H"]


class TestSize:
    def test_relative_sizes_follow_the_current_size(self) -> None:
        # 見本 tag09 <s*2> は設定欄の 100 の倍、続く <s+20> はその 200 に足して 220
        (line,) = runs("H<s50>H<s>H<s*2>H<s+20>H")
        assert [style.size for _part, style in line] == [None, 50.0, None, 200.0, 220.0]

    def test_a_size_carries_over_a_line_break(self) -> None:
        # 見本 tag10 2 行目も 200 のまま描かれた
        first, second, third = runs("H<s200>H\nHH\n<s>HH")
        assert first[-1][1].size == 200.0
        assert second[0][1].size == 200.0
        assert third[0][1].size is None

    def test_the_size_tag_can_change_the_font_and_reset_takes_it_back(self) -> None:
        # 見本 tag11 <s80,Meiryo> の字は Meiryo で探され、<s> の後は設定欄の書体に戻った
        (line,) = runs("H<s80,メイリオ,B,3>H<s>H")
        assert [(style.size, style.font) for _part, style in line] == [
            (None, None),
            (80.0, "メイリオ"),
            (None, None),
        ]

    def test_size_reset_returns_to_the_font_chosen_before_it(self) -> None:
        # 見本 tag24 tag25 <@メイリオ> の後の <s> は、設定欄の Arial ではなくメイリオへ戻った
        # （4 文字目の H の上端がメイリオの 497 Arial なら 499）
        (first,) = runs("H<@メイリオ>H<s50>H<s>H")
        (second,) = runs("H<@メイリオ>H<s50,ＭＳ ゴシック>H<s>H")
        assert [style.font for _part, style in first] == [None, "メイリオ", "メイリオ", "メイリオ"]
        assert [style.font for _part, style in second] == [
            None,
            "メイリオ",
            "ＭＳ ゴシック",
            "メイリオ",
        ]

    def test_font_reset_drops_the_size_font_but_keeps_the_size(self) -> None:
        # 見本 tag45 tag46 <@> の後の字は設定欄の Arial で、大きさは <s> の 50 と 80 のまま
        (first,) = runs("H<s50,ＭＳ ゴシック>H<@>H")
        (second,) = runs("H<@メイリオ>H<s80,ＭＳ ゴシック>H<@>H")
        assert (first[-1][1].font, first[-1][1].size) == (None, 50.0)
        assert (second[-1][1].font, second[-1][1].size) == (None, 80.0)

    def test_an_outline_width_that_is_not_a_number_stays_as_text(self) -> None:
        # float は inf や nan も受け取る inf は描けない縁になり、nan は黙って 0 になって
        # 読めない指定が消える（#183 のレビュー）
        for width in ("inf", "nan", "1e999"):
            text = f"H<s,,,{width}>H"
            (line,) = runs(text)
            assert "".join(part for part, _style in line) == text

    def test_size_styles_and_outline_width_are_undone_by_the_reset(self) -> None:
        # 見本 tag30 tag44 <s,,B> の字は太字、<s,,,8> の字は縁の太さ 8、<s> の後は元へ戻った
        (line,) = runs("H<s,,B>H<s>H<s,,,8>H<s>H")
        assert [(style.bold, style.edge_width, style.size) for _part, style in line] == [
            (None, None, None),
            (True, None, None),
            (None, None, None),
            (None, 8.0, None),
            (None, None, None),
        ]

    def test_an_empty_line_remembers_its_size(self) -> None:
        # 見本 tag22 文字の無い真ん中の行は、その行で指定した 200 の高さだった
        lines = parse_tags("H\n<s200>\nH", 100.0)
        assert lines[1].runs == []
        assert lines[1].end.size == 200.0

    def test_a_size_never_drops_to_zero(self) -> None:
        # 0 以下の大きさで書体を作ると、Qt が既定の大きさで描き、指定と逆に大きくなる
        (line,) = runs("<s-500>H")
        assert line[0][1].size == 1.0

    def test_words_starting_with_s_are_not_tags(self) -> None:
        # <s の後ろを緩く読むと <span> が大きさのタグとして消え、本文の字が欠ける
        (line,) = runs("<span>x")
        assert line == [("<span>x", TextStyle())]


class TestColor:
    def test_text_and_edge_colours_and_reset(self) -> None:
        # 見本 tag08 2 文字目は赤で縁は設定欄の灰、3 文字目は緑で縁は青、4 文字目は元の色
        (line,) = runs("H<#ff0000>H<#00ff00,0000ff>H<#>H")
        assert [(style.color, style.edge) for _part, style in line] == [
            (None, None),
            ("ff0000", None),
            ("00ff00", "0000ff"),
            (None, None),
        ]

    def test_preset_names_use_the_default_palette(self) -> None:
        # 見本 tag23 <#red> は ff0000 で描かれた
        (line,) = runs("H<#red>H<#>H")
        assert line[1][1].color == "ff0000"

    def test_an_unreadable_colour_stays_as_text(self) -> None:
        # 読めない色を黙って捨てると、描けていない指定が画面から消えて気付けない
        # 白へ倒すと、元の色のつもりの字が白で出る
        (line,) = runs("A<#zzz>B")
        assert [part for part, _style in line] == ["A<#zzz>B"]


class TestSpacingAndShape:
    def test_letter_gap_replaces_the_object_setting_until_reset(self) -> None:
        # 見本 tag32 tag47 <gw40> は設定欄の字間 20 に足さず 40 に置き換わった
        (line,) = runs("HH<gw40>HH<gw>HH")
        assert [style.letter_gap for _part, style in line] == [None, 40.0, None]

    def test_line_gap_is_carried_by_the_end_of_the_line(self) -> None:
        # 見本 tag33 tag48 2 行目の頭の <gh40> で 2 行目と 3 行目の間だけが 40 空いた
        # 行間は行の終わりの値で決まるので、3 行目の途中の <gh> で 3 行目の後は元へ戻る
        lines = parse_tags("H\n<gh40>H\nH<gh>\nH", 100.0)
        assert [line.end.line_gap for line in lines] == [None, 40.0, None, None]

    def test_glyph_shapes_and_relative_turns(self) -> None:
        # 見本 tag36 tag49 tag50 横・縦の倍と時計回りの角度 <tr+15> は今の角度に足す
        (line,) = runs("H<tw0.5>H<tw><th2>H<th><tr30>H<tr+15>H<tr>H")
        assert [(style.scale_x, style.scale_y, style.turn) for _part, style in line] == [
            (None, None, None),
            (0.5, None, None),
            (None, 2.0, None),
            (None, None, 30.0),
            (None, None, 45.0),
            (None, None, None),
        ]

    def test_comments_and_block_marks_leave_no_text(self) -> None:
        # 見本 tag37 H が 3 つだけ出た 改行を含むコメントも改行ごと消える
        lines = parse_tags("H<//コメント\n2 行目//>H</>H", 100.0)
        assert len(lines) == 1
        assert lines[0].text == "HHH"


class TestLeftAsText:
    """測った振る舞いを写せていない制御文字は文字のまま残す 黙って捨てると気付けない"""

    def test_timing_position_and_ruby_tags_stay(self) -> None:
        # <r> <w> <c> は時間で出る字が変わり、<p> は組み方の枠まで変える（見本 tag34 tag35 tag52）
        for text in ("H<r5>H", "H<w1>H", "HH<c>H", "H<p+100>H", "漢字<!>かんじ"):
            (line,) = runs(text)
            assert "".join(part for part, _style in line) == text


class TestDroppedParts:
    """読まずに捨てた所を互換性の記録に数える

    数えないと、取り消し線の付くはずの字が素の字で出ていても本人が気付けない
    """

    def test_a_strike_through_is_counted(self) -> None:
        # 3 通りの書き方のどれかで数え漏れると、取り消し線のはずの字が素の字で出ても
        # 互換性レポートに残らず気付けない
        report = CompatibilityReport()
        parse_tags("<@メイリオ,3S>H<@>H<s,,S>H<@+S>H", 100.0, report)
        assert report.missing[DROPPED_STRIKE] == 3
        assert any(DROPPED_STRIKE in line for line in report.lines())

    def test_tags_that_are_read_whole_leave_no_record(self) -> None:
        # 読めている制御文字まで記録すると、互換性レポートが写せている物の行で埋まり、
        # 本当に足りない穴が多い順の上に来ない
        report = CompatibilityReport()
        parse_tags(
            "<@メイリオ,3B>H<@><#ff0000,0000ff>H<#><s*2,,I,4>H<s><gw4>H<tr+5>H", 100.0, report
        )
        assert report.is_empty
