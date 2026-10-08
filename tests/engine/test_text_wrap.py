"""決めた幅での折り返しの位置（#249）

幅は 1 字 10 の決まった物差しで測る 書体に頼らず、折り返す位置の決まり（禁則・英単語・
長すぎる語・手で入れた改行）だけを見る 字の実寸で測るのは描く所の試験（test_text_wrap_render）
"""

from __future__ import annotations

import unicodedata

from sashimono.asr.cleanup import wrap_text
from sashimono.compat.aviutl.text_tags import parse_tags
from sashimono.engine.sources import _revealed, _revealed_lines
from sashimono.engine.text_wrap import graphemes, wrap_lines


def measure(text: str) -> float:
    return 10.0 * len(text)


class TestWrap:
    def test_zero_width_keeps_the_text(self) -> None:
        # 0 は折り返さない 既にある作品と、手で改行したテキストの見た目を変えない
        assert wrap_lines("あいうえおかきくけこ\nさし", 0, measure) == [
            "あいうえおかきくけこ",
            "さし",
        ]

    def test_japanese_breaks_between_any_characters(self) -> None:
        assert wrap_lines("あいうえおかきくけこ", 40, measure) == ["あいうえ", "おかきく", "けこ"]

    def test_a_fitting_line_stays(self) -> None:
        assert wrap_lines("あいう", 30, measure) == ["あいう"]

    def test_manual_breaks_are_kept_and_each_line_wraps(self) -> None:
        # 手で入れた改行は残し、その間の行をさらに幅で折り返す
        assert wrap_lines("あいうえお\nかき", 30, measure) == ["あいう", "えお", "かき"]

    def test_punctuation_does_not_start_a_line(self) -> None:
        # 行頭禁則 句点を次の行の頭に置かず、前の字と一緒に次の行へ送る
        assert wrap_lines("あいう。えお", 30, measure) == ["あい", "う。え", "お"]
        assert wrap_lines("あいう、」えお", 30, measure) == ["あい", "う、」", "えお"]

    def test_small_kana_and_long_vowel_do_not_start_a_line(self) -> None:
        assert wrap_lines("あいっていー", 30, measure) == ["あいっ", "ていー"]

    def test_an_opening_bracket_does_not_end_a_line(self) -> None:
        # 行末禁則 「「」を行の終わりに残さず、次の字と一緒に送る
        assert wrap_lines("あい「うえ」", 30, measure) == ["あい", "「う", "え」"]

    def test_english_words_are_not_split(self) -> None:
        assert wrap_lines("hello big world", 100, measure) == ["hello big", "world"]

    def test_a_word_longer_than_the_width_is_split(self) -> None:
        # 1 語が幅より長いときだけ語の途中で切る 切らないと画面からはみ出す
        assert wrap_lines("abcdefghij xy", 40, measure) == ["abcd", "efgh", "ij", "xy"]

    def test_japanese_and_english_mixed(self) -> None:
        # 和文の中の英単語は和文との間で切れるが、単語の中では切れない
        assert wrap_lines("これはPythonです", 60, measure) == ["これは", "Python", "です"]

    def test_spaces_at_the_break_are_dropped(self) -> None:
        # 行の端に空白が残ると、中央揃えで字が半角ずれる
        assert wrap_lines("ab cd ef", 50, measure) == ["ab cd", "ef"]

    def test_a_very_narrow_width_keeps_every_character(self) -> None:
        # 1 字も収まらない幅でも字は捨てない
        assert "".join(wrap_lines("あいう", 5, measure)) == "あいう"
        assert len(wrap_lines("あいう", 5, measure)) == 3

    def test_an_empty_line_is_kept(self) -> None:
        assert wrap_lines("あ\n\nい", 30, measure) == ["あ", "", "い"]


class TestWithCharacterWrap:
    """字幕の整形（文字数で改行を本文に書き込む）と、描くときの幅の折り返しを両方使ったとき"""

    def test_lines_shorter_than_the_width_stay_as_the_cleanup_made_them(self) -> None:
        # 整形が入れた改行を尊重する 幅の折り返しが先の改行を消したり詰め直したりすると、
        # 整形の設定で決めた行の分け方が効かなくなる
        cleaned = wrap_text("今日はいい天気ですね。明日も晴れるといいですね", 12)
        assert "\n" in cleaned
        assert wrap_lines(cleaned, 200, measure) == cleaned.split("\n")

    def test_a_cleaned_line_longer_than_the_width_is_wrapped_again(self) -> None:
        # 文字数では収まっていても、画面の幅（大きな字）には収まらない行はさらに折り返す
        cleaned = wrap_text("今日はいい天気ですね。明日も晴れるといいですね", 12)
        wrapped = wrap_lines(cleaned, 60, measure)
        assert all(measure(line) <= 60 for line in wrapped)
        assert "".join(wrapped) == cleaned.replace("\n", "")


def by_grapheme(text: str) -> float:
    """見た目の 1 字を 10 で測る 結合文字や絵文字のつなぎを幅に数えない（字の実寸と同じ向き）"""
    return 10.0 * len(graphemes(text))


#: 1 字として見えるのに、コードポイントでは 2 つ以上になる字
ACCENTED = "e" + chr(0x0301)  # é（e と結合アクセント）
FAMILY = chr(0x1F468) + chr(0x200D) + chr(0x1F469) + chr(0x200D) + chr(0x1F467)  # ZWJ の家族
THUMB = chr(0x1F44D) + chr(0x1F3FD)  # 肌の色を付けた絵文字
JAPAN, FRANCE = chr(0x1F1EF) + chr(0x1F1F5), chr(0x1F1EB) + chr(0x1F1F7)  # 国旗（地域指示子の組）
KUZU = "葛" + chr(0xE0100)  # 異体字セレクタの付いた字
VOICED = "か" + chr(0x3099)  # 結合の濁点


def unjoined(text: str) -> float:
    """つなぎの字形を持たない書体の物差し 結合文字・ZWJ・異体字セレクタは幅 0 で、
    ほかのコードポイントは 1 つ 10 ZWJ の家族は 3 人並んで 30、国旗は 2 字の記号で 20、
    肌の色は色の四角が並んで 20 になる（書体が組を 1 字に描けないときの実際の出方）
    幅を超える 1 字が出るので、1 字の中で切るかどうかが分かる
    """
    zero = {0x200D} | set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))
    return 10.0 * sum(1 for c in text if not unicodedata.combining(c) and ord(c) not in zero)


class TestGraphemes:
    """見た目の 1 字の中では折り返さない（CodeRabbit の指摘 #266）

    前はコードポイントで切っていて、幅を超えた所で基の字と結合文字・ZWJ の前後・国旗の
    2 字・絵文字と肌の色が別の行に分かれた
    """

    def test_a_long_accented_word_is_split_between_letters(self) -> None:
        # 幅を超える 1 語はなお切る ただし é の e とアクセントは離さない
        lines = wrap_lines(ACCENTED * 7, 30, by_grapheme)
        assert lines == [ACCENTED * 3, ACCENTED * 3, ACCENTED]

    def test_a_zwj_sequence_stays_whole(self) -> None:
        # 1 字で幅を超えても割らない 割ると家族の 1 人ずつが別の行に出る
        assert wrap_lines(FAMILY * 2, 20, unjoined) == [FAMILY, FAMILY]

    def test_a_skin_tone_stays_with_its_emoji(self) -> None:
        assert wrap_lines(THUMB * 2, 10, unjoined) == [THUMB, THUMB]

    def test_a_flag_is_not_halved(self) -> None:
        assert wrap_lines(JAPAN + FRANCE, 10, unjoined) == [JAPAN, FRANCE]

    def test_a_variation_selector_stays_with_its_kanji(self) -> None:
        assert wrap_lines(KUZU * 3, 20, by_grapheme) == [KUZU * 2, KUZU]

    def test_a_long_plain_word_is_still_split(self) -> None:
        # 1 語が幅より長いときに切る約束は変えない
        assert wrap_lines("abcdefgh", 30, by_grapheme) == ["abc", "def", "gh"]


class TestRevealByGraphemes:
    """文字送りも見た目の 1 字で数える 途中で止まったときに基の字だけ・国旗の片方が出ない"""

    def test_a_voiced_kana_appears_whole(self) -> None:
        # 前はコードポイントの 3 割（3 つのうち 1 つ）で、濁点の無い「か」が出た
        assert _revealed(VOICED + "き", {"reveal": 34.0}) == VOICED

    def test_a_flag_appears_whole(self) -> None:
        assert _revealed(JAPAN + FRANCE, {"reveal": 30.0}) == JAPAN

    def test_the_aviutl_layout_counts_the_same_way(self) -> None:
        lines = _revealed_lines(parse_tags(VOICED + "き", 64.0), {"reveal": 34.0})
        assert "".join(run.text for line in lines for run in line.runs) == VOICED
