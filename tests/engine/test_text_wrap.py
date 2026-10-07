"""決めた幅での折り返しの位置（#249）

幅は 1 字 10 の決まった物差しで測る 書体に頼らず、折り返す位置の決まり（禁則・英単語・
長すぎる語・手で入れた改行）だけを見る 字の実寸で測るのは描く所の試験（test_text_wrap_render）
"""

from __future__ import annotations

from sashimono.asr.cleanup import wrap_text
from sashimono.engine.text_wrap import wrap_lines


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
