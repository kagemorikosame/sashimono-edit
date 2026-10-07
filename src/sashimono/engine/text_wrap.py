"""テキストを決めた幅で折り返す（#249）

折り返す位置は、日本語の禁則と英単語の切れ目で決める 幅は描く所が字の実寸で測る
（``measure``） ここは字を数えるだけで Qt に頼らないので、書体が無くても位置の決まりを
試験で確かめられる

手で入れた改行はそのまま残し、その間の 1 行ずつをさらに幅で折り返す 字幕の整形
（:func:`sashimono.asr.cleanup.wrap_text`）が文字数で入れた改行もここでは手の改行と同じ
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

__all__ = ["NO_LINE_END", "NO_LINE_START", "wrap_lines"]

#: 行頭に置かない字（行頭禁則） 句読点・閉じ括弧・小さい仮名・長音・繰り返し記号
#: 前の字と一緒に前の行へ残す 句点はこの文字列の中だけ番号で書く（文章の句点を見張る
#: tools/punctuation.py が、ここの句点をデータと見分けられない）
NO_LINE_START = frozenset(
    "、\u3002，．,.!?！？:;：；)]}）］｝〕〉》」』】〙〗〟’”»ー―‐-–—…‥・゠"
    "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
    "々〻ゝゞヽヾ%％℃°′″"
)

#: 行末に置かない字（行末禁則） 開き括弧 次の字と一緒に次の行へ送る
NO_LINE_END = frozenset("([{（［｛〔〈《「『【〘〖〝‘“«")


def wrap_lines(text: str, width: float, measure: Callable[[str], float]) -> list[str]:
    """``text`` を、``measure`` で測った幅が ``width`` を超えないように行へ分ける

    ``width`` が 0 以下なら折り返さない（改行で分けるだけ） 手で入れた改行は残す
    折り返した所の空白は落とす（行の端に空白が残ると、揃え方で字が半角ずれる）
    1 語や禁則でつないだ塊が幅より長いときだけ、その塊を字の途中で切る
    1 行に 1 字も収まらない狭さでも、1 字ずつは置く（字を捨てない）
    """
    paragraphs = text.split("\n")
    if width <= 0:
        return paragraphs
    lines: list[str] = []
    for paragraph in paragraphs:
        lines.extend(_wrap_paragraph(paragraph, width, measure))
    return lines


def _wrap_paragraph(paragraph: str, width: float, measure: Callable[[str], float]) -> list[str]:
    if not paragraph or measure(paragraph) <= width:
        return [paragraph]
    lines: list[str] = []
    line = ""
    for chunk in _chunks(paragraph):
        candidate = line + chunk
        if not line or measure(candidate.rstrip()) <= width:
            line = candidate
        else:
            lines.append(line.rstrip())
            line = chunk.lstrip()
        # 1 つの塊だけで幅を超えるなら、塊の中を字で切る 切った残りは次の塊とつなげる
        while line and measure(line.rstrip()) > width:
            head, line = _split_long(line, width, measure)
            lines.append(head)
    if line.strip() or not lines:
        lines.append(line.rstrip())
    return lines


def _split_long(line: str, width: float, measure: Callable[[str], float]) -> tuple[str, str]:
    """幅を超える 1 塊を、収まる所までの頭と残りに分ける 頭は少なくとも 1 字"""
    cut = 1
    while cut < len(line) and measure(line[: cut + 1]) <= width:
        cut += 1
    if cut >= len(line):
        return line.rstrip(), ""
    # 切った所でも禁則は守る 収まる所まで詰めた結果が禁則に当たるなら、前へ寄せる
    # 寄せられない（頭の 1 字しか無い）ときは、字を捨てないことを優先して切る
    while cut > 1 and (line[cut] in NO_LINE_START or line[cut - 1] in NO_LINE_END):
        cut -= 1
    return line[:cut], line[cut:].lstrip()


def _chunks(paragraph: str) -> list[str]:
    """間で折り返してよい塊に分ける 折り返せるのは塊と塊の間だけ

    - 空白の後ろで切れる（空白は前の塊の終わりに付ける）
    - 英数字の続き（英単語・数字）の中では切らない
    - 行頭禁則の字の前と、行末禁則の字の後ろでは切らない
    - それ以外の字（漢字・仮名など）の間ではどこでも切れる
    """
    chunks: list[str] = []
    current = ""
    previous = ""
    for character in paragraph:
        if current and _breakable(previous, character):
            chunks.append(current)
            current = ""
        current += character
        previous = character
    if current:
        chunks.append(current)
    return chunks


def _breakable(before: str, after: str) -> bool:
    """``before`` と ``after`` の間で折り返せるか"""
    if after.isspace():
        # 空白の前では切らない 空白は前の行の終わりに付けて落とす
        return False
    if before.isspace():
        return True
    if after in NO_LINE_START or before in NO_LINE_END:
        return False
    if unicodedata.combining(after) or after in "‍️":
        # 濁点などの結合文字と絵文字のつなぎは、前の字と 1 字として扱う
        return False
    # 英単語は途中で切らない 和文の中の英単語も前後の和文とは切れる
    return not (_is_word(before) and _is_word(after))


def _is_word(character: str) -> bool:
    """英単語の一部になる字（ラテン文字・数字と、語の中に入るアポストロフィなど）"""
    if character in "'’_-":
        return True
    if not character.isalnum():
        return False
    # 漢字や仮名も isalnum が真になるので、ラテン文字などのアルファベットの字に絞る
    name = unicodedata.name(character, "")
    return character.isascii() or name.startswith(("LATIN", "GREEK", "CYRILLIC", "FULLWIDTH"))
