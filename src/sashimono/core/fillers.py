"""言いよどみ（えーと・あのー）の音を探す（#414）

通しの確かめで、無音は詰めても「えーと」が残った 字幕の文言からは整形
（:mod:`sashimono.asr.cleanup`）が落とすが、音は残る 語の時刻（#420）から言いよどみの語を探し、
素材の秒の範囲で返す 切るのは無音カットと同じ命令（呼ぶ側）

限り: Whisper は言いよどみを書かないことが多い（文を読みやすく直して返す） 書かなかった言いよどみは
語が無いので見つからない 見つかるのは認識が書いた物だけ

Qt も音も知らない 字幕と語の時刻だけを読む
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from sashimono.core.model import Transcript, Word
from sashimono.core.words import PHRASE_GAP

__all__ = [
    "AUDIO_FILLERS",
    "FillerSpan",
    "filler_key",
    "filler_spans",
]

#: 音から切る既定の言いよどみ 字幕の整形の既定（``asr.cleanup.DEFAULT_FILLERS``）から、驚きの
#: 「えっ」を外した物 字幕の文字から落とすのと違い、音から切ると反応の声まで消える
#: 「まあ」「なんか」のように意味を持ちうる語は入れない（呼ぶ側が足せる）
AUDIO_FILLERS: tuple[str, ...] = (
    "えーと",
    "えっと",
    "ええと",
    "えーっと",
    "あのー",
    "あの一",
    "そのー",
    "えー",
    "あー",
    "うー",
    "うーん",
    "んー",
    "uh",
    "um",
    "erm",
)

#: 束ねて読む語の数の上限 Whisper の日本語の語は 1 字ほどなので「えーっと」で 4 つになる
_MAX_WORDS = 6
#: これより短い言いよどみは返さない（秒） 語の時刻の誤り（0.06 秒ほど）と見分けが付かない
_MIN_SECONDS = Fraction(1, 10)
#: 比べるときに外す字 句読点と空白は語に付いたり付かなかったりする
_MARKS = frozenset(" 　、\u3002，．！？!?,.…・「」『』（）()")
#: 伸ばす音として書かれる小さい母音 「えぇと」「あぁ」は「えーと」「あー」と同じ音
_SMALL_VOWELS = frozenset("ぁぃぅぇぉ")


@dataclass(frozen=True, slots=True)
class FillerSpan:
    """見つかった言いよどみ 1 つ 時刻は素材の秒"""

    start: Fraction
    end: Fraction
    text: str


def filler_key(text: str) -> str:
    """比べる形 互換の字をそろえ、カタカナをひらがなに、伸ばす記号の続きを 1 つにする

    認識は同じ「えーと」を「エート」「えーーと」「えぇと」とも書く
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    for letter in folded:
        if letter in _MARKS:
            continue
        if letter in "〜~":
            letter = "ー"
        code = ord(letter)
        if 0x30A1 <= code <= 0x30F6:
            letter = chr(code - 0x60)
        if letter in _SMALL_VOWELS and out:
            letter = "ー"
        if letter == "ー" and out and out[-1] == "ー":
            continue
        out.append(letter)
    return "".join(out)


def filler_spans(
    transcript: Transcript | None, fillers: Iterable[str] = AUDIO_FILLERS
) -> list[FillerSpan]:
    """字幕の語から言いよどみを探す 語の時刻の無い行は見ない 並びは素材の時刻の順

    1 つの行の中で、続く語を束ねた文字が言いよどみと同じ所を探す 長い方を先に取る（「えー」を
    先に取ると「えーと」の「と」が残る） 束ねは語の切れ目でだけ始めて終える 語の途中で切ると、
    隣の語の音まで削る 間の空いた語は束ねない（語の区切りと同じ :data:`PHRASE_GAP`） 束ねると、
    離れた「え」と「ーと」の間の別の声まで切る
    """
    if transcript is None:
        return []
    wanted = {key for key in (filler_key(text) for text in fillers) if key}
    if not wanted:
        return []
    longest = max(len(key) for key in wanted)
    found: list[FillerSpan] = []
    for segment in transcript.segments:
        found.extend(_in_line(segment.words, wanted, longest))
    found.sort(key=lambda span: (span.start, span.end))
    return found


def _in_line(words: Sequence[Word], wanted: set[str], longest: int) -> Iterable[FillerSpan]:
    keys = [filler_key(word.text) for word in words]
    index = 0
    while index < len(words):
        joined = ""
        last: int | None = None
        for stop in range(index, min(len(words), index + _MAX_WORDS)):
            if stop > index and words[stop].start - words[stop - 1].end > PHRASE_GAP:
                break
            # 束ねた後もそろえる 「え」「ー」「ー」「と」と分かれると、伸ばす記号が続いて残る
            joined = filler_key(joined + keys[stop])
            if len(joined) > longest:
                break
            if joined in wanted:
                last = stop
        if last is None or not keys[index]:
            index += 1
            continue
        start, end = words[index].start, words[last].end
        if end - start >= _MIN_SECONDS:
            text = "".join(word.text for word in words[index : last + 1]).strip()
            yield FillerSpan(start, end, text)
        index = last + 1
