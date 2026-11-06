"""語の時刻を読む 切る所と字幕の文言を、語の単位で確かめる（#420）

AI は音を聞けない 15 秒のハイライトの 16 点の入りと出のうち 3 点が言葉の途中で、音量の測りで
見つかったのは 1 点だけだった 語の時刻はここで素材の秒のまま読み、タイムラインのフレームへは
:func:`~sashimono.core.projection.heard_pieces` で行き来する

語の時刻は認識の見積もりで、前後に 0.1 秒ほどずれる 音の波形と合わせて補うのは
:mod:`sashimono.engine.audio.word_cuts` の役目で、ここは波形を知らない

Whisper の語は日本語では語にならない 実物の素材で調べると、語の時刻の 1 単位は認識の 1 字
（「編|集|機|能」「カ|メ|ラ」）で、中央値が 0.17 秒しかなく、音の切れ目とも合わない これを
そのまま語の境目と読むと、語の途中を境目として勧める そこで :func:`phrases` が、字の種類の
変わり目と音の間から、切ってよい単位（文節に近い）に束ねる 形態素の辞書は入れていない
（配る大きさに比べて、切る位置を選ぶのに要る精度は字の種類で足りる）
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import accumulate

from sashimono.core.model import MediaId, Project, Transcript, TranscriptSegment, Word
from sashimono.core.projection import HeardPiece, heard_pieces

__all__ = [
    "CUT_IN",
    "CUT_OUT",
    "WORD_MARGIN",
    "CutEdge",
    "HeardWord",
    "WordMap",
    "attach_words",
    "cut_edges",
    "heard_between",
    "heard_words",
    "phrases",
    "split_seam",
    "transcript_words",
    "unheard_words",
    "untimed_sounds",
    "wordless_lines",
]

#: クリップの音の頭と終わり どちらの端かで、音を残す側が変わる
CUT_IN = "in"
CUT_OUT = "out"

#: 語の途中と読む、語の端からの距離（秒） 実物の素材で、語の境目と音の谷の差は中央値 0.06 秒・
#: 9 割が 0.09 秒以内だった 端ちょうどを途中と読むと、語の境目で切った所まで問題になる
WORD_MARGIN = Fraction(8, 100)
#: 語の間がこれより空いていれば、字の種類によらず区切る 息継ぎより短く、語の中の促音（「っ」の
#: 詰まり）より長い
_PHRASE_GAP = Fraction(12, 100)
#: 後ろの語に付く頭の 1 字 ここで区切ると「お|茶」の間を語の境目として勧める
_PREFIXES = frozenset("おご御")
#: 文の終わりの形 ひらがなが続くと字の種類では区切れない所
_SENTENCE_ENDS = ("です", "ます", "ました", "ません", "でした", "ください")
#: 文の終わりの形の後に付く助詞 ここでは区切らない（「ですか」「ますね」「ですが」）
_ENDING_PARTICLES = "かねよなわぞが"
#: 語の無い所と読む、語からの距離（秒） 語の時刻のずれ（9 割が 0.09 秒以内）の 3 倍ほど
#: 短いと語の時刻のずれだけの所を語の無い所と読み、語の末尾の声を未検査にする
_GAP_REACH = Fraction(3, 10)
#: 等速の音のつなぎ目で、素材の時刻の端数と読む差（秒） 取り込みの時刻の刻み（ミリ秒）
_SOURCE_ROUNDING = Fraction(1, 1000)
_HIRAGANA, _KATAKANA, _KANJI, _LATIN, _MARK = range(5)


def transcript_words(transcript: Transcript | None) -> tuple[Word, ...]:
    """字幕の語を素材の時刻の順に並べる 語の時刻が無い字幕なら空"""
    if transcript is None:
        return ()
    return tuple(
        sorted(
            (word for segment in transcript.segments for word in segment.words),
            key=lambda word: (word.start, word.end),
        )
    )


def _letter_kind(letter: str) -> int:
    if "\u3040" <= letter <= "\u309f":
        return _HIRAGANA
    if "\u30a0" <= letter <= "\u30ff" or "\uff66" <= letter <= "\uff9f":
        return _KATAKANA
    if "\u4e00" <= letter <= "\u9fff" or "\u3400" <= letter <= "\u4dbf" or letter == "\u3005":
        return _KANJI
    return _LATIN if letter.isalnum() else _MARK


def _breaks(so_far: str, word: Word, following: Word, later: Word | None) -> bool:
    """語 ``word`` と次の語の間で区切るか ``so_far`` は今の束の文字 ``later`` は次の次の語"""
    if following.start - word.end >= _PHRASE_GAP:
        return True
    if not word.text or not following.text:
        return False
    before, after = _letter_kind(word.text[-1]), _letter_kind(following.text[0])
    if before == _MARK:
        return True
    if word.text in _PREFIXES:
        return False
    if (
        following.text in _PREFIXES
        and later is not None
        and later.text
        and _letter_kind(later.text[0]) == _KANJI
    ):
        return True
    # 助詞の後ろも文の切れ目 「ですか|はい」を束ねると、その間で切った所まで語の途中と読む
    if (
        so_far.rstrip(_ENDING_PARTICLES).endswith(_SENTENCE_ENDS)
        and following.text[0] not in _ENDING_PARTICLES
    ):
        return True
    # ひらがなの後に漢字・カタカナ・英数が来る所が文節の頭になることが多い 送り仮名と助詞は
    # 前の語の後ろに付くので、逆の向きでは区切らない
    return before == _HIRAGANA and after in (_KATAKANA, _KANJI, _LATIN)


def _segment_phrases(words: Sequence[Word]) -> Iterator[Word]:
    text, start = "", words[0].start
    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else None
        later = words[index + 2] if index + 2 < len(words) else None
        text += word.text
        if following is None or _breaks(text, word, following, later):
            yield Word(start=start, end=max(start, word.end), text=text)
            text = ""
            if following is not None:
                start = following.start


def phrases(transcript: Transcript | None) -> tuple[Word, ...]:
    """字幕の語を、切ってよい単位に束ねて並べる 語の時刻が無ければ空

    字幕の行の境目はいつも区切る Whisper は文の切れ目で行を分けるので、語を束ねる手掛かりより
    確か 束ねるのは日本語だけ（字幕の ``language`` が ``ja`` か空） ほかの言語の Whisper の語は
    そのまま語になっていて、束ねると文が 1 語になる
    """
    if transcript is None:
        return ()
    if transcript.language and not transcript.language.startswith("ja"):
        return transcript_words(transcript)
    grouped = [
        phrase
        for segment in transcript.segments
        if segment.words
        for phrase in _segment_phrases(segment.words)
    ]
    return tuple(sorted(grouped, key=lambda word: (word.start, word.end)))


def _reaching(
    starts: Sequence[Fraction], reach: Sequence[Fraction], low: Fraction, high: Fraction
) -> range:
    """頭の順に並んだ区間のうち、``[low, high]`` に届きうる物の番号を後ろから

    区間は重なることがある すぐ前の区間の終わりで見切ると、先に始まる長い区間が陰に隠れるので、
    そこまでの一番遅い終わり（``reach``）で見切る
    """
    last = bisect.bisect_right(starts, high) - 1
    first = bisect.bisect_right(reach, low)
    return range(last, first - 1, -1)


def attach_words(transcript: Transcript, recognized: Transcript) -> Transcript:
    """今の字幕の文言を変えずに、起こし直した語の時刻だけを写す（#420）

    語の時刻が無い字幕を持つ作品は、語の時刻のためだけに起こし直すと人が直した文言と話者が消える
    語は重なりが一番長い字幕の行へ入れる どの行にも重ならない語は捨てる 行を消した所の語を
    残すと、消した言葉が字幕の語として返る
    """
    # 認識が語を 1 つも返さないときは写さない 全部の行の語を空にすると、持っていた語の時刻が消える
    if not transcript_words(recognized):
        return transcript
    groups: list[list[Word]] = [[] for _ in transcript.segments]
    starts = [segment.start for segment in transcript.segments]
    reach = list(accumulate((segment.end for segment in transcript.segments), max))
    for word in transcript_words(recognized):
        best: int | None = None
        best_overlap = Fraction(0)
        # 行は重なることがある 語の頭の直前の行から数えると、先に始まる長い行を候補から落とす
        # 同じだけ重なるときは前の行へ入れるので、頭の順に見る
        for index in reversed(_reaching(starts, reach, word.start, word.end)):
            segment = transcript.segments[index]
            overlap = min(segment.end, word.end) - max(segment.start, word.start)
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        if best is not None:
            groups[best].append(word)
    return replace(
        transcript,
        segments=tuple(
            replace(segment, words=tuple(group))
            for segment, group in zip(transcript.segments, groups, strict=True)
        ),
    )


@dataclass(frozen=True, slots=True)
class HeardWord:
    """タイムラインで鳴る語 1 つ"""

    word: Word
    piece: HeardPiece
    #: 語の真ん中のタイムラインのフレーム
    frame: Fraction


def heard_words(project: Project, start: int = 0, end: int | None = None) -> list[HeardWord]:
    """タイムラインの ``[start, end)`` で鳴る語を、鳴る順に返す

    語の真ん中が音の範囲に入る物を鳴る語と数える 語の時刻は前後へずれるので、端が少し掛かった
    だけの語まで数えると、切り落とした語が残っていると読める
    """
    found: list[HeardWord] = []
    known: dict[tuple[MediaId, int], tuple[tuple[Word, ...], list[Fraction]]] = {}
    for piece in heard_pieces(project):
        if piece.end <= start or (end is not None and piece.start >= end):
            continue
        key = (piece.media_id, piece.stream)
        if key not in known:
            media = project.find_media(piece.media_id)
            words = transcript_words(None if media is None else media.transcript_for(piece.stream))
            middles = sorted((word.start + word.end) / 2 for word in words)
            ordered = tuple(sorted(words, key=lambda word: word.start + word.end))
            known[key] = (ordered, middles)
        ordered, middles = known[key]
        # 細かく切った作品はクリップが何千本にもなる 語を毎回頭から見ると、調べ直すたびに待たされる
        low = bisect.bisect_left(middles, piece.source_start)
        high = bisect.bisect_left(middles, piece.source_end)
        for word, middle in zip(ordered[low:high], middles[low:high], strict=True):
            frame = piece.frame_at(middle, project.rate)
            if frame >= start and (end is None or frame < end):
                found.append(HeardWord(word, piece, frame))
    found.sort(key=lambda item: item.frame)
    return found


@dataclass(frozen=True, slots=True)
class CutEdge:
    """音が切れる所 1 つ"""

    piece: HeardPiece
    #: :data:`CUT_IN` か :data:`CUT_OUT`
    side: str
    #: 切れる所の素材の秒
    seconds: Fraction
    #: タイムラインのフレーム
    frame: Fraction


#: 音の並びの引き方 同じ素材の同じ音の、タイムラインのコマの番号から引く
type _Index = dict[tuple[MediaId, int, int], list[HeardPiece]]


def _index(pieces: Iterable[HeardPiece], edge: Callable[[HeardPiece], Fraction]) -> _Index:
    found: _Index = {}
    for piece in pieces:
        found.setdefault((piece.media_id, piece.stream, round(edge(piece))), []).append(piece)
    return found


def _neighbours(index: _Index, piece: HeardPiece, frame: Fraction) -> Iterator[HeardPiece]:
    """``frame`` に端がある、同じ素材の同じ音の並び"""
    for other in index.get((piece.media_id, piece.stream, round(frame)), ()):
        if other is not piece:
            yield other


def _rounding(piece: HeardPiece, other: HeardPiece, frame_seconds: Fraction) -> Fraction:
    """つなぎ目の素材の時刻の差のうち、丸めと読む大きさ

    等速の分割は分数のまま割るので差が出ない 許すのはミリ秒へ丸めた取り込みの端数だけ 1 コマも
    許すと、素材を 1 コマ飛ばしたつなぎ目を分割と読み、語が欠けても返さない 速度を変えると長さを
    コマへ丸めるので、速度を掛けた音だけ 1 コマ分の素材の時刻を許す
    """
    speed = max(piece.speed, other.speed)
    return _SOURCE_ROUNDING if speed == 1 and piece.speed == other.speed else frame_seconds * speed


def split_seam(earlier: HeardPiece, later: HeardPiece, frame_seconds: Fraction) -> bool:
    """``earlier`` の終わりに ``later`` が分割しただけでつながるか 音が途切れない所

    同じ素材の同じ音で、タイムラインの端がちょうど合い（1 コマ空いた所は無音が出る）、素材の
    時刻が丸めの差の内で続く所 :func:`cut_edges` と字幕の文言の確かめが同じ答えを使う
    """
    return (
        earlier.media_id == later.media_id
        and earlier.stream == later.stream
        and earlier.end == later.start
        and abs(later.source_start - earlier.source_end) <= _rounding(later, earlier, frame_seconds)
    )


def cut_edges(project: Project, start: int, end: int) -> list[CutEdge]:
    """タイムラインの ``[start, end]`` の中で、素材の音が途切れる端を返す

    分割しただけのつなぎ目（前の終わりと後ろの頭が素材の続き）は音が切れないので数えない
    素材の頭と終わりも切った所ではないので数えない 語の途中で始まる素材は録音の都合で、
    編集では直せない
    """
    pieces = heard_pieces(project)
    frame_seconds = project.rate.frame_duration
    heads = _index(pieces, lambda piece: piece.start)
    tails = _index(pieces, lambda piece: piece.end)
    edges: list[CutEdge] = []
    for piece in pieces:
        media = project.find_media(piece.media_id)
        if media is None:
            continue
        continued_from = any(
            split_seam(other, piece, frame_seconds)
            for other in _neighbours(tails, piece, piece.start)
        )
        continues = any(
            split_seam(piece, other, frame_seconds)
            for other in _neighbours(heads, piece, piece.end)
        )
        ends_media = media.duration > 0 and piece.source_end >= media.duration - frame_seconds
        if start <= piece.start < end and piece.source_start > 0 and not continued_from:
            edges.append(CutEdge(piece, CUT_IN, piece.source_start, piece.start))
        if start < piece.end <= end and not ends_media and not continues:
            edges.append(CutEdge(piece, CUT_OUT, piece.source_end, piece.end))
    return edges


def wordless_lines(transcript: Transcript | None) -> tuple[TranscriptSegment, ...]:
    """語の時刻が無い字幕の行 手で足した行や、語の時刻が入る前の行"""
    if transcript is None:
        return ()
    return tuple(segment for segment in transcript.segments if not segment.words)


def _near(words: Iterable[Word], seconds: Fraction) -> bool:
    return any(word.start - _GAP_REACH < seconds < word.end + _GAP_REACH for word in words)


class WordMap:
    """1 つの音の語の読み取り 語の途中か・語で確かめられる所かを、道具と検査が同じ答えで読む

    別々に判定すると、語の時刻の無い行と、語が行の一部にしか付かない行の扱いが所ごとに食い違う
    """

    def __init__(self, transcript: Transcript | None) -> None:
        self.phrases = phrases(transcript)
        self.lines = () if transcript is None else transcript.segments
        self._line_starts = [line.start for line in self.lines]
        self._line_reach = list(accumulate((line.end for line in self.lines), max))
        self._word_starts = [word.start for word in self.phrases]
        self._word_reach = list(accumulate((word.end for word in self.phrases), max))

    @property
    def units(self) -> tuple[Word, ...]:
        """語と、語の時刻の無い行を時刻の順に並べた物 切り所を探す単位

        語の時刻の無い行も行の切れ目と波形で切り所は読める 外すと起こし直すまで AI が手探りで切る
        """
        bare = (Word(line.start, line.end, line.text) for line in self.lines if not line.words)
        return tuple(sorted((*self.phrases, *bare), key=lambda word: (word.start, word.end)))

    def word_at(self, seconds: Fraction) -> Word | None:
        """``seconds`` を中に含む語 語の頭と終わりから :data:`WORD_MARGIN` より内側に入る物だけ

        語の時刻は見積もりなので、端ちょうどは語の途中と読まない
        """
        for at in _reaching(self._word_starts, self._word_reach, seconds, seconds):
            word = self.phrases[at]
            if word.start + WORD_MARGIN < seconds < word.end - WORD_MARGIN:
                return word
        return None

    def unworded_at(self, seconds: Fraction) -> bool:
        """語の時刻から語の途中かを言えない所か 語から :data:`_GAP_REACH` より離れた所

        行の中は行の語で見る 語の時刻の無い行と、語が行の一部にしか付かない行の語の無い所がこれに
        当たる 行の外は作品の語で見る 字幕を消した所や認識が拾わなかった所でも話していることがある
        話しているかは語の時刻から分からないので、波形と合わせて読む
        （:func:`~sashimono.engine.audio.word_cuts.read_cut`）
        """
        here = _reaching(self._line_starts, self._line_reach, seconds, seconds)
        groups = [self.lines[at].words for at in here if seconds < self.lines[at].end]
        return not all(_near(words, seconds) for words in groups or [self._nearby(seconds)])

    def _nearby(self, seconds: Fraction) -> tuple[Word, ...]:
        low, high = seconds - _GAP_REACH, seconds + _GAP_REACH
        reached = _reaching(self._word_starts, self._word_reach, low, high)
        return tuple(self.phrases[at] for at in reached)


def untimed_sounds(project: Project, start: int, end: int) -> set[tuple[MediaId, int]]:
    """タイムラインの ``[start, end)`` で鳴る音のうち、語の時刻が揃っていない音

    字幕の一部の行だけ語の時刻を持つ音も入れる 語の無い行の所は切れ目も文言も比べられないので、
    入れないと、調べていない所を無事と読む 見るのは鳴っている行だけ 使っていない所の行まで見ると、
    置いていない行 1 本で作品全体が調べられない扱いになる
    """
    found: set[tuple[MediaId, int]] = set()
    lines: dict[tuple[MediaId, int], tuple[TranscriptSegment, ...] | None] = {}
    for piece in heard_pieces(project):
        if piece.end <= start or piece.start >= end:
            continue
        key = (piece.media_id, piece.stream)
        if key not in lines:
            media = project.find_media(piece.media_id)
            transcript = None if media is None else media.transcript_for(piece.stream)
            # 細かく切った作品で行を毎回なめないよう、語の無い行だけを控える ふつうは空
            lines[key] = wordless_lines(transcript) if phrases(transcript) else None
        wordless = lines[key]
        low = piece.seconds_at(max(piece.start, Fraction(start)), project.rate)
        high = piece.seconds_at(min(piece.end, Fraction(end)), project.rate)
        if wordless is None or any(line.overlaps(low, high) for line in wordless):
            found.add(key)
    return found


def heard_between(
    heard: Sequence[HeardWord], media_id: MediaId, stream: int, start: int, end: int
) -> list[HeardWord]:
    """:func:`heard_words` の戻り値（鳴る順）から、1 つの音の ``[start, end)`` の語を取り出す"""
    low = bisect.bisect_left(heard, Fraction(start), key=lambda item: item.frame)
    high = bisect.bisect_left(heard, Fraction(end), key=lambda item: item.frame)
    return [
        item
        for item in heard[low:high]
        if item.piece.media_id == media_id and item.piece.stream == stream
    ]


def unheard_words(
    segment: TranscriptSegment,
    heard: Sequence[HeardWord],
    media_id: MediaId,
    stream: int,
    start: int,
    end: int,
) -> list[Word]:
    """字幕の行の語のうち、タイムラインの ``[start, end)`` で鳴らない語

    ``heard`` は :func:`heard_words` の戻り値（鳴る順） 字幕の行の数だけ作り直さず使い回す
    """
    kept = {item.word for item in heard_between(heard, media_id, stream, start, end)}
    return [word for word in segment.words if word not in kept]
