"""字幕起こしの結果

時刻はすべて**素材内のソース秒**で持つ タイムライン上の位置は持たない これが
この設計の要で、クリップをどう切っても並べ替えても、字幕は投影で位置が決まるため
同期処理そのものが不要になる（:mod:`sashimono.core.projection` を参照）
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate

from sashimono.core.model.ids import SegmentId, new_segment_id

__all__ = ["Transcript", "TranscriptSegment", "Word"]


@dataclass(frozen=True, slots=True)
class Word:
    """語の単位のタイムスタンプ

    クリップ分割の境界の決定と、AI が語の境目で切るのに使う（#420）
    Whisper の語は日本語では 1 字ほどの単位で、語の境目としては
    :func:`sashimono.core.words.phrases` が束ねた物を読む
    """

    start: Fraction
    end: Fraction
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """1 つの発話区間 字幕 1 枚に対応する"""

    start: Fraction
    end: Fraction
    text: str
    words: tuple[Word, ...] = ()
    speaker: str | None = None
    #: 人手または AI で編集済みか 再起こしの際に上書きしてよいかの判断に使う
    edited: bool = False
    id: SegmentId = field(default_factory=new_segment_id)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"終了が開始より前: {self.start} .. {self.end}")

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    def overlaps(self, start: Fraction, end: Fraction) -> bool:
        """``[start, end)`` と重なるか 接するだけの場合は重ならない扱い"""
        return self.start < end and start < self.end

    def with_text(self, text: str) -> TranscriptSegment:
        """本文を差し替え、編集済みとして印を付けた新しいセグメントを返す"""
        return TranscriptSegment(
            start=self.start,
            end=self.end,
            text=text,
            words=self.words,
            speaker=self.speaker,
            edited=True,
            id=self.id,
        )


@dataclass(frozen=True, slots=True)
class Transcript:
    """1 つの素材に対する起こし結果"""

    segments: tuple[TranscriptSegment, ...] = ()
    language: str = ""
    #: 生成に使ったモデル名 結果の再現性と、再起こしの要否判断のために残す
    model: str = ""
    #: :meth:`overlapping` の索引 各セグメントの頭と、そこまでの終わりの最大
    #: 作るときに 1 度だけ求める 選択のたびに字幕パネルがクリップごとに引くので、毎回
    #: 頭からなめると 1 時間の録画の字幕で 1 回の選択が 100 ms を超えた（#455 の実プロジェクト）
    _starts: tuple[Fraction, ...] = field(init=False, repr=False, compare=False)
    _reach: tuple[Fraction, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        starts = tuple(s.start for s in self.segments)
        if list(starts) != sorted(starts):
            raise ValueError("セグメントは開始時刻の昇順でなければならない")
        # 凍った型なので object.__setattr__ で入れる 比べ・表示・保存には出ない
        object.__setattr__(self, "_starts", starts)
        object.__setattr__(self, "_reach", tuple(accumulate((s.end for s in self.segments), max)))

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[TranscriptSegment]:
        return iter(self.segments)

    def overlapping(self, start: Fraction, end: Fraction) -> Iterator[TranscriptSegment]:
        """ソース時刻 ``[start, end)`` に重なるセグメントを順に返す

        頭が ``end`` 以降の物と、そこまでの終わりの最大が ``start`` 以前の物は重ならないので、
        その間だけを見る 終わりは並んでいない（長い字幕の後に短い字幕が来る）ので、
        終わりそのものではなく最大で切る
        """
        low = bisect_right(self._reach, start)
        high = bisect_left(self._starts, end)
        for segment in self.segments[low:high]:
            if segment.overlaps(start, end):
                yield segment

    def replace_segment(self, segment: TranscriptSegment) -> Transcript:
        """同じ ID のセグメントを差し替えた新しい :class:`Transcript` を返す"""
        found = False
        replaced = []
        for existing in self.segments:
            if existing.id == segment.id:
                replaced.append(segment)
                found = True
            else:
                replaced.append(existing)
        if not found:
            raise KeyError(f"セグメントが見つからない: {segment.id}")
        return Transcript(tuple(replaced), language=self.language, model=self.model)
