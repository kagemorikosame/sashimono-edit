"""目印（マーカー）からチャプター（目次）を作る（#330）

チャプターは名前の付いた目印 1 つにつき 1 つ 名前の無い目印は、編集の途中の覚え書きとして
置くことが多いので、目次には入れない（入れると「無題」の行が並ぶ）

時刻は書き出す範囲の頭から数える 書き出し範囲（:attr:`Timeline.work_area`）を決めてあれば
書き出した動画はその頭から始まるので、タイムラインの時刻のまま書くと目次がずれる
範囲の外の目印は、書き出した動画に無い場面なので入れない

YouTube の概要欄の目次（``0:00 はじめに`` を行ごとに並べた文）は :func:`youtube_text` で作る
YouTube が目次として読む条件（先頭が 0:00・3 つ以上・どれも 10 秒以上）に合わないときは、
勝手に足したり詰めたりせず、:attr:`Chapters.warnings` で知らせる 何を足すか・どれを
まとめるかは編集の判断で、黙って変えると本人の付けた名前と位置が食い違う
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from sashimono.core.commands.work_area import export_range
from sashimono.core.model import Timeline
from sashimono.core.timebase import frame_to_seconds

__all__ = [
    "YOUTUBE_MIN_CHAPTERS",
    "YOUTUBE_MIN_SECONDS",
    "Chapter",
    "Chapters",
    "chapters_of",
    "clock",
    "youtube_text",
]

#: YouTube が概要欄の時刻を目次として読む条件 外れると、ただの時刻のリンクとして扱われ、
#: 再生バーが区切られない（YouTube ヘルプの「動画のチャプター」）
YOUTUBE_MIN_CHAPTERS = 3
YOUTUBE_MIN_SECONDS = 10


@dataclass(frozen=True, slots=True)
class Chapter:
    """目次の 1 行 時刻は書き出す動画の頭からの秒（分数）"""

    #: 元の目印のタイムラインの位置 AI が目印を直すときに指す
    frame: int
    start: Fraction
    end: Fraction
    title: str

    @property
    def length(self) -> Fraction:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Chapters:
    chapters: tuple[Chapter, ...]
    #: 名前が無くて目次に入れなかった目印の数
    unnamed: int
    #: 書き出す範囲の外で入れなかった目印の数
    outside: int
    #: 書き出す動画の長さ（秒）
    duration: Fraction

    @property
    def warnings(self) -> tuple[str, ...]:
        """YouTube が目次として読まない理由 空なら読める"""
        notes: list[str] = []
        if not self.chapters:
            return ("名前の付いた目印がありません",)
        if self.chapters[0].start != 0:
            notes.append(
                "先頭の目次が 0:00 から始まっていません（YouTube は 0:00 の行が無いと目次に"
                "しません）"
            )
        if len(self.chapters) < YOUTUBE_MIN_CHAPTERS:
            notes.append(
                f"目次が {len(self.chapters)} 行です（YouTube は {YOUTUBE_MIN_CHAPTERS} 行以上で"
                "目次にします）"
            )
        short = [c for c in self.chapters if c.length < YOUTUBE_MIN_SECONDS]
        if short:
            names = "・".join(f"{clock(c.start)} {c.title}" for c in short)
            notes.append(
                f"{YOUTUBE_MIN_SECONDS} 秒より短い目次があります: {names}（YouTube は"
                "どれも 10 秒以上で目次にします）"
            )
        seen: set[int] = set()
        for chapter in self.chapters:
            # 秒の切り捨てで同じ時刻になると、概要欄では同じ行が 2 つに見える
            second = math.floor(chapter.start)
            if second in seen:
                notes.append(f"同じ秒（{clock(chapter.start)}）に目次が 2 つあります")
                break
            seen.add(second)
        return tuple(notes)


def chapters_of(timeline: Timeline) -> Chapters:
    """タイムラインの目印から目次を作る"""
    rate = timeline.rate
    span = export_range(timeline)
    start, end = span if span is not None else (0, timeline.duration)
    unnamed = sum(1 for marker in timeline.markers if not marker.label.strip())
    named = sorted(
        (marker for marker in timeline.markers if marker.label.strip()),
        key=lambda marker: marker.frame,
    )
    inside = [marker for marker in named if start <= marker.frame < end]
    duration = frame_to_seconds(max(0, end - start), rate)
    rows: list[Chapter] = []
    for index, marker in enumerate(inside):
        until = inside[index + 1].frame if index + 1 < len(inside) else end
        rows.append(
            Chapter(
                frame=marker.frame,
                start=frame_to_seconds(marker.frame - start, rate),
                end=frame_to_seconds(until - start, rate),
                title=marker.label.strip(),
            )
        )
    return Chapters(tuple(rows), unnamed, len(named) - len(inside), duration)


def clock(seconds: Fraction) -> str:
    """``1:05`` ``1:02:03`` の形 秒は切り捨て

    繰り上げると、目次を押したときに場面の頭より後ろから再生が始まり、頭が欠ける
    """
    whole = math.floor(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def youtube_text(chapters: Chapters) -> str:
    """YouTube の概要欄に貼る目次 1 行に 1 つ ``0:00 はじめに`` の形"""
    return "\n".join(f"{clock(c.start)} {c.title}" for c in chapters.chapters)
