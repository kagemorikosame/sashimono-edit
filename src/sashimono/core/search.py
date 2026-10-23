"""素材の意味検索（#326 A8） 字幕と場面の説明から「あの場面」を引く

2026-10-20 に利用者が決めた既定の案（部品を足さない 2 段）を形にした

1. 作品の中の素材すべての字幕を、表記のゆれ（NFKC・ひらがなとカタカナ・空白）を揃えた
   部分一致で引く タイムラインに置いていない素材も引く 言い換え（「食べた」と「食事」）は
   引き方では拾わず、AI が語を増やして何度か引く
2. 頼まれたときだけ、AI が場面ごとの絵を見て作った説明文（:class:`SceneNote`）を覚え、
   字幕と同じ引き方で引く 説明は作品の外（``engine/cache``）に覚える

索引（SQLite の FTS5 など）は持たない 字幕は作品の中にあり、数時間分でも数万行で、
揃えた文の部分一致で一瞬に引ける（``tests/core/test_media_search.py`` が測る） 索引を
持つと、字幕を直すたびに作り直す手間と、作品と食い違う控えが増える
"""

from __future__ import annotations

import functools
import itertools
import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from sashimono.core.model import (
    Clip,
    MediaId,
    MediaItem,
    Project,
    SegmentId,
    Track,
    heard_stream,
)
from sashimono.core.timebase import Rounding, seconds_to_frame

__all__ = [
    "KINDS",
    "SCENE",
    "SUBTITLE",
    "Hit",
    "SceneNote",
    "fold",
    "hears_stream",
    "hit_places",
    "media_places",
    "prepared_queries",
    "scene_spans",
    "search_media",
]

#: 当たった物の種類 字幕の行と、場面の説明
SUBTITLE = "subtitle"
SCENE = "scene"
KINDS = (SUBTITLE, SCENE)

#: カタカナ（ァ〜ヶ）をひらがなへ 長音「ー」はどちらでも同じ字なので触らない
_HIRAGANA = str.maketrans({chr(code): chr(code - 0x60) for code in range(0x30A1, 0x30F7)})


@functools.lru_cache(maxsize=262144)
def fold(text: str) -> str:
    """表記のゆれを揃えた形 引く語と引かれる文の両方をこれに通して比べる

    NFKC で全角の英数字と半角のカナを揃え、大文字と小文字を揃え、カタカナをひらがなへ
    寄せ、空白を落とす 空白を落とすのは、起こしが語の間に入れた空白で同じ語が割れないため
    同じ字幕を何度も引くので覚えておく（数万行を引き直すたびに揃え直すと遅い）
    """
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_HIRAGANA)
    return "".join(folded.split())


@dataclass(frozen=True, slots=True)
class SceneNote:
    """場面 1 つの説明 AI が絵を見て作った文 時刻は素材の頭から数えた秒"""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError(f"場面の時刻が数ではない: {self.start}〜{self.end}")
        if not 0 <= self.start < self.end:
            # 長さの無い場面は引いても行けない 逆さの場面はタイムラインの位置が決まらない
            raise ValueError(f"場面の時刻が不正: {self.start}〜{self.end}")
        if not self.text.strip():
            raise ValueError("場面の説明が空です")


@dataclass(frozen=True, slots=True)
class Hit:
    """当たった字幕の行か場面の説明 1 つ 時刻は素材の頭から数えた秒"""

    kind: str
    media_id: MediaId
    start: float
    end: float
    text: str
    #: 当たった語（渡された順） 言い換えを並べたとき、多く当たった物を先に並べる
    matched: tuple[str, ...]
    #: 字幕の音声ストリームの番号（素材の中の ``index``） 場面の説明は ``None``
    stream: int | None = None
    segment_id: SegmentId | None = None


def prepared_queries(raw: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """引く語（渡された形, 揃えた形）の並び 揃えて同じになる語は 1 つにまとめる

    揃えると空になる語（空白だけ・記号の消える語）は断る 空の語はどの文にも当たり、
    全部の行が返って何も絞れない
    """
    prepared: list[tuple[str, str]] = []
    seen: set[str] = set()
    for word in raw:
        folded = fold(word)
        if not folded:
            raise ValueError(f"空の語は引けません（{word!r}）")
        if folded in seen:
            continue
        seen.add(folded)
        prepared.append((word, folded))
    return tuple(prepared)


def search_media(
    project: Project,
    queries: Sequence[str],
    *,
    notes_of: Callable[[MediaItem], Sequence[SceneNote]],
    media_id: MediaId | None = None,
    kinds: Sequence[str] = KINDS,
) -> list[Hit]:
    """作品の素材すべて（``media_id`` を渡せばその素材だけ）から語を含む物を引く

    並びは当たった語の多い順、同じなら素材の並び（素材一覧と同じ）と時刻の順
    場面の説明（``notes_of``）は ``kinds`` に :data:`SCENE` があるときだけ読む 説明は
    作品の外の控えにあり、字幕だけを引くときに素材の数だけ読みに行かないため
    """
    words = prepared_queries(queries)
    found: list[tuple[int, int, Hit]] = []
    for order, media in enumerate(project.media):
        if media_id is not None and media.id != media_id:
            continue
        if SUBTITLE in kinds:
            for stream, transcript in media.transcripts:
                for segment in transcript.segments:
                    matched = _matched(segment.text, words)
                    if matched:
                        hit = Hit(
                            kind=SUBTITLE,
                            media_id=media.id,
                            start=_seconds(segment.start),
                            end=_seconds(segment.end),
                            text=segment.text,
                            matched=matched,
                            stream=stream,
                            segment_id=segment.id,
                        )
                        found.append((order, 0, hit))
        if SCENE in kinds:
            for note in notes_of(media):
                matched = _matched(note.text, words)
                if matched:
                    hit = Hit(SCENE, media.id, note.start, note.end, note.text, matched)
                    found.append((order, 1, hit))
    found.sort(key=lambda entry: (-len(entry[2].matched), entry[0], entry[2].start, entry[1]))
    return [hit for _order, _kind, hit in found]


def _matched(text: str, words: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    folded = fold(text)
    return tuple(word for word, wanted in words if wanted in folded)


def _seconds(value: Fraction) -> float:
    # 返す秒は 3 桁に丸める AI に渡す値で、フレームより細かい桁は読み違いの元になるだけ
    return round(float(value), 3)


def hears_stream(
    project: Project, media: MediaItem, stream: int | None
) -> Callable[[Track, Clip], bool] | None:
    """当たりの音声 ``stream`` を実際に鳴らすクリップだけを通す :func:`media_places` の ``accept``

    字幕の当たり（:attr:`Hit.stream`）の行き先を決めるときに使う 照らさないと、同じ素材の
    同じ時刻を含む別の言語の音のクリップへ行き、探した言葉が聞こえない所を示す
    鳴らす番号は :func:`~sashimono.core.model.heard_stream` で引き、字幕の鍵と同じく
    :meth:`~sashimono.core.model.MediaItem.transcript_stream` でそろえて
    比べる（#394 と同じ照らし方）
    場面の説明の当たり（``None``）は音を持たないので限らない（``None`` を返す）
    """
    if stream is None:
        return None
    wanted = media.transcript_stream(stream)

    def hears(track: Track, clip: Clip) -> bool:
        return (
            project.plays_sound(track, clip)
            and media.transcript_stream(heard_stream(track, clip)) == wanted
        )

    return hears


def media_places(
    project: Project,
    media: MediaItem,
    seconds: float,
    accept: Callable[[Track, Clip], bool] | None = None,
) -> list[tuple[Clip, int]]:
    """素材の ``seconds`` が流れる、今のタイムラインの（クリップ, フレーム） フレームの順

    見る（``accept``）のは素材のクリップだけ 止めた絵（``hold_at``）より後ろの時刻と、
    クリップの頭より前の時刻は流れないので外す フレームを CEIL で取るのは、その時刻の絵か
    音が最初に出るフレームにするため（場面の切り替わり #319 と同じ）
    """
    moment = Fraction(str(seconds))
    places: list[tuple[Clip, int]] = []
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.media_id != media.id or (accept is not None and not accept(track, clip)):
                continue
            if moment < clip.source_in or (clip.hold_at is not None and moment > clip.hold_at):
                continue
            offset = seconds_to_frame(
                (moment - clip.source_in) / clip.speed, project.rate, Rounding.CEIL
            )
            frame = clip.timeline_start + offset
            if frame < clip.timeline_end:
                places.append((clip, frame))
    return sorted(places, key=lambda place: place[1])


def hit_places(
    project: Project, media: MediaItem, seconds: float, stream: int | None
) -> list[tuple[Clip, int]]:
    """当たりの行き先（:func:`media_places`） 字幕の当たりは、その音を鳴らすクリップを選ぶ

    人の〔探す…〕の窓と AI の search_media が同じここを通る

    - 当たりの音声 ``stream`` を鳴らすクリップ（:func:`hears_stream`）があればそれだけ
    - 無ければ音を鳴らさないクリップ（映像のトラックに絵だけを置いた物）
      絵だけを置いた作品でも、言葉の出る場面の絵は示せる 外すと「置いていない」と答え、
      AI が同じ素材を置き直す
    - 別の音声を鳴らすクリップは選ばない 選ぶと、探した言葉の聞こえない所へ行く

    場面の説明の当たり（``stream`` が ``None``）は限らない
    """
    hears = hears_stream(project, media, stream)
    if hears is None:
        return media_places(project, media, seconds)
    heard = media_places(project, media, seconds, hears)
    if heard:
        return heard
    return media_places(
        project, media, seconds, lambda track, clip: not project.plays_sound(track, clip)
    )


def scene_spans(cuts: Sequence[float], end: float, longest: float) -> list[tuple[float, float]]:
    """切り替わりの秒 ``cuts`` で ``0〜end`` を場面に分け、``longest`` 秒より長い場面は等分する

    場面の説明を作る単位 切り替わりの無い長い 1 カット（話しっぱなしの動画）を 1 つの場面に
    すると、1 枚の絵で何十分を説明することになり、途中の出来事を引けない
    """
    if end <= 0:
        return []
    marks = [0.0, *(cut for cut in sorted(cuts) if 0 < cut < end), end]
    spans: list[tuple[float, float]] = []
    for head, tail in itertools.pairwise(marks):
        pieces = max(1, math.ceil((tail - head) / longest - 1e-9))
        width = (tail - head) / pieces
        spans += [
            (round(head + width * index, 3), round(head + width * (index + 1), 3))
            for index in range(pieces)
        ]
    return spans
