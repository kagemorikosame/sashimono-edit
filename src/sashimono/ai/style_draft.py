"""本人の過去の作品から演出の癖を数にし、編集の方針（#343）の下書きを作る（#470 段 2 #477）

方針は本人が手で書くしかなく、書かれていない作品では AI が演出の加減を決める材料が無い
本人が仕上げた作品（``.sme``）を読むだけで開き、書体・色・効果・字幕の見た目・カットの長さ・
演出の密度を数え、方針の窓の欄へ下書きとして入れる 確定は本人が窓で直して OK を押したとき
（``SetStyleGuide`` 取り消し 1 回で戻る）

書き出した動画は読まない（2026-11-16 利用者の決定） 測れるのはカットの長さくらいで、手間に
見合わない 下書きの文には数と書体・色・効果の種類だけを書き、作品の名前・素材のファイル名・
テキストの本文は書かない 方針は AI へ送る物で、作品の外へ出ていく

数える中身は照らし（``ai/style_check.py`` の ``clip_styles``）と同じ物を使う 別々に読むと、
下書きのまま確定した作品を照らしたときに、下書きが拾わなかった書体や色が「方針の外」と返る
エフェクトの定義（塗りの既定の色）を読むので、照らしと同じく AI の層に置く
"""

from __future__ import annotations

import itertools
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from sashimono.ai.style_check import COLOR, EFFECT, FONT, clip_styles, track_styles
from sashimono.core.direction import COLOR as COLOR_KIND
from sashimono.core.direction import (
    CUT,
    KINDS_BY_KEY,
    SHAKE,
    SOUND,
    TELOP,
    TRANSITION,
    ZOOM,
    DirectionReport,
    measure_direction,
)
from sashimono.core.io.serialize import load_project
from sashimono.core.model import AnimatedValue, Clip, Project, StyleGuide, Timeline
from sashimono.core.timebase import frame_to_seconds
from sashimono.effects import registry
from sashimono.effects.sources import TEXT

__all__ = [
    "StyleHabits",
    "SubtitleLook",
    "Tally",
    "draft_guide",
    "measure_habits",
    "read_habits",
]

#: 下書きの文に 1 分あたりの回数を書く型の並び 色は覆う割合で、カットは長さで別に書く
_PER_MINUTE = (ZOOM, SHAKE, TRANSITION, TELOP, SOUND)
#: 色のエフェクトの分類 ``measure_direction`` の道具と同じ物を色に数える
_COLOR_CATEGORY = "色"
#: 字の大きさの項目と、項目が無いときの大きさ（``effects/sources.py`` の ``TEXT`` の既定）
_SIZE = "size"
_DEFAULT_SIZE = 64.0
_FILL_WHERE = "字の塗り"
_STROKE_WHERE = "縁取りの層 1"


@dataclass(frozen=True, slots=True)
class Tally:
    """使った値 1 つと、使ったクリップの数"""

    value: str
    count: int


@dataclass(frozen=True, slots=True)
class SubtitleLook:
    """いちばん多く使った字幕の見た目"""

    font: str
    size: float
    color: str | None
    #: 縁取りの層 1 の色 縁取りが無ければ ``None``
    outline: str | None
    #: この見た目の字幕の数
    count: int
    #: 字幕の数
    total: int


@dataclass(frozen=True, slots=True)
class StyleHabits:
    """1 本の作品から数えた癖"""

    #: メインのタイムラインの長さ（秒）
    seconds: float
    #: 多い順 同じ数なら作品の頭から先に出た順
    fonts: tuple[Tally, ...]
    colors: tuple[Tally, ...]
    effects: tuple[Tally, ...]
    subtitle: SubtitleLook | None
    #: カットの長さ（秒）の 4 分の 1・中央・4 分の 3 絵が無ければ ``None``
    shot_seconds: tuple[float, float, float] | None
    #: 型ごとの 1 分あたりの回数（``core/direction.py`` の型の名前で引く）
    per_minute: Mapping[str, float]
    #: 色のエフェクトが覆う長さの割合（0〜1）
    color_coverage: float | None


def _timelines(project: Project) -> Iterator[Timeline]:
    yield project.timeline
    for scene in project.scenes:
        yield scene.timeline


def _ranked(counter: Counter[str]) -> tuple[Tally, ...]:
    # most_common は同じ数なら先に数えた順を保つ 作品の頭から読んだ順になる
    return tuple(Tally(value, count) for value, count in counter.most_common())


def _shot_seconds(report: DirectionReport, project: Project) -> tuple[float, float, float] | None:
    """カットの長さの 4 分の 1・中央・4 分の 3

    中央だけだと、短く詰める癖と長く見せる癖の幅が分からない
    """
    cut = next(k for k in report.kinds if k.kind.key == CUT)
    if cut.median_gap_seconds is None:
        # 絵の無い作品 カットの長さを書くと、無いカットの長さを方針に入れる
        return None
    starts = sorted({e.start_frame for e in report.events if e.kind == CUT})
    bounds = [report.start_frame, *starts, report.end_frame]
    rate = project.timeline.rate
    lengths = [float(frame_to_seconds(b - a, rate)) for a, b in itertools.pairwise(bounds)]
    if len(lengths) == 1:
        return lengths[0], lengths[0], lengths[0]
    low, middle, high = statistics.quantiles(lengths, n=4, method="inclusive")
    return low, middle, high


def _size(clip: Clip) -> float:
    """字の大きさ 線なら頭の値（字幕の大きさを動かす人は少なく、頭で見た目を代表させる）"""
    value = clip.source.params.get(_SIZE) if clip.source is not None else None
    if isinstance(value, AnimatedValue):
        return float(value.at(0))
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return _DEFAULT_SIZE


def _subtitle_look(project: Project) -> SubtitleLook | None:
    looks: Counter[tuple[str, float, str | None, str | None]] = Counter()
    for timeline in _timelines(project):
        for track in timeline.tracks:
            for clip in track.clips:
                source = clip.source
                if not clip.enabled or source is None or source.kind != TEXT.kind:
                    continue
                if clip.subtitle_origin is None:
                    continue
                font = fill = outline = None
                for kind, value, where in clip_styles(clip):
                    if kind == FONT:
                        font = value
                    elif kind == COLOR and where == _FILL_WHERE and fill is None:
                        fill = value
                    elif kind == COLOR and where == _STROKE_WHERE:
                        outline = value
                looks[(font or "", _size(clip), fill, outline)] += 1
    if not looks:
        return None
    (font, size, color, outline), count = looks.most_common(1)[0]
    return SubtitleLook(font, size, color, outline, count, sum(looks.values()))


def measure_habits(project: Project) -> StyleHabits:
    """作品の癖を数える 作品は変えない

    書体・色・効果はメインとすべてのシーンを数える（照らしもすべてを見る） 演出の密度とカットの
    長さはメインだけ（シーンの中を数えると、同じシーンを何度も置いた作品で回数が膨らむ）
    """
    tallies: dict[str, Counter[str]] = {FONT: Counter(), COLOR: Counter(), EFFECT: Counter()}
    for timeline in _timelines(project):
        for track in timeline.tracks:
            for kind, value, _ in track_styles(track):
                tallies[kind][value] += 1
            for clip in track.clips:
                if not clip.enabled:
                    # 切ったクリップは絵にも音にも出ない 本人が外した物を癖に数えない
                    continue
                # 1 本のクリップで同じ値が何か所かに出ても 1 回 縁取りの層が多いだけで数が膨らまない
                # 集合でなく dict で重なりを除く 同じ数の値の並びが、走らせるたびに変わらないように
                for kind, value in dict.fromkeys((k, v) for k, v, _ in clip_styles(clip)):
                    tallies[kind][value] += 1
    color_kinds = frozenset(d.kind for d in registry.all() if d.category == _COLOR_CATEGORY)
    report = measure_direction(project, project.timeline, color_kinds=color_kinds)
    color = next(k for k in report.kinds if k.kind.key == COLOR_KIND)
    return StyleHabits(
        seconds=report.seconds,
        fonts=_ranked(tallies[FONT]),
        colors=_ranked(tallies[COLOR]),
        effects=_ranked(tallies[EFFECT]),
        subtitle=_subtitle_look(project),
        shot_seconds=_shot_seconds(report, project),
        per_minute={k.kind.key: k.per_minute for k in report.kinds},
        color_coverage=color.coverage,
    )


def read_habits(path: Path) -> StyleHabits:
    """``.sme`` を読むだけで開いて数える 読めなければ ``ProjectFileError``

    ファイルへは何も書かない（古い形式の写し ``.sme.bak`` も作らない） 本人の仕上げた作品で、
    更新時刻が変わると、どれをいつ直したかの手がかりが消える
    """
    return measure_habits(load_project(Path(path)))


def _number(value: float) -> str:
    """文に書く数 小数 1 桁まで 整数なら小数を付けない（3.0 回 と書かない）"""
    rounded = round(value, 1)
    return str(int(rounded)) if rounded == int(rounded) else f"{rounded}"


def _tempo(habits: StyleHabits) -> str:
    lines = [f"過去の作品（{_number(habits.seconds / 60)} 分）から測った下書き"]
    if habits.shot_seconds is not None:
        low, middle, high = habits.shot_seconds
        lines.append(
            f"1 カットは中央 {_number(middle)} 秒"
            f"（短い 4 分の 1 は {_number(low)} 秒まで・長い 4 分の 1 は {_number(high)} 秒から）"
        )
    counts = (
        f"{KINDS_BY_KEY[key].name} {_number(habits.per_minute.get(key, 0.0))} 回"
        for key in _PER_MINUTE
    )
    lines.append("1 分あたり " + "・".join(counts))
    if habits.color_coverage:
        lines.append(f"色を整えたのは全体の {round(habits.color_coverage * 100)}%")
    return "\n".join(lines)


def _subtitles(look: SubtitleLook | None) -> str:
    if look is None:
        return ""
    parts = [f"書体 {look.font}", f"大きさ {_number(look.size)}px"]
    if look.color is not None:
        parts.append(f"字の色 {look.color}")
    parts.append(f"縁取り {look.outline}" if look.outline is not None else "縁取りなし")
    return "・".join(parts) + f"（字幕 {look.total} 本のうち {look.count} 本がこの見た目）"


def _values(tallies: Iterable[Tally]) -> tuple[str, ...]:
    return tuple(t.value for t in tallies)


def draft_guide(habits: StyleHabits) -> StyleGuide:
    """数えた癖を方針の下書きにする 多い順に入れ、持てない分（上限を越えた分）は落とす

    書体・色・効果は使った物を全部入れる 一部だけを入れると、下書きのまま確定した作品を照らすと
    本人が使った物が「方針の外」と返る 減らすのは本人が窓で決める そのほかの欄は空のまま
    （数から書ける事が無く、本人の文を潰さない）
    """
    return StyleGuide(
        fonts=_values(habits.fonts),
        colors=_values(habits.colors),
        subtitles=_subtitles(habits.subtitle),
        tempo=_tempo(habits),
        effects=_values(habits.effects),
    ).fitted()
