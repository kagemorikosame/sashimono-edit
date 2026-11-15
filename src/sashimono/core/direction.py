"""演出の型と、作品に付いた演出の密度を数える計算（#470 段 1 #476）

AI に「いい感じに演出して」と頼んだとき、付けすぎ・足りないを言葉でなく数で言えるようにする
演出を付ける道具（寄る・揺らす・場面切り替え・テロップ・効果音・色・速さ）はどれも専用の
種類を作らず、配置の値の線・ランダム移動・テキストのクリップなど人と同じ物で表している
なので数えるときも、作品の中のその物を読んで「演出の出来事」に直す

ここは Qt もエフェクトの定義も知らない 色の補正に当たるエフェクトの種類（定義の分類が「色」の物）
は呼び手が渡す 定義を読むとコア層がエフェクトの登録に縛られ、試験も AI も同じ API で呼べなくなる

数えるのは「人がそれを演出として足した」と読める物だけ 次は数えない
- 無効にしたクリップとエフェクト 書き出しに出ない物を数えると、外した演出がまだ残って見える
- 起こしから置いた字幕（:attr:`Clip.subtitle_origin` のあるテキスト） 字幕は全体に付くので、
  テロップに数えると 1 分あたりの回数が字幕の数で埋まる
- 映像に付いた音・長い音（BGM）・ナレーションの声 効果音は短い音だけの素材で、
  ほかを数えると話している間ずっと効果音が鳴っているように見える
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    ParamValue,
    Project,
    Timeline,
    Track,
)
from sashimono.core.timebase import FrameRate, frame_to_seconds

__all__ = [
    "COLOR",
    "CUT",
    "DIRECTION_KINDS",
    "KINDS_BY_KEY",
    "NARRATION_FOLDER",
    "SHAKE",
    "SHAKE_EFFECT_KIND",
    "SOUND",
    "SOUND_EFFECT_MAX_SECONDS",
    "TELOP",
    "TRANSITION",
    "ZOOM",
    "CloseCall",
    "DirectionEvent",
    "DirectionKind",
    "DirectionReport",
    "KindDensity",
    "direction_events",
    "measure_direction",
]

ZOOM = "zoom"
SHAKE = "shake"
TRANSITION = "transition"
TELOP = "telop"
SOUND = "sound"
COLOR = "color"
CUT = "cut"

#: 揺れに使う動きのエフェクト（``effects/emphasis.py`` の ``SHAKE_EFFECT_KIND`` と同じ）
#: エフェクトの層を読むとコア層が定義に縛られるので、名前だけを持つ 食い違いは試験が見る
SHAKE_EFFECT_KIND = "random_move"
#: ナレーションの声を書くフォルダ（``engine/audio/voicevox.py`` と同じ） 声は短い音だけの
#: 素材なので、ここで除かないと 1 文ごとに効果音を 1 回と数える 食い違いは試験が見る
NARRATION_FOLDER = "narration"
#: 効果音とみなす長さの上限（秒） 効果音は長くても数秒で、10 秒を超える音だけの素材は
#: BGM か環境音として流している物 BGM を数えると、全体に 1 回の演出が「1 回」に入ってしまう
SOUND_EFFECT_MAX_SECONDS = 10

#: 配置の値の線のうち、寄る・ずらすに使う欄
#: （``core/emphasis.py`` の ``zoom_curves`` ``pan_curves``）
_TRANSFORM_KIND = "transform"
_SCALE = "scale"
_POSITIONS = ("pos_x", "pos_y")
_SHAKE_RANGES = ("range_x", "range_y")
_TEXT_KIND = "text"
_TRANSITION_KIND = "transition"
_FILTER_KIND = "filter"
#: 値が動いたとみなす差 画面の欄の行き来で 0.001 ほどずれる物を、動きと読まない
_MOVED = 1e-3


@dataclass(frozen=True, slots=True)
class DirectionKind:
    """演出の型 1 つ

    ``min_gap_seconds`` は、同じ種類の演出がこれより近く続くと「やりすぎ」とみなす間隔
    既定の値は、知らない人が見て落ち着かなくなる手前に置いた 好みで変わる強さの加減
    （控えめ・ふつう・派手）は段 3（#478）で設定に出すので、ここは測るための線にとどめる
    ``None`` は間隔で測らない型（色は場面ごとに付くので、近さはやりすぎの印にならない）
    """

    key: str
    name: str
    #: 作品の中の何を数えるか（AI と人へそのまま見せる）
    counts: str
    min_gap_seconds: float | None
    #: 強さの単位 強さを持たない型は空
    unit: str = ""


#: 演出の型の一覧 並びは Issue #470 の例の順（寄る → 揺らす → 場面切り替え → テロップ →
#: 効果音 → 色 → テンポ） 型を足すときはここと :func:`direction_events` の両方へ足す
DIRECTION_KINDS: tuple[DirectionKind, ...] = (
    DirectionKind(
        ZOOM,
        "寄る・ずらす",
        "素材とシーンのクリップの配置（拡大率・位置）のキーフレームが動く所 寄って戻すまでで 1 回",
        # 寄るのが 5 秒おきでは見ている人が酔う（#470 の例） 8 秒あれば話の 1 区切りに 1 回ほど
        8.0,
        "%",
    ),
    DirectionKind(
        SHAKE,
        "揺らす",
        "ランダム移動のエフェクト（幅にキーフレームがあれば幅が 0 より大きい所ごと）",
        # 揺れは驚き・笑いの一瞬に使う物で、続くと字幕が読めない
        10.0,
        "px",
    ),
    DirectionKind(
        TRANSITION,
        "場面切り替え",
        "場面切り替えのクリップ",
        # 話題の変わり目に付ける物で、10 秒より短い間に何度も入ると切り替えが目立つだけになる
        10.0,
        "秒",
    ),
    DirectionKind(
        TELOP,
        "テロップ",
        "テキストのクリップ（起こしから置いた字幕は除く）",
        # 要点の文字は読む時間が要る 3 秒に満たない間で次が出ると読み切れない
        3.0,
    ),
    DirectionKind(
        SOUND,
        "効果音",
        f"{SOUND_EFFECT_MAX_SECONDS} 秒以下の音だけの素材のクリップ"
        "（映像の音・BGM・ナレーションは除く）",
        # 2 秒おきより詰めると、話す声に被って聞き取りにくい
        2.0,
    ),
    DirectionKind(
        COLOR,
        "色",
        "色の分類のエフェクトを持つ絵のクリップ（フィルタのクリップも含む）",
        None,
    ),
    DirectionKind(
        CUT,
        "テンポ（カット）",
        "素材とシーンのクリップの頭（タイムラインの頭は除く） 間隔はカットの長さ",
        # 1 秒に満たないカットが続くと、何が映ったかを追えない
        1.0,
    ),
)

KINDS_BY_KEY: Mapping[str, DirectionKind] = {kind.key: kind for kind in DIRECTION_KINDS}


@dataclass(frozen=True, slots=True)
class DirectionEvent:
    """作品の中の演出 1 回 フレームはタイムラインの位置"""

    kind: str
    start_frame: int
    end_frame: int
    clip_id: str
    track_id: str
    #: 強さ（型の単位） 持たない型・読めない物は ``None``
    strength: float | None = None
    #: 寄る・ずらすのうち拡大率が動かずに位置だけが動いた物
    pan_only: bool = False


@dataclass(frozen=True, slots=True)
class CloseCall:
    """下限より近く続いた 2 回"""

    first_frame: int
    second_frame: int
    gap_seconds: float


@dataclass(frozen=True, slots=True)
class KindDensity:
    """1 つの型の数"""

    kind: DirectionKind
    count: int
    per_minute: float
    #: 続く 2 回の頭の間隔（秒） 2 回に満たなければ ``None``
    min_gap_seconds: float | None
    median_gap_seconds: float | None
    strength_min: float | None
    strength_median: float | None
    strength_max: float | None
    #: 下限より近い所 型が間隔で測らなければ空
    too_close: tuple[CloseCall, ...]
    #: 使った下限（引数で上書きした値を含む）
    limit_seconds: float | None
    #: 色だけ 色のエフェクトを持つクリップが覆う長さの割合（0〜1）
    coverage: float | None = None


@dataclass(frozen=True, slots=True)
class DirectionReport:
    """測った範囲と、型ごとの数"""

    start_frame: int
    end_frame: int
    seconds: float
    kinds: tuple[KindDensity, ...]
    events: tuple[DirectionEvent, ...]
    #: 中を数えていないシーンのクリップの数 シーンの中の演出は、そのシーンを測ると数えられる
    scene_clips: int = 0
    #: 速さを変えた素材のクリップの数（テンポの添え）
    speed_changes: int = 0


def _curve(params: Mapping[str, ParamValue], name: str) -> AnimatedValue | None:
    value = params.get(name)
    return value if isinstance(value, AnimatedValue) else None


def _away_spans(curves: Sequence[tuple[AnimatedValue, float]]) -> list[tuple[int, int]]:
    """どれかの線が休みの値から離れている区間（クリップの頭からのフレーム）

    ``curves`` は線と休みの値の組 キーフレームの間ごとに、両端のどちらかが休みから離れていれば
    その間を「動いている」とし、続く間をつなげて 1 回にする 寄って・止まって・戻すまでを
    1 回に数えるため（点ごとに数えると、強調 1 回が 3 回に見える）
    """
    frames = sorted({k.frame for curve, _ in curves for k in curve.keyframes})

    def away(frame: int) -> bool:
        return any(abs(curve.at(frame) - rest) > _MOVED for curve, rest in curves)

    spans: list[tuple[int, int]] = []
    for left, right in itertools.pairwise(frames):
        if not (away(left) or away(right)):
            continue
        if spans and spans[-1][1] == left:
            spans[-1] = (spans[-1][0], right)
        else:
            spans.append((left, right))
    return spans


def _zoom_events(track: Track, clip: Clip) -> Iterator[DirectionEvent]:
    transform = next(
        (e for e in clip.effects if e.kind == _TRANSFORM_KIND and e.enabled and e.fixed), None
    )
    if transform is None:
        return
    scale = _curve(transform.params, _SCALE)
    positions = [c for c in (_curve(transform.params, n) for n in _POSITIONS) if c is not None]
    curves = [(c, c.keyframes[0].value) for c in (scale, *positions) if c and c.is_animated]
    for start, end in _away_spans(curves):
        strength: float | None = None
        pan_only = True
        if scale is not None and scale.is_animated:
            rest = scale.keyframes[0].value
            inside = [k.value for k in scale.keyframes if start <= k.frame <= end]
            if rest > 0 and any(abs(value - rest) > _MOVED for value in inside):
                pan_only = False
                # 休みの大きさからの比 寄る（120%）と引く（80%）の両方で、離れた方の端を返す
                strength = max((value / rest * 100 for value in inside), key=lambda p: abs(p - 100))
        yield _event(ZOOM, track, clip, start, end, strength, pan_only=pan_only)


def _shake_events(track: Track, clip: Clip) -> Iterator[DirectionEvent]:
    for effect in clip.effects:
        if effect.kind != SHAKE_EFFECT_KIND or not effect.enabled:
            continue
        animated = [c for c in (_curve(effect.params, n) for n in _SHAKE_RANGES) if c]
        if any(c.is_animated for c in animated):
            curves = [(c, 0.0) for c in animated if c.is_animated]
            for start, end in _away_spans(curves):
                peak = max(
                    (k.value for c, _ in curves for k in c.keyframes if start <= k.frame <= end),
                    default=None,
                )
                yield _event(SHAKE, track, clip, start, end, peak)
            continue
        # 幅が動かないランダム移動は、クリップの間ずっと揺れる 1 回の演出
        widths = [_static(effect, name) for name in _SHAKE_RANGES]
        known = [w for w in widths if w is not None]
        if known and max(known) <= 0:
            continue
        yield _event(SHAKE, track, clip, 0, clip.duration, max(known) if known else None)


def _static(effect: Effect, name: str) -> float | None:
    value = effect.params.get(name)
    if isinstance(value, AnimatedValue):
        return value.static
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    # 項目を持たない（定義の既定のまま）物は、幅を知らないので強さを返さない
    return None


def _event(
    kind: str,
    track: Track,
    clip: Clip,
    start: int,
    end: int,
    strength: float | None = None,
    *,
    pan_only: bool = False,
) -> DirectionEvent:
    """クリップの頭からの範囲をタイムラインの位置に直した出来事 クリップの外は切る"""
    begin = clip.timeline_start + max(0, min(start, clip.duration))
    finish = clip.timeline_start + max(0, min(end, clip.duration))
    return DirectionEvent(
        kind, begin, max(begin, finish), str(clip.id), str(track.id), strength, pan_only=pan_only
    )


def _is_sound_effect(project: Project, track: Track, clip: Clip, rate: FrameRate) -> bool:
    if clip.media_id is None or not project.plays_sound(track, clip):
        return False
    media = project.find_media(clip.media_id)
    if media is None or media.has_video or media.is_still:
        return False
    if media.path.parent.name == NARRATION_FOLDER:
        return False
    return frame_to_seconds(clip.duration, rate) <= SOUND_EFFECT_MAX_SECONDS


def _has_color(clip: Clip, color_kinds: frozenset[str]) -> bool:
    return any(e.enabled and not e.fixed and e.kind in color_kinds for e in clip.effects)


def direction_events(
    project: Project, timeline: Timeline, *, color_kinds: Iterable[str] = ()
) -> list[DirectionEvent]:
    """``timeline`` に付いた演出を、頭の位置の順に並べる

    ``color_kinds`` は色の補正に当たるエフェクトの種類（定義の分類が「色」の物）
    シーンのクリップは 1 本の絵として数え（カット・寄る・揺らす・色）、中は辿らない
    中を辿ると、同じシーンを 2 か所に置いたときに中の演出を 2 倍に数え、測る範囲も合わない
    """
    colors = frozenset(color_kinds)
    rate = timeline.rate
    found: list[DirectionEvent] = []
    for track in timeline.tracks:
        for clip in track.clips:
            if not clip.enabled:
                continue
            source = clip.source
            kind = source.kind if source is not None else None
            if kind == _TRANSITION_KIND:
                seconds = float(frame_to_seconds(clip.duration, rate))
                found.append(_event(TRANSITION, track, clip, 0, clip.duration, seconds))
                continue
            if kind == _TEXT_KIND:
                if clip.subtitle_origin is None:
                    found.append(_event(TELOP, track, clip, 0, clip.duration))
                continue
            if _is_sound_effect(project, track, clip, rate):
                found.append(_event(SOUND, track, clip, 0, clip.duration))
                continue
            picture = project.draws_picture(track, clip)
            if picture and _has_color(clip, colors):
                found.append(_event(COLOR, track, clip, 0, clip.duration))
            if picture and kind == _FILTER_KIND:
                found.extend(_shake_events(track, clip))
                continue
            # 寄る・揺らす・カットは素材とシーンのクリップだけ テキストや図形の配置の動きは
            # 登場の動きとして付けた物で、話者や要所へ寄る演出ではない
            if not picture or source is not None:
                continue
            found.append(_event(CUT, track, clip, 0, clip.duration))
            found.extend(_zoom_events(track, clip))
            found.extend(_shake_events(track, clip))
    found.sort(key=lambda e: (e.start_frame, e.kind, e.track_id))
    return found


def _cut_starts(events: Sequence[DirectionEvent], start: int) -> list[int]:
    """カットの頭 同じ位置で重ねた何本か（絵と上の素材）は 1 回に数え、範囲の頭は除く"""
    return sorted({e.start_frame for e in events if e.start_frame > start})


def _seconds(frames: int, rate: FrameRate) -> float:
    return float(frame_to_seconds(frames, rate))


def _density(
    kind: DirectionKind,
    events: Sequence[DirectionEvent],
    *,
    rate: FrameRate,
    start: int,
    end: int,
    limit: float | None,
) -> KindDensity:
    minutes = max(_seconds(end - start, rate), 1e-9) / 60
    if kind.key == CUT:
        starts = _cut_starts(events, start)
        # カットの長さは、範囲の頭・各カットの頭・範囲の終わりの間 最後のカットも長さに入れる
        bounds = [start, *starts, end]
        gaps = [b - a for a, b in itertools.pairwise(bounds) if b > a]
        pairs = list(itertools.pairwise(bounds))
        count = len(starts)
    else:
        starts = [e.start_frame for e in events]
        gaps = [b - a for a, b in itertools.pairwise(starts)]
        pairs = list(itertools.pairwise(starts))
        count = len(events)
    gap_seconds = [_seconds(g, rate) for g in gaps]
    close: list[CloseCall] = []
    if limit is not None:
        close = [
            CloseCall(a, b, round(_seconds(b - a, rate), 3))
            for a, b in pairs
            if _seconds(b - a, rate) < limit
        ]
    strengths = [e.strength for e in events if e.strength is not None]
    coverage = None
    if kind.key == COLOR:
        coverage = _covered(events, start, end) / max(end - start, 1)
    return KindDensity(
        kind=kind,
        count=count,
        per_minute=round(count / minutes, 2),
        min_gap_seconds=round(min(gap_seconds), 3) if gap_seconds else None,
        median_gap_seconds=round(statistics.median(gap_seconds), 3) if gap_seconds else None,
        strength_min=round(min(strengths), 3) if strengths else None,
        strength_median=round(statistics.median(strengths), 3) if strengths else None,
        strength_max=round(max(strengths), 3) if strengths else None,
        too_close=tuple(close),
        limit_seconds=limit,
        coverage=round(coverage, 3) if coverage is not None else None,
    )


def _covered(events: Sequence[DirectionEvent], start: int, end: int) -> int:
    """重なりを 1 回に数えた、出来事が覆うフレームの数"""
    total = 0
    reach = start
    for event in sorted(events, key=lambda e: e.start_frame):
        begin, finish = max(event.start_frame, reach), min(event.end_frame, end)
        if finish > begin:
            total += finish - begin
            reach = finish
    return total


def measure_direction(
    project: Project,
    timeline: Timeline,
    *,
    color_kinds: Iterable[str] = (),
    start_frame: int | None = None,
    end_frame: int | None = None,
    min_gaps: Mapping[str, float] | None = None,
) -> DirectionReport:
    """``timeline`` の演出を型ごとに数える 範囲の頭が入っている出来事だけを数える

    範囲を省けば全体（0〜タイムラインの長さ） ``min_gaps`` は型ごとのやりすぎの下限（秒）の上書き
    """
    end_of_timeline = timeline.duration
    start = max(0, start_frame if start_frame is not None else 0)
    end = min(end_of_timeline, end_frame if end_frame is not None else end_of_timeline)
    if end <= start:
        raise ValueError(
            f"測る範囲が空です（start_frame {start} end_frame {end}"
            f" タイムラインの長さ {end_of_timeline}）"
        )
    overrides = dict(min_gaps or {})
    unknown = sorted(set(overrides) - set(KINDS_BY_KEY))
    if unknown:
        raise ValueError(f"知らない型: {'、'.join(unknown)}（{'・'.join(KINDS_BY_KEY)} のどれか）")
    every = direction_events(project, timeline, color_kinds=color_kinds)
    # 範囲の頭に置いたクリップはカットでなく始まり 一覧に残すと、数（count）と一覧の行が食い違う
    inside = [
        e
        for e in every
        if start <= e.start_frame < end and not (e.kind == CUT and e.start_frame == start)
    ]
    rate = timeline.rate
    densities = tuple(
        _density(
            kind,
            [e for e in inside if e.kind == kind.key],
            rate=rate,
            start=start,
            end=end,
            limit=overrides.get(kind.key, kind.min_gap_seconds),
        )
        for kind in DIRECTION_KINDS
    )
    scene_clips = 0
    speed_changes = 0
    for track in timeline.tracks:
        for clip in track.clips:
            if not clip.enabled or not clip.overlaps(start, end):
                continue
            if clip.scene_id is not None:
                scene_clips += 1
            if clip.media_id is not None and clip.speed != Fraction(1):
                speed_changes += 1
    return DirectionReport(
        start_frame=start,
        end_frame=end,
        seconds=round(_seconds(end - start, rate), 3),
        kinds=densities,
        events=tuple(inside),
        scene_clips=scene_clips,
        speed_changes=speed_changes,
    )
