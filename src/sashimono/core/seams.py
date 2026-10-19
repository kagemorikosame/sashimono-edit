"""ジャンプカットのつなぎ目を目立たなくする（#332 B6）

無音を詰めた（ジェットカット）所や、同じ素材を飛び飛びにつないだ所は、話者の顔や手が
1 フレームで跳び、音の波形が途中で切れて「プツッ」と鳴る ここでは、そのつなぎ目へ人が
手で付けるのと同じ物を付ける

- 映像のクロスフェード 場面切り替え（クロスフェード）のクリップをつなぎ目に重ねる
  人はタイムラインの〔場面切り替えを追加〕で同じ物を置き、設定パネルで長さと切り替え方を直せる
- 映像の寄り（パンチイン） つなぎ目の後ろのクリップを 1 本おきに少し拡大する 跳びが
  「カメラを替えた」ように見える 配置の拡大率の欄の値を変えるだけ
- 音の継ぎ目のフェード つなぎ目の前のクリップの終わりと後ろのクリップの頭へ、音量の
  フェードの欄（設定パネルの「フェードイン」「フェードアウト」）に短い値を入れる

どれも人の操作と同じ命令を通すので、1 回の呼び出しは取り消し 1 回で戻る ここは Qt も
エフェクトの定義も知らない 場面切り替えの中身（:class:`GeneratedSource`）は呼ぶ側が作って渡す
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import AddEffect, ParamPath, SetParam
from sashimono.core.commands.fixed import FADE_EFFECT_KIND, fixed_effect, fixed_slot
from sashimono.core.commands.insert import insert_clip
from sashimono.core.emphasis import can_emphasize, fixed_transform, transform_commands
from sashimono.core.model import (
    TRANSITION_KIND,
    AnimatedValue,
    Clip,
    ClipId,
    GeneratedSource,
    Project,
    Track,
    TrackId,
)
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "DEFAULT_SEAM_AUDIO_FADE_MS",
    "DEFAULT_SEAM_CROSSFADE_MS",
    "DEFAULT_SEAM_VIDEO",
    "DEFAULT_SEAM_ZOOM_PERCENT",
    "FADE_ALREADY",
    "FADE_MOVING",
    "SEAM_AUDIO_FADE_MS_RANGE",
    "SEAM_CROSSFADE",
    "SEAM_CROSSFADE_MS_RANGE",
    "SEAM_NONE",
    "SEAM_VIDEOS",
    "SEAM_VIDEO_NAMES",
    "SEAM_ZOOM",
    "SEAM_ZOOM_PERCENT_RANGE",
    "UNFIT_SEAM",
    "SeamPlan",
    "SeamScope",
    "SeamStyle",
    "SmoothCuts",
    "find_seams",
    "plan_smoothing",
    "seams_after_cuts",
    "seams_touching",
    "touching_scope",
]

#: 映像に何もしない（音だけ処理する）
SEAM_NONE = "none"
#: 前の絵から後ろの絵へ短く溶かす
SEAM_CROSSFADE = "crossfade"
#: 後ろのクリップを 1 本おきに少し拡大する（パンチイン）
SEAM_ZOOM = "zoom"
SEAM_VIDEOS = (SEAM_NONE, SEAM_CROSSFADE, SEAM_ZOOM)
#: 設定画面に出す名前
SEAM_VIDEO_NAMES: dict[str, str] = {
    SEAM_NONE: "何もしない（音だけ）",
    SEAM_CROSSFADE: "短いクロスフェード",
    SEAM_ZOOM: "1 本おきに少し寄る（パンチイン）",
}
#: 既定は短いクロスフェード 寄りは画面の端（焼き込んだ字幕・頭の上）が欠けることがあり、
#: 知らない人が気付かないまま書き出すと困る 溶かすだけなら絵の中身は欠けない
DEFAULT_SEAM_VIDEO = SEAM_CROSSFADE
#: クロスフェードの長さ（ミリ秒） 100 ms は 30 fps で 3 フレーム 跳びが和らぎ、前の絵が
#: 止まって見えるほど長くない 長いと止めた前の絵が残って二重に見える
DEFAULT_SEAM_CROSSFADE_MS = 100
SEAM_CROSSFADE_MS_RANGE = (10, 1000)
#: 寄りの拡大率（%） 110% は寄ったと分かり、1920x1080 で端が 96 画素ずつ欠けるだけで済む
DEFAULT_SEAM_ZOOM_PERCENT = 110
SEAM_ZOOM_PERCENT_RANGE = (101, 150)
#: 音の継ぎ目のフェード（ミリ秒） 20 ms は「プツッ」が消え、話の頭が欠けて聞こえない長さ
#: 0 は音に何もしない
DEFAULT_SEAM_AUDIO_FADE_MS = 20
SEAM_AUDIO_FADE_MS_RANGE = (0, 200)
#: 対はあるが、選んだ処理を付けられる絵か音の対が無いつなぎ目の理由
UNFIT_SEAM = "選んだ処理を付けられる絵か音のつなぎ目ではない"
#: 音の対に、頼んだ長さ以上のフェードがもう付いている
FADE_ALREADY = "もう頼んだ長さ以上の音のフェードが付いている"
#: 音の対のフェードの長さをキーフレームで動かしている（人が付けた動きは触らない）
FADE_MOVING = "音のフェードの長さがキーフレームで動いている"


@dataclass(frozen=True, slots=True)
class SeamStyle:
    """つなぎ目の処理の既定（本人の設定 AI が値を言わなければこれを使う）"""

    video: str = DEFAULT_SEAM_VIDEO
    crossfade_ms: int = DEFAULT_SEAM_CROSSFADE_MS
    zoom_percent: int = DEFAULT_SEAM_ZOOM_PERCENT
    audio_fade_ms: int = DEFAULT_SEAM_AUDIO_FADE_MS


@dataclass(frozen=True, slots=True)
class SeamScope:
    """つなぎ目で処理する前後のクリップの対の決まり

    フレームだけで探すと、同じフレームで区切れている別のトラックの関係の無いクリップまで
    寄せたりフェードを入れたりし、そこの短いクリップのせいでクロスフェードごと断られる
    ``jumps_only`` は同じ素材を飛び飛びにつないだ対（:func:`find_seams` と同じ決まり）だけ
    ``clip_ids`` を渡せば、前か後ろがその中にある対だけ
    """

    jumps_only: bool = False
    clip_ids: frozenset[ClipId] | None = None

    def takes(self, rate: FrameRate, left: Clip, right: Clip) -> bool:
        if self.jumps_only and not _is_jump(rate, left, right):
            return False
        return self.clip_ids is None or left.id in self.clip_ids or right.id in self.clip_ids


@dataclass(frozen=True, slots=True)
class SeamPlan:
    """つなぎ目の処理の中身と、何をしたか"""

    commands: tuple[Command, ...]
    #: 命令をすべて当てた後のプロジェクト
    project: Project
    #: 処理したつなぎ目（何か 1 つでも付けた所）
    handled: tuple[int, ...] = ()
    #: クロスフェードを置いたつなぎ目と、溶かすフレーム数
    crossfades: tuple[tuple[int, int], ...] = ()
    #: 寄せたクリップ
    zoomed: tuple[ClipId, ...] = ()
    #: 音のフェードを付けたつなぎ目
    audio: tuple[int, ...] = ()
    #: 付けなかった物（つなぎ目, 理由）
    skipped: tuple[tuple[int, str], ...] = field(default=())
    #: 前後のクリップが接していない（つなぎ目ではない）フレーム 詰めた範囲がクリップの頭や
    #: 終わりに掛かっていた所はここに来る
    missing: tuple[int, ...] = ()


def seams_after_cuts(ranges: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """削って詰める範囲（:class:`~sashimono.core.commands.RippleCut`）の後にできるつなぎ目

    詰めると後ろの範囲は前で削った長さだけ前へ寄る 範囲は重なりをまとめた物を渡す
    （:func:`~sashimono.core.jetcut.merge_ranges`）
    """
    seams: list[int] = []
    removed = 0
    for start, end in sorted(ranges):
        seams.append(start - removed)
        removed += end - start
    return tuple(frame for frame in dict.fromkeys(seams) if frame > 0)


def find_seams(
    project: Project,
    *,
    start: int = 0,
    end: int | None = None,
    track_ids: tuple[TrackId, ...] | None = None,
) -> tuple[int, ...]:
    """同じ素材を飛び飛びにつないだ所（ジャンプカット）のフレーム

    同じトラックで前のクリップの終わりと後ろのクリップの頭が接し、どちらも同じ素材で、
    素材の中の時刻が飛んでいる所 ただ分割しただけ（時刻が続いている）所は跳ばないので数えない
    数えると、跳びの無い所にまでクロスフェードが掛かる 別の素材へ切り替えた所も数えない
    （場面を替えたのでジャンプカットではない） 止めた（無効の）クリップとロックしたトラックも見ない
    """
    rate = project.rate
    found: set[int] = set()
    for track in project.timeline.tracks:
        if track.locked or (track_ids is not None and track.id not in track_ids):
            continue
        ordered = sorted(track.clips, key=lambda clip: clip.timeline_start)
        for left, right in pairwise(ordered):
            frame = right.timeline_start
            if left.timeline_end != frame or frame < start or (end is not None and frame >= end):
                continue
            if left.enabled and right.enabled and _is_jump(rate, left, right):
                found.add(frame)
    return tuple(sorted(found))


def _is_jump(rate: FrameRate, left: Clip, right: Clip) -> bool:
    """接した前後のクリップが、同じ素材を飛び飛びにつないだ物か"""
    if left.media_id is None or left.media_id != right.media_id:
        return False
    if left.scene_id or right.scene_id:
        return False
    # 半フレーム分より近ければ続いている 分割の計算は分数なので普通はちょうど一致する
    step = rate.frame_duration * left.speed / 2
    return abs(right.source_in - left.source_out(rate)) > step


def _pairs_at(project: Project, frame: int, scope: SeamScope) -> list[tuple[Track, Clip, Clip]]:
    """``frame`` で接する前後のクリップ（ロックしていないトラック、どちらも有効な物で、
    ``scope`` に入る物）"""
    pairs: list[tuple[Track, Clip, Clip]] = []
    for track in project.timeline.tracks:
        if track.locked:
            continue
        left = next((c for c in track.clips if c.timeline_end == frame), None)
        right = next((c for c in track.clips if c.timeline_start == frame), None)
        if left is None or right is None or not (left.enabled and right.enabled):
            continue
        if scope.takes(project.rate, left, right):
            pairs.append((track, left, right))
    return pairs


def _static(value: object, default: float) -> float | None:
    """動かない値 キーフレームで動いていれば ``None``"""
    if isinstance(value, AnimatedValue):
        return None if value.is_animated else value.static
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _fade_commands(clip: Clip, name: str, seconds: float) -> list[Command]:
    """音量フェードの ``name``（``fade_in`` か ``fade_out``）を ``seconds`` 以上にする命令

    今の値の方が長ければ変えない（人が付けた長いフェードを縮めない） キーフレームで
    動かしていれば触らない 固定のフェードをまだ持たない古いクリップは、値を入れて足す
    1 つの命令にする（:func:`~sashimono.core.commands.volume.volume_commands` と同じ理由）
    """
    effect = next((e for e in clip.effects if e.fixed and e.kind == FADE_EFFECT_KIND), None)
    if effect is None:
        added = fixed_effect(FADE_EFFECT_KIND).with_param(name, AnimatedValue(static=seconds))
        return [AddEffect(clip.id, added, index=fixed_slot(clip.effects, FADE_EFFECT_KIND))]
    now = _static(effect.params.get(name), 0.0)
    if now is None or now >= seconds:
        return []
    return [SetParam(ParamPath.of_effect(clip.id, effect.id, name), AnimatedValue(static=seconds))]


def _fade_note(clip: Clip, name: str) -> str:
    """:func:`_fade_commands` が何も返さなかった訳"""
    effect = next((e for e in clip.effects if e.fixed and e.kind == FADE_EFFECT_KIND), None)
    value = effect.params.get(name) if effect is not None else None
    if isinstance(value, AnimatedValue) and value.is_animated:
        return FADE_MOVING
    return FADE_ALREADY


class _Builder:
    """命令を 1 つずつ当てながら積む 後の命令は前の命令を当てた形を見て組む"""

    def __init__(self, project: Project) -> None:
        self.project = project
        self.commands: list[Command] = []

    def add(self, commands: list[Command]) -> None:
        for command in commands:
            self.project = command.apply(self.project)
            self.commands.append(command)

    def clip(self, clip_id: ClipId) -> Clip:
        located = self.project.timeline.locate_clip(clip_id)
        if located is None:  # pragma: no cover - 前の命令はクリップを消さない
            raise KeyError(f"クリップが見つからない: {clip_id}")
        return located[1]


def plan_smoothing(
    project: Project,
    frames: tuple[int, ...],
    style: SeamStyle,
    transition: GeneratedSource | None = None,
    scope: SeamScope | None = None,
) -> SeamPlan:
    """``frames`` のつなぎ目を ``style`` で処理する命令を組む

    ``transition`` はクロスフェードに置く場面切り替えの中身（クロスフェードの時だけ要る）
    コアはエフェクトの定義を読めないので、呼ぶ側が既定の値を埋めて渡す
    ``scope`` は処理する前後のクリップの対の決まり（:class:`SeamScope`） 省けば、そのフレームで
    接する全トラックの対（AI が位置を名指ししたとき）
    """
    scope = scope or SeamScope()
    if style.video not in SEAM_VIDEOS:
        raise ValueError(f"知らない映像の処理です: {style.video}")
    if style.video == SEAM_CROSSFADE and transition is None:
        raise ValueError("クロスフェードには場面切り替えの中身が要る")
    builder = _Builder(project)
    ordered = tuple(sorted(dict.fromkeys(frames)))
    skipped: list[tuple[int, str]] = []
    handled: set[int] = set()
    audio: list[int] = []
    zoomed: list[ClipId] = []
    crossfades: list[tuple[int, int]] = []
    missing = tuple(frame for frame in ordered if not _pairs_at(project, frame, scope))
    #: 音の対はあったが、もう付いていた（または動かしている）ので何もしなかったつなぎ目の理由
    #: 映像の処理を付けたつなぎ目では言わない（付けたのに断ったように見える）
    sound_notes: dict[int, str] = {}

    if style.audio_fade_ms > 0:
        seconds = style.audio_fade_ms / 1000
        for frame in ordered:
            touched = False
            for track, left, right in _pairs_at(builder.project, frame, scope):
                if not (project.plays_sound(track, left) and project.plays_sound(track, right)):
                    continue
                rate = project.rate
                commands: list[Command] = []
                for clip, name in ((left, "fade_out"), (right, "fade_in")):
                    # 短いクリップでは頭と終わりのフェードが重ならないよう半分までにする
                    limit = float(clip.duration * rate.frame_duration) / 2
                    current = builder.clip(clip.id)
                    planned = _fade_commands(current, name, min(seconds, limit))
                    if not planned:
                        sound_notes.setdefault(frame, _fade_note(current, name))
                    commands += planned
                if commands:
                    builder.add(commands)
                    touched = True
            if touched:
                audio.append(frame)
                handled.add(frame)

    if style.video == SEAM_ZOOM:
        raised: set[ClipId] = set()
        for frame in ordered:
            for track, left, right in _pairs_at(builder.project, frame, scope):
                if not (
                    can_emphasize(project, track, left) and can_emphasize(project, track, right)
                ):
                    continue
                if left.id in raised:
                    # 前のクリップを寄せたので、後ろは元の大きさのまま（1 本おき）
                    # 寄った絵から元の絵へ戻るのも跳びを隠すので、処理したつなぎ目に数える
                    handled.add(frame)
                    continue
                left_scale = _scale(builder.clip(left.id))
                right_scale = _scale(builder.clip(right.id))
                if left_scale is None or right_scale is None:
                    skipped.append((frame, f"{right.id} の拡大率がキーフレームで動いている"))
                    continue
                if not math.isclose(left_scale, right_scale, rel_tol=1e-9):
                    # 前と後ろの大きさが違えば、もう跳びに寄りが付いている 掛け直すと
                    # 何度も呼んだときに拡大が重なっていく
                    skipped.append((frame, f"{right.id} はもう前のクリップと大きさが違う"))
                    continue
                current = builder.clip(right.id)
                builder.add(
                    # 掛けてから割る 先に 1.1 を作ると 110.00000000000001 になり、設定パネルに
                    # 半端な値が出る
                    transform_commands(
                        current,
                        {"scale": AnimatedValue(static=right_scale * style.zoom_percent / 100)},
                    )
                )
                raised.add(right.id)
                zoomed.append(right.id)
                handled.add(frame)

    if style.video == SEAM_CROSSFADE and transition is not None:
        wanted = max(
            1, seconds_to_frame(Fraction(style.crossfade_ms, 1000), project.rate, Rounding.NEAREST)
        )
        for frame in ordered:
            reason = _crossfade(builder, project, frame, wanted, transition, crossfades, scope)
            if reason is None:
                handled.add(frame)
            elif reason:
                skipped.append((frame, reason))

    # 対はあったのに、選んだ処理を付けられる物が無かったつなぎ目（寄りを頼んだのに音だけの対など）
    # 理由を残さないと、呼ぶ側は「付けられなかった」としか言えず、人も AI も何を変えればよいか
    # 分からない
    # 音の対にもう付いていた所は「つなぎ目ではない」と言わず、付いていたと言う（掛け直したときに
    # 間違った案内をしない）
    noted = {frame for frame, _ in skipped}
    for frame in ordered:
        if frame in handled or frame in missing or frame in noted:
            continue
        skipped.append((frame, sound_notes.get(frame, UNFIT_SEAM)))

    return SeamPlan(
        commands=tuple(builder.commands),
        project=builder.project,
        handled=tuple(sorted(handled)),
        crossfades=tuple(crossfades),
        zoomed=tuple(zoomed),
        audio=tuple(audio),
        skipped=tuple(skipped),
        missing=missing,
    )


def _scale(clip: Clip) -> float | None:
    transform = fixed_transform(clip)
    return _static(transform.params.get("scale") if transform else None, 100.0)


def _crossfade(
    builder: _Builder,
    project: Project,
    frame: int,
    wanted: int,
    transition: GeneratedSource,
    placed: list[tuple[int, int]],
    scope: SeamScope,
) -> str | None:
    """つなぎ目に場面切り替えを置く 置いたら ``None``、置けなければ理由を返す
    つなぎ目ではない所（前で数えた）は空の理由

    場面切り替えは、頭より前に始まって範囲の中で終わるクリップを「前の場面」にし、終わりの
    直前の絵で止めて後の場面へ溶かす（:meth:`~sashimono.engine.render.renderer.Renderer._draw_transition`）
    つなぎ目の 1 フレーム前から置くと、つなぎ目の前は今までどおり動き、つなぎ目から
    ``wanted`` フレームで後ろの絵へ溶ける 真ん中に置くと、つなぎ目の前は前後が同じ絵なので
    何も起きず、つなぎ目で急に半分まで溶けた絵が出る
    """
    pictures = [
        (track, left, right)
        for track, left, right in _pairs_at(builder.project, frame, scope)
        if project.draws_picture(track, left) and project.draws_picture(track, right)
    ]
    if not pictures:
        return (
            "" if not _pairs_at(builder.project, frame, scope) else "絵のつなぎ目ではない（音だけ）"
        )
    start = frame - 1
    if any(
        clip.source is not None
        and clip.source.kind == TRANSITION_KIND
        and clip.overlaps(start, frame + 1)
        for track in builder.project.timeline.tracks
        for clip in track.clips
    ):
        return "もう場面切り替えが掛かっている"
    # 前のクリップは置く頭より前に始まっている必要がある（2 フレーム以上） 後ろのクリップの中で
    # 溶かし終える（次のつなぎ目の 1 フレーム前に置く切り替えと重ねない）
    if min(left.duration for _, left, _ in pictures) < 2:
        return "前のクリップが 1 フレームしかない"
    length = min(wanted, *(right.duration - 1 for _, _, right in pictures))
    if length < 1:
        return "後ろのクリップが 1 フレームしかない"
    clip = Clip(timeline_start=start, duration=length + 1, source=transition)
    builder.add(insert_clip(builder.project, clip, at_frame=start))
    placed.append((frame, length))
    return None


@dataclass(frozen=True, slots=True)
class SmoothCuts(Command):
    """つなぎ目の処理を、当てる時のプロジェクトで組んで当てる

    無音を詰める命令（:class:`~sashimono.core.commands.RippleCut`）と 1 回の取り消しで
    まとめるために要る 詰めた後のクリップは詰める命令が当たるまで ID が決まらないので、
    前もって組んだ命令の列では後ろのクリップを指せない フレームだけを持ち、当てる時に
    前後のクリップを探す
    """

    frames: tuple[int, ...]
    style: SeamStyle
    transition: GeneratedSource | None = None
    #: 処理する対の決まり 無音カットは同じ素材を飛び飛びにつないだ対だけ（詰めてできた所）
    scope: SeamScope = SeamScope()

    @property
    def label(self) -> str:
        return f"つなぎ目を目立たなくする: {len(self.frames)} か所"

    def apply(self, project: Project) -> Project:
        return plan_smoothing(project, self.frames, self.style, self.transition, self.scope).project


def seams_touching(project: Project, clip_ids: tuple[ClipId, ...]) -> tuple[int, ...]:
    """``clip_ids`` の頭か終わりにあるジャンプカットのつなぎ目（右クリックの相手）

    選んだクリップのトラックと、リンクした相手のトラックで探す 絵だけを選んでも、音の
    継ぎ目は同じフレームにあるので一緒に処理される 処理する対は :func:`touching_scope` で絞る
    選んでいないクリップどうしのつなぎ目は入れない 範囲で探すと、選んだクリップの間に挟まった
    別の所まで変わる
    """
    timeline = project.timeline
    tracks: set[TrackId] = set()
    edges: set[int] = set()
    for clip_id in clip_ids:
        located = timeline.locate_clip(clip_id)
        if located is None:
            continue
        track, clip = located
        tracks.add(track.id)
        edges.update((clip.timeline_start, clip.timeline_end))
        if clip.link_group is not None:
            tracks.update(partner.id for partner, _ in timeline.linked_clips(clip.link_group))
    if not tracks:
        return ()
    return tuple(frame for frame in find_seams(project, track_ids=tuple(tracks)) if frame in edges)


def touching_scope(project: Project, clip_ids: tuple[ClipId, ...]) -> SeamScope:
    """右クリックで選んだクリップ（とリンクした相手）が前か後ろにいるジャンプカットの対だけ

    フレームだけを渡すと、同じフレームで区切れている別のトラックの、選んでいないクリップまで
    寄せたりフェードを入れたりする
    """
    timeline = project.timeline
    chosen: set[ClipId] = set()
    for clip_id in clip_ids:
        located = timeline.locate_clip(clip_id)
        if located is None:
            continue
        clip = located[1]
        chosen.add(clip.id)
        if clip.link_group is not None:
            chosen.update(partner.id for _, partner in timeline.linked_clips(clip.link_group))
    return SeamScope(jumps_only=True, clip_ids=frozenset(chosen))
