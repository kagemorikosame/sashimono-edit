"""クリップの再生速度を変える命令（#332 B6）

設定パネルの「再生速度」は長さを変えずに速度だけを変える（同じ長さに 2 倍の素材が入る）
AI に「2 倍速にして」と頼むと、多くの人は中身を保ったまま短くなる方を思い浮かべる
どちらも 1 つの命令で持ち、``keep_length`` で選ぶ

リンクした相手（同じ素材の絵と音）にも同じ速度と長さを入れる 片方だけ変えると、絵と音が
ずれていく（設定パネルの :meth:`~sashimono.ui.inspector.panel.InspectorPanel._set_linked` と同じ）
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.commands.base import Command
from sashimono.core.commands.edit import STOP_NEXT, _linked_group, _reads_source_time
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    Keyframe,
    ParamValue,
    Project,
    Track,
)

__all__ = [
    "SPEED_PERCENT_RANGE",
    "ChangeSpeed",
    "can_change_speed",
    "sped_duration",
    "speed_members",
]

#: 速度の範囲（%） 設定パネルの「再生速度」の欄と同じ 1% より遅いと 1 フレームの絵が
#: 何秒も続き、1000% より速いと音が聞き取れない
SPEED_PERCENT_RANGE = (1, 1000)


def can_change_speed(project: Project, track: Track, clip: Clip) -> bool:
    """速度を変えられるクリップか（AI の set_speed と右クリックで同じ決まりにする）

    素材の時間を持たないクリップ（テキスト・図形・静止画）は速度が効かない ロックは人が
    触らせないために掛けた物なので、右クリックからも AI からも変えない
    """
    return not track.locked and _reads_source_time(project, clip)


def speed_members(project: Project, clip_ids: tuple[ClipId, ...]) -> list[tuple[Track, Clip]]:
    """``clip_ids`` の速度を変えると一緒に変わるクリップ

    リンクした相手を足し、重なりは 1 つにする
    """
    timeline = project.timeline
    members: dict[ClipId, tuple[Track, Clip]] = {}
    for clip_id in clip_ids:
        located = timeline.locate_clip(clip_id)
        if located is None:
            raise KeyError(f"クリップが見つからない: {clip_id}")
        for track_id, member in _linked_group(project, located[1]):
            track = timeline.find_track(track_id)
            if track is not None:
                members.setdefault(member.id, (track, member))
    return list(members.values())


def sped_duration(clip: Clip, speed: Fraction) -> int:
    """中身（素材の範囲）を保ったまま速度を ``speed`` にしたときの長さ（フレーム） 1 以上

    近いフレームへ丸める 切り捨てると 1.5 倍速を何度か掛け直すたびに最後の絵が欠けていく
    """
    return max(1, round(clip.duration * clip.speed / speed))


@dataclass(frozen=True, slots=True)
class ChangeSpeed(Command):
    """クリップの再生速度を変える

    ``keep_length`` が偽なら、素材の同じ範囲を新しい速度で流す長さにする（2 倍速で半分の長さ）
    真なら長さを変えず、流す素材の範囲が変わる（設定パネルの「再生速度」と同じ）

    長くなって同じトラックの後ろのクリップにぶつかるなら断る（:class:`TrimClip` と同じく
    重なる所で止める決まり） 黙って短くすると、頼んだ速度と長さの釣り合いが崩れる
    ロックしたトラックのクリップ（リンクの相手も）と、素材の時間を持たないクリップ（テキスト・
    図形・静止画）も断る ロックした相手を飛ばすと絵と音がずれ、時間を持たないクリップは
    速度を変えても何も変わらない

    中身を保つ変え方では、キーフレームも中身の同じ時刻へ動かす（2 倍速なら頭からのフレームが半分）
    動かさないと、話の要所に付けた寄りや字幕の出が中身からずれ、短くした後ろの点は見えなくなる
    長さを保つ変え方では動かさない（設定パネルと同じ 長さが変わらないので点の位置も変えない）
    """

    clip_ids: tuple[ClipId, ...]
    speed: Fraction
    keep_length: bool = False

    @property
    def label(self) -> str:
        return f"再生速度を変更: {float(self.speed * 100):g}%"

    def apply(self, project: Project) -> Project:
        if self.speed <= 0:
            raise ValueError(f"再生速度は正でなければならない: {self.speed}")
        members = speed_members(project, self.clip_ids)
        for track, clip in members:
            if track.locked:
                raise ValueError(
                    f"{clip.id} のトラック {track.name} はロックされているので速度を変えられない"
                )
            if not _reads_source_time(project, clip):
                raise ValueError(
                    f"{clip.id} は素材の時間を持たない（テキスト・図形・静止画）ので速度が効かない"
                )
        timeline = project.timeline
        changed = False
        for track, clip in members:
            duration = clip.duration if self.keep_length else sped_duration(clip, self.speed)
            if duration == clip.duration and self.speed == clip.speed:
                continue
            current = timeline.find_track(track.id)
            if current is None:  # pragma: no cover - 上で見つけたトラック
                continue
            growth = duration - clip.duration
            if growth > 0:
                room = _room_after(current, clip)
                if room is not None and growth > room:
                    raise ValueError(
                        f"{clip.id} は{STOP_NEXT}にぶつかるので {duration} フレームまで伸ばせない"
                        f" 後ろへ伸ばせるのはあと {room} フレームまで"
                        "（後ろのクリップをずらすか、長さを変えない速度の変え方にする）"
                    )
            sped = replace(clip, speed=self.speed, duration=duration)
            if not self.keep_length:
                sped = _retimed(sped, clip.speed / self.speed)
            others = tuple(c for c in current.clips if c.id != clip.id)
            timeline = timeline.replace_track(current.with_clips((*others, sped)))
            changed = True
        if not changed:
            return project
        return project.with_timeline(timeline)


def _room_after(track: Track, clip: Clip) -> int | None:
    """同じトラックの後ろのクリップの頭までの空き 後ろに何も無ければ ``None``"""
    following = [
        other.timeline_start
        for other in track.clips
        if other.id != clip.id and other.timeline_start >= clip.timeline_end
    ]
    return min(following) - clip.timeline_end if following else None


def _retimed(clip: Clip, ratio: Fraction) -> Clip:
    """クリップの時間で動く値（不透明度・エフェクト・中身・縁取りの層）の点を ``ratio`` 倍の位置へ

    近いフレームへ丸める 速くして同じフレームに重なった点は後ろの点を残す（動き切った値が
    残る 前の点を残すと、寄り切る前の値で止まって見える） 分割（``_split_animation``）と
    同じ所を辿る 1 つでも辿り忘れると、その欄だけ動きが中身からずれる
    """

    def moved(value: AnimatedValue) -> AnimatedValue:
        if not value.keyframes:
            return value
        placed: dict[int, Keyframe] = {}
        for key in value.keyframes:
            placed[round(key.frame * ratio)] = replace(key, frame=round(key.frame * ratio))
        return replace(value, keyframes=tuple(placed[frame] for frame in sorted(placed)))

    def params(values: dict[str, ParamValue]) -> dict[str, ParamValue]:
        return {
            name: moved(value) if isinstance(value, AnimatedValue) else value
            for name, value in values.items()
        }

    def effects(stack: tuple[Effect, ...]) -> tuple[Effect, ...]:
        return tuple(replace(effect, params=params(effect.params)) for effect in stack)

    source = clip.source
    if source is not None:
        source = replace(
            source,
            params=params(source.params),
            strokes=tuple(
                replace(stroke, params=params(stroke.params), effects=effects(stroke.effects))
                for stroke in source.strokes
            ),
        )
    return replace(
        clip,
        opacity=moved(clip.opacity),
        effects=effects(clip.effects),
        after_effects=effects(clip.after_effects),
        source=source,
    )
