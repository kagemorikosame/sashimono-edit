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
from sashimono.core.model import Clip, ClipId, Project, Track

__all__ = ["SPEED_PERCENT_RANGE", "ChangeSpeed", "sped_duration", "speed_members"]

#: 速度の範囲（%） 設定パネルの「再生速度」の欄と同じ 1% より遅いと 1 フレームの絵が
#: 何秒も続き、1000% より速いと音が聞き取れない
SPEED_PERCENT_RANGE = (1, 1000)


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

    キーフレームはクリップの頭からのフレームで持つので、長さが変わっても同じフレームに残る
    （トリムと同じ） 短くした後ろに残った点は見えなくなるだけで消えない
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
