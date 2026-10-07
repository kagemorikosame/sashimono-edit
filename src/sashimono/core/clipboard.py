"""クリップのコピー・切り取り・貼り付け

Qt のクリップボードは使わない 中身は ID とリンクを持ったモデルのままで、文字や
画像にしてから戻すと、リンクや素材の参照を作り直すことになる アプリの中だけで
行き来する

どれもコマンドの列を返すだけで、実行はしない 呼び出し側がチェックポイントで
括れば、貼り付けは何本でも 1 回の Undo で戻る
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from sashimono.core.commands import AddClip, AddTrack, Command, InsertGap, RemoveClip
from sashimono.core.commands.layers import shows_picture_on_layer, solo_for_new_track
from sashimono.core.model import (
    Clip,
    ClipId,
    GroupId,
    Project,
    Track,
    TrackId,
    TrackKind,
    default_track_name,
    new_clip_id,
    new_group_id,
)

__all__ = [
    "ClipboardContent",
    "CopiedClip",
    "copy_clips",
    "cut_commands",
    "insert_paste_commands",
    "paste_commands",
]


@dataclass(frozen=True, slots=True)
class CopiedClip:
    """コピーしたクリップと、元にいたトラック"""

    track_id: TrackId
    kind: TrackKind
    clip: Clip


@dataclass(frozen=True, slots=True)
class ClipboardContent:
    """コピーした中身 開始位置の順に並ぶ"""

    clips: tuple[CopiedClip, ...]

    @property
    def origin(self) -> int:
        """いちばん早い開始位置 貼り付けではここを再生ヘッドに合わせる"""
        return min(copied.clip.timeline_start for copied in self.clips)

    @property
    def length(self) -> int:
        """いちばん早い頭からいちばん遅い尻までの長さ 挿入貼り付けで後ろを押す量

        トラックごとの長さ（映像だけ長い、など）で押さないのは、押す量がトラックで
        違うと、リンクした映像と音声やほかのトラックの字幕が貼った後ろでずれるため
        """
        return max(copied.clip.timeline_end for copied in self.clips) - self.origin


def copy_clips(project: Project, clip_ids: Iterable[ClipId]) -> ClipboardContent:
    """クリップをコピーする リンクした相手（映像なら音声）も一緒に入れる

    相手を置いていくと、貼り付けた映像に音が付いてこない 画面で選んでいるのは
    片方だけでも、編集の単位は組のほう
    """
    found: dict[ClipId, CopiedClip] = {}
    timeline = project.timeline
    for clip_id in clip_ids:
        located = timeline.locate_clip(clip_id)
        if located is None:
            continue
        track, clip = located
        members = (
            list(timeline.linked_clips(clip.link_group))
            if clip.link_group is not None
            else [(track, clip)]
        )
        for member_track, member in members:
            found.setdefault(member.id, CopiedClip(member_track.id, member_track.kind, member))
    ordered = sorted(found.values(), key=lambda copied: copied.clip.timeline_start)
    return ClipboardContent(tuple(ordered))


def cut_commands(project: Project, content: ClipboardContent) -> list[Command]:
    """コピーしたものを消すコマンド 詰めはしない（隙間はそのまま残す）

    リンクした組は :class:`RemoveClip` 1 つで両方消える 組の両方に出すと、
    2 つ目が「見つからない」で失敗する
    """
    commands: list[Command] = []
    groups: set[GroupId] = set()
    for copied in content.clips:
        clip = copied.clip
        if project.timeline.locate_clip(clip.id) is None:
            continue
        if clip.link_group is not None:
            if clip.link_group in groups:
                continue
            groups.add(clip.link_group)
        commands.append(RemoveClip(clip.id))
    return commands


def paste_commands(project: Project, content: ClipboardContent, at_frame: int) -> list[Command]:
    """``at_frame`` を先頭にして貼り付けるコマンド

    並びの間隔はコピーしたときのまま 行き先は元のトラックを優先し、塞がって
    いれば同じ種類の別のトラック、それも無ければ新しいトラックを作る 重ねて置く
    ことはできない（:class:`Track` はクリップの重なりを許さない）

    リンクは新しいグループに付け替える 元のままだと、貼ったものと元のものが
    同じ組になり、片方を動かすともう片方まで動く
    """
    if not content.clips:
        return []
    offset = max(0, at_frame) - content.origin
    groups: dict[GroupId, GroupId] = {}
    #: 束ね（グループ）も新しく付け替える 元のままだと、貼ったものを選ぶと元の
    #: クリップまで一緒に選ばれて動く
    bundles: dict[GroupId, GroupId] = {}
    taken: dict[TrackId, list[tuple[int, int]]] = {}
    commands: list[Command] = []

    for copied in content.clips:
        clip = copied.clip
        if clip.media_id is not None and project.find_media(clip.media_id) is None:
            raise ValueError("コピーした素材がプロジェクトから外されているので貼り付けられない")
        start = clip.timeline_start + offset
        end = start + clip.duration
        track = _landing_track(project, copied, start, end, taken, commands)
        taken.setdefault(track.id, []).append((start, end))
        group = (
            groups.setdefault(clip.link_group, new_group_id())
            if clip.link_group is not None
            else None
        )
        bundle = (
            bundles.setdefault(clip.group_id, new_group_id()) if clip.group_id is not None else None
        )
        pasted = replace(
            clip, id=new_clip_id(), timeline_start=start, link_group=group, group_id=bundle
        )
        commands.append(AddClip(track.id, pasted))
    return commands


def insert_paste_commands(
    project: Project, content: ClipboardContent, at_frame: int, *, all_tracks: bool = True
) -> list[Command]:
    """``at_frame`` から後ろを貼る長さぶん押し出してから貼り付けるコマンド（挿入貼り付け）

    先頭は :class:`InsertGap`（再生ヘッドをまたぐクリップは割ってから押す）、続けて
    空いた所へ :func:`paste_commands` と同じ決まりで置く 呼び出し側が 1 つの
    チェックポイントで括れば、押し出しと貼り付けが 1 回の取り消しで戻る

    ``all_tracks`` が真なら全トラックとマーカーを押す（Premiere Pro の既定 全トラックの
    同期ロックが入っている） 偽なら、コピー元のトラックと、そこで押されるクリップの
    リンクの相手・グループの仲間・焼き込んだ字幕のトラックだけを押す

    押す中身のあるトラックがロックされていれば ``ValueError`` で断る 何も実行しないので、
    タイムラインは元のまま
    """
    if not content.clips:
        return []
    at = max(0, at_frame)
    targets = None if all_tracks else _insert_targets(project, content)
    gap = InsertGap(at, content.length, targets)
    # 貼る先は押し出した後の姿で決める 押す前の姿で決めると、元のトラックの再生ヘッドの
    # 後ろが塞がって見え、空けた所ではなく新しいトラックへ置かれる
    # 押し出しは純関数で、実行したときも同じ姿になる（割った後ろ半分の ID だけは変わるが、
    # 貼り付けのコマンドはそれを指さない）
    return [gap, *paste_commands(gap.apply(project), content, at)]


def _insert_targets(project: Project, content: ClipboardContent) -> tuple[TrackId, ...]:
    """挿入貼り付けで押すトラック 今のタイムラインで実際に貼る先になるトラック

    コピー元のトラックが今のタイムラインにあればそれ 無ければ（別のシーンでコピーした物）
    :func:`_landing_track` が次に選ぶ、同じ種類のロックしていないトラックを並びの順に
    割り当てる（コピー元のトラックごとに別の 1 本） コピー元の ID のまま渡すと、
    :class:`InsertGap` が何も押さず、普通の貼り付けと同じく別のトラックへ逃げる
    """
    tracks = project.timeline.tracks
    known = {track.id for track in tracks}
    chosen: dict[TrackId, TrackId] = {}
    used = {copied.track_id for copied in content.clips if copied.track_id in known}
    for copied in content.clips:
        if copied.track_id in chosen:
            continue
        if copied.track_id in known:
            chosen[copied.track_id] = copied.track_id
            continue
        spare = [t.id for t in tracks if t.kind is copied.kind and not t.locked]
        # 同じ種類が足りなければ、ほかのコピー元と同じトラックを使う（:func:`_landing_track` も
        # 重ならなければ同じトラックへ置く） 1 本も無ければ貼り付けが作るので押す物は無い
        pick = next((t for t in spare if t not in used), spare[0] if spare else None)
        if pick is not None:
            chosen[copied.track_id] = pick
            used.add(pick)
    return tuple(dict.fromkeys(chosen.values()))


def _landing_track(
    project: Project,
    copied: CopiedClip,
    start: int,
    end: int,
    taken: dict[TrackId, list[tuple[int, int]]],
    commands: list[Command],
) -> Track:
    """``[start, end)`` が空いている行き先 この貼り付けで先に置いた分も数える"""
    created = [c.track for c in commands if isinstance(c, AddTrack)]
    same_kind = [t for t in (*project.timeline.tracks, *created) if t.kind is copied.kind]
    ordered = sorted(same_kind, key=lambda track: track.id != copied.track_id)

    for track in ordered:
        if track.locked:
            continue
        busy = any(clip.overlaps(start, end) for clip in track.clips) or any(
            s < end and start < e for s, e in taken.get(track.id, [])
        )
        if not busy:
            return track

    names = {t.name for t in (*project.timeline.tracks, *created)}
    name = default_track_name(copied.kind, len(same_kind) + 1, names)
    # 作るのが元と同じ種類なのは方式を問わない レイヤーのクリップ（絵と音の 1 本）を
    # 映像トラックへ貼ると音が消え、映像と音声の組をレイヤーへ貼ると絵が 2 枚重なる
    # ソロで絞っている間は、作ったトラックにもソロを付ける 付けないと貼った物が出ない
    # レイヤーは、貼る物が絵を描くなら絵の側・音だけなら音の側のソロを見る
    solo = solo_for_new_track(
        project, copied.kind, picture=shows_picture_on_layer(project, copied.clip)
    )
    track = Track(kind=copied.kind, name=name, solo=solo)
    commands.append(AddTrack(track))
    return track
