"""基本的な編集コマンド

どれも純関数で、失敗するときは例外を投げる トラック内でクリップが重ならないことなどの
不変条件は :class:`~sashimono.core.model.Track` 側で検査されるので、ここで作った
おかしな状態はモデル構築時点で弾かれる
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sashimono.core.commands.base import Command
from sashimono.core.model import (
    AnimatedValue,
    Blending,
    Clip,
    ClipId,
    Effect,
    GroupId,
    LayerMode,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    Timeline,
    Track,
    TrackId,
    TrackKind,
    Transcript,
    default_track_name,
    new_clip_id,
    new_group_id,
)
from sashimono.core.timebase import FrameRate

__all__ = [
    "AddClip",
    "AddMedia",
    "AddTrack",
    "InsertGap",
    "MoveClip",
    "MoveClips",
    "RemoveClip",
    "RemoveClips",
    "RemoveMedia",
    "RemoveTrack",
    "RenameProject",
    "RenameTrack",
    "RippleCut",
    "SetBlending",
    "SetLayerMode",
    "SetResolution",
    "SetTrackHeights",
    "SetTrackState",
    "SetTranscript",
    "SplitClip",
    "TrimClip",
    "shifted_track",
]


@dataclass(frozen=True, slots=True)
class AddMedia(Command):
    """メディアプールに素材を追加する"""

    item: MediaItem

    @property
    def label(self) -> str:
        return f"素材を追加: {self.item.name}"

    def apply(self, project: Project) -> Project:
        if project.find_media(self.item.id) is not None:
            raise ValueError(f"すでに登録されている素材: {self.item.id}")
        return project.with_media((*project.media, self.item))


@dataclass(frozen=True, slots=True)
class RemoveMedia(Command):
    """素材をメディアプールから外す

    その素材を使っているクリップが 1 つでもあれば失敗する 参照だけ残して
    素材を消すと、あとから原因の分からない再生エラーになるため
    """

    media_id: MediaId

    @property
    def label(self) -> str:
        return "素材を削除"

    def apply(self, project: Project) -> Project:
        item = project.require_media(self.media_id)
        # シーンの中で実行されても、メインとほかのシーンの参照まで数える
        # 開いているタイムラインだけを見ると、別の場所のクリップが消えた素材を指す
        timelines = [project.timeline, *(scene.timeline for scene in project.scenes)]
        in_use = [
            clip.id
            for timeline in timelines
            for track in timeline.tracks
            for clip in track.clips
            if clip.media_id == self.media_id
        ]
        if in_use:
            raise ValueError(f"素材 {item.name!r} はタイムラインで {len(in_use)} 箇所使われている")
        remaining = tuple(m for m in project.media if m.id != self.media_id)
        return project.with_media(remaining)


@dataclass(frozen=True, slots=True)
class SetTranscript(Command):
    """素材の字幕起こし結果を差し替える

    字幕はトラックではなく素材に紐付くので、この 1 操作でその素材を使っている
    すべての箇所の字幕が同時に変わる
    """

    media_id: MediaId
    transcript: Transcript | None
    #: 音声ストリームの番号 ``None`` なら 1 本目 ほかの音の字幕はそのまま残す
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕を更新" if self.transcript is not None else "字幕を削除"

    def apply(self, project: Project) -> Project:
        item = project.require_media(self.media_id)
        return project.replace_media(item.with_transcript(self.transcript, self.stream))


@dataclass(frozen=True, slots=True)
class AddTrack(Command):
    """トラックを追加する ``index`` が ``None`` なら末尾"""

    track: Track
    index: int | None = None

    @property
    def label(self) -> str:
        return f"トラックを追加: {self.track.name or self.track.kind.value}"

    def apply(self, project: Project) -> Project:
        timeline = project.timeline
        if timeline.find_track(self.track.id) is not None:
            raise ValueError(f"すでに存在するトラック: {self.track.id}")
        tracks = list(timeline.tracks)
        tracks.insert(len(tracks) if self.index is None else self.index, self.track)
        return project.with_timeline(replace(timeline, tracks=tuple(tracks)))


@dataclass(frozen=True, slots=True)
class RemoveTrack(Command):
    """トラックを、載っているクリップごと削除する"""

    track_id: TrackId

    @property
    def label(self) -> str:
        return "トラックを削除"

    def apply(self, project: Project) -> Project:
        timeline = project.timeline
        if timeline.find_track(self.track_id) is None:
            raise KeyError(f"トラックが見つからない: {self.track_id}")
        tracks = tuple(t for t in timeline.tracks if t.id != self.track_id)
        return project.with_timeline(replace(timeline, tracks=tracks))


@dataclass(frozen=True, slots=True)
class AddClip(Command):
    """トラックにクリップを置く 既存のクリップと重なる場合は失敗する"""

    track_id: TrackId
    clip: Clip

    @property
    def label(self) -> str:
        return "クリップを追加"

    def apply(self, project: Project) -> Project:
        timeline = project.timeline
        track = _require_track(project, self.track_id)
        if track.locked:
            raise ValueError(f"トラック {track.name!r} はロックされている")
        _validate_clip_media(project, track, self.clip)
        updated = track.with_clips((*track.clips, self.clip))
        return project.with_timeline(timeline.replace_track(updated))


@dataclass(frozen=True, slots=True)
class RemoveClip(Command):
    """クリップを削除する

    ``ripple`` が真なら、同じトラックの後続クリップを詰める リンクされた
    映像・音声も同時に削除される
    """

    clip_id: ClipId
    ripple: bool = False

    @property
    def label(self) -> str:
        return "クリップを削除（詰める）" if self.ripple else "クリップを削除"

    def apply(self, project: Project) -> Project:
        located = project.timeline.locate_clip(self.clip_id)
        if located is None:
            raise KeyError(f"クリップが見つからない: {self.clip_id}")
        _, clip = located

        targets = _linked_group(project, clip)
        # 消す前に組の全員のトラックを見る 移動とトリムはロックを見ていたのに、
        # 削除だけ素通しで、ロックしたトラックのクリップも消えていた
        # 片方だけ消すと映像と音声の組が壊れるので、1 本でもロックなら丸ごと止める
        for track_id, _ in targets:
            locked = _require_track(project, track_id)
            if locked.locked:
                raise ValueError(f"トラック {locked.name!r} はロックされている")
        timeline = project.timeline
        for track_id, target in targets:
            track = _require_track(project, track_id)
            remaining = [c for c in track.clips if c.id != target.id]
            if self.ripple:
                remaining = [
                    c.moved_to(c.timeline_start - target.duration)
                    if c.timeline_start >= target.timeline_end
                    else c
                    for c in remaining
                ]
            timeline = timeline.replace_track(track.with_clips(tuple(remaining)))
            project = project.with_timeline(timeline)
        return project


@dataclass(frozen=True, slots=True)
class MoveClip(Command):
    """クリップを別の位置、必要なら別のトラックへ動かす

    リンクされた映像・音声は同じだけ動く トラックの移動は掴んだクリップだけで、
    相手は自分のトラックに残る（音声が映像トラックへ飛んでは困る）
    """

    clip_id: ClipId
    timeline_start: int
    track_id: TrackId | None = None

    @property
    def label(self) -> str:
        return "クリップを移動"

    def apply(self, project: Project) -> Project:
        located = project.timeline.locate_clip(self.clip_id)
        if located is None:
            raise KeyError(f"クリップが見つからない: {self.clip_id}")
        source_track, clip = located
        if self.timeline_start < 0:
            raise ValueError(f"開始位置が負: {self.timeline_start}")

        target_track = (
            source_track if self.track_id is None else _require_track(project, self.track_id)
        )
        if source_track.locked or target_track.locked:
            raise ValueError("ロックされたトラックのクリップは動かせない")
        carried = _carried_across(project, source_track, target_track, clip)
        _validate_clip_media(project, target_track, carried)

        delta = self.timeline_start - clip.timeline_start
        timeline = project.timeline
        without = tuple(c for c in source_track.clips if c.id != clip.id)
        timeline = timeline.replace_track(source_track.with_clips(without))

        # 同一トラック内の移動では、直前の replace_track で反映済みのトラックを
        # 取り直さないと、取り除いたはずのクリップが復活する
        destination = timeline.find_track(target_track.id)
        if destination is None:
            raise KeyError(f"トラックが見つからない: {target_track.id}")
        moved = carried.moved_to(self.timeline_start)
        timeline = timeline.replace_track(destination.with_clips((*destination.clips, moved)))

        for track_id, partner in _linked_group(project, clip):
            if partner.id == clip.id:
                continue
            track = timeline.find_track(track_id)
            if track is None or track.locked:
                continue
            start = partner.timeline_start + delta
            if start < 0:
                raise ValueError("リンクされたクリップがタイムラインの先頭より前へ出る")
            others = tuple(c for c in track.clips if c.id != partner.id)
            timeline = timeline.replace_track(track.with_clips((*others, partner.moved_to(start))))
        return project.with_timeline(timeline)


@dataclass(frozen=True, slots=True)
class MoveClips(Command):
    """選んだクリップをまとめて ``delta`` フレームずらす トラックは変えない

    :class:`MoveClip` を並べると、動かす途中で前のクリップが後ろのクリップの元の
    場所へ入り、まだ動いていない相手と重なって失敗する（全体としては重ならない
    動かし方でも） 全員をいったん外してから置き直す

    リンクした相手も同じだけ動く 動く全員（選んだものと相手）のうち 1 本でも
    ロックしたトラックにいれば、何も動かさない :class:`MoveClip` は相手を残して
    動くが、まとめて動かすときに同じことをすると、選んだ中のどれかの映像と音声が
    黙ってずれる 何本も動かすと、どれがずれたのかを見つけにくい
    """

    clip_ids: tuple[ClipId, ...]
    delta: int
    #: 動かすトラックの本数（同じ種類のトラックの並びで数える） 正で下へ
    track_delta: int = 0

    @property
    def label(self) -> str:
        return f"{len(self.clip_ids)} 本を移動"

    def apply(self, project: Project) -> Project:
        timeline = project.timeline
        targets: dict[ClipId, tuple[TrackId, Clip]] = {}
        for clip_id in self.clip_ids:
            located = timeline.locate_clip(clip_id)
            if located is None:
                raise KeyError(f"クリップが見つからない: {clip_id}")
            track, clip = located
            if track.locked:
                raise ValueError(f"トラック {track.name!r} はロックされている")
            for track_id, member in _linked_group(project, clip):
                partner_track = _require_track(project, track_id)
                if partner_track.locked:
                    raise ValueError(
                        f"リンクした相手のトラック {partner_track.name!r} がロックされている"
                    )
                targets.setdefault(member.id, (track_id, member))
        if (self.delta == 0 and self.track_delta == 0) or not targets:
            return project
        if any(clip.timeline_start + self.delta < 0 for _, clip in targets.values()):
            raise ValueError("タイムラインの先頭より前へは動かせない")

        by_track: dict[TrackId, list[Clip]] = {}
        for track_id, clip in targets.values():
            by_track.setdefault(track_id, []).append(clip)

        # まず全員を元のトラックから外す 先に置き直すと、行き先のトラックにまだ残って
        # いる元のクリップと重なって失敗する
        arrivals: dict[TrackId, list[Clip]] = {}
        for track_id, moving in by_track.items():
            track = _require_track(project, track_id)
            leaving = {clip.id for clip in moving}
            staying = tuple(c for c in track.clips if c.id not in leaving)
            timeline = timeline.replace_track(track.with_clips(staying))
            destination = shifted_track(project, track_id, self.track_delta)
            arrivals.setdefault(destination, []).extend(
                c.moved_to(c.timeline_start + self.delta) for c in moving
            )
        for track_id, coming in arrivals.items():
            arrival = timeline.find_track(track_id)
            if arrival is None:
                raise KeyError(f"トラックが見つからない: {track_id}")
            if arrival.locked:
                raise ValueError(f"トラック {arrival.name!r} はロックされている")
            timeline = timeline.replace_track(arrival.with_clips((*arrival.clips, *coming)))
        return project.with_timeline(timeline)


def shifted_track(project: Project, track_id: TrackId, delta: int) -> TrackId:
    """同じ種類のトラックの並びで ``delta`` 本ずらした先 端を越えれば断る

    :class:`MoveClips` の行き先 タイムラインの画面もドラッグ中の枠をこれで求める
    別々に数えると、枠を出した所と離したときに入る所が食い違う
    """
    if delta == 0:
        return track_id
    track = _require_track(project, track_id)
    same = [t for t in project.timeline.tracks if t.kind is track.kind]
    index = next(i for i, t in enumerate(same) if t.id == track_id) + delta
    if not 0 <= index < len(same):
        raise ValueError("トラックの並びの外へは動かせない")
    return same[index].id


@dataclass(frozen=True, slots=True)
class TrimClips(Command):
    """選んだクリップの端をまとめて動かす

    1 本ずつ :class:`TrimClip` を当てる 途中で失敗すれば、コマンドごと失敗するので
    タイムラインは元のまま残る（履歴も 1 段）
    """

    clip_ids: tuple[ClipId, ...]
    head_delta: int = 0
    tail_delta: int = 0

    @property
    def label(self) -> str:
        return f"{len(self.clip_ids)} 本をトリム"

    def apply(self, project: Project) -> Project:
        if self.head_delta == 0 and self.tail_delta == 0:
            return project
        done: set[ClipId] = set()
        for clip_id in self.clip_ids:
            located = project.timeline.locate_clip(clip_id)
            if located is None:
                raise KeyError(f"クリップが見つからない: {clip_id}")
            track, clip = located
            if track.locked:
                raise ValueError(f"トラック {track.name!r} はロックされている")
            if clip_id in done:
                continue
            # リンクした映像と音声は 1 回で両方が削れる 2 回当てると相手だけ余分に縮む
            done.update(member.id for _, member in _linked_group(project, clip))
            project = TrimClip(
                clip_id, head_delta=self.head_delta, tail_delta=self.tail_delta
            ).apply(project)
        return project


@dataclass(frozen=True, slots=True)
class RemoveClips(Command):
    """選んだクリップをまとめて消す ``ripple`` なら消したぶんを詰める

    後ろのクリップから順に消す 前から消して詰めると、後ろのクリップの位置が
    ずれ、詰める量の計算がずれる リンクした組は 1 回で両方消えるので、2 本目に
    来たら飛ばす
    """

    clip_ids: tuple[ClipId, ...]
    ripple: bool = False

    @property
    def label(self) -> str:
        suffix = "（詰める）" if self.ripple else ""
        return f"{len(self.clip_ids)} 本を削除{suffix}"

    def apply(self, project: Project) -> Project:
        located = [project.timeline.locate_clip(clip_id) for clip_id in self.clip_ids]
        if any(entry is None for entry in located):
            raise KeyError("消すクリップの一部が見つからない")
        ordered = sorted(
            (entry[1] for entry in located if entry is not None),
            key=lambda clip: clip.timeline_start,
            reverse=True,
        )
        for clip in ordered:
            if project.timeline.locate_clip(clip.id) is not None:
                project = RemoveClip(clip.id, ripple=self.ripple).apply(project)
        return project


@dataclass(frozen=True, slots=True)
class SplitClip(Command):
    """クリップを ``frame`` の位置で 2 つに割る

    左側は元の ID を保ち、右側が新しい ID を得る リンクされた映像・音声も
    同じ位置で割られるので、片方だけずれることはない
    """

    clip_id: ClipId
    frame: int
    #: 右側（後ろ）の片割れを移すグループ（元のグループ, 新しいグループ）の組
    #: グループの何本かをまとめて割るとき、呼ぶ側が 1 つの新しいグループを決めて、同じ組を
    #: 全部の割る命令へ渡す 渡さなければ右側も元のグループに残る（1 本だけ割ったとき）
    #: 右側を元のグループに残すと、割った後もグループ全体が 1 つのままで、前と後ろを
    #: 別々に選べない（利用者の報告）
    new_groups: tuple[tuple[GroupId, GroupId], ...] = ()

    @property
    def label(self) -> str:
        return "クリップを分割"

    def apply(self, project: Project) -> Project:
        located = project.timeline.locate_clip(self.clip_id)
        if located is None:
            raise KeyError(f"クリップが見つからない: {self.clip_id}")
        _, clip = located
        if not (clip.timeline_start < self.frame < clip.timeline_end):
            raise ValueError(
                f"分割位置がクリップの内側にない: {self.frame} は "
                f"[{clip.timeline_start}, {clip.timeline_end}) の外"
            )

        rate = project.rate
        timeline = project.timeline
        # 右側は新しいリンクグループにする 元のままだと、分割してできた左右が
        # 同じグループに残り、片方を削除するともう片方まで消える 新しいグループを
        # 映像・音声の右側どうしで共有するので、分割後もリンクは保たれる
        right_group = new_group_id() if clip.link_group is not None else None

        for track_id, target in _linked_group(project, clip):
            if not target.contains(self.frame):
                # リンク先の長さが違う場合 片方だけ割ると同期が崩れるので何もしない
                continue
            track = timeline.find_track(track_id)
            if track is None:
                continue
            left, right = _halves(
                target,
                self.frame,
                rate,
                link_group=right_group,
                group_id=dict(self.new_groups).get(target.group_id, target.group_id)
                if target.group_id is not None
                else None,
            )
            others = tuple(c for c in track.clips if c.id != target.id)
            timeline = timeline.replace_track(track.with_clips((*others, left, right)))
        return project.with_timeline(timeline)


def _halves(
    target: Clip,
    frame: int,
    rate: FrameRate,
    *,
    link_group: GroupId | None,
    group_id: GroupId | None,
) -> tuple[Clip, Clip]:
    """``target`` を ``frame`` で割った前後 後ろは新しい ID で、渡したリンクとグループに入る

    分割（:class:`SplitClip`）と挿入（:class:`InsertGap`）が同じ割り方をする 別々に書くと、
    片方だけキーフレームや素材の読み始めの数え直しを忘れる
    """
    left_duration = frame - target.timeline_start
    # キーフレームはクリップの頭から数える 後ろは割った所を頭にして数え直す
    # （:meth:`AnimatedValue.split_at`） そのまま写すと、後ろのクリップにも前と同じ
    # 位置付近に同じキーが入り、割っただけで動きが変わる
    before, after = _split_animation(target, left_duration)
    left = replace(before, duration=left_duration)
    right = replace(
        after,
        id=new_clip_id(),
        timeline_start=frame,
        duration=target.timeline_end - frame,
        # 右側は、左側が消費したソース時間の分だけ後ろから始まる
        source_in=target.source_in + left_duration * rate.frame_duration * target.speed,
        link_group=link_group,
        group_id=group_id,
    )
    return left, right


@dataclass(frozen=True, slots=True)
class InsertGap(Command):
    """``frame`` から後ろを ``length`` フレーム後ろへずらし、間を空ける 挿入貼り付けの前半

    ``frame`` をまたぐクリップは、そこで割って後ろ半分だけをずらす（Premiere Pro の挿入と
    同じ） 割らずに丸ごと押すと、再生ヘッドより前に見えていた絵まで後ろへ逃げる
    ちょうど ``frame`` から始まるクリップもずらす（貼った物の後ろに来る）

    ``track_ids`` が ``None`` なら全トラックとマーカーをずらし、書き出し範囲も合わせる
    （Premiere の同期ロックを全部入れたときと同じ :func:`_pushed_area`） トラックを渡したときは
    範囲を動かさない（ほかのトラックの中身が動かないので）
    トラックを渡せば、そのトラックに加えて、そこでずれるクリップの
    リンクの相手・グループの仲間・焼き込んだ字幕がいるトラックもずらす 相手を置いていくと、
    映像と音声の組や、話している所と字幕が貼った長さぶんずれる

    ずらす中身のある（``frame`` より後ろに掛かるクリップを持つ）トラックが 1 本でも
    ロックされていれば何もせずに止める 飛ばすと、そのトラックだけ後ろが残り、
    以降の同期がすべて崩れる（:class:`RemoveClip` と同じ決まり）
    """

    frame: int
    length: int
    track_ids: tuple[TrackId, ...] | None = None

    @property
    def label(self) -> str:
        return "間を空ける"

    def apply(self, project: Project) -> Project:
        if self.frame < 0:
            raise ValueError(f"間を空ける位置が負: {self.frame}")
        if self.length <= 0:
            return project
        frame = self.frame
        timeline = project.timeline
        shifted = self._shifted_tracks(timeline)
        moving = [t for t in timeline.tracks if t.id in shifted]
        for track in moving:
            if track.locked and any(clip.timeline_end > frame for clip in track.clips):
                raise ValueError(
                    f"トラック {track.name!r} はロックされているので、後ろをずらせない"
                )

        # 割った後ろ半分と、丸ごと後ろにいる仲間を、新しいリンクとグループへまとめる
        # 前のままだと、前に残った半分と後ろへ行った半分が同じ組で、片方を動かすと
        # 間を空けた所を越えてもう片方まで動く
        straddling = [
            clip
            for track in moving
            for clip in track.clips
            if clip.timeline_start < frame < clip.timeline_end
        ]
        links = {
            clip.link_group: new_group_id() for clip in straddling if clip.link_group is not None
        }
        behind: dict[GroupId, int] = {}
        for track in moving:
            for clip in track.clips:
                if clip.group_id is not None and clip.timeline_end > frame:
                    behind[clip.group_id] = behind.get(clip.group_id, 0) + 1
        # 後ろ側が 1 本だけなら元のグループに残す（:class:`SplitClip` を 1 本だけ割ったときと同じ）
        # 1 本だけのグループを作っても束ねる意味が無い
        groups = {
            clip.group_id: new_group_id()
            for clip in straddling
            if clip.group_id is not None and behind[clip.group_id] >= 2
        }

        def after(clip: Clip, start: int) -> Clip:
            return replace(
                clip,
                timeline_start=start,
                link_group=links.get(clip.link_group, clip.link_group)
                if clip.link_group is not None
                else None,
                group_id=groups.get(clip.group_id, clip.group_id)
                if clip.group_id is not None
                else None,
            )

        for track in moving:
            if track.locked:
                continue
            pieces: list[Clip] = []
            for clip in track.clips:
                if clip.timeline_end <= frame:
                    pieces.append(clip)
                elif clip.timeline_start >= frame:
                    pieces.append(after(clip, clip.timeline_start + self.length))
                else:
                    left, right = _halves(clip, frame, project.rate, link_group=None, group_id=None)
                    pieces.append(left)
                    pieces.append(
                        after(
                            replace(right, link_group=clip.link_group, group_id=clip.group_id),
                            frame + self.length,
                        )
                    )
            timeline = timeline.replace_track(track.with_clips(tuple(pieces)))

        if self.track_ids is None:
            # 目印も中身と一緒にずらす 残すと、話の区切りに打った印が貼った物の上に来る
            # （ジェットカット :class:`RippleCut` が詰めるときと同じ扱い）
            markers = tuple(
                replace(marker, frame=marker.frame + self.length)
                if marker.frame >= frame
                else marker
                for marker in timeline.markers
            )
            timeline = replace(
                timeline,
                markers=markers,
                work_area=_pushed_area(timeline.work_area, frame, self.length),
            )
        return project.with_timeline(timeline)

    def _shifted_tracks(self, timeline: Timeline) -> set[TrackId]:
        """ずらすトラック 渡したトラックから、後ろでずれるクリップの相手をたどって広げる"""
        if self.track_ids is None:
            return {track.id for track in timeline.tracks}
        known = {track.id for track in timeline.tracks}
        shifted = {track_id for track_id in self.track_ids if track_id in known}
        frontier = list(shifted)
        while frontier:
            track = timeline.find_track(frontier.pop())
            if track is None:
                continue
            for clip in track.clips:
                if clip.timeline_end <= self.frame:
                    continue
                for partner_track, partner in _companions(timeline, clip):
                    if partner.timeline_end > self.frame and partner_track.id not in shifted:
                        shifted.add(partner_track.id)
                        frontier.append(partner_track.id)
        return shifted


def _pushed_area(area: tuple[int, int] | None, frame: int, length: int) -> tuple[int, int] | None:
    """全トラックを ``frame`` から ``length`` 押したときの書き出し範囲

    範囲の前（頭ちょうども含む クリップやマーカーと同じく ``frame`` から後ろは押す）なら
    両端を押し、途中なら終わりだけを延ばす（Premiere Pro の挿入と同じ） 範囲を残すと、
    中身だけが後ろへ移り、書き出しの頭に意図しない部分が入って末尾が欠ける

    ジェットカット（:class:`RippleCut`）は範囲を外すが、こちらは外さない 挿入は押した後の
    位置がどこへ移ったかが 1 通りに決まり、外すと決めた範囲を黙って捨てることになる
    """
    if area is None:
        return None
    start, end = area
    if frame <= start:
        return start + length, end + length
    if frame < end:
        return start, end + length
    return area


def _companions(timeline: Timeline, clip: Clip) -> list[tuple[Track, Clip]]:
    """``clip`` と一緒にずれないと困る物 リンクの相手・グループの仲間・焼き込んだ字幕"""
    found: list[tuple[Track, Clip]] = []
    if clip.link_group is not None:
        found.extend(timeline.linked_clips(clip.link_group))
    if clip.group_id is not None:
        found.extend(timeline.grouped_clips(clip.group_id))
    if clip.media_id is not None:
        found.extend(
            (track, other)
            for track in timeline.tracks
            for other in track.clips
            if other.subtitle_origin is not None and other.subtitle_origin.media_id == clip.media_id
        )
    return found


def _split_animation(clip: Clip, cut: int) -> tuple[Clip, Clip]:
    """``clip`` の時間で動く値（不透明度・エフェクト・中身）を、頭から ``cut`` の所で分けた前後

    長さと位置は変えない（割る側が決める） キーの無い値はそのまま両方へ渡す
    """

    def pair(value: ParamValue) -> tuple[ParamValue, ParamValue]:
        if isinstance(value, AnimatedValue) and value.keyframes:
            return value.split_at(cut)
        return value, value

    def params(
        values: dict[str, ParamValue],
    ) -> tuple[dict[str, ParamValue], dict[str, ParamValue]]:
        split = {name: pair(value) for name, value in values.items()}
        return (
            {name: halves[0] for name, halves in split.items()},
            {name: halves[1] for name, halves in split.items()},
        )

    def effects(stack: tuple[Effect, ...]) -> tuple[tuple[Effect, ...], tuple[Effect, ...]]:
        halves = [params(effect.params) for effect in stack]
        return (
            tuple(replace(e, params=h[0]) for e, h in zip(stack, halves, strict=True)),
            tuple(replace(e, params=h[1]) for e, h in zip(stack, halves, strict=True)),
        )

    opacity_before, opacity_after = clip.opacity.split_at(cut)
    effects_before, effects_after = effects(clip.effects)
    after_before, after_after = effects(clip.after_effects)
    source_before = source_after = clip.source
    if clip.source is not None:
        params_before, params_after = params(clip.source.params)
        source_before = replace(clip.source, params=params_before)
        source_after = replace(clip.source, params=params_after)
    return (
        replace(
            clip,
            opacity=opacity_before,
            effects=effects_before,
            after_effects=after_before,
            source=source_before,
        ),
        replace(
            clip,
            opacity=opacity_after,
            effects=effects_after,
            after_effects=after_after,
            source=source_after,
        ),
    )


@dataclass(frozen=True, slots=True)
class TrimClip(Command):
    """クリップの端を動かす

    ``head`` を動かすとソース範囲の開始位置も一緒にずれる（素材の中身は動かない）
    ``tail`` は長さだけを変える
    """

    clip_id: ClipId
    #: 先頭を動かす量（フレーム） 正で短く、負で長くなる
    head_delta: int = 0
    #: 末尾を動かす量（フレーム） 正で長く、負で短くなる
    tail_delta: int = 0

    @property
    def label(self) -> str:
        return "クリップをトリム"

    def apply(self, project: Project) -> Project:
        located = project.timeline.locate_clip(self.clip_id)
        if located is None:
            raise KeyError(f"クリップが見つからない: {self.clip_id}")
        _, clip = located

        timeline = project.timeline
        # リンクされた映像・音声は同じだけ削る 片方だけ縮めると音がずれる
        for track_id, target in _linked_group(project, clip):
            track = timeline.find_track(track_id)
            if track is None or track.locked:
                continue
            others = tuple(c for c in track.clips if c.id != target.id)
            trimmed = _trimmed(project, target, self.head_delta, self.tail_delta)
            timeline = timeline.replace_track(track.with_clips((*others, trimmed)))
        return project.with_timeline(timeline)


@dataclass(frozen=True, slots=True)
class SetTrackState(Command):
    """トラックのミュート・ソロ・ロックを切り替える ``None`` の項目は触らない

    ロック中のトラックでも切り替えられる ロックはクリップを守るためのもので、
    聞こえ方まで固めると「ロックしたら消音できない」になる
    """

    track_id: TrackId
    muted: bool | None = None
    solo: bool | None = None
    locked: bool | None = None

    @property
    def label(self) -> str:
        names = [
            (on if value else off)
            for value, on, off in (
                (self.muted, "ミュート", "ミュートを解除"),
                (self.solo, "ソロ", "ソロを解除"),
                (self.locked, "ロック", "ロックを解除"),
            )
            if value is not None
        ]
        return "、".join(names) or "トラックの状態を変更"

    def apply(self, project: Project) -> Project:
        track = _require_track(project, self.track_id)
        if self.muted is None and self.solo is None and self.locked is None:
            return project
        updated = replace(
            track,
            muted=track.muted if self.muted is None else self.muted,
            solo=track.solo if self.solo is None else self.solo,
            locked=track.locked if self.locked is None else self.locked,
        )
        return project.with_timeline(project.timeline.replace_track(updated))


@dataclass(frozen=True, slots=True)
class RenameTrack(Command):
    """トラック（レイヤー）の名前を変える

    前後の空白は落とす 空にしたら、その位置の既定の名前（``V2`` ``レイヤー 3`` など
    :func:`~sashimono.core.model.default_track_name`）へ戻す 空の名前を残すと、見出しに
    種類の名前しか出ず、どのレイヤーなのかを並びの位置で数えることになる
    ほかのトラックと同じ名前も断らない（本人が付けた名前で、並びが違えば見分けられる）
    既定へ戻すときだけは、ほかのトラックの名前と重ならない番号まで進める
    """

    track_id: TrackId
    name: str

    @property
    def label(self) -> str:
        return "トラックの名前を変更"

    def apply(self, project: Project) -> Project:
        track = _require_track(project, self.track_id)
        name = self.name.strip()
        if not name:
            timeline = project.timeline
            same = [t.id for t in timeline.tracks if t.kind is track.kind]
            others = {t.name for t in timeline.tracks if t.id != track.id}
            name = default_track_name(track.kind, same.index(track.id) + 1, others)
        if name == track.name:
            return project
        return project.with_timeline(project.timeline.replace_track(replace(track, name=name)))


#: トラックの高さ（画素） 下はトラック名とボタンが 1 行で収まる高さ、上は
#: 1 本で画面を占領しない程度 既定は :class:`Track` の既定と同じ
MIN_TRACK_HEIGHT = 28
MAX_TRACK_HEIGHT = 240
DEFAULT_TRACK_HEIGHT = 60


@dataclass(frozen=True, slots=True)
class SetTrackHeights(Command):
    """トラックの高さを変える 範囲の外は端へ寄せる

    1 本でも全部でも同じコマンドで扱う 全トラックをまとめて変えたときに、
    トラックの数だけ取り消し段ができると戻すのが大変になる
    """

    heights: tuple[tuple[TrackId, int], ...]

    @property
    def label(self) -> str:
        return "トラックの高さを変更"

    def apply(self, project: Project) -> Project:
        timeline = project.timeline
        for track_id, height in self.heights:
            track = _require_track(project, track_id)
            clamped = min(max(height, MIN_TRACK_HEIGHT), MAX_TRACK_HEIGHT)
            if clamped != track.height:
                timeline = timeline.replace_track(replace(track, height=clamped))
                project = project.with_timeline(timeline)
        return project


#: 解像度として受け付ける範囲（画素）
#: 下は縮小プレビューが潰れない程度、上は 8K まで GPU のテクスチャ上限もこのあたり
MIN_RESOLUTION = 16
MAX_RESOLUTION = 8192


@dataclass(frozen=True, slots=True)
class SetResolution(Command):
    """出力の解像度を変える

    クリップは動かさない 位置は画面中央からの画素数で持っているので、中央に
    置いたものは中央のまま残る 端に寄せたものは、広げれば内側へ、縮めれば外へ出る

    縦横とも偶数に限る 書き出しの yuv420p は色を 2x2 画素ごとに持つので、奇数だと
    エンコーダが断る 書き出しの最後で分かっても遅いので、決める時点で止める
    """

    width: int
    height: int

    @property
    def label(self) -> str:
        return f"解像度を変更: {self.width}x{self.height}"

    def apply(self, project: Project) -> Project:
        for name, value in (("横", self.width), ("縦", self.height)):
            if not MIN_RESOLUTION <= value <= MAX_RESOLUTION:
                raise ValueError(
                    f"{name}の画素数は {MIN_RESOLUTION}〜{MAX_RESOLUTION} にしてください: {value}"
                )
            if value % 2:
                raise ValueError(f"{name}の画素数は偶数にしてください（書き出せないため）: {value}")
        settings = replace(project.settings, width=self.width, height=self.height)
        return replace(project, settings=settings)


@dataclass(frozen=True, slots=True)
class SetBlending(Command):
    """半透明の重ね合わせの方法（:class:`~sashimono.core.model.Blending`）を変える

    プロジェクトの設定として持つ 作品の見た目そのものなので、同じ作品を別の機械で
    開いても同じ絵にならなければならない（本人の好み ``Preferences`` とは混ぜない）
    """

    blending: str

    @property
    def label(self) -> str:
        name = "sRGB" if self.blending == Blending.SRGB else "リニア"
        return f"重ね合わせを変更: {name}"

    def apply(self, project: Project) -> Project:
        if self.blending not in Blending.ALL:
            raise ValueError(f"重ね合わせの方法が不正: {self.blending!r}")
        settings = replace(project.settings, blending=self.blending)
        return replace(project, settings=settings)


@dataclass(frozen=True, slots=True)
class SetLayerMode(Command):
    """素材を置くトラックの方式（:class:`~sashimono.core.model.LayerMode`）を変える

    変えるのは設定だけで、置いてあるトラックは動かさない どちらの方式のトラックも
    同じように描いて鳴らせるので、設定を変えただけで絵や音は変わらない
    置いてあるトラックまで変えるのは別の命令（変換）にする 1 つにまとめると、
    設定だけ戻したいときにも変換まで戻ってしまう
    """

    layer_mode: str

    @property
    def label(self) -> str:
        name = "混合" if self.layer_mode == LayerMode.MIXED else "映像と音声に分ける"
        return f"トラックの方式を変更: {name}"

    def apply(self, project: Project) -> Project:
        if self.layer_mode not in LayerMode.ALL:
            raise ValueError(f"トラックの方式が不正: {self.layer_mode!r}")
        settings = replace(project.settings, layer_mode=self.layer_mode)
        return replace(project, settings=settings)


@dataclass(frozen=True, slots=True)
class RenameProject(Command):
    """プロジェクト名を変える"""

    name: str

    @property
    def label(self) -> str:
        return "プロジェクト名を変更"

    def apply(self, project: Project) -> Project:
        return project.renamed(self.name)


def _trimmed(project: Project, clip: Clip, head_delta: int, tail_delta: int) -> Clip:
    """端を動かしたクリップを返す 無理な指定は例外にする"""
    duration = clip.duration - head_delta + tail_delta
    if duration <= 0:
        raise ValueError(f"トリム後の長さが 0 以下: {duration}")

    source_in = clip.source_in + head_delta * project.rate.frame_duration * clip.speed
    if source_in < 0:
        raise ValueError("素材の先頭より前はトリムできない")
    if clip.timeline_start + head_delta < 0:
        raise ValueError("タイムラインの先頭より前へは動かせない")

    return replace(
        clip,
        timeline_start=clip.timeline_start + head_delta,
        duration=duration,
        source_in=source_in,
    )


def _require_track(project: Project, track_id: TrackId) -> Track:
    track = project.timeline.find_track(track_id)
    if track is None:
        raise KeyError(f"トラックが見つからない: {track_id}")
    return track


def _carried_across(project: Project, source: Track, target: Track, clip: Clip) -> Clip:
    """音声トラックと混合トラックの間で動かすクリップの、鳴らす音声ストリームを移し替える

    音声トラックは :attr:`Clip.stream_index` を、混合トラックは :attr:`Clip.audio_stream` を
    鳴らす 移し替えないと、音声クリップをレイヤーへ移しただけで音が消え、逆向きでは
    選んだ音ではなく絵のストリームの番号で音を開く

    混合トラックから音声トラックへ移せないのは 2 つ 絵を描くクリップ（絵が黙って消える
    映像トラックへ音を鳴らすクリップを移せないのと同じ）と、音を鳴らさないクリップ
    （音声トラックでは鳴り出してしまう） 映像トラックへ移すときの決まり（音を鳴らす
    クリップと絵を隠したクリップを断る）は :func:`_validate_clip_media` が持つ
    """
    if source.kind is target.kind:
        return clip
    # 素材を持たないシーンやテキストもレイヤーでは絵を描く 素材の有無より先に見る
    if (
        source.kind is TrackKind.MIXED
        and target.kind is TrackKind.AUDIO
        and project.draws_picture(source, clip)
    ):
        raise ValueError("絵を描くクリップは音声トラックへ置けない（絵が消える）")
    if clip.media_id is None:
        if source.kind is TrackKind.AUDIO and target.kind is TrackKind.MIXED:
            # 音声トラックのシーンは音だけを出している レイヤーでも音だけのまま置く
            return replace(clip, show_picture=False)
        return clip
    if source.kind is TrackKind.AUDIO and target.kind is TrackKind.MIXED:
        media = project.require_media(clip.media_id)
        # 絵の番号は素材の映像ストリームへ向け直す 音の番号のまま残すと、後で絵を
        # 出したときに映像ではない番号でデコーダを開く
        picture = media.video_streams[0].index if media.has_video else clip.stream_index
        # 絵は出さないまま移す 音声トラックでは描いていなかった 描き始めると、リンクした
        # 映像クリップの絵がもう 1 枚重なり、手前なら映像トラックの位置やエフェクトを隠す
        return replace(
            clip, audio_stream=clip.stream_index, stream_index=picture, show_picture=False
        )
    if source.kind is TrackKind.MIXED and target.kind is TrackKind.AUDIO:
        if clip.audio_stream is None:
            raise ValueError("音を鳴らさないクリップは音声トラックへ置けない（鳴り出してしまう）")
        return replace(clip, stream_index=clip.audio_stream, audio_stream=None)
    return clip


def _validate_clip_media(project: Project, track: Track, clip: Clip) -> None:
    """クリップの素材がトラックの種類に合っているかを確かめる

    映像トラックに音声しか持たない素材を置くと、再生時に何も出ない無音の穴になる
    置いた時点で気付ける方がよい

    混合トラックは素材の種類を問わない（何でも置けるのが混合の意味）
    鳴らす音声ストリーム（:attr:`Clip.audio_stream`）を持つクリップは映像トラックへ
    置けない 映像トラックはクリップの音を鳴らさないので、混合トラックから移しただけで
    音が黙って消える
    """
    if track.kind is TrackKind.MIXED:
        if clip.media_id is None:
            return
        # 音を鳴らさないクリップでも素材があるかは確かめる 飛ばすと、プロジェクトに無い
        # 素材を指すクリップが置けてしまう（モデルは参照を確かめない）
        media = project.require_media(clip.media_id)
        if clip.audio_stream is None:
            return
        if all(stream.index != clip.audio_stream for stream in media.audio_streams):
            # デコーダは無い番号を頼まれると先頭の音へ逃げる 選んでいない言語が鳴る
            raise ValueError(f"素材 {media.name!r} に音声ストリーム {clip.audio_stream} は無い")
        return
    if track.kind is TrackKind.VIDEO and clip.audio_stream is not None:
        raise ValueError("音を鳴らすクリップは映像トラックへ置けない（音が鳴らなくなる）")
    if track.kind is TrackKind.VIDEO and not clip.show_picture:
        # 映像トラックは show_picture を読まずに必ず描く 素材を持たないテキストも同じで、
        # 混合トラックで隠していた絵が黙って映り出す
        raise ValueError("絵を隠したクリップは映像トラックへ置けない（絵が出てしまう）")
    if clip.media_id is None:
        return
    item = project.require_media(clip.media_id)
    if track.kind is TrackKind.VIDEO and not (item.has_video or item.is_still):
        raise ValueError(f"素材 {item.name!r} に映像が無いので映像トラックには置けない")
    if track.kind is TrackKind.AUDIO and not item.has_audio:
        raise ValueError(f"素材 {item.name!r} に音声が無いので音声トラックには置けない")


def _linked_group(project: Project, clip: Clip) -> list[tuple[TrackId, Clip]]:
    """リンクされたクリップをまとめて返す リンクが無ければ自分だけ"""
    if clip.link_group is None:
        located = project.timeline.locate_clip(clip.id)
        if located is None:
            return []
        track, found = located
        return [(track.id, found)]
    return [(track.id, found) for track, found in project.timeline.linked_clips(clip.link_group)]


@dataclass(frozen=True, slots=True)
class RippleCut(Command):
    """タイムラインの範囲をまとめて削除し、後ろを詰める

    ジェットカットの実体 範囲を 1 つずつ「分割して削除して詰める」形で組み立てる
    こともできるが、分割で生まれるクリップの ID が実行するまで分からないため、
    コマンドの列としては書けない 範囲の一覧を受け取って一度に処理する

    範囲に掛かったクリップは端が削られ、範囲をまたぐクリップは 2 つに分かれる
    ロックされたトラックには触れない 触れてしまうと、そのトラックだけ長さが
    変わらず、以降すべてがずれる
    """

    ranges: tuple[tuple[int, int], ...]

    @property
    def label(self) -> str:
        return f"無音をカット: {len(self.ranges)} か所"

    def apply(self, project: Project) -> Project:
        # 後ろから切る 前から切ると、切るたびに残りの範囲がずれて計算し直しになる
        for start, end in sorted(_normalized(self.ranges), reverse=True):
            project = _cut_range(project, start, end)
        return project


def _normalized(ranges: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in ranges if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _cut_range(project: Project, start: int, end: int) -> Project:
    """``[start, end)`` を全トラックから取り除き、後ろを詰める"""
    length = end - start
    rate = project.rate
    timeline = project.timeline
    # 範囲をまたいだクリップの右側に配り直すリンクグループ 左右が同じ
    # グループに残ると、片方を消したときにもう片方まで消える 映像と音声で
    # 同じ新グループを共有させたいので、この範囲の処理を通して覚えておく
    regrouped: dict[GroupId, GroupId] = {}

    for track in timeline.tracks:
        if track.locked:
            continue
        pieces: list[Clip] = []
        for clip in track.clips:
            pieces.extend(_cut_clip(clip, start, end, length, rate, regrouped))
        timeline = timeline.replace_track(track.with_clips(tuple(pieces)))

    markers = tuple(
        replace(marker, frame=marker.frame - length) if marker.frame >= end else marker
        for marker in timeline.markers
        if not (start <= marker.frame < end)
    )
    return project.with_timeline(replace(timeline, markers=markers, work_area=None))


def _cut_clip(
    clip: Clip,
    start: int,
    end: int,
    length: int,
    rate: FrameRate,
    regrouped: dict[GroupId, GroupId],
) -> list[Clip]:
    """1 つのクリップから範囲を抜く 残るのは 0 個・1 個・2 個のどれか"""
    if clip.timeline_end <= start:
        return [clip]
    if clip.timeline_start >= end:
        return [clip.moved_to(clip.timeline_start - length)]

    head = start - clip.timeline_start
    tail = clip.timeline_end - end
    if head <= 0 and tail <= 0:
        return []
    # 残る部分のキーフレームは、割るとき（:class:`SplitClip`）と同じく分けて数え直す
    # そのまま写すと、後ろに残った部分の動きが削った長さだけ前へずれる
    if head > 0 and tail <= 0:
        return [replace(_split_animation(clip, head)[0], duration=head)]

    # 範囲より後ろに残る部分 素材のどこから始まるかを計算し直す
    consumed = (end - clip.timeline_start) * rate.frame_duration * clip.speed
    right = replace(
        _split_animation(clip, end - clip.timeline_start)[1],
        id=new_clip_id() if head > 0 else clip.id,
        timeline_start=start,
        duration=tail,
        source_in=clip.source_in + consumed,
        link_group=_regroup(clip.link_group, regrouped) if head > 0 else clip.link_group,
    )
    if head <= 0:
        return [right]
    return [replace(_split_animation(clip, head)[0], duration=head), right]


def _regroup(group: GroupId | None, regrouped: dict[GroupId, GroupId]) -> GroupId | None:
    if group is None:
        return None
    if group not in regrouped:
        regrouped[group] = new_group_id()
    return regrouped[group]
