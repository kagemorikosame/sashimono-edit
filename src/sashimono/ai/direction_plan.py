"""演出の計画（#470 段 4 #479）

AI がレシピ（:mod:`sashimono.core.direction_recipes`）で選んだ演出を、項目の一覧として
計画の欄に出す 人は要らない項目を外し、残りを 1 回で付ける 取り消し 1 回で全部戻る
（2026-11-16 に本人が決めた確かめ方）

演出は数が多く、1 つずつ確かめると人の手間が増える AI がそのまま付けると、外したい 1 つの
ために全部を取り消すことになる 一覧で見せて外せるようにし、付けるのは人の〔付ける〕にする

計画は作品ごとに 1 つ、記憶の中だけに持つ（:mod:`sashimono.ai.plan` と同じく、別の作品を
開いたら閉じる） AI の道具は会話のスレッドから、欄は編集画面のスレッドから触るので、
中身は錠の内で読み書きする
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction

from sashimono.core.commands import AddClip, AddTrack, Command
from sashimono.core.commands.history import Author, Document
from sashimono.core.direction import SHAKE, SOUND_EFFECT_MAX_SECONDS, ZOOM, direction_events
from sashimono.core.direction_recipes import (
    ACTION_SHAKE,
    ACTION_SOUND,
    ACTION_ZOOM,
    ACTIONS,
    PlannedDirection,
)
from sashimono.core.model import (
    Clip,
    ClipId,
    MediaId,
    MediaItem,
    Project,
    Track,
    TrackId,
    TrackKind,
)
from sashimono.core.timebase import frame_to_seconds, seconds_to_frame
from sashimono.effects.emphasis import EmphasisRequest, emphasis_commands

__all__ = [
    "APPLY_LABEL",
    "SOUND_TRACK_NAME",
    "ApplyResult",
    "DirectionItem",
    "DirectionProposal",
    "ProposalState",
    "ProposalView",
    "apply_items",
    "build_items",
    "proposal_for",
    "sound_problem",
]

#: 取り消しの段の名前 1 回で全部戻ることが履歴の名前から分かるようにする
APPLY_LABEL = "演出の計画を付ける"
#: 効果音を置くトラックの名前 本人の音声トラックへ混ぜると、BGM やナレーションと重なる
SOUND_TRACK_NAME = "効果音"
#: 一覧に出す項目の上限 多すぎると欄で外す手間が 1 つずつ確かめるのと変わらない
ITEMS_LIMIT = 200


class ProposalState(Enum):
    """計画の状態"""

    #: 人が外して〔付ける〕を押すのを待っている
    PROPOSED = "proposed"
    #: 付けた
    APPLIED = "applied"
    #: 人が〔やめる〕を押した
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DirectionItem:
    """計画の項目 1 つ フレームはタイムラインの位置"""

    number: int
    recipe: str
    action: str
    kind: str
    start_frame: int
    end_frame: int
    amount: int | None
    reason: str
    #: 寄る・揺らすを付けるクリップ 効果音は持たない
    clip_id: ClipId | None = None
    #: 効果音に使う本人の素材 寄る・揺らすは持たない
    media_id: MediaId | None = None
    #: 欄に出す時刻（分:秒） 欄はフレームの速さを知らないので、作るときに決めておく
    at: str = ""

    @property
    def action_name(self) -> str:
        return ACTIONS.get(self.action, self.action)


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """付けた結果 付けられなかった項目は理由を添える"""

    applied: tuple[int, ...]
    skipped: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ProposalView:
    """欄と道具へ渡す計画の写し"""

    state: ProposalState
    items: tuple[DirectionItem, ...]
    excluded: frozenset[int]
    #: 計画を作るときに外した物（クリップが無い・手で整えた配置がある など）と理由
    dropped: tuple[tuple[str, str], ...]
    strength_name: str
    version: int
    result: ApplyResult | None = None

    @property
    def chosen(self) -> tuple[DirectionItem, ...]:
        return tuple(item for item in self.items if item.number not in self.excluded)


@dataclass
class DirectionProposal:
    """作品 1 つの演出の計画 無い間は ``view()`` が ``None``"""

    generation: int
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _view: ProposalView | None = None
    _version: int = 0

    def follow(self, generation: int) -> bool:
        """別の作品を開いていたら計画を閉じる 閉じたら真

        前の作品の計画を次の作品で付けると、別の作品のクリップへ演出を付けかねない
        """
        with self._lock:
            if generation == self.generation:
                return False
            self.generation = generation
            closed = self._view is not None
            self._view = None
            self._version += 1
            return closed

    def propose(
        self,
        items: Sequence[DirectionItem],
        dropped: Sequence[tuple[str, str]],
        strength_name: str,
    ) -> ProposalView:
        """計画を出す 前の計画は捨てる（頼み直したら最新の 1 つだけを見せる）"""
        with self._lock:
            self._version += 1
            self._view = ProposalView(
                state=ProposalState.PROPOSED,
                items=tuple(items),
                excluded=frozenset(),
                dropped=tuple(dropped),
                strength_name=strength_name,
                version=self._version,
            )
            return self._view

    def view(self) -> ProposalView | None:
        with self._lock:
            return self._view

    def exclude(self, number: int, excluded: bool) -> None:
        """項目を外す（``excluded`` が偽なら戻す） 出している間だけ"""
        with self._lock:
            view = self._view
            if view is None or view.state is not ProposalState.PROPOSED:
                return
            numbers = {item.number for item in view.items}
            if number not in numbers:
                return
            chosen = set(view.excluded)
            if excluded:
                chosen.add(number)
            else:
                chosen.discard(number)
            self._version += 1
            self._view = replace(view, excluded=frozenset(chosen), version=self._version)

    def discard(self) -> None:
        """欄の〔×〕 付けた後・やめた後の計画を閉じる 出している間は閉じない
        （閉じると、人が外した印も〔付ける〕も消え、AI は計画が消えた訳を読めない）
        """
        with self._lock:
            if self._view is None or self._view.state is ProposalState.PROPOSED:
                return
            self._view = None
            self._version += 1

    def cancel(self) -> None:
        """〔やめる〕 何も付けずに閉じる"""
        self._finish(ProposalState.CANCELLED, None)

    def apply(self, document: Document) -> ApplyResult | None:
        """外していない項目を 1 回の取り消しで戻せるまとまりとして付ける 出していなければ ``None``

        編集画面のスレッドから呼ぶ（文書を変えるため）
        """
        with self._lock:
            view = self._view
            if view is None or view.state is not ProposalState.PROPOSED:
                return None
            chosen = view.chosen
        result = apply_items(document, chosen)
        self._finish(ProposalState.APPLIED, result)
        return result

    def _finish(self, state: ProposalState, result: ApplyResult | None) -> None:
        with self._lock:
            view = self._view
            if view is None or view.state is not ProposalState.PROPOSED:
                return
            self._version += 1
            self._view = replace(view, state=state, result=result, version=self._version)


_PROPOSALS: weakref.WeakKeyDictionary[Document, DirectionProposal] = weakref.WeakKeyDictionary()
_PROPOSALS_LOCK = threading.Lock()


def proposal_for(document: Document) -> DirectionProposal:
    """``document`` の演出の計画 開いている作品が変わっていれば、前の計画は閉じてある"""
    with _PROPOSALS_LOCK:
        proposal = _PROPOSALS.get(document)
        if proposal is None:
            proposal = DirectionProposal(document.generation)
            _PROPOSALS[document] = proposal
    proposal.follow(document.generation)
    return proposal


def sound_problem(media: MediaItem) -> str | None:
    """効果音に使えない素材なら理由 使えれば ``None``

    段 1 の測り方が効果音と数える物（音だけ・短い）に限る 映像や長い音を置くと、
    付けた後に数えても効果音に入らず、計画と数が食い違う
    """
    if media.has_video:
        return "映像のある素材は効果音に使えません 音だけの素材を選んでください"
    if not media.has_audio:
        return "音の無い素材は効果音に使えません"
    if media.duration > SOUND_EFFECT_MAX_SECONDS:
        return f"効果音に使えるのは {SOUND_EFFECT_MAX_SECONDS} 秒以下の音です"
    return None


def build_items(
    project: Project,
    planned: Sequence[PlannedDirection],
    clips: Sequence[ClipId | None],
    sound: MediaItem | None,
) -> tuple[list[DirectionItem], list[tuple[str, str]]]:
    """レシピの選んだ演出を、付ける先の決まった計画の項目にする

    ``clips`` は ``planned`` と同じ並びの、手がかりを拾ったクリップ（無ければ ``None`` で、
    その時刻に映る素材のクリップを選ぶ） 付けられない物は外して理由を返す
    外す物 クリップが無い・すでに同じ型の動きがある（人が手で整えた配置や前に付けた動きを
    上書きしない #427 #428 と同じく、人が作った物を AI が黙って壊さない）・効果音の素材が無い
    """
    rate = project.timeline.rate
    existing = direction_events(project, project.timeline)
    items: list[DirectionItem] = []
    dropped: list[tuple[str, str]] = []
    for plan, clip_hint in zip(planned, clips, strict=True):
        start = seconds_to_frame(Fraction(plan.start).limit_denominator(1000), rate)
        end = max(start + 1, seconds_to_frame(Fraction(plan.end).limit_denominator(1000), rate))
        where = _seconds_text(start, project)
        if plan.action == ACTION_SOUND:
            if sound is None:
                dropped.append((where, "効果音に使う本人の素材が選ばれていません"))
                continue
            end = start + max(1, seconds_to_frame(sound.duration, rate))
            clip_id: ClipId | None = None
            media_id: MediaId | None = sound.id
        else:
            clip = _target(project, clip_hint, start)
            if clip is None:
                dropped.append((where, "その時刻に寄る・揺らす先の素材のクリップがありません"))
                continue
            end = min(end, clip.timeline_end)
            if any(
                e.kind == plan.kind
                and e.clip_id == str(clip.id)
                and e.start_frame < end
                and start < e.end_frame
                for e in existing
            ):
                dropped.append((where, f"すでに{ACTIONS[plan.action]}動きがあるので上書きしません"))
                continue
            clip_id, media_id = clip.id, None
        items.append(
            DirectionItem(
                number=len(items) + 1,
                recipe=plan.recipe,
                action=plan.action,
                kind=plan.kind,
                start_frame=start,
                end_frame=end,
                amount=plan.amount,
                reason=plan.reason,
                clip_id=clip_id,
                media_id=media_id,
                at=where,
            )
        )
        if len(items) >= ITEMS_LIMIT:
            break
    return items, dropped


def _seconds_text(frame: int, project: Project) -> str:
    seconds = float(frame_to_seconds(frame, project.timeline.rate))
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _target(project: Project, hint: ClipId | None, frame: int) -> Clip | None:
    """寄る・揺らす先 手がかりを拾ったクリップがその時刻に映っていればそれ、無ければ
    その時刻に映る素材のクリップのうち一番奥（本編）

    テキストや図形を寄せると、話者でなく文字が動く 一番手前の素材（顔の小窓など）を
    寄せると、本編はそのままで小窓だけが動く
    """
    timeline = project.timeline
    shown = {track.id for track in timeline.active_picture_tracks()}
    if hint is not None:
        located = timeline.locate_clip(hint)
        if located is not None:
            track, clip = located
            if _footage(project, track, clip, shown) and clip.contains(frame):
                return clip
    for track in timeline.tracks:
        found = track.clip_at(frame)
        if found is not None and _footage(project, track, found, shown):
            return found
    return None


def _footage(project: Project, track: Track, clip: Clip, shown: set[TrackId]) -> bool:
    return (
        track.id in shown
        and clip.enabled
        and clip.source is None
        and clip.media_id is not None
        and project.draws_picture(track, clip)
    )


def apply_items(document: Document, items: Sequence[DirectionItem]) -> ApplyResult:
    """``items`` を 1 回の取り消しで戻せるまとまりとして付ける

    項目ごとに今の作品から命令を組む 先に全部を組むと、同じクリップへの 2 つ目の寄りが
    1 つ目を付ける前の配置から組まれ、1 つ目を消してしまう
    付けられなかった項目（計画を出した後でクリップが消えた など）は飛ばして理由を返す
    1 つの失敗で全部をやめると、外さなかった残りまで付かない
    """
    applied: list[int] = []
    skipped: list[tuple[int, str]] = []
    if not items:
        return ApplyResult((), ())
    with document.checkpoint(APPLY_LABEL, by=Author.AI):
        for item in items:
            try:
                commands = _commands(document.project, item)
            except ValueError as exc:  # EmphasisError も ValueError
                skipped.append((item.number, str(exc)))
                continue
            for command in commands:
                document.execute(command)
            applied.append(item.number)
    return ApplyResult(tuple(applied), tuple(skipped))


def _commands(project: Project, item: DirectionItem) -> list[Command]:
    if item.action == ACTION_SOUND:
        return _sound_commands(project, item)
    if item.clip_id is None:
        raise ValueError("付ける先のクリップがありません")
    located = project.timeline.locate_clip(item.clip_id)
    if located is None:
        raise ValueError("付ける先のクリップが見つかりません（計画の後で消えた）")
    _, clip = located
    start = max(0, item.start_frame - clip.timeline_start)
    end = min(clip.duration, item.end_frame - clip.timeline_start)
    if end <= start:
        raise ValueError("付ける先のクリップがその時刻に映っていません（計画の後で動いた）")
    kind = {ACTION_ZOOM: ZOOM, ACTION_SHAKE: SHAKE}[item.action]
    request = EmphasisRequest(kind=kind, start=start, end=end, strength=float(item.amount or 0))
    commands, _ = emphasis_commands(clip, request, project.timeline.rate)
    return commands


def _sound_commands(project: Project, item: DirectionItem) -> list[Command]:
    """効果音を効果音のトラックへ置く 空いていなければ足したトラックへ置く"""
    if item.media_id is None or project.find_media(item.media_id) is None:
        raise ValueError("効果音の素材が見つかりません（計画の後で外された）")
    clip = Clip(
        timeline_start=item.start_frame,
        duration=item.end_frame - item.start_frame,
        media_id=item.media_id,
    )
    for track in project.timeline.tracks:
        free = all(
            other.timeline_end <= clip.timeline_start or clip.timeline_end <= other.timeline_start
            for other in track.clips
        )
        if track.kind is TrackKind.AUDIO and track.name == SOUND_TRACK_NAME and free:
            return [AddClip(track.id, clip)]
    track = Track(kind=TrackKind.AUDIO, name=SOUND_TRACK_NAME)
    return [AddTrack(track), AddClip(track.id, clip)]
