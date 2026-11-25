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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path

from sashimono.core.commands import AddClip, AddMedia, AddTrack, Command
from sashimono.core.commands.history import Author, Document
from sashimono.core.direction import (
    NARRATION_FOLDER,
    SHAKE,
    SOUND_EFFECT_MAX_SECONDS,
    ZOOM,
    direction_events,
)
from sashimono.core.direction_recipes import (
    ACTION_SHAKE,
    ACTION_SOUND,
    ACTION_ZOOM,
    ACTIONS,
    PlannedDirection,
)
from sashimono.core.material_library import same_place
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
from sashimono.core.timebase import Rounding, frame_to_seconds, seconds_to_frame
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
    #: 効果音の素材の名前 欄に出し、何の音を付けるかを〔付ける〕の前に見せる
    sound_name: str = ""
    #: エクスプローラーに登録したフォルダから選んだ効果音（#522） まだ作品の素材ではないので、
    #: 〔付ける〕で素材へ入れてから置く 計画を出すだけでは作品を変えない（素材一覧にも出さない）
    sound_media: MediaItem | None = None
    #: ``sound_media`` を調べたときのファイルの版（大きさ, 更新時刻） 〔付ける〕までに差し替わった
    #: ファイルを、計画を出したときの長さと流れのまま取り込まない
    sound_version: tuple[int, int] | None = None

    @property
    def action_name(self) -> str:
        return ACTIONS.get(self.action, self.action)


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """付けた結果 付けられなかった項目は理由を添える"""

    applied: tuple[int, ...]
    skipped: tuple[tuple[int, str], ...] = ()
    #: 付けた効果音の項目が実際に使った素材（項目の番号, 素材） 計画の後で同じファイルを取り込んで
    #: いたときは、計画の素材でなくその素材を使うので、計画の返事にはこちらを出す
    media: tuple[tuple[int, MediaId], ...] = ()


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

    def apply(
        self, document: Document, allowed: Callable[[MediaItem], bool] | None = None
    ) -> ApplyResult | None:
        """外していない項目を 1 回の取り消しで戻せるまとまりとして付ける 出していなければ ``None``

        編集画面のスレッドから呼ぶ（文書を変えるため） ``allowed`` は登録したフォルダから
        取り込む効果音を、付ける今もまだ AI に使ってよいか（:func:`apply_items`）
        """
        with self._lock:
            view = self._view
            if view is None or view.state is not ProposalState.PROPOSED:
                return None
            chosen = view.chosen
        result = apply_items(document, chosen, allowed)
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
    if media.path.parent.name == NARRATION_FOLDER:
        # 段 1 はナレーションの録音を効果音に数えない 置けても数えられず、計画と数が食い違う
        return "ナレーションの録音は効果音に使えません 効果音の素材を選んでください"
    return None


def build_items(
    project: Project,
    planned: Sequence[PlannedDirection],
    clips: Sequence[ClipId | None],
    sound: MediaItem | None,
    limit: int,
    sound_version: tuple[int, int] | None = None,
) -> tuple[list[DirectionItem], list[tuple[str, str]]]:
    """レシピの選んだ演出を、付ける先の決まった計画の項目にする

    ``limit`` は計画した範囲の終わりのフレーム 効果音がこれを越えるなら外す

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
        # 計画の秒は手がかりのフレームから作った物 切り捨てで戻すと、29.97 fps のように
        # 秒が割り切れないレートで手がかりより 1 フレーム早くなる
        start = seconds_to_frame(Fraction(plan.start), rate, Rounding.NEAREST)
        end = max(start + 1, seconds_to_frame(Fraction(plan.end), rate, Rounding.NEAREST))
        where = _seconds_text(start, project)
        if plan.action == ACTION_SOUND:
            if sound is None:
                dropped.append((where, "効果音に使う本人の素材が選ばれていません"))
                continue
            end = start + max(1, seconds_to_frame(sound.duration, rate))
            if end > limit:
                # 作品の終わりを越えて置くと、作品が延びて書き出しに絵の無い末尾が付く
                # 音を途中で切ると笑いや驚きの音として聞こえないので、切り詰めずに外す
                dropped.append((where, "効果音が範囲の終わりを越えます"))
                continue
            if any(
                e.kind == plan.kind and e.start_frame < end and start < e.end_frame
                for e in existing
            ):
                # 前に付けた効果音や人が置いた効果音と重ねると、同じ音が二重に鳴って大きくなる
                # 計画を出し直すたびに増えないよう、寄る・揺らすと同じく上書きせず外す
                dropped.append((where, "その時刻にはもう効果音があるので重ねません"))
                continue
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
                sound_name=sound.name if sound is not None and media_id is not None else "",
                sound_media=(
                    sound
                    if sound is not None
                    and media_id is not None
                    and project.find_media(sound.id) is None
                    else None
                ),
                sound_version=sound_version,
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


def apply_items(
    document: Document,
    items: Sequence[DirectionItem],
    allowed: Callable[[MediaItem], bool] | None = None,
) -> ApplyResult:
    """``items`` を 1 回の取り消しで戻せるまとまりとして付ける

    項目ごとに今の作品から命令を組む 先に全部を組むと、同じクリップへの 2 つ目の寄りが
    1 つ目を付ける前の配置から組まれ、1 つ目を消してしまう
    付けられなかった項目（計画を出した後でクリップが消えた など）は飛ばして理由を返す
    1 つの失敗で全部をやめると、外さなかった残りまで付かない
    """
    applied: list[int] = []
    skipped: list[tuple[int, str]] = []
    used: list[tuple[int, MediaId]] = []
    if not items:
        return ApplyResult((), ())
    with document.checkpoint(APPLY_LABEL, by=Author.AI):
        for item in items:
            sound = item.sound_media
            if (
                sound is not None
                and allowed is not None
                and _held(document.project, sound) is None
                and not allowed(sound)
            ):
                # 計画の後で登録を外した・AI に使わせない分類にした 計画を出したときの確かめだけで
                # 取り込むと、本人が使わせないと決めた後の音が作品に入る
                skipped.append(
                    (item.number, "効果音のフォルダが AI に使わせない設定になったので付けません")
                )
                continue
            if (
                sound is not None
                and item.sound_version is not None
                and _held(document.project, sound) is None
                and file_version(sound.path) != item.sound_version
            ):
                # 計画の後でファイルが差し替わった（消えた） 計画を出したときに調べた長さと流れの
                # まま取り込むと、置いたクリップが今のファイルと合わない
                skipped.append(
                    (
                        item.number,
                        "計画の後で効果音のファイルが変わったので付けません 計画し直してください",
                    )
                )
                continue
            try:
                commands = _commands(document.project, item)
                # 先に写しへ当てて、項目の命令がすべて通るかを見る 通らない物（鍵の掛かった
                # トラックへ置く など）を途中まで実行すると、その例外で外の段ごと戻り、
                # 先に付けた項目まで消える
                trial = document.project
                for command in commands:
                    trial = command.apply(trial)
            except ValueError as exc:  # EmphasisError も ValueError
                skipped.append((item.number, str(exc)))
                continue
            # 文書の側で断る物（持ち出し用パッケージの範囲の外 など）は命令の試しでは分からない
            refused = document.refusal(trial, by=Author.AI)
            if refused is not None:
                skipped.append((item.number, refused))
                continue
            for command in commands:
                document.execute(command)
            applied.append(item.number)
            if item.action == ACTION_SOUND:
                used.extend(
                    (item.number, command.clip.media_id)
                    for command in commands
                    if isinstance(command, AddClip) and command.clip.media_id is not None
                )
    return ApplyResult(tuple(applied), tuple(skipped), tuple(used))


def file_version(path: Path) -> tuple[int, int] | None:
    """ファイルの版（大きさ, 更新時刻） 読めなければ ``None``"""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _held(project: Project, sound: MediaItem) -> MediaItem | None:
    """作品の素材のうち ``sound`` と同じ物（同じ ID か、同じ場所のファイル）"""
    found = project.find_media(sound.id)
    if found is not None:
        return found
    return next((media for media in project.media if same_place(media.path, sound.path)), None)


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
    """効果音を効果音のトラックへ置く 空いていなければ足したトラックへ置く

    登録したフォルダから選んだ音（:attr:`DirectionItem.sound_media`）は、まだ作品の素材で
    なければ素材へ入れてから置く 入れるのも同じ取り消しの段なので、1 回で素材ごと戻る
    """
    added: list[Command] = []
    sound = item.sound_media
    media_id = item.media_id
    if (
        media_id is not None
        and project.find_media(media_id) is None
        and sound is not None
        and sound.id == media_id
    ):
        held = _held(project, sound)
        if held is not None:
            # 計画の後で同じファイルを取り込んでいた（別の ID） その素材を使う もう 1 本足すと
            # 素材一覧に同じ音が 2 つ並ぶ
            media_id = held.id
        else:
            added.append(AddMedia(sound))
            project = added[0].apply(project)
    if media_id is None or project.find_media(media_id) is None:
        raise ValueError("効果音の素材が見つかりません（計画の後で外された）")
    # 計画を作った時の検査だけでは、その後に人が同じ区間へ置いた効果音を見落とす
    # 足すのは別のトラックなので、重なると両方が鳴って大きくなる 付ける直前の作品でも見る
    if any(
        event.kind == item.kind
        and event.start_frame < item.end_frame
        and item.start_frame < event.end_frame
        for event in direction_events(project, project.timeline)
    ):
        raise ValueError("その時刻にはもう効果音があるので重ねません（計画の後で置かれた）")
    clip = Clip(
        timeline_start=item.start_frame,
        duration=item.end_frame - item.start_frame,
        media_id=media_id,
    )
    timeline = project.timeline
    # 音のソロがあると、ソロでない音のトラックは鳴らない 足すトラックもソロにそろえる
    # ソロは役割の中で決まるので、音を鳴らすトラック（音声と混合）の中だけで見る 映像トラックの
    # ソロまで見ると、足したトラックのソロで本人の BGM やナレーションが鳴らなくなる
    # 人のトラックのソロや消音は変えない（人が聞き分けのために付けた物）
    solo = any(track.solo and not track.muted for track in timeline.sound_tracks())
    sounding = {track.id for track in timeline.active_sound_tracks()}
    for track in timeline.tracks:
        free = all(
            other.timeline_end <= clip.timeline_start or clip.timeline_end <= other.timeline_start
            for other in track.clips
        )
        # 消音・鍵・ソロの外のトラックへ置くと、付いたことになっても鳴らない（置けない）
        audible = track.id in sounding and not track.locked
        if track.kind is TrackKind.AUDIO and track.name == SOUND_TRACK_NAME and free and audible:
            return [*added, AddClip(track.id, clip)]
    track = Track(kind=TrackKind.AUDIO, name=SOUND_TRACK_NAME, solo=solo)
    return [*added, AddTrack(track), AddClip(track.id, clip)]
