"""テキストの縁取りの層を足す・消す・並べ替える（#272）

層の項目の値とキーフレームは :class:`~sashimono.core.commands.effects.ParamPath` の
``stroke_id`` で指して、ふつうの値と同じ命令（``SetParam`` など）で変える 層に掛ける
エフェクトも ``AddEffect`` などの ``stroke_id`` で指す（#273）

層を持たないテキストは前からの項目（``border_width`` ``border_color``）で縁を描く
（:mod:`sashimono.core.model.stroke`） 初めて層を足すときは、その縁を 1 つ目の層へ移して
から足す 移さずに足すと、前からの縁が層の並びの外に残り、どこにも出ないまま描かれ続ける
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import _update_clip, stroke_of, with_stroke
from sashimono.core.model import (
    LEGACY_BORDER_WIDTH,
    MAX_STROKES,
    AnimatedValue,
    Clip,
    ClipId,
    GeneratedSource,
    Project,
    Stroke,
    StrokeId,
    legacy_in_use,
    legacy_stroke,
    new_stroke_id,
)

__all__ = [
    "AddStroke",
    "AdoptLegacyStroke",
    "MoveStroke",
    "RemoveStroke",
    "SetStrokeEnabled",
    "adopted_source",
]

#: 縁取りの層を持てる生成オブジェクトの種類 描く所（``engine/sources.py``）はテキストの
#: 縁取りにだけ層を読む 図形に持たせても描かれず、保存だけされて残る
_TEXT_KIND = "text"


def _text_source(clip: Clip) -> GeneratedSource:
    if clip.source is None or clip.source.kind != _TEXT_KIND:
        raise ValueError("縁取りの層を持てるのはテキストだけ")
    return clip.source


def adopted_source(source: GeneratedSource, stroke_id: StrokeId) -> GeneratedSource:
    """前からの縁取りを、ID ``stroke_id`` の 1 つ目の層へ移した中身

    前からの太さは 0 にする 残すと、層を全部消したときに前の縁がまた出てくる
    色は残す（太さ 0 なら描かれず、前の版の本体に戻したときの手がかりにもなる）
    """
    if source.strokes:
        raise ValueError("もう縁取りの層を持っている")
    params = {**source.params, LEGACY_BORDER_WIDTH: AnimatedValue(0.0)}
    return replace(source, params=params, strokes=(legacy_stroke(source.params, stroke_id),))


@dataclass(frozen=True, slots=True)
class AdoptLegacyStroke(Command):
    """前からの縁取り（1 組の項目）を、縁取りの層 1 つへ移す

    設定パネルは、層を持たないテキストの縁を仮の層として見せ、触ったときにこれで移してから
    値を入れる（1 回の取り消しで戻る） 開いただけで移すと、見ただけのクリップまで変わり、
    保存を促される
    """

    clip_id: ClipId
    stroke_id: StrokeId

    @property
    def label(self) -> str:
        return "縁取りを層にする"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            return replace(clip, source=adopted_source(_text_source(clip), self.stroke_id))

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class AddStroke(Command):
    """縁取りの層を足す ``index`` を省けば並びの末尾（一番下）

    層を持たないテキストに前からの縁があれば、先にそれを 1 つ目の層へ移す
    層の数は :data:`~sashimono.core.model.MAX_STROKES` まで
    """

    clip_id: ClipId
    stroke: Stroke
    index: int | None = None

    @property
    def label(self) -> str:
        return "縁取りを追加"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            source = _text_source(clip)
            if not source.strokes and legacy_in_use(source.params):
                source = adopted_source(source, new_stroke_id())
            strokes = list(source.strokes)
            if len(strokes) >= MAX_STROKES:
                raise ValueError(f"縁取りの層は {MAX_STROKES} つまで")
            if any(stroke.id == self.stroke.id for stroke in strokes):
                raise ValueError(f"同じ ID の層がもうある: {self.stroke.id}")
            position = len(strokes) if self.index is None else self.index
            strokes.insert(max(0, min(position, len(strokes))), self.stroke)
            return replace(clip, source=source.with_strokes(strokes))

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class RemoveStroke(Command):
    """縁取りの層を消す 層に掛けたエフェクトも一緒に消える"""

    clip_id: ClipId
    stroke_id: StrokeId

    @property
    def label(self) -> str:
        return "縁取りを削除"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            source = _text_source(clip)
            stroke_of(clip, self.stroke_id)
            remaining = [stroke for stroke in source.strokes if stroke.id != self.stroke_id]
            return replace(clip, source=source.with_strokes(remaining))

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class MoveStroke(Command):
    """縁取りの層の順番を変える 並びの頭が一番上に描かれる"""

    clip_id: ClipId
    stroke_id: StrokeId
    index: int

    @property
    def label(self) -> str:
        return "縁取りの順番を変更"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            source = _text_source(clip)
            moving = stroke_of(clip, self.stroke_id)
            strokes = [stroke for stroke in source.strokes if stroke.id != self.stroke_id]
            strokes.insert(max(0, min(self.index, len(strokes))), moving)
            return replace(clip, source=source.with_strokes(strokes))

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class SetStrokeEnabled(Command):
    """縁取りの層を隠す・出す 消さずに切れると、層を足す前と後を見比べられる"""

    clip_id: ClipId
    stroke_id: StrokeId
    enabled: bool

    @property
    def label(self) -> str:
        return "縁取りを有効化" if self.enabled else "縁取りを無効化"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            _text_source(clip)
            return with_stroke(clip, replace(stroke_of(clip, self.stroke_id), enabled=self.enabled))

        return _update_clip(project, self.clip_id, update)
