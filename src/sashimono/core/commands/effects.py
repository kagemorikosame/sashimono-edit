"""エフェクトとパラメータの操作

パラメータの指し方を :class:`ParamPath` に統一してある クリップ自身の値も、
エフェクトの値も、生成オブジェクトの値も同じ形で指せるので、設定 UI も
キーフレーム編集も AI エージェントも 1 種類のコマンドで済む

指す先ごとにコマンドを分けると、キーフレームの追加だけで 3 種類を書くことになり、
どれか 1 つの実装が遅れて「エフェクトはアニメーションするがテキストはしない」
といったちぐはぐが生まれる
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import cast

from sashimono.core.commands.base import Command
from sashimono.core.commands.fixed import fixed_slot, loose_slot
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    EffectId,
    GeneratedSource,
    Interpolation,
    Keyframe,
    ParamValue,
    Project,
    Stroke,
    StrokeId,
)

__all__ = [
    "AddEffect",
    "ClearKeyframes",
    "MoveEffect",
    "MoveKeyframe",
    "ParamPath",
    "ParamTarget",
    "RemoveEffect",
    "RemoveKeyframe",
    "SetClipProperty",
    "SetEffectEnabled",
    "SetKeyframe",
    "SetParam",
    "SetSource",
    "resolve_param",
]


class ParamTarget(Enum):
    """パラメータがどこに属しているか"""

    #: クリップに積んだエフェクトの値
    EFFECT = "effect"
    #: 生成オブジェクト（テキスト・図形）の値
    SOURCE = "source"
    #: クリップ自身の値（不透明度など）
    CLIP = "clip"


@dataclass(frozen=True, slots=True)
class ParamPath:
    """1 つのパラメータの在りか"""

    clip_id: ClipId
    target: ParamTarget
    name: str
    #: ``target`` が :attr:`ParamTarget.EFFECT` のときだけ意味を持つ
    effect_id: EffectId | None = None
    #: 場面切り替えの「後の場面」に積んだエフェクトを指すか
    after: bool = False
    #: テキストの縁取りの層（#272） ``SOURCE`` なら層の項目、``EFFECT`` なら層に掛けた
    #: エフェクト（#273）の項目を指す 指す先を同じ形にしておくと、値の変更・キーフレーム・
    #: 初期値へ戻す操作を、層のために別に書かずに済む
    stroke_id: StrokeId | None = None

    def __post_init__(self) -> None:
        if self.target is ParamTarget.EFFECT and self.effect_id is None:
            raise ValueError("エフェクトのパラメータには effect_id が要る")
        if self.stroke_id is not None and (self.target is ParamTarget.CLIP or self.after):
            raise ValueError("縁取りの層はテキストの中身にだけある")

    @classmethod
    def of_effect(
        cls, clip_id: ClipId, effect_id: EffectId, name: str, *, after: bool = False
    ) -> ParamPath:
        return cls(clip_id, ParamTarget.EFFECT, name, effect_id, after=after)

    @classmethod
    def of_source(cls, clip_id: ClipId, name: str) -> ParamPath:
        return cls(clip_id, ParamTarget.SOURCE, name)

    @classmethod
    def of_clip(cls, clip_id: ClipId, name: str) -> ParamPath:
        return cls(clip_id, ParamTarget.CLIP, name)

    @classmethod
    def of_stroke(cls, clip_id: ClipId, stroke_id: StrokeId, name: str) -> ParamPath:
        """縁取りの層の項目（太さ・色・位置など）"""
        return cls(clip_id, ParamTarget.SOURCE, name, stroke_id=stroke_id)

    @classmethod
    def of_stroke_effect(
        cls, clip_id: ClipId, stroke_id: StrokeId, effect_id: EffectId, name: str
    ) -> ParamPath:
        """縁取りの層に掛けたエフェクトの項目"""
        return cls(clip_id, ParamTarget.EFFECT, name, effect_id, stroke_id=stroke_id)


def stack_of(clip: Clip, after: bool, stroke_id: StrokeId | None = None) -> tuple[Effect, ...]:
    """エフェクトの置き場 場面切り替えの「後の場面」と、縁取りの層ごとの物は別に持つ"""
    if stroke_id is not None:
        return stroke_of(clip, stroke_id).effects
    return clip.after_effects if after else clip.effects


def with_stack(
    clip: Clip, after: bool, effects: tuple[Effect, ...], stroke_id: StrokeId | None = None
) -> Clip:
    if stroke_id is not None:
        stroke = stroke_of(clip, stroke_id)
        return with_stroke(clip, replace(stroke, effects=effects))
    return replace(clip, after_effects=effects) if after else replace(clip, effects=effects)


def stroke_of(clip: Clip, stroke_id: StrokeId) -> Stroke:
    """クリップの縁取りの層 無ければ ``KeyError``"""
    found = clip.source.find_stroke(stroke_id) if clip.source is not None else None
    if found is None:
        raise KeyError(f"縁取りの層が見つからない: {stroke_id}")
    return found


def with_stroke(clip: Clip, stroke: Stroke) -> Clip:
    """同じ ID の層を ``stroke`` に差し替えたクリップ"""
    assert clip.source is not None  # 層を引けた（:func:`stroke_of`）クリップだけが来る
    strokes = tuple(stroke if s.id == stroke.id else s for s in clip.source.strokes)
    return replace(clip, source=clip.source.with_strokes(strokes))


def resolve_param(project: Project, path: ParamPath) -> ParamValue | None:
    """今の値を読む 見つからなければ ``None``"""
    located = project.timeline.locate_clip(path.clip_id)
    if located is None:
        return None
    _, clip = located

    if path.target is ParamTarget.CLIP:
        return getattr(clip, path.name, None)
    stroke = (
        clip.source.find_stroke(path.stroke_id)
        if path.stroke_id is not None and clip.source is not None
        else None
    )
    if path.stroke_id is not None and stroke is None:
        return None
    if path.target is ParamTarget.SOURCE:
        if stroke is not None:
            return stroke.params.get(path.name)
        return clip.source.params.get(path.name) if clip.source is not None else None

    stack = stroke.effects if stroke is not None else stack_of(clip, path.after)
    effect = next((e for e in stack if e.id == path.effect_id), None)
    return effect.params.get(path.name) if effect is not None else None


@dataclass(frozen=True, slots=True)
class SetParam(Command):
    """パラメータの値を差し替える

    キーフレームの付いた値に対して呼ぶと、アニメーションを捨てて静的な値になる
    スライダーを触ったときの挙動としてはそれが自然（キーフレームを残したまま
    値だけ変えると、次のフレームで元へ戻って「効かない」ように見える）
    """

    path: ParamPath
    value: ParamValue

    @property
    def label(self) -> str:
        return f"{self.path.name} を変更"

    def apply(self, project: Project) -> Project:
        return _update_param(project, self.path, lambda _: self.value)


@dataclass(frozen=True, slots=True)
class SetKeyframe(Command):
    """指定フレームにキーフレームを置く すでにあれば差し替える

    ``interpolation`` を省くと、同じフレームに点があればその出方（補間方法・制御点・
    曲線の名前）を引き継ぎ、無ければ直線にする 値だけを直す操作（インスペクターで
    数を打ち直す）で出方まで直線へ戻すと、YMM4 から読んだ Back や Expo の点が
    値を触っただけで別の動きになる 出方を変えたいときは ``interpolation`` を渡す
    """

    path: ParamPath
    frame: int
    value: float
    interpolation: Interpolation | None = None
    control_points: tuple[float, float, float, float] | None = None
    curve: str = ""

    @property
    def label(self) -> str:
        return f"{self.path.name} にキーフレーム"

    def _keyframe(self, existing: Keyframe | None) -> Keyframe:
        if self.interpolation is None:
            # 出方を渡さない呼び方は値だけを直すもの 新しい点は直線で置き、
            # 渡された制御点や曲線の名前は使わない（直線の点に持たせても効かない）
            if existing is not None:
                return replace(existing, value=self.value)
            return Keyframe(frame=self.frame, value=self.value)
        return Keyframe(
            frame=self.frame,
            value=self.value,
            interpolation=self.interpolation,
            control_points=self.control_points,
            curve=self.curve,
        )

    def apply(self, project: Project) -> Project:
        def update(current: ParamValue | None) -> ParamValue:
            animated = _as_animated(current)
            existing = next((k for k in animated.keyframes if k.frame == self.frame), None)
            keyframe = self._keyframe(existing)
            others = tuple(k for k in animated.keyframes if k.frame != self.frame)
            merged = tuple(sorted((*others, keyframe), key=lambda k: k.frame))
            return AnimatedValue(static=animated.static, keyframes=merged)

        return _update_param(project, self.path, update)


@dataclass(frozen=True, slots=True)
class RemoveKeyframe(Command):
    """指定フレームのキーフレームを消す

    最後の 1 つを消したときは、その値を静的値として残す 0 に戻ると、
    キーフレームを消した瞬間に絵が飛ぶ
    """

    path: ParamPath
    frame: int

    @property
    def label(self) -> str:
        return f"{self.path.name} のキーフレームを削除"

    def apply(self, project: Project) -> Project:
        def update(current: ParamValue | None) -> ParamValue:
            animated = _as_animated(current)
            remaining = tuple(k for k in animated.keyframes if k.frame != self.frame)
            if not remaining:
                return AnimatedValue(static=animated.at(self.frame))
            return AnimatedValue(static=animated.static, keyframes=remaining)

        return _update_param(project, self.path, update)


@dataclass(frozen=True, slots=True)
class MoveKeyframe(Command):
    """キーフレームを別のフレームへ動かし、値も変える

    グラフエディタで点をつまんで動かす操作 削除と設置の 2 手に分けると、
    ドラッグ 1 回で履歴が 2 段積まれる
    """

    path: ParamPath
    from_frame: int
    to_frame: int
    value: float

    @property
    def label(self) -> str:
        return f"{self.path.name} のキーフレームを移動"

    def apply(self, project: Project) -> Project:
        def update(current: ParamValue | None) -> ParamValue:
            animated = _as_animated(current)
            moving = next((k for k in animated.keyframes if k.frame == self.from_frame), None)
            if moving is None:
                raise KeyError(f"キーフレームが見つからない: {self.from_frame}")

            # 移動先に別の点があれば、それを置き換える 重なった 2 点は
            # モデル側の検査で弾かれる
            others = tuple(
                k for k in animated.keyframes if k.frame not in (self.from_frame, self.to_frame)
            )
            moved = replace(moving, frame=max(0, self.to_frame), value=self.value)
            merged = tuple(sorted((*others, moved), key=lambda k: k.frame))
            return AnimatedValue(static=animated.static, keyframes=merged)

        return _update_param(project, self.path, update)


@dataclass(frozen=True, slots=True)
class ClearKeyframes(Command):
    """アニメーションを解除し、その時点の値で固定する"""

    path: ParamPath
    frame: int = 0

    @property
    def label(self) -> str:
        return f"{self.path.name} のアニメーションを解除"

    def apply(self, project: Project) -> Project:
        def update(current: ParamValue | None) -> ParamValue:
            animated = _as_animated(current)
            return AnimatedValue(static=animated.at(self.frame))

        return _update_param(project, self.path, update)


@dataclass(frozen=True, slots=True)
class AddEffect(Command):
    """クリップにエフェクトを積む

    ``index`` が ``None`` なら、ふつうのエフェクトは最初の固定の項目（位置・反転・音量）の
    前へ入る（:func:`~sashimono.core.commands.fixed.loose_slot`） YMM4 と同じく、足した
    エフェクトを掛けてから置く 末尾へ積むと、足したぼかしが置いた後の絵に掛かり、
    YMM4 で同じ設定にした絵と違う
    固定の項目を足すとき（前の版のファイルで欄が無かった物を、触ったときに足す）は、
    固定の項目どうしの並び（反転 → 配置 → 音量 → フェード）の所へ入る
    """

    clip_id: ClipId
    effect: Effect
    index: int | None = None
    #: 場面切り替えの「後の場面」へ積むか
    after: bool = False
    #: 縁取りの層へ積むか（#273） 層の列は固定の項目を持たないので、位置を渡さなければ末尾
    #: どのエフェクトを層に掛けてよいかは定義を読む側（設定パネルと AI の道具）が決める
    #: コア層はエフェクトの定義を読まない
    stroke_id: StrokeId | None = None

    @property
    def label(self) -> str:
        return "エフェクトを追加"

    def apply(self, project: Project) -> Project:
        if self.stroke_id is not None and self.effect.fixed:
            raise ValueError("縁取りの層にクリップの欄（固定の項目）は積めない")

        def update(clip: Clip) -> Clip:
            effects = list(stack_of(clip, self.after, self.stroke_id))
            if self.index is not None:
                position = self.index
            elif self.stroke_id is not None:
                position = len(effects)
            elif self.effect.fixed:
                position = fixed_slot(effects, self.effect.kind)
            else:
                position = loose_slot(effects)
            effects.insert(position, self.effect)
            return with_stack(clip, self.after, tuple(effects), self.stroke_id)

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class RemoveEffect(Command):
    """エフェクトを外す 固定の項目（:attr:`Effect.fixed`）は断る

    固定の項目はクリップが最初から持つ欄で、外せると YMM4 と同じ並びのパネルが
    クリップごとに崩れる 効かせたくないときは無効にする
    """

    clip_id: ClipId
    effect_id: EffectId
    after: bool = False
    stroke_id: StrokeId | None = None

    @property
    def label(self) -> str:
        return "エフェクトを削除"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            stack = stack_of(clip, self.after, self.stroke_id)
            target = next((e for e in stack if e.id == self.effect_id), None)
            if target is None:
                raise KeyError(f"エフェクトが見つからない: {self.effect_id}")
            if target.fixed:
                raise ValueError(
                    f"{target.kind} はクリップが最初から持つ項目なので外せない 無効にはできる"
                )
            remaining = tuple(e for e in stack if e.id != self.effect_id)
            return with_stack(clip, self.after, remaining, self.stroke_id)

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class MoveEffect(Command):
    """エフェクトの順番を変える

    順番は結果に効く ぼかしてから色を変えるのと、色を変えてからぼかすのは
    別の絵になる

    固定の項目（:attr:`Effect.fixed`）は動かせず、ふつうのエフェクトが固定の項目を
    またぐ動きも断る 固定の項目どうしの並びは YMM4 の欄の並び（描画→動画→音声）で、
    間へ別のエフェクトを割り込ませると、どこまでが最初からある欄か見分けが付かなくなる
    """

    clip_id: ClipId
    effect_id: EffectId
    index: int
    after: bool = False
    stroke_id: StrokeId | None = None

    @property
    def label(self) -> str:
        return "エフェクトの順番を変更"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            effects = list(stack_of(clip, self.after, self.stroke_id))
            for position, effect in enumerate(effects):
                if effect.id == self.effect_id:
                    if effect.fixed:
                        raise ValueError(
                            f"{effect.kind} はクリップが最初から持つ項目なので並べ替えられない"
                        )
                    effects.pop(position)
                    destination = max(0, min(self.index, len(effects)))
                    # 抜いた後の列で、元の位置と行き先の間にある物が「またぐ」相手
                    low, high = sorted((position, destination))
                    if any(e.fixed for e in effects[low:high]):
                        raise ValueError("最初から持つ項目をまたいで並べ替えられない")
                    effects.insert(destination, effect)
                    return with_stack(clip, self.after, tuple(effects), self.stroke_id)
            raise KeyError(f"エフェクトが見つからない: {self.effect_id}")

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class SetEffectEnabled(Command):
    """エフェクトの有効・無効を切り替える

    消さずに切れるようにしておくと、掛ける前と後を見比べられる
    """

    clip_id: ClipId
    effect_id: EffectId
    enabled: bool
    after: bool = False
    stroke_id: StrokeId | None = None

    @property
    def label(self) -> str:
        return "エフェクトを有効化" if self.enabled else "エフェクトを無効化"

    def apply(self, project: Project) -> Project:
        def update(clip: Clip) -> Clip:
            effects = tuple(
                replace(e, enabled=self.enabled) if e.id == self.effect_id else e
                for e in stack_of(clip, self.after, self.stroke_id)
            )
            return with_stack(clip, self.after, effects, self.stroke_id)

        return _update_clip(project, self.clip_id, update)


@dataclass(frozen=True, slots=True)
class SetSource(Command):
    """生成オブジェクトを差し替える テキストや図形を置くときに使う"""

    clip_id: ClipId
    source: GeneratedSource | None

    @property
    def label(self) -> str:
        return "内容を変更"

    def apply(self, project: Project) -> Project:
        return _update_clip(project, self.clip_id, lambda clip: replace(clip, source=self.source))


@dataclass(frozen=True, slots=True)
class SetClipProperty(Command):
    """クリップ自身の設定を変える 合成方法や速度など"""

    clip_id: ClipId
    name: str
    value: object

    #: 変更を許す項目 任意の属性を書き換えられると、位置や長さを
    #: 検査なしで壊せてしまう
    #: ``hold_at`` はインスペクタの「解除」が ``None`` を書く 止める時刻の値そのものは
    #: 素材の中の時刻なので、検査はクリップ自身（負を断る）に任せる
    #: ``source_in``（再生開始位置）・``clip_to_below``（クリッピング）・``native_size``
    #: （素材の画素で置く）は設定パネルの描画・動画・音声の組から変える
    ALLOWED = (
        "blend_mode",
        "speed",
        "enabled",
        "stream_index",
        "hold_at",
        "source_in",
        "clip_to_below",
        "native_size",
    )

    @property
    def label(self) -> str:
        return f"クリップの {self.name} を変更"

    def apply(self, project: Project) -> Project:
        if self.name not in self.ALLOWED:
            raise ValueError(f"変更できない項目: {self.name}")
        if self.name == "hold_at" and not (self.value is None or isinstance(self.value, Fraction)):
            # 整数や小数を通すと、モデルには入るが保存の所で分数として書けずに落ちる
            # 保存できないプロジェクトを作るより、変える所で断る
            raise ValueError(f"絵を止める時刻は分数か None: {self.value!r}")
        if self.name in ("source_in", "speed") and not isinstance(self.value, Fraction | int):
            # 小数のまま入れると保存で分数に直せず落ちる（hold_at と同じ） 負や 0 の検査は
            # クリップ自身が行う
            raise ValueError(f"{self.name} は分数: {self.value!r}")
        if self.name in ("clip_to_below", "native_size", "enabled") and not isinstance(
            self.value, bool
        ):
            raise ValueError(f"{self.name} は真偽: {self.value!r}")
        return _update_clip(
            project, self.clip_id, lambda clip: _replace_named(clip, **{self.name: self.value})
        )


# --- 補助 -----------------------------------------------------------------

#: 項目名が実行時に決まる差し替え ``dataclasses.replace`` に静的な型は付かないので、
#: ここ 1 箇所で外す 呼び出し側は名前の妥当性を自分で確かめること
_replace_named = cast("Callable[..., Clip]", replace)


def _find_effect(clip: Clip, effect_id: EffectId | None) -> Effect | None:
    return next((e for e in clip.effects if e.id == effect_id), None)


def _as_animated(value: ParamValue | None) -> AnimatedValue:
    """数値パラメータをアニメーション値として扱う

    静的な数値にキーフレームを打つ操作を、特別扱いせずに書けるようにする
    """
    if isinstance(value, AnimatedValue):
        return value
    if isinstance(value, bool):
        return AnimatedValue(static=float(value))
    if isinstance(value, int | float):
        return AnimatedValue(static=float(value))
    return AnimatedValue()


def _update_clip(project: Project, clip_id: ClipId, update: Callable[[Clip], Clip]) -> Project:
    """クリップ 1 つを差し替える"""
    located = project.timeline.locate_clip(clip_id)
    if located is None:
        raise KeyError(f"クリップが見つからない: {clip_id}")
    track, clip = located

    updated = update(clip)
    others = tuple(c for c in track.clips if c.id != clip.id)
    return project.with_timeline(
        project.timeline.replace_track(track.with_clips((*others, updated)))
    )


def _update_param(
    project: Project, path: ParamPath, update: Callable[[ParamValue | None], ParamValue]
) -> Project:
    """パラメータ 1 つを差し替える 指す先ごとの違いをここに閉じ込める"""

    def change(clip: Clip) -> Clip:
        if path.target is ParamTarget.CLIP:
            if not hasattr(clip, path.name):
                raise KeyError(f"クリップにその項目は無い: {path.name}")
            return _replace_named(clip, **{path.name: update(getattr(clip, path.name))})

        if path.target is ParamTarget.SOURCE:
            if clip.source is None:
                raise KeyError("生成オブジェクトを持たないクリップ")
            if path.stroke_id is not None:
                stroke = stroke_of(clip, path.stroke_id)
                changed = stroke.with_param(path.name, update(stroke.params.get(path.name)))
                return with_stroke(clip, changed)
            current = clip.source.params.get(path.name)
            return replace(clip, source=clip.source.with_param(path.name, update(current)))

        stack = stack_of(clip, path.after, path.stroke_id)
        effect = next((e for e in stack if e.id == path.effect_id), None)
        if effect is None:
            raise KeyError(f"エフェクトが見つからない: {path.effect_id}")
        updated = effect.with_param(path.name, update(effect.params.get(path.name)))
        effects = tuple(updated if e.id == effect.id else e for e in stack)
        return with_stack(clip, path.after, effects, path.stroke_id)

    return _update_clip(project, path.clip_id, change)
