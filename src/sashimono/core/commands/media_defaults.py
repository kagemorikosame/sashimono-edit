"""素材の置き方の既定（#524） クリップから集めて素材に覚え、次に置くときの初めの値にする

覚えるのは固定の項目（:mod:`sashimono.core.commands.fixed` の反転・配置・音量・フェード）と、
不透明度・画素で置くか・素材の頭のずれだけ ふつうのエフェクトは覚えない（一度だけ掛けた
ぼかしが、以後置くたびに付いてくるため）

集めるのはリンクした仲間（同じ素材を絵と音に分けて置いた組）も含める 分ける方式では
絵と音が別のクリップなので、絵のクリップで〔既定にする〕を選んだときに音量を取りこぼすと、
「覚えたはずの音量が効かない」になる
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.commands.base import Command
from sashimono.core.commands.fixed import (
    PICTURE_FIXED,
    SOUND_FIXED,
    fixed_slot,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    MediaDefaults,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    Track,
    new_effect_id,
)
from sashimono.core.model.timeline import heard_stream

__all__ = [
    "SetMediaDefaults",
    "defaults_from_clip",
    "placed_offset",
    "with_picture_defaults",
    "with_sound_defaults",
]


def _still(value: ParamValue) -> ParamValue:
    # クリップの頭の値で止める 長さの違うクリップへキーフレームを写すと、終わりの側が
    # 外へはみ出して効かないか、短いクリップで途中までしか動かない
    if isinstance(value, AnimatedValue) and value.is_animated:
        return AnimatedValue(static=value.at(0))
    return value


def _remembered(effect: Effect) -> Effect:
    return replace(effect, params={name: _still(value) for name, value in effect.params.items()})


def _fixed_of(clip: Clip, kinds: tuple[str, ...]) -> tuple[Effect, ...]:
    return tuple(_remembered(e) for e in clip.effects if e.fixed and e.kind in kinds)


def _partners(project: Project, track: Track, clip: Clip) -> list[tuple[Track, Clip]]:
    """``clip`` と、同じ素材でリンクした仲間 選んだクリップを先頭に置く"""
    if clip.link_group is None:
        return [(track, clip)]
    others = [
        (t, c)
        for t, c in project.timeline.linked_clips(clip.link_group)
        if c.id != clip.id and c.media_id == clip.media_id
    ]
    return [(track, clip), *others]


def defaults_from_clip(project: Project, clip_id: ClipId) -> tuple[MediaId, MediaDefaults]:
    """開いているタイムラインの ``clip_id`` とリンクした仲間から、素材の置き方の既定を集める

    素材を持たないクリップ（テキスト・図形・シーンなど）と、覚える物が何も無いクリップは
    :class:`ValueError` 黙って空の既定を覚えると、本人は覚えたつもりで何も効かない
    """
    found = project.timeline.locate_clip(clip_id)
    if found is None:
        raise ValueError(f"クリップが見つからない: {clip_id}")
    track, clip = found
    if clip.media_id is None:
        raise ValueError(
            "素材を置いたクリップではない（テキスト・図形・シーンは素材の既定を持たない）"
        )
    media = project.require_media(clip.media_id)

    picture: tuple[Effect, ...] = ()
    opacity: float | None = None
    native_size: bool | None = None
    sounds: dict[int, tuple[Effect, ...]] = {}
    for t, c in _partners(project, track, clip):
        if (media.has_video or media.is_still) and not picture and project.draws_picture(t, c):
            picture = _fixed_of(c, PICTURE_FIXED)
            opacity = min(1.0, max(0.0, c.opacity.at(0)))
            native_size = c.native_size
        stream = heard_stream(t, c)
        if media.has_audio and stream is not None and project.plays_sound(t, c):
            effects = _fixed_of(c, SOUND_FIXED)
            if effects and stream not in sounds:
                sounds[stream] = effects

    # 頭のずれは動く素材だけ 静止画は素材の中の位置を持たない 範囲だけを入れた素材は
    # 入っている区間に合わせて置くので、ずれを覚えても置くときに使わない
    offset = clip.source_in if clip.source_in > 0 and not media.is_still else None
    defaults = MediaDefaults(
        picture=picture,
        opacity=opacity,
        native_size=native_size,
        sounds=tuple(sorted(sounds.items())),
        offset=offset,
    )
    if defaults.is_empty:
        raise ValueError(f"素材 {media.name!r} のクリップに覚える項目が無い")
    return media.id, defaults


def _applied(clip: Clip, remembered: tuple[Effect, ...]) -> Clip:
    """``clip`` の固定の項目を、覚えた値で差し替える 覚えていない種類はそのまま"""
    if not remembered:
        return clip
    effects = list(clip.effects)
    for wanted in remembered:
        # 写しごとに新しい ID を振る 同じ ID のエフェクトが何本ものクリップに並ぶと、
        # ID でエフェクトを指す命令（値の変更・キーフレーム）が別のクリップの物を掴む
        fresh = replace(wanted, id=new_effect_id(), params=dict(wanted.params))
        at = next(
            (i for i, e in enumerate(effects) if e.fixed and e.kind == wanted.kind),
            None,
        )
        if at is None:
            effects.insert(fixed_slot(effects, wanted.kind), fresh)
        else:
            effects[at] = fresh
    return replace(clip, effects=tuple(effects))


def with_picture_defaults(clip: Clip, defaults: MediaDefaults | None) -> Clip:
    """絵のクリップへ、覚えた描画の欄・不透明度・画素で置くかを当てる"""
    if defaults is None:
        return clip
    placed = _applied(clip, defaults.picture)
    if defaults.opacity is not None:
        placed = replace(placed, opacity=AnimatedValue(defaults.opacity))
    if defaults.native_size is not None:
        placed = replace(placed, native_size=defaults.native_size)
    return placed


def with_sound_defaults(clip: Clip, defaults: MediaDefaults | None, stream: int) -> Clip:
    """音のクリップ（音声ストリーム ``stream``）へ、覚えた音声の欄を当てる"""
    if defaults is None:
        return clip
    return _applied(clip, defaults.sound_for(stream))


def placed_offset(media: MediaItem) -> Fraction:
    """素材を全体で置くときの、素材の中の開始位置（秒） 覚えていなければ 0

    素材の長さ以上のずれは使わない 置いても何も映らず何も鳴らない長さ 0 のクリップになる
    （素材を短い物へ差し替えたのに、同じ名前の既定が残ったときに起きる）
    範囲だけを入れた素材と静止画にも使わない（集める側の :func:`defaults_from_clip` を見る）
    """
    defaults = media.defaults
    if defaults is None or defaults.offset is None or media.packed or media.is_still:
        return Fraction(0)
    if defaults.offset >= media.duration:
        return Fraction(0)
    return defaults.offset


@dataclass(frozen=True, slots=True)
class SetMediaDefaults(Command):
    """素材の置き方の既定を覚える ``None`` なら消す

    素材に覚えるだけで、置いてあるクリップは変えない 既に置いた物まで書き換えると、
    場面ごとに位置を変えて置いた立ち絵が全部同じ位置へ動く
    """

    media_id: MediaId
    defaults: MediaDefaults | None

    @property
    def label(self) -> str:
        return "素材の既定を消す" if self.defaults is None else "素材の既定を覚える"

    def apply(self, project: Project) -> Project:
        item = project.require_media(self.media_id)
        kept = None if self.defaults is None or self.defaults.is_empty else self.defaults
        return project.replace_media(replace(item, defaults=kept))
