"""クリップが最初から持つ項目（固定の項目 :attr:`Effect.fixed`）の中身と並び

YMM4 のアイテムは、置いた時点で 描画（位置・拡大率・回転・左右反転）と 音声（音量・
パン・フェード）の欄を持つ こちらではその欄を印付きのエフェクトで表し、置いたときに付ける

並びは YMM4 と同じく **足したエフェクトの後ろ** YMM4 はエフェクトを掛けた絵を最後に
反転して置く（:func:`sashimono.compat.ymm4.template._video_chain`） 置いてから掛けると、
回した絵にぼかしが掛かって縁が画面の端で切れるなど、同じ設定でも絵が変わる
固定の項目どうしは 反転 → 配置 → 音量 → フェード（YMM4 の欄の並び 描画 → 音声）

コア層はエフェクトの定義（:mod:`sashimono.effects`）を読めないので、既定の値をここに
書く 定義の既定と食い違わないことは試験で見る
"""

from __future__ import annotations

from dataclasses import replace

from sashimono.core.model import FILTER_KIND, AnimatedValue, Clip, Effect, ParamValue

__all__ = [
    "FADE_EFFECT_KIND",
    "FIXED_ORDER",
    "FLIP_EFFECT_KIND",
    "LAST_KINDS",
    "LIMITER_EFFECT_KIND",
    "PICTURE_FIXED",
    "SOUND_FIXED",
    "TRANSFORM_EFFECT_KIND",
    "VOLUME_EFFECT_KIND",
    "fixed_effect",
    "fixed_rank",
    "fixed_slot",
    "loose_slot",
    "takes_picture_items",
    "with_fixed_items",
]

#: 左右反転（YMM4 の描画の「左右反転」）
FLIP_EFFECT_KIND = "flip"
#: 位置・拡大率・回転（YMM4 の描画の X・Y・拡大率・回転角）
TRANSFORM_EFFECT_KIND = "transform"
#: 音量とパン（YMM4 の音声の「音量」「パン」）
VOLUME_EFFECT_KIND = "audio_volume"
#: 音のフェード（YMM4 の音声の「フェードイン」「フェードアウト」）
FADE_EFFECT_KIND = "audio_fade"
#: リミッター（#414） 固定の項目ではないが、音量とフェードの**後**に積む 前に積むと、
#: 後から掛かる音量の欄で山が上限を越え、抑えたつもりの音が割れる
LIMITER_EFFECT_KIND = "audio_limiter"
#: 足すときに固定の項目の後（末尾）へ入る種類
LAST_KINDS = (LIMITER_EFFECT_KIND,)

#: 絵を持つクリップの固定の項目 反転してから置く（YMM4 は裏返してから回す）
PICTURE_FIXED = (FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND)
#: 音を持つクリップの固定の項目
SOUND_FIXED = (VOLUME_EFFECT_KIND, FADE_EFFECT_KIND)
#: 固定の項目どうしの並び 描画 → 音声
FIXED_ORDER = (*PICTURE_FIXED, *SOUND_FIXED)

#: 置いたときの値 どれも絵も音も変えない値にする 反転の定義の既定は「左右 真」
#: （足したら裏返ってほしいので）だが、最初から持つ欄は裏返さない値で持つ
_DEFAULTS: dict[str, tuple[tuple[str, float | bool | str], ...]] = {
    FLIP_EFFECT_KIND: (("horizontal", False), ("vertical", False)),
    TRANSFORM_EFFECT_KIND: (
        ("pos_x", 0.0),
        ("pos_y", 0.0),
        ("scale", 100.0),
        ("scale_y", 100.0),
        ("scale_x", 100.0),
        ("rotation", 0.0),
        ("rotation_x", 0.0),
        ("rotation_y", 0.0),
        ("anchor_x", 0.0),
        ("anchor_y", 0.0),
        ("pivot_h", "screen"),
        ("pivot_v", "screen"),
        ("move_to_pivot", False),
    ),
    VOLUME_EFFECT_KIND: (("volume", 100.0), ("pan", 0.0)),
    FADE_EFFECT_KIND: (("fade_in", 0.0), ("fade_out", 0.0)),
}


def _param(value: float | bool | str) -> ParamValue:
    # 数は動かせる値として持つ（定義の TrackSpec の既定と同じ形） 真偽と選択肢はそのまま
    if isinstance(value, bool | str):
        return value
    return AnimatedValue(static=value)


def fixed_effect(kind: str) -> Effect:
    """``kind`` の固定の項目を、絵も音も変えない値で作る"""
    defaults = _DEFAULTS.get(kind)
    if defaults is None:
        raise KeyError(f"固定の項目ではない: {kind}")
    return Effect(kind=kind, params={name: _param(value) for name, value in defaults}, fixed=True)


def fixed_rank(kind: str) -> int:
    """固定の項目どうしの並びでの順番 知らない種類は一番後ろ"""
    return FIXED_ORDER.index(kind) if kind in FIXED_ORDER else len(FIXED_ORDER)


def fixed_slot(effects: tuple[Effect, ...] | list[Effect], kind: str) -> int:
    """``kind`` の固定の項目を差し込む位置

    自分より後ろに並ぶはずの固定の項目の前 無ければ列の末尾 途中に割り込ませると、
    音量のあとに反転が来るなど、パネルの欄の並びとエフェクトの掛かる順が食い違う
    """
    rank = fixed_rank(kind)
    return next(
        (
            index
            for index, effect in enumerate(effects)
            if effect.fixed and fixed_rank(effect.kind) > rank
        ),
        len(effects),
    )


def loose_slot(effects: tuple[Effect, ...] | list[Effect]) -> int:
    """ふつうのエフェクトを足す位置 最初の固定の項目の前

    末尾へ足すと、足したぼかしが置いた後の絵に掛かり、YMM4 で同じ設定にした絵と変わる
    （回した絵の縁がぼけずに切れる 拡大した絵のぼかしが拡大率で太らない）
    """
    return next((index for index, effect in enumerate(effects) if effect.fixed), len(effects))


def takes_picture_items(clip: Clip) -> bool:
    """描画の欄（反転・配置）を持たせるクリップか

    フィルタは自分の絵を持たず下の絵へ掛けるだけで、場面切り替えは前と後の場面を
    自分で動かす どちらも置く位置を持たないので、欄を付けると効かない欄が並ぶ
    """
    if clip.source is None:
        return True
    return clip.source.kind not in (FILTER_KIND, "transition")


def with_fixed_items(clip: Clip, *, picture: bool = False, sound: bool = False) -> Clip:
    """足りない固定の項目を足したクリップ すでに印の付いた同じ種類があれば足さない

    ``picture`` は描画の欄（反転・配置）、``sound`` は音声の欄（音量・フェード）
    互換の読み込みがすでに付けた物（YMM4 の配置など）は、そこへ印を付けてから渡す
    同じ種類を 2 つ並べると、どちらがパネルの欄なのか分からなくなる
    """
    wanted = (*(PICTURE_FIXED if picture else ()), *(SOUND_FIXED if sound else ()))
    effects = list(clip.effects)
    for kind in wanted:
        if any(effect.fixed and effect.kind == kind for effect in effects):
            continue
        effects.insert(fixed_slot(effects, kind), fixed_effect(kind))
    if len(effects) == len(clip.effects):
        return clip
    return replace(clip, effects=tuple(effects))
