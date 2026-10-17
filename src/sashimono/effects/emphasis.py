"""強調の動き（#331）を命令にする所 AI の道具とタイムラインの右クリックが同じ所を通る

線の形は :mod:`sashimono.core.emphasis` が組む ここはエフェクトの定義を読む側の仕事
（揺れに使うランダム移動を定義の既定で作る・欄の範囲を超えないかを見る）を足して命令にする
入口ごとに書くと、AI が付けた動きと人が右クリックで付けた動きが食い違う
"""

from __future__ import annotations

from dataclasses import dataclass

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import AddEffect
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, fixed_effect
from sashimono.core.emphasis import (
    EMPHASIS_KIND_NAMES,
    PAN,
    SHAKE,
    ZOOM,
    EmphasisError,
    Envelope,
    envelope,
    fixed_transform,
    pan_curves,
    ramp_frames,
    shake_curve,
    transform_commands,
    zoom_curves,
)
from sashimono.core.model import AnimatedValue, Clip
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.effects.spec import TrackSpec

__all__ = ["SHAKE_EFFECT_KIND", "EmphasisRequest", "emphasis_commands", "emphasis_label"]

#: 揺れに使う動きのエフェクト 設定パネルの〔エフェクトを追加…〕→ 動き →〔ランダム移動〕と同じ物
#: 揺れのために新しい種類を作らないのは、人が設定パネルで幅や間隔を直せるようにするため
SHAKE_EFFECT_KIND = "random_move"


@dataclass(frozen=True, slots=True)
class EmphasisRequest:
    """付ける動き 1 つ ``start``・``end`` はクリップの頭からのフレーム

    ``strength`` は種類ごとの強さ（zoom と pop は拡大率の %、shake は揺れの幅の画素）
    pan は ``offset``（画素 Y は上が正）を使い ``strength`` は見ない
    ``keep`` のときは ``end`` を見ない（クリップの終わりまで残す）
    """

    kind: str
    start: int
    end: int
    strength: float = 0.0
    ramp_ms: int = 250
    easing: str = "smooth"
    focus: tuple[float, float] | None = None
    offset: tuple[float, float] = (0.0, 0.0)
    keep: bool = False


def emphasis_label(kind: str) -> str:
    """取り消しの段の名前 AI と右クリックで同じにする（履歴で見分けが付かなくならないよう）"""
    return f"強調の動き: {EMPHASIS_KIND_NAMES.get(kind, kind)}"


def emphasis_commands(
    clip: Clip, request: EmphasisRequest, rate: FrameRate
) -> tuple[list[Command], Envelope]:
    """``clip`` に ``request`` の動きを付ける命令と、その強さの形

    組めなければ :class:`EmphasisError`
    """
    shape = envelope(
        request.kind,
        request.start,
        request.end,
        ramp_frames(request.ramp_ms, rate),
        request.easing,
        keep=request.keep,
    )
    transform = fixed_transform(clip) or fixed_effect(TRANSFORM_EFFECT_KIND)
    if not transform.enabled:
        # 無効の配置は描くときに飛ばされるので、点を打っても動かない
        raise EmphasisError(
            "このクリップの配置（位置・拡大率）の欄が無効です 有効にしてから付けてください"
        )
    if request.kind == SHAKE:
        return [_shake_command(clip, shape, request.strength, request.start)], shape
    if request.kind == PAN:
        curves = pan_curves(transform, shape, request.offset)
    else:
        focus = request.focus if request.kind == ZOOM else None
        curves = zoom_curves(transform, shape, request.strength / 100.0, focus)
    _within_spec(curves)
    return transform_commands(clip, curves), shape


def _within_spec(curves: dict[str, AnimatedValue]) -> None:
    """線の値が配置の欄の範囲（設定パネルと同じ）に収まるかを見る 収まらなければ断る

    端へ寄せると、寄せた所だけ動きが止まり、ズームの寄る先がずれる
    """
    definition = registry.get(TRANSFORM_EFFECT_KIND)
    for name, curve in curves.items():
        spec = definition.spec(name) if definition is not None else None
        if not isinstance(spec, TrackSpec):
            continue
        for keyframe in curve.keyframes:
            if not spec.minimum <= keyframe.value <= spec.maximum:
                raise EmphasisError(
                    f"{name} が欄の範囲（{spec.minimum:g}〜{spec.maximum:g}）を超えます"
                    f"（{keyframe.value:.1f}） 強さを下げてください"
                )


def _shake_command(clip: Clip, shape: Envelope, amplitude: float, start: int) -> Command:
    """揺れ 範囲の外では幅 0 のランダム移動を足す命令

    ランダム移動はクリップ全体に掛かるので、幅の線で範囲の外を止める 種は範囲の頭から
    取る 同じクリップへ 2 度揺れを付けたとき、同じ揺れ方が繰り返されて機械的に見えないように
    """
    definition = registry.get(SHAKE_EFFECT_KIND)
    if definition is None:
        raise EmphasisError(f"揺れに使うエフェクト（{SHAKE_EFFECT_KIND}）が見つかりません")
    curve = shake_curve(shape, amplitude)
    effect = definition.create(seed=start % 10000)
    effect = effect.with_param("range_x", curve).with_param("range_y", curve)
    return AddEffect(clip.id, effect)
