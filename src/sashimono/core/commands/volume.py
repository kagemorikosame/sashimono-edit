"""クリップの固定の音量（``audio_volume`` の ``volume`` と ``pan``）を読む・変える命令

タイムラインの音量の線（:mod:`sashimono.ui.timeline.value_line`）と AI の音量の道具
（#329）が同じ所を通る 入口ごとに書くと、固定の音量を持たない古いクリップの扱いが
入口で食い違う（片方だけ 2 段の取り消しになる など）
"""

from __future__ import annotations

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import AddEffect, ParamPath, SetParam
from sashimono.core.commands.fixed import VOLUME_EFFECT_KIND, fixed_effect, fixed_slot
from sashimono.core.model import AnimatedValue, Clip, Effect

__all__ = ["clip_volume", "fixed_volume", "volume_command", "volume_commands"]


def fixed_volume(clip: Clip) -> Effect | None:
    """クリップの固定の音量 まだ持たない古いクリップは ``None``"""
    return next((e for e in clip.effects if e.fixed and e.kind == VOLUME_EFFECT_KIND), None)


def clip_volume(clip: Clip) -> AnimatedValue:
    """クリップの音量の線（%） 固定の音量をまだ持たないクリップは 100%（音を変えない値）"""
    effect = fixed_volume(clip)
    value = effect.params.get("volume") if effect is not None else None
    if isinstance(value, AnimatedValue):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return AnimatedValue(static=float(value))
    return AnimatedValue(static=100.0)


def volume_commands(
    clip: Clip, volume: AnimatedValue | None, pan: AnimatedValue | None = None
) -> list[Command]:
    """音量の線と左右（``None`` は変えない）を丸ごと差し替える命令

    固定の音量をまだ持たない古いクリップは、値を入れた固定の音量を足す 1 つの命令にする
    足すのと値を変えるのを分けると、1 回戻すと 100% の音量調整だけが残る
    """
    if volume is None and pan is None:
        return []
    effect = fixed_volume(clip)
    if effect is None:
        # 差し込む位置は置いたときと同じ決まり（fixed_slot） 自前で決めると、固定の項目どうしの
        # 並び（反転 → 配置 → 音量 → フェード）が置いたクリップと食い違う
        added = fixed_effect(VOLUME_EFFECT_KIND)
        if volume is not None:
            added = added.with_param("volume", volume)
        if pan is not None:
            added = added.with_param("pan", pan)
        return [AddEffect(clip.id, added, index=fixed_slot(clip.effects, VOLUME_EFFECT_KIND))]
    commands: list[Command] = []
    if volume is not None:
        commands.append(SetParam(ParamPath.of_effect(clip.id, effect.id, "volume"), volume))
    if pan is not None:
        commands.append(SetParam(ParamPath.of_effect(clip.id, effect.id, "pan"), pan))
    return commands


def volume_command(clip: Clip, value: AnimatedValue) -> Command:
    """音量の線を丸ごと差し替える命令 1 つ（:func:`volume_commands` の音量だけ）"""
    (command,) = volume_commands(clip, value)
    return command
