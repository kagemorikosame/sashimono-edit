"""灰色の欄を使えるようにする切り替え（:class:`~sashimono.effects.sources.Unlock`）の命令

設定パネルのボタン（〔組み方を「標準」にする〕など #301）と AI の道具（#349）が同じ命令を
作るよう 1 か所に置く 別々に持つと、どのクリップを切り替えるか・どの値を入れるかが食い違う
"""

from __future__ import annotations

from collections.abc import Iterable

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import ParamPath, SetParam
from sashimono.core.model import Clip
from sashimono.effects.sources import SourceDefinition, Unlock

__all__ = ["unlock_commands"]


def unlock_commands(
    clips: Iterable[Clip], definition: SourceDefinition, unlock: Unlock
) -> tuple[list[Command], int]:
    """``clips`` のうち、``unlock`` で灰色が解ける物を切り替える命令と、切り替える本数

    値はクリップごとに作る 組み方と一緒に変える横の基準が、それぞれの行揃えで決まるため
    主のクリップの値を写すと、行揃えの違うクリップの字が跳ぶ 灰色になっていないクリップ
    （縦書きなど）は触らない 押したのは灰色を解くためで、効いている設定まで書き換えると、
    選んでいただけのクリップの見た目が変わる
    """
    commands: list[Command] = []
    switched = 0
    for clip in clips:
        source = clip.source
        if source is None or source.kind != definition.kind:
            continue
        reasons = definition.locked_reasons(source.params).values()
        if not any(definition.unlock_for(reason) == unlock for reason in reasons):
            continue
        switched += 1
        for name, value in unlock.switch(source.params).items():
            commands.append(SetParam(ParamPath.of_source(clip.id, name), value))
    return commands, switched
