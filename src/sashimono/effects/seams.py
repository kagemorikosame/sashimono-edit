"""ジャンプカットのつなぎ目（#332）に置く場面切り替えの中身

コア（:mod:`sashimono.core.seams`）はエフェクトの定義を読めないので、置く中身はここで作って渡す
AI の道具と、タイムラインの右クリック・無音カットの窓が同じ所を通る 入口ごとに作ると、
片方だけ切り替え方の既定の値が違う場面切り替えが置かれる
"""

from __future__ import annotations

from sashimono.core.model import GeneratedSource
from sashimono.core.seams import SEAM_CROSSFADE, SeamStyle
from sashimono.effects.sources import TRANSITION

__all__ = ["seam_transition"]


def seam_transition(style: SeamStyle) -> GeneratedSource | None:
    """クロスフェードに置く場面切り替え 人が〔場面切り替えを追加〕で置く物と同じ既定の中身

    クロスフェードでなければ ``None``（置かない）
    """
    return TRANSITION.create(style="fade") if style.video == SEAM_CROSSFADE else None
