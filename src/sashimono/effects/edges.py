"""絵のまっすぐな縁が、エフェクトの並びの途中でどこにあるか（#493）

変形は透明な地に置いた絵を地ごと拡大する 縁が画面の端に重なったとき、縁の外の透明を拾わない
ためには縁の場所が要る 画素の不透明度だけからは読めない 半端な画素へ置いた絵（奇数の大きさ・
画質を落としたプレビュー）は塗られた画素が本当の縁から半画素ずれ、前の拡大でなまった縁は
どこが縁かの手掛かりが残らない そこで置いた所から、縁を動かすエフェクトの分だけ数で追う

``u_object`` を使い回さない あちらは互換の都合で変形では動かさない（中心基準の動きや切り抜きが
元の範囲を基準にする）ので、書き換えると読み込んだテンプレートの見た目が変わる
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

__all__ = [
    "EdgeStep",
    "MoveEdges",
    "PictureEdges",
    "cropped_edges",
    "flipped_edges",
    "placed_edges",
    "transformed_edges",
]

_Box = tuple[float, float, float, float]

#: 画素の真ん中に乗ったかを見るときの遊び 計算の誤差で 1 画素隣を縁と取り違えないため
_SLACK = 0.001


@dataclass(frozen=True, slots=True)
class PictureEdges:
    """絵の縁（合成の画素 左・下・右・上 Y は上が正）"""

    box: _Box
    #: 縁から何画素離れれば、透明とも中身とも混ざっていないか（横・縦）
    soft: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EdgeStep:
    """エフェクト 1 つぶんの、縁を動かす手掛かり 値はシェーダへ渡す物と同じ読み方"""

    number: Callable[[str], float]
    checked: Callable[[str], bool]
    #: 変形の支点（合成の画素） 支点を持たないエフェクトでは使わない
    pivot: Callable[[], tuple[float, float]]
    #: 絵の置かれた範囲（``u_object``）の中央
    center: tuple[float, float]


#: エフェクトの後の縁を返す関数 追えなくなったら ``None``
type MoveEdges = Callable[[PictureEdges, EdgeStep], PictureEdges | None]


def placed_edges(box: _Box, rect: _Box, source: tuple[int, int]) -> PictureEdges:
    """``source`` の大きさの絵を ``rect`` へ置いた直後の縁 ``box`` は中身のある範囲"""

    def softness(axis: int) -> float:
        low, high = rect[axis], rect[axis + 2]
        zoom = abs(high - low) / max(source[axis], 1)
        # 置いた枠そのものの縁は四角で切れるのでなまらない なまると見ると、端から少し
        # はみ出した絵（拡大率 301 など）の外周の中身まで読む位置が動く
        on_rim = abs(box[axis] - low) < _SLACK and abs(box[axis + 2] - high) < _SLACK
        if on_rim or abs(zoom - 1.0) < _SLACK:
            return 0.5
        # 枠の中の縁（透明な余白のある絵） 伸ばして置けば補間の幅の半分、縮めて置けば
        # 平均した範囲の分だけなまる
        return zoom / 2.0 if zoom > 1.0 else (1.0 + zoom) / 2.0

    return PictureEdges(box, (softness(0), softness(1)))


def cropped_edges(edges: PictureEdges, step: EdgeStep) -> PictureEdges:
    """切り抜きで内へ入った縁 ぼかして切った縁は本物の濃淡なので追わない"""
    if step.number("feather") > 0.0:
        return edges
    left, bottom, right, top = edges.box
    box = (
        left + max(step.number("left"), 0.0),
        bottom + max(step.number("bottom"), 0.0),
        right - max(step.number("right"), 0.0),
        top - max(step.number("top"), 0.0),
    )
    return replace(edges, box=box)


def flipped_edges(edges: PictureEdges, step: EdgeStep) -> PictureEdges:
    """反転で入れ替わった縁 片側だけ切った絵は、裏返すと縁の場所が変わる"""
    left, bottom, right, top = edges.box
    x, y = step.center
    if step.checked("horizontal"):
        left, right = 2.0 * x - right, 2.0 * x - left
    if step.checked("vertical"):
        bottom, top = 2.0 * y - top, 2.0 * y - bottom
    return replace(edges, box=(left, bottom, right, top))


def transformed_edges(edges: PictureEdges, step: EdgeStep) -> PictureEdges | None:
    """変形の後の縁 シェーダ（``builtin.py`` の ``_TRANSFORM``）の逆算の裏返し

    縦でも横でもなくなった縁（傾き・90 度の倍数でない回転）は追わない
    """
    if step.number("rotation_x") != 0.0 or step.number("rotation_y") != 0.0:
        return None
    angle = math.radians(-step.number("rotation"))
    c, s = math.cos(angle), math.sin(angle)
    if min(abs(c), abs(s)) > 1e-5:
        return None
    sideways = abs(c) <= 1e-5

    whole = max(step.number("scale") / 100.0, 0.0001)
    zoom = (
        max(step.number("scale_x") / 100.0, 0.0001) * whole,
        max(step.number("scale_y") / 100.0, 0.0001) * whole,
    )
    anchor = step.pivot()
    shift = [step.number("pos_x"), step.number("pos_y")]
    if step.checked("move_to_pivot"):
        shift = [shift[axis] - (anchor[axis] - step.center[axis]) for axis in (0, 1)]

    def moved(x: float, y: float) -> tuple[float, float]:
        x, y = (x - anchor[0]) * zoom[0], (y - anchor[1]) * zoom[1]
        return (anchor[0] + shift[0] + c * x - s * y, anchor[1] + shift[1] + s * x + c * y)

    first, second = moved(*edges.box[:2]), moved(*edges.box[2:])
    box = (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[0], second[0]),
        max(first[1], second[1]),
    )

    # 縮めるときに読む点の散らばり（:data:`~sashimono.effects.sampling.AREA_SAMPLING` と同じ数）
    count = min(max(math.ceil(1.0 / min(zoom) - _SLACK), 1), 8)
    soft = []
    for axis in (0, 1):
        # 読んだ所が混ざらないのは、縁から離れた最初の画素の真ん中より奥 そこまでが次の縁の幅
        reach = max(
            _clear_of(edges.box[axis], edges.soft[axis]),
            _clear_of(-edges.box[axis + 2], edges.soft[axis]),
        )
        spread = (1.0 - 1.0 / count) / (2.0 * zoom[axis])
        soft.append(max((reach + spread) * zoom[axis], 0.5))
    return PictureEdges(box, (soft[1], soft[0]) if sideways else (soft[0], soft[1]))


def _clear_of(edge: float, soft: float) -> float:
    """縁から、混ざっていない最初の画素の真ん中までの長さ（大きい方が内）"""
    return math.ceil(edge + soft - 0.5 - _SLACK) + 0.5 - edge
