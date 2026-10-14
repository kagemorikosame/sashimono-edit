"""テキストの塗り・縁取りの層・影を、グラデーションで塗る筆を作る（#312）

項目の意味は :mod:`sashimono.effects.fills` が決める ここは Qt の筆（``QLinearGradient``
``QRadialGradient``）へ直すだけ テキストの絵は CPU（``QPainter``）で描くので、グラデーションも
同じ所で塗る GPU のエフェクトにすると、塗りだけでなく縁取りと影まで巻き込む

座標は字を描く座標（画面の画素 Y は下が正） 画質を落としたプレビューでは描く座標ごと
縮めてある（:func:`sashimono.engine.sources.render_source_framed`）ので、筆も字の枠に合わせて
置けば、書き出しと同じ所に同じ色が出る
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QGradient, QLinearGradient, QRadialGradient

from sashimono.effects.spec import GradientSpec

__all__ = ["Paint", "gradient_brush", "paint_of"]

#: 止まりの並びを整える仕様 名前は使わない（整え方だけを借りる）
_STOPS = GradientSpec("gradient", "グラデーション")


@dataclass(frozen=True, slots=True)
class Paint:
    """グラデーションの塗り 1 つの、その時刻の値

    字の広がりの見積もり（:func:`sashimono.engine.sources._text_reach`）の鍵に入るので、
    比べられて覚えられる値だけを持つ
    """

    #: ``linear`` か ``radial``
    kind: str
    #: 止まりの並び（:class:`~sashimono.effects.spec.GradientSpec` の値）
    stops: tuple[float, ...]
    angle: float = 0.0
    length: float = 100.0
    shift: float = 0.0
    span: str = "extent"
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 100.0
    #: 掛け直す単位 ``box``（字の枠全体）・``line``（1 行ごと）・``char``（1 文字ごと）
    basis: str = "box"

    @property
    def visible(self) -> bool:
        """どこかに色が出るか 止まりがどれも透明なら塗っても何も出ない"""
        return any(colour[3] > 0.0 for _, colour in GradientSpec.stops(self.stops))


def _number(values: Mapping[str, object], name: str, default: float) -> float:
    """解いた値から数を読む 壊れた値（NaN や無限大・数でない物）は既定へ戻す"""
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def paint_of(values: Mapping[str, object], prefix: str) -> Paint | None:
    """``values``（時刻で解いた値）の ``prefix`` の塗り 単色なら ``None``

    単色のときは前からの色の項目で塗る（:data:`None` を返して呼ぶ側に任せる） 単色の道を
    グラデーションの筆に通すと、色が 1 つでも塗り方が変わり、前の作品の字が 1 画素ずれうる
    """
    kind = values.get(f"{prefix}_type")
    if kind not in ("linear", "radial"):
        return None
    span = values.get(f"{prefix}_span")
    raw = values.get(f"{prefix}_gradient")
    basis = values.get(f"{prefix}_basis")
    return Paint(
        kind=str(kind),
        stops=_STOPS.coerce(raw if isinstance(raw, tuple) else None),
        angle=_number(values, f"{prefix}_angle", 0.0),
        length=_number(values, f"{prefix}_length", 100.0),
        shift=_number(values, f"{prefix}_shift", 0.0),
        span=span if span in ("extent", "width", "pixels") else "extent",
        center_x=_number(values, f"{prefix}_center_x", 0.0),
        center_y=_number(values, f"{prefix}_center_y", 0.0),
        radius=_number(values, f"{prefix}_radius", 100.0),
        basis=basis if basis in ("box", "line", "char") else "box",
    )


def gradient_brush(paint: Paint, box: QRectF, opacity: float = 1.0) -> QBrush:
    """``box``（字の枠 描く座標）に合わせて置いた ``paint`` の筆

    ``opacity`` は止まりの不透明度に掛ける（縁取りの層の不透明度）

    線形 枠の真ん中を通る線の上で、角度の向きに長さぶん 0 から 1 へ進む 長さの基準は
    枠をその向きに測った長さ（100% で枠の端から端まで）か、枠の横の幅（YMM4 の割合の指定）
    ずらしは同じ基準の割合で、向きの先へ動かす
    円形 枠の真ん中から中心をずらし（枠の幅・高さに対する割合 Y は上が正）、半径 100% で
    枠の対角の半分（真ん中に置けば枠の角まで届く）
    止まりの外は端の色のまま（YMM4 の Clamp と同じ）
    """
    width = max(box.width(), 1e-3)
    height = max(box.height(), 1e-3)
    centre = box.center()
    gradient: QGradient
    if paint.kind == "radial":
        middle = QPointF(
            centre.x() + width * paint.center_x / 100.0,
            centre.y() - height * paint.center_y / 100.0,
        )
        reach = max(math.hypot(width, height) / 2.0 * paint.radius / 100.0, 1e-3)
        # 中心をほんの少しずらす 画素の真ん中と円の中心がちょうど重なると、Qt はその 1 画素を
        # 塗らずに透明な穴を空ける（字の枠の真ん中が画素の真ん中に来る字で、中心に点が抜けた）
        # 1/97 画素なら色は 8 bit で変わらず、画素の真ん中（x.5）にちょうど乗ることも無い
        middle += QPointF(1.0 / 97.0, 1.0 / 97.0)
        gradient = QRadialGradient(middle, reach)
    else:
        radians = math.radians(paint.angle)
        # 「グラデーション」エフェクトと同じ向き 0 で下へ、90 で左へ（画面の上で時計回り）
        dx, dy = -math.sin(radians), math.cos(radians)
        if paint.span == "pixels":
            # 長さとずらしをそのまま画素で数える（AviUtl2 の幅） 描く座標なのでプレビューでも縮む
            basis = 100.0
        elif paint.span == "width":
            basis = width
        else:
            basis = abs(width * dx) + abs(height * dy)
        unit = max(basis, 1e-3) / 100.0
        half = max(paint.length * unit, 1e-3) / 2.0
        middle_x = centre.x() + dx * paint.shift * unit
        middle_y = centre.y() + dy * paint.shift * unit
        gradient = QLinearGradient(
            QPointF(middle_x - dx * half, middle_y - dy * half),
            QPointF(middle_x + dx * half, middle_y + dy * half),
        )
    gradient.setSpread(QGradient.Spread.PadSpread)
    # 色と不透明度を別々に混ぜる（YMM4・AviUtl2・Photoshop と同じ） Qt の既定は事前乗算で
    # 混ぜ、透明な止まりの色を無視する 赤から透明な青へのグラデーションの真ん中が、
    # YMM4 では紫になるのに Qt の既定では赤のまま薄くなる
    gradient.setInterpolationMode(QGradient.InterpolationMode.ComponentInterpolation)
    strength = min(max(opacity, 0.0), 1.0)
    # 並びごと渡す ``setColorAt`` は同じ位置の止まりを前へ差し込むので、位置の重なった
    # 止まり（その位置で色が切り替わる段）の色が入れ替わる
    gradient.setStops(
        [
            (
                position,
                QColor(
                    round(red * 255),
                    round(green * 255),
                    round(blue * 255),
                    round(alpha * strength * 255),
                ),
            )
            for position, (red, green, blue, alpha) in GradientSpec.stops(paint.stops)
        ]
    )
    return QBrush(gradient)
