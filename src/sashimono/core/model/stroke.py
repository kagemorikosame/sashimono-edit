"""テキストの縁取りの層（#272 #273）

テキストは縁取りを、順に並べた層として持てる（Photoshop のレイヤースタイルの境界線を
いくつも足すのと同じ） 並びの頭が一番上（字の塗りに近い側）で、後ろほど下に描く
Photoshop の一覧と同じ向きにしておくと、上へ動かす・下へ動かすが見た目の上下と合う

太さは層ごとに字の輪郭から数える 前の層の外側から数えると、内側の層を細くしただけで
外側の層まで内へ寄り、1 つの層だけを直すことができない

層を 1 つも持たないテキストは、前からある 1 組の項目（``border_width`` ``border_color``）を
縁取りとして描く 前の版のプロジェクト・エイリアス・プリセット・AviUtl と YMM4 の読み込みは
どれもこの形で、層を足すまでは見た目が 1 画素も変わらない
層を持つテキストは前からの項目を読まない 層へ移すとき（:func:`legacy_stroke`）に、
命令の側で太さを 0 にする 残すと、層を全部消したときに前の縁がまた出てくる
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from sashimono.core.model.effect import AnimatedValue, Effect, ParamValue
from sashimono.core.model.ids import StrokeId, new_stroke_id

__all__ = [
    "LEGACY_BORDER_COLOR",
    "LEGACY_BORDER_WIDTH",
    "MAX_STROKES",
    "Stroke",
    "legacy_in_use",
    "legacy_stroke",
]

#: 1 つのテキストに足せる層の数 層 1 つごとに字の輪郭から縁を作り直すので、増やすほど
#: 字幕を作り直す時間が延びる YMM4 の実物（88 本・テキスト 227 個）では 1 つの字に縁取りが
#: 多くて 2 つで、8 あれば手で重ねる飾りにも足りる
MAX_STROKES = 8

#: 層を持たないテキストの縁取りの太さと色（:data:`sashimono.effects.sources.TEXT` の項目）
LEGACY_BORDER_WIDTH = "border_width"
LEGACY_BORDER_COLOR = "border_color"

#: 色の項目が無いときの縁取りの色 テキストの定義の既定（黒）と同じ
#: コア層はエフェクトの定義を読まないので、ここにも書く
_LEGACY_DEFAULT_COLOR: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Stroke:
    """縁取りの層 1 つ

    ``params`` の項目と範囲は :data:`sashimono.effects.strokes.STROKE` が決める
    ``effects`` はこの層の絵だけに掛けるエフェクト（#273） 層を消すと一緒に消える
    """

    params: dict[str, ParamValue] = field(default_factory=dict)
    enabled: bool = True
    effects: tuple[Effect, ...] = ()
    id: StrokeId = field(default_factory=new_stroke_id)

    def with_param(self, name: str, value: ParamValue) -> Stroke:
        """項目を 1 つ差し替えた層 ``replace`` で作るので、後で足す欄も落ちない"""
        return replace(self, params={**self.params, name: value})


def legacy_in_use(params: Mapping[str, ParamValue]) -> bool:
    """層を持たないテキストが、前からの項目で縁取りを描くか

    キーフレームで動く太さは、途中で 0 でなくなるので使っている側に数える
    """
    width = params.get(LEGACY_BORDER_WIDTH)
    if isinstance(width, AnimatedValue):
        return width.is_animated or width.static > 0.0
    if isinstance(width, bool):
        return False
    if isinstance(width, int | float):
        return width > 0
    return False


def legacy_stroke(params: Mapping[str, ParamValue], stroke_id: StrokeId | None = None) -> Stroke:
    """前からの項目（太さと色）を写した層 位置は外側・角は丸で、前と同じ絵になる

    太さのキーフレームもそのまま持っていく 写した層が前の縁と違う動きをすると、
    層を足しただけで縁の動きが変わる
    """
    width = params.get(LEGACY_BORDER_WIDTH)
    colour = params.get(LEGACY_BORDER_COLOR)
    if isinstance(width, bool) or not isinstance(width, AnimatedValue | int | float):
        width = AnimatedValue(0.0)
    elif not isinstance(width, AnimatedValue):
        width = AnimatedValue(float(width))
    if not isinstance(colour, tuple):
        colour = _LEGACY_DEFAULT_COLOR
    stroke = Stroke(params={"width": width, "color": colour})
    return stroke if stroke_id is None else replace(stroke, id=stroke_id)
