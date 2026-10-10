"""テキストの縁取りの層の項目と、層に掛けられるエフェクト（#272 #273）

層そのもの（並び・入り切り・掛けたエフェクト）は :class:`sashimono.core.model.Stroke`
ここは層が持つ項目の定義（設定パネルの欄・AI の道具・描く所が同じ物を読む）と、
層の絵に掛けてよいエフェクトの決まりを持つ
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from sashimono.core.model import (
    LEGACY_BORDER_COLOR,
    LEGACY_BORDER_WIDTH,
    AnimatedValue,
    Effect,
    GeneratedSource,
    ParamValue,
    Stroke,
    legacy_in_use,
)
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.sources import SourceDefinition
from sashimono.effects.spec import ColorSpec, SelectSpec, TrackSpec

__all__ = [
    "JOINS",
    "POSITIONS",
    "STROKE",
    "STROKE_EFFECT_KINDS",
    "STROKE_EFFECT_NOTE",
    "next_stroke",
    "pixel_reach",
    "stroke_effect_definitions",
    "takes_stroke_effect",
]

#: 線を引く所 外側は前からの縁取りと同じ（字の輪郭から外へ太さぶん）
#: 内側は字の塗りの内側だけ、中央は輪郭をまたいで半分ずつ（Photoshop の境界線と同じ 3 つ）
POSITIONS = (("outside", "外側"), ("center", "中央"), ("inside", "内側"))

#: 線の角 丸は前からの縁取りと同じ 角は尖らせ、面取りは角を斜めに落とす
JOINS = (("round", "丸"), ("miter", "角"), ("bevel", "面取り"))

#: 層の項目 太さの上限は前からの縁取り（64）より広くする 縁を何重にも重ねると、外側の層は
#: 字の輪郭から内側の層の太さの和だけ離れる（YMM4 の 2 つ重ねで 15 画素）
STROKE = SourceDefinition(
    kind="stroke",
    label="縁取り",
    parameters=(
        TrackSpec("width", "太さ", 0, 256, 4, step=1, unit="px"),
        ColorSpec("color", "色", (0.0, 0.0, 0.0, 1.0)),
        SelectSpec("position", "位置", POSITIONS, "outside"),
        TrackSpec("opacity", "不透明度", 0, 100, 100, step=1, unit="%"),
        SelectSpec("join", "角", JOINS, "round"),
    ),
)

#: 縁取りの層の絵に掛けてよいエフェクト
#:
#: 層の絵だけで結果が決まり、時刻でも変わらない物に限る（#273 の「まず対象を絞る」）
#: 層の絵はテキストの絵を作るときに 1 度だけ掛けて覚えるので、時刻で動く物（ノイズ・
#: 登場の動き・揺れ）を許すと、止まった字幕では動かず、動く字幕では毎フレーム作り直しになる
#: 下の絵を読む物（画像合成・ディスプレイスメント）や、クリップ全体の位置で動く物（変形・
#: 部分フィルタ）は、層の絵の中で掛けても思った所に効かない
STROKE_EFFECT_KINDS: tuple[str, ...] = (
    "blur",
    "glow",
    "lens_blur",
    "directional_blur",
    "border_blur",
    "sharpen",
    "opacity",
    "fill",
    "color",
    "color_correct",
    "tint",
    "invert",
    "exposure",
    "color_grade",
    "gradient",
    "gradient_map",
    "mosaic",
    "shadow",
    "border",
    "inner_shadow",
    "inner_outline",
    "bevel_light",
    "morphology",
)

#: 層に掛けられない物があることの断り 設定パネルのメニューと AI の道具の答えに出す
STROKE_EFFECT_NOTE = (
    "縁取りの層には、層の絵だけで決まり時間で変わらないエフェクト（ぼかし・グロー・色・"
    "グラデーションなど）を掛けられます 時間で動く物や、下の絵・クリップの位置を使う物は、"
    "クリップのエフェクトとして足してください"
)


def takes_stroke_effect(kind: str) -> bool:
    """``kind`` のエフェクトを縁取りの層に掛けてよいか"""
    return kind in STROKE_EFFECT_KINDS and registry.get(kind) is not None


def stroke_effect_definitions() -> list[EffectDefinition]:
    """層に掛けられるエフェクトの定義 一覧（:data:`STROKE_EFFECT_KINDS`）の順"""
    found = (registry.get(kind) for kind in STROKE_EFFECT_KINDS)
    return [definition for definition in found if definition is not None]


def pixel_reach(effects: Iterable[Effect], frame: int) -> float:
    """``effects`` が絵を外へ運びうる量（画面の画素）

    画素で決める項目（影のずれ・ぼかしの範囲・縁取りの太さなど）はどれもその値より遠くへは
    絵を運ばない 入れ物を広げる効果（``expands_object``）は広げる量のうち大きい方 順に
    掛かるので、効果ごとの量を足す 定義の無い物と切ってある物は数えない
    """
    reach = 0.0
    for effect in effects:
        definition = registry.get(effect.kind)
        if definition is None or not effect.enabled:
            continue
        grows = set(definition.expands_object or ())
        growth = 0.0
        for spec in definition.parameters:
            if not isinstance(spec, TrackSpec) or not (spec.in_pixels or spec.name in grows):
                continue
            amount = abs(_pixels(spec, effect, frame))
            if not math.isfinite(amount):
                continue
            if spec.name in grows:
                # 上と下は別の側へ広げる 足すと 1 つの側に要る量の倍を取る
                growth = max(growth, amount)
            else:
                reach += amount
        reach += growth
    return reach


def _pixels(spec: TrackSpec, effect: Effect, frame: int) -> float:
    """項目の値を、シェーダへ渡すのと同じ読み方で（画面の画素のまま）

    読み方を :func:`~sashimono.engine.gpu.effects._number` と揃えないと、壊れた値や
    キーフレームで余白と実際の動く量が食い違う
    """
    raw = effect.params.get(spec.name)
    value = spec.default_value() if raw is None else raw
    return float(spec.scaled_at(spec.coerce(value), frame, 1.0))


#: 新しい層を、今ある一番太い層よりどれだけ外へ出すか（画面の画素）
_NEXT_STEP = 4.0


def next_stroke(source: GeneratedSource) -> Stroke:
    """設定パネルと AI が足す新しい層 今ある縁の外側に見える太さと色にする

    層の太さは字の輪郭から数えるので、既定の太さのまま末尾（一番下）へ足すと、今ある太い縁の
    下に隠れて何も変わらないように見える 一番太い層より :data:`_NEXT_STEP` だけ太くし、
    色は一番外の縁と明るさが逆の白か黒にする（黒い縁の外に黒を足しても太くなっただけに見える）
    縁が 1 つも無ければ、字の色と明るさが逆の色にする（白い字に白い縁では見えない）
    """
    outer_width = 0.0
    fill = source.params.get("color")
    outer_colour: tuple[float, ...] = (
        fill if isinstance(fill, tuple) and len(fill) >= 3 else (1.0, 1.0, 1.0, 1.0)
    )
    candidates: list[Mapping[str, ParamValue]] = [s.params for s in source.strokes]
    if not source.strokes and legacy_in_use(source.params):
        candidates.append(
            {
                "width": source.params.get(LEGACY_BORDER_WIDTH, AnimatedValue(0.0)),
                "color": source.params.get(LEGACY_BORDER_COLOR, outer_colour),
            }
        )
    width_spec = STROKE.parameters[0]
    assert isinstance(width_spec, TrackSpec)  # 太さは数のスライダー
    for params in candidates:
        width = width_spec.coerce(params.get("width"))
        peak = max([width.static, *(key.value for key in width.keyframes)])
        if peak >= outer_width:
            outer_width = peak
            colour = params.get("color")
            outer_colour = (
                colour if isinstance(colour, tuple) and len(colour) >= 3 else (0.0, 0.0, 0.0)
            )
    dark = 0.2126 * outer_colour[0] + 0.7152 * outer_colour[1] + 0.0722 * outer_colour[2] < 0.5
    colour = (1.0, 1.0, 1.0, 1.0) if dark else (0.0, 0.0, 0.0, 1.0)
    defaults = STROKE.default_params()
    defaults["width"] = AnimatedValue(
        width_spec.clamp(outer_width + _NEXT_STEP) if candidates else width_spec.default
    )
    defaults["color"] = colour
    return Stroke(params=defaults)
