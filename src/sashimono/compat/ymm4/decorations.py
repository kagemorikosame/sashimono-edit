"""YMM4 の文字装飾と映像エフェクトを、こちらの持ち物へ写す

**実物を見て分かったこと** 配布テンプレートの ``Decorations`` は空で、飾りは
次の 2 か所に入っていた

* ``Style`` / ``StyleColor`` — テキストアイテム自身が持つ文字装飾 AviUtl2 の
  ``文字装飾`` に似るが、太さは YMM4 に描かせて測った別の表（:data:`_STYLES`）で持つ
* ``VideoEffects`` — 積まれた映像エフェクトの列 手元の 2 本では
  ``OutlineEffect``（縁取り）が 152 回と圧倒的に多く、これが YMM4 の縁取りの
  実体だった

だから ``Decorations`` だけを見ていると、**縁取りが 1 つも出ない** ここでは
3 つとも読む

YMM4 の縁取りは映像エフェクトなので、2 つ目の縁取りは 1 つ目の縁取りを付けた絵の外側に
付く（太さが足し合わさる） テキストに載せられる縁取りが 2 つ以上あるときは、縁取りの層
（#272）として内側から順に並べ、層の太さを字の輪郭からの和にする 1 つだけのときは前と同じく
テキストの縁取りの項目へ載せる（前からの作品と同じ絵のまま）
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.decoration import PLAIN, TextDecoration, decoration_params
from sashimono.compat.ymm4.brushes import fill_foreground, gradient_effect, is_solid, noise_mask
from sashimono.compat.ymm4.effects import CenterPoint, center_point, map_effect, mapped_names
from sashimono.compat.ymm4.values import (
    animated,
    brush_colour,
    colour,
    number,
    reporting,
    type_name,
)
from sashimono.core.model import MAX_STROKES, AnimatedValue, Effect, ParamValue, Stroke
from sashimono.effects.definition import registry

__all__ = [
    "DecorationResult",
    "has_outline",
    "map_decorations",
    "map_video_effects",
    "outlines_fit_text",
]

Colour = tuple[float, float, float, float]

#: YMM4 の ``Style``（テキストの文字装飾） 名前は YMM4 本体（4.56.1.1）の列挙
#: ``YukkuriMovieMaker.Project.Items.Style`` をメタデータの表から読んだ
#: 前は AviUtl2 の呼び名から推した名前（ThinBorder など）で引いていて、本体に無い名前だった
#:
#: 太さと影のずれは、Arial 100 の H を YMM4 に描かせて測った（#184 の探り 50 と 200 でも
#: 同じ割合） 縁取りは片側 8（50 で 4、200 で 15.5）、Light の付く縁取りは 4、影は右と下へ
#: 4 ずれ、ShadowLight は同じずれで濃さが半分 Sharp の付く縁取りは太さが同じで角が尖る
#: （角の形までは写さない） AviUtl2 の文字装飾とは太さが違うので、表を分けて持つ
_STYLES: dict[str, TextDecoration] = {
    "Normal": PLAIN,
    "Shadow": TextDecoration("Shadow", shadow=0.04),
    "ShadowLight": TextDecoration("ShadowLight", shadow=0.04, shadow_opacity=0.5),
    "Border": TextDecoration("Border", border=0.08),
    "BorderLight": TextDecoration("BorderLight", border=0.04),
    "SharpBorder": TextDecoration("SharpBorder", border=0.08),
    "SharpBorderLight": TextDecoration("SharpBorderLight", border=0.04),
}


@dataclass(slots=True)
class DecorationResult:
    """装飾を分けた結果"""

    #: テキストオブジェクトへ直接載せる設定
    params: dict[str, ParamValue] = field(default_factory=dict)
    #: 外側に積むエフェクト 内側から外側の順
    effects: list[Effect] = field(default_factory=list)
    #: 最後に効いている中心点 アイテムの位置・拡大・回転の支点にもなる
    pivot: CenterPoint | None = None
    #: テキストの縁取りの層（並びの頭が一番内側で一番上） 縁取りが 2 つ以上あるときだけ
    strokes: list[Stroke] = field(default_factory=list)


def map_decorations(
    decorations: Any,
    report: CompatibilityReport,
    *,
    size: float = 64.0,
    style: str = "",
    style_colour: Any = None,
) -> DecorationResult:
    """文字の飾りを読む

    ``decorations`` は ``Decorations`` の列（実物では空のことが多い）、
    ``style`` は ``Style``、``style_colour`` は ``StyleColor``
    """
    result = DecorationResult()

    tint = colour(style_colour, (0.0, 0.0, 0.0, 1.0))
    if style:
        decoration = _STYLES.get(style)
        if decoration is None:
            report.note_missing(f"YMM4 の文字装飾: {style}")
        else:
            if style.startswith("Sharp"):
                # 太さは合わせたが角は丸いまま 黙って丸めると尖った角が消えたことに気付けない
                report.note_missing(f"YMM4 の文字装飾の尖った角: {style}（丸い角で描いた）")
            result.params.update(decoration_params(decoration, size, tint))

    if not isinstance(decorations, list):
        return result

    borders: list[tuple[AnimatedValue, Colour]] = []
    for entry in decorations:
        if not isinstance(entry, dict):
            continue
        name = type_name(entry)
        if name.endswith("BorderDecoration") or name in ("Border", "Outline"):
            borders.append(_border(entry))
        elif name.endswith("ShadowDecoration") or name == "Shadow":
            _shadow(entry, result, size)
        else:
            report.note_missing(f"YMM4 の装飾: {name or '種類不明'}")

    _place_borders(borders, result)
    return result


#: ``VideoEffects`` の種類と、こちらのエフェクト種別
#:
#: 手元の配布テンプレート 2 本に出てきた 50 種あまりのうち、同じ絵になるものだけ
#: 残りは記録に残して素通しにする 似た別のもので代用すると、直したつもりの
#: 無い違いが出る
_VIDEO_EFFECTS: dict[str, str] = {
    "GaussianBlurEffect": "blur",
    "BlurEffect": "blur",
    "DirectionalBlurEffect": "directional_blur",
    "UnidirectionalBlurEffect": "directional_blur",
    "ColorCorrectionEffect": "color",
    "MonocolorizationEffect": "monochrome",
    "MosaicEffect": "mosaic",
    "NoiseEffect": "noise",
    "CropEffect": "crop",
    "ZoomEffect": "zoom",
    "RotateEffect": "rotate",
    "DrawPositionEffect": "position",
    "OpacityEffect": "opacity",
    "LuminanceKeyEffect": "luminance_key",
}


def map_video_effects(
    effects: Any,
    report: CompatibilityReport,
    *,
    length: int = 1,
    keyframes: Any = None,
    text: bool = False,
) -> DecorationResult:
    """``VideoEffects`` の列を読む

    ``text`` が真なら、縁取り（``OutlineEffect``）をテキストの飾りとして
    :class:`DecorationResult` の ``params`` へ分けて返す 偽なら並びの位置のまま
    縁取りのエフェクトにする
    """
    # 知らない移動方法の形を、渡された記録へ書く（値を読む所は記録を受け取らない）
    with reporting(report):
        return _map_video_effects(effects, report, length=length, keyframes=keyframes, text=text)


def _map_video_effects(
    effects: Any, report: CompatibilityReport, *, length: int, keyframes: Any, text: bool
) -> DecorationResult:
    result = DecorationResult()
    if not isinstance(effects, list):
        return result

    borders: list[tuple[AnimatedValue, Colour]] = []
    # テキスト以外は縁取りの設定を持たないので、縁取りはいつも並びのその位置へ
    # エフェクトとして置く テキストの設定の形で返すと、読む側が拾わない図形や素材では
    # 縁が黙って落ちる（矢印_ピンク・リボンのテロップの図形の縁 #179）
    # テキストでも、載せられない縁取り（縁だけ・動く不透明度・ぼかし）が 1 つでもあれば
    # すべてをエフェクトにする 一部だけをテキストへ載せると、載せた方が並びの頭へ動いて
    # 重なり順が YMM4 と変わる
    in_place = not text or not outlines_fit_text(effects, length=length, keyframes=keyframes)
    pivot: CenterPoint | None = None
    for entry in effects:
        if not isinstance(entry, dict):
            continue
        if entry.get("IsEnabled") is False:
            continue

        name = type_name(entry)
        if name == "ShowOnlyPreviewEffect":
            # 掛かったアイテムごと書き出しから外す（アイテムを読む所で見る）
            continue
        if name == "CenterPointEffect":
            # 後ろに続く回転と拡大の支点になる 位置を保たないなら絵もずらす
            # 「位置を保つ」を切ったときのずらしは、アイテムを最後に置く変形
            # （:func:`sashimono.compat.ymm4.template._placement`）がまとめて行う 途中で
            # ずらすと、後ろの変形が元の範囲を支点にしたまま回り、支点が合わない
            pivot, _ = center_point(entry, report, length=length, keyframes=keyframes)
            continue
        if name == "OutlineEffect":
            outline = _outline(entry, length, keyframes, report)
            if in_place:
                # 太さ 0 でも縁だけなら置く 元の絵を消すのは縁だけの役目で、落とすと
                # 消えるはずの塗りが残る（#175）
                if _peak(outline.width) > 0.0 or outline.outline_only:
                    result.effects.append(outline.effect())
                continue
            borders.append(outline.baked())
            continue

        if name == "FillForegroundEffect":
            result.effects.append(
                _or_skip(fill_foreground(entry, report, length=length, keyframes=keyframes))
            )
            continue
        if name == "NoiseEffect" and isinstance(entry.get("NoiseParameter"), dict):
            # 新しい形のノイズは模様の値で不透明度か色を薄める 前は古い形の強さ（Intensity）を
            # 探して既定の 20 で粒を足していて、画面効果/雨 の縞が薄まらずに全面へ出た（#177）
            result.effects.append(
                _or_skip(noise_mask(entry, report, length=length, keyframes=keyframes))
            )
            continue
        if name == "GradientEffect":
            result.effects.append(
                _or_skip(gradient_effect(entry, report, length=length, keyframes=keyframes))
            )
            continue
        if name == "MosaicEffect" and str(entry.get("MosaicType") or "Rectangle") != "Rectangle":
            # 三角（Triangle 4 本）とドロネー（Delaunay 2 本）は四角の粒しか持たないこちらでは
            # 同じ絵にならない 粒の大きさは合わせて四角で描き、違うことを数えて残す
            report.note_missing(f"YMM4 のモザイクの形: {entry.get('MosaicType')}")
        built = _video_effect(name, entry, length, keyframes)
        if built is None:
            built = map_effect(name, entry, report, length=length, keyframes=keyframes)
        if built is None:
            # 写し方を持っている種類で None なら、形の問題としてすでに記録してある
            if name not in mapped_names():
                report.note_missing(f"YMM4 の映像エフェクト: {name or '種類不明'}")
            continue
        if pivot is not None and built.kind in _PIVOTED_KINDS:
            built = with_pivot(built, pivot)
        result.effects.append(built)

    _place_borders(borders, result, layered=True)
    result.pivot = pivot
    return result


#: 支点（中心点エフェクト）を受け取れる変形
#: 角度で切り抜きは支点から帯の位置を測る（SFっぽい吹き出し(右) の名札 #179）
_PIVOTED_KINDS = frozenset(
    {
        "crop_angle",
        "transform",
        "inout_zoom",
        # 登場の回転も支点を受け取る 木製看板テロップは釘の所を支点に振れて出入りする（#177）
        "inout_rotate",
        "random_rotate",
        "random_zoom",
        "repeat_rotate",
        "inout_getup",
        "spiral",
    }
)


def _or_skip(effect: Effect | None) -> Effect:
    """写せなかったブラシは素通しのエフェクトにする（記録は写す側が残してある）"""
    if effect is not None:
        return effect
    definition = registry.get("opacity")
    assert definition is not None  # 標準エフェクトは必ずある
    return definition.create(amount=100.0)


def with_pivot(effect: Effect, pivot: CenterPoint) -> Effect:
    """変形の支点を、前にあった中心点に合わせる"""
    definition = registry.get(effect.kind)
    if definition is None:  # pragma: no cover - 変形は標準エフェクト
        return effect
    params = dict(effect.params)
    for name, value in pivot.params().items():
        spec = definition.spec(name)
        if spec is not None:
            params[name] = spec.coerce(value)
    return replace(effect, params=params)


def _video_effect(name: str, entry: dict[str, Any], length: int, keyframes: Any) -> Effect | None:
    kind = _VIDEO_EFFECTS.get(name)
    if kind is None:
        return None

    def value(key: str, default: float = 0.0, scale: float = 1.0) -> AnimatedValue:
        return animated(entry.get(key), default, length=length, keyframes=keyframes, scale=scale)

    if kind == "blur":
        definition = registry.get("blur")
        return None if definition is None else definition.create(radius=value("Blur", 8.0))
    if kind == "mosaic":
        definition = registry.get("mosaic")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # 粒の大きさは形ごとの設定（MosaicParameter）の中にある 手元の実物 7 本は
        # どれもそこにしか持たず、上の段を見ていたころは全部が既定の 16 で描かれていた
        # （ドット絵風加工は 4 のはずが 4 倍の粗さ） 上の段は古い形のために残す
        parameter = entry.get("MosaicParameter")
        if isinstance(parameter, dict) and "Size" in parameter:
            size = animated(parameter.get("Size"), 16.0, length=length, keyframes=keyframes)
            return definition.create(size=size)
        return definition.create(size=value("Size", 16.0))
    if kind == "noise":
        definition = registry.get("noise")
        return None if definition is None else definition.create(strength=value("Intensity", 20.0))
    if kind == "color":
        definition = registry.get("color")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # YMM4 は 100 を「変化なし」にする百分率 こちらは 0 が変化なし
        return definition.create(
            brightness=AnimatedValue(number(entry.get("Lightness"), 100.0) - 100.0),
            contrast=AnimatedValue(number(entry.get("Contrast"), 100.0) - 100.0),
            saturation=AnimatedValue(number(entry.get("Saturation"), 100.0) - 100.0),
            hue=value("HueRotation"),
            # 輝度（Brightness）は明るさ（Lightness）と別の項目 読まずにいると、
            # 場面切り替えで明るく飛ばす動き（ペイントトランジション）が消える
            gain=value("Brightness", 100.0),
        )
    if kind == "directional_blur":
        definition = registry.get("directional_blur")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        return definition.create(radius=value("StandardDeviation", 16.0), angle=value("Angle"))
    if kind == "fill":
        definition = registry.get("fill")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # 合成モードまでは写せない 塗る色と強さだけを合わせる
        return definition.create(
            color=brush_colour(entry.get("Brush"), (1.0, 1.0, 1.0, 1.0)),
            amount=value("Opacity", 100.0),
        )
    if kind == "opacity":
        definition = registry.get("opacity")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        return definition.create(amount=value("Opacity", 100.0))
    if kind == "luminance_key":
        definition = registry.get("luminance_key")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # ``Mode`` が ``Dark`` なら暗いところを抜く ``IsInvert`` はその反転
        dark = str(entry.get("Mode") or "") == "Dark"
        return definition.create(
            threshold=value("Threshold", 50.0),
            smoothness=value("Smoothness", 10.0),
            invert=dark is bool(entry.get("IsInvert")),
        )
    if kind == "position":
        definition = registry.get("transform")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # YMM4 の Y は下向き こちらは上向き
        return definition.create(pos_x=value("X"), pos_y=value("Y", scale=-1.0))
    if kind == "monochrome":
        definition = registry.get("fill")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # 単色化は色を塗る 明るさを保つ（KeepBrightness）なら塗る色の明るさを元の絵へそろえる
        # 前は彩度を抜くだけで色を塗らず、レトロなカウントダウン3秒 の暗い茶色
        # （#292110 明るさを保たない）の数字が白いまま残った（#177）
        return definition.create(
            color=colour(entry.get("Color"), (1.0, 1.0, 1.0, 1.0)),
            amount=value("Strength", 100.0),
            keep_luma=entry.get("KeepBrightness") is not False,
        )
    if kind == "zoom":
        definition = registry.get("transform")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # 拡大率は ``Zoom`` に縦横それぞれの ``ZoomX`` ``ZoomY`` が掛かる
        # どれも動きうるので、素の数で読むと登場アニメーションが止まる
        # 前は ``ZoomX`` を読まず、お辞儀(120F) の横 105% が掛からずに縁の差が 40 を
        # 超えていた（#205）
        return definition.create(
            scale=value("Zoom", 100.0),
            scale_x=value("ZoomX", 100.0),
            scale_y=value("ZoomY", 100.0),
        )
    if kind == "rotate":
        definition = registry.get("transform")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        # Z は平面の回転、X と Y は板を傾ける立体の回転 X と Y は Sashimono と向きが逆
        return definition.create(
            rotation=value("Z"),
            rotation_x=value("X", scale=-1.0),
            rotation_y=value("Y", scale=-1.0),
        )
    if kind == "crop":
        definition = registry.get("crop")
        if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
            return None
        return definition.create(
            top=value("Top"), bottom=value("Bottom"), left=value("Left"), right=value("Right")
        )
    return None


def _border(entry: dict[str, Any]) -> tuple[AnimatedValue, Colour]:
    thickness = number(entry.get("Thickness"), number(entry.get("StrokeThickness"), 4.0))
    tint = brush_colour(entry.get("StrokeBrush"), colour(entry.get("Color"), (0.0, 0.0, 0.0, 1.0)))
    return AnimatedValue(max(0.0, thickness)), tint


@dataclass(frozen=True, slots=True)
class _Outline:
    """``OutlineEffect`` 1 つ分"""

    width: AnimatedValue
    tint: Colour
    #: 0〜100 動くことがある
    opacity: AnimatedValue
    outline_only: bool
    #: 縁のぼかし（画素）
    blur: AnimatedValue
    #: 縁だけのずれ（画素 Y は上が正） YMM4 の X と Y 配布物の 3 つは Y が -1
    offset_x: AnimatedValue = field(default_factory=lambda: AnimatedValue(0.0))
    offset_y: AnimatedValue = field(default_factory=lambda: AnimatedValue(0.0))

    @property
    def fits_text(self) -> bool:
        """テキストの縁取りの設定（太さと色）へ載せられるか

        縁だけは文字の塗りと一緒に描かれる縁取りでは表せない 動く不透明度は、色の濃さへ
        焼き込むと最初の値で止まる テキストの縁取りはぼかせない
        """
        return (
            not self.outline_only
            and not self.opacity.keyframes
            and _peak(self.blur) <= 0.0
            and _still_zero(self.offset_x)
            and _still_zero(self.offset_y)
        )

    def baked(self) -> tuple[AnimatedValue, Colour]:
        """動かない不透明度を色の濃さへ焼き込んだ太さと色"""
        red, green, blue, alpha = self.tint
        return self.width, (red, green, blue, alpha * _unit(self.opacity.static / 100.0))

    def effect(self) -> Effect:
        return _border_effect(
            self.width,
            self.tint,
            opacity=self.opacity,
            outline_only=self.outline_only,
            blur=self.blur,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
        )


def outlines_fit_text(effects: Any, *, length: int = 1, keyframes: Any = None) -> bool:
    """列の中の効いている縁取りが、どれもテキストの縁取りの設定へ載せられるか

    描画を遅らせる印で列を分けて読むときも、分ける前の列全体で 1 度だけ決める 区間ごとに
    決めると、片側の縁取りだけがテキストへ移って並びの頭へ動き、重なり順が変わる
    """
    if not isinstance(effects, list):
        return True
    return all(
        _outline(entry, length, keyframes).fits_text
        for entry in effects
        if isinstance(entry, dict)
        and entry.get("IsEnabled") is not False
        and type_name(entry) == "OutlineEffect"
    )


def has_outline(effects: Any) -> bool:
    """列に効いている縁取りがあるか"""
    return isinstance(effects, list) and any(
        isinstance(entry, dict)
        and entry.get("IsEnabled") is not False
        and type_name(entry) == "OutlineEffect"
        for entry in effects
    )


def _outline(
    entry: dict[str, Any],
    length: int,
    keyframes: Any,
    report: CompatibilityReport | None = None,
) -> _Outline:
    """`OutlineEffect` を読む `report` を渡したときだけ写せない所を数える

    載せられるかを確かめるだけの読み（:func:outlines_fit_text）では渡さない 渡すと
    同じ縁取りを 2 度数える
    """

    def value(key: str, default: float) -> AnimatedValue:
        return animated(entry.get(key), default, length=length, keyframes=keyframes)

    # 4.56 の縁取りは太さを Thickness、色を Brush に持つ（実物はキラリンエフェクトの 1 件）
    # 古い名前だけを見ると、既定の太さ 4 の黒い縁になる
    newer = "StrokeThickness" not in entry and "Thickness" in entry
    brush = entry.get("Brush" if newer else "StrokeBrush")
    if report is not None and not is_solid(brush):
        # 縁取りのエフェクトは色 1 つしか持たない 模様のブラシは色 1 つ（無ければ黒）で
        # 描くので、違うことを数えて残す
        report.note_missing("YMM4 の縁取りの単色以外のブラシ")
    return _Outline(
        width=_non_negative(value("Thickness" if newer else "StrokeThickness", 4.0)),
        tint=brush_colour(brush),
        # 縁の不透明度 読まずにいると、薄く光らせるつもりのグループの縁（SFっぽい
        # 吹き出し(右) は 50.9）が、格子の隙間を濃く埋める
        opacity=value("Opacity", 100.0),
        outline_only=entry.get("IsOutlineOnly") is True,
        blur=_non_negative(value("Blur", 0.0)),
        # 縁のずれ YMM4 に Y -20 を描かせると縁が 20 上へ、X 30 で 30 右へ動いた（#192）
        # グループの縁も同じ 元の絵は動かない
        offset_x=value("X", 0.0),
        offset_y=animated(entry.get("Y"), 0.0, length=length, keyframes=keyframes, scale=-1.0),
    )


def _still_zero(value: AnimatedValue) -> bool:
    return not value.keyframes and value.static == 0.0


def _non_negative(value: AnimatedValue) -> AnimatedValue:
    if value.keyframes:
        return value
    return AnimatedValue(max(0.0, value.static))


def _peak(value: AnimatedValue) -> float:
    return max((key.value for key in value.keyframes), default=value.static)


def _place_borders(
    borders: list[tuple[AnimatedValue, Colour]],
    result: DecorationResult,
    *,
    layered: bool = False,
) -> None:
    """縁取りをテキストへ載せる

    ``layered`` なら（映像エフェクトの縁取り）、2 つ以上の縁取りを縁取りの層にする
    YMM4 の縁取りは前の縁取りを付けた絵の外側に付くので、層の太さは字の輪郭から数えた和
    （1 つ目 7.3 と 2 つ目 1.0 なら 7.3 と 8.3） 並びの頭が一番内側で、上に描かれる
    太さが動く縁取りが 2 つ以上あるときは層にしない 動く値どうしの和はキーの位置が違うと
    1 つの動く値で表せない 層の数の上限を超えるときも同じ（どちらも実物には無い）

    層にしないときは、一番太いものをテキストに持たせ、残りを縁取りエフェクトにする
    一番太いものを選ぶのは、それが文字の形をいちばん強く決めるから 細いほうをテキストに
    載せると、太いほうをエフェクトで足したときに二重の縁の間隔が変わる
    文字装飾（``Decorations``）の縁取りは実物に 1 つも無く、重なり方を確かめていないので層にしない
    """
    usable = [item for item in borders if _peak(item[0]) > 0.0]
    if not usable:
        return
    moving = sum(1 for thickness, _ in usable if thickness.keyframes)
    if layered and 2 <= len(usable) <= MAX_STROKES and moving <= 1:
        reach = AnimatedValue(0.0)
        for thickness, tint in usable:
            reach = _plus(reach, thickness)
            result.strokes.append(Stroke(params={"width": reach, "color": tint}))
        return

    widest = max(usable, key=lambda item: _peak(item[0]))
    result.params["border_width"] = widest[0]
    result.params["border_color"] = widest[1]

    seen_widest = False
    for thickness, tint in usable:
        if not seen_widest and (thickness, tint) == widest:
            seen_widest = True
            continue
        result.effects.append(_border_effect(thickness, tint))


def _plus(first: AnimatedValue, second: AnimatedValue) -> AnimatedValue:
    """2 つの太さの和 動くのはどちらか 1 つまで（動く側の点ごとに、動かない側を足す）"""
    if first.keyframes and second.keyframes:
        raise ValueError("動く値どうしは足せない")
    moving, still = (first, second) if first.keyframes else (second, first)
    return AnimatedValue(
        first.static + second.static,
        tuple(replace(key, value=key.value + still.static) for key in moving.keyframes),
    )


def _border_effect(
    thickness: AnimatedValue,
    tint: Colour,
    *,
    opacity: AnimatedValue | None = None,
    outline_only: bool = False,
    blur: AnimatedValue | None = None,
    offset_x: AnimatedValue | None = None,
    offset_y: AnimatedValue | None = None,
) -> Effect:
    definition = registry.get("border")
    assert definition is not None  # 標準エフェクトは必ずある
    return definition.create(
        width=thickness,
        color=tint,
        opacity=opacity if opacity is not None else AnimatedValue(100.0),
        outline_only=outline_only,
        blur=blur if blur is not None else AnimatedValue(0.0),
        offset_x=offset_x if offset_x is not None else AnimatedValue(0.0),
        offset_y=offset_y if offset_y is not None else AnimatedValue(0.0),
    )


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _shadow(entry: dict[str, Any], result: DecorationResult, size: float) -> None:
    if "shadow_x" in result.params:
        # 2 つ目以降の影は載せられない 黙って捨てず、エフェクトの影として積む
        definition = registry.get("shadow")
        if definition is not None:
            result.effects.append(
                definition.create(
                    offset_x=number(entry.get("X"), 0.0),
                    offset_y=number(entry.get("Y"), 0.0),
                    blur=number(entry.get("Blur"), 0.0),
                    color=colour(entry.get("Color"), (0.0, 0.0, 0.0, 1.0)),
                )
            )
        return

    default = size * 0.06
    result.params["shadow_x"] = AnimatedValue(number(entry.get("X"), default))
    result.params["shadow_y"] = AnimatedValue(-number(entry.get("Y"), default))
    result.params["shadow_blur"] = AnimatedValue(number(entry.get("Blur"), 0.0))
    result.params["shadow_color"] = colour(entry.get("Color"), (0.0, 0.0, 0.0, 1.0))
