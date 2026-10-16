"""YMM4 の設定の意味を確かめるための試験テンプレートを作る

配布物の値の並びだけでは、角度の向きや大きさの基準（画素か割合か）が決まらない
値を 1 つずつ変えたテンプレートを作り、``tools/ymm4_compare.py`` で YMM4 本体に
描かせて、並んだ絵から意味を読み取る

ひな形（エフェクトや図形の JSON の形）は、手元の配布テンプレート
（``tests/fixtures/ymm4``）から取り出す 配布物そのものは使わず、形だけを借りる

使い方::

    .venv\\Scripts\\python.exe tools\\ymm4_probes.py .work\\probes\\probes.ymmt
    .venv\\Scripts\\python.exe tools\\ymm4_compare.py --work .work\\probe \
        build .work\\probes\\probes.ymmt

2 つ目の引数で何回目の試験かを選ぶ（``first``〜``seventh``） ``seventh`` は画面の大きさで
意味の変わる物を見るので、``build --size 1280x720`` で並べる
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ymm4_compare import base_shape  # noqa: E402

from sashimono.compat.ymm4.template import load_template  # noqa: E402
from sashimono.compat.ymm4.values import type_name  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "ymm4"
EFFECTS = "YukkuriMovieMaker.Project.Effects"
BRUSHES = "YukkuriMovieMaker.Brush"


def still(value: float) -> dict[str, Any]:
    return {"Values": [{"Value": value}], "Span": 0.0, "AnimationType": "なし"}


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def collect_samples() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """配布テンプレートから、種類ごとに 1 つずつ JSON の形を取り出す"""
    shapes: dict[str, dict[str, Any]] = {}
    brushes: dict[str, dict[str, Any]] = {}
    for path in sorted(FIXTURES.rglob("*.ymmt")):
        for template in load_template(path):
            for item in template.items:
                for node in _walk(item):
                    if "$type" in node:
                        shapes.setdefault(type_name(node), copy.deepcopy(node))
                    kind = str(node.get("Type") or "")
                    if "Brush" in kind and isinstance(node.get("Parameter"), dict):
                        brushes.setdefault(
                            kind.partition(",")[0].rpartition(".")[2], copy.deepcopy(node)
                        )
    return shapes, brushes


def _cases(table: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """変える項目の組の一覧 組ごとに値の型が違うので、どの値でも受ける形で持つ"""
    return list(table.items())


def set_values(node: dict[str, Any], **values: Any) -> dict[str, Any]:
    """数はアニメーションの形、そのほかはそのまま入れる"""
    for key, value in values.items():
        current = node.get(key)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isinstance(current, dict)
        ):
            node[key] = still(float(value))
        else:
            node[key] = value
    return node


def solid(color: str) -> dict[str, Any]:
    return {
        "Type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushPlugin, YukkuriMovieMaker",
        "Parameter": {
            "$type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushParameter, YukkuriMovieMaker",
            "Color": color,
        },
    }


def build(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    probes: list[tuple[str, dict[str, Any]]] = []

    def shape(
        width: float = 1200.0, height: float = 600.0, color: str = "#FFFFFFFF"
    ) -> dict[str, Any]:
        item = base_shape(0, 0, 60)
        item["ShapeParameter"]["Width"] = still(width)
        item["ShapeParameter"]["Height"] = still(height)
        item["ShapeParameter"]["Brush"] = solid(color)
        return item

    def with_effect(name: str, effect: dict[str, Any], base: dict[str, Any] | None = None) -> None:
        item = base if base is not None else shape()
        item["VideoEffects"] = [effect]
        probes.append((name, item))

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        found = samples.get(kind)
        if found is None:
            raise KeyError(kind)
        entry = copy.deepcopy(found)
        entry["IsEnabled"] = True
        return set_values(entry, **values)

    def brush(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(brushes[kind])
        set_values(entry["Parameter"], **values)
        return entry

    two_stops = [{"Offset": 0.0, "Color": "#FFFF0000"}, {"Offset": 1.0, "Color": "#FF0000FF"}]
    three_stops = [
        {"Offset": 0.0, "Color": "#FFFF0000"},
        {"Offset": 0.25, "Color": "#FF00FF00"},
        {"Offset": 1.0, "Color": "#800000FF"},
    ]

    # --- グラデーション（エフェクト） ---
    gradient = {
        "GradientType": "Linear",
        "ExtendMode": "Clamp",
        "Blend": "Normal",
        "X": 0,
        "Y": 0,
        "Rotation": 0,
        "Size": 400,
        "Opacity": 100,
        "Stops": two_stops,
    }
    for name, change in _cases(
        {
            "linear": {},
            "rot45": {"Rotation": 45},
            "rot90": {"Rotation": 90},
            "size800": {"Size": 800},
            "x300": {"X": 300},
            "y200": {"Y": 200},
            "round": {"GradientType": "Round"},
            "convex": {"GradientType": "Convex"},
            "wrap": {"ExtendMode": "Wrap", "Size": 200},
            "mirror": {"ExtendMode": "Mirror", "Size": 200},
            "multiply": {"Blend": "Multiply"},
            "opacity50": {"Opacity": 50},
            "stops3": {"Stops": three_stops},
        }
    ):
        with_effect(f"gradient_{name}", effect("GradientEffect", **{**gradient, **change}))

    # --- 前景の塗りつぶし（ブラシ） ---
    fill = {"Opacity": 100, "BlendMode": "Normal", "IsBrushOnly": False}
    linear = {
        "Stops": two_stops,
        "CoordinateMode": "Pixel",
        "Size": 600,
        "Offset": 0,
        "Angle": 0,
        "ExtendMode": "Clamp",
    }
    for name, change in _cases(
        {
            "linear": {},
            "linear_angle45": {"Angle": 45},
            "linear_offset200": {"Offset": 200},
            "linear_wrap": {"ExtendMode": "Wrap", "Size": 200},
            "linear_relative": {"CoordinateMode": "Relative", "Size": 50},
            "linear_stops3": {"Stops": three_stops},
        }
    ):
        with_effect(
            f"fill_{name}",
            effect(
                "FillForegroundEffect",
                **fill,
                Brush=brush("LinearGradientBrushPlugin", **{**linear, **change}),
            ),
        )
    radial = {
        "Stops": two_stops,
        "CenterX": 0,
        "CenterY": 0,
        "OriginX": 0,
        "OriginY": 0,
        "RadiusX": 300,
        "RadiusY": 300,
        "Zoom": 100,
        "Angle": 0,
        "Aspect": 0,
        "ExtendMode": "Clamp",
        "IsInverted": False,
    }
    for name, change in _cases(
        {
            "radial": {},
            "radial_center": {"CenterX": 200, "CenterY": 100},
            "radial_origin": {"OriginX": 150},
            "radial_aspect": {"Aspect": 50},
            "radial_zoom": {"Zoom": 200},
            "radial_inverted": {"IsInverted": True},
        }
    ):
        with_effect(
            f"fill_{name}",
            effect(
                "FillForegroundEffect",
                **fill,
                Brush=brush("RadiulGradientBrushPlugin", **{**radial, **change}),
            ),
        )
    stripe = {
        "Color1": "#FFFF0000",
        "Width1": 40,
        "Color2": "#FF0000FF",
        "Width2": 80,
        "Offset": 0,
        "Zoom": 100,
        "Angle": 0,
    }
    for name, change in _cases(
        {
            "stripe": {},
            "stripe_angle45": {"Angle": 45},
            "stripe_offset20": {"Offset": 20},
            "stripe_zoom200": {"Zoom": 200},
        }
    ):
        with_effect(
            f"fill_{name}",
            effect(
                "FillForegroundEffect",
                **fill,
                Brush=brush("StripeBrushPlugin", **{**stripe, **change}),
            ),
        )
    dot = {
        "Foreground": "#FFFF0000",
        "Background": "#FF0000FF",
        "Radius": 10,
        "Span": 40,
        "Zoom": 100,
        "X": 0,
        "Y": 0,
        "Angle": 0,
        "Aspect": 0,
        "IsInverted": False,
    }
    for name, change in _cases(
        {
            "dot": {},
            "dot_angle45": {"Angle": 45},
            "dot_xy": {"X": 10, "Y": 20},
            "dot_aspect": {"Aspect": 50},
            "dot_inverted": {"IsInverted": True},
        }
    ):
        with_effect(
            f"fill_{name}",
            effect(
                "FillForegroundEffect", **fill, Brush=brush("DotBrushPlugin", **{**dot, **change})
            ),
        )
    grid = {
        "StrokeColor": "#FFFF0000",
        "BackgroundColor": "#FF0000FF",
        "Thickness": 4,
        "Width": 40,
        "Height": 60,
        "Zoom": 100,
        "X": 0,
        "Y": 0,
        "Angle": 0,
        "Aspect": 0,
        "IsInverted": False,
    }
    for name, change in {"grid": {}, "grid_angle30": {"Angle": 30}}.items():
        with_effect(
            f"fill_{name}",
            effect(
                "FillForegroundEffect",
                **fill,
                Brush=brush("GridLineBrushPlugin", **{**grid, **change}),
            ),
        )
    for mode in (
        "Multiply",
        "Overlay",
        "Screen",
        "Subtract",
        "Add",
        "LighterColor",
        "Darker",
        "SoftLight",
        "Difference",
        "ColorBurn",
        "PinLight",
        "VividLight",
        "LinearLight",
        "Division",
        "Lighter",
        "ColorDodge",
        "HardMix",
        "LinearDodge",
        "HardLight",
        "Hue",
    ):
        base = shape(color="#FF808080")
        with_effect(
            f"fill_blend_{mode}",
            effect(
                "FillForegroundEffect",
                Opacity=100,
                BlendMode=mode,
                IsBrushOnly=False,
                Brush=solid("#FFFF8040"),
            ),
            base,
        )
    with_effect(
        "fill_opacity50",
        effect(
            "FillForegroundEffect",
            Opacity=50,
            BlendMode="Normal",
            IsBrushOnly=False,
            Brush=solid("#FFFF0000"),
        ),
    )
    dotted = shape()
    dotted["ShapeParameter"]["Brush"] = brush("DotBrushPlugin", **dot)
    probes.append(("shape_brush_dot", dotted))

    # --- 影 ---
    red = solid("#FFFF0000")

    def small() -> dict[str, Any]:
        # 影がはみ出すので小さめの下地
        return shape(600, 300)

    for name, change in _cases(
        {
            "plain": {},
            "blur10": {"Blur": 10},
            "zoom120": {"Zoom": 120},
            "angle30": {"Angle": 30},
            "angle30_center": {"Angle": 30, "IsRotateAtCenter": True},
            "opacity50": {"Opacity": 50},
        }
    ):
        values = {
            "X": 40,
            "Y": 30,
            "Opacity": 100,
            "Zoom": 100,
            "Angle": 0,
            "Blur": 0,
            "IsRotateAtCenter": False,
            "Brush": red,
        }
        with_effect(f"shadow_{name}", effect("ShadowEffect", **{**values, **change}), small())
    for name, change in _cases(
        {
            "plain": {},
            "blur10": {"Blur": 10},
            "multiply": {"BlendMode": "Multiply"},
        }
    ):
        values = {"X": 40, "Y": 30, "Opacity": 100, "Blur": 0, "BlendMode": "Normal", "Brush": red}
        with_effect(
            f"inner_shadow_{name}", effect("InnerShadowEffect", **{**values, **change}), small()
        )
    for name, change in _cases(
        {
            "solid": {},
            "angle45": {"Angle": 45},
            "image": {"ShadowType": "Image"},
            "attenuation50": {"Attenuation": 50},
        }
    ):
        values = {
            "Angle": 135,
            "Length": 60,
            "Opacity": 100,
            "Attenuation": 0,
            "ShadowType": "Solid",
            "Color1": "#FFFF0000",
            "Color2": "#FF0000FF",
        }
        with_effect(
            f"long_shadow_{name}", effect("LongShadowEffect", **{**values, **change}), small()
        )

    # --- マスク ---
    circle = copy.deepcopy(samples["MaskEffect"])
    for name, change in _cases(
        {
            "circle": {},
            "circle_invert": {"InvertMask": True},
            "circle_blur30": {"Blur": 30},
            "circle_x200": {"X": 200, "Y": 100},
        }
    ):
        entry = copy.deepcopy(circle)
        entry["ShapeType2"] = "YukkuriMovieMaker.Shape.CircleShapePlugin, YukkuriMovieMaker"
        entry["ShapeParameter"] = copy.deepcopy(samples["CircleShapeParameter"])
        set_values(
            entry["ShapeParameter"],
            SizeMode="SizeAspect",
            Size=400,
            AspectRate=0,
            StrokeThickness=4000,
        )
        set_values(entry, X=0, Y=0, Opacity=100, Angle=0, Blur=0, InvertMask=False, IsEnabled=True)
        set_values(entry, **change)
        with_effect(f"mask_{name}", entry)
    fan = copy.deepcopy(circle)
    fan["ShapeType2"] = "YukkuriMovieMaker.Shape.FanShapePlugin, YukkuriMovieMaker"
    fan["ShapeParameter"] = set_values(
        copy.deepcopy(samples["FanShapeParameter"]),
        CenterAngle=90,
        CenterAngleBase=100,
        SizeMode="SizeAspect",
        Size=500,
        AspectRate=0,
        StrokeThickness=4000,
    )
    set_values(fan, X=0, Y=0, Opacity=100, Angle=0, Blur=0, InvertMask=False, IsEnabled=True)
    with_effect("mask_fan90", fan)

    # --- 反転コピー・背景の塗り ---
    for name, change in _cases(
        {
            "right": {},
            "bottom": {"Position": "Bottom"},
            "distance100": {"Distance": 100},
            "not_centering": {"IsCentering": False},
        }
    ):
        values = {
            "Position": "Right",
            "Distance": 0,
            "IsTopBottomReversed": False,
            "IsLeftRightReversed": True,
            "IsCentering": True,
        }
        with_effect(
            f"copy_reverse_{name}",
            effect("CopyAndReverseEffect", **{**values, **change}),
            shape(400, 300),
        )
    for name, change in _cases(
        {
            "margins": {},
            "round40": {"Round": 40},
            "negative": {"Right": -50},
        }
    ):
        values = {
            "Opacity": 100,
            "Round": 0,
            "BlendMode": "Normal",
            "IsBackgroundOnly": False,
            "Brush": red,
            "Top": 10,
            "Bottom": 20,
            "Left": 30,
            "Right": 40,
        }
        with_effect(
            f"fill_background_{name}",
            effect("FillBackgroundEffect", **{**values, **change}),
            shape(400, 300),
        )

    # --- 図形 ---
    def shaped(name: str, plugin: str, parameter: str, **values: Any) -> None:
        item = base_shape(0, 0, 60)
        item["ShapeType2"] = f"YukkuriMovieMaker.Shape.{plugin}, YukkuriMovieMaker"
        item["ShapeParameter"] = set_values(copy.deepcopy(samples[parameter]), **values)
        if "Brush" in item["ShapeParameter"]:
            item["ShapeParameter"]["Brush"] = solid("#FFFFFFFF")
        probes.append((f"shape_{name}", item))

    shaped(
        "circle_aspect50",
        "CircleShapePlugin",
        "CircleShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=50,
        StrokeThickness=4000,
    )
    shaped(
        "circle_aspect_minus50",
        "CircleShapePlugin",
        "CircleShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=-50,
        StrokeThickness=4000,
    )
    shaped(
        "circle_stroke20",
        "CircleShapePlugin",
        "CircleShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        StrokeThickness=20,
    )
    shaped(
        "triangle",
        "TriangleShapePlugin",
        "TriangleShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        Round=0,
        StrokeThickness=4000,
    )
    shaped(
        "hexagon",
        "HexagonShapePlugin",
        "HexagonShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        Round=0,
        StrokeThickness=4000,
    )
    shaped(
        "star",
        "StarShapePlugin",
        "StarShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        Round=0,
        InflationRatio=50,
        StrokeThickness=4000,
    )
    shaped(
        "arrow",
        "ArrowShapePlugin",
        "ArrowShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        Dent=0,
        BarLength=50,
        BarThickness=50,
        StrokeThickness=4000,
    )
    shaped(
        "fan90",
        "FanShapePlugin",
        "FanShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        CenterAngle=90,
        CenterAngleBase=100,
        StrokeThickness=4000,
    )
    shaped(
        "superformula",
        "SuperformulaShapePlugin",
        "SuperformulaShapeParameter",
        SizeMode="SizeAspect",
        Size=400,
        AspectRate=0,
        M=5,
        N=1,
        StrokeThickness=4000,
    )
    shaped(
        "concentration",
        "ConcentrationLineShapePlugin",
        "ConcentrationLineShapeParameter",
        Size=800,
        Density=50,
        Thickness=50,
        Length=50,
        CenterWidth=50,
        Speed=0,
        Stroke="#FFFFFFFF",
    )
    line = set_values(
        copy.deepcopy(samples["LineShapeParameter"]),
        LineType="Straight",
        DashStyle="Solid",
        Thickness=10,
        LengthRate=100,
        IsClosed=False,
    )
    line["Points"] = [{"X": -300.0, "Y": -100.0}, {"X": 0.0, "Y": 100.0}, {"X": 300.0, "Y": -100.0}]
    line["Brush"] = solid("#FFFFFFFF")
    item = base_shape(0, 0, 60)
    item["ShapeType2"] = "YukkuriMovieMaker.Shape.LineShapePlugin, YukkuriMovieMaker"
    item["ShapeParameter"] = line
    probes.append(("shape_line", item))
    closed = copy.deepcopy(item)
    closed["ShapeParameter"]["IsClosed"] = True
    closed["ShapeParameter"]["FillBrush"] = solid("#FFFF0000")
    probes.append(("shape_line_closed", closed))

    # --- 絵を加工するもの（明暗のある下地に掛ける） ---
    def gradient_base() -> dict[str, Any]:
        base = shape()
        base["ShapeParameter"]["Brush"] = brush(
            "LinearGradientBrushPlugin",
            Stops=[
                {"Offset": 0.0, "Color": "#FF000000"},
                {"Offset": 0.5, "Color": "#FFFF2020"},
                {"Offset": 1.0, "Color": "#FFFFFFFF"},
            ],
            CoordinateMode="Pixel",
            Size=1200,
            Offset=0,
            Angle=0,
            ExtendMode="Clamp",
        )
        return base

    simple: dict[str, Callable[[], dict[str, Any]]] = {
        "border_blur20": lambda: effect("BorderBlurEffect", Blur=20),
        "binarization50": lambda: effect(
            "BinarizationEffect", Threshold=50, IsInverted=False, KeepColor=False
        ),
        "binarization50_keep": lambda: effect(
            "BinarizationEffect", Threshold=50, IsInverted=False, KeepColor=True
        ),
        "sharpen": lambda: effect("SharpenEffect", Sharpness=9, Threshold=12),
        "chroma_key_black": lambda: effect(
            "ChromaKeyEffect", Color="#FF000000", Tolerance=30, Feather=True, IsInvert=False
        ),
        "linear_transfer": lambda: effect(
            "LinearTransferEffect",
            RedYIntercept=0,
            RedSlope=50,
            GreenYIntercept=30,
            GreenSlope=100,
            BlueYIntercept=0,
            BlueSlope=100,
            AlphaYIntercept=0,
            AlphaSlope=100,
        ),
        "bloom": lambda: effect(
            "BloomEffect",
            Strength=100,
            Threshold=50,
            Blur=30,
            IsFixedSizeEnabled=False,
            IsColorizationEnabled=False,
        ),
    }
    for name, make in simple.items():
        with_effect(name, make(), gradient_base())
    return [
        {"Name": f"probe_{name}", "Path": ["probe", name], "Items": [item]} for name, item in probes
    ]


def build_second(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """2 回目の試験 登場と退場（時間で変わる）、ノイズ、変形、光の当て方など

    登場は 2 秒（60 フレーム）にして、枠の中の 3 枚（入り・真ん中・終わり）で途中の
    進み具合が見えるようにする
    """
    probes: list[tuple[str, dict[str, Any]]] = []

    def base(width: float = 600.0, height: float = 300.0) -> dict[str, Any]:
        item = base_shape(0, 0, 60)
        item["ShapeParameter"]["Width"] = still(width)
        item["ShapeParameter"]["Height"] = still(height)
        item["ShapeParameter"]["Brush"] = solid("#FFFFFFFF")
        return item

    def textured() -> dict[str, Any]:
        item = base(800.0, 400.0)
        item["ShapeParameter"]["Brush"] = copy.deepcopy(brushes["StripeBrushPlugin"])
        set_values(
            item["ShapeParameter"]["Brush"]["Parameter"],
            Color1="#FFFF4040",
            Width1=40,
            Color2="#FF4040FF",
            Width2=40,
            Offset=0,
            Zoom=100,
            Angle=0,
        )
        return item

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(samples[kind])
        entry["IsEnabled"] = True
        return set_values(entry, **values)

    def add(name: str, *effects: dict[str, Any], item: dict[str, Any] | None = None) -> None:
        target = item if item is not None else base()
        target["VideoEffects"] = list(effects)
        probes.append((name, target))

    timing = {
        "IsInEffect": True,
        "IsOutEffect": False,
        "EffectTimeSeconds": 2.0,
        "EasingType": "Linear",
        "EasingMode": "In",
    }
    add("inout_fade", effect("InOutFadeEffect", Value=0.0, **timing))
    add(
        "inout_fade_out",
        effect(
            "InOutFadeEffect", Value=0.0, **{**timing, "IsInEffect": False, "IsOutEffect": True}
        ),
    )
    add(
        "inout_rotate_z180",
        effect("InOutRotateEffect", ValueX=0.0, ValueY=0.0, ValueZ=180.0, Is3D=False, **timing),
    )
    add(
        "inout_rotate_x90_3d",
        effect("InOutRotateEffect", ValueX=90.0, ValueY=0.0, ValueZ=0.0, Is3D=True, **timing),
    )
    add(
        "inout_rotate_y90_3d",
        effect("InOutRotateEffect", ValueX=0.0, ValueY=90.0, ValueZ=0.0, Is3D=True, **timing),
    )
    add("inout_move_x600", effect("InOutMoveEffect", Value=600.0, Value2=0.0, Value3=0.0, **timing))
    add(
        "inout_move_v2_90",
        effect("InOutMoveEffect", Value=600.0, Value2=90.0, Value3=0.0, **timing),
    )
    add(
        "inout_move_v3_300",
        effect("InOutMoveEffect", Value=600.0, Value2=0.0, Value3=300.0, **timing),
    )
    add(
        "inout_skew_x30",
        effect(
            "InOutSkewEffect",
            AngleX=30.0,
            AngleY=0.0,
            CenterPoint="Center",
            CenterX=0.0,
            CenterY=0.0,
            **timing,
        ),
    )
    add(
        "inout_skew_y30",
        effect(
            "InOutSkewEffect",
            AngleX=0.0,
            AngleY=30.0,
            CenterPoint="Center",
            CenterX=0.0,
            CenterY=0.0,
            **timing,
        ),
    )
    add("inout_blur20", effect("InOutGaussianBlurEffect", Value=20.0, **timing))
    add(
        "repeat_zoom",
        effect(
            "RepeatZoomEffect",
            Zoom=150.0,
            ZoomX=100.0,
            ZoomY=100.0,
            Span=2.0,
            EasingType="Linear",
            EasingMode="In",
            IsCentering=True,
        ),
    )
    add(
        "center_custom_rotate",
        effect(
            "CenterPointEffect",
            Horizontal="Custom",
            Vertical="Custom",
            X=150.0,
            Y=100.0,
            IsKeepPosition=True,
        ),
        effect("RotateEffect", X=0.0, Y=0.0, Z=45.0)
        if "RotateEffect" in samples
        else effect("CenterPointEffect"),
    )
    add(
        "center_origin_rotate",
        effect(
            "CenterPointEffect",
            Horizontal="Origin",
            Vertical="Origin",
            X=0.0,
            Y=0.0,
            IsKeepPosition=True,
        ),
        effect("RotateEffect", X=0.0, Y=0.0, Z=45.0)
        if "RotateEffect" in samples
        else effect("CenterPointEffect"),
    )
    add(
        "reel_spin",
        effect("ReelSpinEffect", Rotation=50.0, Direction=0.0, Blur=0.0, Tile=False),
        item=textured(),
    )
    add(
        "fish_eye",
        effect("FishEyeLensEffect", Projection="Orthographic", Angle=60.0, Zoom=100.0),
        item=textured(),
    )
    add(
        "ripple",
        effect("RippleEffect", X=0.0, Y=0.0, Amplitude=20.0, WaveLength=200.0, Period=1.0),
        item=textured(),
    )
    add(
        "stretch",
        effect(
            "StretchEffect",
            X=0.0,
            Y=0.0,
            Angle=0.0,
            StretchLength=300.0,
            Range=0.0,
            IsCentering=True,
        ),
    )
    add("polar", effect("PolarTransformEffect", CoreWidth=100.0, TwistAngle=0.0), item=textured())
    add(
        "inner_outline",
        effect(
            "InnerOutlineEffect",
            Thickness=20.0,
            Opacity=100.0,
            Blur=0.0,
            Quality=64.0,
            Smoothness=100.0,
            Blend="Normal",
            IsOutlineOnly=False,
            IsAngular=False,
            Brush=solid("#FFFF0000"),
        ),
    )
    add(
        "inner_halftone",
        effect(
            "InnerHalfToneShadowEffect",
            X=40.0,
            Y=30.0,
            Opacity=100.0,
            Blur=0.0,
            BlendMode="Normal",
            Layout="Rhombus",
            Distance=10.0,
            Size=100.0,
            Color="#FFFF0000",
            Strength=100.0,
        ),
    )
    add(
        "three_dimensional",
        effect(
            "ThreeDimensionalEffect",
            X=100.0,
            Y=100.0,
            Length=30.0,
            Opacity=100.0,
            Attenuation=0.0,
            ShadowType="Solid",
            Color1="#FFFF0000",
            Color2="#FF0000FF",
            IsAbsolutePoint=False,
        ),
    )
    add("reflection_bevel", effect("ReflectionAndExtrusionEffect", Blur=0.0, IsInvert=False))
    add("show_only_preview", effect("ShowOnlyPreviewEffect"))
    for noise in ("Perlin", "Voronoi", "Cellular", "Curl", "Random", "Block"):
        brush = copy.deepcopy(brushes["NoiseBrushPlugin"])
        brush["Parameter"]["NoiseType"] = noise
        brush["Parameter"]["Color1"] = "#FF000000"
        brush["Parameter"]["Color2"] = "#FFFFFFFF"
        item = base(800.0, 400.0)
        item["ShapeParameter"]["Brush"] = brush
        probes.append((f"noise_brush_{noise.lower()}", item))
    return [
        {"Name": f"probe2_{name}", "Path": ["probe2", name], "Items": [item]}
        for name, item in probes
    ]


def _noise_parameters() -> dict[str, dict[str, Any]]:
    """配布テンプレートに出てくるノイズの設定を、ノイズの種類ごとに 1 つずつ"""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(FIXTURES.rglob("*.ymmt")):
        for template in load_template(path):
            for item in template.items:
                for node in _walk(item):
                    parameter = node.get("NoiseParameter")
                    kind = node.get("NoiseType")
                    if isinstance(parameter, dict) and isinstance(kind, str):
                        found.setdefault(kind, copy.deepcopy(parameter))
    return found


def build_third(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """3 回目の試験 ノイズのブラシとノイズで歪める設定の意味、立体の回転の向き"""
    probes: list[tuple[str, dict[str, Any]]] = []
    noises = _noise_parameters()

    def base(width: float = 800.0, height: float = 400.0) -> dict[str, Any]:
        item = base_shape(0, 0, 60)
        item["ShapeParameter"]["Width"] = still(width)
        item["ShapeParameter"]["Height"] = still(height)
        item["ShapeParameter"]["Brush"] = solid("#FFFFFFFF")
        return item

    def parameter(kind: str, **values: Any) -> dict[str, Any]:
        source = noises.get(kind) or noises["Perlin"]
        node = copy.deepcopy(source)
        defaults: dict[str, Any] = {
            "Strength": 100.0,
            "Threshold": 0.0,
            "Levels": 256.0,
            "Octaves": 1.0,
            "Lacunarity": 2.0,
            "Gain": 0.5,
            "FractalMode": "Normal",
            "WarpStrength": 0.0,
            "WarpScale": 100.0,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "SpeedX": 0.0,
            "SpeedY": 0.0,
            "SpeedZ": 0.0,
            "Size": 100.0,
            "ScaleX": 100.0,
            "ScaleY": 100.0,
            "ScaleZ": 100.0,
            "Angle": 0.0,
        }
        for key, value in {**defaults, **values}.items():
            if key in node or key in values:
                node[key] = still(float(value)) if isinstance(value, float) else value
        return node

    def noise_brush(
        kind: str,
        *,
        colors: tuple[str, str] = ("#FF000000", "#FFFFFFFF"),
        is_color: bool = False,
        **values: Any,
    ) -> dict[str, Any]:
        item = base()
        brush = copy.deepcopy(brushes["NoiseBrushPlugin"])
        brush["Parameter"]["NoiseType"] = kind
        brush["Parameter"]["NoiseParameter"] = parameter(kind, **values)
        brush["Parameter"]["Color1"], brush["Parameter"]["Color2"] = colors
        brush["Parameter"]["IsColor"] = is_color
        item["ShapeParameter"]["Brush"] = brush
        return item

    def textured() -> dict[str, Any]:
        item = base()
        stripe = copy.deepcopy(brushes["StripeBrushPlugin"])
        set_values(
            stripe["Parameter"],
            Color1="#FFFF4040",
            Width1=40,
            Color2="#FF4040FF",
            Width2=40,
            Offset=0,
            Zoom=100,
            Angle=0,
        )
        item["ShapeParameter"]["Brush"] = stripe
        return item

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(samples[kind])
        entry["IsEnabled"] = True
        return set_values(entry, **values)

    def add(name: str, item: dict[str, Any], *effects: dict[str, Any]) -> None:
        if effects:
            item["VideoEffects"] = list(effects)
        probes.append((name, item))

    add("perlin", noise_brush("Perlin"))
    add("perlin_scale400", noise_brush("Perlin", ScaleX=400.0, ScaleY=400.0))
    add("perlin_scalex400", noise_brush("Perlin", ScaleX=400.0))
    add("perlin_size400", noise_brush("Perlin", Size=400.0))
    add("perlin_octaves5", noise_brush("Perlin", Octaves=5.0))
    add("perlin_levels4", noise_brush("Perlin", Levels=4.0))
    add("perlin_threshold50", noise_brush("Perlin", Threshold=50.0))
    add("perlin_strength50", noise_brush("Perlin", Strength=50.0))
    add("perlin_strength200", noise_brush("Perlin", Strength=200.0))
    add("perlin_turbulence", noise_brush("Perlin", Octaves=5.0, FractalMode="Turbulence"))
    add("perlin_x200", noise_brush("Perlin", X=200.0))
    add("perlin_z1", noise_brush("Perlin", Z=1.0))
    add("perlin_angle45", noise_brush("Perlin", ScaleX=400.0, Angle=45.0))
    add("perlin_speedx100", noise_brush("Perlin", SpeedX=100.0))
    add("perlin_speedz1", noise_brush("Perlin", SpeedZ=1.0))
    add("perlin_red_blue", noise_brush("Perlin", colors=("#FFFF0000", "#FF0000FF")))
    add("perlin_is_color", noise_brush("Perlin", is_color=True))
    for kind in ("Random", "Fractal", "Curl", "Voronoi", "Cellular", "Block"):
        add(kind.lower(), noise_brush(kind))
    add("voronoi_scale400", noise_brush("Voronoi", ScaleX=400.0, ScaleY=400.0))
    add("random_scale400", noise_brush("Random", ScaleX=400.0, ScaleY=400.0))

    def displaced(kind: str, x: float, y: float, **values: Any) -> dict[str, Any]:
        entry = effect("NoiseDisplacementMapEffect", NoiseType=kind)
        entry["NoiseParameter"] = parameter(kind, **values)
        transform = entry.get("Transform")
        if isinstance(transform, dict):
            transform["XScale"] = still(x)
            transform["YScale"] = still(y)
        return entry

    add("move_perlin_x100", textured(), displaced("Perlin", 100.0, 0.0))
    add("move_perlin_y100", textured(), displaced("Perlin", 0.0, 100.0))
    add(
        "move_perlin_x100_scale400",
        textured(),
        displaced("Perlin", 100.0, 0.0, ScaleX=400.0, ScaleY=400.0),
    )
    add("move_random_x100", textured(), displaced("Random", 100.0, 0.0))
    add("move_voronoi_x100", textured(), displaced("Voronoi", 100.0, 0.0))

    for axis, angle in (("X", 45.0), ("Y", 45.0), ("Z", 30.0)):
        values = {"X": 0.0, "Y": 0.0, "Z": 0.0, axis: angle}
        add(f"rotate_{axis.lower()}", base(600.0, 300.0), effect("RotateEffect", **values))
    add("reflection_sample", base(600.0, 300.0), effect("ReflectionAndExtrusionEffect"))
    add("three_dimensional_sample", base(600.0, 300.0), effect("ThreeDimensionalEffect"))
    return [
        {"Name": f"probe3_{name}", "Path": ["probe3", name], "Items": [item]}
        for name, item in probes
    ]


def _transition_parameters() -> dict[str, dict[str, Any]]:
    """配布テンプレートに出てくる切り替えの設定を、切り替えの種類ごとに 1 つずつ"""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(FIXTURES.rglob("*.ymmt")):
        for template in load_template(path):
            for item in template.items:
                if type_name(item) != "TransitionItem":
                    continue
                kind = str(item.get("TransitionType") or "").partition(",")[0]
                found.setdefault(kind.rpartition(".")[2], copy.deepcopy(item))
    return found


def build_fourth(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """4 回目の試験 場面切り替え（TransitionItem）の効き方

    レイヤー 0 に前の場面 A（赤い四角、左）と後の場面 B（青い丸、右）を 90 フレームずつ
    並べ、切り替えをレイヤー 1 の 60〜120 フレームに置く 切り替えの前後と真ん中を撮る
    """
    del brushes
    transitions = _transition_parameters()
    probes: list[tuple[str, list[dict[str, Any]]]] = []

    def scene(
        frame: int, layer: int, colour: str, x: float, *, ellipse: bool = False
    ) -> dict[str, Any]:
        item = base_shape(frame, layer, 90)
        item["ShapeParameter"]["Width"] = still(500.0)
        item["ShapeParameter"]["Height"] = still(300.0)
        item["ShapeParameter"]["Brush"] = solid(colour)
        item["X"] = still(x)
        if ellipse:
            circle = (
                copy.deepcopy(samples["CircleShapeParameter"])
                if "CircleShapeParameter" in samples
                else None
            )
            if circle is not None:
                item["ShapeType2"] = "YukkuriMovieMaker.Shape.CircleShapePlugin, YukkuriMovieMaker"
                circle["Size"] = still(400.0)
                circle["StrokeThickness"] = still(4000.0)
                circle["Brush"] = solid(colour)
                item["ShapeParameter"] = circle
        return item

    def rotate(values: list[float]) -> dict[str, Any]:
        entry = copy.deepcopy(samples["RotateEffect"])
        entry["IsEnabled"] = True
        entry["X"] = still(0.0)
        entry["Y"] = still(0.0)
        entry["Is3D"] = False
        entry["Z"] = {
            "Values": [{"Value": v} for v in values],
            "Span": 0.0,
            "AnimationType": "直線移動",
        }
        return entry

    def transition(
        kind: str,
        *,
        frame: int = 60,
        length: int = 60,
        layer: int = 1,
        before: list[dict[str, Any]] | None = None,
        after: list[dict[str, Any]] | None = None,
        **parameter: Any,
    ) -> dict[str, Any]:
        item = copy.deepcopy(transitions[kind])
        item["Frame"] = frame
        item["Length"] = length
        item["Layer"] = layer
        item["Group"] = 0
        item["IsLocked"] = False
        item["BeforeVideoEffects"] = before or []
        item["AfterVideoEffects"] = after or []
        item["VideoEffects"] = []
        item["TransitionParameter"].update(parameter)
        return item

    def add(name: str, *items: dict[str, Any], a_layer: int = 0) -> None:
        probes.append(
            (
                name,
                [
                    scene(0, a_layer, "#FFFF3030", -300.0),
                    scene(90, a_layer, "#FF3060FF", 300.0, ellipse=True),
                    *items,
                ],
            )
        )

    linear: dict[str, Any] = {"EasingType": "Linear", "EasingMode": "In"}
    add("switch", transition("SwitchTransitionPlugin"))
    add("fade", transition("FadeTransitionPlugin", **linear))
    add("push_0", transition("PushTransitionPlugin", Angle=0.0, **linear))
    add("push_90", transition("PushTransitionPlugin", Angle=90.0, **linear))
    add("slide_before_0", transition("SlideTransitionPlugin", Target="Before", Angle=0.0, **linear))
    add("slide_after_0", transition("SlideTransitionPlugin", Target="After", Angle=0.0, **linear))
    add("none_before", transition("NoneTransitionPlugin", OverlayTarget="Before"))
    add("none_after", transition("NoneTransitionPlugin", OverlayTarget="After"))
    add(
        "switch_rotate",
        transition(
            "SwitchTransitionPlugin", before=[rotate([0.0, 90.0])], after=[rotate([-90.0, 0.0])]
        ),
    )
    add(
        "none_rotate",
        transition(
            "NoneTransitionPlugin",
            OverlayTarget="Before",
            before=[rotate([0.0, 90.0])],
            after=[rotate([-90.0, 0.0])],
        ),
    )
    add("switch_early", transition("SwitchTransitionPlugin", frame=30, length=90))
    add("fade_early", transition("FadeTransitionPlugin", frame=30, length=90, **linear))
    # 切り替えより上のレイヤーに置いた場面は変わるか（緑の小さな四角をずっと出す）
    marker = base_shape(0, 2, 180)
    marker["ShapeParameter"]["Width"] = still(200.0)
    marker["ShapeParameter"]["Height"] = still(200.0)
    marker["ShapeParameter"]["Brush"] = solid("#FF30C030")
    marker["Y"] = still(300.0)
    add("push_with_marker_above", transition("PushTransitionPlugin", Angle=0.0, **linear), marker)
    # 切り替えより上に場面を置いたとき
    add(
        "push_scenes_above",
        transition("PushTransitionPlugin", Angle=0.0, layer=0, **linear),
        a_layer=1,
    )
    return [
        {"Name": f"probe4_{name}", "Path": ["probe4", name], "Items": items}
        for name, items in probes
    ]


def _fixture_item(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    """配布テンプレートから条件に合うアイテムを 1 つ写して返す"""
    for path in sorted(FIXTURES.rglob("*.ymmt")):
        for template in load_template(path):
            for item in template.items:
                if predicate(item):
                    return copy.deepcopy(item)
    return None


def build_fifth(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """5 回目の試験 残りの映像エフェクトと図形 どれも 2 秒（60 フレーム）"""
    probes: list[tuple[str, dict[str, Any]]] = []

    def base(
        width: float = 600.0, height: float = 300.0, colour: str = "#FFE08A2C"
    ) -> dict[str, Any]:
        item = base_shape(0, 0, 60)
        item["ShapeParameter"]["Width"] = still(width)
        item["ShapeParameter"]["Height"] = still(height)
        item["ShapeParameter"]["Brush"] = solid(colour)
        return item

    def brushed(kind: str, **values: Any) -> dict[str, Any]:
        brush = copy.deepcopy(brushes[kind])
        set_values(brush["Parameter"], **values)
        return brush

    def gradient_item() -> dict[str, Any]:
        item = base(800.0, 400.0)
        brush = copy.deepcopy(brushes["LinearGradientBrushPlugin"])
        brush["Parameter"]["Stops"] = [
            {"Offset": 0.0, "Color": "#FFFF3030"},
            {"Offset": 1.0, "Color": "#FF3060FF"},
        ]
        set_values(brush["Parameter"], Size=800.0, Offset=0.0, Angle=0.0)
        brush["Parameter"]["CoordinateMode"] = "Pixel"
        brush["Parameter"]["ExtendMode"] = "Clamp"
        item["ShapeParameter"]["Brush"] = brush
        return item

    def textured() -> dict[str, Any]:
        item = base(800.0, 400.0)
        item["ShapeParameter"]["Brush"] = brushed(
            "StripeBrushPlugin",
            Color1="#FFFF4040",
            Width1=40,
            Color2="#FF4040FF",
            Width2=40,
            Offset=0,
            Zoom=100,
            Angle=30,
        )
        return item

    def dots() -> dict[str, Any]:
        item = base(1600.0, 800.0)
        item["ShapeParameter"]["Brush"] = brushed(
            "DotBrushPlugin",
            Foreground="#FFFFFFFF",
            Background="#FF000000",
            Radius=6,
            Span=120,
            Zoom=100,
            X=0,
            Y=0,
            Angle=0,
            Aspect=0,
        )
        return item

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(samples[kind])
        entry["IsEnabled"] = True
        return set_values(entry, **values)

    def add(name: str, item: dict[str, Any], *effects: dict[str, Any]) -> None:
        item["VideoEffects"] = list(effects)
        probes.append((name, item))

    def shape_of(word: str) -> dict[str, Any]:
        found = _fixture_item(
            lambda item: type_name(item) == "ShapeItem" and word in str(item.get("ShapeType2"))
        )
        assert found is not None, word
        found.update(
            {
                "Frame": 0,
                "Layer": 0,
                "Length": 60,
                "X": still(0.0),
                "Y": still(0.0),
                "Zoom": still(100.0),
                "Rotation": still(0.0),
                "Opacity": still(100.0),
                "Blend": "Normal",
                "Group": 0,
                "IsInverted": False,
                "KeyFrames": {"Frames": [], "Count": 0},
            }
        )
        return found

    # アイテムの反転（左右か上下か）
    inverted = gradient_item()
    inverted["IsInverted"] = True
    inverted["Rotation"] = still(20.0)
    add("item_inverted", inverted)
    add("item_not_inverted_rotated", set_values(gradient_item(), Rotation=20.0))

    # 反射と押し出し（光の当て方）
    reflection = samples["ReflectionAndExtrusionEffect"]
    add("reflection_sample", base(), effect("ReflectionAndExtrusionEffect"))
    for label, change in (
        ("azimuth90", ("LightSource", "Azimuth", 90.0)),
        ("elevation45", ("LightSource", "Elevation", 45.0)),
        ("constant100", ("Highlight", "Constant", 100.0)),
    ):
        entry = effect("ReflectionAndExtrusionEffect", Blur=0.0)
        entry["Lighting"][change[0]][change[1]] = change[2]
        add(f"reflection_{label}", base(), entry)
    thick = effect("ReflectionAndExtrusionEffect", Blur=0.0)
    thick["Heightmap"]["Thickness"] = still(40.0)
    add("reflection_thick40", base(), thick)
    inverted_light = effect("ReflectionAndExtrusionEffect", Blur=0.0, IsInvert=True)
    add("reflection_invert", base(), inverted_light)
    for mode in ("Round", "InvertedRound", "Straight"):
        entry = effect("ReflectionAndExtrusionEffect", Blur=0.0)
        entry["Heightmap"]["BevelMode"] = mode
        entry["Heightmap"]["Thickness"] = still(40.0)
        add(f"reflection_bevel_{mode.lower()}", base(), entry)
    diffuse = _fixture_item(
        lambda item: '"DistantDiffuse"' in json.dumps(item.get("VideoEffects", []))
    )
    if diffuse is not None:
        entry = next(
            e for e in diffuse["VideoEffects"] if type_name(e) == "ReflectionAndExtrusionEffect"
        )
        entry["IsEnabled"] = True
        add("reflection_diffuse", base(), entry)
    del reflection

    # 縦横の指定がある反復拡大
    add(
        "repeat_zoom_x",
        base(),
        effect(
            "RepeatZoomEffect",
            Zoom=100.0,
            ZoomX=50.0,
            ZoomY=100.0,
            Span=2.0,
            EasingType="Linear",
            EasingMode="In",
            IsCentering=False,
        ),
    )
    add(
        "repeat_zoom_150_nocenter",
        base(),
        effect(
            "RepeatZoomEffect",
            Zoom=150.0,
            ZoomX=100.0,
            ZoomY=100.0,
            Span=2.0,
            EasingType="Linear",
            EasingMode="In",
            IsCentering=False,
        ),
    )

    # 色で方向を見て抜く
    keyed = base(800.0, 400.0)
    keyed["ShapeParameter"]["Brush"] = brushed(
        "StripeBrushPlugin",
        Color1="#FF282828",
        Width1=60,
        Color2="#FFE7E7E7",
        Width2=60,
        Offset=0,
        Zoom=100,
        Angle=0,
    )
    color_keys = _all_effects("DirectionalColorKeyEffect")
    for index, entry in enumerate(color_keys[:2]):
        add(f"directional_key_{index}", copy.deepcopy(keyed), entry)

    # 集中線
    lines = shape_of("ConcentrationLine")
    add("concentration_sample", lines)
    for label, values in (
        ("density30", {"Density": 30.0}),
        ("thickness10", {"Thickness": 10.0}),
        ("length20", {"Length": 20.0}),
        ("center0", {"CenterWidth": 0.0}),
        ("speed0", {"Speed": 0.0}),
    ):
        item = shape_of("ConcentrationLine")
        set_values(item["ShapeParameter"], **values)
        add(f"concentration_{label}", item)

    # 跳ねる
    add(
        "jump_sample",
        base(),
        effect(
            "JumpEffect",
            JumpHeight=100.0,
            Stretch=0.0,
            Period=1.0,
            Distortion=0.0,
            Interval=0.0,
            X=0.0,
            Y=0.0,
        ),
    )
    add(
        "jump_stretch",
        base(),
        effect(
            "JumpEffect",
            JumpHeight=100.0,
            Stretch=1.0,
            Period=1.0,
            Distortion=0.0,
            Interval=0.0,
            X=0.0,
            Y=0.0,
        ),
    )
    add(
        "jump_distortion",
        base(),
        effect(
            "JumpEffect",
            JumpHeight=100.0,
            Stretch=0.0,
            Period=1.0,
            Distortion=1.0,
            Interval=0.5,
            X=0.0,
            Y=0.0,
        ),
    )

    # パーティクル（配布物の雨と雪）
    for index, entry in enumerate(_all_effects("ParticleOutputEffect")[:2]):
        add(f"particle_{index}", base(20.0, 60.0, "#FFFFFFFF"), entry)

    # レンズぼかし
    add(
        "lens_blur_20",
        dots(),
        effect("LensBlurEffect", BlurRadius=20.0, Brightness=100.0, EdgeStrength=2.0, Quality=16.0),
    )
    add(
        "lens_blur_edge0",
        dots(),
        effect("LensBlurEffect", BlurRadius=20.0, Brightness=100.0, EdgeStrength=0.0, Quality=16.0),
    )
    add(
        "lens_blur_bright200",
        dots(),
        effect("LensBlurEffect", BlurRadius=20.0, Brightness=200.0, EdgeStrength=2.0, Quality=16.0),
    )

    # 画像のワイプ（YMM4 に付いている切り替え画像）
    resources = "D:\\Program\\YukkuriMovieMaker_v4\\Resources\\Transition\\"
    wipe = {
        "Tolerance": 3.0,
        "Angle": 0.0,
        "KeepAspect": True,
        "IsFixedCoveringScale": False,
        "IsInEffect": True,
        "IsReversedInEffect": False,
        "IsOutEffect": False,
        "IsReversedOutEffect": False,
        "EffectTimeSeconds": 2.0,
        "EasingType": "Linear",
        "EasingMode": "In",
    }
    for name in ("ワイプ横", "円", "四角", "時計回り"):
        add(
            f"wipe_{name}",
            base(800.0, 400.0),
            effect("InOutTransitionEffect", File=resources + name + ".png", **wipe),
        )
    add(
        "wipe_reversed",
        base(800.0, 400.0),
        effect(
            "InOutTransitionEffect",
            File=resources + "ワイプ横.png",
            **{**wipe, "IsReversedInEffect": True},
        ),
    )
    add(
        "wipe_tolerance50",
        base(800.0, 400.0),
        effect(
            "InOutTransitionEffect", File=resources + "ワイプ横.png", **{**wipe, "Tolerance": 50.0}
        ),
    )
    add(
        "wipe_angle90",
        base(800.0, 400.0),
        effect("InOutTransitionEffect", File=resources + "ワイプ横.png", **{**wipe, "Angle": 90.0}),
    )
    add(
        "wipe_missing_file",
        base(800.0, 400.0),
        effect(
            "InOutTransitionEffect",
            File="C:\\tools\\YukkuriMovieMaker4_Lite\\Resources\\Transition\\ワイプ横.png",
            **wipe,
        ),
    )
    add(
        "wipe_out",
        base(800.0, 400.0),
        effect(
            "InOutTransitionEffect",
            File=resources + "ワイプ横.png",
            **{**wipe, "IsInEffect": False, "IsOutEffect": True},
        ),
    )

    # 並べる
    add("tiling_sample", textured(), effect("TilingEffect", X=1.0, Y=41.0))
    add("tiling_2_3", textured(), effect("TilingEffect", X=2.0, Y=3.0))

    # 立体
    for shadow in ("Image", "Solid", "Gradient"):
        add(
            f"three_dimensional_{shadow.lower()}",
            gradient_item(),
            effect(
                "ThreeDimensionalEffect",
                X=100.0,
                Y=60.0,
                Length=30.0,
                Opacity=100.0,
                Attenuation=0.0,
                ShadowType=shadow,
                Color1="#FFFFFFFF",
                Color2="#FF00A000",
                IsAbsolutePoint=False,
            ),
        )

    # 虹色のブラシの幅を割合で
    for width in (400.0, 800.0):
        item = base(width, 300.0)
        item["ShapeParameter"]["Brush"] = copy.deepcopy(brushes["RainbowLinearGradientBrushPlugin"])
        set_values(item["ShapeParameter"]["Brush"]["Parameter"], Width=100.0, Offset=0.0, Angle=0.0)
        item["ShapeParameter"]["Brush"]["Parameter"]["CoordinateMode"] = "Relative"
        add(f"rainbow_relative_{int(width)}", item)

    # リール回転
    add(
        "reel_direction71",
        textured(),
        effect("ReelSpinEffect", Rotation=30.0, Direction=71.0, Blur=0.0, Tile=False),
    )
    add(
        "reel_tile",
        textured(),
        effect("ReelSpinEffect", Rotation=30.0, Direction=0.0, Blur=0.0, Tile=True),
    )
    add(
        "reel_blur50",
        textured(),
        effect("ReelSpinEffect", Rotation=30.0, Direction=0.0, Blur=50.0, Tile=False),
    )
    add(
        "reel_rotation100",
        textured(),
        effect("ReelSpinEffect", Rotation=100.0, Direction=0.0, Blur=0.0, Tile=False),
    )

    # 残像（動く四角で）
    for mode in ("Front", "Back"):
        item = base(200.0, 200.0)
        item["X"] = {
            "Values": [{"Value": -600.0}, {"Value": 600.0}],
            "Span": 0.0,
            "AnimationType": "直線移動",
        }
        add(
            f"after_image_{mode.lower()}",
            item,
            effect("AfterImageEffect", Strength=50.0, Mode=mode),
        )

    # 中心点と登場の拡大
    add(
        "inout_zoom_pivot",
        base(),
        effect(
            "CenterPointEffect",
            Horizontal="Left",
            Vertical="Top",
            X=0.0,
            Y=0.0,
            IsKeepPosition=True,
        ),
        effect(
            "InOutZoomEffect",
            Value=100.0,
            X=100.0,
            Y=100.0,
            IsInEffect=True,
            IsOutEffect=False,
            EffectTimeSeconds=2.0,
            EasingType="Linear",
            EasingMode="In",
        ),
    )

    # 反復回転の Jump
    add(
        "repeat_rotate_jump",
        base(),
        effect(
            "RepeatRotateEffect",
            X=0.0,
            Y=0.0,
            Z=90.0,
            Is3D=False,
            Span=1.0,
            EasingType="Jump",
            EasingMode="In",
            IsCentering=False,
        ),
    )

    # 魚眼
    for label, fish in (
        ("angle120", {"Projection": "Orthographic", "Angle": 120.0, "Zoom": 100.0}),
        ("zoom50", {"Projection": "Orthographic", "Angle": 60.0, "Zoom": 50.0}),
        ("equidistant", {"Projection": "Equidistant", "Angle": 120.0, "Zoom": 100.0}),
        ("stereographic", {"Projection": "Stereographic", "Angle": 120.0, "Zoom": 100.0}),
    ):
        add(f"fish_eye_{label}", textured(), effect("FishEyeLensEffect", **fish))

    # 波紋
    add(
        "ripple_center",
        textured(),
        effect("RippleEffect", X=0.0, Y=0.0, Amplitude=20.0, WaveLength=100.0, Period=2.0),
    )
    add(
        "ripple_offset",
        textured(),
        effect("RippleEffect", X=300.0, Y=100.0, Amplitude=20.0, WaveLength=100.0, Period=2.0),
    )
    add(
        "ripple_negative",
        textured(),
        effect("RippleEffect", X=0.0, Y=0.0, Amplitude=-20.0, WaveLength=300.0, Period=2.0),
    )

    # 描画を遅らせる（位置と拡大をこの位置で当てる）
    lazy = base()
    lazy["X"] = still(400.0)
    lazy["Zoom"] = still(50.0)
    circle_mask = effect("MaskEffect")
    add(
        "draw_lazy_mask",
        copy.deepcopy(lazy),
        effect(
            "DrawLazyEffectEffect",
            IsXYZ=True,
            IsOpacity=False,
            IsZoom=True,
            IsRotation=True,
            IsInvert=False,
            IsCamera=False,
        ),
        copy.deepcopy(circle_mask),
    )
    add("draw_lazy_none_mask", copy.deepcopy(lazy), copy.deepcopy(circle_mask))

    # ブルームの色付け
    add(
        "bloom_colorize",
        dots(),
        effect(
            "BloomEffect",
            Strength=134.2,
            Threshold=45.8,
            Blur=39.1,
            IsFixedSizeEnabled=True,
            IsColorizationEnabled=True,
            Color="#FFFFC039",
        ),
    )
    add(
        "bloom_plain",
        dots(),
        effect(
            "BloomEffect",
            Strength=134.2,
            Threshold=45.8,
            Blur=39.1,
            IsFixedSizeEnabled=True,
            IsColorizationEnabled=False,
            Color="#FFFFC039",
        ),
    )
    add(
        "bloom_not_fixed",
        dots(),
        effect(
            "BloomEffect",
            Strength=134.2,
            Threshold=45.8,
            Blur=39.1,
            IsFixedSizeEnabled=False,
            IsColorizationEnabled=False,
            Color="#FFFFC039",
        ),
    )

    # タイマー
    timer = shape_of("Timer")
    add("timer_sample", timer)

    # 引き伸ばし
    for label, values in (
        (
            "sample",
            {
                "X": 0.0,
                "Y": 0.0,
                "Angle": 0.0,
                "StretchLength": 4000.0,
                "Range": 0.0,
                "IsCentering": True,
            },
        ),
        (
            "range100",
            {
                "X": 0.0,
                "Y": 0.0,
                "Angle": 0.0,
                "StretchLength": 300.0,
                "Range": 100.0,
                "IsCentering": True,
            },
        ),
        (
            "angle45",
            {
                "X": 0.0,
                "Y": 0.0,
                "Angle": 45.0,
                "StretchLength": 300.0,
                "Range": 0.0,
                "IsCentering": True,
            },
        ),
        (
            "offset",
            {
                "X": 100.0,
                "Y": 50.0,
                "Angle": 0.0,
                "StretchLength": 300.0,
                "Range": 0.0,
                "IsCentering": False,
            },
        ),
    ):
        add(f"stretch_{label}", gradient_item(), effect("StretchEffect", **values))

    # 極座標
    add("polar_core0", textured(), effect("PolarTransformEffect", CoreWidth=0.0, TwistAngle=0.0))
    add(
        "polar_twist90",
        textured(),
        effect("PolarTransformEffect", CoreWidth=100.0, TwistAngle=90.0),
    )

    # 破片の回転
    for amount in (0.0, 100.0):
        add(
            f"crash_rotate{int(amount)}",
            gradient_item(),
            effect(
                "CrashEffect",
                StartTime=0.0,
                PlaybackRate=100.0,
                Size=80.0,
                X=0.0,
                Y=0.0,
                Z=0.0,
                FlySpeed=30.0,
                FallSpeed=0.0,
                Delay=0.0,
                Impact=0.0,
                RandomRotate=amount,
                RandomVector=0.0,
            ),
        )

    # ペン
    add("pen_sample", shape_of("PenShape"))
    return [
        {"Name": f"probe5_{name}", "Path": ["probe5", name], "Items": [item]}
        for name, item in probes
    ]


#: 6 回目の探りの文字 縁の太さを測るので、縁の付く辺が長くまっすぐな H にする
PROBE_TEXT = "H"
#: 文字装飾の列挙 YMM4 本体（4.56.1.1）の ``YukkuriMovieMaker.Project.Items.Style`` を
#: メタデータの表から読んだ 配布物の 125 個はどれも Normal だった
TEXT_STYLES = (
    "Normal",
    "Shadow",
    "ShadowLight",
    "Border",
    "BorderLight",
    "SharpBorder",
    "SharpBorderLight",
)


def put(node: dict[str, Any], **values: Any) -> dict[str, Any]:
    """:func:`set_values` と同じ入れ方で、ひな形に無い項目は断る

    無い項目を足すと、動く値の形を知らないまま素の数で書くことになり、YMM4 がプロジェクトを
    開けない 書き出しの道具は読み込みの失敗の窓を見られないので、ここで止める
    """
    missing = [key for key in values if key not in node]
    if missing:
        raise KeyError(f"{node.get('$type', '?')} に無い項目: {', '.join(missing)}")
    return set_values(node, **values)


#: ペンの探りの線 1920x1080 の画面の真ん中を曲がり角にした L の字（画面の左上が原点の画素）
PEN_POINTS: tuple[tuple[float, float], ...] = (
    *((860.0, 440.0 + step * 10.0) for step in range(21)),
    *((870.0 + step * 10.0, 640.0) for step in range(20)),
)


def pen_item(points: tuple[tuple[float, float], ...] = PEN_POINTS) -> dict[str, Any]:
    """配布物のペンの図形の形を借り、線を 1 本だけにして ``points`` を通す"""
    found = _fixture_item(
        lambda item: type_name(item) == "ShapeItem" and "PenShape" in str(item.get("ShapeType2"))
    )
    assert found is not None, "PenShape"
    strokes = found["ShapeParameter"]["Strokes"]
    first = copy.deepcopy(strokes[0])
    template = first["StylusPoints"][0]
    first["StylusPoints"] = [{**template, "X": x, "Y": y} for x, y in points]
    found["ShapeParameter"]["Strokes"] = [first]
    found.update(
        {
            "Frame": 0,
            "Layer": 0,
            "Length": 60,
            "X": still(0.0),
            "Y": still(0.0),
            "Zoom": still(100.0),
            "Rotation": still(0.0),
            "Opacity": still(100.0),
            "Blend": "Normal",
            "Group": 0,
            "IsInverted": False,
            "VideoEffects": [],
            "KeyFrames": {"Frames": [], "Count": 0},
        }
    )
    for key in ("Offset", "Length", "Thickness"):
        if isinstance(found["ShapeParameter"].get(key), dict):
            found["ShapeParameter"][key] = still({"Offset": 0.0, "Length": 100.0}.get(key, 100.0))
    return found


def build_sixth(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """6 回目の試験 #184・#192・#198 の値の意味と、#177 で見つけた退場と単色化とノイズ

    最後に置くのは フレームバッファ にリール回転を掛けた物 #177 の書き出しは
    シュバッと演出素材（フレームバッファにリール回転）の頭で 2 度止まった 止まっても
    前の探りが残るように、疑わしい物ほど後ろに置く
    """
    probes: list[tuple[str, list[dict[str, Any]]]] = []

    def rectangle(
        width: float = 300.0,
        height: float = 300.0,
        colour: str = "#FFFFFFFF",
        *,
        stroke: float = 10000.0,
        layer: int = 0,
    ) -> dict[str, Any]:
        item = base_shape(0, layer, 60)
        item["ShapeParameter"]["Width"] = still(width)
        item["ShapeParameter"]["Height"] = still(height)
        item["ShapeParameter"]["StrokeThickness"] = still(stroke)
        item["ShapeParameter"]["Brush"] = solid(colour)
        return item

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(samples[kind])
        entry["IsEnabled"] = True
        return put(entry, **values)

    def add(name: str, *items: dict[str, Any]) -> None:
        probes.append((name, list(items)))

    def text(
        style: str = "Normal",
        size: float = 100.0,
        content: str = PROBE_TEXT,
        base_point: str = "CenterCenter",
    ) -> dict[str, Any]:
        found = _fixture_item(lambda item: type_name(item) == "TextItem")
        assert found is not None
        found.update(
            {
                "Text": content,
                "Decorations": [],
                "Font": "Arial",
                "FontSize": still(size),
                "LineHeight2": still(100.0),
                "LetterSpacing2": still(0.0),
                "BasePoint": base_point,
                "FontColor": "#FFFFFFFF",
                "Style": style,
                "StyleColor": "#FFFF0000",
                "Bold": False,
                "Italic": False,
                "X": still(0.0),
                "Y": still(0.0),
                "Zoom": still(100.0),
                "Rotation": still(0.0),
                "Opacity": still(100.0),
                "IsInverted": False,
                "VideoEffects": [],
                "Frame": 0,
                "Layer": 0,
                "Length": 60,
                "Group": 0,
                "KeyFrames": {"Frames": [], "Count": 0},
            }
        )
        return found

    def outline(**values: Any) -> dict[str, Any]:
        entry = effect(
            "OutlineEffect",
            StrokeThickness=4.0,
            Blur=0.0,
            X=0.0,
            Y=0.0,
            Opacity=100.0,
            Zoom=100.0,
            Rotation=0.0,
            IsOutlineOnly=False,
            IsAngular=False,
        )
        entry["StrokeBrush"] = solid("#FF6FEEED")
        return set_values(entry, **values)

    def group(*effects: dict[str, Any], span: int = 2) -> dict[str, Any]:
        found = _fixture_item(lambda item: type_name(item) == "GroupItem")
        assert found is not None
        found.update(
            {
                "GroupRange": span,
                "IsGroupOnly": False,
                "X": still(0.0),
                "Y": still(0.0),
                "Z": still(0.0),
                "Zoom": still(100.0),
                "Rotation": still(0.0),
                "Opacity": still(100.0),
                "IsInverted": False,
                "VideoEffects": list(effects),
                "Frame": 0,
                "Layer": 0,
                "Length": 60,
                "Group": 0,
                "KeyFrames": {"Frames": [], "Count": 0},
            }
        )
        return found

    def hollow_frames(layer: int) -> list[dict[str, Any]]:
        """太さ 2 の線だけの四角を 2 つ 細い線の周りの縁がどう付くかを見る"""
        return [
            rectangle(600.0, 300.0, stroke=2.0, layer=layer),
            rectangle(300.0, 150.0, stroke=2.0, layer=layer + 1),
        ]

    # --- #184 テキストの文字装飾 縁と影の太さ 大きさ 100 と 50 ---
    for style in TEXT_STYLES:
        add(f"text_style_{style.lower()}", text(style))
    add("text_style_border_size50", text("Border", size=50.0))
    add("text_style_border_size200", text("Border", size=200.0))

    # --- #192 細い線の周りの縁 グループの縁と、図形そのものの縁 ---
    add("thin_group_outline_sf", group(outline(Blur=2.5, Opacity=50.9)), *hollow_frames(1))
    add("thin_group_outline_plain", group(outline()), *hollow_frames(1))
    add("thin_group_outline_blur", group(outline(Blur=2.5)), *hollow_frames(1))
    add("thin_group_outline_half", group(outline(Opacity=50.9)), *hollow_frames(1))
    framed = hollow_frames(0)
    framed[0]["VideoEffects"] = [outline(Blur=2.5, Opacity=50.9)]
    framed[1]["VideoEffects"] = [outline(Blur=2.5, Opacity=50.9)]
    add("thin_item_outline_sf", *framed)
    # 縁の位置のずらし Y の向き（配布物の 3 つは -1）
    add(
        "outline_offset_y_minus20",
        rectangle(400.0, 200.0, layer=0)
        | {"VideoEffects": [outline(Y=-20.0, StrokeThickness=10.0)]},
    )
    add(
        "outline_offset_x30",
        rectangle(400.0, 200.0, layer=0)
        | {"VideoEffects": [outline(X=30.0, StrokeThickness=10.0)]},
    )
    add(
        "group_outline_offset_y_minus20",
        group(outline(Y=-20.0, StrokeThickness=10.0), span=1),
        rectangle(400.0, 200.0, layer=1),
    )

    # --- #198 登場の移動の Value3 ---
    for value3 in (0.0, 300.0, -300.0):
        add(
            f"inout_move_value3_{int(value3)}",
            rectangle()
            | {
                "VideoEffects": [
                    effect(
                        "InOutMoveEffect",
                        Value=0.0,
                        Value2=0.0,
                        Value3=value3,
                        IsInEffect=True,
                        IsOutEffect=False,
                        EffectTimeSeconds=2.0,
                        EasingType="Linear",
                        EasingMode="In",
                    )
                ]
            },
        )

    # --- #198 中心点の原点 左上を基準にしたテキストを回す ---
    for horizontal, vertical in (("Origin", "Origin"), ("Center", "Center"), ("Left", "Top")):
        item = text(content="HHHH", base_point="LeftTop")
        item["X"] = still(-200.0)
        item["Y"] = still(-100.0)
        item["VideoEffects"] = [
            effect(
                "CenterPointEffect",
                Horizontal=horizontal,
                Vertical=vertical,
                X=0.0,
                Y=0.0,
                IsKeepPosition=True,
            ),
            effect("RotateEffect", X=0.0, Y=0.0, Z=30.0, Is3D=False),
        ]
        add(f"center_text_lefttop_{horizontal.lower()}_{vertical.lower()}", item)

    # --- #198 反射と押し出しの光の方位 ---
    for azimuth in (0.0, 45.0, 90.0, 180.0, 270.0):
        entry = effect("ReflectionAndExtrusionEffect", Blur=0.0)
        entry["Lighting"]["LightSource"]["Azimuth"] = still(azimuth)
        add(
            f"reflection_azimuth{int(azimuth)}",
            rectangle(600.0, 300.0, "#FFE08A2C") | {"VideoEffects": [entry]},
        )

    # --- #198 リール回転の向きと極座標の角度の基準 上が赤・下が青の縦のグラデーション ---
    def vertical_gradient() -> dict[str, Any]:
        item = rectangle(600.0, 300.0)
        brush = copy.deepcopy(brushes["LinearGradientBrushPlugin"])
        brush["Parameter"]["Stops"] = [
            {"Offset": 0.0, "Color": "#FFFF3030"},
            {"Offset": 1.0, "Color": "#FF3060FF"},
        ]
        set_values(brush["Parameter"], Size=300.0, Offset=0.0, Angle=90.0)
        brush["Parameter"]["CoordinateMode"] = "Pixel"
        brush["Parameter"]["ExtendMode"] = "Clamp"
        item["ShapeParameter"]["Brush"] = brush
        return item

    def horizontal_gradient() -> dict[str, Any]:
        item = vertical_gradient()
        set_values(item["ShapeParameter"]["Brush"]["Parameter"], Size=600.0, Angle=0.0)
        return item

    for direction in (0.0, 90.0, -90.0):
        add(
            f"reel_rotation10_direction{int(direction)}",
            vertical_gradient()
            | {
                "VideoEffects": [
                    effect(
                        "ReelSpinEffect", Rotation=10.0, Direction=direction, Blur=0.0, Tile=False
                    )
                ]
            },
        )
    add(
        "polar_horizontal_gradient",
        horizontal_gradient()
        | {"VideoEffects": [effect("PolarTransformEffect", CoreWidth=0.0, TwistAngle=0.0)]},
    )
    add(
        "polar_vertical_gradient",
        vertical_gradient()
        | {"VideoEffects": [effect("PolarTransformEffect", CoreWidth=0.0, TwistAngle=0.0)]},
    )

    # --- #198 残像 アイテムが終わった後の区間に残るか 30 フレームのアイテムを 60 の枠に ---
    moving = rectangle(200.0, 200.0)
    moving["Length"] = 30
    moving["X"] = {
        "Values": [{"Value": -600.0}, {"Value": 600.0}],
        "Span": 0.0,
        "AnimationType": "直線移動",
    }
    moving["VideoEffects"] = [effect("AfterImageEffect", Strength=50.0, Mode="Front")]
    add("after_image_tail", moving)

    # --- #177 退場のイージングの向き 回転 90 度 Back の Out と In ---
    for mode in ("Out", "In"):
        add(
            f"inout_rotate_out_back_{mode.lower()}",
            rectangle(400.0, 100.0)
            | {
                "VideoEffects": [
                    effect(
                        "InOutRotateEffect",
                        ValueX=0.0,
                        ValueY=0.0,
                        ValueZ=90.0,
                        Is3D=False,
                        IsInEffect=False,
                        IsOutEffect=True,
                        EffectTimeSeconds=1.0,
                        EasingType="Back",
                        EasingMode=mode,
                    )
                ]
            },
        )

    # --- #177 単色化 明るさを保つか ---
    for keep in (False, True):
        add(
            f"monocolor_keep_{str(keep).lower()}",
            horizontal_gradient()
            | {
                "VideoEffects": [
                    effect(
                        "MonocolorizationEffect",
                        Strength=100.0,
                        Color="#FF292110" if not keep else "#FF40A0FF",
                        KeepBrightness=keep,
                    )
                ]
            },
        )

    # --- #177 ノイズ（NoiseEffect）の透明度 乱数なので面の平均と散らばりで読む ---
    def noise_effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = effect("NoiseEffect", IsColor=False, IsAlpha=True, IsUniqueSeed=False)
        entry["NoiseType"] = kind
        parameter = copy.deepcopy(samples[f"{kind}NoiseParameter"])
        set_values(
            parameter,
            Strength=100.0,
            Threshold=0.0,
            Levels=256.0,
            Octaves=5.0,
            X=0.0,
            Y=0.0,
            Z=0.0,
            SpeedX=0.0,
            SpeedY=0.0,
            SpeedZ=0.0,
            ScaleX=1000.0,
            ScaleY=1000.0,
            ScaleZ=100.0,
            Angle=0.0,
        )
        if "Size" in parameter:
            set_values(parameter, Size=100.0)
        set_values(parameter, **values)
        entry["NoiseParameter"] = parameter
        return entry

    for label, kind, values in (
        ("random", "Random", {}),
        ("random_threshold50", "Random", {"Threshold": 50.0}),
        ("random_strength200", "Random", {"Strength": 200.0}),
        ("random_strength50", "Random", {"Strength": 50.0}),
        ("perlin", "Perlin", {}),
    ):
        add(
            f"noise_alpha_{label}",
            rectangle(800.0, 400.0) | {"VideoEffects": [noise_effect(kind, **values)]},
        )
    colour_noise = noise_effect("Random")
    colour_noise["IsAlpha"] = False
    add(
        "noise_colour_random",
        rectangle(800.0, 400.0, "#FF808080") | {"VideoEffects": [colour_noise]},
    )

    # --- #198 ペン（1920x1080 の基準 1280x720 は 7 回目） ---
    add("pen_cross", pen_item())

    # --- #177 最後に フレームバッファへリール回転（シュバッと演出素材と同じ形） ---
    buffer = _fixture_item(lambda item: type_name(item) == "FrameBufferItem")
    assert buffer is not None
    buffer.update(
        {
            "Frame": 0,
            "Layer": 1,
            "Length": 60,
            "IsHidden": False,
            "IsLocked": False,
            "VideoEffects": [
                effect("ReelSpinEffect", Rotation=30.0, Direction=0.0, Blur=50.0, Tile=False)
            ],
        }
    )
    add("frame_buffer_reel_spin", vertical_gradient(), buffer)

    return [
        {"Name": f"probe6_{name}", "Path": ["probe6", name], "Items": items}
        for name, items in probes
    ]


def build_seventh(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """7 回目の試験 画面を 1280x720 にして書き出す（``ymm4_compare.py build --size 1280x720``）

    - ペンの点の座標の基準（#198） 1920x1080 では 6 回目の ``pen_cross`` と一致した
      点が画面の左上からの画素なら、L の角が画面の真ん中から右下へずれる
    - 細かい格子の周りのグループの縁（#192 SFっぽい吹き出し(右) と同じ格子と縁）
    - ノイズの強さ 100 より上（6 回目の 200 は全部が消えた）
    - 最後に、シュバッと演出素材と同じ形のフレームバッファ（頭のずれと動くリール回転）
    """
    probes: list[tuple[str, list[dict[str, Any]]]] = []

    def add(name: str, *items: dict[str, Any]) -> None:
        probes.append((name, list(items)))

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(samples[kind])
        entry["IsEnabled"] = True
        return put(entry, **values)

    def grid(cell: float = 12.0, layer: int = 0) -> dict[str, Any]:
        item = base_shape(0, layer, 60)
        item["ShapeParameter"]["Width"] = still(600.0)
        item["ShapeParameter"]["Height"] = still(300.0)
        brush = copy.deepcopy(brushes["GridLineBrushPlugin"])
        brush["Parameter"]["StrokeColor"] = "#FF1C9F9D"
        brush["Parameter"]["BackgroundColor"] = "#00FFFFFF"
        put(
            brush["Parameter"],
            Thickness=1.3,
            Width=cell,
            Height=cell,
            Zoom=100.0,
            X=0.0,
            Y=0.0,
            Angle=-45.0,
            Aspect=0.0,
        )
        brush["Parameter"]["IsInverted"] = False
        item["ShapeParameter"]["Brush"] = brush
        item["Opacity"] = still(75.2)
        return item

    def outline(**values: Any) -> dict[str, Any]:
        entry = effect(
            "OutlineEffect",
            StrokeThickness=4.0,
            Blur=2.5,
            X=0.0,
            Y=0.0,
            Opacity=50.9,
            Zoom=100.0,
            Rotation=0.0,
            IsOutlineOnly=False,
            IsAngular=False,
        )
        entry["StrokeBrush"] = solid("#FF6FEEED")
        return put(entry, **values)

    def group(*effects: dict[str, Any]) -> dict[str, Any]:
        found = _fixture_item(lambda item: type_name(item) == "GroupItem")
        assert found is not None
        found.update(
            {
                "GroupRange": 1,
                "X": still(0.0),
                "Y": still(0.0),
                "Z": still(0.0),
                "Zoom": still(100.0),
                "Rotation": still(0.0),
                "Opacity": still(100.0),
                "IsInverted": False,
                "VideoEffects": list(effects),
                "Frame": 0,
                "Layer": 0,
                "Length": 60,
                "Group": 0,
                "KeyFrames": {"Frames": [], "Count": 0},
            }
        )
        return found

    add("pen_l_1280x720", pen_item())

    add("grid_group_outline_sf", group(outline()), grid(layer=1))
    add("grid_group_outline_sharp", group(outline(Blur=0.0, Opacity=100.0)), grid(layer=1))
    add("grid_item_outline_sf", grid() | {"VideoEffects": [outline()]})
    add("grid_group_outline_cell24", group(outline()), grid(24.0, layer=1))
    add("grid_plain", grid())

    for strength in (120.0, 150.0, 200.0):
        entry = effect("NoiseEffect", IsColor=False, IsAlpha=True, IsUniqueSeed=False)
        entry["NoiseType"] = "Random"
        parameter = copy.deepcopy(samples["RandomNoiseParameter"])
        put(
            parameter,
            Strength=strength,
            Threshold=0.0,
            Levels=256.0,
            SpeedX=0.0,
            SpeedY=0.0,
            SpeedZ=0.0,
            ScaleX=1000.0,
            ScaleY=1000.0,
            Angle=0.0,
        )
        entry["NoiseParameter"] = parameter
        square = base_shape(0, 0, 60)
        square["ShapeParameter"]["Width"] = still(600.0)
        square["ShapeParameter"]["Height"] = still(300.0)
        square["ShapeParameter"]["Brush"] = solid("#FFFFFFFF")
        square["VideoEffects"] = [entry]
        add(f"noise_alpha_random_strength{int(strength)}", square)

    # シュバッと演出素材と同じ形 頭を 4 秒ずらしたフレームバッファに、0 → 100 と動くリール回転
    buffer = _fixture_item(lambda item: type_name(item) == "FrameBufferItem")
    assert buffer is not None
    reel = effect("ReelSpinEffect", Direction=0.0, Tile=False)
    reel["Rotation"] = {
        "Values": [{"Value": 0.0}, {"Value": 100.0}, {"Value": 100.0}],
        "Span": 0.0,
        "AnimationType": "Expo_Out",
    }
    reel["Blur"] = {
        "Values": [{"Value": 50.0}, {"Value": 50.0}, {"Value": 50.0}],
        "Span": 0.0,
        "AnimationType": "Expo_Out",
    }
    buffer.update(
        {
            "Frame": 0,
            "Layer": 1,
            "Length": 60,
            "IsHidden": False,
            "IsLocked": False,
            "ContentOffset": "00:00:04",
            "VideoEffects": [reel],
        }
    )
    add("frame_buffer_offset_reel_keys", grid(), buffer)
    return [
        {"Name": f"probe7_{name}", "Path": ["probe7", name], "Items": items}
        for name, items in probes
    ]


def _fixture_template(word: str) -> list[dict[str, Any]]:
    """名前に ``word`` を含む配布テンプレートのアイテム 頭を 0 フレーム・0 レイヤーへ寄せる"""
    for path in sorted(FIXTURES.rglob("*.ymmt")):
        for template in load_template(path):
            if word in template.name:
                items = [copy.deepcopy(item) for item in template.items]
                first = min(int(item.get("Frame", 0)) for item in items)
                top = min(int(item.get("Layer", 0)) for item in items)
                for item in items:
                    item["Frame"] = int(item.get("Frame", 0)) - first
                    item["Layer"] = int(item.get("Layer", 0)) - top
                return items
    raise KeyError(word)


def build_eighth(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """8 回目の試験（#210 #216） どれも 2 秒（60 フレーム）

    - 縁取りの濃さが元の絵の不透明度に従うか（細かい格子の上の縁 #210 の 5）
    - ノイズで歪めるときのしきい値の効き方（水の中風の横の帯 #210 の 3）
    - 除算の重ね方 下に何も無い所・黒・灰の上（周辺カラークリスタルノイズ #210 の 4）
    - 画面外から登場の入り始めの位置 図形・グループ・ずらした図形（#216 の 2）
    - 最後に、書き出しを止めるシュバッと演出素材を 1 つずつ組み立てる（#210 の 1）
      軽い物から順に足していくので、止まったコマでどの足し方が止めたかが分かる
    """
    probes: list[tuple[str, list[dict[str, Any]]]] = []

    def add(name: str, *items: dict[str, Any]) -> None:
        probes.append((name, list(items)))

    def effect(kind: str, **values: Any) -> dict[str, Any]:
        entry = copy.deepcopy(samples[kind])
        entry["IsEnabled"] = True
        return put(entry, **values)

    def rectangle(
        width: float,
        height: float,
        colour: str = "#FFFFFFFF",
        *,
        layer: int = 0,
        x: float = 0.0,
    ) -> dict[str, Any]:
        item = base_shape(0, layer, 60)
        item["ShapeParameter"]["Width"] = still(width)
        item["ShapeParameter"]["Height"] = still(height)
        item["ShapeParameter"]["Brush"] = solid(colour)
        item["X"] = still(x)
        return item

    def outline(**values: Any) -> dict[str, Any]:
        entry = effect(
            "OutlineEffect",
            StrokeThickness=6.0,
            Blur=0.0,
            X=0.0,
            Y=0.0,
            Opacity=100.0,
            Zoom=100.0,
            Rotation=0.0,
            IsOutlineOnly=False,
            IsAngular=False,
        )
        entry["StrokeBrush"] = solid("#FF6FEEED")
        return put(entry, **values)

    def group(*effects: dict[str, Any], span: int = 1) -> dict[str, Any]:
        found = _fixture_item(lambda item: type_name(item) == "GroupItem")
        assert found is not None
        found.update(
            {
                "GroupRange": span,
                "IsGroupOnly": False,
                "X": still(0.0),
                "Y": still(0.0),
                "Z": still(0.0),
                "Zoom": still(100.0),
                "Rotation": still(0.0),
                "Opacity": still(100.0),
                "IsInverted": False,
                "VideoEffects": list(effects),
                "Frame": 0,
                "Layer": 0,
                "Length": 60,
                "Group": 0,
                "KeyFrames": {"Frames": [], "Count": 0},
            }
        )
        return found

    # --- #210 の 5 縁の濃さ 不透明度 50 の四角・色の不透明度 50 の四角・細い線 ---
    half = rectangle(300.0, 300.0)
    half["Opacity"] = still(50.0)
    half["VideoEffects"] = [outline()]
    add("outline_item_opacity50", half)
    add(
        "outline_colour_alpha50",
        rectangle(300.0, 300.0, "#80FFFFFF") | {"VideoEffects": [outline()]},
    )
    add(
        "outline_group_over_opacity50",
        group(outline()),
        rectangle(300.0, 300.0, layer=1) | {"Opacity": still(50.0)},
    )
    for thickness in (1.0, 2.0, 4.0):
        add(
            f"outline_thin_line{int(thickness)}",
            rectangle(800.0, thickness) | {"VideoEffects": [outline(StrokeThickness=4.0)]},
        )

    # --- #210 の 3 ノイズで歪めるときのしきい値 横の細い線を縦にずらす ---
    def wobble(threshold: float, levels: float = 256.0) -> dict[str, Any]:
        entry = effect("NoiseDisplacementMapEffect", Blur=0.0)
        entry["Mode"] = "Move"
        entry["Transform"] = {
            "$type": f"{EFFECTS}.DisplacementMapMoveTransform, YukkuriMovieMaker",
            "XScale": still(0.0),
            "YScale": still(200.0),
        }
        entry["NoiseType"] = "Perlin"
        parameter = copy.deepcopy(samples["PerlinNoiseParameter"])
        put(
            parameter,
            Strength=100.0,
            Threshold=threshold,
            Levels=levels,
            Octaves=1.0,
            X=0.0,
            Y=0.0,
            Z=0.0,
            SpeedX=0.0,
            SpeedY=0.0,
            SpeedZ=0.0,
            ScaleX=400.0,
            ScaleY=400.0,
            ScaleZ=100.0,
        )
        entry["NoiseParameter"] = parameter
        return entry

    for threshold in (0.0, 30.0, 60.0):
        add(
            f"wobble_threshold{int(threshold)}",
            rectangle(1800.0, 4.0) | {"VideoEffects": [wobble(threshold)]},
        )
    add("wobble_levels4", rectangle(1800.0, 4.0) | {"VideoEffects": [wobble(0.0, 4.0)]})

    # --- #210 の 4 除算 下に何も無い・黒・灰の上 ---
    for label, below in (("empty", None), ("black", "#FF000000"), ("grey", "#FF808080")):
        items: list[dict[str, Any]] = []
        if below is not None:
            items.append(rectangle(1920.0, 1080.0, below, layer=0))
        top = rectangle(600.0, 300.0, "#FF4080C0", layer=1)
        top["Blend"] = "Division"
        items.append(top)
        add(f"division_{label}", *items)
    faint = rectangle(600.0, 300.0, "#FF4080C0", layer=1) | {
        "Blend": "Division",
        "Opacity": still(63.5),
    }
    add("division_grey_opacity63", rectangle(1920.0, 1080.0, "#FF808080", layer=0), faint)

    # --- #216 の 2 画面外から登場（右から 直線 2 秒） ---
    def enter(direction: str = "Right") -> dict[str, Any]:
        return effect(
            "InOutMoveFromOutsideFrameEffect",
            Value=direction,
            IsInEffect=True,
            IsOutEffect=False,
            EffectTimeSeconds=2.0,
            EasingType="Linear",
            EasingMode="In",
        )

    add("enter_right_shape", rectangle(640.0, 360.0, "#FFE08A2C") | {"VideoEffects": [enter()]})
    add(
        "enter_right_shape_x300",
        rectangle(640.0, 360.0, "#FFE08A2C", x=300.0) | {"VideoEffects": [enter()]},
    )
    add(
        "enter_left_shape", rectangle(640.0, 360.0, "#FFE08A2C") | {"VideoEffects": [enter("Left")]}
    )
    add("enter_right_group", group(enter()), rectangle(640.0, 360.0, "#FFE08A2C", layer=1))
    add(
        "enter_right_group_x300",
        group(enter()),
        rectangle(640.0, 360.0, "#FFE08A2C", layer=1, x=300.0),
    )

    # --- #210 の 1 シュバッと演出素材を組み立てる 最後に置く ---
    whole = _fixture_template("シュバッと演出素材(横)")
    by_layer = {int(item["Layer"]): item for item in whole}
    stripes_group, band_a, band_b = (by_layer[i] for i in range(1, 4))

    def only(item: dict[str, Any], layer: int, keep: tuple[str, ...]) -> dict[str, Any]:
        copied = copy.deepcopy(item)
        copied["Layer"] = layer
        copied["VideoEffects"] = [e for e in copied["VideoEffects"] if type_name(e) in keep]
        return copied

    add("shuba_band_brush_only", only(band_a, 0, ()))
    add("shuba_band_blur", only(band_a, 0, ("GaussianBlurEffect",)))
    add("shuba_band_invert", only(band_a, 0, ("GaussianBlurEffect", "InvertEffect")))
    add("shuba_band_b_sharpen", only(band_b, 0, ("GaussianBlurEffect", "SharpenEffect")))
    add("shuba_band_key", only(band_a, 0, ("GaussianBlurEffect", "DirectionalColorKeyEffect")))
    add(
        "shuba_band_b_key",
        only(band_b, 0, ("GaussianBlurEffect", "DirectionalColorKeyEffect", "SharpenEffect")),
    )
    add("shuba_bands_grouped", *(copy.deepcopy(i) for i in (stripes_group, band_a, band_b)))
    add("shuba_whole", *(copy.deepcopy(i) for i in whole))
    del brushes
    return [
        {"Name": f"probe8_{name}", "Path": ["probe8", name], "Items": items}
        for name, items in probes
    ]


def build_ninth(
    samples: dict[str, dict[str, Any]], brushes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """9 回目の試験（#216 の 2） 画面外から登場の上下 どれも 2 秒（60 フレーム）

    横は画面の幅だけずらすと 8 回目で分かった 縦も画面の高さだけか、絵の端が画面の端を
    越えるだけか（今の写し方）を、置き場を変えた四角で見分ける
    """
    probes: list[tuple[str, list[dict[str, Any]]]] = []

    def enter(direction: str) -> dict[str, Any]:
        entry = copy.deepcopy(samples["InOutMoveFromOutsideFrameEffect"])
        entry["IsEnabled"] = True
        return put(
            entry,
            Value=direction,
            IsInEffect=True,
            IsOutEffect=False,
            EffectTimeSeconds=2.0,
            EasingType="Linear",
            EasingMode="In",
        )

    def rectangle(y: float = 0.0) -> dict[str, Any]:
        item = base_shape(0, 0, 60)
        item["ShapeParameter"]["Width"] = still(640.0)
        item["ShapeParameter"]["Height"] = still(360.0)
        item["ShapeParameter"]["Brush"] = solid("#FFE08A2C")
        item["Y"] = still(y)
        return item

    for direction in ("Top", "Bottom"):
        label = direction.lower()
        probes.append(
            (f"enter_{label}_shape", [rectangle() | {"VideoEffects": [enter(direction)]}])
        )
        probes.append(
            (f"enter_{label}_shape_y200", [rectangle(200.0) | {"VideoEffects": [enter(direction)]}])
        )
    del brushes
    return [
        {"Name": f"probe9_{name}", "Path": ["probe9", name], "Items": items}
        for name, items in probes
    ]


def _all_effects(name: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(FIXTURES.rglob("*.ymmt")):
        for template in load_template(path):
            for item in template.items:
                for node in _walk(item):
                    if type_name(node) == name:
                        entry = copy.deepcopy(node)
                        entry["IsEnabled"] = True
                        found.append(entry)
    return found


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".work" / "probes" / "probes.ymmt"
    which = sys.argv[2] if len(sys.argv) > 2 else "first"
    samples, brushes = collect_samples()
    builders = {
        "first": build,
        "second": build_second,
        "third": build_third,
        "fourth": build_fourth,
        "fifth": build_fifth,
        "sixth": build_sixth,
        "seventh": build_seventh,
        "eighth": build_eighth,
        "ninth": build_ninth,
    }
    templates = builders[which](samples, brushes)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"ItemTemplates": templates}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(templates)} 本の試験テンプレートを {target} に書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
