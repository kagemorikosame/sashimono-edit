"""YMM4 のブラシと塗りのエフェクトの写し方

値の意味は YMM4 本体に描かせた絵で確かめた（``tools/ymm4_probes.py``） ここでは
写した結果の値が、その確かめた決まりどおりに並ぶことを押さえる
"""

from __future__ import annotations

from typing import Any

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.ymm4.brushes import brush_effect, fill_foreground, gradient_effect
from sashimono.compat.ymm4.template import map_template
from sashimono.core.model import AnimatedValue


def _still(value: float) -> dict[str, Any]:
    return {"Values": [{"Value": value}], "Span": 0.0, "AnimationType": "なし"}


def _brush(kind: str, **parameter: Any) -> dict[str, Any]:
    return {"Type": f"YukkuriMovieMaker.Brush.{kind}, YukkuriMovieMaker", "Parameter": parameter}


def _static(value: object) -> float:
    assert isinstance(value, AnimatedValue)
    return value.static


class TestGradientEffect:
    def test_angle_zero_runs_top_to_bottom(self) -> None:
        # 模様の角度 0 は左から右 グラデーションエフェクトの 0 は上から下 ずらさないと 90 度回る
        entry = {"GradientType": "Linear", "Rotation": _still(0.0), "Size": _still(400.0)}
        effect = gradient_effect(entry, CompatibilityReport())
        assert effect is not None
        assert _static(effect.params["angle"]) == 90.0
        assert effect.params["pattern"] == "linear"

    def test_round_reaches_the_end_at_half_the_size(self) -> None:
        # 円形は半径が Size の半分 Size をそのまま半径にすると、倍の大きさの円になる
        entry = {"GradientType": "Round", "Size": _still(400.0)}
        effect = gradient_effect(entry, CompatibilityReport())
        assert effect is not None
        assert _static(effect.params["radius_x"]) == 200.0

    def test_stops_are_sorted_and_thinned(self) -> None:
        # 位置の順に並んでいない配布物がある 上限を超えた分は間引いて記録に残す
        stops = [{"Offset": i / 19, "Color": "#FFFFFFFF"} for i in reversed(range(20))]
        report = CompatibilityReport()
        effect = gradient_effect({"Stops": stops}, report)
        assert effect is not None
        assert effect.params["stops"] == 16
        assert _static(effect.params["offset0"]) == 0.0
        assert _static(effect.params["offset15"]) == 1.0
        assert any("間引いた" in line for line in report.lines())


class TestBrushes:
    def test_unknown_blend_is_recorded(self) -> None:
        # YMM4 に無い名前を通常にして黙ると、見た目の違いに気付けない
        report = CompatibilityReport()
        effect = fill_foreground(
            {"BlendMode": "Darken", "Brush": _brush("SolidColorBrushPlugin", Color="#FFFF0000")},
            report,
        )
        assert effect is not None
        assert effect.params["blend"] == "normal"
        assert any("Darken" in line for line in report.lines())

    def test_real_blend_names(self) -> None:
        # 実物の比較(明)(暗)は Lighter と Darker
        for name, expected in (
            ("Lighter", "lighten"),
            ("Darker", "darken"),
            ("LinearDodge", "add"),
        ):
            entry = {
                "BlendMode": name,
                "Brush": _brush("SolidColorBrushPlugin", Color="#FFFF0000"),
            }
            effect = fill_foreground(entry, CompatibilityReport())
            assert effect is not None, f"{name} が写せていない"
            assert effect.params["blend"] == expected

    def test_dot_brush_keeps_its_spacing(self) -> None:
        # ドットの間隔（Span）を写し損ねると、
        # YMM4 の水玉模様で塗ったテンプレートの点が違う間隔で並ぶ
        brush = _brush(
            "DotBrushPlugin",
            Foreground="#FFFF0000",
            Background="#FF0000FF",
            Radius=_still(10.0),
            Span=_still(40.0),
        )
        effect = brush_effect(brush, CompatibilityReport())
        assert effect is not None
        assert effect.params["pattern"] == "dot"
        assert _static(effect.params["span"]) == 40.0

    def test_noise_brush_becomes_a_noise_pattern(self) -> None:
        # 大きさ（Size）は横と縦の大きさに掛かる 乱流は FractalMode で決まる
        report = CompatibilityReport()
        brush = _brush(
            "NoiseBrushPlugin",
            Color1="#FF000000",
            Color2="#FFFFFFFF",
            NoiseType="Voronoi",
            NoiseParameter={
                "Size": 200.0,
                "ScaleX": _still(150.0),
                "ScaleY": _still(100.0),
                "Levels": _still(4.0),
                "Octaves": 3.0,
                "FractalMode": "Turbulence",
            },
        )
        effect = brush_effect(brush, report)
        assert effect is not None
        assert effect.params["pattern"] == "noise"
        assert effect.params["noise_kind"] == "voronoi"
        assert _static(effect.params["noise_scale_x"]) == 300.0
        assert _static(effect.params["noise_levels"]) == 4.0
        assert effect.params["noise_octaves"] == 3
        assert effect.params["turbulence"] is True
        assert not report.lines()

    def test_a_broken_noise_size_falls_back_and_is_recorded(self) -> None:
        # 大きさが数として読めないと、掛けた先の横縦の大きさまで NaN になり模様が消える
        report = CompatibilityReport()
        brush = _brush(
            "NoiseBrushPlugin",
            Color1="#FF000000",
            Color2="#FFFFFFFF",
            NoiseParameter={"Size": float("nan"), "ScaleX": _still(150.0)},
        )
        effect = brush_effect(brush, report)
        assert effect is not None
        assert _static(effect.params["noise_scale_x"]) == 150.0
        assert any("大きさ" in line for line in report.lines())

    def test_a_shape_with_a_pattern_is_painted_white_then_patterned(self) -> None:
        # 図形の色がブラシの模様なら、形は白で描き、模様だけで塗る
        item = {
            "$type": "YukkuriMovieMaker.Project.Items.ShapeItem, YukkuriMovieMaker",
            "ShapeType2": "YukkuriMovieMaker.Shape.QuadrilateralShapePlugin, YukkuriMovieMaker",
            "ShapeParameter": {
                "$type": "YukkuriMovieMaker.Project.Items.RectangleShapeParameter, A",
                "Width": _still(100.0),
                "Height": _still(100.0),
                "StrokeThickness": _still(4000.0),
                "Brush": _brush("StripeBrushPlugin", Color1="#FFFF0000", Color2="#FF0000FF"),
            },
            "Frame": 0,
            "Length": 30,
            "Layer": 0,
        }
        (mapped,) = map_template([item], report=CompatibilityReport())
        assert mapped.clip.source is not None
        assert mapped.clip.source.params["color"] == (1.0, 1.0, 1.0, 1.0)
        assert mapped.clip.effects[0].kind == "brush_fill"
        assert mapped.clip.effects[0].params["pattern_only"] is True
