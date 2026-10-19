"""YMM4 の残りの映像エフェクトと図形の写し方

値の意味は YMM4 本体に描かせた試験（``tools/ymm4_probes.py`` の 3〜5 回目）の絵から
読み取った ここで見るのは「読み取った決まりどおりに写せているか」
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import place
from sashimono.compat.ymm4.decorations import map_video_effects
from sashimono.compat.ymm4.template import map_template
from sashimono.core.model import AnimatedValue, Effect, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.sources import centred_points


def _still(amount: float) -> dict[str, Any]:
    return {"Values": [{"Value": amount}], "Span": 0.0, "AnimationType": "なし"}


def _moving(first: float, last: float, style: str = "直線移動") -> dict[str, Any]:
    return {
        "Values": [{"Value": first}, {"Value": last}],
        "Span": 0.0,
        "AnimationType": style,
    }


def _value(value: object, frame: int = 0) -> float:
    assert isinstance(value, AnimatedValue)
    return value.at(frame)


def _map(name: str, **fields: Any) -> tuple[Effect, CompatibilityReport]:
    entry = {"$type": f"N.{name}, YukkuriMovieMaker", "IsEnabled": True, **fields}
    report = CompatibilityReport()
    result = map_video_effects([entry], report, length=60)
    (mapped,) = result.effects
    return mapped, report


def _shape_item(plugin: str, parameter: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "$type": "YukkuriMovieMaker.Project.Items.ShapeItem, YukkuriMovieMaker",
        "ShapeType2": f"N.{plugin}, YukkuriMovieMaker",
        "ShapeParameter": parameter,
        "Length": 60,
        "Frame": 0,
        "Layer": 0,
        **fields,
    }


class TestLightAndLens:
    def test_reflection_reads_the_nested_lighting(self) -> None:
        # 入れ子の Lighting と Heightmap を読み落とすと、
        # YMM4 の反射と押し出しの光の向きや形や色が既定のまま描かれる
        effect, report = _map(
            "ReflectionAndExtrusionEffect",
            LightingMode="DistantDiffuse",
            HeightmapMode="Bevel",
            Lighting={
                "LightSource": {"Azimuth": _still(-85.0), "Elevation": _still(30.0)},
                "Highlight": {
                    "Exponent": _still(2.0),
                    "Constant": _still(45.0),
                    "Color": "#FFFF0000",
                    "Blend": "Add",
                },
                "SurfaceScale": _still(8.0),
            },
            Heightmap={"BevelMode": "InvertedRound", "Thickness": _still(12.0)},
            Blur=_still(4.0),
            IsInvert=True,
        )
        assert not report.lines()
        assert effect.kind == "bevel_light"
        assert effect.params["lighting"] == "diffuse"
        assert _value(effect.params["azimuth"]) == -85.0
        assert _value(effect.params["constant"]) == 45.0
        assert effect.params["profile"] == "inverted_round"
        assert effect.params["color"] == (1.0, 0.0, 0.0, 1.0)
        assert effect.params["inverted"] is True

    def test_lens_blur(self) -> None:
        # BlurRadius を半径へ写し損ねると、YMM4 のレンズぼかしのぼけの強さが変わる
        effect, _ = _map("LensBlurEffect", BlurRadius=_still(20.0), Brightness=_still(150.0))
        assert effect.kind == "lens_blur"
        assert _value(effect.params["radius"]) == 20.0

    def test_bloom_can_colour_the_light(self) -> None:
        # 色付けの旗と色を読み落とすと、YMM4 で緑に色付けしたブルームの光が元の色のまま光る
        effect, report = _map(
            "BloomEffect",
            Strength=_still(120.0),
            Threshold=_still(50.0),
            Blur=_still(30.0),
            IsColorizationEnabled=True,
            Color="#FF00FF00",
        )
        assert not report.lines()
        assert effect.kind == "glow"
        assert effect.params["tinted"] is True
        assert effect.params["tint"] == (0.0, 1.0, 0.0, 1.0)


class TestDistortion:
    def test_fish_eye(self) -> None:
        # 射影の方式や角度を写し損ねると、YMM4 の魚眼レンズの歪み方が変わる
        effect, _ = _map(
            "FishEyeLensEffect", Projection="Equidistant", Angle=_still(120.0), Zoom=_still(90.0)
        )
        assert effect.kind == "fish_eye"
        assert effect.params["projection"] == "equidistant"
        assert _value(effect.params["angle"]) == 120.0

    def test_ripple_flips_y(self) -> None:
        # 波紋の中心の Y を裏返さないと、YMM4 で中心より下に置いた波紋が上で広がる
        effect, _ = _map(
            "RippleEffect",
            X=_still(100.0),
            Y=_still(40.0),
            Amplitude=_still(-20.0),
            WaveLength=_still(300.0),
            Period=_still(3.0),
        )
        assert effect.kind == "ripple"
        assert _value(effect.params["center_y"]) == -40.0

    def test_stretch(self) -> None:
        # 引き伸ばしの長さと中心に揃える旗を写し損ねると、YMM4 の引き伸ばしの量や伸びる起点が変わる
        effect, _ = _map(
            "StretchEffect",
            X=_still(0.0),
            Y=_still(0.0),
            Angle=_still(45.0),
            StretchLength=_still(300.0),
            Range=_still(50.0),
            IsCentering=False,
        )
        assert effect.kind == "stretch"
        assert _value(effect.params["stretch"]) == 300.0
        assert effect.params["centering"] is False

    def test_polar_and_tiling(self) -> None:
        # 芯の幅やタイルの縦の数を写し損ねると、YMM4 の極座標変換やタイル状に並べた絵の形が変わる
        polar, _ = _map("PolarTransformEffect", CoreWidth=_still(200.0), TwistAngle=_still(90.0))
        assert (polar.kind, _value(polar.params["core"])) == ("polar", 200.0)
        tiling, _ = _map("TilingEffect", X=_still(2.0), Y=_still(41.0))
        assert (tiling.kind, _value(tiling.params["count_y"])) == ("tile", 41.0)


class TestTiming:
    def test_repeat_zoom(self) -> None:
        # 間隔・イージング・中心の旗を写し損ねると、
        # YMM4 で繰り返し拡大するテンプレートの速さや動き方が変わる
        effect, _ = _map(
            "RepeatZoomEffect",
            Zoom=_still(150.0),
            ZoomX=_still(100.0),
            ZoomY=_still(100.0),
            Span=_still(2.0),
            EasingType="Sine",
            EasingMode="InOut",
            IsCentering=True,
        )
        assert effect.kind == "repeat_zoom"
        assert _value(effect.params["interval"]) == 2.0
        assert effect.params["easing"] == "sine"
        assert effect.params["centering"] is True

    def test_jump(self) -> None:
        # 跳ねる高さや周期を写し損ねると、YMM4 で弾ませたテンプレートの跳ね方が変わる
        effect, _ = _map(
            "JumpEffect",
            JumpHeight=_still(40.0),
            Stretch=_still(1.0),
            Period=_still(0.53),
            Distortion=_still(0.6),
            Interval=_still(0.07),
        )
        assert effect.kind == "jump"
        assert _value(effect.params["height"]) == 40.0
        assert _value(effect.params["period"]) == 0.53

    def test_the_jump_easing_waits_until_the_end(self) -> None:
        # Jump を知らない形として扱うか加減速へ丸めると、繰り返し回転が終わりで一度に回らず、
        # なめらかに回る
        effect, report = _map(
            "RepeatRotateEffect",
            X=_still(0.0),
            Y=_still(0.0),
            Z=_still(90.0),
            Is3D=False,
            Span=_still(1.0),
            EasingType="Jump",
            EasingMode="In",
        )
        assert not report.lines()
        assert effect.params["easing"] == "jump"

    def test_after_image(self) -> None:
        # 残像の強さとモードを写し損ねると、YMM4 の残像の濃さや前後の重なり方が変わる
        effect, _ = _map("AfterImageEffect", Strength=_still(53.6), Mode="Front")
        assert effect.kind == "after_image"
        assert _value(effect.params["strength"]) == pytest.approx(53.6)
        assert effect.params["mode"] == "front"

    def test_the_three_dimensional_effect_adds_nothing(self) -> None:
        # YMM4 に描かせても絵が変わらなかった（記録にも残さない）
        entry = {
            "$type": "N.ThreeDimensionalEffect, YukkuriMovieMaker",
            "IsEnabled": True,
            "X": _still(0.0),
            "Y": _still(300.0),
            "Length": _still(4.0),
        }
        report = CompatibilityReport()
        assert not map_video_effects([entry], report, length=60).effects
        assert not report.lines()


class TestWipeAndKey:
    @pytest.mark.parametrize(
        ("name", "pattern"),
        [
            ("ワイプ横", "horizontal"),
            ("円", "circle"),
            ("四角", "square"),
            ("時計回り", "clockwise"),
        ],
    )
    def test_the_bundled_wipe_images_become_patterns(self, name: str, pattern: str) -> None:
        # YMM4 に付いてくるワイプ画像の名前を模様へ引けないと、
        # 横や円のワイプが別の形かただのフェードで出る
        effect, report = _map(
            "InOutTransitionEffect",
            File=f"D:\\Program\\YukkuriMovieMaker_v4\\Resources\\Transition\\{name}.png",
            Tolerance=_still(3.0),
            Angle=_still(0.0),
            IsInEffect=True,
            IsOutEffect=False,
            EffectTimeSeconds=1.0,
            EasingType="Quart",
            EasingMode="InOut",
        )
        assert not report.lines()
        assert effect.kind == "inout_wipe"
        assert effect.params["pattern"] == pattern
        assert effect.params["easing"] == "quart"

    def test_an_unknown_wipe_image_falls_back_to_a_fade(self) -> None:
        # 自作のワイプ画像を黙ってフェードへ代えると、
        # ワイプの形が消えた理由がテンプレートの窓の注意書きに出ない
        effect, report = _map("InOutTransitionEffect", File="C:\\素材\\自作ワイプ.png")
        assert effect.params["pattern"] == "fade"
        assert any("自作ワイプ" in line for line in report.lines())

    def test_directional_key(self) -> None:
        # 背景色を写し損ねると、YMM4 の色で抜くキーで抜く色が変わり、抜けるはずの背景が残る
        effect, _ = _map(
            "DirectionalColorKeyEffect",
            BackgroundColor="#FF282828",
            ForegroundColor="#FFE7E7E7",
            EdgeSoftness=_still(3.0),
            NoiseThreshold=_still(0.02),
            OutputForeground=True,
        )
        assert effect.kind == "directional_key"
        background = effect.params["background"]
        assert isinstance(background, tuple)
        assert background[0] == pytest.approx(0x28 / 255)
        assert effect.params["output_foreground"] is True


class TestParticles:
    def test_the_emitter_keeps_the_screen_direction(self) -> None:
        # 放つ位置の Y をほかと同じく裏返すと、画面の上から降らせるパーティクルが画面の下から出る
        effect, report = _map(
            "ParticleOutputEffect",
            Rate=_still(400.0),
            Lifetime=_still(2.0),
            Preroll=_still(10.0),
            Size=_still(86.0),
            X=_still(0.0),
            Y=_still(-2000.0),
            EmitRange=_still(1300.0),
            EmitAngle=_still(-106.0),
            Speed=_still(1000.0),
            Gravity=_still(18000.0),
            EndScale=_still(67.0),
            Fade=_still(100.0),
            Randomness=_still(100.0),
        )
        assert effect.kind == "particles"
        # 放つ位置の Y は画面と同じ下向き（パーティクルのシェーダの中で裏返す）
        assert _value(effect.params["emitter_y"]) == -2000.0
        assert _value(effect.params["gravity"]) == 18000.0
        assert not report.lines()

    def test_particles_past_the_limit_are_recorded(self) -> None:
        """同時に居る粒（1 秒の数 x 寿命）が上限を超える設定は、互換性レポートに残る

        上限を超えた古い粒は描かれない 記録が無いと、粒が少なく見える理由を追えない
        """
        _, report = _map(
            "ParticleOutputEffect",
            Rate=_still(100.0),
            Lifetime={
                "Values": [{"Value": 2.0}, {"Value": 60.0}],
                "Span": 0.0,
                "AnimationType": "直線移動",
            },
        )
        assert any("パーティクル" in line and "上限" in line for line in report.lines())

    def test_peaks_at_different_times_are_not_recorded(self) -> None:
        """1 秒の数と寿命の山が別の時刻なら、2 つの山の積では記録しない

        1 秒の数が 100 から 10 へ減り、寿命が 5 から 50 へ延びる 同時に居る粒は
        どの時刻でも 1600 に届かない 山どうしを掛けると 5000 になり、上限を超えたと誤る
        """
        _, report = _map(
            "ParticleOutputEffect",
            Rate=_moving(100.0, 10.0),
            Lifetime=_moving(5.0, 50.0),
        )
        assert not report.lines()

    def test_an_overshooting_curve_is_recorded(self) -> None:
        """行き過ぎて戻る曲線（Back）で、端の値より多い粒が居る時刻も記録する

        1 秒の数が 100 から 2000（寿命 2 秒）なら端では 4000 で上限の中 Back_Out は
        途中で 2000 を超えるので、その間は上限を超える
        """
        _, report = _map(
            "ParticleOutputEffect",
            Rate=_moving(100.0, 2000.0, "Back_Out"),
            Lifetime=_still(2.0),
        )
        assert any("パーティクル" in line and "上限" in line for line in report.lines())

    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), 1e308])
    def test_a_broken_particle_count_is_recorded_not_raised(self, rate: float) -> None:
        """数でない粒の数（NaN・無限大・掛けて溢れる値）でも、読み込みは止まらず記録に残る"""
        _, report = _map("ParticleOutputEffect", Rate=_still(rate), Lifetime=_still(1e308))
        assert any("パーティクル" in line for line in report.lines())

    def test_particles_within_the_limit_are_not_recorded(self) -> None:
        """配布物の雪（1 秒に 100 粒・寿命 15 秒）は上限の中に収まり、記録は出ない"""
        _, report = _map("ParticleOutputEffect", Rate=_still(100.0), Lifetime=_still(15.0))
        assert not report.lines()


class TestShapes:
    def test_the_pen_points_move_to_the_centre(self) -> None:
        # ペンの線の点・太さ・描く範囲を写し損ねると、
        # YMM4 の手描きの線のテンプレートが違う位置・太さ・長さで出る
        item = _shape_item(
            "PenShapePlugin",
            {
                "Strokes": [
                    {
                        "StylusPoints": [{"X": 960.0, "Y": 540.0}, {"X": 1060.0, "Y": 640.0}],
                        "DrawingAttributes": {"Color": "#FF74C7CB", "Width": 640.0},
                    }
                ],
                "Thickness": _still(50.0),
                "Length": _still(80.0),
                "Offset": _still(10.0),
            },
        )
        report = CompatibilityReport()
        source = map_template([item], report=report)[0].clip.source
        assert source is not None
        # 点は画面の左上からの座標のまま持ち、中心を引いて Y を上向きへ直すのは描くとき
        # この印が抜けると、線が中心からずれたうえ上下が逆さまに出る
        assert source.params["points"] == "960,540;1060,640"
        assert source.params["points_from"] == "corner"
        values = dict(source.params)
        assert centred_points(values, 1920, 1080) == [(0.0, 0.0), (100.0, -100.0)]
        assert _value(source.params["line_width"]) == pytest.approx(320.0)
        assert _value(source.params["trim_end"]) == pytest.approx(90.0)

    def test_the_timer_counts_down_to_its_first_value(self) -> None:
        # 数え下げの旗や長さを写し損ねると、
        # YMM4 のカウントダウンのテンプレートが数え上げで出るか違う時刻で終わる
        item = _shape_item(
            "TimerShapePlugin",
            {
                "Format": "s",
                "Direction": "CountDown",
                "InitialValue": _still(0.99),
                "PlaybackRate": _still(100.0),
                "FontSize": 520.0,
                "FontColor": "#FFFFFFFF",
                "BasePoint": "CenterCenter",
            },
        )
        source = map_template([item], report=CompatibilityReport())[0].clip.source
        assert source is not None and source.kind == "text"
        assert source.params["timer_format"] == "s"
        assert source.params["timer_countdown"] is True
        assert source.params["timer_length"] == 60

    def test_the_concentration_lines(self) -> None:
        # 密度や中心の幅を写し損ねると、YMM4 の集中線のテンプレートの線の混み方や柔らかさが変わる
        item = _shape_item(
            "ConcentrationLineShapePlugin",
            {
                "Size": _still(1000.0),
                "Density": _still(90.0),
                "Thickness": _still(60.0),
                "Length": _still(70.0),
                "CenterWidth": _still(50.0),
                "Speed": _still(4.0),
                "Stroke": "#FFFFEBC4",
            },
        )
        report = CompatibilityReport()
        source = map_template([item], report=report)[0].clip.source
        assert source is not None
        assert source.params["shape"] == "concentration"
        assert _value(source.params["density"]) == 90.0
        assert _value(source.params["softness"]) == 50.0
        assert not report.lines()

    def test_an_inverted_item_is_flipped_before_it_is_placed(self) -> None:
        # 反転を配置の後へ回すと、ずらして置いた後の絵を裏返すことになり、
        # X で右へ置いた反転アイテムが YMM4 と違う所に出る
        item = _shape_item(
            "QuadrilateralShapePlugin",
            {"Width": _still(100.0), "Height": _still(50.0)},
            IsInverted=True,
            X=_still(200.0),
        )
        clip = map_template([item], report=CompatibilityReport())[0].clip
        kinds = [effect.kind for effect in clip.effects]
        assert kinds == ["flip", "transform"]
        assert clip.effects[0].params["horizontal"] is True


def _drawn(item: dict[str, Any], size: int = 200) -> np.ndarray:
    """YMM4 のアイテム 1 つを読み、画面の真ん中に置いて 1 枚描く"""
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    # 読み込みや置く所で落ちてもコンテキストは返す 返さないと後ろの GPU の試験が道連れになる
    try:
        settings = ProjectSettings(width=size, height=size, frame_rate=FrameRate(30))
        project = Project.create(settings)
        for command in place(map_template([item], report=CompatibilityReport()), project):
            project = command.apply(project)
        renderer = FrameRenderer(project, context=context)
        try:
            return renderer.render(0)
        finally:
            renderer.close()
    finally:
        context.release()


def _cropped_by_angle(angle: float) -> np.ndarray:
    white = {
        "$type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushParameter, YukkuriMovieMaker",
        "Color": "#FFFFFFFF",
    }
    return _drawn(
        _shape_item(
            "QuadrilateralShapePlugin",
            {
                "SizeMode": "WidthHeight",
                "Width": _still(160.0),
                "Height": _still(160.0),
                "Brush": {"Type": "N.SolidColorBrushPlugin, YukkuriMovieMaker", "Parameter": white},
            },
            VideoEffects=[
                {
                    "$type": "N.CropByAngleEffect, YukkuriMovieMaker",
                    "IsEnabled": True,
                    "X": _still(0.0),
                    "Y": _still(0.0),
                    "Angle": _still(angle),
                    "Blur": _still(0.0),
                    "Width": _still(40.0),
                }
            ],
        )
    )


class TestCropByAngle:
    """角度で切り抜きは、角度の向きに伸びる帯を残す（YMM4 の書き出しから読んだ）

    ドッグタグ風テロップ（角度 0・幅 304 で円を切る）は、YMM4 では横長の札になる
    帯を角度に直交する向きに取ると縦長の細い板になり、SFっぽい吹き出しの 45 度の
    切り欠きも反対の角へ付く（``tools/ymm4_compare.py`` の差 12.6 と 26.5）
    """

    def test_an_angle_of_zero_keeps_a_level_band(self) -> None:
        # 角度 0 の帯を縦に取ると、ドッグタグ風テロップが YMM4 の横長の札ではなく縦長の細い板になる
        image = _cropped_by_angle(0.0)
        assert image[100, 30, 0] > 200, "横の帯の端が切れている"
        assert image[30, 100, 0] < 20, "帯の上まで残っている（縦の帯になっている）"

    def test_a_positive_angle_turns_the_band_clockwise(self) -> None:
        # YMM4 の角度は画面の上で時計回り 45 度なら左上から右下へ下がる帯
        image = _cropped_by_angle(45.0)
        assert image[45, 45, 0] > 200, "左上が切れている"
        assert image[155, 155, 0] > 200, "右下が切れている"
        assert image[45, 155, 0] < 20, "右上が残っている（帯が反対の斜めに伸びている）"

    @pytest.mark.parametrize("keep", [True, False])
    def test_the_band_is_measured_from_the_centre_point(self, keep: bool) -> None:
        """前に中心点があれば、帯の位置（X）は絵の中央ではなく中心点から測る

        SFっぽい吹き出し(右) の名札は、左端を中心点にしてから角度で切り抜く YMM4 の
        書き出しでは、帯が左端から X の所にあった（位置を保つ・保たないのどちらも）
        絵の中央から測ると、名札が 233 と 500 画素ずれた所に出る（#179）
        """
        white = {
            "$type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushParameter, YukkuriMovieMaker",
            "Color": "#FFFFFFFF",
        }
        image = _drawn(
            _shape_item(
                "QuadrilateralShapePlugin",
                {
                    "SizeMode": "WidthHeight",
                    "Width": _still(160.0),
                    "Height": _still(40.0),
                    "Brush": {
                        "Type": "N.SolidColorBrushPlugin, YukkuriMovieMaker",
                        "Parameter": white,
                    },
                },
                VideoEffects=[
                    {
                        "$type": "N.CenterPointEffect, YukkuriMovieMaker",
                        "IsEnabled": True,
                        "Horizontal": "Left",
                        "Vertical": "Center",
                        "X": _still(0.0),
                        "Y": _still(0.0),
                        "IsKeepPosition": keep,
                    },
                    {
                        "$type": "N.CropByAngleEffect, YukkuriMovieMaker",
                        "IsEnabled": True,
                        "X": _still(20.0),
                        "Y": _still(0.0),
                        # 縦に伸びる帯 横の位置だけで見分けられる
                        "Angle": _still(90.0),
                        "Blur": _still(0.0),
                        "Width": _still(20.0),
                    },
                ],
            ),
            size=400,
        )
        # 左端は、位置を保てば 120、保たなければ中心点が絵の置き場（200）へ寄る
        left = 120 if keep else 200
        row = image[200, :, 0]
        assert row[left + 20] > 200, "中心点から X の所の帯が無い"
        assert row[left + 80 + 20] < 20, "絵の中央から X の所に帯が残っている"


class TestBrokenValues:
    def test_a_noise_brush_with_a_broken_octave_still_loads(self) -> None:
        # NaN を round へ渡すと例外になり、同じテンプレートのほかのアイテムまで読めなくなる
        item = _shape_item(
            "QuadrilateralShapePlugin",
            {
                "Width": _still(100.0),
                "Height": _still(100.0),
                "Brush": {
                    "Type": "YukkuriMovieMaker.Brush.NoiseBrushPlugin, YukkuriMovieMaker",
                    "Parameter": {
                        "NoiseType": "Perlin",
                        "Color1": "#FF000000",
                        "Color2": "#FFFFFFFF",
                        "NoiseParameter": {"Octaves": float("nan")},
                    },
                },
            },
        )
        report = CompatibilityReport()
        (mapped,) = map_template([item], report=report)
        painted = next(e for e in mapped.clip.effects if e.kind == "brush_fill")
        assert painted.params["noise_octaves"] == 5
        assert any("重ね数" in line for line in report.lines())

    def test_an_unknown_transition_setting_is_recorded(self) -> None:
        # 知らないイージングや向きを黙って直線にすると、
        # トランジションの動き方が変わった理由がテンプレートの窓の注意書きに出ない
        item = {
            "$type": "YukkuriMovieMaker.Project.Items.TransitionItem, YukkuriMovieMaker",
            "TransitionType": "N.FadeTransitionPlugin, YukkuriMovieMaker",
            "TransitionParameter": {"EasingType": "ぬるっと", "EasingMode": "Middle"},
            "Frame": 0,
            "Length": 30,
        }
        report = CompatibilityReport()
        (mapped,) = map_template([item], report=report)
        assert mapped.clip.source is not None
        assert mapped.clip.source.params["easing"] == "linear"
        lines = report.lines()
        assert any("イージング" in line for line in lines)
        assert any("向き" in line for line in lines)
