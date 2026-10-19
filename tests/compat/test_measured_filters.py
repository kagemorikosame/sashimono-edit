"""YMM4 と AviUtl2 に描かせて測った値の意味（#177・#184・#188・#192・#195・#198）

測った数はどれも合成の見本（白い四角・色の升・Arial の H）を本体に書き出させて読んだ物
YMM4 は ``tools/ymm4_probes.py`` の 6 回目と 7 回目、AviUtl2 は ``tools/aviutl_filter_probes.py``
測った数をここへ定数で持ち、読み込みと描き方がその数を出すかを見る
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pytest

from sashimono.compat.aviutl.exo import ExoEntry
from sashimono.compat.aviutl.mapping import _filter, script_filter_effects
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.decoration import DECORATIONS, decoration_params
from sashimono.compat.ymm4.decorations import map_decorations, map_video_effects
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import PREVIOUS_OBJECT, SHAPE, TEXT
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.sources import centred_points

WIDTH, HEIGHT = 400, 400


def _still(amount: float) -> dict[str, Any]:
    return {"Values": [{"Value": amount}], "Span": 0.0, "AnimationType": "なし"}


def _value(value: object) -> float:
    assert isinstance(value, AnimatedValue)
    return value.at(0)


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture
def draw(gl: OffscreenGLContext) -> Callable[..., np.ndarray]:
    """1 クリップだけのプロジェクト（400x400・30fps・60 フレーム）を描く"""

    def render(
        source: GeneratedSource, effects: tuple[Effect, ...] = (), *, frame: int = 0
    ) -> np.ndarray:
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
        clip = Clip(timeline_start=0, duration=60, source=source, effects=effects)
        track = Track(kind=TrackKind.VIDEO, clips=(clip,))
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            return np.asarray(renderer.render(frame))[..., :3]
        finally:
            renderer.close()

    return render


def _square(
    size: int = 200, colour: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> GeneratedSource:
    return SHAPE.create(shape="rect", width=size, height=size, color=(*colour, 1.0))


def _box(image: np.ndarray, threshold: int = 128) -> tuple[int, int, int, int] | None:
    """明るい所の外形 左・上・右・下（右と下は 1 つ外）"""
    lit = image.max(axis=2) >= threshold
    rows = np.nonzero(lit.any(axis=1))[0]
    columns = np.nonzero(lit.any(axis=0))[0]
    if rows.size == 0:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


# --- YMM4 ---------------------------------------------------------------------------------


class TestYmm4TextStyles:
    """Arial 100 の H で測った YMM4 の文字装飾（#184） 名前は本体の列挙の名前"""

    #: 片側の縁の太さ（画素） Light は細い縁、Sharp は角の尖った同じ太さの縁
    BORDERS: ClassVar[dict[str, float]] = {
        "Border": 8.0,
        "BorderLight": 4.0,
        "SharpBorder": 8.0,
        "SharpBorderLight": 4.0,
    }

    @pytest.mark.parametrize(("style", "width"), sorted(BORDERS.items()))
    def test_the_border_is_as_thick_as_ymm4(self, style: str, width: float) -> None:
        # 前は AviUtl2 の呼び名から推した名前（ThinBorder など）で引いていて、本体の名前は
        # Border 以外どれも「知らない文字装飾」になり、縁が付かなかった
        report = CompatibilityReport()
        result = map_decorations([], report, size=100.0, style=style, style_colour="#FFFF0000")
        assert not any("文字装飾:" in line for line in report.lines())
        assert _value(result.params["border_width"]) == pytest.approx(width)

    def test_the_sharp_corner_is_recorded(self) -> None:
        # Sharp の付く縁取りは太さだけ合わせ、角は丸いまま描く 記録が無いと、尖った角が
        # 丸まったことに互換性レポートで気付けない
        report = CompatibilityReport()
        map_decorations([], report, size=100.0, style="SharpBorder")
        assert any("尖った角: SharpBorder" in line for line in report.lines())

    def test_the_shadow_moves_four_pixels(self) -> None:
        # 前は AviUtl2 の表（0.06）から引いて 6 画素ずれ、影が字から 2 画素離れて太く見えた
        result = map_decorations([], CompatibilityReport(), size=100.0, style="Shadow")
        assert _value(result.params["shadow_x"]) == pytest.approx(4.0)
        # 右下へずれる こちらの Y は上が正
        assert _value(result.params["shadow_y"]) == pytest.approx(-4.0)

    def test_the_light_shadow_is_half_as_dark(self) -> None:
        # 前は ShadowLight を知らない名前として影を付けず、薄い影の字がただの白い字になった
        result = map_decorations(
            [], CompatibilityReport(), size=100.0, style="ShadowLight", style_colour="#FF000000"
        )
        colour = result.params["shadow_color"]
        assert isinstance(colour, tuple)
        assert colour[3] == pytest.approx(0.5)


def _outline(**fields: Any) -> dict[str, Any]:
    return {
        "$type": "YukkuriMovieMaker.Project.Effects.OutlineEffect, YukkuriMovieMaker",
        "IsEnabled": True,
        "StrokeThickness": _still(10.0),
        "Blur": _still(0.0),
        "Opacity": _still(100.0),
        "X": _still(0.0),
        "Y": _still(0.0),
        **fields,
    }


class TestYmm4Outline:
    def test_the_outline_moves_up_with_a_negative_y(self) -> None:
        # YMM4 に Y -20 を描かせると縁だけが 20 上へ動いた（#192） 読まずにいると、配布物の
        # グループの縁（Y -1）が 1 画素下にずれて描かれる
        result = map_video_effects(
            [_outline(Y=_still(-20.0), X=_still(30.0))], CompatibilityReport()
        )
        (effect,) = result.effects
        assert effect.kind == "border"
        assert _value(effect.params["offset_y"]) == pytest.approx(20.0)
        assert _value(effect.params["offset_x"]) == pytest.approx(30.0)

    def test_only_the_edge_moves(self, draw: Callable[..., np.ndarray]) -> None:
        # 縁だけでなく本体まで動くか、縁が動かないと、YMM4 で縁をずらした字や図形の
        # 縁の位置が YMM4 と食い違う
        border = registry.require("border").create(
            width=10, color=(0.0, 1.0, 1.0, 1.0), offset_y=20
        )
        image = draw(_square(), (border,))
        cyan = (image[..., 0] < 80) & (image[..., 1] > 150)
        rows = np.nonzero(cyan.any(axis=1))[0]
        # 白い四角は 100〜300 縁は上へ 20 ずれて 70〜290 の範囲に出る（下の縁は四角に隠れる）
        assert int(rows[0]) == pytest.approx(70, abs=1)
        white = np.where((image.min(axis=2) > 250)[..., None], image, 0)
        assert _box(white, 250) == pytest.approx((100, 100, 300, 300), abs=1)


class TestYmm4InOut:
    def test_value3_brings_the_picture_forward(self, draw: Callable[..., np.ndarray]) -> None:
        # InOutMove の Value3 は手前へ出す量 YMM4 は 300 の四角が 300 で 428、-300 で 230 から
        # 始まった（奥行き 1000 ほどの遠近） 前は読まずに記録だけしていた
        report = CompatibilityReport()
        entry = {
            "$type": "YukkuriMovieMaker.Project.Effects.InOutMoveEffect, YukkuriMovieMaker",
            "IsEnabled": True,
            "Value": 0.0,
            "Value2": 0.0,
            "Value3": 100.0,
            "IsInEffect": True,
            "IsOutEffect": False,
            "EffectTimeSeconds": 2.0,
            "EasingType": "Linear",
            "EasingMode": "In",
        }
        (effect,) = map_video_effects([entry], report, length=60).effects
        assert not report.lines()
        assert _value(effect.params["offset_z"]) == 100.0
        # 動く値の形で来ても動きを落とさない 素の数で読むと最初の値で止まる
        moving = {
            **entry,
            "Value3": {
                "Values": [{"Value": 0.0}, {"Value": 200.0}],
                "Span": 0.0,
                "AnimationType": "直線移動",
            },
        }
        (animated_effect,) = map_video_effects([moving], CompatibilityReport(), length=60).effects
        depth = animated_effect.params["offset_z"]
        assert isinstance(depth, AnimatedValue)
        assert depth.at(59) > depth.at(0) + 150.0
        box = _box(draw(_square(100), (effect,)))
        assert box is not None
        # 1024 / (1024 - 100) で 108 前後（YMM4 の奥行き 1000 なら 111）
        assert box[2] - box[0] == pytest.approx(110, abs=4)

    def test_the_exit_runs_the_entrance_backwards(self, draw: Callable[..., np.ndarray]) -> None:
        # 退場は終わりから数えた残りで登場と同じ曲線を引く Back・Out の退場で残り 0.23 なら
        # 1 - Back_Out(0.23) = 0.22 だけずれる 進み具合を曲線へそのまま渡していたころは、
        # 行き過ぎて切り詰めた 1 になり、終わりの 7 フレーム前でもう退場しきっていた
        move = registry.require("inout_offset").create(
            offset_x=100,
            offset_y=0,
            effect_in=False,
            effect_out=True,
            effect_time=1.0,
            easing="back",
            easing_mode="out",
        )
        box = _box(draw(_square(100), (move,), frame=53))
        assert box is not None
        shift = (box[0] + box[2]) / 2 - WIDTH / 2
        assert shift == pytest.approx(22, abs=3)


class TestYmm4Monocolor:
    def test_without_keeping_brightness_it_paints_the_colour(self) -> None:
        # レトロなカウントダウン3秒 の数字は暗い茶色（#292110）に塗られていた 前は彩度を抜く
        # だけで白いまま残った
        entry = {
            "$type": "YukkuriMovieMaker.Project.Effects.MonocolorizationEffect, YukkuriMovieMaker",
            "IsEnabled": True,
            "Strength": _still(100.0),
            "Color": "#FF292110",
            "KeepBrightness": False,
        }
        (effect,) = map_video_effects([entry], CompatibilityReport()).effects
        assert effect.kind == "fill"
        assert effect.params["keep_luma"] is False
        assert effect.params["color"] == pytest.approx((0x29 / 255, 0x21 / 255, 0x10 / 255, 1.0))


def _noise(strength: float, **extra: Any) -> dict[str, Any]:
    return {
        "$type": "YukkuriMovieMaker.Project.Effects.NoiseEffect, YukkuriMovieMaker",
        "IsEnabled": True,
        "NoiseType": "Random",
        "NoiseParameter": {
            "$type": "N.RandomNoiseParameter, YukkuriMovieMaker",
            "Strength": _still(strength),
            "Threshold": _still(0.0),
            "Levels": _still(256.0),
            "ScaleX": _still(200.0),
            "ScaleY": _still(200.0),
        },
        "IsColor": False,
        "IsAlpha": True,
        **extra,
    }


class TestYmm4Noise:
    """``NoiseEffect`` の不透明度 乱数なので面の平均で見る（#177 雨）"""

    #: 白い四角の不透明度の平均 YMM4 に書き出させて測った値
    MEANS: ClassVar[dict[float, float]] = {
        50.0: 0.75,
        100.0: 0.50,
        120.0: 0.405,
        150.0: 0.25,
        200.0: 0.0,
    }

    @pytest.mark.parametrize(("strength", "mean"), sorted(MEANS.items()))
    def test_the_mean_opacity_matches_ymm4(
        self, draw: Callable[..., np.ndarray], strength: float, mean: float
    ) -> None:
        # 強さと不透明度の対応がずれると、YMM4 の雨や砂嵐のノイズを読み込んだ絵が濃すぎるか薄すぎる
        (effect,) = map_video_effects([_noise(strength)], CompatibilityReport()).effects
        assert effect.kind == "brush_fill"
        assert effect.params["noise_mask"] == "alpha"
        image = draw(_square(360), (effect,)).astype(float)
        region = image[30:370, 30:370].mean(axis=2) / 255.0
        assert float(region.mean()) == pytest.approx(mean, abs=0.05)

    def test_the_random_grains_are_not_one_grey(self, draw: Callable[..., np.ndarray]) -> None:
        # 砂嵐の値は升ごとにばらばら 前は勾配ノイズを升の角で引いて、どこでも 0.5 の一枚だった
        (effect,) = map_video_effects([_noise(100.0)], CompatibilityReport()).effects
        region = draw(_square(200), (effect,)).astype(float)[120:280, 120:280, 0] / 255.0
        assert float(region.std()) > 0.15


class TestYmm4Pen:
    def test_the_points_count_from_the_screen_corner(self) -> None:
        # YMM4 は 1920x1080 と 1280x720 のどちらでも、同じ点を画面の左上から同じ画素の所に
        # 描いた（#198） 1920x1080 の真ん中を決め打ちで引くと、1280x720 で線がずれる
        values: dict[str, object] = {"points": "860,440;1060,640", "points_from": "corner"}
        assert centred_points(values, 1920, 1080) == [(-100.0, 100.0), (100.0, -100.0)]
        assert centred_points(values, 1280, 720) == [(220.0, -80.0), (420.0, -280.0)]


class TestVerticalTextAnchor:
    def test_a_vertical_text_follows_the_anchor(self, draw: Callable[..., np.ndarray]) -> None:
        # 縦書きは横の基準と縦の基準を見ずに、いつも塊の真ん中を位置に置いていた
        # 左・上を基準にしたら、塊の左の端と上の端が位置に来る
        text = TEXT.create(
            text="HH", size=60, font="Arial", vertical=True, anchor="left", valign="top"
        )
        box = _box(draw(text))
        assert box is not None
        assert box[0] >= WIDTH / 2 - 2
        assert box[1] >= HEIGHT / 2 - 2


class TestYmm4TextAnchor:
    def test_the_left_base_point_puts_the_left_edge_on_the_position(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # YMM4 の基準位置 LeftTop は文字の塊の左の端が位置 真ん中に置くと 4 文字の H が
        # 130 画素ほど左へずれた（#198 フルバレ○トファイアー再現文字装飾）
        text = TEXT.create(
            text="HHHH", size=60, font="Arial", align="left", anchor="left", valign="top"
        )
        box = _box(draw(text))
        assert box is not None
        assert box[0] == pytest.approx(WIDTH / 2, abs=6)
        assert box[1] >= HEIGHT / 2 - 2


class TestYmm4Polar:
    def test_the_left_edge_starts_at_the_top_going_counterclockwise(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 左が赤・右が青の横のグラデーションを YMM4 に巻かせると、真上の継ぎ目の左が赤、右が青
        # だった（#198） 時計回りに巻くと左右が裏返る
        gradient = registry.require("brush_fill").create(
            pattern="linear",
            stops=2,
            color0=(1.0, 0.0, 0.0, 1.0),
            color1=(0.0, 0.0, 1.0, 1.0),
            size=300,
            angle=0,
        )
        polar = registry.require("polar").create(core=0, twist=0)
        image = draw(SHAPE.create(shape="rect", width=300, height=100), (gradient, polar))
        left, right = image[125, 190].astype(int), image[125, 210].astype(int)
        assert left[0] > left[2]
        assert right[2] > right[0]


# --- AviUtl2 ------------------------------------------------------------------------------


def _entry(name: str, **params: str) -> ExoEntry:
    return ExoEntry(name=name, params=params)


class TestAviUtlDecorations:
    """Arial 100 の H で測った AviUtl2 の文字装飾（#184）"""

    #: 片側の縁の太さ（画素）
    BORDERS: ClassVar[dict[str, float]] = {
        "縁取り文字": 8.0,
        "縁取り文字（細）": 4.0,
        "縁取り文字（太）": 12.0,
        "縁取り文字（角）": 4.0,
    }

    @pytest.mark.parametrize(("name", "width"), sorted(BORDERS.items()))
    def test_the_border_is_as_thick_as_aviutl2(self, name: str, width: float) -> None:
        # 前の表は AviUtl2 より片側 2〜3 画素細かった（縁取り文字 5・太 9）
        params = decoration_params(DECORATIONS[name], 100.0, (1.0, 0.0, 0.0, 1.0))
        assert _value(params["border_width"]) == pytest.approx(width)

    def test_the_shadow_moves_five_pixels(self) -> None:
        # 前の 0.06 では 6 画素ずれ、AviUtl2 より影が 1 画素字から離れた
        params = decoration_params(DECORATIONS["影付き文字"], 100.0, (0.0, 0.0, 0.0, 1.0))
        assert _value(params["shadow_x"]) == pytest.approx(5.0)

    def test_the_edge_of_a_tall_letter_is_cut_too(self, draw: Callable[..., np.ndarray]) -> None:
        # 字の形だけを先に切ると、後から付ける縁が枠の外へ広がる 縁も枠で切る
        text = TEXT.create(
            text="H<th2>H<th>H",
            size=100,
            font="Arial",
            layout="aviutl",
            border_width=8,
            border_color=(1.0, 0.0, 0.0, 1.0),
        )
        plain = TEXT.create(
            text="HHH",
            size=100,
            font="Arial",
            layout="aviutl",
            border_width=8,
            border_color=(1.0, 0.0, 0.0, 1.0),
        )
        tall, flat = _box(draw(text), 40), _box(draw(plain), 40)
        assert tall is not None and flat is not None
        # 枠（高さ 114）で切れる 字の形だけを切ると縁が上下へ 8 ずつ出て 130 になる
        assert tall[3] - tall[1] <= 116
        assert flat[3] - flat[1] < tall[3] - tall[1]

    def test_a_tall_letter_is_cut_by_the_text_frame(self, draw: Callable[..., np.ndarray]) -> None:
        # AviUtl2 は ``H<th2>H<th>H`` の真ん中の字を文字の枠（高さ 114）の上下で切った
        # 切らずに描くと、縦に 2 倍の H が枠の上下へ 20 画素ずつはみ出す
        text = TEXT.create(text="H<th2>H<th>H", size=100, font="Arial", layout="aviutl")
        box = _box(draw(text))
        assert box is not None
        assert box[3] - box[1] <= 116


class TestAviUtlColorCorrection:
    """3 色の升に掛けた 色調補正 の測り値（#188） 色は 0〜255 の sRGB"""

    NEUTRAL: ClassVar[dict[str, str]] = {
        "明るさ": "100.0",
        "コントラスト": "100.0",
        "色相": "0.0",
        "輝度": "100.0",
        "彩度": "100.0",
    }
    ORANGE = (200, 100, 50)
    #: 項目を 1 つずつ動かしたときの橙の升の色
    MEASURED: ClassVar[dict[tuple[str, str], tuple[int, int, int]]] = {
        ("明るさ", "150.0"): (255, 227, 177),
        ("明るさ", "50.0"): (72, 0, 0),
        ("コントラスト", "150.0"): (236, 86, 11),
        ("輝度", "150.0"): (255, 162, 112),
        ("彩度", "50.0"): (162, 112, 87),
        ("色相", "90.0"): (65, 173, 28),
    }

    def _effect(self, **change: str) -> Effect:
        effect = _filter(
            _entry("色調補正", **{**self.NEUTRAL, **change}), (), CompatibilityReport()
        )
        assert effect is not None
        return effect

    def test_all_hundreds_leave_the_colour(self, draw: Callable[..., np.ndarray]) -> None:
        # 100 が元のまま 前は Sashimono の色調補正へ値のまま入れ、明るさ +100% で白く飛んだ
        colour = (self.ORANGE[0] / 255.0, self.ORANGE[1] / 255.0, self.ORANGE[2] / 255.0)
        image = draw(_square(200, colour), (self._effect(),))
        assert tuple(int(v) for v in image[200, 200]) == pytest.approx(self.ORANGE, abs=2)

    @pytest.mark.parametrize(("change", "expected"), sorted(MEASURED.items()))
    def test_each_item_matches_aviutl2(
        self,
        draw: Callable[..., np.ndarray],
        change: tuple[str, str],
        expected: tuple[int, int, int],
    ) -> None:
        # 項目ごとの換算がずれると、AviUtl2 で色調補正を掛けたオブジェクトが
        # 読み込み後に明るさや色相の違う色で出る
        colour = (self.ORANGE[0] / 255.0, self.ORANGE[1] / 255.0, self.ORANGE[2] / 255.0)
        image = draw(_square(200, colour), (self._effect(**{change[0]: change[1]}),))
        assert tuple(int(v) for v in image[200, 200]) == pytest.approx(expected, abs=4)

    def test_the_script_reads_the_same_way(self) -> None:
        # スクリプトの obj.effect だけ別の写し方をしていたころは、同じ値でもエイリアスと
        # スクリプトで色が違った（アクリル矩形の板の明るさが変わる）
        (effect,) = script_filter_effects("色調補正", {"明るさ": 135, "輝度": 30})
        assert effect.kind == "color_correct"
        assert _value(effect.params["brightness"]) == 135.0
        assert _value(effect.params["luma"]) == 30.0

    def test_the_aviutl1_contrast_is_read(self) -> None:
        # AviUtl1 の .exa はコントラストを半角で書く 読まずにいると、コントラストを上げた
        # エイリアスが元のままの色で描かれ、読めなかったことも記録にしか残らない
        effect = self._effect(**{"ｺﾝﾄﾗｽﾄ": "150.0"})
        assert _value(effect.params["contrast"]) == 150.0


class TestAviUtlSlantClip:
    def _clipped(self, draw: Callable[..., np.ndarray], width: str) -> np.ndarray:
        effect = _filter(
            _entry("斜めクリッピング", 中心X="0.0", 中心Y="0.0", 角度="0.0", ぼかし="0", 幅=width),
            (),
            CompatibilityReport(),
        )
        assert effect is not None
        return draw(_square(200), (effect,))

    def test_zero_width_drops_the_bottom(self, draw: Callable[..., np.ndarray]) -> None:
        # 幅 0 は線の片側を落とす 幅を帯として読み違えると、線 1 本しか残らない（前の crop_angle）
        assert _box(self._clipped(draw, "0")) == pytest.approx((100, 100, 300, 200), abs=1)

    def test_a_positive_width_keeps_a_band(self, draw: Callable[..., np.ndarray]) -> None:
        # AviUtl2 は 300 の四角に幅 100 で、線を真ん中にした 98 の帯だけを残した
        assert _box(self._clipped(draw, "100")) == pytest.approx((100, 150, 300, 250), abs=2)

    def test_a_negative_width_drops_a_band(self, draw: Callable[..., np.ndarray]) -> None:
        # 負の幅で帯を消せないと、AviUtl2 で真ん中の帯を抜いた斜めクリッピングが
        # 帯の残った絵や片側の欠けた絵で読み込まれる
        image = self._clipped(draw, "-100")
        assert _box(image) == pytest.approx((100, 100, 300, 300), abs=1)
        assert int(image[200, 200].max()) < 20


class TestAviUtlBlurAndBorder:
    def test_a_fixed_size_blur_keeps_the_square(self, draw: Callable[..., np.ndarray]) -> None:
        # サイズ固定 の四角は 範囲 30 でも端が薄れず、300 のまま白だった
        effect = _filter(_entry("ぼかし", 範囲="30", サイズ固定="1"), (), CompatibilityReport())
        assert effect is not None
        image = draw(_square(200), (effect,))
        assert _box(image, 250) == pytest.approx((100, 100, 300, 300), abs=1)

    def test_a_full_blur_still_draws_the_edge(self, draw: Callable[..., np.ndarray]) -> None:
        # ぼかし 100% は太さを全部ぼかしに回して太さ 0 になる 太さ 0 で何も描かない作りの
        # ままだと、縁が黙って消える
        effect = _filter(
            _entry("縁取り", サイズ="10", ぼかし="100", 縁色="ff0000"), (), CompatibilityReport()
        )
        assert effect is not None
        image = draw(_square(200), (effect,))
        assert int(image[200, 95, 0]) > 20

    def test_the_border_blur_is_a_share_of_its_size(self) -> None:
        # ぼかし 20 の縁（サイズ 10）は外の端から 3 画素かけて落ちた ぼかしは太さの割合
        report = CompatibilityReport()
        effect = _filter(_entry("縁取り", サイズ="10", ぼかし="20", 縁色="ff0000"), (), report)
        assert effect is not None
        assert not report.lines()
        assert _value(effect.params["blur"]) == pytest.approx(2.0)
        assert _value(effect.params["width"]) == pytest.approx(8.0)


class TestAviUtlContents:
    def test_the_framebuffer_becomes_the_screen_copy(self) -> None:
        # 前は未対応の中身として何も置かず、下の絵に掛けるはずの効果がまるごと消えた
        from sashimono.compat.aviutl.mapping import _content

        report = CompatibilityReport()
        source, path, kind = _content(
            _entry("フレームバッファ", フレームバッファをクリア="0"), (), report
        )
        assert source is not None and source.kind == "framebuffer"
        assert (path, kind) == ("", "framebuffer")
        assert not report.lines()

    def test_clearing_the_framebuffer_is_recorded(self) -> None:
        # こちらは下の絵を消せない 記録が無いと、AviUtl2 では消える下の絵が残っていることに
        # 気付けない
        from sashimono.compat.aviutl.mapping import _content

        report = CompatibilityReport()
        _content(_entry("フレームバッファ", フレームバッファをクリア="1"), (), report)
        assert any("フレームバッファをクリア" in line for line in report.lines())

    def test_the_previous_object_becomes_the_copy_of_the_one_below(self) -> None:
        # 前は未対応の中身として何も置かず、下の絵の写しが丸ごと消えていた
        from sashimono.compat.aviutl.mapping import _content

        report = CompatibilityReport()
        source, path, kind = _content(_entry("直前オブジェクト"), (), report)
        assert source is not None and source.kind == PREVIOUS_OBJECT.kind
        assert (path, kind) == ("", PREVIOUS_OBJECT.kind)
        assert not report.lines()

    def test_unknown_items_of_the_previous_object_are_recorded(self) -> None:
        # 測った 直前オブジェクト は項目を 1 つも持たなかった 知らない項目を黙って捨てると、
        # 写せていない設定があることに互換性レポートで気付けない
        from sashimono.compat.aviutl.mapping import _content

        report = CompatibilityReport()
        _content(_entry("直前オブジェクト", 何か="1"), (), report)
        assert any("何か" in line for line in report.lines())


def _object_text(*entries: tuple[str, dict[str, str]]) -> str:
    """AviUtl2 の ``.object`` と同じ書き方の 1 オブジェクト（0 から 5 フレーム）"""
    lines = ["[Object]", "frame=0,5"]
    for index, (name, params) in enumerate(entries):
        lines += [f"[Object.{index}]", f"effect.name={name}"]
        lines += [f"{key}={value}" for key, value in params.items()]
    return "\n".join(lines) + "\n"


def _aviutl_draw(x: float) -> tuple[str, dict[str, str]]:
    return ("標準描画", {"X": f"{x:.2f}", "Y": "0.00", "拡大率": "100.000", "透明度": "0.00"})


def _aviutl_square(*filters: tuple[str, dict[str, str]]) -> tuple[tuple[str, dict[str, str]], ...]:
    figure = {"図形の種類": "四角形", "サイズ": "100", "色": "ffffff", "ライン幅": "4000"}
    return (("図形", figure), *filters)


class TestAviUtlPreviousObject:
    """``tools/aviutl_filter_probes.py`` の fp52・fp53 と同じ組み（#195）

    AviUtl2 は下のレイヤーの四角（X=-400）を、直前オブジェクトの位置（X=400）へ同じ大きさで
    写し、下の四角に掛けた単色化の赤も写した 外形は 460〜660 と 1260〜1460（1920 幅）で、
    明るさの平均は四角 1 つの 2 倍 ここでは 400x400 に収まるよう X を ±100、四角を 100 にする
    """

    def _render(self, gl: OffscreenGLContext, *texts: str) -> np.ndarray:
        from sashimono.compat.aviutl.exo import parse_exo
        from sashimono.compat.aviutl.mapping import map_object

        rate = FrameRate(30)
        report = CompatibilityReport()
        clips = []
        for text in texts:
            mapped = map_object(parse_exo(text).objects[0], rate, report=report)
            assert mapped is not None
            clips.append(mapped.clip)
        assert not report.lines()
        project = Project.create(ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=rate))
        tracks = tuple(Track(kind=TrackKind.VIDEO, clips=(clip,)) for clip in clips)
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=tracks)
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            return np.asarray(renderer.render(0))[..., :3]
        finally:
            renderer.close()

    def test_the_copy_is_placed_at_its_own_position(self, gl: OffscreenGLContext) -> None:
        # 下の四角の位置（X=-100）まで足すと写しが X=0 に来る 自分の位置だけで置くこと
        below = _object_text(*_aviutl_square(), _aviutl_draw(-100.0))
        above = _object_text(("直前オブジェクト", {}), _aviutl_draw(100.0))
        image = self._render(gl, below, above)
        half = WIDTH // 2
        assert _box(image[:, :half]) == (50, 150, 150, 250)
        right = _box(image[:, half:])
        assert right is not None
        assert (right[0] + half, right[1], right[2] + half, right[3]) == (250, 150, 350, 250)
        # 明るい所は四角 2 つぶん（AviUtl2 の明るさの平均が 2 倍だったのと同じ）
        # 写しを同じ大きさで置かないと数が合わない
        assert int((image.max(axis=2) >= 128).sum()) == 2 * 100 * 100

    def test_the_copy_keeps_the_filters_of_the_one_below(self, gl: OffscreenGLContext) -> None:
        # AviUtl2 は下の四角に掛けた単色化の赤まで写した 素の図形を写すと写しだけ白く出る
        paint = ("単色化", {"強さ": "100.0", "色": "ff0000", "輝度を保持する": "0"})
        below = _object_text(*_aviutl_square(paint), _aviutl_draw(-100.0))
        above = _object_text(("直前オブジェクト", {}), _aviutl_draw(100.0))
        image = self._render(gl, below, above)
        assert tuple(int(v) for v in image[200, 100]) == (255, 0, 0)
        assert tuple(int(v) for v in image[200, 300]) == (255, 0, 0)

    def test_a_chain_copies_the_copy(self, gl: OffscreenGLContext) -> None:
        # 直前オブジェクトを 2 つ重ねると、2 つ目は 1 つ目（下の四角の写し）を写す
        # 1 つ目を写す相手が無いものとして扱うと、2 つ目が何も出さない
        # 1 つ目の位置は写さないので、2 つ目は自分の位置（真ん中から上へ 120）にだけ出る
        below = _object_text(*_aviutl_square(), _aviutl_draw(-100.0))
        first = _object_text(("直前オブジェクト", {}), _aviutl_draw(100.0))
        second = _object_text(("直前オブジェクト", {}), ("標準描画", {"Y": "-120.00"}))
        image = self._render(gl, below, first, second)
        assert int(image[80, 200].max()) == 255
        assert int((image.max(axis=2) >= 128).sum()) == 3 * 100 * 100

    def test_nothing_below_draws_nothing(self, gl: OffscreenGLContext) -> None:
        # 写す相手の無い直前オブジェクトは何も描かない 画面の写し取りのように黒で覆うと、
        # 置いただけで画面が黒くなる
        above = _object_text(("直前オブジェクト", {}), _aviutl_draw(100.0))
        image = self._render(gl, above)
        assert int(image.max()) == 0


def _render_clips(gl: OffscreenGLContext, *clips: Clip, frame: int = 0) -> np.ndarray:
    """1 本ずつ別のトラックに置いたクリップを下から重ねて描く（400x400・30fps）"""
    project = Project.create(ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30)))
    tracks = tuple(Track(kind=TrackKind.VIDEO, clips=(clip,)) for clip in clips)
    project = project.with_timeline(project.timeline.__class__(rate=project.rate, tracks=tracks))
    renderer = FrameRenderer(project, context=gl)
    try:
        return np.asarray(renderer.render(frame))[..., :3]
    finally:
        renderer.close()


def _placed(x: float = 0.0, y: float = 0.0) -> Effect:
    """クリップが最初から持つ配置の欄（印付き）"""
    return replace(registry.require("transform").create(pos_x=x, pos_y=y), fixed=True)


class TestPreviousObjectOfSpecialClips:
    """下のクリップが別の道で描かれるときの直前オブジェクト（#195 PR #219 の指摘）"""

    def test_the_trail_of_the_one_below_is_copied(self, gl: OffscreenGLContext) -> None:
        # 残像は本体とは別に前のフレームを描く 写すときにそこを通らないと、写しにだけ
        # 残像が付かない 下の四角は横へ動き、配置の欄で上へ 120 に置いてある
        moving = registry.require("transform").create(
            pos_x=AnimatedValue(
                keyframes=(Keyframe(frame=0, value=-100.0), Keyframe(frame=10, value=100.0))
            )
        )
        trail = registry.require("after_image").create(strength=90.0, samples=10)
        below = Clip(
            timeline_start=0,
            duration=30,
            source=_square(40),
            effects=(trail, moving, _placed(y=120.0)),
        )
        copy = Clip(timeline_start=0, duration=30, source=PREVIOUS_OBJECT.create())
        image = _render_clips(gl, below, copy, frame=10)
        # 下の本体は X=100（列 280〜320）で、2 フレーム前の残像が列 250 に残る
        assert int(image[80, 250].max()) > 100
        # 写しは下の配置を外して真ん中の行に出る 残像も同じ所に出る
        assert int(image[200, 300].max()) == 255
        assert int(image[200, 250].max()) > 100

    def test_a_framebuffer_below_is_copied_from_the_screen(self, gl: OffscreenGLContext) -> None:
        # フレームバッファを写すときに、下の絵を溜めた合成先ではなく空の写し先を読むと、
        # 真っ黒な画面で全体を覆って下の四角まで消える
        square = Clip(
            timeline_start=0, duration=30, source=_square(100), effects=(_placed(-100.0),)
        )
        grab = Clip(timeline_start=0, duration=30, source=GeneratedSource(kind="framebuffer"))
        copy = Clip(
            timeline_start=0,
            duration=30,
            source=PREVIOUS_OBJECT.create(),
            effects=(_placed(100.0),),
        )
        image = _render_clips(gl, square, grab, copy)
        # 画面の写しを右へ 100 ずらして置くので、四角（列 50〜150）が列 150〜250 に出る
        assert int(image[200, 200].max()) == 255
        # 写しに覆われない左の 100 列には元の四角が残る
        assert int(image[200, 60].max()) == 255
