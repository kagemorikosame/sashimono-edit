"""YMM4 のテキストの頭の「前景を塗りつぶし」を、テキストの塗りのグラデーションへ写す（#312）

写すのは塗りのエフェクトと同じ絵になると言える物だけ（:func:`~sashimono.compat.ymm4.brushes.text_fill`）
実物（あおもや式のポップおこ・ポップグラデ・確定演出的な虹色字幕 19 個）の形をここで作って見る
YMM4 本体との差は ``tools/ymm4_compare.py`` で測り、写す前と同じだった（docs/development.md）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.ymm4.brushes import fill_foreground, text_fill
from sashimono.compat.ymm4.decorations import map_video_effects
from sashimono.compat.ymm4.template import map_template
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import AnimatedValue, Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from tests.compat.test_ymm4 import still, text_item

WHITE = (1.0, 1.0, 1.0, 1.0)


def fill(*stops: tuple[float, str], **fields: Any) -> dict[str, Any]:
    """実物（ポップグラデ）と同じ形の前景の塗り 線形・割合・端の色・角度 90"""
    parameter: dict[str, Any] = {
        "$type": "YukkuriMovieMaker.Brush.LinearGradientBrushParameter, YukkuriMovieMaker",
        "CoordinateMode": "Relative",
        "Size": still(80.0),
        "Offset": still(8.0),
        "Angle": still(90.0),
        "ExtendMode": "Clamp",
        "Stops": [{"Offset": offset, "Color": colour} for offset, colour in stops],
    }
    entry: dict[str, Any] = {
        "$type": "YukkuriMovieMaker.Plugin.Effects.FillForegroundEffect, YukkuriMovieMaker",
        "IsEnabled": True,
        "BlendMode": "Normal",
        "Opacity": still(100.0),
        "IsBrushOnly": False,
        "Brush": {
            "Type": "YukkuriMovieMaker.Plugin.Brush.LinearGradientBrushPlugin, YukkuriMovieMaker",
            "Parameter": parameter,
        },
    }
    for key, value in fields.items():
        (parameter if key in parameter else entry)[key] = value
    return entry


def outline(size: float, colour: str) -> dict[str, Any]:
    return {
        "$type": "YukkuriMovieMaker.Plugin.Effects.OutlineEffect, YukkuriMovieMaker",
        "IsEnabled": True,
        "StrokeThickness": still(size),
        "StrokeColor": colour,
        "Blur": still(0.0),
        "IsOutlineOnly": False,
        "IsAngular": False,
    }


#: ポップグラデ濃緑の塗り 色が同じで不透明度だけが 0x50 から 0xFF へ
GREEN_FADE = ((0.0, "#5000A95F"), (1.0, "#FF00A95F"))


class TestWhatIsFolded:
    def test_the_stops_are_laid_over_the_text_colour(self) -> None:
        # 止まりの色をそのまま塗ると、字の白が透けるはずの上の方が濃い緑になる
        params = text_fill(fill(*GREEN_FADE), WHITE, CompatibilityReport())
        assert params is not None
        assert params["fill_type"] == "linear"
        assert params["fill_span"] == "width"
        top = 0x50 / 255
        gradient = params["fill_gradient"]
        assert isinstance(gradient, tuple)
        expected_top = (1.0 - top, 1.0 * (1 - top) + 0xA9 / 255 * top, 1 - top + 0x5F / 255 * top)
        assert gradient[:5] == pytest.approx((0.0, *expected_top, 1.0))
        assert gradient[5:] == pytest.approx((1.0, 0.0, 0xA9 / 255, 0x5F / 255, 1.0))

    def test_the_angle_length_and_shift(self) -> None:
        # ブラシの角度 90（上から下）を塗りの 0 へ直さないと、縦のグラデーションが横に倒れる
        params = text_fill(fill(*GREEN_FADE), WHITE, CompatibilityReport())
        assert params is not None
        assert params["fill_angle"] == AnimatedValue(0.0)
        assert params["fill_length"] == AnimatedValue(80.0)
        assert params["fill_shift"] == AnimatedValue(8.0)

    def test_a_one_step_colour_difference_still_folds(self) -> None:
        # 実物（ポップおこ赤 #50E95532 → #FFEA5532）は色が 1 段だけ違う 断ると赤だけ写らない
        params = text_fill(
            fill((0.0, "#50E95532"), (1.0, "#FFEA5532")), WHITE, CompatibilityReport()
        )
        assert params is not None

    @pytest.mark.parametrize(
        "changed",
        [
            # 乗算・オーバーレイは字の色との混ぜ方が違う（ポップおこの 2 つ目の塗り）
            {"BlendMode": "Multiply"},
            # 薄くした塗りは止まりの色に混ぜ込めるが、実物に無く確かめていない
            {"Opacity": still(50.0)},
            # 画素で長さを決める物は字の大きさで割合が変わる
            {"CoordinateMode": "Absolute"},
            # 繰り返しは塗りに無い
            {"ExtendMode": "Wrap"},
            # 動く値は塗りの値で表せる形か確かめていない
            {"Angle": {"Values": [{"Value": 0.0}, {"Value": 90.0}], "AnimationType": "直線移動"}},
        ],
    )
    def test_what_cannot_be_the_same_picture_stays_an_effect(self, changed: dict[str, Any]) -> None:
        # 違う絵になる物まで塗りにすると、読み込んだ字幕の色が YMM4 と変わる
        assert text_fill(fill(*GREEN_FADE, **changed), WHITE, CompatibilityReport()) is None

    def test_colour_and_opacity_changing_together_stays_an_effect(self) -> None:
        # 色と不透明度が両方変わると、字の色に重ねた色は直線で並ばず、止まりだけでは表せない
        bending = fill((0.0, "#00FF0000"), (1.0, "#FF0000FF"))
        assert text_fill(bending, WHITE, CompatibilityReport()) is None

    def test_brush_only_keeps_the_colours(self) -> None:
        # ブラシだけの塗りは字の色を捨てる 重ねると、字の色の分だけ明るい字になる
        params = text_fill(
            fill((0.0, "#FFFF0000"), (1.0, "#800000FF"), IsBrushOnly=True),
            (1.0, 1.0, 1.0, 0.5),
            CompatibilityReport(),
        )
        assert params is not None
        gradient = params["fill_gradient"]
        assert isinstance(gradient, tuple)
        assert gradient == pytest.approx((0.0, 1, 0, 0, 0.5, 1.0, 0, 0, 1, 0x80 / 255 * 0.5))


class TestTheChain:
    def test_only_the_head_fill_folds_and_the_outlines_keep_their_order(self) -> None:
        # 縁取りを層にすると、塗りを写した後も YMM4 と縁の付き方が変わる（ポップグラデの差 1.02 が
        # 層にした形では 2.46） 並びの位置に残す
        chain = [fill(*GREEN_FADE), outline(6, "#FFFFFFFF"), outline(9, "#FF000000")]
        result = map_video_effects(chain, CompatibilityReport(), text=True, fill_base=WHITE)
        assert result.params["fill_type"] == "linear"
        assert [effect.kind for effect in result.effects] == ["border", "border"]
        assert result.strokes == []

    def test_a_fill_after_an_outline_stays_an_effect(self) -> None:
        # 縁取りの後ろの塗りは縁まで塗る 字の塗りへ移すと、縁が単色に戻る
        chain = [outline(6, "#FFFFFFFF"), outline(9, "#FF000000"), fill(*GREEN_FADE)]
        result = map_video_effects(chain, CompatibilityReport(), text=True, fill_base=WHITE)
        assert "fill_type" not in result.params
        assert result.effects[-1].kind == "brush_fill"

    def test_the_second_fill_stays_an_effect(self) -> None:
        # ポップおこは頭の普通の塗りの後ろに乗算の塗りを重ねる 2 つ目まで字の塗りにはできない
        chain = [fill(*GREEN_FADE), fill(*GREEN_FADE, BlendMode="Multiply")]
        result = map_video_effects(chain, CompatibilityReport(), text=True, fill_base=WHITE)
        assert result.params["fill_type"] == "linear"
        assert [effect.kind for effect in result.effects] == ["brush_fill"]

    def test_a_styled_text_is_not_folded(self) -> None:
        # 文字装飾（Style）の縁と影まで YMM4 は塗る 字の塗りへ移すと縁が塗られなくなる
        styled = text_item(Style="Border", VideoEffects=[fill(*GREEN_FADE)])
        (mapped,) = map_template([styled])
        assert mapped.clip.source is not None
        assert mapped.clip.source.params.get("fill_type") != "linear"
        assert any(effect.kind == "brush_fill" for effect in mapped.clip.effects)

    def test_a_plain_text_item_gets_the_fill(self) -> None:
        # 読み込みで写らないと、YMM4 の字幕のグラデーションを設定パネルのバーで直せない
        # 塗りのエフェクトも残ると、同じ塗りが 2 度掛かって色が濃くなる
        plain = text_item(VideoEffects=[fill(*GREEN_FADE)])
        (mapped,) = map_template([plain])
        assert mapped.clip.source is not None
        assert mapped.clip.source.params["fill_type"] == "linear"
        assert not any(effect.kind == "brush_fill" for effect in mapped.clip.effects)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _render(clip: Clip, context: OffscreenGLContext) -> np.ndarray:
    base = Project.create(ProjectSettings(width=640, height=360, frame_rate=FrameRate(30)))
    track = Track(TrackKind.VIDEO, "V1", (with_fixed_items(clip, picture=True),))
    project = base.with_timeline(replace(base.timeline, tracks=(track,)))
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(0).astype(np.int16)
    finally:
        renderer.close()


@pytest.mark.parametrize("stops", [GREEN_FADE, ((0.0, "#FFFDAE47"), (1.0, "#FFF6FFD1"))])
def test_the_fold_draws_the_same_picture(
    gl_context: OffscreenGLContext, stops: tuple[tuple[float, str], ...]
) -> None:
    # 写した塗りとエフェクトの塗りで絵が違うと、読み込んだ字幕の色が前の版と変わる
    entry = fill(*stops)
    text = TEXT.create(text="■■■", size=120, color=WHITE)
    effect = fill_foreground(entry, CompatibilityReport())
    assert effect is not None
    params = text_fill(entry, WHITE, CompatibilityReport())
    assert params is not None
    as_effect = _render(
        Clip(timeline_start=0, duration=30, source=text, effects=(effect,)), gl_context
    )
    as_fill = _render(
        Clip(
            timeline_start=0,
            duration=30,
            source=replace(text, params={**text.params, **params}),
        ),
        gl_context,
    )
    difference = np.abs(as_effect - as_fill)[..., :3]
    assert difference.mean() < 0.2, difference.mean()
    assert np.percentile(difference, 99.9) <= 6
