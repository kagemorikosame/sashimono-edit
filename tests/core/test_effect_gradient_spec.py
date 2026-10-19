"""エフェクトの色の止まりの並び（#364）の値の決まり

並びが空の間は前からの色の項目で塗り、画面のバーは前からの項目を並べて見せる
"""

from __future__ import annotations

from sashimono.ai.operations import _describe_spec
from sashimono.compat.aviutl.text_fill import text_gradient_fill
from sashimono.core.model import AnimatedValue, Keyframe
from sashimono.effects import GradientSpec, SelectSpec, registry
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.gradient import colour_at

RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def _spec(kind: str) -> GradientSpec:
    spec = registry.require(kind).spec("gradient")
    assert isinstance(spec, GradientSpec)
    return spec


class TestEmptyMeansTheOldFields:
    def test_a_new_effect_starts_empty(self) -> None:
        # 既定を色の入った並びにすると、作るときに既定で埋める道（読み込み・〔＋ エフェクト〕）が
        # 前からの色の項目を無視して白から黒で塗る
        for kind in ("gradient", "brush_fill"):
            assert registry.require(kind).create().params["gradient"] == ()

    def test_a_broken_list_falls_back_to_the_old_fields(self) -> None:
        # 壊れた並びを既定の白から黒にすると、手で直したファイルの誤りで色が全部変わる
        spec = _spec("gradient")
        assert spec.coerce((0.0, 1.0, 0.0)) == ()
        assert spec.coerce(None) == ()

    def test_the_bar_shows_the_start_and_end_colours(self) -> None:
        # バーに開始色と終了色が出ないと、前からの作品の設定パネルに既定の白から黒が並び、
        # 触っただけで色が置き換わる
        params = registry.require("gradient").create(start_color=GREEN, end_color=RED).params
        assert _spec("gradient").shown(params) == (0.0, *GREEN, 1.0, *RED)

    def test_the_bar_shows_the_numbered_fields_in_order(self) -> None:
        # 配布物の並びの順のまま見せると、つまみの順と色の順が食い違う
        params = (
            registry.require("brush_fill")
            .create(stops=3, color0=RED, offset0=0.8, color1=GREEN, offset1=0.1, color2=BLUE)
            .params
        )
        shown = GradientSpec.stops(_spec("brush_fill").shown(params))
        assert [position for position, _ in shown] == [0.1, 0.8, 1.0]
        assert [colour for _, colour in shown] == [GREEN, RED, BLUE]

    def test_a_single_colour_becomes_two_stops_of_it(self) -> None:
        # バーは 2 つより少ない止まりを持てない 1 つのまま渡すと既定の白から黒が出る
        params = registry.require("brush_fill").create(stops=1, color0=RED).params
        assert _spec("brush_fill").shown(params) == (0.0, *RED, 1.0, *RED)

    def test_the_bar_wins_once_it_has_stops(self) -> None:
        # 止まりを並べた後も前からの 2 色をバーに出すと、バーで足した色が画面から消え、直せなくなる
        stops = (0.0, *GREEN, 1.0, *BLUE)
        params = registry.require("gradient").create(gradient=stops).params
        assert _spec("gradient").shown(params) == stops


class TestTheSCurve:
    def test_only_the_gradient_effect_mixes_with_an_s_curve(self) -> None:
        # 模様で塗るは YMM4 と同じく直線で混ぜる 見本を S 字で描くと絵と食い違う
        assert not _spec("brush_fill").smooth_in({})
        spec = _spec("gradient")
        # 項目の無い前の版の作品と AviUtl2 の読み込みは S 字のまま 直線にすると前の作品の
        # グラデーションの帯の端の濃さが変わる
        assert spec.smooth_in({})
        assert spec.smooth_in(registry.require("gradient").create().params)
        assert not spec.smooth_in(registry.require("gradient").create(mixing="linear").params)

    def test_the_mixing_is_a_choice_not_a_keyframed_value(self) -> None:
        # 動かせる欄にすると、混ぜ方の切り替わりの途中のコマで止まりの色がどちらでもなくなる
        spec = registry.require("gradient").spec("mixing")
        assert isinstance(spec, SelectSpec)
        assert spec.default == "smooth"
        assert [value for value, _ in spec.choices] == ["smooth", "linear"]

    def test_the_smoothed_stops_follow_the_s_curve(self) -> None:
        # 見本を直線で刻むと、設定パネルのバーの色の移り方が書き出した絵の S 字の混ざり方と食い違う
        stops = [(0.0, RED), (1.0, BLUE)]
        smoothed = GradientSpec.smoothed(stops)
        assert smoothed[0] == stops[0] and smoothed[-1] == stops[-1]
        quarter = next(colour for position, colour in smoothed if position == 0.25)
        # S 字の 1/4 は 0.15625 直線なら 0.25
        assert abs(quarter[2] - 0.15625) < 1e-9

    def test_a_new_stop_takes_the_colour_already_shown_there(self) -> None:
        # 足した止まりの色が見本の色と違うと、足しただけで色が跳ねる
        stops = [(0.0, RED), (1.0, BLUE)]
        assert abs(colour_at(stops, 0.25, smooth=True)[2] - 0.15625) < 1e-9
        assert abs(colour_at(stops, 0.25)[2] - 0.25) < 1e-9


class TestWhoReadsTheOldColours:
    def test_the_aviutl_text_fill_leaves_a_bar_gradient_as_an_effect(self) -> None:
        # バーで止まりを並べたグラデーションを開始色と終了色の 2 色として字の塗りへ写すと、
        # 並べた色が消える
        source = TEXT.create(text="字")
        plain = registry.require("gradient").create(angle=0)
        assert text_gradient_fill(plain, source) is not None
        barred = registry.require("gradient").create(
            angle=0, gradient=(0.0, *RED, 0.5, *GREEN, 1.0, *BLUE)
        )
        assert text_gradient_fill(barred, source) is None
        # 直線で混ぜる物を S 字で刻んで写すと、帯の中ほどの色が変わる
        straight = registry.require("gradient").create(angle=0, mixing="linear")
        assert text_gradient_fill(straight, source) is None

    def test_the_ai_is_told_what_an_empty_list_means(self) -> None:
        # 空の意味を伝えないと、AI が空を「色が無い」と読んで止まりを足し直す
        described = _describe_spec(_spec("brush_fill"))
        assert "空の並び" in described["format"]
        text_spec = TEXT.spec("fill_gradient")
        assert "空の並び" not in _describe_spec(text_spec)["format"]


def test_a_moving_offset_is_read_at_its_resting_value() -> None:
    # 動く位置はバーに出さない（:func:`sashimono.effects.paint._paint_unused`） 読むときに
    # 落ちないことだけを見る 落ちると設定パネルを開いただけで止まる
    moving = AnimatedValue(0.3, (Keyframe(0, 0.2), Keyframe(30, 0.8)))
    params = registry.require("brush_fill").create(offset0=moving).params
    assert GradientSpec.stops(_spec("brush_fill").shown(params))[0][0] == 0.3
