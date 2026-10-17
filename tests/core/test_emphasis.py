"""強調の動き（#331）の線を組む計算

AI の道具（tests/ai/test_emphasis_tools.py）が通す線の形を、Qt もエフェクトの定義も
無しで確かめる 範囲の外の動きを変えないこと、寄る先の点が動かないこと、人の点を
潰さないことを見る
"""

from __future__ import annotations

import pytest

from sashimono.core.commands.effects import AddEffect, SetParam
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, fixed_effect
from sashimono.core.emphasis import (
    EMPHASIS_EASING_NAMES,
    EMPHASIS_EASINGS,
    EmphasisError,
    envelope,
    pan_curves,
    ramp_frames,
    scaled_curve,
    shake_curve,
    transform_commands,
    zoom_curves,
)
from sashimono.core.model import AnimatedValue, Clip, Interpolation, Keyframe
from sashimono.core.timebase import FrameRate


def _still(value: float) -> AnimatedValue:
    return AnimatedValue(static=value)


class TestEnvelope:
    def test_zoom_rises_holds_and_returns(self) -> None:
        # 壊れると、寄ったまま戻らないか、寄り切る前に戻り始めて要所で寄っていない
        shape = envelope("zoom", 10, 40, 5, "smooth")
        assert [(frame, amount) for frame, amount, *_ in shape.points] == [
            (10, 0.0),
            (15, 1.0),
            (35, 1.0),
            (40, 0.0),
        ]

    def test_a_short_range_peaks_in_the_middle(self) -> None:
        # 壊れると、動き切る長さより短い範囲で点の順が入れ替わり、線を作れずに断られる
        shape = envelope("zoom", 10, 14, 5, "smooth")
        assert [(frame, amount) for frame, amount, *_ in shape.points] == [
            (10, 0.0),
            (12, 1.0),
            (14, 0.0),
        ]

    def test_pop_does_not_hold(self) -> None:
        # 壊れると、一瞬ふくらむはずのポップが範囲の間ずっとふくらんだまま止まる
        shape = envelope("pop", 0, 12, 4, "back")
        assert [(frame, amount) for frame, amount, *_ in shape.points] == [
            (0, 0.0),
            (4, 1.0),
            (12, 0.0),
        ]

    def test_overshooting_easings_land_with_a_bounce(self) -> None:
        # 壊れると、back を選んでも動き出しで逆へ振れ、着いた所で跳ねない
        start, *_ = envelope("zoom", 0, 30, 5, "back").points
        assert start[2:] == (Interpolation.EASE_OUT, "back")
        smooth, *_ = envelope("zoom", 0, 30, 5, "smooth").points
        assert smooth[2:] == (Interpolation.EASE_IN_OUT, "")

    def test_keep_only_for_zoom_and_pan(self) -> None:
        # 壊れると、揺れやポップがクリップの終わりまで残り、残りの全部が揺れ続ける
        assert envelope("zoom", 0, 0, 5, "smooth", keep=True).keep
        with pytest.raises(EmphasisError):
            envelope("shake", 0, 0, 5, "smooth", keep=True)

    def test_unknown_names_are_refused(self) -> None:
        # 壊れると、綴りを間違えた動き方が黙って直線になり、頼んだ動きにならない
        with pytest.raises(EmphasisError):
            envelope("spin", 0, 30, 5, "smooth")
        with pytest.raises(EmphasisError):
            envelope("zoom", 0, 30, 5, "wobble")

    def test_ramp_is_at_least_one_frame(self) -> None:
        # 壊れると、0 ms のときに動く前と動き切った点が同じフレームに重なり、線を作れない
        assert ramp_frames(0, FrameRate(30)) == 1
        assert ramp_frames(250, FrameRate(30)) == 8

    def test_every_easing_has_a_name_for_the_settings(self) -> None:
        # 壊れると、設定画面の動き方の一覧に名前の無い項目が出て、選べないか落ちる
        assert set(EMPHASIS_EASING_NAMES) == set(EMPHASIS_EASINGS)
        for easing in EMPHASIS_EASINGS:
            envelope("zoom", 0, 30, 5, easing)


class TestCurves:
    def test_outside_the_range_nothing_changes(self) -> None:
        # 壊れると、要所の前後まで拡大率が動き、強調したい所以外でも絵が揺れる
        base = AnimatedValue(static=100.0, keyframes=(Keyframe(0, 100.0), Keyframe(100, 200.0)))
        curve = scaled_curve(base, envelope("zoom", 40, 60, 5, "smooth"), 1.2)
        for frame in [*range(0, 41), *range(60, 101)]:
            assert curve.at(frame) == pytest.approx(base.at(frame)), frame
        assert curve.at(50) == pytest.approx(base.at(50) * 1.2)

    def test_a_still_value_rises_to_the_ratio(self) -> None:
        # 壊れると、動いていない拡大率に寄る動きが付かないか、寄る量が頼んだ値と違う
        curve = scaled_curve(_still(100.0), envelope("zoom", 10, 40, 5, "smooth"), 1.25)
        assert curve.at(0) == 100.0
        assert curve.at(20) == pytest.approx(125.0)
        assert curve.at(50) == 100.0

    def test_a_keyframe_inside_the_range_is_refused(self) -> None:
        # 壊れると、人が範囲の中に打った点が消えるか、強調の形と混ざってどちらでもなくなる
        base = AnimatedValue(static=100.0, keyframes=(Keyframe(0, 100.0), Keyframe(20, 150.0)))
        with pytest.raises(EmphasisError):
            scaled_curve(base, envelope("zoom", 10, 40, 5, "smooth"), 1.2)
        with pytest.raises(EmphasisError):
            scaled_curve(base, envelope("zoom", 5, 0, 5, "smooth", keep=True), 1.2)

    def test_keep_stays_until_the_end(self) -> None:
        # 壊れると、残すと頼んだ寄りが範囲の後ろで元の大きさへ戻る
        curve = scaled_curve(_still(100.0), envelope("zoom", 10, 0, 5, "smooth", keep=True), 1.2)
        assert curve.at(5) == 100.0
        assert curve.at(15) == pytest.approx(120.0)
        assert curve.at(10_000) == pytest.approx(120.0)

    def test_the_focus_point_stays_put(self) -> None:
        # 壊れると、話者の顔へ寄ったつもりが画面の中央へ寄り、顔が画面の外へずれていく
        transform = fixed_effect(TRANSFORM_EFFECT_KIND).with_param("pos_x", _still(30.0))
        focus = (200.0, 120.0)
        curves = zoom_curves(transform, envelope("zoom", 0, 30, 6, "back"), 1.5, focus)
        # 支点から Q の絵の点は pos + 拡大率 x Q に描かれる 寄る前に focus にあった点を辿る
        q = ((focus[0] - 30.0) / 1.0, (focus[1] - 0.0) / 1.0)
        for frame in range(31):
            k = curves["scale"].at(frame) / 100.0
            x = curves["pos_x"].at(frame) + k * q[0]
            y = curves["pos_y"].at(frame) + k * q[1]
            assert (x, y) == pytest.approx(focus), frame
        assert curves["scale"].at(15) == pytest.approx(150.0)

    def test_without_focus_the_position_is_untouched(self) -> None:
        # 壊れると、絵の中心へ寄るだけの頼みでも位置の欄に点が並び、人が直すときに迷う
        curves = zoom_curves(
            fixed_effect(TRANSFORM_EFFECT_KIND), envelope("pop", 0, 9, 3, "back"), 1.1, None
        )
        assert set(curves) == {"scale"}

    def test_pan_up_raises_y(self) -> None:
        # 壊れると、「上へ」と頼んだパンで絵が下へ動く（Y は上が正）
        curves = pan_curves(
            fixed_effect(TRANSFORM_EFFECT_KIND), envelope("pan", 0, 30, 5, "smooth"), (0.0, 80.0)
        )
        assert set(curves) == {"pos_y"}
        assert curves["pos_y"].at(15) == pytest.approx(80.0)
        assert curves["pos_y"].at(30) == pytest.approx(0.0)

    def test_shake_is_silent_outside_the_range(self) -> None:
        # 壊れると、揺れを付けた所の前後までクリップ全体が揺れる
        curve = shake_curve(envelope("shake", 20, 50, 3, "linear"), 12.0)
        assert curve.at(0) == 0.0
        assert curve.at(35) == pytest.approx(12.0)
        assert curve.at(80) == 0.0


class TestCommands:
    def test_an_old_clip_gets_the_transform_in_one_step(self) -> None:
        # 壊れると、配置を持たない古いクリップで取り消しが 2 段になり、1 回戻すと動かない配置が残る
        clip = Clip(timeline_start=0, duration=30)
        curve = scaled_curve(_still(100.0), envelope("zoom", 0, 30, 5, "smooth"), 1.2)
        (command,) = transform_commands(clip, {"scale": curve})
        assert isinstance(command, AddEffect)
        assert command.effect.fixed and command.effect.params["scale"] == curve

    def test_a_clip_with_a_transform_gets_set_param(self) -> None:
        # 壊れると、配置を持つクリップに 2 つ目の配置が積まれ、どちらがパネルの欄か分からなくなる
        transform = fixed_effect(TRANSFORM_EFFECT_KIND)
        clip = Clip(timeline_start=0, duration=30, effects=(transform,))
        curve = scaled_curve(_still(100.0), envelope("zoom", 0, 30, 5, "smooth"), 1.2)
        (command,) = transform_commands(clip, {"scale": curve})
        assert isinstance(command, SetParam)
        assert command.path.effect_id == transform.id
