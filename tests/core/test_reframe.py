"""縦動画への再構図（#334）の計算

枠の大きさ・配置の式・場面ごとの枠の決め方を、手で組んだ値で確かめる 素材を読む所と
命令にする所は tests/engine/test_reframe.py が見る
"""

from __future__ import annotations

import pytest

from sashimono.core.model import AnimatedValue, Interpolation, Keyframe
from sashimono.core.reframe import (
    AUTO,
    CENTER,
    FOLLOW,
    STEADY,
    ReframeError,
    ReframeStyle,
    Sample,
    Shot,
    ShotPlan,
    affine,
    background_changes,
    clamp_span,
    crop_window,
    focus_curves,
    overlay_changes,
    parse_size,
    plan_shot,
    window_of,
)

OLD, NEW = (1920, 1080), (1080, 1920)
WINDOW = crop_window(OLD, NEW)
#: 枠の中心を置ける範囲（1920 の幅に 607.5 の枠）
BOUNDS = (-656.25, 656.25, 0.0, 0.0)


def _shot(points: list[tuple[int, float, float]], *, start: int = 0, end: int = 120) -> Shot:
    return Shot(
        start=start,
        end=end,
        samples=tuple(Sample(frame, x, 0.0, weight) for frame, x, weight in points),
        bounds=BOUNDS,
        home=(0.0, 0.0),
    )


def _drawn(pos: float, scale: float, pivot: float, point: float) -> float:
    """配置の式（シェーダの _TRANSFORM と同じ） 支点 + 位置 + 拡大 x (絵の点 - 支点)"""
    return pivot + pos + scale * (point - pivot)


class TestTheWindow:
    def test_the_window_is_the_tallest_that_fits(self) -> None:
        # 枠が小さいと拡げる倍率が上がり、縦の作品の絵が要らないほど粗くなる
        assert WINDOW.scale == pytest.approx(1920 / 1080)
        assert WINDOW.size == pytest.approx((607.5, 1080.0))

    def test_a_square_from_a_wide_picture_uses_the_full_height(self) -> None:
        # 縦横どちらを基準にするかを取り違えると、正方形の枠が画面の外まで伸びて黒い帯が出る
        square = crop_window(OLD, (1080, 1080))
        assert square.size == pytest.approx((1080.0, 1080.0))
        assert square.scale == pytest.approx(1.0)

    def test_the_same_size_is_refused(self) -> None:
        # 断らないと、何も変わらない段が履歴に積まれ、作り替えたつもりで横のまま書き出す
        with pytest.raises(ReframeError, match="もう"):
            crop_window(OLD, OLD)

    def test_the_size_is_read_as_width_by_height(self) -> None:
        # 横と縦を逆に読むと、縦の作品のつもりが横長の作品になる
        assert parse_size("1080x1920") == (1080, 1920)
        assert ReframeStyle().resolution == (1080, 1920)
        with pytest.raises(ReframeError):
            parse_size("縦長")


class TestTheBackground:
    def test_the_focus_point_lands_in_the_middle(self) -> None:
        # 壊れると、被写体に合わせたはずの枠の中身が縦の画面の端へずれて、人が半分切れる
        changes = background_changes(
            {}, (AnimatedValue(400.0), AnimatedValue(0.0)), WINDOW, 1.0, (0.0, 0.0)
        )
        pos = changes["pos_x"].static
        scale = changes["scale"].static / 100.0
        assert _drawn(pos, scale, 0.0, 400.0) == pytest.approx(0.0)
        # 枠の右端（400 + 303.75）は縦の画面の右端（540）へ行く
        assert _drawn(pos, scale, 0.0, 400.0 + 303.75) == pytest.approx(540.0)

    def test_a_shifted_and_scaled_clip_keeps_its_place_in_the_window(self) -> None:
        # 前からの位置と拡大率を数えないと、少し寄せて置いた動画で枠と中身がずれる
        params = {"pos_x": AnimatedValue(100.0), "scale": AnimatedValue(120.0)}
        changes = background_changes(
            params, (AnimatedValue(250.0), AnimatedValue(0.0)), WINDOW, 1.0, (0.0, 0.0)
        )
        before = _drawn(100.0, 1.2, 0.0, 50.0)  # 前の画面で絵の点 50 が描かれた所
        after = _drawn(changes["pos_x"].static, changes["scale"].static / 100.0, 0.0, 50.0)
        assert after == pytest.approx(WINDOW.scale * (before - 250.0))

    def test_a_pivot_on_the_picture_edge_keeps_the_same_mapping(self) -> None:
        # 支点を絵の端にした（YMM4 の中心点）動画で支点のずれを数えないと、枠が端の分だけずれる
        placement = 0.5625  # 画面に収めて置く動画は縦の画面で小さく収まる
        pivot = -960.0
        params = {"pos_x": AnimatedValue(30.0), "scale": AnimatedValue(110.0)}
        changes = background_changes(
            params, (AnimatedValue(-200.0), AnimatedValue(0.0)), WINDOW, placement, (pivot, 0.0)
        )
        point = 300.0  # 前の画面での絵の点（支点の基準の置き方で）
        before = _drawn(30.0, 1.1, pivot, point)
        after = _drawn(
            changes["pos_x"].static,
            changes["scale"].static / 100.0,
            pivot * placement,
            point * placement,
        )
        assert after == pytest.approx(WINDOW.scale * (before + 200.0))

    def test_a_moving_window_becomes_position_keyframes(self) -> None:
        # 枠の動きが位置の線に移らないと、追うはずの枠が頭の位置で止まる
        focus = AnimatedValue(
            0.0,
            (
                Keyframe(0, -300.0, Interpolation.EASE_IN_OUT),
                Keyframe(60, 300.0, Interpolation.HOLD),
            ),
        )
        changes = background_changes({}, (focus, AnimatedValue(0.0)), WINDOW, 1.0, (0.0, 0.0))
        moved = changes["pos_x"]
        for frame in (0, 17, 30, 60):
            assert moved.at(frame) == pytest.approx(-WINDOW.scale * focus.at(frame))

    def test_both_moving_is_refused(self) -> None:
        # 黙って片方を捨てると、人が付けた寄りかパンが消える
        moving = AnimatedValue(0.0, (Keyframe(0, 0.0), Keyframe(10, 50.0)))
        with pytest.raises(ReframeError):
            background_changes(
                {"pos_x": moving}, (moving, AnimatedValue(0.0)), WINDOW, 1.0, (0.0, 0.0)
            )

    def test_a_scale_beyond_the_field_is_refused(self) -> None:
        # 欄の上限を越えた拡大率を入れると、人が設定パネルで直せず、触ると上限へ縮む
        with pytest.raises(ReframeError, match="拡大率"):
            background_changes(
                {"scale": AnimatedValue(500.0)},
                (AnimatedValue(0.0), AnimatedValue(0.0)),
                WINDOW,
                1.0,
                (0.0, 0.0),
            )


class TestTheCurves:
    def test_affine_keeps_the_easing_shape(self) -> None:
        # 曲がり方まで書き換えると、人が付けたベジェの寄りが作り替えで別の動きになる
        curve = AnimatedValue(
            0.0,
            (
                Keyframe(0, 10.0, Interpolation.BEZIER, control_points=(0.2, 0.9, 0.3, 1.0)),
                Keyframe(40, 90.0),
            ),
        )
        moved = affine(curve, -2.0, 7.0, 0.0)
        for frame in (0, 5, 13, 29, 40):
            assert moved.at(frame) == pytest.approx(-2.0 * curve.at(frame) + 7.0)

    def test_a_window_cut_keeps_the_values_inside(self) -> None:
        # 切り出しでクリップの頭からの数え方を間違えると、後ろのクリップで枠が別の場面へ動く
        curve = AnimatedValue(
            0.0,
            (Keyframe(0, 0.0, Interpolation.EASE_IN_OUT), Keyframe(100, 100.0)),
        )
        local = window_of(curve, 30, 80)
        for frame in range(0, 50, 7):
            assert local.at(frame) == pytest.approx(curve.at(30 + frame))

    def test_a_shot_change_jumps_without_sliding(self) -> None:
        # 場面の切り替わりで枠が流れると、前の場面の絵が横へ滑って見える
        plans = [
            ShotPlan(0, 60, STEADY, ((0, -200.0, 0.0),)),
            ShotPlan(60, 120, STEADY, ((60, 300.0, 0.0),)),
        ]
        across, _down = focus_curves(plans)
        assert across.at(59) == pytest.approx(-200.0)
        assert across.at(60) == pytest.approx(300.0)


class TestTheSpeed:
    def test_a_jumping_clue_does_not_whip_the_window(self) -> None:
        # 抑えないと、散った動きの偏りや速い被写体で枠が 1 秒に画面の幅ほど振り回される
        points = [
            (frame, -600.0 if (frame // 8) % 2 else 600.0, 0.01) for frame in range(0, 120, 8)
        ]
        plan = plan_shot(_shot(points), FOLLOW, WINDOW.size, 8, max_step=5.0)
        for (frame_a, x_a, _), (frame_b, x_b, _) in zip(plan.points, plan.points[1:], strict=False):
            assert abs(x_b - x_a) <= 5.0 * (frame_b - frame_a) + 1e-6


class TestTheShots:
    def test_a_subject_that_stays_gets_a_still_window(self) -> None:
        # 少し動くだけの話し手を追うと、枠が手振りで揺れ続けて見にくい
        shot = _shot([(frame, 300.0 + (frame % 3) * 20.0, 0.01) for frame in range(0, 120, 8)])
        plan = plan_shot(shot, AUTO, WINDOW.size, 30)
        assert plan.mode == STEADY
        assert len(plan.points) == 1
        assert plan.points[0][1] == pytest.approx(320.0, abs=1.0)

    def test_a_subject_that_walks_across_is_followed(self) -> None:
        # 止めたままにすると、画面を横切る人が枠から出ていく
        shot = _shot([(frame, -600.0 + frame * 10.0, 0.01) for frame in range(0, 120, 8)])
        plan = plan_shot(shot, AUTO, WINDOW.size, 30)
        assert plan.mode == FOLLOW
        assert plan.points[0][0] == 0
        assert plan.points[0][1] < plan.points[-1][1]
        for frame, x, _y in plan.points:
            assert BOUNDS[0] - 1e-6 <= x <= BOUNDS[1] + 1e-6, frame

    def test_steady_is_kept_when_asked(self) -> None:
        # 止めてと頼まれて追うと、本人の好み（三脚の見え方）が効かない
        shot = _shot([(frame, -600.0 + frame * 10.0, 0.01) for frame in range(0, 120, 8)])
        assert plan_shot(shot, STEADY, WINDOW.size, 30).mode == STEADY

    def test_center_does_not_look_at_the_subject(self) -> None:
        # 真ん中を頼まれて被写体へ寄せると、解析を待たない約束と結果が食い違う
        shot = _shot([(0, 500.0, 0.05)])
        plan = plan_shot(shot, CENTER, WINDOW.size, 30)
        assert plan.mode == CENTER
        assert plan.points == ((0, 0.0, 0.0),)

    def test_no_clue_falls_back_to_the_middle(self) -> None:
        # 手がかりの無い場面でざらつきを追うと、止まった風景で枠がふらふら動く
        plan = plan_shot(_shot([(0, 500.0, 0.0), (8, -500.0, 0.0)]), AUTO, WINDOW.size, 30)
        assert plan.no_clue
        assert plan.points == ((0, 0.0, 0.0),)

    def test_the_window_stays_on_the_picture(self) -> None:
        # 枠が絵の外へ出ると、縦の画面の端に黒い帯が出る
        plan = plan_shot(_shot([(0, 950.0, 0.05)]), AUTO, WINDOW.size, 30)
        assert plan.points[0][1] == pytest.approx(BOUNDS[1])

    def test_an_unknown_mode_is_refused(self) -> None:
        # 黙って既定に倒すと、打ち間違えた頼みが別の動かし方で通る
        with pytest.raises(ReframeError):
            plan_shot(_shot([(0, 0.0, 0.01)]), "zoom", WINDOW.size, 30)


class TestTheOverlays:
    def test_a_bottom_caption_stays_at_the_bottom(self) -> None:
        # 位置を画素のまま残すと、横の画面の下の字幕が縦の画面の真ん中より上に浮く
        changes = overlay_changes({"pos_y": AnimatedValue(-380.0)}, (0.5625, 1920 / 1080), 1.0)
        assert changes["pos_y"].static / 960.0 == pytest.approx(-380.0 / 540.0)
        assert "scale" not in changes

    def test_a_fitted_image_keeps_its_pixel_size(self) -> None:
        # 画面に収めて置く画像は縦の画面で小さく収まる 戻さないとロゴが半分の大きさになる
        changes = overlay_changes({}, (0.5625, 1920 / 1080), 0.5)
        assert changes["scale"].static == pytest.approx(200.0)

    def test_a_wide_thing_is_pulled_inside(self) -> None:
        # 寄せないと、右寄せのテロップが縦の画面の右端で切れる
        assert clamp_span(400.0, (-200.0, 200.0), 1080.0) == pytest.approx(340.0)
        assert clamp_span(0.0, (0.0, 2000.0), 1080.0) == pytest.approx(-1000.0)
