"""プレビューで部分モザイク・ぼかしと部分フィルタの範囲をつまんで動かす（利用者の要望）

範囲の中を掴んで移動、角で幅と高さ、辺の真ん中で片方の大きさ、上の丸で回転
離したら 1 段で取り消せる キーフレームのある値は再生位置のキーを動かす（◆ と同じ）
範囲は絵の原点（素材は絵の中央、フィルタのクリップは画面の中央）から数え、Y は上が正
後ろに積んだ配置（拡大・回転）も範囲ごと絵を動かすので、枠もそれに付いていく

プレビューは GL を作らずに使う（窓に出さない） 画面の座標がそのまま合成の画素になるよう
320 × 180 で作る（:mod:`tests.ui.test_preview_handles` と同じ）
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent, Qt

from sashimono.core.commands import Document, SetKeyframe
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND
from sashimono.core.model import (
    FILTER_KIND,
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.region import PARTIAL_FILTER, REGION_BLUR
from sashimono.engine.render.region_outline import region_frame
from sashimono.ui.preview import PreviewWidget
from tests.ui.test_preview_handles import (
    MakeWidget,
    _apply,
    _clip,
    _drag,
    _media,
    _project,
    _Recorder,
    _send,
    make_widget,
)

__all__ = ["make_widget"]


def _region(kind: str = REGION_BLUR, **values: float) -> Effect:
    params = {
        "center_x": 20.0,
        "center_y": 10.0,
        "region_width": 40.0,
        "region_height": 20.0,
        **values,
    }
    return registry.require(kind).create(
        **{name: AnimatedValue(value) for name, value in params.items()}
    )


def _with(clip: Clip, *effects: Effect) -> Clip:
    """ふつうのエフェクトを固定の欄（反転・配置）の前へ積む（AddEffect と同じ所）"""
    loose = tuple(e for e in clip.effects if not e.fixed)
    fixed = tuple(e for e in clip.effects if e.fixed)
    return replace(clip, effects=(*loose, *effects, *fixed))


def _placed(clip: Clip, **values: float) -> Clip:
    """固定の配置に値を入れる"""
    effects = []
    for effect in clip.effects:
        if effect.fixed and effect.kind == TRANSFORM_EFFECT_KIND:
            for name, value in values.items():
                effect = effect.with_param(name, AnimatedValue(value))
        effects.append(effect)
    return replace(clip, effects=tuple(effects))


def _number(project: Project, clip_id: ClipId, effect_id: str, name: str, frame: int = 0) -> float:
    located = project.timeline.locate_clip(clip_id)
    assert located is not None
    effect = next(e for e in located[1].effects if e.id == effect_id)
    value = effect.params[name]
    assert isinstance(value, AnimatedValue)
    return value.at(frame)


def _open(make_widget: MakeWidget, clip: Clip) -> tuple[PreviewWidget, Project, _Recorder]:
    project, _ = _project(clip)
    widget = make_widget(project, clip.id)
    return widget, project, _Recorder(widget)


class TestTheFrame:
    def test_the_frame_sits_on_the_region(self, make_widget: MakeWidget) -> None:
        # 素材は画面の真ん中 (80, 45)-(240, 135) 原点は絵の中央 (160, 90) Y は上が正
        clip = _with(_clip(_media()), region := _region())
        widget, _, _ = _open(make_widget, clip)
        assert widget.region_effect == region.id
        found = widget._region()
        assert found is not None
        corners = found[3].corners
        assert corners[0] == pytest.approx((160.0, 70.0))
        assert corners[2] == pytest.approx((200.0, 90.0))

    def test_a_later_placement_moves_the_frame_with_the_picture(self) -> None:
        # 配置で 2 倍にして右へ 30 動かすと、範囲も絵と一緒に 2 倍になって動く
        clip = _placed(_with(_clip(_media()), region := _region()), scale=200.0, pos_x=30.0)
        project, _ = _project(clip)
        frame = region_frame(project, clip, region.id, 0)
        assert frame is not None
        # 中心は (160 + 20 × 2 + 30, 90 - 10 × 2) 幅と高さは 80 × 40
        assert frame.corners[0] == pytest.approx((190.0, 50.0))
        assert frame.corners[2] == pytest.approx((270.0, 90.0))

    def test_a_filter_clip_counts_from_the_screen_centre(self, make_widget: MakeWidget) -> None:
        # フィルタのクリップは外枠を持たないが、範囲の枠は出す 原点は画面の中央
        background = _clip(_media())
        filter_clip = Clip(
            timeline_start=0,
            duration=60,
            source=GeneratedSource(kind=FILTER_KIND),
            effects=(region := _region(PARTIAL_FILTER, center_x=-40.0, center_y=30.0),),
        )
        project, _ = _project(background, filter_clip)
        widget = make_widget(project, filter_clip.id)
        assert widget.region_effect == region.id
        found = widget._region()
        assert found is not None
        # 中心は (160 - 40, 90 - 30)
        assert found[3].corners[0] == pytest.approx((100.0, 50.0))
        assert found[3].corners[2] == pytest.approx((140.0, 70.0))

    def test_a_sound_only_clip_has_no_region_frame(self, make_widget: MakeWidget) -> None:
        # 絵を出さない音声クリップに枠を残すと、見えない範囲を動かせてしまう
        visible = _with(_clip(_media()), _region())
        project, _ = _project(visible)
        hidden = replace(visible, show_picture=False)
        track = replace(project.timeline.tracks[0], kind=TrackKind.MIXED, clips=(hidden,))
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))
        widget = make_widget(project, hidden.id)
        assert widget._region() is None

    def test_the_last_touched_region_is_shown(self, make_widget: MakeWidget) -> None:
        # 2 つ積んだとき、設定パネルで触った方の枠を出す
        first, second = _region(), _region(center_x=-50.0)
        clip = _with(_clip(_media()), first, second)
        widget, _, _ = _open(make_widget, clip)
        assert widget.region_effect == first.id
        widget.set_region_effect(second.id)
        assert widget.region_effect == second.id
        # 範囲を持たないエフェクトを触っても、出している枠は変えない
        widget.set_region_effect(clip.effects[-1].id)
        assert widget.region_effect == second.id


class TestDragging:
    def test_dragging_inside_moves_the_region_not_the_clip(self, make_widget: MakeWidget) -> None:
        # 範囲はクリップの枠の中にある 枠を先に見ると、範囲を掴めずにクリップが動く
        clip = _with(_clip(_media()), region := _region())
        widget, project, seen = _open(make_widget, clip)
        _drag(widget, (180, 80), (185, 75), (190, 70))
        assert len(seen.committed) == 1
        commands, label = seen.committed[0]
        assert label == "プレビューで範囲を移動"
        moved = _apply(project, commands)
        assert _number(moved, clip.id, region.id, "center_x") == pytest.approx(30.0)
        assert _number(moved, clip.id, region.id, "center_y") == pytest.approx(20.0)

    def test_outside_the_region_the_clip_still_moves(self, make_widget: MakeWidget) -> None:
        # 範囲の外を掴んでもクリップが動かないと、部分モザイクを掛けた素材を位置合わせできない
        clip = _with(_clip(_media()), _region())
        widget, _, seen = _open(make_widget, clip)
        _drag(widget, (100, 120), (110, 120))
        assert seen.committed[0][1] == "プレビューで位置を変更"

    def test_a_corner_resizes_and_keeps_the_opposite_corner(self, make_widget: MakeWidget) -> None:
        # 右下を引くと左上 (160, 70) は動かない 中心は新しい四角の真ん中へ
        clip = _with(_clip(_media()), region := _region())
        widget, project, seen = _open(make_widget, clip)
        _drag(widget, (200, 90), (220, 100))
        moved = _apply(project, seen.committed[0][0])
        assert seen.committed[0][1] == "プレビューで範囲の大きさを変更"
        assert _number(moved, clip.id, region.id, "region_width") == pytest.approx(60.0)
        assert _number(moved, clip.id, region.id, "region_height") == pytest.approx(30.0)
        assert _number(moved, clip.id, region.id, "center_x") == pytest.approx(30.0)
        assert _number(moved, clip.id, region.id, "center_y") == pytest.approx(5.0)

    def test_alt_resizes_around_the_centre(self, make_widget: MakeWidget) -> None:
        # Alt で中心を保てないと、顔に合わせた範囲を広げるたびに中心がずれて置き直すことになる
        clip = _with(_clip(_media()), region := _region())
        widget, project, seen = _open(make_widget, clip)
        _drag(widget, (200, 90), (210, 95), modifiers=Qt.KeyboardModifier.AltModifier)
        moved = _apply(project, seen.committed[0][0])
        assert _number(moved, clip.id, region.id, "region_width") == pytest.approx(60.0)
        assert _number(moved, clip.id, region.id, "center_x") == pytest.approx(20.0)

    def test_an_edge_changes_one_side(self, make_widget: MakeWidget) -> None:
        # 辺の真ん中は向きの 1 つだけ 右の辺を引いても高さは変わらない
        clip = _with(_clip(_media()), region := _region())
        widget, project, seen = _open(make_widget, clip)
        _drag(widget, (200, 80), (210, 90))
        moved = _apply(project, seen.committed[0][0])
        assert _number(moved, clip.id, region.id, "region_width") == pytest.approx(50.0)
        assert _number(moved, clip.id, region.id, "region_height") == pytest.approx(20.0)
        assert _number(moved, clip.id, region.id, "center_x") == pytest.approx(25.0)
        assert _number(moved, clip.id, region.id, "center_y") == pytest.approx(10.0)

    def test_the_knob_turns_clockwise(self, make_widget: MakeWidget) -> None:
        # 上の辺の真ん中 (180, 70) から 22 上の丸を、中心 (180, 80) の右へ回すと時計回り 90 度
        clip = _with(_clip(_media()), region := _region())
        widget, project, seen = _open(make_widget, clip)
        _drag(widget, (180, 48), (200, 55), (212, 80))
        moved = _apply(project, seen.committed[0][0])
        assert _number(moved, clip.id, region.id, "rotation") == pytest.approx(90.0, abs=0.5)

    def test_a_doubled_picture_moves_the_region_half_as_far(self, make_widget: MakeWidget) -> None:
        # 画面で 20 動かしても、2 倍に拡げた絵の中では 10 画素 そのまま足すと範囲が指から逃げる
        clip = _placed(_with(_clip(_media()), region := _region()), scale=200.0)
        widget, project, seen = _open(make_widget, clip)
        # 中心は (200, 70)
        _drag(widget, (200, 70), (220, 70))
        moved = _apply(project, seen.committed[0][0])
        assert _number(moved, clip.id, region.id, "center_x") == pytest.approx(30.0)

    def test_a_keyed_value_moves_the_key_under_the_playhead(self, make_widget: MakeWidget) -> None:
        # 点を全部捨てて止まった値にすると、動かしていた範囲が止まる（◆ と同じく再生位置へ）
        keyed = AnimatedValue(20.0, keyframes=(Keyframe(0, 20.0), Keyframe(40, 60.0)))
        clip = _with(_clip(_media()), region := _region())
        clip = replace(
            clip,
            effects=tuple(
                e.with_param("center_x", keyed) if e.id == region.id else e for e in clip.effects
            ),
        )
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        widget.set_frame(20)
        seen = _Recorder(widget)
        # 20 フレームの中心は X 40 → 画面の (200, 80)
        _drag(widget, (200, 80), (210, 80))
        commands = seen.committed[0][0]
        assert all(isinstance(c, SetKeyframe) for c in commands)
        moved = _apply(project, commands)
        assert _number(moved, clip.id, region.id, "center_x", 20) == pytest.approx(50.0)
        assert _number(moved, clip.id, region.id, "center_x", 0) == pytest.approx(20.0)
        assert _number(moved, clip.id, region.id, "center_x", 40) == pytest.approx(60.0)

    def test_it_is_one_undo_step(self, make_widget: MakeWidget) -> None:
        # 範囲の移動が何段にも分かれると、ずらしすぎた範囲を戻すのに取り消しを何回も押すことになる
        clip = _with(_clip(_media()), _region())
        widget, project, seen = _open(make_widget, clip)
        _drag(widget, (180, 80), (190, 80), (200, 80))
        document = Document(project)
        commands, label = seen.committed[0]
        with document.checkpoint(label):
            for command in commands:
                document.execute(command)
        document.undo()
        assert document.project == project

    def test_escape_cancels(self, make_widget: MakeWidget) -> None:
        # Esc で止められないと、掴み間違えた範囲がそのまま動き、取り消しで戻すしかなくなる
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QApplication

        clip = _with(_clip(_media()), _region())
        widget, _, seen = _open(make_widget, clip)
        _send(widget, QEvent.Type.MouseButtonPress, (180, 80))
        _send(widget, QEvent.Type.MouseMove, (190, 80))
        QApplication.sendEvent(
            widget,
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
        )
        _send(widget, QEvent.Type.MouseButtonRelease, (190, 80))
        assert seen.committed == []
        assert seen.previewed[-1] == []


def test_the_rotated_region_frame_matches_the_shader(qt_application: object) -> None:
    # 範囲の回転は正で時計回り（シェーダの region_local の逆） 逆に回すと、枠と効く所が
    # 逆へ傾く 枠の四隅をシェーダの式へ戻すと、範囲の端（±幅/2, ±高さ/2）になる
    del qt_application
    region = _region(rotation=30.0)
    clip = _with(_clip(_media()), region)
    project, _ = _project(clip)
    frame = region_frame(project, clip, region.id, 0)
    assert frame is not None
    angle = math.radians(30.0)
    for (x, y), (lx, ly) in zip(
        frame.corners, ((-20, 10), (20, 10), (20, -10), (-20, -10)), strict=True
    ):
        # 画面の座標から、原点 (160, 90) を引いて Y を上向きに直し、中心を引く
        dx, dy = x - 160.0 - 20.0, (90.0 - y) - 10.0
        local = (
            math.cos(angle) * dx - math.sin(angle) * dy,
            math.sin(angle) * dx + math.cos(angle) * dy,
        )
        assert local == pytest.approx((lx, ly), abs=1e-6)


def test_touching_a_region_in_the_panel_tells_which_one(qt_application: object) -> None:
    # 設定パネルで 2 つ目の範囲を直しているのに 1 つ目の枠が出ると、どれを直しているか分からない
    from sashimono.core.commands import ParamPath
    from sashimono.ui.inspector.panel import InspectorPanel

    del qt_application
    first, second = _region(), _region(center_x=-50.0)
    clip = _with(_clip(_media()), first, second)
    project, _ = _project(clip)
    panel = InspectorPanel()
    try:
        panel.set_project(project)
        panel.set_selection((clip.id,))
        seen: list[str] = []
        panel.effect_focused.connect(seen.append)
        panel._on_value_changed(
            ParamPath.of_effect(clip.id, second.id, "center_x"), AnimatedValue(-40.0)
        )
        assert seen == [str(second.id)]
    finally:
        panel.deleteLater()
