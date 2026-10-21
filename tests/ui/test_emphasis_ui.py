"""強調の動き（#331）の設定画面と、本物の窓で AI が付けた動き

偽のホストの試験（tests/ai/test_emphasis_tools.py）は道具が作る線を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに付けた動きが取り消し 1 回で
戻ること、タイムラインの右クリック〔強調の動きを付ける…〕が AI と同じ動きを付けることを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.core.commands import AddClip, SetTrackState, SetWorkArea
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, fixed_effect
from sashimono.core.emphasis import EMPHASIS_EASINGS, EmphasisStyle, fixed_transform
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui import main_window as main_window_module
from sashimono.ui.emphasis_dialog import EmphasisChoice, EmphasisDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.timeline.view import TimelineView
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ui.ai_tool_calls import call_tool


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    強調の動きは取り消せる編集で、確かめはブリッジの承認（ここでは自動で承認）だけ
    窓が出たら作りが変わったということで、開いたまま待つと試験が人の操作を待って止まる
    """

    def refuse(*_arguments: object, **_options: object) -> None:
        raise AssertionError("試験の途中で窓が開いた")

    for owner, name in (
        (QDialog, "exec"),
        (QDialog, "open"),
        (QDialog, "show"),
        (QMainWindow, "show"),
        (QMessageBox, "exec"),
        (QMessageBox, "warning"),
        (QMessageBox, "information"),
        (QMessageBox, "question"),
        (QMessageBox, "critical"),
        (QFileDialog, "getSaveFileName"),
        (QFileDialog, "getOpenFileName"),
    ):
        monkeypatch.setattr(owner, name, refuse)


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    windows: list[MainWindow] = []
    yield windows
    for window in windows:
        window.close()
        window.deleteLater()


def _project() -> tuple[Project, Clip]:
    """図形を 1 本（3 秒）置いた 30 fps の作品"""
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
    track = Track(TrackKind.VIDEO, "V1")
    project = base.with_timeline(replace(base.timeline, tracks=(track,)))
    square = Clip(
        timeline_start=0,
        duration=90,
        source=GeneratedSource(kind="shape", params={"shape": "rect"}),
        effects=(fixed_effect(TRANSFORM_EFFECT_KIND),),
    )
    return AddClip(track.id, square).apply(project), square


def _call(window: MainWindow, name: str, arguments: dict[str, Any]) -> Any:
    """本物の AI と同じ道で道具を呼ぶ（変更の承認と待ちの上限は tests/ui/ai_tool_calls.py）"""
    return call_tool(window, name, arguments)


class TestWindow:
    def test_the_tool_uses_the_saved_preference_and_undoes_once(
        self, opened: list[MainWindow]
    ) -> None:
        # 壊れると、設定画面で選んだ寄る量が AI の強調の動きに効かないか、付けた動きを
        # 取り消し 1 回で戻せない
        PreferenceStore().save(Preferences(emphasis_zoom_percent=140, emphasis_ramp_ms=0))
        project, square = _project()
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        assert window.emphasis_style == EmphasisStyle(zoom_percent=140, ramp_ms=0)
        result = _call(
            window, "emphasize", {"clip_id": str(square.id), "kind": "zoom", "end_frame": 30}
        )
        assert result["scale_percent"] == 140.0
        located = window.document.project.timeline.locate_clip(square.id)
        assert located is not None
        transform = fixed_transform(located[1])
        assert transform is not None
        scale = transform.params["scale"]
        assert isinstance(scale, AnimatedValue)
        assert scale.at(15) == pytest.approx(140.0)
        window.document.undo()
        assert window.document.project == project


class TestPreference:
    def test_the_defaults_are_moderate(self) -> None:
        # 壊れると、何も選んでいない人の動画が大きく寄りすぎて端が欠けるか、ほとんど動かない
        plain = Preferences()
        assert plain.emphasis_style == EmphasisStyle()
        assert (plain.emphasis_zoom_percent, plain.emphasis_ramp_ms, plain.emphasis_easing) == (
            120,
            250,
            "smooth",
        )

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        # 壊れると、手で書き換えた値（100% の寄り・負の長さ・知らない動き方）のまま、
        # 寄りが効かないか、道具が毎回断る
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"emphasis_zoom_percent": 100, "emphasis_ramp_ms": -5,'
            ' "emphasis_easing": "wobble", "emphasis_shake_px": "大きく"}',
            encoding="utf-8",
        )
        assert PreferenceStore().load().emphasis_style == EmphasisStyle()

    def test_the_dialog_offers_and_returns_the_values(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面で強調の動きの強さや動き方を変えても保存されない
        del qt_application
        dialog = PreferencesDialog(
            Preferences(emphasis_zoom_percent=130, emphasis_easing="back", emphasis_shake_px=20)
        )
        try:
            chosen = dialog.preferences()
            assert chosen.emphasis_style == EmphasisStyle(
                zoom_percent=130, easing="back", shake_px=20
            )
            dialog._emphasis_zoom.setValue(150)
            dialog._emphasis_pop.setValue(110)
            dialog._emphasis_pan.setValue(300)
            dialog._emphasis_ramp.setValue(500)
            dialog._emphasis_easing.setCurrentIndex(EMPHASIS_EASINGS.index("bounce"))
            assert dialog.preferences().emphasis_style == EmphasisStyle(
                zoom_percent=150,
                pop_percent=110,
                shake_px=20,
                pan_px=300,
                ramp_ms=500,
                easing="bounce",
            )
        finally:
            dialog.deleteLater()


EMPHASIS_TEXT = "強調の動きを付ける…"


def _clip_point(view: TimelineView, track: Track, frame: int) -> QPoint:
    for band in view.view_layout.bands(view.project.timeline):
        if band.track.id == track.id:
            return QPoint(int(view.view_layout.frame_to_x(frame)), (band.top + band.bottom) // 2)
    raise AssertionError(f"トラックが無い: {track.name}")


def _menu_action(window: MainWindow, track: Track, frame: int) -> QAction:
    view = window._timeline
    view.resize(1200, 400)
    menu = view.build_context_menu(_clip_point(view, track, frame))
    found = next((a for a in menu.actions() if a.text().startswith(EMPHASIS_TEXT)), None)
    assert found is not None, [a.text() for a in menu.actions()]
    return found


def _scale(window: MainWindow, clip: Clip) -> AnimatedValue:
    located = window.document.project.timeline.locate_clip(clip.id)
    assert located is not None
    transform = fixed_transform(located[1])
    assert transform is not None
    scale = transform.params["scale"]
    assert isinstance(scale, AnimatedValue)
    return scale


class TestRightClick:
    """タイムラインの右クリックから人が使う（#331 2026-10-18 利用者の決定）

    窓は main_window の ask_emphasis を差し替えて答える 差し替えずに窓が開くと
    no_windows が落とす
    """

    def test_the_item_is_on_a_picture_clip(self, opened: list[MainWindow]) -> None:
        # 壊れると、人は AI に頼まないと強調の動きをまとめて付けられず、点を 1 つずつ打つ
        project, _square = _project()
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        assert _menu_action(window, project.timeline.tracks[0], 30).isEnabled()

    def test_a_locked_track_greys_it_out(self, opened: list[MainWindow]) -> None:
        # 壊れると、人が触らせないためにロックしたトラックの絵が右クリックで動かされる
        project, _square = _project()
        track = project.timeline.tracks[0]
        project = SetTrackState(track.id, locked=True).apply(project)
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        assert not _menu_action(window, track, 30).isEnabled()

    def test_the_menu_matches_the_tool_and_undoes_once(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が右クリックで付けた寄りと AI が付けた寄りの線が食い違うか、
        # 取り消し 1 回で戻らない 範囲は書き出し範囲が無ければ再生ヘッドから 1 秒
        PreferenceStore().save(Preferences(emphasis_zoom_percent=130))
        project, square = _project()
        asked: list[EmphasisStyle] = []

        def answer(_parent: object, style: EmphasisStyle, _span: str) -> EmphasisChoice:
            asked.append(style)
            return EmphasisChoice("zoom", 130, style.ramp_ms, style.easing)

        monkeypatch.setattr(main_window_module, "ask_emphasis", answer)
        by_hand = MainWindow(project, confirm_unsaved=False)
        opened.append(by_hand)
        by_hand.seek(20)
        _menu_action(by_hand, project.timeline.tracks[0], 30).trigger()
        # 窓の既定は本人の設定
        assert asked == [EmphasisStyle(zoom_percent=130)]
        by_tool = MainWindow(project, confirm_unsaved=False)
        opened.append(by_tool)
        _call(
            by_tool,
            "emphasize",
            {"clip_id": str(square.id), "kind": "zoom", "start_frame": 20, "end_frame": 50},
        )
        assert _scale(by_hand, square) == _scale(by_tool, square)
        assert _scale(by_hand, square).at(35) == pytest.approx(130.0)
        by_hand.document.undo()
        assert by_hand.document.project == project

    def test_the_work_area_is_the_range(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、書き出し範囲で要所を選んだのに再生ヘッドの所へ動きが付く
        project, square = _project()
        project = SetWorkArea((40, 80)).apply(project)
        monkeypatch.setattr(
            main_window_module,
            "ask_emphasis",
            lambda _parent, style, _span: EmphasisChoice("pan", 50, 0, "linear", "up"),
        )
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        window.seek(0)
        _menu_action(window, project.timeline.tracks[0], 30).trigger()
        located = window.document.project.timeline.locate_clip(square.id)
        assert located is not None
        transform = fixed_transform(located[1])
        assert transform is not None
        pos_y = transform.params["pos_y"]
        assert isinstance(pos_y, AnimatedValue)
        assert pos_y.at(30) == 0.0
        # 上へ（Y は上が正）
        assert pos_y.at(60) == pytest.approx(50.0)

    def test_cancelling_changes_nothing(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、窓でやめたのに動きが付き、取り消しの段が増える
        project, _square = _project()
        monkeypatch.setattr(main_window_module, "ask_emphasis", lambda *_args: None)
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        _menu_action(window, project.timeline.tracks[0], 30).trigger()
        assert window.document.project == project
        assert not window.document.can_undo

    def test_the_dialog_starts_from_the_preference(self, qt_application: QApplication) -> None:
        # 壊れると、窓の既定が本人の設定と違い、毎回値を直してから付けることになる
        # 種類を替えると強さもその種類の既定へ替わる（120 % のまま揺れへ替えて 120 px で揺れない）
        del qt_application
        dialog = EmphasisDialog(EmphasisStyle(zoom_percent=140, shake_px=9), "再生ヘッドから 1 秒")
        try:
            assert dialog.choice() == EmphasisChoice("zoom", 140, 250, "smooth", "left")
            dialog.kind_box.setCurrentIndex(dialog.kind_box.findData("shake"))
            assert dialog.choice().strength == 9
            assert not dialog.direction_box.isEnabled()
        finally:
            dialog.deleteLater()
