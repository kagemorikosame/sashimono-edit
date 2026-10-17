"""強調の動き（#331）の設定画面と、本物の窓で AI が付けた動き

偽のホストの試験（tests/ai/test_emphasis_tools.py）は道具が作る線を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに付けた動きが取り消し 1 回で
戻ることを見る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip
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
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending


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
    """AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道） 変更は自動で承認する"""
    operation = find_operation(name)
    assert operation is not None
    bridge = allow_sending(EditorBridge(window, auto_approve=True))
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while thread.is_alive():
        bridge.pump()
        QApplication.processEvents()
        thread.join(0.01)
        if time.monotonic() > deadline:
            bridge.cancel()
            thread.join(5.0)
            pytest.fail("道具が返らなかった")
    if "error" in box:
        raise box["error"]
    return box["result"]


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
