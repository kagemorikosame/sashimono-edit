"""検査で見つけた問題を AI が直して調べ直す繰り返し（#339）を、本物の窓と本物の検査で確かめる

偽のホストの試験（tests/ai/test_fix_loop.py）は数え方・区切り方・断り方を見る ここでは
AI パネルと同じく指示の段を開き、小さい音の作品を check_output で調べて normalize_loudness で
直し、調べ直す 直した回が「AI: 検査の直し（1 回目）」の段になり、〔元に戻す〕1 回でその回だけ
戻ること、設定画面の回数が本物の窓の道具に効くことを見る 音だけを扱うので GPU は要らない
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
from sashimono.ai.fixes import AUTO, fix_label
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, Author
from sashimono.core.model import Clip, MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode.probe import probe_media
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending
from tests.check_media import tone, write_tone


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    調べる・そろえる・断るのどれも窓を出さない（確かめはブリッジの自動の承認だけ）
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


@pytest.fixture(scope="module")
def quiet(tmp_path_factory: pytest.TempPathFactory) -> MediaItem:
    """-20 dBFS の 1 kHz（両方のチャンネル）は -20 LUFS 配信の基準 -14 より 6 LU 小さい"""
    folder = tmp_path_factory.mktemp("fix")
    return probe_media(write_tone(folder / "小さい.wav", tone(6.0, -20.0)))


def _project(quiet: MediaItem) -> Project:
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(quiet,))
    track = Track(kind=TrackKind.AUDIO, name="A1")
    project = base.with_timeline(replace(base.timeline, tracks=(track,)))
    return AddClip(track.id, Clip(timeline_start=0, duration=180, media_id=quiet.id)).apply(project)


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    windows: list[MainWindow] = []
    yield windows
    for window in windows:
        window.close()
        window.deleteLater()


def _window(opened: list[MainWindow], project: Project) -> MainWindow:
    window = MainWindow(project, confirm_unsaved=False)
    opened.append(window)
    return window


def _call(bridge: EditorBridge, name: str, arguments: dict[str, Any]) -> Any:
    """AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道）"""
    operation = find_operation(name)
    assert operation is not None
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120.0
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


_CHECK = {"checks": ["loudness"], "wait_seconds": 60}


class TestLoop:
    def test_a_fix_round_is_one_undo_step(self, opened: list[MainWindow], quiet: MediaItem) -> None:
        """壊れると、AI が音量を直した回を人が〔元に戻す〕1 回で戻せず、指示の前まで
        戻るか何も戻らない"""
        project = _project(quiet)
        window = _window(opened, project)
        bridge = allow_sending(EditorBridge(window, auto_approve=True))
        # AI パネルが指示ごとに開く段と同じ
        window.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        try:
            # 調べる前の指示の編集 直した回を戻しても、これは残る
            _call(bridge, "add_track", {"kind": "audio", "name": "BGM"})
            edited = window.document.project
            first = _call(bridge, "check_output", _CHECK)
            (problem,) = first["problems"]
            assert problem["kind"] == "loudness"
            assert problem["fix"]["action"] == AUTO
            assert first["fix_round"]["next_round"] == 1
            for name in problem["fix"]["tools"][:1]:
                _call(bridge, name, {"wait_seconds": 60})
            second = _call(bridge, "check_output", _CHECK)
        finally:
            window.document.end_checkpoint()

        assert second["problems"] == []
        assert second["fix_round"]["used"] == 1
        assert window.document.undo_label == fix_label(1)
        window.undo()
        assert window.document.project == edited
        assert window.document.undo_label == "AI: 仕上げて"
        window.undo()
        assert window.document.project == project

    def test_zero_rounds_refuses_the_fix(self, opened: list[MainWindow], quiet: MediaItem) -> None:
        """壊れると、設定で「直さない」を選んだ人の作品の音量を、AI が調べた後に変えてしまう"""
        PreferenceStore().save(Preferences(ai_fix_rounds=0))
        project = _project(quiet)
        window = _window(opened, project)
        assert window.fix_rounds == 0
        bridge = allow_sending(EditorBridge(window, auto_approve=True))
        window.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        try:
            result = _call(bridge, "check_output", _CHECK)
            assert result["fix_round"]["closed"] is True
            with pytest.raises(ToolError, match="上限"):
                _call(bridge, "normalize_loudness", {"wait_seconds": 60})
        finally:
            window.document.end_checkpoint()
        assert window.document.project == project


class TestPreference:
    def test_the_default_is_three_rounds(self) -> None:
        """壊れると、何も選んでいない人の所で AI が直さずに止まるか、止まらずに直し続ける"""
        assert Preferences().ai_fix_rounds == 3

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        """壊れると、手で書き換えた値（負の数や文字）のまま、直す回数が決まらない"""
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        for broken in ('"三"', "-1", "11", "true"):
            path.write_text(f'{{"ai_fix_rounds": {broken}}}', encoding="utf-8")
            assert PreferenceStore().load().ai_fix_rounds == 3, broken

    def test_the_dialog_offers_and_returns_the_value(self, qt_application: QApplication) -> None:
        """壊れると、設定画面で直す回数を変えても保存されず、AI は前の回数で直し続ける"""
        del qt_application
        dialog = PreferencesDialog(Preferences(ai_fix_rounds=5))
        try:
            assert dialog.preferences().ai_fix_rounds == 5
            dialog._ai_fix_rounds.setValue(0)
            assert dialog.preferences().ai_fix_rounds == 0
            assert dialog._ai_fix_rounds.maximum() == 10
        finally:
            dialog.deleteLater()
