"""AI の取り消し・やり直しを本物の窓で確かめる（B7 #333）

AI の道具が履歴を動かしたら、人が Ctrl+Z で取り消したときと同じく、タイムラインと
メニューの「元に戻す: 〜」「やり直す: 〜」が今の形に追い付くこと AI の指示の途中で
人が Ctrl+Z を押したときも、段の外の履歴を壊さないこと
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.ui.main_window import MainWindow


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 取り消しとやり直しは、確かめの窓もファイルの窓も出さない"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _small_project() -> Project:
    track = Track(TrackKind.VIDEO, "V1")
    project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    project = AddTrack(track).apply(project)
    return AddClip(track.id, Clip(timeline_start=0, duration=90)).apply(project)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_small_project(), confirm_unsaved=False)
    yield created
    created.close()
    created.deleteLater()


def _tool(window: MainWindow, name: str, **arguments: Any) -> Any:
    operation = find_operation(name)
    assert operation is not None
    return operation(window, arguments)


def _split(window: MainWindow, frame: int) -> None:
    clip = window.document.project.timeline.tracks[0].clips[0].id
    _tool(window, "split_clip", clip_id=str(clip), frame=frame)


def _shown_clips(window: MainWindow) -> int:
    return len(window._timeline.project.timeline.tracks[0].clips)


class TestTheScreenFollowsTheAi:
    def test_the_timeline_and_menu_follow_an_ai_undo(self, window: MainWindow) -> None:
        """壊れると、AI が取り消してもタイムラインに切った跡が残り、メニューも前の名前のまま"""
        _split(window, 30)
        assert _shown_clips(window) == 2

        _tool(window, "undo")

        assert _shown_clips(window) == 1
        assert window._undo_action.isEnabled() is False
        assert window._redo_action.text() == "やり直す: クリップを分割"

    def test_the_timeline_and_menu_follow_an_ai_redo(self, window: MainWindow) -> None:
        """壊れると、AI がやり直しても画面は取り消したままに見え、人が同じ所をもう 1 度切る"""
        _split(window, 30)
        window.undo()

        _tool(window, "redo")

        assert _shown_clips(window) == 2
        assert window._undo_action.text() == "元に戻す: クリップを分割"
        assert window._redo_action.isEnabled() is False


class TestThePersonUndoesDuringAnInstruction:
    def test_ctrl_z_during_an_instruction_keeps_the_earlier_history(
        self, window: MainWindow
    ) -> None:
        """壊れると、AI の指示の途中で人が Ctrl+Z を押すと、前の段が履歴から消えて戻れなくなる"""
        _split(window, 60)
        window.document.begin_checkpoint("AI: もっと切って")
        _split(window, 30)

        window.undo()

        assert _shown_clips(window) == 2
        window.document.end_checkpoint()
        assert window.document.history_labels == ("クリップを分割",)
        window.undo()
        assert _shown_clips(window) == 1
