"""画面と同じ道具（#349）を本物の窓で動かす

人の Ctrl+C・Ctrl+V と AI のコピー・貼り付けが同じ置き場を使うか、AI の挿入貼り付けが人の
設定に従うか、開いたシーンを消したときに窓がメインへ戻るかを見る 道具の決まりそのものは
``tests/ai/test_screen_parity_tools.py`` が偽のホストで見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox

from sashimono.ai.operations import find_operation
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.ui.main_window import MainWindow


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 道具はどれも取り消せる編集で、人に何も尋ねないはず"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QInputDialog, "getText", popped)
    monkeypatch.setattr(QInputDialog, "getItem", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _text(text: str, start: int = 0) -> Clip:
    return with_fixed_items(
        Clip(timeline_start=start, duration=90, source=TEXT.create(text=text)), picture=True
    )


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    base = Project.create()
    track = Track(TrackKind.VIDEO, "V1", (_text("一つ目"), _text("二つ目", start=100)))
    created = MainWindow(
        base.with_timeline(replace(base.timeline, tracks=(track,))), confirm_unsaved=False
    )
    yield created
    created.close()
    created.deleteLater()


def _run(window: MainWindow, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(window, arguments)


def _texts(window: MainWindow) -> list[tuple[int, str]]:
    return sorted(
        (clip.timeline_start, str(clip.source.params["text"]))
        for track in window.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None
    )


class TestTheClipboardIsShared:
    def test_what_the_assistant_copies_is_pasted_by_ctrl_v(self, window: MainWindow) -> None:
        # 壊れると、AI に「コピーしておいて」と頼んだ物を人が Ctrl+V で貼れない
        first = window.document.project.timeline.tracks[0].clips[0]
        _run(window, "copy_clips", clip_ids=[str(first.id)])
        window.seek(150)
        assert window._timeline.paste_at_playhead()
        assert (150, "一つ目") in _texts(window)

    def test_what_the_person_copies_is_pasted_by_the_assistant(self, window: MainWindow) -> None:
        # 壊れると、人がコピーした物を AI に「3 秒の所へ貼って」と頼んでも貼れない
        second = window.document.project.timeline.tracks[0].clips[1]
        window._timeline.set_selection([second.id])
        assert window._timeline.copy_selected()
        pasted = _run(window, "paste_clips", at_frame=400)["pasted"]
        assert (400, "二つ目") in _texts(window)
        assert [str(clip_id) for clip_id in window.selected_clips] == pasted

    def test_a_cut_by_the_assistant_goes_back_with_one_ctrl_z(self, window: MainWindow) -> None:
        # 壊れると、AI が切り取った物を人が 1 回の Ctrl+Z で戻せない
        before = window.document.project
        first = window.document.project.timeline.tracks[0].clips[0]
        _run(window, "cut_clips", clip_ids=[str(first.id)])
        assert (0, "一つ目") not in _texts(window)
        window.undo()
        assert window.document.project == before

    @pytest.mark.parametrize("all_tracks", [True, False])
    def test_insert_paste_reads_the_person_s_setting(
        self, window: MainWindow, all_tracks: bool
    ) -> None:
        # 壊れると、設定の「挿入貼り付けで押し出す」を変えても、AI の挿入貼り付けだけ前のまま
        window._timeline.set_insert_all_tracks(all_tracks)
        assert window.insert_paste_all_tracks is all_tracks


class TestScenesInTheWindow:
    def test_removing_the_open_scene_shows_the_main_again(self, window: MainWindow) -> None:
        # 壊れると、AI が開いているシーンを消した後、窓が消えたシーンを開いたまま何も描けない
        scene_id = window.create_scene("場面")
        assert scene_id is not None and window.active_scene == scene_id
        _run(window, "remove_scene")
        assert window.active_scene is None
        assert window.document.project.scenes == ()
        assert window.view_project.timeline == window.document.project.timeline

    def test_renaming_the_open_scene_keeps_it_open(self, window: MainWindow) -> None:
        # 壊れると、名前を変えただけで開いているシーンが閉じ、AI の次の道具がメインに当たる
        scene_id = window.create_scene("場面")
        _run(window, "rename_scene", name="冒頭")
        assert window.active_scene == scene_id
        assert [scene.name for scene in window.document.project.scenes] == ["冒頭"]
