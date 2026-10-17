"""画面からシーンとグループを扱う

シーンを開いたあとの編集が、開いたシーンの中へ入ることと、取り消しでシーンが
消えたときにメインへ戻ることが肝 グループは、1 本クリックすると仲間ごと選ばれること
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import GroupClips
from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.ui.main_window import MainWindow


def _project() -> Project:
    base = Project.create()
    clips = (
        Clip(timeline_start=0, duration=30, source=TEXT.create()),
        Clip(timeline_start=40, duration=30, source=TEXT.create()),
    )
    return base.with_timeline(replace(base.timeline, tracks=(Track(TrackKind.VIDEO, "V1", clips),)))


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    yield created
    created.close()


class TestScenes:
    def test_text_added_in_a_scene_stays_in_the_scene(self, window: MainWindow) -> None:
        # メインに入ると、シーンを開いて作ったはずのテロップが本編に出る
        scene_id = window.create_scene("オープニング")
        assert scene_id is not None and window.active_scene == scene_id
        window.add_text()
        root = window.document.project
        assert len(root.require_scene(scene_id).timeline.tracks) == 1
        assert len(root.timeline.tracks[0].clips) == 2
        # タイムラインの画面は開いたシーンを映す
        assert len(window._timeline.project.timeline.tracks) == 1

    def test_placing_a_scene_in_main(self, window: MainWindow) -> None:
        # シーンを本編へ置けないと、作ったシーンを動画に使えない
        scene_id = window.create_scene("挿入")
        assert scene_id is not None
        window.open_scene(None)
        window._seek(100)
        window.place_scene(scene_id)
        placed = [
            clip
            for track in window.document.project.timeline.tracks
            for clip in track.clips
            if clip.scene_id == scene_id
        ]
        assert len(placed) == 1

    def test_undoing_the_scene_returns_to_main(self, window: MainWindow) -> None:
        # 消えたシーンを開いたままだと、次の操作が「シーンが見つからない」で全部失敗する
        scene_id = window.create_scene("消える")
        assert window.active_scene == scene_id
        window.undo()
        assert window.active_scene is None

    def test_a_placed_scene_is_not_removed(self, window: MainWindow) -> None:
        # 使っているシーンを消せてしまうと、本編に置いたクリップの中身が空になり絵が消える
        scene_id = window.create_scene("使用中")
        assert scene_id is not None
        window.open_scene(None)
        window.place_scene(scene_id)
        window.open_scene(scene_id)
        window.remove_active_scene()
        assert window.document.project.find_scene(scene_id) is not None


class TestGroups:
    def test_one_click_selects_the_whole_group(self, window: MainWindow) -> None:
        # 仲間ごと選ばれないと、束ねても 1 本ずつ動かすことになる
        clips = window.document.project.timeline.tracks[0].clips
        window.execute(GroupClips((clips[0].id, clips[1].id)))
        view = window._timeline
        view.resize(900, 300)
        band = view._layout.bands(view.project.timeline)[0]
        point = QPoint(int(view._layout.frame_to_x(10)), band.top + band.height // 2)
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=point)
        assert set(view.selected_clips) == {clips[0].id, clips[1].id}

    def test_right_click_selects_the_whole_group(self, window: MainWindow) -> None:
        # 1 本だけ選び直すと、右クリックの削除や切り取りで束が裂ける
        clips = window.document.project.timeline.tracks[0].clips
        window.execute(GroupClips((clips[0].id, clips[1].id)))
        view = window._timeline
        view.resize(900, 300)
        band = view._layout.bands(view.project.timeline)[0]
        point = QPoint(int(view._layout.frame_to_x(50)), band.top + band.height // 2)
        view.build_context_menu(point)
        assert set(view.selected_clips) == {clips[0].id, clips[1].id}
        assert view.selected_clip == clips[1].id

    def test_the_place_button_needs_another_scene(self, window: MainWindow) -> None:
        # 開いているシーン自身しか無いのに押せると、押してから断られて壊れて見える
        scene_id = window.create_scene("ひとつだけ")
        assert scene_id is not None
        window.open_scene(scene_id)
        assert not window._scene_bar._place_button.isEnabled()
        window.open_scene(None)
        assert window._scene_bar._place_button.isEnabled()

    def test_group_and_ungroup_from_the_selection(self, window: MainWindow) -> None:
        # 選んだ物をグループにできないか解けないと、まとめて動かすことも 1 本だけ直すこともできない
        view = window._timeline
        view.select_all()
        assert view.group_selected()
        assert {c.group_id for c in window.document.project.timeline.tracks[0].clips} != {None}
        assert view.ungroup_selected()
        assert {c.group_id for c in window.document.project.timeline.tracks[0].clips} == {None}
