"""仕上がりの穴を画面と AI に知らせる（#494 小さく入った動画・書体に無い字）"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from sashimono.ai.operations import find_operation
from sashimono.core.commands.insert import insert_media
from sashimono.core.model import Clip, MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import FakeHost


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    made: list[MainWindow] = []
    yield made
    for window in made:
        window.close()


def _selected(opened: list[MainWindow], project: Project, clip: Clip) -> MainWindow:
    window = MainWindow(project, confirm_unsaved=False)
    opened.append(window)
    window._timeline.select(clip.id)
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()
    return window


def _small(media: MediaItem) -> MediaItem:
    stream = replace(media.video_streams[0], width=640, height=360)
    return replace(media, video_streams=(stream,))


def _picture(project: Project) -> Clip:
    return next(c for t in project.timeline.tracks for c in t.clips if c.show_picture)


def test_the_button_fits_a_small_video(opened: list[MainWindow], video_media: MediaItem) -> None:
    """壊れると、1/3 で入った動画を画面いっぱいにする手段が見つからない"""
    project = Project.create(ProjectSettings(width=1920, height=1080))
    for command in insert_media(project, _small(video_media), at_frame=0):
        project = command.apply(project)
    window = _selected(opened, project, _picture(project))
    (button,) = (
        b for b in window._inspector.findChildren(QPushButton) if b.objectName() == "fit_to_screen"
    )
    button.click()
    assert not _picture(window.document.project).native_size


def test_the_assistant_is_told_of_a_small_video(video_media: MediaItem) -> None:
    """壊れると、AI は絵を描いて見るまで小さく入ったことに気付かず、そのまま仕上げる"""
    small = _small(video_media)
    host = FakeHost(Project.create(ProjectSettings(width=1920, height=1080), media=(small,)))
    operation = find_operation("place_media")
    assert operation is not None
    result = operation(host, {"media_id": str(small.id)})
    assert isinstance(result, dict)
    assert result["small"][0]["screen_percent"] == 33
    assert "native_size" in result["small_note"]


def test_a_full_size_video_is_not_told(video_media: MediaItem) -> None:
    """足りている素材でも知らせると、AI が頼まれていない置き方の変更を始める"""
    host = FakeHost(Project.create(ProjectSettings(width=1920, height=1080), media=(video_media,)))
    operation = find_operation("place_media")
    assert operation is not None
    result = operation(host, {"media_id": str(video_media.id)})
    assert isinstance(result, dict)
    assert "small" not in result


def test_the_panel_tells_absent_characters(opened: list[MainWindow]) -> None:
    """壊れると、代わりの書体で出た字に書き出すまで気付かない"""
    if "Bahnschrift" not in QFontDatabase.families():
        pytest.skip("Bahnschrift の無い機械")
    clip = Clip(timeline_start=0, duration=30, source=TEXT.create(text="字", font="Bahnschrift"))
    project = Project.create()
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    project = project.with_timeline(replace(project.timeline, tracks=(track,)))
    window = _selected(opened, project, clip)
    notes = [label.text() for label in window._inspector.findChildren(QLabel)]
    assert any("この書体に無い字「字」" in note for note in notes)
