"""タイムラインの目印（マーカー） 目盛りに旗で見え、右クリックで置く・名前を変える・消す（#330）

AI アシスタントが置いた目印を人が見て手直しできないと、書き出しの目次を確かめる手段が無い
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from sashimono.core.commands import (
    AddClip,
    AddMarkers,
    AddTrack,
    Command,
    RemoveMarkers,
    UpdateMarker,
)
from sashimono.core.model import Clip, Marker, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.theme import Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.markers import DELETE_TEXT, PLACE_TEXT, RENAME_TEXT

_RULER_Y = Metrics.RULER_HEIGHT // 2


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす 開いたまま待つと、試験が人の操作を待って止まる

    名前の入力欄を試す所だけは、試験の中で ``QInputDialog.getText`` を差し替える
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
        (QInputDialog, "getText"),
    ):
        monkeypatch.setattr(owner, name, refuse)


def _project(markers: tuple[Marker, ...] = ()) -> Project:
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create())
    project = AddClip(track.id, Clip(timeline_start=0, duration=300, source=TEXT.create())).apply(
        project
    )
    return project.with_timeline(replace(project.timeline, markers=markers))


@pytest.fixture
def view(qt_application: QApplication) -> Iterator[TimelineView]:
    del qt_application
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    created = TimelineView(_project(), analyzer)
    created.resize(900, 300)
    yield created
    analyzer.close()


def _x(view: TimelineView, frame: int) -> int:
    return round(view._layout.frame_to_x(frame))


def _received(view: TimelineView) -> list[list[Command]]:
    received: list[list[Command]] = []
    view.commands_requested.connect(lambda commands, _label: received.append(commands))
    return received


def _trigger(view: TimelineView, position: QPoint, text: str) -> None:
    menu = view.build_context_menu(position)
    action = next((a for a in menu.actions() if a.text() == text), None)
    assert action is not None, f"右クリックに {text} が無い"
    action.trigger()


def _texts(view: TimelineView, position: QPoint) -> set[str]:
    return {action.text() for action in view.build_context_menu(position).actions()}


class TestPaint:
    def test_a_marker_shows_on_the_ruler_in_its_colour(self, view: TimelineView) -> None:
        # 描かないと、AI が置いた目印が見えないまま書き出しの目次にだけ効く
        before = view.grab().toImage()
        view.set_project(_project((Marker(60, "本編", "#ff0000"),)))
        after = view.grab().toImage()
        ratio = after.width() / view.width()

        def pixel(image: QImage, x: int, y: int) -> QColor:
            return image.pixelColor(round(x * ratio), round(y * ratio))

        x = _x(view, 60)
        assert pixel(after, x, 2) != pixel(before, x, 2)
        flag = pixel(after, x, 2)
        assert flag.red() > 200 and flag.green() < 80 and flag.blue() < 80
        # 目印の無い所の目盛りは変わらない
        away = _x(view, 200)
        assert pixel(after, away, 2) == pixel(before, away, 2)


class TestRightClick:
    def test_place_on_an_empty_part_of_the_ruler(self, view: TimelineView) -> None:
        # 壊れると、人は AI に頼まないと目印を置けない
        received = _received(view)
        _trigger(view, QPoint(_x(view, 90), _RULER_Y), PLACE_TEXT)
        assert received == [[AddMarkers((Marker(90),))]]

    def test_delete_on_a_marker(self, view: TimelineView) -> None:
        # 壊れると、AI が置き間違えた目印を人が消せず、目次に残る
        view.set_project(_project((Marker(60, "本編"),)))
        received = _received(view)
        _trigger(view, QPoint(_x(view, 60) + 2, _RULER_Y), DELETE_TEXT)
        assert received == [[RemoveMarkers((60,))]]

    def test_rename_on_a_marker(self, view: TimelineView, monkeypatch: pytest.MonkeyPatch) -> None:
        # 壊れると、目次の見出しの誤りを直すのに AI へ頼み直すことになる
        view.set_project(_project((Marker(60, "仮", "#00ff00"),)))
        asked: list[str] = []

        def answer(_parent: object, _title: str, _label: str, text: str = "") -> tuple[str, bool]:
            asked.append(text)
            return ("本編", True)

        monkeypatch.setattr(QInputDialog, "getText", answer)
        received = _received(view)
        _trigger(view, QPoint(_x(view, 60), _RULER_Y), RENAME_TEXT)
        assert asked == ["仮"]
        assert received == [[UpdateMarker(60, Marker(60, "本編", "#00ff00"))]]

    def test_a_marker_offers_no_second_place(self, view: TimelineView) -> None:
        # 目印の上で「置く」を出すと、押しても同じ位置なので断られるだけになる
        view.set_project(_project((Marker(60),)))
        texts = _texts(view, QPoint(_x(view, 60), _RULER_Y))
        assert PLACE_TEXT not in texts
        assert {RENAME_TEXT, DELETE_TEXT} <= texts

    def test_tracks_do_not_get_marker_items(self, view: TimelineView) -> None:
        # クリップの操作の中に目印の項目が毎回混ざると、目当ての項目を探しにくい
        texts = _texts(view, QPoint(_x(view, 90), Metrics.RULER_HEIGHT + 20))
        assert not {PLACE_TEXT, RENAME_TEXT, DELETE_TEXT} & texts
