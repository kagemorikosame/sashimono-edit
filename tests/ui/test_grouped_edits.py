"""グループ化したクリップどうしで、設定パネルの値が連動しない（実際の部品を押して確かめる）

ee1f11e は「グループの仲間として引き込まれた」クリップを覚えて外したが、利用者の手元では
まだ連動した 2 本を選んでからグループ化すると、どちらも「自分で選んだ」物のまま残り、
その後 1 本を押し直しても選びが変わらない（選んだ中の 1 本を押すと選びを保つ）ので、
設定パネルは 2 本へ値を当て続けた
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.model import AnimatedValue, Clip, ClipId, Project, Track, TrackKind
from sashimono.effects.sources import SHAPE
from sashimono.ui.inspector.widgets import TrackEditor
from sashimono.ui.main_window import MainWindow
from sashimono.ui.timeline import TimelineView

_LEFT = Qt.MouseButton.LeftButton
_CTRL = Qt.KeyboardModifier.ControlModifier


def _shape(start: int) -> Clip:
    clip = Clip(timeline_start=start, duration=30, source=SHAPE.create(shape="rect"))
    return with_fixed_items(clip, picture=True)


def _project() -> Project:
    base = Project.create()
    tracks = (
        Track(TrackKind.VIDEO, "V1", (_shape(0),)),
        Track(TrackKind.VIDEO, "V2", (_shape(0),)),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    created._timeline.resize(900, 400)
    yield created
    created.close()


def _point(view: TimelineView, track: int, frame: int) -> QPoint:
    wanted = view.project.timeline.tracks[track].id
    band = next(b for b in view._layout.bands(view.project.timeline) if b.track.id == wanted)
    return QPoint(int(view._layout.frame_to_x(frame)), band.top + band.height // 2)


def _ids(window: MainWindow) -> tuple[ClipId, ClipId]:
    tracks = window.document.project.timeline.tracks
    return tracks[0].clips[0].id, tracks[1].clips[0].id


def _scale(window: MainWindow, clip_id: ClipId) -> float:
    located = window.document.project.timeline.locate_clip(clip_id)
    assert located is not None
    effect = next(e for e in located[1].effects if e.kind == TRANSFORM_EFFECT_KIND)
    value = effect.params["scale"]
    assert isinstance(value, AnimatedValue)
    return value.static


def _type_scale(window: MainWindow, value: float) -> None:
    """設定パネルの拡大率の数値欄へ打ち込む（利用者の操作と同じ部品を通す）"""
    editor = next(e for (_, name), e in window._inspector._editors.items() if name == "scale")
    assert isinstance(editor, TrackEditor)
    number = editor.findChild(QDoubleSpinBox)
    assert number is not None
    number.setValue(value)


def _select_both_then_group(window: MainWindow) -> tuple[ClipId, ClipId]:
    view = window._timeline
    a, b = _ids(window)
    QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
    QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 1, 10))
    assert view.group_selected()
    return a, b


class TestTypingAValue:
    def test_right_after_grouping(self, window: MainWindow) -> None:
        # 利用者の手順 2 本を選んでグループ化し、そのまま拡大率を変える
        a, b = _select_both_then_group(window)
        primary = window._timeline.selected_clip
        other = b if primary == a else a
        _type_scale(window, 150.0)
        assert _scale(window, primary) == pytest.approx(150.0)  # type: ignore[arg-type]
        assert _scale(window, other) == pytest.approx(100.0)

    def test_after_pressing_one_of_them_again(self, window: MainWindow) -> None:
        # 選んだ中の 1 本を押すと選びを保つ（まとめて動かすため） それで仲間へ当たっていた
        a, b = _select_both_then_group(window)
        view = window._timeline
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
        assert window._timeline.selected_clip == a
        _type_scale(window, 150.0)
        assert _scale(window, a) == pytest.approx(150.0)
        assert _scale(window, b) == pytest.approx(100.0)

    def test_pressing_the_second_member(self, window: MainWindow) -> None:
        a, b = _select_both_then_group(window)
        view = window._timeline
        view.select(None)
        QTest.mouseClick(view, _LEFT, pos=_point(view, 1, 10))
        assert set(view.selected_clips) == {a, b}
        _type_scale(window, 60.0)
        assert _scale(window, b) == pytest.approx(60.0)
        assert _scale(window, a) == pytest.approx(100.0)

    def test_the_preview_pick_changes_only_that_clip(self, window: MainWindow) -> None:
        a, b = _select_both_then_group(window)
        window._preview.clip_picked.emit(str(b))
        _type_scale(window, 70.0)
        assert _scale(window, b) == pytest.approx(70.0)
        assert _scale(window, a) == pytest.approx(100.0)
