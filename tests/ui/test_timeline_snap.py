"""タイムラインの磁石（吸着 利用者の要望）

クリップを動かす・端を伸び縮みさせる・置くときに、ほかのクリップの頭と終わり、再生位置、
キーフレームのコマと書き出し範囲の端へ吸い付く 吸い付く距離は画面の画素（既定 8）
動かしている途中で Shift を押している間は吸い付かない ツールバーの〔磁石〕と設定で切れる
窓は表示しない（オフスクリーン）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDragMoveEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import Command
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.media_pool import MEDIA_MIME
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.scene_bar import SceneBar
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.snap import nearest_snap, snap_targets
from sashimono.ui.timeline.view import DragKind
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.conftest import release_modifier_keys

_LEFT = Qt.MouseButton.LeftButton
type Made = tuple[list[TimelineView], MediaAnalyzer]


def _text(start: int, duration: int = 60, opacity: AnimatedValue | None = None) -> Clip:
    clip = Clip(timeline_start=start, duration=duration, source=TEXT.create())
    return clip if opacity is None else replace(clip, opacity=opacity)


def _project(*layers: tuple[Clip, ...], media: tuple[MediaItem, ...] = ()) -> Project:
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED), media=media)
    tracks = tuple(
        Track(TrackKind.MIXED, f"レイヤー {n + 1}", clips) for n, clips in enumerate(layers)
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def made(qt_application: QApplication) -> Iterator[Made]:
    del qt_application
    views: list[TimelineView] = []
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    yield views, analyzer
    for view in views:
        view.deleteLater()
    analyzer.close()


def _open(made: Made, project: Project) -> tuple[TimelineView, list[list[Command]]]:
    views, analyzer = made
    view = TimelineView(project, analyzer)
    view.resize(1000, 300)
    # 1 フレーム 2 画素 吸い付く距離 8 画素は 4 フレーム
    view._layout = TimelineLayout(pixels_per_frame=2.0)
    received: list[list[Command]] = []

    def apply(commands: list[Command], _label: str) -> None:
        received.append(list(commands))
        updated = view.project
        for command in commands:
            updated = command.apply(updated)
        view.set_project(updated)

    view.commands_requested.connect(apply)
    views.append(view)
    return view, received


def _point(view: TimelineView, layer: int, frame: int) -> QPoint:
    band = view.view_layout.bands(view.project.timeline)[layer]
    return QPoint(int(view.view_layout.frame_to_x(frame)), band.top + band.height // 2)


def _drag(
    view: TimelineView,
    start: QPoint,
    end: QPoint,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    QTest.mousePress(view, _LEFT, Qt.KeyboardModifier.NoModifier, start)
    middle = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
    QTest.mouseMove(view, middle)
    QTest.mouseMove(view, end)
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent

    # 動かしている途中で押しているキー（QTest.mouseMove はキーを渡せない）
    QApplication.sendEvent(
        view,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(end),
            QPointF(end),
            Qt.MouseButton.NoButton,
            _LEFT,
            modifiers,
        ),
    )
    QTest.mouseRelease(view, _LEFT, modifiers, end)


def _start(view: TimelineView, layer: int) -> int:
    return view.project.timeline.tracks[layer].clips[0].timeline_start


class TestTargets:
    def test_edges_playhead_keyframes_and_the_work_area(self) -> None:
        keyed = _text(
            200,
            opacity=AnimatedValue(1.0, keyframes=(Keyframe(0, 0.0), Keyframe(15, 1.0))),
        )
        project = _project((_text(0), keyed), (moving := _text(400),))
        project = project.with_timeline(replace(project.timeline, work_area=(30, 330)))
        targets = snap_targets(project, 123, exclude={moving.id})
        assert {0, 60, 200, 215, 260, 123, 30, 330} <= set(targets)
        # 動かしている物の端には吸い付かない 吸い付くと動かした量が 0 に戻される
        assert 400 not in targets and 460 not in targets

    def test_the_nearest_within_reach_wins(self) -> None:
        assert nearest_snap((62, 122), [0, 60, 125], 4.0) == (-2, 60)
        assert nearest_snap((70,), [0, 60, 125], 4.0) is None

    @pytest.mark.parametrize("kind", [DragKind.MOVE_CLIP, DragKind.TRIM_HEAD])
    def test_a_locked_selection_stays_a_target(self, made: Made, kind: DragKind) -> None:
        # 選んでいてもロックで動かない物まで外すと、その物の端へ吸着できない
        locked, moving = _text(100), _text(300)
        project = _project((locked,), (moving,))
        tracks = (replace(project.timeline.tracks[0], locked=True), project.timeline.tracks[1])
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        view, _ = _open(made, project)
        view.set_selection((locked.id, moving.id))
        view._drag.kind = kind
        view._drag.clip_id = moving.id
        targets = view._snap_targets()
        assert {100, 160} <= set(targets)
        assert 300 not in targets and 360 not in targets


class TestDragging:
    def test_a_moved_clip_snaps_to_the_end_of_another(self, made: Made) -> None:
        # 頭を 62 へ落とすと、上のレイヤーのクリップの終わり 60 へ吸い付く
        view, _ = _open(made, _project((_text(0),), (_text(200),)))
        _drag(view, _point(view, 1, 220), _point(view, 1, 82))
        assert _start(view, 1) == 60
        assert view.snap_line == 60

    def test_shift_while_dragging_turns_it_off(self, made: Made) -> None:
        view, _ = _open(made, _project((_text(0),), (_text(200),)))
        _drag(
            view,
            _point(view, 1, 220),
            _point(view, 1, 82),
            modifiers=Qt.KeyboardModifier.ShiftModifier,
        )
        assert _start(view, 1) == 62

    def test_a_shift_left_by_an_earlier_test_is_released(self, made: Made) -> None:
        """前の試験が Shift を押したまま終えても、次の試験では吸い付く（#239）

        アプリが覚えている修飾キーは試験をまたいで残る Shift のクリックで終わった試験の
        直後に磁石の試験が来ると、押したままに見えて吸い付かずに落ちた 並列で試験の
        並びが変わって表に出た 試験の間で戻す所（conftest）が効いているかを、
        間に挟む片付けと同じ関数を呼んで確かめる
        """
        view, _ = _open(made, _project((_text(0),), (_text(200),)))
        QTest.mouseClick(view, _LEFT, Qt.KeyboardModifier.ShiftModifier, QPoint(1, 1))
        assert QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier
        release_modifier_keys()
        assert QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier
        view.set_selection(())
        _drag(view, _point(view, 1, 220), _point(view, 1, 82))
        assert _start(view, 1) == 60

    def test_the_toolbar_setting_turns_it_off(self, made: Made) -> None:
        view, _ = _open(made, _project((_text(0),), (_text(200),)))
        view.set_snap(False)
        _drag(view, _point(view, 1, 220), _point(view, 1, 82))
        assert _start(view, 1) == 62

    def test_the_end_snaps_to_the_playhead(self, made: Made) -> None:
        # 終わりの端を伸ばして、再生位置 150 の近く（148）で離す
        view, _ = _open(made, _project((_text(0),), (_text(200),)))
        view.set_playhead(150, follow=False)
        edge = QPoint(int(view.view_layout.frame_to_x(60)) - 1, _point(view, 0, 30).y())
        _drag(view, edge, _point(view, 0, 148))
        assert view.project.timeline.tracks[0].clips[0].timeline_end == 150

    def test_a_keyframe_attracts_too(self, made: Made) -> None:
        keyed = _text(0, duration=120, opacity=AnimatedValue(1.0, keyframes=(Keyframe(90, 0.5),)))
        view, _ = _open(made, _project((keyed,), (_text(300),)))
        _drag(view, _point(view, 1, 320), _point(view, 1, 112))
        assert _start(view, 1) == 90


def test_dropping_media_snaps_the_start(made: Made, video_media: MediaItem) -> None:
    # 素材を引いてきて置くときも、置く頭が近くの端へ吸い付く
    target = _text(0)
    view, _ = _open(made, _project((target,), (), media=(video_media,)))
    view.set_selection((target.id,))
    mime = QMimeData()
    mime.setData(MEDIA_MIME, str(video_media.id).encode("utf-8"))
    view.dragMoveEvent(
        QDragMoveEvent(
            _point(view, 1, 62),
            Qt.DropAction.CopyAction,
            mime,
            _LEFT,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    guide = view.drop_guide
    assert guide is not None and guide.spot.frame == 60


def test_the_setting_is_kept_and_shown(qt_application: QApplication, tmp_path: Path) -> None:
    # 既定は入 切った状態と距離は次に開いたときも残す
    del qt_application
    assert Preferences().timeline_snap
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(timeline_snap=False, snap_distance=16))
    loaded = store.load()
    assert (loaded.timeline_snap, loaded.snap_distance) == (False, 16)
    dialog = PreferencesDialog(loaded)
    try:
        assert dialog.preferences().snap_distance == 16
        assert not dialog.preferences().timeline_snap
    finally:
        dialog.deleteLater()
    bar = SceneBar()
    try:
        seen: list[bool] = []
        bar.snap_toggled.connect(seen.append)
        bar.snap_button.click()
        assert seen == [False]
        bar.set_snap(True)
        assert seen == [False], "窓が合わせただけで知らせると、設定を 2 度書く"
    finally:
        bar.deleteLater()
