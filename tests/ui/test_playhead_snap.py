"""目盛りで動かす再生ヘッドの磁石（Issue #278）

既定では、目盛りをドラッグしている途中で Shift を押している間だけ、近くのクリップの端・
キーフレーム・書き出し範囲の端へ吸い付く（クリップの磁石とは Shift の向きが逆）
吸い付き方（Shift の間だけ・常に・切）と吸い付く先は設定で変えられる
窓は表示しない（オフスクリーン）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    LayerMode,
    Marker,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.theme import Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.snap import PlayheadSnap, snap_targets
from sashimono.ui.workspace import (
    PLAYHEAD_SNAP_ALWAYS,
    PLAYHEAD_SNAP_OFF,
    PLAYHEAD_SNAP_SHIFT,
    Preferences,
    PreferenceStore,
)

_LEFT = Qt.MouseButton.LeftButton
_SHIFT = Qt.KeyboardModifier.ShiftModifier
_NONE = Qt.KeyboardModifier.NoModifier
type Made = tuple[list[TimelineView], MediaAnalyzer]


def _text(start: int, duration: int = 60, opacity: AnimatedValue | None = None) -> Clip:
    clip = Clip(timeline_start=start, duration=duration, source=TEXT.create())
    return clip if opacity is None else replace(clip, opacity=opacity)


def _project(*clips: Clip) -> Project:
    """0〜60 と 200〜260 のような、離れたクリップを 1 本のレイヤーに並べる"""
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    track = Track(TrackKind.MIXED, "レイヤー 1", clips)
    return base.with_timeline(replace(base.timeline, tracks=(track,)))


@pytest.fixture
def made(qt_application: QApplication) -> Iterator[Made]:
    del qt_application
    views: list[TimelineView] = []
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    yield views, analyzer
    for view in views:
        view.deleteLater()
    analyzer.close()


def _open(made: Made, project: Project, pixels_per_frame: float = 2.0) -> TimelineView:
    views, analyzer = made
    view = TimelineView(project, analyzer)
    view.resize(1000, 300)
    # 既定は 1 フレーム 2 画素 吸い付く距離 8 画素は 4 フレーム
    view._layout = TimelineLayout(pixels_per_frame=pixels_per_frame)
    views.append(view)
    return view


def _ruler(view: TimelineView, frame: int) -> QPoint:
    return QPoint(round(view.view_layout.frame_to_x(frame)), Metrics.RULER_HEIGHT // 2)


def _move(view: TimelineView, at: QPoint, modifiers: Qt.KeyboardModifier) -> None:
    """ボタンを押したまま動かす 途中で押しているキーを渡す（QTest.mouseMove は渡せない）"""
    QApplication.sendEvent(
        view,
        QMouseEvent(QEvent.Type.MouseMove, QPointF(at), QPointF(at), _LEFT, _LEFT, modifiers),
    )


def _scrub(view: TimelineView, start: int, end: int, modifiers: Qt.KeyboardModifier = _NONE) -> int:
    """目盛りを ``start`` で掴み、``end`` まで動かして離す 離す前の再生位置を返す"""
    QTest.mousePress(view, _LEFT, _NONE, _ruler(view, start))
    _move(view, _ruler(view, end), modifiers)
    frame = view.playhead
    QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, end))
    return frame


class TestShiftMode:
    """既定（Shift を押している間だけ吸い付く）"""

    def test_shift_while_dragging_snaps_to_the_end_of_a_clip(self, made: Made) -> None:
        # 吸い付かないと、クリップの終わりへ合わせるのに拡大して 1 コマずつ寄せることになる
        view = _open(made, _project(_text(0), _text(200)))
        assert _scrub(view, 10, 62, _SHIFT) == 60
        assert view.snap_line == 60

    def test_without_shift_it_moves_freely(self, made: Made) -> None:
        # 既定で吸い付くと、クリップの端の近くで 1 コマずつ動かせない
        view = _open(made, _project(_text(0), _text(200)))
        assert _scrub(view, 10, 62) == 62
        assert view.snap_line is None

    def test_releasing_shift_goes_back_to_the_plain_move(self, made: Made) -> None:
        view = _open(made, _project(_text(0), _text(200)))
        QTest.mousePress(view, _LEFT, _NONE, _ruler(view, 10))
        _move(view, _ruler(view, 62), _SHIFT)
        assert view.playhead == 60
        _move(view, _ruler(view, 63), _NONE)
        assert view.playhead == 63
        QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, 63))

    def test_it_does_not_snap_to_its_own_position(self, made: Made) -> None:
        # 再生位置そのものを先に入れると、動かし始めた所へ引き戻されて離れられない
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead(100, follow=False)
        assert 100 not in view._playhead_targets()
        assert _scrub(view, 100, 102, _SHIFT) == 102

    def test_nothing_near_means_no_snap(self, made: Made) -> None:
        # 6 フレームは 12 画素 吸い付く距離（8 画素）の外
        view = _open(made, _project(_text(0), _text(200)))
        assert _scrub(view, 10, 66, _SHIFT) == 66

    def test_the_reach_is_counted_in_screen_pixels(self, made: Made) -> None:
        # 拡大して 1 フレーム 4 画素にすると、8 画素は 2 フレーム 拡大しても吸い付く
        # 強さ（指の感覚）が変わらない
        view = _open(made, _project(_text(0), _text(200)), pixels_per_frame=4.0)
        assert _scrub(view, 10, 63, _SHIFT) == 63
        assert _scrub(view, 10, 62, _SHIFT) == 60

    def test_it_works_with_the_magnet_button_off(self, made: Made) -> None:
        # Shift を押すのは、いま吸い付かせたいと本人が言ったのと同じ
        view = _open(made, _project(_text(0), _text(200)))
        view.set_snap(False)
        assert _scrub(view, 10, 62, _SHIFT) == 60

    def test_a_shift_the_app_remembers_does_not_snap(self, made: Made) -> None:
        # 日本語入力や別の窓で押した Shift をアプリが覚えていても、マウスの知らせに
        # 載っていなければ吸い付かない 押していないのに吸い付くと、壊れたように見える
        view = _open(made, _project(_text(0), _text(200)))
        QTest.mouseClick(view, _LEFT, _SHIFT, QPoint(1, 1))
        assert QApplication.keyboardModifiers() & _SHIFT
        assert _scrub(view, 10, 62) == 62

    def test_pressing_an_empty_track_area_does_not_snap(self, made: Made) -> None:
        # 空いた所の押下は囲んで選ぶ操作の始まり 押した所から再生ヘッドがずれると戸惑う
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(mode=PLAYHEAD_SNAP_ALWAYS))
        band = view.view_layout.bands(view.project.timeline)[0]
        empty = QPoint(round(view.view_layout.frame_to_x(62)), band.top + band.height // 2)
        QTest.mousePress(view, _LEFT, _NONE, empty)
        assert view.playhead == 62
        QTest.mouseRelease(view, _LEFT, _NONE, empty)

    def test_edge_scrolling_keeps_the_snap(self, made: Made) -> None:
        # 端で表示を送ったときも、進め直した位置で吸い付く 送る前の位置に取り残さない
        view = _open(made, _project(_text(0), _text(200)))
        QTest.mousePress(view, _LEFT, _NONE, _ruler(view, 10))
        _move(view, _ruler(view, 49), _SHIFT)
        # 表示を 10 フレームぶん（20 画素）右へ送る 同じマウスの位置は 59 になり 60 へ吸い付く
        view._on_edge_scroll(20.0, 0.0, _ruler(view, 49))
        assert view.playhead == 60
        assert view.snap_line == 60
        QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, 50))


class TestModes:
    def test_always_snaps_and_shift_turns_it_off(self, made: Made) -> None:
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(mode=PLAYHEAD_SNAP_ALWAYS))
        assert _scrub(view, 10, 62) == 60
        assert _scrub(view, 10, 62, _SHIFT) == 62

    def test_always_follows_the_magnet_button(self, made: Made) -> None:
        # 〔磁石〕を切っても吸い付くと、ボタンを押した意味が無い
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(mode=PLAYHEAD_SNAP_ALWAYS))
        view.set_snap(False)
        assert _scrub(view, 10, 62) == 62

    def test_off_never_snaps(self, made: Made) -> None:
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(mode=PLAYHEAD_SNAP_OFF))
        assert _scrub(view, 10, 62, _SHIFT) == 62
        assert _scrub(view, 10, 62) == 62


class TestTargets:
    def test_the_default_targets(self) -> None:
        keyed = _text(
            200, opacity=AnimatedValue(1.0, keyframes=(Keyframe(0, 0.0), Keyframe(15, 1.0)))
        )
        project = _project(_text(0), keyed)
        timeline = replace(project.timeline, work_area=(30, 330), markers=(Marker(500),))
        project = project.with_timeline(timeline)
        targets = snap_targets(project, None)
        assert {0, 60, 200, 215, 260, 30, 330} <= set(targets)
        # 目印は既定で入れない（タイムラインに描かないので見えない所で引っ掛かる）
        assert 500 not in targets

    def test_a_marker_attracts_when_turned_on(self, made: Made) -> None:
        project = _project(_text(0))
        project = project.with_timeline(replace(project.timeline, markers=(Marker(150),)))
        view = _open(made, project)
        assert _scrub(view, 100, 152, _SHIFT) == 152
        view.set_playhead_snap(PlayheadSnap(markers=True))
        assert _scrub(view, 100, 152, _SHIFT) == 150

    def test_clip_edges_can_be_left_out(self, made: Made) -> None:
        # 切ったのに吸い付くと、設定がある方が質が悪い
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(clip_edges=False))
        assert _scrub(view, 10, 62, _SHIFT) == 62

    def test_the_clip_magnet_keeps_the_playhead_as_a_target(self) -> None:
        # 再生ヘッドの吸い付きを足しても、クリップを動かすときは今までどおり再生位置へ吸い付く
        assert 123 in snap_targets(_project(_text(0)), 123)


class TestPreferences:
    def test_the_default_is_shift_with_the_clip_magnet_targets(self) -> None:
        plain = Preferences()
        assert plain.playhead_snap == PLAYHEAD_SNAP_SHIFT
        assert PlayheadSnap.from_preferences(plain) == PlayheadSnap()

    def test_it_is_kept_and_shown(self, qt_application: QApplication, tmp_path: Path) -> None:
        del qt_application
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(
            playhead_snap=PLAYHEAD_SNAP_ALWAYS,
            playhead_snap_keyframes=False,
            playhead_snap_markers=True,
        )
        store.save(chosen)
        loaded = store.load()
        assert loaded == chosen
        dialog = PreferencesDialog(loaded)
        try:
            assert dialog.preferences() == chosen
        finally:
            dialog.deleteLater()

    def test_a_broken_value_falls_back_to_the_default(self, tmp_path: Path) -> None:
        path = tmp_path / "preferences.json"
        path.write_text('{"playhead_snap": "magnet", "playhead_snap_clips": 3}', encoding="utf-8")
        loaded = PreferenceStore(path).load()
        assert loaded.playhead_snap == PLAYHEAD_SNAP_SHIFT
        assert loaded.playhead_snap_clips

    def test_off_disables_the_target_boxes(self, qt_application: QApplication) -> None:
        # 吸い付かないのに先を選べると、選んでも何も変わらない欄が残る
        del qt_application
        dialog = PreferencesDialog(Preferences(playhead_snap=PLAYHEAD_SNAP_OFF))
        try:
            assert not dialog._playhead_snap_markers.isEnabled()
            dialog._playhead_snap.setCurrentIndex(
                dialog._playhead_snap.findData(PLAYHEAD_SNAP_SHIFT)
            )
            assert dialog._playhead_snap_markers.isEnabled()
        finally:
            dialog.deleteLater()
