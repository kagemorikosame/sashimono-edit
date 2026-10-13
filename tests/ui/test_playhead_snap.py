"""目盛りで動かす再生ヘッドの吸い付き（Issue #278 #313）

入り切りはタイムラインの上の〔磁石〕の隣のボタン 既定は入で、目盛りをドラッグすると何も
押さずに近くのクリップの端・キーフレーム・書き出し範囲の端へ吸い付き、掴んだ後に Shift を
押している間は外す（クリップの磁石と同じ向き） 〔磁石〕とは互いに影響しない
吸い付く先は設定で変えられる 窓は表示しない（オフスクリーン）
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.core.commands import Command, SetWorkArea
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
from sashimono.resources import BUNDLED_FILES, PLAYHEAD_SNAP_ICONS, path_to
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.recovery_dialog import RecoveryDialog
from sashimono.ui.scene_bar import SceneBar
from sashimono.ui.theme import Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.snap import PlayheadSnap, snap_targets
from sashimono.ui.workspace import Preferences, PreferenceStore

_LEFT = Qt.MouseButton.LeftButton
_SHIFT = Qt.KeyboardModifier.ShiftModifier
_NONE = Qt.KeyboardModifier.NoModifier
type Made = tuple[list[TimelineView], MediaAnalyzer]


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """尋ねる窓・知らせる窓・設定の窓が開いたらその場で落とす

    開いたまま待つと、試験は誰も押さないボタンを待って止まる 設定の窓を通す試験は、
    この後で自分の答えに差し替える
    """

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)
    monkeypatch.setattr(RecoveryDialog, "exec", popped)
    monkeypatch.setattr(PreferencesDialog, "exec", popped)


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
    """目盛りを ``start`` で掴み、``end`` まで動かして離す 離す前の再生位置を返す

    ``modifiers`` は掴んだ後の動きで押しているキー（掴む押下には付けない）
    """
    QTest.mousePress(view, _LEFT, _NONE, _ruler(view, start))
    _move(view, _ruler(view, end), modifiers)
    frame = view.playhead
    QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, end))
    return frame


def _received(view: TimelineView) -> list[list[Command]]:
    received: list[list[Command]] = []
    view.commands_requested.connect(lambda commands, _label: received.append(commands))
    return received


def _shift_drag_on_ruler(view: TimelineView, start: int, end: int) -> None:
    """Shift を押してから目盛りを押し、そのまま動かして離す（書き出し範囲の指定）"""
    QTest.mousePress(view, _LEFT, _SHIFT, _ruler(view, start))
    _move(view, _ruler(view, end), _SHIFT)
    QTest.mouseRelease(view, _LEFT, _SHIFT, _ruler(view, end))


class TestOn:
    """既定（ボタンが入）"""

    def test_dragging_snaps_to_the_end_of_a_clip(self, made: Made) -> None:
        # 吸い付かないと、知らない人はクリップの終わりへ合わせるのに拡大して 1 コマずつ寄せる
        view = _open(made, _project(_text(0), _text(200)))
        assert _scrub(view, 10, 62) == 60
        assert view.snap_line == 60

    def test_shift_after_grabbing_lets_it_go(self, made: Made) -> None:
        # 外せないと、クリップの端のすぐ隣のコマへ再生ヘッドを置けない
        view = _open(made, _project(_text(0), _text(200)))
        assert _scrub(view, 10, 62, _SHIFT) == 62

    def test_releasing_shift_snaps_again(self, made: Made) -> None:
        # 離しても外れたままだと、一度 Shift を押したドラッグでは端へ合わせられない
        view = _open(made, _project(_text(0), _text(200)))
        QTest.mousePress(view, _LEFT, _NONE, _ruler(view, 10))
        _move(view, _ruler(view, 62), _SHIFT)
        assert view.playhead == 62
        _move(view, _ruler(view, 63), _NONE)
        assert view.playhead == 60
        QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, 63))

    def test_shift_before_pressing_the_ruler_still_sets_the_range(self, made: Made) -> None:
        # 吸い付きを入れても、Shift を押してから目盛りを押す操作は書き出し範囲のまま
        # 再生ヘッドの操作に変わると、範囲を決められなくなり見ていた所も失う
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead(150, follow=False)
        received = _received(view)
        _shift_drag_on_ruler(view, 30, 90)
        assert received == [[SetWorkArea((30, 90))]]
        assert view.playhead == 150

    def test_it_does_not_snap_to_its_own_position(self, made: Made) -> None:
        # 再生位置そのものを先に入れると、動かし始めた所へ引き戻されて離れられない
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead(100, follow=False)
        assert 100 not in view._playhead_targets()
        assert _scrub(view, 100, 102) == 102

    def test_it_can_come_back_to_the_edge_it_started_on(self, made: Made) -> None:
        # 掴んだ所がクリップの終わりでも、そこは吸い付く先のまま残す 外すと、行き過ぎて
        # 戻したいときにドラッグの間ずっとその端へ合わせられない（Premiere Pro も戻せる）
        view = _open(made, _project(_text(0), _text(200)))
        QTest.mousePress(view, _LEFT, _NONE, _ruler(view, 60))
        assert view.playhead == 60
        _move(view, _ruler(view, 80), _NONE)
        assert view.playhead == 80
        _move(view, _ruler(view, 62), _NONE)
        assert view.playhead == 60
        QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, 62))

    def test_nothing_near_means_no_snap(self, made: Made) -> None:
        # 6 フレームは 12 画素 吸い付く距離（8 画素）の外 外でも引き寄せると細かく動かせない
        view = _open(made, _project(_text(0), _text(200)))
        assert _scrub(view, 10, 66) == 66
        assert view.snap_line is None

    def test_the_reach_is_counted_in_screen_pixels(self, made: Made) -> None:
        # 拡大して 1 フレーム 4 画素にすると、8 画素は 2 フレーム 拡大しても吸い付く
        # 強さ（指の感覚）が変わらない
        view = _open(made, _project(_text(0), _text(200)), pixels_per_frame=4.0)
        assert _scrub(view, 10, 63) == 63
        assert _scrub(view, 10, 62) == 60

    def test_it_works_with_the_magnet_button_off(self, made: Made) -> None:
        # 〔磁石〕を切ったら再生ヘッドまで止まると、押していないボタンが効いたように見える
        view = _open(made, _project(_text(0), _text(200)))
        view.set_snap(False)
        assert _scrub(view, 10, 62) == 60

    def test_a_shift_the_app_remembers_does_not_let_it_go(
        self, made: Made, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 日本語入力や別の窓で押した Shift をアプリが覚えていても、マウスの知らせに
        # 載っていなければ吸い付く 押していないのに外れると、壊れたように見える
        # 覚えを Qt の入力で作ると、試験が配る押下（キー無し）で覚えが消えてしまい、
        # アプリの覚えを見る作りに戻しても落ちない なので覚えの方を差し替える
        view = _open(made, _project(_text(0), _text(200)))
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: _SHIFT))
            assert QApplication.keyboardModifiers() & _SHIFT
            assert _scrub(view, 10, 62) == 60

    def test_a_shift_from_the_hover_before_the_press_does_not_let_it_go(self, made: Made) -> None:
        # 掴む前のマウスの動きで覚えた Shift を押下に持ち越すと、離したはずの Shift で
        # 押した所で吸い付かない
        view = _open(made, _project(_text(0), _text(200)))
        hover = _ruler(view, 30)
        QApplication.sendEvent(
            view,
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(hover),
                QPointF(hover),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                _SHIFT,
            ),
        )
        QTest.mousePress(view, _LEFT, _NONE, _ruler(view, 62))
        assert view.playhead == 60
        QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, 62))

    def test_pressing_an_empty_track_area_does_not_snap(self, made: Made) -> None:
        # 空いた所の押下は囲んで選ぶ操作の始まり 押した所から再生ヘッドがずれると戸惑う
        view = _open(made, _project(_text(0), _text(200)))
        band = view.view_layout.bands(view.project.timeline)[0]
        empty = QPoint(round(view.view_layout.frame_to_x(62)), band.top + band.height // 2)
        QTest.mousePress(view, _LEFT, _NONE, empty)
        assert view.playhead == 62
        QTest.mouseRelease(view, _LEFT, _NONE, empty)

    def test_edge_scrolling_keeps_the_snap(self, made: Made) -> None:
        # 端で表示を送ったときも、進め直した位置で吸い付く 送る前の位置に取り残さない
        view = _open(made, _project(_text(0), _text(200)))
        QTest.mousePress(view, _LEFT, _NONE, _ruler(view, 10))
        _move(view, _ruler(view, 49), _NONE)
        # 表示を 10 フレームぶん（20 画素）右へ送る 同じマウスの位置は 59 になり 60 へ吸い付く
        view._on_edge_scroll(20.0, 0.0, _ruler(view, 49))
        assert view.playhead == 60
        assert view.snap_line == 60
        QTest.mouseRelease(view, _LEFT, _NONE, _ruler(view, 50))


class TestOff:
    def test_off_never_snaps(self, made: Made) -> None:
        # 切ったのに吸い付くと、1 コマずつ合わせたい人がクリップの端の近くで合わせられない
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(enabled=False))
        assert _scrub(view, 10, 62) == 62
        assert _scrub(view, 10, 62, _SHIFT) == 62
        assert view.snap_line is None

    def test_the_magnet_button_does_not_turn_it_back_on(self, made: Made) -> None:
        # 〔磁石〕が入でも、再生ヘッドのボタンを切ったら吸い付かない（互いに影響しない）
        view = _open(made, _project(_text(0), _text(200)))
        view.set_snap(True)
        view.set_playhead_snap(PlayheadSnap(enabled=False))
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
        # 入れても吸い付かないと、目印へ合わせるのに拡大して寄せることになる
        project = _project(_text(0))
        project = project.with_timeline(replace(project.timeline, markers=(Marker(150),)))
        view = _open(made, project)
        assert _scrub(view, 100, 152) == 152
        view.set_playhead_snap(PlayheadSnap(markers=True))
        assert _scrub(view, 100, 152) == 150

    def test_clip_edges_can_be_left_out(self, made: Made) -> None:
        # 切ったのに吸い付くと、設定がある方が質が悪い
        view = _open(made, _project(_text(0), _text(200)))
        view.set_playhead_snap(PlayheadSnap(clip_edges=False))
        assert _scrub(view, 10, 62) == 62

    def test_keyframes_can_be_left_out(self, made: Made) -> None:
        # 切ったのにキーフレームへ吸い付くと、キーの隣のコマへ置けない
        keyed = _text(0, duration=120, opacity=AnimatedValue(1.0, keyframes=(Keyframe(90, 0.5),)))
        view = _open(made, _project(keyed))
        assert _scrub(view, 10, 92) == 90
        view.set_playhead_snap(PlayheadSnap(keyframes=False))
        assert _scrub(view, 10, 92) == 92

    def test_the_work_area_can_be_left_out(self, made: Made) -> None:
        # 切ったのに範囲の端へ吸い付くと、範囲の外の隣のコマへ置けない
        project = _project(_text(0))
        project = project.with_timeline(replace(project.timeline, work_area=(150, 300)))
        view = _open(made, project)
        assert _scrub(view, 100, 152) == 150
        view.set_playhead_snap(PlayheadSnap(work_area=False))
        assert _scrub(view, 100, 152) == 152

    def test_the_clip_magnet_keeps_the_playhead_as_a_target(self) -> None:
        # 再生ヘッドの吸い付きを足しても、クリップを動かすときは今までどおり再生位置へ吸い付く
        assert 123 in snap_targets(_project(_text(0)), 123)


class TestPreferences:
    def test_the_default_is_on_with_the_clip_magnet_targets(self) -> None:
        # 既定が切だと、知らない人は吸い付くことに気付かない（利用者の決定は「磁石のように」入）
        plain = Preferences()
        assert plain.playhead_snap is True
        assert PlayheadSnap.from_preferences(plain) == PlayheadSnap()
        assert PlayheadSnap().enabled

    @pytest.mark.parametrize(
        ("preferences", "expected"),
        [
            (Preferences(playhead_snap=False), PlayheadSnap(enabled=False)),
            (Preferences(playhead_snap_clips=False), PlayheadSnap(clip_edges=False)),
            (Preferences(playhead_snap_keyframes=False), PlayheadSnap(keyframes=False)),
            (Preferences(playhead_snap_work_area=False), PlayheadSnap(work_area=False)),
            (Preferences(playhead_snap_markers=True), PlayheadSnap(markers=True)),
        ],
    )
    def test_each_setting_reaches_the_timeline(
        self, preferences: Preferences, expected: PlayheadSnap
    ) -> None:
        # 1 つだけ既定と逆にして、その旗だけが逆になることを見る 取り違えると、
        # 設定でキーフレームを切ったのにクリップの端が切れる
        assert PlayheadSnap.from_preferences(preferences) == expected

    def test_it_is_kept_and_shown(self, qt_application: QApplication, tmp_path: Path) -> None:
        # 残らないと、次に開いたときに切ったはずの吸い付きが戻っている
        del qt_application
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(
            playhead_snap=False,
            playhead_snap_keyframes=False,
            playhead_snap_markers=True,
        )
        store.save(chosen)
        loaded = store.load()
        assert loaded == chosen
        dialog = PreferencesDialog(loaded)
        try:
            assert not dialog._playhead_snap.isChecked()
            assert dialog.preferences() == chosen
        finally:
            dialog.deleteLater()

    @pytest.mark.parametrize(
        ("saved", "expected"),
        [
            # 前の版（0.2.1 まで）の 3 択 常には入、吸い付かないは切、Shift の間だけは既定（入）
            ("always", True),
            ("off", False),
            ("shift", True),
            # 今の版の入切
            (True, True),
            (False, False),
            # 壊れた値は既定へ
            ("magnet", True),
            (3, True),
            (None, True),
        ],
    )
    def test_an_older_value_is_read_as_on_or_off(
        self, tmp_path: Path, saved: object, expected: bool
    ) -> None:
        # 読み替えないと、前の版で「吸い付かない」にしていた人のところで黙って吸い付き始める
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"playhead_snap": saved}), encoding="utf-8")
        assert PreferenceStore(path).load().playhead_snap is expected

    def test_a_broken_target_falls_back_to_the_default(self, tmp_path: Path) -> None:
        # 壊れた値で吸い付く先が消えると、設定を触っていないのに吸い付かなくなる
        path = tmp_path / "preferences.json"
        path.write_text('{"playhead_snap_clips": 3}', encoding="utf-8")
        assert PreferenceStore(path).load().playhead_snap_clips

    def test_off_disables_the_target_boxes(self, qt_application: QApplication) -> None:
        # 吸い付かないのに先を選べると、選んでも何も変わらない欄が残る
        del qt_application
        dialog = PreferencesDialog(Preferences(playhead_snap=False))
        try:
            assert not dialog._playhead_snap_markers.isEnabled()
            dialog._playhead_snap.setChecked(True)
            assert dialog._playhead_snap_markers.isEnabled()
        finally:
            dialog.deleteLater()


class TestButton:
    def test_it_sits_next_to_the_magnet_and_says_what_it_is(
        self, qt_application: QApplication
    ) -> None:
        # 離れた所に置くと、〔磁石〕で切れると思った人が見つけられない 補足が無いと、
        # 磁石と何が違うのか、Shift をいつ押すのかが分からない
        del qt_application
        bar = SceneBar()
        try:
            layout = bar.layout()
            assert layout is not None
            magnet = layout.indexOf(bar.snap_button)
            assert layout.indexOf(bar.playhead_snap_button) == magnet + 1
            button = bar.playhead_snap_button
            assert button.isCheckable() and button.isChecked()
            assert button.text() == ""
            tip = button.toolTip()
            assert tip.startswith("再生ヘッドの吸い付き（目盛りで動かすとき")
            assert "Shift" in tip and "入" in tip
            on = button.icon().pixmap(18, 18).toImage()
            button.click()
            assert "切" in button.toolTip()
            off = button.icon().pixmap(18, 18).toImage()
            assert on != off, "入と切で同じ印だと、どちらなのか分からない"
            magnet_on = bar.snap_button.icon().pixmap(18, 18).toImage()
            button.click()
            assert button.icon().pixmap(18, 18).toImage() != magnet_on, (
                "〔磁石〕と同じ印だと、どちらを押したのか分からない"
            )
        finally:
            bar.deleteLater()

    def test_matching_the_window_does_not_signal(self, qt_application: QApplication) -> None:
        # 窓が合わせただけで知らせると、設定画面で変えたときに設定を 2 度書く
        del qt_application
        bar = SceneBar()
        try:
            seen: list[bool] = []
            bar.playhead_snap_toggled.connect(seen.append)
            bar.playhead_snap_button.click()
            assert seen == [False]
            bar.set_playhead_snap(True)
            assert seen == [False]
            assert bar.playhead_snap_button.isChecked()
        finally:
            bar.deleteLater()

    def test_the_icons_are_bundled(self) -> None:
        # 配る版に積み忘れると印が黙って消え、ボタンが空の四角になる
        assert set(PLAYHEAD_SNAP_ICONS) <= set(BUNDLED_FILES)
        assert all(path_to(name).is_file() for name in PLAYHEAD_SNAP_ICONS)


def _window() -> MainWindow:
    window = MainWindow(_project(_text(0), _text(200)), confirm_unsaved=False)
    window.resize(1200, 800)
    # 窓を出さないので、タイムラインは並べ直されず小さいまま 押す所が幅の外へ出ないよう広げる
    window._timeline.resize(1000, 300)
    # 試験の中で吸い付く距離をフレームで数えられるよう、1 フレーム 2 画素にそろえる
    window._timeline._layout = TimelineLayout(pixels_per_frame=2.0)
    return window


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _window()
    yield created
    created.close()


class TestWindow:
    """実際に窓を組み、タイムラインの上のボタンを押して確かめる"""

    def test_the_button_turns_it_on_and_off(self, window: MainWindow) -> None:
        # ボタンを押しても目盛りの動きが変わらないと、ボタンがある意味が無い
        button = window._scene_bar.playhead_snap_button
        timeline = window._timeline
        assert button.isChecked()
        assert _scrub(timeline, 10, 62) == 60
        button.click()
        assert not button.isChecked()
        assert _scrub(timeline, 10, 62) == 62
        button.click()
        assert _scrub(timeline, 10, 62) == 60
        assert _scrub(timeline, 10, 62, _SHIFT) == 62

    def test_shift_before_pressing_sets_the_range_in_the_window(self, window: MainWindow) -> None:
        # 吸い付きが入でも、Shift を押してから目盛りを押すと書き出し範囲が決まる
        timeline = window._timeline
        timeline.set_playhead(150, follow=False)
        _shift_drag_on_ruler(timeline, 30, 90)
        assert window.document.project.timeline.work_area == (30, 90)
        assert timeline.playhead == 150

    def test_the_two_buttons_do_not_touch_each_other(self, window: MainWindow) -> None:
        # 片方を押したらもう片方まで変わると、押していないボタンが効いたように見える
        magnet = window._scene_bar.snap_button
        playhead = window._scene_bar.playhead_snap_button
        timeline = window._timeline
        magnet.click()
        assert not magnet.isChecked() and playhead.isChecked()
        assert not timeline.snap_enabled
        assert _scrub(timeline, 10, 62) == 60
        magnet.click()
        playhead.click()
        assert magnet.isChecked() and not playhead.isChecked()
        assert timeline.snap_enabled
        assert _scrub(timeline, 10, 62) == 62

    def test_the_button_is_kept_for_the_next_time(self, qt_application: QApplication) -> None:
        # 残らないと、切った人は開くたびに押し直すことになる
        del qt_application
        first = _window()
        try:
            first._scene_bar.playhead_snap_button.click()
        finally:
            first.close()
        assert PreferenceStore().load().playhead_snap is False
        again = _window()
        try:
            assert not again._scene_bar.playhead_snap_button.isChecked()
            assert not again._timeline.playhead_snap.enabled
            assert _scrub(again._timeline, 10, 62) == 62
        finally:
            again.close()

    def test_the_button_keeps_the_chosen_targets(self, qt_application: QApplication) -> None:
        # ボタンで入れ直したときに吸い付く先が既定へ戻ると、設定で選んだ先が黙って変わる
        del qt_application
        PreferenceStore().save(Preferences(playhead_snap=False, playhead_snap_clips=False))
        created = _window()
        try:
            created._scene_bar.playhead_snap_button.click()
            assert created._timeline.playhead_snap == PlayheadSnap(clip_edges=False)
        finally:
            created.close()

    def test_the_settings_dialog_and_the_button_agree(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 設定画面で切ってもボタンが入のままだと、どちらが本当なのか分からない
        def turn_off(dialog: PreferencesDialog) -> int:
            assert dialog._playhead_snap.isChecked(), "ボタンの入が設定画面に出ていない"
            dialog._playhead_snap.setChecked(False)
            return QDialog.DialogCode.Accepted.value

        monkeypatch.setattr(PreferencesDialog, "exec", turn_off)
        window.edit_preferences()
        assert not window._scene_bar.playhead_snap_button.isChecked()
        assert _scrub(window._timeline, 10, 62) == 62
        assert PreferenceStore().load().playhead_snap is False

        # ボタンで入れ直すと、次に開く設定画面にも入で出る
        window._scene_bar.playhead_snap_button.click()
        shown: list[bool] = []

        def look(dialog: PreferencesDialog) -> int:
            shown.append(dialog._playhead_snap.isChecked())
            return QDialog.DialogCode.Rejected.value

        monkeypatch.setattr(PreferencesDialog, "exec", look)
        window.edit_preferences()
        assert shown == [True]
