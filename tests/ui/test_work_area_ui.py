"""タイムラインの書き出し範囲と、書き出しの画面の「全体 / 指定した範囲」（Issue #27）

目盛りの Shift+ドラッグで範囲が決まり、Shift の無い操作は今までどおり再生ヘッドを動かす
決めた範囲は帯で見え、右クリックと編集メニューから解除でき、書き出しの既定になる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from sashimono.core.commands import AddClip, AddTrack, Command, SetWorkArea
from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui import export_dialog
from sashimono.ui.export_dialog import RANGE_ALL, RANGE_WORK_AREA, ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.work_area import CLEAR_TEXT, HINT_TEXT

_LEFT = Qt.MouseButton.LeftButton
_SHIFT = Qt.KeyboardModifier.ShiftModifier
_NONE = Qt.KeyboardModifier.NoModifier
_RULER_Y = Metrics.RULER_HEIGHT // 2


def _project(frames: int = 300) -> Project:
    """``frames`` コマのテキストを 1 本置いた作品"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create())
    return AddClip(track.id, Clip(timeline_start=0, duration=frames, source=TEXT.create())).apply(
        project
    )


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


def _drag_on_ruler(
    view: TimelineView, start: int, end: int, modifier: Qt.KeyboardModifier = _SHIFT
) -> None:
    QTest.mousePress(view, _LEFT, modifier, QPoint(_x(view, start), _RULER_Y))
    QTest.mouseMove(view, QPoint(_x(view, end), _RULER_Y))
    QTest.mouseRelease(view, _LEFT, modifier, QPoint(_x(view, end), _RULER_Y))


def _labels(view: TimelineView, position: QPoint) -> dict[str, bool]:
    menu = view.build_context_menu(position)
    return {action.text(): action.isEnabled() for action in menu.actions() if action.text()}


def _with_area(view: TimelineView, area: tuple[int, int]) -> None:
    view.set_project(SetWorkArea(area).apply(view.project))


class TestRulerDrag:
    def test_shift_drag_on_the_ruler_sets_the_range(self, view: TimelineView) -> None:
        """Shift+ドラッグで、押した所から離した所までが 1 つのコマンドになる

        途中をコマンドにすると、動かした分だけ取り消しの段が積もる
        """
        received = _received(view)
        _drag_on_ruler(view, 30, 90)
        assert received == [[SetWorkArea((30, 90))]]

    def test_dragging_leftwards_gives_the_same_range(self, view: TimelineView) -> None:
        # 始まりと終わりを押した順のまま使うと、左へ引いた範囲はモデルに断られる
        received = _received(view)
        _drag_on_ruler(view, 90, 30)
        assert received == [[SetWorkArea((30, 90))]]

    def test_shift_drag_does_not_move_the_playhead(self, view: TimelineView) -> None:
        """範囲を引いている間に再生ヘッドが付いて来ると、見ていた場所を失う"""
        view.set_playhead(200)
        _drag_on_ruler(view, 30, 90)
        assert view.playhead == 200

    def test_a_plain_drag_still_moves_the_playhead(self, view: TimelineView) -> None:
        """Shift の無い目盛りの操作は今までどおり再生ヘッドを動かし、範囲は触らない"""
        received = _received(view)
        _drag_on_ruler(view, 30, 90, _NONE)
        assert view.playhead == 90
        assert received == []

    def test_a_shift_click_keeps_the_range(self, view: TimelineView) -> None:
        """幅の無い Shift+クリックで範囲を消さない クリックのつもりで決めた範囲が消える"""
        _with_area(view, (30, 90))
        received = _received(view)
        QTest.mouseClick(view, _LEFT, _SHIFT, QPoint(_x(view, 150), _RULER_Y))
        assert received == []

    def test_the_range_edge_snaps_to_the_nearest_frame_boundary(self, view: TimelineView) -> None:
        """縁は近い方の境目へ寄せる 切り捨てると、右へ引いたとき最後のコマがこぼれる"""
        received = _received(view)
        # 1 コマ 2 画素 境目の 1 画素右（コマの中ほど）で離す
        QTest.mousePress(view, _LEFT, _SHIFT, QPoint(_x(view, 10), _RULER_Y))
        QTest.mouseMove(view, QPoint(_x(view, 40) - 1, _RULER_Y))
        QTest.mouseRelease(view, _LEFT, _SHIFT, QPoint(_x(view, 40) - 1, _RULER_Y))
        assert received == [[SetWorkArea((10, 40))]]


class TestLook:
    def test_the_range_is_drawn_on_the_ruler_and_the_tracks(self, view: TimelineView) -> None:
        """範囲の中だけ目盛りに帯が出て、トラックにも薄く色が乗る

        帯が出ないと、書き出しが範囲だけになる理由が画面から読めない
        """
        before = view.grab().toImage()
        _with_area(view, (30, 90))
        after = view.grab().toImage()
        # 高 DPI の画面では grab の画素が論理座標より多い 同じ割合で読み替える
        ratio = after.width() / view.width()

        def pixel(image: QImage, x: int, y: int) -> QColor:
            return image.pixelColor(round(x * ratio), round(y * ratio))

        band_y = Metrics.RULER_HEIGHT - 3
        inside, outside = _x(view, 60), _x(view, 150)
        assert pixel(after, inside, band_y) != pixel(before, inside, band_y)
        assert pixel(after, outside, band_y) == pixel(before, outside, band_y)
        # トラックの空いた所（テキストの下の段） 範囲の中だけ色が変わる
        track_y = view.height() - 10
        assert pixel(after, inside, track_y) != pixel(before, inside, track_y)
        assert pixel(after, outside, track_y) == pixel(before, outside, track_y)
        # 帯は黄緑 再生ヘッド（赤）と見分けが付かないと、帯を再生ヘッドの影と読む
        band = pixel(after, inside, band_y)
        assert band.green() > band.red()
        assert band.green() > Colors.TIMELINE_RULER.green()


class TestMenus:
    def test_the_ruler_offers_to_clear_the_range(self, view: TimelineView) -> None:
        # 目盛りの右クリックに無いと、帯を見ている所から範囲を消す手が無くなる
        _with_area(view, (30, 90))
        received = _received(view)
        menu = view.build_context_menu(QPoint(_x(view, 150), _RULER_Y))
        clear = next(action for action in menu.actions() if action.text() == CLEAR_TEXT)
        clear.trigger()
        assert received == [[SetWorkArea(None)]]

    def test_inside_the_range_on_a_track_the_clear_is_offered(self, view: TimelineView) -> None:
        # 薄い色の上で右クリックしても出ないと、色の付いた所が何なのかを消して確かめられない
        _with_area(view, (30, 90))
        assert CLEAR_TEXT in _labels(view, QPoint(_x(view, 60), view.height() - 10))

    def test_outside_the_range_on_a_track_it_is_not(self, view: TimelineView) -> None:
        # クリップの操作の中に毎回関係の無い項目が混ざると、目当ての項目を探しにくい
        _with_area(view, (30, 90))
        assert CLEAR_TEXT not in _labels(view, QPoint(_x(view, 150), view.height() - 10))

    def test_without_a_range_the_ruler_tells_how_to_make_one(self, view: TimelineView) -> None:
        """範囲が無ければ作り方を案内する Shift+ドラッグは見ただけでは分からない"""
        labels = _labels(view, QPoint(_x(view, 150), _RULER_Y))
        assert labels.get(HINT_TEXT) is False
        assert CLEAR_TEXT not in labels

    def test_clearing_without_a_range_does_nothing(self, view: TimelineView) -> None:
        # 無い範囲を消すコマンドを出すと、何も変わらない段が取り消しの履歴に積もる
        received = _received(view)
        assert not view.clear_work_area()
        assert received == []


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    created.resize(1200, 800)
    yield created
    created.close()


class TestWindow:
    def test_the_edit_menu_clears_the_range_and_undo_brings_it_back(
        self, window: MainWindow
    ) -> None:
        # 取り消しで戻らないと、誤って解除した範囲を引き直すことになる
        window.execute(SetWorkArea((30, 90)))
        action, _ = window._actions["編集/書き出し範囲を解除"]
        action.trigger()
        assert window.document.project.timeline.work_area is None
        window.undo()
        assert window.document.project.timeline.work_area == (30, 90)

    def test_in_a_scene_the_range_is_the_scenes_and_export_uses_the_main(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """シーンを開いて引いた範囲はシーンの物 書き出しはメインの範囲を使い、そう断る

        シーンの範囲をメインへ書くと、シーンで引いた帯の時刻でメインを切って書き出す
        """
        window.execute(SetWorkArea((0, 60)))
        scene_id = window.create_scene("OP")
        assert scene_id is not None
        window.execute(
            AddTrack(Track(TrackKind.VIDEO, "V1", (Clip(0, 150, source=TEXT.create()),)))
        )
        window._timeline.commands_requested.emit([SetWorkArea((10, 20))], "書き出し範囲を指定")
        project = window.document.project
        assert project.timeline.work_area == (0, 60)
        assert project.require_scene(scene_id).timeline.work_area == (10, 20)

        seen: list[ExportDialog] = []

        def capture(dialog: ExportDialog) -> int:
            seen.append(dialog)
            return 0

        monkeypatch.setattr(ExportDialog, "exec", capture)
        window.export()
        # _settings() はコーデックが 1 つも開けない機械で None になる 範囲とは関係が無いので、
        # 画面が選んだ範囲を直に見る
        assert seen[0]._frame_range() == (0, 60)
        notes = [label.text() for label in seen[0].findChildren(QLabel)]
        assert any("OP" in note and "メイン" in note for note in notes)


class TestExportDialog:
    @pytest.fixture(autouse=True)
    def one_codec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """開けるコーデックが 1 つある機械にそろえる

        1 つも開けない機械では ``_settings()`` が ``None`` を返し、範囲と関係の無い理由で
        落ちる 範囲が設定まで届くかを見たいので、コーデックの有無はここで決めてしまう
        """
        monkeypatch.setattr(export_dialog, "available_video_codecs", lambda _priority: ["libx264"])

    def test_with_a_range_the_range_is_the_default(self) -> None:
        """範囲があれば既定は範囲 帯を引いた人は、その所を出したくて引いている"""
        dialog = ExportDialog(SetWorkArea((30, 90)).apply(_project()))
        try:
            assert dialog._range.currentData() == RANGE_WORK_AREA
            settings = dialog._settings()
            assert settings is not None
            assert settings.frame_range == (30, 90)
        finally:
            dialog.deleteLater()

    def test_choosing_the_whole_exports_everything(self) -> None:
        # 範囲を決めたまま忘れていた人が、全体へ戻せる
        dialog = ExportDialog(SetWorkArea((30, 90)).apply(_project()))
        try:
            dialog._range.setCurrentIndex(dialog._range.findData(RANGE_ALL))
            settings = dialog._settings()
            assert settings is not None
            assert settings.frame_range is None
        finally:
            dialog.deleteLater()

    def test_without_a_range_the_whole_is_exported(self) -> None:
        # 範囲が無いのに範囲の選びが出ると、選んでも何も書き出せない項目が並ぶ
        dialog = ExportDialog(_project())
        try:
            assert dialog._range.count() == 1
            settings = dialog._settings()
            assert settings is not None
            assert settings.frame_range is None
        finally:
            dialog.deleteLater()

    def test_a_range_past_the_end_is_cut_to_the_end(self) -> None:
        """終わりより後ろへはみ出した分は書き出さない 出すと黒と無音が後ろに付く"""
        dialog = ExportDialog(SetWorkArea((200, 900)).apply(_project(300)))
        try:
            settings = dialog._settings()
            assert settings is not None
            assert settings.frame_range == (200, 300)
        finally:
            dialog.deleteLater()

    def test_a_range_wholly_past_the_end_cannot_be_chosen(self) -> None:
        # 選べると、何も映らない所だけを書き出す
        project = _project(300)
        project = project.with_timeline(replace(project.timeline, work_area=(400, 500)))
        dialog = ExportDialog(project)
        try:
            assert dialog._range.currentData() == RANGE_ALL
            settings = dialog._settings()
            assert settings is not None
            assert settings.frame_range is None
        finally:
            dialog.deleteLater()
