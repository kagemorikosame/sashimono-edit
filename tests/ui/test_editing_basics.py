"""右クリック、コピー・貼り付け、トラックの高さ、2 つの窓

どれも操作の入口 中身の正しさは core 側のテスト（test_clipboard.py など）で見るので、
ここでは「入口から正しいコマンドが出るか」と「窓の組み立てに繋がっているか」を見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from sashimono.core.commands import AddClip, Command, InsertGap, SetTrackHeights
from sashimono.core.io import others_holding, project_presence_dir
from sashimono.core.model import Clip, MediaItem, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_pool import MediaPoolWidget
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.theme import Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.workspace import (
    INSERT_ALL_TRACKS,
    INSERT_TARGET_TRACKS,
    Preferences,
    PreferenceStore,
)
from tests.fake_clipboard import FakeClipboard

CTRL_SHIFT = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier


def _project() -> Project:
    base = Project.create()
    text = Clip(timeline_start=0, duration=30, source=TEXT.create())
    tracks = (Track(TrackKind.VIDEO, "V1", (text,)), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def view(qt_application: QApplication) -> Iterator[TimelineView]:
    del qt_application
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    created = TimelineView(_project(), analyzer)
    created.resize(900, 300)
    yield created
    analyzer.close()


def _received(view: TimelineView) -> list[list[Command]]:
    received: list[list[Command]] = []
    view.commands_requested.connect(lambda commands, _label: received.append(commands))
    return received


def _clip_point(view: TimelineView) -> QPoint:
    band = view._layout.bands(view.project.timeline)[0]
    x = int(view._layout.frame_to_x(10))
    return QPoint(x, band.top + band.height // 2)


def _labels(view: TimelineView, position: QPoint) -> dict[str, bool]:
    menu = view.build_context_menu(position)
    return {action.text(): action.isEnabled() for action in menu.actions() if action.text()}


class TestContextMenu:
    def test_a_clip_offers_the_editing_actions(self, view: TimelineView) -> None:
        # 壊れると、右クリックしても編集の操作が出ず、メニューを探しに行くことになる
        labels = _labels(view, _clip_point(view))
        for expected in ("コピー", "切り取り", "削除", "削除して詰める", "再生ヘッドで分割"):
            assert expected in labels

    def test_right_clicking_a_clip_selects_it(self, view: TimelineView) -> None:
        # 選んでいた別のクリップが対象になると、見ていないものを消す
        view.build_context_menu(_clip_point(view))
        assert view.selected_clip == view.project.timeline.tracks[0].clips[0].id

    def test_paste_is_greyed_out_until_something_is_copied(self, view: TimelineView) -> None:
        # 押せるのに何も起きない項目は、壊れているように見える
        empty = QPoint(800, _clip_point(view).y())
        assert _labels(view, empty)["貼り付け（再生ヘッドの位置）"] is False
        view.select(view.project.timeline.tracks[0].clips[0].id)
        assert view.copy_selected()
        assert _labels(view, empty)["貼り付け（再生ヘッドの位置）"] is True

    def test_the_track_toggles_are_there(self, view: TimelineView) -> None:
        # 壊れると、右クリックからトラックをミュートできない
        labels = _labels(view, _clip_point(view))
        assert {"V1 をミュート", "V1 をソロ", "V1 をロック"} <= labels.keys()


class TestCopyPaste:
    def test_paste_goes_to_the_playhead(self, view: TimelineView) -> None:
        # 壊れると、クリップが再生ヘッドとは違う時刻に置かれる
        received = _received(view)
        view.select(view.project.timeline.tracks[0].clips[0].id)
        view.copy_selected()
        view.set_playhead(90)
        assert view.paste_at_playhead()
        (commands,) = received
        assert [c.clip.timeline_start for c in commands if isinstance(c, AddClip)] == [90]

    def test_nothing_copied_says_so(self, view: TimelineView) -> None:
        # 黙って何も起きないと、貼り付けが壊れているのか区別がつかない
        messages: list[str] = []
        view.status_message.connect(messages.append)
        assert not view.paste_at_playhead()
        assert messages


class TestInsertPaste:
    def test_the_menu_offers_it_once_something_is_copied(self, view: TimelineView) -> None:
        # 右クリックに無いと、挿入貼り付けを知らない人は Ctrl+V の後に手で詰め直す
        empty = QPoint(800, _clip_point(view).y())
        assert _labels(view, empty)["挿入して貼り付け"] is False
        view.select(view.project.timeline.tracks[0].clips[0].id)
        assert view.copy_selected()
        assert _labels(view, empty)["挿入して貼り付け"] is True

    def test_it_pushes_what_is_behind_the_playhead(self, view: TimelineView) -> None:
        # 壊れると、後ろのクリップが動かず、貼った物が新しいトラックへ逃げる
        received = _received(view)
        view.select(view.project.timeline.tracks[0].clips[0].id)
        view.copy_selected()
        view.set_playhead(10)
        assert view.insert_paste_at_playhead()
        (commands,) = received
        assert isinstance(commands[0], InsertGap)
        assert commands[0].length == 30
        assert [c.clip.timeline_start for c in commands if isinstance(c, AddClip)] == [10]

    def test_the_preference_narrows_the_pushed_tracks(self, view: TimelineView) -> None:
        # 設定で貼り先だけを選んでも全トラックを押すと、設定が効いていない
        received = _received(view)
        view.select(view.project.timeline.tracks[0].clips[0].id)
        view.copy_selected()
        view.set_insert_all_tracks(False)
        assert view.insert_paste_at_playhead()
        gap = received[0][0]
        assert isinstance(gap, InsertGap)
        assert gap.track_ids == (view.project.timeline.tracks[0].id,)

    def test_a_locked_track_is_reported(self, view: TimelineView) -> None:
        # ロックで断ったのに黙っていると、何が起きなかったのか分からない
        timeline = view.project.timeline
        view.set_project(
            view.project.with_timeline(
                timeline.replace_track(replace(timeline.tracks[0], locked=True))
            )
        )
        view.select(view.project.timeline.tracks[0].clips[0].id)
        view.copy_selected()
        received = _received(view)
        messages: list[str] = []
        view.status_message.connect(messages.append)
        assert not view.insert_paste_at_playhead()
        assert not received
        assert any("ロック" in message for message in messages)


_FIELDS = (QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox)


def _clip_count(window: MainWindow) -> int:
    return sum(len(track.clips) for track in window.document.project.timeline.tracks)


def _press(widget: QWidget, key: Qt.Key, modifiers: Qt.KeyboardModifier) -> None:
    QTest.keyClick(widget, key, modifiers)
    QApplication.processEvents()


def _field_text(field: QWidget) -> str:
    if isinstance(field, (QPlainTextEdit, QTextEdit)):
        return field.toPlainText()
    if isinstance(field, QSpinBox):
        return field.text()
    assert isinstance(field, QLineEdit)
    return field.text()


class TestKeysWhileTyping:
    """入力欄に打っている間は、欄が使うキーで窓のショートカットを動かさない"""

    @pytest.fixture
    def shown(self, window: MainWindow) -> MainWindow:
        window.show()
        window.activateWindow()
        timeline = window._timeline
        timeline.select(window.document.project.timeline.tracks[0].clips[0].id)
        assert timeline.copy_selected()
        window.seek(10)
        QApplication.processEvents()
        return window

    @pytest.mark.parametrize("kind", _FIELDS)
    def test_ctrl_shift_v_in_a_field_pastes_into_the_field(
        self, shown: MainWindow, kind: type[QWidget], fake_clipboard: FakeClipboard
    ) -> None:
        # 直す前は窓の挿入貼り付けが動き、字幕や設定の欄に打っている途中でクリップが増えた
        field = kind(shown)
        field.show()
        field.setFocus()
        if isinstance(field, QSpinBox):
            # 数の欄は入っている 0 を選んだ所へ貼る（後ろへ足すと 012 で範囲の外になる）
            field.selectAll()
        fake_clipboard.setText("12")
        before = _clip_count(shown)
        _press(field, Qt.Key.Key_V, CTRL_SHIFT)
        assert _clip_count(shown) == before
        assert "12" in _field_text(field)

    @pytest.mark.parametrize(
        ("key", "modifiers"),
        [
            (Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier),
        ],
    )
    def test_the_other_editing_keys_stay_in_the_field(
        self, shown: MainWindow, key: Qt.Key, modifiers: Qt.KeyboardModifier
    ) -> None:
        # Qt の入力欄が自分で受け取るキー 守りが外れると、文字を消すつもりでクリップが消える
        field = QLineEdit(shown)
        field.show()
        field.setFocus()
        before = shown.document.project
        _press(field, key, modifiers)
        assert shown.document.project == before

    def test_outside_a_field_ctrl_shift_v_still_inserts(self, shown: MainWindow) -> None:
        # 欄を守るついでに窓のショートカットまで止めると、挿入貼り付けが使えない
        shown._timeline.setFocus()
        before = _clip_count(shown)
        _press(shown._timeline, Qt.Key.Key_V, CTRL_SHIFT)
        # 再生ヘッドの下のクリップが割れて 1 本、貼った物で 1 本増える
        assert _clip_count(shown) == before + 2

    def test_a_read_only_field_does_not_hold_the_key(self, shown: MainWindow) -> None:
        # 読むだけの欄は貼れない 欄が取ると、押しても何も起きない
        field = QPlainTextEdit(shown)
        field.setReadOnly(True)
        field.show()
        field.setFocus()
        before = _clip_count(shown)
        _press(field, Qt.Key.Key_V, CTRL_SHIFT)
        assert _clip_count(shown) == before + 2


class TestInsertPastePreference:
    def test_the_default_pushes_every_track(self) -> None:
        # Premiere の既定と同じ 貼り先だけを既定にすると、別のトラックの字幕や BGM が黙ってずれる
        assert Preferences().insert_paste == INSERT_ALL_TRACKS
        assert Preferences().inserts_on_all_tracks

    def test_it_is_kept_and_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        # 次の起動で戻ると、選び直すたびに設定画面を開くことになる
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(insert_paste=INSERT_TARGET_TRACKS))
        assert not store.load().inserts_on_all_tracks
        store.path.write_text('{"insert_paste": "sideways"}', encoding="utf-8")
        assert store.load().insert_paste == INSERT_ALL_TRACKS

    @pytest.mark.parametrize("mode", [INSERT_ALL_TRACKS, INSERT_TARGET_TRACKS])
    def test_the_dialog_carries_it(self, qt_application: QApplication, mode: str) -> None:
        # 画面が値を返さないと、設定を開いて OK を押しただけで既定へ戻る
        del qt_application
        dialog = PreferencesDialog(Preferences(insert_paste=mode))
        try:
            assert dialog.preferences().insert_paste == mode
        finally:
            dialog.deleteLater()


class TestTrackHeight:
    def test_dragging_the_border_resizes_that_track(self, view: TimelineView) -> None:
        received = _received(view)
        band = view._layout.bands(view.project.timeline)[0]
        start = QPoint(40, band.bottom)
        QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view, QPoint(40, band.bottom + 30))
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(40, band.bottom + 30))

        track = band.track
        assert received == [[SetTrackHeights(((track.id, track.height + 30),))]]
        # 途中の高さは描画のためだけ 離したあと、自分では書き換えていない
        assert view.project.timeline.tracks[0].height == track.height

    def test_the_border_is_only_grabbed_in_the_header(self, view: TimelineView) -> None:
        # タイムラインの側まで広げると、クリップの下端を掴んだつもりが高さの変更になる
        band = view._layout.bands(view.project.timeline)[0]
        assert view._resize_band_at(QPoint(Metrics.TRACK_HEADER_WIDTH + 50, band.bottom)) is None

    def test_all_tracks_move_together(self, view: TimelineView) -> None:
        # 1 本ずつのコマンドになると、まとめて変えたのに取り消しをトラックの数だけ押す
        received = _received(view)
        view.adjust_track_heights(12)
        (commands,) = received
        (command,) = commands
        assert isinstance(command, SetTrackHeights)
        assert len(command.heights) == 2


class TestMediaPoolMenu:
    def test_removing_asks_the_window(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(video_media,)))
        asked: list[str] = []
        pool.remove_requested.connect(asked.append)
        menu = pool.build_menu(video_media.id)
        remove = next(action for action in menu.actions() if action.text() == "プールから外す")
        remove.trigger()
        assert asked == [str(video_media.id)]


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    yield created
    created.close()


class TestWindow:
    def test_ctrl_v_pastes_as_one_undo_step(self, window: MainWindow) -> None:
        # 壊れると、貼り付けを 1 回の取り消しで戻せない
        timeline = window._timeline
        timeline.select(window.document.project.timeline.tracks[0].clips[0].id)
        timeline.copy_selected()
        window.seek(30)
        timeline.paste_at_playhead()
        assert len(window.document.project.timeline.tracks[0].clips) == 2
        window.undo()
        assert len(window.document.project.timeline.tracks[0].clips) == 1

    def test_ctrl_shift_v_inserts_as_one_undo_step(self, window: MainWindow) -> None:
        # 押し出しと貼り付けが別の段だと、1 回取り消すと間だけ空いたまま残る
        action, default = window._actions["編集/貼り付け（挿入）"]
        assert default == "Ctrl+Shift+V"
        timeline = window._timeline
        before = window.document.project
        timeline.select(before.timeline.tracks[0].clips[0].id)
        timeline.copy_selected()
        window.seek(10)
        action.trigger()
        spans = sorted(
            (c.timeline_start, c.timeline_end)
            for c in window.document.project.timeline.tracks[0].clips
        )
        assert spans == [(0, 10), (10, 40), (40, 60)]
        window.undo()
        assert window.document.project == before

    def test_the_preference_reaches_the_timeline(self, window: MainWindow) -> None:
        # 設定画面で選んでも窓が渡さないと、挿入貼り付けは既定のまま全トラックを押す
        window._apply_preferences(replace(window._preferences, insert_paste=INSERT_TARGET_TRACKS))
        assert window._timeline._insert_all_tracks is False
        window._apply_preferences(replace(window._preferences, insert_paste=INSERT_ALL_TRACKS))
        assert window._timeline._insert_all_tracks is True

    def test_a_file_open_in_another_window_is_noticed(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del qt_application
        path = tmp_path / "本編.sme"
        folder = project_presence_dir(path)
        first = MainWindow(_project(), path=path, confirm_unsaved=False)
        try:
            assert others_holding(folder)
            second = MainWindow(_project(), path=path, confirm_unsaved=False)
            # 尋ねない設定なので開けるが、先の窓がいることは見えている 見えないと、
            # 2 つの窓が警告なしで同じ作品を開き、あとから保存した方の内容だけが残る
            assert second._project_lock is not None
            assert others_holding(folder, second._project_lock.path)
            second.close()
            assert others_holding(folder)
        finally:
            first.close()
        assert not others_holding(folder)

    def test_a_window_opened_anyway_is_still_seen_after_the_first_closes(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 「それでも開く」の窓が数に入らないと、先の窓が閉じたあと、まだ開いて
        # いるのに 3 つ目の窓が警告なしで開けてしまう 待ち時間なしで見えること
        del qt_application
        path = tmp_path / "本編.sme"
        folder = project_presence_dir(path)
        first = MainWindow(_project(), path=path, confirm_unsaved=False)
        second = MainWindow(_project(), path=path, confirm_unsaved=False)
        try:
            first.close()
            assert others_holding(folder)
        finally:
            second.close()
        assert not others_holding(folder)
