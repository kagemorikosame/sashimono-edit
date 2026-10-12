"""削除して詰めたあと、再生ヘッドをつなぎ目（消した所の頭）へ動かす（Issue #299）

入口は 3 つある 編集メニューの「削除して詰める」（Shift+Del）、右クリックの
「削除して詰める」、タイムラインに焦点があるときの Shift+Delete どれも同じ動きになることを見る
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from sashimono.core.model import (
    Clip,
    ClipId,
    MediaItem,
    Project,
    Track,
    TrackKind,
    new_group_id,
)
from sashimono.effects.sources import TEXT
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """試験の間に確認や警告の窓が開いたら、その場で止めて試験を落とす

    開いた窓は誰も閉じないので、試験が止まったまま返ってこない
    メニューから呼ぶ所は Qt の信号の先で、そこで投げた例外は Qt が飲み込む
    投げるだけでは試験が通ってしまうので、開いた窓を覚えておき、終わりに落とす
    """
    opened: list[str] = []

    def refuse(name: str) -> Callable[..., None]:
        def record(*_arguments: object, **_options: object) -> None:
            opened.append(name)
            pytest.fail(f"試験の途中で窓が開いた: {name}")

        return record

    monkeypatch.setattr(QDialog, "exec", refuse("QDialog.exec"))
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, refuse(f"QMessageBox.{name}"))
    yield
    assert not opened, f"試験の途中で窓が開いた: {opened}"


def _text(start: int, duration: int) -> Clip:
    return Clip(timeline_start=start, duration=duration, source=TEXT.create())


def _project(*tracks: Track, media: tuple[MediaItem, ...] = ()) -> Project:
    base = Project.create()
    return replace(base.with_timeline(replace(base.timeline, tracks=tracks)), media=media)


def _open(qt_application: QApplication, project: Project) -> MainWindow:
    del qt_application
    window = MainWindow(project, confirm_unsaved=False)
    window._apply_preferences(replace(window._preferences, ripple_playhead=True))
    return window


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    # V1: 0-30 / 30-60 / 60-90 が隙間なく並ぶ V2: 45-75
    created = _open(
        qt_application,
        _project(
            Track(TrackKind.VIDEO, "V1", (_text(0, 30), _text(30, 30), _text(60, 30))),
            Track(TrackKind.VIDEO, "V2", (_text(45, 30),)),
        ),
    )
    yield created
    created.close()


def _clip(window: MainWindow, track: int, index: int) -> ClipId:
    return window.document.project.timeline.tracks[track].clips[index].id


def _ripple_action(window: MainWindow) -> QAction:
    return next(a for name, (a, _) in window._actions.items() if name.endswith("/削除して詰める"))


def _ripple_by_menu(window: MainWindow) -> None:
    _ripple_action(window).trigger()


def _ripple_by_context_menu(window: MainWindow) -> None:
    view = window._timeline
    clip = view.project.timeline.locate_clip(view.selected_clips[0])
    assert clip is not None
    band = next(b for b in view._layout.bands(view.project.timeline) if b.track.id == clip[0].id)
    point = QPoint(int(view._layout.frame_to_x(clip[1].timeline_start + 1)), band.top + 2)
    menu = view.build_context_menu(point)
    action = next(a for a in menu.actions() if a.text().startswith("削除して詰める"))
    action.trigger()


def _ripple_by_key(window: MainWindow) -> None:
    QTest.keyClick(window._timeline, Qt.Key.Key_Delete, Qt.KeyboardModifier.ShiftModifier)


ENTRANCES: list[Callable[[MainWindow], None]] = [
    _ripple_by_menu,
    _ripple_by_context_menu,
    _ripple_by_key,
]


class TestLanding:
    def test_the_menu_key_is_shift_delete(self, window: MainWindow) -> None:
        # 割り当てが変わると、Shift+Del を押しても何も動かない（ここで動きを見ている入口が外れる）
        assert _ripple_action(window).shortcut().toString() == "Shift+Del"

    @pytest.mark.parametrize("entrance", ENTRANCES)
    def test_every_entrance_lands_on_the_head(
        self, window: MainWindow, entrance: Callable[[MainWindow], None]
    ) -> None:
        # 入口ごとに動きが違うと、右クリックで消したときだけつなぎ目を探し直すことになる
        window._seek(80)
        window.select_clip(_clip(window, 0, 1))
        entrance(window)
        assert [c.timeline_start for c in window.document.project.timeline.tracks[0].clips] == [
            0,
            30,
        ]
        assert window._timeline.playhead == 30

    def test_the_preview_shows_the_head(self, window: MainWindow) -> None:
        # プレビューと再生位置が付いてこないと、再生ヘッドだけ動いて画面は元の場面のまま
        window._seek(80)
        window.select_clip(_clip(window, 0, 1))
        _ripple_by_menu(window)
        assert window._preview._frame == 30
        assert window._playback.frame == 30

    def test_the_earliest_head_across_tracks_wins(self, window: MainWindow) -> None:
        # 遅い方の頭へ寄せると、早い方のつなぎ目を通り過ぎた所から映る
        # 後から選んだ方（V1 の 30）が早い 選んだ順の先頭を取ると 45 になる
        window._seek(5)
        window.select_clips([_clip(window, 1, 0), _clip(window, 0, 1)])
        _ripple_by_menu(window)
        assert window._timeline.playhead == 30

    def test_the_linked_partner_counts(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 映像だけ選んでも音声は一緒に消える 音声の方が早く始まっていれば、つなぎ目は音声の頭
        # 選んだ映像の頭へ寄せると、音のつなぎ目を通り過ぎる
        link = new_group_id()
        picture = Clip(timeline_start=20, duration=30, media_id=video_media.id, link_group=link)
        sound = Clip(timeline_start=10, duration=30, media_id=video_media.id, link_group=link)
        window = _open(
            qt_application,
            _project(
                Track(TrackKind.VIDEO, "V1", (_text(0, 20), picture, _text(50, 30))),
                Track(TrackKind.AUDIO, "A1", (sound,)),
                media=(video_media,),
            ),
        )
        try:
            window._seek(70)
            window.select_clip(picture.id)
            _ripple_by_menu(window)
            assert window.document.project.timeline.locate_clip(sound.id) is None
            assert window._timeline.playhead == 10
        finally:
            window.close()

    def test_the_end_is_clamped_to_the_new_length(self, qt_application: QApplication) -> None:
        # 末尾を消すと頭が新しい長さの先に出る そのままだと何も映らない所に再生ヘッドが残る
        window = _open(
            qt_application,
            _project(Track(TrackKind.VIDEO, "V1", (_text(0, 30), _text(60, 30)))),
        )
        try:
            window._seek(80)
            window.select_clip(_clip(window, 0, 1))
            _ripple_by_menu(window)
            assert window.document.project.duration == 30
            assert window._timeline.playhead == 30
        finally:
            window.close()


class TestLeftAlone:
    def test_a_plain_delete_keeps_the_playhead(self, window: MainWindow) -> None:
        # 詰めない削除は隙間が残るだけ 動かすと、見ていた場面から勝手に離れる
        window._seek(80)
        window.select_clip(_clip(window, 0, 1))
        window._timeline.delete_selected()
        assert len(window.document.project.timeline.tracks[0].clips) == 2
        assert window._timeline.playhead == 80

    def test_turning_it_off_keeps_the_playhead(self, window: MainWindow) -> None:
        # 切っても動くなら、設定がある方が質が悪い
        window._apply_preferences(replace(window._preferences, ripple_playhead=False))
        window._seek(80)
        window.select_clip(_clip(window, 0, 1))
        _ripple_by_menu(window)
        assert len(window.document.project.timeline.tracks[0].clips) == 2
        assert window._timeline.playhead == 80

    def test_a_locked_track_keeps_the_playhead(self, qt_application: QApplication) -> None:
        # 何も消えていないのに再生ヘッドだけが飛ぶと、消えたように見える
        locked = Track(TrackKind.VIDEO, "V1", (_text(0, 30), _text(30, 30)), locked=True)
        window = _open(qt_application, _project(locked))
        try:
            window._seek(50)
            window.select_clip(_clip(window, 0, 1))
            _ripple_by_menu(window)
            assert len(window.document.project.timeline.tracks[0].clips) == 2
            assert window._timeline.playhead == 50
        finally:
            window.close()

    def test_a_locked_group_keeps_the_playhead(self, qt_application: QApplication) -> None:
        # グループの仲間がロックで止まったとき（_refuse_locked_group）も同じ
        group = new_group_id()
        free = replace(_text(30, 30), group_id=group)
        held = replace(_text(30, 30), group_id=group)
        window = _open(
            qt_application,
            _project(
                Track(TrackKind.VIDEO, "V1", (_text(0, 30), free)),
                Track(TrackKind.VIDEO, "V2", (held,), locked=True),
            ),
        )
        try:
            window._seek(50)
            window.select_clips([free.id, held.id])
            _ripple_by_menu(window)
            assert window.document.project.timeline.locate_clip(free.id) is not None
            assert window._timeline.playhead == 50
        finally:
            window.close()


class TestUndo:
    def test_one_undo_brings_the_clip_back(self, window: MainWindow) -> None:
        # 再生ヘッドの動きが別の段になると、1 回の取り消しでクリップが戻らない
        before = window.document.project.timeline
        window._seek(80)
        window.select_clip(_clip(window, 0, 1))
        _ripple_by_menu(window)
        assert window.document.history_labels == ("クリップを削除（詰める）",)
        window.undo()
        assert window.document.project.timeline == before
        assert not window.document.can_undo
        # 取り消しでは再生ヘッドを動かさない（決めたこと） 戻ったクリップの頭に残る
        assert window._timeline.playhead == 30


class TestPlaying:
    def test_playback_goes_on_from_the_head(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 編集を当てると再生は 1 度止まる 鳴らし直さないと、つなぎ目の前で止まったままになる
        # 音の出口は機械によって開けないので、鳴らす所だけ差し替える
        started: list[int] = []
        player = window._playback._player
        monkeypatch.setattr(player, "start", lambda sample, **_: started.append(sample))
        monkeypatch.setattr(player, "stop", lambda: None)
        window._seek(80)
        window._playback.play()
        assert window._playback.is_playing
        window.select_clip(_clip(window, 0, 1))
        _ripple_by_menu(window)
        assert window._playback.is_playing
        assert window._playback.frame == 30
        # 最初に鳴らした 1 回と、頭から鳴らし直した 1 回
        assert len(started) == 2
        window._playback.stop()

    def test_a_stopped_playback_stays_stopped(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 止めていたのに鳴り出すと、消しただけで音が出て驚く
        # 鳴らし始めたかは数えて見る 差し替えた所で投げても、信号の先なので Qt が飲み込む
        started: list[int] = []
        player = window._playback._player
        monkeypatch.setattr(player, "start", lambda sample, **_: started.append(sample))
        monkeypatch.setattr(player, "stop", lambda: None)
        window._seek(80)
        window.select_clip(_clip(window, 0, 1))
        _ripple_by_menu(window)
        assert started == []
        assert not window._playback.is_playing
        assert window._timeline.playhead == 30

    def test_landing_on_the_end_stops_there(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 末尾に着いたところで鳴らし直すと、続きが無いので作品の頭（0）へ巻き戻って
        # 頭から鳴り出す 消した所を見たかったのに、まったく別の場面へ飛ぶ
        window = _open(
            qt_application,
            _project(Track(TrackKind.VIDEO, "V1", (_text(0, 30), _text(60, 30)))),
        )
        try:
            started: list[int] = []
            player = window._playback._player
            monkeypatch.setattr(player, "start", lambda sample, **_: started.append(sample))
            monkeypatch.setattr(player, "stop", lambda: None)
            window._seek(80)
            window._playback.play()
            window.select_clip(_clip(window, 0, 1))
            _ripple_by_menu(window)
            assert not window._playback.is_playing
            assert window._timeline.playhead == 30
            assert window._playback.frame == 30
            # 最初に鳴らした 1 回だけ 末尾から鳴らし直していない
            assert len(started) == 1
        finally:
            window.close()


class TestPreferences:
    def test_the_default_moves_it(self) -> None:
        # 既定はつなぎ目を確かめたい人の側（Issue #299）
        assert Preferences().ripple_playhead is True

    def test_the_choice_is_saved(self, tmp_path: Path) -> None:
        # 保存されないと、切った人が起動するたびに切り直すことになる
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(ripple_playhead=False))
        assert store.load().ripple_playhead is False

    def test_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        # 壊れた値で起動を止めない 項目ごとに既定へ戻す
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"ripple_playhead": "はい"}), encoding="utf-8")
        assert PreferenceStore(path).load().ripple_playhead is True

    def test_the_dialog_shows_and_returns_it(self, qt_application: QApplication) -> None:
        # 画面に出ていない、または OK で返らないと、設定画面から切れない
        del qt_application
        chosen = Preferences(ripple_playhead=False)
        dialog = PreferencesDialog(chosen)
        assert dialog.preferences() == chosen
        dialog._ripple_playhead.setChecked(True)
        assert dialog.preferences().ripple_playhead is True
