"""タイムラインへのドラッグ＆ドロップと、素材一覧の表示（Issue #27）

- エクスプローラーのファイルをタイムラインへ落とすと、落とした所へ置かれる
  新しくプロジェクトを作らなくても（起動した直後の空のプロジェクトのままでも）落とせる
- 素材一覧の素材をタイムラインへ引いて落とすと、落とした所へ置かれ、1 回の取り消しで戻る
- 素材一覧は一覧とアイコンを切り替えられ、選んだ方を次に開いたときも覚えている
  どちらの表示でも行の頭に絵（映像・画像はサムネイル、音声は音声の印）が出る

窓は表示しない 落とし込みは QDropEvent を作って直に渡す 素材を調べる所は
差し替えて、実物のファイルが無くても読み込みの流れ（裏で調べてから置く）を通す
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QListView

from sashimono.core.model import (
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    new_media_id,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_icons import GRID_ICON_SIZE
from sashimono.ui.media_pool import (
    MEDIA_MIME,
    VIEW_ICONS,
    VIEW_LIST,
    MediaPoolWidget,
    media_ids_in,
)
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.theme import Colors
from sashimono.ui.timeline import TimelineView
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.media_fixtures import SampleMedia
from tests.waiting import wait_until

# --- 組み立て ---


def _two_tracks() -> Project:
    base = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    tracks = (Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _file_mime(*paths: Path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def _media_mime(*media: MediaItem) -> QMimeData:
    mime = QMimeData()
    mime.setData(MEDIA_MIME, "\n".join(str(item.id) for item in media).encode("utf-8"))
    return mime


def _drop(view: TimelineView, mime: QMimeData, position: QPointF) -> QDropEvent:
    event = QDropEvent(
        position,
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.dropEvent(event)
    return event


def _move(view: TimelineView, mime: QMimeData, position: QPoint) -> QDragMoveEvent:
    event = QDragMoveEvent(
        position,
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.dragMoveEvent(event)
    return event


def _point(view: TimelineView, frame: int, track_index: int) -> QPointF:
    """そのトラック（上から数えた並び）の、そのフレームの位置"""
    band = view._layout.bands(view.project.timeline)[track_index]
    return QPointF(view._layout.frame_to_x(frame) + 1, band.top + band.height / 2)


@pytest.fixture
def view(qt_application: QApplication) -> Iterator[TimelineView]:
    del qt_application
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    created = TimelineView(_two_tracks(), analyzer)
    created.resize(900, 300)
    yield created
    analyzer.close()


class _FakeProbe:
    """素材を調べる所の差し替え パスごとに別の素材として返す"""

    def __init__(self, template: MediaItem) -> None:
        self.template = template

    def __call__(self, path: Path) -> MediaItem:
        return replace(self.template, path=path, id=new_media_id())


def _quiet(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    """置いた素材は本物のファイルではない 解析と控えを頼まないようにする"""
    monkeypatch.setattr(window, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(window._analyzer, "request", lambda _media, **_kwargs: None)


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_two_tracks(), confirm_unsaved=False)
    _quiet(created, monkeypatch)
    yield created
    created.close()


# --- タイムラインが受け取る所 ---


class TestTheTimelineTakesDrops:
    def test_files_are_handed_over_with_the_frame_and_track(self, view: TimelineView) -> None:
        # 前はタイムラインが落とし込みを受け付けず、一覧へ落とすしかなかった
        received: list[tuple[list[Path], int, str]] = []
        view.files_dropped.connect(
            lambda paths, frame, track: received.append((paths, frame, track))
        )
        audio = view._layout.bands(view.project.timeline)[1].track
        event = _drop(view, _file_mime(Path("C:/素材/声.wav")), _point(view, 90, 1))
        assert event.isAccepted()
        ((paths, frame, track),) = received
        assert paths == [Path("C:/素材/声.wav")]
        assert frame == 90
        assert track == str(audio.id)

    def test_media_from_the_pool_is_handed_over_as_ids(
        self, view: TimelineView, video_media: MediaItem
    ) -> None:
        # ID で渡らないと、素材一覧から引いた素材がタイムラインのどこにも置かれない
        received: list[tuple[list[str], int, str]] = []
        view.media_dropped.connect(lambda ids, frame, track: received.append((ids, frame, track)))
        files: list[object] = []
        view.files_dropped.connect(lambda *args: files.append(args))
        _drop(view, _media_mime(video_media), _point(view, 30, 0))
        ((ids, frame, _track),) = received
        assert ids == [str(video_media.id)]
        assert frame == 30
        # 素材一覧から来た物をファイルとして読み込み直すと、同じ素材が一覧に 2 つ並ぶ
        assert files == []

    def test_it_is_taken_as_a_copy(self, view: TimelineView) -> None:
        # 動かす（Move）で受け取ると、引いてきた側（素材一覧）が元の行を消す
        event = _drop(view, _file_mime(Path("C:/素材/a.mp4")), _point(view, 0, 0))
        assert event.dropAction() == Qt.DropAction.CopyAction

    def test_something_else_is_refused(self, view: TimelineView) -> None:
        # 文字を落とされて「読み込めない」と出すより、落とせない印を出す方が分かる
        mime = QMimeData()
        mime.setText("ただの文字")
        event = _move(view, mime, _point(view, 0, 0).toPoint())
        assert not event.isAccepted()
        assert view.drop_guide is None

    def test_a_drop_below_the_tracks_has_no_track(self, view: TimelineView) -> None:
        # トラックの無い所へ落としたら、置く側が合う種類のトラックを探す（作る）
        spot = view.drop_spot_at(QPointF(400, 290))
        assert spot.track_id is None

    def test_a_drop_on_the_header_uses_the_left_edge(self, view: TimelineView) -> None:
        # 名前の欄はフレームを持たない そこへ落とした人は「このトラックの頭へ」と思っている
        band = view._layout.bands(view.project.timeline)[0]
        spot = view.drop_spot_at(QPointF(10, band.top + 5))
        assert spot.frame == 0
        assert spot.track_id == band.track.id


class TestTheGuide:
    def test_dragging_shows_where_it_lands(self, view: TimelineView) -> None:
        # 目安が出ないと、離すまでどのトラックのどこへ入るのか分からない
        _move(view, _file_mime(Path("C:/素材/a.mp4")), _point(view, 120, 0).toPoint())
        guide = view.drop_guide
        assert guide is not None
        assert guide.spot.frame == 120
        image = view.grab().toImage()
        x = round(view._layout.frame_to_x(120))
        band = view._layout.bands(view.project.timeline)[1]
        colors = {image.pixelColor(x + dx, band.top + band.height // 2).name() for dx in (-1, 0, 1)}
        assert Colors.ACCENT.name() in colors

    def test_leaving_removes_the_guide(self, view: TimelineView) -> None:
        # 外へ出たのに線が残ると、落としていないのに置かれたように見える
        _move(view, _file_mime(Path("C:/素材/a.mp4")), _point(view, 120, 0).toPoint())
        view.dragLeaveEvent(QDragLeaveEvent())
        assert view.drop_guide is None

    def test_dropping_removes_the_guide(self, view: TimelineView) -> None:
        # 落とした後も線が残ると、置いたクリップの上に余計な縦線が出たままになる
        mime = _file_mime(Path("C:/素材/a.mp4"))
        _move(view, mime, _point(view, 120, 0).toPoint())
        _drop(view, mime, _point(view, 120, 0))
        assert view.drop_guide is None

    def test_entering_is_accepted_for_files(self, view: TimelineView) -> None:
        # 入ったときに受け取ると言わないと、Qt は動かしている間の知らせも落とす知らせも寄こさない
        # 中身は手元で持つ 出来事は中身を持たず、渡した直後に Python の側で消える
        mime = _file_mime(Path("C:/素材/a.mp4"))
        event = QDragEnterEvent(
            _point(view, 5, 0).toPoint(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dragEnterEvent(event)
        assert event.isAccepted()


# --- 窓が置く所 ---


def _clips_by_track(window: MainWindow) -> dict[str, list[tuple[int, str]]]:
    project = window.document.project
    names = {media.id: media.path.name for media in project.media}
    return {
        track.name: [
            (clip.timeline_start, names.get(clip.media_id, "")) if clip.media_id else (0, "")
            for clip in track.clips
        ]
        for track in project.timeline.tracks
    }


class TestFilesDroppedOnTheTimeline:
    def test_they_land_where_they_were_dropped(
        self,
        window: MainWindow,
        audio_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 落とした位置を捨てると、狙ったトラックと時刻でなく末尾へ置かれて並べ直す羽目になる
        monkeypatch.setattr(main_window_module, "probe_media", _FakeProbe(audio_media))
        timeline = window._timeline
        _drop(timeline, _file_mime(tmp_path / "声.wav"), _point(timeline, 75, 1))
        # 調べるのは裏のスレッド 落とした所で待たされない
        assert window.importing
        assert window.wait_for_imports()
        assert _clips_by_track(window) == {"V1": [], "A1": [(75, "声.wav")]}

    def test_several_files_line_up_and_undo_together(
        self,
        window: MainWindow,
        audio_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 何本かを落として何回も取り消すことになると、落とし込みが使いにくい
        monkeypatch.setattr(main_window_module, "probe_media", _FakeProbe(audio_media))
        timeline = window._timeline
        _drop(timeline, _file_mime(tmp_path / "1.wav", tmp_path / "2.wav"), _point(timeline, 30, 1))
        assert window.wait_for_imports()
        length = window.document.project.timeline.tracks[1].clips[0].duration
        assert _clips_by_track(window)["A1"] == [(30, "1.wav"), (30 + length, "2.wav")]
        window.undo()
        assert window.document.project.media == ()
        assert _clips_by_track(window) == {"V1": [], "A1": []}

    def test_it_works_without_making_a_new_project(
        self,
        qt_application: QApplication,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 起動した直後（プロジェクトを渡さずに開いた窓）はトラックを 1 本も持たない
        # それでも落とせないと、〔新規〕を押してからでないと何も始められない
        del qt_application
        started = MainWindow(confirm_unsaved=False)
        try:
            _quiet(started, monkeypatch)
            monkeypatch.setattr(main_window_module, "probe_media", _FakeProbe(video_media))
            assert len(started.document.project.timeline.tracks) == 0
            timeline = started._timeline
            timeline.resize(900, 300)
            _drop(timeline, _file_mime(tmp_path / "本編.mp4"), QPointF(400, 120))
            assert started.wait_for_imports()
            after: Project = started.document.project
            placed = {
                track.kind: [clip.timeline_start for clip in track.clips]
                for track in after.timeline.tracks
            }
            frame = timeline.drop_spot_at(QPointF(400, 120)).frame
            # 起動した直後のプロジェクトは好みの方式（既定は混合）なので、音付きの動画は
            # 1 本のレイヤーに 1 本のクリップで入る
            assert placed == {TrackKind.MIXED: [frame]}
        finally:
            started.close()

    def test_a_queued_drop_keeps_its_own_spot(
        self,
        window: MainWindow,
        audio_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 読み込みの最中に落とした分は待たされる 待っている間に位置を忘れると、
        # 末尾へ並んで落とした所と違う所に出る
        monkeypatch.setattr(main_window_module, "probe_media", _FakeProbe(audio_media))
        window.import_media([tmp_path / "先.wav"])
        timeline = window._timeline
        _drop(timeline, _file_mime(tmp_path / "後.wav"), _point(timeline, 600, 1))
        assert window.wait_for_imports()
        # 先の分（30 秒）が 0 から置かれるので、600 の所は A1 が埋まっている 落とした
        # 時刻は保ったまま、空いたトラック（無ければ新しく作ったもの）へ回る
        rows = {
            name: (track, start)
            for track, clips in _clips_by_track(window).items()
            for start, name in clips
        }
        assert rows["先.wav"] == ("A1", 0)
        assert rows["後.wav"] == ("A2", 600)


class TestMediaDroppedFromThePool:
    def test_it_lands_where_it_was_dropped_in_one_undo(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 映像と音声が別々の手になると、1 回の取り消しで片方だけ残り音ずれの元になる
        from sashimono.core.commands import AddMedia

        window.execute(AddMedia(video_media))
        timeline = window._timeline
        _drop(timeline, _media_mime(video_media), _point(timeline, 150, 0))
        tracks = window.document.project.timeline.tracks
        assert [clip.timeline_start for clip in tracks[0].clips] == [150]
        assert [clip.timeline_start for clip in tracks[1].clips] == [150]
        # 取り消し 1 回で映像と音声の両方が消える 素材は一覧に残る
        window.undo()
        assert all(track.clips == () for track in window.document.project.timeline.tracks)
        assert window.document.project.media == (video_media,)

    def test_the_pool_rows_carry_their_ids(
        self, qt_application: QApplication, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 行を引いたときに載せる中身 ID が載らないと、タイムラインは何を置くのか分からない
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(video_media, audio_media)))
        items = [pool._list.item(row) for row in range(pool._list.count())]
        mime = pool._list.mimeData(items)
        assert media_ids_in(mime) == [video_media.id, audio_media.id]
        # ファイルの URL は載せない 載せると一覧の上に落としたときに読み込み直す
        assert not mime.hasUrls()

    def test_the_pool_only_drags_out(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 一覧の中で動かせると、画面の並びが素材の並び（プロジェクト）と食い違う
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(video_media,)))
        for mode in (VIEW_ICONS, VIEW_LIST):
            pool.set_view_mode(mode)
            assert pool._list.dragEnabled()
            assert pool._list.dragDropMode() == QListView.DragDropMode.DragOnly
            assert pool._list.movement() == QListView.Movement.Static


# --- 素材一覧の表示 ---


def _solid_tile(color: tuple[int, int, int]) -> np.ndarray:
    tile = np.zeros((72, 128, 4), dtype=np.uint8)
    tile[:, :, :3] = color
    tile[:, :, 3] = 255
    return tile


def _center(pool: MediaPoolWidget, row: int) -> str:
    icon = pool._list.item(row).icon()
    image = icon.pixmap(GRID_ICON_SIZE).toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2).name()


class TestThePoolView:
    def test_it_switches_to_icons(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # ボタンが効かないか知らせが出ないと、アイコン表示へ変えられず選んだ表示も覚えない
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(video_media,)))
        chosen: list[str] = []
        pool.view_mode_changed.connect(chosen.append)
        icons = next(
            b for b in pool._view_buttons.buttons() if b.property("view_mode") == VIEW_ICONS
        )
        icons.click()
        assert pool._list.viewMode() == QListView.ViewMode.IconMode
        assert chosen == [VIEW_ICONS]
        # アイコン表示では名前だけにする 長さと大きさまで並べると枠の中で折り返して読めない
        assert pool.row_text(video_media.id) == video_media.name
        assert "1920x1080" in pool._list.item(0).toolTip()

    def test_setting_it_from_outside_does_not_announce(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 覚えた表示を当てただけで知らせると、起動のたびに設定ファイルを書き直す
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(video_media,)))
        chosen: list[str] = []
        pool.view_mode_changed.connect(chosen.append)
        pool.set_view_mode(VIEW_ICONS)
        assert chosen == []
        pool.set_view_mode("知らない表示")
        assert pool.view_mode == VIEW_LIST

    def test_rows_show_a_thumbnail_or_the_audio_mark(
        self, qt_application: QApplication, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 絵が出ないと、名前の似た素材を 1 本ずつ置いて確かめることになる
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(video_media, audio_media)))
        pending = _center(pool, 0)
        tiles: dict[str, np.ndarray] = {}
        pool.set_thumbnail_source(lambda media: tiles.get(media.path.name))
        # まだ絵が無い間は印を出す 空の行にすると、表示が壊れたように見える
        assert not pool._list.item(0).icon().isNull()
        assert _center(pool, 0) == pending
        tiles["本編.mp4"] = _solid_tile((200, 30, 40))
        pool.refresh_thumbnails()
        assert _center(pool, 0) == "#c81e28"
        # 音声だけの素材はサムネイルを持たない 音声の印を出す
        audio_center = _center(pool, 1)
        assert audio_center not in (pending, "#c81e28", "#000000")

    def test_the_choice_is_remembered(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 閉じて開き直すたびに一覧へ戻ると、アイコンで使う人が毎回押し直す
        del qt_application
        first = MainWindow(Project.create(media=(video_media,)), confirm_unsaved=False)
        try:
            pool = first._media_pool
            icons = next(
                b for b in pool._view_buttons.buttons() if b.property("view_mode") == VIEW_ICONS
            )
            icons.click()
        finally:
            first.close()
        assert PreferenceStore().load().media_view == VIEW_ICONS
        second = MainWindow(Project.create(media=(video_media,)), confirm_unsaved=False)
        try:
            assert second._media_pool.view_mode == VIEW_ICONS
            assert second._media_pool._list.viewMode() == QListView.ViewMode.IconMode
        finally:
            second.close()

    def test_the_settings_dialog_keeps_it(self, qt_application: QApplication) -> None:
        # 設定を開いて OK を押しただけで、ボタンで選んだ表示が一覧へ戻ってはいけない
        del qt_application
        dialog = PreferencesDialog(Preferences(media_view=VIEW_ICONS))
        assert dialog.preferences().media_view == VIEW_ICONS

    def test_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        # 設定ファイルの表示の値が壊れていると、素材一覧が知らない表示で開けず空のまま残る
        path = tmp_path / "preferences.json"
        path.write_text('{"media_view": "巨大"}', encoding="utf-8")
        assert PreferenceStore(path).load().media_view == VIEW_LIST


class TestRealThumbnails:
    def test_an_imported_video_gets_its_picture(
        self, qt_application: QApplication, sample_av: SampleMedia
    ) -> None:
        """本物の素材を読み込むと、裏で作った絵の並びから行の絵が載る

        一覧のために素材を開き直していないことも見る（絵の並びが無い間は印のまま）
        """
        del qt_application
        window = MainWindow(_two_tracks(), confirm_unsaved=False)
        try:
            window.import_media([sample_av.path])
            assert window.wait_for_imports()
            (media,) = window.document.project.media
            pool = window._media_pool

            def turn() -> None:
                QApplication.processEvents()
                window._flush_analysis()

            # 解析が終わる時間は機械の速さで変わる 秒で区切らず、絵が届くのを待つ（#405）
            wait_until(lambda: media.id in pool._icons, "素材の絵", step=turn, pause=0.02)
            assert window._analyzer.filmstrip(media) is not None
        finally:
            window.close()
