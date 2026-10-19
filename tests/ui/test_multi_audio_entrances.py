"""動画の映像と音声の置き方の設定が、置く入口のすべてに効くこと（Issue #27 の流れ）

入口はファイルのドロップ・メニューからの読み込み・素材一覧からのドラッグ・素材一覧の
〔タイムラインへ置く〕・AI の置く道具・ドラッグ中の目安 1 つでも設定を見ないと、
同じ動画でも置き方によってレイヤーの数が変わる

窓は表示しない（オフスクリーン） 素材は作り物で、解析と控えは頼まない
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, AddMedia
from sashimono.core.model import (
    AudioStreamInfo,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
    new_media_id,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_pool import MEDIA_MIME
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.timeline import TimelineView
from sashimono.ui.workspace import (
    MEDIA_SPLIT,
    MEDIA_TOGETHER,
    Preferences,
    PreferenceStore,
)

_RATE = FrameRate(30)


def _two_voices() -> MediaItem:
    """映像 1 本と音声 2 本の 5 秒の動画（作り物 ファイルは開かない）"""
    return MediaItem(
        path=Path("C:/素材/録画.mkv"),
        duration=Fraction(5),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=320,
                height=240,
                frame_rate=_RATE,
                time_base=Fraction(1, 15360),
                codec="h264",
            ),
        ),
        audio_streams=tuple(
            AudioStreamInfo(
                index=index,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="aac",
            )
            for index in (1, 2)
        ),
    )


def _mixed() -> Project:
    base = Project.create(
        ProjectSettings(width=320, height=240, frame_rate=_RATE, layer_mode=LayerMode.MIXED)
    )
    return base.with_timeline(
        replace(base.timeline, tracks=(Track(TrackKind.MIXED, "レイヤー 1"),))
    )


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_mixed(), confirm_unsaved=False)
    # 置いた素材は本物のファイルではない 解析と控えを頼まない
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    monkeypatch.setattr(
        main_window_module,
        "probe_media",
        lambda path: replace(_two_voices(), path=path, id=new_media_id()),
    )
    yield created
    created.close()


def _choose(window: MainWindow, mode: str) -> None:
    """設定画面で OK を押したときと同じ道で設定を当てる"""
    window._apply_preferences(replace(window._preferences, media_split=mode))


def _layers(window: MainWindow) -> int:
    return len(window.document.project.timeline.tracks)


def _pooled(window: MainWindow) -> MediaItem:
    media = _two_voices()
    window.execute(AddMedia(media))
    return media


def _file_drop(window: MainWindow, path: Path) -> None:
    view = window._timeline
    band = view._layout.bands(view.project.timeline)[0]
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(
        QPointF(view._layout.frame_to_x(0) + 1, band.top + band.height / 2),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.dropEvent(event)
    assert window.wait_for_imports()


_EXPECTED = [(MEDIA_SPLIT, 3), (MEDIA_TOGETHER, 1)]


class TestEveryEntrance:
    @pytest.mark.parametrize(("mode", "layers"), _EXPECTED)
    def test_a_file_dropped_on_the_timeline(
        self, window: MainWindow, tmp_path: Path, mode: str, layers: int
    ) -> None:
        # 落としたファイルだけ設定を見ないと、ドロップの時だけ置き方が違う
        _choose(window, mode)
        _file_drop(window, tmp_path / "録画.mkv")
        assert _layers(window) == layers

    @pytest.mark.parametrize(("mode", "layers"), _EXPECTED)
    def test_importing_from_the_menu(
        self, window: MainWindow, tmp_path: Path, mode: str, layers: int
    ) -> None:
        # 〔ファイル → 読み込み〕は末尾へ置く別の道（insert_media）を通る
        _choose(window, mode)
        window.import_media([tmp_path / "録画.mkv"])
        assert window.wait_for_imports()
        assert _layers(window) == layers

    @pytest.mark.parametrize(("mode", "layers"), _EXPECTED)
    def test_dragging_from_the_media_pool(self, window: MainWindow, mode: str, layers: int) -> None:
        # プールから引いた時だけ設定を見ないと、入れ方によって音声のレイヤーの数が変わる
        _choose(window, mode)
        media = _pooled(window)
        window._on_media_dropped([str(media.id)], 0, "")
        assert _layers(window) == layers

    @pytest.mark.parametrize(("mode", "layers"), _EXPECTED)
    def test_placing_from_the_media_pool(self, window: MainWindow, mode: str, layers: int) -> None:
        # プールの〔置く〕だけ設定を見ないと、選んだ置き方と違う数のレイヤーが並ぶ
        _choose(window, mode)
        media = _pooled(window)
        window._insert_media_by_id(str(media.id))
        assert _layers(window) == layers

    @pytest.mark.parametrize(("mode", "expected"), [(MEDIA_SPLIT, True), (MEDIA_TOGETHER, False)])
    def test_the_assistant_reads_the_same_setting(
        self, window: MainWindow, mode: str, expected: bool
    ) -> None:
        # AI の置く道具が設定を見ないと、頼み方によってレイヤーの数が変わる
        _choose(window, mode)
        assert window.splits_media is expected

    def test_one_undo_takes_back_the_split(self, window: MainWindow) -> None:
        # 足したレイヤーが取り消しで残ると、何も無いレイヤーが並んだままになる
        media = _pooled(window)
        window._on_media_dropped([str(media.id)], 0, "")
        assert _layers(window) == 3
        window.undo()
        assert _layers(window) == 1
        assert window.document.project.timeline.tracks[0].clips == ()


class TestTheDropGuide:
    @pytest.mark.parametrize(("mode", "added"), [(MEDIA_SPLIT, 3), (MEDIA_TOGETHER, 1)])
    def test_the_guide_matches_the_drop(self, window: MainWindow, mode: str, added: int) -> None:
        # 目安だけ設定を見ないと、目安に無いレイヤーが落とした後に増える
        _choose(window, mode)
        media = _pooled(window)
        view: TimelineView = window._timeline
        view.resize(900, 300)
        band = view._layout.bands(view.project.timeline)[0]
        mime = QMimeData()
        mime.setData(MEDIA_MIME, str(media.id).encode("utf-8"))
        view.dragMoveEvent(
            QDragMoveEvent(
                QPointF(view._layout.frame_to_x(0) + 1, band.top + band.height / 2).toPoint(),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        preview = view.drop_preview
        assert preview is not None
        assert sum(isinstance(c, AddClip) for c in preview.commands) == added


class TestThePreference:
    def test_the_default_splits_and_is_kept(self, tmp_path: Path) -> None:
        # 利用者の要望 既定は分ける 1 本目だけを選んだ人は次の起動でもそのまま
        assert Preferences().media_split == MEDIA_SPLIT
        assert Preferences().splits_media
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(media_split=MEDIA_TOGETHER))
        loaded = store.load()
        assert loaded.media_split == MEDIA_TOGETHER
        assert not loaded.splits_media

    def test_an_unknown_value_falls_back_to_split(self, tmp_path: Path) -> None:
        # 知らない値のまま持つと、どちらの置き方にもならない値が設定画面に残る
        path = tmp_path / "preferences.json"
        path.write_text('{"media_split": "layered"}', encoding="utf-8")
        assert PreferenceStore(path).load().media_split == MEDIA_SPLIT
        path.write_text('{"multi_audio": "layered"}', encoding="utf-8")
        assert PreferenceStore(path).load().media_split == MEDIA_SPLIT

    @pytest.mark.parametrize(("old", "new"), [("first", MEDIA_TOGETHER), ("split", MEDIA_SPLIT)])
    def test_the_old_setting_is_carried_over(self, tmp_path: Path, old: str, new: str) -> None:
        # 前の版の「音声が複数ある動画の置き方」を読まないと、1 本目だけ（分けない）を選んでいた
        # 人の置き方が、更新しただけで黙って分ける側へ変わる
        path = tmp_path / "preferences.json"
        path.write_text(f'{{"multi_audio": "{old}"}}', encoding="utf-8")
        assert PreferenceStore(path).load().media_split == new

    def test_the_new_name_wins_over_the_old_one(self, tmp_path: Path) -> None:
        # 新しい版で保存した後も古い名前が残る（前の版で開いて保存したファイル）
        # 古い方を優先すると、設定画面で選び直しても元へ戻る
        path = tmp_path / "preferences.json"
        path.write_text('{"multi_audio": "first", "media_split": "split"}', encoding="utf-8")
        assert PreferenceStore(path).load().media_split == MEDIA_SPLIT

    @pytest.mark.parametrize("mode", [MEDIA_SPLIT, MEDIA_TOGETHER])
    def test_the_dialog_carries_it(self, qt_application: QApplication, mode: str) -> None:
        # 画面が値を返さないと、設定を開いて OK を押しただけで選んだ置き方が既定へ戻る
        del qt_application
        dialog = PreferencesDialog(Preferences(media_split=mode))
        try:
            assert dialog.preferences().media_split == mode
        finally:
            dialog.deleteLater()
