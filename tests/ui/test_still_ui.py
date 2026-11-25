"""プレビューの 1 コマを編集の中で使う（#525）の画面の入口

止める命令・名前・AI の道具の中身は tests/core/test_still.py と tests/ai/test_still_tools.py で見る
ここではメニューと右クリックから正しい所へ届くか、人のメニューが窓を出さずに書いて置くかを見る
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from sashimono.core.commands import STILLS_FOLDER, AddClip, AddTrack, insert_media
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.timeline import TimelineView
from sashimono.ui.workspace import Preferences

FREEZE = "このコマで止める"


def _menu(view: TimelineView, clip: Clip) -> QMenu:
    view.resize(900, 300)
    band = next(
        b
        for b in view._layout.bands(view.project.timeline)
        if any(c.id == clip.id for c in b.track.clips)
    )
    x = int(view._layout.frame_to_x(clip.timeline_start + 5))
    return view.build_context_menu(QPoint(x, band.top + band.height // 2))


@pytest.fixture
def video_window(
    qt_application: QApplication, video_media: MediaItem, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    del qt_application
    project = Project.create()
    for command in insert_media(project, video_media):
        project = command.apply(project)
    created = MainWindow(project, confirm_unsaved=False)
    # 置いた素材は本物の動画ではない 解析と控えを頼まないようにする
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    yield created
    created.close()


def _video_clip(window: MainWindow) -> Clip:
    project = window.document.project
    return next(c for t in project.timeline.tracks for c in t.clips if t.kind is TrackKind.VIDEO)


class TestFreezeMenu:
    def test_it_is_offered_while_the_playhead_is_on_the_clip(
        self, video_window: MainWindow
    ) -> None:
        # 壊れると、動画のクリップの右クリックに〔このコマで止める〕が出ない
        clip = _video_clip(video_window)
        video_window.seek(30)
        labels = {a.text(): a.isEnabled() for a in _menu(video_window._timeline, clip).actions()}
        assert labels[FREEZE]

    def test_it_is_greyed_out_when_the_playhead_is_elsewhere(
        self, video_window: MainWindow
    ) -> None:
        # 隠すと在ることに気付けない 押しても止める所が無いので灰色で出す
        clip = _video_clip(video_window)
        video_window.seek(clip.timeline_end + 30)
        labels = {a.text(): a.isEnabled() for a in _menu(video_window._timeline, clip).actions()}
        assert labels[FREEZE] is False

    def test_choosing_it_splits_and_holds(self, video_window: MainWindow) -> None:
        # 壊れると、右クリックから選んでも止まらないか、取り消しで戻らない
        clip = _video_clip(video_window)
        video_window.seek(30)
        menu = _menu(video_window._timeline, clip)
        next(a for a in menu.actions() if a.text() == FREEZE).trigger()
        pictures = [
            c
            for t in video_window.document.project.timeline.tracks
            for c in t.clips
            if t.kind is TrackKind.VIDEO
        ]
        assert sorted((c.timeline_start, c.hold_at is not None) for c in pictures) == [
            (0, False),
            (30, True),
        ]
        video_window.undo()
        assert _video_clip(video_window) == clip


class TestStillToMedia:
    def test_an_unsaved_project_asks_to_save_first(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 保存していない作品にはまだ書く場所が無い アプリのキャッシュへ書くと、作品を移したときに
        # 素材が見つからなくなる
        del qt_application
        told: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "information", lambda _parent, _title, text: told.append(text)
        )
        created = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            created.still_to_media()
        finally:
            created.close()
        assert told and STILLS_FOLDER in told[0]
        assert created.document.project.media == ()

    @pytest.mark.skipif(sys.platform != "win32", reason="書く先を持てるのは Windows だけ")
    @pytest.mark.usefixtures("gpu")
    def test_placing_writes_the_png_and_puts_it_at_the_playhead(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、〔静止画をタイムラインへ置く〕が画面と違う絵を書くか、窓を出して止まるか、
        # 置く位置と長さがずれる
        del qt_application

        def popped(*_args: object, **_kwargs: object) -> object:
            pytest.fail("試験の途中で窓が開いた")

        for name in ("warning", "information"):
            monkeypatch.setattr(QMessageBox, name, popped)
        folder = tmp_path / "作品"
        folder.mkdir()
        track = Track(TrackKind.VIDEO, "V1")
        project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
        project = AddTrack(track).apply(project)
        project = AddClip(track.id, Clip(timeline_start=0, duration=60)).apply(project)
        window = MainWindow(project, path=folder / "作品.sme", confirm_unsaved=False)
        try:
            window._apply_preferences(replace(window._preferences, still_seconds=2))
            window.seek(10)
            window.still_to_media(place=True)
            (written,) = (folder / STILLS_FOLDER).iterdir()
            # 〔静止画を保存…〕と同じ描き方
            by_hand = window.save_snapshot(tmp_path / "人.png")
            assert by_hand is not None
            assert QImage(str(written)) == QImage(str(by_hand))
            still = window.document.project.media[-1]
            assert still.path == written
            placed = [
                c
                for t in window.document.project.timeline.tracks
                for c in t.clips
                if c.media_id == still.id
            ]
            assert [(c.timeline_start, c.duration) for c in placed] == [(10, 60)]
            # 取り消し 1 回で読み込みと置いた物が戻り、書いた PNG は残る
            window.undo()
            assert window.document.project == project
            assert written.is_file()
        finally:
            window.close()


class TestPreference:
    def test_the_default_matches_placing_a_still(self) -> None:
        # 壊れると、知らない人の静止画の長さがふつうに置いたときと食い違う
        assert Preferences().still_seconds == 5

    @pytest.mark.parametrize("value", [1, 12])
    def test_the_dialog_carries_it(self, qt_application: QApplication, value: int) -> None:
        # 壊れると、設定画面を開いて OK を押しただけで値が既定へ戻る
        del qt_application
        dialog = PreferencesDialog(Preferences(still_seconds=value))
        try:
            assert dialog.preferences().still_seconds == value
        finally:
            dialog.deleteLater()
