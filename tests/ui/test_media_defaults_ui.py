"""素材の置き方の既定（#524）の画面の入口

覚える・置く・保存する中身は core の試験（tests/core/test_media_defaults.py）で見る
ここでは右クリックから正しい所へ届くか、本人の置き場（ほかの作品の分）を設定どおりに
書くか、読み込みで写るかを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QMenu

from sashimono.core.commands import insert_media
from sashimono.core.io.media_memory import MediaMemory, fingerprint
from sashimono.core.model import (
    Clip,
    MediaDefaults,
    MediaItem,
    Project,
    Track,
    TrackKind,
    new_media_id,
)
from sashimono.effects.sources import TEXT
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_pool import MediaPoolWidget
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.timeline import TimelineView
from sashimono.ui.workspace import Preferences

REMEMBER = "この設定を素材の既定にする"
FORGET = "素材の既定を消す"


@pytest.fixture
def material(tmp_path: Path, video_media: MediaItem) -> MediaItem:
    """本物のファイルを指す素材 中身で見分けるので、ファイルが無いと覚えられない"""
    path = tmp_path / "立ち絵.mp4"
    path.write_bytes(bytes(range(256)) * 64)
    return replace(video_media, path=path)


def _placed(media: MediaItem) -> Project:
    project = Project.create(media=())
    for command in insert_media(project, media):
        project = command.apply(project)
    return project


@pytest.fixture
def window(
    qt_application: QApplication, material: MediaItem, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_placed(material), confirm_unsaved=False)
    # 置いた素材は本物の動画ではない 解析と控えを頼まないようにする
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    yield created
    created.close()


def _media_clip(window: MainWindow) -> Clip:
    project = window.document.project
    return next(c for t in project.timeline.tracks for c in t.clips if c.media_id is not None)


def _menu(view: TimelineView, clip: Clip) -> QMenu:
    """``clip`` の上の右クリックメニュー"""
    view.resize(900, 300)
    band = next(
        b
        for b in view._layout.bands(view.project.timeline)
        if any(c.id == clip.id for c in b.track.clips)
    )
    x = int(view._layout.frame_to_x(clip.timeline_start + 5))
    return view.build_context_menu(QPoint(x, band.top + band.height // 2))


def _menu_labels(view: TimelineView, clip: Clip) -> dict[str, bool]:
    menu = _menu(view, clip)
    return {action.text(): action.isEnabled() for action in menu.actions() if action.text()}


def _remembered_outside(material: MediaItem) -> MediaDefaults | None:
    key = fingerprint(material.path)
    assert key is not None
    return MediaMemory().recall(key)


class TestTimelineMenu:
    def test_a_media_clip_offers_to_remember(self, window: MainWindow) -> None:
        # 壊れると、素材のクリップの右クリックに〔既定にする〕が出ず、覚える手段が無い
        assert _menu_labels(window._timeline, _media_clip(window))[REMEMBER]

    def test_a_text_clip_does_not(self, qt_application: QApplication) -> None:
        # 素材を持たないクリップに出すと、押しても覚える先が無い
        del qt_application
        base = Project.create()
        text = Clip(timeline_start=0, duration=30, source=TEXT.create())
        track = Track(TrackKind.VIDEO, "V1", (text,))
        project = base.with_timeline(replace(base.timeline, tracks=(track,)))
        created = MainWindow(project, confirm_unsaved=False)
        try:
            assert REMEMBER not in _menu_labels(created._timeline, text)
        finally:
            created.close()

    def test_the_menu_asks_the_window(self, window: MainWindow) -> None:
        # 窓へ伝わらないと、押しても何も覚えない
        asked: list[str] = []
        window._timeline.media_defaults_requested.connect(asked.append)
        clip = _media_clip(window)
        menu = _menu(window._timeline, clip)
        next(a for a in menu.actions() if a.text() == REMEMBER).trigger()
        assert asked == [str(clip.id)]


class TestWindow:
    def test_remembering_fills_the_project_and_the_memory(
        self, window: MainWindow, material: MediaItem
    ) -> None:
        # 壊れると、覚えたつもりの既定が作品にもほかの作品にも残らない
        window.remember_media_defaults(_media_clip(window).id)
        defaults = window.document.project.media[0].defaults
        assert defaults is not None
        assert _remembered_outside(material) == defaults

    def test_with_the_setting_off_only_the_project_remembers(
        self, window: MainWindow, material: MediaItem
    ) -> None:
        # 切ったのに作品の外へ書くと、別の作品で思わぬ置き方になる
        window._apply_preferences(replace(window._preferences, media_defaults_global=False))
        window.remember_media_defaults(_media_clip(window).id)
        assert window.document.project.media[0].defaults is not None
        assert _remembered_outside(material) is None

    def test_one_undo_takes_back_the_project_defaults(self, window: MainWindow) -> None:
        # 壊れると、間違えて押した〔既定にする〕を取り消しで戻せない
        window.remember_media_defaults(_media_clip(window).id)
        window.undo()
        assert window.document.project.media[0].defaults is None

    def test_clearing_forgets_both(self, window: MainWindow, material: MediaItem) -> None:
        # 壊れると、〔素材の既定を消す〕を押してもほかの作品ではまだ効く
        window.remember_media_defaults(_media_clip(window).id)
        window.clear_media_defaults(window.document.project.media[0].id)
        assert window.document.project.media[0].defaults is None
        assert _remembered_outside(material) is None

    def test_imported_media_receives_the_remembered_defaults(
        self,
        window: MainWindow,
        material: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 別の作品で同じファイルを読み込んだときにも効くのが、アプリ全体に覚える訳
        window.remember_media_defaults(_media_clip(window).id)
        remembered = window.document.project.media[0].defaults
        monkeypatch.setattr(
            main_window_module,
            "probe_media",
            lambda path: replace(material, path=path, id=new_media_id(), defaults=None),
        )
        window.import_media([material.path])
        assert window.wait_for_imports()
        imported = window.document.project.media[-1]
        assert imported.id != material.id
        assert imported.defaults == remembered

    def test_imports_ignore_the_memory_when_the_setting_is_off(
        self,
        window: MainWindow,
        material: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、設定を切ってもほかの作品の既定が勝手に当たる
        window.remember_media_defaults(_media_clip(window).id)
        window._apply_preferences(replace(window._preferences, media_defaults_global=False))
        monkeypatch.setattr(
            main_window_module,
            "probe_media",
            lambda path: replace(material, path=path, id=new_media_id(), defaults=None),
        )
        window.import_media([material.path])
        assert window.wait_for_imports()
        assert window.document.project.media[-1].defaults is None


class TestMediaPool:
    def test_forgetting_is_greyed_out_until_something_is_remembered(
        self, qt_application: QApplication, material: MediaItem
    ) -> None:
        # 隠さずに灰色で出す 隠すと、既定を覚えられることに気付けない
        del qt_application
        pool = MediaPoolWidget(Project.create(media=(material,)))
        forget = next(a for a in pool.build_menu(material.id).actions() if a.text() == FORGET)
        assert not forget.isEnabled()

        remembered = replace(material, defaults=MediaDefaults(opacity=0.5))
        pool = MediaPoolWidget(Project.create(media=(remembered,)))
        asked: list[str] = []
        pool.defaults_clear_requested.connect(asked.append)
        forget = next(a for a in pool.build_menu(material.id).actions() if a.text() == FORGET)
        assert forget.isEnabled()
        forget.trigger()
        assert asked == [str(material.id)]


class TestPreference:
    def test_the_default_remembers_across_projects(self) -> None:
        # 本人が〔既定にする〕を選んだのだから、別の作品でも効くと思うのが自然
        assert Preferences().media_defaults_global

    @pytest.mark.parametrize("value", [True, False])
    def test_the_dialog_carries_it(self, qt_application: QApplication, value: bool) -> None:
        # 画面が値を返さないと、設定を開いて OK を押しただけで既定へ戻る
        del qt_application
        dialog = PreferencesDialog(Preferences(media_defaults_global=value))
        try:
            assert dialog.preferences().media_defaults_global is value
        finally:
            dialog.deleteLater()
