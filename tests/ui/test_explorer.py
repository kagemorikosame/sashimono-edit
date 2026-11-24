"""エクスプローラー（#521） 登録・外し・消えたフォルダ・絞り込み・下見と試聴・タイムラインへ置く

登録と見て回る所そのものは tests/core/test_material_library.py
AI の道具は tests/ai/test_library_tools.py
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from sashimono.core.material_library import (
    CATEGORY_IMAGE,
    CATEGORY_MUSIC,
    CATEGORY_SOUND_EFFECT,
    DEFAULT_AI_CATEGORIES,
    LibraryFolder,
)
from sashimono.core.model import Project
from sashimono.ui.explorer import ExplorerPanel
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import Preferences, PreferenceStore


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


@pytest.fixture
def sounds(tmp_path: Path) -> Path:
    root = tmp_path / "効果音"
    _touch(root / "ドン.wav")
    _touch(root / "爆発" / "ドカン.mp3")
    _touch(root / "説明.txt")
    return root


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[ExplorerPanel]:
    del qt_application
    created = ExplorerPanel()
    yield created
    created.deleteLater()


class TestRegistering:
    def test_registering_lists_the_files_and_tells_the_window(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 知らせが無いと、登録が設定に残らず、次に開いたときに消えている
        told: list[object] = []
        panel.folders_changed.connect(told.append)
        assert "登録されていません" in panel.status_text()
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        assert told == [(LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT),)]
        assert sorted(p.name for p in panel.visible_files()) == ["ドカン.mp3", "ドン.wav"]

    def test_removing_keeps_the_files(self, panel: ExplorerPanel, sounds: Path) -> None:
        # 壊れると、登録を外しただけで本人の素材が消えるか、外したことが設定に残らない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        told: list[object] = []
        panel.folders_changed.connect(told.append)
        panel._remove_button.click()
        assert told == [()]
        assert panel.visible_files() == []
        assert (sounds / "ドン.wav").exists()

    def test_a_vanished_folder_is_shown_not_raised(
        self, panel: ExplorerPanel, tmp_path: Path
    ) -> None:
        # 壊れると、外付けのドライブを外しただけでパネルが落ちるか、黙って空になる
        panel.set_folders([LibraryFolder(str(tmp_path / "無い"), CATEGORY_MUSIC)])
        assert "見つかりません" in panel.status_text()
        assert panel.visible_files() == []

    def test_the_filter_narrows_by_name_and_folder(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、数千の効果音から名前で絞れず、一覧を目で追うことになる
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        panel._filter.setText("爆発")
        assert [p.name for p in panel.visible_files()] == ["ドカン.mp3"]
        panel._filter.setText("")
        assert len(panel.visible_files()) == 2


class TestUsingAFile:
    def test_dragging_carries_the_file_like_the_os_explorer(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # タイムラインはファイルの URL を受けて読み込んで置く（取り消し 1 回で戻る） ID を載せると
        # まだ作品の素材でないので置けない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        assert panel.select_file(sounds / "ドン.wav")
        mime = panel.drag_data()
        assert [Path(url.toLocalFile()) for url in mime.urls()] == [sounds / "ドン.wav"]

    def test_the_place_button_asks_the_window(self, panel: ExplorerPanel, sounds: Path) -> None:
        # 壊れると、〔タイムラインへ〕を押しても何も置かれない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        asked: list[list[Path]] = []
        panel.place_requested.connect(asked.append)
        panel.select_file(sounds / "ドン.wav")
        panel._place_button.click()
        assert asked == [[sounds / "ドン.wav"]]

    def test_listening_plays_at_the_chosen_volume_and_stops(
        self, qt_application: QApplication, sounds: Path
    ) -> None:
        # 壊れると、設定した試聴の音量が効かないか、鳴り始めた音を止められない
        del qt_application
        played: list[tuple[Path, float]] = []
        stopped: list[bool] = []
        panel = ExplorerPanel(
            [LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT)],
            play=lambda path, volume: played.append((path, volume)),
            stop=lambda: stopped.append(True),
        )
        panel.set_preview_volume(40)
        panel.select_file(sounds / "ドン.wav")
        panel._listen.click()
        assert played == [(sounds / "ドン.wav", 0.4)]
        panel._listen.click()
        assert stopped == [True]
        assert panel._listen.text() == "試聴"
        panel.deleteLater()

    def test_pictures_get_a_preview(self, panel: ExplorerPanel, tmp_path: Path) -> None:
        # 下見が出ないと、画像の素材を名前だけで選ぶことになる
        folder = tmp_path / "画像"
        folder.mkdir()
        image = QImage(64, 36, QImage.Format.Format_RGBA8888)
        image.fill(0xFF3366FF)
        assert image.save(str(folder / "青.png"))
        panel.register(str(folder), CATEGORY_IMAGE)
        panel.select_file(folder / "青.png")
        assert not panel._preview.isHidden()
        pixmap = panel._preview.pixmap()
        assert pixmap is not None and not pixmap.isNull()
        assert not panel._listen.isEnabled()


class TestSettings:
    def test_round_trip_and_broken_values(self, tmp_path: Path) -> None:
        # 壊れた値で起動を止めない 項目ごとに既定へ戻す
        store = PreferenceStore(tmp_path / "preferences.json")
        plain = store.load()
        assert plain.explorer_folders == ()
        assert plain.explorer_ai_categories == DEFAULT_AI_CATEGORIES
        assert plain.explorer_ai_bgm is False
        assert plain.explorer_preview_volume == 100
        chosen = Preferences(
            explorer_folders=(LibraryFolder("D:/SE", CATEGORY_SOUND_EFFECT),),
            explorer_ai_categories=(CATEGORY_SOUND_EFFECT,),
            explorer_ai_bgm=True,
            explorer_preview_volume=30,
        )
        store.save(chosen)
        loaded = PreferenceStore(tmp_path / "preferences.json").load()
        assert loaded.explorer_folders == chosen.explorer_folders
        assert loaded.explorer_ai_categories == (CATEGORY_SOUND_EFFECT,)
        assert loaded.explorer_ai_bgm is True
        assert loaded.explorer_preview_volume == 30
        (tmp_path / "broken.json").write_text(
            json.dumps(
                {
                    "explorer_folders": "D:/SE",
                    "explorer_ai_categories": "music",
                    "explorer_ai_bgm": "yes",
                    "explorer_preview_volume": 900,
                }
            ),
            encoding="utf-8",
        )
        broken = PreferenceStore(tmp_path / "broken.json").load()
        assert broken.explorer_folders == ()
        assert broken.explorer_ai_categories == DEFAULT_AI_CATEGORIES
        assert broken.explorer_ai_bgm is False
        assert broken.explorer_preview_volume == 100


class TestTheWindow:
    def test_the_host_follows_the_settings_and_remembers_the_folders(
        self, qt_application: QApplication, sounds: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、設定画面で外した分類を AI が使い続けるか、登録したフォルダの場所が
        # 名前と場所を送らないと決めた人の所で伏せずに送られる
        del qt_application
        window = MainWindow(Project.create(), confirm_unsaved=False)
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, _prefs: None)
        try:
            folder = LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT)
            window._explorer.register(folder.path, folder.category)
            library = window.material_library
            assert library.folders == (folder,)
            assert library.ai_categories == DEFAULT_AI_CATEGORIES
            assert str(sounds) in window.remembered_names()
            settings = window.app_settings()
            preferences = settings["preferences"]
            assert isinstance(preferences, dict)
            assert preferences["explorer_folders"] == (
                {"path": str(sounds), "category": CATEGORY_SOUND_EFFECT},
            )
            window._apply_preferences(
                Preferences(explorer_folders=(), explorer_ai_categories=(CATEGORY_MUSIC,))
            )
            assert window._explorer.folders == ()
            assert window.material_library.ai_categories == (CATEGORY_MUSIC,)
        finally:
            window.close()
            window.deleteLater()

    def test_the_place_button_imports_at_the_playhead(
        self, qt_application: QApplication, sounds: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、置いた音が再生位置でなく作品の末尾へ行き、探して動かし直すことになる
        del qt_application
        window = MainWindow(Project.create(), confirm_unsaved=False)
        asked: list[tuple[list[Path], int | None]] = []
        monkeypatch.setattr(
            window,
            "import_media",
            lambda paths, *, at=None: asked.append((paths, at.frame if at else None)),
        )
        try:
            frame = window._timeline.playhead
            window._place_explorer_files([sounds / "ドン.wav"])
            assert asked == [([sounds / "ドン.wav"], frame)]
        finally:
            window.close()
            window.deleteLater()
