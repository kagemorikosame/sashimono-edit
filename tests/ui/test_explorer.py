"""エクスプローラー（#521） 木と中身・列と並べ替え・アイコン・下見のプレーヤー・置く・読み込む

登録と見て回る所そのものは tests/core/test_material_library.py
下見の再生と測る所は tests/engine/test_library_sound.py
AI の道具は tests/ai/test_library_tools.py
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QModelIndex, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from sashimono.core.material_library import (
    CATEGORY_MUSIC,
    CATEGORY_SOUND_EFFECT,
    DEFAULT_AI_CATEGORIES,
    LibraryFile,
    LibraryFolder,
    Measure,
    MeasureBook,
)
from sashimono.core.model import Project
from sashimono.engine.audio.library_sound import PreviewPlayer
from sashimono.ui import explorer as explorer_module
from sashimono.ui.explorer import (
    COLUMNS,
    DOUBLE_CLICK_TIMELINE,
    FAVORITES,
    STAR_OFF,
    STAR_ON,
    VIEW_ICONS,
    ExplorerPanel,
)
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import (
    EXPLORER_DOUBLE_CLICKS,
    EXPLORER_ICON_SIZES,
    EXPLORER_VIEWS,
    Preferences,
    PreferenceStore,
)
from tests.check_media import tone, write_tone
from tests.waiting import wait_until

_LENGTH = COLUMNS.index("長さ")
_LOUDNESS = COLUMNS.index("大きさ")


class _Sink:
    def write(self, samples: np.ndarray) -> None:
        del samples

    def close(self, drain: bool) -> None:
        del drain


def _quiet_player() -> PreviewPlayer:
    """機械の出口から鳴らさない下見 試験で本人の机の上に音を出さない"""
    return PreviewPlayer(sink=lambda _rate, _channels: _Sink())


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _lengths(path: Path) -> Measure:
    seconds = {"ドン.wav": 0.8, "ドカン.mp3": 3.0, "長い.wav": 95.0, "中.wav": 3.0}.get(
        path.name, 1.0
    )
    return Measure(seconds=seconds, has_audio=True, has_video=False, loudness=-20.0 + seconds)


@pytest.fixture
def sounds(tmp_path: Path) -> Path:
    root = tmp_path / "効果音"
    _touch(root / "ドン.wav")
    _touch(root / "長い.wav")
    _touch(root / "中.wav")
    _touch(root / "爆発" / "ドカン.mp3")
    _touch(root / "説明.txt")
    return root


def _process() -> None:
    QApplication.processEvents()


def _settle(panel: ExplorerPanel) -> None:
    """裏で読む中身と下見の調べが終わるまで待つ（どちらも画面のスレッドを止めずに裏で行う）"""
    wait_until(lambda: not panel.busy, "中身と下見", step=_process)


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[ExplorerPanel]:
    del qt_application
    created = ExplorerPanel(player=_quiet_player(), measure=_lengths)
    created.resize(900, 600)
    yield created
    created.shutdown()
    created.deleteLater()


class TestTheTreeAndTheContents:
    def test_registering_builds_the_tree_and_lists_one_level(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 知らせが無いと、登録が設定に残らず、次に開いたときに消えている
        # 下まで全部を並べると、根を開くたびに数万のファイルを待つ
        told: list[object] = []
        panel.folders_changed.connect(told.append)
        assert "登録されていません" in panel.status_text()
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        assert told == [(LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT),)]
        assert sorted(p.name for p in panel.visible_files()) == ["ドン.wav", "中.wav", "長い.wav"]
        assert panel.location_text() == "効果音 › 効果音"
        assert panel.open_folder(str(sounds / "爆発"))
        _settle(panel)
        assert [p.name for p in panel.visible_files()] == ["ドカン.mp3"]
        assert panel.location_text().endswith("› 爆発")

    def test_the_filter_searches_below(self, panel: ExplorerPanel, sounds: Path) -> None:
        # 壊れると、数千の効果音から名前で探すのに、下のフォルダを 1 つずつ開くことになる
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel._filter.setText("ドカン")
        _settle(panel)
        assert [p.name for p in panel.visible_files()] == ["ドカン.mp3"]
        panel._filter.setText("")
        _settle(panel)
        panel._kind_box.setCurrentIndex(panel._kind_box.findData("video"))
        _settle(panel)
        assert panel.visible_files() == []

    def test_removing_keeps_the_files(self, panel: ExplorerPanel, sounds: Path) -> None:
        # 壊れると、登録を外しただけで本人の素材が消えるか、外したことが設定に残らない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
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
        _settle(panel)
        assert "見つかりません" in panel.status_text()
        root = panel._tree.topLevelItem(0)
        item = root.child(0) if root is not None else None
        assert item is not None and item.text(0).endswith("（見つかりません）")

    def test_columns_fill_in_the_background_and_sort_by_number(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、長さと大きさの列が空のままか、文字の順（95 秒が 3 秒の前）に並ぶ
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        # 測る仕事は 2 本で並ぶので、どちらが先に届くかは決まらない 両方を待つ
        wait_until(
            lambda: all(
                panel.cell(sounds / name, _LENGTH) for name in ("長い.wav", "中.wav", "ドン.wav")
            ),
            "長さの列",
            step=_process,
        )
        assert panel.cell(sounds / "ドン.wav", _LENGTH) == "0.8 秒"
        assert panel.cell(sounds / "長い.wav", _LENGTH) == "1:35"
        assert panel.cell(sounds / "長い.wav", _LOUDNESS) == "75.0 LUFS"
        panel._table.sortByColumn(_LENGTH, Qt.SortOrder.DescendingOrder)
        # 95 秒・3 秒・0.8 秒 文字の順（「1:35」「3.0 秒」「0.8 秒」）とは違う並びになる
        assert [p.name for p in panel.visible_files()] == ["長い.wav", "中.wav", "ドン.wav"]

    def test_icons_get_waveforms_made_in_the_background(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、アイコン表示で音の素材がみな同じ印になり、どんな音かが目で分からない
        # 作った波形を覚えないと、開くたびに全部の音を読み直す
        del qt_application
        root = tmp_path / "音"
        write_tone(_touch(root / "a.wav"), tone(1.0, -12.0))
        made: list[Path] = []
        original = explorer_module.sound_peaks

        def counted(path: Path, columns: int) -> np.ndarray | None:
            made.append(path)
            return original(path, columns)

        monkeypatch.setattr(explorer_module, "sound_peaks", counted)
        thumbs = tmp_path / "thumbs"

        def open_icons() -> None:
            panel = ExplorerPanel(
                [LibraryFolder(str(root), CATEGORY_MUSIC)],
                player=_quiet_player(),
                measure=None,
                thumbs=thumbs,
            )
            panel.set_view_mode(VIEW_ICONS)
            _settle(panel)
            wait_until(lambda: bool(panel._icons), "波形のアイコン", step=_process)
            panel.shutdown()
            panel.deleteLater()

        open_icons()
        open_icons()
        assert made == [root / "a.wav"]
        assert list(thumbs.iterdir())


class TestUsingAFile:
    def test_dragging_carries_the_file_like_the_os_explorer(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # タイムラインはファイルの URL を受けて読み込んで置く（取り消し 1 回で戻る） ID を載せると
        # まだ作品の素材でないので置けない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        assert panel.select_file(sounds / "ドン.wav")
        _settle(panel)
        mime = panel.drag_data()
        assert [Path(url.toLocalFile()) for url in mime.urls()] == [sounds / "ドン.wav"]

    def test_the_place_button_and_the_menu_ask_the_window(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、〔タイムラインへ〕や右クリックを押しても何も置かれず、読み込まれない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        placed: list[list[Path]] = []
        pooled: list[list[Path]] = []
        panel.place_requested.connect(placed.append)
        panel.import_requested.connect(pooled.append)
        panel.select_file(sounds / "ドン.wav")
        _settle(panel)
        panel._place_button.click()
        menu = panel.build_menu([sounds / "ドン.wav"])
        names = [action.text() for action in menu.actions() if action.text()]
        assert names == [
            "タイムラインへ置く",
            "素材一覧へ読み込む",
            "エクスプローラーで開く",
            "お気に入りに入れる",
            "お気に入りから外す",
        ]
        menu.actions()[0].trigger()
        menu.actions()[1].trigger()
        assert placed == [[sounds / "ドン.wav"], [sounds / "ドン.wav"]]
        assert pooled == [[sounds / "ドン.wav"]]
        menu.deleteLater()

    def test_double_click_plays_or_places_as_chosen(
        self, panel: ExplorerPanel, tmp_path: Path
    ) -> None:
        # 壊れると、選んだ好みと違う動きになり、下見したいだけなのにタイムラインへ置かれる
        root = tmp_path / "音"
        path = write_tone(_touch(root / "a.wav"), tone(2.0, -12.0))
        panel.register(str(root), CATEGORY_MUSIC)
        _settle(panel)
        panel.select_file(path)
        _settle(panel)
        assert panel.preview.player.path == path
        placed: list[list[Path]] = []
        panel.place_requested.connect(placed.append)
        index = panel._table.currentIndex()
        panel._on_double_click(index)
        assert panel.preview.player.playing and placed == []
        panel.preview.stop()
        panel.set_double_click(DOUBLE_CLICK_TIMELINE)
        panel._on_double_click(index)
        assert placed == [[path]]

    def test_space_toggles_the_preview_only_inside(
        self, panel: ExplorerPanel, tmp_path: Path
    ) -> None:
        # 壊れると、Space で作品の再生が始まって下見と 2 つの音が重なるか、下見が止められない
        root = tmp_path / "音"
        path = write_tone(_touch(root / "a.wav"), tone(2.0, -12.0))
        panel.register(str(root), CATEGORY_MUSIC)
        _settle(panel)
        panel.select_file(path)
        _settle(panel)
        override = QKeyEvent(
            QEvent.Type.ShortcutOverride, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "
        )
        assert panel.eventFilter(panel._table, override)
        assert override.isAccepted()
        press = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "
        )
        assert panel.eventFilter(panel._table, press)
        assert panel.preview.player.playing
        panel.eventFilter(panel._table, press)
        assert not panel.preview.player.playing

    def test_the_preview_shows_pictures(self, panel: ExplorerPanel, tmp_path: Path) -> None:
        # 壊れると、画像や動画の下見が黒いまま、名前だけで選ぶことになる
        from PySide6.QtGui import QImage

        root = tmp_path / "画像"
        root.mkdir()
        image = QImage(64, 36, QImage.Format.Format_RGBA8888)
        image.fill(0xFF3366FF)
        assert image.save(str(root / "青.png"))
        panel.register(str(root), "image")
        _settle(panel)
        panel.select_file(root / "青.png")
        _settle(panel)
        wait_until(panel.preview._picture.has_picture, "画像の下見", step=_process)
        # 静止画は再生しない
        assert not panel.preview.play_button.isEnabled()

    def test_narrow_panels_stack_the_tree_above(self, panel: ExplorerPanel) -> None:
        # 壊れると、素材一覧と重ねた狭い左の列で木と中身が横に並び、名前の列が欠ける
        panel.resize(400, 700)
        panel.show()
        _process()
        assert panel._outer.orientation() == Qt.Orientation.Vertical
        panel.resize(1000, 600)
        _process()
        assert panel._outer.orientation() == Qt.Orientation.Horizontal
        panel.hide()


class TestSettings:
    def test_the_values_match_the_panel(self) -> None:
        # 壊れると、設定に覚えた表示やダブルクリックの値をパネルが知らず、既定へ戻る
        assert EXPLORER_VIEWS == explorer_module.VIEW_MODES
        assert EXPLORER_ICON_SIZES == explorer_module.ICON_SIZE_RANGE
        assert tuple(explorer_module.DOUBLE_CLICK_ACTIONS) == EXPLORER_DOUBLE_CLICKS

    def test_round_trip_and_broken_values(self, tmp_path: Path) -> None:
        # 壊れた値で起動を止めない 項目ごとに既定へ戻す
        store = PreferenceStore(tmp_path / "preferences.json")
        plain = store.load()
        assert plain.explorer_folders == ()
        assert plain.explorer_ai_categories == DEFAULT_AI_CATEGORIES
        assert plain.explorer_ai_bgm is False
        assert plain.explorer_preview_volume == 100
        assert plain.explorer_view == "list"
        assert plain.explorer_double_click == "preview"
        chosen = Preferences(
            explorer_folders=(LibraryFolder("D:/SE", CATEGORY_SOUND_EFFECT),),
            explorer_ai_categories=(CATEGORY_SOUND_EFFECT,),
            explorer_ai_bgm=True,
            explorer_preview_volume=30,
            explorer_view="icons",
            explorer_icon_size=200,
            explorer_double_click="timeline",
        )
        store.save(chosen)
        loaded = PreferenceStore(tmp_path / "preferences.json").load()
        assert loaded.explorer_folders == chosen.explorer_folders
        assert loaded.explorer_ai_categories == (CATEGORY_SOUND_EFFECT,)
        assert loaded.explorer_ai_bgm is True
        assert loaded.explorer_preview_volume == 30
        assert (loaded.explorer_view, loaded.explorer_icon_size) == ("icons", 200)
        assert loaded.explorer_double_click == "timeline"
        (tmp_path / "broken.json").write_text(
            json.dumps(
                {
                    "explorer_folders": "D:/SE",
                    "explorer_ai_categories": "music",
                    "explorer_ai_bgm": "yes",
                    "explorer_preview_volume": 900,
                    "explorer_view": "grid",
                    "explorer_icon_size": 9000,
                    "explorer_double_click": "delete",
                }
            ),
            encoding="utf-8",
        )
        broken = PreferenceStore(tmp_path / "broken.json").load()
        assert broken.explorer_folders == ()
        assert broken.explorer_ai_categories == DEFAULT_AI_CATEGORIES
        assert broken.explorer_ai_bgm is False
        assert broken.explorer_preview_volume == 100
        assert (broken.explorer_view, broken.explorer_icon_size) == ("list", 128)
        assert broken.explorer_double_click == "preview"


class TestTheWindow:
    def test_the_host_follows_the_settings_and_remembers_the_folders(
        self, qt_application: QApplication, sounds: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、設定画面で外した分類を AI が使い続けるか、登録したフォルダの場所が
        # 名前と場所を送らないと決めた人の所で伏せずに送られる
        del qt_application
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, _prefs: None)
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            folder = LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT)
            window._explorer.register(folder.path, folder.category)
            _settle(window._explorer)
            library = window.material_library
            assert library.folders == (folder,)
            assert library.ai_categories == DEFAULT_AI_CATEGORIES
            assert str(sounds) in window.remembered_names()
            preferences = window.app_settings()["preferences"]
            assert isinstance(preferences, dict)
            assert preferences["explorer_folders"] == [
                {"path": str(sounds), "category": CATEGORY_SOUND_EFFECT},
            ]
            # AI に使わせない分類のフォルダは、設定の写しにも載せない
            window._explorer.register(str(sounds.parent), "video")
            listed = window.app_settings()["preferences"]
            assert isinstance(listed, dict)
            assert [f["category"] for f in listed["explorer_folders"]] == [CATEGORY_SOUND_EFFECT]
            window._apply_preferences(
                Preferences(explorer_ai_categories=(CATEGORY_MUSIC,), explorer_view=VIEW_ICONS)
            )
            assert window._explorer.folders == ()
            assert window._explorer.view_mode == VIEW_ICONS
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
        asked: list[tuple[list[Path], int | None, bool]] = []
        monkeypatch.setattr(
            window,
            "import_media",
            lambda paths, *, at=None, pool_only=False: asked.append(
                (paths, at.frame if at else None, pool_only)
            ),
        )
        try:
            frame = window._timeline.playhead
            window._place_explorer_files([sounds / "ドン.wav"])
            window._explorer.import_requested.emit([sounds / "ドン.wav"])
            assert asked == [
                ([sounds / "ドン.wav"], frame, False),
                ([sounds / "ドン.wav"], None, True),
            ]
        finally:
            window.close()
            window.deleteLater()

    def test_reading_into_the_pool_does_not_place(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、〔素材一覧へ読み込む〕でもタイムラインの末尾に置かれ、外す手間が要る
        del qt_application
        path = write_tone(tmp_path / "音.wav", tone(1.0, -12.0))
        window = MainWindow(Project.create(), confirm_unsaved=False)
        monkeypatch.setattr(window, "_request_proxy", lambda _media: None)
        monkeypatch.setattr(window._analyzer, "request", lambda _media, **_kwargs: None)
        try:
            window.import_media([path], pool_only=True)
            wait_until(lambda: not window.importing, "読み込み", step=_process)
            project = window.document.project
            assert [media.path for media in project.media] == [path]
            assert all(not track.clips for track in project.timeline.tracks)
            window.document.undo()
            assert window.document.project.media == ()
        finally:
            window.close()
            window.deleteLater()


class TestFavorites:
    def test_marking_files_and_folders_puts_them_on_top(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、いつも使う効果音を毎回フォルダを辿って探すことになる
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        told: list[object] = []
        panel.favorites_changed.connect(told.append)
        star = COLUMNS.index("☆")
        assert panel.cell(sounds / "ドン.wav", star) == STAR_OFF
        # 一覧の印の列を押すと入れる・外すが切り替わる
        panel.select_file(sounds / "ドン.wav")
        _settle(panel)
        row = panel._table.currentIndex().row()
        panel._on_cell_clicked(panel._proxy.index(row, star))
        _settle(panel)
        assert told == [(str(sounds / "ドン.wav"),)]
        assert panel.cell(sounds / "ドン.wav", star) == STAR_ON
        menu = panel.build_tree_menu(sounds / "爆発")
        enabled = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
        assert enabled == {"お気に入りに入れる": True, "お気に入りから外す": False}
        menu.actions()[1].trigger()
        _settle(panel)
        menu.deleteLater()
        root = panel._tree.topLevelItem(0)
        assert root is not None and root.text(0) == f"{STAR_ON} {FAVORITES}"
        # 根はお気に入りのファイル、フォルダは木から中へ辿れる
        panel._tree.setCurrentItem(root)
        _settle(panel)
        assert [p.name for p in panel.visible_files()] == ["ドン.wav"]
        folder = root.child(0)
        assert folder is not None and folder.text(0) == "爆発"
        panel._tree.setCurrentItem(folder)
        _settle(panel)
        assert [p.name for p in panel.visible_files()] == ["ドカン.mp3"]
        assert panel.location_text() == f"{FAVORITES} › 爆発"

    def test_vanished_favorites_stay_and_say_so(self, panel: ExplorerPanel, tmp_path: Path) -> None:
        # 壊れると、外付けのドライブを外しただけでお気に入りが消えるか、パネルが落ちる
        panel.set_favorites([str(tmp_path / "無い"), str(tmp_path / "無い.wav")])
        _settle(panel)
        root = panel._tree.topLevelItem(0)
        assert root is not None
        child = root.child(0)
        assert child is not None and child.text(0).endswith("（見つかりません）")
        panel._tree.setCurrentItem(root)
        _settle(panel)
        # 消えた物はファイルかフォルダか分からない どちらも木に残し、数も知らせる
        assert root.childCount() == 2
        assert "2 個が見つかりません" in panel.status_text()
        assert len(panel.favorites) == 2

    def test_icons_show_the_mark_too(self, panel: ExplorerPanel, sounds: Path) -> None:
        # 壊れると、アイコン表示に切り替えたときにどれがお気に入りか分からない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.set_favorite([sounds / "ドン.wav"], True)
        _settle(panel)
        panel.set_view_mode(VIEW_ICONS)
        _settle(panel)
        panel.select_file(sounds / "ドン.wav")
        _settle(panel)
        index = panel._icons_view.currentIndex()
        option = QStyleOptionViewItem()
        panel._icons_view.itemDelegate().initStyleOption(option, index)  # type: ignore[attr-defined]
        assert option.text.startswith(STAR_ON)


class TestLastPlace:
    def test_the_window_reopens_where_it_was(
        self, qt_application: QApplication, sounds: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、起動するたびに木の先頭から辿り直すことになる
        del qt_application
        saved: list[Preferences] = []
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, prefs: saved.append(prefs))
        monkeypatch.setattr(
            PreferenceStore,
            "load",
            lambda _self: Preferences(
                explorer_folders=(LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT),)
            ),
        )
        first = MainWindow(Project.create(), confirm_unsaved=False)
        # 窓はエクスプローラーを初めて見せるまで中身を読まない 見せたのと同じにする
        first._explorer.set_active(True)
        first._explorer.open_folder(str(sounds / "爆発"))
        _settle(first._explorer)
        first._explorer.select_file(sounds / "爆発" / "ドカン.mp3")
        _settle(first._explorer)
        first.close()
        first.deleteLater()
        assert saved
        remembered = saved[-1]
        assert remembered.explorer_last_folder == str(sounds / "爆発")
        assert remembered.explorer_last_file == str(sounds / "爆発" / "ドカン.mp3")
        monkeypatch.setattr(PreferenceStore, "load", lambda _self: remembered)
        second = MainWindow(Project.create(), confirm_unsaved=False)
        second._explorer.set_active(True)
        try:
            _settle(second._explorer)
            assert second._explorer.location_text().endswith("› 爆発")
            assert second._explorer.selected_files() == [sounds / "爆発" / "ドカン.mp3"]
        finally:
            second.close()
            second.deleteLater()

    def test_a_vanished_place_opens_from_the_top(
        self, panel: ExplorerPanel, sounds: Path, tmp_path: Path
    ) -> None:
        # 壊れると、前回のフォルダが消えていたときに空の場所が開くか落ちる
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        before = panel.location_text()
        panel.restore_location(str(tmp_path / "無い"), "")
        _settle(panel)
        assert panel.location_text() == before
        panel.restore_location(str(sounds / "爆発"), "")
        _settle(panel)
        assert panel.location_text().endswith("› 爆発")


class TestWhileReading:
    def test_the_old_list_cannot_be_placed_while_reading(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、応答の遅い場所を開いている間に〔タイムラインへ〕で前の場所の物が置かれる
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.select_file(sounds / "ドン.wav")
        _settle(panel)
        placed: list[list[Path]] = []
        panel.place_requested.connect(placed.append)
        panel._lister.forget()
        panel._show_files()
        assert panel.listing
        panel._place_selected()
        assert placed == []
        assert not panel._place_button.isEnabled()
        _settle(panel)
        panel._place_selected()
        assert placed == [[sounds / "ドン.wav"]]

    def test_restoring_an_open_folder_selects_at_once(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、もう開いていた場所の前回のファイルが選ばれず、次の読み直しまで残る
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.restore_location(str(sounds), str(sounds / "長い.wav"))
        _settle(panel)
        assert panel.selected_files() == [sounds / "長い.wav"]

    def test_stopping_cancels_a_play_asked_while_opening(
        self, panel: ExplorerPanel, tmp_path: Path
    ) -> None:
        # 壊れると、作品を鳴らし始めて下見を止めた後に、調べ終えた下見が鳴り出す
        root = tmp_path / "音"
        path = write_tone(_touch(root / "a.wav"), tone(2.0, -12.0))
        panel.register(str(root), CATEGORY_MUSIC)
        _settle(panel)
        panel._preview.show_file(None)
        panel._preview.show_file(path)
        panel.preview.play()
        panel.preview.stop()
        _settle(panel)
        assert panel.preview.player.path == path
        assert not panel.preview.player.playing


class TestThirdReview:
    def test_removing_a_favorite_keeps_the_favorites_open(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、お気に入りの根で 1 つ外すたびに既定の一覧へ戻り、残りを整理しづらい
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.set_favorite([sounds / "ドン.wav", sounds / "長い.wav"], True)
        _settle(panel)
        root = panel._tree.topLevelItem(0)
        assert root is not None
        panel._tree.setCurrentItem(root)
        _settle(panel)
        panel.set_favorite([sounds / "ドン.wav"], False)
        _settle(panel)
        assert panel.location_text() == FAVORITES
        assert [p.name for p in panel.visible_files()] == ["長い.wav"]

    def test_overlapping_registrations_show_a_file_once(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、親と子を同じ分類で登録したとき、分類の根で同じ行が 2 つ並ぶ
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.register(str(sounds / "爆発"), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        root = panel._tree.topLevelItem(0)
        assert root is not None
        panel._tree.setCurrentItem(root)
        _settle(panel)
        names = [p.name for p in panel.visible_files()]
        assert names.count("ドカン.mp3") == 1

    def test_the_settings_copy_hides_places_inside_hidden_categories(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、AI に使わせない分類のフォルダの中の場所が、お気に入りと前回の場所から外へ出る
        del qt_application
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, _prefs: None)
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            movie = str(tmp_path / "動画" / "空.mp4")
            sound = str(tmp_path / "SE" / "ドン.wav")
            window._apply_preferences(
                Preferences(
                    explorer_folders=(
                        LibraryFolder(str(tmp_path / "動画"), "video"),
                        LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT),
                    ),
                    explorer_favorite_paths=(movie, sound),
                    explorer_last_folder=str(tmp_path / "動画"),
                    explorer_last_file=movie,
                )
            )
            copied = window.app_settings()["preferences"]
            assert isinstance(copied, dict)
            assert copied["explorer_favorite_paths"] == [sound]
            assert copied["explorer_last_folder"] == ""
            assert copied["explorer_last_file"] == ""
        finally:
            window.close()
            window.deleteLater()


class TestFourthReview:
    def test_a_category_root_stays_chosen_after_marking(
        self, panel: ExplorerPanel, sounds: Path, tmp_path: Path
    ) -> None:
        # 壊れると、分類の根で印を付けるたびに選びが外れ、すべてのフォルダの一覧に戻る
        music = tmp_path / "曲"
        _touch(music / "のんびり.wav")
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.register(str(music), CATEGORY_MUSIC)
        _settle(panel)
        root = panel._category_root(CATEGORY_SOUND_EFFECT)
        assert root is not None
        panel._tree.setCurrentItem(root)
        _settle(panel)
        panel.set_favorite([sounds / "ドン.wav"], True)
        _settle(panel)
        assert panel.location_text() == "効果音"
        assert all(p.parent != music for p in panel.visible_files())

    def test_several_chosen_files_clear_the_preview(
        self, panel: ExplorerPanel, tmp_path: Path
    ) -> None:
        # 壊れると、1 つを下見した後に別の 2 つを選んで Space を押すと、選んでいない音が鳴る
        root = tmp_path / "音"
        a = write_tone(_touch(root / "a.wav"), tone(1.0, -12.0))
        b = write_tone(_touch(root / "b.wav"), tone(1.0, -12.0))
        c = write_tone(_touch(root / "c.wav"), tone(1.0, -12.0))
        panel.register(str(root), CATEGORY_MUSIC)
        _settle(panel)
        panel.select_file(a)
        _settle(panel)
        assert panel.preview.player.path == a
        panel.select_files([b, c])
        _settle(panel)
        assert panel.preview.player.path is None

    def test_a_nested_registration_lists_with_its_own_category(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、親を効果音・子を画像で登録したとき、子の中の音が親の効果音として出る
        _touch(sounds / "絵" / "隠れた音.wav")
        _touch(sounds / "絵" / "青.png")
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.register(str(sounds / "絵"), "image")
        _settle(panel)
        # すべてのフォルダ（木で何も選ばない）
        panel._tree.setCurrentIndex(QModelIndex())
        _settle(panel)
        names = [p.name for p in panel.visible_files()]
        assert "隠れた音.wav" not in names
        assert "青.png" in names

    def test_files_that_cannot_be_measured_are_skipped_next_time(
        self, qt_application: QApplication, sounds: Path
    ) -> None:
        # 壊れると、測れない物が頭に並ぶ一覧で〔読み直す〕を押しても、同じ物をまた測りに行く
        del qt_application
        asked: list[Path] = []

        def broken(path: Path) -> Measure | None:
            asked.append(path)
            return None

        panel = ExplorerPanel(player=_quiet_player(), measure=broken)
        try:
            panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
            _settle(panel)
            wait_until(lambda: len(asked) >= 3, "測る仕事", step=_process)
            wait_until(lambda: not panel._worker.busy, "測り終わり", step=_process)
            _process()
            count = len(asked)
            panel.rescan()
            _settle(panel)
            _process()
            assert len(asked) == count
        finally:
            panel.shutdown()
            panel.deleteLater()

    def test_a_dotted_registration_is_judged_by_its_real_depth(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、`..` を含む書き方の親が子より深いと数えられ、使わせない子の中の場所が外へ出る
        del qt_application
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, _prefs: None)
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            parent = tmp_path / "親"
            dotted = str(parent / "a" / "b" / ".." / "..")
            movie = str(parent / "動画" / "空.mp4")
            window._apply_preferences(
                Preferences(
                    explorer_folders=(
                        LibraryFolder(dotted, CATEGORY_SOUND_EFFECT),
                        LibraryFolder(str(parent / "動画"), "video"),
                    ),
                    explorer_favorite_paths=(movie,),
                )
            )
            copied = window.app_settings()["preferences"]
            assert isinstance(copied, dict)
            assert copied["explorer_favorite_paths"] == []
        finally:
            window.close()
            window.deleteLater()


class TestFifthReview:
    def test_a_still_picture_does_not_stop_the_work(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 壊れると、作品の再生中に画像をダブルクリックすると作品が止まり、どちらも鳴らない
        del qt_application
        from PySide6.QtGui import QImage

        stopped: list[bool] = []
        root = tmp_path / "画像"
        root.mkdir()
        image = QImage(16, 9, QImage.Format.Format_RGBA8888)
        image.fill(0xFF3366FF)
        assert image.save(str(root / "青.png"))
        panel = ExplorerPanel(
            player=_quiet_player(), before_play=lambda: stopped.append(True), measure=None
        )
        try:
            panel.register(str(root), "image")
            _settle(panel)
            panel.select_file(root / "青.png")
            _settle(panel)
            panel.preview.toggle()
            assert stopped == []
        finally:
            panel.shutdown()
            panel.deleteLater()

    def test_nothing_is_read_until_shown(self, qt_application: QApplication, sounds: Path) -> None:
        # 壊れると、エクスプローラーを開かずに編集を始めただけで、裏で数百の素材を測り始める
        del qt_application
        asked: list[Path] = []

        def measure(path: Path) -> Measure | None:
            asked.append(path)
            return None

        panel = ExplorerPanel(
            [LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT)],
            player=_quiet_player(),
            measure=measure,
            active=False,
        )
        try:
            panel.restore_location(str(sounds / "爆発"), "")
            for _ in range(20):
                _process()
            assert panel.visible_files() == [] and asked == []
            panel.set_active(True)
            _settle(panel)
            assert panel.location_text().endswith("› 爆発")
        finally:
            panel.shutdown()
            panel.deleteLater()

    def test_choosing_in_the_tree_cancels_the_last_place(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、起動直後に人が選んだ場所が、遅れて確かめ終えた前回の場所に置き換わる
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.restore_location(str(sounds / "爆発"), "")
        root = panel._category_root(CATEGORY_SOUND_EFFECT)
        assert root is not None
        panel._tree.setCurrentItem(root)
        _settle(panel)
        assert panel.location_text() == "効果音"

    def test_names_inside_an_opened_subfolder_count_from_the_registration(
        self, panel: ExplorerPanel, sounds: Path
    ) -> None:
        # 壊れると、「爆発」を開いて「爆発」と探すと、その中のドカンが見つからない
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        panel.open_folder(str(sounds / "爆発"))
        _settle(panel)
        panel._filter.setText("爆発")
        _settle(panel)
        assert [p.name for p in panel.visible_files()] == ["ドカン.mp3"]

    def test_hiding_the_panel_stops_the_preview(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、パネルを閉じても下見の BGM が鳴り続け、止めるにはまた開くことになる
        del qt_application
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, _prefs: None)
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            stopped: list[bool] = []
            monkeypatch.setattr(window._explorer.preview, "stop", lambda: stopped.append(True))
            window._on_explorer_visible(False)
            assert stopped == [True]
        finally:
            window.close()
            window.deleteLater()


class TestSelfReviewFive:
    def test_places_outside_every_registration_are_not_sent(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、登録を外した後に残ったお気に入りと前回の場所が、設定の写しから外へ出る
        del qt_application
        monkeypatch.setattr(PreferenceStore, "save", lambda _self, _prefs: None)
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            left = str(tmp_path / "外した" / "ドン.wav")
            window._apply_preferences(
                Preferences(
                    explorer_folders=(LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT),),
                    explorer_favorite_paths=(left,),
                    explorer_last_folder=str(tmp_path / "外した"),
                    explorer_last_file=left,
                )
            )
            copied = window.app_settings()["preferences"]
            assert isinstance(copied, dict)
            assert copied["explorer_favorite_paths"] == []
            assert copied["explorer_last_folder"] == ""
            assert copied["explorer_last_file"] == ""
        finally:
            window.close()
            window.deleteLater()

    def test_a_replaced_file_is_opened_again(self, panel: ExplorerPanel, sounds: Path) -> None:
        # 壊れると、読み直した後も、差し替える前のファイルの長さと絵のまま下見する
        panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
        _settle(panel)
        target = sounds / "ドン.wav"
        panel.select_file(target)
        _settle(panel)
        opened: list[Path | None] = []
        original = panel.preview.show_file

        def recorded(path: Path | None, *rest: object) -> None:
            opened.append(path)
            original(path, *rest)  # type: ignore[arg-type]

        panel.preview.show_file = recorded  # type: ignore[method-assign, assignment]
        target.write_bytes(b"replaced and longer")
        os.utime(target, ns=(1_000_000_000, 1_000_000_000))
        panel.rescan()
        _settle(panel)
        panel.select_file(target)
        _settle(panel)
        assert panel.preview._version is not None
        assert opened and opened[-1] == target

    def test_a_measure_that_arrives_after_moving_is_kept(
        self, qt_application: QApplication, sounds: Path
    ) -> None:
        # 壊れると、測り終える前に別の場所を開くと、測った長さを捨て、戻るたびに測り直す
        del qt_application
        release = threading.Event()
        asked: list[Path] = []

        def slow(path: Path) -> Measure | None:
            asked.append(path)
            release.wait()
            return _lengths(path)

        book = MeasureBook()
        panel = ExplorerPanel(player=_quiet_player(), measure=slow, book=book)
        try:
            panel.register(str(sounds), CATEGORY_SOUND_EFFECT)
            wait_until(lambda: bool(asked), "測り始め", step=_process)
            panel.open_folder(str(sounds / "爆発"))
            release.set()
            _settle(panel)
            measured = list(asked)
            assert measured
            file = LibraryFile(
                path=measured[0],
                folder=LibraryFolder(str(sounds), CATEGORY_SOUND_EFFECT),
                size=measured[0].stat().st_size,
                modified_ns=measured[0].stat().st_mtime_ns,
            )
            assert book.get(file) is not None
        finally:
            release.set()
            panel.shutdown()
            panel.deleteLater()

    def test_a_job_asked_again_while_running_is_not_run_twice(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、場所を移って戻るたびに、まだ作っている途中の波形をもう 1 本作り始める
        del qt_application
        worker = explorer_module._Worker(workers=2)
        release = threading.Event()
        started: list[str] = []

        def job() -> object:
            started.append("x")
            release.wait()
            return None

        try:
            worker.submit("peaks:a", job)
            wait_until(lambda: bool(started), "走り始め", step=_process)
            worker.forget()
            worker.submit("peaks:a", job)
            release.set()
            wait_until(lambda: not worker.busy, "終わり", step=_process)
            assert started == ["x"]
        finally:
            release.set()
            worker.close()
