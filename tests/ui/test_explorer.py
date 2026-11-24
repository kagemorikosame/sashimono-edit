"""エクスプローラー（#521） 木と中身・列と並べ替え・アイコン・下見のプレーヤー・置く・読み込む

登録と見て回る所そのものは tests/core/test_material_library.py
下見の再生と測る所は tests/engine/test_library_sound.py
AI の道具は tests/ai/test_library_tools.py
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from sashimono.core.material_library import (
    CATEGORY_MUSIC,
    CATEGORY_SOUND_EFFECT,
    DEFAULT_AI_CATEGORIES,
    LibraryFolder,
    Measure,
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
    seconds = {"ドン.wav": 0.8, "ドカン.mp3": 3.0, "長い.wav": 95.0}.get(path.name, 1.0)
    return Measure(seconds=seconds, has_audio=True, has_video=False, loudness=-20.0 + seconds)


@pytest.fixture
def sounds(tmp_path: Path) -> Path:
    root = tmp_path / "効果音"
    _touch(root / "ドン.wav")
    _touch(root / "長い.wav")
    _touch(root / "爆発" / "ドカン.mp3")
    _touch(root / "説明.txt")
    return root


def _process() -> None:
    QApplication.processEvents()


def _settle(panel: ExplorerPanel) -> None:
    """裏で読む中身と下見の調べが終わるまで待つ（どちらも画面のスレッドを止めずに裏で行う）"""
    wait_until(lambda: not panel.listing and not panel.preview.opening, "中身と下見", step=_process)


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
        assert sorted(p.name for p in panel.visible_files()) == ["ドン.wav", "長い.wav"]
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
        wait_until(
            lambda: panel.cell(sounds / "長い.wav", _LENGTH) != "", "長さの列", step=_process
        )
        assert panel.cell(sounds / "ドン.wav", _LENGTH) == "0.8 秒"
        assert panel.cell(sounds / "長い.wav", _LENGTH) == "1:35"
        assert panel.cell(sounds / "長い.wav", _LOUDNESS) == "75.0 LUFS"
        panel._table.sortByColumn(_LENGTH, Qt.SortOrder.DescendingOrder)
        assert [p.name for p in panel.visible_files()] == ["長い.wav", "ドン.wav"]

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
            assert preferences["explorer_folders"] == (
                {"path": str(sounds), "category": CATEGORY_SOUND_EFFECT},
            )
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
        assert not panel.restore_location(str(tmp_path / "無い"), "")
        assert panel.restore_location(str(sounds / "爆発"), "")
