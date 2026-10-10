"""プリセットとエイリアスの一覧の窓（#276 #277）

前は名前だけのメニューから選ぶだけで、消す・名前を変える・分類を変える道が画面に無く、
どんな見た目かは当ててから確かめるしかなかった
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QListView,
    QMessageBox,
)

from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.io import Preset, PresetStore
from sashimono.core.io.aliases import Alias, AliasStore
from sashimono.core.io.library import ALIAS, PRESET, Library, LibraryEntry, TrashEntry
from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.ui import library_dialog
from sashimono.ui.library_dialog import CONFLICT_BOTH, LibraryDialog
from sashimono.ui.library_thumbnails import (
    THUMBNAIL_SIZE,
    THUMBNAILS_FULL,
    THUMBNAILS_OFF,
    THUMBNAILS_SIMPLE,
    LookThumbnails,
)
from sashimono.ui.library_view import LibraryOptions
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore

RED = (1.0, 0.0, 0.0, 1.0)


def _clip(text: str = "見出し", **params: object) -> Clip:
    source = TEXT.create(text=text, **params)  # type: ignore[arg-type]
    return with_fixed_items(Clip(timeline_start=0, duration=60, source=source), picture=True)


@pytest.fixture(autouse=True)
def no_modal_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """小窓を開かせない 思わぬ所で出ると、CI は時間切れまで止まる 開いたらその場で落とす

    知らせと問いのほか、名前の窓・分類の窓・ファイルの窓・読み込みの重なりの問い
    （``QMessageBox`` を組んで ``exec``）・一覧の窓そのもの（``QDialog.exec``）も止める
    答える試験は、窓の ``ask_*`` ``confirm`` ``choose_*`` を自分で差し替える
    """

    def refuse(*args: object, **_kwargs: object) -> object:
        pytest.fail(f"思わぬ小窓が出た: {args[2] if len(args) > 2 else args}")

    for name in ("information", "warning", "question", "critical"):
        monkeypatch.setattr(QMessageBox, name, refuse)
    monkeypatch.setattr(QMessageBox, "exec", refuse)
    monkeypatch.setattr(QDialog, "exec", refuse)
    for name in ("getText", "getItem"):
        monkeypatch.setattr(QInputDialog, name, refuse)
    for name in ("getExistingDirectory", "getOpenFileNames"):
        monkeypatch.setattr(QFileDialog, name, refuse)


@pytest.fixture
def library(tmp_path: Path) -> Library:
    created = Library(
        presets=PresetStore(tmp_path / "presets"), aliases=AliasStore(tmp_path / "aliases")
    )
    created.presets.save(Preset.capture("赤", _clip("赤", color=RED), category="見出し"))
    created.presets.save(Preset.capture("青", _clip("青")))
    created.aliases.save(Alias.of("テロップ", _clip("テロップ")))
    return created


@pytest.fixture
def thumbnails(tmp_path: Path) -> Iterator[LookThumbnails]:
    created = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
    yield created
    created.release()


def _dialog(
    library: Library,
    thumbnails: LookThumbnails,
    *,
    kind: str = PRESET,
    pick: str | None = None,
    options: LibraryOptions | None = None,
) -> LibraryDialog:
    dialog = LibraryDialog(
        library,
        kind=kind,  # type: ignore[arg-type]
        pick=pick,  # type: ignore[arg-type]
        thumbnails=thumbnails,
        options=options or LibraryOptions(thumbnails=thumbnails.mode),
    )
    # 見える物の判定と並べ替えに窓の大きさが要る
    dialog.show()
    QApplication.processEvents()
    return dialog


def _close(dialog: LibraryDialog) -> None:
    dialog.reject()
    shiboken6.delete(dialog)


def _names(dialog: LibraryDialog) -> list[str]:
    return [entry.name for entry in dialog.entries_shown()]


def _answering(asked: list[str], answer: bool) -> Callable[[str, str], bool]:
    """確かめの問いの代わり 問いの文を覚えて `answer` を返す"""

    def confirm(_title: str, text: str) -> bool:
        asked.append(text)
        return answer

    return confirm


def _wait(condition: object, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:  # type: ignore[operator]
        QApplication.processEvents()
        time.sleep(0.01)


class TestOrganize:
    def test_both_kinds_are_listed(self, library: Library, thumbnails: LookThumbnails) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            assert _names(dialog) == ["赤", "青"]
            dialog.set_kind(ALIAS)
            assert _names(dialog) == ["テロップ"]
        finally:
            _close(dialog)

    def test_rename_from_the_window(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.select("赤")
            monkeypatch.setattr(dialog, "ask_text", lambda *_args: "朱")
            dialog.rename_selected()
            assert _names(dialog) == ["朱", "青"]
            assert {p.name for p in library.presets.all()} == {"朱", "青"}
        finally:
            _close(dialog)

    def test_a_taken_name_is_asked_and_kept_when_refused(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library.presets.save(Preset.capture("緑", _clip(), category="見出し"))
        dialog = _dialog(library, thumbnails)
        try:
            dialog.select("赤")
            asked: list[str] = []
            monkeypatch.setattr(dialog, "ask_text", lambda *_args: "緑")
            monkeypatch.setattr(dialog, "confirm", _answering(asked, False))
            dialog.rename_selected()
            assert asked and "緑" in asked[0]
            assert sorted(_names(dialog)) == ["緑", "赤", "青"]
        finally:
            _close(dialog)

    def test_a_new_category_can_be_typed(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.select("赤", "青")
            monkeypatch.setattr(dialog, "ask_category", lambda *_args: "お気に入り")
            dialog.recategorize_selected()
            assert {p.category for p in library.presets.all()} == {"お気に入り"}
            # 移した分類を開いて、移した物を選んだまま見せる
            assert _names(dialog) == ["赤", "青"]
        finally:
            _close(dialog)

    def test_duplicate(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialog = _dialog(library, thumbnails, kind=ALIAS)
        try:
            dialog.select("テロップ")
            monkeypatch.setattr(dialog, "ask_text", lambda _t, _l, text: text)
            dialog.duplicate_selected()
            assert _names(dialog) == ["テロップ", "テロップ のコピー"]
        finally:
            _close(dialog)


class TestDelete:
    def test_it_asks_then_goes_to_the_trash_and_comes_back(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.select("赤")
            asked: list[str] = []
            monkeypatch.setattr(dialog, "confirm", _answering(asked, True))
            dialog.delete_selected()
            assert len(asked) == 1 and "戻せる" in asked[0]
            assert _names(dialog) == ["青"]

            dialog.show_folder("trash")
            (trashed,) = dialog.entries_shown()
            assert isinstance(trashed, TrashEntry)
            dialog.select("赤")
            dialog.restore_selected()
            dialog.show_folder("all")
            assert _names(dialog) == ["赤", "青"]
        finally:
            _close(dialog)

    def test_the_question_can_be_turned_off(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        options = LibraryOptions(thumbnails=thumbnails.mode, confirm_delete=False)
        dialog = _dialog(library, thumbnails, options=options)
        try:
            monkeypatch.setattr(dialog, "confirm", lambda *_a: pytest.fail("尋ねないはず"))
            dialog.select("赤")
            dialog.delete_selected()
            assert _names(dialog) == ["青"]
            # 切っても消えるのではなく、ごみ箱へ移る
            assert [t.name for t in library.trashed(PRESET)] == ["赤"]
        finally:
            _close(dialog)

    def test_emptying_always_asks(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        options = LibraryOptions(thumbnails=thumbnails.mode, confirm_delete=False)
        library.delete(next(e for e in library.entries(PRESET) if e.name == "赤"))
        dialog = _dialog(library, thumbnails, options=options)
        try:
            dialog.show_folder("trash")
            asked: list[str] = []
            monkeypatch.setattr(dialog, "confirm", _answering(asked, False))
            dialog.empty_trash()
            # 戻せなくなる操作は、確かめを切っていても尋ねる
            assert asked
            assert len(library.trashed(PRESET)) == 1
            # 選んだ物だけをごみ箱から消すときも同じ
            dialog.select("赤")
            dialog.purge_selected()
            assert len(asked) == 2
            assert len(library.trashed(PRESET)) == 1
        finally:
            _close(dialog)

    def test_the_trash_is_counted_apart(self, library: Library, thumbnails: LookThumbnails) -> None:
        # ごみ箱の物が一覧や分類に混ざると、消したはずの物が当てる候補に戻ってくる
        library.delete(next(e for e in library.entries(PRESET) if e.name == "赤"))
        dialog = _dialog(library, thumbnails)
        try:
            assert _names(dialog) == ["青"]
            labels = [dialog._folders.item(r).text() for r in range(dialog._folders.count())]
            assert labels == ["すべて", "ユーザー", "ごみ箱（1）"]
        finally:
            _close(dialog)


class TestExchange:
    def test_export_then_import_keeping_both(
        self,
        library: Library,
        thumbnails: LookThumbnails,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            out = tmp_path / "out"
            told: list[str] = []
            monkeypatch.setattr(dialog, "tell", lambda _t, text: told.append(text))
            monkeypatch.setattr(dialog, "choose_folder", lambda: out)
            dialog.select("赤")
            dialog.export_selected()
            assert (out / "赤.smep").is_file()

            monkeypatch.setattr(dialog, "choose_files", lambda: [out / "赤.smep"])
            monkeypatch.setattr(dialog, "ask_conflict", lambda *_a: CONFLICT_BOTH)
            dialog.import_files()
            assert sorted(_names(dialog)) == ["赤", "赤 (2)", "青"]
            assert "1 件を読み込んだ" in told[-1]
        finally:
            _close(dialog)

    def test_an_alias_file_opens_the_alias_tab(
        self,
        library: Library,
        thumbnails: LookThumbnails,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 読み込んだのに一覧に出ないと、失敗したように見える
        source = AliasStore(tmp_path / "elsewhere")
        path = source.save(Alias.of("届いた", _clip()))
        dialog = _dialog(library, thumbnails)
        try:
            monkeypatch.setattr(dialog, "tell", lambda *_a: None)
            monkeypatch.setattr(dialog, "choose_files", lambda: [path])
            dialog.import_files()
            assert dialog.kind == ALIAS
            assert _names(dialog) == ["テロップ", "届いた"]
        finally:
            _close(dialog)


class TestPick:
    def test_the_chosen_preset_is_handed_back(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails, pick=PRESET)
        try:
            dialog.select("青")
            dialog._pick_selected()
            assert isinstance(dialog.chosen, LibraryEntry)
            assert dialog.chosen.name == "青"
        finally:
            shiboken6.delete(dialog)

    def test_the_other_kind_cannot_be_picked(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        # プリセットを当てる窓でエイリアスを選んでも、当てる物が無い
        dialog = _dialog(library, thumbnails, pick=PRESET)
        try:
            dialog.set_kind(ALIAS)
            dialog.select("テロップ")
            dialog._pick_selected()
            assert dialog.chosen is None
        finally:
            _close(dialog)


class TestThumbnails:
    def test_visible_items_get_a_picture_and_it_is_kept(
        self, library: Library, thumbnails: LookThumbnails, tmp_path: Path
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            preset = next(p for p in library.presets.all() if p.name == "赤")
            _wait(lambda: thumbnails.cached(preset) is not None)
            found = thumbnails.cached(preset)
            assert found is not None and found.image is not None
            assert found.image.size() == THUMBNAIL_SIZE
            assert found.simple
        finally:
            _close(dialog)
        # 作り置き 別の窓（新しい係）でも描かずに出る
        again = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
        assert again.cached(preset) is not None
        again.release()

    def test_a_rename_reuses_the_picture_and_a_change_does_not(
        self, thumbnails: LookThumbnails
    ) -> None:
        first = Preset.capture("赤", _clip("あ"))
        assert thumbnails.key_for(first) == thumbnails.key_for(Preset.capture("朱", _clip("あ")))
        assert thumbnails.key_for(first) != thumbnails.key_for(Preset.capture("赤", _clip("い")))

    def test_a_broken_picture_is_drawn_again(self, thumbnails: LookThumbnails) -> None:
        preset = Preset.capture("赤", _clip())
        thumbnails.request(preset)
        _wait(lambda: thumbnails.cached(preset) is not None)
        path = next((thumbnails._path(thumbnails.key_for(preset))).parent.glob("*.png"))
        path.write_bytes(b"not a png")
        fresh = LookThumbnails(path.parents[2], mode=THUMBNAILS_SIMPLE)
        try:
            assert fresh.cached(preset) is None
            assert not path.exists()
        finally:
            fresh.release()

    def test_off_means_no_drawing_and_a_plain_list(self, library: Library, tmp_path: Path) -> None:
        # 切ったのに描き続けると、設定がある方が質が悪い
        off = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_OFF)
        dialog = _dialog(library, off)
        try:
            assert off.request(Preset.capture("赤", _clip())) is None
            assert off._worker is None
            assert dialog._list.viewMode() == QListView.ViewMode.ListMode
            assert not (tmp_path / "cache").exists()
        finally:
            _close(dialog)
            off.release()

    def test_the_simple_drawing_says_effects_are_missing(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        # 出せていない物を出せているように見せない（テンプレートの棚の下絵と同じ）
        dialog = _dialog(library, thumbnails)
        try:
            assert "エフェクトは出ていない" in dialog._note.text()
        finally:
            _close(dialog)


class TestPreferences:
    def test_the_settings_come_back(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(
            Preferences(
                library_thumbnails=THUMBNAILS_OFF,
                library_backdrop="dark",
                library_confirm_delete=False,
            )
        )
        loaded = store.load()
        assert loaded.library_options == LibraryOptions(
            thumbnails=THUMBNAILS_OFF, backdrop="dark", confirm_delete=False
        )

    def test_the_defaults_suit_someone_who_never_opened_the_settings(self) -> None:
        options = Preferences().library_options
        assert options.thumbnails == "full"
        assert options.backdrop == "checker"
        assert options.confirm_delete

    def test_the_dialog_offers_them(self) -> None:
        dialog = PreferencesDialog(Preferences(library_thumbnails=THUMBNAILS_SIMPLE))
        try:
            assert dialog.preferences().library_thumbnails == THUMBNAILS_SIMPLE
            dialog._library_confirm_delete.setChecked(False)
            dialog._library_backdrop.setCurrentIndex(dialog._library_backdrop.findData("light"))
            chosen = dialog.preferences()
            assert chosen.library_confirm_delete is False
            assert chosen.library_backdrop == "light"
        finally:
            shiboken6.delete(dialog)


class TestEntrances:
    """設定パネルの〔プリセット…〕と、〔追加〕→〔エイリアス〕から開いて選んだ物が届く"""

    @pytest.fixture
    def window(self, library: Library) -> Iterator[MainWindow]:
        base = Project.create()
        track = Track(TrackKind.VIDEO, "V1", (_clip("元の文字"),))
        created = MainWindow(
            base.with_timeline(replace(base.timeline, tracks=(track,))), confirm_unsaved=False
        )
        created._inspector.set_preset_store(library.presets)
        created._inspector.set_alias_store(library.aliases)
        created._timeline.add_sources.presets = library.presets
        created._timeline.add_sources.aliases = library.aliases
        yield created
        created.close()

    def test_the_preset_menu_opens_the_window_and_applies_the_pick(
        self, window: MainWindow, library: Library, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = window.document.project.timeline.tracks[0].clips[0]
        window._timeline.set_selection((clip.id,))
        QApplication.processEvents()
        opened: list[tuple[str, str | None]] = []

        def fake_open(
            _parent: object, shown: Library, *, kind: str, pick: str | None = None
        ) -> LibraryEntry:
            opened.append((kind, pick))
            # 窓に渡す置き場は、設定パネルが持っている物と同じ
            assert shown.presets.root == library.presets.root
            return next(e for e in shown.entries(PRESET) if e.name == "赤")

        monkeypatch.setattr(library_dialog, "open_library", fake_open)
        menu = window._inspector.preset_menu()
        assert menu is not None
        manage = next(a for a in menu.actions() if a.data() == "manage_presets")
        window._inspector.run_preset_action(manage)
        QApplication.processEvents()
        assert opened == [(PRESET, PRESET)]
        placed = window.document.project.timeline.tracks[0].clips[0]
        assert placed.source is not None
        # 文字は残し（当て方の既定）、見た目（色）だけが当たる
        assert placed.source.params["color"] == RED
        assert placed.source.params["text"] == "元の文字"

    def test_the_alias_menu_opens_the_window_and_places_the_pick(
        self, window: MainWindow, library: Library
    ) -> None:
        def fake_open(
            _parent: object, shown: Library, *, kind: str, pick: str | None = None
        ) -> LibraryEntry:
            assert (kind, pick) == (ALIAS, ALIAS)
            return next(e for e in shown.entries(ALIAS) if e.name == "テロップ")

        window._timeline.add_sources.open_library = fake_open
        before = sum(len(t.clips) for t in window.document.project.timeline.tracks)
        window._timeline._add_menus.manage_aliases(200, None)
        QApplication.processEvents()
        clips = [c for t in window.document.project.timeline.tracks for c in t.clips]
        assert len(clips) == before + 1
        placed = next(c for c in clips if c.timeline_start == 200)
        assert placed.source is not None
        assert placed.source.params["text"] == "テロップ"


class TestWithoutGpu:
    def test_pictures_fall_back_to_simple_and_are_not_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GPU の無い機械 空の見本を並べず簡易の描き方で出し、エフェクトが出ていないと書く
        # GPU で頼んだ鍵へ簡易の絵を置くと、GPU の使える機械へ移ったときに掴み続ける
        def refuse(_self: OffscreenGLContext) -> None:
            raise GLContextError("試験で GPU を断る")

        monkeypatch.setattr(OffscreenGLContext, "complete", refuse)
        service = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_FULL)
        try:
            key = service.request(Preset.capture("赤", _clip()))
            assert key is not None
            _wait(lambda: service.thumbnail(key) is not None)
            found = service.thumbnail(key)
            assert found is not None and found.simple and found.image is not None
            assert service.simple
            assert not list((tmp_path / "cache").rglob("*.png"))
        finally:
            service.release()
