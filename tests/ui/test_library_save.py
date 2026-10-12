"""プリセットとエイリアスを保存する窓（#276 の追記）

前は名前だけの入力窓で、同じ名前があると閉じた後に上書きを尋ね、上書きすると前の内容は
戻せないまま消えた 上書きは最初の窓で指定させ、前の内容はごみ箱へ移す
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.io import Preset, PresetStore
from sashimono.core.io.aliases import Alias, AliasStore
from sashimono.core.io.library import ALIAS, PRESET, Library, LibraryKind
from sashimono.core.model import Clip
from sashimono.effects.sources import TEXT
from sashimono.ui import library_save
from sashimono.ui.library_dialog import LibraryDialog
from sashimono.ui.library_save import (
    UNDO_NOTICE_MS,
    OverwriteUndo,
    SaveChoice,
    SaveDialog,
    UndoNotice,
    offer_undo,
    save_look,
    undo_offers,
)
from sashimono.ui.library_thumbnails import (
    THUMBNAIL_SIZE,
    THUMBNAILS_SIMPLE,
    LookThumbnails,
    Thumbnail,
)
from sashimono.ui.library_view import LibraryOptions
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def _clip(text: str = "見出し", **params: object) -> Clip:
    source = TEXT.create(text=text, **params)  # type: ignore[arg-type]
    return with_fixed_items(Clip(timeline_start=0, duration=60, source=source), picture=True)


@pytest.fixture(autouse=True)
def no_modal_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """小窓を開かせない 思わぬ所で出ると、CI は時間切れまで止まる 開いたらその場で落とす"""

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
    created.presets.save(Preset.capture("赤", _clip(color=RED), category="見出し"))
    created.aliases.save(Alias.of("テロップ", _clip("前の文字"), category="字幕"))
    return created


@pytest.fixture
def thumbnails(tmp_path: Path) -> Iterator[LookThumbnails]:
    created = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
    yield created
    created.release()


def _preset_maker(color: tuple[float, ...] = BLUE) -> Callable[[str, str], Preset]:
    return lambda name, category: Preset.capture(name, _clip(color=color), category=category)


def _alias_maker(text: str = "いまの文字") -> Callable[[str, str], Alias]:
    return lambda name, category: Alias.of(name, _clip(text), category=category)


def _dialog(
    library: Library,
    thumbnails: LookThumbnails,
    kind: LibraryKind = PRESET,
    *,
    suggestion: str = "新しい",
    category: str = "ユーザー",
    options: LibraryOptions | None = None,
) -> SaveDialog:
    make = _preset_maker() if kind == PRESET else _alias_maker()
    dialog = SaveDialog(
        library,
        kind,
        make=make,
        suggestion=suggestion,
        category=category,
        thumbnails=thumbnails,
        options=options or LibraryOptions(thumbnails=thumbnails.mode),
    )
    dialog.show()
    QApplication.processEvents()
    return dialog


def _close(dialog: SaveDialog) -> None:
    dialog.reject()
    shiboken6.delete(dialog)


def _wait(condition: Callable[[], bool], seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


class TestTheWindow:
    def test_a_taken_name_warns_and_needs_the_mark(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            assert dialog.can_save()
            assert dialog.warning() == ""
            dialog.set_category("見出し")
            dialog.set_name("赤")
            # 打った時点で窓の中に出す 閉じた後に不意の問いを出さない
            assert "この名前はもうあります" in dialog.warning()
            assert "ごみ箱" in dialog.warning()
            assert not dialog.can_save()
            assert dialog.choice().replacing is None
            dialog.set_overwrite(True)
            assert dialog.can_save()
            replacing = dialog.choice().replacing
            assert replacing is not None and replacing.name == "赤"
        finally:
            _close(dialog)

    def test_the_mark_starts_off_unless_the_setting_says_so(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            assert not dialog._overwrite.isChecked()
        finally:
            _close(dialog)
        options = LibraryOptions(thumbnails=thumbnails.mode, overwrite_without_asking=True)
        dialog = _dialog(library, thumbnails, options=options)
        try:
            assert dialog._overwrite.isChecked()
        finally:
            _close(dialog)

    def test_picking_from_the_list_means_overwriting_it(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.pick("赤")
            choice = dialog.choice()
            assert (choice.name, choice.category) == ("赤", "見出し")
            assert choice.replacing is not None
            assert dialog.can_save()
        finally:
            _close(dialog)

    def test_an_alias_in_another_category_is_still_overwritten_and_says_so(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        # エイリアスの置き場は名前で決まる 分類だけ変えても別の物にはならない
        dialog = _dialog(library, thumbnails, ALIAS, category="ユーザー")
        try:
            dialog.set_name("テロップ")
            assert "分類" in dialog.warning()
            assert not dialog.can_save()
        finally:
            _close(dialog)

    def test_a_new_category_can_be_typed(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.set_category("新しい分類")
            assert dialog.choice().category == "新しい分類"
        finally:
            _close(dialog)

    def test_enter_alone_does_not_save(self, library: Library, thumbnails: LookThumbnails) -> None:
        # Enter だけで上書きまで進まない 印を入れた後でも〔保存〕を押すまで閉じない
        dialog = _dialog(library, thumbnails)
        try:
            dialog.set_category("見出し")
            dialog.set_name("赤")
            dialog.set_overwrite(True)
            for widget in (dialog._name, dialog._overwrite, dialog):
                QTest.keyClick(widget, Qt.Key.Key_Return)
                QTest.keyClick(widget, Qt.Key.Key_Enter)
            QApplication.processEvents()
            assert dialog.isVisible()
            assert dialog.result() != QDialog.DialogCode.Accepted
            dialog._save.click()
            assert dialog.result() == QDialog.DialogCode.Accepted
        finally:
            shiboken6.delete(dialog)

    def test_before_and_after_pictures_are_shown_side_by_side(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.set_category("見出し")
            dialog.set_name("赤")
            assert dialog._before.isVisible()
            _wait(
                lambda: not dialog._before.pixmap().isNull() and not dialog._after.pixmap().isNull()
            )
            assert not dialog._before.pixmap().isNull()
            assert not dialog._after.pixmap().isNull()
            # 一覧の見本より大きく見せる
            assert dialog._before.width() > dialog._list.iconSize().width()
        finally:
            _close(dialog)

    def test_it_fits_a_small_screen(self, library: Library, thumbnails: LookThumbnails) -> None:
        # 1280x720 の画面（タスクバーと題名の帯を引いて中身 1280x640）に収まる
        for _ in range(30):
            library.aliases.save(Alias.of(f"テロップ {_:02}", _clip()))
        dialog = _dialog(library, thumbnails, ALIAS)
        try:
            hint = dialog.sizeHint().expandedTo(dialog.minimumSizeHint())
            assert hint.width() <= 1280
            assert hint.height() <= 640
        finally:
            _close(dialog)


class TestSaving:
    def _ask(self, dialog: SaveDialog) -> Callable[..., SaveChoice]:
        return lambda *_a, **_k: dialog.choice()

    def test_overwriting_moves_the_old_one_to_the_trash_and_it_comes_back(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails)
        try:
            dialog.pick("赤")
            saved = save_look(
                None, library, PRESET, make=_preset_maker(), suggestion="", ask=self._ask(dialog)
            )
        finally:
            _close(dialog)
        assert saved is not None
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["color"] == BLUE
        (trashed,) = library.trashed(PRESET)
        library.restore(trashed, overwrite=True)
        (back,) = library.presets.all()
        assert back.source is not None and back.source.params["color"] == RED

    def test_a_save_elsewhere_after_the_window_opened_is_not_overwritten(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        # 窓を出した後で、ほかの窓が同じ名前で保存し直した 見せた前の見本と違う物を消さない
        dialog = _dialog(library, thumbnails)
        try:
            dialog.pick("赤")
            library.presets.save(Preset.capture("赤", _clip("ほかの窓"), category="見出し"))
            told: list[str] = []
            saved = save_look(
                None,
                library,
                PRESET,
                make=_preset_maker(),
                suggestion="",
                ask=self._ask(dialog),
                tell=told.append,
            )
        finally:
            _close(dialog)
        assert saved is None
        assert told and "ほかで変わった" in told[0]
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["text"] == "ほかの窓"
        assert library.trashed(PRESET) == ()

    def test_a_name_taken_elsewhere_without_the_mark_is_not_overwritten(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = _dialog(library, thumbnails, ALIAS, suggestion="新しい")
        try:
            library.aliases.save(Alias.of("新しい", _clip("ほかの窓")))
            told: list[str] = []
            saved = save_look(
                None,
                library,
                ALIAS,
                make=_alias_maker(),
                suggestion="",
                ask=self._ask(dialog),
                tell=told.append,
            )
        finally:
            _close(dialog)
        assert saved is None
        assert told and "もうある" in told[0]
        found = next(a for a in library.aliases.all() if a.name == "新しい")
        assert found.clip.source is not None
        assert found.clip.source.params["text"] == "ほかの窓"


class TestReplaceFromTheList:
    def _list_dialog(
        self, library: Library, thumbnails: LookThumbnails, capture: object
    ) -> LibraryDialog:
        dialog = LibraryDialog(
            library,
            kind=ALIAS,
            capture=capture,  # type: ignore[arg-type]
            thumbnails=thumbnails,
            options=LibraryOptions(thumbnails=thumbnails.mode),
        )
        dialog.show()
        QApplication.processEvents()
        return dialog

    def test_the_selected_one_is_updated_with_the_current_clip(
        self, library: Library, thumbnails: LookThumbnails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def capture(kind: LibraryKind, name: str, category: str) -> Alias:
            assert kind == ALIAS
            return Alias.of(name, _clip("いまの文字"), category=category)

        dialog = self._list_dialog(library, thumbnails, capture)
        try:
            asked: list[str] = []
            monkeypatch.setattr(dialog, "confirm", lambda _t, text: asked.append(text) or True)  # type: ignore[func-returns-value]
            dialog.select("テロップ")
            assert dialog._replace_button.isVisible() and dialog._replace_button.isEnabled()
            dialog.replace_with_current()
            assert asked and "ごみ箱" in asked[0]
            (now,) = library.aliases.all()
            assert now.category == "字幕"
            assert now.clip.source is not None and now.clip.source.params["text"] == "いまの文字"
            (trashed,) = library.trashed(ALIAS)
            library.restore(trashed, overwrite=True)
            (back,) = library.aliases.all()
            assert back.clip.source is not None and back.clip.source.params["text"] == "前の文字"
        finally:
            dialog.reject()
            shiboken6.delete(dialog)

    def test_without_a_current_clip_the_button_is_not_shown(
        self, library: Library, thumbnails: LookThumbnails
    ) -> None:
        dialog = self._list_dialog(library, thumbnails, None)
        try:
            dialog.select("テロップ")
            assert not dialog._replace_button.isVisible()
        finally:
            dialog.reject()
            shiboken6.delete(dialog)


class TestPresetSaveUsesTheSameWindow:
    def test_the_panel_opens_the_save_window(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        from sashimono.core.model import Project, Track, TrackKind
        from sashimono.ui.main_window import MainWindow

        base = Project.create()
        clip = _clip()
        project = base.with_timeline(
            replace(base.timeline, tracks=(Track(TrackKind.VIDEO, "V1", (clip,)),))
        )
        window = MainWindow(project, confirm_unsaved=False)
        try:
            window._inspector.set_preset_store(PresetStore(tmp_path / "presets"))
            window._timeline.set_selection((clip.id,))
            QApplication.processEvents()
            asked: list[str] = []

            def ask(*_a: object, **kwargs: object) -> SaveChoice | None:
                asked.append(str(_a[2]))
                return None

            monkeypatch.setattr(library_save, "ask_save", ask)
            menu = window._inspector.preset_menu()
            assert menu is not None
            # 保存の項目はメニューの頭にある（分類の段の中ではない）
            save = next(a for a in menu.actions() if a.data() == "save_preset")
            window._inspector.run_preset_action(save)
            assert asked == [PRESET]
        finally:
            window.close()


class TestPreference:
    def test_it_comes_back(self, tmp_path: Path) -> None:
        assert not Preferences().library_options.overwrite_without_asking
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(save_overwrite_without_asking=True))
        assert store.load().library_options.overwrite_without_asking

    def test_the_dialog_offers_it(self) -> None:
        dialog = PreferencesDialog(Preferences())
        try:
            assert not dialog._save_overwrite.isChecked()
            dialog._save_overwrite.setChecked(True)
            assert dialog.preferences().save_overwrite_without_asking
        finally:
            shiboken6.delete(dialog)


class TestUndoNotice:
    """上書きして保存した直後に、知らせの〔元に戻す〕で前の内容へ戻す（利用者の決定）"""

    @pytest.fixture
    def offers(self) -> Iterator[list[OverwriteUndo]]:
        found: list[OverwriteUndo] = []
        undo_offers().offered.connect(found.append)
        yield found
        undo_offers().offered.disconnect(found.append)

    def _overwrite(self, library: Library) -> None:
        item = Preset.capture("赤", _clip(color=BLUE), category="見出し")
        shown = library.occupant(PRESET, item)
        assert shown is not None

        def ask(*_a: object, **_k: object) -> SaveChoice:
            return SaveChoice(name="赤", category="見出し", replacing=shown)

        saved = save_look(
            None,
            library,
            PRESET,
            make=lambda name, category: Preset.capture(name, _clip(color=BLUE), category=category),
            suggestion="",
            ask=ask,
        )
        assert saved is not None

    def _color(self, library: Library) -> object:
        (now,) = library.presets.all()
        assert now.source is not None
        return now.source.params["color"]

    def test_the_button_brings_the_old_one_back_once(
        self, library: Library, offers: list[OverwriteUndo]
    ) -> None:
        self._overwrite(library)
        (offer,) = offers
        notice = UndoNotice()
        try:
            notice.offer(offer)
            assert notice.isVisible() and notice.button.isEnabled()
            results: list[str] = []
            notice.finished.connect(results.append)
            notice.button.click()
            assert self._color(library) == RED
            # 上書きした新しい方はごみ箱へ（そこからまた戻せる）
            assert [t.name for t in library.trashed(PRESET)] == ["赤"]
            assert results and "戻した" in results[0]
            # 戻した後はもう押せない 押せると、戻した物をまた戻して行き来する
            assert not notice.isVisible()
            assert notice.current is None
            assert "もう" in offer.undo()
            assert self._color(library) == RED
        finally:
            shiboken6.delete(notice)

    def test_a_new_save_offers_nothing(self, library: Library, offers: list[OverwriteUndo]) -> None:
        saved = save_look(
            None,
            library,
            PRESET,
            make=_preset_maker(),
            suggestion="",
            ask=lambda *_a, **_k: SaveChoice(name="新しい", category="ユーザー", replacing=None),
        )
        assert saved is not None
        assert offers == []

    def test_a_change_elsewhere_after_saving_is_not_undone(
        self, library: Library, offers: list[OverwriteUndo]
    ) -> None:
        # 保存した後でほかの窓が同じ名前を変えた そちらの変更を戻すつもりで消さない
        self._overwrite(library)
        (offer,) = offers
        library.presets.save(Preset.capture("赤", _clip("ほかの窓"), category="見出し"))
        text = offer.undo()
        assert "戻さなかった" in text
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["text"] == "ほかの窓"

    def test_the_notice_goes_away_and_cannot_be_used_after(
        self, library: Library, offers: list[OverwriteUndo]
    ) -> None:
        self._overwrite(library)
        notice = UndoNotice()
        try:
            notice.offer(offers[0])
            assert notice._timer.interval() == UNDO_NOTICE_MS
            notice._timer.timeout.emit()
            assert not notice.isVisible()
            assert notice.current is None
            notice.button.click()
            assert self._color(library) == BLUE
        finally:
            shiboken6.delete(notice)

    def test_replacing_from_the_list_also_offers_it(
        self,
        library: Library,
        thumbnails: LookThumbnails,
        offers: list[OverwriteUndo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def capture(kind: LibraryKind, name: str, category: str) -> Alias:
            return Alias.of(name, _clip("いまの文字"), category=category)

        dialog = LibraryDialog(
            library,
            kind=ALIAS,
            capture=capture,
            thumbnails=thumbnails,
            options=LibraryOptions(thumbnails=thumbnails.mode, overwrite_without_asking=True),
        )
        try:
            dialog.show()
            QApplication.processEvents()
            dialog.select("テロップ")
            dialog.replace_with_current()
            (offer,) = offers
            assert offer.kind == ALIAS
            offer.undo()
            (back,) = library.aliases.all()
            assert back.clip.source is not None
            assert back.clip.source.params["text"] == "前の文字"
        finally:
            dialog.reject()
            shiboken6.delete(dialog)

    def test_the_main_window_shows_it_in_the_status_bar(self, library: Library) -> None:
        from sashimono.core.model import Project
        from sashimono.ui.main_window import MainWindow

        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            window.show()
            item = Preset.capture("赤", _clip(color=BLUE), category="見出し")
            shown = library.occupant(PRESET, item)
            saved, trashed = library.save_keeping(PRESET, item, replacing=shown)
            assert trashed is not None
            offer_undo(library, PRESET, saved, trashed)
            assert window._undo_notice.isVisible()
            window._undo_notice.button.click()
            assert self._color(library) == RED
        finally:
            window.close()


class _HeldThumbnails(QObject):
    """絵が届く順を試験で決める見本の係 頼まれた物は覚えておき、届けるのは試験が呼んだ時"""

    ready = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.drawn: dict[str, Thumbnail] = {}

    def key_for(self, item: Preset | Alias) -> str:
        return item.name

    def cached(self, item: Preset | Alias) -> Thumbnail | None:
        return None

    def request(self, item: Preset | Alias) -> str:
        return item.name

    def thumbnail(self, key: str) -> Thumbnail | None:
        return self.drawn.get(key)

    def deliver(self, key: str) -> None:
        image = QImage(THUMBNAIL_SIZE, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0xFFFF0000)
        self.drawn[key] = Thumbnail(image)
        self.ready.emit(key)


class TestLatePictures:
    def test_a_late_picture_of_the_old_name_does_not_land_on_before(self, library: Library) -> None:
        # 名前を「赤」から、描き終わる前に「緑」へ打ち替える 後から届いた赤の絵が
        # 「前（ごみ箱へ移る）」に貼られると、警告（緑）と見本（赤）が食い違った
        library.presets.save(Preset.capture("緑", _clip(), category="見出し"))
        held = _HeldThumbnails()
        dialog = SaveDialog(
            library,
            PRESET,
            make=_preset_maker(),
            suggestion="新しい",
            category="見出し",
            thumbnails=held,  # type: ignore[arg-type]
            options=LibraryOptions(),
        )
        try:
            dialog.set_name("赤")
            dialog._request_pictures()
            dialog.set_name("緑")
            dialog._request_pictures()
            held.deliver("赤")
            assert dialog._before.pixmap().isNull()
            held.deliver("緑")
            assert not dialog._before.pixmap().isNull()
        finally:
            shiboken6.delete(dialog)
