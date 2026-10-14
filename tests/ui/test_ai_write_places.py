"""AI が書ける場所と、アプリの設定を読む口（#350）を、本物の窓で確かめる

AI が書けるのは作品のフォルダと、書き出しの窓で前に選んだフォルダだけ（2026-10-14 利用者の
決定） 書き出しの窓で人が書き出したフォルダを覚えることと、AI がアプリの設定を読めるが
変えられないことを押さえる
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai.operations import find_operation
from sashimono.core.model import Project
from sashimono.engine.encode import ExportSettings
from sashimono.ui import export_dialog, main_window
from sashimono.ui.export_dialog import ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ui.test_export_dialog import _non_empty_project


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 答える試験は、この後で自分の答えに差し替える"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()
    created.deleteLater()


class _ChoosingDialog(ExportDialog):
    """人が書き出しの窓で ``chosen`` へ書き出した、の代わり

    本物の書き出しは走らせない 試験で見たいのは、人が選んだ所を窓の外が覚えるかだけ
    """

    chosen: Path | None = None
    opened_at: list[str] = []  # noqa: RUF012 - 試験の間だけ、開いた窓の初めの場所を並べる

    def exec(self) -> int:
        _ChoosingDialog.opened_at.append(self._path.text())
        self.started_folder = _ChoosingDialog.chosen
        return 1


class TestExportFolder:
    def test_a_folder_exported_to_is_remembered_and_offered_next_time(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、人が書き出したフォルダへ AI が書けず（利用者の決定の 2 つ目）、
        # 書き出しの窓も毎回ビデオのフォルダから選び直しになる
        monkeypatch.setattr(main_window, "ExportDialog", _ChoosingDialog)
        _ChoosingDialog.opened_at = []
        _ChoosingDialog.chosen = tmp_path
        assert window.export_folder is None

        window.export()
        assert window.export_folder == tmp_path
        assert PreferenceStore().load().export_folder == str(tmp_path)

        window.export()
        assert Path(_ChoosingDialog.opened_at[-1]).parent == tmp_path

    def test_closing_without_exporting_remembers_nothing(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、窓を開いて閉じただけで、人が選んでいないフォルダまで AI が書ける所になる
        monkeypatch.setattr(main_window, "ExportDialog", _ChoosingDialog)
        _ChoosingDialog.chosen = None
        window.export()
        assert window.export_folder is None

    def test_a_relative_folder_in_the_settings_is_ignored(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、設定のファイルに書かれた相対の場所が、起動した所しだいで別のフォルダを指す
        del qt_application
        PreferenceStore().save(Preferences(export_folder="相対/フォルダ"))
        created = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert created.export_folder is None
        finally:
            created.close()
            created.deleteLater()

    def test_the_dialog_notes_the_folder_when_the_export_starts(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、〔書き出し〕を押しても窓の外へフォルダが伝わらず、覚えられない
        del qt_application
        finished: list[str] = []

        def fake_export(_project: Project, settings: ExportSettings, **_options: Any) -> Path:
            return settings.path

        monkeypatch.setattr(export_dialog, "run_export", fake_export)
        monkeypatch.setattr(
            QMessageBox,
            "information",
            lambda _parent, _title, text: finished.append(text),
        )
        dialog = ExportDialog(_non_empty_project(), folder=tmp_path)
        try:
            assert Path(dialog._path.text()).parent == tmp_path
            assert dialog.started_folder is None
            dialog._start()
            assert dialog.started_folder == tmp_path
            thread = dialog._thread
            if thread is not None:
                thread.wait(5000)
            for _ in range(50):
                QApplication.processEvents()
                if finished:
                    break
            assert finished
        finally:
            dialog.deleteLater()


class TestAppSettings:
    def test_the_assistant_reads_settings_shortcuts_layout_and_help(
        self, window: MainWindow
    ) -> None:
        # 壊れると、AI が本人の設定やショートカットを知らずに、画面と違う手順を案内する
        operation = find_operation("get_app_settings")
        assert operation is not None
        result = operation(window, {})
        assert isinstance(result, dict)
        assert set(result) >= {"version", "preferences", "shortcuts", "layout", "help"}
        preferences = result["preferences"]
        assert isinstance(preferences, dict)
        assert preferences["theme"] == window._preferences.theme
        shortcuts = result["shortcuts"]
        assert isinstance(shortcuts, dict)
        assert any(name.endswith("/保存") for name in shortcuts)

    def test_what_the_assistant_read_cannot_change_the_settings(self, window: MainWindow) -> None:
        # 壊れると、読んだ値を書き換えただけで本人の設定が変わり、次の起動から動きが変わる
        operation = find_operation("get_app_settings")
        assert operation is not None
        before = window._preferences
        result = operation(window, {})
        assert isinstance(result, dict)
        preferences = result["preferences"]
        assert isinstance(preferences, dict)
        preferences["update_check"] = not before.update_check
        assert window._preferences == before
        assert PreferenceStore().load().update_check is Preferences().update_check

    def test_the_project_folder_is_the_saved_file(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 壊れると、AI が書ける場所の基準（作品のフォルダ）が、開いている作品と食い違う
        del qt_application
        path = tmp_path / "作品.sme"
        created = MainWindow(Project.create(), path=path, confirm_unsaved=False)
        try:
            assert created.project_path == path
        finally:
            created.close()
            created.deleteLater()
