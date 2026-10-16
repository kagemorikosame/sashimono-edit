"""設定画面の OK で、画面に出していない項目（覚えた書き出しのフォルダなど）を消さない

設定画面は画面の部品から :class:`Preferences` を組み立てる 一から作り直すと、画面に
出していない項目（窓の外で覚える物）は既定へ戻る 書き出しのフォルダが消えると、
書き出しの窓は毎回ビデオのフォルダから始まり、AI はそこへ書けなくなる（#350 の決定）
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox

from sashimono.core.model import Project
from sashimono.ui import preferences_dialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす 設定画面だけは試験が OK の代わりをする"""

    def refuse(*_arguments: object, **_options: object) -> None:
        raise AssertionError("試験の途中で窓が開いた")

    for owner, name in (
        (QDialog, "open"),
        (QDialog, "show"),
        (QMainWindow, "show"),
        (QMessageBox, "exec"),
        (QMessageBox, "warning"),
        (QMessageBox, "information"),
        (QMessageBox, "question"),
        (QMessageBox, "critical"),
    ):
        monkeypatch.setattr(owner, name, refuse)
    monkeypatch.setattr(QDialog, "exec", refuse)


@pytest.fixture
def window(
    qt_application: QApplication, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[MainWindow]:
    del qt_application
    folder = tmp_path_factory.mktemp("書き出し")
    PreferenceStore().save(Preferences(export_folder=str(folder)))
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()
    created.deleteLater()


class TestHiddenItemsSurvive:
    def test_the_dialog_keeps_the_export_folder(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面で OK を押しただけで覚えた書き出しのフォルダが消える
        del qt_application
        dialog = PreferencesDialog(Preferences(export_folder="C:/書き出し"))
        try:
            assert dialog.preferences().export_folder == "C:/書き出し"
        finally:
            dialog.deleteLater()

    def test_an_untouched_dialog_returns_what_it_was_given(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、何も変えずに OK を押しただけで、どれかの設定が既定へ戻る
        del qt_application
        given = Preferences(export_folder="C:/書き出し", check_loudness_target=-23, theme="light")
        dialog = PreferencesDialog(given)
        try:
            assert dialog.preferences() == given
        finally:
            dialog.deleteLater()

    def test_the_dialog_starts_from_the_given_settings(self) -> None:
        # 壊れると、次に画面の外で覚える項目を足したとき、設定画面の OK でまた消える
        # （一から Preferences(...) を組むと、渡し忘れた項目は既定に戻る）
        source = inspect.getsource(preferences_dialog.PreferencesDialog.preferences)
        tree = ast.parse(source.strip().replace("\n    ", "\n"))
        built = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Preferences"
        ]
        assert built == []

    def test_ok_in_the_window_keeps_the_export_folder(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、設定を変えた後の書き出しで前のフォルダが出ず、AI もそこへ書けない
        folder = window.export_folder
        assert folder is not None
        monkeypatch.setattr(
            PreferencesDialog, "exec", lambda _self: QDialog.DialogCode.Accepted.value
        )
        window.edit_preferences()
        assert window.export_folder == folder
        assert PreferenceStore().load().export_folder == str(folder)
