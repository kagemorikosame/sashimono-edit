"""書き出しの窓の「マーカーをチャプターとして埋める」と、その設定（#330）

2026-10-17 に利用者が決めた 既定は入 窓で選んだ値は本人の設定に覚え、表示 → 設定… にも出す
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.encode import ExportSettings
from sashimono.ui.export_dialog import ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


def test_on_by_default_and_kept_after_restart(tmp_path: Path) -> None:
    # 既定が設定と書き出しでずれると、設定を触っていない人の動画に章が付いたり付かなかったりする
    # 保存した「切る」を読み戻せないと、次の起動で入りに戻る
    assert Preferences().export_chapters is True
    assert ExportSettings(path=tmp_path / "a.mp4").chapters is True
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(export_chapters=False))
    assert store.load().export_chapters is False


def test_the_preferences_dialog_shows_and_keeps_it(qt_application: QApplication) -> None:
    # 壊れると、設定画面で〔OK〕を押しただけで、切っていた設定が入りに戻る
    del qt_application
    dialog = PreferencesDialog(Preferences(export_chapters=False))
    try:
        assert dialog.preferences().export_chapters is False
        dialog._export_chapters.setChecked(True)
        assert dialog.preferences().export_chapters is True
    finally:
        dialog.deleteLater()


def test_the_export_dialog_passes_the_choice(qt_application: QApplication) -> None:
    # 壊れると、窓で切っても書き出した動画に章が入る
    del qt_application
    dialog = ExportDialog(Project.create(), chapters=False)
    try:
        assert dialog.embeds_chapters is False
        settings = dialog._settings()
        if settings is None:
            pytest.skip("映像のコーデックが無い")
        assert settings.chapters is False
        dialog._chapters.setChecked(True)
        settings = dialog._settings()
        assert settings is not None and settings.chapters is True
    finally:
        dialog.deleteLater()


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


def test_the_window_remembers_what_was_chosen(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 覚えないと、毎回の書き出しで切り直すことになる
    assert window._preferences.export_chapters is True

    def choose_off(dialog: ExportDialog) -> int:
        dialog._chapters.setChecked(False)
        return 0

    monkeypatch.setattr(ExportDialog, "exec", choose_off)
    window.export()
    assert window._preferences.export_chapters is False
    assert PreferenceStore().load().export_chapters is False

    seen: list[bool] = []

    def look(dialog: ExportDialog) -> int:
        seen.append(dialog.embeds_chapters)
        return 0

    monkeypatch.setattr(ExportDialog, "exec", look)
    window.export()
    assert seen == [False]
