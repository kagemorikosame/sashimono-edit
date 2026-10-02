"""編集画面から開く窓を、閉じたあとに捨てること

窓は編集画面を親にして作るので、捨てないと開くたびに編集画面の子として残り続ける
（設定の窓は 30 を超える欄、テンプレートの棚は一覧と下絵を持つ） 開いて閉じるを
繰り返しても、窓が増えないことを見る 書き出しの窓は、書き出し中に〔閉じる〕で閉じても
走っているスレッドを畳んでから閉じること（畳まずに捨てると、スレッドごと壊れて落ちる）
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QDialog

from sashimono.core.model import Project
from sashimono.ui.compat_dialog import CompatibilityDialog
from sashimono.ui.export_dialog import ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.project_settings_dialog import ProjectSettingsDialog
from sashimono.ui.shortcut_dialog import ShortcutDialog
from sashimono.ui.template_dialog import TemplateDialog


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


def test_opening_and_closing_does_not_pile_up(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 開く所だけを差し替える 開くと押す人を待って止まる 取り消したことにする
    dialogs = (
        PreferencesDialog,
        ShortcutDialog,
        ProjectSettingsDialog,
        TemplateDialog,
        CompatibilityDialog,
        ExportDialog,
    )
    for dialog in dialogs:
        monkeypatch.setattr(dialog, "exec", lambda self: 0)
    for _ in range(3):
        window.edit_preferences()
        window.customize_shortcuts()
        window.edit_settings()
        window.new_project()
        window.show_templates()
        window.show_compatibility()
        window.export()
    opened = {type(child) for child in window.findChildren(QDialog)}
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    # 窓が本当に開いたこと 開いていなければ、残らないのは当たり前で何も言えない
    assert opened == set(dialogs)
    assert window.findChildren(QDialog) == []


def test_closing_the_export_window_stops_the_export(qt_application: QApplication) -> None:
    """〔閉じる〕（reject）でも、書き出しを止めてから閉じること

    reject は closeEvent を通らない 止めずに閉じると、開いた側が窓を捨てたときに
    走っているスレッドごと壊れる
    """
    del qt_application
    dialog = ExportDialog(Project.create())
    calls: list[str] = []

    class Worker:
        def cancel(self) -> None:
            calls.append("cancel")

    class Thread:
        def quit(self) -> None:
            calls.append("quit")

        def wait(self, timeout: int) -> bool:
            del timeout
            calls.append("wait")
            return True

    # 書き出しの途中の形にする 本物のスレッドは走らせない
    dialog._worker = Worker()  # type: ignore[assignment]
    dialog._thread = Thread()  # type: ignore[assignment]
    dialog.reject()
    dialog._thread = None
    dialog._worker = None
    dialog.deleteLater()
    assert calls == ["cancel", "quit", "wait"]
