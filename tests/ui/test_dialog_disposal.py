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
from PySide6.QtGui import QCloseEvent
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


class _Worker:
    """書き出しの途中のワーカーの代わり 止めるよう頼まれたかを覚える"""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _Thread:
    """止まらないスレッドの代わり 止まったと言うのは、ワーカーの知らせの後だけ"""

    def __init__(self) -> None:
        self.stopped = False

    def quit(self) -> None:
        return None

    def wait(self, timeout: int | None = None) -> bool:
        del timeout
        return self.stopped


@pytest.fixture
def exporting(qt_application: QApplication) -> Iterator[tuple[ExportDialog, _Worker, _Thread]]:
    """書き出しの途中の窓 本物のスレッドは走らせない"""
    del qt_application
    dialog = ExportDialog(Project.create())
    worker, thread = _Worker(), _Thread()
    dialog._worker = worker  # type: ignore[assignment]
    dialog._thread = thread  # type: ignore[assignment]
    yield dialog, worker, thread
    dialog._worker = None
    dialog._thread = None
    dialog.deleteLater()


@pytest.mark.parametrize("closing", ["reject", "close"])
def test_closing_while_exporting_waits_for_the_stop(
    exporting: tuple[ExportDialog, _Worker, _Thread], closing: str
) -> None:
    """書き出し中に閉じても、止まった知らせが来るまで閉じないこと

    止まったと確かめずに閉じると、開いた側が窓を捨てたときに走っているスレッドごと
    壊れて落ちる 前は 5 秒だけ待ち、止まらなくても閉じていた 〔閉じる〕と Esc
    （reject）は closeEvent を通らないので、どちらの道も見る
    """
    dialog, worker, thread = exporting
    closed: list[bool] = []
    dialog.rejected.connect(lambda: closed.append(True))
    if closing == "reject":
        dialog.reject()
    else:
        event = QCloseEvent()
        dialog.closeEvent(event)
        assert not event.isAccepted()
    assert worker.cancelled
    assert closed == []

    # 止まった知らせ（中止は失敗として届く）が来たら、そこで閉じる
    thread.stopped = True
    dialog._on_failed("中止した")
    assert closed == [True]
    assert dialog._thread is None
