"""試験の窓が本人の画面へ出ないこと（``tests/offscreen_windows.py`` の ``OffScreenWindows``）

壊れると、試験を走らせるたびに編集画面が数百回出ては消え、本人の作業の前へ割り込む
Qt の上で表示中かどうかは変わらないので、ほかの試験は落ちずに気付けない
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from ctypes import wintypes

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMenu, QWidget

from sashimono.core.model import Project
from sashimono.ui.main_window import MainWindow
from tests.offscreen_windows import windows_on_screen

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Windows の窓の見え方を見る"),
    pytest.mark.skipif(windows_on_screen(), reason="窓を画面へ出す指定で走っている"),
]


def _on_screen(widget: QWidget) -> bool:
    """Windows が窓を今映しているか Qt の ``isVisible`` ではなく OS に尋ねる

    引数と戻り値の型を決めておく 決めないと ctypes は窓の番号を 32 ビットの整数として
    渡し、上の桁が落ちた番号で別の窓を尋ねてしまう（映っていても通る）
    """
    is_window_visible = ctypes.WinDLL("user32").IsWindowVisible
    is_window_visible.argtypes = [wintypes.HWND]
    is_window_visible.restype = wintypes.BOOL
    return bool(is_window_visible(int(widget.winId())))


class _ShowWatch(QObject):
    """表示の知らせを受けた時点で、画面へ映さない印が無かった窓を控える

    Qt は表示の知らせの後に、この印を見て窓を映すかを決める 印の無い窓は、すぐ隠れても
    一瞬は映る 終わった後に OS へ尋ねるだけでは、その一瞬を取り逃がす
    """

    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        if (
            event.type() == QEvent.Type.Show
            and isinstance(watched, QWidget)
            and watched.isWindow()
            and not watched.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        ):
            self.shown.append(watched.metaObject().className())
        return False


@pytest.fixture
def made(qt_application: QApplication) -> Iterator[list[QWidget]]:
    """作った窓を閉じる 壊すのは conftest の後片付け（``widgets_left_behind``）に任せる

    ここで壊すと、編集画面の時計やスレッドが畳まれる前に消え、後の試験で落ちうる
    試験の間に一瞬でも映った窓があれば、閉じた後に落とす
    """
    watch = _ShowWatch()
    qt_application.installEventFilter(watch)
    widgets: list[QWidget] = []
    try:
        yield widgets
    finally:
        for widget in reversed(widgets):
            if shiboken6.isValid(widget):
                widget.close()
        qt_application.removeEventFilter(watch)
    assert not watch.shown, f"画面へ映った窓がある: {watch.shown}"


def test_a_plain_window_stays_off_the_screen(made: list[QWidget]) -> None:
    # 壊れると、試験の中で作った小さな窓が本人の画面の左上に出ては消える
    widget = QWidget()
    made.append(widget)
    widget.show()
    QApplication.processEvents()
    # 試験が見るのは Qt の上の表示 ここが偽になると、表示を確かめる試験が全部変わる
    assert widget.isVisible()
    assert not _on_screen(widget)


def test_showing_again_and_maximizing_stays_off_the_screen(made: list[QWidget]) -> None:
    # 壊れると、1 度隠してから出し直した窓や最大化した窓だけが、画面いっぱいに映る
    window = QMainWindow()
    made.append(window)
    window.show()
    window.hide()
    window.showMaximized()
    QApplication.processEvents()
    assert window.isVisible()
    assert not _on_screen(window)


def test_a_dialog_and_a_menu_over_a_window_stay_off_the_screen(made: list[QWidget]) -> None:
    # 子の窓は親とは別の窓として作られる 親だけ止めても、尋ねる窓やメニューは映る
    window = QMainWindow()
    made.append(window)
    window.show()
    dialog = QDialog(window)
    dialog.show()
    menu = QMenu(window)
    menu.addAction("x")
    menu.popup(window.mapToGlobal(window.rect().center()))
    QApplication.processEvents()
    assert dialog.isVisible()
    assert menu.isVisible()
    assert not _on_screen(dialog)
    assert not _on_screen(menu)
    menu.close()


def test_a_window_shown_and_hidden_at_once_never_reaches_the_screen(
    made: list[QWidget],
) -> None:
    # 出してすぐ閉じる窓は、終わった後に OS へ尋ねても映っていない 一瞬のちらつきは
    # 表示の知らせの時点で見る（``made`` が控えて落とす）
    widget = QWidget()
    made.append(widget)
    widget.show()
    widget.hide()
    QApplication.processEvents()
    assert not _on_screen(widget)


def test_the_editing_window_stays_off_the_screen(made: list[QWidget]) -> None:
    # 一番よく出ては消えていたのが編集画面 GL のプレビューを持つ窓でも止まること
    window = MainWindow(Project.create())
    made.append(window)
    window.show()
    QApplication.processEvents()
    assert window.isVisible()
    assert window.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    assert not _on_screen(window)
