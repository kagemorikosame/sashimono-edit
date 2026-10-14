"""窓を出した直後の大きさが、頼んだ大きさのまま変わらないか（Issue #316）

AI のパネルは、折り返す部品（モデルの並び・ログインの案内など）の今の幅での高さを最小の
高さに入れる（#303） 出る前はパネルの幅が作ったときの仮の 100 画素のままで、そこで
数えるとログインの案内だけで 460 画素を超え、窓の最小の高さが 972 画素になった 窓は
出るときにその最小まで縦に伸び、落ち着いて最小が 556 画素に下がっても伸びたまま残った
（タイムラインが狭くなり、1280x800 で撮る Wiki の写真ではレイヤー 2 が写らなかった）

ログインの済んでいない機械と、AI の部品が入っていない機械の両方で確かめる どちらも
本人の機械の状態に任せると、手元では起きず別の機械でだけ起きる
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from sashimono.core.model import Project
from sashimono.ui.main_window import DEFAULT_SIZE, MainWindow, initial_size
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.recovery_dialog import RecoveryDialog
from sashimono.ui.theme import style_sheet
from sashimono.ui.workspace import Workspace

#: Windows 11 のタスクバー（画素） 画面の高さから引いた残りが窓に使える高さ
TASKBAR = 48

#: 頼む大きさ 既定の大きさ・撮る道具（tools/shots.py）の大きさ・高さ 900 前後の
#: ノート PC の画面に合わせた初めての大きさ（1440x900 と 1600x900）
SIZES = {
    "既定": DEFAULT_SIZE,
    "撮る道具": QSize(1280, 800),
    "1440x900 の画面": initial_size(QSize(1440, 900 - TASKBAR)),
    "1600x900 の画面": initial_size(QSize(1600, 900 - TASKBAR)),
}


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """尋ねる窓・知らせる窓・設定の窓が開いたらその場で落とす

    開いたまま待つと、試験は誰も押さないボタンを待って止まる
    """

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)
    monkeypatch.setattr(RecoveryDialog, "exec", popped)
    monkeypatch.setattr(PreferencesDialog, "exec", popped)


@pytest.fixture(params=["ログインがまだ", "部品が入っていない"])
def machine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """AI のパネルの状態を決める 窓を作る前に当てる

    ログインがまだの機械では、文を折り返すログインの案内が出る（#316 で伸びた形）
    部品が入っていない機械では、導入の欄が開いたままになる
    """
    from sashimono.ai import AI_PACK
    from sashimono.runtime import PackageStatus, PackStatus
    from sashimono.ui.chat import panel as panel_module
    from sashimono.ui.setup import SetupSection

    installed = request.param == "ログインがまだ"
    version = "0.2.158" if installed else None
    status = PackStatus(pack=AI_PACK, packages=(PackageStatus("claude-agent-sdk", version),))
    monkeypatch.setattr(SetupSection, "status", property(lambda _self: status))
    monkeypatch.setattr(panel_module, "credentials_found", lambda: False)
    return str(request.param)


@pytest.fixture
def styled(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    """本番と同じスタイルシートを当て、作った窓を試験の後で壊す

    スタイルシートはボタンの余白を変えるので、当てずに測ると本番と違う高さになる
    外す前に窓を壊す 捨てた編集画面にスタイルシートが当たったまま後のごみ集めで壊れると、
    GPU の無い CI でプロセスごと落ちる（tests/ui/test_narrow_window.py と同じ作法）
    """
    gc.collect()
    saved = qt_application.styleSheet()
    qt_application.setStyleSheet(style_sheet())
    windows: list[MainWindow] = []
    try:
        yield windows
    finally:
        for window in windows:
            window.close()
            window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        for window in windows:
            if shiboken6.isValid(window):
                shiboken6.delete(window)
        gc.collect()
        qt_application.setStyleSheet(saved)


def _window(windows: list[MainWindow]) -> MainWindow:
    window = MainWindow(Project.create(), confirm_unsaved=False)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    windows.append(window)
    return window


def _settle() -> None:
    for _ in range(6):
        QApplication.processEvents()


@pytest.mark.parametrize("chat_in_front", [False, True])
@pytest.mark.parametrize("name", sorted(SIZES))
def test_the_window_keeps_the_size_it_was_given(
    machine: str, styled: list[MainWindow], name: str, chat_in_front: bool
) -> None:
    """初めて開いた窓は、頼んだ大きさのまま出る

    壊れると、初めて開いたときに窓が縦に伸びて画面の下へはみ出し、タイムラインが狭くなる
    """
    del machine
    size = SIZES[name]
    window = _window(styled)
    window.reset_layout()
    if chat_in_front:
        window.show_chat()
    window.resize(size)
    assert window.size() == size
    window.show()
    assert window.size() == size
    _settle()
    assert window.size() == size


@pytest.mark.parametrize("name", sorted(SIZES))
def test_the_remembered_size_comes_back_unchanged(
    machine: str, styled: list[MainWindow], name: str
) -> None:
    """2 回目からの起動でも、覚えている大きさのまま出る

    壊れると、起動するたびに覚えた大きさより縦に伸びて出て、タイムラインが狭くなる
    （閉じると伸びた大きさを覚えるので、直したつもりでも次の起動で戻らない）
    """
    del machine
    size = SIZES[name]
    first = _window(styled)
    first.resize(size)
    first.show()
    _settle()
    Workspace().save(first)
    first.hide()

    window = _window(styled)
    assert window.size() == size
    window.show()
    assert window.size() == size
    _settle()
    assert window.size() == size


@pytest.mark.parametrize("machine", ["ログインがまだ"], indirect=True)
def test_the_wrapped_parts_still_count_once_shown(machine: str, styled: list[MainWindow]) -> None:
    """出た後は、折り返す部品の今の幅での高さが、AI のパネルの最小の高さに入る

    部品が入っている機械だけで見る（入っていない間は並びもログインの案内も隠れている）
    壊れると、狭いパネルを最小まで縮めたときにログインの案内やモデルの並びが下の部品に
    重なって読めない（#303 で直した重なり）
    """
    from sashimono.ui.chat import ChatPanel

    window = _window(styled)
    window.reset_layout()
    window.show_chat()
    window.resize(SIZES["撮る道具"])
    window.show()
    _settle()
    chat = window.findChild(ChatPanel)
    assert chat is not None
    for part in (chat._choices, chat._login_box):
        assert not part.isHidden(), f"{machine} なのに出ていない"
        assert part.minimumHeight() > 0, part.objectName() or type(part).__name__
