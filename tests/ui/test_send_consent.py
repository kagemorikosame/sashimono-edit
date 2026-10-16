"""AI が外へ送る前の同意（#342 D4）を、本物の窓（MainWindow）で確かめる

AI の道具の呼び出しはスレッドを 1 本立てて行い、画面の側はパネルの見張り（``_poll``）を
回す 本物のアプリと同じ道で、窓が出る・断ると送らない・許すと覚える・設定で取り消すと
また尋ねる、を通す 尋ねる窓は試験の中で差し替えて答える（本物の窓が開いたら落とす）
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QMessageBox, QWidget

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import ImageResult, find_operation
from sashimono.ai.outbound import DESTINATION, Outbound, SendAnswer, SendChoice, SendRequest
from sashimono.core.model import Project
from sashimono.ui.chat.send_consent import (
    SEND_ALWAYS_TEXT,
    SEND_DENY_TEXT,
    SEND_ONCE_TEXT,
    ask_send,
)
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.test_bridge import call_in_thread


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 送る前の確かめはパネルの ``ask_send`` を差し替えて答える"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created._chat.close_session()
    created.close()
    created.deleteLater()


class _Answers:
    """尋ねる窓の代わり 出た確かめを覚え、決めておいた答えを返す"""

    def __init__(self, answer: SendAnswer | None) -> None:
        self.answer = answer
        self.asked: list[SendRequest] = []

    def __call__(self, _parent: QWidget | None, request: SendRequest) -> SendAnswer | None:
        self.asked.append(request)
        return self.answer


def _bridge(window: MainWindow) -> EditorBridge:
    return window._chat._bridge


def _preview(window: MainWindow) -> dict[str, object]:
    """AI が preview_frame を呼んだことにして、画面の見張りを回しながら終わりを待つ"""
    operation = find_operation("preview_frame")
    assert operation is not None
    done, box = call_in_thread(_bridge(window), operation, {"frame": 0, "width": 160})
    deadline = time.monotonic() + 10
    while not done.is_set() and time.monotonic() < deadline:
        window._chat._poll()
        QApplication.processEvents()
        time.sleep(0.005)
    window._chat._poll()
    assert done.is_set(), "道具が終わらなかった（送る前の確かめを待ったまま止まっている）"
    return box


@pytest.fixture
def rendered(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """絵を描いた回数 描いたら外へ出る 描かなければ出ない"""
    drawn: list[int] = []

    def draw(frame: int, *, width: int) -> bytes:
        del width
        drawn.append(frame)
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(window, "render_png", draw)
    return drawn


def _answer_with(window: MainWindow, answer: SendAnswer | None) -> _Answers:
    answers = _Answers(answer)
    window._chat.ask_send = answers
    return answers


class TestPreviewConsentInTheWindow:
    def test_the_first_preview_shows_the_window_and_denying_sends_nothing(
        self, window: MainWindow, rendered: list[int]
    ) -> None:
        # 壊れると、本人が知らないうちに映像の絵が Anthropic へ送られる
        answers = _answer_with(window, SendAnswer.DENY)
        box = _preview(window)

        assert [request.kinds for request in answers.asked] == [(Outbound.FRAMES,)]
        assert isinstance(box.get("error"), ToolError)
        assert rendered == []
        # 断ったことは覚えない（次に起動したらまた尋ねる）
        assert PreferenceStore().load().ai_send_frames == SendChoice.ASK.value
        assert "映像の絵 を送りません" in window._chat._view.toPlainText()

    def test_allowing_is_remembered_in_the_settings(
        self, window: MainWindow, rendered: list[int]
    ) -> None:
        # 壊れると、許した後も絵を見るたびに窓が出るか、次に起動したときにまた尋ねる
        answers = _answer_with(window, SendAnswer.ALWAYS)
        box = _preview(window)
        assert isinstance(box.get("result"), ImageResult)
        assert len(answers.asked) == 1
        assert PreferenceStore().load().ai_send_frames == SendChoice.ALLOW.value
        assert window._preferences.ai_send_frames == SendChoice.ALLOW.value

        again = _preview(window)
        assert isinstance(again.get("result"), ImageResult)
        assert len(answers.asked) == 1
        assert rendered == [0, 0]

    def test_revoking_in_the_settings_asks_again(
        self, window: MainWindow, rendered: list[int]
    ) -> None:
        # 壊れると、設定で取り消しても、前に許したまま黙って絵を送り続ける
        answers = _answer_with(window, SendAnswer.ALWAYS)
        _preview(window)
        assert len(answers.asked) == 1

        dialog = PreferencesDialog(window._preferences, window)
        try:
            box = dialog._ai_sends[Outbound.FRAMES]
            assert box.currentData() == SendChoice.ALLOW.value
            box.setCurrentIndex(box.findData(SendChoice.ASK.value))
            window._apply_preferences(dialog.preferences())
        finally:
            dialog.deleteLater()
        assert PreferenceStore().load().ai_send_frames == SendChoice.ASK.value

        answers.answer = SendAnswer.DENY
        box_after = _preview(window)
        assert len(answers.asked) == 2
        assert isinstance(box_after.get("error"), ToolError)
        assert rendered == [0]

    def test_a_setting_saved_before_starting_is_used(self, qt_application: QApplication) -> None:
        # 壊れると、前に「尋ねずに送る」にした人にも、起動するたびに窓が出る
        del qt_application
        PreferenceStore().save(Preferences(ai_send_frames=SendChoice.ALLOW.value))
        created = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert _bridge(created).may_send(Outbound.FRAMES)
            assert not _bridge(created).may_send(Outbound.TEXT)
        finally:
            created.close()
            created.deleteLater()


class TestSettings:
    def test_each_kind_has_a_row_and_the_default_is_ask(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面から送る物を選べないか、知らない人が尋ねられずに送ってしまう
        del qt_application
        dialog = PreferencesDialog(Preferences())
        try:
            assert set(dialog._ai_sends) == {Outbound.FRAMES, Outbound.TEXT, Outbound.FILES}
            for box in dialog._ai_sends.values():
                assert isinstance(box, QComboBox)
                assert box.currentData() == SendChoice.ASK.value
                assert DESTINATION in box.toolTip()
            chosen = dialog.preferences()
            assert chosen.send_choices() == dict.fromkeys(
                (Outbound.FRAMES, Outbound.TEXT, Outbound.FILES), SendChoice.ASK
            )
        finally:
            dialog.deleteLater()

    def test_ok_keeps_what_the_window_does_not_show(self, qt_application: QApplication) -> None:
        # 壊れると、設定の窓で OK を押しただけで書き出しのフォルダが消え、AI がそこへ書けなくなる
        del qt_application
        dialog = PreferencesDialog(replace(Preferences(), export_folder="D:/書き出し"))
        try:
            assert dialog.preferences().export_folder == "D:/書き出し"
        finally:
            dialog.deleteLater()

    def test_a_broken_value_falls_back_to_ask(self, tmp_path: Path) -> None:
        # 壊れると、設定のファイルの書き損じで「尋ねずに送る」扱いになるか、起動が止まる
        store = PreferenceStore(tmp_path / "preferences.json")
        store.path.write_text('{"ai_send_frames": "yes", "ai_send_text": "deny"}', "utf-8")
        loaded = store.load()
        assert loaded.ai_send_frames == SendChoice.ASK.value
        assert loaded.ai_send_text == SendChoice.DENY.value


class TestTheRealWindow:
    @pytest.mark.parametrize(
        ("button", "expected"),
        [
            (SEND_ALWAYS_TEXT, SendAnswer.ALWAYS),
            (SEND_ONCE_TEXT, SendAnswer.UNTIL_CLOSE),
            (SEND_DENY_TEXT, SendAnswer.DENY),
        ],
    )
    def test_it_says_what_and_where_and_maps_the_buttons(
        self,
        qt_application: QApplication,
        monkeypatch: pytest.MonkeyPatch,
        button: str,
        expected: SendAnswer,
    ) -> None:
        # 壊れると、窓に何がどこへ送られるかが出ないまま押させるか、押したボタンと逆の答えになる
        del qt_application
        seen: dict[str, str] = {}

        def press(box: QMessageBox) -> int:
            seen["text"] = box.informativeText()
            default = box.defaultButton()
            seen["default"] = default.text() if default is not None else ""
            for candidate in box.buttons():
                if candidate.text() == button:
                    candidate.click()
            return 0

        monkeypatch.setattr(QMessageBox, "exec", press)
        request = SendRequest(tool="preview_frame", kinds=(Outbound.FRAMES,))
        assert ask_send(None, request) is expected
        assert DESTINATION in seen["text"]
        assert "映像の絵" in seen["text"]
        assert "preview_frame" in seen["text"]
        # Enter を押しただけで送らない
        assert seen["default"] == SEND_DENY_TEXT
