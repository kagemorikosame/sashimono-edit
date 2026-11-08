"""AI が外へ送る前の同意（#342 D4）を、本物の窓（MainWindow）で確かめる

AI の道具の呼び出しはスレッドを 1 本立てて行い、画面の側はパネルの見張り（``_poll``）を
回す 本物のアプリと同じ道で、窓が出る・断ると送らない・許すと覚える・設定で取り消すと
また尋ねる、を通す 尋ねる窓は試験の中で差し替えて答える（本物の窓が開いたら落とす）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRect
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QWidget,
)

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import ImageResult, find_operation
from sashimono.ai.outbound import (
    DESTINATION,
    SAMPLE_LINES,
    Outbound,
    Sample,
    SendAnswer,
    SendChoice,
    SendRequest,
)
from sashimono.core.model import Project
from sashimono.ui.chat.send_consent import (
    SEND_ALWAYS_TEXT,
    SEND_DENY_TEXT,
    SEND_ONCE_TEXT,
    SEND_THIS_TIME_TEXT,
    THUMBNAIL_WIDTH,
    SendConsentDialog,
    ask_send,
)
from sashimono.ui.chat.sent_record import SentRecordDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.test_bridge import call_in_thread
from tests.waiting import wait_until

#: 描いた絵の代わり PNG の頭だけ
PNG_HEAD = b"\x89PNG\r\n\x1a\n"


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

    def turn() -> None:
        window._chat._poll()
        QApplication.processEvents()

    # 返る時間は機械の速さで変わる 秒で区切らず、道具が返るのを待つ（#405） 送る前の確かめを
    # 待ったまま止まる崩れは、上限（tests/waiting.py）に当たって居場所と一緒に落ちる
    wait_until(done.is_set, "道具（送る前の確かめを待ったまま止まっていないか）", step=turn)
    window._chat._poll()
    return box


@pytest.fixture
def rendered(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """絵を描いた回数 描いたら外へ出る 描かなければ出ない"""
    drawn: list[int] = []

    def draw(frame: int, *, width: int) -> bytes:
        del width
        drawn.append(frame)
        return PNG_HEAD

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
        # 下見のために 1 度描くが、断った絵は AI へ返らない（#425）
        assert rendered == [0]
        assert "result" not in box
        assert answers.asked[0].samples[0].images == (PNG_HEAD,)
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
        assert "result" not in box_after
        assert rendered == [0, 0]

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

        def press(dialog: SendConsentDialog) -> int:
            seen["text"] = dialog.info.text()
            seen["default"] = next(
                candidate.text()
                for candidate in (dialog.always, dialog.once, dialog.deny)
                if candidate.isDefault()
            )
            for candidate in (dialog.always, dialog.once, dialog.deny):
                if candidate.text() == button:
                    candidate.click()
            return 0

        monkeypatch.setattr(SendConsentDialog, "exec", press)
        request = SendRequest(tool="preview_frame", kinds=(Outbound.FRAMES,))
        assert ask_send(None, request) is expected
        assert DESTINATION in seen["text"]
        assert "映像の絵" in seen["text"]
        assert "preview_frame" in seen["text"]
        # Enter を押しただけで送らない
        assert seen["default"] == SEND_DENY_TEXT

    def test_closing_with_escape_sends_nothing_and_remembers_nothing(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、Esc で閉じただけで「送らない」が残り、次に尋ねてもらえない
        del qt_application

        def escape(dialog: SendConsentDialog) -> int:
            dialog.reject()
            return 0

        monkeypatch.setattr(SendConsentDialog, "exec", escape)
        request = SendRequest(tool="preview_frame", kinds=(Outbound.FRAMES,))
        assert ask_send(None, request) is None

    def test_each_call_says_this_time(self, qt_application: QApplication) -> None:
        # 壊れると、呼び出しごとに尋ねる人が「アプリを閉じるまで」と読み、1 回の許しを長く思う
        del qt_application
        request = SendRequest(tool="get_subtitles", kinds=(Outbound.TEXT,), each_call=True)
        dialog = SendConsentDialog(request)
        try:
            assert dialog.once.text() == SEND_THIS_TIME_TEXT
        finally:
            dialog.deleteLater()


def _png(width: int = 64, height: int = 36) -> bytes:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")  # type: ignore[call-overload]  # QBuffer を受ける形が型の定義に無い
    return bytes(data.data())


class TestPreviewInTheWindow:
    def test_counts_and_head_lines_show_without_opening(self, qt_application: QApplication) -> None:
        # 壊れると、窓に送る文の量も頭の行も出ず、何が送られるかを見ずに答えさせる
        del qt_application
        lines = tuple(f"字幕 {number}" for number in range(10))
        sample = Sample(Outbound.TEXT, count=10, chars=40, lines=lines)
        request = SendRequest(tool="get_subtitles", kinds=(Outbound.TEXT,), samples=(sample,))
        dialog = SendConsentDialog(request)
        try:
            assert "字幕と文字（10 行 40 文字）" in dialog.info.text()
            (heading,) = dialog.headings
            assert "字幕 0" in heading.text()
            assert "字幕 2" in heading.text()
            assert "字幕 3" not in heading.text()
            assert "ほか 7 件" in heading.text()
            # 詳細は既定で閉じ、開くまで作らない（字幕の多い作品で重くしない）
            assert dialog.details is None
            dialog.toggle.setChecked(True)
            assert dialog.details is not None
            text = dialog.details.findChild(QPlainTextEdit)
            assert text is not None
            assert text.toPlainText().splitlines() == list(lines)
        finally:
            dialog.deleteLater()

    def test_frames_show_as_thumbnails(self, qt_application: QApplication) -> None:
        # 壊れると、どのコマの絵が送られるのかを見ずに答えさせる
        del qt_application
        sample = Sample(Outbound.FRAMES, count=1, lines=("3.0 秒",), images=(_png(),))
        request = SendRequest(tool="preview_frame", kinds=(Outbound.FRAMES,), samples=(sample,))
        dialog = SendConsentDialog(request)
        try:
            dialog.toggle.setChecked(True)
            assert dialog.details is not None
            pictures = [
                label
                for label in dialog.details.findChildren(QLabel)
                if not label.pixmap().isNull()
            ]
            assert len(pictures) == 1
            assert pictures[0].pixmap().width() <= THUMBNAIL_WIDTH
        finally:
            dialog.deleteLater()

    @pytest.mark.parametrize("opened", [False, True])
    def test_a_long_preview_fits_the_screen_and_keeps_the_buttons(
        self, qt_application: QApplication, opened: bool
    ) -> None:
        # 壊れると、字幕の多い作品や大きな字で窓が画面からはみ出し、ボタンを押せない
        del qt_application
        lines = tuple(f"とても長い字幕の行 {number} " * 4 for number in range(SAMPLE_LINES))
        text = Sample(Outbound.TEXT, count=5000, chars=90000, lines=lines)
        places = tuple(
            f"C:/Users/本人/動画/素材 {number}.mp4 → 〔素材 m_{number}〕" for number in range(60)
        )
        files = Sample(
            Outbound.FILES,
            count=60,
            lines=places,
            note="ユーザー名「本人」を含む物が 60 件あります",
        )
        request = SendRequest(
            tool="get_subtitles", kinds=(Outbound.TEXT, Outbound.FILES), samples=(text, files)
        )
        dialog = SendConsentDialog(request)
        try:
            font = dialog.font()
            font.setPointSizeF(font.pointSizeF() * 2)
            dialog.setFont(font)
            dialog.show()
            dialog.toggle.setChecked(opened)
            QApplication.processEvents()
            screen = dialog.screen()
            assert screen is not None
            available = screen.availableGeometry()
            assert dialog.frameGeometry().height() <= available.height()
            for button in (dialog.always, dialog.once, dialog.deny):
                assert button.isVisible()
                inside = button.mapTo(dialog, button.rect().bottomRight())
                assert dialog.rect().contains(inside), button.text()
            if opened:
                shown = dialog.details.findChild(QPlainTextEdit) if dialog.details else None
                assert shown is not None
                assert "ほか 4800 件" in shown.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_a_narrow_screen_wraps_the_buttons(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、幅の狭い画面で窓が画面より広くなり、端のボタンを押せない
        # （CodeRabbit の指摘 #450）
        del qt_application
        request = SendRequest(tool="preview_frame", kinds=(Outbound.FRAMES,))
        dialog = SendConsentDialog(request)
        try:
            screen = dialog.screen()
            assert screen is not None
            real = screen.availableGeometry()
            narrow = QRect(real.x(), real.y(), 360, real.height())
            monkeypatch.setattr(type(screen), "availableGeometry", lambda _self: narrow)
            dialog._fit_to_screen()
            dialog.show()
            QApplication.processEvents()
            assert dialog.width() <= narrow.width()
            for button in (dialog.always, dialog.once, dialog.deny):
                assert button.isVisible()
                corner = button.mapTo(dialog, button.rect().bottomRight())
                assert dialog.rect().contains(corner), button.text()
        finally:
            dialog.close()
            dialog.deleteLater()


class TestSentRecord:
    def test_the_settings_show_the_last_sent_and_the_record(
        self, window: MainWindow, rendered: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、「尋ねずに送る」にした後は、何が出ているかを確かめる手段が無い
        del rendered
        _answer_with(window, SendAnswer.ALWAYS)
        _preview(window)
        dialog = PreferencesDialog(window._preferences, window, send_log=window._chat.send_log)
        try:
            assert dialog._send_log_box.isChecked()
            assert not dialog._send_each_call.isChecked()
            record = SentRecordDialog(window._chat.send_log)
            try:
                assert record.last_views
                assert "preview_frame" in record.entries.toPlainText()
                assert "映像の絵 1 枚" in record.entries.toPlainText()
                monkeypatch.setattr(
                    QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes
                )
                record.clear.click()
                assert record.entries.toPlainText() == "記録はありません"
                assert window._chat.send_log.last() == {}
            finally:
                record.deleteLater()
        finally:
            dialog.deleteLater()

    def test_the_settings_reach_the_bridge(self, window: MainWindow) -> None:
        # 壊れると、設定画面で呼び出しごとに尋ねると選んでも、窓は 1 回しか出ない
        dialog = PreferencesDialog(window._preferences, window, send_log=window._chat.send_log)
        try:
            dialog._send_each_call.setChecked(True)
            dialog._send_log_box.setChecked(False)
            window._apply_preferences(dialog.preferences())
        finally:
            dialog.deleteLater()
        assert _bridge(window).ask_each_call
        assert not _bridge(window).send_log.keep
        loaded = PreferenceStore().load()
        assert loaded.ai_send_ask_each_call
        assert not loaded.ai_send_log
