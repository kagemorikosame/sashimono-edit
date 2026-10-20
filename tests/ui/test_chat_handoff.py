"""AI のパネルの引き継ぎの欄（#346 D8）

止まる道（例外・中断・上限・断られた・接続が切れた）のどれでも、何をやって・どこで止まったかの
要約が会話の欄に残り、欄からコピーと続きの頼みができることを見る 偽の会話へ出来事を流すので、
本物の Claude Code は要らない 要約の組み方そのものは tests/ai/test_handoff.py で見る
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from sashimono.ai.bridge import Approval, Question
from sashimono.ai.handoff import CONTINUE_PROMPT, Handoff, StopReason, shelf_for
from sashimono.ai.operations import question_spec
from sashimono.ai.outbound import Outbound, SendAnswer, SendRequest
from sashimono.ai.session import LIMIT_DETAIL, AgentEvent, EventKind
from sashimono.core.commands import SplitClip
from sashimono.core.model import MediaItem, Transcript
from sashimono.ui.chat import ChatPanel
from sashimono.ui.chat.panel import HANDOFF_WHO
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import FakeHost, make_loaded
from tests.fake_clipboard import FakeClipboard
from tests.ui.test_chat_panel import _ready, _RecordingSession


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """試験の間に確かめの窓や問い合わせの窓が開いたら、その場で落とす

    開いたままだと、無人の CI では誰も閉じずに試験が止まる 引き継ぎの欄はパネルの中に
    出す物で、窓を開かないことも合わせて押さえる
    """

    def refuse(*_arguments: object, **_options: object) -> int:
        raise AssertionError("試験の間に窓が開いた")

    monkeypatch.setattr(QDialog, "exec", refuse)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, refuse)


@pytest.fixture
def recorded(
    qt_application: QApplication,
    video_media: MediaItem,
    transcript: Transcript,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[ChatPanel, list[_RecordingSession]]]:
    """偽の会話（Claude へは繋がない）で動くパネル"""
    from sashimono.ui.chat import panel as panel_module

    del qt_application
    _ready(monkeypatch)
    _RecordingSession.made = []
    monkeypatch.setattr(panel_module, "AgentSession", _RecordingSession)
    widget = ChatPanel(FakeHost(make_loaded(video_media, transcript)))
    widget._refresh_availability()
    yield widget, _RecordingSession.made
    widget.close_session()
    widget.deleteLater()


def _send(widget: ChatPanel, prompt: str) -> None:
    widget._input.setPlainText(prompt)
    widget.send()


def _host(widget: ChatPanel) -> FakeHost:
    host = widget._host
    assert isinstance(host, FakeHost)
    return host


def _shelved(widget: ChatPanel) -> Handoff | None:
    document = widget._host.document
    return shelf_for(document).current(document)


def _cut_twice(widget: ChatPanel) -> None:
    """AI が指示の中で 2 回切った（道具 2 回分の段）"""
    host = _host(widget)
    first = host.document.project.timeline.tracks[0].clips[0].id
    host.apply_commands([SplitClip(first, 90)], "分割")
    host.apply_commands([SplitClip(first, 60)], "分割")


def _logged(widget: ChatPanel) -> list[str]:
    return [text for who, text in widget._log if who == HANDOFF_WHO]


class TestEveryWayOfStopping:
    def test_an_error_hands_off(self, recorded: tuple[ChatPanel, list[_RecordingSession]]) -> None:
        # 壊れると、例外で止まった指示が「失敗」の区切りだけで終わり、何を変えたか・どう戻すかを
        # 人が履歴を辿って探すことになる
        widget, _ = recorded
        _send(widget, "冒頭を切って")
        _cut_twice(widget)
        widget._handle(AgentEvent(EventKind.ERROR, text="ProcessError: 落ちました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert handoff.reason is StopReason.ERROR
        assert "落ちました" in handoff.detail
        assert handoff.changes == ("分割", "分割")
        assert widget._handoff_box.isHidden() is False
        assert len(_logged(widget)) == 1
        assert "Ctrl+Z" in _logged(widget)[0]
        # 書いた段の名前が、本当に Ctrl+Z で戻る段の名前
        assert widget._host.document.undo_label == handoff.checkpoint == "AI: 冒頭を切って"

    def test_an_interrupt_hands_off(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、〔中断〕で止めた所が分からず、半端に切った作品の上に次の指示を重ねる
        widget, _ = recorded
        _send(widget, "冒頭を切って")
        _cut_twice(widget)
        widget.interrupt()
        widget._handle(AgentEvent(EventKind.ERROR, text="ToolError: 中断されました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert handoff.reason is StopReason.INTERRUPTED

    def test_a_limit_hands_off(self, recorded: tuple[ChatPanel, list[_RecordingSession]]) -> None:
        # 壊れると、上限で止まった指示がエラーと書かれ、送り直せば続くことに気付かない
        widget, _ = recorded
        _send(widget, "全部の字幕を直して")
        widget._handle(AgentEvent(EventKind.ERROR, text="上限に達しました", detail=LIMIT_DETAIL))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert handoff.reason is StopReason.LIMIT
        assert handoff.changed is False

    def test_a_denied_approval_hands_off(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、許さなかった操作を AI が「済みました」と言って終えたとき、何がされずに
        # 残ったかが要約に出ない
        widget, _ = recorded
        _send(widget, "分けて保存して")
        widget._bridge._approvals.put(Approval(tool="split_clip", summary="分割", arguments={}))
        widget._check_approval()
        widget._answer(False)
        widget._handle(AgentEvent(EventKind.TEXT, text="全部済みました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert handoff.reason is StopReason.DENIED
        assert any("split_clip" in item for item in handoff.refused)
        # AI の言葉は AI の言葉として分けて残す
        assert handoff.ai_words == "全部済みました"

    def test_an_approval_folded_before_an_answer_hands_off(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、確かめの返事が無いまま時間切れで断られた操作が、要約から抜ける
        widget, _ = recorded
        _send(widget, "保存して")
        approval = Approval(tool="save_project", summary="保存", arguments={})
        widget._bridge._approvals.put(approval)
        widget._check_approval()
        approval.deny("確認の返事がありませんでした")
        widget._check_approval()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert any("返事がありません" in item for item in handoff.refused)

    def test_a_refused_send_hands_off(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、絵を送るのを断ったせいで確かめられなかった所が、要約に出ない
        widget, _ = recorded
        _send(widget, "見て直して")
        widget.ask_send = lambda _parent, _request: SendAnswer.DENY
        widget._bridge._send_requests.put(
            SendRequest(tool="preview_frame", kinds=(Outbound.FRAMES,))
        )
        widget._check_send()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert any("preview_frame" in item for item in handoff.refused)

    def test_an_unanswered_question_hands_off(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、答えずに閉じた質問のせいで止まったことが、要約から読めない
        widget, _ = recorded
        _send(widget, "切って")
        question = Question(
            question_spec(
                {"question": "どれを切りますか", "choices": [{"label": "A"}, {"label": "B"}]}
            )
        )
        widget._bridge._questions.put(question)
        widget._check_question()
        question.give_up("質問への返事がありませんでした")
        widget._check_question()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        handoff = _shelved(widget)
        assert handoff is not None
        assert any("どれを切りますか" in item for item in handoff.refused)

    def test_a_lost_connection_hands_off_with_the_dropped_prompts(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、応えている途中で会話が閉じたとき、続けて送ってあった指示が黙って消える
        widget, _ = recorded
        _send(widget, "冒頭を切って")
        _send(widget, "字幕を付けて")
        widget._handle(AgentEvent(EventKind.ERROR, text="ConnectionError: 切れました"))
        widget._handle(AgentEvent(EventKind.CLOSED))
        handoff = _shelved(widget)
        assert handoff is not None
        assert handoff.reason is StopReason.DISCONNECTED
        # 接続が切れる前のエラーの文も、止まった訳に残す
        assert "切れました" in handoff.detail
        assert any("字幕を付けて" in item for item in handoff.remaining)

    def test_a_turn_that_finished_hands_nothing_off(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、ふつうに終えた指示のたびに要約が出て、本当に止まった時に読まれない
        widget, _ = recorded
        _send(widget, "冒頭を切って")
        _cut_twice(widget)
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert _shelved(widget) is None
        assert widget._handoff_box.isHidden() is True
        assert _logged(widget) == []


class TestTheBox:
    def _stopped(self, widget: ChatPanel) -> None:
        _send(widget, "冒頭を切って")
        _cut_twice(widget)
        widget._handle(AgentEvent(EventKind.ERROR, text="落ちました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))

    def test_the_summary_can_be_copied(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]], fake_clipboard: FakeClipboard
    ) -> None:
        # 壊れると、要約を Issue や別の窓の AI へ渡すのに、会話の欄から手で拾い集める
        widget, _ = recorded
        self._stopped(widget)
        widget._handoff_copy.click()
        handoff = _shelved(widget)
        assert handoff is not None
        assert fake_clipboard.text() == handoff.text()

    def test_continuing_fills_the_input_without_sending(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、押しただけで続きが走り出すか、書きかけの指示が消える 中身を入力欄へ写すと、
        # 伏せるはずの素材の名前が指示の文として外へ出る
        widget, made = recorded
        self._stopped(widget)
        sent = list(made[0].prompts)
        widget._input.setPlainText("BGM も下げて")
        widget._handoff_continue.click()
        written = widget._input.toPlainText()
        assert written.startswith(CONTINUE_PROMPT)
        assert "BGM も下げて" in written
        assert "落ちました" not in written
        assert made[0].prompts == sent

    def test_the_next_instruction_lowers_the_box_but_keeps_the_summary(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、新しい指示も止まったように見えるか、続きを頼んだ AI が要約を読めない
        widget, _ = recorded
        self._stopped(widget)
        widget._handoff_continue.click()
        widget.send()
        assert widget._handoff_box.isHidden() is True
        assert _shelved(widget) is not None
        assert len(_logged(widget)) == 1

    def test_a_queued_instruction_starting_keeps_the_box(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、続けて送ってあった指示が始まった途端に欄が消え、止まった指示の要約を
        # コピーできない 欄にはどの指示の要約かも出す（走っている指示と取り違えない）
        widget, _ = recorded
        _send(widget, "冒頭を切って")
        _send(widget, "字幕を付けて")
        widget._handle(AgentEvent(EventKind.ERROR, text="落ちました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert widget._handoff_box.isHidden() is False
        assert "AI: 冒頭を切って" in widget._handoff_text.text()

    def test_finishing_later_turns_are_counted(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、止まった後の指示で形が変わったのに、AI が要約の形を今の形と思って続ける
        widget, _ = recorded
        self._stopped(widget)
        _send(widget, "字幕を付けて")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert shelf_for(widget._host.document).later == 1

    def test_the_summary_survives_a_theme_change(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、テーマを替えたときに会話の欄を書き直し、要約が欄から消える
        widget, _ = recorded
        self._stopped(widget)
        widget._redraw_log()
        assert "引き継ぎ" in widget._view.toPlainText()


class TestPreference:
    def test_turning_it_off_keeps_the_summary_for_the_ai(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 切った人には出さない 切っても要約は作り、続きを頼まれた AI は読める
        widget, _ = recorded
        widget.apply_preferences(Preferences(ai_handoff=False))
        _send(widget, "冒頭を切って")
        widget._handle(AgentEvent(EventKind.ERROR, text="落ちました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert widget._handoff_box.isHidden() is True
        assert _logged(widget) == []
        assert _shelved(widget) is not None

    def test_the_default_shows_it(self) -> None:
        # 既定は出す 知らない人が、止まったことに気付かないまま次の指示を重ねないように
        assert Preferences().ai_handoff is True

    def test_the_choice_survives_a_restart(self, tmp_path: Path) -> None:
        # 保存し忘れると、切ったはずの要約が次の起動で戻る
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(ai_handoff=False)
        store.save(chosen)
        assert store.load() == chosen

    def test_the_settings_dialog_keeps_it(self, qt_application: QApplication) -> None:
        # 出し忘れると、設定を開いて OK を押しただけで既定へ戻る
        from sashimono.ui.preferences_dialog import PreferencesDialog

        del qt_application
        chosen = Preferences(ai_handoff=False)
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences() == chosen
        finally:
            dialog.deleteLater()


@pytest.mark.parametrize("subtype_detail", ["", LIMIT_DETAIL])
def test_the_status_line_returns_to_idle(
    subtype_detail: str, recorded: tuple[ChatPanel, list[_RecordingSession]]
) -> None:
    # 壊れると、引き継ぎを出した後も状態の行が実行中のまま残り、まだ動いていると思って待ち続ける
    widget, _ = recorded
    _send(widget, "切って")
    widget._handle(AgentEvent(EventKind.ERROR, text="止まった", detail=subtype_detail))
    widget._handle(AgentEvent(EventKind.TURN_DONE))
    assert widget._status_text.text() == "待機中"
