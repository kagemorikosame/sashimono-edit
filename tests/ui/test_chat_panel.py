"""AI チャットパネル

Claude そのものは呼ばない パネルの仕事は「出来事を見せる」「確認を取る」
「1 つの指示をまとめて 1 段の履歴にする」の 3 つなので、そこを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, cast

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent, QImage, QInputMethodEvent, QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)

from sashimono.ai.bridge import Approval, EditorBridge, Question
from sashimono.ai.models import MODELS
from sashimono.ai.operations import ASK_USER, question_spec
from sashimono.ai.safety import Consent
from sashimono.ai.session import (
    ASK_RULES,
    SYSTEM_PROMPT,
    AgentEvent,
    AgentSession,
    EventKind,
    system_prompt,
)
from sashimono.core.commands import SetLayerMode, SplitClip
from sashimono.core.model import LayerMode, MediaItem, Project, Transcript
from sashimono.ui.chat import ChatPanel
from sashimono.ui.chat.panel import IRREVERSIBLE_ALLOW_TEXT, IRREVERSIBLE_HEAD, _Input
from sashimono.ui.theme import PALETTES, THEME_DARK, THEME_LIGHT, theme_signals, use_palette
from sashimono.ui.workspace import Preferences
from tests.ai.conftest import FakeHost, make_loaded


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """試験の間に確かめの窓や問い合わせの窓が開いたら、その場で落とす

    開いたままだと、無人の CI では誰も閉じずに試験が止まる 質問の箱（#296）は
    パネルの中に出す物で、窓を開かないことも合わせて押さえる
    """

    def refuse(*_arguments: object, **_options: object) -> int:
        raise AssertionError("試験の間に窓が開いた")

    monkeypatch.setattr(QDialog, "exec", refuse)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, refuse)


@pytest.fixture
def loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    return make_loaded(video_media, transcript)


@pytest.fixture
def panel(qt_application: QApplication, loaded: Project) -> Iterator[tuple[ChatPanel, FakeHost]]:
    del qt_application
    host = FakeHost(loaded)
    created = ChatPanel(host)
    yield created, host
    created.close_session()
    created.deleteLater()


def _text(panel: ChatPanel) -> str:
    return panel._view.toPlainText()


class TestConversationView:
    def test_assistant_text_is_shown(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、AI の説明や問い返しが会話の欄に出ず、何をしたのか本人に伝わらない
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TEXT, text="切りました"))
        assert "切りました" in _text(widget)
        assert "Claude" in _text(widget)

    def test_tool_calls_are_visible(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 何をされているか分からないまま編集が進むのが一番怖い
        widget, _ = panel
        widget._handle(
            AgentEvent(EventKind.TOOL_USE, tool="split_clip", detail="clip_id=abc, frame=30")
        )
        shown = _text(widget)
        assert "split_clip" in shown
        assert "frame=30" in shown

    def test_failed_tool_results_are_shown(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、AI の操作が失敗しても理由が出ず、編集が済んだと思い込んで先へ進む
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TOOL_RESULT, text="失敗", detail="見つかりません"))
        assert "見つかりません" in _text(widget)

    def test_successful_tool_results_stay_quiet(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 成功のたびに中身を出すと、会話が JSON で埋まる
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TOOL_RESULT, detail='{"ok": true}'))
        assert "ok" not in _text(widget)

    def test_the_conversation_is_rewritten_in_the_new_theme(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 色は HTML に焼き込んである 書き直さないと、テーマを切り替えても前の色が残り、
        # 暗いテーマの白に近い字が明るい地の上で読めなくなる
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TEXT, text="切りました"))
        widget._handle(AgentEvent(EventKind.TOOL_USE, tool="split_clip", detail="frame=30"))
        try:
            use_palette(THEME_LIGHT)
            theme_signals().changed.emit()
            shown = widget._view.toHtml()
            assert PALETTES[THEME_LIGHT]["ACCENT"].name() in shown
            assert PALETTES[THEME_LIGHT]["TEXT_MUTED"].name() in shown
            assert PALETTES[THEME_DARK]["ACCENT"].name() not in shown
            assert "切りました" in _text(widget)
            assert "split_clip" in _text(widget)
        finally:
            use_palette(THEME_DARK)

    def test_bold_and_code_in_the_reply_are_formatted(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # Claude の返事は素の Markdown で来る 組まないと ** と ` がそのまま本文に混ざる
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TEXT, text="**強調** と `set_param`"))
        assert "**" not in _text(widget)
        assert "`" not in _text(widget)
        # 本人の指示は書いたとおりに見せる 組むと、書いた ** が消えて伝わらない
        widget._say("あなた", "**そのまま**")
        assert "**そのまま**" in _text(widget)

    def test_errors_are_labelled(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 印が無いと、Claude Code が見つからない等の失敗が AI の返事に紛れて見落とされる
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.ERROR, text="Claude Code が見つかりません"))
        assert "エラー" in _text(widget)
        assert "Claude Code" in _text(widget)

    def test_html_in_the_text_is_escaped(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 返事の中の < > を書式として読むと、タグに見える部分の文字が消え、AI の言ったことが欠ける
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TEXT, text="<b>太字にはしない</b>"))
        assert "<b>太字にはしない</b>" in _text(widget)


class TestApproval:
    def _pending(self, widget: ChatPanel) -> Approval:
        approval = Approval(tool="split_clip", summary="クリップを分割\nframe=30", arguments={})
        widget._bridge._approvals.put(approval)
        widget._check_approval()
        return approval

    def test_a_request_is_shown_with_its_arguments(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、どのクリップの何フレーム目を切るのか分からないまま許可を押すことになる
        widget, _ = panel
        self._pending(widget)
        # パネル自体を画面に出していないので isVisible は使えない 表示の指示だけを見る
        assert widget._approval_box.isHidden() is False
        assert "frame=30" in widget._approval_text.text()

    def test_allowing_releases_the_waiting_tool(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、許可を押しても AI の操作が待ったまま進まず、確かめの欄も残り続ける
        widget, _ = panel
        approval = self._pending(widget)
        widget._answer(True)
        assert approval.allowed is True
        assert approval.done.is_set()
        assert widget._approval_box.isHidden() is True

    def test_denying_is_recorded(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、拒んだはずの操作が走るか、AI が答えを待ち続けて会話が止まる
        widget, _ = panel
        approval = self._pending(widget)
        widget._answer(False)
        assert approval.allowed is False
        assert approval.done.is_set()

    def test_always_allow_covers_the_next_call(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、「以降は確認しない」を押しても同じ操作のたびに確かめが出続ける
        widget, _ = panel
        approval = self._pending(widget)
        widget._answer(True, always=True)
        assert approval.allowed is True
        assert "split_clip" in widget._bridge._always

    def test_auto_approve_is_off_by_default(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 既定で入っていると、知らずに使い始めた人の作品を AI が確かめ無しに書き換える
        widget, _ = panel
        assert widget._bridge.auto_approve is False
        widget._auto.setChecked(True)
        assert widget._bridge.auto_approve is True

    def test_turning_auto_approve_off_forgets_the_exceptions(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、自動の承認を切って確かめに戻したつもりでも、
        # 前に「以降は確認しない」にした操作だけは黙って走る
        widget, _ = panel
        widget._bridge.allow_always("split_clip")
        widget._auto.setChecked(True)
        widget._auto.setChecked(False)
        assert widget._bridge._always == set()

    def test_an_interrupted_request_disappears(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、中断した後も答える相手の無い確かめの欄が残り、押しても何も起きない
        widget, _ = panel
        self._pending(widget)
        widget._bridge.cancel()
        widget._check_approval()
        assert widget._approval_box.isHidden() is True

    def _irreversible(self, widget: ChatPanel) -> Approval:
        consent = Consent(action="プロジェクトを上書き保存します", target="C:/作品/作品.sme")
        approval = Approval(
            tool="save_project", summary=consent.text(), arguments={}, consent=consent
        )
        widget._bridge._approvals.put(approval)
        widget._check_approval()
        return approval

    def test_an_irreversible_request_says_so_and_offers_no_always(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、ふだんの確かめと見分けが付かず、戻せない上書きを編集のつもりで許す
        # 「以降は確認しない」が出ていると、押しても毎回確かめるので効かないボタンになる
        widget, _ = panel
        self._irreversible(widget)
        assert widget._approval_text.text().startswith(IRREVERSIBLE_HEAD)
        assert "C:/作品/作品.sme" in widget._approval_text.text()
        assert widget._approval_always.isHidden() is True
        assert widget._approval_allow.text() == IRREVERSIBLE_ALLOW_TEXT

    def test_the_next_ordinary_request_gets_its_buttons_back(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、1 度取り消せない確かめが出た後、ふだんの編集で「以降は確認しない」が消えたまま
        widget, _ = panel
        self._irreversible(widget)
        widget._answer(True)
        self._pending(widget)
        assert widget._approval_always.isHidden() is False
        assert widget._approval_allow.text() == "許可"

    def test_always_on_an_irreversible_request_is_not_remembered(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、隠したボタンの経路から押された「以降は確認しない」で、次の上書きが黙って走る
        widget, _ = panel
        approval = self._irreversible(widget)
        widget._answer(True, always=True)
        assert approval.allowed is True
        assert widget._bridge._always == set()

    def test_the_auto_approve_box_tells_what_it_does_not_skip(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、自動で承認に任せて席を外した人が、確かめで止まっていたことに後で気付く
        widget, _ = panel
        assert "取り消せない操作" in widget._auto.toolTip()


class TestCheckpoint:
    def test_one_prompt_becomes_one_undo_step(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、AI が 1 つの指示で行った編集を戻すのに何度も取り消しを押し、
        # 途中で数え違えると半端に切れた状態が残る
        widget, host = panel
        clip = host.document.project.timeline.tracks[0].clips[0].id

        widget._open_checkpoint("冒頭を切って")
        # AI が 3 回操作した、という想定
        host.apply_commands([SplitClip(clip, 60)], "分割")
        host.apply_commands([SplitClip(clip, 30)], "分割")
        second = host.document.project.timeline.tracks[0].clips[-1].id
        host.apply_commands([SplitClip(second, 120)], "分割")
        widget._close_checkpoint()

        assert len(host.document.project.timeline.tracks[0].clips) == 4
        # 3 回の編集が 1 段 取り消し 1 回で最初の状態へ戻る
        assert host.document.history_labels == ("AI: 冒頭を切って",)
        host.document.undo()
        assert len(host.document.project.timeline.tracks[0].clips) == 1

    def test_the_label_is_trimmed(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 長い指示の全文を段の名前にすると、履歴の一覧や「元に戻す」の文字が横に伸びて読めない
        widget, host = panel
        widget._open_checkpoint("あ" * 100)
        clip = host.document.project.timeline.tracks[0].clips[0].id
        host.apply_commands([SplitClip(clip, 60)], "分割")
        widget._close_checkpoint()
        assert len(host.document.history_labels[0]) <= 28

    def test_a_turn_that_changed_nothing_leaves_no_step(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、AI が何も変えなかった指示にも段ができ、
        # 取り消しを押しても何も戻らない 1 回が混ざる
        widget, host = panel
        widget._open_checkpoint("何もしないで")
        widget._close_checkpoint()
        assert host.document.can_undo is False

    def test_the_turn_ending_closes_the_checkpoint(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、応答が終わっても段が開いたままで、その後の手の編集まで AI の段に飲み込まれる
        widget, host = panel
        widget._open_checkpoint("切って")
        clip = host.document.project.timeline.tracks[0].clips[0].id
        host.apply_commands([SplitClip(clip, 60)], "分割")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert host.document.can_undo is True

    def test_closing_the_session_closes_an_open_checkpoint(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 開いたまま終わると、以降の編集が全部その 1 段に飲み込まれる
        widget, host = panel
        widget._open_checkpoint("途中で閉じる")
        clip = host.document.project.timeline.tracks[0].clips[0].id
        host.apply_commands([SplitClip(clip, 60)], "分割")
        widget.close_session()
        assert host.document.in_checkpoint is False
        assert host.document.can_undo is True


class TestSending:
    def test_an_empty_prompt_does_nothing(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、空のまま送りを押しただけで空の指示が Claude へ飛び、中身の無い履歴の段が開く
        widget, host = panel
        widget._input.setPlainText("   ")
        widget.send()
        assert host.document.in_checkpoint is False
        assert _text(widget).strip() == "" or "案内" in _text(widget)


class TestFormatting:
    def test_bold_and_code_are_rendered(self) -> None:
        # 壊れると、返事の中の道具や値の名前が本文と見分けられず、何を変えたのか読み取りにくい
        from sashimono.ui.chat.panel import _to_html

        # Claude の返事は素の Markdown で来る そのまま出すと ** が本文に混ざる
        rendered = _to_html("**強調** と `set_param`")
        assert "<b>強調</b>" in rendered
        assert "<code" in rendered and "set_param" in rendered

    def test_tags_in_the_text_are_neutralised_first(self) -> None:
        # 組む前に無害にしないと、返事に混ざった <b> などが書式として効き、
        # 文字が消えたり崩れたりする
        from sashimono.ui.chat.panel import _to_html

        rendered = _to_html("<b>これは太字にしない</b>")
        assert "&lt;b&gt;" in rendered


def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """導入が済んだことにする 本物の SDK と claude の有無に左右されないように"""
    from sashimono.ai import AI_PACK
    from sashimono.runtime import PackageStatus, PackStatus
    from sashimono.ui.setup import SetupSection

    status = PackStatus(pack=AI_PACK, packages=(PackageStatus("claude-agent-sdk", "0.2.158"),))
    monkeypatch.setattr(SetupSection, "status", property(lambda _self: status))


def _press(widget: QWidget, key: Qt.Key, modifiers: Qt.KeyboardModifier) -> None:
    QTest.keyClick(widget, key, modifiers)


class TestSendKey:
    """Enter で送り、Shift+Enter で改行する 日本語の変換を確定する Enter では送らない"""

    @pytest.fixture
    def typed(self, panel: tuple[ChatPanel, FakeHost]) -> tuple[ChatPanel, list[bool]]:
        widget, _ = panel
        sent: list[bool] = []
        widget._input.submitted.disconnect()
        widget._input.submitted.connect(lambda: sent.append(True))
        widget._input.setEnabled(True)
        widget._input.setPlainText("冒頭を切って")
        widget._input.moveCursor(QTextCursor.MoveOperation.End)
        return widget, sent

    def test_enter_sends(self, typed: tuple[ChatPanel, list[bool]]) -> None:
        # 壊れると、Enter を押しても改行が入るだけで、指示が送れないように見える
        widget, sent = typed
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert sent == [True]
        assert "\n" not in widget._input.toPlainText()

    def test_the_keypad_enter_also_sends(self, typed: tuple[ChatPanel, list[bool]]) -> None:
        # テンキーの Enter は修飾（Keypad）付きで届く 素の Enter と同じに扱わないと、
        # テンキーで押す人だけ送れずに改行が入る
        widget, sent = typed
        _press(widget._input, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
        assert sent == [True]

    def test_shift_enter_breaks_the_line(self, typed: tuple[ChatPanel, list[bool]]) -> None:
        # 壊れると、改行のつもりの Shift+Enter で書きかけの指示が送られる
        widget, sent = typed
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        assert sent == []
        # 送った文の中で改行として読まれる、ふつうの改行が入る
        assert widget._input.toPlainText() == "冒頭を切って\n"

    def test_ctrl_enter_still_sends(self, typed: tuple[ChatPanel, list[bool]]) -> None:
        # 前の版で覚えた押し方も効く 壊れると、前の版の Ctrl+Enter で送っていた人が
        # 押しても何も起きず、送れなくなったように見える
        widget, sent = typed
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
        assert sent == [True]

    def test_enter_while_converting_does_not_send(
        self, typed: tuple[ChatPanel, list[bool]]
    ) -> None:
        """変換を確定する Enter で送ると、書きかけの文が飛んでいく"""
        widget, sent = typed
        QApplication.sendEvent(widget._input, QInputMethodEvent("へんかん", []))
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert sent == []

        committed = QInputMethodEvent("", [])
        committed.setCommitString("変換")
        QApplication.sendEvent(widget._input, committed)
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert sent == [True]

    def test_the_old_way_can_be_chosen(self, typed: tuple[ChatPanel, list[bool]]) -> None:
        # 壊れると、Ctrl+Enter で送る設定にした人が、改行のつもりの Enter で
        # 長い指示を途中まで送ってしまう
        widget, sent = typed
        widget.apply_preferences(Preferences(chat_enter_sends=False))
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert sent == []
        assert "Ctrl+Enter" in widget._input.placeholderText()
        _press(widget._input, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
        assert sent == [True]

    def test_the_placeholder_tells_the_keys(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 書いていないと、前の版の Ctrl+Enter を覚えた人も、改行したい人も押し方が分からない
        widget, _ = panel
        assert "Enter で送信" in widget._input.placeholderText()
        assert "Shift+Enter で改行" in widget._input.placeholderText()


def _preedit(widget: QWidget, text: str) -> None:
    """日本語入力の変換中の文字（まだ確定していない読み）を送る 空なら変換の終わり"""
    QApplication.sendEvent(widget, QInputMethodEvent(text, []))


def _drawn(widget: QWidget) -> QImage:
    return widget.grab().toImage()


class TestPlaceholderWhileConverting:
    """日本語の変換中は案内の文を隠し、確定・取り消し・欄を離れたら戻す（#297）

    Qt は文書が空なら案内の文を描く 変換中の読みは文書の外の層にあるので、
    何もしないと読みと案内の文が重なって描かれる
    """

    def test_the_placeholder_is_hidden_while_converting(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、打っている読み「てきすと」の上に薄い案内の文が重なって読めない
        widget, _ = panel
        placeholder = widget._input.placeholder
        _preedit(widget._input, "てきすと")
        assert widget._input.toPlainText() == ""
        assert widget._input.placeholderText() == ""
        assert widget._input.placeholder == placeholder

    def test_what_is_drawn_while_converting_has_no_placeholder(
        self, qt_application: QApplication
    ) -> None:
        # 文字の値だけでなく描かれた絵で見る 壊れると、案内の文のある欄だけ変換中の絵が違う
        del qt_application
        with_text, without = _Input(), _Input()
        with_text.set_placeholder("編集の指示を書いて Enter で送信")
        try:
            for field in (with_text, without):
                field.resize(320, 60)
                _preedit(field, "てきすと")
            assert _drawn(with_text) == _drawn(without)
        finally:
            with_text.deleteLater()
            without.deleteLater()

    def test_committing_brings_the_placeholder_back(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、確定して送った後の空の欄に案内の文が出ず、何を書く欄か分からなくなる
        widget, _ = panel
        placeholder = widget._input.placeholder
        _preedit(widget._input, "てきすと")
        committed = QInputMethodEvent("", [])
        committed.setCommitString("テキスト")
        QApplication.sendEvent(widget._input, committed)
        assert widget._input.toPlainText() == "テキスト"
        widget._input.clear()
        assert widget._input.placeholderText() == placeholder

    def test_cancelling_brings_the_placeholder_back(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # Esc で取り消すと、何も確定せずに読みだけが消える 壊れると空の欄の案内の文が消えたまま
        widget, _ = panel
        placeholder = widget._input.placeholder
        _preedit(widget._input, "てきすと")
        _preedit(widget._input, "")
        assert widget._input.toPlainText() == ""
        assert widget._input.placeholderText() == placeholder

    def test_leaving_the_field_brings_the_placeholder_back(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 変換中に別の所を押した 壊れると、案内の文が消えたままになり、戻ってきて
        # 押す Enter も変換の確定と取られて送れない
        widget, _ = panel
        placeholder = widget._input.placeholder
        _preedit(widget._input, "てきすと")
        QApplication.sendEvent(widget._input, QFocusEvent(QEvent.Type.FocusOut))
        assert widget._input.placeholderText() == placeholder
        assert widget._input._composing is False

    def test_changing_the_send_key_while_converting_keeps_it_hidden(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 変換中に設定を変えた 壊れると、新しい文面の案内の文が読みの上へ戻って重なる
        widget, _ = panel
        _preedit(widget._input, "てきすと")
        widget.apply_preferences(Preferences(chat_enter_sends=False))
        assert widget._input.placeholderText() == ""
        _preedit(widget._input, "")
        # 戻すときは、変えた後の正しい文面で戻る 壊れると前の押し方を案内したまま
        assert "Ctrl+Enter で送信" in widget._input.placeholderText()
        assert "Shift+Enter" not in widget._input.placeholderText()

    def test_the_transcription_hint_hides_its_placeholder_too(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 字幕の起こしの「ヒント」欄も同じ症状が出ないかを絵で見る 1 行の欄（QLineEdit）は
        # 変換中に案内の文を描かない作り 欄を複数行の物へ替えると、壊れて重なる
        from sashimono.asr.service import TranscriptionService
        from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog

        del qt_application
        dialog = TranscribeDialog(video_media, cast(TranscriptionService, object()))
        try:
            field = dialog._prompt
            assert field.placeholderText()
            _preedit(field, "てきすと")
            with_placeholder = _drawn(field)
            field.setPlaceholderText("")
            assert with_placeholder == _drawn(field)
        finally:
            dialog.deleteLater()


class _RecordingSession:
    """会話の差し替え Claude へは繋がない 何で始めたかだけを覚える"""

    made: ClassVar[list[_RecordingSession]] = []

    def __init__(
        self,
        bridge: object,
        *,
        model: str | None,
        effort: str | None,
        system_prompt: str = SYSTEM_PROMPT,
        ask: bool = True,
    ) -> None:
        del bridge
        self.model = model
        self.effort = effort
        self.system_prompt = system_prompt
        self.ask = ask
        self.prompts: list[str] = []
        self.closed = False
        self.busy = False
        self.acknowledged = 0
        _RecordingSession.made.append(self)

    def acknowledge_turn(self) -> None:
        self.acknowledged += 1

    def send(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def poll(self) -> list[AgentEvent]:
        return []

    def close(self, *, wait: bool = True) -> None:
        del wait
        self.closed = True

    def interrupt(self) -> None:
        return


@pytest.fixture
def recorded(
    panel: tuple[ChatPanel, FakeHost], monkeypatch: pytest.MonkeyPatch
) -> tuple[ChatPanel, list[_RecordingSession]]:
    from sashimono.ui.chat import panel as panel_module

    _ready(monkeypatch)
    _RecordingSession.made = []
    monkeypatch.setattr(panel_module, "AgentSession", _RecordingSession)
    widget, _ = panel
    widget._refresh_availability()
    return widget, _RecordingSession.made


def _choose(box: QComboBox, value: str) -> None:
    box.setCurrentIndex(box.findData(value))


class TestModelChoice:
    def test_the_default_is_what_claude_code_picks(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 選べるようになる前と同じ動き 何も渡さない
        # 壊れると、そのモデルを使えないアカウントでは、何も選んでいないのに
        # 会話が始まらない
        widget, made = recorded
        widget._input.setPlainText("切って")
        widget.send()
        assert (made[0].model, made[0].effort) == (None, None)

    def test_a_mixed_project_gets_the_mixed_instructions(
        self,
        recorded: tuple[ChatPanel, list[_RecordingSession]],
        panel: tuple[ChatPanel, FakeHost],
    ) -> None:
        # 分ける方式の指示のまま混合の作品を触らせると、AI が無い組の片方を探し回る
        # モデルの選択と一緒に渡すので、どちらかを落とすとここで分かる
        widget, made = recorded
        _, host = panel
        host.apply_commands([SetLayerMode(LayerMode.MIXED)], "方式")
        _choose(widget._model, "claude-sonnet-5")
        widget._input.setPlainText("切って")
        widget.send()
        assert made[0].system_prompt == system_prompt(LayerMode.MIXED)
        assert made[0].model == "claude-sonnet-5"

    def test_a_mode_change_restarts_before_the_next_instruction(
        self,
        recorded: tuple[ChatPanel, list[_RecordingSession]],
        panel: tuple[ChatPanel, FakeHost],
    ) -> None:
        # 指示の文は会話を作るときにしか渡せない 作り直さないと、混合にした後も
        # 分ける方式の説明のまま話し続け、AI が無い組の片方を探し回る
        widget, made = recorded
        _, host = panel
        widget._input.setPlainText("切って")
        widget.send()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        host.apply_commands([SetLayerMode(LayerMode.MIXED)], "方式")
        widget._input.setPlainText("もう少し")
        widget.send()
        assert len(made) == 2
        assert made[0].closed
        assert made[1].system_prompt == system_prompt(LayerMode.MIXED)
        assert made[1].prompts == ["もう少し"]

    def test_the_same_mode_keeps_the_conversation(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 方式が同じなのに作り直すと、送るたびにそれまでのやり取りが消える
        widget, made = recorded
        for prompt in ("切って", "もう少し"):
            widget._input.setPlainText(prompt)
            widget.send()
            widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert len(made) == 1

    def test_a_mode_change_waits_for_the_pending_instruction(
        self,
        recorded: tuple[ChatPanel, list[_RecordingSession]],
        panel: tuple[ChatPanel, FakeHost],
    ) -> None:
        # 応答を待っている間に畳むと、送った指示が消える 終わった所で作り直す
        widget, made = recorded
        _, host = panel
        widget._input.setPlainText("切って")
        widget.send()
        host.apply_commands([SetLayerMode(LayerMode.MIXED)], "方式")
        widget._input.setPlainText("もう少し")
        widget.send()
        assert len(made) == 1
        assert made[0].prompts == ["切って", "もう少し"]
        for _ in range(2):
            widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert made[0].closed

    def test_the_listed_models_are_the_current_ones(self) -> None:
        # 一覧はネットに取りに行かない定数 欠けると、そのモデルを選ぶ手段が無くなる
        ids = {model.id for model in MODELS}
        assert {
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-haiku-4-5-20251001",
            "claude-fable-5-1",
        } <= ids

    def test_the_chosen_model_and_effort_are_used(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 渡し忘れると、欄の上には選んだモデルが出ているのに既定のモデルで話す
        widget, made = recorded
        _choose(widget._model, "claude-sonnet-5")
        _choose(widget._effort, "high")
        widget._input.setPlainText("切って")
        widget.send()
        assert (made[0].model, made[0].effort) == ("claude-sonnet-5", "high")

    def test_haiku_does_not_get_an_effort(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        """Haiku 4.5 はエフォートを受け付けない 渡すと会話が始まる前に失敗する"""
        widget, made = recorded
        _choose(widget._effort, "max")
        _choose(widget._model, "claude-haiku-4-5-20251001")
        assert widget._effort.isEnabled() is False
        widget._input.setPlainText("切って")
        widget.send()
        assert (made[0].model, made[0].effort) == ("claude-haiku-4-5-20251001", None)

    def test_changing_the_model_starts_a_new_conversation(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 繋ぎ直さないと、選んだモデルが効かないまま前のモデルで話し続ける
        widget, made = recorded
        widget._input.setPlainText("切って")
        widget.send()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        _choose(widget._model, "claude-opus-5-5")
        assert made[0].closed is True
        widget._input.setPlainText("もう一度")
        widget.send()
        assert made[1].model == "claude-opus-5-5"
        assert "新しい会話" in _text(widget)

    def test_a_change_right_after_sending_keeps_the_prompt(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        """送った直後は Claude Code の起動中で busy が立っていない

        そこで畳むと、送った指示が処理されないまま消え、履歴の段も開いたまま残る
        """
        widget, made = recorded
        widget._input.setPlainText("切って")
        widget.send()
        _choose(widget._model, "claude-opus-5-5")
        assert made[0].closed is False

        # 応答を待つ間に続けて送った指示も、前の会話で最後まで処理させる
        widget._input.setPlainText("続けて")
        widget.send()
        assert made[0].prompts == ["切って", "続けて"]
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert made[0].closed is False
        assert widget._stop_button.isEnabled() is True
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert made[0].closed is True
        assert widget._stop_button.isEnabled() is False

    def test_a_change_during_a_reply_waits_for_the_reply(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 応答の途中で畳むと、その応答が途中で切れる
        widget, made = recorded
        widget._input.setPlainText("切って")
        widget.send()
        made[0].busy = True
        _choose(widget._model, "claude-opus-5-5")
        assert made[0].closed is False
        made[0].busy = False
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert made[0].closed is True

    def test_choices_are_announced_for_saving(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、欄の上で選んだモデルが保存されず、次に起動したとき既定へ戻る
        widget, _ = panel
        announced: list[tuple[str, str]] = []
        widget.choices_changed.connect(lambda model, effort: announced.append((model, effort)))
        _choose(widget._model, "claude-fable-5-1")
        assert announced[-1] == ("claude-fable-5-1", "")

    def test_preferences_are_shown_without_announcing(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 読んだだけで「選び直した」と知らせると、起動のたびに保存が走る
        widget, _ = panel
        announced: list[tuple[str, str]] = []
        widget.choices_changed.connect(lambda model, effort: announced.append((model, effort)))
        widget.apply_preferences(Preferences(ai_model="claude-sonnet-5", ai_effort="low"))
        assert (widget.model, widget.effort) == ("claude-sonnet-5", "low")
        assert announced == []


class TestQueuedPrompts:
    def test_each_queued_prompt_gets_its_own_undo_step(
        self,
        recorded: tuple[ChatPanel, list[_RecordingSession]],
        loaded: Project,
    ) -> None:
        """応答を待つ間に続けて送った指示も、指示ごとに 1 回の取り消しで戻せる

        前の指示の段を閉じる前に次の指示の編集が入ると、2 つの指示の編集が
        1 つの段にまとまり、取り消し 1 回で両方が戻る
        """
        del loaded
        widget, made = recorded
        host = cast(FakeHost, widget._host)
        clip = host.document.project.timeline.tracks[0].clips[0].id

        widget._input.setPlainText("1 つ目")
        widget.send()
        widget._input.setPlainText("2 つ目")
        widget.send()
        host.apply_commands([SplitClip(clip, 60)], "分割")  # 1 つ目の指示の編集
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        # 段を付け替え終えてから、次の指示を始めさせる
        assert made[0].acknowledged == 1
        assert host.document.in_checkpoint is True
        host.apply_commands([SplitClip(clip, 30)], "分割")  # 2 つ目の指示の編集
        widget._handle(AgentEvent(EventKind.TURN_DONE))

        assert host.document.history_labels == ("AI: 1 つ目", "AI: 2 つ目")
        assert host.document.in_checkpoint is False

    def test_the_session_waits_for_the_boundary(self) -> None:
        # 区切りを待たずに次の指示を始めると、次の指示の編集が前の段へ混ざる
        session = AgentSession(cast(EditorBridge, object()))
        session._boundary.clear()  # 1 つ目の指示を始めた所
        assert session._boundary.is_set() is False
        session.acknowledge_turn()
        assert session._boundary.is_set() is True


class TestLogin:
    def test_the_login_guide_shows_when_no_credentials_are_found(
        self,
        recorded: tuple[ChatPanel, list[_RecordingSession]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、ログインしていない人に案内が出ないか、ログイン済みの人に
        # 案内が出続けて、済んだのかどうか分からない
        from sashimono.ui.chat import panel as panel_module

        widget, _ = recorded
        monkeypatch.setattr(panel_module, "credentials_found", lambda: False)
        widget._refresh_availability()
        assert widget._login_box.isHidden() is False
        assert "ANTHROPIC_API_KEY" in widget._login_text.text()

        monkeypatch.setattr(panel_module, "credentials_found", lambda: True)
        widget._refresh_availability()
        assert widget._login_box.isHidden() is True

    def test_the_login_button_opens_claude_code(
        self,
        recorded: tuple[ChatPanel, list[_RecordingSession]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # ログインは Claude Code 自身の画面で済ませる このソフトは鍵を受け取らない
        # 壊れると、「ログイン…」を押しても何も開かず、ソフトの中からログインする手段が無い
        from sashimono.ui.chat import panel as panel_module

        widget, _ = recorded
        cli = tmp_path / "claude.exe"
        opened: list[Path] = []
        monkeypatch.setattr(panel_module, "claude_cli", lambda: cli)
        monkeypatch.setattr(panel_module, "open_login_window", opened.append)
        widget._login_button.click()
        assert opened == [cli]


class TestLoginHint:
    def test_a_login_failure_says_where_to_log_in(self) -> None:
        # 壊れると、英語の「/login を実行して」だけが出て、ソフトの中のどこで
        # ログインすればよいのか分からない
        from sashimono.ai.session import with_login_hint

        text = with_login_hint("Invalid API key · Please run /login")
        assert "ログイン…" in text

    def test_other_failures_are_left_alone(self) -> None:
        # 壊れると、ログインと関係の無い失敗にもログインの案内が付き、原因を取り違える
        from sashimono.ai.session import with_login_hint

        assert with_login_hint("接続が切れました") == "接続が切れました"


class _Clock:
    """経過秒を思いどおりに進める時計 本物の時計だと秒の境目で試験が揺れる"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def timed(
    recorded: tuple[ChatPanel, list[_RecordingSession]],
) -> tuple[ChatPanel, list[_RecordingSession], _Clock]:
    widget, made = recorded
    clock = _Clock()
    widget._clock = clock
    return widget, made, clock


def _status(widget: ChatPanel) -> str:
    return widget._status_text.text()


def _send(widget: ChatPanel, prompt: str) -> None:
    widget._input.setPlainText(prompt)
    widget.send()


class TestRunningStatus:
    """入力欄の上の 1 行で、いま何をしているかが文字で分かる（#250）

    偽の会話へ出来事を流し、状態の行・区切りの行・送信ボタン・ドックの名前が
    移り変わるのを見る 本物の Claude Code は要らない
    """

    def test_idle_is_written_out(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 色だけに頼らない 何も出ていないと、止まっているのか待っているのか分からない
        widget, _, _ = timed
        assert _status(widget) == "待機中"
        assert widget._busy_mark.isHidden() is True

    def test_connecting_shows_right_after_sending(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # Claude Code の起動と接続の数秒も、送ったことが見える 壊れると固まったように見える
        widget, _, clock = timed
        _send(widget, "切って")
        assert _status(widget).startswith("接続しています…")
        assert widget._busy_mark.isHidden() is False
        clock.now += 3
        widget._poll()
        assert "3 秒" in _status(widget)

    def test_running_shows_the_tool_and_the_seconds(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 道具と秒が出ないと、止まっているのか動いているのかが分からない
        widget, _, clock = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.READY))
        assert _status(widget).startswith("実行中")
        widget._handle(AgentEvent(EventKind.TOOL_USE, tool="split_clip", detail="frame=30"))
        clock.now += 12
        widget._poll()
        assert _status(widget) == "実行中: split_clip（12 秒）"

    def test_a_finished_turn_leaves_a_divider(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 中断ボタンが灰色に戻るだけだと、終わったことに気付かない
        widget, _, clock = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.READY))
        for _ in range(5):
            widget._handle(AgentEvent(EventKind.TOOL_USE, tool="split_clip"))
        clock.now += 38
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "完了（38 秒・操作 5 件）" in _text(widget)
        assert _status(widget) == "待機中"
        assert widget._busy_mark.isHidden() is True

    def test_long_turns_are_written_in_minutes(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 125 秒と出されても長さが掴みにくい
        widget, _, clock = timed
        _send(widget, "切って")
        clock.now += 125
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "完了（2 分 5 秒・操作 0 件）" in _text(widget)

    def test_failed_operations_are_counted_in_the_divider(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 失敗を数えないと、一部の操作が効いていないことに気付かず書き出してしまう
        widget, _, _ = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.TOOL_USE, tool="split_clip"))
        widget._handle(AgentEvent(EventKind.TOOL_RESULT, text="失敗", detail="見つかりません"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "操作 1 件・失敗 1 件" in _text(widget)

    def test_an_error_ends_with_a_failure_divider(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # エラーで終わった指示を「完了」と出すと、編集が済んだと思い込む
        widget, _, _ = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.ERROR, text="接続が切れました"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "失敗（0 秒・操作 0 件）" in _text(widget)
        assert "完了（" not in _text(widget)

    def test_a_closed_session_ends_the_pending_turn(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 会話が落ちたとき、実行中のまま残ると、いつまでも待たされているように見える
        widget, _, _ = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.ERROR, text="Claude Code が見つかりません"))
        widget._handle(AgentEvent(EventKind.CLOSED))
        assert "失敗（" in _text(widget)
        assert _status(widget) == "待機中"

    def test_a_turn_ending_without_a_prompt_adds_no_divider(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 送っていないのに区切りが出ると、何かが終わったように見えて紛らわしい
        widget, _, _ = timed
        widget._handle(AgentEvent(EventKind.CLOSED))
        assert "完了（" not in _text(widget)
        assert "失敗（" not in _text(widget)

    def test_interrupting_shows_stopping_then_an_interrupted_divider(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 中断の途中と終わりが出ないと、止まったのか分からず中断を押し続ける
        widget, _, _ = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.READY))
        widget.interrupt()
        assert _status(widget).startswith("中断しています…")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "中断（0 秒・操作 0 件）" in _text(widget)
        assert _status(widget) == "待機中"

    def test_waiting_for_permission_is_distinct(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 欄の外から「止まっている」のか「許可を待っている」のかを見分けられるように
        widget, _, _ = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.READY))
        approval = Approval(tool="split_clip", summary="クリップを分割", arguments={})
        widget._bridge._approvals.put(approval)
        widget._poll()
        assert _status(widget).startswith("許可待ち")
        assert "split_clip" in _status(widget)
        widget._answer(True)
        assert _status(widget).startswith("実行中")

    def test_queued_prompts_are_counted(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 続けて送った指示が見えないと、送れていないと思ってもう一度送る
        widget, _, _ = timed
        assert widget._send_button.text() == "送信"
        _send(widget, "1 つ目")
        assert widget._send_button.text() == "追加で送る"
        _send(widget, "2 つ目")
        _send(widget, "3 つ目")
        assert "待ち 2 件" in _status(widget)
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "待ち 1 件" in _status(widget)
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "待ち" not in _status(widget)
        assert _status(widget).startswith("実行中")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert _status(widget) == "待機中"
        assert widget._send_button.text() == "送信"
        assert _text(widget).count("完了（") == 3

    def test_the_next_queued_turn_counts_its_own_time(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 前の指示の秒数と操作の数を引き継ぐと、2 つ目の区切りの数が大きく出る
        widget, _, clock = timed
        _send(widget, "1 つ目")
        _send(widget, "2 つ目")
        widget._handle(AgentEvent(EventKind.TOOL_USE, tool="split_clip"))
        clock.now += 10
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        clock.now += 4
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "完了（10 秒・操作 1 件）" in _text(widget)
        assert "完了（4 秒・操作 0 件）" in _text(widget)

    def test_the_dock_tab_is_marked_while_running(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # パネルが別のタブの裏に隠れていても、動いていると分かる
        from PySide6.QtWidgets import QDockWidget

        widget, _, _ = timed
        dock = QDockWidget("AI アシスタント")
        try:
            dock.setWidget(widget)
            _send(widget, "切って")
            assert dock.windowTitle() == "● AI アシスタント"
            widget._handle(AgentEvent(EventKind.TURN_DONE))
            assert dock.windowTitle() == "AI アシスタント"
        finally:
            # パネルは試験の後始末で消すので、ドックから外しておく
            dock.setWidget(QWidget())
            widget.setParent(None)
            dock.deleteLater()

    def test_ticking_seconds_does_not_rebuild_the_panel(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        """秒の更新は状態の行の文字だけを変える

        パネルや会話の欄を作り直すと、書きかけの指示の入力位置が飛び、重くもなる（#251）
        """
        widget, _, clock = timed
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.READY))
        widget._input.setPlainText("書きかけ")
        cursor = widget._input.textCursor()
        cursor.setPosition(2)
        widget._input.setTextCursor(cursor)
        label = widget._status_text
        logged = len(widget._log)
        html_before = widget._view.toHtml()

        set_calls: list[str] = []
        original = label.setText

        def counting(text: str) -> None:
            set_calls.append(text)
            original(text)

        label.setText = counting  # type: ignore[method-assign]
        for _ in range(3):
            widget._poll()  # 同じ秒のうちは書き換えない
        assert set_calls == []
        clock.now += 1
        widget._poll()
        assert len(set_calls) == 1

        assert widget._status_text is label
        assert len(widget._log) == logged
        assert widget._view.toHtml() == html_before
        assert widget._input.toPlainText() == "書きかけ"
        assert widget._input.textCursor().position() == 2

    def test_no_timer_is_left_after_closing(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 窓を閉じた後に時計が残ると、消えた部品を触って落ちる
        from PySide6.QtCore import QTimer

        widget, _, _ = timed
        _send(widget, "切って")
        widget.close_session()
        assert all(not timer.isActive() for timer in widget.findChildren(QTimer))

    def test_the_moving_mark_can_be_turned_off(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 動く物が苦手な人は切れる 切っても文字の状態は残す
        widget, _, _ = timed
        widget.apply_preferences(Preferences(ai_busy_animation=False))
        _send(widget, "切って")
        assert widget._busy_mark.isHidden() is True
        assert _status(widget).startswith("接続しています…")

    def test_the_mark_is_drawn_in_both_themes(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 色を焼き込むと、テーマを替えたとき地の色に溶けて見えなくなる
        from PySide6.QtGui import QColor, QImage

        widget, _, _ = timed
        mark = widget._busy_mark
        try:
            for theme in (THEME_DARK, THEME_LIGHT):
                use_palette(theme)
                image = QImage(mark.size(), QImage.Format.Format_ARGB32)
                image.fill(QColor(0, 0, 0, 0))
                mark.render(image)
                accent = PALETTES[theme]["ACCENT"].rgb()
                found = any(
                    image.pixel(x, y) == accent
                    for x in range(image.width())
                    for y in range(image.height())
                )
                assert found, theme
        finally:
            use_palette(THEME_DARK)

    def test_the_taskbar_is_flashed_only_when_asked(
        self,
        timed: tuple[ChatPanel, list[_RecordingSession], _Clock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 既定は切 入れた人には、別の窓を触っている間に終わったら知らせる
        widget, _, _ = timed
        alerted: list[object] = []
        monkeypatch.setattr(QApplication, "alert", lambda w, ms=0: alerted.append(w))
        _send(widget, "切って")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert alerted == []

        widget.apply_preferences(Preferences(ai_done_alert=True))
        _send(widget, "もう一度")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert len(alerted) == 1


class TestRunningStatusPreferences:
    def test_the_choices_survive_a_restart(self, tmp_path: Path) -> None:
        # 保存し忘れると、切ったはずの動く印が次の起動で戻る
        from sashimono.ui.workspace import PreferenceStore

        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(ai_busy_animation=False, ai_done_alert=True)
        store.save(chosen)
        assert store.load() == chosen

    def test_the_defaults_suit_someone_who_does_not_know(self) -> None:
        # 動く印は入、タスクバーの点滅は切（知らない人が驚かない側）
        plain = Preferences()
        assert (plain.ai_busy_animation, plain.ai_done_alert) == (True, False)

    def test_the_settings_dialog_keeps_them(self, qt_application: QApplication) -> None:
        # 出し忘れると、設定を開いて OK を押しただけで既定へ戻る
        from sashimono.ui.preferences_dialog import PreferencesDialog

        del qt_application
        chosen = Preferences(ai_busy_animation=False, ai_done_alert=True)
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences() == chosen
        finally:
            dialog.deleteLater()


def _question(*, multiple: bool = False, free_text: bool = True) -> Question:
    return Question(
        question_spec(
            {
                "question": "どのクリップを切りますか",
                "choices": [
                    {"label": "選択中", "description": "いま選んでいるクリップ"},
                    {"label": "全部", "description": "タイムラインのすべて"},
                    {"label": "BGM & 効果音", "description": "音声トラックの物"},
                ],
                "multiple": multiple,
                "free_text": free_text,
            }
        )
    )


def _shown(widget: ChatPanel, question: Question) -> None:
    """ブリッジに質問を積み、見張りの 1 回で画面に出させる"""
    widget._bridge._questions.put(question)
    widget._poll()


def _choice_button(widget: ChatPanel, label: str) -> QPushButton:
    return next(
        button
        for button in widget._question_choices.findChildren(QPushButton)
        if button.text().replace("&&", "&") == label
    )


class TestQuestion:
    """AI からの質問の箱（#296） 選ぶ・複数選ぶ・自由に書くの 3 通りで答えが返る"""

    def test_one_choice_answers_at_once(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 壊れると、ボタンを押しても AI に答えが届かず、返事を待ったまま止まる
        widget, _, _ = timed
        _send(widget, "ここを切って")
        question = _question()
        _shown(widget, question)
        assert widget._question_box.isHidden() is False
        assert widget._question_text.text() == "どのクリップを切りますか"
        _choice_button(widget, "全部").click()
        assert question.answered is True
        assert question.selected == ("全部",)
        assert widget._question_box.isHidden() is True
        # 会話の欄に「質問」と「答え」を残す 壊れると、後から何を聞かれて何と答えたか分からない
        shown = _text(widget)
        assert "質問" in shown
        assert "・全部: タイムラインのすべて" in shown
        assert shown.rstrip().endswith("全部")

    def test_an_ampersand_in_a_label_is_shown_and_answered_as_written(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # Qt は & を押し下げの印にして消す 壊れると「BGM  効果音」と見え、答えも食い違う
        widget, _ = panel
        question = _question()
        _shown(widget, question)
        button = _choice_button(widget, "BGM & 効果音")
        assert button.text() == "BGM && 効果音"
        button.click()
        assert question.selected == ("BGM & 効果音",)

    def test_several_choices_are_answered_together(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、複数選べる質問で 1 つ目を押した所で答えが出てしまうか、何も返らない
        widget, _ = panel
        question = _question(multiple=True)
        _shown(widget, question)
        assert widget._question_send.isHidden() is False
        # 何も選んでいない間は答えられない 押せると空の答えで AI が「どれでもよい」と読む
        assert widget._question_send.isEnabled() is False
        checks = dict(widget._question_checks)
        checks["全部"].setChecked(True)
        checks["選択中"].setChecked(True)
        assert question.done.is_set() is False
        widget._question_send.click()
        assert question.selected == ("選択中", "全部")

    def test_text_sent_from_the_input_is_the_answer(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 質問が出ている間に送った文は新しい指示にしない 壊れると、答えのつもりの文が
        # 次の指示として並び、今の指示は返事を待ったまま、並べた指示もいつまでも始まらない
        widget, made, _ = timed
        _send(widget, "ここを切って")
        question = _question()
        _shown(widget, question)
        _send(widget, "先頭の 3 秒だけ")
        assert question.answered is True
        assert question.free_text == "先頭の 3 秒だけ"
        assert question.selected == ()
        assert made[0].prompts == ["ここを切って"]
        assert list(widget._queued) == ["ここを切って"]
        assert widget._input.toPlainText() == ""

    def test_checked_choices_go_with_the_written_answer(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 複数選べる質問で、選んだうえに書き足した 壊れると、選んだ物が答えから落ちる
        widget, _ = panel
        question = _question(multiple=True)
        _shown(widget, question)
        dict(widget._question_checks)["全部"].setChecked(True)
        widget._input.setPlainText("音声は残す")
        widget.send()
        assert question.selected == ("全部",)
        assert question.free_text == "音声は残す"

    def test_the_box_says_where_the_written_answer_goes(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、入力欄に書いた文が答えになるのか新しい指示になるのか、送るまで分からない
        widget, _ = panel
        first = _question()
        _shown(widget, first)
        assert "入力欄から送った文は、この質問への答えになります" in widget._question_note.text()
        assert widget._question_other.isHidden() is False
        _choice_button(widget, "全部").click()
        _shown(widget, _question(free_text=False))
        assert widget._question_other.isHidden() is True
        assert "選択肢から選んでください" in widget._question_note.text()
        assert "入力欄からは答えられません" in widget._question_note.text()

    def test_choices_only_questions_do_not_take_the_input(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 選択肢に無い答えを断った質問 壊れると、AI が決めた答えの形の外の文が答えとして届く
        widget, made, _ = timed
        _send(widget, "ここを切って")
        question = _question(free_text=False)
        _shown(widget, question)
        assert widget._send_button.text() == "選択肢から答える"
        _send(widget, "先頭の 3 秒だけ")
        assert question.done.is_set() is False
        assert widget._question is question
        assert widget._question_box.isHidden() is False
        # 新しい指示にもしない 書いた文は消さずに残す（答えた後に送り直せるように）
        assert made[0].prompts == ["ここを切って"]
        assert widget._input.toPlainText() == "先頭の 3 秒だけ"
        assert "選択肢から答えてください" in _text(widget)
        _choice_button(widget, "全部").click()
        assert question.selected == ("全部",)
        assert question.free_text == ""

    def test_a_question_folded_before_it_is_shown_is_not_shown(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 中断の直後に積まれた・画面が塞がって時間が切れた質問 壊れると、押しても
        # 「もう閉じていた」になる箱が出て、状態の行も「質問待ち」のまま
        widget, _, _ = timed
        _send(widget, "ここを切って")
        question = _question()
        question.give_up("中断されました")
        _shown(widget, question)
        assert widget._question is None
        assert widget._question_box.isHidden() is True
        assert "質問を閉じました（中断されました）" in _text(widget)
        assert not _status(widget).startswith("質問待ち")

    def test_the_status_says_it_waits_for_an_answer(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 「実行中」のままだと、待たされているのがこちらの返事だと気付かない（#250 の表示）
        widget, _, _ = timed
        _send(widget, "ここを切って")
        widget._handle(AgentEvent(EventKind.READY))
        _shown(widget, _question())
        assert _status(widget).startswith("質問待ち")
        assert widget._send_button.text() == "答えを送る"
        _choice_button(widget, "選択中").click()
        assert _status(widget).startswith("実行中")
        assert widget._send_button.text() == "追加で送る"

    def test_a_folded_question_leaves_the_panel(
        self, timed: tuple[ChatPanel, list[_RecordingSession], _Clock]
    ) -> None:
        # 中断と時間切れでブリッジが畳んだ 壊れると、押しても届かない箱が残り続ける
        widget, _, _ = timed
        _send(widget, "ここを切って")
        _shown(widget, _question())
        widget._bridge.cancel()
        widget._poll()
        assert widget._question_box.isHidden() is True
        assert widget._question is None
        assert "質問を閉じました（中断されました）" in _text(widget)
        assert not _status(widget).startswith("質問待ち")

    def test_the_raw_call_line_is_not_shown(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 引数の 1 行（choices=[{...}]）は読めず、同じ質問が 2 回出たように見える
        widget, _ = panel
        widget._handle(AgentEvent(EventKind.TOOL_USE, tool=ASK_USER, detail="choices=[{...}]"))
        assert "choices=" not in _text(widget)
        assert widget._turn_tools == 1


class TestAskPreference:
    """設定の「あいまいなときは、AI が質問する」（#296）"""

    def test_turning_it_off_removes_the_rules_and_the_tool(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 壊れると、切ったのに聞き返され続ける
        widget, made = recorded
        widget.apply_preferences(Preferences(ai_ask_questions=False))
        _send(widget, "ここを切って")
        assert made[0].ask is False
        assert ASK_RULES not in made[0].system_prompt

    def test_it_is_on_from_the_start(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 既定は入（知らない人が、意図と違う操作をされて困らない側）
        widget, made = recorded
        assert Preferences().ai_ask_questions is True
        _send(widget, "ここを切って")
        assert made[0].ask is True
        assert ASK_RULES in made[0].system_prompt

    def test_changing_it_starts_a_new_conversation_when_idle(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 指示の文と道具は会話を作るときにしか渡せない 壊れると、切っても今の会話では聞かれる
        widget, made = recorded
        _send(widget, "ここを切って")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        widget.apply_preferences(Preferences(ai_ask_questions=False))
        assert made[0].closed is True
        _send(widget, "次")
        assert made[1].ask is False

    def test_changing_it_during_a_reply_moves_the_waiting_prompts(
        self, recorded: tuple[ChatPanel, list[_RecordingSession]]
    ) -> None:
        # 応えている間に切った 壊れると、続けて送った指示が古い会話（質問の道具と基準あり）で
        # 走り、切った後も質問される 待ちの指示は消さずに新しい会話へ渡す
        widget, made = recorded
        _send(widget, "1 つ目")
        _send(widget, "2 つ目")
        widget.apply_preferences(Preferences(ai_ask_questions=False))
        # 応えている最中の指示は止めない
        assert made[0].closed is False
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert made[0].closed is True
        # 古い会話は区切りを待ったまま畳む 区切りを渡すと 2 つ目を古い会話で始めてしまう
        assert made[0].acknowledged == 0
        assert len(made) == 2
        assert made[1].ask is False
        assert ASK_RULES not in made[1].system_prompt
        assert made[1].prompts == ["2 つ目"]
        assert list(widget._queued) == ["2 つ目"]
        # 2 つ目の取り消しの段も開いている 壊れると 2 つ目の編集が取り消しで戻らない
        assert widget._checkpoint_open is True

    def test_the_choice_survives_a_restart(self, tmp_path: Path) -> None:
        # 保存し忘れると、切ったはずの質問が次の起動で戻る
        from sashimono.ui.workspace import PreferenceStore

        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(ai_ask_questions=False))
        assert store.load().ai_ask_questions is False

    def test_the_settings_dialog_keeps_it(self, qt_application: QApplication) -> None:
        # 出し忘れると、設定を開いて OK を押しただけで入に戻る
        from sashimono.ui.preferences_dialog import PreferencesDialog

        del qt_application
        dialog = PreferencesDialog(Preferences(ai_ask_questions=False))
        try:
            assert dialog.preferences().ai_ask_questions is False
        finally:
            dialog.deleteLater()
