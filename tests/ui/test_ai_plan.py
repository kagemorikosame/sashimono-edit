"""AI のパネルの計画の欄（#340 D2）

偽の会話へ出来事を流し、AI の道具が書いた計画と進み具合が欄と状態の行に出ること、
〔区切りで止める〕〔続ける〕〔×〕が計画と会話に効くことを見る 本物の Claude Code は要らない
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from sashimono.ai.bridge import Approval
from sashimono.ai.budget import LimitAction, Limits
from sashimono.ai.fixes import fixes_closed
from sashimono.ai.handoff import Handoff, StopReason, shelf_for
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, find_operation
from sashimono.ai.plan import PAUSED_MESSAGE, RESUME_PROMPT, Stopped, plan_for
from sashimono.ai.session import LIMIT_DETAIL, PLAN_RULES, AgentEvent, EventKind
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.ui.chat import ChatPanel
from sashimono.ui.chat.plan_box import _STOPPED_HINTS, PAUSE_TEXT, PAUSING_TEXT
from sashimono.ui.workspace import Preferences
from tests.ai.conftest import FakeHost, make_loaded
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.ai.test_fix_loop import _MIXED_CLIP, _finish
from tests.ui.test_chat_panel import _ready, _RecordingSession


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 計画の欄はパネルの中に出す物で、どの操作も窓を出さない

    開いたままだと、無人の CI では誰も閉じずに試験が止まる
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
def made(
    qt_application: QApplication, loaded: Project, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ChatPanel, FakeHost, list[_RecordingSession]]]:
    """会話を差し替えたパネル 導入は済んだことにする"""
    from sashimono.ui.chat import panel as panel_module

    del qt_application
    _ready(monkeypatch)
    _RecordingSession.made = []
    monkeypatch.setattr(panel_module, "AgentSession", _RecordingSession)
    host = FakeHost(loaded)
    widget = ChatPanel(host)
    widget._refresh_availability()
    yield widget, host, _RecordingSession.made
    widget.close_session()
    widget.deleteLater()


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


def _send(widget: ChatPanel, prompt: str) -> None:
    widget._input.setPlainText(prompt)
    widget.send()


def _plan(host: FakeHost, *steps: str) -> None:
    """AI が set_plan を呼んだ（道具は編集画面のスレッドで動く）"""
    _operation("set_plan")(host, {"goal": "無音を詰めて書き出す", "steps": list(steps)})


def _mark(host: FakeHost, step: int, state: str) -> dict[str, Any]:
    result = _operation("update_plan_step")(host, {"step": step, "state": state})
    assert isinstance(result, dict)
    return result


def _shelved(host: FakeHost) -> Handoff | None:
    """棚に置いた今の作品の引き継ぎ（#346）"""
    return shelf_for(host.document).current(host.document)


def _log(widget: ChatPanel) -> str:
    return widget._view.toPlainText()


def _over_the_limit(widget: ChatPanel) -> None:
    """この指示でトークンの上限（尋ねずに止める）を使い切った所にする

    尋ねずに止める設定なら、道具を呼ぶとその場で断る 尋ねる設定だと画面のスレッドの返事を待つ
    """
    widget._bridge.usage.limits = Limits(turn_tokens=100, action=LimitAction.STOP)
    widget._bridge.usage.record_message("m", {"output_tokens": 100})


class TestPlanBox:
    def test_no_plan_no_box(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、計画の無い 1〜2 手の頼みでも空の欄が会話の欄の高さを食う
        widget, _, _ = made
        _send(widget, "ここを切って")
        widget._poll()
        assert widget._plan_box.isHidden() is True

    def test_the_plan_and_its_progress_are_shown(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、AI が長い作業の何をどの順でやるのか、どこまで済んだのかが見えない
        widget, host, _ = made
        _send(widget, "無音を詰めて書き出して")
        _plan(host, "無音を探す", "詰める", "書き出す")
        _mark(host, 1, "done")
        _mark(host, 2, "running")
        widget._poll()
        box = widget._plan_box
        assert box.isHidden() is False
        assert "無音を詰めて書き出す" in box.title()
        assert "手順 2/3" in box.title()
        assert box.steps_text().splitlines() == [
            "✓ 1. 無音を探す（済み）",
            "▸ 2. 詰める（取りかかり中）",
            "・ 3. 書き出す",
        ]
        # 状態の行でも、欄を巻き戻さずに今どこかが読める
        assert "手順 2/3" in widget._status_text.text()

    def test_the_plan_keeps_the_instruction_it_was_made_for(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、会話を作り直した後に〔続ける〕を押しても、AI は元の頼みが分からない
        widget, host, _ = made
        _send(widget, "全体の無音を詰めて、字幕を付けて書き出して")
        _plan(host, "a", "b")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        widget.resume_plan()
        _plan(host, "b")
        read = _operation("get_plan")(host, {})
        assert isinstance(read, dict)
        assert read["instruction"] == "全体の無音を詰めて、字幕を付けて書き出して"

    def test_a_long_goal_does_not_widen_the_panel(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、目標の長さで AI のパネルの最小の幅が広がり、1366 の画面に窓が収まらない
        widget, host, _ = made
        before = widget.minimumSizeHint().width()
        _send(widget, "長い作業")
        _operation("set_plan")(host, {"goal": "長" * 200, "steps": ["手" * 80]})
        widget._poll()
        assert widget.minimumSizeHint().width() <= before


class TestPauseAndResume:
    def test_pausing_stops_at_the_boundary_and_offers_to_resume(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、〔区切りで止める〕を押しても止まらないか、止まった後に続け方が分からない
        widget, host, sessions = made
        _send(widget, "無音を詰めて書き出して")
        _plan(host, "無音を探す", "詰める", "書き出す")
        _mark(host, 1, "running")
        widget._poll()
        box = widget._plan_box
        assert box.pause_button.isVisibleTo(box) and box.pause_button.isEnabled()
        assert box.resume_button.isVisibleTo(box) is False
        box.pause_button.click()
        widget._poll()
        assert box.pause_button.text() == PAUSING_TEXT
        assert box.pause_button.isEnabled() is False
        assert "区切りで止めます" in widget._status_text.text()
        assert _mark(host, 1, "done")["paused"] is True
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "計画を区切りで止めました（1/3 済み）" in _log(widget)
        assert Stopped.PAUSED.value in box.stopped_text()
        assert box.resume_button.isVisibleTo(box) and box.resume_button.isEnabled()
        box.resume_button.click()
        assert sessions[0].prompts[-1] == RESUME_PROMPT
        assert list(widget._queued) == [RESUME_PROMPT]
        # 続きも 1 つの指示として取り消しの段を開く 壊れると続きの編集が取り消しで戻らない
        assert widget._checkpoint_open is True
        # 止めは外れている 壊れると続きの初めの道具から断られる
        assert plan_for(host.document).refusal("split_clip") == ""
        widget._poll()
        assert box.pause_button.text() == PAUSE_TEXT

    def test_interrupting_leaves_the_step_marked_as_maybe_half_done(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、〔中断〕で手順の途中で止まったことが残らず、続きの AI が確かめずに進める
        widget, host, _ = made
        _send(widget, "無音を詰めて書き出して")
        _plan(host, "無音を探す", "詰める")
        _mark(host, 1, "running")
        widget.interrupt()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        box = widget._plan_box
        assert Stopped.INTERRUPTED.value in box.stopped_text()
        assert "途中" in box.stopped_text()
        assert box.resume_button.isEnabled()

    def test_a_limit_stop_is_kept_as_the_reason(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、使う量の上限（#344）で止めた指示が計画の欄に「応え終えた」と残り、会話の欄の
        # 「上限で停止」と食い違う 続きの AI も途中の手順を確かめずに進める
        widget, host, _ = made
        _send(widget, "無音を詰めて書き出して")
        _plan(host, "無音を探す", "詰める")
        _mark(host, 1, "running")
        _over_the_limit(widget)
        with pytest.raises(ToolError, match="上限"):
            widget._bridge.invoke(_operation("get_selection"), {})
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "上限で停止" in _log(widget)
        box = widget._plan_box
        assert Stopped.LIMIT.value in box.stopped_text()
        assert "AI の上限" in box.stopped_text()
        assert box.resume_button.isEnabled()

    def test_a_paused_plan_refuses_before_the_limit(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、区切りで止めた後の道具にも上限の確かめが先に効き、止めた人に「上限を広げて
        # 続けますか」を尋ねる（広げても区切りで断るだけ） 上限で止めた印も残り、区切りで止めた
        # 指示が「上限で停止」と書かれる
        widget, host, _ = made
        _send(widget, "無音を詰めて書き出して")
        _plan(host, "無音を探す", "詰める")
        _mark(host, 1, "running")
        assert plan_for(host.document).request_pause()
        _mark(host, 1, "done")
        _over_the_limit(widget)
        with pytest.raises(ToolError) as refused:
            widget._bridge.invoke(_operation("get_selection"), {})
        assert str(refused.value) == PAUSED_MESSAGE
        assert widget._bridge.usage.stopped is None
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert Stopped.PAUSED.value in widget._plan_box.stopped_text()

    def test_a_fix_limit_stop_is_kept_as_the_reason(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、検査の直しの回数の上限（#339）で残りの編集を断られた計画が、欄に「応え終えた」
        # と残る 設定で回数を増やせば続けられることも、手順の途中かもしれないことも読めない
        widget, host, _ = made
        host.fix_limit = 0
        _send(widget, "仕上がりを直して書き出して")
        _plan(host, "直す", "書き出す")
        _mark(host, 1, "running")
        _finish(host, [_MIXED_CLIP])
        assert fixes_closed(host.document)
        # 断るかの確かめは編集画面のスレッドへ頼むので、AI のスレッドの代わりに別のスレッドで呼ぶ
        done, result = call_in_thread(widget._bridge, _operation("add_track"), {"kind": "video"})
        pump_until(widget._bridge, done)
        error = result.get("error")
        assert isinstance(error, ToolError) and "上限" in str(error)
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        box = widget._plan_box
        assert Stopped.FIX_LIMIT.value in box.stopped_text()
        assert "AI が仕上がりを直す回数" in box.stopped_text()
        assert box.resume_button.isEnabled()

    def test_the_handoff_names_the_same_fix_limit(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、直しの回数の上限で手順を残した計画に、AI が言葉で締めて終えただけで引き継ぎが
        # 作られず、計画は「直しの回数の上限」と言うのに引き継ぎは何も止まっていないことになる
        widget, host, _ = made
        host.fix_limit = 0
        _send(widget, "仕上がりを直して書き出して")
        _plan(host, "直す", "書き出す")
        _mark(host, 1, "running")
        _finish(host, [_MIXED_CLIP])
        widget._handle(AgentEvent(EventKind.TEXT, text="直す回数の上限なのでここまでにします"))
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert Stopped.FIX_LIMIT.value in widget._plan_box.stopped_text()
        handoff = _shelved(host)
        assert handoff is not None and handoff.reason is StopReason.LIMIT
        assert "直す回数の上限" in handoff.detail
        assert "計画の手順が 2 個残っています（get_plan で読めます）" in handoff.remaining

    def test_resume_and_close_wait_while_answering(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、応えている間に「続けて」が後ろへ並ぶか、AI が書く先の計画が消える
        widget, host, sessions = made
        _send(widget, "無音を詰めて")
        _plan(host, "a", "b")
        widget._poll()
        box = widget._plan_box
        assert box.close_button.isEnabled() is False
        widget.resume_plan()
        widget.close_plan()
        assert sessions[0].prompts == ["無音を詰めて"]
        assert plan_for(host.document).view() is not None

    def test_closing_the_plan_when_idle_hides_the_box(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、終えた計画や要らない計画の欄が残り続ける
        widget, host, _ = made
        _send(widget, "無音を詰めて")
        _plan(host, "a", "b")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        box = widget._plan_box
        assert box.close_button.isEnabled()
        box.close_button.click()
        assert box.isHidden() is True
        assert plan_for(host.document).view() is None

    def test_a_finished_plan_offers_no_resume(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、終えた計画に〔続ける〕が出て、押すと済んだ作業を頼み直す
        widget, host, sessions = made
        _send(widget, "無音を詰めて")
        _plan(host, "a")
        _mark(host, 1, "done")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert widget._plan_box.resume_button.isEnabled() is False
        widget.resume_plan()
        assert sessions[0].prompts == ["無音を詰めて"]

    def test_pausing_during_the_last_step_ends_as_finished(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、最後の手順の途中で止めを頼んで AI が終えたとき、会話の欄に
        # 「計画を区切りで止めました」と出て、押せない〔続ける〕を案内する
        widget, host, _ = made
        _send(widget, "無音を詰めて")
        _plan(host, "a", "b")
        _mark(host, 1, "done")
        _mark(host, 2, "running")
        widget._poll()
        box = widget._plan_box
        box.pause_button.click()
        _mark(host, 2, "done")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "計画を区切りで止めました" not in _log(widget)
        assert box.stopped_text() == ""
        assert box.resume_button.isEnabled() is False

    def test_opening_another_project_closes_the_plan(
        self,
        made: tuple[ChatPanel, FakeHost, list[_RecordingSession]],
        video_media: MediaItem,
        transcript: Transcript,
    ) -> None:
        # 壊れると、〔続ける〕で前の作品の計画を次の作品へ当て、別の作品を手順どおりに変える
        widget, host, _ = made
        _send(widget, "無音を詰めて")
        _plan(host, "a", "b")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        host.document.reset(make_loaded(video_media, transcript))
        widget._poll()
        assert widget._plan_box.isHidden() is True
        assert "前の作品の計画を閉じました" in _log(widget)


def _end_by_interrupt(widget: ChatPanel, host: FakeHost) -> None:
    widget.interrupt()
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_by_usage_limit(widget: ChatPanel, host: FakeHost) -> None:
    _over_the_limit(widget)
    with pytest.raises(ToolError, match="上限"):
        widget._bridge.invoke(_operation("get_selection"), {})
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_by_agent_limit(widget: ChatPanel, host: FakeHost) -> None:
    widget._handle(AgentEvent(EventKind.ERROR, text="上限に達しました", detail=LIMIT_DETAIL))
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_by_fix_limit_refusal(widget: ChatPanel, host: FakeHost) -> None:
    host.fix_limit = 0
    _finish(host, [_MIXED_CLIP])
    done, result = call_in_thread(widget._bridge, _operation("add_track"), {"kind": "video"})
    pump_until(widget._bridge, done)
    assert isinstance(result.get("error"), ToolError)
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_after_the_fix_limit(widget: ChatPanel, host: FakeHost) -> None:
    host.fix_limit = 0
    _finish(host, [_MIXED_CLIP])
    widget._handle(AgentEvent(EventKind.TEXT, text="直す回数の上限なのでここまでにします"))
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_by_lost_connection(widget: ChatPanel, host: FakeHost) -> None:
    widget._handle(AgentEvent(EventKind.CLOSED))


def _end_by_error(widget: ChatPanel, host: FakeHost) -> None:
    widget._handle(AgentEvent(EventKind.ERROR, text="ProcessError: 落ちました"))
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_by_denial(widget: ChatPanel, host: FakeHost) -> None:
    widget._bridge._approvals.put(Approval(tool="split_clip", summary="分割", arguments={}))
    widget._check_approval()
    widget._answer(False)
    widget._handle(AgentEvent(EventKind.TEXT, text="許されなかったのでここまでにします"))
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_by_pause(widget: ChatPanel, host: FakeHost) -> None:
    assert plan_for(host.document).request_pause()
    _mark(host, 1, "done")
    widget._handle(AgentEvent(EventKind.TURN_DONE))


def _end_on_its_own(widget: ChatPanel, host: FakeHost) -> None:
    widget._handle(AgentEvent(EventKind.TEXT, text="どちらにしますか"))
    widget._handle(AgentEvent(EventKind.TURN_DONE))


class TestMatchesTheHandoff:
    """計画の止まった訳（#340）と、止まった指示の引き継ぎ（#346）の訳が食い違わないこと"""

    @pytest.mark.parametrize(
        ("end", "stopped", "reason", "hint"),
        [
            (_end_by_interrupt, Stopped.INTERRUPTED, StopReason.INTERRUPTED, "確かめてから"),
            (_end_by_usage_limit, Stopped.LIMIT, StopReason.LIMIT, "「AI の上限」"),
            # Claude Code の上限は設定の「AI の上限」では変わらないので、そこへ案内しない
            (_end_by_agent_limit, Stopped.AGENT_LIMIT, StopReason.LIMIT, "Claude Code の 1 回"),
            (_end_by_fix_limit_refusal, Stopped.FIX_LIMIT, StopReason.LIMIT, "直す回数"),
            (_end_after_the_fix_limit, Stopped.FIX_LIMIT, StopReason.LIMIT, "直す回数"),
            (_end_by_lost_connection, Stopped.DISCONNECTED, StopReason.DISCONNECTED, "つなぎ直し"),
            (_end_by_error, Stopped.ERROR, StopReason.ERROR, "エラーを確かめて"),
            (_end_by_denial, Stopped.DENIED, StopReason.DENIED, "許さなかった物"),
            # 人が区切りで止めた・AI が自分で応え終えたときは、止まった指示として引き継がない
            (_end_by_pause, Stopped.PAUSED, None, "止めた所から"),
            (_end_on_its_own, Stopped.ENDED, None, "残りの手順"),
        ],
    )
    def test_every_way_of_stopping_names_the_same_reason(
        self,
        made: tuple[ChatPanel, FakeHost, list[_RecordingSession]],
        end: Callable[[ChatPanel, FakeHost], None],
        stopped: Stopped,
        reason: StopReason | None,
        hint: str,
    ) -> None:
        # 壊れると、同じ指示が計画の欄では「手順を残して応え終えました」、引き継ぎでは「エラーで
        # 止まりました」のように食い違い、どちらの訳を信じて続ければよいか分からない 欄の添え書きも
        # 訳に合わないと、Claude Code の上限なのに設定の上限を広げに行くなど、違う直し方を案内する
        widget, host, _ = made
        _send(widget, "仕上がりを直して書き出して")
        _plan(host, "直す", "書き出す")
        _mark(host, 1, "running")
        end(widget, host)
        view = plan_for(host.document).view()
        assert view is not None and view.stopped is stopped
        assert stopped.value in widget._plan_box.stopped_text()
        assert hint in widget._plan_box.stopped_text()
        handoff = _shelved(host)
        if reason is None:
            assert handoff is None
        else:
            assert handoff is not None and handoff.reason is reason
            assert "計画の手順が" in " ".join(handoff.remaining)

    def test_every_reason_has_a_hint(self) -> None:
        # 壊れると、訳を足したときに添え書きが無く、欄を出す所で落ちて計画の欄が描けない
        assert set(_STOPPED_HINTS) == set(Stopped)

    def test_a_person_interrupt_after_the_limit_is_an_interrupt_in_both(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、使う量の上限の後に人が〔中断〕を押した指示が、引き継ぎでは「中断」、計画の欄
        # では「使う量の上限」と食い違い、どちらの訳で続ければよいか分からない
        widget, host, _ = made
        _send(widget, "無音を詰めて書き出して")
        _plan(host, "無音を探す", "詰める")
        _mark(host, 1, "running")
        _over_the_limit(widget)
        with pytest.raises(ToolError, match="上限"):
            widget._bridge.invoke(_operation("get_selection"), {})
        widget.interrupt()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert Stopped.INTERRUPTED.value in widget._plan_box.stopped_text()
        handoff = _shelved(host)
        assert handoff is not None and handoff.reason is StopReason.INTERRUPTED


class TestPlanPreference:
    def test_it_is_on_from_the_start(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 既定は入（知らない人ほど、長い作業の途中で何をしているのかが分からず困る）
        widget, _, sessions = made
        assert Preferences().ai_plan_first is True
        _send(widget, "無音を詰めて")
        assert sessions[0].plan is True
        assert PLAN_RULES in sessions[0].system_prompt

    def test_turning_it_off_removes_the_rules_and_the_tools(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、切った人の会話でも計画を書かせ、道具の呼び出しが増える
        widget, _, sessions = made
        widget.apply_preferences(Preferences(ai_plan_first=False))
        _send(widget, "無音を詰めて")
        assert sessions[0].plan is False
        assert PLAN_RULES not in sessions[0].system_prompt

    def test_changing_it_starts_a_new_conversation_when_idle(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 指示の文と道具は会話を作るときにしか渡せない 壊れると、切っても今の会話では計画を書く
        widget, _, sessions = made
        _send(widget, "無音を詰めて")
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        widget.apply_preferences(Preferences(ai_plan_first=False))
        assert sessions[0].closed is True
        _send(widget, "次")
        assert sessions[1].plan is False

    def test_changing_it_during_a_reply_moves_the_waiting_prompts(
        self, made: tuple[ChatPanel, FakeHost, list[_RecordingSession]]
    ) -> None:
        # 壊れると、続けて送った指示が古い会話（計画の道具と決まりあり）で走る
        widget, _, sessions = made
        _send(widget, "1 つ目")
        _send(widget, "2 つ目")
        widget.apply_preferences(Preferences(ai_plan_first=False))
        assert sessions[0].closed is False
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert sessions[0].closed is True
        assert sessions[1].plan is False
        assert sessions[1].prompts == ["2 つ目"]

    def test_the_choice_survives_a_restart(self, tmp_path: Path) -> None:
        # 保存し忘れると、切ったはずの計画が次の起動で戻る
        from sashimono.ui.workspace import PreferenceStore

        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(ai_plan_first=False))
        assert store.load().ai_plan_first is False

    def test_the_settings_dialog_keeps_it(self, qt_application: QApplication) -> None:
        # 出し忘れると、設定を開いて OK を押しただけで入に戻る
        from sashimono.ui.preferences_dialog import PreferencesDialog

        del qt_application
        dialog = PreferencesDialog(Preferences(ai_plan_first=False))
        try:
            assert dialog.preferences().ai_plan_first is False
        finally:
            dialog.deleteLater()
