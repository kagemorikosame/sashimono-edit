"""AI の長い作業の計画・進み具合・区切りで止める・続ける（#340 D2）

計画は作品を変えない（パネルの計画の欄だけ） 人が区切りで止めた後は、ブリッジが計画の
道具のほかを断る 止めた訳と済んだ手順は get_plan で読め、会話を作り直した後も続けられる
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.ledger import AI_ONLY
from sashimono.ai.operations import OPERATIONS, Operation, find_operation
from sashimono.ai.outbound import ASKED, Outbound, SendAnswer, SendChoice, SendRequest
from sashimono.ai.plan import (
    GOAL_LIMIT,
    PAUSED_MESSAGE,
    PLAN_TOOLS,
    RESUME_PROMPT,
    STEP_LIMIT,
    STEPS_RANGE,
    StepState,
    Stopped,
    WorkPlan,
    plan_for,
)
from sashimono.ai.safety import Safety
from sashimono.ai.session import PLAN_RULES, AgentSession, system_prompt
from sashimono.core.model import MediaItem, Transcript
from tests.ai.conftest import FakeHost, make_loaded
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.ai.test_outbound import answer_sends, undeclared

SECRET_PATH = Path("C:/Users/秘密の人/動画/結婚式の撮影.mp4")


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


def _plan(host: FakeHost, steps: int = 3) -> dict[str, Any]:
    result = _operation("set_plan")(
        host, {"goal": "無音を詰めて書き出す", "steps": [f"手順 {n}" for n in range(1, steps + 1)]}
    )
    assert isinstance(result, dict)
    return result


def _mark(host: FakeHost, step: int, state: str, note: str = "") -> dict[str, Any]:
    result = _operation("update_plan_step")(host, {"step": step, "state": state, "note": note})
    assert isinstance(result, dict)
    return result


def _read(host: FakeHost) -> dict[str, Any]:
    result = _operation("get_plan")(host, {})
    assert isinstance(result, dict)
    return result


class TestPlanTools:
    def test_a_plan_is_shown_and_read_back(self, host: FakeHost) -> None:
        # 壊れると、AI が書いた計画が欄に出ず、人は長い作業の中身と順を先に見られない
        made = _plan(host)
        assert made["steps"] == 3 and made["next_step"] == 1
        view = plan_for(host.document).view()
        assert view is not None
        assert view.goal == "無音を詰めて書き出す"
        assert [step.title for step in view.steps] == ["手順 1", "手順 2", "手順 3"]
        read = _read(host)
        assert [step["state"] for step in read["steps"]] == ["pending"] * 3

    def test_progress_is_written_step_by_step(self, host: FakeHost) -> None:
        # 壊れると、欄の進み具合が動かず、どこまで済んだかが分からない
        _plan(host)
        _mark(host, 1, "done")
        result = _mark(host, 2, "running")
        assert (result["done"], result["total"], result["next_step"]) == (1, 3, 3)
        view = plan_for(host.document).view()
        assert view is not None
        assert view.running == 2
        assert [step.state for step in view.steps] == [
            StepState.DONE,
            StepState.RUNNING,
            StepState.PENDING,
        ]

    def test_the_plan_does_not_touch_the_project(self, host: FakeHost) -> None:
        # 壊れると、計画を書くだけで取り消しの段が積まれ、人の Ctrl+Z が計画の書き込みで潰れる
        before = host.document.project
        _plan(host)
        _mark(host, 1, "done", "済んだ")
        assert host.document.project is before
        assert not host.document.can_undo

    def test_writing_tools_do_not_echo_the_plan(self, host: FakeHost) -> None:
        # 壊れると、書いた手順の文（素材の名前・字幕の写し）を返し、指示と答えだけと書いた
        # 宣言の外の物を、断った人にも尋ねず送る
        result = _operation("set_plan")(
            host, {"goal": str(SECRET_PATH), "steps": [f"{SECRET_PATH} を読む"]}
        )
        assert str(SECRET_PATH) not in str(result)
        marked = _mark(host, 1, "failed", f"{SECRET_PATH.name} が開けない")
        assert SECRET_PATH.name not in str(marked)

    def test_a_new_plan_says_how_much_of_the_old_one_was_left(self, host: FakeHost) -> None:
        # 壊れると、前の計画の残りを黙って捨て、やり終えたつもりで忘れる
        _plan(host)
        _mark(host, 1, "done")
        assert _plan(host, 2)["replaced_unfinished"] == 2

    @pytest.mark.parametrize(
        ("arguments", "words"),
        [
            ({"goal": "", "steps": ["a"]}, "goal"),
            ({"goal": "x" * (GOAL_LIMIT + 1), "steps": ["a"]}, "goal"),
            ({"goal": "目標", "steps": []}, "steps"),
            ({"goal": "目標", "steps": ["a"] * (STEPS_RANGE[1] + 1)}, "steps"),
            ({"goal": "目標", "steps": ["x" * (STEP_LIMIT + 1)]}, "steps"),
            ({"goal": "目標", "steps": [""]}, "空"),
            ({"goal": "目標", "steps": "a,b"}, "steps"),
            ({"goal": "目標", "steps": [1]}, "文字列"),
        ],
    )
    def test_bad_plans_are_refused_not_cut(
        self, host: FakeHost, arguments: dict[str, object], words: str
    ) -> None:
        # 切って受けると、手順の違いを書いた所が消え、どこまで済んだかが読めなくなる
        with pytest.raises(ToolError, match=words):
            _operation("set_plan")(host, arguments)
        assert plan_for(host.document).view() is None

    @pytest.mark.parametrize(
        ("step", "state"), [(0, "done"), (4, "done"), (True, "done"), ("1", "done"), (1, "済み")]
    )
    def test_bad_marks_are_refused(self, host: FakeHost, step: object, state: object) -> None:
        # 壊れると、無い手順や読めない進み具合を受け、欄の数が合わなくなる
        _plan(host)
        with pytest.raises(ToolError):
            _operation("update_plan_step")(host, {"step": step, "state": state})

    def test_marking_without_a_plan_says_to_make_one(self, host: FakeHost) -> None:
        # 壊れると、AI は何が足りないのか分からず、同じ呼び出しを繰り返す
        with pytest.raises(ToolError, match="set_plan"):
            _mark(host, 1, "done")

    def test_reading_without_a_plan_says_so(self, host: FakeHost) -> None:
        # 壊れると、計画の無い会話で続きを頼まれた AI が、無い計画を探し回る
        assert _read(host)["plan"] is None

    def test_the_plan_is_read_with_the_instruction_it_was_made_for(self, host: FakeHost) -> None:
        # 壊れると、会話を作り直した後に〔続ける〕を押しても、元の頼みが分からず続けられない
        plan = plan_for(host.document)
        plan.begin_turn("全体の無音を詰めて、字幕を付けて書き出して")
        _plan(host)
        plan.end_turn(interrupted=True)
        plan.begin_turn(RESUME_PROMPT)
        assert _read(host)["instruction"] == "全体の無音を詰めて、字幕を付けて書き出して"
        # 続きの中で計画を作り直しても、元の頼みは「続けて」で上書きされない
        _plan(host, 2)
        assert _read(host)["instruction"] == "全体の無音を詰めて、字幕を付けて書き出して"

    def test_the_tools_are_read_only_and_in_the_ledger(self) -> None:
        # 壊れると（取り消せる編集と書くと）、計画を書くたびに許可の確かめが出て、長い作業ほど
        # 人が席を外せない 台帳に無いと、画面の操作との突き合わせの見張りから外れる
        for name in PLAN_TOOLS:
            operation = _operation(name)
            assert operation.safety is Safety.READ_ONLY
            assert name in AI_ONLY


class TestPause:
    def test_pausing_stops_at_the_next_step_boundary(self, host: FakeHost) -> None:
        # 壊れると、〔区切りで止める〕を押しても AI が次の手順へ進み続ける
        plan = plan_for(host.document)
        _plan(host)
        _mark(host, 1, "running")
        assert plan.request_pause() is True
        # 区切りに来るまでは今の手順を続ける
        assert plan.refusal("split_clip") == ""
        result = _mark(host, 1, "done")
        assert result["paused"] is True and result["message"] == PAUSED_MESSAGE
        assert plan.refusal("split_clip") == PAUSED_MESSAGE
        # 済んだ印は書いてから止める 書かずに止めると、続きで済んだ手順をやり直す
        view = plan.view()
        assert view is not None and view.steps[0].state is StepState.DONE

    def test_a_step_is_not_started_after_the_pause(self, host: FakeHost) -> None:
        # 壊れると、止めた後に次の手順が「取りかかり中」になり、続きで途中の手順と見誤る
        plan = plan_for(host.document)
        _plan(host)
        plan.request_pause()
        result = _mark(host, 2, "running")
        assert result["paused"] is True
        view = plan.view()
        assert view is not None and view.steps[1].state is StepState.PENDING

    def test_a_new_plan_cannot_lift_the_pause(self, host: FakeHost) -> None:
        # 壊れると、AI が計画を作り直すだけで人の止めを外して進める
        plan = plan_for(host.document)
        _plan(host)
        plan.request_pause()
        _mark(host, 1, "done")
        with pytest.raises(ToolError, match="区切りで止め"):
            _plan(host)

    def test_the_bridge_refuses_other_tools_without_asking(self, host: FakeHost) -> None:
        # 壊れると、止めた後も AI が編集を続け、止めた人に許可の確かめの窓まで出る
        bridge = EditorBridge(host)
        _plan(host)
        bridge.plan.request_pause()
        _mark(host, 1, "done")
        clips = [clip for track in host.document.project.timeline.tracks for clip in track.clips]
        done, box = call_in_thread(
            bridge, _operation("split_clip"), {"clip_id": str(clips[0].id), "frame": 30}
        )
        pump_until(bridge, done)
        assert bridge.take_approval() is None
        error = box.get("error")
        assert isinstance(error, ToolError) and str(error) == PAUSED_MESSAGE
        assert not host.document.can_undo

    def test_plan_tools_still_pass_after_the_pause(self, host: FakeHost) -> None:
        # 壊れると、止めた後に済んだ印も書けず、止めた所も読めない
        bridge = EditorBridge(host)
        _plan(host)
        bridge.plan.request_pause()
        _mark(host, 1, "done")
        done, box = call_in_thread(
            bridge, _operation("update_plan_step"), {"step": 2, "state": "done"}
        )
        pump_until(bridge, done)
        assert box.get("error") is None
        done, box = call_in_thread(bridge, _operation("get_plan"), {})
        answer_sends(bridge, done, lambda _request: SendAnswer.ALWAYS)
        assert box.get("error") is None

    def test_a_new_instruction_lifts_the_pause(self, host: FakeHost) -> None:
        # 壊れると、止めた後に送った指示（〔続ける〕も）まで全部断られる
        plan = plan_for(host.document)
        _plan(host)
        plan.request_pause()
        _mark(host, 1, "done")
        plan.end_turn(interrupted=False)
        plan.begin_turn(RESUME_PROMPT)
        assert plan.refusal("split_clip") == ""
        assert _mark(host, 2, "running").get("paused") is None

    def test_pausing_a_finished_plan_does_nothing(self, host: FakeHost) -> None:
        # 壊れると、終わった計画に止めが残り、次の指示の初めの道具が断られる
        plan = plan_for(host.document)
        _plan(host, 1)
        _mark(host, 1, "done")
        assert plan.request_pause() is False

    @pytest.mark.parametrize(
        ("interrupted", "pause", "stopped"),
        [
            (True, False, Stopped.INTERRUPTED),
            (False, True, Stopped.PAUSED),
            (False, False, Stopped.ENDED),
        ],
    )
    def test_the_reason_for_stopping_is_kept(
        self, host: FakeHost, interrupted: bool, pause: bool, stopped: Stopped
    ) -> None:
        # 壊れると、中断で途中まで済んだ手順を、続きの AI が済んでいない物として最初からやり直す
        plan = plan_for(host.document)
        _plan(host)
        _mark(host, 1, "running")
        if pause:
            plan.request_pause()
        plan.end_turn(interrupted=interrupted)
        assert _read(host)["stopped"] == stopped.value
        view = plan.view()
        assert view is not None and view.stopped is stopped

    def test_a_finished_plan_has_no_reason_to_resume(self, host: FakeHost) -> None:
        # 壊れると、終えた計画にも〔続ける〕が出て、押すと済んだ作業を頼み直す
        plan = plan_for(host.document)
        _plan(host, 1)
        _mark(host, 1, "done")
        plan.end_turn(interrupted=True)
        view = plan.view()
        assert view is not None and view.finished and view.stopped is None


class TestBelongsToTheProject:
    def test_opening_another_project_closes_the_plan(
        self, host: FakeHost, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、前の作品の計画を次の作品で続け、別の作品のクリップを手順どおりに変える
        _plan(host)
        host.document.reset(make_loaded(video_media, transcript))
        assert plan_for(host.document).view() is None
        assert _read(host)["plan"] is None

    def test_the_bridge_sees_the_same_plan_after_reopening(
        self, host: FakeHost, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、作品を開き直した後の計画の止めが、ブリッジに届かない（別の物を見ている）
        bridge = EditorBridge(host)
        host.document.reset(make_loaded(video_media, transcript))
        plan = plan_for(host.document)
        assert plan is bridge.plan

    def test_follow_reports_only_a_dropped_plan(self) -> None:
        # 壊れると、計画の無い作品を開くたびに「計画を閉じました」と出て紛らわしい
        plan = WorkPlan(0)
        assert plan.follow(1) is False
        plan.make("目標", ["a"])
        assert plan.follow(1) is False
        assert plan.follow(2) is True


@pytest.fixture
def secret_host(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    return FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))


def _secret_plan(host: FakeHost) -> None:
    """AI が名前と字幕の文を手順に書き写した計画（許されていた間に読んだ物）"""
    _operation("set_plan")(
        host,
        {"goal": "「編集ソフトを」の字幕を直す", "steps": [f"{SECRET_PATH} の字幕を読む"]},
    )


class TestWhatIsSent:
    def test_reading_the_plan_declares_names_and_text(self, secret_host: FakeHost) -> None:
        # 壊れると（指示と答えだけと書くと）、手順に写した素材の場所と字幕の文を、
        # 断った人にも宣言の外で送る 見張り（undeclared）は弱めた宣言で落ちる
        _secret_plan(secret_host)
        operation = _operation("get_plan")
        result = operation(secret_host, {})
        sentences = {"編集ソフトを"}
        places = {str(SECRET_PATH)}
        assert not undeclared(operation, result, sentences=sentences, places=places)
        weakened = replace(operation, sends=frozenset({Outbound.CONVERSATION}), hidden=frozenset())
        assert len(undeclared(weakened, result, sentences=sentences, places=places)) == 2

    def test_writing_tools_send_only_the_conversation(self) -> None:
        # 壊れると、計画を書くだけで名前や字幕を送ってよいかを尋ねられ、長い作業が窓で止まる
        for name in ("set_plan", "update_plan_step"):
            assert _operation(name).sends == frozenset({Outbound.CONVERSATION})

    def test_refused_names_and_text_are_hidden_without_asking_about_text(
        self, secret_host: FakeHost
    ) -> None:
        # 壊れると、字幕と文字を送らない人は止めた所から続けられない（断ると道具が動かない）か、
        # 断ったはずの名前と字幕の文が、続きを読む道具でそのまま送られる
        _secret_plan(secret_host)
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.DENY))
        asked: list[SendRequest] = []

        def reply(request: SendRequest) -> SendAnswer:
            asked.append(request)
            return SendAnswer.DENY

        done, box = call_in_thread(bridge, _operation("get_plan"), {})
        answer_sends(bridge, done, reply)
        assert asked == []
        result = box.get("result")
        text = str(result)
        assert str(SECRET_PATH) not in text and SECRET_PATH.name not in text
        assert "編集ソフトを" not in text
        assert isinstance(result, dict) and len(result["steps"]) == 1

    def test_writing_a_plan_never_asks(self, host: FakeHost) -> None:
        # 壊れると、計画を書く前に送ってよいかの窓が出て、長い作業の頭で止まる
        bridge = EditorBridge(host)
        done, box = call_in_thread(
            bridge, _operation("set_plan"), {"goal": "目標", "steps": ["a", "b"]}
        )
        asked = answer_sends(bridge, done, lambda _request: SendAnswer.DENY)
        assert asked == [] and box.get("error") is None


class TestSessionSetting:
    @pytest.mark.parametrize("plan", [True, False])
    def test_the_rules_follow_the_setting(self, plan: bool) -> None:
        # 壊れると、切った人の会話でも計画を書かせて道具の呼び出しが増える（入れた人には書かせない）
        assert (PLAN_RULES in system_prompt(plan=plan)) is plan

    @pytest.mark.parametrize("plan", [True, False])
    def test_the_session_passes_the_tools_only_when_on(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch, plan: bool
    ) -> None:
        # 壊れると、設定で切っても計画の道具が見え、AI は計画を書きに来る
        from sashimono.ai import session as session_module

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(
            session_module, "build_server", lambda _bridge, **options: seen.append(options)
        )
        AgentSession(EditorBridge(host), plan=plan).server()
        assert seen == [{"ask": True, "plan": plan}]

    @pytest.mark.parametrize("plan", [True, False])
    def test_the_server_leaves_the_tools_out_when_off(self, host: FakeHost, plan: bool) -> None:
        # 壊れると、指示の文から決まりを外しても、道具が見えていれば AI は計画を書く
        pytest.importorskip("claude_agent_sdk", reason="AI 連携の実行環境が未導入")
        from sashimono.ai.server import build_tools

        names = {tool.name for tool in build_tools(EditorBridge(host), plan=plan)}
        assert (names >= PLAN_TOOLS) is plan
        assert not plan or len(names) == len(OPERATIONS)
        assert plan or not names & PLAN_TOOLS
