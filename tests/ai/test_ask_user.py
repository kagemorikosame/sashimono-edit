"""AI からの質問（ask_user #296）

引数の確かめ・ブリッジでの待ち合わせ・道具と指示の文の出し入れを見る
画面の質問の箱は tests/ui/test_chat_panel.py で見る
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from sashimono.ai import bridge as bridge_module
from sashimono.ai.bridge import EditorBridge, Question
from sashimono.ai.host import ToolError
from sashimono.ai.operations import (
    ASK_USER,
    CHOICES_RANGE,
    DESCRIPTION_LIMIT,
    LABEL_LIMIT,
    OPERATIONS,
    Choice,
    Operation,
    QuestionSpec,
    find_operation,
    question_spec,
)
from sashimono.ai.session import ASK_RULES, AgentSession, system_prompt
from tests.ai.conftest import FakeHost
from tests.ai.test_bridge import call_in_thread, pump_until


def _ask() -> Operation:
    found = find_operation(ASK_USER)
    assert found is not None
    return found


def _arguments(**changes: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "question": "どのクリップを切りますか",
        "choices": [
            {"label": "選択中", "description": "いま選んでいるクリップ"},
            {"label": "全部", "description": "タイムラインのすべてのクリップ"},
        ],
    }
    arguments.update(changes)
    return arguments


def _take(bridge: EditorBridge, done: threading.Event) -> Question:
    """UI スレッドの代わりに質問を取り出す 出るまで待つ"""
    taken: list[Question] = []

    def grab() -> None:
        question = bridge.take_question()
        if question is not None:
            taken.append(question)

    for _ in range(1000):
        grab()
        if taken or done.is_set():
            break
        threading.Event().wait(0.005)
    assert taken, "質問が画面へ届かない"
    return taken[0]


class TestArguments:
    def test_a_well_formed_question_is_read(self) -> None:
        # 壊れると、正しく組んだ質問まで断られ、AI はいつまでも聞けない
        spec = question_spec(_arguments(multiple=True, free_text=False))
        assert spec == QuestionSpec(
            text="どのクリップを切りますか",
            choices=(
                Choice("選択中", "いま選んでいるクリップ"),
                Choice("全部", "タイムラインのすべてのクリップ"),
            ),
            multiple=True,
            free_text=False,
        )

    def test_free_text_is_allowed_unless_said_otherwise(self) -> None:
        # 壊れると、選択肢に無い答えを書きたい人が書けない（AI が書き忘れたときも）
        spec = question_spec(_arguments())
        assert spec.free_text is True
        assert spec.multiple is False

    @pytest.mark.parametrize("count", [1, CHOICES_RANGE[1] + 1])
    def test_the_number_of_choices_is_bounded(self, count: int) -> None:
        # 1 つは質問ではなく、5 つ以上はチャット欄に収まらない 壊れると箱が画面からあふれる
        choices = [{"label": f"案{n}", "description": ""} for n in range(count)]
        with pytest.raises(ToolError, match="個にしてください"):
            question_spec(_arguments(choices=choices))

    def test_a_long_label_is_refused_not_cut(self) -> None:
        # 切ると選択肢の違いが消える 壊れると、ボタンの名前がパネルの幅からはみ出す
        long = "あ" * (LABEL_LIMIT + 1)
        with pytest.raises(ToolError, match="description に書きます"):
            question_spec(_arguments(choices=[{"label": long}, {"label": "短い"}]))

    def test_a_long_description_is_refused(self) -> None:
        # 壊れると、説明が何行にも折り返して、狭い画面で質問の箱が入力欄を押しつぶす
        long = "い" * (DESCRIPTION_LIMIT + 1)
        with pytest.raises(ToolError, match="description は"):
            question_spec(_arguments(choices=[{"label": "a", "description": long}, {"label": "b"}]))

    def test_the_same_label_twice_is_refused(self) -> None:
        # 同じ名前が並ぶと、答えからどちらを選んだのか AI が見分けられない
        with pytest.raises(ToolError, match="重なって"):
            question_spec(_arguments(choices=[{"label": "a"}, {"label": "a"}]))

    @pytest.mark.parametrize("key", ["multiple", "free_text"])
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_a_flag_that_is_not_a_boolean_is_refused(self, key: str, value: object) -> None:
        # bool("false") は真 壊れると、AI が文字列で "false" と渡したとき逆の設定で質問が出る
        with pytest.raises(ToolError, match="true か false"):
            question_spec(_arguments(**{key: value}))

    @pytest.mark.parametrize(
        "changes",
        [
            {"question": 3},
            {"choices": [{"label": 1, "description": ""}, {"label": "b"}]},
            {"choices": [{"label": "a", "description": ["説明"]}, {"label": "b"}]},
        ],
    )
    def test_text_that_is_not_a_string_is_refused(self, changes: dict[str, object]) -> None:
        # 壊れると、数やリストを文字にした中身の分からない質問や選択肢が並ぶ
        with pytest.raises(ToolError, match="文字列で渡してください"):
            question_spec(_arguments(**changes))

    def test_an_empty_question_is_refused(self) -> None:
        # 壊れると、何を聞かれているのか分からない箱が出る
        with pytest.raises(ToolError, match="空"):
            question_spec(_arguments(question="  "))

    def test_asking_is_not_a_change(self) -> None:
        # 変更系にすると、質問の前に「この操作を許可しますか？」が出て、答える前に許可を求められる
        assert _ask().writes is False
        assert _ask().asks is True
        assert [operation.name for operation in OPERATIONS if operation.asks] == [ASK_USER]


class TestWaitingForTheAnswer:
    def test_the_answer_comes_back_as_the_tool_result(self, host: FakeHost) -> None:
        # 壊れると、選んだ答えが AI に届かず、聞いた意味が無い
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, _ask(), _arguments())
        question = _take(bridge, done)
        assert question.spec.text == "どのクリップを切りますか"
        assert bridge.take_approval() is None  # 許可は挟まない
        assert question.reply(("全部",), " 先頭だけ残して ")
        pump_until(bridge, done)
        assert box["result"] == {"selected": ["全部"], "free_text": "先頭だけ残して"}

    def test_the_selection_follows_the_order_of_the_choices(self, host: FakeHost) -> None:
        # 押した順で返すと、同じ選び方でも答えが揺れる 選択肢の並びで返す
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, _ask(), _arguments(multiple=True))
        question = _take(bridge, done)
        question.reply(("全部", "選択中", "無い名前"))
        pump_until(bridge, done)
        assert box["result"] == {"selected": ["選択中", "全部"], "free_text": ""}

    def test_auto_approve_still_asks(self, host: FakeHost) -> None:
        # 質問は操作の許可ではない 壊れると「変更を自動で承認」の人には聞かずに進む
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, _ask(), _arguments())
        question = _take(bridge, done)
        question.reply(("選択中",))
        pump_until(bridge, done)
        assert box["result"] == {"selected": ["選択中"], "free_text": ""}

    def test_bad_arguments_fail_without_asking(self, host: FakeHost) -> None:
        # 壊れると、選択肢の無い箱が出て答えようがない AI には直し方を返す
        bridge = EditorBridge(host)
        with pytest.raises(ToolError, match="個にしてください"):
            bridge.invoke(_ask(), _arguments(choices=[]))
        assert bridge.take_question() is None


class TestFoldingTheQuestion:
    @pytest.mark.parametrize("taken", [True, False])
    def test_cancelling_folds_a_waiting_question(self, host: FakeHost, taken: bool) -> None:
        # 画面に出した後でも、まだ出す前でも 壊れると中断しても AI のスレッドが 10 分止まる
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, _ask(), _arguments())
        question = _take(bridge, done) if taken else None
        if not taken:
            # 画面がまだ取り出していないうちに中断する 積まれるのを待ってから
            for _ in range(1000):
                if not bridge._questions.empty():
                    break
                threading.Event().wait(0.005)
        bridge.cancel()
        assert done.wait(5.0), "中断しても質問の待ちがほどけない"
        assert isinstance(box["error"], ToolError)
        assert "中断" in str(box["error"])
        if question is not None:
            # 畳んだ後に押された答えは届かない 届いたことにすると画面と AI の話が食い違う
            assert question.reply(("全部",)) is False

    def test_no_answer_in_time_is_reported(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、席を外した人の返事を AI のスレッドがいつまでも待つか、空の答えを
        # 「どれでもよい」と読んで推測で進む
        monkeypatch.setattr(bridge_module, "QUESTION_TIMEOUT", 0.05)
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, _ask(), _arguments())
        assert done.wait(5.0)
        assert isinstance(box["error"], ToolError)
        assert "返事がありませんでした" in str(box["error"])
        assert "推測で進めず" in str(box["error"])

    def test_the_timeout_follows_the_approval(self) -> None:
        # 承認と同じ考え方で決めた 短くすると、少し席を外しただけで質問が消える
        assert bridge_module.QUESTION_TIMEOUT == bridge_module.APPROVAL_TIMEOUT

    def test_after_cancel_a_new_question_fails_fast(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 中断の後に積むと、誰も畳まずに待ち続ける 壊れると中断が効かない
        # 待ちを短くしておく 壊れたときに 10 分待たずに「返事が無い」で落ちるように
        monkeypatch.setattr(bridge_module, "QUESTION_TIMEOUT", 0.05)
        bridge = EditorBridge(host)
        bridge.cancel()
        with pytest.raises(ToolError, match="中断"):
            bridge.invoke(_ask(), _arguments())


class TestTurningItOff:
    def test_the_prompt_carries_the_rules_only_when_on(self) -> None:
        # 切った人に基準だけ残すと、無い道具を呼ぼうとして失敗する 入れた人に基準が
        # 無いと、あいまいな指示でも聞かずに進む
        on = system_prompt(ask=True)
        off = system_prompt(ask=False)
        assert ASK_RULES in on
        assert ASK_USER in on
        assert ASK_RULES not in off
        assert ASK_USER not in off
        assert "{ask_rules}" not in on + off
        assert system_prompt() == on

    def test_the_rules_keep_questions_few(self) -> None:
        # 聞く基準の要（あいまいで結果が大きく変わるときだけ・1 つの話題・確かめられることは
        # 聞かない） 抜けると毎回のように聞かれて手間が増える
        assert "結果が大きく変わるときだけ" in ASK_RULES
        assert "1 つの話題だけ" in ASK_RULES
        assert "聞かずに進めます" in ASK_RULES

    @pytest.mark.parametrize("ask", [True, False])
    def test_the_session_passes_the_tool_only_when_on(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch, ask: bool
    ) -> None:
        # 壊れると、設定で切っても質問の道具が渡り、聞かれ続ける
        from sashimono.ai import session as session_module

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(
            session_module, "build_server", lambda _bridge, **options: seen.append(options)
        )
        AgentSession(EditorBridge(host), ask=ask).server()
        assert seen == [{"ask": ask, "plan": True}]

    @pytest.mark.parametrize("ask", [True, False])
    def test_the_server_leaves_the_tool_out_when_off(self, host: FakeHost, ask: bool) -> None:
        # 壊れると、指示の文から基準を外しても、道具が見えていれば AI は聞きに来る
        pytest.importorskip("claude_agent_sdk", reason="AI 連携の実行環境が未導入")
        from sashimono.ai.server import build_tools

        names = [tool.name for tool in build_tools(EditorBridge(host), ask=ask)]
        assert (ASK_USER in names) is ask
        assert len(names) == len(OPERATIONS) - (0 if ask else 1)
