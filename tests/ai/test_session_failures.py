"""会話が失敗したときの知らせ（#412）

Claude Code が作った言葉（``model`` が ``<synthetic>``）を AI の言葉として流さない・ログインの失敗を
見分ける・1 通が大きすぎても分かる文で止まり、次の指示で繋ぎ直せることを見る
CI には SDK が入らないので、SDK の形は偽物で作る 値の形は通しの確かめの記録に合わせた
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from sashimono.ai import session as session_module
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.session import (
    LOGIN_DETAIL,
    AgentEvent,
    AgentSession,
    EventKind,
    result_error,
    with_login_hint,
)
from tests.waiting import wait_until

SYNTHETIC = "<synthetic>"
EXPIRED = "Failed to authenticate: OAuth session expired and could not be refreshed"


@dataclass
class _Text:
    text: str


@dataclass
class _Assistant:
    content: list[object]
    model: str = "claude-opus-5-5"
    error: str | None = None
    usage: dict[str, int] | None = None
    message_id: str | None = None


@dataclass
class _Result:
    is_error: bool = False
    subtype: str = "success"
    result: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class _Unused:
    """この試験の知らせには出てこない SDK の形"""

    content: list[object] = field(default_factory=list)


class _Usage:
    def record_message(self, message_id: object, usage: object) -> None:
        del message_id, usage

    def settle_output(self, usage: object) -> None:
        del usage


class _Bridge:
    usage = _Usage()

    def resume(self) -> None:
        return

    def cancel(self) -> None:
        return


class _Client:
    """ClaudeSDKClient の代わり 1 回の指示ごとに ``replies`` の頭を 1 つ使う"""

    replies: ClassVar[list[list[object] | BaseException]] = []
    made: ClassVar[int] = 0
    #: 立つまで終了処理を終えない 無ければすぐ終える
    leaving: ClassVar[threading.Event | None] = None

    def __init__(self, options: object) -> None:
        del options
        _Client.made += 1

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        leaving = _Client.leaving
        if leaving is not None:
            await asyncio.to_thread(leaving.wait)

    async def query(self, prompt: str) -> None:
        del prompt

    async def receive_response(self) -> AsyncIterator[object]:
        reply = _Client.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        for message in reply:
            yield message

    async def interrupt(self) -> None:
        return


@pytest.fixture(autouse=True)
def sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    fake: Any = types.ModuleType("claude_agent_sdk")
    fake.AssistantMessage = _Assistant
    fake.ResultMessage = _Result
    fake.TextBlock = _Text
    fake.ThinkingBlock = _Unused
    fake.ToolResultBlock = _Unused
    fake.ToolUseBlock = _Unused
    fake.UserMessage = _Unused
    fake.ClaudeAgentOptions = lambda **_kwargs: object()
    fake.ClaudeSDKClient = _Client
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    monkeypatch.setattr(session_module, "build_server", lambda _bridge, **_options: None)
    monkeypatch.setattr(session_module, "hide_cli_console", lambda: None)
    monkeypatch.setattr(AgentSession, "option_values", lambda _self: {})
    _Client.replies = []
    _Client.made = 0
    _Client.leaving = None
    _sessions.clear()
    yield
    # 落ちた試験の会話も畳む 残すと、会話のスレッドが指示を待ったままで、試験の終わりに止まる
    if _Client.leaving is not None:
        _Client.leaving.set()
    for made in _sessions:
        made.close()


_sessions: list[AgentSession] = []


def _session() -> AgentSession:
    made = AgentSession(cast(EditorBridge, _Bridge()))
    _sessions.append(made)
    return made


def _events(*messages: object) -> list[AgentEvent]:
    session = _session()
    for message in messages:
        session._translate(message)
    return session.poll()


def _until(session: AgentSession, kind: EventKind) -> list[AgentEvent]:
    """``kind`` の知らせが来るまで集める"""
    seen: list[AgentEvent] = []
    wait_until(
        lambda: any(event.kind is kind for event in seen),
        f"{kind} の知らせ",
        step=lambda: seen.extend(session.poll()),
        pause=0.01,
    )
    return seen


def _expired_turn() -> list[object]:
    """ログインが切れたときに流れてくる形（Claude Code が作った言葉と、失敗の結果）"""
    return [
        _Assistant([_Text(EXPIRED)], model=SYNTHETIC, error="authentication_failed"),
        _Result(is_error=True, result=EXPIRED),
    ]


class TestWordsClaudeCodeMade:
    def test_a_made_up_failure_is_not_the_ai_speaking(self) -> None:
        # 壊れると、同じ失敗の文が「Claude」の言葉として会話の欄に出て、引き継ぎにも
        # 「AI の最後の言葉」として載り、1 つの失敗が 3 回並ぶ
        events = _events(*_expired_turn())
        assert [event.kind for event in events] == [EventKind.ERROR]

    def test_the_real_ai_is_still_heard(self) -> None:
        # 壊れると、本物の返事まで捨てて、AI が何も言わないように見える
        events = _events(_Assistant([_Text("切りました")]), _Result())
        assert [(event.kind, event.text) for event in events] == [(EventKind.TEXT, "切りました")]

    def test_a_failure_without_a_failed_result_is_still_told(self) -> None:
        # 結果が失敗の印を持たなくても、作られた言葉が失敗を言っていれば失敗として出す
        # 捨てると、応えられなかった指示が黙って「完了」になる
        events = _events(
            _Assistant([_Text("Rate limited")], model=SYNTHETIC, error="rate_limit"),
            _Result(is_error=False),
        )
        assert [event.kind for event in events] == [EventKind.ERROR]

    def test_the_text_of_the_failure_comes_from_the_made_up_words(self) -> None:
        # 結果に文が無いと「失敗しました」だけになり、何が起きたか分からない
        events = _events(
            _Assistant([_Text("API Error: 400 Bad things")], model=SYNTHETIC, error="unknown"),
            _Result(is_error=True, result=None),
        )
        assert "Bad things" in events[0].text

    def test_a_notice_that_is_not_a_failure_stays_quiet(self) -> None:
        # 応えなかった印の短い文などを AI の言葉として並べない
        assert _events(_Assistant([_Text("No response requested.")], model=SYNTHETIC)) == []

    def test_a_failure_is_told_even_when_no_result_comes(self) -> None:
        # 結果が来ないまま終わると、失敗の知らせが消えて「完了」と書かれる
        session = _session()
        _Client.replies = [
            [_Assistant([_Text("Server down")], model=SYNTHETIC, error="server_error")]
        ]
        session.send("指示")
        events = _until(session, EventKind.TURN_DONE)
        session.close()
        assert EventKind.ERROR in [event.kind for event in events]

    def test_an_outdated_claude_code_is_still_seen(self) -> None:
        # 画面は失敗の文から「部品が古い」を見分ける 訳の文で包んで元の文が消えると、
        # 〔AI の部品を更新〕が出ない
        text = "API Error: 400 Claude Code 2.1.259 does not support this model"
        events = _events(
            _Assistant([_Text(text)], model=SYNTHETIC, error="invalid_request"),
            _Result(is_error=True, result=text),
        )
        assert session_module.outdated_claude_code(events[0].text) is not None


class TestFailuresInJapanese:
    def test_the_instructions_cover_the_words_between_tools(self) -> None:
        # 「返事は」だけだと、道具の合間の説明や計画の手順は別の物と読まれ、英語で書かれる
        for word in ("道具の合間", "計画の手順", "英語"):
            assert word in session_module.system_prompt()

    @pytest.mark.parametrize(
        ("code", "word"),
        [
            ("rate_limit", "上限"),
            ("billing_error", "契約"),
            ("server_error", "Claude の側"),
            ("invalid_request", "受け付け"),
        ],
    )
    def test_known_failures_are_explained(self, code: str, word: str) -> None:
        # 壊れると、英語の文だけが出て、待てばよいのか直すのかが分からない
        event = result_error("success", "Some English text", code=code)
        assert word in event.text
        assert "Some English text" in event.text

    def test_an_unknown_exception_says_what_failed_in_japanese(self) -> None:
        # 壊れると、例外の名前と英語の文だけが会話の欄に出る
        text = session_module._explain(RuntimeError("boom"))
        assert "Claude Code" in text
        assert "boom" in text


class TestLoginFailure:
    def test_the_failure_is_marked(self) -> None:
        # 壊れると、画面が見分けられず〔ログイン…〕が出ないまま「ログイン…で」と言う
        event = _events(*_expired_turn())[0]
        assert event.detail == LOGIN_DETAIL

    def test_the_code_alone_is_enough(self) -> None:
        # 文面が変わっても、Claude Code が付けた印で見分ける
        event = result_error("success", "Something new", code="authentication_failed")
        assert event.detail == LOGIN_DETAIL

    def test_the_english_is_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 壊れると、英語の文が残り、案内の前に読めない 1 行が出る
        monkeypatch.setattr(session_module, "uses_saved_login", lambda: True)
        assert "Failed" not in _events(*_expired_turn())[0].text

    def test_an_expired_login_says_to_type_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 切れた人の窓では Claude Code が案内を出さないかもしれない 打つ物まで言う
        # 「案内に従って」だけだと、何も出ない窓の前で止まる
        monkeypatch.setattr(session_module, "uses_saved_login", lambda: False)
        text = with_login_hint(EXPIRED)
        assert "切れて" in text
        assert "/login" in text

    def test_someone_who_never_logged_in_gets_the_first_guide(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 初めての人に「切れています」と言うと、何かを壊したように読める
        monkeypatch.setattr(session_module, "uses_saved_login", lambda: False)
        text = with_login_hint("Not logged in · Please run /login")
        assert "切れて" not in text
        assert "ログイン…" in text

    def test_a_wrong_api_key_is_not_called_an_expired_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、環境変数の鍵を直す人に「切れています」と言い、/login へ向かわせる
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        (tmp_path / ".credentials.json").write_text("{}", encoding="utf-8")
        text = with_login_hint("Invalid API key · Please run /login")
        assert "切れて" not in text
        assert "ANTHROPIC_API_KEY" in text

    def test_a_saved_login_with_other_words_is_still_expired(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 文面は版で変わる 壊れると、保存したログインの人に /login が案内されない
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        (tmp_path / ".credentials.json").write_text("{}", encoding="utf-8")
        assert "/login" in with_login_hint("API Error: 401 Unauthorized")

    def test_a_number_that_contains_401_is_not_a_login_failure(self) -> None:
        # 壊れると、番号に 401 を含むだけの失敗でログインを促し、原因を取り違える
        assert with_login_hint("frame 14012 failed") == "frame 14012 failed"
        assert "ログイン" in with_login_hint("API Error: 401 Unauthorized")

    def test_a_failed_start_is_marked_too(self) -> None:
        # ログインがまだの機械では、指示の前に起動で落ちることがある 案内の文に置き換えた後も
        # 印が付かないと、〔続きを頼む〕が出て先にログインへ向かわせられない
        session = _session()
        _Client.replies = [RuntimeError("Not logged in · Please run /login")]
        session.send("指示")
        events = _until(session, EventKind.CLOSED)
        details = [event.detail for event in events if event.kind is EventKind.ERROR]
        assert details == [LOGIN_DETAIL]

    def test_the_conversation_is_closed_so_the_new_login_is_read(self) -> None:
        # 動き続ける Claude Code が、別の窓で入れ直したログインを読むかは分からない
        # 畳んで次の指示で起動し直せば、必ず新しいログインで始まる
        session = _session()
        _Client.replies = [_expired_turn(), [_Result()]]
        session.send("1 つ目")
        _until(session, EventKind.CLOSED)
        assert session.running is False
        session.send("2 つ目")
        _until(session, EventKind.TURN_DONE)
        session.close()
        assert _Client.made == 2


# SDK と同じ名前にする 見分けは名前で行う（SDK を読まずに文を作る所なので）
_TooLargeError: type[Exception] = type("CLIJSONDecodeError", (Exception,), {})


class TestMessageTooLarge:
    def _failure(self) -> Exception:
        return _TooLargeError(
            "Failed to decode JSON: JSON message exceeded maximum buffer size of 1048576 bytes..."
        )

    def test_it_stops_with_words_a_person_can_read(self) -> None:
        # 壊れると、例外の名前と英語の文だけが出て、何が起きたか分からない
        text = session_module._explain(self._failure())
        assert "CLIJSONDecodeError" not in text
        assert "大きすぎ" in text
        assert "続きを頼む" in text

    def test_the_next_prompt_starts_a_new_conversation(self) -> None:
        # あふれると SDK の読み取りが止まる（実物で確かめた） 同じ会話へ次の指示を送ると、
        # 返事が 1 つも来ないまま「完了」と出る
        session = _session()
        _Client.replies = [self._failure(), [_Assistant([_Text("続けます")]), _Result()]]
        session.send("1 つ目")
        events = _until(session, EventKind.CLOSED)
        assert [event.kind for event in events].count(EventKind.ERROR) == 1
        assert session.running is False
        session.send("2 つ目")
        later = _until(session, EventKind.TURN_DONE)
        session.close()
        assert _Client.made == 2
        assert [event.text for event in later if event.kind is EventKind.TEXT] == ["続けます"]

    def _held_shutdown(self) -> threading.Event:
        """Claude Code の終了処理を止めておく 戻り値を立てると終わる"""
        release = threading.Event()
        _Client.leaving = release
        return release

    def test_a_prompt_sent_while_closing_runs_in_the_next_conversation(self) -> None:
        # 壊れると、失敗の後すぐに〔続きを頼む〕を送った指示が、終わる会話のキューに入って捨てられ、
        # 画面も「閉じた」と待ちの指示を消す
        release = self._held_shutdown()
        session = _session()
        _Client.replies = [self._failure(), [_Assistant([_Text("続けます")]), _Result()]]
        session.send("1 つ目")
        seen = _until(session, EventKind.TURN_DONE)
        session.acknowledge_turn()
        session.send("続きを頼む")
        release.set()
        seen += _until(session, EventKind.TEXT)
        assert _Client.made == 2
        assert EventKind.CLOSED not in [event.kind for event in seen]

    def test_a_prompt_sent_while_closing_after_a_login_failure_is_not_lost(self) -> None:
        # ログインの失敗の後も同じ畳み方をする 壊れると、ログインし直して送った指示が消える
        release = self._held_shutdown()
        session = _session()
        _Client.replies = [_expired_turn(), [_Assistant([_Text("直りました")]), _Result()]]
        session.send("1 つ目")
        _until(session, EventKind.TURN_DONE)
        session.acknowledge_turn()
        session.send("もう一度")
        release.set()
        later = _until(session, EventKind.TEXT)
        assert [event.text for event in later if event.kind is EventKind.TEXT] == ["直りました"]

    def test_a_prompt_queued_during_the_failed_turn_runs_next(self) -> None:
        # 応えている間に続けて送った指示も同じ 壊れると、失敗の巻き添えで黙って消える
        release = self._held_shutdown()
        session = _session()
        _Client.replies = [self._failure(), [_Assistant([_Text("続けます")]), _Result()]]
        session.send("1 つ目")
        session.send("2 つ目")
        _until(session, EventKind.TURN_DONE)
        session.acknowledge_turn()
        release.set()
        later = _until(session, EventKind.TEXT)
        assert [event.text for event in later if event.kind is EventKind.TEXT] == ["続けます"]

    def test_a_prompt_sent_before_the_screen_saw_the_close_is_not_undone(self) -> None:
        # 畳み終えた知らせを画面が拾う前に送ると、古い知らせが新しい指示を「閉じた」と消す
        session = _session()
        _Client.replies = [self._failure(), [_Assistant([_Text("続けます")]), _Result()]]
        session.send("1 つ目")
        wait_until(lambda: _Client.made == 1 and not session.running, "畳み終わり")
        session.send("2 つ目")
        seen = _until(session, EventKind.TEXT)
        assert EventKind.CLOSED not in [event.kind for event in seen]

    def test_the_real_sdk_error_is_recognized(self) -> None:
        """SDK が名前か文面を変えると、見分けられずに英語のまま出る"""
        del sys.modules["claude_agent_sdk"]
        sdk = pytest.importorskip("claude_agent_sdk")
        failure = sdk.CLIJSONDecodeError(
            "JSON message exceeded maximum buffer size of 1048576 bytes", ValueError("over")
        )
        assert "大きすぎ" in session_module._explain(failure)
