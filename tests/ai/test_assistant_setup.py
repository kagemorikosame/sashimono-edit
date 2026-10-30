"""アシスタントの実行環境 同梱の Claude Code・ログインの手掛かり・会話へ渡す設定

SDK は import しない 配る先の機械と同じく、SDK の中身は偽物の置き場で作る
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar, cast

import pytest

from sashimono.ai import environment
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.environment import AI_PACK
from sashimono.ai.models import effort_for
from sashimono.ai.operations import ImageResult
from sashimono.ai.server import _content
from sashimono.ai.session import AUTO_MEMORY_OFF, AgentSession, EventKind
from sashimono.ai.sheet import MAX_SHEET_EDGE
from sashimono.runtime import install_command
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.test_runtime_after_install import write_distribution


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """SDK も claude も入っていない機械 戻り値は、導入先に見立てたフォルダ"""
    site = tmp_path / "site"
    site.mkdir()
    # 標準ライブラリは残す メタデータの読み取りが初めて読む部品（email など）を
    # 引けなくなる 開発機に入っている SDK が見える site-packages だけを外す
    kept = [p for p in sys.path if "site-packages" not in p.replace("\\", "/")]
    monkeypatch.setattr(sys, "path", [str(site), *kept])
    for name in list(sys.modules):
        if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
            monkeypatch.delitem(sys.modules, name)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    for variable in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
    return site


def _install_sdk(site: Path, *, bundled: bool = True) -> Path:
    write_distribution(site, "claude-agent-sdk", "claude_agent_sdk", "0.2.174")
    cli = site / "claude_agent_sdk" / "_bundled" / "claude.exe"
    if bundled:
        cli.parent.mkdir(parents=True)
        cli.write_bytes(b"")
    return cli


class TestBundledClaudeCode:
    def test_nothing_is_found_before_installing(self, machine: Path) -> None:
        # 入っていないのに見つかったことにすると、導入の案内が出ず、送った瞬間に失敗する
        del machine
        assert environment.bundled_claude_cli() is None
        assert environment.runtime_status().installed is False

    def test_installing_the_sdk_alone_is_enough(self, machine: Path) -> None:
        """直す前は PATH の claude を要求し、SDK だけでは「使えない」と出ていた

        SDK は Claude Code 本体を同梱している Node で別に入れていない人は、
        導入を済ませても入力欄が開かなかった
        """
        cli = _install_sdk(machine)
        assert environment.bundled_claude_cli() == cli
        status = environment.runtime_status()
        assert status.missing_commands == ()
        assert status.ready is True

    def test_an_sdk_older_than_required_asks_to_be_replaced(self, machine: Path) -> None:
        """前の条件（0.2 以上）で入れた古い SDK を「導入済み」と見ないこと

        見てしまうと入力欄が開き、考える深さを選んだ所で、古い SDK が知らない
        引数を渡されて会話を始めた瞬間に落ちる
        """
        write_distribution(machine, "claude-agent-sdk", "claude_agent_sdk", "0.2.10")
        status = environment.runtime_status()
        assert status.ready is False
        assert status.needs_upgrade is True
        assert "古い版" in status.summary()
        # 配布版の導入先（--target）では --upgrade が無いと入れ替わらない
        command = install_command(AI_PACK, extra=False, upgrade=status.needs_upgrade)
        assert "--upgrade" in command

    def test_a_prerelease_of_the_required_version_is_older(self, machine: Path) -> None:
        # 0.2.174rc1 は 0.2.174 より前の版 同じと読むと、足りない SDK で入力欄が開く
        write_distribution(machine, "claude-agent-sdk", "claude_agent_sdk", "0.2.174rc1")
        status = environment.runtime_status()
        assert status.needs_upgrade is True
        assert status.ready is False

    def test_a_newer_version_is_accepted(self, machine: Path) -> None:
        # 新しい版まで古いと読むと、入れ直しても入力欄が開かない
        _install_sdk(machine)
        (machine / "claude_agent_sdk-0.2.174.dist-info").rename(
            machine / "claude_agent_sdk-0.10.0.dist-info"
        )
        (machine / "claude_agent_sdk-0.10.0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: claude-agent-sdk\nVersion: 0.10.0\n", encoding="utf-8"
        )
        assert environment.runtime_status().ready is True

    def test_an_old_sdk_without_the_bundle_still_asks_for_claude(self, machine: Path) -> None:
        # 壊れると、Claude Code が無いのに入力欄が開き、最初の送信が
        # CLINotFoundError で落ちる 入れ方の案内も出ない
        _install_sdk(machine, bundled=False)
        status = environment.runtime_status()
        assert status.ready is False
        assert "npm install" in status.summary()


class TestCredentials:
    def test_no_login_is_detected(self, machine: Path) -> None:
        # 壊れると、ログインしていない人にログインの案内が出ず、送った指示が
        # 英語の認証エラーで返るまで何をすればよいか分からない
        del machine
        assert environment.credentials_found() is False

    def test_a_logged_in_claude_code_is_detected(self, machine: Path, tmp_path: Path) -> None:
        # 壊れると、ログイン済みの人にもログインの案内が出続け、済んだのか分からない
        del machine
        folder = tmp_path / "home" / ".claude"
        folder.mkdir()
        (folder / ".credentials.json").write_text("{}", encoding="utf-8")
        assert environment.credentials_found() is True

    def test_an_api_key_in_the_environment_counts(
        self, machine: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、API キーで使っている人にも、要らないログインの案内が出続ける
        # 値の中身は見ない 在るかどうかだけ
        del machine
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        assert environment.credentials_found() is True

    def test_the_login_window_runs_only_claude_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、ログインのつもりで余計な引数や shell を通した起動が混ざる
        # 起動するのは見つけた Claude Code の実行ファイルただ 1 つに限る
        started: list[list[str]] = []

        def popen(argv: list[str], **_kwargs: object) -> None:
            started.append(argv)

        monkeypatch.setattr(subprocess, "Popen", popen)
        cli = tmp_path / "claude.exe"
        environment.open_login_window(cli)
        assert started == [[str(cli)]]


def _session(model: str | None = None, effort: str | None = None) -> AgentSession:
    # 渡す設定を組み立てるだけで、ブリッジは使わない
    return AgentSession(cast(EditorBridge, object()), model=model, effort=effort)


class TestSessionOptions:
    def test_nothing_is_forced_by_default(self, machine: Path) -> None:
        # 既定で何かを渡すと、そのモデルを使えないアカウントでは会話が始まらない
        del machine
        values = _session().option_values()
        assert values["model"] is None
        assert "effort" not in values

    def test_the_chosen_model_and_effort_are_passed(self, machine: Path) -> None:
        # 渡し忘れると、選んだ深さが効かないまま Claude Code の既定で考える
        del machine
        values = _session("claude-opus-5-5", "xhigh").option_values()
        assert values["model"] == "claude-opus-5-5"
        assert values["effort"] == "xhigh"

    def test_haiku_gets_no_effort(self, machine: Path) -> None:
        # Haiku 4.5 はエフォートを受け付けない 渡すと会話が始まる前に失敗する
        del machine
        assert "effort" not in _session("claude-haiku-4-5-20251001", "high").option_values()
        assert effort_for("claude-haiku-4-5-20251001", "high") is None

    def test_the_bundled_claude_code_is_left_to_the_sdk(self, machine: Path) -> None:
        # ここで npm の claude.cmd を渡すと、SDK が起動を断る
        _install_sdk(machine)
        assert "cli_path" not in _session().option_values()

    def test_an_old_sdk_gets_the_separately_installed_claude(
        self, machine: Path, tmp_path: Path
    ) -> None:
        # 壊れると、エクスプローラから起動した画面では PATH に無い ~/.local/bin の
        # claude を SDK が見つけられず、入れてあるのに会話が始まらない
        _install_sdk(machine, bundled=False)
        local = tmp_path / "home" / ".local" / "bin" / "claude.exe"
        local.parent.mkdir(parents=True)
        local.write_bytes(b"")
        assert _session().option_values()["cli_path"] == str(local)

    def test_the_options_match_what_the_sdk_accepts(self) -> None:
        """名前を 1 つでも違えると、会話を始めた瞬間に TypeError で落ちる"""
        sdk = pytest.importorskip("claude_agent_sdk")
        fields = set(sdk.ClaudeAgentOptions.__dataclass_fields__)
        values = _session("claude-sonnet-5", "low").option_values()
        assert set(values) <= fields

    def test_the_automatic_memo_is_turned_off(self, machine: Path) -> None:
        # 壊れると、本人のメモが「AI へ送る物」の同意の外で Anthropic へ渡る
        # setting_sources が空でも止まらない（実物で確かめた）
        del machine
        assert _session().option_values()["env"][AUTO_MEMORY_OFF] == "1"

    def test_the_conversation_is_not_written_to_disk(self, machine: Path) -> None:
        # 壊れると、字幕と絵を含む会話の全文が利用者のホームの下に残る
        del machine
        assert _session().option_values()["extra_args"] == {"no-session-persistence": None}

    def test_the_bundled_claude_code_knows_the_flag(self) -> None:
        """知らない引数は Claude Code が断るので、会話が始まらなくなる"""
        cli = environment.bundled_claude_cli()
        if cli is None:
            pytest.skip("同梱の Claude Code が無い")
        shown = subprocess.run([str(cli), "--help"], capture_output=True, check=True)
        assert b"--no-session-persistence" in shown.stdout

    def test_the_conversation_has_its_own_folder(
        self, machine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、起動した場所（リポジトリや本人のフォルダ）を受け継ぎ、そこに紐付く物が混ざる
        del machine
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        monkeypatch.chdir(tmp_path)
        folder = Path(_session().option_values()["cwd"])
        assert folder == environment.conversation_folder()
        assert (tmp_path / "local") in folder.parents

    def test_a_chosen_folder_still_wins(self, machine: Path, tmp_path: Path) -> None:
        # 壊れると、作り手が場所を渡しても黙って無視され、試験や別の使い方で場所を選べない
        del machine
        session = AgentSession(cast(EditorBridge, object()), cwd=tmp_path)
        assert session.option_values()["cwd"] == str(tmp_path)

    def test_a_single_message_can_hold_the_largest_picture(self, machine: Path) -> None:
        # 壊れると、Claude Code が絵を 2 回書いた 1 通が上限を越えて、編集の途中で会話が落ちる
        # 絵は上限の大きさのまま、縮まらない中身（乱数）で作る
        del machine
        side = MAX_SHEET_EDGE
        png = _png(side, side)
        sent = _content(ImageResult(png=png, caption="説明"))
        # 実物の 1 通は道具の結果と tool_use_result に同じ絵が入る
        line = json.dumps({"content": sent["content"], "tool_use_result": sent["content"]})
        assert len(png) > side * side * 4
        assert len(line) <= _session().option_values()["max_buffer_size"]


def _png(width: int, height: int) -> bytes:
    """縮まらない RGBA の PNG 上限を見る試験で、一番重い絵として使う"""
    row = bytes(1 + width * 4)
    # 行ごとに乱数を作ると 1000 万バイトの乱数で遅い 圧縮せずに置くので同じ行の繰り返しでよい
    raw = (row[:1] + os.urandom(width * 4)) * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    head = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    body = zlib.compress(raw, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", body) + chunk(b"IEND", b"")


class _QuietBridge:
    """会話が送るときと畳むときに呼ぶ所だけを持つ 編集の依頼はここへ来ない"""

    def resume(self) -> None:
        return

    def cancel(self) -> None:
        return


class TestSettingsDialogEffort:
    def test_the_effort_is_disabled_for_haiku(self, qt_application: object) -> None:
        """設定画面でも、Haiku 4.5 では考える深さを選べなくする

        選べたままだと、変えても Haiku には渡さないので、応答に何も効かず、
        設定が壊れているように見える
        """
        del qt_application
        from sashimono.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(Preferences(ai_model="claude-haiku-4-5-20251001"))
        assert dialog._ai_effort.isEnabled() is False  # 開いた時点で

        dialog._ai_model.setCurrentIndex(dialog._ai_model.findData("claude-sonnet-5"))
        assert dialog._ai_effort.isEnabled() is True  # 受け付けるモデルへ替えたら戻る

        dialog._ai_model.setCurrentIndex(dialog._ai_model.findData("claude-haiku-4-5-20251001"))
        assert dialog._ai_effort.isEnabled() is False
        dialog.deleteLater()


class TestReconnect:
    def test_a_prompt_left_by_a_failed_connection_is_not_run_later(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """繋がる前に落ちた会話の指示を、次に繋いだときに実行しない

        画面は落ちた時点でその指示を捨てている 残っていると、次の指示より先に
        黙って実行され、その編集が次の指示の取り消しの段へ入る
        """
        queried: list[str] = []
        attempts: list[int] = []

        async def main(self: AgentSession) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                # ログインがまだ、などで接続に失敗した 指示は取られないまま
                raise RuntimeError("接続に失敗しました")
            prompt = self._prompts.get()
            if prompt is not None:
                queried.append(prompt)

        monkeypatch.setattr(AgentSession, "_main", main)
        session = AgentSession(cast(EditorBridge, _QuietBridge()))
        session.send("前の指示")
        assert session._thread is not None
        session._thread.join(timeout=5.0)
        kinds = [event.kind for event in session.poll()]
        assert kinds[-1] is EventKind.CLOSED

        session.send("次の指示")
        session._thread.join(timeout=5.0)
        assert queried == ["次の指示"]


class _FakeClient:
    """ClaudeSDKClient の代わり 受けた指示を覚え、すぐに応え終える"""

    queried: ClassVar[list[str]] = []

    def __init__(self, options: object) -> None:
        del options

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def query(self, prompt: str) -> None:
        _FakeClient.queried.append(prompt)

    async def receive_response(self) -> AsyncIterator[object]:
        return
        yield

    async def interrupt(self) -> None:
        return


class TestClosingDropsPendingPrompts:
    def test_a_closed_session_does_not_run_the_next_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """畳んだ会話が、区切り待ちの間に積まれた次の指示を走らせない（PR #265 の Codex の指摘）

        部品が古くて断られた指示の後に 2 つ目を送ってあると、畳んだ後に古い会話が
        2 つ目を取り出して走らせていた 画面はもうその会話を見ていないので、返事も
        区切りも拾われず、編集だけが古い部品で進む 2 つ目は画面が新しい会話へ渡し直す
        """
        import types

        from sashimono.ai import session as session_module

        fake = types.ModuleType("claude_agent_sdk")
        fake.ClaudeAgentOptions = lambda **_kwargs: object()  # type: ignore[attr-defined]
        fake.ClaudeSDKClient = _FakeClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        monkeypatch.setattr(session_module, "build_server", lambda _bridge, **_options: None)
        monkeypatch.setattr(session_module, "hide_cli_console", lambda: None)
        monkeypatch.setattr(AgentSession, "option_values", lambda _self: {})
        _FakeClient.queried = []

        session = AgentSession(cast(EditorBridge, _QuietBridge()))
        session.send("1 つ目")
        deadline = time.monotonic() + 5.0
        events: list[EventKind] = []
        while EventKind.TURN_DONE not in events and time.monotonic() < deadline:
            events.extend(event.kind for event in session.poll())
            time.sleep(0.01)
        assert EventKind.TURN_DONE in events
        # 画面はまだ区切りを付け終えていない（acknowledge_turn を呼んでいない）所で 2 つ目
        session.send("2 つ目")
        session.close(wait=False)
        assert session._thread is not None
        session._thread.join(timeout=5.0)
        assert session._thread.is_alive() is False
        assert _FakeClient.queried == ["1 つ目"]

    def test_the_conversation_folder_is_made_before_starting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、初めての人の機械に作業場所が無く、Claude Code が起動できずに会話が始まらない
        import types

        from sashimono.ai import session as session_module

        fake = types.ModuleType("claude_agent_sdk")
        fake.ClaudeAgentOptions = lambda **_kwargs: object()  # type: ignore[attr-defined]
        fake.ClaudeSDKClient = _FakeClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        monkeypatch.setattr(session_module, "build_server", lambda _bridge, **_options: None)
        monkeypatch.setattr(session_module, "hide_cli_console", lambda: None)
        _FakeClient.queried = []

        folder = tmp_path / "まだ無い" / "作業場所"
        session = AgentSession(cast(EditorBridge, _QuietBridge()), cwd=folder)
        session.send("指示")
        deadline = time.monotonic() + 5.0
        while not _FakeClient.queried and time.monotonic() < deadline:
            time.sleep(0.01)
        session.close(wait=False)
        assert session._thread is not None
        session._thread.join(timeout=5.0)
        assert folder.is_dir()


class TestPreferences:
    def test_the_choices_survive_a_restart(self, tmp_path: Path) -> None:
        # 壊れると、選んだモデルや送り方が次に起動したとき既定へ戻る
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(ai_model="claude-fable-5-1", ai_effort="max", chat_enter_sends=False)
        store.save(chosen)
        assert store.load() == chosen

    def test_unknown_values_fall_back_to_the_default(self, tmp_path: Path) -> None:
        # 手で書き換えた値や、なくなったモデルで起動を止めない
        path = tmp_path / "preferences.json"
        path.write_text(
            json.dumps({"ai_model": "claude-2", "ai_effort": 3, "chat_enter_sends": "yes"}),
            encoding="utf-8",
        )
        loaded = PreferenceStore(path).load()
        assert (loaded.ai_model, loaded.ai_effort, loaded.chat_enter_sends) == ("", "", True)

    def test_the_settings_dialog_keeps_them(self, qt_application: object) -> None:
        # 壊れると、設定を開いて OK を押しただけで、欄の上で選んだモデルが既定へ戻る
        del qt_application
        from sashimono.ui.preferences_dialog import PreferencesDialog

        chosen = Preferences(ai_model="claude-sonnet-5", ai_effort="medium", chat_enter_sends=False)
        assert PreferencesDialog(chosen).preferences() == chosen
