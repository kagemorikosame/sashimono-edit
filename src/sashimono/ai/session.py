"""Claude Agent SDK との会話を、UI から扱える形に包む

SDK は asyncio で動き、``claude`` コマンドを子プロセスとして起動する Qt の
イベントループとは混ぜられないので、専用スレッドで asyncio を回し、やり取りは
キュー越しにする（:mod:`sashimono.ai.bridge` と同じ考え方）

UI がやることは 3 つだけ :meth:`AgentSession.send` で送り、:meth:`AgentSession.poll`
で溜まった出来事を拾い、:meth:`AgentSession.interrupt` で止める
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.console import hide_cli_console
from sashimono.ai.environment import bundled_claude_cli, find_claude_cli
from sashimono.ai.models import effort_for
from sashimono.ai.server import SERVER_NAME, build_server
from sashimono.core.model import LayerMode

__all__ = ["SYSTEM_PROMPT", "AgentEvent", "AgentSession", "EventKind", "system_prompt"]

#: 置き方の方式ごとの、素材の絵と音の持ち方の説明 :func:`system_prompt` が差し込む
#: 方式を取り違えて伝えると、混合の作品で AI がリンクした音声クリップを探し回ったり、
#: 分ける方式の作品で組の両方に同じ操作をして 2 回目で失敗したりする
#: 方式が決めるのは**これから置く物**だけ 方式を途中で変えた作品や、分ける方式で
#: レイヤーを足した作品には両方の形が並ぶので、置いてある物はクリップごとに確かめさせる
_CLIP_SHAPES = """\
- 置いてあるクリップの形は、方式ではなくクリップごとに確かめます list_clips の
  link_group が同じクリップ（映像トラックと音声トラックの組や、レイヤーに分けた絵と
  音の組）は組で、1 本を分割・削除・移動・トリムすると、ほかも同じように動きます
  **組のどれにも同じ操作をしないでください**（2 回目は失敗します） 組の 1 本に
  1 回だけ操作します track_kind が mixed で link_group の無いクリップは、絵と音を
  1 本で持ちます レイヤー（mixed のトラック）は番号が大きいほど手前に描かれます"""

# 混合の方式で音付きの動画を分けて置くかは本人の設定で変わる 片方の形だけを言い切ると、
# 設定を変えた人の作品で AI が無い組を探し回るか、組の両方に同じ操作をして失敗する
_LINKED_CLIPS = {
    LayerMode.SEPARATED: f"""\
- このプロジェクトは分ける方式です これから置く音付きの動画は、映像トラックと
  音声トラックの 2 本に分かれ、リンクで組になります
{_CLIP_SHAPES}""",
    LayerMode.MIXED: f"""\
- このプロジェクトは混合の方式です これから置く音付きの動画は、本人の設定により、
  絵のレイヤーと音ごとのレイヤーに分かれてリンクで組になるか、絵と音を 1 本の
  クリップで持ってレイヤーに置かれます 置いた後は list_clips で形を確かめます
{_CLIP_SHAPES}""",
}

_PROMPT = """\
あなたは動画編集ソフト Sashimono Edit の中で動く編集アシスタントです ユーザーの指示を、
用意されたツールで実際の編集操作に変えてください

守ること:

- タイムライン上の時刻はすべて**プロジェクト fps 基準の整数フレーム**です 秒で
  言われたら fps を掛けて frame に直します（get_project で fps が分かります）
- 素材やクリップは ID で指します まず list_media / list_clips で ID を確かめてから
  操作してください 当てずっぽうの ID は失敗します
- 何かを変えたら preview_frame でその位置を描いて、**自分の目で結果を確かめて**
  ください 数値が正しくても見た目が意図と違うことがあります
{linked_clips}
- 置き方の方式は get_project の layer_mode で分かります 途中で変わっていたら、
  そちらに従ってください
- 字幕は素材に紐付いていて、カットや分割には自動で追従します 字幕の位置を手で
  合わせ直す必要はありません
- 色は #RRGGBB で指定します
- 大きく変える前に、何をするつもりかを 1〜2 文で伝えてください 変更系の操作には
  ユーザーの確認が入ります
- 失敗したら、エラーの文面を読んで直してから再試行してください 同じ操作を
  そのまま繰り返さないこと

返事は日本語で、簡潔に 作業の実況ではなく、やったことと結果を伝えてください
"""


def system_prompt(layer_mode: str = LayerMode.SEPARATED) -> str:
    """エージェントへの指示 ``layer_mode`` はプロジェクトの置き方の方式

    会話を始めるときの方式で書く 会話の途中で方式を変えることもあるので、確かめ方
    （get_project の layer_mode）も添えてある
    """
    linked = _LINKED_CLIPS.get(layer_mode, _LINKED_CLIPS[LayerMode.SEPARATED])
    # format ではなく置き換えにする 指示の文に波括弧を書いたときに壊れないように
    return _PROMPT.replace("{linked_clips}", linked)


#: 分ける方式（モデルの既定）での指示 会話を作る側が方式を渡さないときに使う
SYSTEM_PROMPT = system_prompt()


class EventKind(Enum):
    READY = "ready"
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    TURN_DONE = "turn_done"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """エージェントからの 1 件の知らせ"""

    kind: EventKind
    text: str = ""
    tool: str = ""
    detail: str = ""


class AgentSession:
    """1 つの会話

    会話は繋ぎっぱなしにする 1 回ごとに繋ぎ直すと、それまでのやり取りを毎回
    渡し直すことになり、待ち時間も費用も増える
    """

    def __init__(
        self,
        bridge: EditorBridge,
        *,
        model: str | None = None,
        effort: str | None = None,
        cwd: Path | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._bridge = bridge
        self._model = model or None
        self._effort = effort_for(model or "", effort or "")
        self._cwd = cwd
        self._system_prompt = system_prompt

        self._events: queue.Queue[AgentEvent] = queue.Queue()
        self._prompts: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self._busy = threading.Event()
        self._closed = threading.Event()
        #: 画面が前の指示の区切り（取り消しの段）を付け終えたか
        #: 付け終える前に次の指示を始めると、次の指示の編集が前の段へ混ざり、
        #: 1 回の取り消しで 2 つの指示の編集がまとめて戻る
        self._boundary = threading.Event()
        self._boundary.set()

    # --- UI スレッドから ---

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def model(self) -> str | None:
        """使うモデル ``None`` は Claude Code の既定"""
        return self._model

    @property
    def effort(self) -> str | None:
        """実際に渡すエフォート 受け付けないモデルでは ``None``"""
        return self._effort

    @property
    def busy(self) -> bool:
        """いま応答を生成している最中か"""
        return self._busy.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._closed.clear()
        # 前の接続が繋がる前に落ちると（Claude Code が見つからない・ログインが
        # まだ、など）、取られなかった指示が残る 残したまま繋ぎ直すと、画面が
        # 捨てた指示を黙って先に実行し、その編集が次の指示の取り消しの段へ入る
        self._prompts = queue.Queue()
        self._boundary.set()
        self._thread = threading.Thread(target=self._run, name="sashimono-agent", daemon=True)
        self._thread.start()

    def send(self, prompt: str) -> None:
        """指示を送る まだ繋がっていなければ繋いでから送る"""
        self.start()
        self._bridge.resume()
        self._prompts.put(prompt)

    def acknowledge_turn(self) -> None:
        """TURN_DONE を受けて区切りを付け終えた 次の指示を始めてよい

        画面が TURN_DONE を拾うのはタイマーの次の回なので、それまで会話の
        スレッドを待たせる
        """
        self._boundary.set()

    def poll(self) -> list[AgentEvent]:
        """溜まった出来事を取り出す ブロックしない"""
        drained: list[AgentEvent] = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                return drained

    def interrupt(self) -> None:
        """生成を止める 待っているツールも諦めさせる"""
        self._bridge.cancel()
        loop, client = self._loop, self._client
        if loop is not None and client is not None:
            asyncio.run_coroutine_threadsafe(_safe_interrupt(client), loop)

    def close(self, *, wait: bool = True) -> None:
        """会話を畳む 終了時に呼ぶ

        ``wait`` を切ると、畳み終わるのを待たずに戻る モデルを選び直したときの
        ように、アプリは続く場面で使う 待つと Claude Code が終わるまで画面が固まる
        """
        if not self.running:
            return
        self._closed.set()
        self._boundary.set()  # 区切り待ちのまま畳まれずに残らないように
        self._bridge.cancel()
        self._prompts.put(None)
        thread = self._thread
        if wait and thread is not None:
            thread.join(timeout=5.0)

    # --- エージェントスレッド ---

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # スレッドの外へ例外を出さない
            self._emit(AgentEvent(EventKind.ERROR, text=_explain(exc)))
        finally:
            self._loop = None
            self._client = None
            self._busy.clear()
            self._emit(AgentEvent(EventKind.CLOSED))

    async def _main(self) -> None:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        # 繋ぐ前に包む 繋いだ後では、claude.exe の黒い窓がもう開いている（#228）
        hide_cli_console()
        self._loop = asyncio.get_running_loop()
        options = ClaudeAgentOptions(
            **self.option_values(),
            mcp_servers={SERVER_NAME: build_server(self._bridge)},
            can_use_tool=self._can_use_tool,
        )

        async with ClaudeSDKClient(options) as client:
            self._client = client
            self._emit(AgentEvent(EventKind.READY))
            while not self._closed.is_set():
                await asyncio.to_thread(self._boundary.wait)
                prompt = await asyncio.to_thread(self._prompts.get)
                if prompt is None:
                    break
                # TURN_DONE を出す前に下ろす 出した後だと、画面が先に区切りを
                # 付け終えて立てた旗を、ここで消してしまうことがある
                self._boundary.clear()
                await self._turn(client, prompt)

    def option_values(self) -> dict[str, Any]:
        """SDK へ渡す設定のうち、SDK を読まずに決まるもの

        分けてあるのは、SDK の無い試験でも「何を渡すか」を確かめるため
        """
        values: dict[str, Any] = {
            "system_prompt": self._system_prompt,
            # 組み込みのツール（ファイル読み書き・シェル・検索）は全部止める
            # 編集の話をしているつもりで、ディスクの中身を読まれては困る
            "tools": [],
            # allowed_tools は使わない 並べると SDK がそちらで先に通してしまい、
            # 門番（can_use_tool）が呼ばれなくなる 許可の判断を 1 か所に寄せる
            # ソフトの中の会話なので、リポジトリの設定やスキルは読み込まない
            # ここを開けると、編集と関係ない指示が混ざる
            "setting_sources": [],
            "permission_mode": "default",
            "model": self._model,
            "cwd": str(self._cwd) if self._cwd is not None else None,
        }
        if self._effort is not None:
            # 指定しないときは渡さない 渡すと Claude Code の既定を上書きしてしまう
            values["effort"] = self._effort
        if bundled_claude_cli() is None:
            # 同梱の Claude Code があれば SDK が自分で見つける ここで別の場所を
            # 渡すと、npm の claude.cmd（SDK が起動を断る）を掴むことがある
            # 同梱の無い古い SDK のときだけ、PATH に無い場所まで探した物を渡す
            cli = find_claude_cli()
            if cli is not None:
                values["cli_path"] = str(cli)
        return values

    async def _turn(self, client: Any, prompt: str) -> None:
        self._busy.set()
        try:
            await client.query(prompt)
            async for message in client.receive_response():
                self._translate(message)
        except Exception as exc:
            self._emit(AgentEvent(EventKind.ERROR, text=_explain(exc)))
        finally:
            self._busy.clear()
            self._emit(AgentEvent(EventKind.TURN_DONE))

    async def _can_use_tool(self, name: str, arguments: dict[str, Any], context: Any) -> Any:
        """この編集ソフトのツール以外は使わせない

        ファイル操作やシェルまで開けると、編集の話をしているつもりが別のことを
        されうる 変更系の確認はブリッジ側で取っているので、ここは門番だけ
        """
        del arguments, context
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        if name.startswith(f"mcp__{SERVER_NAME}__"):
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"{name} はこの画面からは使えません")

    def _translate(self, message: Any) -> None:
        """SDK のメッセージを、UI が知っている形へ"""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    self._emit(AgentEvent(EventKind.TEXT, text=block.text))
                elif isinstance(block, ThinkingBlock):
                    self._emit(AgentEvent(EventKind.THINKING, text=block.thinking))
                elif isinstance(block, ToolUseBlock):
                    self._emit(
                        AgentEvent(
                            EventKind.TOOL_USE,
                            tool=_short_name(block.name),
                            detail=_summarize(block.input),
                        )
                    )
        elif isinstance(message, UserMessage):
            for block in message.content if isinstance(message.content, list) else ():
                if isinstance(block, ToolResultBlock):
                    self._emit(
                        AgentEvent(
                            EventKind.TOOL_RESULT,
                            detail=_summarize(block.content),
                            text="失敗" if block.is_error else "",
                        )
                    )
        elif isinstance(message, ResultMessage) and message.is_error:
            self._emit(
                AgentEvent(
                    EventKind.ERROR, text=with_login_hint(str(message.result or "失敗しました"))
                )
            )

    def _emit(self, event: AgentEvent) -> None:
        self._events.put(event)


async def _safe_interrupt(client: Any) -> None:
    try:
        await client.interrupt()
    except Exception:  # 止める処理で落ちても意味が無い
        return


def _short_name(name: str) -> str:
    prefix = f"mcp__{SERVER_NAME}__"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _summarize(value: object, limit: int = 160) -> str:
    """ツールの引数や結果を 1 行に畳む"""
    if isinstance(value, dict):
        text = ", ".join(f"{k}={v}" for k, v in value.items())
    elif isinstance(value, list):
        parts = []
        for entry in value:
            if isinstance(entry, dict) and entry.get("type") == "text":
                parts.append(str(entry.get("text", "")))
            elif isinstance(entry, dict) and entry.get("type") == "image":
                parts.append("（画像）")
            else:
                parts.append(str(entry))
        text = " ".join(parts)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _explain(exc: BaseException) -> str:
    """例外を、次にどうすればよいか分かる文へ"""
    missing = _missing_part(exc)
    if missing is not None:
        return missing
    name = type(exc).__name__
    if name == "CLINotFoundError":
        return (
            "Claude Code が見つかりません npm install -g @anthropic-ai/claude-code で"
            "入れてから、もう一度お試しください"
        )
    if name == "ProcessError":
        return with_login_hint(f"Claude Code の起動に失敗しました: {exc}")
    return with_login_hint(f"{name}: {exc}")


#: 部品が読めないときの案内に書く、環境を入れ直す所
REINSTALL_HINT = "アシスタント欄の「環境を更新」から入れ直してください"


def _missing_part(exc: BaseException) -> str | None:
    """部品が読めないための失敗なら、何が足りないかと次にすることを書いた文 違えば ``None``

    ``ModuleNotFoundError`` のままでは、使う人は何をすればよいか分からない（利用者の画面）
    標準ライブラリが無いのは配布版の不具合なので入れ直しでは直らない 新しい版を案内する
    """
    if not isinstance(exc, ImportError):
        return None
    module = exc.name or ""
    head = module.split(".")[0]
    if head and head in sys.stdlib_module_names:
        return (
            f"配布版に Python の標準の部品 {module} が入っていないため、アシスタントを動かせません"
            " 配布版の不具合です お手数ですが新しい版への更新をお願いします"
            f"（不具合の報告に「{module} が無い」と添えてください）"
        )
    what = module or str(exc)
    return f"アシスタントの部品 {what} を読めませんでした {REINSTALL_HINT}"


#: ログインが済んでいないときに Claude Code が返す文面の手掛かり（小文字で比べる）
_LOGIN_MARKERS = ("/login", "api key", "not logged in", "authentication", "oauth", "401")

#: ログインが要るときに添える案内
LOGIN_HINT = (
    "Claude へのログインが要ります アシスタント欄の「ログイン…」で Claude Code を開き、"
    "案内に従ってログインしてから送り直してください"
    "（API キーを使う場合は環境変数 ANTHROPIC_API_KEY を設定し、ソフトを再起動します）"
)


def with_login_hint(text: str) -> str:
    """ログインが済んでいないための失敗なら、次にすることを書き足す

    Claude Code の文面は英語で「/login を実行して」と言うだけで、ソフトの中から
    どこでそれをすればよいのかが分からない
    """
    lowered = text.lower()
    if any(marker in lowered for marker in _LOGIN_MARKERS):
        return f"{text}\n{LOGIN_HINT}"
    return text
