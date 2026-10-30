"""編集操作を MCP のツールとして公開する

外部プロセスではなく**アプリ内の MCP サーバ**にしてある 編集中のプロジェクトは
メモリ上にしかないので、プロセスをまたぐとファイル経由で受け渡すことになり、
保存していない状態を AI が触れなくなる

:mod:`sashimono.ai.operations` に並べた操作を、ここで 1 つずつ MCP ツールへ包む
包む処理は全部同じなので、ツールを足すときに触るのは operations だけで済む

**SDK の import は関数の中で行う** AI 連携の実行環境は同梱しておらず、
ソフト内の導入ボタンで後から入れる（計画書 F-9-9） ここで表に import すると、
未導入の環境では**このモジュールを読み込んだだけで落ち**、繋がっている
チャットパネル経由で UI 全体が起動しなくなる
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import SdkMcpTool
    from claude_agent_sdk.types import McpSdkServerConfig

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import OPERATIONS, ImageResult, Operation
from sashimono.ai.plan import PLAN_TOOLS
from sashimono.ai.sheet import MAX_SHEET_EDGE
from sashimono.core.io.serialize import json_text

__all__ = [
    "MAX_MESSAGE_BYTES",
    "SERVER_NAME",
    "build_server",
    "build_tools",
    "tool_name",
    "tool_names",
]

#: MCP サーバの名前 Claude 側でのツール名は ``mcp__sashimono__<ツール名>`` になる
SERVER_NAME = "sashimono"

#: 道具が返す 1 枚の絵の、縮まらないときの大きさ 長い辺の上限の正方形を RGBA のまま持つ形
_LARGEST_IMAGE_BYTES = MAX_SHEET_EDGE * MAX_SHEET_EDGE * 4
#: Claude Code から届く 1 通の上限（SDK の ``max_buffer_size``）
#: 既定の 1 MiB では編集の途中で会話が落ちた Claude Code は道具の結果と ``tool_use_result`` に
#: 同じ絵の base64 を書くので、絵が 2 回入り、PNG が 39 万バイトを超えるだけで既定を越える
#: 絵の base64（4/3 倍）2 回分に、説明の文と JSON の分として 1 MiB を足す 絵を描く道具は
#: どれも長い辺を :data:`MAX_SHEET_EDGE` までに収める 1 指示の絵の枚数（既定 30）は掛けない
#: 道具の結果は 1 つずつ別の 1 通で届く（通しの確かめの記録で、並べて呼んだ道具も別の通だった）
#: 上限は読むときの見張りで、前もって確保する量ではないので、大きくしても使う記憶は増えない
MAX_MESSAGE_BYTES = (_LARGEST_IMAGE_BYTES * 4 // 3 + 4) * 2 + 1024 * 1024


def tool_name(operation: Operation | str) -> str:
    """Claude から見えるツール名"""
    name = operation if isinstance(operation, str) else operation.name
    return f"mcp__{SERVER_NAME}__{name}"


def tool_names(*, writes: bool | None = None) -> list[str]:
    """ツール名の一覧 ``writes`` を指定すると読み取り／変更で絞る"""
    return [
        tool_name(operation)
        for operation in OPERATIONS
        if writes is None or operation.writes is writes
    ]


def build_tools(
    bridge: EditorBridge, *, ask: bool = True, plan: bool = True
) -> list[SdkMcpTool[Any]]:
    """ブリッジに繋がった MCP ツールを組み立てる

    サーバとは別に取り出せるようにしてある ``create_sdk_mcp_server`` が返す
    設定からはツールを覗けないので、試験からはこちらを使う
    ``ask`` を切ると質問の道具（ask_user）を渡さない 設定で「AI が質問する」を
    切った人に、道具だけ残って聞かれることが無いように
    ``plan`` を切ると計画の道具（#340）を渡さない 設定で計画を切った人の会話で、
    道具だけ見えて計画を書きに来ることが無いように
    """
    return [
        _wrap(operation, bridge)
        for operation in OPERATIONS
        if (ask or not operation.asks) and (plan or operation.name not in PLAN_TOOLS)
    ]


def build_server(
    bridge: EditorBridge, *, ask: bool = True, plan: bool = True
) -> McpSdkServerConfig:
    """ブリッジに繋がった MCP サーバを作る"""
    from claude_agent_sdk import create_sdk_mcp_server

    return create_sdk_mcp_server(
        name=SERVER_NAME, version="1.0.0", tools=build_tools(bridge, ask=ask, plan=plan)
    )


def _wrap(operation: Operation, bridge: EditorBridge) -> SdkMcpTool[Any]:
    """1 つの操作を MCP ツールにする"""
    from claude_agent_sdk import SdkMcpTool

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        try:
            # ブリッジの中で UI スレッドの返事を待つ ここで直接待つと、
            # SDK のメッセージ受信まで止まって中断が効かなくなる
            result = await loop.run_in_executor(
                None, lambda: bridge.invoke(operation, dict(arguments))
            )
        except ToolError as exc:
            return _error(str(exc))
        except (ValueError, KeyError) as exc:
            # 編集コマンドが弾いた場合 理由はそのまま返す方が次の手を選びやすい
            return _error(str(exc))
        return _content(result)

    return SdkMcpTool(
        name=operation.name,
        description=operation.description,
        input_schema=operation.schema,
        handler=handler,
    )


def _content(result: object) -> dict[str, Any]:
    if isinstance(result, ImageResult):
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "data": base64.b64encode(result.png).decode("ascii"),
                "mimeType": "image/png",
            }
        ]
        if result.caption:
            blocks.append({"type": "text", "text": result.caption})
        return {"content": blocks}

    # 効果の値には読めないバイトを持つ文字が入りうる そのまま渡すと、返事を
    # UTF-8 にする所で落ちる（json_text を参照）
    text = result if isinstance(result, str) else json_text(result, default=str)
    return {"content": [{"type": "text", "text": text}]}


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}
