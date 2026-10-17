"""操作を MCP ツールへ包むところ

SDK は追加導入なので、入っていない環境ではこのファイルごと飛ばす ツールの
中身そのものは :mod:`tests.ai.test_operations` で見ているので、ここは包み方だけ
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk", reason="AI 連携の実行環境が未導入")

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.server import (
    SERVER_NAME,
    build_server,
    build_tools,
    tool_name,
    tool_names,
)
from tests.ai.conftest import FakeHost, allow_sending


def call_tool(bridge: EditorBridge, tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """ツールを呼びつつ、UI スレッドの代わりに仕事を拾う"""
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["result"] = asyncio.run(tool.handler(arguments))
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    deadline = time.monotonic() + 10.0
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        approval = bridge.take_approval()
        if approval is not None:
            approval.allow()
        time.sleep(0.005)
    bridge.pump()
    assert done.is_set(), "ツールが返ってこない"
    result: dict[str, Any] = box["result"]
    return result


def find(tools: list[Any], name: str) -> Any:
    return next(tool for tool in tools if tool.name == name)


class TestNaming:
    def test_tool_names_carry_the_server_prefix(self) -> None:
        # 接頭辞が食い違うと、門番が自分のツールを他所の物と見なして断り、
        # AI が何も編集できない
        assert tool_name("split_clip") == f"mcp__{SERVER_NAME}__split_clip"

    def test_names_can_be_filtered_by_kind(self) -> None:
        # 読み取りと変更の分け目が崩れると、名前で許す範囲を絞ったときに変更の道具が混じる
        reads = tool_names(writes=False)
        writes = tool_names(writes=True)
        assert tool_name("list_clips") in reads
        assert tool_name("list_clips") not in writes
        assert set(reads) | set(writes) == set(tool_names())


class TestServer:
    def test_every_operation_becomes_a_tool(self, host: FakeHost) -> None:
        # 包み損ねた操作は AI から見えず、画面でできることを頼んでもできないと返る
        from sashimono.ai.operations import OPERATIONS

        assert len(build_tools(EditorBridge(host))) == len(OPERATIONS)
        # サーバそのものも組み立てられること
        assert build_server(EditorBridge(host))["type"] == "sdk"

    def test_a_read_returns_json_text(self, host: FakeHost) -> None:
        # JSON の文字で返さないと、AI は結果を読めず、尺やクリップの数を取り違えて答える
        bridge = allow_sending(EditorBridge(host))
        tool = find(build_tools(bridge), "get_project")

        result = call_tool(bridge, tool, {})
        payload = json.loads(result["content"][0]["text"])
        assert payload["duration_frames"] == 300

    def test_an_image_comes_back_as_an_image_block(self, host: FakeHost) -> None:
        # 絵を文字として返すと、AI は画面を見られず、見た目の確認を頼んでも憶測で答える
        bridge = allow_sending(EditorBridge(host))
        tool = find(build_tools(bridge), "preview_frame")

        result = call_tool(bridge, tool, {"frame": 30})
        image = result["content"][0]
        assert image["mimeType"] == "image/png"
        assert base64.b64decode(image["data"]).startswith(b"\x89PNG")
        # 説明文も付ける 画像だけだと、それがどのフレームか分からない
        assert "00:00:01:00" in result["content"][1]["text"]

    @pytest.mark.parametrize("name", ["contact_sheet", "preview_range"])
    def test_a_sheet_comes_back_as_one_image_block(self, host: FakeHost, name: str) -> None:
        # 壊れると、続きのある道具の途中の印が AI へ返り、絵の代わりに読めない文字が届く
        bridge = allow_sending(EditorBridge(host))
        tool = find(build_tools(bridge), name)

        result = call_tool(bridge, tool, {})
        image = result["content"][0]
        assert image["mimeType"] == "image/png"
        assert base64.b64decode(image["data"]).startswith(b"\x89PNG")
        assert "00:00:00" in result["content"][1]["text"]

    def test_a_failure_is_reported_as_an_error_result(self, host: FakeHost) -> None:
        # 例外を投げると会話が止まる 失敗も結果として返し、次の手を選ばせる
        bridge = EditorBridge(host)
        tool = find(build_tools(bridge), "split_clip")

        result = call_tool(bridge, tool, {"clip_id": "ないよ", "frame": 10})
        assert result["is_error"] is True
        assert "list_clips" in result["content"][0]["text"]

    def test_writes_pass_through_the_approval(self, host: FakeHost) -> None:
        # 確認を通った変更が当たらないと、利用者が許可しても AI の編集が反映されない
        bridge = EditorBridge(host)
        tool = find(build_tools(bridge), "split_clip")
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)

        # call_tool は届いた確認を許可する 許可した結果として分割される
        call_tool(bridge, tool, {"clip_id": clip, "frame": 100})
        assert len(host.document.project.timeline.tracks[0].clips) == 2

    def test_schemas_are_passed_through(self, host: FakeHost) -> None:
        # 引数の形が SDK へ渡らないと、AI は text が要ると知らずに文字の無いテロップを頼んで失敗する
        tool = find(build_tools(EditorBridge(host)), "add_text")
        assert "text" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["text"]
