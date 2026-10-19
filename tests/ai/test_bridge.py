"""AI スレッドと UI スレッドの橋渡し

実際にスレッドを 2 本使って確かめる 片方だけで書くと、待ち合わせの取りこぼしが
すり抜ける
"""

from __future__ import annotations

import threading
import time

import pytest

from sashimono.ai.bridge import EditorBridge, describe_call
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, find_operation
from tests.ai.conftest import FakeHost


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


# 外へ送る物の同意（#342）が要らない、編集の構成だけを返す道具
READ = _operation("get_selection")
WRITE = _operation("split_clip")


def pump_until(bridge: EditorBridge, done: threading.Event, timeout: float = 5.0) -> None:
    """UI スレッドの代わりに、終わるまで仕事を拾い続ける"""
    deadline = time.monotonic() + timeout
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        time.sleep(0.005)
    bridge.pump()


def call_in_thread(
    bridge: EditorBridge, operation: Operation, arguments: dict[str, object]
) -> tuple[threading.Event, dict[str, object]]:
    """AI スレッドのつもりでツールを呼ぶ"""
    done = threading.Event()
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except BaseException as exc:  # 例外もテストの観察対象
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    return done, box


class TestReadOnlyCalls:
    def test_the_work_runs_on_the_pumping_thread(self, host: FakeHost) -> None:
        # UI の外のスレッドで編集を走らせると、画面の描画と取り合って
        # ソフトが落ちたり表示が壊れたりする
        bridge = EditorBridge(host)
        pumped_on: list[int] = []

        done = threading.Event()
        box: dict[str, object] = {}

        def run() -> None:
            box["result"] = bridge.call(lambda: pumped_on.append(threading.get_ident()))
            done.set()

        threading.Thread(target=run, daemon=True).start()
        pump_until(bridge, done)

        assert done.is_set()
        # 仕事をしたのは pump を呼んだ側（＝UI スレッド役）
        assert pumped_on == [threading.get_ident()]

    def test_reads_need_no_approval(self, host: FakeHost) -> None:
        # 中身を見るだけの問い合わせにも許可を尋ねると、AI が様子を調べるたびに押す羽目になる
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, READ, {})
        pump_until(bridge, done)
        assert bridge.take_approval() is None
        assert box["result"]["playhead"] == 0  # type: ignore[index]

    def test_a_frozen_ui_times_out_instead_of_hanging(self, host: FakeHost) -> None:
        # 画面が固まったときに AI の側が待ち続けると、会話が止まったまま戻らなくなる
        from sashimono.ai import bridge as bridge_module

        bridge = EditorBridge(host)
        # pump を一度も呼ばない＝UI が固まっている状況
        original = bridge_module.CALL_TIMEOUT
        bridge_module.CALL_TIMEOUT = 0.05
        try:
            with pytest.raises(ToolError, match="応答しません"):
                bridge.call(lambda: None)
        finally:
            bridge_module.CALL_TIMEOUT = original


class TestApproval:
    def test_writes_wait_for_a_decision(self, host: FakeHost) -> None:
        # 許可を押す前に分割が走ると、利用者が確かめる前にタイムラインが勝手に変わる
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)
        bridge = EditorBridge(host)
        done, _ = call_in_thread(bridge, WRITE, {"clip_id": clip, "frame": 100})

        # 許可を出すまでは何も起きない
        approval = None
        deadline = time.monotonic() + 5.0
        while approval is None and time.monotonic() < deadline:
            bridge.pump()
            approval = bridge.take_approval()
            time.sleep(0.005)

        assert approval is not None
        assert approval.tool == "split_clip"
        assert not done.is_set()
        assert len(host.document.project.timeline.tracks[0].clips) == 1

        approval.allow()
        pump_until(bridge, done)
        assert len(host.document.project.timeline.tracks[0].clips) == 2

    def test_denial_reaches_the_tool_as_an_error(self, host: FakeHost) -> None:
        # 断った事が AI に伝わらないと、AI は成功したつもりで次の編集へ進む
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, WRITE, {"clip_id": clip, "frame": 100})

        approval = None
        while approval is None:
            bridge.pump()
            approval = bridge.take_approval()
            time.sleep(0.005)
        approval.deny()
        pump_until(bridge, done)

        assert isinstance(box["error"], ToolError)
        assert "許可しません" in str(box["error"])
        assert len(host.document.project.timeline.tracks[0].clips) == 1

    def test_auto_approve_skips_the_question(self, host: FakeHost) -> None:
        # 自動で許可にしても毎回尋ねられると、まとめて任せたい人の手が止まり続ける
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)
        bridge = EditorBridge(host, auto_approve=True)
        done, _ = call_in_thread(bridge, WRITE, {"clip_id": clip, "frame": 100})
        pump_until(bridge, done)
        assert bridge.take_approval() is None
        assert len(host.document.project.timeline.tracks[0].clips) == 2

    def test_always_allow_covers_later_calls(self, host: FakeHost) -> None:
        # 〔常に許可〕を選んでも次の呼び出しでまた尋ねられると、同じ問いに何度も答えることになる
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)
        bridge = EditorBridge(host)
        bridge.allow_always("split_clip")
        done, _ = call_in_thread(bridge, WRITE, {"clip_id": clip, "frame": 100})
        pump_until(bridge, done)
        assert bridge.take_approval() is None

    def test_forgetting_always_brings_the_question_back(self, host: FakeHost) -> None:
        # 〔常に許可〕を取り消しても効いたままだと、利用者の知らないうちに AI が編集を続ける
        bridge = EditorBridge(host)
        bridge.allow_always("split_clip")
        bridge.forget_always()
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)
        done, _ = call_in_thread(bridge, WRITE, {"clip_id": clip, "frame": 100})

        approval = None
        while approval is None:
            bridge.pump()
            approval = bridge.take_approval()
            time.sleep(0.005)
        approval.deny()
        pump_until(bridge, done)


class TestCancel:
    def test_cancelling_releases_a_waiting_approval(self, host: FakeHost) -> None:
        # 承認待ちのまま中断すると、AI スレッドが 10 分止まる 畳んで返す
        clip = str(host.document.project.timeline.tracks[0].clips[0].id)
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, WRITE, {"clip_id": clip, "frame": 100})

        while bridge.take_approval() is None:
            bridge.pump()
            time.sleep(0.005)
        bridge.cancel()
        pump_until(bridge, done)

        assert isinstance(box["error"], ToolError)
        assert len(host.document.project.timeline.tracks[0].clips) == 1

    def test_after_cancel_new_calls_fail_fast(self, host: FakeHost) -> None:
        # 中断した後の呼び出しが UI を待ちに入ると、止めたはずの AI が応答待ちで固まる
        bridge = EditorBridge(host)
        bridge.cancel()
        with pytest.raises(ToolError, match="中断"):
            bridge.call(lambda: None)

    def test_resume_lets_work_through_again(self, host: FakeHost) -> None:
        # 中断の後で戻せないと、一度止めた会話では AI がもう何もできなくなる
        bridge = EditorBridge(host)
        bridge.cancel()
        bridge.resume()
        done, box = call_in_thread(bridge, READ, {})
        pump_until(bridge, done)
        assert "error" not in box


class TestDescribeCall:
    def test_arguments_are_shown_with_the_description(self) -> None:
        # ツール名だけでは何が起きるか分からない 押す前に見えるようにする
        text = describe_call(WRITE, {"clip_id": "abc", "frame": 100})
        assert "clip_id=abc" in text
        assert "frame=100" in text

    def test_long_values_are_trimmed(self) -> None:
        # 長い値をそのまま並べると、許可を尋ねる欄が文字で埋まり、何をする呼び出しか読み取れない
        text = describe_call(WRITE, {"text": "あ" * 200})
        assert "…" in text
        assert len(text) < 200
