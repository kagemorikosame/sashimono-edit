"""AI が使う量の上限（#344）の画面

AI のパネルが上限に当たったときに尋ねる箱を出して答えを運ぶこと、使った量を見せること、
設定画面の値が保存されて数えと道具に効くことを見る 数え方そのものは tests/ai/test_budget.py
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.budget import (
    DEFAULT_IMAGE_EDGE,
    DEFAULT_TURN_TOKENS,
    LimitAction,
    LimitKind,
    Limits,
)
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.session import AgentEvent, EventKind
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.ui.chat import ChatPanel
from sashimono.ui.chat.panel import BUDGET_HEAD, USAGE_FILE
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore, config_root
from tests.ai.conftest import FakeHost, allow_sending, make_loaded


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす

    上限の確かめはパネルの中の箱で、窓は出さない 窓が出たら作りが変わったということで、
    開いたまま待つと試験が人の操作を待って止まる
    """

    def refuse(*_arguments: object, **_options: object) -> None:
        raise AssertionError("試験の途中で窓が開いた")

    for owner, name in (
        (QDialog, "exec"),
        (QDialog, "open"),
        (QDialog, "show"),
        (QMainWindow, "show"),
        (QMessageBox, "exec"),
        (QMessageBox, "warning"),
        (QMessageBox, "information"),
        (QMessageBox, "question"),
        (QMessageBox, "critical"),
        (QFileDialog, "getSaveFileName"),
        (QFileDialog, "getOpenFileName"),
    ):
        monkeypatch.setattr(owner, name, refuse)


@pytest.fixture
def loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    return make_loaded(video_media, transcript)


@pytest.fixture
def panel(qt_application: QApplication, loaded: Project) -> Iterator[tuple[ChatPanel, FakeHost]]:
    del qt_application
    host = FakeHost(loaded)
    created = ChatPanel(host)
    allow_sending(created._bridge)
    yield created, host
    created.close_session()
    created.deleteLater()


def _call_in_thread(widget: ChatPanel, name: str) -> tuple[threading.Event, dict[str, object]]:
    """AI のスレッドのつもりで道具を呼ぶ 画面のスレッドは :func:`_poll_until` が回す"""
    operation = find_operation(name)
    assert operation is not None
    done = threading.Event()
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["result"] = widget._bridge.invoke(operation, {})
        except BaseException as exc:  # 例外も試験の観察対象
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    return done, box


def _poll_until(widget: ChatPanel, condition: object, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        widget._poll()
        if callable(condition) and condition():
            return
        time.sleep(0.005)
    pytest.fail("待っていた様子にならなかった")


def _over_tokens(widget: ChatPanel, limits: Limits) -> None:
    """指示を始め、トークンの上限を使い切った所にする"""
    widget._bridge.usage.limits = limits
    widget._begin_turn()
    widget._bridge.usage.record_message("m", {"output_tokens": limits.turn_tokens})


class TestAsking:
    def test_the_box_appears_and_continuing_runs_the_tool(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、上限に当たっても箱が出ず、AI のスレッドが返事を 10 分待って止まったままになる
        widget, _ = panel
        _over_tokens(widget, Limits(turn_tokens=100))
        done, box = _call_in_thread(widget, "get_selection")
        _poll_until(widget, lambda: widget._budget is not None)
        assert widget._budget_box.isHidden() is False
        assert BUDGET_HEAD in widget._budget_text.text()
        assert "200 トークン" in widget._budget_go.text()
        assert widget._status_line().startswith("上限の確かめ待ち")
        assert not done.is_set()
        widget._budget_go.click()
        _poll_until(widget, done.is_set)
        assert "result" in box
        assert widget._budget_box.isHidden() is True
        assert "続けます" in widget._view.toPlainText()

    def test_stopping_refuses_and_says_why(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、〔ここで止める〕を押しても道具が動くか、止めた訳が会話の欄に残らない
        widget, _ = panel
        _over_tokens(widget, Limits(turn_tokens=100))
        done, box = _call_in_thread(widget, "get_selection")
        _poll_until(widget, lambda: widget._budget is not None)
        widget._budget_stop.click()
        _poll_until(widget, done.is_set)
        assert isinstance(box.get("error"), ToolError)
        shown = widget._view.toPlainText()
        assert "上限" in shown
        assert "表示 → 設定…" in shown
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "上限で停止" in widget._view.toPlainText()

    def test_interrupting_takes_the_box_down(self, panel: tuple[ChatPanel, FakeHost]) -> None:
        # 壊れると、中断した後も押しても何も起きない箱が残る
        widget, _ = panel
        _over_tokens(widget, Limits(turn_tokens=100))
        done, _box = _call_in_thread(widget, "get_selection")
        _poll_until(widget, lambda: widget._budget is not None)
        widget._bridge.cancel()
        _poll_until(widget, lambda: widget._budget is None and done.is_set())
        assert widget._budget_box.isHidden() is True
        assert "中断されました" in widget._view.toPlainText()

    def test_stopping_without_asking_tells_in_the_log(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、尋ねずに止める設定の人には、なぜ AI が途中でやめたのかが分からない
        widget, _ = panel
        _over_tokens(widget, Limits(turn_tokens=100, action=LimitAction.STOP))
        done, box = _call_in_thread(widget, "get_selection")
        _poll_until(widget, done.is_set)
        widget._poll()
        assert widget._budget_box.isHidden() is True
        assert isinstance(box.get("error"), ToolError)
        assert "1 回の指示のトークンが上限" in widget._view.toPlainText()


class TestShowingUsage:
    def test_the_line_shows_this_instruction_against_its_limits(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、どれだけ使ったかが見えず、上限に当たるまで重い指示だと気付けない
        widget, _ = panel
        widget._bridge.usage.limits = Limits(turn_tokens=1_000_000, turn_images=30)
        widget._begin_turn()
        widget._bridge.usage.record_message("m", {"output_tokens": 32_000})
        widget._bridge.usage.record_image()
        widget._refresh_status()
        line = widget._usage_text.text()
        assert line.startswith("この指示:")
        assert "3.2 万 / 100 万 トークン" in line
        assert "絵 1 / 30 枚" in line
        assert "今日:" in widget._usage_text.toolTip()

    def test_idle_shows_today_and_no_limit_has_no_slash(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、上限なしの人に「/ 0」と出て、何も使えないように読める
        widget, _ = panel
        widget._bridge.usage.limits = Limits(day_tokens=0, day_images=0)
        widget._refresh_status()
        assert widget._usage_text.text() == "今日: 0 トークン・絵 0 枚"

    def test_the_divider_keeps_what_the_instruction_used(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、終わった後にどの指示が重かったかを見返せない
        # 送る所は通さない（AI の部品の無い機械では送らずに案内を出すので、指示が始まらない）
        widget, _ = panel
        widget._queued.append("切って")
        widget._begin_turn()
        widget._bridge.usage.record_message("m", {"output_tokens": 12_345})
        widget._bridge.usage.record_image()
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert "1.2 万 トークン・絵 1 枚" in widget._view.toPlainText()

    def test_time_waiting_for_the_parts_update_is_not_counted(
        self, panel: tuple[ChatPanel, FakeHost], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、AI の部品の更新中に送った指示が、更新を待つ間だけで 1 回の時間の上限に
        # 当たり、AI が指示を受け取った直後の最初の道具で止まる
        widget, _ = panel
        now = [100.0]
        widget._bridge.usage._clock = lambda: now[0]
        dispatched: list[str] = []
        # 会話は作らない 部品の無い機械でも試験できるように、渡した指示だけを記録する
        monkeypatch.setattr(widget, "_dispatch", dispatched.append)
        widget._retrying = True
        widget._queued.append("切って")
        widget._begin_turn()
        assert widget._bridge.usage.turn_usage() is None
        now[0] += 600.0
        widget._retrying = False
        widget._pass_held()
        assert dispatched == ["切って"]
        now[0] += 5.0
        turn = widget._bridge.usage.turn_usage()
        assert turn is not None
        assert turn.seconds == pytest.approx(5.0)

    def test_the_day_is_written_to_the_settings_folder(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、アプリを閉じるたびに 1 日の量が消え、1 日の上限が効かない
        widget, _ = panel
        widget._begin_turn()
        widget._bridge.usage.record_message("m", {"output_tokens": 77})
        widget._finish_turn()
        stored = json.loads((config_root() / USAGE_FILE).read_text(encoding="utf-8"))
        assert [day["tokens"] for day in stored.values()] == [77]

    def test_closing_while_answering_keeps_what_was_used(
        self, qt_application: QApplication, loaded: Project
    ) -> None:
        # 壊れると、AI が応えている途中で窓を閉じた指示の量が 1 日の置き場に残らず、
        # 起動し直すとその分だけ 1 日の上限を越えて使える
        del qt_application
        widget = ChatPanel(FakeHost(loaded))
        try:
            widget._queued.append("切って")
            widget._begin_turn()
            widget._bridge.usage.record_message("m", {"output_tokens": 55})
            widget._bridge.usage.record_image()
            widget.close_session()
        finally:
            widget.deleteLater()
        stored = json.loads((config_root() / USAGE_FILE).read_text(encoding="utf-8"))
        assert [(day["tokens"], day["images"]) for day in stored.values()] == [(55, 1)]


class TestPreferences:
    def test_the_panel_uses_the_limits_from_the_settings(
        self, panel: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 壊れると、設定画面で変えた上限が AI に効かず、既定の量で止まり続ける
        widget, _ = panel
        chosen = Preferences(
            ai_limit_turn_tokens=50_000,
            ai_limit_turn_images=3,
            ai_limit_turn_minutes=7,
            ai_limit_day_tokens=0,
            ai_limit_day_images=9,
            ai_limit_action=LimitAction.STOP.value,
            ai_image_edge=1024,
        )
        widget.apply_preferences(chosen)
        assert widget._bridge.usage.limits == Limits(
            turn_tokens=50_000,
            turn_images=3,
            turn_seconds=420,
            day_tokens=0,
            day_images=9,
            action=LimitAction.STOP,
            image_edge=1024,
        )

    def test_the_dialog_round_trips_the_limits(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面を開いて OK を押しただけで、選んだ上限が既定へ戻る
        del qt_application
        chosen = Preferences(
            ai_limit_turn_tokens=300_000,
            ai_limit_turn_images=0,
            ai_limit_turn_minutes=90,
            ai_limit_day_tokens=2_500_000,
            ai_limit_day_images=40,
            ai_limit_action=LimitAction.STOP.value,
            ai_image_edge=768,
        )
        dialog = PreferencesDialog(chosen)
        assert dialog._limit_turn_images.text() == "上限なし"
        assert dialog.preferences() == chosen

    def test_an_untouched_odd_token_limit_is_kept(self, qt_application: QApplication) -> None:
        # 壊れると、手で書いた万で割り切れない上限が、設定を開いて OK を押しただけで変わる
        del qt_application
        dialog = PreferencesDialog(Preferences(ai_limit_turn_tokens=123_456))
        assert dialog.preferences().ai_limit_turn_tokens == 123_456
        dialog._limit_turn_tokens.setValue(20)
        assert dialog.preferences().ai_limit_turn_tokens == 200_000

    def test_broken_values_go_back_to_the_defaults(self) -> None:
        # 壊れると、手で書き換えた設定ファイル 1 つで AI が止まらなくなるか、起動で落ちる
        store = PreferenceStore()
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            json.dumps(
                {
                    "ai_limit_turn_tokens": -5,
                    "ai_limit_turn_images": "多い",
                    "ai_limit_turn_minutes": True,
                    "ai_limit_action": "explode",
                    "ai_image_edge": 999,
                }
            ),
            encoding="utf-8",
        )
        loaded = store.load()
        plain = Preferences()
        assert loaded.ai_limit_turn_tokens == DEFAULT_TURN_TOKENS
        assert loaded.ai_limit_turn_images == plain.ai_limit_turn_images
        assert loaded.ai_limit_turn_minutes == plain.ai_limit_turn_minutes
        assert loaded.ai_limit_action == LimitAction.ASK.value
        assert loaded.ai_image_edge == DEFAULT_IMAGE_EDGE

    def test_the_window_hands_the_image_edge_to_the_tools(
        self, qt_application: QApplication, loaded: Project
    ) -> None:
        # 壊れると、設定で絵を小さくしても、本物の窓の AI は 1568 画素の絵を送り続ける
        del qt_application
        PreferenceStore().save(Preferences(ai_image_edge=1024))
        window = MainWindow(loaded, confirm_unsaved=False)
        try:
            assert window.image_edge == 1024
        finally:
            window.close()
            window.deleteLater()


def test_every_limit_has_a_label() -> None:
    # 壊れると、箱と会話の欄に「None が上限に達しました」と出て、どの上限か分からない
    for kind in LimitKind:
        assert kind.label
