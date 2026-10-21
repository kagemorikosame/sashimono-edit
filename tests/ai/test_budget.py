"""AI が使う量の上限（#344）

数え方（:class:`UsageMeter`）と 1 日の量の置き場（:class:`DailyUsageStore`）、ブリッジが
道具を動かす前に止めること、絵の長い辺の上限が絵を返す道具に効くことを見る
画面（尋ねる箱・使った量の行）は tests/ui/test_ai_budget.py が見る
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from sashimono.ai.bridge import BudgetRequest, EditorBridge
from sashimono.ai.budget import (
    KEEP_DAYS,
    DailyUsageStore,
    LimitAction,
    LimitKind,
    Limits,
    Usage,
    UsageMeter,
    format_tokens,
)
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, find_operation
from sashimono.ai.sheet import MAX_SHEET_EDGE
from sashimono.core.model import Project
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread, pump_until


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


#: 道具が返るのを待つ上限（秒） 返ればすぐ戻る 並べて走らせた CI でスレッドの切り替えが
#: 遅れても、待ちの長さで落ちないように取る（結果を見る試験で、速さは見ない）
WAIT = 60.0

PREVIEW = _operation("preview_frame")
SHEET = _operation("contact_sheet")
READ = _operation("get_selection")


class _Clock:
    """進めた分だけ進む時計 本物の時計だと秒の境目で揺れる"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Today:
    def __init__(self) -> None:
        self.day = date(2026, 10, 19)

    def __call__(self) -> date:
        return self.day


def _meter(limits: Limits | None = None, store: DailyUsageStore | None = None) -> UsageMeter:
    meter = UsageMeter(store)
    if limits is not None:
        meter.limits = limits
    return meter


class TestCounting:
    def test_split_messages_with_one_id_are_counted_once(self) -> None:
        # 壊れると、1 つの応答が中身の塊ごとに何通にも分かれて届くたびに足され、
        # 使った量が 2〜3 倍に見えて、上限の半分も使わないうちに止まる
        meter = _meter()
        meter.begin_turn()
        usage = {"input_tokens": 100, "cache_creation_input_tokens": 400, "output_tokens": 30}
        meter.record_message("msg_1", usage)
        meter.record_message("msg_1", {**usage, "output_tokens": 50})
        meter.record_message("msg_1", {**usage, "output_tokens": 50})
        turn = meter.turn_usage()
        assert turn is not None
        assert turn.tokens == 100 + 400 + 50

    def test_cache_reads_are_not_counted(self) -> None:
        # 壊れると、API を呼ぶたびに読み直す会話の全部（値段は 1/10）が数に入り、
        # 道具を 20 回使っただけで数十万になって、ふつうの指示が上限で止まる
        meter = _meter()
        meter.begin_turn()
        meter.record_message("m", {"input_tokens": 10, "cache_read_input_tokens": 50_000})
        turn = meter.turn_usage()
        assert turn is not None
        assert turn.tokens == 10

    def test_each_api_call_adds_up(self) -> None:
        # 壊れると、2 回目からの API の呼び出しが数に入らず、長い指示が上限を素通りする
        meter = _meter()
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 5})
        meter.record_message("b", {"output_tokens": 7})
        meter.record_message(None, {"output_tokens": 11})
        turn = meter.turn_usage()
        assert turn is not None
        assert turn.tokens == 23

    def test_counts_start_over_with_each_instruction(self) -> None:
        # 壊れると、前の指示で使った量のせいで、次の短い指示が始めから上限で止まる
        meter = _meter(Limits(turn_tokens=100))
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 100})
        assert meter.overrun() is not None
        assert meter.end_turn().tokens == 100
        meter.begin_turn()
        assert meter.overrun() is None

    def test_what_arrives_between_instructions_counts_for_the_day_only(self) -> None:
        # 壊れると、指示の区切りの間に届いた知らせが 1 日の量にも入らずに消えるか、
        # 次の指示の量に混ざって、短い指示が前の量で 1 回の上限に当たる
        meter = _meter(Limits(turn_tokens=10, day_tokens=0))
        meter.record_message("a", {"output_tokens": 9})
        meter.record_image()
        assert meter.day_usage().tokens == 9
        # 指示の外で置き場へ書いた後も、今日の量から抜けない
        meter.flush()
        assert (meter.day_usage().tokens, meter.day_usage().images) == (9, 1)
        meter.begin_turn()
        meter.record_message("b", {"output_tokens": 9})
        assert meter.overrun() is None
        assert meter.end_turn().tokens == 9
        assert (meter.day_usage().tokens, meter.day_usage().images) == (18, 1)

    def test_tokens_are_shown_in_man(self) -> None:
        # 壊れると、AI のパネルに 1234567 のような桁の多い数が並び、どれだけ使ったか掴めない
        assert format_tokens(8_500) == "8,500"
        assert format_tokens(32_000) == "3.2 万"
        assert format_tokens(1_000_000) == "100 万"


class TestLimits:
    def test_the_instruction_token_limit_stops(self) -> None:
        # 壊れると、同じ道具を繰り返して止まらなくなった指示が、人が気付くまで使い続ける
        meter = _meter(Limits(turn_tokens=1000))
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 999})
        assert meter.overrun() is None
        meter.record_message("b", {"output_tokens": 1})
        overrun = meter.overrun()
        assert overrun is not None
        assert overrun.kind is LimitKind.TURN_TOKENS
        assert "1 回の指示のトークン" in overrun.text()

    def test_zero_means_no_limit(self) -> None:
        # 壊れると、設定で「上限なし」にした人の AI が、1 回目の道具で止まる
        meter = _meter(
            Limits(turn_tokens=0, turn_images=0, turn_seconds=0, day_tokens=0, day_images=0)
        )
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 10**9})
        for _ in range(100):
            meter.record_image()
        assert meter.overrun(adding_images=1) is None

    def test_the_image_that_would_cross_the_limit_is_stopped_before_drawing(self) -> None:
        # 壊れると、上限の 1 枚先まで描いて送ってから止まり、設定より 1 枚多く送る
        meter = _meter(Limits(turn_images=2))
        meter.begin_turn()
        meter.record_image()
        assert meter.overrun(adding_images=1) is None
        meter.record_image()
        overrun = meter.overrun(adding_images=1)
        assert overrun is not None
        assert overrun.kind is LimitKind.TURN_IMAGES
        # 絵を送らない道具は止めない 止めると、編集の後片付けもできなくなる
        assert meter.overrun() is None

    def test_the_time_limit_uses_the_time_since_the_instruction_began(self) -> None:
        # 壊れると、席を外している間に AI が何十分も作業を続ける
        clock = _Clock()
        meter = UsageMeter(clock=clock)
        meter.limits = Limits(turn_seconds=60)
        meter.begin_turn()
        clock.now += 59
        assert meter.overrun() is None
        clock.now += 1
        overrun = meter.overrun()
        assert overrun is not None
        assert overrun.kind is LimitKind.TURN_SECONDS

    def test_continuing_widens_by_the_set_amount_and_asks_again(self) -> None:
        # 壊れると、〔続ける〕を 1 度押しただけで上限が消え、止まらなくなった指示を
        # もう止められない
        meter = _meter(Limits(turn_tokens=100))
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 100})
        first = meter.overrun()
        assert first is not None
        assert first.extended() == "200 トークン"
        meter.extend(first.kind)
        assert meter.overrun() is None
        meter.record_message("b", {"output_tokens": 100})
        again = meter.overrun()
        assert again is not None
        assert again.limit == 200

    def test_a_widened_daily_limit_lasts_the_day(self) -> None:
        # 壊れると、1 日の上限を越えて〔続ける〕と答えた人が、指示を送るたびに尋ねられる
        today = _Today()
        meter = UsageMeter(today=today)
        meter.limits = Limits(day_tokens=100, turn_tokens=0)
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 100})
        overrun = meter.overrun()
        assert overrun is not None
        assert overrun.kind is LimitKind.DAY_TOKENS
        meter.extend(overrun.kind)
        meter.end_turn()
        meter.begin_turn()
        assert meter.overrun() is None
        meter.end_turn()
        # 次の日には広げた分を持ち越さない 持ち越すと、上限が日ごとに育つ
        today.day += timedelta(days=1)
        meter.begin_turn()
        meter.record_message("b", {"output_tokens": 100})
        assert meter.overrun() is not None

    def test_a_stopped_instruction_stays_stopped(self) -> None:
        # 壊れると、止めると答えた後の道具で、また「続けますか」と尋ねられる
        meter = _meter(Limits(turn_tokens=10))
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 10})
        overrun = meter.overrun()
        assert overrun is not None
        meter.stop(overrun)
        meter.limits = Limits(turn_tokens=0)
        assert meter.overrun() == overrun
        meter.end_turn()
        assert meter.last_stop == overrun


class TestDailyStore:
    def test_the_day_survives_closing_the_app(self, tmp_path: Path) -> None:
        # 壊れると、アプリを開き直すたびに 1 日の量が 0 に戻り、1 日の上限が効かない
        path = tmp_path / "ai_usage.json"
        today = _Today()
        first = UsageMeter(DailyUsageStore(path), today=today)
        first.begin_turn()
        first.record_message("a", {"output_tokens": 70})
        first.record_image()
        first.end_turn()
        second = UsageMeter(DailyUsageStore(path), today=today)
        second.limits = Limits(day_tokens=100, turn_tokens=0)
        assert second.day_usage().tokens == 70
        second.begin_turn()
        second.record_message("b", {"output_tokens": 30})
        overrun = second.overrun()
        assert overrun is not None
        assert overrun.kind is LimitKind.DAY_TOKENS

    def test_two_windows_add_up_instead_of_overwriting(self, tmp_path: Path) -> None:
        # 壊れると、2 つの窓が自分の合計を書き合い、後から書いた窓が先の窓の分を消す
        path = tmp_path / "ai_usage.json"
        today = _Today()
        one = UsageMeter(DailyUsageStore(path), today=today)
        two = UsageMeter(DailyUsageStore(path), today=today)
        one.begin_turn()
        two.begin_turn()
        one.record_message("a", {"output_tokens": 5})
        two.record_message("b", {"output_tokens": 7})
        one.end_turn()
        two.end_turn()
        assert DailyUsageStore(path).load(today.day).tokens == 12

    def test_two_windows_at_once_share_the_daily_limit(self, tmp_path: Path) -> None:
        # 壊れると、2 つの窓で同時に指示したとき、それぞれが始めた時の量で確かめ、
        # 1 日の上限が 100 なら合わせて 200 まで使える
        path = tmp_path / "ai_usage.json"
        today = _Today()
        one = UsageMeter(DailyUsageStore(path), today=today)
        two = UsageMeter(DailyUsageStore(path), today=today)
        for meter in (one, two):
            meter.limits = Limits(turn_tokens=0, day_tokens=100)
            meter.begin_turn()
        one.record_message("a", {"output_tokens": 60})
        assert one.overrun() is None
        two.record_message("b", {"output_tokens": 60})
        overrun = two.overrun()
        assert overrun is not None
        assert overrun.kind is LimitKind.DAY_TOKENS
        assert overrun.used == 120
        # 先に確かめた窓も、次の道具の前にはもう一方の分を読む
        assert one.overrun() is not None
        # 確かめるたびに足しても、終わりに 2 重に足さない
        one.end_turn()
        two.end_turn()
        assert DailyUsageStore(path).load(today.day).tokens == 120

    def test_what_was_used_is_saved_before_the_instruction_ends(self, tmp_path: Path) -> None:
        # 壊れると、指示の途中でアプリが落ちたとき、それまでに使った量が 1 日の置き場に残らず、
        # 起動し直すとその分だけ 1 日の上限を越えて使える
        path = tmp_path / "ai_usage.json"
        today = _Today()
        meter = UsageMeter(DailyUsageStore(path), today=today)
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 40})
        meter.record_image()
        meter.flush()
        assert DailyUsageStore(path).load(today.day).tokens == 40
        meter.record_message("b", {"output_tokens": 2})
        meter.overrun()
        stored = DailyUsageStore(path).load(today.day)
        assert (stored.tokens, stored.images) == (42, 1)

    def test_what_could_not_be_written_is_written_later(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、錠やファイルが一時的に開けなかった差を足し終えた扱いにして二度と足さず、
        # 1 日の量を少なく数えて、もう 1 つの窓や起動し直した後に上限を越えて使える
        from sashimono.ai import budget as budget_module

        path = tmp_path / "ai_usage.json"
        today = _Today()
        meter = UsageMeter(DailyUsageStore(path), today=today)
        meter.limits = Limits(turn_tokens=0, day_tokens=100)
        meter.begin_turn()
        from sashimono.core.io.atomic import write_atomic as real

        def refuse(*_arguments: object, **_options: object) -> None:
            raise OSError("書けない")

        monkeypatch.setattr(budget_module, "write_atomic", refuse)
        meter.record_message("a", {"output_tokens": 70})
        meter.flush()
        meter.record_message("b", {"output_tokens": 30})
        # 書けない間も、この指示の分を 1 日の量に入れて確かめる
        assert meter.overrun() is not None
        assert meter.end_turn().tokens == 100
        assert DailyUsageStore(path).load(today.day) == Usage()
        # 書けないまま次の指示へ進んでも、前の指示の量は次の指示の量に入らない
        meter.limits = Limits(turn_tokens=50, day_tokens=0)
        meter.begin_turn()
        fresh = meter.turn_usage()
        assert fresh is not None
        assert fresh.tokens == 0
        assert meter.overrun() is None
        assert meter.day_usage().tokens == 100
        meter.end_turn()
        # 書けなかった分は、その日の分として持ち続ける 日が変わっても次の日へ移さない
        today.day += timedelta(days=1)
        monkeypatch.setattr(budget_module, "write_atomic", real)
        meter.begin_turn()
        meter.end_turn()
        assert DailyUsageStore(path).load(today.day - timedelta(days=1)).tokens == 100
        assert DailyUsageStore(path).load(today.day) == Usage()

    def test_an_instruction_across_midnight_counts_for_each_day(self, tmp_path: Path) -> None:
        # 壊れると、0 時をまたいだ指示が昨日の量で 1 日の上限に当たり、今日の分まで昨日へ書かれる
        path = tmp_path / "ai_usage.json"
        today = _Today()
        meter = UsageMeter(DailyUsageStore(path), today=today)
        meter.limits = Limits(turn_tokens=0, day_tokens=100)
        meter.begin_turn()
        meter.record_message("a", {"output_tokens": 80})
        assert meter.overrun() is None
        today.day += timedelta(days=1)
        meter.record_message("b", {"output_tokens": 30})
        assert meter.overrun() is None
        assert meter.day_usage().tokens == 30
        meter.end_turn()
        store = DailyUsageStore(path)
        assert store.load(today.day - timedelta(days=1)).tokens == 80
        assert store.load(today.day).tokens == 30

    def test_a_new_day_starts_from_zero(self, tmp_path: Path) -> None:
        # 壊れると、昨日の量で今日の最初の指示が止まる
        store = DailyUsageStore(tmp_path / "ai_usage.json")
        day = date(2026, 10, 19)
        store.add(day, Usage(tokens=500))
        assert store.load(day + timedelta(days=1)) == Usage()

    def test_old_days_are_dropped(self, tmp_path: Path) -> None:
        # 壊れると、毎日 1 行ずつファイルが育ち続ける
        path = tmp_path / "ai_usage.json"
        store = DailyUsageStore(path)
        day = date(2026, 10, 19)
        store.add(day - timedelta(days=KEEP_DAYS + 5), Usage(tokens=1))
        store.add(day, Usage(tokens=2))
        kept = json.loads(path.read_text(encoding="utf-8"))
        assert list(kept) == [day.isoformat()]

    def test_a_broken_file_counts_from_zero_and_is_rewritten(self, tmp_path: Path) -> None:
        # 壊れると、手で書き換えて壊したファイル 1 つで AI の作業が始められなくなる
        path = tmp_path / "ai_usage.json"
        path.write_text("{壊れた", encoding="utf-8")
        store = DailyUsageStore(path)
        day = date(2026, 10, 19)
        assert store.load(day) == Usage()
        path.write_text(json.dumps({day.isoformat(): {"tokens": "多い", "images": -3}}), "utf-8")
        assert store.load(day) == Usage()
        store.add(day, Usage(tokens=4))
        assert store.load(day).tokens == 4


def _wait_for_request(bridge: EditorBridge, timeout: float = WAIT) -> BudgetRequest:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bridge.pump()
        request = bridge.take_budget_request()
        if request is not None:
            return request
        time.sleep(0.005)
    pytest.fail("「続けますか」が来なかった")


def _bridge(host: FakeHost, limits: Limits) -> EditorBridge:
    bridge = allow_sending(EditorBridge(host))
    bridge.usage.limits = limits
    bridge.begin_turn()
    return bridge


class TestBridge:
    def test_a_picture_over_the_limit_is_not_drawn(self, host: FakeHost) -> None:
        # 壊れると、上限を越えた絵も描いて送り、設定した枚数を守らない
        bridge = _bridge(host, Limits(turn_images=1, action=LimitAction.STOP))
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 0})
        pump_until(bridge, done, WAIT)
        assert "result" in box
        sent = bridge.usage.turn_usage()
        assert sent is not None
        assert sent.images == 1
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 1})
        pump_until(bridge, done, WAIT)
        error = box.get("error")
        assert isinstance(error, ToolError)
        assert "上限" in str(error)
        assert "これ以上道具を使わず" in str(error)
        assert host.rendered == [(0, 640)]
        notice = bridge.take_budget_notice()
        assert notice is not None
        assert notice.kind is LimitKind.TURN_IMAGES

    def test_after_stopping_every_tool_is_refused(self, host: FakeHost) -> None:
        # 壊れると、上限で止めた後も絵を使わない道具は動き続け、指示が終わらない
        bridge = _bridge(host, Limits(turn_tokens=10, action=LimitAction.STOP))
        bridge.usage.record_message("a", {"output_tokens": 10})
        for _ in range(2):
            done, box = call_in_thread(bridge, READ, {})
            pump_until(bridge, done, WAIT)
            assert isinstance(box.get("error"), ToolError)
        # 断った訳を返しても呼び続けたら、画面に会話を止めさせる
        assert bridge.wants_halt

    def test_asking_waits_and_continuing_runs_the_tool(self, host: FakeHost) -> None:
        # 壊れると、尋ねる箱を出す前に道具が動くか、〔続ける〕を押しても動かない
        bridge = _bridge(host, Limits(turn_images=1))
        bridge.usage.record_image()
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 3})
        request = _wait_for_request(bridge)
        assert not done.is_set()
        assert host.rendered == []
        assert request.overrun.kind is LimitKind.TURN_IMAGES
        assert request.reply(True)
        pump_until(bridge, done, WAIT)
        assert "result" in box
        assert host.rendered == [(3, 640)]
        # 1 枚分だけ広がった 次の絵ではまた尋ねる
        overrun = bridge.usage.overrun(adding_images=1)
        assert overrun is not None
        assert overrun.limit == 2

    def test_stopping_refuses_and_is_not_asked_again(self, host: FakeHost) -> None:
        # 壊れると、〔ここで止める〕の後の道具ごとに、また同じ箱が出る
        bridge = _bridge(host, Limits(turn_tokens=10))
        bridge.usage.record_message("a", {"output_tokens": 10})
        done, box = call_in_thread(bridge, READ, {})
        request = _wait_for_request(bridge)
        assert request.reply(False)
        pump_until(bridge, done, WAIT)
        assert isinstance(box.get("error"), ToolError)
        done, box = call_in_thread(bridge, READ, {})
        pump_until(bridge, done, WAIT)
        assert isinstance(box.get("error"), ToolError)
        assert bridge.take_budget_request() is None

    def test_interrupting_folds_the_question(self, host: FakeHost) -> None:
        # 壊れると、中断を押しても AI のスレッドが返事を 10 分待ち続ける
        bridge = _bridge(host, Limits(turn_tokens=10))
        bridge.usage.record_message("a", {"output_tokens": 10})
        done, box = call_in_thread(bridge, READ, {})
        request = _wait_for_request(bridge)
        bridge.cancel()
        assert done.wait(5.0)
        assert request.reason == "中断されました"
        assert isinstance(box.get("error"), ToolError)

    def test_the_limit_is_checked_before_asking_to_send(self, host: FakeHost) -> None:
        # 壊れると、上限で動かない道具のために「絵を送ってよいか」の窓が先に出る
        bridge = EditorBridge(host)
        bridge.usage.limits = Limits(turn_images=1, action=LimitAction.STOP)
        bridge.begin_turn()
        bridge.usage.record_image()
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 0})
        pump_until(bridge, done, WAIT)
        assert isinstance(box.get("error"), ToolError)
        assert bridge.take_send_request() is None

    def test_pictures_are_counted_only_when_sent(self, host: FakeHost) -> None:
        # 壊れると、描けずに断った絵も数えて、まだ送っていないのに上限で止まる
        bridge = _bridge(host, Limits())
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 0})
        pump_until(bridge, done, WAIT)
        done, box = call_in_thread(bridge, SHEET, {"count": 37})
        pump_until(bridge, done, WAIT)
        assert isinstance(box.get("error"), ToolError)
        turn = bridge.usage.turn_usage()
        assert turn is not None
        assert turn.images == 1

    def test_without_an_instruction_nothing_is_stopped(self, host: FakeHost) -> None:
        # 壊れると、指示の外から道具を呼ぶ所（試験・CLI）が、数え始めていないのに止まる
        bridge = allow_sending(EditorBridge(host))
        bridge.usage.limits = Limits(turn_images=0, day_images=0, turn_tokens=1, day_tokens=1)
        bridge.usage.record_message("a", {"output_tokens": 50})
        done, box = call_in_thread(bridge, READ, {})
        pump_until(bridge, done, WAIT)
        assert "result" in box


class TestImageEdge:
    def test_preview_frame_fits_the_long_side_of_a_tall_project(self, loaded: Project) -> None:
        # 壊れると、縦長の作品で横 640 の絵が縦 1138 になり、設定で小さくした上限を越えて送る
        tall = replace(loaded, settings=replace(loaded.settings, width=1080, height=1920))
        host = FakeHost(tall)
        host.image_edge = 768
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 0})
        pump_until(bridge, done, WAIT)
        assert host.rendered == [(0, 768 * 1080 // 1920)]
        caption = box["result"].caption  # type: ignore[attr-defined]
        assert "768" in caption

    def test_preview_frame_is_untouched_within_the_edge(self, host: FakeHost) -> None:
        # 壊れると、上限の内の絵まで縮めて送り、細かい所が見えなくなる
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, PREVIEW, {"frame": 0, "width": 1280})
        pump_until(bridge, done, WAIT)
        assert host.rendered == [(0, 1280)]
        assert "縮め" not in box["result"].caption  # type: ignore[attr-defined]

    def test_contact_sheet_uses_the_set_edge(self, host: FakeHost) -> None:
        # 壊れると、長い辺を小さくした人にも 1568 画素のコマの 1 枚を送る
        host.image_edge = 1024
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, SHEET, {"count": 12, "cell_width": 640})
        pump_until(bridge, done, WAIT)
        assert "result" in box
        layout = host.sheets[0].layout
        assert max(layout.width, layout.height) <= 1024
        host.image_edge = MAX_SHEET_EDGE
        done, box = call_in_thread(bridge, SHEET, {"count": 12, "cell_width": 640})
        pump_until(bridge, done, WAIT)
        wide = host.sheets[1].layout
        assert max(wide.width, wide.height) > 1024


def test_the_session_counts_what_the_sdk_reports(host: FakeHost) -> None:
    # 壊れると、SDK の知らせを読まずに捨て、トークンの上限がいつまでも効かない
    sdk = pytest.importorskip("claude_agent_sdk", reason="AI 連携の実行環境が未導入")
    from sashimono.ai.session import AgentSession

    bridge = EditorBridge(host)
    bridge.begin_turn()
    session = AgentSession(bridge)
    usage = {"input_tokens": 3, "cache_read_input_tokens": 900, "output_tokens": 4}
    for _ in range(2):
        session._translate(
            sdk.AssistantMessage(
                content=[sdk.TextBlock(text="切ります")],
                model="claude",
                usage=usage,
                message_id="msg_1",
            )
        )
    turn = bridge.usage.turn_usage()
    assert turn is not None
    assert turn.tokens == 7


@pytest.mark.parametrize("is_error", [False, True])
def test_the_output_is_settled_by_the_result(host: FakeHost, is_error: bool) -> None:
    # 壊れると、途中の知らせの仮の出力（応答の始めの数トークン）だけを数え、長い返事を書いた
    # 指示でも出力がほぼ 0 になり、1 回と 1 日のトークンの上限が遅れて効く
    # 値は claude-agent-sdk 0.2.164 で実際に流れてきた形（途中の知らせは 6 結果は 332）
    sdk = pytest.importorskip("claude_agent_sdk", reason="AI 連携の実行環境が未導入")
    from sashimono.ai.session import AgentSession, EventKind

    bridge = EditorBridge(host)
    bridge.begin_turn()
    session = AgentSession(bridge)
    for _ in range(2):
        session._translate(
            sdk.AssistantMessage(
                content=[sdk.TextBlock(text="切ります")],
                model="claude",
                usage={"input_tokens": 3455, "output_tokens": 6},
                message_id="msg_1",
            )
        )
    result = sdk.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s",
        usage={"input_tokens": 3455, "output_tokens": 332},
        result="失敗" if is_error else "済み",
    )
    # 同じ結果を 2 度受けても 2 重に足さない
    session._translate(result)
    session._translate(result)
    turn = bridge.usage.turn_usage()
    assert turn is not None
    assert turn.tokens == 3455 + 332
    kinds = []
    while not session._events.empty():
        kinds.append(session._events.get_nowait().kind)
    # 失敗の知らせは今のまま出す
    assert (EventKind.ERROR in kinds) is is_error


def test_a_result_never_lowers_the_count() -> None:
    # 壊れると、結果の出力が途中の知らせの合計より小さいとき（結果の欄が無い古い CLI など）、
    # 数えた量を減らして上限を越えて使える
    meter = _meter()
    meter.begin_turn()
    meter.record_message("a", {"output_tokens": 50})
    meter.settle_output({"output_tokens": 20})
    meter.settle_output(None)
    turn = meter.turn_usage()
    assert turn is not None
    assert turn.tokens == 50


def test_the_lock_keeps_threads_from_losing_counts() -> None:
    # 壊れると、AI のスレッドが数える間に画面が指示を締め、数えた分が 1 日の量から消える
    meter = _meter()
    meter.begin_turn()

    def add() -> None:
        for index in range(1000):
            meter.record_message(f"{threading.get_ident()}-{index}", {"output_tokens": 1})

    threads = [threading.Thread(target=add) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert meter.end_turn().tokens == 4000
