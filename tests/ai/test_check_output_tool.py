"""AI の道具 check_output の引数・待ち方・返事の形（#337）

検査そのものは試験しない（tests/engine/test_check_*.py と tests/ui/test_check_output.py）
ここでは偽のホストに検査の様子を入れて、道具が頼む物と返す物を見る
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Pending, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.commands import RemoveClip
from sashimono.core.model import SceneId
from sashimono.engine.checks import CheckJob, CheckLimits, CheckRequest, CheckState
from sashimono.engine.checks.job import _Progress
from sashimono.engine.checks.report import assemble
from sashimono.engine.checks.scan import BLACK, Run
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.waiting import GUARD_SECONDS


def _operation() -> Any:
    operation = find_operation("check_output")
    assert operation is not None
    return operation


def _state(
    request: CheckRequest,
    runs: list[Run] | None = None,
    *,
    running: bool = False,
    stopped: bool = False,
    failure: str | None = None,
    analyzed: int | None = None,
) -> CheckState:
    end = request.end or 300
    report = assemble(
        runs or [],
        start=request.start,
        end=end,
        analyzed=end if analyzed is None else analyzed,
        fps=30.0,
        complete=not running and not stopped,
        limits=request.limits,
        index=None,
        loudness=-14.2,
        peak=0.5,
        length_seconds=None,
    )
    return CheckState(
        request=request,
        report=report,
        running=running,
        stopped=stopped,
        failure=failure,
    )


def _finished(host: FakeHost, arguments: dict[str, Any], runs: list[Run] | None = None) -> Any:
    """1 度呼んで頼みを取り、その頼みで終わった様子を入れて、もう 1 度呼ぶ"""
    _operation()(host, {**arguments, "wait_seconds": 0})
    request = host.check_requests[-1]
    host.check = _state(request, runs)
    return _operation()(host, {**arguments, "wait_seconds": 0})


class TestKind:
    def test_it_only_reads(self) -> None:
        # 壊れると、調べるだけの道具に確かめの窓が出るか、取り消しの履歴に何かが載る
        assert _operation().safety is Safety.READ_ONLY


class TestRequest:
    def test_the_defaults_check_the_whole_timeline_with_the_preference_target(
        self, host: FakeHost
    ) -> None:
        # 壊れると、AI が範囲を言わないと一部しか調べないか、本人の選んだ音量の基準を使わない
        host.loudness = -23.0
        _operation()(host, {"wait_seconds": 0})
        (request,) = host.check_requests
        assert request.project == host.project
        assert (request.start, request.end) == (0, host.project.duration)
        assert request.limits.loudness_target == -23.0
        assert request.limits == CheckLimits(loudness_target=-23.0)

    def test_arguments_become_the_limits(self, host: FakeHost) -> None:
        # 壊れると、AI が渡したしきい値が効かず、頼んだ物と違う基準で調べる
        _operation()(
            host,
            {
                "from_frame": 30,
                "to_frame": 90,
                "checks": ["black", "silence"],
                "black_level": 0.05,
                "silence_db": -50,
                "loudness_target": -16,
                "safe_margin_percent": 5,
                "expected_seconds": 2,
                "clip_samples": 5,
                "wait_seconds": 0,
            },
        )
        (request,) = host.check_requests
        assert (request.start, request.end) == (30, 90)
        limits = request.limits
        assert limits.kinds == frozenset({"black", "silence"})
        assert limits.black_level == 0.05
        assert limits.silence_db == -50
        assert limits.loudness_target == -16
        assert limits.safe_margin == pytest.approx(0.05)
        assert limits.expected_seconds == 2
        assert limits.clip_samples == 5

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"checks": ["black", "ノイズ"]}, "知らない種類"),
            ({"checks": []}, "checks"),
            ({"black_level": 0.9}, "black_level"),
            ({"silence_db": "-60"}, "数で"),
            ({"min_freeze_seconds": float("nan")}, "min_freeze_seconds"),
            ({"clip_samples": 2.5}, "整数"),
            ({"from_frame": 100, "to_frame": 50}, "範囲"),
            ({"to_frame": 100000}, "範囲"),
            ({"from_frame": "0"}, "整数"),
            ({"wait_seconds": 999}, "wait_seconds"),
        ],
    )
    def test_bad_arguments_are_refused_with_the_range(
        self, host: FakeHost, arguments: dict[str, Any], message: str
    ) -> None:
        # 壊れると、範囲の外の値を黙って丸め、AI は頼んだ条件で調べたと思い込む
        with pytest.raises(ToolError, match=message):
            _operation()(host, arguments)
        assert host.check_requests == []

    def test_an_empty_timeline_is_refused(self, host: FakeHost) -> None:
        # 壊れると、何も無い作品を調べて「問題無し」と返す
        clip = host.project.timeline.tracks[0].clips[0]
        host.apply_commands([RemoveClip(clip.id)], "消す")
        with pytest.raises(ToolError, match="空"):
            _operation()(host, {})


class TestFiles:
    def test_a_written_file_in_the_export_folder_can_be_checked(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、AI が書き出した動画を調べられず、書き出した後の確かめができない
        host.exported_to = tmp_path
        written = tmp_path / "完成.mp4"
        written.write_bytes(b"x" * 10)
        _operation()(host, {"path": str(written), "wait_seconds": 0})
        (request,) = host.check_requests
        assert request.path == written.resolve()
        assert request.project is None
        assert request.stamp is not None and request.stamp[0] == 10
        host.check = _state(request)
        result = _operation()(host, {"path": str(written), "wait_seconds": 0})
        assert result["target"] == str(written.resolve())
        assert "frame_note" in result

    @pytest.mark.parametrize("name", ["../外.mp4", "完成.mp4:隠し", "無い.mp4"])
    def test_odd_or_missing_files_are_refused(
        self, host: FakeHost, tmp_path: Path, name: str
    ) -> None:
        # 壊れると、書き出しのフォルダの外や、無いファイルを調べに行って中身の無い結果を返す
        host.exported_to = tmp_path
        with pytest.raises(ToolError):
            _operation()(host, {"path": str(tmp_path / name)})
        assert host.check_requests == []

    def test_a_file_outside_the_folders_is_refused(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、作品と関係の無い動画の中身（長さや音の大きさ）まで AI へ渡る
        allowed = tmp_path / "書き出し"
        allowed.mkdir()
        host.exported_to = allowed
        outside = tmp_path / "よそ.mp4"
        outside.write_bytes(b"x")
        with pytest.raises(ToolError, match="調べられる場所の外"):
            _operation()(host, {"path": str(outside)})

    def test_a_range_is_refused_for_a_file(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、ファイルを調べるのにタイムラインの範囲を黙って捨て、頼んだ範囲と違う所を返す
        host.exported_to = tmp_path
        written = tmp_path / "完成.mp4"
        written.write_bytes(b"x")
        with pytest.raises(ToolError, match="タイムライン"):
            _operation()(host, {"path": str(written), "from_frame": 0})


class TestAnswer:
    def test_a_problem_carries_time_kind_and_degree(self, host: FakeHost) -> None:
        # 壊れると、AI が問題の位置を preview_frame へ渡せず、どれだけ悪いかも分からない
        result = _finished(host, {}, [Run(BLACK, 90, 93, {"seconds": 0.1}, clip_ids=())])
        (row,) = result["problems"]
        assert row["kind"] == "black"
        assert row["name"] == "黒いコマ"
        assert row["severity"] == "medium"
        assert (row["start_frame"], row["end_frame"]) == (90, 93)
        assert row["start_seconds"] == 3.0
        assert row["timecode"] == "00:00:03:00"
        assert row["seconds"] == 0.1
        assert result["complete"] is True
        assert result["summary"]["integrated_lufs"] == -14.2

    def test_no_problems_says_so(self, host: FakeHost) -> None:
        # 壊れると、空の一覧を「まだ調べていない」と読み違え、AI が何度も呼び直す
        result = _finished(host, {})
        assert result["problems"] == []
        assert "問題は見つかりませんでした" in result["note"]

    def test_a_failure_is_an_error_with_the_reason(self, host: FakeHost) -> None:
        # 壊れると、調べられなかったのに空の一覧を返し、AI が問題無しと伝える
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, failure="InvalidDataError: 読めない")
        with pytest.raises(ToolError, match="読めない"):
            _operation()(host, {"wait_seconds": 0})

    def test_a_stopped_check_says_it_was_stopped(self, host: FakeHost) -> None:
        # 壊れると、人が止めた途中までの結果を、全部調べた結果として AI が伝える
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, stopped=True, analyzed=100)
        result = _operation()(host, {"wait_seconds": 0})
        assert result["complete"] is False
        assert "止められました" in result["note"]

    def test_an_edit_during_the_check_is_noted(self, host: FakeHost) -> None:
        # 壊れると、調べている間に変えた作品の結果を、今の作品の結果として返す
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=30)
        pending = _operation()(host, {"wait_seconds": 5})
        assert isinstance(pending, Pending)
        clip = host.project.timeline.tracks[0].clips[0]
        move = find_operation("move_clip")
        assert move is not None
        move(host, {"clip_id": str(clip.id), "timeline_start": 10})
        host.check = _state(request)
        result = pending.then()
        assert isinstance(result, dict)
        assert "作品が変わりました" in result["note"]

    def test_the_scene_is_the_one_the_check_started_in(self, host: FakeHost) -> None:
        # 壊れると、待つ間に人が別のシーンを開くと、メインを調べた結果に開いたシーンの名前が付く
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=30)
        pending = _operation()(host, {"wait_seconds": 5})
        assert isinstance(pending, Pending)
        host.scene = SceneId("あとで開いたシーン")
        host.check = _state(request)
        result = pending.then()
        assert isinstance(result, dict)
        assert result["scene"] is None

    def test_calling_again_after_an_edit_asks_for_a_new_check(self, host: FakeHost) -> None:
        # 壊れると、AI が直した後に呼んでも前の作品の結果が返り、直ったか確かめられない
        _operation()(host, {"wait_seconds": 0})
        first = host.check_requests[-1]
        clip = host.project.timeline.tracks[0].clips[0]
        move = find_operation("move_clip")
        assert move is not None
        move(host, {"clip_id": str(clip.id), "timeline_start": 10})
        _operation()(host, {"wait_seconds": 0})
        assert host.check_requests[-1] != first
        assert host.check_requests[-1].project == host.project


class TestWaiting:
    def test_a_running_check_is_waited_for_on_the_ai_thread(self, host: FakeHost) -> None:
        # 壊れると、画面のスレッドで待って画面が固まるか、待たずに途中の結果だけを返す
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=60)
        pending = _operation()(host, {"wait_seconds": 5})
        assert isinstance(pending, Pending)
        assert pending.seconds == 5
        assert host.check_waits == []
        assert pending.wait(1.0)
        host.check = _state(request)
        result = pending.then()
        assert isinstance(result, dict)
        assert result["complete"] is True

    def test_partial_results_say_how_far_and_how_to_continue(self, host: FakeHost) -> None:
        # 壊れると、AI が途中の結果を全部と読み違えるか、続きの受け取り方が分からない
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=60)
        result = _operation()(host, {"wait_seconds": 0})
        assert result["complete"] is False
        assert "2.0 秒まで" in result["note"]
        assert "同じ引数で呼ぶと続きを返します" in result["note"]

    def test_the_bridge_waits_without_blocking_and_stops_on_cancel(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、待つ間に画面の仕事を拾えず固まるか、中断を押しても最長 2 分待ち続ける

        時間で区切って「1 秒後にも生きているか」を見ると、遅い機械では待ちに入る前に区切りが
        来て揺れる（CI の 3.14 で落ちた #382） 待ちに入ったことを印（Event）で確かめてから止める
        中断が効くまでの上限は、待つ長さ（120 秒）よりずっと短いことだけを見る
        """
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=60)
        entered = threading.Event()
        released = threading.Event()

        def blocking_waiter() -> Any:
            # 本物と同じく、頼まれた秒だけ本当に止まる（終わらない検査） 試験の終わりには放す
            def wait(seconds: float) -> bool:
                entered.set()
                released.wait(seconds)
                return False

            return wait

        monkeypatch.setattr(host, "check_waiter", blocking_waiter)
        bridge = allow_sending(EditorBridge(host))
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["result"] = bridge.invoke(_operation(), {"wait_seconds": 120})
            except ToolError as exc:
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        try:
            # 待ちに入るまで画面の仕事を回し続ける 回さないと道具の頭（画面のスレッドの仕事）が
            # 進まず、待ちに入らない（画面を止めないことの確かめ）
            patience = time.monotonic() + 30.0
            while not entered.is_set() and thread.is_alive() and time.monotonic() < patience:
                bridge.pump()
                thread.join(0.01)
            assert entered.is_set(), f"待ちに入らなかった: {box}"
            # 検査は終わらないので、中断するまで待ち続けている
            assert thread.is_alive(), f"止める前に抜けた: {box}"
            started = time.monotonic()
            bridge.cancel()
            while thread.is_alive() and time.monotonic() - started < 30.0:
                bridge.pump()
                thread.join(0.01)
            # 待つ長さ（120 秒）まで待たずに抜けたこと 上限は遅い機械でも揺れない広さにする
            assert not thread.is_alive(), "中断を押しても待ち続けた"
            assert "中断" in str(box.get("error", ""))
        finally:
            released.set()

    def test_a_worker_holding_the_lock_does_not_outlast_the_wait(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、検査の走り係が鍵を持ったままの間、ブリッジの待ちが渡した秒を越えて
        止まり続け、道具が返らない（#405 で 600 秒返らなかった道の 1 つ）

        本物の係（:class:`CheckJob`）の待つ関数を使い、別のスレッドに鍵を持たせて
        順番を決め打ちにする 鍵は片付けまで放さない その間に途中までの結果が返ればよい
        かかった秒では見ない 遅い機械では、返るのが速くても試験の側が止まって秒が伸びる
        """
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=60)
        job = CheckJob()
        progress = _Progress(request, lambda: None)
        job._progress = progress
        monkeypatch.setattr(host, "check_waiter", job.waiter)
        taken = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with progress.lock:
                taken.set()
                # 上限を付けない 上限で放すと、返ったかを確かめる前に鍵が空き、鍵が空いた後に
                # 返った物を「持たれている間に返った」と読む 放すのは片付け（finally）だけ
                release.wait()

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        try:
            assert taken.wait(GUARD_SECONDS)
            bridge = allow_sending(EditorBridge(host))
            done, box = call_in_thread(bridge, _operation(), {"wait_seconds": 0.5})
            pump_until(bridge, done, timeout=GUARD_SECONDS)
            assert done.is_set(), "鍵を持たれている間、道具が返らなかった"
            # 返ったのは鍵が放される前（鍵を持つスレッドはまだ持っている）
            assert holder.is_alive()
            result = box["result"]
            assert isinstance(result, dict)
            assert result["complete"] is False
        finally:
            release.set()
            holder.join(GUARD_SECONDS)

    def test_a_check_that_ends_while_waiting_returns_through_the_bridge(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると（ブリッジが道具の続きを辿り切らないと）、待つ印がそのまま AI へ返り、
        # 検査の結果の代わりに読めない答えが届く（#320 で続きの辿り方を 1 つにまとめた）
        _operation()(host, {"wait_seconds": 0})
        request = host.check_requests[-1]
        host.check = _state(request, running=True, analyzed=60)

        def finishing_waiter() -> Any:
            def wait(seconds: float) -> bool:
                del seconds
                host.check = _state(request)
                return True

            return wait

        monkeypatch.setattr(host, "check_waiter", finishing_waiter)
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, _operation(), {"wait_seconds": 5})
        pump_until(bridge, done)
        result = box.get("result")
        assert isinstance(result, dict), box.get("error")
        assert result["complete"] is True
