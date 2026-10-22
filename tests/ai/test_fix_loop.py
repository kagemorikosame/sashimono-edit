"""検査で見つけた問題を AI が直して調べ直す繰り返し（#339 C3）

2026-10-20 に本人が決めたこと

- 直すのは今の道具のまま 回数を数えて上限で止めるのはアプリ（check_output とブリッジ）
- 自動で直す範囲は ai/fixes.py の表のとおり
- 直す回ごとに「AI: 検査の直し（n 回目）」の段へ分け、Ctrl+Z 1 回でその回だけを戻せる
- 上限は Preferences.ai_fix_rounds（既定 3 0 は調べるだけ）

偽のホストに検査の結果を入れ、道具とブリッジが数え・区切り・断るかを見る
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.fixes import (
    ASK,
    AUTO,
    FIXES,
    HANDOFF,
    LEAVE,
    PASSES_AFTER_LIMIT,
    fix_for,
    fix_label,
    fixes_closed,
)
from sashimono.ai.host import ToolError
from sashimono.ai.operations import OPERATIONS, Operation, Pending, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import Author, RenameProject
from sashimono.engine.checks import KINDS, CheckRequest, CheckState
from sashimono.engine.checks.report import HIGH, LOW, MEDIUM, assemble
from sashimono.engine.checks.scan import (
    BLACK,
    CLIPPING,
    FREEZE,
    LENGTH,
    LOUDNESS,
    SILENCE,
    TEXT_OVERFLOW,
    Run,
)
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread, pump_until


def _tool(name: str) -> Operation:
    operation = find_operation(name)
    assert operation is not None
    return operation


#: 自動で直す音割れ 混ぜた音が最大を超えた（下げれば消える）
_MIXED_CLIP = Run(CLIPPING, 30, 40, {"over_full_scale": True})
#: 人へ渡す音割れ 素材の音が割れている（下げても割れは残る）
_SOURCE_CLIP = Run(CLIPPING, 50, 60, {"over_full_scale": False})


def _finish(host: FakeHost, runs: list[Run]) -> dict[str, Any]:
    """検査を頼み、その頼みで終わった様子を入れてから結果を読む（AI が呼び直す形）"""
    _tool("check_output")(host, {"wait_seconds": 0})
    request: CheckRequest = host.check_requests[-1]
    end = request.end or 300
    report = assemble(
        runs,
        start=request.start,
        end=end,
        analyzed=end,
        fps=30.0,
        complete=True,
        limits=request.limits,
        index=None,
        loudness=-14.2,
        peak=0.5,
        length_seconds=None,
    )
    host.check = CheckState(request=request, report=report)
    result = _tool("check_output")(host, {"wait_seconds": 0})
    assert isinstance(result, dict)
    return result


def _fix(host: FakeHost, name: str) -> None:
    """直す道具の代わりに作品を 1 か所変える（AI の道具と同じ apply_commands を通す）"""
    host.apply_commands([RenameProject(name)], "直す")


class TestTable:
    def test_every_kind_has_a_way_to_fix(self) -> None:
        """壊れると、検査に種類を足したときに直し方を決め忘れ、AI が黙って直すか放っておく"""
        assert {kind for kind, _severity, _variant in FIXES} == set(KINDS)

    @pytest.mark.parametrize(
        ("kind", "severity", "measure", "action"),
        [
            (LOUDNESS, HIGH, {}, AUTO),
            (LOUDNESS, MEDIUM, {}, AUTO),
            (CLIPPING, HIGH, {"over_full_scale": True}, AUTO),
            (CLIPPING, HIGH, {}, HANDOFF),
            (TEXT_OVERFLOW, HIGH, {"beyond_screen": True}, AUTO),
            (TEXT_OVERFLOW, MEDIUM, {}, AUTO),
            (TEXT_OVERFLOW, LOW, {"moving": True}, LEAVE),
            (BLACK, HIGH, {}, ASK),
            (BLACK, MEDIUM, {}, HANDOFF),
            (BLACK, LOW, {}, LEAVE),
            (FREEZE, MEDIUM, {}, HANDOFF),
            (FREEZE, LOW, {}, LEAVE),
            (SILENCE, HIGH, {}, HANDOFF),
            (SILENCE, MEDIUM, {}, ASK),
            (SILENCE, LOW, {}, LEAVE),
            (LENGTH, MEDIUM, {}, HANDOFF),
        ],
    )
    def test_the_decided_table(
        self, kind: str, severity: str, measure: dict[str, object], action: str
    ) -> None:
        """壊れると、本人が尋ねると決めた詰め・削りを AI が黙ってする（時刻が全部動く）か、
        直してよいと決めた音量や字幕を直さずに人へ回す"""
        assert fix_for(kind, severity, measure).action == action

    def test_an_unknown_combination_goes_to_the_person(self) -> None:
        """壊れると、表に無い問題を AI が自分の判断で直しに行く"""
        assert fix_for("noise", HIGH, {}).action == HANDOFF
        assert fix_for(FREEZE, HIGH, {}).action == HANDOFF

    def test_the_tools_exist_and_can_be_undone(self) -> None:
        """壊れると、表が無い道具か取り消せない道具を指し、直した回を Ctrl+Z で戻せない"""
        for fix in FIXES.values():
            assert fix.action in (AUTO, ASK) or not fix.tools
            for name in fix.tools:
                assert _tool(name).safety is Safety.UNDOABLE, name

    def test_after_the_limit_nothing_is_fixed(self) -> None:
        """壊れると、上限に達した後の結果にも直してよい印が残り、AI が直しに行って断られ続ける"""
        for (kind, severity, variant), fix in FIXES.items():
            measure = {"over_full_scale": variant == "mixed"}
            closed = fix_for(kind, severity, measure, closed=True)
            assert closed.action in (HANDOFF, LEAVE)
            assert closed.tools == () or fix.action == LEAVE


class TestDeclarations:
    def test_the_check_still_only_reads(self) -> None:
        """壊れると、調べるたびに確かめの窓が出るか、「変更を自動で承認」で調べること自体が省かれる"""
        assert _tool("check_output").safety is Safety.READ_ONLY

    def test_the_limit_is_declared_as_an_app_setting(self) -> None:
        """壊れると、本人の設定（直す回数の上限）が結果に載るのに、送る物の宣言から漏れる"""
        assert Outbound.APP_SETTINGS in _tool("check_output").sends

    def test_the_ai_cannot_raise_its_own_limit(self) -> None:
        """壊れると、AI が引数で上限を広げ、止まらない繰り返しを自分で許せる"""
        properties = _tool("check_output").schema["properties"]
        assert not [name for name in properties if "round" in name or "fix" in name]


def _finish_file(host: FakeHost, path: Path, runs: list[Run]) -> dict[str, Any]:
    """書き出した動画のファイルを調べ、終わった様子を入れてから結果を読む"""
    arguments = {"path": str(path), "wait_seconds": 0}
    _tool("check_output")(host, arguments)
    request: CheckRequest = host.check_requests[-1]
    report = assemble(
        runs,
        start=0,
        end=300,
        analyzed=300,
        fps=30.0,
        complete=True,
        limits=request.limits,
        index=None,
        loudness=-20.0,
        peak=0.5,
        length_seconds=None,
    )
    host.check = CheckState(request=request, report=report)
    result = _tool("check_output")(host, arguments)
    assert isinstance(result, dict)
    return result


class TestFiles:
    """書き出した動画のファイルを調べたときは、今の作品を変える直し方を返さない（Codex P1）

    ファイルが今の作品から書き出した物かは確かめていない 古い書き出しや別の作品の動画でも
    作品のフォルダか書き出しのフォルダにあれば調べられる
    """

    @pytest.fixture
    def written(self, host: FakeHost, tmp_path: Path) -> Path:
        host.exported_to = tmp_path
        path = tmp_path / "前の書き出し.mp4"
        path.write_bytes(b"x" * 10)
        return path

    def test_a_file_check_offers_no_fix_on_the_open_work(
        self, host: FakeHost, written: Path
    ) -> None:
        """壊れると、古い書き出しや別の作品の動画の音量を見て、今開いている関係の無い作品の
        音量を AI が normalize_loudness で変えてしまう"""
        result = _finish_file(host, written, [_MIXED_CLIP, Run(BLACK, 100, 110, {})])
        assert result["problems"]
        for problem in result["problems"]:
            assert problem["fix"]["action"] in (HANDOFF, LEAVE), problem
            assert problem["fix"]["tools"] == []
        assert result["summary"]["fixes"][AUTO] == 0
        assert result["summary"]["fixes"][ASK] == 0
        assert result["fix_round"] is None
        assert "タイムラインを調べ直" in result["note"]

    def test_a_file_check_does_not_count_or_split(self, host: FakeHost, written: Path) -> None:
        """壊れると、書き出した動画を確かめただけで直す回が減るか、上限に達して編集を断られる"""
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 確かめて", by=Author.AI)
        _fix(host, "編集")
        _finish_file(host, written, [_MIXED_CLIP])
        _fix(host, "続き")
        _finish_file(host, written, [_MIXED_CLIP])
        assert not fixes_closed(host.document)
        assert host.document.checkpoint_label == "AI: 確かめて"
        host.document.end_checkpoint()
        assert host.document.history_labels[-1] == "AI: 確かめて"


def _finish_after_edit(host: FakeHost, runs: list[Run]) -> dict[str, Any]:
    """検査を頼んで待つ間に人がタイムラインを変え、その後で始めたときの作品の結果を読む"""
    pending = _tool("check_output")(host, {"wait_seconds": 30})
    assert isinstance(pending, Pending)
    request: CheckRequest = host.check_requests[-1]
    host.document.execute(RenameProject("待つ間の人の編集"), by=Author.PERSON)
    report = assemble(
        runs,
        start=request.start,
        end=request.end or 300,
        analyzed=request.end or 300,
        fps=30.0,
        complete=True,
        limits=request.limits,
        index=None,
        loudness=-14.2,
        peak=0.5,
        length_seconds=None,
    )
    host.check = CheckState(request=request, report=report)
    result = pending.then()
    assert isinstance(result, dict)
    return result


class TestChangedWhileWaiting:
    """検査を待つ間に作品が変わったら、古い作品の問題から直しを返さない（Codex P2）

    結果の問題とクリップ ID は調べ始めたときの作品の物 今の作品へ当てると、消えた・動いた
    クリップへ set_param や clean_subtitles を当ててしまう
    """

    def test_no_fix_is_offered_for_a_stale_result(self, host: FakeHost) -> None:
        """壊れると、待つ間に人が動かしたクリップの古い ID や位置へ、AI が直しを当てる"""
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        result = _finish_after_edit(host, [_MIXED_CLIP, Run(BLACK, 100, 110, {})])
        host.document.end_checkpoint()
        assert result["problems"]
        for problem in result["problems"]:
            assert problem["fix"]["action"] in (HANDOFF, LEAVE), problem
            assert problem["fix"]["tools"] == []
        assert result["summary"]["fixes"][AUTO] == 0
        assert "1 回目の直し" not in result["note"]
        assert "調べ直" in result["note"]

    def test_a_stale_result_does_not_use_up_a_round(self, host: FakeHost) -> None:
        """壊れると、待つ間に人が編集しただけで直す回が上限に達し、調べ直した後に直せない"""
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "AI の直し")
        stale = _finish_after_edit(host, [_MIXED_CLIP])
        assert stale["fix_round"]["used"] == 0
        assert not fixes_closed(host.document)
        fresh = _finish(host, [_MIXED_CLIP])
        host.document.end_checkpoint()
        assert fresh["fix_round"]["used"] == 1
        assert fresh["fix_round"]["closed"] is True


class TestUnusableResults:
    def test_a_failed_check_does_not_use_up_a_round(self, host: FakeHost) -> None:
        """壊れると、検査が失敗して何も分からなかったのに直す回を使い、上限 1 なら調べ直した
        後に直せない"""
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "AI の直し")
        pending = _tool("check_output")(host, {"wait_seconds": 30})
        assert isinstance(pending, Pending)
        request: CheckRequest = host.check_requests[-1]
        host.check = CheckState(request=request, failure="読めない")
        with pytest.raises(ToolError, match="調べられません"):
            pending.then()
        assert not fixes_closed(host.document)
        # 本物の窓は失敗した頼みを同じ頼みでも調べ直す 偽のホストでは様子を空へ戻して同じにする
        host.check = CheckState()
        fresh = _finish(host, [_MIXED_CLIP])
        host.document.end_checkpoint()
        assert fresh["fix_round"]["used"] == 1

    def test_a_missing_report_does_not_use_up_a_round(self, host: FakeHost) -> None:
        """壊れると、結果がまだ無いと断った呼び出しで直す回を使い、上限に達して直せなくなる"""
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "AI の直し")
        pending = _tool("check_output")(host, {"wait_seconds": 30})
        assert isinstance(pending, Pending)
        request: CheckRequest = host.check_requests[-1]
        host.check = CheckState(request=request)
        with pytest.raises(ToolError, match="まだありません"):
            pending.then()
        host.document.end_checkpoint()
        assert not fixes_closed(host.document)


class TestAnswer:
    def test_each_problem_says_how_to_fix_it(self, host: FakeHost) -> None:
        """壊れると、AI はどの道具で直すか・尋ねるべきかを知らずに、見つけた物を
        手当たり次第に直す"""
        result = _finish(host, [_MIXED_CLIP, _SOURCE_CLIP])
        mixed, source = result["problems"]
        assert mixed["fix"]["action"] == AUTO
        assert "normalize_loudness" in mixed["fix"]["tools"]
        assert source["fix"]["action"] == HANDOFF
        assert result["summary"]["fixes"] == {AUTO: 1, ASK: 0, HANDOFF: 1, LEAVE: 0}
        assert result["fix_round"] == {"used": 0, "limit": 3, "next_round": 1, "closed": False}
        assert "1 回目の直し" in result["note"]


class TestRounds:
    def test_rounds_are_counted_and_stop_at_the_limit(self, host: FakeHost) -> None:
        """壊れると、直しても消えない問題を AI が上限なく直し続け、時間と費用が膨らむ"""
        host.fix_limit = 2
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        assert _finish(host, [_MIXED_CLIP])["fix_round"]["next_round"] == 1
        _fix(host, "1 回目")
        assert _finish(host, [_MIXED_CLIP])["fix_round"]["next_round"] == 2
        _fix(host, "2 回目")
        result = _finish(host, [_MIXED_CLIP])
        host.document.end_checkpoint()

        assert result["fix_round"] == {"used": 2, "limit": 2, "next_round": None, "closed": True}
        (problem,) = result["problems"]
        assert problem["fix"]["action"] == HANDOFF
        assert "上限（2 回）に達しました" in result["note"]

    def test_reading_the_same_result_again_is_not_a_round(self, host: FakeHost) -> None:
        """壊れると、途中までの結果の続きを読むだけで直す回が減り、1 回も直さずに止まる"""
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        for _ in range(3):
            result = _finish(host, [_MIXED_CLIP])
        host.document.end_checkpoint()
        assert result["fix_round"]["used"] == 0
        assert result["fix_round"]["closed"] is False

    def test_edits_by_the_person_are_not_a_round(self, host: FakeHost) -> None:
        """壊れると、調べ直す間に人が手で変えた分まで AI の直した回に数え、AI が 1 回も
        直さずに止まる"""
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        host.document.execute(RenameProject("人の直し"), by=Author.PERSON)
        result = _finish(host, [_MIXED_CLIP])
        assert result["fix_round"]["used"] == 0
        assert result["fix_round"]["closed"] is False
        _fix(host, "AI の直し")
        result = _finish(host, [_MIXED_CLIP])
        host.document.end_checkpoint()
        assert result["fix_round"]["used"] == 1
        assert result["fix_round"]["closed"] is True

    def test_a_new_instruction_starts_counting_again(self, host: FakeHost) -> None:
        """壊れると、前の指示で上限に達した後、次の指示でも編集を断り続ける"""
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        assert _finish(host, [])["fix_round"]["closed"] is True
        assert fixes_closed(host.document)
        host.document.end_checkpoint()
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        assert not fixes_closed(host.document)
        host.document.end_checkpoint()

    def test_zero_only_checks(self, host: FakeHost) -> None:
        """壊れると、直さないと決めた人の作品でも、AI が見つけた物を直してしまう"""
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        result = _finish(host, [_MIXED_CLIP])
        host.document.end_checkpoint()
        assert result["problems"][0]["fix"]["action"] == HANDOFF
        assert "直さないことになっています" in result["note"]

    def test_each_round_is_its_own_undo_step(self, host: FakeHost) -> None:
        """壊れると、直した回を戻すつもりの Ctrl+Z 1 回で、指示の編集まで消える"""
        before = host.document.project
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _fix(host, "編集")
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 1")
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 2")
        _finish(host, [])
        host.document.end_checkpoint()

        assert host.document.history_labels[-3:] == (
            "AI: 仕上げて",
            fix_label(1),
            fix_label(2),
        )
        host.document.undo()
        assert host.document.project.name == "直し 1"
        host.document.undo()
        assert host.document.project.name == "編集"
        host.document.undo()
        assert host.document.project == before


class TestBridge:
    def _closed(self, host: FakeHost) -> None:
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])

    def test_edits_are_refused_after_the_limit(self, host: FakeHost) -> None:
        """壊れると、上限に達したと伝えても AI が編集を続け、回数を守るかが AI 任せになる"""
        self._closed(host)
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        before = host.document.project
        done, box = call_in_thread(bridge, _tool("add_track"), {"kind": "video"})
        pump_until(bridge, done)
        host.document.end_checkpoint()
        assert isinstance(box.get("error"), ToolError)
        assert "上限" in str(box["error"])
        assert host.document.project is before

    def test_it_refuses_before_asking_the_person(self, host: FakeHost) -> None:
        """壊れると、確かめの窓で人が許したのに、その後で断られて何も起きない"""
        self._closed(host)
        bridge = allow_sending(EditorBridge(host, auto_approve=False))
        done, box = call_in_thread(bridge, _tool("add_track"), {"kind": "video"})
        pump_until(bridge, done)
        host.document.end_checkpoint()
        assert isinstance(box.get("error"), ToolError)
        assert bridge.take_approval() is None

    def test_undo_still_works_after_the_limit(self, host: FakeHost) -> None:
        """壊れると、上限の後に人が「直しを戻して」と頼んでも、AI が戻せない"""
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し")
        _finish(host, [_MIXED_CLIP])
        assert fixes_closed(host.document)
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        done, box = call_in_thread(bridge, _tool("undo"), {})
        pump_until(bridge, done)
        host.document.end_checkpoint()
        assert "error" not in box, box
        assert host.document.project.name != "直し"

    def test_it_refuses_before_asking_to_send(self, host: FakeHost) -> None:
        """壊れると、上限で断ると決まっている道具のために「外へ送ってよいか」の窓が出て、
        人が答えた後で断られる"""
        self._closed(host)
        # 送る物は既定の「尋ねる」のまま 文を送る道具を呼ぶ
        bridge = EditorBridge(host, auto_approve=True)
        operation = _tool("add_text")
        assert Outbound.TEXT in operation.sends
        done, box = call_in_thread(bridge, operation, {"text": "字幕", "frame": 0})
        pump_until(bridge, done, timeout=2.0)
        asked = bridge.take_send_request()
        bridge.cancel()
        done.wait(5.0)
        host.document.end_checkpoint()
        assert asked is None
        assert isinstance(box.get("error"), ToolError)
        assert "上限" in str(box["error"])

    def test_cancelling_an_export_still_works_after_the_limit(self, host: FakeHost) -> None:
        """壊れると、直す回数の上限に達した後、AI が始めた書き出しを AI が止められない"""
        self._closed(host)
        host.export_state = {**host.export_state, "state": "rendering"}
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        done, box = call_in_thread(bridge, _tool("cancel_export"), {})
        pump_until(bridge, done)
        host.document.end_checkpoint()
        assert "error" not in box, box
        assert host.export_state["state"] == "cancelled"

    def test_only_the_listed_tools_pass_after_the_limit(self, host: FakeHost) -> None:
        """壊れると、上限の後も作品を変える道具が通るか、通すと決めた道具が断られる"""
        self._closed(host)
        passed: set[str] = set()
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        changing = [op for op in OPERATIONS if op.writes and not op.irreversible]
        done = threading.Event()

        def look() -> None:
            # 断るかの確かめは編集画面のスレッドへ頼むので、AI のスレッドの代わりに
            # 別のスレッドで呼ぶ
            try:
                for operation in changing:
                    try:
                        bridge._refuse_after_fix_limit(operation)
                    except ToolError:
                        continue
                    passed.add(operation.name)
            finally:
                done.set()

        threading.Thread(target=look, daemon=True).start()
        pump_until(bridge, done, timeout=30.0)
        host.document.end_checkpoint()
        assert done.is_set()
        assert passed == set(PASSES_AFTER_LIMIT)

    def test_the_document_names_what_passes_after_the_limit(self) -> None:
        """壊れると、開発の文書に書いた「上限の後も通す道具」と、実際にブリッジが通す道具が
        食い違い、直す人が文書を信じて取り違える"""
        text = (Path(__file__).parents[2] / "docs" / "development.md").read_text(encoding="utf-8")
        section = text[text.index("### 検査で見つけた問題を AI が直す繰り返し") :]
        bullet = section[section.index("- 上限は `Preferences.ai_fix_rounds`") :]
        bullet = bullet[: bullet.index("\n- ", 1)]
        named = {
            name for name in re.findall(r"`([a-z_]+)`", bullet) if find_operation(name) is not None
        }
        irreversible = {operation.name for operation in OPERATIONS if operation.irreversible}
        assert named == set(PASSES_AFTER_LIMIT) | irreversible

    def test_edits_before_any_check_are_not_refused(self, host: FakeHost) -> None:
        """壊れると、直さないと決めた人の所で、調べる前の普通の編集まで断られる"""
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 足して", by=Author.AI)
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        done, box = call_in_thread(bridge, _tool("add_track"), {"kind": "video"})
        pump_until(bridge, done)
        host.document.end_checkpoint()
        assert "error" not in box, box
