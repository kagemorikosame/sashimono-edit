"""検査で見つけた問題を AI が直して調べ直す繰り返し（#339 C3）

2026-10-20 に本人が決めたこと

- 直すのは今の道具のまま 回数を数えて上限で止めるのはアプリ（check_output とブリッジ）
- 自動で直す範囲は ai/fixes.py の表のとおり
- 直す回ごとに「AI: 検査の直し（n 回目）」の段へ分け、Ctrl+Z 1 回でその回だけを戻せる
- 上限は Preferences.ai_fix_rounds（既定 3 0 は調べるだけ）

偽のホストに検査の結果を入れ、道具とブリッジが数え・区切り・断るかを見る
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.fixes import (
    ASK,
    AUTO,
    FIXES,
    HANDOFF,
    LEAVE,
    fix_for,
    fix_label,
    fixes_closed,
)
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, find_operation
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

    def test_edits_before_any_check_are_not_refused(self, host: FakeHost) -> None:
        """壊れると、直さないと決めた人の所で、調べる前の普通の編集まで断られる"""
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 足して", by=Author.AI)
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        done, box = call_in_thread(bridge, _tool("add_track"), {"kind": "video"})
        pump_until(bridge, done)
        host.document.end_checkpoint()
        assert "error" not in box, box
