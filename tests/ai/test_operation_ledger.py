"""操作の台帳（:mod:`sashimono.ai.ledger`）の中身が、AI の道具の一覧と食い違わないか（#348）

画面の側から見た突き合わせは ``tests/ui/test_operation_ledger.py`` ここは道具の側から見る
"""

from __future__ import annotations

from collections import Counter

import pytest

from sashimono.ai import ledger
from sashimono.ai.ledger import AI_ONLY, CHILD_ISSUES, LEDGER, Entry, Status
from sashimono.ai.operations import OPERATIONS

_NAMES = {operation.name for operation in OPERATIONS}


def unknown_tools(rows: tuple[Entry, ...], only: dict[str, str]) -> list[str]:
    """台帳と AI_ONLY に書いてあるのに、``ai/operations.py`` に無い道具の名前"""
    named = {tool for entry in rows for tool in entry.tools} | set(only)
    return sorted(named - _NAMES)


def unmapped_tools(rows: tuple[Entry, ...], only: dict[str, str]) -> list[str]:
    """``ai/operations.py`` にあるのに、台帳のどの行にも AI_ONLY にも無い道具の名前"""
    named = {tool for entry in rows for tool in entry.tools} | set(only)
    return sorted(_NAMES - named)


def malformed(rows: tuple[Entry, ...]) -> list[str]:
    """状態に要る欄が欠けた行"""
    wrong: list[str] = []
    for entry in rows:
        if entry.status in (Status.SAME, Status.PARTIAL) and not entry.tools:
            wrong.append(f"{entry.key}: 道具の名前が無い")
        if entry.status in (Status.MISSING, Status.PARTIAL, Status.HIDDEN) and (
            entry.issue not in CHILD_ISSUES
        ):
            wrong.append(f"{entry.key}: 足す先の Issue が #298 の子（#319〜#352）でない")
        if entry.status in (Status.MISSING, Status.HIDDEN) and entry.tools:
            wrong.append(f"{entry.key}: AI にできないはずなのに道具の名前がある")
        if entry.status in (Status.PARTIAL, Status.HIDDEN) and not entry.note:
            wrong.append(f"{entry.key}: 足りない所か、出さない理由の案が無い")
    return wrong


class TestToolsAgreeWithTheLedger:
    def test_every_tool_named_in_the_ledger_exists(self) -> None:
        # 落ちると: 台帳が「AI でできる」と言う操作の道具が無く、AI に頼んでも何も起きない
        assert not unknown_tools(LEDGER, AI_ONLY)

    def test_every_tool_is_in_the_ledger(self) -> None:
        # 落ちると: 道具を足した PR が台帳を直さず、「まだできない」と数えたままの操作が残る
        left = unmapped_tools(LEDGER, AI_ONLY)
        assert not left, (
            "src/sashimono/ai/ledger.py に、この道具を使う画面の操作の行を same / partial で"
            f"書くか、画面に当たらなければ AI_ONLY に訳を書く: {left}"
        )

    def test_a_renamed_tool_is_caught(self) -> None:
        # 落ちると: 道具の名前を変えても台帳の古い名前に気付かず、上の見張りが効かない
        rows = (*LEDGER, ledger.same("メニュー/試し", "no_such_tool"))
        assert unknown_tools(rows, AI_ONLY) == ["no_such_tool"]

    def test_a_new_tool_without_a_row_is_caught(self) -> None:
        # 落ちると: 道具を足して台帳に書かなくても通り、台帳が実物とずれる
        rows = tuple(entry for entry in LEDGER if "undo" not in entry.tools)
        assert "undo" in unmapped_tools(rows, AI_ONLY)


class TestRowsAreWellFormed:
    def test_keys_are_unique(self) -> None:
        # 落ちると: 同じ操作に 2 つの答えが付き、どちらが今の状態か分からなくなる
        twice = [key for key, n in Counter(entry.key for entry in LEDGER).items() if n > 1]
        assert not twice

    def test_each_status_has_what_it_needs(self) -> None:
        # 落ちると: 「まだできない」のに足す先が無い行や、理由の無い「出さない」行が残り、
        # 抜けを数えて埋める進め方が回らない
        assert not malformed(LEDGER)

    def test_hidden_rows_are_still_only_candidates(self) -> None:
        # 落ちると: 利用者が決めていない線引きを、決まったことのように読ませる
        # 線引きは #350（E4）で決める 決まったら状態の名前ごと変える
        assert Status.HIDDEN.value == "出さない候補"
        assert all(e.issue == 350 for e in LEDGER if e.status is Status.HIDDEN)

    @pytest.mark.parametrize("status", list(Status))
    def test_counts_cover_every_row(self, status: Status) -> None:
        # 落ちると: PR の本文や棚卸しに書く数が、台帳の行の数と食い違う
        assert ledger.counts()[status] == sum(1 for e in LEDGER if e.status is status)
