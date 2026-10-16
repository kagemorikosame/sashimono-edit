"""操作の台帳（:mod:`sashimono.ai.ledger`）の中身が、AI の道具の一覧と食い違わないか（#348）

画面の側から見た突き合わせは ``tests/ui/test_operation_ledger.py`` ここは道具の側から見る
"""

from __future__ import annotations

from collections import Counter

import pytest

from sashimono.ai import ledger
from sashimono.ai.ledger import (
    AI_ONLY,
    CHILD_ISSUES,
    DECIDED_READ_ONLY,
    LEDGER,
    Entry,
    Status,
)
from sashimono.ai.operations import OPERATIONS, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety

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
        if entry.status in (Status.MISSING, Status.PARTIAL, Status.READ_ONLY, Status.HIDDEN) and (
            entry.issue not in CHILD_ISSUES
        ):
            wrong.append(f"{entry.key}: 足す先の Issue が #298 の子（#319〜#352）でない")
        if entry.status in (Status.MISSING, Status.HIDDEN) and entry.tools:
            wrong.append(f"{entry.key}: AI にできないはずなのに道具の名前がある")
        if entry.status in (Status.PARTIAL, Status.READ_ONLY, Status.HIDDEN) and not entry.note:
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


def safety_mismatches(rows: tuple[Entry, ...]) -> list[str]:
    """行の区分（#350 の 読むだけ・取り消せる・取り消せない）と道具の区分が違う所"""
    wrong: list[str] = []
    for entry in rows:
        expected = entry.expected_safety
        for tool in entry.tools:
            operation = find_operation(tool)
            if expected is None or operation is None or operation.safety is expected:
                continue
            wrong.append(
                f"{entry.key}: 行は{expected.label}なのに {tool} は{operation.safety.label}"
            )
    return wrong


class TestSafetyAgreesWithTheLedger:
    """台帳の行の区分と、道具の区分（``Operation.safety`` #350）をそろえる"""

    def test_each_tool_has_the_safety_of_its_row(self) -> None:
        # 落ちると: 取り消せない画面の操作（保存・書き出し）を取り消せる道具で受け、
        # 「変更を自動で承認」で確かめずに上書きする 読むだけと決めた操作を変える道具で受けても同じ
        assert not safety_mismatches(LEDGER)

    def test_a_mismatched_tool_is_caught(self) -> None:
        # 落ちると: 区分の違う道具を行に書いても見張りが気付かない
        rows = (ledger.missing("メニュー/ファイル/保存", 327, safety=Safety.IRREVERSIBLE),)
        wrong = (
            Entry(rows[0].key, Status.SAME, ("delete_clips",), safety=Safety.IRREVERSIBLE),
            ledger.read_only("メニュー/表示/設定…", "設定"),
            Entry("メニュー/表示/設定…", Status.READ_ONLY, ("set_param",), 350, "設定"),
        )
        assert not safety_mismatches(rows)
        assert len(safety_mismatches(wrong)) == 2

    def test_read_only_rows_use_the_reading_tool(self) -> None:
        # 落ちると: アプリ自身の操作を「読むだけ」と書きながら、読む道具が台帳から外れ、
        # AI が設定を読めるかどうかが台帳から分からない
        rows = [e for e in LEDGER if e.status is Status.READ_ONLY]
        assert all(e.tools == ("get_app_settings",) for e in rows)
        assert Safety.READ_ONLY.label in Status.READ_ONLY.value

    def test_reading_tools_outside_the_ledger_only_read(self) -> None:
        # 落ちると: 「画面を見て分かることを読む道具」と書いた道具が、実は何かを変える
        reading = [name for name, why in AI_ONLY.items() if why == ledger.READING]
        assert reading
        for name in reading:
            operation = find_operation(name)
            assert operation is not None and operation.safety is Safety.READ_ONLY, name


def sends_mismatches(rows: tuple[Entry, ...], only: dict[str, str]) -> list[str]:
    """台帳の行と、道具が外へ送る物の宣言（``Operation.sends`` #342）が食い違う所

    - アプリ自身の操作（読むだけと決めた行）を受ける道具は、アプリの設定を送ると宣言し、
      作品の中身（絵・字幕と文字）は送らない 送ると、設定を読ませただけで作品の中身が外へ出る
    - 「画面を見て分かることを読む道具」も含め、台帳に出てくる道具はどれも何かを送ると書く
    """
    wrong: list[str] = []
    for entry in rows:
        for tool in entry.tools:
            operation = find_operation(tool)
            if operation is None:
                continue
            if not operation.sends:
                wrong.append(f"{entry.key}: {tool} は送る物を書いていない")
            if entry.status is Status.READ_ONLY and (
                Outbound.APP_SETTINGS not in operation.sends
                or operation.sends & {Outbound.FRAMES, Outbound.TEXT}
            ):
                wrong.append(
                    f"{entry.key}: アプリ自身の操作を受ける {tool} の送る物が"
                    f" {sorted(kind.value for kind in operation.sends)}"
                )
    for tool in only:
        operation = find_operation(tool)
        if operation is not None and not operation.sends:
            wrong.append(f"AI_ONLY: {tool} は送る物を書いていない")
    return wrong


class TestSendsAgreeWithTheLedger:
    """台帳の行と、道具が外へ送る物の宣言（``Operation.sends`` #342）をそろえる"""

    def test_each_tool_sends_what_its_row_says(self) -> None:
        # 落ちると: アプリの設定を読むだけのはずの道具が作品の絵や文を送り、設定を読ませた
        # だけで、同意の窓に出ない物が外へ出る（または送る物の書かれていない道具が台帳に残る）
        assert not sends_mismatches(LEDGER, AI_ONLY)

    def test_a_mismatch_is_caught(self) -> None:
        # 落ちると: 絵を返す道具をアプリ自身の操作の行に書いても、見張りが気付かない
        wrong = (Entry("メニュー/表示/設定…", Status.READ_ONLY, ("preview_frame",), 350, "設定"),)
        assert len(sends_mismatches(wrong, {})) == 1

    def test_the_settings_reader_is_declared_as_sending_settings_and_paths(self) -> None:
        # 落ちると: 読むだけと決めた行の道具（get_app_settings）が、書き出しのフォルダなどの
        # 場所を返すのに名前と場所を送ると書かず、尋ねずに送る
        operation = find_operation("get_app_settings")
        assert operation is not None
        assert operation.sends == {Outbound.APP_SETTINGS, Outbound.FILES}


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
        # 線引きは #350（E4）で決める 決まった物は READ_ONLY へ移す
        assert Status.HIDDEN.value == "出さない候補"
        hidden = [e for e in LEDGER if e.status is Status.HIDDEN]
        assert all(e.issue == 350 and DECIDED_READ_ONLY not in e.note for e in hidden)

    def test_read_only_rows_cite_the_decision(self) -> None:
        # 落ちると: 「AI は読むだけ」が誰がいつ決めたことか分からず、候補と見分けが付かない
        rows = [e for e in LEDGER if e.status is Status.READ_ONLY]
        assert rows
        assert all(e.issue == 350 and DECIDED_READ_ONLY in e.note for e in rows)

    @pytest.mark.parametrize("status", list(Status))
    def test_counts_cover_every_row(self, status: Status) -> None:
        # 落ちると: PR の本文や棚卸しに書く数が、台帳の行の数と食い違う
        assert ledger.counts()[status] == sum(1 for e in LEDGER if e.status is status)
