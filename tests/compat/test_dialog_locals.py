"""``--dialog`` の ``local 名前=初期値`` をスクリプトのローカル変数として渡すこと（#190）

PSDToolKit の 吹き出し は ``--dialog:余白 横,local mlr=24;…`` と書き、本文で ``mlr`` を読む
AviUtl はこの形を本文の頭で宣言したローカル変数として渡す ``local mlr`` という名前の
大域変数として置くと、本文の ``mlr`` は nil のままで、吹き出し が 1 行目の計算で落ちていた
"""

from __future__ import annotations

import numpy as np

from sashimono.compat.aviutl.control import parse_control
from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime

SOURCE = """--dialog:余白 横,local mlr=24;色/col,local col=0xffffff;普通,plain=3
obj.ox = mlr
obj.oy = col
obj.oz = plain
"""


def _state(values: dict[str, object]) -> ObjectState:
    return ObjectState(image=np.zeros((1, 1, 4), np.uint8), values=values)


def _run(runtime: LuaScriptRuntime, source: str, values: dict[str, object]) -> ObjectState:
    state = _state(values)
    result = runtime.run(source, state, header=parse_control(source))
    assert not result.failed, result.message
    return state


class TestDialogLocals:
    def test_local_items_reach_the_body(self) -> None:
        # local の欄の値が本文へ届かないと、PSDToolKit の 吹き出し が余白を nil で読んで
        # 1 行目の計算で落ちる
        runtime = LuaScriptRuntime(report=CompatibilityReport())
        state = _run(runtime, SOURCE, {"local mlr": 30, "local col": 0xFF0000, "plain": 5})
        assert (state.ox, state.oy, state.oz) == (30.0, float(0xFF0000), 5.0)

    def test_they_do_not_leak_into_the_next_script(self) -> None:
        # ローカル変数なので、同じランタイムで次に走るスクリプトからは見えない
        runtime = LuaScriptRuntime(report=CompatibilityReport())
        _run(runtime, SOURCE, {"local mlr": 30, "local col": 1, "plain": 5})
        after = _run(runtime, "obj.ox = mlr == nil and 1 or 0", {})
        assert after.ox == 1.0

    def test_a_nil_local_does_not_read_a_global_of_the_same_name(self) -> None:
        # 初期値 nil の欄と値の無い欄も本文のローカル変数 宣言しないと、同じランタイムで前に
        # 走ったスクリプトが残した同じ名前の大域変数を読み、描き方が前のスクリプトで変わる
        source = (
            "--dialog:表,local tbl=nil;余白,local mlr=24\n"
            "obj.ox = tbl == nil and 1 or 0\n"
            "obj.oy = mlr == nil and 1 or 0\n"
        )
        runtime = LuaScriptRuntime(report=CompatibilityReport())
        _run(runtime, "tbl = 5 mlr = 7", {})
        state = _run(runtime, source, {"local mlr": None})
        assert (state.ox, state.oy) == (1.0, 1.0)

    def test_a_body_starting_with_a_bracket_still_runs(self) -> None:
        # 宣言と本文の間に文の区切りが無いと、括弧で始まる本文が直前の値の呼び出しに読まれ、
        # スクリプトが 1 行目で落ちる
        runtime = LuaScriptRuntime(report=CompatibilityReport())
        state = _run(runtime, "(function() obj.ox = mlr end)()", {"local mlr": 3})
        assert state.ox == 3.0

    def test_line_numbers_in_errors_stay_the_same(self) -> None:
        # 頭に宣言を足しても、失敗したときの行番号がスクリプトの行とずれない
        runtime = LuaScriptRuntime(report=CompatibilityReport())
        state = _state({"local mlr": 1})
        result = runtime.run("local a = 1\nerror('x')", state)
        assert result.failed and ":2:" in result.message
