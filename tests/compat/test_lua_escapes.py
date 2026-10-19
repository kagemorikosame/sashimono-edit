"""Lua の文字の逃がし書きを、AviUtl1 の Lua（LuaJIT）と同じに戻す

``--dialog`` の初期値と ``.exa`` の ``param=`` は Lua の書き方のまま入っている
戻し方が 1 つでも違うと、スクリプトが受け取る文字が AviUtl1 と変わる

比べる相手は lupa に入っている LuaJIT 同じ文字を LuaJIT に読ませ、出来たバイト列を
そのまま突き合わせる（こちらが規則を覚え違えていても、表の側で気付ける）
"""

from __future__ import annotations

import faulthandler
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from _pytest.faulthandler import fault_handler_stderr_fd_key

from sashimono.compat.aviutl.control import lua_string, lua_value, parse_control
from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image
from sashimono.effects.spec import TextSpec

#: LuaJIT がそのまま読める書き方 どれも引用符の中身（``\\`` は Lua の逆斜線 1 つ）
ESCAPES = (
    "a\\ab",
    "\\b\\f\\v",
    "\\r\\n\\t",
    "\\\\\\'\\\"",
    "\\065\\66C",
    "\\0\\9\\255",
    "\\1234",
    "\\x41\\xe3\\x81\\x82",
    "a\\z  \n  b",
    "a\\zb",
    "\\u{3042}\\u{41}",
    "行\\\n次",
    "行\\\r\n次",
    "あ\\255い",
)

#: LuaJIT が読み込みで止める書き方 引用符ごと書かれたまま返る決まり
BAD_ESCAPES = (
    "\\q",
    "\\256",
    "\\x4",
    "\\xg0",
    "\\u41",
    "\\u{}",
    "a\\",
    # 代用符号の範囲と U+10FFFF より先 LuaJIT は invalid escape sequence で止める
    "\\u{D800}",
    "\\u{DFFF}",
    "\\u{110000}",
)


def _luajit() -> Any:
    """比べる相手の LuaJIT AviUtl1 の Lua はこれ

    ソフトが走らせる埋め込みの Lua は 5.1 で、``\\x`` と ``\\z`` を知らず、``\\q`` を
    黙って ``q`` にする AviUtl1 の読み方とは違うので、比べる相手にはしない
    無い機械では LuaJIT との突き合わせだけを飛ばす
    """
    # 型の情報を持たない包みなので、名前で読み込む（runtime.py と同じやり方）
    try:
        luajit21 = import_module("lupa.luajit21")
    except ImportError:
        return None
    return luajit21.LuaRuntime(encoding=None, register_builtins=False)


LUAJIT = _luajit()

#: LuaJIT が Lua のエラーを投げるときの Windows の例外の番号の頭 最後の桁が Lua の
#: 状態の番号（3 が文法の誤り、2 が実行時の誤り） 間の ``4c4a`` は "LJ"
_LUAJIT_EXCEPTION_CODE = "0xe24c4a0"


@contextmanager
def _faulthandler_paused(config: pytest.Config) -> Iterator[None]:
    """LuaJIT にわざとエラーを出させる間だけ、faulthandler の書き出しを止める

    Windows の LuaJIT は Lua のエラーを必ず SEH の例外（``0xe24c4a0X``）で投げて
    自分で捕まえる Lua の ``pcall`` の中でも同じ faulthandler は捕まるかどうかを
    見ずに、上の位が立った番号をすべて致命的な物として書き出すので、``LuaError`` に
    なるだけの誤りでも ``Windows fatal exception`` が出る これが試験の出力に並ぶと、
    本物の落ち方が起きても紛れて見落とす（#222）

    番号で選んで黙らせる口は faulthandler に無いので、止める間を狭くして済ませる
    戻すときは pytest が開いた書き出し先（複製した標準エラー）へ戻す 番号 2 の
    標準エラーへ戻すと、試験の間は取り込みの一時ファイルへ向いていて誰も読まない
    戻し先が分からないときは止めない 止めたまま戻せないと、以降の本物の落ち方が
    1 つも書き出されなくなる
    """
    stderr = config.stash.get(fault_handler_stderr_fd_key, None)
    if not faulthandler.is_enabled() or stderr is None:
        yield
        return
    faulthandler.disable()
    try:
        yield
    finally:
        faulthandler.enable(file=stderr)


@pytest.mark.parametrize("body", ESCAPES)
def test_the_same_bytes_as_luajit(body: str) -> None:
    # 戻し方が LuaJIT とずれると、ダイアログの初期値や .exa の param= の文字が
    # AviUtl1 と違う文字でスクリプトに届く
    if LUAJIT is None:
        pytest.skip("LuaJIT が無い")
    literal = '"' + body + '"'
    expected = LUAJIT.execute(b"return " + literal.encode("utf-8"))
    assert lua_string(literal).encode("utf-8", "surrogateescape") == expected


@pytest.mark.parametrize("body", BAD_ESCAPES)
def test_what_luajit_rejects_is_left_as_written(body: str, pytestconfig: pytest.Config) -> None:
    # 推し量って直すと、AviUtl1 では読めない値が動く値として紛れる
    literal = '"' + body + '"'
    if LUAJIT is not None:
        with (
            _faulthandler_paused(pytestconfig),
            pytest.raises(Exception),  # noqa: B017 - lupa の例外の種類は版で違う
        ):
            LUAJIT.execute(b"return " + literal.encode("utf-8"))
    assert lua_string(literal) == literal


def test_luajit_errors_do_not_print_a_fatal_exception() -> None:
    """LuaJIT が捕まえる誤りが、試験の出力に致命的な例外として並ばない

    並ぶと、本物の落ち方（読み書きの違反など）が起きても同じ見た目の行に紛れて
    見落とす faulthandler の書き出しは pytest の取り込みを通らないので、別の
    プロセスで走らせて出力を見る
    """
    if LUAJIT is None:
        pytest.skip("LuaJIT が無い")
    target = f"{Path(__file__)}::test_what_luajit_rejects_is_left_as_written"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", target],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert _LUAJIT_EXCEPTION_CODE not in output, output


def test_the_faulthandler_is_back_after_a_luajit_error(pytestconfig: pytest.Config) -> None:
    # 戻し損ねると、それより後の試験で本物が落ちても何も書き出されない
    if LUAJIT is None:
        pytest.skip("LuaJIT が無い")
    # -p no:faulthandler で走らせると書き出し先が無く、戻ったかどうかを確かめられない
    # 切れているかどうかで飛ばす条件にすると、先に走った試験が戻し損ねた回帰まで
    # 飛ばしてしまうので、書き出し先があるのに切れていれば落とす
    if fault_handler_stderr_fd_key not in pytestconfig.stash:
        pytest.skip("faulthandler の pytest プラグインが無効")
    assert faulthandler.is_enabled(), "先に走った試験が faulthandler を戻し損ねている"
    with _faulthandler_paused(pytestconfig), pytest.raises(Exception):  # noqa: B017
        LUAJIT.execute(b'error("x")')
    assert faulthandler.is_enabled()


def test_the_old_shortcuts_are_gone() -> None:
    # 以前は \n と \t 以外を次の 1 文字にしていた \r が r、\065 が 065 になっていた
    assert lua_string('"\\r"') == "\r"
    assert lua_string('"\\065"') == "A"
    assert lua_string("'\\x41\\z   B'") == "AB"


def test_a_dialog_default_is_unescaped() -> None:
    # --dialog の文字の初期値の逃がし書きを戻さないと、設定欄の既定の文字が
    # A と改行ではなく \065\r のまま出る
    (spec,) = parse_control('--dialog:本文,_1="\\065\\r";').parameters
    assert isinstance(spec, TextSpec)
    assert spec.default == "A\r"


#: 文字の中身をバイトの番号の並びにする Lua の書き方 LuaJIT とソフトの Lua の両方で走らせる
_BYTES_OF = (
    "local t = {} for i = 1, #s do t[#t + 1] = string.byte(s, i) end return table.concat(t, ',')"
)


@pytest.mark.parametrize("body", ["a\\255b", "\\xFF\\xFE", "\\xe3\\x81", "あ\\128"])
def test_a_high_byte_reaches_the_script_as_the_same_bytes(body: str) -> None:
    """ダイアログの値が、LuaJIT が持つのと同じバイト列でスクリプトに届く

    読めないバイトを持つ値をそのまま lupa へ渡すと UTF-8 へ直せずに例外になり、
    スクリプトが 1 行も走らなかった 大域変数と ``obj.名前`` の両方の入口を見る
    """
    (spec,) = parse_control(f'--dialog:本文,_1="{body}";').parameters
    value = lua_value(spec, spec.default_value())
    runtime = LuaScriptRuntime(instruction_limit=100_000)
    state = ObjectState(image=blank_image(8, 8))
    state.values["_1"] = value
    script = (
        "local function bytes(s) " + _BYTES_OF + " end "
        "obj.global_bytes = bytes(_1) obj.field_bytes = bytes(obj._1)"
    )
    result = runtime.run(script, state)
    assert not result.failed, result.message
    got = state.values["global_bytes"]
    assert got == state.values["field_bytes"]
    if LUAJIT is None:
        pytest.skip("LuaJIT が無い")
    expected = LUAJIT.execute(('local s = "' + body + '" ' + _BYTES_OF).encode("utf-8"))
    assert str(got).encode("ascii") == expected
