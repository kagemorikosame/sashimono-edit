"""AviUtl2 のスクリプトモジュール（DLL）を呼ぶ所

表（``SCRIPT_MODULE_PARAM``）の並びが 1 つでもずれると、DLL は別の関数を呼んで
黙って誤った値を読む 絵は出るが、どこかが少しずつおかしい、という壊れ方になる
配布物の DLL はリポジトリに入れられないので、**Python で作った関数を C の関数として
渡し**、DLL と同じ作法（表の関数を呼んで引数を読み、結果を積む）で確かめる
"""

from __future__ import annotations

import ctypes
import struct
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.compat.aviutl import native
from sashimono.compat.aviutl.native import (
    NativeModule,
    NativeModuleError,
    PixelData,
    _FunctionEntry,
    _ModuleFunction,
    _ModuleTable,
    _Param,
    is_native_x64,
)
from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image


def _module(functions: dict[str, Callable[[Any], None]]) -> tuple[NativeModule, list[Any]]:
    """Python の関数を並べた、DLL の代わりのモジュール 2 つ目は生かしておく物"""
    keep: list[Any] = []
    entries = (_FunctionEntry * (len(functions) + 1))()
    for index, (name, body) in enumerate(functions.items()):
        function = _ModuleFunction(lambda param, body=body: body(param.contents))
        keep.append(function)
        entries[index].name = name
        entries[index].func = function
    table = _ModuleTable(information="試験用", functions=entries)
    keep.extend([entries, table])
    return NativeModule(Path("試験.mod2"), None, table), keep


@pytest.fixture(autouse=True)
def allowed() -> Iterator[None]:
    """試験のたびに設定を戻す 切った試験のあとに、ほかの試験が読めなくなる"""
    native.set_enabled(True)
    yield
    native.set_enabled(True)


class TestReadingTheArguments:
    """DLL が引数を読む関数 C の並び（0 から）と Lua の並び（1 から）の食い違いも見る"""

    def test_numbers_strings_and_counts(self) -> None:
        seen: dict[str, Any] = {}

        def body(p: _Param) -> None:
            seen["count"] = p.get_param_num()
            seen["int"] = p.get_param_int(0)
            seen["double"] = p.get_param_double(1)
            seen["string"] = ctypes.string_at(p.get_param_string(2)).decode("utf-8")
            seen["missing"] = p.get_param_int(9)
            seen["bool"] = p.get_param_boolean(3)

        module, _keep = _module({"f": body})
        module.call("f", [7, 2.5, "字幕", True])
        assert seen == {
            "count": 4,
            "int": 7,
            "double": 2.5,
            "string": "字幕",
            "missing": 0,
            "bool": True,
        }

    def test_pixel_data_arrives_as_its_address(self) -> None:
        """画素は番地で渡る DLL はその番地を直に読み書きする"""
        data = PixelData(np.zeros((2, 3, 4), np.uint8))
        seen: list[int] = []
        module, _keep = _module({"f": lambda p: seen.append(p.get_param_data(0))})
        module.call("f", [data])
        assert seen == [data.address]

    def test_what_the_dll_writes_reaches_the_pixels(self) -> None:
        """DLL が番地へ書いた画素は、そのまま Sashimono から見える（複製を渡していない）"""
        data = PixelData(np.zeros((1, 1, 4), np.uint8))

        def body(p: _Param) -> None:
            ctypes.memmove(p.get_param_data(0), bytes([1, 2, 3, 4]), 4)

        module, _keep = _module({"f": body})
        module.call("f", [data])
        assert data.pixels[0, 0].tolist() == [1, 2, 3, 4]

    def test_tables_by_key_and_by_position(self) -> None:
        """配列の位置は C では 0 から Lua の表は 1 から 取り違えると 1 つずれて読む"""
        seen: dict[str, Any] = {}

        def body(p: _Param) -> None:
            seen["key"] = p.get_param_table_int(0, b"size")
            seen["length"] = p.get_param_array_num(1)
            seen["first"] = p.get_param_array_int(1, 0)
            seen["last"] = p.get_param_array_double(1, 2)

        module, _keep = _module({"f": body})
        module.call("f", [{"size": 48}, {1: 10, 2: 20, 3: 30.5}])
        assert seen == {"key": 48, "length": 3, "first": 10, "last": 30.5}

    def test_the_type_of_each_argument(self) -> None:
        # PARAM_TYPE の値（module2.h） DLL はこれを見て読み方を変える
        seen: list[int] = []

        def body(p: _Param) -> None:
            seen.extend(p.get_param_type(i) for i in range(6))

        module, _keep = _module({"f": body})
        module.call("f", [None, True, 1.0, "文字", {1: 2}])
        assert seen == [0, 1, 3, 4, 5, -1]


class TestTheResults:
    def test_results_come_back_in_order(self) -> None:
        def body(p: _Param) -> None:
            p.push_result_int(3)
            values = (ctypes.c_int * 4)(10, 5, 73, 30)
            p.push_result_array_int(values, 4)
            p.push_result_string("板".encode())

        module, _keep = _module({"f": body})
        assert module.call("f", []) == [3, [10, 5, 73, 30], "板"]

    def test_a_table_result(self) -> None:
        def body(p: _Param) -> None:
            keys = (ctypes.c_char_p * 2)(b"w", b"h")
            values = (ctypes.c_double * 2)(1.5, 2.5)
            p.push_result_table_double(keys, values, 2)

        module, _keep = _module({"f": body})
        assert module.call("f", []) == [{"w": 1.5, "h": 2.5}]


class TestWhenItGoesWrong:
    def test_an_error_from_the_dll_is_raised(self) -> None:
        """DLL が失敗を伝えてきたら投げる 黙って空の結果を返すと、原因が分からない"""
        module, _keep = _module({"f": lambda p: p.set_error("読めない画像".encode())})
        with pytest.raises(NativeModuleError, match="読めない画像"):
            module.call("f", [])

    def test_an_unsupported_return_is_refused(self) -> None:
        """関数を返すモジュールには対応していない 呼ばれても落とさずに断る"""
        module, _keep = _module({"f": lambda p: p.push_result_function(None, None)})
        with pytest.raises(NativeModuleError, match="対応していない"):
            module.call("f", [])

    def test_an_unknown_function(self) -> None:
        module, _keep = _module({"f": lambda p: None})
        with pytest.raises(NativeModuleError, match="無い"):
            module.call("g", [])


def _pe(path: Path, machine: int) -> Path:
    """PE の見出しだけの物 読み込めるかの見分けに使う"""
    head = bytearray(0x100)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    head[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<H", head, 0x44, machine)
    path.write_bytes(bytes(head))
    return path


class TestWhatIsLoaded:
    def test_a_64bit_dll_is_recognised(self, tmp_path: Path) -> None:
        assert is_native_x64(_pe(tmp_path / "a.mod2", 0x8664))

    def test_a_32bit_dll_is_not(self, tmp_path: Path) -> None:
        """32bit の DLL は 64bit の Sashimono へ読み込めない 読みに行くと落ちる"""
        assert not is_native_x64(_pe(tmp_path / "a.mod2", 0x014C))

    def test_text_is_not(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mod2"
        path.write_text("return {}", "utf-8")
        assert not is_native_x64(path)

    def test_turned_off_it_is_not_loaded(self, tmp_path: Path) -> None:
        """切ったら読まない 読んでから使わない、では DLL の初期化が走ってしまう"""
        native.set_enabled(False)
        with pytest.raises(NativeModuleError, match="読まない設定"):
            native.load(_pe(tmp_path / "a.mod2", 0x8664))

    def test_only_on_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        with pytest.raises(NativeModuleError, match="Windows"):
            native.load(_pe(tmp_path / "a.mod2", 0x8664))


def _state() -> ObjectState:
    return ObjectState(image=blank_image(8, 8))


class TestFromLua:
    """``obj.module`` から DLL の関数を呼ぶ道"""

    def test_obj_module_takes_the_mod2_and_require_the_lua(self, tmp_path: Path) -> None:
        """同じ名前の ``.lua`` と ``.mod2`` があるとき、``obj.module`` は ``.mod2`` を読む

        ``require`` と同じく ``.lua`` を先に拾うと、テレビ字幕の ``obj.module``
        に Lua の方（``create_polygons``）が返って、DLL の ``scan`` が nil になっていた
        """
        (tmp_path / "共通.lua").write_text('return { kind = "lua" }', "utf-8")
        (tmp_path / "共通.mod2").write_text('return { kind = "mod2" }', "utf-8")
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        state = _state()
        runtime.run(
            'kinds = obj.module("共通").kind .. "," .. require("共通").kind',
            state,
        )
        assert runtime._lua.globals()["kinds"] == "mod2,lua"

    def test_without_a_mod2_it_falls_back(self, tmp_path: Path) -> None:
        # これまで obj.module で .lua を読んでいたスクリプトを壊さない
        (tmp_path / "共通.lua").write_text('return { kind = "lua" }', "utf-8")
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        runtime.run('kind = obj.module("共通").kind', _state())
        assert runtime._lua.globals()["kind"] == "lua"

    def test_a_dll_function_is_called_with_lua_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lua の表は辞書に、DLL の配列は Lua の表に 1 から数える形で渡る"""
        (tmp_path / "板.mod2").write_bytes(b"MZ")
        seen: list[Any] = []

        class Fake:
            names = ("scan",)

            def call(self, name: str, args: list[Any]) -> list[Any]:
                seen.append(args)
                return [[10, 5, 73, 30]]

        monkeypatch.setattr(native, "load", lambda path: Fake())
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        state = _state()
        runtime.run(
            'local r = obj.module("板").scan(3, {1, 2}, {a = 5})\nobj.ox = r[1] + r[4]', state
        )
        assert seen == [[3, {1: 1, 2: 2}, {"a": 5}]]
        assert state.ox == 40

    def test_a_dll_error_fails_only_that_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DLL が失敗したら、そのスクリプトの失敗として記録する（フレームごと落とさない）"""
        (tmp_path / "板.mod2").write_bytes(b"MZ")

        class Broken:
            names = ("scan",)

            def call(self, name: str, args: list[Any]) -> list[Any]:
                raise NativeModuleError("板.mod2 の scan: 読めない")

        monkeypatch.setattr(native, "load", lambda path: Broken())
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        result = runtime.run('obj.module("板").scan()', _state(), script="テレビ字幕")
        assert result.failed
        assert any("読めない" in line for line in report.lines())

    def test_turned_off_it_says_so(self, tmp_path: Path) -> None:
        """切ってあるときは、理由を残して nil を返す 黙って nil だと原因が分からない"""
        _pe(tmp_path / "板.mod2", 0x8664)
        native.set_enabled(False)
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        runtime.run('m = obj.module("板")', _state())
        assert any("読まない設定" in line for line in report.lines())

    def test_turning_it_off_later_stops_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """読んだあとで切っても、もう使わない 読んだ物を控えから返し続けると、切った意味が無い"""
        (tmp_path / "板.mod2").write_bytes(b"MZ")

        class Fake:
            names = ("scan",)

            def call(self, name: str, args: list[Any]) -> list[Any]:
                return [1]

        monkeypatch.setattr(native, "load", lambda path: Fake())
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        runtime.run('first = obj.module("板")', _state())
        native.set_enabled(False)
        runtime.run('second = obj.module("板")', _state())
        assert runtime._lua.globals()["first"] is not None
        assert runtime._lua.globals()["second"] is None


class TestRoundTwo:
    """レビューで見つかった穴 どれも直す前の作りで落ちることを確かめた"""

    def test_a_kept_function_stops_when_turned_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """取っておいた関数も、切ったら呼べない

        スクリプトが関数の表を大域変数へ取っておくと、設定を切ったあとも
        同じ表から DLL を呼べてしまう（切った意味が無い）
        """
        (tmp_path / "板.mod2").write_bytes(b"MZ")
        calls: list[str] = []

        class Fake:
            names = ("scan",)
            path = tmp_path / "板.mod2"

            def call(self, name: str, args: list[Any]) -> list[Any]:
                calls.append(name)
                return [1]

        monkeypatch.setattr(native, "load", lambda path: Fake())
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        runtime.run('kept = obj.module("板")', _state())
        native.set_enabled(False)
        result = runtime.run("kept.scan()", _state())
        assert calls == []
        assert result.failed

    def test_the_pe_header_can_be_far_away(self, tmp_path: Path) -> None:
        """見出しが先頭から遠くにある正しい DLL も読む（前置きの長い DLL）"""
        head = bytearray(0x1000)
        head[:2] = b"MZ"
        struct.pack_into("<I", head, 0x3C, 0x800)
        head[0x800:0x804] = b"PE\0\0"
        struct.pack_into("<H", head, 0x804, 0x8664)
        path = tmp_path / "遠い.mod2"
        path.write_bytes(bytes(head))
        assert is_native_x64(path)

    def test_the_same_name_in_two_folders_stays_apart(self, tmp_path: Path) -> None:
        """別の配布物の同じ名前のモジュールは、それぞれ自分の物を受け取る

        名前だけで控えると、後から走ったスクリプトにも先に読んだ方（DLL を含む）が渡る
        """
        first, second = tmp_path / "一", tmp_path / "二"
        for folder, kind in ((first, "first"), (second, "second")):
            folder.mkdir()
            (folder / "共通.mod2").write_text(f'return {{ kind = "{kind}" }}', "utf-8")
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.run('a = obj.module("共通").kind', _state(), folder=first)
        runtime.run('b = obj.module("共通").kind', _state(), folder=second)
        names = runtime._lua.globals()
        assert (names["a"], names["b"]) == ("first", "second")

    def test_a_module_without_a_list_is_refused(self) -> None:
        """関数の一覧を持たない DLL は、落とさずに断る"""
        with pytest.raises(NativeModuleError, match="一覧"):
            NativeModule(Path("空.mod2"), None, _ModuleTable(information="", functions=None))

    def test_a_list_without_an_end_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """終わりの印が無い一覧は、どこまでも読みに行かずに断る"""
        monkeypatch.setattr(native, "MAX_FUNCTIONS", 2)
        keep: list[Any] = []
        entries = (_FunctionEntry * 3)()
        for index in range(3):
            function = _ModuleFunction(lambda param: None)
            keep.append(function)
            entries[index].name = f"f{index}"
            entries[index].func = function
        table = _ModuleTable(information="", functions=entries)
        with pytest.raises(NativeModuleError, match="終わり"):
            NativeModule(Path("長い.mod2"), None, table)


class TestRoundThree:
    def test_exactly_the_limit_of_functions_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """上限ちょうどの数の関数と、その次の終わりの印 正しい一覧なので断らない"""
        monkeypatch.setattr(native, "MAX_FUNCTIONS", 2)
        module, _keep = _module({"f0": lambda p: None, "f1": lambda p: None})
        assert module.names == ("f0", "f1")

    def test_a_table_too_deep_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """深すぎる表は、黙って空にせず呼ぶのをやめる

        空にして渡すと、DLL は切れた引数を正しい値として読む
        """
        (tmp_path / "板.mod2").write_bytes(b"MZ")
        calls: list[Any] = []

        class Fake:
            names = ("scan",)
            path = tmp_path / "板.mod2"

            def call(self, name: str, args: list[Any]) -> list[Any]:
                calls.append(args)
                return []

        monkeypatch.setattr(native, "load", lambda path: Fake())
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        result = runtime.run(
            'local t = {}\nt.self = t\nobj.module("板").scan(t)', _state(), script="深い"
        )
        assert calls == []
        assert result.failed


class TestEditSection:
    """``SCRIPT_MODULE_PARAM.edit`` の先 plugin2.h の ``EDIT_SECTION`` の並び

    DLL と同じ読み方をする 先頭の枠を ``EDIT_INFO*`` として読み、関数の枠は
    番地として読む（呼ぶと落ちうるので、中身は番地の一致で確かめる）
    """

    @staticmethod
    def _seen_by_dll() -> dict[str, Any]:
        seen: dict[str, Any] = {}

        def body(p: _Param) -> None:
            section = ctypes.cast(p.edit, ctypes.POINTER(ctypes.c_void_p * 83)).contents
            # EDIT_INFO は int が 15 個 float int float int と 4 バイトの色（80 バイト）
            info = ctypes.cast(section[0] or 0, ctypes.POINTER(ctypes.c_int * 20)).contents
            seen["ints"] = list(info[:15])
            seen["first_function"] = section[1]
            seen["last_function"] = section[82]
            # 同じ関数を呼んで 0 が返ることも見る 先頭の関数は必ず表の中にある
            stub = ctypes.CFUNCTYPE(ctypes.c_int64)(section[1] or 0)
            seen["returned"] = stub()

        module, _keep = _module({"f": body})
        module.call("f", [])
        return seen

    def test_the_first_slot_is_edit_info_not_a_function(self) -> None:
        """先頭に関数の番地を置くと、``edit->info->width`` を読むプラグインが
        関数の機械語を解像度として読み、でたらめな大きさで組む
        """
        ints = self._seen_by_dll()["ints"]
        assert ints[:13] == [0] * 13

    def test_no_selection_is_minus_one(self) -> None:
        """0 にすると「0 フレーム目を選んでいる」と読まれる plugin2.h では未選択は -1"""
        assert self._seen_by_dll()["ints"][13:15] == [-1, -1]

    def test_the_last_function_slot_is_inside_the_table(self) -> None:
        """枠を 1 つ少なく作ると、最後の関数を引くプラグインは表の外の番地へ飛ぶ"""
        seen = self._seen_by_dll()
        assert seen["last_function"] == seen["first_function"]
        assert seen["returned"] == 0


class TestKeepingTheRuntime:
    def test_a_runtime_that_read_a_dll_is_never_let_go(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DLL のモジュールを読んだ Lua を手放すと、片付けの中でアクセス違反で落ちた
        # （AviUtl2 の配布物の DLL 棚の見本で続けて描いた後のごみ集め #277）
        from sashimono.compat.aviutl import runtime as runtime_module

        module, keep = _module({"noop": lambda _param: None})
        monkeypatch.setattr(native, "load", lambda _path: module)
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        assert runtime._native_module(tmp_path / "試験.mod2", "試験") is not None
        assert runtime in runtime_module._PINNED_RUNTIMES
        # 2 度読んでも 1 つだけ持つ
        runtime._native_module(tmp_path / "試験.mod2", "試験")
        assert runtime_module._PINNED_RUNTIMES.count(runtime) == 1
        assert keep

    def test_a_plain_runtime_is_let_go(self) -> None:
        # DLL を読まないランタイムまで持ち続けると、書き出しや静止画のたびに溜まっていく
        from sashimono.compat.aviutl import runtime as runtime_module

        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.run("x = 1", _state())
        assert runtime not in runtime_module._PINNED_RUNTIMES
