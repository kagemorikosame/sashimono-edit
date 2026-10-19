"""実際に配布されているスクリプトが使っている書き方

資料に載っている書き方と、現物で使われている書き方は食い違うことがある
ここに並んでいるのは全部、実配布スクリプトを動かして見つかったもの
どれか 1 つ欠けるだけでスクリプトは 1 行目で落ちる
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.aviutl.control import lua_value, parse_control
from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image
from sashimono.effects.spec import CheckSpec, ColorSpec, SelectSpec, TextSpec


@pytest.fixture(scope="module")
def runtime() -> LuaScriptRuntime:
    return LuaScriptRuntime(instruction_limit=200_000)


def state(**fields: object) -> ObjectState:
    return ObjectState(image=blank_image(32, 32), **fields)  # type: ignore[arg-type]


class TestHeaderSyntax:
    def test_select_with_the_default_on_the_label(self) -> None:
        # --select@font_style:文字修飾=3,標準文字=0,影付き文字=1,…
        spec = parse_control(
            "--select@font_style:文字修飾=3,標準文字=0,影付き文字=1,縁取り文字=3"
        ).parameters[0]
        assert isinstance(spec, SelectSpec)
        assert spec.label == "文字修飾"
        assert spec.default == "3"
        assert dict(spec.choices)["0"] == "標準文字"

    def test_check_accepts_true_and_false(self) -> None:
        # --check@bold:太字,true
        spec = parse_control("--check@bold:太字,true").parameters[0]
        assert isinstance(spec, CheckSpec)
        assert spec.default is True
        other = parse_control("--check@x:X,false").parameters[0]
        assert isinstance(other, CheckSpec)
        assert other.default is False

    def test_the_text_control(self) -> None:
        # --text@caption:字幕,ここを強調
        spec = parse_control("--text@caption:字幕,ここを強調").parameters[0]
        assert isinstance(spec, TextSpec)
        assert (spec.name, spec.label, spec.default) == ("caption", "字幕", "ここを強調")

    def test_the_font_control_keeps_its_default(self) -> None:
        # 既定の書体を落とすと、配布スクリプトのフォントの欄が作者の選んだ書体ではなく
        # Yu Gothic UI で始まる
        spec = parse_control("--font@font_name:フォント,Meiryo UI").parameters[0]
        assert spec.default_value() == "Meiryo UI"

    def test_information_is_treated_as_a_label(self) -> None:
        # --information を見出しとして受けないと、作者と版の行が ScriptHeader.labels に入らず、
        # 呼び手がスクリプトの説明を出せない
        header = parse_control("--information:マーカー字幕ミニ v1.1.0 by やまたな")
        assert header.labels and "やまたな" in header.labels[0]


class TestValueTypes:
    def test_colours_arrive_as_numbers(self) -> None:
        # AviUtl では色は 0xRRGGBB の数値 タプルのまま渡すと
        # ``color / 65536`` のような計算でいきなり落ちる
        spec = ColorSpec("col", "色", (1.0, 0.5, 0.0, 1.0))
        assert lua_value(spec, (1.0, 0.5, 0.0, 1.0)) == 0xFF7F00

    def test_checks_arrive_as_booleans(self) -> None:
        # 配布スクリプトは ``if bold then`` と書く
        assert lua_value(CheckSpec("bold", "太字", True), 1) is True

    def test_numeric_choices_arrive_as_numbers(self) -> None:
        # 番号の選択肢を文字のまま渡すと、if style == 1 then と比べる配布スクリプトが
        # 選んだ選択肢の分かれ道に入らない
        spec = SelectSpec("style", "型", (("0", "A"), ("1", "B")), "1")
        assert lua_value(spec, "1") == 1

    def test_non_numeric_choices_stay_text(self) -> None:
        # 数字でない識別子を数へ直そうとして崩すと、mode == "curve" と比べるスクリプトに
        # 選んだ選択肢が届かない
        spec = SelectSpec("mode", "型", (("line", "直線"), ("curve", "曲線")), "line")
        assert lua_value(spec, "curve") == "curve"


class TestGlobalVariables:
    def test_named_parameters_are_plain_globals(self, runtime: LuaScriptRuntime) -> None:
        # --track@offset_x:… と書いたスクリプトは obj.offset_x ではなく
        # 素の offset_x を読む 実配布スクリプトはどれもこの書き方
        target = state()
        target.values["offset_x"] = 25.0
        runtime.run("obj.ox = offset_x * 2", target)
        assert target.ox == 50.0

    def test_values_from_a_previous_run_do_not_leak(self, runtime: LuaScriptRuntime) -> None:
        # 残っていると、自分では設定していない値を読めてしまい、
        # 動いたり動かなかったりする
        first = state()
        first.values["only_here"] = 5.0
        runtime.run("obj.ox = only_here", first)

        second = state()
        second.values["marker"] = 1.0
        runtime.run("obj.oy = only_here ~= nil and 1 or 0", second)
        assert second.oy == 0.0

    def test_text_values_reach_the_script(self, runtime: LuaScriptRuntime) -> None:
        # 文字の値を大域変数へ渡せないと、--text の欄に入れた文字がスクリプトでは nil になり、
        # 文字を扱う行で落ちる
        target = state()
        target.values["caption"] = "見出し"
        runtime.run("obj.ox = string.len(caption)", target)
        assert target.ox == 9.0  # UTF-8 のバイト数 Lua の string.len と同じ


class TestScaleFields:
    def test_sx_and_sy_are_writable(self, runtime: LuaScriptRuntime) -> None:
        # 拡張描画のスクリプトは軸ごとの倍率を触る
        target = state()
        runtime.run("obj.sx = 2  obj.sy = 0.5", target)
        assert (target.sx, target.sy) == (2.0, 0.5)

    def test_reading_sx_before_writing_gives_one(self, runtime: LuaScriptRuntime) -> None:
        # nil だと ``keep_sx * scale`` のような計算で落ちる
        target = state()
        runtime.run("obj.ox = obj.sx", target)
        assert target.ox == 1.0

    def test_the_scale_reaches_the_draw(self, runtime: LuaScriptRuntime) -> None:
        # obj.sx を描く物へ渡し損ねると、軸ごとに引き伸ばすスクリプトの絵が元の倍率で描かれる
        result = runtime.run("obj.sx = 3", state())
        assert result.draws[0].sx == 3.0


class TestBufferNames:
    def test_object_and_obj_mean_the_same(self, runtime: LuaScriptRuntime) -> None:
        # "object" を obj と同じ物と見ないと、写すはずの絵が仮想バッファへ入らず、
        # あとで tmp を読み戻すスクリプトの絵が欠ける
        target = state()
        result = runtime.run('obj.copybuffer("tmp", "object")', target)
        assert result.failed is False
        assert "tmp" in target.buffers

    def test_tempbuffer_is_the_temporary_one(self, runtime: LuaScriptRuntime) -> None:
        # "tempbuffer" を tmp と同じ物と見ないと、写した絵が別の名前の置き場へ入り、
        # tmp から読み戻すスクリプトの絵が欠ける
        target = state()
        runtime.run('obj.copybuffer("tempbuffer", "obj")', target)
        assert "tmp" in target.buffers

    def test_copying_back(self, runtime: LuaScriptRuntime) -> None:
        # 仮想バッファから obj へ戻すときに絵の大きさが崩れると、取っておいた絵を戻して描く
        # スクリプトの絵がずれるか欠ける
        # 白紙を写し戻すだけだと、戻す処理が何もしなくても中身が崩れても見分けられない
        # 模様の付いた絵を取っておき、obj を別の大きさへ差し替えてから戻す
        target = state()
        pattern = np.arange(32 * 32 * 4, dtype=np.uint32).reshape(32, 32, 4).astype(np.uint8)
        target.image = pattern.copy()
        runtime.run('obj.copybuffer("tmp", "obj")', target)
        target.image = blank_image(8, 8)
        result = runtime.run('obj.copybuffer("obj", "tmp")', target)
        assert result.failed is False
        assert target.image.shape == (32, 32, 4)
        assert np.array_equal(target.image, pattern)


class TestModules:
    def test_a_lua_module_is_loaded(self, tmp_path: Path) -> None:
        # 置き場の .lua を obj.module で読めないと、共通の部品を使う配布スクリプトが
        # nil を引いて落ちる
        (tmp_path / "共通.lua").write_text(
            "return { twice = function(v) return v * 2 end }", "utf-8"
        )
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))

        target = state()
        result = runtime.run('local m = obj.module("共通")  obj.ox = m.twice(21)', target)
        assert result.failed is False
        assert target.ox == 42.0

    def test_require_works_the_same_way(self, tmp_path: Path) -> None:
        # require で置き場の .lua を引けないと、部品を require する配布スクリプトが頭で止まる
        (tmp_path / "共通.lua").write_text("return { value = 7 }", "utf-8")
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        target = state()
        runtime.run('obj.ox = require("共通").value', target)
        assert target.ox == 7.0

    def test_an_unusable_native_module_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        # AviUtl2 の .mod2 は中身が DLL のことがある 読めない物（64bit の Windows の
        # DLL でない）で黙って nil を返すと「なぜか動かない」で終わる
        (tmp_path / "ネイティブ.mod2").write_bytes(b"MZ\x90\x00" + b"\x00" * 64)
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=200_000)
        runtime.set_roots((tmp_path,))

        runtime.run('local m = obj.module("ネイティブ")', state())
        assert any("ネイティブ" in line and "DLL" in line for line in report.lines())

    def test_modules_outside_the_script_folder_are_refused(self, tmp_path: Path) -> None:
        # 任意のパスを開けると、読み込んだだけでディスクを読まれうる
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        runtime.run('obj.module("../../秘密")', state())
        assert any("見つかりません" in line for line in report.lines())

    def test_a_module_is_only_executed_once(self, tmp_path: Path) -> None:
        # 読むたびに部品を走らせ直すと、部品の頭で数えた値や作った表が呼ぶたびにやり直され、
        # Lua の require と違う結果になる
        (tmp_path / "数え.lua").write_text(
            "count = (count or 0) + 1\nreturn { n = count }", "utf-8"
        )
        runtime = LuaScriptRuntime(instruction_limit=200_000)
        runtime.set_roots((tmp_path,))
        target = state()
        runtime.run(
            'local a = obj.module("数え")  local b = obj.module("数え")  obj.ox = b.n', target
        )
        assert target.ox == 1.0


class TestIdentifiers:
    def test_obj_id_is_a_number(self, runtime: LuaScriptRuntime) -> None:
        # nil だと比較や演算で落ちる
        target = state(index=3)
        runtime.run("obj.ox = obj.id", target)
        assert target.ox == 3.0

    def test_getvalue_accepts_the_track_prefix(self, runtime: LuaScriptRuntime) -> None:
        # 頭の track. を外さないと、そう書く AviUtl2 のスクリプトが自分の欄の値ではなく 0 を受け、
        # 欄を動かしても絵が変わらない
        target = state()
        target.values["animation_size"] = 120.0
        runtime.run('obj.ox = obj.getvalue("track.animation_size")', target)
        assert target.ox == 120.0
