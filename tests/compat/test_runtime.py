"""Lua ランタイムと ``obj`` API

配布スクリプトは他人が書いたコードで、読み込むだけで走る サンドボックスと
実行時間の上限は「動くこと」と同じくらい大事なので、そこも見る
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.aviutl.objapi import MAX_BUFFERS, ObjectState
from sashimono.compat.aviutl.report import MAX_KINDS, CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image, lua_available


@pytest.fixture(scope="module")
def runtime() -> LuaScriptRuntime:
    """使い回すランタイム 作るのに数ミリ秒かかる"""
    return LuaScriptRuntime(instruction_limit=200_000)


def state(width: int = 64, height: int = 64, **fields: object) -> ObjectState:
    return ObjectState(image=blank_image(width, height), **fields)  # type: ignore[arg-type]


class TestAvailability:
    def test_lua_is_there_by_default(self) -> None:
        # P5 の前提 追加導入なしで動くこと
        assert lua_available() is True

    def test_it_is_lua_5_1(self, runtime: LuaScriptRuntime) -> None:
        # AviUtl 本体が 5.1 配布スクリプトは unpack や setfenv を普通に使う
        target = state()
        runtime.run("obj.ox = (_VERSION == 'Lua 5.1') and 1 or 0", target)
        assert target.ox == 1.0

    def test_five_one_only_functions_exist(self, runtime: LuaScriptRuntime) -> None:
        # unpack と setfenv が無いと、それを使う配布スクリプトが落ち、そのクリップは
        # 効果の無い元の絵で描かれる
        target = state()
        runtime.run("obj.ox = (unpack ~= nil and setfenv ~= nil) and 1 or 0", target)
        assert target.ox == 1.0


class TestFields:
    def test_writable_fields_round_trip(self, runtime: LuaScriptRuntime) -> None:
        # obj.ox などの代入が状態へ戻らないと、スクリプトで動かした位置・拡大・透明度・回転が
        # プレビューにも書き出しにも出ない
        target = state()
        runtime.run("obj.ox=1 obj.oy=2 obj.zoom=3 obj.alpha=0.5 obj.rz=90 obj.aspect=0.25", target)
        assert (target.ox, target.oy, target.zoom) == (1.0, 2.0, 3.0)
        assert (target.alpha, target.rz, target.aspect) == (0.5, 90.0, 0.25)

    def test_size_is_read_from_the_image(self, runtime: LuaScriptRuntime) -> None:
        # obj.w と obj.h が絵の大きさを返さないと、大きさに合わせて並べるスクリプトが位置を誤る
        target = state(width=120, height=80)
        runtime.run("obj.ox = obj.w  obj.oy = obj.h", target)
        assert (target.ox, target.oy) == (120.0, 80.0)

    def test_time_comes_from_the_frame_and_rate(self, runtime: LuaScriptRuntime) -> None:
        # obj.time を frame と fps から作らないと、時間で動くスクリプトが止まるか速さが変わる
        target = state(frame=15, totalframe=60, framerate=30.0)
        runtime.run("obj.ox = obj.time  obj.oy = obj.totaltime  obj.oz = obj.frame", target)
        assert (target.ox, target.oy, target.oz) == (0.5, 2.0, 15.0)

    def test_track_values_are_visible(self, runtime: LuaScriptRuntime) -> None:
        # obj.track0〜3 で設定欄の値が読めないと、スライダーを動かしても効果が変わらない
        target = state()
        target.track = [10.0, 20.0, 30.0, 40.0]
        runtime.run("obj.ox = obj.track0 + obj.track3", target)
        assert target.ox == 50.0

    def test_check0_is_a_number(self, runtime: LuaScriptRuntime) -> None:
        # AviUtl では 0 か 1 真偽値で返すと ``obj.check0 == 1`` が偽になる
        target = state()
        target.check0 = True
        runtime.run("obj.ox = obj.check0", target)
        assert target.ox == 1.0

    def test_named_parameters_are_visible(self, runtime: LuaScriptRuntime) -> None:
        # 名前付きの値は、大域変数としても obj 越しにも読める 配布スクリプトは
        # 前者を使うが、どちらで書かれていても動くようにしてある
        target = state()
        target.values["amount"] = 7.0
        runtime.run("obj.ox = amount  obj.oy = obj.amount", target)
        assert (target.ox, target.oy) == (7.0, 7.0)

    def test_writing_a_read_only_field_is_recorded(self) -> None:
        # obj.w への代入を黙って捨てると、大きさが変わらない理由が互換性レポートに出ない
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        # 記録だけでなく、代入が効かないことも見る 効くと絵の大きさが AviUtl と違ってしまう
        # 既定の 64 と違う大きさにして、読み戻しで元の幅が返ることを確かめる
        target = state(32, 48)
        runtime.run("obj.w = 999  obj.ox = obj.w", target)
        assert any("obj.w" in line for line in report.lines())
        assert (target.width, target.height) == (32, 48)
        assert target.ox == 32.0
        assert "w" not in target.values


class TestDrawing:
    def test_without_an_explicit_draw_the_state_is_drawn_once(
        self, runtime: LuaScriptRuntime
    ) -> None:
        # obj.draw を呼ばないスクリプトで自動の 1 回が描かれないと、位置を動かすだけの効果で
        # クリップが消える
        result = runtime.run("obj.ox = 40", state())
        assert len(result.draws) == 1
        assert result.draws[0].x == 40.0

    def test_explicit_draws_replace_the_automatic_one(self, runtime: LuaScriptRuntime) -> None:
        # 残像や複製を作るスクリプトはこの仕組みで動いている
        result = runtime.run("for i=1,4 do obj.draw(i*10, 0, 0, 1, 0.25) end", state())
        assert [call.x for call in result.draws] == [10.0, 20.0, 30.0, 40.0]
        assert all(call.alpha == 0.25 for call in result.draws)

    def test_draw_without_arguments_uses_the_current_values(
        self, runtime: LuaScriptRuntime
    ) -> None:
        # 引数を省いた obj.draw がいまの obj.ox や obj.rz を使わないと、動かしてから描く
        # スクリプトが元の位置と向きで描く
        result = runtime.run("obj.ox = 12 obj.rz = 30 obj.draw()", state())
        assert (result.draws[0].x, result.draws[0].rz) == (12.0, 30.0)

    def test_drawpoly_becomes_a_draw_on_a_quad(self) -> None:
        # 記録だけで描かないと、立体的に動く配布スクリプトが何も映さない
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        result = runtime.run(
            "obj.ox = 5 obj.drawpoly(-10,-10,0, 10,-10,0, 10,10,20, -10,10,20,"
            " 0,0, 4,0, 4,4, 0,4, 0.5)",
            state(),
        )
        (call,) = result.draws
        assert call.quad == (
            (-5.0, -10.0, 0.0),
            (15.0, -10.0, 0.0),
            (15.0, 10.0, 20.0),
            (-5.0, 10.0, 20.0),
        )
        assert call.uv == ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
        assert call.alpha == 0.5
        assert not any("drawpoly" in line for line in report.lines())

    def test_each_draw_keeps_the_picture_of_its_time(self, runtime: LuaScriptRuntime) -> None:
        # 同じ配列を共有すると、後の putpixel が前に描いた分まで書き換える
        result = runtime.run(
            "obj.drawpoly(0,0,0, 1,0,0, 1,1,0, 0,1,0) obj.putpixel(0, 0, 0xff0000, 1) obj.draw()",
            state(),
        )
        first, second = result.draws
        assert tuple(second.image[0, 0]) == (255, 0, 0, 255)
        assert tuple(first.image[0, 0]) != (255, 0, 0, 255)

    def test_drawpoly_with_a_broken_uv_is_recorded(self) -> None:
        # 途中で切れた UV の先頭を透明度として黙って読むと、絵が薄くなった理由が追えない
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        result = runtime.run("obj.drawpoly(0,0,0, 1,0,0, 1,1,0, 0,1,0, 0,0, 4,0)", state())
        assert result.draws[0].uv is None
        # 先頭の UV の 0 を透明度と読むと、板ごと消える
        assert result.draws[0].alpha == 1.0
        assert any("drawpoly" in line for line in report.lines())

    def test_python_is_out_of_reach(self) -> None:
        # python.builtins が見えると、配布ファイルの Lua から手元のファイルを読み書きできる
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        reachable = "obj.ox = (python ~= nil and python.builtins ~= nil) and 1 or 0"
        result = runtime.run(reachable, state())
        assert result.state is not None
        assert result.state.ox == 0.0

    @pytest.mark.parametrize(
        "code",
        [
            'python.eval("1")',
            "local g = RGB.__globals__",
            "local g = debug_print.__globals__",
        ],
    )
    def test_python_internals_cannot_be_walked(self, code: str) -> None:
        # 渡した関数の __globals__ を辿れると、__import__ から手元のファイル操作や
        # プロセスの起動まで届く python.eval が戻っても同じ
        runtime = LuaScriptRuntime(report=CompatibilityReport(), instruction_limit=100_000)
        assert runtime.run(code, state()).failed

    def test_a_script_cannot_eat_all_the_memory(self) -> None:
        # 配布ファイルを開いただけで、巨大な文字列を作られてソフトごと落ちないこと
        runtime = LuaScriptRuntime(report=CompatibilityReport(), instruction_limit=100_000)
        result = runtime.run('local s = string.rep("a", 1024 * 1024 * 1024)', state())
        assert result.failed
        assert "memory" in (result.message or "")
        # 断ったあとも同じランタイムで描ける
        assert runtime.run("obj.draw()", state()).draws

    def test_drawpoly_with_too_few_points_is_recorded(self) -> None:
        # 四隅が揃わないまま描くと、潰れた面が画面に広がる
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        result = runtime.run("obj.drawpoly(0,0,0, 1,0,0)", state())
        assert all(call.quad is None for call in result.draws)
        assert any("drawpoly" in line for line in report.lines())


class TestEffects:
    def test_a_known_filter_is_queued(self, runtime: LuaScriptRuntime) -> None:
        # obj.effect のぼかしが積まれないと、スクリプトが頼んでもプレビューと書き出しで
        # ぼけない
        target = state()
        result = runtime.run('obj.effect("ぼかし", "範囲", 20)', target)
        assert [request.kind for request in result.draws[0].effects] == ["blur"]
        assert result.draws[0].effects[0].params == {"範囲": 20.0}

    def test_an_unknown_filter_is_recorded(self) -> None:
        # 無いフィルタを黙って捨てると、効果が掛からない理由が互換性レポートに出ず、
        # 次に足すフィルタを数で選べない
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        runtime.run('obj.effect("まだ無いフィルタ")', state())
        assert any("まだ無いフィルタ" in line for line in report.lines())


class TestHelpers:
    def test_rgb_and_hsv(self, runtime: LuaScriptRuntime) -> None:
        # RGB と HSV の数がずれると、色を組み立てて渡すスクリプトが違う色で描く
        target = state()
        runtime.run("obj.ox = RGB(255,128,0)  obj.oy = HSV(0,255,255)", target)
        assert target.ox == 0xFF8000
        assert target.oy == 0xFF0000

    def test_bit_operations(self, runtime: LuaScriptRuntime) -> None:
        # OR・AND・XOR がずれると、色や旗をビットで組むスクリプトが違う値を読む
        target = state()
        runtime.run("obj.ox = OR(5,2) obj.oy = AND(6,3) obj.oz = XOR(5,3)", target)
        assert (target.ox, target.oy, target.oz) == (7.0, 2.0, 6.0)

    def test_shift(self, runtime: LuaScriptRuntime) -> None:
        # SHIFT の正負の向きを取り違えると、左へずらすはずの値を右へずらし、スクリプトが
        # 組む数が変わる
        target = state()
        runtime.run("obj.ox = SHIFT(1, 4)  obj.oy = SHIFT(16, -2)", target)
        assert (target.ox, target.oy) == (16.0, 4.0)

    def test_rand_is_stable_within_a_frame(self, runtime: LuaScriptRuntime) -> None:
        # 毎回ばらつくと、1 フレーム描き直すたびに絵が変わる
        first = state(frame=7)
        runtime.run("obj.ox = obj.rand(0, 1000)", first)
        second = state(frame=7)
        runtime.run("obj.ox = obj.rand(0, 1000)", second)
        assert first.ox == second.ox

    def test_rand_changes_between_frames(self, runtime: LuaScriptRuntime) -> None:
        # フレームで種が変わらないと、obj.rand で散らす揺れや粒が全フレームで同じ値に止まる
        values = set()
        for frame in range(8):
            target = state(frame=frame)
            runtime.run("obj.ox = obj.rand(0, 100000)", target)
            values.add(target.ox)
        assert len(values) > 1

    def test_interpolation_passes_through_the_points(self, runtime: LuaScriptRuntime) -> None:
        # 端の点を通らないと、obj.interpolation で道筋を描くスクリプトの始めと終わりが
        # 指定した位置からずれる
        target = state()
        runtime.run("obj.ox = obj.interpolation(0.0, 10, 20, 30)", target)
        assert target.ox == 10.0
        runtime.run("obj.oy = obj.interpolation(1.0, 10, 20, 30)", target)
        assert target.oy == 30.0


class TestTextAndFigures:
    def test_setfont_and_mes_make_an_image(self, qt_application: object) -> None:
        # obj.setfont と obj.mes で文字の絵ができないと、文字を描くスクリプトのクリップに
        # 何も映らない
        del qt_application
        from sashimono.core.model import GeneratedSource
        from sashimono.engine.sources import render_source

        def draw(kind: str, params: dict[str, object], width: int, height: int) -> np.ndarray:
            from sashimono.core.model import AnimatedValue

            wrapped = {
                name: AnimatedValue(float(value)) if isinstance(value, int | float) else value
                for name, value in params.items()
            }
            image = render_source(GeneratedSource(kind=kind, params=wrapped), width, height)  # type: ignore[arg-type]
            assert image is not None
            return image

        runtime = LuaScriptRuntime(render_source=draw, instruction_limit=200_000)
        target = state(width=200, height=100)
        target.screen_w, target.screen_h = 200, 100
        runtime.run('obj.setfont("Yu Gothic UI", 40, 1, 0xffffff)  obj.mes("あ")', target)
        assert target.image.shape == (100, 200, 4)
        assert int((target.image[..., 3] > 0).sum()) > 0

    def test_load_figure_uses_the_aviutl_names(self, qt_application: object) -> None:
        # AviUtl の図形名を写せないと、obj.load で 六角形 を頼んだスクリプトが円を描き、
        # 色もずれる
        del qt_application
        from sashimono.core.model import AnimatedValue, GeneratedSource
        from sashimono.engine.sources import render_source

        seen: dict[str, object] = {}

        def draw(kind: str, params: dict[str, object], width: int, height: int) -> np.ndarray:
            seen.update(params)
            wrapped = {
                name: AnimatedValue(float(value)) if isinstance(value, int | float) else value
                for name, value in params.items()
            }
            image = render_source(GeneratedSource(kind=kind, params=wrapped), width, height)  # type: ignore[arg-type]
            assert image is not None
            return image

        runtime = LuaScriptRuntime(render_source=draw, instruction_limit=200_000)
        runtime.run('obj.load("figure", "六角形", 0xff0000, 80)', state(200, 200))
        assert seen["shape"] == "hexagon"
        assert seen["color"] == (1.0, 0.0, 0.0, 1.0)


class TestSandbox:
    def test_file_access_is_closed(self, runtime: LuaScriptRuntime) -> None:
        # io が開いていると、配布スクリプトを読み込んだだけで手元のファイルを読み書きされうる
        result = runtime.run('local f = io.open("秘密.txt")', state())
        assert result.failed is True

    def test_process_access_is_closed(self, runtime: LuaScriptRuntime) -> None:
        # os が開いていると、配布スクリプトから別のプログラムを起動されうる
        assert runtime.run('os.execute("calc")', state()).failed is True

    def test_require_only_reaches_the_script_folders(self, runtime: LuaScriptRuntime) -> None:
        # require は塞がずに、スクリプトフォルダの中だけへ向けてある 配布物は
        # 共通処理を別ファイルへ切り出しており、塞ぐと 1 行目で落ちる
        target = state()
        result = runtime.run('obj.ox = require("どこにも無い") == nil and 1 or 0', target)
        assert result.failed is False
        assert target.ox == 1.0

    def test_require_cannot_escape_the_folders(self, runtime: LuaScriptRuntime) -> None:
        # ../ でスクリプトフォルダの外へ出られると、配布スクリプトの require でディスク上の
        # 別の Lua を走らせられる
        target = state()
        runtime.run('obj.ox = require("../../秘密") == nil and 1 or 0', target)
        assert target.ox == 1.0

    def test_an_infinite_loop_is_cut_off(self, runtime: LuaScriptRuntime) -> None:
        # 掛けておかないと、編集画面が戻ってこなくなる
        result = runtime.run("while true do end", state())
        assert result.failed is True
        assert "長すぎ" in result.message

    def test_a_failure_still_draws_the_original(self, runtime: LuaScriptRuntime) -> None:
        # 1 つのスクリプトの失敗でフレームが真っ黒になる方が困る
        result = runtime.run("error('わざと')", state())
        assert result.failed is True
        assert len(result.draws) == 1

    def test_a_syntax_error_is_reported(self, runtime: LuaScriptRuntime) -> None:
        # 構文の誤りで failed が立たないと、シーンチェンジのスクリプトが走らなかったと
        # 分からず、前の場面のまま切り替わらない
        result = runtime.run("this is not lua", state())
        assert result.failed is True

    def test_failures_are_recorded(self) -> None:
        # 失敗をスクリプト名つきで記録しないと、互換性レポートにどのスクリプトが落ちたかが
        # 出ない
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
        runtime.run("error('だめ')", state(), script="ためし.anm")
        assert any("ためし.anm" in line for line in report.lines())


#: 別のプロセスで 1 本のスクリプトを走らせ、失敗したかを出す 守りが壊れると
#: 戻ってこないので、同じプロセスで走らせるとテスト全体が止まる
_ISOLATED = """
import sys
from pathlib import Path
from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image
runtime = LuaScriptRuntime(report=CompatibilityReport(), instruction_limit=200_000)
if len(sys.argv) > 2:
    runtime.set_roots((Path(sys.argv[2]),))
result = runtime.run(sys.argv[1], ObjectState(image=blank_image(8, 8)))
print("failed" if result.failed else "ok")
"""


def _run_isolated(code: str, *roots: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _ISOLATED, code, *map(str, roots)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


class TestLimits:
    """スクリプトが上限から逃げられないこと 逃げられると編集画面が戻らない"""

    @pytest.mark.parametrize(
        "code",
        [
            "coroutine.wrap(function() while true do end end)()",
            "local co = coroutine.create(function() while true do end end)"
            " assert(coroutine.resume(co))",
            "while true do pcall(function() while true do end end) end",
            "while true do xpcall(function() while true do end end,"
            " function() while true do end end) end",
            "debug.sethook() while true do end",
            'string.find(string.rep("a", 3000), "a-a-a-a-a-b")',
            'string.match(string.rep("a", 200), ".-.-.-.-.-.-b")',
        ],
    )
    def test_the_limit_cannot_be_escaped(self, code: str) -> None:
        # コルーチン、pcall、xpcall の後始末、フックの取り外し、パターン照合（C の中で回る）
        assert _run_isolated(code) == "failed"

    def test_a_module_with_an_endless_loop_is_cut_off(self, tmp_path: Path) -> None:
        # Python から直に実行すると命令数のフックが掛からず、require の 1 行で固まる
        (tmp_path / "ループ.lua").write_text("while true do end return {}", encoding="utf-8")
        assert _run_isolated('local m = require("ループ") obj.ox = 1', tmp_path) == "ok"

    def test_ordinary_patterns_and_xpcall_still_work(self, runtime: LuaScriptRuntime) -> None:
        # 守りのために、配布スクリプトがよく使う書き方まで断ると本末転倒
        target = state()
        code = (
            'local a, b, c = string.match("12,34,56", "(%d+),(%d+),(%d+)") '
            'local ok, message = xpcall(function() error("x") end, function() return "拾った" end) '
            'obj.ox = a + b + c + ((not ok and message == "拾った") and 1 or 0)'
        )
        assert runtime.run(code, target).failed is False
        assert target.ox == 103.0

    def test_the_report_does_not_grow_without_end(self) -> None:
        # 毎回違う名前を記録させると、記録だけでメモリを使い切る
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report, instruction_limit=5_000_000)
        runtime.run("for i = 1, 5000 do debug_print(i) end", state())
        assert len(report.missing) <= MAX_KINDS + 1

    def test_named_buffers_are_limited(self) -> None:
        # 毎回違う名前で写されると、画面 1 枚ぶんずつ溜まる
        runtime = LuaScriptRuntime(report=CompatibilityReport(), instruction_limit=200_000)
        target = state()
        runtime.run('for i = 1, 100 do obj.copybuffer("cache:" .. i, "obj") end', target)
        assert len(target.buffers) <= MAX_BUFFERS


class TestSetup:
    def test_param_code_runs_before_the_script(self, runtime: LuaScriptRuntime) -> None:
        # --param の初期化が本文より先に走らないと、そこで決めた値を読む本文が nil の計算で
        # 落ちる
        from sashimono.compat.aviutl.control import parse_control

        header = parse_control("--param:base=5;")
        target = state()
        runtime.run("obj.ox = base * 2", target, header=header)
        assert target.ox == 10.0
