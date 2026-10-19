"""``require`` がスクリプトフォルダの深い所に置かれたモジュールも見つけること（#190）

PSDToolKit の字幕表示はテキスト欄の Lua で ``subobj=require("PSDToolKit").subtitle:mes(…)``
と書き、同じオブジェクトに積んだ 吹き出し がその ``subobj`` を読む テキスト欄の Lua には
スクリプト自身のフォルダが無いので、探すのは置き場とその 1 段下だけだった 配布物を
リポジトリのまま置く（``aviutl_psdtoolkit/src/lua/PSDToolKit.lua``）と見つからず、
``subobj`` が作られないまま 吹き出し が 1 行目で落ちていた
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime


def _state() -> ObjectState:
    return ObjectState(image=np.zeros((1, 1, 4), np.uint8), screen_w=320, screen_h=180)


def _module(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'return {{ name = "{value}" }}', encoding="utf-8")


def _runtime(root: Path, report: CompatibilityReport | None = None) -> LuaScriptRuntime:
    runtime = LuaScriptRuntime(report=report or CompatibilityReport(), instruction_limit=200_000)
    runtime.set_roots((root,))
    return runtime


class TestDeepModules:
    def test_text_lua_finds_a_module_three_folders_down(self, tmp_path: Path) -> None:
        # 置き場の深い所を探さないと、配布のまま置いた PSDToolKit の require が見つからず、
        # 字幕表示が subobj を作れない
        _module(tmp_path / "aviutl_psdtoolkit" / "src" / "lua" / "Deep.lua", "deep")
        report = CompatibilityReport()
        text = _runtime(tmp_path, report).expand_text('<?mes(require("Deep").name)?>', _state())
        assert text == "deep"
        assert not report.missing

    def test_a_global_set_by_the_text_reaches_the_effect(self, tmp_path: Path) -> None:
        # AviUtl の Lua は 1 つの環境を全部のスクリプトで分け合う 吹き出し はテキスト欄が
        # 置いた subobj を読む
        _module(tmp_path / "a" / "b" / "Deep.lua", "deep")
        runtime = _runtime(tmp_path)
        runtime.expand_text('<?held=require("Deep")?>', _state())
        state = _state()
        result = runtime.run("obj.ox = held.name == 'deep' and 1 or 0", state)
        assert not result.failed, result.message
        assert state.ox == 1.0

    def test_the_shallow_one_wins(self, tmp_path: Path) -> None:
        # 同じ名前が浅い所にもあれば、今までどおりそちらを読む 探し方を変えただけで
        # 読むモジュールが入れ替わると、動いていた配布物が別の物を読む
        _module(tmp_path / "x" / "Same.lua", "shallow")
        _module(tmp_path / "a" / "b" / "c" / "Same.lua", "deep")
        assert (
            _runtime(tmp_path).expand_text('<?mes(require("Same").name)?>', _state()) == "shallow"
        )

    def test_the_folders_are_walked_once_for_many_missing_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 名前ごとに置き場全体を歩くと、毎回違う無い名前を require するスクリプトが、Lua の
        # 命令数の上限の外で描画を止められる 歩くのは置き場ごとに 1 度だけ
        _module(tmp_path / "a" / "b" / "c" / "Deep.lua", "deep")
        walks: list[Path] = []
        original_glob, original_rglob = Path.glob, Path.rglob

        def glob(self: Path, pattern: str, *args: Any, **kwargs: Any) -> Any:
            if "**" in pattern:
                walks.append(self)
            return original_glob(self, pattern, *args, **kwargs)

        def rglob(self: Path, pattern: str, *args: Any, **kwargs: Any) -> Any:
            walks.append(self)
            return original_rglob(self, pattern, *args, **kwargs)

        monkeypatch.setattr(Path, "glob", glob)
        monkeypatch.setattr(Path, "rglob", rglob)
        runtime = _runtime(tmp_path)
        source = (
            "for i = 1, 20 do require('none' .. i) end"
            " obj.ox = require('Deep').name == 'deep' and 1 or 0"
        )
        state = _state()
        result = runtime.run(source, state)
        assert not result.failed, result.message
        assert state.ox == 1.0
        assert len(walks) <= 1

    def test_a_missing_module_is_still_recorded(self, tmp_path: Path) -> None:
        # 深く探しても無い名前を記録しないと、互換性レポートにどのモジュールが足りないかが
        # 出ない
        report = CompatibilityReport()
        _runtime(tmp_path, report).expand_text('<?mes(tostring(require("Nowhere")))?>', _state())
        assert any("Nowhere" in line for line in report.missing)
