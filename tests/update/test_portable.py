"""exe の隣の ``scripts`` に本人が置いた物を ``%APPDATA%`` 側へ移す（Issue #244）

移すのは、zip を手で展開し直してフォルダごと入れ替えると消える所から、消えない所へ
上書きしない・写し終えて中身を照らしてから元を消す・途中で失敗しても元を消さない
読む順（同じ名前でどちらが勝つか）を変えない
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from sashimono.compat.aviutl import catalog as catalog_module
from sashimono.compat.aviutl.catalog import ScriptCatalog
from sashimono.update.package import carry_user_files
from sashimono.update.portable import (
    BUNDLED_SCRIPT_FILES,
    MODULE_FILE_SUFFIXES,
    PORTABLE_SCRIPTS_DIR,
    bundles,
    move_user_scripts,
    unoffered,
    user_script_files,
)

ROOT = Path(__file__).resolve().parents[2]


def _script(name: str, value: int = 0) -> str:
    return f"@{name}\n--track0:量,0,100,{value}\nobj.ox = obj.track0\n"


@pytest.fixture
def install(tmp_path: Path) -> Path:
    """配った zip を展開した形 ``scripts`` には同梱の説明だけがある"""
    folder = tmp_path / "Programs" / "Sashimono"
    (folder / PORTABLE_SCRIPTS_DIR).mkdir(parents=True)
    (folder / "Sashimono.exe").write_bytes(b"MZ")
    (folder / PORTABLE_SCRIPTS_DIR / "README.txt").write_text("同梱の説明", encoding="utf-8")
    return folder


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return tmp_path / "roaming" / "Sashimono" / PORTABLE_SCRIPTS_DIR


def _put(folder: Path, relative: str, text: str) -> Path:
    path = folder / PORTABLE_SCRIPTS_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestFindingTheirFiles:
    def test_only_the_bundled_readme_is_not_theirs(self, install: Path) -> None:
        """同梱の説明を数えると、何も置いていない人にまで移すよう勧める"""
        assert user_script_files(install) == []
        _put(install, "自分の/効果.anm2", _script("揺れ"))
        _put(install, "自分の/共通.mod2", "-- 共通")
        _put(install, "README.md", "自分の説明")
        assert [p.as_posix() for p in user_script_files(install)] == [
            "README.md",
            "自分の/共通.mod2",
            "自分の/効果.anm2",
        ]

    def test_a_readme_in_a_subfolder_is_theirs(self, install: Path) -> None:
        """同梱の説明は直下の 1 つだけ 配布物のフォルダの中の説明は本人の物"""
        _put(install, "配布物/README.txt", "配布物の説明")
        assert [p.as_posix() for p in user_script_files(install)] == ["配布物/README.txt"]

    def test_the_bundled_name_is_matched_without_case(self, install: Path) -> None:
        (install / PORTABLE_SCRIPTS_DIR / "README.txt").rename(
            install / PORTABLE_SCRIPTS_DIR / "readme.txt"
        )
        assert user_script_files(install) == []

    def test_no_folder_means_nothing(self, tmp_path: Path) -> None:
        assert user_script_files(tmp_path / "無い") == []

    def test_offered_ones_are_not_offered_again(self) -> None:
        files = [Path("a.anm2"), Path("b/c.obj2")]
        assert unoffered(files, ()) == files
        assert unoffered(files, ("A.anm2",)) == [Path("b/c.obj2")]
        assert unoffered(files, ("a.anm2", "b/c.obj2")) == []


class TestMoving:
    def test_their_files_move_with_the_folders(self, install: Path, target: Path) -> None:
        _put(install, "自分の/効果.anm2", _script("揺れ"))
        _put(install, "自分の/画像/星.png", "png")
        result = move_user_scripts(install, target)
        assert [p.as_posix() for p in result.moved] == ["自分の/効果.anm2", "自分の/画像/星.png"]
        assert (target / "自分の" / "効果.anm2").read_text(encoding="utf-8") == _script("揺れ")
        assert (target / "自分の" / "画像" / "星.png").is_file()
        # 元は消え、空になったフォルダも片付く 同梱の説明と scripts そのものは残す
        scripts = install / PORTABLE_SCRIPTS_DIR
        assert sorted(p.name for p in scripts.iterdir()) == ["README.txt"]
        assert not (target / "README.txt").exists()
        assert not list(target.rglob("*.moving"))

    def test_nothing_is_overwritten(self, install: Path, target: Path) -> None:
        """移し先に同じ名前があれば、移し先も元も触らない

        移し先は本人が後から置いた物かもしれない
        """
        mine = _put(install, "効果.anm2", _script("古い"))
        target.mkdir(parents=True)
        (target / "効果.anm2").write_text(_script("新しい"), encoding="utf-8")
        result = move_user_scripts(install, target)
        assert result.moved == () and result.kept == (Path("効果.anm2"),)
        assert (target / "効果.anm2").read_text(encoding="utf-8") == _script("新しい")
        assert mine.read_text(encoding="utf-8") == _script("古い")

    def test_a_failed_copy_keeps_the_original(
        self, install: Path, target: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mine = _put(install, "効果.anm2", _script("揺れ"))

        def broken(source: Path, destination: Path) -> None:
            Path(destination).write_text("途中", encoding="utf-8")
            raise OSError("ディスクがいっぱい")

        monkeypatch.setattr(shutil, "copy2", broken)
        result = move_user_scripts(install, target)
        assert [p.as_posix() for p, _reason in result.failed] == ["効果.anm2"]
        assert mine.read_text(encoding="utf-8") == _script("揺れ")
        # 途中の写しを本来の名前で残さない 残すと次から「もう在る」と見て移さず、欠けた方が読まれる
        assert not (target / "効果.anm2").exists()
        assert not list(target.rglob("*.moving"))

    def test_a_copy_that_differs_keeps_the_original(
        self, install: Path, target: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """大きさが同じでも中身が違えば元を消さない 消した後では欠けた写しに気付いても戻せない"""
        mine = _put(install, "効果.anm2", _script("揺れ"))

        def corrupt(source: Path, destination: Path) -> None:
            data = Path(source).read_bytes()
            Path(destination).write_bytes(bytes(reversed(data)))

        monkeypatch.setattr(shutil, "copy2", corrupt)
        result = move_user_scripts(install, target)
        assert len(result.failed) == 1
        assert mine.is_file()
        assert not (target / "効果.anm2").exists()

    def test_an_original_that_cannot_be_removed_is_counted(
        self, install: Path, target: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mine = _put(install, "効果.anm2", _script("揺れ"))
        real_unlink = Path.unlink

        def refuse(self: Path, missing_ok: bool = False) -> None:
            if self == mine:
                raise PermissionError("使用中")
            real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", refuse)
        result = move_user_scripts(install, target)
        assert result.left == (Path("効果.anm2"),) and result.moved == ()
        assert mine.is_file() and (target / "効果.anm2").is_file()


class TestReadingOrder:
    """移しても、同じ名前のスクリプトでどれが勝つかは変わらない

    読む順は exe の隣 → ``%APPDATA%`` → AviUtl2 で、後に読んだ方が勝つ 移すのは exe の隣
    （いちばん負ける所）から次の所へだけで、移し先に同じ名前があれば移さない
    """

    def _winners(self, roots: tuple[Path, ...]) -> dict[str, str]:
        found = ScriptCatalog(roots).scan()
        return {entry.identifier: entry.source for entry in found}

    def test_winners_stay_the_same(self, install: Path, target: Path, tmp_path: Path) -> None:
        aviutl = tmp_path / "ProgramData" / "aviutl2" / "Script"
        aviutl.mkdir(parents=True)
        _put(install, "一/ひとつ.anm2", _script("ひとつ", 1))
        # %APPDATA% と同じ名前 前から %APPDATA% が勝っている
        _put(install, "重/重なる.anm2", _script("重なる", 2))
        (target / "重").mkdir(parents=True)
        (target / "重" / "重なる.anm2").write_text(_script("重なる", 3), encoding="utf-8")
        # AviUtl2 と同じ名前 前から AviUtl2 が勝っている
        _put(install, "上/上がある.anm2", _script("上がある", 4))
        (aviutl / "上").mkdir()
        (aviutl / "上" / "上がある.anm2").write_text(_script("上がある", 5), encoding="utf-8")
        roots = (install / PORTABLE_SCRIPTS_DIR, target, aviutl)

        before = self._winners(roots)
        result = move_user_scripts(install, target)
        after = self._winners(roots)

        assert after == before
        assert sorted(p.as_posix() for p in result.moved) == ["一/ひとつ.anm2", "上/上がある.anm2"]
        assert result.kept == (Path("重/重なる.anm2"),)

    def test_the_order_of_roots_is_what_this_relies_on(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """exe の隣がいちばん先（いちばん負ける）でなければ、移すと勝ち負けが変わる"""
        monkeypatch.setattr(catalog_module, "app_dir", lambda: tmp_path / "Sashimono")
        roots = catalog_module.default_script_roots()
        assert roots[0] == tmp_path / "Sashimono" / PORTABLE_SCRIPTS_DIR
        assert roots[1].name == PORTABLE_SCRIPTS_DIR and roots[1].parent.name == "Sashimono"


class TestBundles:
    """一緒に動く物（束）は一式で移すか、一式で残す（PR #245 の Codex の指摘）

    スクリプトはモジュールを自分のフォルダから先に探す 効果だけ移してモジュールを残すと、
    移した先の同じフォルダにある別のモジュールを読み、更新しただけで描画が変わる
    """

    def _run(self, roots: tuple[Path, ...], script: Path) -> float:
        """本物の Lua の実行で、スクリプトが読んだモジュールの値を返す"""
        import numpy as np

        from sashimono.compat.aviutl.objapi import ObjectState
        from sashimono.compat.aviutl.report import CompatibilityReport
        from sashimono.compat.aviutl.runtime import LuaScriptRuntime

        runtime = LuaScriptRuntime(report=CompatibilityReport(), instruction_limit=200_000)
        runtime.set_roots(roots)
        state = ObjectState(image=np.zeros((1, 1, 4), np.uint8), screen_w=320, screen_h=180)
        result = runtime.run(
            script.read_text(encoding="utf-8"), state, script=script.name, folder=script.parent
        )
        assert not result.failed, result.message
        return float(state.ox)

    def _where(self, install: Path, target: Path, relative: str) -> Path:
        for base in (target, install / PORTABLE_SCRIPTS_DIR):
            if (base / relative).is_file():
                return base / relative
        raise AssertionError(relative)

    def test_a_clashing_module_keeps_its_effect_beside_it(
        self, install: Path, target: Path
    ) -> None:
        """移し先に中身の違う同じ名前のモジュールがあれば、効果もモジュールも一式残す"""
        _put(install, "配布物/効果.anm2", 'local m = require("common")\nobj.ox = m.v\n')
        _put(install, "配布物/common.lua", "return { v = 1 }")
        (target / "配布物").mkdir(parents=True)
        (target / "配布物" / "common.lua").write_text("return { v = 2 }", encoding="utf-8")
        roots = (install / PORTABLE_SCRIPTS_DIR, target)
        before = self._run(roots, self._where(install, target, "配布物/効果.anm2"))

        result = move_user_scripts(install, target)

        assert result.moved == ()
        assert result.kept == (Path("配布物/common.lua"),)
        assert result.held == (Path("配布物/効果.anm2"),)
        assert self._run(roots, self._where(install, target, "配布物/効果.anm2")) == before == 1

    def test_the_same_bytes_are_not_a_clash(self, install: Path, target: Path) -> None:
        """移し先に同じ中身の物があれば一式移す 移す側の重なった物は消すだけ"""
        _put(install, "配布物/効果.anm2", 'local m = require("common")\nobj.ox = m.v\n')
        _put(install, "配布物/common.mod2", "return { v = 1 }")
        (target / "配布物").mkdir(parents=True)
        (target / "配布物" / "common.mod2").write_text("return { v = 1 }", encoding="utf-8")

        result = move_user_scripts(install, target)

        assert sorted(p.as_posix() for p in result.moved) == [
            "配布物/common.mod2",
            "配布物/効果.anm2",
        ]
        assert not (install / PORTABLE_SCRIPTS_DIR / "配布物").exists()
        assert (target / "配布物" / "common.mod2").read_text(encoding="utf-8") == "return { v = 1 }"

    def test_a_module_name_used_elsewhere_in_appdata_keeps_the_bundle(
        self, install: Path, target: Path
    ) -> None:
        """モジュールは名前で探す 移し先の別のフォルダに同じ名前の物があれば、移した後で
        どちらが先に見つかるかが変わりうる
        """
        _put(install, "配布物/効果.anm2", 'local m = require("common")\nobj.ox = m.v\n')
        _put(install, "配布物/common.lua", "return { v = 1 }")
        (target / "別の物").mkdir(parents=True)
        (target / "別の物" / "common.mod2").write_text("return { v = 3 }", encoding="utf-8")

        result = move_user_scripts(install, target)

        assert result.moved == () and len(result.held) == 2

    def test_a_bundle_sharing_a_module_name_with_a_kept_one_stays(
        self, install: Path, target: Path
    ) -> None:
        """残した束と同じ名前のモジュールを持つ束も残す 片方だけ移すと、exe の隣の中で
        決まっていた勝ち負けが置き場の順で決まるように変わる
        """
        _put(install, "甲/common.lua", "return { v = 1 }")
        _put(install, "甲/効果.anm2", "obj.ox = 1\n")
        _put(install, "乙/common.lua", "return { v = 2 }")
        _put(install, "丙/単独.anm2", "obj.ox = 3\n")
        (target / "甲").mkdir(parents=True)
        (target / "甲" / "効果.anm2").write_text("obj.ox = 9\n", encoding="utf-8")

        result = move_user_scripts(install, target)

        assert [p.as_posix() for p in result.moved] == ["丙/単独.anm2"]
        assert sorted(p.as_posix() for p in result.held) == [
            "乙/common.lua",
            "甲/common.lua",
        ]

    def test_loose_files_are_one_bundle(self, install: Path, target: Path) -> None:
        """置き場の直下のファイルは同じフォルダを分け合うので、まとめて 1 束"""
        _put(install, "効果.anm2", 'local m = require("common")\nobj.ox = m.v\n')
        _put(install, "common.lua", "return { v = 1 }")
        target.mkdir(parents=True)
        (target / "common.lua").write_text("return { v = 2 }", encoding="utf-8")
        result = move_user_scripts(install, target)
        assert result.moved == () and result.held == (Path("効果.anm2"),)

    def test_the_bundles(self) -> None:
        found = bundles(
            [Path("a.anm2"), Path("b.lua"), Path("X/c.anm2"), Path("x/d/e.lua"), Path("Y/f")]
        )
        assert found == {
            "*": [Path("a.anm2"), Path("b.lua")],
            "x": [Path("X/c.anm2"), Path("x/d/e.lua")],
            "y": [Path("Y/f")],
        }

    def test_the_module_suffixes_match_the_runtime(self) -> None:
        """名前が食い違うと、探される物をモジュールと見なさずに束を割る"""
        from sashimono.compat.aviutl import runtime

        assert set(MODULE_FILE_SUFFIXES) == {*runtime.MODULE_SUFFIXES, runtime.C_MODULE_SUFFIX}


class TestSurvivingUpdates:
    """移した物は、自動更新でも手で入れ替えても残る"""

    def test_a_replaced_folder_keeps_moved_scripts(self, install: Path, target: Path) -> None:
        """zip を展開し直してフォルダごと入れ替える（手で入れ替える）"""
        _put(install, "自分の/効果.anm2", _script("揺れ"))
        move_user_scripts(install, target)
        shutil.rmtree(install)
        (install / PORTABLE_SCRIPTS_DIR).mkdir(parents=True)
        (install / PORTABLE_SCRIPTS_DIR / "README.txt").write_text("新しい説明", encoding="utf-8")

        found = ScriptCatalog((install / PORTABLE_SCRIPTS_DIR, target)).scan()
        assert [entry.name for entry in found] == ["揺れ"]

    def test_without_moving_a_replaced_folder_loses_them(self, install: Path, target: Path) -> None:
        """移さないと消える（この試験が通らなくなったら、勧める理由が無くなっている）"""
        _put(install, "自分の/効果.anm2", _script("揺れ"))
        shutil.rmtree(install)
        (install / PORTABLE_SCRIPTS_DIR).mkdir(parents=True)
        assert ScriptCatalog((install / PORTABLE_SCRIPTS_DIR, target)).scan() == ()

    def test_an_automatic_update_after_moving_has_nothing_to_carry(
        self, install: Path, target: Path, tmp_path: Path
    ) -> None:
        _put(install, "自分の/効果.anm2", _script("揺れ"))
        move_user_scripts(install, target)
        staged = tmp_path / "Programs" / "Sashimono.new"
        (staged / PORTABLE_SCRIPTS_DIR).mkdir(parents=True)
        (staged / PORTABLE_SCRIPTS_DIR / "README.txt").write_text("新しい説明", encoding="utf-8")
        assert carry_user_files(install, staged) == 0
        assert (target / "自分の" / "効果.anm2").is_file()


class TestNames:
    def test_the_folder_name_matches_the_catalog(self) -> None:
        """名前が食い違うと、移す元が読まれている所と違う"""
        assert PORTABLE_SCRIPTS_DIR == catalog_module.PORTABLE_SCRIPTS_DIR

    def test_the_bundled_files_match_what_the_zip_puts_there(self, tmp_path: Path) -> None:
        """配る zip の ``scripts`` に入れる物が増えたら、ここにも足す 足さないと、全員に
        「自分で置いた物がある」と言って同梱の物を移す
        """
        spec = importlib.util.spec_from_file_location(
            "build_package_for_portable", ROOT / "tools" / "build_package.py"
        )
        assert spec is not None and spec.loader is not None
        builder: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = builder
        try:
            spec.loader.exec_module(builder)
            bundle = tmp_path / "bundle"
            bundle.mkdir()
            builder.assemble(bundle)
        finally:
            sys.modules.pop(spec.name, None)
        placed = {p.name for p in (bundle / PORTABLE_SCRIPTS_DIR).iterdir()}
        assert placed == set(BUNDLED_SCRIPT_FILES)
        assert user_script_files(bundle) == []
