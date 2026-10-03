"""配る zip（パッケージ版）でだけ通る道

開発環境では ``sys.frozen`` が無いので、ここにある道は普段まったく通らない
壊れても誰も気づかないまま配ることになるので、固めた状態を真似て確かめる
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from sashimono.app import SELF_CHECK_FLAG, main
from sashimono.compat.aviutl.catalog import PORTABLE_SCRIPTS_DIR, default_script_roots
from sashimono.core.model import Project
from sashimono.runtime import app_dir, install_command, pip_arguments, run_pip
from sashimono.selfcheck import CheckResult, format_results, run_self_check

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """固めた exe として動いているふりをする 返すのは exe の場所"""
    executable = tmp_path / "Sashimono" / "Sashimono.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    return executable


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_package", ROOT / "tools" / "build_package.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheScriptFolderBesideTheExe:
    def test_it_comes_first_in_the_package(self, frozen: Path) -> None:
        """同梱の見本と、前の版の案内どおりそこへ置いた人のスクリプトを読み続ける

        外すと、置き場を移していない人のスクリプトが更新の前から読まれなくなる
        置き場として案内するのは ``%APPDATA%`` の側（Issue #138）
        """
        assert default_script_roots()[0] == frozen.parent / PORTABLE_SCRIPTS_DIR

    def test_the_readme_points_to_the_folder_an_update_keeps(self, builder: ModuleType) -> None:
        # exe の隣を置き場として案内すると、新しい版へフォルダごと入れ替えた人の
        # スクリプトが消える 案内は入れ替えても残る %APPDATA% の側にする（Issue #138）
        text = builder.README_TEXT
        assert "%APPDATA%\\Sashimono\\scripts" in text
        assert "スクリプトフォルダを開く" in text
        assert "中身が消えます" in text

    def test_it_is_not_looked_at_in_development(self) -> None:
        # 開発環境で .venv の隣を探しに行くと、関係の無いフォルダを読む
        assert app_dir() is None
        assert all(
            root.name != PORTABLE_SCRIPTS_DIR or "Sashimono" in root.parts
            for root in default_script_roots()
        )


class TestPipInsideThePackage:
    """配布版には Python の本体が無い 導入ボタンは exe 自身に pip を走らせる"""

    def test_the_install_button_calls_the_exe_itself(self, frozen: Path) -> None:
        # 導入ボタンが組み立てるコマンド 先頭は exe（ここで受けないと Sashimono が 2 つ立つ）
        from sashimono.runtime import FeaturePack

        command = install_command(FeaturePack(key="x", label="x", required=("pkg",)))
        assert command[:4] == [str(frozen), "-m", "pip", "install"]

    def test_the_exe_hands_it_to_pip(self, frozen: Path) -> None:
        assert pip_arguments([str(frozen), "-m", "pip", "install", "pkg"]) == ["install", "pkg"]

    def test_an_ordinary_start_is_not_taken_for_pip(self, frozen: Path) -> None:
        """プロジェクトを開く起動（``Sashimono.exe 作品.sme``）を pip と取り違えない"""
        assert pip_arguments([str(frozen), "作品.sme"]) is None

    def test_development_leaves_it_to_python(self) -> None:
        """開発環境の ``sys.executable`` は本物の Python なので、こちらは受けない

        受けると、``python -m sashimono -m pip`` のような書き方まで pip に回る
        """
        assert pip_arguments(["python", "-m", "pip", "list"]) is None

    def test_main_runs_pip_instead_of_the_window(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """窓を作る前に pip へ渡す 窓を作ってからだと、導入のたびに窓が開く"""
        called: list[list[str]] = []

        def fake_pip(arguments: list[str]) -> int:
            called.append(arguments)
            return 0

        monkeypatch.setattr("pip._internal.cli.main.main", fake_pip)
        assert main([str(frozen), "-m", "pip", "--version"]) == 0
        assert called == [["--version"]]

    def test_distlib_learns_how_to_find_its_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pip の中の distlib に、配布版の読み込み方式でも部品を探せるよう教える

        教えないと ``pip install`` が「Unable to locate finder」で落ちる
        ``pip --version`` は部品を探さないので通ってしまい、ここまで気づけなかった
        （配布版で実際に入れてみて見つかった）
        """
        from pip._vendor import distlib
        from pip._vendor.distlib import resources

        class FrozenLoader:
            """PyInstaller の読み込み方式の代わり distlib の一覧に無い型"""

        monkeypatch.setattr(distlib, "__loader__", FrozenLoader(), raising=False)
        monkeypatch.setattr("pip._internal.cli.main.main", lambda arguments: 0)
        # 試験のあとに登録を残さない 残すと、ほかの試験が別の探し方で動く
        registry = resources._finder_registry
        monkeypatch.setattr(resources, "_finder_registry", dict(registry))

        assert run_pip(["--version"]) == 0
        assert resources._finder_registry.get(FrozenLoader) is resources.ResourceFinder


class TestTheSelfCheck:
    # 描く・書き出す項目は OpenGL 4.3 が要る GPU の無い CI では飛ばす
    # （自己診断そのものは動くが、その 2 項目が NG になるのは正しい結果）
    @pytest.mark.usefixtures("gpu")
    def test_everything_works_here(self) -> None:
        """開発環境では全部動く ここで落ちるなら、配る前から壊れている"""
        results = run_self_check()
        failed = [r for r in results if not r.ok and not r.optional]
        assert failed == [], format_results(results)

    def test_the_flag_reaches_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--self-check`` は窓を作らずに終わる

        窓を作ると、使う人の機械で確かめてもらうときに、閉じるまで結果が出ない
        """
        monkeypatch.setattr("sashimono.selfcheck.run_self_check", lambda: [CheckResult("x", True)])
        assert main(["sashimono", SELF_CHECK_FLAG]) == 0

    def test_a_project_with_the_flag_is_opened_not_checked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """プロジェクトと一緒に渡されたら、自己診断ではなく開く起動として扱う

        自己診断を優先すると、頼んだプロジェクトが開かずに黙って終わる
        """
        checked: list[bool] = []

        def fake_check() -> int:
            checked.append(True)
            return 0

        monkeypatch.setattr("sashimono.selfcheck.main", fake_check)

        class ReachedTheWindowError(Exception):
            """いつもの起動の最初の段まで来たら止める ここまで来れば開く起動"""

        def stop() -> None:
            raise ReachedTheWindowError

        # main 自身が見ている名前を差し替える ほかの試験がモジュールを読み直すと、
        # 「sashimono.app」という名前の先と、ここで握っている main の先が別物になる
        monkeypatch.setitem(main.__globals__, "activate_runtime", stop)
        with pytest.raises(ReachedTheWindowError):
            main(["sashimono", "作品.sme", SELF_CHECK_FLAG])
        assert checked == []

    def test_a_missing_part_fails_the_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """1 つでも動かなければ終了コード 1 組み立ての道具はこれを見て止まる"""
        monkeypatch.setattr(
            "sashimono.selfcheck.run_self_check",
            lambda: [CheckResult("GL で描く", False, "積み忘れ")],
        )
        assert main(["sashimono", SELF_CHECK_FLAG]) == 1

    def test_an_optional_part_does_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """音の出口が無い機械（リモート接続など）でも編集はできる 落とさない"""
        monkeypatch.setattr(
            "sashimono.selfcheck.run_self_check",
            lambda: [CheckResult("音の出口", False, "無い", optional=True)],
        )
        assert main(["sashimono", SELF_CHECK_FLAG]) == 0

    def test_a_failure_says_which_part(self) -> None:
        # どの項目が動かないかが 1 行で分かる 分からないと組み立て直しを繰り返す
        text = format_results([CheckResult("書き出す（FFmpeg）", False, "DLL が無い")])
        assert "[NG] 書き出す（FFmpeg）: DLL が無い" in text


class TestTheZip:
    def test_it_unpacks_into_one_folder(self, builder: ModuleType, tmp_path: Path) -> None:
        """展開すると ``Sashimono\\`` が 1 つできる

        ばらで入れると、展開した場所（デスクトップなど）に部品が散らばる
        """
        bundle = tmp_path / "bundle"
        (bundle / "_internal").mkdir(parents=True)
        (bundle / "Sashimono.exe").write_bytes(b"MZ")
        (bundle / "_internal" / "part.dll").write_bytes(b"x")
        builder.assemble(bundle)
        archive = builder.make_zip(bundle, tmp_path / "out.zip")
        with zipfile.ZipFile(archive) as opened:
            names = opened.namelist()
        assert all(name.startswith("Sashimono/") for name in names)
        assert "Sashimono/Sashimono.exe" in names

    def test_the_script_folder_is_already_there(self, builder: ModuleType, tmp_path: Path) -> None:
        """空でも置き場を作っておく 無いと、どこへ置けばいいのかが分からない

        zip は空のフォルダを持てないので、説明書きを 1 つ入れて残す
        """
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        builder.assemble(bundle)
        archive = builder.make_zip(bundle, tmp_path / "out.zip")
        with zipfile.ZipFile(archive) as opened:
            assert f"Sashimono/{PORTABLE_SCRIPTS_DIR}/README.txt" in opened.namelist()

    def test_an_unfinished_zip_is_not_left_as_the_real_one(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """途中で止まった zip を完成品と取り違えない"""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "Sashimono.exe").write_bytes(b"MZ")

        def broken(self: zipfile.ZipFile, *args: object, **kwargs: object) -> None:
            raise OSError("書けない")

        monkeypatch.setattr(zipfile.ZipFile, "write", broken)
        with pytest.raises(OSError):
            builder.make_zip(bundle, tmp_path / "out.zip")
        assert not (tmp_path / "out.zip").exists()
        # 書きかけも残さない 100 MB ずつ溜まるうえ、手で配るときに紛れる
        assert list(tmp_path.glob("*.writing")) == []

    def test_a_failed_rename_leaves_nothing_behind(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """書き終えても、名前を付けられなければ書いた物を残さない

        そこで残すと、書き終えた 100 MB がそのまま溜まる
        """
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "Sashimono.exe").write_bytes(b"MZ")
        target = tmp_path / "out.zip"

        def locked(self: Path, other: Path) -> Path:
            raise PermissionError("使用中")

        monkeypatch.setattr(Path, "replace", locked)
        with pytest.raises(PermissionError):
            builder.make_zip(bundle, target)
        assert list(tmp_path.glob("*.writing")) == []

    def test_the_previous_zip_does_not_survive_a_failure(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """前の組み立ての zip を、今回の版の名前で残さない

        今回が途中で落ちたときに前の物が残ると、新しい物と取り違えて配る
        """
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "Sashimono.exe").write_bytes(b"MZ")
        target = tmp_path / "out.zip"
        target.write_bytes(b"previous build")

        def broken(self: zipfile.ZipFile, *args: object, **kwargs: object) -> None:
            raise OSError("書けない")

        monkeypatch.setattr(zipfile.ZipFile, "write", broken)
        with pytest.raises(OSError):
            builder.make_zip(bundle, target)
        assert not target.exists(), "前の組み立ての zip が今回の名前で残っている"

    def test_the_check_does_not_borrow_the_developers_path(self, builder: ModuleType) -> None:
        """確かめるときは開発機の PATH を使わない

        開発機の PATH には Python や FFmpeg が載っている 残すと、zip に
        積み忘れた DLL をそちらから拾って通ってしまう
        """
        environment = builder.minimal_environment(
            {"PATH": r"C:\Python314;C:\ffmpeg\bin", "SystemRoot": r"C:\Windows", "TEMP": "t"}
        )
        assert "Python" not in environment["PATH"]
        assert "ffmpeg" not in environment["PATH"]
        assert environment["SYSTEMROOT"] == r"C:\Windows"

    def test_the_optional_features_are_left_out(self, builder: ModuleType) -> None:
        """字幕起こしと AI 連携は積まない 開発機に入っていると拾われて 2 GB を超える"""
        arguments = builder.pyinstaller_arguments(Path("w"), Path("d"))
        excluded = {
            arguments[i + 1] for i, value in enumerate(arguments) if value == "--exclude-module"
        }
        assert {"faster_whisper", "claude_agent_sdk"} <= excluded

    def test_the_dynamically_loaded_parts_are_collected(self, builder: ModuleType) -> None:
        """名前で読む部品はまとめて積む

        lupa は Lua の実体を名前で選び（``lupa.lua51``）、pip は導入ボタンが使う
        PyInstaller は名前で読む import を辿れないので、積み忘れても組み立ては通る
        """
        arguments = builder.pyinstaller_arguments(Path("w"), Path("d"))
        collected = {
            arguments[i + 1] for i, value in enumerate(arguments) if value == "--collect-all"
        }
        assert {"lupa", "pip"} <= collected


def _listing(*names: str) -> str:
    """いま入っている版で書いた一覧（THIRD_PARTY_NOTICES.md の表の形）

    試験を走らせる機械の版で書く リポジトリの一覧の版で比べると、CI が新しい版を
    入れた日に、写しを集める試験まで落ちる
    """
    import importlib.metadata

    return "".join(f"| `{name}` | {importlib.metadata.version(name)} |\n" for name in names)


class TestTheNotices:
    """配る zip は GPL の部品（x264・x265）を積むので、全体を GPL の条件で配る

    使用許諾の全文と、部品ごとの一覧・ソースの入手先を一緒に渡さなければならない
    欠けていても zip は作れて動くので、道具の側で止める
    """

    def test_the_zip_carries_the_notices(self, builder: ModuleType, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        builder.assemble(bundle)
        archive = builder.make_zip(bundle, tmp_path / "out.zip")
        with zipfile.ZipFile(archive) as opened:
            names = set(opened.namelist())
        assert "Sashimono/THIRD_PARTY_NOTICES.txt" in names
        assert "Sashimono/LICENSE.txt" in names
        for text in ("GPL-2.0.txt", "GPL-3.0.txt", "LGPL-2.1.txt", "LGPL-3.0.txt"):
            assert f"Sashimono/licenses/{text}" in names, f"{text} の全文が zip に無い"

    def test_the_dll_licenses_the_wheel_lacks_go_in(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        """PyAV の wheel の DLL と LuaJIT の写しは、リポジトリに置いた物を zip へ入れる

        wheel が写しを持っていないので、dist-info から集めるだけでは入らない
        BSD や MIT もバイナリと一緒に表記を渡す条件なので、欠けたまま配れない
        """
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        builder.assemble(bundle)
        archive = builder.make_zip(bundle, tmp_path / "out.zip")
        with zipfile.ZipFile(archive) as opened:
            names = set(opened.namelist())
        for copy in (
            "ffmpeg-8.1.2/LICENSE.md",
            "x264-b35605ac/COPYING",
            "x265-4.2/COPYING",
            "dav1d-1.5.3/COPYING",
            "opus-1.6.1/COPYING",
            "SVT-AV1-4.1.0/LICENSE.md",
            "libvpx-1.16.0/LICENSE",
            "libwebp-1.6.0/COPYING",
            "libvpl-2.16.0/LICENSE",
            "lame-3.100/COPYING",
            "opencore-amr-0.1.6/LICENSE",
            "zlib-1.3.2/LICENSE",
            "libiconv-1.19/COPYING.LIB",
            "gcc-16.1.0/COPYING.RUNTIME",
            "winpthreads-mingw-w64-14.0.0/COPYING",
            "LuaJIT-2.0-e4c7d8b3/COPYRIGHT",
            "LuaJIT-2.1-18b087cd/COPYRIGHT",
        ):
            assert f"Sashimono/licenses/{copy}" in names, f"{copy} が zip に無い"
        # zip から確かめる段も、リポジトリに置いた写しを全部見本にする
        assert "x264-b35605ac/COPYING" in builder.repository_license_files()

    def test_the_gnu_texts_are_the_real_ones(self) -> None:
        # 名前だけの空のファイルや別の版の全文を置いても、有無の確認は通ってしまう
        for name, title, version in (
            ("GPL-2.0.txt", "GNU GENERAL PUBLIC LICENSE", "Version 2, June 1991"),
            ("GPL-3.0.txt", "GNU GENERAL PUBLIC LICENSE", "Version 3, 29 June 2007"),
            ("LGPL-2.1.txt", "GNU LESSER GENERAL PUBLIC LICENSE", "Version 2.1, February 1999"),
            ("LGPL-3.0.txt", "GNU LESSER GENERAL PUBLIC LICENSE", "Version 3, 29 June 2007"),
        ):
            head = (ROOT / "licenses" / name).read_text(encoding="utf-8")[:200]
            assert title in head and version in head, name

    def test_the_gnu_texts_are_whole(self) -> None:
        """全文が 1 文字も欠けていない（FSF の配る全文と sha256 が同じ）

        見出しだけを見ると、途中で切れた全文でも通って zip に入る
        """
        expected = {
            "GPL-2.0.txt": "edaef632cbb643e4e7a221717a6c441a4c1a7c918e6e4d56debc3d8739b233f6",
            "GPL-3.0.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
            "LGPL-2.1.txt": "20e50fe7aae3e56378ebf0417d9de904f55a0e61e4df315333e632a4d3555d95",
            "LGPL-3.0.txt": "da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768",
        }
        for name, digest in expected.items():
            # 改行は LF にそろえて比べる リポジトリは LF で持つが、Windows で取り出すと
            # 設定によって CRLF になる（CI の取り出しがそうだった） 文面は同じ
            data = (ROOT / "licenses" / name).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(data).hexdigest() == digest, name

    def test_a_bundled_package_brings_its_license(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        """積んだファイルから包みを辿り、その包みの写しを集める

        包みの名前を決め打ちで持つと、組み立てる機械に入っている包みが変わったとき
        （PyInstaller は入っていれば拾う）に写しの無い物を黙って配る
        """
        import numpy

        problems = builder.collect_licenses(
            tmp_path, [Path(numpy.__file__)], (), notices=_listing("numpy", "pyinstaller")
        )
        assert not [p for p in problems if "numpy" in p], problems
        copied = tmp_path / "licenses" / f"numpy-{numpy.__version__}" / "LICENSE.txt"
        assert copied.is_file(), "numpy の使用許諾の写しが集まっていない"

    def test_a_file_from_nowhere_stops_it(self, builder: ModuleType, tmp_path: Path) -> None:
        """開発機の PATH から拾った DLL のように、どの包みの物でもないファイルは止める

        どの使用許諾で配るのか決められない 実際に Git for Windows の OpenSSL が
        積まれていた
        """
        stray = tmp_path / "elsewhere" / "libssl-3-x64.dll"
        problems = builder.collect_licenses(tmp_path / "bundle", [stray], ())
        assert any("libssl-3-x64.dll" in p for p in problems), problems

    def test_sashimonos_own_files_are_not_taken_for_strays(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        """Sashimono 自身のソースは出どころの分からない物として止めない

        止めると、正しい組み立てでも毎回 zip を作れなくなる
        """
        own = ROOT / "src" / "sashimono" / "__init__.py"
        problems = builder.collect_licenses(tmp_path, [own], (ROOT / "src",))
        assert not [p for p in problems if "__init__.py" in p], problems

    def test_a_package_without_a_license_file_stops_it(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """写しを持たない包みは、一覧に書いたうえで名指しで許した物だけ通す"""
        import OpenGL

        source = [Path(OpenGL.__file__)]
        listing = _listing("PyOpenGL", "pyinstaller")
        allowed = builder.collect_licenses(tmp_path / "a", source, (), notices=listing)
        assert not [p for p in allowed if "PyOpenGL" in p], allowed

        monkeypatch.setattr(builder, "WITHOUT_LICENSE_FILES", frozenset())
        refused = builder.collect_licenses(tmp_path / "b", source, (), notices=listing)
        assert any("PyOpenGL" in p and "写し" in p for p in refused), refused

    def test_a_package_missing_from_the_list_stops_it(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        # 一覧に無い包みは、ソースの入手先も書いていない
        import numpy

        problems = builder.collect_licenses(tmp_path, [Path(numpy.__file__)], (), notices="")
        assert any("numpy" in p and "一覧" in p for p in problems), problems

    def test_the_list_names_what_the_real_build_bundled(self, builder: ModuleType) -> None:
        """開発機の組み立てで数えた包みが、一覧に全部載っている

        一覧に無い包みがあると、組み立ての最後で止まって zip を作れない
        """
        listed = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").lower()
        for name in (
            "PySide6_Essentials",
            "PySide6_Addons",
            "shiboken6",
            "av",
            "numpy",
            "lupa",
            "PyOpenGL",
            "sounddevice",
            "pip",
            "pyinstaller",
        ):
            assert f"`{name.lower()}`" in listed, name

    def test_the_record_includes_the_archive(self, builder: ModuleType, tmp_path: Path) -> None:
        """exe の中の書庫（PYZ）に入った純 Python の包みも数える

        exe の隣のフォルダだけを見ると、pip や setuptools の写しを集め損ねる
        """
        (tmp_path / "COLLECT-00.toc").write_text(
            repr(([("Sashimono.exe", r"C:\work\Sashimono.exe", "EXECUTABLE")],)), encoding="utf-8"
        )
        (tmp_path / "PYZ-00.toc").write_text(
            repr(
                (
                    r"C:\work\PYZ-00.pyz",
                    [
                        ("pip", r"C:\venv\pip\__init__.py", "PYMODULE"),
                        ("ns", "-", "PYMODULE"),
                    ],
                )
            ),
            encoding="utf-8",
        )
        assert builder.bundled_sources(tmp_path) == [
            Path(r"C:\work\Sashimono.exe"),
            Path(r"C:\venv\pip\__init__.py"),
        ]

    def test_the_unpacked_zip_is_checked(self, builder: ModuleType, tmp_path: Path) -> None:
        """zip から確かめる段でも見る 途中の段を飛ばしても zip は作れてしまう"""
        home = tmp_path / "Sashimono"
        home.mkdir()
        missing = builder.missing_notices(home)
        assert "THIRD_PARTY_NOTICES.txt" in missing
        assert "licenses/GPL-3.0.txt" in missing

        # 組み立てと同じ順（写しを集めてから説明書きと全文を置く）
        builder.collect_licenses(home, [], ())
        builder.assemble(home)
        if not (Path(sys.base_prefix) / "LICENSE.txt").exists():
            pytest.skip("この Python には LICENSE.txt が無い")
        assert builder.missing_notices(home) == []

    def test_the_build_does_not_see_the_developers_path(
        self, builder: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """組み立てる間は PATH を Windows の分にし、終わったら戻す

        戻さないと、そのあとの zip の確認や後続の処理が別の PATH で動く
        """
        developer = r"C:\Program Files\Git\mingw64\bin;C:\Windows\System32"
        monkeypatch.setenv("PATH", developer)
        with builder._without_developer_path():
            assert "Git" not in os.environ["PATH"]
        assert os.environ["PATH"] == developer


def _distribution(root: Path, declared: list[str], present: list[str]) -> object:
    """dist-info を手で作った包み ``declared`` を METADATA に書き、``present`` だけ置く"""
    import importlib.metadata

    info = root / "sample-1.0.dist-info"
    (info / "licenses").mkdir(parents=True)
    metadata = "Metadata-Version: 2.4\nName: sample\nVersion: 1.0\n"
    metadata += "".join(f"License-File: {name}\n" for name in declared)
    (info / "METADATA").write_text(metadata, encoding="utf-8")
    record = ["sample-1.0.dist-info/METADATA,,"]
    for name in present:
        (info / "licenses" / name).write_text("license text", encoding="utf-8")
        record.append(f"sample-1.0.dist-info/licenses/{name},,")
    (info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
    return importlib.metadata.PathDistribution(info)


class TestTheNoticesStayExact:
    """一覧と写しが、実際に積んだ物と食い違わない"""

    def test_a_declared_license_that_is_missing_stops_it(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        """包みが書いている写しが 1 つでも無ければ止める

        1 つ見つかれば良しとすると、写しの欠けた一式を黙って配る
        """
        distribution = _distribution(tmp_path, ["LICENSE", "NOTICE"], ["LICENSE"])
        found, missing = builder.license_files(distribution)
        assert [name for name, _ in found] == ["LICENSE"]
        assert missing == ["NOTICE"]

        source = tmp_path / "sample" / "__init__.py"
        owners = {builder._key(source): distribution}
        problems = builder.collect_licenses(
            tmp_path / "bundle", [source], (), owners=owners, notices="| `sample` | 1.0 |\n"
        )
        assert any("NOTICE" in p for p in problems), problems

    def test_a_version_the_list_does_not_have_stops_it(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        """積んだ版と一覧の版が違えば止める

        依存は下限だけで指定しているので、新しい版が黙って入る 一覧の版と使用許諾が
        古いまま配ることになる
        """
        distribution = _distribution(tmp_path, ["LICENSE"], ["LICENSE"])
        source = tmp_path / "sample" / "__init__.py"
        problems = builder.collect_licenses(
            tmp_path / "bundle",
            [source],
            (),
            owners={builder._key(source): distribution},
            notices="| `sample` | 0.9 |\n",
        )
        assert any("0.9" in p and "1.0" in p for p in problems), problems

    def test_the_name_may_be_spelled_either_way(self, builder: ModuleType) -> None:
        """``PySide6-Essentials`` と ``PySide6_Essentials`` は同じ包み

        揃えないと、一覧に載っているのに「一覧に無い」と言って止まる
        """
        listed = builder.listed_versions("| `PySide6_Essentials` | 6.11.2 | LGPL |\n")
        assert listed[builder.canonical_name("PySide6-Essentials")] == "6.11.2"

    def test_the_real_list_is_read(self, builder: ModuleType) -> None:
        # リポジトリの一覧の表が読めること 読めないと、どの包みも「一覧に無い」になる
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        listed = builder.listed_versions(notices)
        assert listed["numpy"] == "2.5.3"
        assert listed["av"] == "18.1.0"
        assert listed["pyside6-essentials"] == "6.11.2"


class TestNativeLicenses:
    """wheel の中の DLL（PyAV の av.libs、lupa の LuaJIT）には、リポジトリの写しが要る"""

    def test_the_version_and_mark_are_stripped(self, builder: ModuleType) -> None:
        assert builder.native_base_name("libx264-165-f3a909470ddc2d85ed21eda3d0fb7954.dll") == (
            "libx264"
        )
        assert builder.native_base_name(
            "libopencore-amrnb-0-b8933128305cd084d40426d0d2c0d9ef.dll"
        ) == ("libopencore-amrnb")
        assert builder.native_base_name("libstdc++-6-2d5c346d47ad531ef9f5185db0e8cef3.dll") == (
            "libstdc++"
        )
        assert builder.native_base_name("luajit21.cp314-win_amd64.pyd") == "luajit21"

    def test_known_dlls_pass(self, builder: ModuleType, tmp_path: Path) -> None:
        internal = tmp_path / "_internal"
        (internal / "av.libs").mkdir(parents=True)
        (internal / "lupa").mkdir()
        for name in (
            "avcodec-62-984de33114b7fa384296817dec999c9d.dll",
            "libx265-efe48a158520a59ef99c0a0b3eb835ae.dll",
            "zlib1-79e0c4f9db71cb511398c504b832d395.dll",
        ):
            (internal / "av.libs" / name).write_bytes(b"MZ")
        (internal / "lupa" / "luajit20.cp314-win_amd64.pyd").write_bytes(b"MZ")
        # Lua 5.x の分は lupa 自身の写しにあるので見張らない
        (internal / "lupa" / "lua54.cp314-win_amd64.pyd").write_bytes(b"MZ")
        assert builder.native_license_problems(internal) == []

    def test_a_new_dll_stops_it(self, builder: ModuleType, tmp_path: Path) -> None:
        """PyAV を上げて DLL が増えたら止める 黙って通すと写しの無い部品を配る"""
        internal = tmp_path / "_internal"
        (internal / "av.libs").mkdir(parents=True)
        (internal / "av.libs" / "libaom-3-0123456789abcdef0123456789abcdef.dll").write_bytes(b"MZ")
        problems = builder.native_license_problems(internal)
        assert len(problems) == 1 and "libaom" in problems[0]

    def test_every_listed_copy_is_in_the_repository(self, builder: ModuleType) -> None:
        for folder in set(builder.NATIVE_LICENSES.values()):
            assert (ROOT / "licenses" / folder).is_dir(), folder


class TestOnlyRecordedFilesGoIn:
    """--skip-build で前の組み立てを使うとき、フォルダに混ざった物を zip に入れない"""

    def _record(self, tmp_path: Path) -> Path:
        record = tmp_path / "record"
        record.mkdir()
        entries = [
            ("Sashimono.exe", r"C:\work\Sashimono.exe", "EXECUTABLE"),
            ("PySide6\\Qt6Core.dll", r"C:\venv\PySide6\Qt6Core.dll", "BINARY"),
        ]
        (record / "COLLECT-00.toc").write_text(repr((entries,)), encoding="utf-8")
        return record

    def test_a_stray_dll_is_found(self, builder: ModuleType, tmp_path: Path) -> None:
        bundle = tmp_path / "Sashimono"
        (bundle / "_internal" / "PySide6").mkdir(parents=True)
        (bundle / "Sashimono.exe").write_bytes(b"MZ")
        (bundle / "_internal" / "PySide6" / "Qt6Core.dll").write_bytes(b"MZ")
        (bundle / "_internal" / "leftover.dll").write_bytes(b"MZ")
        builder.assemble(bundle)
        found = builder.untracked_files(bundle, self._record(tmp_path))
        # 説明書きや写しは記録に無くても入れてよい 混ざった DLL だけを挙げる
        assert found == ["_internal/leftover.dll"]


class TestTheUnpackedCopiesAreTheSame:
    def test_a_broken_copy_is_found(self, builder: ModuleType, tmp_path: Path) -> None:
        """展開した zip の写しが、組み立てたときの物と同じか

        名前だけを見ると、途中で消えた写しや壊れた写しでも通る
        """
        bundle = tmp_path / "Sashimono"
        bundle.mkdir()
        builder.assemble(bundle)
        expected = builder.notice_digests(bundle)
        assert "licenses/x264-b35605ac/COPYING" in expected
        assert "THIRD_PARTY_NOTICES.txt" in expected

        (bundle / "licenses" / "x264-b35605ac" / "COPYING").write_text("cut", encoding="utf-8")
        (bundle / "licenses" / "GPL-3.0.txt").unlink()
        changed = builder.changed_notices(bundle, expected)
        assert "中身が違う: licenses/x264-b35605ac/COPYING" in changed
        assert "無い: licenses/GPL-3.0.txt" in changed


class TestUnusedQtIsLeftOut:
    """PyInstaller がプラグインごと積む Qt の部品のうち、使わない物を外す

    外さないと zip が 18 MB ほど膨らみ、LGPL の部品として qtwebengine（580 MB）の
    ソースまで添付しなければならない
    """

    def test_the_listed_parts_are_removed(self, builder: ModuleType, tmp_path: Path) -> None:
        internal = tmp_path / "_internal"
        for relative in [*builder.UNUSED_QT_PARTS, "PySide6/Qt6Core.dll"]:
            (internal / relative).parent.mkdir(parents=True, exist_ok=True)
            (internal / relative).write_bytes(b"MZ")
        removed = builder.drop_unused_qt(tmp_path)
        assert set(removed) == set(builder.UNUSED_QT_PARTS)
        assert not (internal / "PySide6" / "Qt6Pdf.dll").exists()
        # 使う物には触らない
        assert (internal / "PySide6" / "Qt6Core.dll").exists()

    def test_a_part_still_in_use_stops_it(self, builder: ModuleType, tmp_path: Path) -> None:
        """外した DLL を残った物が読むなら止める 使った時点で落ちる zip になる"""
        internal = tmp_path / "_internal"
        (internal / "PySide6").mkdir(parents=True)
        (internal / "PySide6" / "QtQuick.pyd").write_bytes(b"MZ")
        (internal / "PySide6" / "Qt6Core.dll").write_bytes(b"MZ")
        imports = {"QtQuick.pyd": ["qt6quick.dll", "qt6core.dll"], "Qt6Core.dll": []}

        def reader(path: Path) -> list[str]:
            return imports[path.name]

        found = builder.dangling_imports(internal, builder.UNUSED_QT_PARTS, reader=reader)
        assert found == [("PySide6/QtQuick.pyd", "qt6quick.dll")]

    def test_nothing_dangles_when_nothing_uses_them(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        internal = tmp_path / "_internal"
        (internal / "PySide6").mkdir(parents=True)
        (internal / "PySide6" / "Qt6Widgets.dll").write_bytes(b"MZ")
        found = builder.dangling_imports(
            internal, builder.UNUSED_QT_PARTS, reader=lambda path: ["qt6core.dll"]
        )
        assert found == []

    def test_the_kept_qt_is_not_on_the_list(self, builder: ModuleType) -> None:
        # 画面・描画・絵の読み込みに使う物を誤って外すと、起動も描画もできない
        names = {Path(relative).name for relative in builder.UNUSED_QT_PARTS}
        for kept in (
            "Qt6Core.dll",
            "Qt6Gui.dll",
            "Qt6Widgets.dll",
            "Qt6OpenGL.dll",
            "qwindows.dll",
        ):
            assert kept not in names


class TestTheEditorCheckLeavesNoTrace:
    """編集画面の組み立ては、本人の設定に触れない

    編集画面は閉じるときに画面の並びを保存する 見せていない窓を閉じて保存すると、
    本人が整えた並びが既定の並びで上書きされる
    """

    def test_the_layout_is_not_written(self) -> None:
        from sashimono.selfcheck import _editor
        from sashimono.ui.workspace import config_root

        layout = config_root() / "workspace.ini"
        before = layout.read_bytes() if layout.exists() else None
        _editor()
        after = layout.read_bytes() if layout.exists() else None
        assert after == before, "確かめただけで画面の並びを書き換えている"

    def test_the_folders_come_back(self) -> None:
        # 向け先を戻し忘れると、そのあとの項目（スクリプト置き場）が一時フォルダを見る
        import os

        from sashimono.selfcheck import USER_FOLDER_VARIABLES, _isolated_user_folders

        before = {name: os.environ.get(name) for name in USER_FOLDER_VARIABLES}
        with _isolated_user_folders():
            assert os.environ["APPDATA"] != before["APPDATA"]
        assert {name: os.environ.get(name) for name in USER_FOLDER_VARIABLES} == before


class TestTheExportCheckUsesTheCpu:
    def test_without_libx264_it_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CPU の符号化器が無ければ落とす GPU の符号化器へ逃げない

        逃げると、GPU の符号化器がある開発機では通り、無い機械では書き出せない
        zip を「動いた」として配ることになる
        """
        from sashimono import selfcheck

        monkeypatch.setattr(
            "sashimono.engine.encode.available_video_codecs", lambda: ["h264_nvenc", "h264_qsv"]
        )
        with pytest.raises(RuntimeError, match="libx264"):
            selfcheck._export()


class TestWithoutAConsole:
    """窓だけの exe をパイプ無しで起動すると、標準出力が無い（``sys.stdout`` が None）

    ダブルクリックやコマンドでそのまま打つとこうなる 書いても誰にも届かず、
    自己診断を頼んだ人には何も起きないように見える
    """

    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
        shown: list[tuple[str, bool]] = []

        def fake(text: str, title: str, *, warning: bool) -> None:
            shown.append((text, warning))

        monkeypatch.setattr("sashimono.selfcheck._native_message", fake)
        monkeypatch.setattr(sys, "stdout", None)
        return shown

    def test_the_result_is_shown_in_a_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shown = self._capture(monkeypatch)
        monkeypatch.setattr(
            "sashimono.selfcheck.run_self_check", lambda: [CheckResult("GL で描く", True, "描けた")]
        )
        assert main(["sashimono", SELF_CHECK_FLAG]) == 0
        assert shown and "GL で描く" in shown[0][0]
        assert shown[0][1] is False

    def test_a_failure_is_shown_as_a_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 落ちた項目があるときは目立つ形で出す 情報の窓だと読み流される
        shown = self._capture(monkeypatch)
        monkeypatch.setattr(
            "sashimono.selfcheck.run_self_check", lambda: [CheckResult("GL で描く", False, "x")]
        )
        assert main(["sashimono", SELF_CHECK_FLAG]) == 1
        assert shown and shown[0][1] is True

    def test_it_does_not_need_qt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Qt が落ちていても結果は見せる

        Qt の DLL を積み忘れたときこそ結果を見せたい Qt の窓で出そうとすると、
        まさにそのときに何も出ない
        """
        shown = self._capture(monkeypatch)
        monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
        monkeypatch.setattr(
            "sashimono.selfcheck.run_self_check", lambda: [CheckResult("Qt", False, "DLL が無い")]
        )
        assert main(["sashimono", SELF_CHECK_FLAG]) == 1
        assert shown and "DLL が無い" in shown[0][0]


class TestOnlyCheckedZipsRemain:
    """dist に zip がある＝確かめ済み

    確かめて落ちた zip を残すと、動かない物を完成品と取り違えて配る
    """

    def _bundle(self, tmp_path: Path) -> Path:
        bundle = tmp_path / "Sashimono"
        bundle.mkdir()
        (bundle / "Sashimono.exe").write_bytes(b"MZ")
        return bundle

    def test_a_failed_check_takes_the_zip_away(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(builder, "smoke_test", lambda archive, notices: 1)
        target = tmp_path / "out.zip"
        assert builder.package(self._bundle(tmp_path), target) == 1
        assert not target.exists(), "確かめて落ちた zip が残っている"

    def test_the_folder_stays_for_a_look(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 何が足りないかを調べるには、組み立てたフォルダの方が要る
        monkeypatch.setattr(builder, "smoke_test", lambda archive, notices: 1)
        bundle = self._bundle(tmp_path)
        builder.package(bundle, tmp_path / "out.zip")
        assert (bundle / "Sashimono.exe").exists()

    def test_a_passed_check_keeps_it(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(builder, "smoke_test", lambda archive, notices: 0)
        target = tmp_path / "out.zip"
        assert builder.package(self._bundle(tmp_path), target) == 0
        assert target.exists()


class TestNothingUncheckedIsLeft:
    """どの段で落ちても、確かめていない zip を完成品の名前で残さない"""

    def test_a_crash_while_checking_takes_the_zip_away(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """展開できない・exe が返ってこない（時間切れ）でも同じ"""
        bundle = tmp_path / "Sashimono"
        bundle.mkdir()
        (bundle / "Sashimono.exe").write_bytes(b"MZ")

        def crash(archive: Path, notices: object) -> int:
            raise TimeoutError("exe が返ってこない")

        monkeypatch.setattr(builder, "smoke_test", crash)
        target = tmp_path / "out.zip"
        with pytest.raises(TimeoutError):
            builder.package(bundle, target)
        assert not target.exists()

    def test_the_previous_zip_goes_before_building(
        self, builder: ModuleType, tmp_path: Path
    ) -> None:
        """組み立てる前に前の zip を消す

        組み立てで落ちると zip を作る所まで進まない そこで消していては、
        前の物が今回の完成品に見えて残る
        """
        from sashimono import __version__

        old = tmp_path / f"SashimonoEdit-{__version__}-windows-x64.zip"
        old.write_bytes(b"previous build")
        # 組み立て済みの exe が無い＝組み立てに失敗した状態
        assert builder.main(["--skip-build"], dist=tmp_path) == 1
        assert not old.exists(), "組み立てに失敗したのに前の zip が残っている"


class TestTheExportCheckCountsFrames:
    # 書き出しは中で GL のコンテキストを作る GPU の無い CI では、切れたかを
    # 見る前に GL で落ちて、確かめたいことを確かめられない
    @pytest.mark.usefixtures("gpu")
    def test_a_truncated_export_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """書き出しが黙って途中で切れたら落とす

        1 コマでも読めれば通す形だと、見本（2 コマ）が 1 コマに切れても気づけない
        """
        from dataclasses import replace

        import sashimono.engine.encode as encode
        from sashimono import selfcheck

        real = encode.export_project

        def truncated(project: Project, settings: encode.ExportSettings, **kwargs: object) -> Path:
            return real(project, replace(settings, frame_range=(0, 1)))

        monkeypatch.setattr(encode, "export_project", truncated)
        with pytest.raises(RuntimeError, match="1 コマ"):
            selfcheck._export()


class TestTheEntryDoesNotNeedQt:
    """配布版は同じ exe が 3 つの役をする（編集画面・自己診断・導入ボタンの pip）

    入口の一番上で Qt を読むと、Qt の部品が欠けた配布版では自己診断にたどり着く
    前に落ちる 欠けたことを知りたいまさにそのときに、結果の窓も出ない
    別のプロセスで確かめる（このプロセスはもう Qt を読んでいる）
    """

    def _run(self, code: str, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
        return subprocess.run(
            [sys.executable, "-c", code, *extra],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            env=environment,
            timeout=300,
            check=False,
        )

    def test_the_entry_does_not_load_qt(self) -> None:
        # pip を走らせるだけのために Qt 一式を読まない
        completed = self._run("import sys, sashimono.app; print('PySide6' in sys.modules)")
        assert completed.stdout.strip() == "False", completed.stderr

    def test_the_self_check_reports_a_missing_qt(self, tmp_path: Path) -> None:
        """Qt が読めなくても自己診断は最後まで走り、結果を窓へ渡して 1 を返す"""
        shown = tmp_path / "shown.txt"
        code = (
            "import sys\n"
            "sys.modules['PySide6'] = None  # Qt の部品が欠けた配布版の代わり\n"
            "import sashimono.selfcheck as check\n"
            "def show(text, title, *, warning):\n"
            "    open(sys.argv[1], 'w', encoding='utf-8').write(text)\n"
            "check._native_message = show\n"
            "sys.stdout = None  # パイプ無しで起動した窓だけの exe の代わり\n"
            "from sashimono.app import main\n"
            "raise SystemExit(main(['sashimono', '--self-check']))\n"
        )
        completed = self._run(code, str(shown))
        assert completed.returncode == 1, completed.stderr
        assert shown.exists(), "結果の窓が出ていない"
        assert "[NG] Qt" in shown.read_text(encoding="utf-8")


def test_the_bundled_pictures_are_collected(builder: ModuleType) -> None:
    """同梱の絵（アイコン・磁石の印）を積む引数が組み立てに入っている

    抜けると、配る版でボタンの印が黙って消える 揃っているかは自己診断（同梱の絵）も見る
    """
    arguments = builder.pyinstaller_arguments(Path("work"), Path("dist"))
    index = arguments.index("--collect-data")
    assert arguments[index + 1] == "sashimono.resources"


def _values(arguments: list[str], flag: str) -> set[str]:
    return {arguments[i + 1] for i, value in enumerate(arguments) if value == flag}


class TestTheStandardLibraryGoesIn:
    """後から入れる部品（AI 連携・字幕起こし）が使う標準ライブラリを配布版に積む

    PyInstaller は本体が import する物しか積まない AI 連携を入れて送った途端に
    ``No module named 'zoneinfo'`` で止まった（pydantic が読む 利用者の画面）
    """

    def test_the_whole_library_is_asked_for(self, builder: ModuleType) -> None:
        arguments = builder.pyinstaller_arguments(Path("w"), Path("d"))
        asked = _values(arguments, "--hidden-import") | _values(arguments, "--collect-submodules")
        assert {"zoneinfo", "email", "xml", "json", "asyncio", "sqlite3", "tomllib"} <= asked
        # 包みは下の部品まで（email.mime.text など）
        assert {"email", "xml", "concurrent"} <= _values(arguments, "--collect-submodules")
        # 画面の部品（Tk）と Python 自身の試験は積まない
        assert not {"tkinter", "test", "idlelib"} & asked

    def _record(self, tmp_path: Path, pyz: list[str]) -> Path:
        (tmp_path / "COLLECT-00.toc").write_text(
            repr(([("select.pyd", r"C:\py\select.pyd", "EXTENSION")],)), encoding="utf-8"
        )
        entries = [(name, rf"C:\py\{name}.py", "PYMODULE") for name in pyz]
        (tmp_path / "PYZ-00.toc").write_text(repr((r"C:\PYZ", entries)), encoding="utf-8")
        with zipfile.ZipFile(tmp_path / "base_library.zip", "w") as base:
            base.writestr("os.pyc", b"")
        return tmp_path

    def test_a_missing_one_stops_the_build(
        self, builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 前の組み立ては zoneinfo を持たず、AI 連携が送った途端に落ちた
        monkeypatch.setattr(builder, "runtime_distributions", lambda: ["x"])
        monkeypatch.setattr(
            builder, "imported_stdlib", lambda d: {"zoneinfo", "json", "os", "select", "fcntl"}
        )
        record = self._record(tmp_path, ["json"])
        # fcntl は Windows に無いので数えない（積めず、Windows では読まれない）
        assert builder.missing_stdlib(record) == ["zoneinfo"]
        assert builder.missing_stdlib(self._record(tmp_path, ["json", "zoneinfo"])) == []

    def test_the_imports_are_read_from_the_real_sdk(self, builder: ModuleType) -> None:
        # この機械に入っている AI 連携の部品から、使う標準ライブラリを字面で集められる
        found = builder.runtime_distributions()
        if not found:
            pytest.skip("AI 連携の部品がこの機械に入っていない")
        needed = builder.imported_stdlib(found)
        assert "zoneinfo" in needed and "asyncio" in needed


class TestTheImportCheck:
    """配布版の exe の中で部品を import してみる口（組み立ての道具が使う）"""

    def test_the_path_files_of_the_runtime_are_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # pywin32 は .pth で win32/lib を探す道へ足す 入れた置き場の .pth を読まないと、
        # 配布版では pywintypes が無いと言って AI 連携が動かなかった（組み立ての確かめで分かった）
        from sashimono.runtime import read_path_files

        inner = tmp_path / "win32" / "lib"
        inner.mkdir(parents=True)
        (inner / "sashimono_pth_probe.py").write_text("VALUE = 1\n", encoding="utf-8")
        (tmp_path / "probe.pth").write_text("win32/lib\n", encoding="utf-8")
        monkeypatch.setattr(sys, "path", list(sys.path))
        read_path_files(str(tmp_path))
        assert str(inner) in sys.path

    def test_a_readable_module_passes(self, tmp_path: Path) -> None:
        (tmp_path / "sashimono_import_probe.py").write_text("VALUE = 1\n", encoding="utf-8")
        assert main(["sashimono", "--import-check", str(tmp_path), "sashimono_import_probe"]) == 0

    def test_a_missing_module_fails_and_says_which(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "sashimono_import_broken.py").write_text(
            "import no_such_stdlib_part\n", encoding="utf-8"
        )
        assert main(["sashimono", "--import-check", str(tmp_path), "sashimono_import_broken"]) == 1
        assert "no_such_stdlib_part" in capsys.readouterr().out


class TestTheAssistantSaysWhatIsMissing:
    """AI アシスタントの窓に ModuleNotFoundError のまま出さない（利用者の画面）"""

    def test_a_missing_standard_part_points_to_an_update(self) -> None:
        from sashimono.ai.session import _explain

        text = _explain(ModuleNotFoundError("No module named 'zoneinfo'", name="zoneinfo"))
        assert "zoneinfo" in text and "標準" in text and "更新" in text
        assert "ModuleNotFoundError" not in text

    def test_a_missing_add_on_points_to_reinstalling(self) -> None:
        from sashimono.ai.session import REINSTALL_HINT, _explain

        text = _explain(ModuleNotFoundError("No module named 'anyio'", name="anyio"))
        assert "anyio" in text and REINSTALL_HINT in text


class TestTheUpdateParts:
    """自動更新の部品が配布版の中で動くかを、zip からの確かめが見る（ネットワークへは出ない）"""

    def test_the_build_info_goes_into_the_zip(self, builder: ModuleType, tmp_path: Path) -> None:
        """書き付けが無いと、配った版がこの zip を新しい版として受け取れない"""
        from sashimono import __version__
        from sashimono.runtime import python_abi
        from sashimono.update.package import read_build_info

        bundle = tmp_path / "Sashimono"
        bundle.mkdir()
        builder.assemble(bundle)
        info = read_build_info(bundle)
        assert info is not None
        assert (info.version, info.python_abi) == (__version__, python_abi())
        # 使用許諾の照合には混ぜない（写しではない）
        assert builder.BUILD_INFO_NAME not in builder.notice_digests(bundle)

    def test_the_build_info_is_not_a_stray_file(self, builder: ModuleType, tmp_path: Path) -> None:
        bundle = tmp_path / "Sashimono"
        bundle.mkdir()
        (bundle / "Sashimono.exe").write_bytes(b"MZ")
        builder.assemble(bundle)
        record = tmp_path / "record"
        record.mkdir()
        entries = [("Sashimono.exe", r"C:\work\Sashimono.exe", "EXECUTABLE")]
        (record / "COLLECT-00.toc").write_text(repr((entries,)), encoding="utf-8")
        assert builder.untracked_files(bundle, record) == []

    def test_the_exe_name_matches(self, builder: ModuleType) -> None:
        """入れ替え係が起こす名前と、組み立てる名前が食い違うと、入れた版を起こせない"""
        from sashimono.update.package import APP_EXE

        assert f"{builder.APP_NAME}.exe" == APP_EXE

    def test_a_failing_update_item_fails_the_zip(self, builder: ModuleType, tmp_path: Path) -> None:
        """自己診断の「自動更新」が通らない zip・書き付けの無い zip は配らない"""
        home = tmp_path / "Sashimono"
        home.mkdir()
        builder.assemble(home)
        passed = f"[ok] {builder.UPDATE_CHECK_NAME}: 確かめた"
        assert builder.update_failures(home, passed) == []
        assert builder.update_failures(home, f"[NG] {builder.UPDATE_CHECK_NAME}: 落ちた")
        (home / builder.BUILD_INFO_NAME).unlink()
        assert builder.update_failures(home, passed)

    def test_the_self_check_rehearses_an_update(self) -> None:
        """使い捨ての鍵と見本のリリースで、署名・照合・展開・入れ替えまでを通す"""
        from sashimono.selfcheck import _update

        detail = _update()
        assert "署名・照合・展開" in detail
        assert "公開鍵" in detail
        if sys.platform == "win32":
            assert "入れ替え（PowerShell）" in detail


class TestTheSelfCheckOnAnEnglishWindows:
    """英語の Windows（CI の Windows も同じ）では、パイプの文字コードが cp1252 になる

    日本語を 1 文字も書けず、結果を出す前に落ちていた 終了コードだけが 1 になり、
    どの部品が動かないのかを知る手段が無くなる
    """

    def test_a_narrow_pipe_is_switched_to_utf8(self) -> None:
        from sashimono.selfcheck import _make_writable

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        text = "[ok] 版: 0.1.0（配布版）"
        _make_writable(stream, text)
        stream.write(text)
        stream.flush()
        assert stream.encoding == "utf-8"

    def test_a_japanese_windows_keeps_its_own(self) -> None:
        """書ける出口は変えない 変えると ``| more`` で読む人の画面が化ける"""
        from sashimono.selfcheck import _make_writable

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
        _make_writable(stream, "[ok] 版: 0.1.0（配布版）")
        assert stream.encoding == "cp932"


class TestTheVcRuntimeCheck:
    """配布版が Visual C++ の実行時の部品を、zip の外から借りていないか

    開発機にも CI の Windows にも再頒布可能パッケージが入っていて、積み忘れても動いてしまう
    入っていない機械では、自己診断にすらたどり着かずに起動の時点で落ちる
    """

    def _bundle(self, tmp_path: Path) -> Path:
        bundle = tmp_path / "展開 先" / "Sashimono"
        (bundle / "_internal" / "PySide6").mkdir(parents=True)
        (bundle / "_internal" / "VCRUNTIME140.dll").write_bytes(b"MZ")
        (bundle / "_internal" / "PySide6" / "MSVCP140.dll").write_bytes(b"MZ")
        return bundle

    def test_parts_from_inside_pass(self, tmp_path: Path) -> None:
        from sashimono.selfcheck import vc_runtime_report

        bundle = self._bundle(tmp_path)
        loaded = [
            bundle / "_internal" / "VCRUNTIME140.dll",
            bundle / "_internal" / "PySide6" / "MSVCP140.dll",
            Path(r"C:\Windows\System32\kernel32.dll"),
        ]
        assert "中の 2 個" in vc_runtime_report(loaded, bundle)

    def test_an_outside_copy_of_a_bundled_part_passes(self, tmp_path: Path) -> None:
        """同じ名前を配布版が持っていれば、入っていない機械ではそちらが読まれる"""
        from sashimono.selfcheck import vc_runtime_report

        bundle = self._bundle(tmp_path)
        loaded = [
            bundle / "_internal" / "VCRUNTIME140.dll",
            Path(r"C:\Windows\System32\msvcp140.dll"),
        ]
        assert "msvcp140.dll" in vc_runtime_report(loaded, bundle)

    def test_a_part_only_windows_has_fails(self, tmp_path: Path) -> None:
        from sashimono.selfcheck import vc_runtime_report

        bundle = self._bundle(tmp_path)
        loaded = [
            bundle / "_internal" / "VCRUNTIME140.dll",
            Path(r"C:\Windows\System32\concrt140.dll"),
        ]
        with pytest.raises(RuntimeError, match=r"concrt140\.dll"):
            vc_runtime_report(loaded, bundle)

    def test_nothing_counted_is_not_taken_as_fine(self, tmp_path: Path) -> None:
        """配布版の Python 自身が vcruntime140.dll を読む 数えられないのに通すと何も見ていない"""
        from sashimono.selfcheck import vc_runtime_report

        with pytest.raises(RuntimeError, match="数えられない"):
            vc_runtime_report([Path(r"C:\Windows\System32\kernel32.dll")], self._bundle(tmp_path))

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows の DLL の一覧を読む")
    def test_this_process_is_counted(self) -> None:
        from sashimono.selfcheck import VC_RUNTIME_PREFIXES, loaded_modules

        names = [path.name.lower() for path in loaded_modules()]
        assert any(name.startswith("python3") for name in names)
        assert any(name.startswith(VC_RUNTIME_PREFIXES) for name in names)

    def test_development_is_not_judged(self) -> None:
        # 開発環境の Python は再頒布可能パッケージ込みで入っていて、照らす配布版が無い
        from sashimono.selfcheck import _vc_runtime

        assert "開発環境" in _vc_runtime()


class TestTheEncodeCheckNeedsNoGL:
    """GPU の無い機械でも FFmpeg の部品は確かめる 書き出す項目は GL で先に落ちる"""

    def test_it_encodes_into_a_japanese_folder(self) -> None:
        from sashimono.selfcheck import JAPANESE_FOLDER, _encode

        assert " " in JAPANESE_FOLDER and not JAPANESE_FOLDER.isascii()
        assert JAPANESE_FOLDER in _encode()

    def test_it_does_not_touch_gl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sashimono import selfcheck

        def no_gl(*args: object, **kwargs: object) -> None:
            raise AssertionError("GL を使った")

        monkeypatch.setattr("sashimono.engine.gpu.OffscreenGLContext", no_gl)
        monkeypatch.setattr("sashimono.engine.gpu.context.OffscreenGLContext", no_gl)
        assert "読み戻せた" in selfcheck._encode()


class TestTheBuildIsPinnedToTheList:
    """CI のまっさらな機械で組むときは、一覧（THIRD_PARTY_NOTICES.md）の版に留めて入れる

    依存は下限だけなので、留めないとその日の最新が入る av 19 が出た日に、写しの無い
    DLL（libvmaf）を積み、一覧と版が食い違って組み立てが止まった（Issue #33 の CI）
    """

    def test_the_listed_packages_are_pinned(self, builder: ModuleType) -> None:
        lines = builder.constraints(builder.NOTICES_SOURCE.read_text(encoding="utf-8"))
        names = {line.split("==")[0] for line in lines}
        assert {"av", "pyside6-essentials", "cryptography", "pyinstaller"} <= names
        assert all(re.fullmatch(r"[a-z0-9-]+==\d[0-9A-Za-z.!+]*", line) for line in lines), lines

    def test_the_file_table_is_not_taken_for_packages(self, builder: ModuleType) -> None:
        # 同梱のファイルの表（| `LICENSE.txt` | 説明 |）も 1 列目が ` で始まる 版の形でない
        # 行まで制約にすると、uv が制約を読めずに止まる
        notices = "| `LICENSE.txt` | 本体の使用許諾 |\n| `av`（PyAV） | 18.1.0 | BSD |\n"
        assert builder.constraints(notices) == ["av==18.1.0"]

    def test_they_are_written_without_building(self, builder: ModuleType, tmp_path: Path) -> None:
        target = tmp_path / "build" / "constraints.txt"
        assert builder.main(["--write-constraints", str(target)], dist=tmp_path) == 0
        assert "av==" in target.read_text(encoding="utf-8")
        assert not (tmp_path / "Sashimono").exists()


class TestTheSoftwareGLIsLeftOut:
    def test_qts_software_gl_is_not_shipped(self, builder: ModuleType) -> None:
        """Qt のソフトウェアの GL（opengl32sw.dll 20 MB）は積まない

        取れるのは OpenGL 3.0 までで描画に要る 4.3 に届かず、描く関数は PyOpenGL が
        Windows の opengl32.dll から引くので、Qt がこちらで作ったコンテキストへ届かない
        積んでいても GPU の無い機械で描けないことは変わらない
        """
        assert "PySide6/opengl32sw.dll" in builder.UNUSED_QT_PARTS
