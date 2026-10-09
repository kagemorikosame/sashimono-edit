"""入れ替え係（PowerShell の台本）を本当に走らせる

本体の代わりに小さな .cmd を置く 新しい版の .cmd は、起動できた印を書くか、書かずに
終わる（起動に失敗した版の代わり） 前の版の .cmd は、起こされたことを隣に書き残す
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from sashimono.update import swap as swap_module
from sashimono.update.package import Layout, write_build_info
from sashimono.update.swap import (
    BUNDLED_SCRIPT,
    HEALTH_ENV,
    HELPER_SCRIPT,
    SWAP_CONTRACT,
    SwapPlan,
    helper_script,
    launch,
    leave_install_folder,
    mark_started,
    take_result,
    wait_started,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="入れ替え係は Windows の PowerShell"
)

APP = "app.cmd"

#: 起動できた印を書く新しい版
HEALTHY = f'@echo ok> "%{HEALTH_ENV}%"\r\n'
#: 印を書かずに落ちる新しい版
BROKEN = "@exit /b 1\r\n"
#: 起こされたことを隣に書き残す前の版
OLD = '@echo old> "%~dp0..\\old-started.txt"\r\n'


def _folder(path: Path, script: str) -> None:
    path.mkdir(parents=True)
    (path / APP).write_text(script, encoding="ascii")


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    # 日本語の名前の場所で通す 本人の名前が日本語の機械はよくある
    install = tmp_path / "利用者" / "Sashimono"
    _folder(install, OLD)
    (install / "version.txt").write_text("old", encoding="ascii")
    return Layout(install)


def _run(plan: SwapPlan, folder: Path) -> list[str]:
    launched = launch(plan, folder)
    assert wait_started(launched)
    launched.process.wait(timeout=120)
    return take_result(folder)


def _wait_for(path: Path, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return path.exists()


def _plan(layout: Layout, mode: str = "apply", **changes: object) -> SwapPlan:
    options: dict[str, object] = {
        "exe_name": APP,
        "hidden": True,
        "wait_seconds": 10,
        "health_seconds": 10,
    }
    options.update(changes)
    return SwapPlan(mode, layout, **options)  # type: ignore[arg-type]


class TestApplying:
    def test_the_new_version_takes_the_place(self, layout: Layout, tmp_path: Path) -> None:
        _folder(layout.staged, HEALTHY)
        (layout.staged / "version.txt").write_text("new", encoding="ascii")
        lines = _run(_plan(layout), tmp_path / "update")
        assert "healthy" in lines, lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "new"
        # 前の版は 1 世代だけ残す（戻すため）
        assert (layout.previous / "version.txt").read_text(encoding="ascii") == "old"
        assert not layout.staged.exists()

    def test_an_older_previous_is_replaced(self, layout: Layout, tmp_path: Path) -> None:
        """残すのは 1 世代だけ 毎回残すと、更新のたびに 200 MB ずつ増える"""
        _folder(layout.previous, OLD)
        (layout.previous / "version.txt").write_text("older", encoding="ascii")
        _folder(layout.staged, HEALTHY)
        _run(_plan(layout), tmp_path / "update")
        assert (layout.previous / "version.txt").read_text(encoding="ascii") == "old"

    def test_a_version_that_cannot_start_is_rolled_back(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """入れた版が 2 回続けて起動できなければ、前の版へ戻して前の版を起こす"""
        _folder(layout.staged, BROKEN)
        (layout.staged / "version.txt").write_text("new", encoding="ascii")
        lines = _run(_plan(layout), tmp_path / "update")
        assert "rolled-back" in lines, lines
        assert lines.count("start-failed 1") == 1 and "start-failed 2" in lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "old"
        assert (layout.failed / "version.txt").read_text(encoding="ascii") == "new"
        assert _wait_for(layout.install.parent / "old-started.txt")

    def test_without_relaunch_nothing_is_started(self, layout: Layout, tmp_path: Path) -> None:
        """自己診断の試しでは入れ替えるだけ 何も起こさない"""
        _folder(layout.staged, BROKEN)
        lines = _run(_plan(layout, relaunch=False, check_start=False), tmp_path / "update")
        assert lines[-1] == "swapped"

    def test_it_waits_for_the_app_to_end(self, layout: Layout, tmp_path: Path) -> None:
        """本体が動いている間はフォルダを動かさない（動かせない）"""
        _folder(layout.staged, HEALTHY)
        import subprocess

        sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3)"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            started = time.monotonic()
            lines = _run(_plan(layout, pid=sleeper.pid, relaunch=False), tmp_path / "update")
            assert time.monotonic() - started >= 2
        finally:
            sleeper.kill()
        assert "swapped" in lines

    def test_a_locked_folder_restores_the_current_version(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """今の版のフォルダを動かせなければ、何も替えずに今の版を起こし直す"""
        _folder(layout.staged, HEALTHY)
        held = (layout.install / "version.txt").open("rb")
        try:
            # Windows は開いているファイルのあるフォルダの改名を断る（動いている exe と同じ）
            lines = _run(_plan(layout), tmp_path / "update")
        finally:
            held.close()
        assert "install-locked" in lines, lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "old"
        assert layout.staged.exists()
        assert _wait_for(layout.install.parent / "old-started.txt")


#: 起動できた印を書き、自分の作業場所と、起こし直した印を隣に書き残す新しい版
HEALTHY_WHERE = (
    '@cd> "%~dp0..\\new-cwd.txt"\r\n'
    # 振り向けを前に書く 後ろに書くと、値の 1 が「1>」として振り向けの番号に読まれる
    '@> "%~dp0..\\new-relaunched.txt" echo %SASHIMONO_UPDATE_RELAUNCHED%\r\n' + HEALTHY
)


def _read_line(path: Path) -> str:
    # cmd の echo は本人の文字コード（日本語の Windows では cp932）で書く
    return path.read_bytes().decode("mbcs", "replace").strip()


class TestTheWorkingFolder:
    """入れ替え係の作業場所がインストール先の中になっても入れ替わる（#279）

    Windows は、どれかのプロセスの作業場所になっているフォルダの名前を変えさせない
    Explorer から起こした本体は作業場所がインストール先で、それを受け継いだ入れ替え係が、
    20 回（10 秒）続けて自分で改名を断らせ、install-locked で終わっていた
    """

    def test_an_app_working_in_the_install_folder_can_still_update(
        self, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """本体の作業場所がインストール先のまま入れ替え係を起こしても入れ替わる"""
        _folder(layout.staged, HEALTHY_WHERE)
        (layout.staged / "version.txt").write_text("new", encoding="ascii")
        folder = tmp_path / "update"
        monkeypatch.chdir(layout.install)
        launched = launch(_plan(layout), folder)
        # 試験のプロセスは終わらないので、起こした直後に作業場所を外へ戻す（本物の本体は終わる）
        # 戻さないと、試験のプロセスが改名を断らせ、入れ替え係の作業場所を確かめられない
        os.chdir(tmp_path)
        assert wait_started(launched)
        launched.process.wait(timeout=120)
        lines = take_result(folder)
        assert "install-locked" not in lines and "healthy" in lines, lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "new"

    def test_the_script_leaves_the_install_folder_by_itself(
        self, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """起こす側が作業場所を渡さない道（ほかの起こし方）でも、台本が自分で外へ移る"""
        _folder(layout.staged, HEALTHY)
        (layout.staged / "version.txt").write_text("new", encoding="ascii")
        import subprocess

        real_popen = subprocess.Popen

        def inside_install(*arguments: Any, **options: Any) -> Any:
            options["cwd"] = layout.install
            return real_popen(*arguments, **options)

        monkeypatch.setattr(subprocess, "Popen", inside_install)
        lines = _run(_plan(layout, relaunch=False), tmp_path / "update")
        assert "install-locked" not in lines and lines[-1] == "swapped", lines

    def test_the_new_version_is_started_outside_the_install_folder(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """起こし直した本体の作業場所もインストール先の外 入れ替え係に起こされた印も渡る

        インストール先で起こすと、前の版へ戻したときの古い版（自分では外へ移さない）が、
        次の更新でまた改名を断らせる 印が無いと、新しい版は入れ替え係の錠を見て終わる
        """
        _folder(layout.staged, HEALTHY_WHERE)
        lines = _run(_plan(layout), tmp_path / "update")
        assert "healthy" in lines, lines
        where = Path(_read_line(layout.install.parent / "new-cwd.txt"))
        assert not str(where).lower().startswith(str(layout.install).lower()), where
        assert _read_line(layout.install.parent / "new-relaunched.txt") == "1"


class TestTheNewVersionsScript:
    """入れ替え係の台本は新しい版の物を使う（#279）

    今の版の台本を使うと、台本の不具合を直しても 1 つ前の版からの更新には効かない
    """

    def _stage_script(self, layout: Layout, script: str, *, contract: int = SWAP_CONTRACT) -> None:
        _folder(layout.staged, HEALTHY)
        (layout.staged / "version.txt").write_text("new", encoding="ascii")
        path = layout.staged / BUNDLED_SCRIPT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script, encoding="utf-8-sig")
        write_build_info(
            layout.staged, "9.0.0", "cp314", swap_contract=contract, swap_script=BUNDLED_SCRIPT
        )

    def test_the_staged_script_runs_the_swap(self, layout: Layout, tmp_path: Path) -> None:
        marked = HELPER_SCRIPT.replace(
            "Write-Result 'holding'\n", "Write-Result 'holding'\n    Write-Result 'new-script'\n", 1
        )
        assert marked != HELPER_SCRIPT
        self._stage_script(layout, marked)
        folder = tmp_path / "update"
        launched = launch(_plan(layout), folder)
        assert launched.from_staged
        assert wait_started(launched)
        launched.process.wait(timeout=120)
        lines = take_result(folder)
        assert "new-script" in lines and "healthy" in lines, lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "new"

    def test_a_different_contract_keeps_the_current_script(self, layout: Layout) -> None:
        """受け渡し（環境変数・結果の言葉・錠）の合わない台本は走らせない"""
        self._stage_script(layout, "exit 1", contract=SWAP_CONTRACT + 1)
        assert helper_script(_plan(layout)) == (HELPER_SCRIPT, False)

    def test_a_version_without_a_script_keeps_the_current_script(self, layout: Layout) -> None:
        """0.2.0 までの版は台本を持たない その版へ戻すときも、今の版の台本で入れ替える"""
        _folder(layout.staged, HEALTHY)
        write_build_info(layout.staged, "9.0.0", "cp314")
        assert helper_script(_plan(layout)) == (HELPER_SCRIPT, False)

    def test_a_missing_or_empty_script_keeps_the_current_script(self, layout: Layout) -> None:
        self._stage_script(layout, "  \n")
        assert helper_script(_plan(layout)) == (HELPER_SCRIPT, False)
        (layout.staged / BUNDLED_SCRIPT).unlink()
        assert helper_script(_plan(layout)) == (HELPER_SCRIPT, False)

    def test_a_script_outside_the_new_version_is_not_run(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """書き付けが .new の外を指していたら使わない 確かめた中身ではない"""
        self._stage_script(layout, HELPER_SCRIPT)
        outside = tmp_path / "outside.ps1"
        outside.write_text("exit 1", encoding="utf-8")
        for pointed in ("../outside.ps1", str(outside), "."):
            write_build_info(
                layout.staged, "9.0.0", "cp314", swap_contract=SWAP_CONTRACT, swap_script=pointed
            )
            assert helper_script(_plan(layout)) == (HELPER_SCRIPT, False), pointed

    def test_rolling_back_uses_the_current_script(self, layout: Layout) -> None:
        self._stage_script(layout, "exit 1")
        assert helper_script(_plan(layout, "rollback")) == (HELPER_SCRIPT, False)


class TestLeavingTheInstallFolder:
    """本体は起動の頭で、作業場所をインストール先の外へ移す（#279）"""

    def test_an_app_started_from_explorer_moves_out(
        self, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.chdir(layout.install / ".")
        assert leave_install_folder(layout.install, home) == home
        assert Path.cwd() == home

    def test_a_folder_inside_the_install_folder_also_moves_out(
        self, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = layout.install / "SCRIPTS"
        inner.mkdir()
        monkeypatch.chdir(inner)
        # Windows は大文字小文字を区別しない 綴りが違っても中と見る
        assert leave_install_folder(Path(str(layout.install).upper()), tmp_path) == tmp_path

    def test_a_command_line_start_elsewhere_stays(
        self, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """コマンドの行から別の所で起こした人は、そこを基準に相対の場所を渡している"""
        monkeypatch.chdir(tmp_path)
        assert leave_install_folder(layout.install, tmp_path / "home") is None
        assert Path.cwd() == tmp_path

    def test_the_development_tree_stays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert leave_install_folder(None, tmp_path / "home") is None
        assert Path.cwd() == tmp_path


class TestTwoWindows:
    def test_only_one_swapper_runs(self, layout: Layout, tmp_path: Path) -> None:
        """2 つの窓で〔今すぐ再起動して入れる〕を選んでも、入れ替えるのは 1 つだけ

        2 つ目も走ると、1 つ目が作った .previous を消し、戻す先まで失う
        """
        _folder(layout.staged, HEALTHY)
        (layout.staged / "version.txt").write_text("new", encoding="ascii")
        import subprocess

        # 1 つ目の入れ替え係は、窓（の代わりの子）が終わるのを待っている間、錠を持ち続ける
        window = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(4)"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        folder = tmp_path / "update"
        try:
            first = launch(_plan(layout, pid=window.pid), folder)
            assert wait_started(first)
            second = launch(_plan(layout), folder)
            assert not wait_started(second)
            assert second.lost_to_another()
            second.process.wait(timeout=60)
            first.process.wait(timeout=120)
        finally:
            window.kill()
        lines = take_result(folder)
        assert lines.count("holding") == 1 and lines.count("already-running") == 1, lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "new"
        assert (layout.previous / "version.txt").read_text(encoding="ascii") == "old"


class TestWhenEvenTheRestoreFails:
    def test_the_previous_version_is_started_where_it_is(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """新しい版へ改名できず、元の名前へ戻すのも断られた（ウイルス対策・一時的な錠）

        戻しの結果を見ずに元の場所を起こすと、そこは空で何も出ない 前の版は
        ``.previous`` に在るので、そこから直に起こす
        """
        _folder(layout.staged, HEALTHY)
        held = (layout.staged / APP).open("rb")  # 新しい版のフォルダを改名させない
        blocker = layout.install
        try:
            launched = launch(_plan(layout), tmp_path / "update")
            assert wait_started(launched)
            # 今の版が previous へよけられたら、元の名前の所をふさいで戻しも断らせる
            assert _wait_for(layout.previous / APP, 30)
            blocker.write_text("ふさぐ", encoding="utf-8")
            launched.process.wait(timeout=120)
        finally:
            held.close()
        lines = take_result(tmp_path / "update")
        assert "restore-failed" in lines and "started-from-aside" in lines, lines
        assert "relaunch-failed" not in lines
        assert _wait_for(layout.install.parent / "old-started.txt")
        # 前の版も新しい版も消えていない
        assert (layout.previous / "version.txt").read_text(encoding="ascii") == "old"
        assert (layout.staged / APP).exists()


class TestRollingBack:
    def test_the_previous_version_comes_back(self, layout: Layout, tmp_path: Path) -> None:
        """メニューから戻す 戻した版は previous へ入れ替わり、もう一度戻せる"""
        _folder(layout.previous, OLD)
        (layout.previous / "version.txt").write_text("previous", encoding="ascii")
        lines = _run(_plan(layout, "rollback"), tmp_path / "update")
        assert "reverted" in lines, lines
        assert (layout.install / "version.txt").read_text(encoding="ascii") == "previous"
        assert (layout.previous / "version.txt").read_text(encoding="ascii") == "old"


class TestTheHandshake:
    def test_a_helper_that_never_starts_is_noticed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """台本の実行が止められた機械で、本体が消えたまま誰も起こし直さない、を防ぐ"""
        monkeypatch.setattr(swap_module, "HELPER_SCRIPT", "exit 1")
        layout = Layout(tmp_path / "Sashimono")
        launched = launch(SwapPlan("apply", layout, relaunch=False), tmp_path / "update")
        assert not wait_started(launched, timeout=30)

    def test_the_new_version_marks_its_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mark = tmp_path / "started.txt"
        monkeypatch.setenv(HEALTH_ENV, str(mark))
        mark_started()
        assert mark.read_text(encoding="utf-8") == "ok"
        # 子（pip など）へ引き継がない
        assert HEALTH_ENV not in os.environ

    def test_a_normal_start_marks_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(HEALTH_ENV, raising=False)
        mark_started()
        assert list(tmp_path.iterdir()) == []

    def test_the_script_is_plain_ascii_outside_comments(self) -> None:
        """台本は場所を環境変数で受け取る 場所を台本へ書き込むと、日本語の名前が化ける"""
        code = [
            line
            for line in swap_module.HELPER_SCRIPT.splitlines()
            if not line.strip().startswith("#")
        ]
        assert all(line.isascii() for line in code)
        assert '"' not in "".join(code)


def test_a_result_is_written_even_while_it_is_being_read(tmp_path: Path) -> None:
    """本体が結果のファイルを読んでいる間に書き足しが断られても、書き直して残す

    本体は走り始めたかを結果のファイルで待つ（何度も読む） 入れ替え係の書き足しが
    ちょうど重なると「ほかのプロセスが使用中」で断られ、入れ替え係が error で止まった
    （#241 の作業中に自己診断の試験で出た） 走り始めたのに走らないと取り違える
    台本の Write-Result だけを取り出し、書き足しが 1 度断られるまで結果のファイルを
    誰にも開かせずに持つ
    """
    import ctypes
    import re
    import subprocess
    from ctypes import wintypes

    match = re.search(r"^function Write-Result.*?^}\n", swap_module.HELPER_SCRIPT, re.M | re.S)
    assert match is not None
    result = tmp_path / "result.txt"
    result.write_bytes(b"")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    generic_read, no_sharing, open_existing = 0x80000000, 0, 3
    handle = kernel.CreateFileW(str(result), generic_read, no_sharing, None, open_existing, 0, None)
    assert handle not in (None, wintypes.HANDLE(-1).value)
    # 書き足しが断られた所（catch）で合図を出させ、それを見てから手放す 書き足す前に合図を
    # 出すと、PowerShell が遅れたときに手放した後で初めて書き足し、書き直さない作りでも通る
    # 書き直さない作り（catch が無い）は合図を出さずに、手放す前に落ちて終わる
    refused = tmp_path / "refused.txt"
    function = match.group(0).replace(
        "} catch {\n",
        "} catch {\nSet-Content -LiteralPath $env:REFUSED -Value 'refused'\n",
        1,
    )
    try:
        code = (
            "$ErrorActionPreference = 'Stop'\n"
            "$result = $env:RESULT\n" + function + "Write-Result 'holding'\n"
        )
        process = subprocess.Popen(
            [str(swap_module.powershell_path()), "-NoProfile", "-NonInteractive", "-Command", code],
            env={**os.environ, "RESULT": str(result), "REFUSED": str(refused)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 60
        while not refused.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        kernel.CloseHandle(handle)
    _, error = process.communicate(timeout=60)
    assert process.returncode == 0, error.decode("cp932", "replace")
    assert refused.exists(), "書き足しが 1 度も断られずに通った（確かめになっていない）"
    assert "holding" in result.read_text(encoding="utf-8-sig")


def test_a_first_powershell_start_is_waited_for() -> None:
    """その利用者が初めて PowerShell 5.1 を起こすときの遅さを、動かないと取り違えない

    CI のまっさらな Windows で、台本を 1 行走らせるだけで 11〜22 秒（変数を削ると 34 秒）
    かかった（Issue #33） 20 秒で見切っていたので、自己診断の入れ替えの項目が落ち、
    使う人の遅い機械では「PowerShell が動かない」と出て更新を入れられなくなる
    """
    assert swap_module.START_SECONDS >= 45
