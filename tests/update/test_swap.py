"""入れ替え係（PowerShell の台本）を本当に走らせる

本体の代わりに小さな .cmd を置く 新しい版の .cmd は、起動できた印を書くか、書かずに
終わる（起動に失敗した版の代わり） 前の版の .cmd は、起こされたことを隣に書き残す
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from sashimono.update import swap as swap_module
from sashimono.update.package import Layout
from sashimono.update.swap import (
    HEALTH_ENV,
    SwapPlan,
    launch,
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
