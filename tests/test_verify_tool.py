"""検証の道具（tools/verify.py）が CI の要約に書くもの

CI は以前、件数を取るためだけに pytest をもう 1 度回していた いまは verify.py が
1 回の実行から要約を作る ここが壊れると、CI で落ちたときにどのテストか分からない
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def verify() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify", ROOT / "tools" / "verify.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_counts_and_the_failed_names_are_kept(verify: ModuleType) -> None:
    output = "\n".join(
        [
            "....F.",
            "FAILED tests/test_x.py::test_y - AssertionError",
            "1 failed, 5 passed, 2 skipped in 3.10s",
        ]
    )
    assert verify.summarize(output, "3.12") == [
        "### Python 3.12: 1 failed, 5 passed, 2 skipped in 3.10s",
        "- FAILED tests/test_x.py::test_y - AssertionError",
    ]


@pytest.mark.parametrize(
    "counts",
    [
        "3 skipped in 0.10s",
        "1 error in 0.50s",
        "no tests ran in 0.01s",
        "2 deselected in 0.02s",
        "1 xfailed, 1 xpassed in 0.30s",
    ],
)
def test_a_finished_run_without_passes_is_not_reported_as_stopped(
    verify: ModuleType, counts: str
) -> None:
    # 通ったものが無い回を「途中で止まった」と書くと、集め損ねた（1 error）のか
    # 本当に止まったのかを CI の要約から見分けられない
    (line,) = verify.summarize(f"=== {counts} ===", "3.12")
    assert line == f"### Python 3.12: {counts}"


def test_a_run_that_stopped_early_says_so(verify: ModuleType) -> None:
    # 集計の行が無いのに「通った」と読める要約を書くと、止まった CI を見落とす
    (line,) = verify.summarize("ImportError while loading conftest", "3.14")
    assert "見つからない" in line


#: pytest-xdist で並べて走らせたときの出力（-rfE --max-worker-restart=0 で、落ちる・
#: 準備で落ちる・プロセスごと落ちる試験を混ぜて手元で取った物を縮めた） 進み具合の
#: 前にワーカーを上げる行が、落ちた試験の前にはワーカーの名前の行が入る
XDIST_OUTPUT = """\
bringing up nodes...

.sFE[gw1] node down: Not properly terminated
F                                                                        [100%]
=================================== ERRORS ====================================
________________________ ERROR at setup of test_error _________________________
[gw0] win32 -- Python 3.14.6 J:\\venv\\Scripts\\python.exe
E       RuntimeError: boom
================================== FAILURES ===================================
__________________________________ test_fail __________________________________
[gw1] win32 -- Python 3.14.6 J:\\venv\\Scripts\\python.exe
E       assert 1 == 2
__________________________________ test_a.py __________________________________
[gw1] win32 -- Python 3.14.6 J:\\venv\\Scripts\\python.exe
worker 'gw1' crashed while running 'test_a.py::test_crash'
============================= slowest 30 durations =============================
3.54s call     tests/test_a.py::test_fail
0.61s setup    tests/test_a.py::test_error
=========================== short test summary info ===========================
FAILED test_a.py::test_fail - assert 1 == 2
FAILED test_a.py::test_crash - worker 'gw1' crashed while running 'test_a.py:...
ERROR test_a.py::test_error - RuntimeError: boom
2 failed, 1 passed, 1 skipped, 1 error in 0.99s
"""


class TestInParallel:
    """CI だけ pytest-xdist で並べて走らせる（#239）"""

    def test_the_counts_and_every_failed_name_are_read_from_xdist(self, verify: ModuleType) -> None:
        """並列の出力から集計の行と、落ちた・準備で落ちた・ワーカーごと落ちた名前を取る

        取り損ねると、CI の要約が「途中で止まった」や名前の無い失敗になり、どのテストが
        落ちたのかをログの中から探すことになる
        """
        assert verify.summarize(XDIST_OUTPUT, "3.14") == [
            "### Python 3.14: 2 failed, 1 passed, 1 skipped, 1 error in 0.99s",
            "- FAILED test_a.py::test_fail - assert 1 == 2",
            "- FAILED test_a.py::test_crash - worker 'gw1' crashed while running 'test_a.py:...",
            "- ERROR test_a.py::test_error - RuntimeError: boom",
        ]

    def test_the_slowest_list_is_not_taken_for_the_counts(self, verify: ModuleType) -> None:
        """--durations の行（秒と名前）を集計の行と取り違えない"""
        output = "\n".join(
            [
                "3 passed in 0.10s",
                "==== slowest 30 durations ====",
                "1.00s call     tests/test_x.py::test_passed_twice",
            ]
        )
        (line,) = verify.summarize(output, "3.12")
        assert line == "### Python 3.12: 3 passed in 0.10s"

    @pytest.mark.parametrize("workers", [None, "", "0", " 0 "])
    def test_without_the_variable_it_runs_in_one_process(
        self, verify: ModuleType, workers: str | None
    ) -> None:
        """手元の既定は直列 0 を入れれば CI でも直列へ戻せる（並列を疑うときの逃げ道）"""
        environment = {} if workers is None else {verify.WORKERS_VARIABLE: workers}
        assert verify.pytest_arguments(["-m", "pytest"], environment) == ["-m", "pytest"]

    def test_the_variable_spreads_the_tests_over_workers(self, verify: ModuleType) -> None:
        """xdist_group を効かせ、落ちたワーカーを立て直さない指定まで付ける

        ``loadgroup`` を付け忘れると同じワーカーにまとめたはずの試験がばらけ、
        立て直しを許すと Windows で落ちたあと pytest が戻らなかった
        """
        arguments = verify.pytest_arguments(["-m", "pytest"], {verify.WORKERS_VARIABLE: "4"})
        assert arguments == [
            "-m",
            "pytest",
            "-n",
            "4",
            "--dist",
            "loadgroup",
            "--max-worker-restart=0",
        ]

    def test_errors_are_listed_at_the_end(self, verify: ModuleType) -> None:
        """-rf だけでは ERROR の行が出ず、準備で落ちた試験の名前が要約に載らない"""
        (pytest_step,) = [arguments for name, arguments in verify.STEPS if name == "pytest"]
        report = next(argument for argument in pytest_step if argument.startswith("-r"))
        assert {"f", "E"} <= set(report[2:])

    def test_the_tests_asked_for_are_handed_to_pytest(
        self, verify: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """環境変数を読む所から pytest を起こす所まで、並列の指定が届くこと"""
        seen: list[list[str]] = []

        def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen.append(command)
            return subprocess.CompletedProcess(command, 0, "1 passed in 0.01s\n", "")

        monkeypatch.setattr(verify.subprocess, "run", run)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        assert verify._run_tests(ROOT, ["-m", "pytest"], {verify.WORKERS_VARIABLE: "3"}) == 0
        assert seen[0][1:5] == ["-m", "pytest", "-n", "3"]


class TestReadingThisTree:
    """worktree で検証しても、その木のコードを試すこと（#102）"""

    def test_this_tree_comes_first(self, verify: ModuleType, tmp_path: Path) -> None:
        """立てないと .pth が指す本体の src が読まれ、worktree の変更を試さない"""
        environment = verify.own_environment(tmp_path, {"PATH": "x"})
        assert environment["PYTHONPATH"] == str(tmp_path / "src")

    def test_a_path_already_set_is_kept_behind(self, verify: ModuleType, tmp_path: Path) -> None:
        """手で立てた置き場を消すと、それを頼りにしていた試験が読めなくなる"""
        environment = verify.own_environment(tmp_path, {"PYTHONPATH": "手で足した"})
        assert environment["PYTHONPATH"].split(os.pathsep) == [str(tmp_path / "src"), "手で足した"]

    def test_a_child_python_imports_this_tree(self, verify: ModuleType, tmp_path: Path) -> None:
        """実際に子の Python を起こして、入っている本体ではなくこの木を読むこと

        環境変数を組み立てるだけの試験では、``.pth`` より先に並ぶかまでは分からない
        """
        package = tmp_path / "src" / "sashimono"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("WHERE = 'この木'\n", encoding="utf-8")
        found = subprocess.run(
            [sys.executable, "-c", "import sashimono; print(sashimono.WHERE)"],
            env={**verify.own_environment(tmp_path), "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        assert found.stdout.strip() == "この木"

    def test_every_step_is_run_with_this_tree_first(
        self, verify: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """組み立てた環境を段へ渡し忘れると、pytest が本体の src を読んだまま通る

        環境を作る所だけを見る試験では、``main`` から ``_run_tests`` を通って
        pytest を起こす道の渡し忘れを見つけられない 全部の段を差し替えて、
        受け取った環境を見る
        """
        seen: list[tuple[str, str]] = []

        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            environment = kwargs.get("env") or {}
            seen.append((" ".join(command[1:3]), environment.get("PYTHONPATH", "")))
            return subprocess.CompletedProcess(command, 0, "1 passed in 0.01s\n", "")

        monkeypatch.setattr(verify.subprocess, "run", run)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        assert verify.main() == 0
        own = str(ROOT / "src")
        assert {step for step, _ in seen} >= {"-m pytest", "-m mypy"}
        for step, path in seen:
            assert path.split(os.pathsep)[0] == own, f"{step} に自分の木の src が渡っていない"
