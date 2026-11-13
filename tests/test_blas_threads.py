"""numpy の行列計算のスレッドの上限（#454）

環境変数は numpy を読む前に決まっている必要があるので、まっさらな Python を起こして確かめる
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import sashimono

SRC = Path(__file__).resolve().parent.parent / "src"

PROBE = "import os, sashimono, numpy\nprint(os.environ.get('OPENBLAS_NUM_THREADS'))\n"


def _run(environ: dict[str, str]) -> str:
    env = {**environ, "PYTHONPATH": str(SRC)}
    # 固定の文字列を、今の Python へ list で渡すだけ（shell は通さない）
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    done = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_the_app_limits_blas_threads_before_numpy_loads() -> None:
    """壊れると、numpy を読むだけで論理コアの数 × 30 MB をコミットし、子のプロセスも同じだけ取る"""
    environ = {key: value for key, value in os.environ.items() if key != "OPENBLAS_NUM_THREADS"}
    assert _run(environ) == str(sashimono.BLAS_THREADS)


def test_a_value_set_by_the_person_wins() -> None:
    """壊れると、行列計算を速くしたくて環境変数を決めた人の値が黙って 4 に戻る"""
    assert _run({**os.environ, "OPENBLAS_NUM_THREADS": "12"}) == "12"
