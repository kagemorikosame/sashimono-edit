"""ruff・mypy・pytest をまとめて走らせる

    .venv\\Scripts\\python.exe tools\\verify.py

1 つでも落ちたら終了コードが非 0 になる シェルでパイプに繋ぐと終了コードが
最後のコマンドのものに化けるので、判定はここで行う
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = ["src", "tests", "tools"]

STEPS: list[tuple[str, list[str]]] = [
    ("ruff (書式)", ["-m", "ruff", "format", "--check", *TARGETS]),
    ("ruff (規約)", ["-m", "ruff", "check", *TARGETS]),
    ("mypy", ["-m", "mypy", "src", "tests"]),
    # -q は pyproject の addopts にもある 2 つ重ねると集計の行まで消えるので、ここでは
    # 足さない -rf は落ちたテストの名前を最後に並べる（CI の要約に写す）
    # --durations は遅いテストの上位を出す 手元と CI で飛ぶテストが違い、手元の内訳では
    # CI の時間を説明できなかった（#239） CI のログに残せば、遅くなったときに原因を追える
    ("pytest", ["-m", "pytest", "-rf", "--durations=30"]),
    # 文章に句点を使わない約束 データとしての句点は見ない（tools/punctuation.py）
    ("句点", ["tools/punctuation.py"]),
]


def own_environment(root: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """この木の ``src`` を先に読ませる環境変数

    ``.venv`` には本体の木が editable install で入っていて、``.pth`` が本体の
    ``src`` を指している worktree で検証しても、立てなければ pytest が読むのは
    本体の ``src`` で、worktree で書いたコードは 1 行も通らないのに「すべて通過」と
    出る（#102） ``PYTHONPATH`` は ``.pth`` より先に並ぶので、ここに立てれば勝つ
    すでに立っている値は後ろに残す（手で足した置き場を消さない）
    """
    environment = dict(os.environ if base is None else base)
    own = str(root / "src")
    rest = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = own + (os.pathsep + rest if rest else "")
    return environment


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    environment = own_environment(root)

    for name, arguments in STEPS:
        print(f"\n=== {name} ===", flush=True)
        if name == "pytest":
            returncode = _run_tests(root, arguments, environment)
        else:
            returncode = subprocess.run(
                [sys.executable, *arguments], cwd=root, check=False, env=environment
            ).returncode
        if returncode != 0:
            failures.append(name)

    print()
    if failures:
        print("失敗: " + "、".join(failures))
        return 1
    print("すべて通過")
    return 0


def _run_tests(root: Path, arguments: list[str], base: dict[str, str]) -> int:
    """pytest を回し、件数と落ちたテストの名前を CI の要約にも残す

    CI はこれとは別に件数を取るためだけにもう 1 度 pytest を回していた 時間が倍に
    なるうえ、たまにだけ落ちるテストを踏む回数も倍になる（PR #12） 1 回で済ませる
    """
    environment = {**base, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        # 要約は見やすくするためのおまけ 書けなくても検証の結果（終了コード）は
        # 変えない ここで例外になると、pytest の結果そのものが見えなくなる
        try:
            with Path(summary).open("a", encoding="utf-8") as handle:
                for line in summarize(completed.stdout, version):
                    handle.write(line + "\n")
        except OSError as exc:
            print(f"CI の要約に書けなかった: {exc}", file=sys.stderr)
    return completed.returncode


#: pytest の集計の行 通ったものが無くても（飛ばしただけ、集め損ねただけ、1 本も
#: 無い）集計の行は出る passed と failed だけを探すと、それを「途中で止まった」と
#: 取り違える
_COUNTS = re.compile(
    r"\b\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b"
    r"|no tests ran"
)


def summarize(output: str, version: str) -> list[str]:
    """pytest の出力から、要約に書く行を作る 集計の行と、落ちたテストの名前"""
    lines = [line for line in output.splitlines() if line.strip()]
    counts = next(
        (line for line in reversed(lines) if _COUNTS.search(line)),
        "集計の行が見つからない（pytest が途中で止まった）",
    )
    failed = [f"- {line}" for line in lines if line.startswith(("FAILED", "ERROR"))]
    return [f"### Python {version}: {counts.strip('= ')}", *failed]


if __name__ == "__main__":
    raise SystemExit(main())
