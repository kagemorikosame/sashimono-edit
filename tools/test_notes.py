r"""試験に「壊れたときに利用者に何が起きるか」の書き添えがあるかを確かめる

    .venv\Scripts\python.exe tools\test_notes.py            # 確かめるだけ
    .venv\Scripts\python.exe tools\test_notes.py --count    # ファイルごとの数を出す

決まり（docs/development.md の「テストは失敗の仕方を書く」）では、試験が落ちたときに
利用者に何が起きるかを書き添える 書き添えが無いと、落ちた試験を「直す」のか
「試験を消す」のかを、次に触る人が判断できない（#363）

書いてあるとみなす場所は次の 3 つ どれか 1 つあればよい

- 試験の関数の docstring
- ``def`` の直前（飾りがあればその上）に空行を挟まずに続くコメント
- 本体の頭 最初の文より前にあるコメント

試験の名前だけでは認めない 名前が利用者の困りごとを言っているかは機械では
見分けられず、認めると「通ること」を言い換えただけの名前も通ってしまう
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

#: 他人が作った配布物と、試験の材料として読むだけのファイルの置き場
SKIP_PREFIXES = ("tests/fixtures/",)

#: 見張る置き場 前からある試験の抜けは 1451 本・207 ファイルあり、1 本の PR に収まらない
#: （CodeRabbit が見るのは 1 PR 100 ファイルまで） 書き添えを埋めた置き場から順に広げ、
#: 全部埋まったら tests 全体にする 抜けた試験の控えを持つ形にしないのは、控えが
#: やり残しの一覧として残り続けるため ``--all`` でまだ見張っていない置き場も数えられる
WATCHED_PREFIXES = ("tests/ai/", "tests/ui/")


@dataclass(frozen=True, slots=True)
class Missing:
    """書き添えの無い試験 1 本"""

    path: Path
    line: int
    name: str

    def __str__(self) -> str:
        return f"{_relative(self.path)}:{self.line}: {self.name}"


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _comment_lines(text: str) -> set[int]:
    """コメントだけが書いてある行の番号

    行の後ろに付けたコメント（``x = 1  # ...``）は数えない 直前の行の後ろの
    コメントを試験の書き添えと取り違えないため
    """
    lines: set[int] = set()
    code_lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            lines.add(token.start[0])
        elif token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            code_lines.update(range(token.start[0], token.end[0] + 1))
    return lines - code_lines


def _is_noted(node: ast.FunctionDef | ast.AsyncFunctionDef, comments: set[int]) -> bool:
    if ast.get_docstring(node) is not None:
        return True
    top = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    if top - 1 in comments:
        return True
    # 本体の頭のコメント 引数を何行かに折った試験もあるので、def の行の次ではなく
    # 最初の文のすぐ上を見る
    return node.body[0].lineno - 1 in comments


def _tests(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """pytest が集める試験の関数 モジュールの直下と ``Test`` で始まるクラスの中"""
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, functions) and node.name.startswith("test"):
            found.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found.extend(
                item
                for item in node.body
                if isinstance(item, functions) and item.name.startswith("test")
            )
    return found


def scan(path: Path) -> list[Missing]:
    text = path.read_bytes().decode("utf-8")
    comments = _comment_lines(text)
    return [
        Missing(path, node.lineno, node.name)
        for node in _tests(ast.parse(text))
        if not _is_noted(node, comments)
    ]


def targets(root: Path = ROOT, *, everything: bool = False) -> list[Path]:
    """試験のファイル pytest の既定の集め方（``test_*.py``）に合わせる"""
    found: list[Path] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(SKIP_PREFIXES) or "__pycache__" in path.parts:
            continue
        if not everything and not relative.startswith(WATCHED_PREFIXES):
            continue
        found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--count", action="store_true", help="ファイルごとの数を多い順に出す")
    parser.add_argument("--all", action="store_true", help="まだ見張っていない置き場も数える")
    arguments = parser.parse_args(argv)

    missing = [item for path in targets(everything=arguments.all) for item in scan(path)]
    if arguments.count:
        for name, count in Counter(_relative(item.path) for item in missing).most_common():
            print(f"{count}\t{name}")
    else:
        for item in missing:
            print(item)
    if missing:
        print(
            f"壊れたときに何が起きるかの書き添えが無い試験が {len(missing)} 本ある"
            "  docstring か、def の直前か本体の頭にコメントで書く（docs/development.md）"
        )
        return 1
    print("どの試験にも書き添えがある")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
