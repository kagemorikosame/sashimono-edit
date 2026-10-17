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


type _Function = ast.FunctionDef | ast.AsyncFunctionDef

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(slots=True)
class _Module:
    """読んだ試験のファイル 1 つ 基底クラスを辿るたびに読み直さないよう持っておく"""

    path: Path
    tree: ast.Module
    comments: set[int]


class Reader:
    """試験のファイルを読み、``Test`` で始まるクラスが受け継ぐ試験まで辿る

    pytest は ``Test`` で始まるクラスが基底クラス（名前が ``Test`` で始まらない mixin や、
    別のモジュールの物も）から受け継いだ ``test_*`` も集める 具象クラスの本体だけを見ると、
    基底クラスに書き添えの無い試験を置けば見張りをすり抜ける（#377 の Codex の指摘）
    pytest の収集を基準にすると、試験のファイルが読み込む Qt やエンジンまで起こすので
    数十秒かかり、部品が無い機械では集める前に落ちる AST で辿れば 1 秒かからず、
    同じ所（tests の木）だけで答えが決まる 辿れない基底（``unittest.TestCase``
    やリポジトリの外の物）は書き添えを求める先が無いので見ない
    """

    def __init__(self, root: Path = ROOT) -> None:
        #: 辿ってよい置き場 外の基底は書き添えを求める先が無い
        self._root = root.resolve()
        self._modules: dict[Path, _Module] = {}

    def module(self, path: Path) -> _Module:
        resolved = path.resolve()
        if resolved not in self._modules:
            text = resolved.read_bytes().decode("utf-8")
            self._modules[resolved] = _Module(resolved, ast.parse(text), _comment_lines(text))
        return self._modules[resolved]

    def tests(self, path: Path) -> list[tuple[_Module, _Function]]:
        """pytest が集める試験の関数と、それが書いてあるファイル"""
        module = self.module(path)
        found: list[tuple[_Module, _Function]] = []
        for node in module.tree.body:
            if isinstance(node, _FUNCTIONS) and node.name.startswith("test"):
                found.append((module, node))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                found.extend(self._class_tests(module, node))
        return found

    def _class_tests(
        self, module: _Module, node: ast.ClassDef, seen: frozenset[int] = frozenset()
    ) -> list[tuple[_Module, _Function]]:
        """クラスが持つ試験 受け継いだ物も含める 上書きした名前は子の物だけを数える

        ``Test`` で始まる入れ子のクラスも pytest は集めるので辿る ``seen`` は基底を
        辿る輪（同じ名前を付け替えた import など）で止まらなくならないため
        """
        if id(node) in seen:
            return []
        seen = seen | {id(node)}
        found: list[tuple[_Module, _Function]] = []
        names: set[str] = set()
        for item in node.body:
            if isinstance(item, _FUNCTIONS) and item.name.startswith("test"):
                found.append((module, item))
                names.add(item.name)
            elif isinstance(item, ast.ClassDef) and item.name.startswith("Test"):
                found.extend(self._class_tests(module, item, seen))
        for base in node.bases:
            located = self._base(module, base)
            if located is None:
                continue
            for owner, function in self._class_tests(*located, seen):
                if function.name not in names:
                    found.append((owner, function))
                    names.add(function.name)
        return found

    def _base(self, module: _Module, base: ast.expr) -> tuple[_Module, ast.ClassDef] | None:
        """基底クラスの定義 同じファイルの直下のクラスか、from で読み込んだクラス"""
        if not isinstance(base, ast.Name):
            return None
        for node in module.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == base.id:
                return module, node
        for node in module.tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if (alias.asname or alias.name) != base.id:
                    continue
                source = self._source(module.path, node)
                if source is None:
                    return None
                imported = self.module(source)
                for item in imported.tree.body:
                    if isinstance(item, ast.ClassDef) and item.name == alias.name:
                        return imported, item
                return None
        return None

    def _source(self, path: Path, node: ast.ImportFrom) -> Path | None:
        """``from ... import`` が指すリポジトリの中のファイル 外の物は None"""
        parts = (node.module or "").split(".") if node.module else []
        if node.level:
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
        else:
            base = self._root
        candidate = base.joinpath(*parts)
        for found in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if found.is_file() and found.resolve().is_relative_to(self._root):
                return found
        return None


def scan(path: Path, reader: Reader | None = None) -> list[Missing]:
    """書き添えの無い試験 受け継いだ試験は、書いてあるファイルと行で出す"""
    reader = reader or Reader()
    missing: list[Missing] = []
    for module, node in reader.tests(path):
        if not _is_noted(node, module.comments):
            item = Missing(module.path, node.lineno, node.name)
            if item not in missing:
                missing.append(item)
    return missing


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

    # 同じ基底を何本ものクラスが受け継ぐと同じ試験が何度も出るので、1 本にまとめる
    reader = Reader()
    missing = list(
        dict.fromkeys(
            item for path in targets(everything=arguments.all) for item in scan(path, reader)
        )
    )
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
