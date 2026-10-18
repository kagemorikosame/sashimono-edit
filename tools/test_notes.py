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
from collections.abc import Iterator
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

#: 中の文も同じモジュールの直下として名前を結ぶ文 ``if TYPE_CHECKING:`` や ``try: import``
#: の中で定義した基底も、実行すればモジュールの名前になる
_BLOCKS = (ast.If, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """モジュールやクラスの直下で名前を結ぶ文 関数とクラスの中には入らない"""
    for node in body:
        yield node
        if isinstance(node, _BLOCKS):
            for field in ("body", "orelse", "finalbody"):
                yield from _statements(getattr(node, field, []))
            for handler in getattr(node, "handlers", []):
                yield from _statements(handler.body)


@dataclass(slots=True, eq=False)
class _Module:
    """読んだファイル 1 つ 基底クラスを辿るたびに読み直さないよう持っておく"""

    path: Path
    tree: ast.Module
    comments: set[int]

    @property
    def package(self) -> bool:
        return self.path.name == "__init__.py"


@dataclass(frozen=True, slots=True)
class _Class:
    module: _Module
    node: ast.ClassDef
    #: 入れ子のクラスなら、それを書いたクラス 基底の式はそこの名前で解く
    outer: ast.ClassDef | None = None


@dataclass(frozen=True, slots=True)
class _Func:
    module: _Module
    node: _Function


@dataclass(frozen=True, slots=True)
class _External:
    """リポジトリの外の物（``unittest`` など） 点でつないだ名前だけを持つ"""

    name: str


type _Symbol = _Module | _Class | _Func | _External

#: 辿っている途中のファイルと名前の組 互いを読み込む輪で止まらなくならないため
type _Visiting = frozenset[tuple[Path, str]]


def _test_flag(body: list[ast.stmt], owner: str | None = None) -> bool | None:
    """本体に書いた ``__test__`` の値 ``owner`` があれば ``owner.__test__ = ...`` の方

    値が定数でなければ（関数の呼び出しなど）分からないので None 分からない物は
    集める側に倒す 外す側に倒すと、書き添えの無い試験を見逃す
    """
    for node in reversed(list(_statements(body))):
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if owner is None:
                hit = isinstance(target, ast.Name) and target.id == "__test__"
            else:
                hit = (
                    isinstance(target, ast.Attribute)
                    and target.attr == "__test__"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == owner
                )
            if hit:
                value = node.value
                return bool(value.value) if isinstance(value, ast.Constant) else None
    return None


def _exports(module: _Module) -> frozenset[str] | None:
    """``__all__`` に文字の並びで直に書いた名前 書いていないか、式で作っていれば None"""
    for node in reversed(list(_statements(module.tree.body))):
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.List | ast.Tuple) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        ):
            return frozenset(
                str(item.value) for item in value.elts if isinstance(item, ast.Constant)
            )
        return None
    return None


def _star_exports(module: _Module, name: str) -> bool:
    """``from module import *`` がその名前を持ち込むか

    ``__all__`` を文字の並びで直に書いてあれば、それで絞る（下線で始まる名前も載って
    いれば持ち込む） 無いか式で作っていれば、下線で始まらない名前を全部持ち込む
    式の ``__all__`` を読み解かないのは、外れたときに多めに数える側へ倒すため
    """
    exported = _exports(module)
    if exported is None:
        return not name.startswith("_")
    return name in exported


class Reader:
    """試験のファイルを読み、pytest が集める試験の関数を、書いてある所まで辿る

    pytest は試験のファイルの名前の表（モジュールの ``__dict__``）を見て集める そのため
    次の物もそのファイルの試験になる 具象クラスの本体や、そのファイルに書いた ``def`` だけを
    見ると、ほかの所に書き添えの無い試験を置けば見張りをすり抜ける（#377 の Codex と
    CodeRabbit が別々の道を 3 度指摘した 1 つずつ塞がず、名前の解き方をまとめて作った）

    - ``Test`` で始まるクラスが受け継いだ ``test_*`` 基底は名前が ``Test`` で始まらなくても
      よく、同じファイル・``from`` や ``import`` で読み込んだ別のファイル・パッケージの
      ``__init__.py`` で公開し直した物（``import *`` も）・``shared.Checks`` のような属性の形・
      ``Base = Checks`` のような付け替えのどれでもよい
    - 別のファイルから読み込んだ ``test_*`` の関数と ``Test*`` のクラス（``import *`` も）
    - 名前に関係なく ``unittest.TestCase`` を受け継ぐクラス
    - クラスの中で ``test_b = helper`` と付け替えた関数

    pytest の収集を基準にすると、試験のファイルが読み込む Qt やエンジンまで起こすので
    数十秒かかり、部品が無い機械では集める前に落ちる AST で辿れば tests の全部を見ても
    数秒で、tests の木だけで答えが決まる リポジトリの外の物（``unittest.TestCase`` など）は
    書き添えを求める先が無いので、``unittest`` の型かどうかを見分けるのにだけ使う
    """

    def __init__(self, root: Path = ROOT) -> None:
        #: 辿ってよい置き場 外の物は書き添えを求める先が無い
        self._root = root.resolve()
        self._modules: dict[Path, _Module] = {}
        self._found: dict[tuple[Path, str], _Symbol | None] = {}

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
        # pytest はモジュールに __test__ = False があると、中を 1 本も集めない
        if _test_flag(module.tree.body) is False:
            return found
        for name in self._names(module, module.tree.body, frozenset()):
            symbol = self.lookup(module, name)
            collected = self._collected(name, symbol, module.tree.body)
            found.extend((owner, node) for _, owner, node in collected)
        return found

    def lookup(
        self, module: _Module, name: str, visiting: _Visiting = frozenset()
    ) -> _Symbol | None:
        """モジュールの直下で名前が指す物

        ``visiting`` はファイル同士が互いを読み込む輪で止まらなくならないため 輪の途中で
        出た答え（None）は輪の外から引くと違うことがあるので、覚えるのは外からの答えだけ
        """
        key = (module.path, name)
        if key in self._found:
            return self._found[key]
        if key in visiting:
            return None
        found = self._find_in(module, module.tree.body, None, name, visiting | {key})
        if not visiting:
            self._found[key] = found
        return found

    def _local(
        self, module: _Module, owner: ast.ClassDef, name: str, visiting: _Visiting
    ) -> _Symbol | None:
        """クラスの本体で名前が指す物 本体に無ければ None（モジュールの直下は見ない）

        ``test_a = test_a`` のように自分の名前を右辺に書くと、Python はまだ結んでいない
        クラスの名前を飛ばしてモジュールの直下を見る 同じ組を 2 度引いたら None を返して
        呼び手にモジュールの直下を見させるのは、その順に合わせるため
        """
        key = (module.path, f"{owner.lineno}:{owner.col_offset}:{name}")
        if key in visiting:
            return None
        return self._find_in(module, owner.body, owner, name, visiting | {key})

    def _find_in(
        self,
        module: _Module,
        body: list[ast.stmt],
        owner: ast.ClassDef | None,
        name: str,
        visiting: _Visiting,
    ) -> _Symbol | None:
        """モジュールの直下（owner が None）かクラスの本体で、名前が指す物

        どちらも同じ手順で解く 名前を結ぶ文（def・class・付け替え・import・from）を、
        if・try・with などの中まで後ろから見る（後から結んだ名前が勝つ） 付け替えの右辺は
        結んだ場所の名前の決まりで解く（クラスの本体なら、まずそのクラスの名前） 両方で
        別々に書くと、片方にだけ足した形がもう片方で抜ける（#377 で 4 度指摘された）
        ``import *`` だけはクラスの中に書けないので、モジュールの直下でだけ辿る
        """
        stars: list[ast.ImportFrom] = []
        for node in reversed(list(_statements(body))):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return _Class(module, node, owner)
            if isinstance(node, _FUNCTIONS) and node.name == name:
                return _Func(module, node)
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                named = any(isinstance(t, ast.Name) and t.id == name for t in targets)
                if named and node.value is not None:
                    return self._expr(module, node.value, visiting, owner)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname == name:
                        return self._module_at(alias.name, 0, module)
                    if alias.asname is None and alias.name.split(".")[0] == name:
                        return self._module_at(name, 0, module)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        stars.append(node)
                    elif (alias.asname or alias.name) == name:
                        return self._from(module, node, alias.name, visiting)
        if owner is not None:
            return None
        for node in stars:
            source = self._module_at(node.module or "", node.level, module)
            if isinstance(source, _Module) and _star_exports(source, name):
                found = self.lookup(source, name, visiting)
                if found is not None:
                    return found
        return None

    def _from(
        self,
        module: _Module,
        node: ast.ImportFrom,
        name: str,
        visiting: _Visiting,
    ) -> _Symbol | None:
        """``from X import name`` の name X の中の名前か、X が包みならその下のモジュール"""
        source = self._module_at(node.module or "", node.level, module)
        if isinstance(source, _External):
            return _External(f"{source.name}.{name}")
        if source is None:
            return None
        return self.lookup(source, name, visiting) or self._submodule(source, name)

    def _expr(
        self,
        module: _Module,
        expr: ast.expr,
        visiting: _Visiting,
        owner: ast.ClassDef | None = None,
    ) -> _Symbol | None:
        """基底や付け替えの右辺に書いた式が指す物 名前と属性（``a.b.C``）だけを辿る

        ``owner`` はその式を書いたクラスの本体 名前はまずそのクラスで、無ければ
        モジュールの直下で引く（Python の名前の決まりと同じ）
        """
        if isinstance(expr, ast.Name):
            if owner is not None:
                found = self._local(module, owner, expr.id, visiting)
                if found is not None:
                    return found
            return self.lookup(module, expr.id, visiting)
        if not isinstance(expr, ast.Attribute):
            return None
        base = self._expr(module, expr.value, visiting, owner)
        if isinstance(base, _External):
            return _External(f"{base.name}.{expr.attr}")
        if isinstance(base, _Module):
            return self.lookup(base, expr.attr, visiting) or self._submodule(base, expr.attr)
        if isinstance(base, _Class):
            return self._attribute(base, expr.attr, visiting, frozenset())
        return None

    def _attribute(
        self, cls: _Class, name: str, visiting: _Visiting, seen: frozenset[int]
    ) -> _Symbol | None:
        """クラスの属性 本体に無ければ基底を前から辿る（``Helpers.check`` など）"""
        if id(cls.node) in seen:
            return None
        found = self._local(cls.module, cls.node, name, visiting)
        if found is not None:
            return found
        for base in cls.node.bases:
            located = self._expr(cls.module, base, visiting, cls.outer)
            if isinstance(located, _Class):
                found = self._attribute(located, name, visiting, seen | {id(cls.node)})
                if found is not None:
                    return found
        return None

    def _module_at(self, dotted: str, level: int, origin: _Module) -> _Module | _External | None:
        """読み込みが指すモジュール リポジトリの中ならファイル、外なら名前だけ"""
        if level:
            base = origin.path.parent
            for _ in range(level - 1):
                base = base.parent
        else:
            base = self._root
        parts = [part for part in dotted.split(".") if part]
        candidate = base.joinpath(*parts)
        files = [candidate / "__init__.py"]
        if parts:
            files.insert(0, candidate.with_name(parts[-1] + ".py"))
        for found in files:
            if found.is_file() and found.resolve().is_relative_to(self._root):
                return self.module(found)
        # 相対の読み込みが外れるのは書き間違いで、外の物でもない
        return None if level else _External(dotted)

    def _submodule(self, package: _Module, name: str) -> _Module | None:
        """包みの下のモジュール ``from tests import shared`` の shared"""
        if not package.package:
            return None
        base = package.path.parent / name
        for found in (base.with_name(name + ".py"), base / "__init__.py"):
            if found.is_file() and found.resolve().is_relative_to(self._root):
                return self.module(found)
        return None

    def _names(self, module: _Module, body: list[ast.stmt], visiting: frozenset[Path]) -> list[str]:
        """モジュールの直下かクラスの本体で結ばれる名前 ``_find_in`` と同じ文を数える

        ``import *`` は読み込み先の名前を足す（クラスの中には書けないので、出るのは
        モジュールの直下だけ）
        """
        names: list[str] = []
        for node in _statements(body):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
            elif isinstance(node, ast.Import):
                names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        names.append(alias.asname or alias.name)
                        continue
                    source = self._module_at(node.module or "", node.level, module)
                    if isinstance(source, _Module) and source.path not in visiting:
                        inner = self._names(source, source.tree.body, visiting | {module.path})
                        names.extend(name for name in inner if _star_exports(source, name))
        return list(dict.fromkeys(names))

    def _collected(
        self, name: str, symbol: _Symbol | None, scope: list[ast.stmt]
    ) -> list[tuple[str, _Module, _Function]]:
        """名前の表の 1 行から pytest が集める試験 モジュールの直下もクラスの本体も同じ決まり

        ``test`` で始まる名前の関数と、``Test`` で始まる名前か ``unittest.TestCase`` を
        受け継ぐクラスの試験 鍵は上書きを見分けるための名前（入れ子のクラスは点でつなぐ）
        ``scope`` はその名前を結んだ本体 ``test_x.__test__ = False`` を探すのに使う
        ``__test__`` が偽の物は pytest が集めないので、書き添えも求めない 求めると、
        試験の部品に ``__test__ = False`` を付けて外した人の verify が落ちる
        """
        if isinstance(symbol, _Func) and name.startswith("test"):
            off = (
                _test_flag(scope, name) is False
                or _test_flag(symbol.module.tree.body, symbol.node.name) is False
            )
            return [] if off else [(name, symbol.module, symbol.node)]
        if isinstance(symbol, _Class) and (
            name.startswith("Test") or self._is_unittest(symbol, frozenset())
        ):
            if _test_flag(scope, name) is False or self._switched_off(symbol, frozenset()):
                return []
            return [
                (f"{name}.{key}", owner, node) for key, owner, node in self._class_tests(symbol)
            ]
        return []

    def _class_tests(
        self, cls: _Class, seen: frozenset[int] = frozenset()
    ) -> list[tuple[str, _Module, _Function]]:
        """クラスが持つ試験 受け継いだ物も含める 本体で結んだ名前は基底の物より勝つ

        本体の名前はモジュールの直下と同じ ``_names`` と ``_find_in`` で読む
        基底は前から辿り、先の基底が持つ名前は後の基底の物より勝つ ``seen`` は基底を
        辿る輪で止まらなくならないため
        """
        if id(cls.node) in seen:
            return []
        seen = seen | {id(cls.node)}
        found: list[tuple[str, _Module, _Function]] = []
        local = self._names(cls.module, cls.node.body, frozenset())
        for name in local:
            symbol = self._local(cls.module, cls.node, name, frozenset())
            found.extend(self._collected(name, symbol, cls.node.body))
        taken = set(local)
        for base in cls.node.bases:
            located = self._expr(cls.module, base, frozenset(), cls.outer)
            if not isinstance(located, _Class):
                continue
            inherited = [
                item
                for item in self._class_tests(located, seen)
                if item[0].split(".")[0] not in taken
            ]
            found.extend(inherited)
            taken.update(key.split(".")[0] for key, _, _ in inherited)
        return found

    def _switched_off(self, cls: _Class, seen: frozenset[int]) -> bool:
        """クラスの ``__test__`` が偽か 本体に無ければ基底から受け継いだ値を見る"""
        return self._class_flag(cls, seen) is False

    def _class_flag(self, cls: _Class, seen: frozenset[int]) -> bool | None:
        """``__test__`` の値 本体に無ければ基底を前から辿り、最初に書いてある物"""
        if id(cls.node) in seen:
            return None
        flag = _test_flag(cls.node.body)
        if flag is not None:
            return flag
        for base in cls.node.bases:
            located = self._expr(cls.module, base, frozenset(), cls.outer)
            if isinstance(located, _Class):
                inherited = self._class_flag(located, seen | {id(cls.node)})
                if inherited is not None:
                    return inherited
        return None

    def _is_unittest(self, cls: _Class, seen: frozenset[int]) -> bool:
        """``unittest.TestCase`` を受け継ぐか pytest は名前に関係なく集める"""
        if id(cls.node) in seen:
            return False
        seen = seen | {id(cls.node)}
        for base in cls.node.bases:
            located = self._expr(cls.module, base, frozenset(), cls.outer)
            if (
                isinstance(located, _External)
                and located.name.startswith("unittest.")
                and located.name.endswith("TestCase")
            ):
                return True
            if isinstance(located, _Class) and self._is_unittest(located, seen):
                return True
        return False


def scan(path: Path, reader: Reader | None = None) -> list[Missing]:
    """書き添えの無い試験 受け継いだり読み込んだりした試験は、書いてあるファイルと行で出す"""
    reader = reader or Reader()
    missing: list[Missing] = []
    for module, node in reader.tests(path):
        if not _is_noted(node, module.comments):
            item = Missing(module.path, node.lineno, node.name)
            if item not in missing:
                missing.append(item)
    return missing


def targets(root: Path = ROOT, *, everything: bool = False) -> list[Path]:
    """試験のファイル pytest の既定の集め方（``test_*.py`` と ``*_test.py``）に合わせる"""
    found: set[Path] = set()
    for pattern in ("test_*.py", "*_test.py"):
        for path in (root / "tests").rglob(pattern):
            relative = path.relative_to(root).as_posix()
            if relative.startswith(SKIP_PREFIXES) or "__pycache__" in path.parts:
                continue
            if not everything and not relative.startswith(WATCHED_PREFIXES):
                continue
            found.add(path)
    return sorted(found)


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
