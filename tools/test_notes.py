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
import builtins
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
WATCHED_PREFIXES = ("tests/ai/", "tests/ui/", "tests/core/", "tests/engine/")


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


def _statements(body: list[ast.stmt], nested: bool = False) -> Iterator[tuple[ast.stmt, bool]]:
    """モジュールやクラスの直下で名前を結ぶ文と、それがブロックの中か 関数とクラスの中には入らない

    ブロック（``if``・``try``・``with``・``for``・``while``）の中の文は、走るかどうかが
    読むだけでは決まらない そこで結んだ名前は「結ぶかもしれない」として扱う
    """
    for node in body:
        yield node, nested
        if isinstance(node, _BLOCKS):
            for field in ("body", "orelse", "finalbody"):
                yield from _statements(getattr(node, field, []), True)
            for handler in getattr(node, "handlers", []):
                yield from _statements(handler.body, True)


def _mentions(body: list[ast.stmt], name: str) -> list[ast.AST]:
    """本体の中で ``name`` が名前・属性・文字として出てくる所 関数の中まで見る

    ``global`` で書き換える関数や ``setattr(x, "__test__", ...)`` も拾うため 拾いすぎた
    ときは読み解かずに多めに数える側へ倒すだけなので、広く拾う

    試験 1 本ごとにファイル全体を歩くと、tests の全部で数十秒かかった 同じ本体と名前は
    1 度だけ歩く 覚えるときに本体そのものも持つのは、本体が消えて同じ番号（``id``）が
    別の本体に使い回されても、取り違えないため
    """
    key = (id(body), name)
    cached = _MENTIONS.get(key)
    if cached is not None and cached[0] is body:
        return cached[1]
    found: list[ast.AST] = [
        node
        for statement in body
        for node in ast.walk(statement)
        if (isinstance(node, ast.Name) and node.id == name)
        or (isinstance(node, ast.Attribute) and node.attr == name)
        or (isinstance(node, ast.Constant) and node.value == name)
    ]
    _MENTIONS[key] = (body, found)
    return found


_MENTIONS: dict[tuple[int, str], tuple[list[ast.stmt], list[ast.AST]]] = {}


def _sole_assignment(body: list[ast.stmt], name: str) -> ast.expr | None:
    """本体で ``name`` が出てくるのが、直下の 1 つの代入の的だけなら、その右辺

    ブロックの中・2 度目の代入・``+=``・呼び出しへ渡すなど、ほかに 1 つでも出てくれば
    None 読み解く側に倒す（値で絞って数を減らす）のは、書かれ方が 1 通りに決まる
    ときだけにする 形ごとに読み解くと、読まない形で書き添えの無い試験を見逃す
    （#377 で CodeRabbit が ``__all__`` の書き換えと ``if TYPE_CHECKING`` の中を指摘した）
    """
    mentions = _mentions(body, name)
    if len(mentions) != 1:
        return None
    for node in body:
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(target is mentions[0] for target in targets):
            return node.value
    return None


@dataclass(slots=True, eq=False)
class _Module:
    """読んだファイル 1 つ 基底クラスを辿るたびに読み直さないよう持っておく"""

    path: Path
    tree: ast.Module
    comments: set[int]
    #: ``__test__`` と ``__all__`` のうち、ファイルの文字に出てくる物 出てこないファイルで
    #: 木を歩かないため（歩くと tests の全部で数秒余計にかかる）
    words: frozenset[str] = frozenset()

    @property
    def package(self) -> bool:
        return self.path.name == "__init__.py"

    def flag(self, body: list[ast.stmt], owner: str | None = None) -> bool | None:
        """このファイルの本体に書いた ``__test__`` の値（``_test_flag``）"""
        return _test_flag(body, owner) if "__test__" in self.words else None


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


@dataclass(frozen=True, slots=True)
class _Unknown:
    """読むだけでは何か決まらない物（呼び出しの結果・見つからない名前など）

    捨てずに候補として残す 捨てると「ほかに何も無い」と読み違え、残った候補だけで
    ``__test__`` を偽と決めたり ``unittest`` の型でないと決めたりして、数を減らす側に倒れる
    （#377 で CodeRabbit が ``Base = object`` の枝を落とす所を指摘した）
    """


#: 組み込みの名前（``object``・``Exception`` など） ファイルの中に無くても解けずに
#: 終わらせず、外の物として持つ ``unittest`` の型でないと分かっているので、数え過ぎない
_BUILTINS = frozenset(dir(builtins))

type _Symbol = _Module | _Class | _Func | _External | _Unknown

#: 辿っている途中のファイルと名前の組 互いを読み込む輪で止まらなくならないため
type _Visiting = frozenset[tuple[Path, str]]


@dataclass(frozen=True, slots=True)
class _Found:
    """名前が指しうる物 ``definite`` は、ブロックの外の文で必ず結ばれるか

    ブロックの中で結ぶ名前は、走らなければ前の値（クラスならモジュールの直下）のまま
    になる どちらになるかは読むだけでは決まらないので、指しうる物を全部持って、全部を
    数える 1 つに決めて読み解くと、決めなかった方の試験を見逃す
    """

    symbols: tuple[_Symbol, ...] = ()
    definite: bool = False


def _test_flag(body: list[ast.stmt], owner: str | None = None) -> bool | None:
    """本体に書いた ``__test__`` の値 ``owner`` があれば ``owner.__test__ = ...`` の方

    外す側（試験を数えない側）へ倒すので、書かれ方が 1 通りに決まるときだけ読む
    本体で ``__test__`` が出てくるのが直下の 1 つの代入だけで、値が定数のとき
    それ以外（ブロックの中・2 度目・呼び出しで決める・``setattr`` など）は None で、
    集める側に倒す 外す側に倒すと、書き添えの無い試験を見逃す
    """
    value = _sole_assignment(body, "__test__")
    if value is None or not isinstance(value, ast.Constant):
        return None
    for node in body:
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is not value:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if owner is None:
                hit = isinstance(target, ast.Name)
            else:
                hit = (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == owner
                )
            if hit:
                return bool(value.value)
    return None


def _exports(module: _Module) -> frozenset[str] | None:
    """``__all__`` に文字の並びで直に書いた名前 書いていないか、読み解けなければ None

    読み解くのは、ファイルの中で ``__all__`` が出てくるのが、直下で文字の並びを
    結ぶ 1 か所だけのときに限る（``_sole_assignment``） ``if TYPE_CHECKING:`` などの
    ブロックの中・``+=``・``extend``・``append``・``del``・添字への代入・``__all__`` を
    渡す呼び出し・``globals()["__all__"]`` など、ほかの形はどれも None（多めに数える側）
    """
    if "__all__" not in module.words:
        return None
    value = _sole_assignment(module.tree.body, "__all__")
    if isinstance(value, ast.List | ast.Tuple) and all(
        isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
    ):
        return frozenset(str(item.value) for item in value.elts if isinstance(item, ast.Constant))
    return None


def _star_exports(module: _Module, name: str) -> bool:
    """``from module import *`` がその名前を持ち込むか

    ``__all__`` を読み解ければ、それで絞る（下線で始まる名前も載っていれば持ち込む）
    読み解けなければ、下線で始まらない名前を全部持ち込む
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
    CodeRabbit が別々の道を何度も指摘した 1 つずつ塞がず、名前の解き方をまとめて作った）

    - ``Test`` で始まるクラスが受け継いだ ``test_*`` 基底は名前が ``Test`` で始まらなくても
      よく、同じファイル・``from`` や ``import`` で読み込んだ別のファイル・パッケージの
      ``__init__.py`` で公開し直した物（``import *`` も）・``shared.Checks`` のような属性の形・
      ``Base = Checks`` のような付け替えのどれでもよい
    - 別のファイルから読み込んだ ``test_*`` の関数と ``Test*`` のクラス（``import *`` も）
    - 名前に関係なく ``unittest.TestCase`` を受け継ぐクラス
    - クラスの中で ``test_b = helper`` と付け替えた関数

    迷ったら多めに数える側へ倒す ブロックの中で結ぶ名前は、結ぶ場合と結ばない場合の
    両方を数え（``_Found``）、``__all__`` と ``__test__`` は書かれ方が 1 通りに決まる
    ときだけ読み解く 数を減らす側に読み解くと、読み違えたときに書き添えの無い試験を
    見逃す 多めに数えて困るのは、要らない書き添えを 1 行求めるだけ

    pytest の収集を基準にすると、試験のファイルが読み込む Qt やエンジンまで起こすので
    数十秒かかり、部品が無い機械では集める前に落ちる AST で辿れば tests の全部を見ても
    数秒で、tests の木だけで答えが決まる リポジトリの外の物（``unittest.TestCase`` など）は
    書き添えを求める先が無いので、``unittest`` の型かどうかを見分けるのにだけ使う
    """

    def __init__(self, root: Path = ROOT) -> None:
        #: 辿ってよい置き場 外の物は書き添えを求める先が無い
        self._root = root.resolve()
        self._modules: dict[Path, _Module] = {}
        self._found: dict[tuple[Path, str], _Found] = {}

    def module(self, path: Path) -> _Module:
        resolved = path.resolve()
        if resolved not in self._modules:
            text = resolved.read_bytes().decode("utf-8")
            words = frozenset(word for word in ("__test__", "__all__") if word in text)
            self._modules[resolved] = _Module(
                resolved, ast.parse(text), _comment_lines(text), words
            )
        return self._modules[resolved]

    def tests(self, path: Path) -> list[tuple[_Module, _Function]]:
        """pytest が集める試験の関数と、それが書いてあるファイル"""
        module = self.module(path)
        found: list[tuple[_Module, _Function]] = []
        # pytest はモジュールに __test__ = False があると、中を 1 本も集めない
        if module.flag(module.tree.body) is False:
            return found
        for name in self._names(module, module.tree.body, frozenset()):
            symbols = self.lookup(module, name).symbols
            collected = self._collected(name, symbols, module, module.tree.body)
            found.extend((owner, node) for _, owner, node in collected)
        return found

    def lookup(self, module: _Module, name: str, visiting: _Visiting = frozenset()) -> _Found:
        """モジュールの直下で名前が指しうる物

        ``visiting`` はファイル同士が互いを読み込む輪で止まらなくならないため 輪の途中で
        出た答えは輪の外から引くと違うことがあるので、覚えるのは外からの答えだけ
        """
        key = (module.path, name)
        if key in self._found:
            return self._found[key]
        if key in visiting:
            return _Found()
        found = self._find_in(module, module.tree.body, None, name, visiting | {key})
        if not visiting:
            self._found[key] = found
        return found

    def _local(
        self, module: _Module, owner: ast.ClassDef, name: str, visiting: _Visiting
    ) -> _Found:
        """クラスの本体で名前が指しうる物 本体で結ばなければ空（モジュールの直下は見ない）

        ``test_a = test_a`` のように自分の名前を右辺に書くと、Python はまだ結んでいない
        クラスの名前を飛ばしてモジュールの直下を見る 同じ組を 2 度引いたら空を返して
        呼び手にモジュールの直下を見させるのは、その順に合わせるため
        """
        key = (module.path, f"{owner.lineno}:{owner.col_offset}:{name}")
        if key in visiting:
            return _Found()
        return self._find_in(module, owner.body, owner, name, visiting | {key})

    def _find_in(
        self,
        module: _Module,
        body: list[ast.stmt],
        owner: ast.ClassDef | None,
        name: str,
        visiting: _Visiting,
    ) -> _Found:
        """モジュールの直下（owner が None）かクラスの本体で、名前が指しうる物

        どちらも同じ手順で解く 名前を結ぶ文（def・class・付け替え・import・from・
        ``import *``）を前から見て、ブロックの外で結んだら前の候補を捨て、ブロックの中で
        結んだら候補に足す 付け替えの右辺は結んだ場所の名前の決まりで解く（クラスの本体
        なら、まずそのクラスの名前） ``import *`` はクラスの中に書けないので直下だけ
        """
        symbols: list[_Symbol] = []
        definite = False
        for node, nested in _statements(body):
            bound = self._binding(module, node, owner, name, visiting)
            if bound is None:
                continue
            if nested or not bound.definite:
                symbols.extend(bound.symbols)
            else:
                symbols = list(bound.symbols)
                definite = True
        return _Found(tuple(dict.fromkeys(symbols)), definite)

    def _binding(
        self,
        module: _Module,
        node: ast.stmt,
        owner: ast.ClassDef | None,
        name: str,
        visiting: _Visiting,
    ) -> _Found | None:
        """1 つの文が名前を結ぶなら、指しうる物 結ばなければ None"""
        if isinstance(node, ast.ClassDef) and node.name == name:
            return _Found((_Class(module, node, owner),), True)
        if isinstance(node, _FUNCTIONS) and node.name == name:
            return _Found((_Func(module, node),), True)
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            named = any(isinstance(t, ast.Name) and t.id == name for t in targets)
            if named and node.value is not None:
                return _Found(self._expr(module, node.value, visiting, owner), True)
            return None
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == name:
                    return _Found(_one(self._module_at(alias.name, 0, module)), True)
                if alias.asname is None and alias.name.split(".")[0] == name:
                    return _Found(_one(self._module_at(name, 0, module)), True)
            return None
        if not isinstance(node, ast.ImportFrom):
            return None
        for alias in node.names:
            if alias.name != "*" and (alias.asname or alias.name) == name:
                return _Found(self._from(module, node, alias.name, visiting), True)
        if owner is None and any(alias.name == "*" for alias in node.names):
            source = self._module_at(node.module or "", node.level, module)
            if isinstance(source, _Module) and _star_exports(source, name):
                found = self.lookup(source, name, visiting)
                # 読み込み先で結ばれるかもしれないだけなら、前の値も残る
                return found if found.symbols else None
        return None

    def _from(
        self,
        module: _Module,
        node: ast.ImportFrom,
        name: str,
        visiting: _Visiting,
    ) -> tuple[_Symbol, ...]:
        """``from X import name`` の name X の中の名前か、X が包みならその下のモジュール"""
        source = self._module_at(node.module or "", node.level, module)
        if isinstance(source, _External):
            return (_External(f"{source.name}.{name}"),)
        if source is None:
            return ()
        found = self.lookup(source, name, visiting)
        submodule = _one(self._submodule(source, name))
        symbols = found.symbols if found.definite else found.symbols + submodule
        return symbols or (_Unknown(),)

    def _expr(
        self,
        module: _Module,
        expr: ast.expr,
        visiting: _Visiting,
        owner: ast.ClassDef | None = None,
    ) -> tuple[_Symbol, ...]:
        """基底や付け替えの右辺に書いた式が指しうる物 名前と属性（``a.b.C``）だけを辿る

        ``owner`` はその式を書いたクラスの本体 名前はまずそのクラスで引き、そこで必ず
        結ばれていなければモジュールの直下の物も足す（Python の名前の決まりと同じ）
        """
        if isinstance(expr, ast.Name):
            found = self._name(module, expr.id, visiting, owner)
        elif isinstance(expr, ast.Attribute):
            found = self._attribute_of(module, expr, visiting, owner)
        else:
            found = ()
        # 何も指さないことは無い 解けなかった物は分からない物として残す
        return found or (_Unknown(),)

    def _name(
        self, module: _Module, name: str, visiting: _Visiting, owner: ast.ClassDef | None
    ) -> tuple[_Symbol, ...]:
        if owner is not None:
            local = self._local(module, owner, name, visiting)
            if local.definite:
                return local.symbols
            symbols = local.symbols + self.lookup(module, name, visiting).symbols
        else:
            symbols = self.lookup(module, name, visiting).symbols
        if not symbols and name in _BUILTINS:
            return (_External(f"builtins.{name}"),)
        return symbols

    def _attribute_of(
        self, module: _Module, expr: ast.Attribute, visiting: _Visiting, owner: ast.ClassDef | None
    ) -> tuple[_Symbol, ...]:
        found: list[_Symbol] = []
        for base in self._expr(module, expr.value, visiting, owner):
            if isinstance(base, _External):
                found.append(_External(f"{base.name}.{expr.attr}"))
            elif isinstance(base, _Module):
                inner = self.lookup(base, expr.attr, visiting)
                found.extend(inner.symbols)
                if not inner.definite:
                    found.extend(_one(self._submodule(base, expr.attr)))
            elif isinstance(base, _Class):
                found.extend(self._attribute(base, expr.attr, visiting, frozenset()))
            else:
                found.append(_Unknown())
        return tuple(dict.fromkeys(found))

    def _attribute(
        self, cls: _Class, name: str, visiting: _Visiting, seen: frozenset[int]
    ) -> tuple[_Symbol, ...]:
        """クラスの属性 本体で必ず結ばれていなければ基底も前から辿る（``Helpers.check`` など）"""
        if id(cls.node) in seen:
            return ()
        local = self._local(cls.module, cls.node, name, visiting)
        if local.definite:
            return local.symbols
        found = list(local.symbols)
        for base_expr in cls.node.bases:
            for located in self._expr(cls.module, base_expr, visiting, cls.outer):
                if isinstance(located, _Class):
                    found.extend(self._attribute(located, name, visiting, seen | {id(cls.node)}))
        return tuple(dict.fromkeys(found))

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
        """モジュールの直下かクラスの本体で結ばれうる名前 ``_find_in`` と同じ文を数える

        ブロックの中で結ぶ名前も入れる ``import *`` は読み込み先の名前を足す（クラスの
        中には書けないので、出るのはモジュールの直下だけ）
        """
        names: list[str] = []
        for node, _ in _statements(body):
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

    def _definite_names(self, body: list[ast.stmt]) -> set[str]:
        """ブロックの外の文で必ず結ばれる名前 基底の試験を上書きしたと言えるのはこれだけ

        ブロックの中でだけ結ぶ名前で基底の試験を外すと、ブロックが走らなかったときに
        集まる基底の試験を見逃す
        """
        names: set[str] = set()
        for node in body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    names.add(node.target.id)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                names.update(
                    alias.asname or alias.name.split(".")[0]
                    for alias in node.names
                    if alias.name != "*"
                )
        # del で消した名前は、基底の物が見えるようになるので上書きとは言えない ブロックの
        # 中の del も、走るかどうかが決まらないので同じ扱い
        for node, _ in _statements(body):
            if isinstance(node, ast.Delete):
                for target in node.targets:
                    names.difference_update(
                        name.id for name in ast.walk(target) if isinstance(name, ast.Name)
                    )
        return names

    def _collected(
        self,
        name: str,
        symbols: tuple[_Symbol, ...],
        home: _Module,
        scope: list[ast.stmt],
    ) -> list[tuple[str, _Module, _Function]]:
        """名前の表の 1 行から pytest が集める試験 モジュールの直下もクラスの本体も同じ決まり

        ``test`` で始まる名前の関数と、``Test`` で始まる名前か ``unittest.TestCase`` を
        受け継ぐクラスの試験 鍵は上書きを見分けるための名前（入れ子のクラスは点でつなぐ）
        名前が指しうる物が幾つかあれば、全部を数える ``scope`` はその名前を結んだ本体で、
        ``test_x.__test__ = False`` を探すのに使う ``__test__`` が偽の物は pytest が
        集めないので、書き添えも求めない
        """
        found: list[tuple[str, _Module, _Function]] = []
        for symbol in symbols:
            if isinstance(symbol, _Func) and name.startswith("test"):
                off = (
                    home.flag(scope, name) is False
                    or symbol.module.flag(symbol.module.tree.body, symbol.node.name) is False
                )
                if not off:
                    found.append((name, symbol.module, symbol.node))
            elif isinstance(symbol, _Class) and (
                name.startswith("Test") or self._is_unittest(symbol, frozenset())
            ):
                if home.flag(scope, name) is False or self._switched_off(symbol, frozenset()):
                    continue
                found.extend(
                    (f"{name}.{key}", owner, node) for key, owner, node in self._class_tests(symbol)
                )
        return found

    def _class_tests(
        self, cls: _Class, seen: frozenset[int] = frozenset()
    ) -> list[tuple[str, _Module, _Function]]:
        """クラスが持つ試験 受け継いだ物も含める 本体で必ず結んだ名前は基底の物より勝つ

        本体の名前はモジュールの直下と同じ ``_names`` と ``_find_in`` で読む
        基底は前から辿り、先の基底が持つ名前は後の基底の物より勝つ ``seen`` は基底を
        辿る輪で止まらなくならないため
        """
        if id(cls.node) in seen:
            return []
        seen = seen | {id(cls.node)}
        found: list[tuple[str, _Module, _Function]] = []
        for name in self._names(cls.module, cls.node.body, frozenset()):
            symbols = self._local(cls.module, cls.node, name, frozenset()).symbols
            found.extend(self._collected(name, symbols, cls.module, cls.node.body))
        taken = self._definite_names(cls.node.body)
        for base in cls.node.bases:
            for located in self._expr(cls.module, base, frozenset(), cls.outer):
                if not isinstance(located, _Class):
                    continue
                inherited = [
                    item
                    for item in self._class_tests(located, seen)
                    if item[0].split(".")[0] not in taken
                ]
                found.extend(inherited)
                taken.update(key.split(".")[0] for key, _, _ in inherited)
        return list(dict.fromkeys(found))

    def _switched_off(self, cls: _Class, seen: frozenset[int]) -> bool:
        """クラスの ``__test__`` が偽と決まるか 本体に無ければ基底から受け継いだ値を見る"""
        return self._class_flag(cls, seen) is False

    def _class_flag(self, cls: _Class, seen: frozenset[int]) -> bool | None:
        """``__test__`` の値 本体に無ければ基底を前から辿り、最初に書いてある物

        基底が幾つかの物を指しうるときは、全部が偽と決まるときだけ偽 1 つでも真か
        分からなければ真（集める側） 本体で読み解けない書き方をしていれば真
        """
        if id(cls.node) in seen:
            return None
        if "__test__" in cls.module.words and _mentions(cls.node.body, "__test__"):
            flag = _test_flag(cls.node.body)
            return True if flag is None else flag
        for base in cls.node.bases:
            # 中を読めない候補（外の物・分からない物）は __test__ を書いていない物として
            # 数に入れる 外すと、残った候補の偽だけでクラスごと外してしまう
            flags = [
                self._class_flag(located, seen | {id(cls.node)})
                if isinstance(located, _Class)
                else None
                for located in self._expr(cls.module, base, frozenset(), cls.outer)
            ]
            defined = [flag for flag in flags if flag is not None]
            if not defined:
                continue
            # 全部が偽と決まるときだけ偽 真や書いていない物が 1 つでも混ざれば真
            return len(defined) != len(flags) or any(defined)
        return None

    def _is_unittest(self, cls: _Class, seen: frozenset[int]) -> bool:
        """``unittest.TestCase`` を受け継ぎうるか pytest は名前に関係なく集める

        外の物は中を読めない ``unittest`` の型そのものでなくても、それを受け継ぐ外の型
        （``django.test.TestCase`` など）かもしれないので、組み込みの型でなければ受け継ぎうると
        見る 分からない物（呼び出しで作る基底など）も同じ 数え過ぎるのは、名前が ``Test`` で
        始まらず ``test_*`` を持つクラスだけ
        """
        if id(cls.node) in seen:
            return False
        seen = seen | {id(cls.node)}
        for base in cls.node.bases:
            for located in self._expr(cls.module, base, frozenset(), cls.outer):
                if isinstance(located, _Unknown):
                    return True
                if isinstance(located, _External) and not located.name.startswith("builtins."):
                    return True
                if isinstance(located, _Class) and self._is_unittest(located, seen):
                    return True
        return False


def _one(symbol: _Symbol | None) -> tuple[_Symbol, ...]:
    return () if symbol is None else (symbol,)


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
