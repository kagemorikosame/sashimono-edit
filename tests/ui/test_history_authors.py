"""文書を変える入口が、誰の変更かの印（by）を書いているかを見張る（#333）

AI の指示の途中に人が変えた物は、指示の「AI: 〜」の段に混ぜず人の段として分ける
（2026-10-18 利用者の決定） 分けられるのは、編集画面の人の操作が ``by=Author.PERSON`` を
書いているときだけ ``by`` を省くと開いている段の印を継ぐので、人の操作の入口で書き忘れると、
AI の指示の途中の人の変更が AI の段に混ざり、AI の指示を取り消したときに一緒に消える

コアの外（画面・AI の層）のソースを読み、文書を変える呼び出しを全部拾って確かめる
- ``checkpoint`` と ``begin_checkpoint`` は、どこで呼んでも ``by`` を書く
- ``execute`` は、``checkpoint`` の ``with`` の中（まとめの印を継ぐ）のほかは ``by`` を書く
"""

from __future__ import annotations

import ast
from pathlib import Path

import sashimono

#: コアの外のソース コアの中（core/）は印を継ぐ側で、ここでは見ない
SOURCE_ROOT = Path(sashimono.__file__).parent
#: 文書を変える呼び出しの名前
OPENERS = frozenset({"checkpoint", "begin_checkpoint"})


def _receiver_is_document(call: ast.Call) -> bool:
    """``self._document.execute`` ``host.document.execute`` などの、文書への呼び出しか

    Lua の ``self._lua.execute`` のような同じ名前の別の物は拾わない
    """
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    receiver = func.value
    name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, "id", "")
    return "document" in name


def _has_by(call: ast.Call) -> bool:
    return any(keyword.arg == "by" for keyword in call.keywords)


def _opens_a_group(node: ast.With) -> bool:
    for item in node.items:
        expression = item.context_expr
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "checkpoint"
            and _receiver_is_document(expression)
        ):
            return True
    return False


def unmarked_calls(source: str, filename: str) -> list[str]:
    """``by`` を書いていない、文書を変える呼び出しの場所（ファイル名:行）"""
    found: list[str] = []

    def visit(node: ast.AST, inside_group: bool) -> None:
        if isinstance(node, ast.Call) and _receiver_is_document(node):
            attribute = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            needs = attribute in OPENERS or (attribute == "execute" and not inside_group)
            if needs and not _has_by(node):
                found.append(f"{filename}:{node.lineno}")
        if isinstance(node, ast.With) and _opens_a_group(node):
            for item in node.items:
                visit(item.context_expr, inside_group)
            for statement in node.body:
                visit(statement, True)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, inside_group)

    visit(ast.parse(source), False)
    return found


def _outside_the_core() -> list[Path]:
    return [
        path
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "core" not in path.relative_to(SOURCE_ROOT).parts
    ]


class TestEveryEntranceMarksWhoChanged:
    def test_every_call_that_changes_the_document_says_who(self) -> None:
        """壊れると、人の操作の入口で印を書き忘れ、AI の指示の途中の人の変更が AI の段に混ざる"""
        missing: list[str] = []
        for path in _outside_the_core():
            relative = path.relative_to(SOURCE_ROOT).as_posix()
            missing += unmarked_calls(path.read_text(encoding="utf-8"), relative)
        assert missing == []

    def test_the_watch_reads_the_screen_and_the_ai_layers(self) -> None:
        """壊れると、探す場所がずれて 1 つも拾わず、見張りが黙って通る"""
        sources = {path.relative_to(SOURCE_ROOT).as_posix() for path in _outside_the_core()}
        assert "ui/main_window.py" in sources
        assert "ui/chat/panel.py" in sources

    def test_a_forgotten_mark_is_caught(self) -> None:
        """壊れると、印を書き忘れた入口を足しても見張りが気付かない"""
        source = (
            "def edit(self, command):\n"
            "    self._document.execute(command)\n"
            "    self._document.begin_checkpoint('AI: 指示')\n"
            "    with self._document.checkpoint('まとめ', by=Author.PERSON):\n"
            "        self._document.execute(command)\n"
            "    self._lua.execute('x')\n"
        )
        assert unmarked_calls(source, "試し.py") == ["試し.py:2", "試し.py:3"]

    def test_a_group_without_a_mark_is_caught(self) -> None:
        """壊れると、まとめの印を書き忘れても、中の execute が継ぐので見落とす"""
        source = (
            "def edit(self, command):\n"
            "    with self._document.checkpoint('まとめ'):\n"
            "        self._document.execute(command)\n"
        )
        assert unmarked_calls(source, "試し.py") == ["試し.py:2"]
