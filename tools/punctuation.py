r"""文章に句点（まる）を使わない、という約束を確かめる

    .venv\Scripts\python.exe tools\punctuation.py          # 確かめるだけ
    .venv\Scripts\python.exe tools\punctuation.py --fix    # 書き換える

対象はコメント・docstring・画面や例外に出す文言・Markdown などの**文章**
**データとしての句点は対象外**で、ここを取り違えると機能が壊れる

- 字幕整形（``src/sashimono/asr/cleanup.py``）が句点として扱う文字の一覧や正規表現
- テストの入力データ（起こし結果の字幕文など）

そのため Python では字句に分けてから、その句点がどこにあるかで決める

=================================  ======================
場所                               扱い
=================================  ======================
コメント                           文章
docstring（式文としての文字列）    文章
``src`` の文字列リテラル           文章（画面や例外に出る）
``tools`` の文字列リテラル         文章（端末に出る）
``tests`` の文字列リテラル         データ（入力と期待値）
``cleanup.py`` の文字列リテラル    データ（句点の一覧）
=================================  ======================

書き換えは、句点の直後で決める 行末や閉じ括弧の前ならただ消し、文の途中なら
半角空白にする 消すだけにすると、次の文と地続きになって読めなくなる
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 句点 コードに直接書くとこのファイル自身が検査に掛かるので、番号で持つ
MARU = chr(0x3002)

ROOT = Path(__file__).resolve().parent.parent

#: 文章として検査するファイルの種類
#: ``.ps1`` は YMM4 を操作する道具（tools/ymm4_export.ps1） 中の文字はどれも文章
TEXT_SUFFIXES = {".md", ".mdc", ".yml", ".yaml", ".toml", ".txt", ".ps1"}

#: 見ないところ
SKIP_PARTS = {
    ".venv",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".work",
}
#: リポジトリ直下だけを外すもの 名前だけで外すと、同じ名前のソースの
#: フォルダ（``src/sashimono/build/`` など）まで黙って検査から抜ける
#: ``build/`` ``dist/`` は配る zip を組み立てた残り（tools/build_package.py）
#: 部品の中には UTF-8 でない文字列ファイルもあり、読んだ時点で検査が落ちる
#: ``.claude/`` は Claude Code が別のブランチを並行して作業する置き場（git の
#: worktree） 中身は別のブランチの写しなので、ここで数えると main の検査が
#: まだ直していない古い句点や、組み立てた残りの UTF-8 でないファイルで落ちる
SKIP_PREFIXES = ("tests/fixtures/", "build/", "dist/", ".claude/")

#: 文字列リテラルがデータとして句点を持つファイル リテラルだけ見逃す
DATA_LITERAL_FILES = {"src/sashimono/asr/cleanup.py"}

#: この文字の前にある句点は、ただ消す（空白を挟むと不自然になる）
_CLOSERS = set("」』）)】〕》’”\"'`*_]|\\") | {" ", "　", "\n", "\r", "\t"}


@dataclass(frozen=True, slots=True)
class Hit:
    """見つかった句点 1 つ"""

    path: Path
    line: int
    column: int
    #: 書き換えるときの置き換え先（空文字か半角空白）
    replacement: str

    def __str__(self) -> str:
        return f"{self.path.relative_to(ROOT).as_posix()}:{self.line}:{self.column + 1}"


def _replacement(text: str, index: int) -> str:
    following = text[index + 1] if index + 1 < len(text) else "\n"
    for mark in ("**", "`"):
        if text.startswith(mark, index + 1):
            # 太字やコードの記号は、閉じなら直前の文にくっ付き、開きなら次の文の頭に
            # なる 開きの前で空白を消すと「書く**次の文**」「受ける`main`」と
            # 詰まって読めない 両方とも一度そうなった
            # 段落の頭からここまでに出た数が奇数なら、いまは記号の中（= 閉じ）
            # 行の頭から数えると、行をまたぐ太字で閉じを開きと取り違える
            inside = text.count(mark, _paragraph_start(text, index), index) % 2 == 1
            return "" if inside else " "
    return "" if following in _CLOSERS else " "


def _paragraph_start(text: str, index: int) -> int:
    """その位置を含む段落の頭

    空行（コメントなら ``#`` だけの行、引用なら ``>`` だけの行も）で区切る
    """
    line_start = text.rfind("\n", 0, index) + 1
    while line_start > 0:
        previous_start = text.rfind("\n", 0, line_start - 1) + 1
        previous = text[previous_start : line_start - 1]
        if not previous.strip().lstrip("#>").strip():
            break
        line_start = previous_start
    return line_start


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_skipped(path: Path) -> bool:
    relative = _relative(path)
    return any(part in SKIP_PARTS for part in path.parts) or relative.startswith(SKIP_PREFIXES)


def _offsets(text: str) -> list[int]:
    """各行の先頭が、全体の何文字目か"""
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


type Span = tuple[tuple[int, int], tuple[int, int]]


def _docstring_spans(tree: ast.AST) -> list[Span]:
    """式文としての文字列（docstring と、属性の説明に置く文字列）の範囲

    開始位置だけで見ると、つないで書いた docstring（"前半" "後半"）の後半を
    取りこぼす ast には 1 つの式だが、字句では文字列が 2 つに分かれる
    """
    found: list[Span] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
            and node.end_col_offset is not None
        ):
            found.append(((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)))
    return found


def _within(position: tuple[int, int], spans: list[Span]) -> bool:
    return any(start <= position < end for start, end in spans)


def scan_python(path: Path) -> list[Hit]:
    text = _read(path)
    if MARU not in text:
        return []

    relative = _relative(path)
    # tools の出力も人が読む文章 ここを外していたので「すべて通過」に句点が残っていた
    literals_are_prose = (
        relative.startswith(("src/", "tools/")) and relative not in DATA_LITERAL_FILES
    )
    docstrings = _docstring_spans(ast.parse(text))
    starts = _offsets(text)

    hits: list[Hit] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if MARU not in token.string:
            continue
        if token.type == tokenize.COMMENT:
            prose = True
        elif token.type == tokenize.STRING:
            prose = _within(token.start, docstrings) or literals_are_prose
        elif token.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            prose = literals_are_prose
        else:
            prose = False
        if not prose:
            continue

        base = starts[token.start[0] - 1] + token.start[1]
        # f 文字列の中身は、字句の文字列と元の文字列で波括弧の数が食い違う
        # ことがある 元の文字列の範囲から直接探す
        end = starts[token.end[0] - 1] + token.end[1]
        for index in range(base, end):
            if text[index] != MARU:
                continue
            line = text.count("\n", 0, index) + 1
            column = index - starts[line - 1]
            hits.append(Hit(path, line, column, _replacement(text, index)))
    return hits


def scan_text(path: Path) -> list[Hit]:
    text = _read(path)
    starts = _offsets(text)
    hits: list[Hit] = []
    for index, character in enumerate(text):
        if character != MARU:
            continue
        line = text.count("\n", 0, index) + 1
        hits.append(Hit(path, line, index - starts[line - 1], _replacement(text, index)))
    return hits


def _read(path: Path) -> str:
    """改行をそのまま保って読む

    read_text は改行を LF に揃え、write_text は Windows で CRLF に戻す 往復させると
    触っていない行まで改行が変わる
    """
    return path.read_bytes().decode("utf-8")


def _is_inside(path: Path, root: Path) -> bool:
    """リポジトリの中の実体か

    シンボリックリンクを辿ると、--fix がリポジトリの外のファイルを書き換える
    PR にリンクを 1 つ混ぜるだけで、手元で走らせた人の別のファイルを壊せてしまう
    """
    return not path.is_symlink() and path.resolve().is_relative_to(root.resolve())


def targets(root: Path = ROOT) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_skipped(path) or not _is_inside(path, root):
            continue
        if path.suffix == ".py" or path.suffix in TEXT_SUFFIXES or path.name == "CODEOWNERS":
            found.append(path)
    return found


def scan(path: Path) -> list[Hit]:
    return scan_python(path) if path.suffix == ".py" else scan_text(path)


#: 文字の範囲（始まりを含み、終わりを含まない）
type Spans = Sequence[tuple[int, int]]


def _prose_marks(text: str, keep: Spans) -> list[int]:
    """句点の位置のうち、残す範囲（貼ったログやコードなどのデータ）の外の物"""
    return [
        index
        for index, character in enumerate(text)
        if character == MARU and not any(start <= index < end for start, end in keep)
    ]


def find_in_text(text: str, keep: Spans = ()) -> list[tuple[int, int]]:
    """リポジトリの外の文章（Issue の本文など）にある句点の行と桁（どちらも 1 から）

    ファイルを介さずに見る口 ``tools/new_issue.py`` が送る前の本文を見る
    Issue の本文は送ってしまうとこの検査の外になり、混ざっても誰も気付かない（#376）
    ``keep`` の中はデータとして見ない 字幕整形の不具合の再現に貼った文から句点を
    消すと、再現できない報告になる 行は LF だけで数える（行の頭も同じ決まりで探す
    splitlines は垂直タブなどでも割るので、行と桁が食い違う）
    """
    found: list[tuple[int, int]] = []
    for index in _prose_marks(text, keep):
        line = text.count("\n", 0, index) + 1
        found.append((line, index - (text.rfind("\n", 0, index) + 1) + 1))
    return found


def fix_text(text: str, keep: Spans = ()) -> str:
    """文章の句点を、ファイルの書き換えと同じ決まりで消すか半角空白にする

    決まりを 2 か所に持つと、Issue の本文とリポジトリの中で直り方が食い違う
    ``keep`` の中は直さない（find_in_text と同じ）
    """
    # 置き換え先は元の文で先に決める（fix と同じ） 書き換えながら決めると、
    # 続いた句点の扱いが前の書き換えに引きずられる
    original = text
    for index in reversed(_prose_marks(original, keep)):
        text = text[:index] + _replacement(original, index) + text[index + 1 :]
    return text


def fix(path: Path, hits: list[Hit]) -> None:
    """後ろから書き換える 前から直すと、後ろの位置がずれる"""
    if not _is_inside(path, ROOT):
        raise ValueError(f"リポジトリの外は書き換えない: {path}")
    text = _read(path)
    starts = _offsets(text)
    for hit in sorted(hits, key=lambda h: (h.line, h.column), reverse=True):
        index = starts[hit.line - 1] + hit.column
        text = text[:index] + hit.replacement + text[index + 1 :]
    path.write_bytes(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="見つけたものを書き換える")
    arguments = parser.parse_args(argv)

    total = 0
    for path in targets():
        hits = scan(path)
        if not hits:
            continue
        total += len(hits)
        if arguments.fix:
            fix(path, hits)
        else:
            for hit in hits[:3]:
                print(hit)
            if len(hits) > 3:
                print(f"  ...ほか {len(hits) - 3} か所（{_relative(path)}）")

    if arguments.fix:
        print(f"{total} か所を書き換えた")
        return 0
    if total:
        print(f"文章に句点が {total} か所ある  tools\\punctuation.py --fix で書き換えられる")
        return 1
    print("文章に句点は無い")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
