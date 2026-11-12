r"""開発者向けの Issue を、そろった形で作る（#376）

    # 1. 雛形を書き出して埋める（種類は feature・bug・investigation）
    .venv\Scripts\python.exe tools\new_issue.py draft feature .work\issue.md

    # 2. 送らずに確かめる（本文の仕上がりと、付けるラベルを出す gh は呼ばない）
    .venv\Scripts\python.exe tools\new_issue.py create feature --title "…" ^
        --area 編集の土台 --size 中 --milestone 0.2.6 --body-file .work\issue.md --dry-run

    # 3. 作る（--dry-run を外す）
    # 作ったあとのロードマップへの登録だけをやり直す
    .venv\Scripts\python.exe tools\new_issue.py register 450

作るときに一度に行うこと

- 雛形の「必須」の節がそろって埋まっているかを見る 空の任意の節は外して知らせる
- 題と本文の句点を見る（``--fix`` で ``tools/punctuation.py`` と同じ決まりで直す）
  コードの囲みとインラインのコードの中はデータとして見ない
- 同じ題の開いた Issue が無いかを見る（やり直しで 2 本作らない）
- ラベルを 4 種類（種類・分野・大きさ・状態）から付ける 分野と大きさは必ず決める
- マイルストーンを付ける
- 公開のロードマップ（プロジェクト 3）に足し、Status を「未着手」、大きさの欄を合わせる

雛形は ``docs/issue-templates/`` に置く ``.github/ISSUE_TEMPLATE/`` に置くと、使う人が
Discussions を通らずに Issue を作れてしまう（docs/development.md の「報告の受け口」）
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# 止めた訳は標準エラーに出す cp932 の端末で日本語が化けたり落ちたりしないよう両方そろえる
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
# 句点の決まりは隣の道具をそのまま使う 書き直すと、リポジトリの中と Issue で直り方が食い違う
sys.path.insert(0, str(Path(__file__).resolve().parent))

import punctuation  # noqa: E402

TEMPLATES = ROOT / "docs" / "issue-templates"

#: Issue を作る先 手元の gh が fork を既定にしていても、本家に作る
#: 推し量らせると、Issue は fork に、ロードマップの登録は本家の板にと食い違う
REPOSITORY = "kagemorikosame/sashimono-edit"

#: 雛形の種類と、そのとき既定で付ける種類のラベル
KINDS: dict[str, str] = {
    "feature": "enhancement",
    "bug": "bug",
    "investigation": "enhancement",
}

# ラベルの 4 種類 リポジトリのラベルと名前を合わせる（gh label list）
# 打ち間違いを gh に任せると、Issue を作る直前まで気付けない
TYPE_LABELS = ("bug", "enhancement", "決めるだけ")
AREA_LABELS = ("編集の土台", "配る準備", "互換: AviUtl", "互換: YMM4")
SIZE_LABELS = ("小", "中", "大")
STATE_LABELS = ("材料待ち", "本人の手が要る")
#: 分野が 4 つのどれにも当たらないとき（開発の段取りの Issue など）に、
#: 付け忘れと区別するために明示で選ぶ値
NO_AREA = "なし"

PROJECT_OWNER = "kagemorikosame"
PROJECT_NUMBER = 3
STATUS_FIELD = "Status"
STATUS_NEW = "未着手"
SIZE_FIELD = "大きさ"

#: 道具を打つときの頭 .py の関連付けが無い Windows でもそのまま打てる形で案内する
COMMAND = r".venv\Scripts\python.exe tools\new_issue.py"

#: gh 1 回を待つ上限（秒） 待ち続けると、止まったのか遅いのかが分からないまま固まる
GH_TIMEOUT = 120

#: 本文の頭（最初の見出しより前） 一行の目的を書く所で、見出しを持たない
LEAD = "（冒頭）"

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_REQUIRED = re.compile(r"<!--\s*必須")
# 引用（>）とリストの印の後ろに置いた囲みも囲みと取る 取らないと、引用して貼ったログの
# 句点や <!-- --> を本文として直してしまう 行の頭の字下げは数えない（狭く取るより、
# コードを本文と取り違えて再現の材料を壊す方が困る）
_FENCE = re.compile(r"^((?:[ \t]*(?:>|(?:[-+*]|\d{1,9}[.)])(?=[ \t])))*)([ \t]*)(`{3,}|~{3,})(.*)$")
#: Markdown の行の区切り（LF・CR・CRLF） splitlines は垂直タブなどでも割るので、
#: 行の途中の ``` を行の頭と取り違え、後ろの文章をコードと見て句点を見逃す
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")
#: インラインのコードが越えない区切り（空行と見出し） 越えると、閉じの無い ` から
#: 次の節の文章までをコードと取り、その間の句点を見逃す
_BLOCK_END = re.compile(r"(?:\r\n|\r|\n)(?=[ \t]*(?:\r\n|\r|\n)|#{1,6}(?:[ \t]|\r|\n|$))")
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"


class IssueError(Exception):
    """ここで止める Issue はまだ作っていない（または作った後の登録で止まった）"""


# --- 本文の読み分け -------------------------------------------------------------


def fenced_lines(lines: Sequence[str]) -> list[bool]:
    """行ごとに、コードの囲み（区切りの行を含む）の中か

    閉じは開きと同じ文字で、開き以上の長さの行だけ ``````` の中に書いた ```` ``` ```` の
    例を閉じと取ると、その後ろのコードの見出しや句点を本文として扱ってしまう
    閉じが無ければ最後まで囲みの中（CommonMark と同じ）
    ただし引用やリストの中で開いた囲みは、その引用やリストが終わった所で閉じる
    閉じないと、閉じ忘れた引用の後ろの見出しや文章までコードと取り、必須の節が無いと止めたり
    句点を見逃したりする
    """
    flags: list[bool] = []
    opener: str | None = None
    quoted = False
    list_indent: int | None = None
    for line in lines:
        raw = line.rstrip("\r\n")
        if opener is not None and _leaves_container(raw, quoted, list_indent):
            opener = None
        match = _FENCE.match(raw)
        if opener is None:
            if match and not (match.group(3)[0] == "`" and "`" in match.group(4)):
                opener = match.group(3)
                prefix = match.group(1)
                quoted = ">" in prefix
                in_list = bool(prefix.strip(" \t>"))
                list_indent = len(prefix) + len(match.group(2)) if in_list else None
                flags.append(True)
                continue
            flags.append(False)
            continue
        flags.append(True)
        if (
            match
            and match.group(3)[0] == opener[0]
            and len(match.group(3)) >= len(opener)
            and not match.group(4).strip()
        ):
            opener = None
    return flags


def _leaves_container(raw: str, quoted: bool, list_indent: int | None) -> bool:
    """引用やリストの中で開いた囲みから、その行で外へ出たか

    引用は ``>`` の無い行（空行を含む）で終わる リストは、項目の字下げより浅い、空でない行で終わる
    """
    if quoted:
        return not raw.lstrip(" \t").startswith(">")
    if list_indent is None or not raw.strip():
        return False
    return len(raw) - len(raw.lstrip(" \t")) < list_indent


def code_lines(lines: Sequence[str]) -> list[bool]:
    """行ごとに、コード（囲みの中か、空行や見出しの後ろの 4 字下げ）か

    4 字下げのコードも Markdown ではコードとして出る ここを本文と取ると、貼った再現用の
    文から句点を消したり、``<!-- -->`` の例を注意書きとして外したりする
    リストの続きの段落も 4 字下げになるが、データを壊すよりは句点を見逃す方を選ぶ
    """
    flags: list[bool] = []
    may_start = True
    previous_code = False
    for line, fenced in zip(lines, fenced_lines(lines), strict=True):
        raw = line.rstrip("\r\n")
        if fenced:
            flags.append(True)
            may_start, previous_code = True, False
            continue
        if not raw.strip():
            flags.append(False)
            may_start = True
            continue
        indented = raw.startswith(("    ", "\t")) and (may_start or previous_code)
        flags.append(indented)
        previous_code = indented
        # 段落の途中の 4 字下げは段落の続きで、コードにならない
        may_start = bool(_HEADING.match(raw))
    return flags


def split_lines(text: str) -> list[str]:
    """Markdown の改行（LF・CR・CRLF）だけで割る 改行は行に残す"""
    return _LINE.findall(text)


def code_spans(text: str) -> list[tuple[int, int]]:
    """コードの囲みとインラインのコードが占める文字の範囲

    ここは貼ったログや再現用の文字列で、データとして元のまま送る
    句点を消したり ``<!-- -->`` を注意書きとして外したりすると、再現できない報告になる
    """
    lines = split_lines(text)
    spans: list[tuple[int, int]] = []
    offset = 0
    prose_start = 0
    for line, fenced in zip(lines, code_lines(lines), strict=True):
        if fenced:
            spans += _inline_spans(text, prose_start, offset)
            spans.append((offset, offset + len(line)))
            prose_start = offset + len(line)
        offset += len(line)
    spans += _inline_spans(text, prose_start, len(text))
    return sorted(spans)


def _inline_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    r"""囲みの外のインラインのコード（CommonMark と同じ読み方）

    閉じは開きと同じ数の ` の並びで、段落（空行や見出しまで）の中なら行をまたいでよい
    行ごとに探すと、行をまたぐコードの中の句点や ``<!-- -->`` を本文として直してしまう
    ``\``` は文字の ` で、コードの開きにしない 開きと取ると、その後ろの文章の句点を見逃す
    """
    spans: list[tuple[int, int]] = []
    index = start
    while index < end:
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character != "`":
            index += 1
            continue
        run = index
        while run < end and text[run] == "`":
            run += 1
        block_end = _BLOCK_END.search(text, run, end)
        close = _closing_run(text, run, block_end.start() if block_end else end, run - index)
        if close is None:
            # 閉じの無い ` はただの文字 後ろの ` を新しい開きとして探し直す
            index = run
            continue
        spans.append((index, close + run - index))
        index = close + run - index
    return spans


def _closing_run(text: str, start: int, end: int, width: int) -> int | None:
    """start から end までで、ちょうど width 個の ` が並ぶ所の頭"""
    index = start
    while index < end:
        if text[index] != "`":
            index += 1
            continue
        run = index
        while run < end and text[run] == "`":
            run += 1
        if run - index == width:
            return index
        index = run
    return None


def _inside(index: int, spans: Sequence[tuple[int, int]]) -> bool:
    """位置が範囲のどれかの中か"""
    return any(start <= index < end for start, end in spans)


@dataclass(frozen=True, slots=True)
class Section:
    """見出し 1 つと、次の見出しまでの中身"""

    title: str
    #: 見出しの深さ 冒頭は 0
    level: int
    #: 見出しの行（冒頭は空）
    heading: str
    text: str


def split_sections(body: str) -> list[Section]:
    """見出しで分ける コードの囲みと注意書き（``<!-- -->``）の中の ``#`` は見出しに数えない

    囲みの中を数えると、貼ったログやシェルのコメントで節が割れ、中身のある節を空と見誤る
    注意書きの中を数えると、公開では消える注意書きの中の節で、必須の節を埋めたことになる
    """
    sections: list[Section] = []
    title, level, heading = LEAD, 0, ""
    collected: list[str] = []
    lines = split_lines(body)
    comments = comment_spans(body)
    offset = 0
    for line, fenced in zip(lines, fenced_lines(lines), strict=True):
        hidden = fenced or _inside(offset, comments)
        offset += len(line)
        match = None if hidden else _HEADING.match(line.rstrip("\r\n"))
        if match is None:
            collected.append(line)
            continue
        sections.append(Section(title, level, heading, "".join(collected)))
        title, level, heading = match.group(2), len(match.group(1)), line
        collected = []
    sections.append(Section(title, level, heading, "".join(collected)))
    return sections


def required_sections(template: str) -> list[str]:
    """雛形で、注意書きが「必須」で始まる節の名前

    冒頭には雛形全体の注意書きが先に来るので、頭の 1 つだけでなく、節の中のどれかで見る
    """
    return [s.title for s in split_sections(template) if _REQUIRED.search(s.text)]


def comment_spans(text: str) -> list[tuple[int, int]]:
    """注意書き（``<!-- -->``）が占める文字の範囲 コードの中に貼ったコメントは含めない

    開きはコードの外の物だけを探す コードの中の ``<!--`` から探し始めると、その後ろの
    注意書きの ``-->`` までを 1 つと取り、まとめて残してしまう
    """
    spans = code_spans(text)
    found: list[tuple[int, int]] = []
    index = text.find(_COMMENT_OPEN)
    while index != -1:
        if _inside(index, spans):
            index = text.find(_COMMENT_OPEN, index + 1)
            continue
        close = text.find(_COMMENT_CLOSE, index + len(_COMMENT_OPEN))
        if close == -1:
            break
        found.append((index, close + len(_COMMENT_CLOSE)))
        index = text.find(_COMMENT_OPEN, close + len(_COMMENT_CLOSE))
    return found


def strip_comments(text: str) -> str:
    """雛形の注意書き（``<!-- -->``）を外す コードの中に貼ったコメントは残す"""
    pieces: list[str] = []
    last = 0
    for start, end in comment_spans(text):
        pieces.append(text[last:start])
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def _empty_flags(sections: list[Section]) -> list[bool]:
    """節ごとに、下の節まで含めて中身が無いか

    「進め方」の下に「### 段階 1」を書く形があるので、見出しの直下が空でも、
    下の節に中身があれば空にしない
    """
    flags = [False] * len(sections)
    for index in range(len(sections) - 1, -1, -1):
        section = sections[index]
        empty = not strip_comments(section.text).strip()
        cursor = index + 1
        # 冒頭は見出しを持たないので、後ろの節を下の節に数えない 数えると、本文のどこかに
        # 中身があるだけで、一行の目的が空でも埋まったことになる
        while empty and section.level and cursor < len(sections):
            if sections[cursor].level <= section.level:
                break
            empty = flags[cursor]
            cursor += 1
        flags[index] = empty
    return flags


@dataclass(slots=True)
class Checked:
    """確かめた結果"""

    body: str
    errors: list[str] = field(default_factory=list)
    #: 止めないが知らせること（外した節など）
    notes: list[str] = field(default_factory=list)


def check_body(body: str, required: Sequence[str]) -> Checked:
    """必須の節を確かめ、注意書きと空の任意の節を外した本文を返す"""
    sections = split_sections(body)
    flags = _empty_flags(sections)
    titles = [s.title for s in sections]
    result = Checked(body="")

    for name in required:
        if name not in titles:
            # 冒頭は見出しが無いので「無い」にはならない 空のときは下で拾う
            result.errors.append(f"必須の節「{name}」が無い")
    kept: list[str] = []
    for section, empty in zip(sections, flags, strict=True):
        if empty and section.title in required:
            what = "一行の目的" if section.title == LEAD else f"節「{section.title}」"
            result.errors.append(f"必須の{what}が空")
        if empty:
            if section.title != LEAD:
                result.notes.append(f"空の節「{section.title}」を外した")
            continue
        kept.append(section.heading + strip_comments(section.text))
    result.body = _tidy("".join(kept))
    return result


def _tidy(text: str) -> str:
    """注意書きを外した跡の空行を詰める 3 行以上空くと、GitHub の表示で節の間が不揃いになる

    コードの囲みの中は詰めない 貼ったログの空行や行末の空白も再現の材料のうち
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    kept: list[bool] = []
    for line, fenced in zip(lines, code_lines(lines), strict=True):
        if fenced:
            out.append(line)
            kept.append(True)
            continue
        line = line.rstrip()
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
        kept.append(False)
    # 末尾の空行は囲みの外の物だけ詰める 閉じの無い囲みで終わる本文は、末尾の空行も
    # 貼ったログのうち（split の最後の空の要素は最後の改行の分で、付け直すので外す）
    if out and not out[-1]:
        out.pop()
        kept.pop()
    while out and not out[-1] and not kept[-1]:
        out.pop()
        kept.pop()
    return "\n".join(out) + "\n"


def check_punctuation(title: str, body: str) -> list[str]:
    """句点の場所 Issue はリポジトリの外で、tools/punctuation.py の検査が届かない"""
    found = [
        f"題の {column} 文字目" for _, column in punctuation.find_in_text(title, code_spans(title))
    ]
    found += [
        f"本文の {line} 行 {column} 文字目"
        for line, column in punctuation.find_in_text(body, code_spans(body))
    ]
    return [f"句点がある（{where}）" for where in found]


def fix_punctuation(text: str) -> str:
    """句点を直す コードの中は直さない（check_punctuation と同じ範囲）"""
    return punctuation.fix_text(text, code_spans(text))


def build_labels(type_label: str, area: str, size: str, states: Sequence[str]) -> list[str]:
    """付けるラベル 4 種類のうち、種類・分野・大きさは 1 つずつ、状態は 0 個以上"""
    if type_label not in TYPE_LABELS:
        raise IssueError(f"種類のラベル {type_label!r} は無い（{' '.join(TYPE_LABELS)}）")
    if area != NO_AREA and area not in AREA_LABELS:
        raise IssueError(f"分野のラベル {area!r} は無い（{' '.join(AREA_LABELS)} {NO_AREA}）")
    if size not in SIZE_LABELS:
        raise IssueError(f"大きさのラベル {size!r} は無い（{' '.join(SIZE_LABELS)}）")
    for state in states:
        if state not in STATE_LABELS:
            raise IssueError(f"状態のラベル {state!r} は無い（{' '.join(STATE_LABELS)}）")
    labels = [type_label]
    if area != NO_AREA:
        labels.append(area)
    labels.append(size)
    labels.extend(dict.fromkeys(states))
    return labels


def size_advice(kind: str, size: str, parent: str | None) -> list[str]:
    """「大」を 1 本の Issue にまとめると、着手の単位が分からなくなる"""
    if size != "大" or kind == "investigation" or parent:
        return []
    return [
        "大きさが「大」 PR 1 本に収まらないなら、investigation の雛形で親の Issue にして、"
        "着手の単位ごとに --parent <親の番号> で子の Issue へ分ける"
    ]


def template_path(kind: str) -> Path:
    """種類の雛形の置き場 知らない種類なら止める"""
    if kind not in KINDS:
        raise IssueError(f"雛形の種類 {kind!r} は無い（{' '.join(KINDS)}）")
    return TEMPLATES / f"{kind}.md"


def read_text(path: Path) -> str:
    """BOM を読み飛ばす PowerShell の Out-File は既定で BOM を付けて書く"""
    return path.read_bytes().decode("utf-8-sig")


def replace_file(path: Path, text: str) -> None:
    """書いてから置き換える 書いている途中で止まっても、元の下書きは残る

    write_bytes は先に中身を空にするので、途中で落ちると書きかけの本文が消える
    """
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(text.encode("utf-8"))
        Path(temporary).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temporary).unlink()
        raise


def _is_inside_repository(path: Path) -> bool:
    """リポジトリの中の実体か（シンボリックリンクは辿らない punctuation.py の fix と同じ）"""
    return not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve())


# --- gh ---------------------------------------------------------------------

Runner = Callable[[Sequence[str]], str]


def run_gh(arguments: Sequence[str]) -> str:
    """gh を走らせて標準出力を返す 失敗したら gh の言い分ごと止める

    渡すのはこの道具が組んだ引数だけ shell は通さない
    """
    what = " ".join(arguments[:2])
    try:
        # 監査済み 引数はこの道具が組んだ物で、外から来るのは題とラベルとマイルストーンの名前だけ
        completed = (
            subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                ["gh", *arguments],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=GH_TIMEOUT,
            )
        )
    except subprocess.TimeoutExpired as error:
        raise IssueError(
            f"gh {what} が {GH_TIMEOUT} 秒で返らなかった GitHub 側では受け付けられたかもしれない"
        ) from error
    if completed.returncode != 0:
        raise IssueError(f"gh {what} が失敗した\n{completed.stderr.strip()}")
    return completed.stdout


@dataclass(frozen=True, slots=True)
class Choice:
    """ロードマップの単一選択の欄と、選ぶ選択肢の ID"""

    field_id: str
    option_id: str


def _option_id(fields: list[dict[str, object]], name: str, option: str) -> Choice:
    """プロジェクトの単一選択の欄と選択肢の ID

    ID は名前から毎回引く 道具に書き込むと、板で欄を作り直したときに黙って外れる
    """
    for entry in fields:
        if entry.get("name") != name:
            continue
        options = entry.get("options")
        if isinstance(options, list):
            for candidate in options:
                if isinstance(candidate, dict) and candidate.get("name") == option:
                    return Choice(str(entry["id"]), str(candidate["id"]))
        raise IssueError(f"ロードマップの欄「{name}」に「{option}」が無い")
    raise IssueError(f"ロードマップに欄「{name}」が無い")


_STATUS_QUERY = (
    "query($item: ID!) { node(id: $item) { ... on ProjectV2Item {"
    f' fieldValueByName(name: "{STATUS_FIELD}") {{'
    " ... on ProjectV2ItemFieldSingleSelectValue { name } } } } }"
)


def _current_status(item_id: str, run: Runner) -> str:
    """ロードマップの項目の今の Status（空なら空の文字）

    項目の ID で引く 板の名前で Issue の projectItems から探すと、同じ名前の別の板の
    Status を見て、こちらの板の Status を付け忘れる
    """
    answer = json.loads(
        run(["api", "graphql", "-f", f"query={_STATUS_QUERY}", "-f", f"item={item_id}"])
    )
    node = (answer.get("data") or {}).get("node") or {}
    value = node.get("fieldValueByName") or {}
    return str(value.get("name") or "")


def register(url: str, size: str, run: Runner = run_gh) -> None:
    """ロードマップに足し、Status を「未着手」、大きさの欄をラベルに合わせる

    足すのは何度やっても 1 つ（gh project item-add は既にあればその項目を返す）
    途中で止まっても、register をやり直せば同じ形になる
    既に Status が付いていれば変えない 作業中やレビュー中の物を「未着手」へ戻さないため
    板の API には「空のときだけ書く」が無いので、Status は足した後の書く直前に読む
    前もって読むと、その間に誰かが進めた Status を「未着手」で上書きしうる
    """
    if size not in SIZE_LABELS:
        raise IssueError(f"大きさ {size!r} はロードマップの欄に無い（{' '.join(SIZE_LABELS)}）")
    common = [str(PROJECT_NUMBER), "--owner", PROJECT_OWNER, "--format", "json"]
    project = json.loads(run(["project", "view", *common]))
    # 既定では先頭の 30 個しか返らない 欄が増えたときに Status や大きさが漏れて止まらないよう、
    # GitHub の上限（100）まで引く
    fields = json.loads(run(["project", "field-list", *common, "--limit", "100"]))["fields"]
    # 選択肢は足す前にすべて引く 足した後で欄が無いと分かると、Status の無い項目が板に残る
    status = _option_id(fields, STATUS_FIELD, STATUS_NEW)
    sized = _option_id(fields, SIZE_FIELD, size)
    item_id = str(json.loads(run(["project", "item-add", *common, "--url", url]))["id"])
    choices = [sized]
    if not _current_status(item_id, run):
        choices.insert(0, status)
    for choice in choices:
        run(
            [
                "project",
                "item-edit",
                "--id",
                item_id,
                "--project-id",
                str(project["id"]),
                "--field-id",
                choice.field_id,
                "--single-select-option-id",
                choice.option_id,
            ]
        )


def size_of(labels: Sequence[str]) -> str:
    """大きさのラベルを 1 つ取り出す 無いか 2 つ以上なら止める

    黙って飛ばすと、板の「大きさ」が空のまま登録が済んだことになる
    """
    found = [label for label in labels if label in SIZE_LABELS]
    if len(found) != 1:
        shown = " ".join(found) or "なし"
        raise IssueError(f"大きさのラベルはちょうど 1 つ要る（今は {shown}）")
    return found[0]


@dataclass(frozen=True, slots=True)
class Request:
    """作る Issue 1 本ぶんの指定"""

    kind: str
    title: str
    body: str
    labels: list[str]
    milestone: str
    parent: str | None = None


def create_arguments(request: Request, body_file: Path) -> list[str]:
    """gh issue create に渡す引数"""
    arguments = [
        "issue",
        "create",
        "--repo",
        REPOSITORY,
        "--title",
        request.title,
        "--body-file",
        str(body_file),
        "--milestone",
        request.milestone,
    ]
    for label in request.labels:
        # まとめて「a,b」と渡すと「互換: AviUtl」のような空白入りの名前で読み違えが起きうる
        arguments += ["--label", label]
    if request.parent:
        # --parent を知らない古い gh は、作る前に「知らない引数」で止まる（作りかけは残らない）
        arguments += ["--parent", request.parent]
    return arguments


def same_titles(title: str, run: Runner) -> list[str]:
    """同じ題の開いた Issue の URL

    作った後の応答が切れてやり直したときや、並行するエージェントが同じ物を立てたときに、
    同じ Issue が 2 本並ぶのを防ぐ 並行するエージェントが同じ時に確かめると両方とも
    通るので、これだけでは防げない（同じ物を立てる担当は 1 人に決める docs/development.md）

    検索（--search）には題を渡さない 「label:bug の表示」のような題は条件として読まれ、
    同じ題の Issue を見逃す 開いた物を最後のページまで引き、題そのもので比べる
    （件数の上限で切ると、上限を超えた所にある同じ題を「無い」と答えてしまう）
    """
    output = run(
        [
            "api",
            "--paginate",
            f"repos/{REPOSITORY}/issues?state=open&per_page=100",
            "--jq",
            ".[] | select(.pull_request | not) | {title, url: .html_url}",
        ]
    )
    found = [json.loads(line) for line in output.splitlines() if line.strip()]
    return [str(issue["url"]) for issue in found if str(issue["title"]).strip() == title]


def _retry_hint(url: str) -> str:
    """作った後の登録で止まったときの、やり直し方の案内"""
    number = url.rstrip("/").rsplit("/", 1)[-1]
    return (
        "gh に project の権限が無いなら gh auth refresh -s project を打ってから"
        "（GH_TOKEN か GITHUB_TOKEN を使っているなら、project の権限のあるトークンに替えてから）、"
        f"{COMMAND} register {number} でやり直す（create を打ち直すと 2 本になる）"
    )


def create(request: Request, run: Runner = run_gh) -> str:
    """Issue を作ってロードマップに登録し、URL を返す"""
    with tempfile.TemporaryDirectory() as folder:
        # 本文は引数ではなくファイルで渡す 長い本文と改行が Windows の引数の上限と引用で崩れない
        body_file = Path(folder) / "body.md"
        body_file.write_bytes(request.body.encode("utf-8"))
        try:
            output = run(create_arguments(request, body_file))
        except IssueError as error:
            raise IssueError(
                f"{error}\n作られたかどうかは gh issue list --repo {REPOSITORY} --author @me "
                "--state all --limit 20 で確かめる（見つからなければ --limit を増やす） "
                f"作られていれば {COMMAND} register <番号> で"
                "ロードマップに登録する（親が付いていなければ GitHub の画面で親に付ける）"
            ) from error
    url = output.strip().splitlines()[-1].strip()
    try:
        register(url, size_of(request.labels), run)
    except (IssueError, KeyError, ValueError) as error:
        raise IssueError(
            f"{url} は作ったが、ロードマップへの登録で止まった\n{error}\n{_retry_hint(url)}"
        ) from error
    return url


# --- 入り口 -------------------------------------------------------------------


def prepare(arguments: argparse.Namespace) -> tuple[Request, list[str]]:
    """引数と本文を確かめて、作る指定と知らせることを返す 問題があれば止める"""
    kind: str = arguments.kind
    template = read_text(template_path(kind))
    body_path = Path(arguments.body_file)
    body = read_text(body_path)
    title: str = arguments.title.strip()
    notes: list[str] = []

    if arguments.fix:
        fixed_title, fixed_body = fix_punctuation(title), fix_punctuation(body)
        if fixed_body != body and not arguments.dry_run:
            # 直した結果を手元のファイルにも残す 送った本文と手元が食い違わないように
            # --dry-run は何も変えない約束なので書かない リポジトリの外のファイルは、
            # 渡し間違いで別の物を書き換えないよう、送る本文だけ直して知らせる
            if _is_inside_repository(body_path):
                replace_file(body_path, fixed_body)
            else:
                notes.append(f"{body_path} はリポジトリの外なので書き換えず、送る本文だけ直した")
        title, body = fixed_title, fixed_body

    errors: list[str] = []
    if not title:
        errors.append("題が空")
    checked = check_body(body, required_sections(template))
    errors += checked.errors
    # 句点は外す前の本文で見る 行の番号が手元のファイルと合い、直す所をすぐ探せる
    errors += check_punctuation(title, body)
    labels = build_labels(
        arguments.type or KINDS[kind], arguments.area, arguments.size, arguments.state
    )
    if not arguments.milestone.strip():
        errors.append("マイルストーンが空")
    if errors:
        raise IssueError("\n".join(f"- {line}" for line in errors))
    notes += checked.notes + size_advice(kind, arguments.size, arguments.parent)
    request = Request(
        kind, title, checked.body, labels, arguments.milestone.strip(), arguments.parent
    )
    return request, notes


def _draft(arguments: argparse.Namespace) -> int:
    text = read_text(template_path(arguments.kind))
    if arguments.path is None:
        print(text, end="")
        return 0
    path = Path(arguments.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arguments.force:
        replace_file(path, text)
    else:
        # 有るかを見てから書くと、その間に別のエージェントが書き始めた下書きを消しうる
        # 無いときだけ作る開き方で 1 回にまとめる
        try:
            with path.open("xb") as stream:
                stream.write(text.encode("utf-8"))
        except FileExistsError as error:
            raise IssueError(
                f"{path} は既にある 書きかけを消さないよう止めた（--force で上書き）"
            ) from error
    print(f"{path} に雛形を書いた 埋めたら create で作る")
    return 0


def _create(arguments: argparse.Namespace, run: Runner) -> int:
    request, notes = prepare(arguments)
    for note in notes:
        print(f"お知らせ: {note}")
    if arguments.dry_run:
        print(f"題: {request.title}")
        print(f"ラベル: {' / '.join(request.labels)}")
        print(f"マイルストーン: {request.milestone}")
        if request.parent:
            print(f"親: {request.parent}")
        print(f"ロードマップ: プロジェクト {PROJECT_NUMBER} に Status「{STATUS_NEW}」で足す")
        print("---- 本文 ----")
        print(request.body, end="")
        print("---- ここまで（--dry-run なので作っていない） ----")
        return 0
    if not arguments.allow_same_title:
        same = same_titles(request.title, run)
        if same:
            raise IssueError(
                f"同じ題の開いた Issue がある: {' '.join(same)}\n"
                "やり直しなら register <番号> で登録だけを行う 別の物なら題を変えるか "
                "--allow-same-title を付ける"
            )
    url = create(request, run)
    print(f"作った: {url}")
    return 0


def _register(arguments: argparse.Namespace, run: Runner) -> int:
    issue = json.loads(
        run(
            [
                "issue",
                "view",
                str(arguments.number),
                "--repo",
                REPOSITORY,
                "--json",
                "url,labels",
            ]
        )
    )
    labels = [str(label["name"]) for label in issue["labels"]]
    register(str(issue["url"]), size_of(labels), run)
    print(f"ロードマップに登録した: {issue['url']}")
    return 0


def parser() -> argparse.ArgumentParser:
    """draft・create・register の 3 つの命令の引数"""
    root = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)

    draft = commands.add_parser("draft", help="雛形を書き出す")
    draft.add_argument("kind", choices=list(KINDS))
    draft.add_argument("path", nargs="?", help="書き出す先 省くと画面に出す")
    draft.add_argument("--force", action="store_true", help="既にあるファイルを上書きする")

    make = commands.add_parser("create", help="本文を確かめて Issue を作る")
    make.add_argument("kind", choices=list(KINDS))
    make.add_argument("--title", required=True)
    make.add_argument("--body-file", required=True, help="雛形を埋めた本文のファイル")
    make.add_argument("--type", choices=TYPE_LABELS, help="種類のラベル 省くと雛形の既定")
    # 分野と大きさは省けない 省けると、急いでいる人や AI が付け忘れたまま作る
    make.add_argument("--area", required=True, choices=[*AREA_LABELS, NO_AREA])
    make.add_argument("--size", required=True, choices=SIZE_LABELS)
    make.add_argument("--state", action="append", default=[], choices=STATE_LABELS)
    make.add_argument("--milestone", required=True, help="マイルストーンの名前（例 0.2.6）")
    make.add_argument("--parent", help="親の Issue の番号 子として作る")
    make.add_argument("--fix", action="store_true", help="題と本文の句点を直してから作る")
    make.add_argument(
        "--allow-same-title", action="store_true", help="同じ題の開いた Issue があっても作る"
    )
    make.add_argument("--dry-run", action="store_true", help="作らずに仕上がりを出す")

    again = commands.add_parser("register", help="作った Issue をロードマップに登録し直す")
    again.add_argument("number", help="Issue の番号か URL")
    return root


def main(argv: Sequence[str] | None = None, run: Runner = run_gh) -> int:
    """入り口 止めたときは訳を標準エラーに出して 1 を返す"""
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "draft":
            return _draft(arguments)
        if arguments.command == "create":
            return _create(arguments, run)
        return _register(arguments, run)
    except IssueError as error:
        print(f"止めた:\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
