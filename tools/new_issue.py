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

#: gh 1 回を待つ上限（秒） 待ち続けると、止まったのか遅いのかが分からないまま固まる
GH_TIMEOUT = 120

#: 本文の頭（最初の見出しより前） 一行の目的を書く所で、見出しを持たない
LEAD = "（冒頭）"

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REQUIRED = re.compile(r"<!--\s*必須")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")


class IssueError(Exception):
    """ここで止める Issue はまだ作っていない（または作った後の登録で止まった）"""


# --- 本文の読み分け -------------------------------------------------------------


def fenced_lines(lines: Sequence[str]) -> list[bool]:
    """行ごとに、コードの囲み（区切りの行を含む）の中か

    閉じは開きと同じ文字で、開き以上の長さの行だけ ``````` の中に書いた ```` ``` ```` の
    例を閉じと取ると、その後ろのコードの見出しや句点を本文として扱ってしまう
    閉じが無ければ最後まで囲みの中（CommonMark と同じ）
    """
    flags: list[bool] = []
    opener: str | None = None
    for line in lines:
        match = _FENCE.match(line.rstrip("\r\n"))
        if opener is None:
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                opener = match.group(1)
                flags.append(True)
                continue
            flags.append(False)
            continue
        flags.append(True)
        if (
            match
            and match.group(1)[0] == opener[0]
            and len(match.group(1)) >= len(opener)
            and not match.group(2).strip()
        ):
            opener = None
    return flags


def code_spans(text: str) -> list[tuple[int, int]]:
    """コードの囲みとインラインのコードが占める文字の範囲

    ここは貼ったログや再現用の文字列で、データとして元のまま送る
    句点を消したり ``<!-- -->`` を注意書きとして外したりすると、再現できない報告になる
    """
    lines = text.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    offset = 0
    for line, fenced in zip(lines, fenced_lines(lines), strict=True):
        if fenced:
            spans.append((offset, offset + len(line)))
        else:
            for match in _INLINE_CODE.finditer(line):
                spans.append((offset + match.start(), offset + match.end()))
        offset += len(line)
    return spans


def _inside(index: int, spans: Sequence[tuple[int, int]]) -> bool:
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
    """見出しで分ける コードの囲みの中の ``#`` は見出しに数えない

    数えると、貼ったログやシェルのコメントで節が割れ、中身のある節を空と見誤る
    """
    sections: list[Section] = []
    title, level, heading = LEAD, 0, ""
    collected: list[str] = []
    lines = body.splitlines(keepends=True)
    for line, fenced in zip(lines, fenced_lines(lines), strict=True):
        match = None if fenced else _HEADING.match(line.rstrip("\r\n"))
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


def strip_comments(text: str) -> str:
    """雛形の注意書き（``<!-- -->``）を外す コードの中に貼ったコメントは残す"""
    spans = code_spans(text)
    pieces: list[str] = []
    last = 0
    for match in _COMMENT.finditer(text):
        if _inside(match.start(), spans):
            continue
        pieces.append(text[last : match.start()])
        last = match.end()
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
    for line, fenced in zip(lines, fenced_lines(lines), strict=True):
        if fenced:
            out.append(line)
            continue
        line = line.rstrip()
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    return "\n".join(out).strip("\n") + "\n"


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


def _current_status(url: str, project_title: str, run: Runner) -> str:
    """その Issue のロードマップでの今の Status（載っていなければ空）"""
    issue = json.loads(run(["issue", "view", url, "--repo", REPOSITORY, "--json", "projectItems"]))
    for item in issue.get("projectItems") or []:
        if item.get("title") == project_title:
            status = item.get("status") or {}
            return str(status.get("name") or "")
    return ""


def register(url: str, size: str, run: Runner = run_gh) -> None:
    """ロードマップに足し、Status を「未着手」、大きさの欄をラベルに合わせる

    足すのは何度やっても 1 つ（gh project item-add は既にあればその項目を返す）
    途中で止まっても、register をやり直せば同じ形になる
    既に Status が付いていれば変えない 作業中やレビュー中の物を「未着手」へ戻さないため
    """
    if size not in SIZE_LABELS:
        raise IssueError(f"大きさ {size!r} はロードマップの欄に無い（{' '.join(SIZE_LABELS)}）")
    common = [str(PROJECT_NUMBER), "--owner", PROJECT_OWNER, "--format", "json"]
    project = json.loads(run(["project", "view", *common]))
    fields = json.loads(run(["project", "field-list", *common]))["fields"]
    # 選択肢は足す前にすべて引く 足した後で欄が無いと分かると、Status の無い項目が板に残る
    status = _option_id(fields, STATUS_FIELD, STATUS_NEW)
    sized = _option_id(fields, SIZE_FIELD, size)
    choices = [sized]
    if not _current_status(url, str(project.get("title", "")), run):
        choices.insert(0, status)
    item_id = str(json.loads(run(["project", "item-add", *common, "--url", url]))["id"])
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
    同じ Issue が 2 本並ぶのを防ぐ 検索は語で当たるので、題がそのまま同じ物だけを拾う
    """
    found = json.loads(
        run(
            [
                "issue",
                "list",
                "--repo",
                REPOSITORY,
                "--state",
                "open",
                "--search",
                f"{title} in:title",
                "--json",
                "title,url",
                "--limit",
                "100",
            ]
        )
    )
    return [str(issue["url"]) for issue in found if str(issue["title"]).strip() == title]


def _retry_hint(url: str) -> str:
    number = url.rstrip("/").rsplit("/", 1)[-1]
    return (
        "gh に project の権限が無いなら gh auth refresh -s project を打ってから、"
        f"tools\\new_issue.py register {number} でやり直す（create を打ち直すと 2 本になる）"
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
                f"{error}\n作られたかどうかは gh issue list --repo {REPOSITORY} --search "
                f'"{request.title} in:title" で確かめる 作られていれば register <番号> で'
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
