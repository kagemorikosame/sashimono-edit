r"""開発者向けの Issue を、そろった形で作る（#376）

    # 1. 雛形を書き出して埋める（種類は feature・bug・investigation）
    .venv\Scripts\python.exe tools\new_issue.py draft feature .work\issue.md

    # 2. 送らずに確かめる（本文の仕上がりと、走らせる gh の命令を出す）
    .venv\Scripts\python.exe tools\new_issue.py create feature --title "…" ^
        --area 編集の土台 --size 中 --milestone 0.2.6 --body-file .work\issue.md --dry-run

    # 3. 作る（--dry-run を外す）
    # 作ったあとのロードマップへの登録だけをやり直す
    .venv\Scripts\python.exe tools\new_issue.py register 450

作るときに一度に行うこと

- 雛形の「必須」の節がそろって埋まっているかを見る 空の任意の節は外して知らせる
- 題と本文の句点を見る（``--fix`` で ``tools/punctuation.py`` と同じ決まりで直す）
- ラベルを 4 種類（種類・分野・大きさ・状態）から付ける 分野と大きさは必ず決める
- マイルストーンを付ける
- 公開のロードマップ（プロジェクト 3）に足し、Status を「未着手」、大きさの欄を合わせる

雛形は ``docs/issue-templates/`` に置く ``.github/ISSUE_TEMPLATE/`` に置くと、使う人が
Discussions を通らずに Issue を作れてしまう（docs/development.md の「報告の受け口」）
"""

from __future__ import annotations

import argparse
import io
import json
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

#: 本文の頭（最初の見出しより前） 一行の目的を書く所で、見出しを持たない
LEAD = "（冒頭）"

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REQUIRED = re.compile(r"<!--\s*必須")
_FENCE = re.compile(r"^\s*(```|~~~)")


class IssueError(Exception):
    """ここで止める Issue はまだ作っていない（または作った後の登録で止まった）"""


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
    lines: list[str] = []
    in_fence = False
    for line in body.splitlines(keepends=True):
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line.rstrip("\r\n"))
        if match is None:
            lines.append(line)
            continue
        sections.append(Section(title, level, heading, "".join(lines)))
        title, level, heading = match.group(2), len(match.group(1)), line
        lines = []
    sections.append(Section(title, level, heading, "".join(lines)))
    return sections


def required_sections(template: str) -> list[str]:
    """雛形で、注意書きが「必須」で始まる節の名前

    冒頭には雛形全体の注意書きが先に来るので、頭の 1 つだけでなく、節の中のどれかで見る
    """
    return [s.title for s in split_sections(template) if _REQUIRED.search(s.text)]


def strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


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
    """注意書きを外した跡の空行を詰める 3 行以上空くと、GitHub の表示で節の間が不揃いになる"""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    for line in lines:
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def check_punctuation(title: str, body: str) -> list[str]:
    """句点の場所 Issue はリポジトリの外で、tools/punctuation.py の検査が届かない"""
    found = [f"題の {column} 文字目" for _, column in punctuation.find_in_text(title)]
    found += [
        f"本文の {line} 行 {column} 文字目" for line, column in punctuation.find_in_text(body)
    ]
    return [f"句点がある（{where}）" for where in found]


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
    if kind not in KINDS:
        raise IssueError(f"雛形の種類 {kind!r} は無い（{' '.join(KINDS)}）")
    return TEMPLATES / f"{kind}.md"


def read_text(path: Path) -> str:
    """BOM を読み飛ばす PowerShell の Out-File は既定で BOM を付けて書く"""
    return path.read_bytes().decode("utf-8-sig")


# --- gh ---------------------------------------------------------------------

Runner = Callable[[Sequence[str]], str]


def run_gh(arguments: Sequence[str]) -> str:
    """gh を走らせて標準出力を返す 失敗したら gh の言い分ごと止める

    渡すのはこの道具が組んだ引数だけ shell は通さない
    """
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
        )
    )
    if completed.returncode != 0:
        raise IssueError(f"gh {arguments[0]} {arguments[1]} が失敗した\n{completed.stderr.strip()}")
    return completed.stdout


def _option_id(fields: list[dict[str, object]], name: str, option: str) -> tuple[str, str]:
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
                    return str(entry["id"]), str(candidate["id"])
        raise IssueError(f"ロードマップの欄「{name}」に「{option}」が無い")
    raise IssueError(f"ロードマップに欄「{name}」が無い")


def register(url: str, size: str | None, run: Runner = run_gh) -> None:
    """ロードマップに足し、Status を「未着手」、大きさの欄をラベルに合わせる

    足すのは何度やっても 1 つ（gh project item-add は既にあればその項目を返す）
    途中で止まっても、register をやり直せば同じ形になる
    """
    common = [str(PROJECT_NUMBER), "--owner", PROJECT_OWNER, "--format", "json"]
    project_id = str(json.loads(run(["project", "view", *common]))["id"])
    fields = json.loads(run(["project", "field-list", *common]))["fields"]
    item_id = str(json.loads(run(["project", "item-add", *common, "--url", url]))["id"])
    choices = [(STATUS_FIELD, STATUS_NEW)]
    if size is not None:
        choices.append((SIZE_FIELD, size))
    for name, option in choices:
        field_id, option_id = _option_id(fields, name, option)
        run(
            [
                "project",
                "item-edit",
                "--id",
                item_id,
                "--project-id",
                project_id,
                "--field-id",
                field_id,
                "--single-select-option-id",
                option_id,
            ]
        )


def size_of(labels: Sequence[str]) -> str | None:
    found = [label for label in labels if label in SIZE_LABELS]
    return found[0] if len(found) == 1 else None


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
    arguments = [
        "issue",
        "create",
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
        arguments += ["--parent", request.parent]
    return arguments


def create(request: Request, run: Runner = run_gh) -> str:
    """Issue を作ってロードマップに登録し、URL を返す"""
    with tempfile.TemporaryDirectory() as folder:
        # 本文は引数ではなくファイルで渡す 長い本文と改行が Windows の引数の上限と引用で崩れない
        body_file = Path(folder) / "body.md"
        body_file.write_bytes(request.body.encode("utf-8"))
        output = run(create_arguments(request, body_file))
    url = output.strip().splitlines()[-1].strip()
    try:
        register(url, size_of(request.labels), run)
    except (IssueError, KeyError, ValueError) as error:
        number = url.rstrip("/").rsplit("/", 1)[-1]
        raise IssueError(
            f"{url} は作ったが、ロードマップへの登録で止まった\n{error}\n"
            "gh に project の権限が無いなら gh auth refresh -s project を打ってから、"
            f"tools\\new_issue.py register {number} でやり直す"
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

    if arguments.fix:
        fixed_title, fixed_body = punctuation.fix_text(title), punctuation.fix_text(body)
        if fixed_body != body and not arguments.dry_run:
            # 直した結果を利用者のファイルにも残す 送った本文と手元が食い違わないように
            # --dry-run は何も変えない約束なので書かない
            body_path.write_bytes(fixed_body.encode("utf-8"))
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
    notes = checked.notes + size_advice(kind, arguments.size, arguments.parent)
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
    if path.exists() and not arguments.force:
        raise IssueError(f"{path} は既にある 書きかけを消さないよう止めた（--force で上書き）")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
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
    url = create(request, run)
    print(f"作った: {url}")
    return 0


def _register(arguments: argparse.Namespace, run: Runner) -> int:
    issue = json.loads(run(["issue", "view", str(arguments.number), "--json", "url,labels"]))
    labels = [str(label["name"]) for label in issue["labels"]]
    register(str(issue["url"]), size_of(labels), run)
    print(f"ロードマップに登録した: {issue['url']}")
    return 0


def parser() -> argparse.ArgumentParser:
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
    make.add_argument("--dry-run", action="store_true", help="作らずに仕上がりを出す")

    again = commands.add_parser("register", help="作った Issue をロードマップに登録し直す")
    again.add_argument("number", help="Issue の番号か URL")
    return root


def main(argv: Sequence[str] | None = None, run: Runner = run_gh) -> int:
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
