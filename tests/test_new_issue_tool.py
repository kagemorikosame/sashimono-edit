"""開発者向けの Issue を作る道具（tools/new_issue.py #376）

この道具が抜けを見逃すと、ラベル・マイルストーン・ロードマップへの登録が抜けた Issue や、
句点の混ざった Issue がまた増える 逆に埋めた本文を止めると、誰も道具で Issue を作らなくなる
その境目と、gh へ渡す命令の形をここで固定する gh は偽物に差し替え、本物の Issue は作らない
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
MARU = chr(0x3002)


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("new_issue", ROOT / "tools" / "new_issue.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass は定義の途中で自分のモジュールを sys.modules から引く
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROJECT = {"id": "P-3", "title": "Sashimono Edit ロードマップ"}
FIELDS = {
    "fields": [
        {"id": "F-title", "name": "Title"},
        {
            "id": "F-status",
            "name": "Status",
            "options": [{"id": "S-new", "name": "未着手"}, {"id": "S-doing", "name": "作業中"}],
        },
        {
            "id": "F-size",
            "name": "大きさ",
            "options": [{"id": "Z-l", "name": "大"}, {"id": "Z-m", "name": "中"}],
        },
    ]
}
URL = "https://github.com/kagemorikosame/sashimono-edit/issues/999"

#: gh issue create が受け付ける引数（gh 2.98 の gh issue create --help で確かめた物のうち
#: 道具が使う物） 知らない引数を通すと、本物の gh では作る前に落ちる組み立てを試験が見逃す
CREATE_FLAGS = {"--repo", "--title", "--body-file", "--milestone", "--label", "--parent"}


class FakeGh:
    """gh の代わり 受けた命令を控え、決めた答えを返す"""

    def __init__(
        self,
        status: str = "",
        labels: Sequence[str] = ("bug", "中"),
        same_title: Sequence[str] = (),
    ) -> None:
        self.calls: list[list[str]] = []
        self.bodies: list[str] = []
        self.status = status
        self.labels = list(labels)
        self.same_title = list(same_title)

    def __call__(self, arguments: Sequence[str]) -> str:
        arguments = list(arguments)
        self.calls.append(arguments)
        key = " ".join(arguments[:2])
        if key == "issue create":
            flags = {value for value in arguments[2:] if value.startswith("--")}
            if not flags <= CREATE_FLAGS:
                raise AssertionError(f"gh issue create が知らない引数: {flags - CREATE_FLAGS}")
            body_file = Path(arguments[arguments.index("--body-file") + 1])
            self.bodies.append(body_file.read_bytes().decode("utf-8"))
            return f"Creating issue\n{URL}\n"
        if key == "issue list":
            return json.dumps([{"title": title, "url": URL} for title in self.same_title])
        if key == "project view":
            return json.dumps(PROJECT)
        if key == "project field-list":
            return json.dumps(FIELDS)
        if key == "project item-edit":
            return ""
        if key == "project item-add":
            return json.dumps({"id": "I-1"})
        if key == "issue view" and arguments[-1] == "projectItems":
            items = [{"title": PROJECT["title"], "status": {"name": self.status}}]
            return json.dumps({"projectItems": items if self.status else []})
        if key == "issue view":
            return json.dumps({"url": URL, "labels": [{"name": name} for name in self.labels]})
        raise AssertionError(f"思っていない gh の命令: {arguments}")


def edits_of(gh: FakeGh) -> set[tuple[str, str]]:
    """ロードマップで選んだ欄と選択肢"""
    return {
        (call[call.index("--field-id") + 1], call[call.index("--single-select-option-id") + 1])
        for call in gh.calls
        if call[:2] == ["project", "item-edit"]
    }


def filled_bug(lead: str = "書き出しで落ちる") -> str:
    return (
        f"{lead}\n\n## 症状\n\n<!-- 必須 -->\n落ちる\n\n## 再現の手順\n\n1. 開く\n\n"
        "## 原因\n\n<!-- 確かめた物と -->\n\n## 確かめること\n\n試験を足す\n"
    )


def run_main(tool: ModuleType, gh: FakeGh, *arguments: str) -> int:
    result: int = tool.main(list(arguments), run=gh)
    return result


def create_bug(body: Path, *extra: str) -> list[str]:
    return [
        "create",
        "bug",
        "--title",
        "落ちる",
        "--area",
        "なし",
        "--size",
        "中",
        "--milestone",
        "0.2.6",
        "--body-file",
        str(body),
        *extra,
    ]


@pytest.mark.parametrize("kind", ["feature", "bug", "investigation"])
def test_each_template_has_required_sections(tool: ModuleType, kind: str) -> None:
    # 必須の印が読めないと、空の雛形のまま Issue が作れてしまう
    required = tool.required_sections(tool.read_text(tool.template_path(kind)))
    assert tool.LEAD in required
    assert len(required) >= 3


def test_feature_requires_preferences_section(tool: ModuleType) -> None:
    # 設定画面に出すかを書き落とすと、好みが分かれる所が決め打ちのまま実装される
    required = tool.required_sections(tool.read_text(tool.template_path("feature")))
    assert "設定画面に出す項目" in required


@pytest.mark.parametrize("kind", ["feature", "bug", "investigation"])
def test_untouched_template_is_refused(tool: ModuleType, kind: str) -> None:
    # 雛形をそのまま送ると、見出しと注意書きだけの Issue が板に並ぶ
    template = tool.read_text(tool.template_path(kind))
    checked = tool.check_body(template, tool.required_sections(template))
    assert len(checked.errors) == len(tool.required_sections(template))


def test_empty_optional_sections_and_notes_are_removed(tool: ModuleType) -> None:
    # 空の節や注意書きが残ると、読む人が「書き忘れか、無いのか」を見分けられない
    template = tool.read_text(tool.template_path("bug"))
    checked = tool.check_body(filled_bug(), tool.required_sections(template))
    assert checked.errors == []
    assert "## 原因" not in checked.body
    assert "<!--" not in checked.body
    assert checked.notes == ["空の節「原因」を外した"]
    assert checked.body.startswith("書き出しで落ちる\n\n## 症状\n\n落ちる\n")


def test_missing_required_heading_is_reported(tool: ModuleType) -> None:
    # 必須の節を消して送ると、再現の手順の無い不具合が起きる
    body = filled_bug().replace("## 再現の手順\n\n1. 開く\n\n", "")
    checked = tool.check_body(body, ["症状", "再現の手順"])
    assert checked.errors == ["必須の節「再現の手順」が無い"]


def test_empty_lead_is_reported(tool: ModuleType) -> None:
    # 一行の目的が無いと、一覧で題だけ見て何の Issue か分からない
    checked = tool.check_body(filled_bug(lead="<!-- 必須 -->"), [tool.LEAD])
    assert checked.errors == ["必須の一行の目的が空"]


def test_heading_with_filled_children_is_kept(tool: ModuleType) -> None:
    # 段階を下の見出しに分けて書いた「進め方」を空と見て外すと、中身ごと消える
    body = "目的\n\n## 進め方\n\n### 段階 1\n\n読む\n\n## 決めること\n\n"
    checked = tool.check_body(body, [])
    assert "## 進め方\n\n### 段階 1\n\n読む\n" in checked.body
    assert checked.notes == ["空の節「決めること」を外した"]


def test_hash_inside_code_fence_is_not_a_heading(tool: ModuleType) -> None:
    # 貼ったログの # を見出しと取ると、中身のある節を空と見誤って外す
    body = "目的\n\n## 症状\n\n```\n# 落ちた\n```\n"
    checked = tool.check_body(body, ["症状"])
    assert checked.errors == []
    assert "# 落ちた" in checked.body


def test_shorter_fence_inside_does_not_close(tool: ModuleType) -> None:
    # 長い囲みの中の短い ``` で閉じたと見ると、コードの中の見出しで節が割れる
    lines = ["````\n", "```\n", "## 中\n", "````\n", "## 外\n"]
    assert tool.fenced_lines(lines) == [True, True, True, True, False]
    assert [s.title for s in tool.split_sections("".join(lines))] == [tool.LEAD, "外"]


def test_code_is_kept_as_data(tool: ModuleType) -> None:
    # 字幕整形の不具合を再現する文やログから句点やコメントを消すと、再現できない報告になる
    body = f"目的 `a{MARU}b`\n\n## 症状\n\n```\n<!-- 貼った -->\n\n\n字幕{MARU}  \n```\n"
    checked = tool.check_body(body, ["症状"])
    assert checked.errors == []
    assert "<!-- 貼った -->\n\n\n" in checked.body
    assert f"字幕{MARU}  \n" in checked.body
    assert tool.check_punctuation("題", body) == []
    assert tool.fix_punctuation(body) == body
    assert tool.fix_punctuation(f"外{MARU}`中{MARU}`") == f"外 `中{MARU}`"


def test_punctuation_in_title_and_body_is_found(tool: ModuleType) -> None:
    # Issue はリポジトリの外で、句点の検査が届かない ここで拾わないと混ざったまま残る
    found = tool.check_punctuation(f"題{MARU}", f"一行目\v続き\n二行目{MARU}次\n")
    assert found == ["句点がある（題の 2 文字目）", "句点がある（本文の 2 行 4 文字目）"]


def test_fix_text_matches_the_file_fixer(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Issue とリポジトリの中で直り方が違うと、同じ文が場所によって違う形になる
    text = f"直す{MARU}次の文{MARU}\n**太字{MARU}**の後{MARU}`code`\n"
    path = tmp_path / "a.md"
    path.write_bytes(text.encode("utf-8"))
    punctuation = sys.modules["punctuation"]
    # fix はリポジトリの外を書き換えない 試験の置き場をリポジトリと見なさせる
    monkeypatch.setattr(punctuation, "ROOT", tmp_path)
    punctuation.fix(path, punctuation.scan_text(path))
    assert punctuation.fix_text(text) == path.read_bytes().decode("utf-8")
    assert MARU not in punctuation.fix_text(text)


def test_labels_are_validated(tool: ModuleType) -> None:
    # 名前を打ち間違えたラベルは gh に渡ると作る直前まで分からない 先に止める
    assert tool.build_labels("bug", "互換: YMM4", "小", ["材料待ち", "材料待ち"]) == [
        "bug",
        "互換: YMM4",
        "小",
        "材料待ち",
    ]
    assert tool.build_labels("enhancement", tool.NO_AREA, "中", []) == ["enhancement", "中"]
    with pytest.raises(tool.IssueError):
        tool.build_labels("bug", "互換:YMM4", "小", [])
    with pytest.raises(tool.IssueError):
        tool.build_labels("bug", "配る準備", "特大", [])


def test_large_issue_suggests_parent_and_children(tool: ModuleType) -> None:
    # 「大」を 1 本にまとめると着手の単位が分からない 親と子に分ける提案を出す
    advice = tool.size_advice("feature", "大", None)
    assert len(advice) == 1
    assert "investigation" in advice[0]
    assert "--parent" in advice[0]
    assert tool.size_advice("investigation", "大", None) == []
    assert tool.size_advice("feature", "大", "376") == []
    assert tool.size_advice("feature", "中", None) == []


def test_create_sends_labels_milestone_and_registers(tool: ModuleType, tmp_path: Path) -> None:
    # 登録が抜けると、ロードマップに載らない Issue が誰にも見つからないまま残る
    body = tmp_path / "issue.md"
    body.write_bytes(("﻿" + filled_bug()).encode("utf-8"))
    gh = FakeGh()
    code = run_main(
        tool,
        gh,
        "create",
        "bug",
        "--title",
        "書き出しで落ちる",
        "--area",
        "互換: AviUtl",
        "--size",
        "大",
        "--state",
        "本人の手が要る",
        "--milestone",
        "0.2.6",
        "--parent",
        "376",
        "--body-file",
        str(body),
    )
    assert code == 0
    assert gh.calls[0][:2] == ["issue", "list"]
    create = gh.calls[1]
    assert create[:2] == ["issue", "create"]
    # fork を既定にした gh でも本家に作る
    assert create[create.index("--repo") + 1] == tool.REPOSITORY
    labels = [create[i + 1] for i, value in enumerate(create) if value == "--label"]
    assert labels == ["bug", "互換: AviUtl", "大", "本人の手が要る"]
    assert create[create.index("--milestone") + 1] == "0.2.6"
    assert create[create.index("--parent") + 1] == "376"
    # BOM と注意書きは送らない 送ると本文の頭に見えない文字が残る
    assert gh.bodies[0].startswith("書き出しで落ちる\n")
    assert "<!--" not in gh.bodies[0]
    assert edits_of(gh) == {("F-status", "S-new"), ("F-size", "Z-l")}
    edits = [call for call in gh.calls if call[:2] == ["project", "item-edit"]]
    assert all(call[call.index("--id") + 1] == "I-1" for call in edits)
    add = next(call for call in gh.calls if call[:2] == ["project", "item-add"])
    assert add[add.index("--url") + 1] == URL


def test_same_title_stops_a_second_issue(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 応答が切れてやり直したときや並行するエージェントが立てたとき、同じ Issue が 2 本並ぶ
    body = tmp_path / "issue.md"
    body.write_bytes(filled_bug().encode("utf-8"))
    gh = FakeGh(same_title=["落ちる"])
    assert run_main(tool, gh, *create_bug(body)) == 1
    assert [call[:2] for call in gh.calls] == [["issue", "list"]]
    assert "register" in capsys.readouterr().err

    # 語が当たっただけの別の題は止めない
    gh = FakeGh(same_title=["落ちる 書き出しで"])
    assert run_main(tool, gh, *create_bug(body)) == 0

    gh = FakeGh(same_title=["落ちる"])
    assert run_main(tool, gh, *create_bug(body, "--allow-same-title")) == 0
    assert gh.calls[0][:2] == ["issue", "create"]


def test_refused_body_creates_nothing(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 止めたはずの本文で Issue が作られると、句点や空の節が直らないまま公開される
    body = tmp_path / "issue.md"
    body.write_bytes(filled_bug(lead=f"落ちる{MARU}").encode("utf-8"))
    gh = FakeGh()
    assert run_main(tool, gh, *create_bug(body)) == 1
    assert gh.calls == []
    assert "句点がある（本文の 1 行 4 文字目）" in capsys.readouterr().err

    # --dry-run は --fix でも手元のファイルを変えない
    assert run_main(tool, gh, *create_bug(body, "--fix", "--dry-run")) == 0
    assert gh.calls == []
    assert MARU in body.read_bytes().decode("utf-8")


def test_fix_writes_back_only_inside_repository(
    tool: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 渡し間違えたリポジトリの外のファイルを書き換えると、別の物を壊す
    # 中のファイルは直して残さないと、送った本文と手元が食い違う
    inside = tmp_path / "repo"
    inside.mkdir()
    body = inside / "issue.md"
    body.write_bytes(filled_bug(lead=f"落ちる{MARU}").encode("utf-8"))
    outside = tmp_path / "outside.md"
    outside.write_bytes(body.read_bytes())

    gh = FakeGh()
    assert run_main(tool, gh, *create_bug(outside, "--fix")) == 0
    assert MARU in outside.read_bytes().decode("utf-8")
    assert MARU not in gh.bodies[0]
    assert "書き換えず" in capsys.readouterr().out

    monkeypatch.setattr(tool, "ROOT", inside)
    gh = FakeGh()
    assert run_main(tool, gh, *create_bug(body, "--fix")) == 0
    assert MARU not in body.read_bytes().decode("utf-8")
    assert sorted(path.name for path in inside.iterdir()) == ["issue.md"]


def test_failed_registration_tells_how_to_retry(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 作った後に止まったことが伝わらないと、板に載らないまま忘れられる
    body = tmp_path / "issue.md"
    body.write_bytes(filled_bug().encode("utf-8"))
    gh = FakeGh()

    def failing(arguments: Sequence[str]) -> str:
        if list(arguments[:2]) == ["project", "item-add"]:
            raise tool.IssueError("project の権限が無い")
        return gh(arguments)

    assert tool.main(create_bug(body), run=failing) == 1
    error = capsys.readouterr().err
    assert URL in error
    assert "new_issue.py register 999" in error


def test_failed_create_tells_how_to_check(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 応答だけが切れたときに打ち直すと 2 本になる 確かめ方と登録のやり直し方を出す
    body = tmp_path / "issue.md"
    body.write_bytes(filled_bug().encode("utf-8"))
    gh = FakeGh()

    def failing(arguments: Sequence[str]) -> str:
        if list(arguments[:2]) == ["issue", "create"]:
            raise tool.IssueError("gh issue create が 120 秒で返らなかった")
        return gh(arguments)

    assert tool.main(create_bug(body), run=failing) == 1
    error = capsys.readouterr().err
    assert "gh issue list" in error
    assert "register" in error


def test_gh_timeout_becomes_issue_error(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    # gh が返らないまま待ち続けると、止まったのか遅いのか分からず固まる
    def hang(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired("gh", tool.GH_TIMEOUT)

    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(tool.IssueError, match="受け付けられたかもしれない"):
        tool.run_gh(["issue", "create"])


def test_register_reads_size_from_issue_labels(tool: ModuleType) -> None:
    # やり直しで大きさの欄が付かないと、板の「大きさ」で並べたときに抜ける
    gh = FakeGh()
    assert run_main(tool, gh, "register", "999") == 0
    assert edits_of(gh) == {("F-status", "S-new"), ("F-size", "Z-m")}


@pytest.mark.parametrize("labels", [("bug",), ("bug", "小", "中")])
def test_register_needs_exactly_one_size(
    tool: ModuleType, labels: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    # 大きさの分からない Issue を登録できたことにすると、板の「大きさ」が空のまま残る
    gh = FakeGh(labels=labels)
    assert run_main(tool, gh, "register", "999") == 1
    assert not [call for call in gh.calls if call[:2] == ["project", "item-add"]]
    assert "大きさのラベル" in capsys.readouterr().err


def test_register_keeps_progressed_status(tool: ModuleType) -> None:
    # やり直しで作業中の物を「未着手」へ戻すと、板の進み具合が嘘になる
    gh = FakeGh(status="作業中")
    assert run_main(tool, gh, "register", "999") == 0
    assert edits_of(gh) == {("F-size", "Z-m")}


def test_unknown_project_option_stops_before_adding(tool: ModuleType) -> None:
    # 板の選択肢の名前が変わったのに足してから止まると、Status の無い項目が板に残る
    gh = FakeGh()
    with pytest.raises(tool.IssueError, match="小"):
        tool.register(URL, "小", run=gh)
    assert not [call for call in gh.calls if call[:2] == ["project", "item-add"]]


def test_draft_does_not_overwrite(tool: ModuleType, tmp_path: Path) -> None:
    # 書きかけの本文を雛形で上書きすると、書いた分が消える
    path = tmp_path / "draft.md"
    assert run_main(tool, FakeGh(), "draft", "feature", str(path)) == 0
    path.write_text("書きかけ", encoding="utf-8")
    assert run_main(tool, FakeGh(), "draft", "feature", str(path)) == 1
    assert path.read_text(encoding="utf-8") == "書きかけ"
    assert run_main(tool, FakeGh(), "draft", "feature", str(path), "--force") == 0
    assert path.read_text(encoding="utf-8").startswith("<!--")
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft.md"]


def test_replace_file_keeps_draft_when_writing_fails(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 書いている途中で落ちて下書きが空になると、埋めた本文がまるごと消える
    path = tmp_path / "draft.md"
    path.write_text("書きかけ", encoding="utf-8")

    def broken(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", broken)
    with pytest.raises(OSError):
        tool.replace_file(path, "新しい中身")
    assert path.read_text(encoding="utf-8") == "書きかけ"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft.md"]


def test_issue_forms_stay_out_of_github_folder() -> None:
    # 雛形を .github/ISSUE_TEMPLATE に置くと、使う人が Discussions を通らずに Issue を作れる
    folder = ROOT / ".github" / "ISSUE_TEMPLATE"
    assert sorted(path.name for path in folder.iterdir()) == ["config.yml"]
    assert "blank_issues_enabled: false" in (folder / "config.yml").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["CLAUDE.md", "AGENTS.md", ".cursor/rules/sashimono.mdc"])
def test_agent_rules_point_to_the_tool(name: str) -> None:
    # AI 向けのルールの 1 つにだけ書くと、その道具を読まないエージェントが手で作って抜ける
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "tools\\new_issue.py" in text
    assert "docs/issue-templates/" in text
