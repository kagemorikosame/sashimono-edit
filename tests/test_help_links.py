"""案内する先と、Issue の雛形が、中身と食い違っていないか

どちらも文字で書いた案内で、名前や置き場を変えたときに一緒に直さないと
嘘の案内が残る 残っても編集は続けられるので、報告が届かなくなるまで気付かない
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from sashimono.core import userdirs
from sashimono.core.userdirs import APP_FOLDER
from sashimono.links import DISCUSSION_CATEGORIES, MANUAL_URL, REPORT_URL, REPOSITORY_URL

ROOT = Path(__file__).resolve().parent.parent
ISSUES = ROOT / ".github" / "ISSUE_TEMPLATE"
TEMPLATES = ROOT / ".github" / "DISCUSSION_TEMPLATE"


def _headings(markdown: str) -> set[str]:
    return {match.group(1).strip() for match in re.finditer(r"^#+\s+(.+)$", markdown, re.M)}


class TestLinks:
    def test_the_manual_is_the_wiki_home(self) -> None:
        # 〔ヘルプ〕→〔使い方〕（F1）は Wiki のホームを開く ホームの目次と横の目次から
        # 手順書へたどれる 壊れると、README の 1 節だけを見せて手順書へ行けない
        parts = urlsplit(MANUAL_URL)
        assert f"{REPOSITORY_URL}/wiki" == MANUAL_URL
        assert not parts.fragment

    def test_the_manual_points_at_a_heading_that_exists(self) -> None:
        # 節を指すように戻したときの備え 節の名前を変えると、GitHub は README の頭を
        # 出すだけで、何も言わずに案内先を失う
        parts = urlsplit(MANUAL_URL)
        if parts.fragment:
            readme = (ROOT / "README.md").read_text(encoding="utf-8")
            assert parts.fragment in _headings(readme)

    def test_every_link_stays_in_the_repository(self) -> None:
        # 壊れると、リポジトリを移したときに 1 つだけ古い先へ飛ばし続ける
        for url in (MANUAL_URL, REPORT_URL):
            assert url.startswith(REPOSITORY_URL)


class TestReportTemplates:
    def test_the_contact_links_point_into_the_repository(self) -> None:
        # 壊れると、雛形を選ぶ画面の「質問はこちら」が別のリポジトリへ飛ぶ
        config = (ISSUES / "config.yml").read_text(encoding="utf-8")
        urls = re.findall(r"^\s*url:\s*(\S+)", config, re.M)
        assert urls
        assert all(url.startswith(REPOSITORY_URL) for url in urls)

    def test_the_bug_report_names_the_real_folders(self) -> None:
        # 置き場の名前は userdirs が決める 変えたのに雛形が古いままだと、
        # 報告する人は無いフォルダを探すことになる
        text = (TEMPLATES / "bug-reports.yml").read_text(encoding="utf-8")
        assert f"%APPDATA%\\{APP_FOLDER}" in text
        assert f"%LOCALAPPDATA%\\{APP_FOLDER}\\recovery" in text

    def test_the_bug_report_names_the_folders_outside_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 動かし方に「ソースから」、OS に「その他」を選べる Windows の場所だけを書くと、
        # それ以外の機械の人は無いフォルダを探すことになる 場所は userdirs から求める
        for name in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_STATE_HOME"):
            monkeypatch.delenv(name, raising=False)
        text = (TEMPLATES / "bug-reports.yml").read_text(encoding="utf-8")
        for root in (userdirs.config_root(), userdirs.state_root()):
            assert f"~/{root.relative_to(Path.home()).as_posix()}" in text
        assert "XDG_CONFIG_HOME" in text
        assert "XDG_STATE_HOME" in text

    def test_the_bug_report_uses_the_labels_of_the_about_box(self) -> None:
        # 雛形は置き場を〔バージョン情報…〕の見出しで指す 見出しを変えたら雛形も直さないと、
        # 案内された見出しが画面に無い
        from sashimono.ui.main_window import about_text

        text = (TEMPLATES / "bug-reports.yml").read_text(encoding="utf-8")
        for label in ("設定・スクリプト・テンプレート", "退避・バックアップ"):
            assert f"{label}:" in about_text()
            assert f"「{label}」" in text

    def test_the_compat_report_asks_for_the_copied_template_notes(self, tmp_path: Path) -> None:
        # 雛形がテンプレートの注意書きを手で打ち写させたままだと、写し漏れと写し間違いが
        # そのまま届く 棚のボタンの名前を変えたら雛形も直さないと、案内したボタンが無い
        from sashimono.compat.catalog import TemplateCatalog
        from sashimono.ui.template_dialog import TemplateDialog

        text = (TEMPLATES / "compatibility.yml").read_text(encoding="utf-8")
        assert "書き写" not in text
        dialog = TemplateDialog(TemplateCatalog(), roots=(tmp_path,))
        try:
            label = dialog._copy_button.text()
        finally:
            dialog.close()
        guide = [line for line in text.splitlines() if "〔互換〕→〔テンプレート…〕" in line]
        assert guide
        assert all(f"〔{label}〕" in line for line in guide)

    def test_the_menu_paths_in_the_templates_exist(self, menu_paths: set[str]) -> None:
        # 雛形は〔メニュー〕→〔項目〕の形で操作を案内する 項目の名前を変えたり、
        # 別のメニューへ移したりしたら雛形も直さないと、案内どおりに探しても見つからない
        guided = {
            f"{template.name}: {path}"
            for template in TEMPLATES.glob("*.yml")
            for path in _guided_paths(template.read_text(encoding="utf-8"))
            if path not in menu_paths
        }
        assert guided == set()

    def test_an_item_under_another_menu_is_caught(self, menu_paths: set[str]) -> None:
        # メニューと項目がそれぞれ在るだけで通すと、〔互換〕→〔設定…〕のような
        # 取り違えた道を案内し続けても気付けない
        assert "表示/設定…" in menu_paths
        assert _guided_paths("〔互換〕→〔設定…〕") == ["互換/設定…"]
        assert "互換/設定…" not in menu_paths


class TestDiscussions:
    """使う人の報告は Discussions で受ける Issue は開発者が直す作業の置き場"""

    def test_the_software_sends_reports_to_discussions(self) -> None:
        # Issue へ飛ばすと、確かめる前の報告と質問が開発の作業の一覧に混ざる
        assert f"{REPOSITORY_URL}/discussions/new/choose" == REPORT_URL

    def test_users_cannot_open_issues_around_discussions(self) -> None:
        # Issue の雛形か白紙の Issue が残っていると、使う人が Discussions を通らずに Issue を作れる
        # 開発者は書き込み権限があるので、閉じていても白紙で作れる
        config = (ISSUES / "config.yml").read_text(encoding="utf-8")
        assert re.search(r"^blank_issues_enabled:\s*false\s*$", config, re.M)
        assert sorted(path.name for path in ISSUES.iterdir()) == ["config.yml"]

    def test_every_category_has_a_form_and_a_way_in(self) -> None:
        # 書き込み欄のファイルの名前がカテゴリのスラッグと違うと、GitHub は欄を出さず白紙になる
        # 雛形を選ぶ画面に行き先が無いと、Issue から来た人がカテゴリに辿り着けない
        assert {path.stem for path in TEMPLATES.glob("*.yml")} == set(DISCUSSION_CATEGORIES)
        config = (ISSUES / "config.yml").read_text(encoding="utf-8")
        linked = set(re.findall(r"discussions/new\?category=([\w-]+)", config))
        assert linked == set(DISCUSSION_CATEGORIES)

    def test_links_in_the_forms_name_real_categories(self) -> None:
        # 書き込み欄の中から別のカテゴリへ案内する 名前を変えたのに古いスラッグのままだと、
        # 案内した先で GitHub が「カテゴリが無い」と出す
        for template in TEMPLATES.glob("*.yml"):
            text = template.read_text(encoding="utf-8")
            for slug in re.findall(r"discussions/new\?category=([\w-]+)", text):
                assert slug in DISCUSSION_CATEGORIES, (template.name, slug)

    def test_the_forms_use_only_keys_discussions_accept(self) -> None:
        # Discussions の書き込み欄は body・labels・title だけを受け付け、Issue の雛形の
        # name・description があると欄ごと読まれない 入力の欄が 1 つも無い物も読まれない
        for template in TEMPLATES.glob("*.yml"):
            text = template.read_text(encoding="utf-8")
            keys = set(re.findall(r"^([A-Za-z_]+):", text, re.M))
            assert keys <= {"body", "labels", "title"}, (template.name, keys)
            assert "body" in keys
            fields = re.findall(r"^\s*- type:\s*(\w+)", text, re.M)
            assert any(kind != "markdown" for kind in fields), template.name


def _guided_paths(text: str) -> list[str]:
    """雛形が案内する〔メニュー〕→〔項目〕の道 画面の操作の名前と同じ「メニュー/項目」の形"""
    return [f"{menu}/{item}" for menu, item in re.findall(r"〔([^〕]+)〕→〔([^〕]+)〕", text)]


@pytest.fixture
def menu_paths() -> Iterator[set[str]]:
    """編集画面に実際にある「メニュー/項目」

    ソースの文字を探すのではなく、窓を組み立てて確かめる 文字で探すと、項目が
    どのメニューの下に足されたかまでは分からない ショートカットの設定が使う名前と
    同じ物なので、メニューの組み立てを変えても追える
    """
    from sashimono.ui.main_window import MainWindow

    window = MainWindow(confirm_unsaved=False)
    try:
        yield set(window._actions)
    finally:
        window.close()
