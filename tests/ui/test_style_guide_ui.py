"""人が編集の方針（#343）を書く所 プロジェクト設定の窓と 表示 → 設定… の新しい作品の方針

方針は作品に付き、プロジェクト設定で変えると取り消し 1 回で戻る 新しい作品の方針は本人の
設定に付き、新しく作る作品と起動した直後の空の作品に入る AI へ同意なしで送るアプリの設定には
方針の文を載せない
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
)

from sashimono.core.commands import Author
from sashimono.core.model import Project, StyleGuide
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.project_settings_dialog import ProjectSettingsDialog
from sashimono.ui.style_guide_dialog import StyleGuideDialog, StyleGuideRow, guide_summary
from sashimono.ui.workspace import Preferences, PreferenceStore

GUIDE = StyleGuide(
    fonts=("源ノ角ゴシック",),
    colors=("#ffffff",),
    tempo="間は 0.3 秒まで",
    effects=("blur",),
    notes="方針の秘密の決まり",
)

#: 試験で並べる効果 本物の一覧は 100 を超え、印を付ける行を探すだけで遅い
EFFECTS = [("blur", "ぼかし"), ("glow", "発光")]


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """試験の間に窓（方針・設定・確かめ・色・ファイル）が開いたら、その場で落とす

    開いたままだと、無人の CI では誰も閉じずに試験が止まる 開いてよい窓は、試験ごとに
    その窓の ``exec`` を答える関数へ差し替える
    """

    def refuse(*_arguments: object, **_options: object) -> int:
        raise AssertionError("試験の間に窓が開いた")

    monkeypatch.setattr(QDialog, "exec", refuse)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, refuse)
    monkeypatch.setattr(QColorDialog, "getColor", refuse)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, refuse)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


def _answer_guide(
    monkeypatch: pytest.MonkeyPatch, fill: Callable[[StyleGuideDialog], None]
) -> list[StyleGuideDialog]:
    """方針の窓を開いた所で ``fill`` を書いて OK を押したことにする"""
    opened: list[StyleGuideDialog] = []

    def fake_exec(self: StyleGuideDialog) -> int:
        opened.append(self)
        fill(self)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(StyleGuideDialog, "exec", fake_exec)
    return opened


def _write_guide(dialog: StyleGuideDialog) -> None:
    dialog.set_fields(fonts="源ノ角ゴシック", colors="#FFFFFF", tempo="間は 0.3 秒まで")
    dialog.set_fields(notes="方針の秘密の決まり")
    for index in range(dialog.effects_list.count()):
        item = dialog.effects_list.item(index)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == "blur":
            item.setCheckState(Qt.CheckState.Checked)


def _ok(dialog: StyleGuideDialog) -> bool:
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None
    button = buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert button is not None
    return button.isEnabled()


class TestDialog:
    def test_the_fields_become_a_guide(self, qt_application: QApplication) -> None:
        """壊れると、窓に書いた方針と違う物が作品に入る（大文字の色や空の行が残る）"""
        del qt_application
        dialog = StyleGuideDialog(StyleGuide(), effects=EFFECTS)
        _write_guide(dialog)
        assert dialog.guide() == GUIDE
        dialog.deleteLater()

    def test_a_bad_colour_stops_ok(self, qt_application: QApplication) -> None:
        """通すと、閉じてからコマンドに断られ、書いた方針が窓ごと消える"""
        del qt_application
        dialog = StyleGuideDialog(StyleGuide(), effects=EFFECTS)
        dialog.set_fields(colors="白")
        assert not _ok(dialog)
        dialog.set_fields(colors="#ffffff")
        assert _ok(dialog)
        dialog.deleteLater()

    def test_an_effect_that_is_not_here_is_kept(self, qt_application: QApplication) -> None:
        """一覧に無い効果（読み込んでいないスクリプト）を落とすと、OK を押しただけで方針が変わる"""
        del qt_application
        dialog = StyleGuideDialog(StyleGuide(effects=("aviutl:よそのスクリプト",)), effects=EFFECTS)
        assert dialog.guide().effects == ("aviutl:よそのスクリプト",)
        dialog.deleteLater()

    def test_cancel_keeps_the_guide(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """取り消したのに書きかけの方針が残ると、やめたつもりの決まりで AI が編集する"""
        del qt_application

        def cancel(self: StyleGuideDialog) -> int:
            self.set_fields(notes="書きかけ")
            return int(QDialog.DialogCode.Rejected)

        monkeypatch.setattr(StyleGuideDialog, "exec", cancel)
        row = StyleGuideRow(GUIDE, title="編集の方針")
        assert not row.edit()
        assert row.guide == GUIDE
        row.deleteLater()

    def test_the_summary_says_what_is_decided(self) -> None:
        """要約が中身を言わないと、窓を開くまで方針があるかどうか分からない"""
        assert "決めていない" in guide_summary(StyleGuide())
        summary = guide_summary(GUIDE)
        assert "書体 1" in summary and "テンポ" in summary
        # 要約は窓の 1 行 本人の文そのものは出さない（長い文で窓が広がる）
        assert GUIDE.notes not in summary


class TestProjectSettings:
    def _accept_with_guide(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answer_guide(monkeypatch, _write_guide)

        def accept(self: ProjectSettingsDialog) -> int:
            assert self.guide_row.edit()
            return int(QDialog.DialogCode.Accepted)

        monkeypatch.setattr(ProjectSettingsDialog, "exec", accept)

    def test_the_guide_is_set_and_undone_in_one_step(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、方針を書いた後に Ctrl+Z 1 回で戻らないか、人の変更が AI の段に数えられる"""
        self._accept_with_guide(monkeypatch)
        window.edit_settings()
        document = window.document
        assert document.project.style_guide == GUIDE
        assert document.undo_by is Author.PERSON
        window.undo()
        assert document.project.style_guide.empty

    def test_ok_without_changes_leaves_no_step(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """何も変えない OK で段を積むと、人の段が増えて AI の取り消しが止まる"""
        monkeypatch.setattr(
            ProjectSettingsDialog, "exec", lambda _self: int(QDialog.DialogCode.Accepted)
        )
        window.edit_settings()
        assert not window.document.can_undo

    def test_the_dialog_shows_the_project_guide(self, qt_application: QApplication) -> None:
        """今の方針を見せずに開くと、OK を押しただけで方針が空に戻る"""
        del qt_application
        dialog = ProjectSettingsDialog(Project.create().settings, guide=GUIDE)
        assert dialog.style_guide() == GUIDE
        dialog.deleteLater()


class TestNewProjects:
    def test_a_blank_project_takes_the_default(self, qt_application: QApplication) -> None:
        """入れないと、起動した直後の空の作品に置いて保存した人の作品が方針を持たない"""
        del qt_application
        PreferenceStore().save(Preferences(new_project_style_guide=GUIDE))
        created = MainWindow(confirm_unsaved=False)
        try:
            assert created.document.project.style_guide == GUIDE
            # 方針を入れただけで「保存していない変更」にしない
            assert not created.is_modified
        finally:
            created.close()

    def test_a_new_project_takes_the_default(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """新規作成の窓が本人の方針を初めの値にしないと、作るたびに方針を書き直す"""
        window._remember_preferences(Preferences(new_project_style_guide=GUIDE))
        seen: list[StyleGuide] = []

        def accept(self: ProjectSettingsDialog) -> int:
            seen.append(self.style_guide())
            return int(QDialog.DialogCode.Accepted)

        monkeypatch.setattr(ProjectSettingsDialog, "exec", accept)
        window.new_project()
        assert seen == [GUIDE]
        assert window.document.project.style_guide == GUIDE

    def test_the_default_survives_a_restart(self, tmp_path: Path) -> None:
        """書いて読み戻すと消えると、次に起動したときに新しい作品の方針が空に戻る"""
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(new_project_style_guide=GUIDE))
        assert PreferenceStore(tmp_path / "preferences.json").load().new_project_style_guide == (
            GUIDE
        )

    @pytest.mark.parametrize("broken", ["方針", {"fonts": 3}, {"colors": ["白"]}])
    def test_a_broken_default_falls_back(self, tmp_path: Path, broken: object) -> None:
        """壊れた値で起動を止めるか、持てない方針のまま新規作成の窓の OK を押せなくなる"""
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"new_project_style_guide": broken}), encoding="utf-8")
        assert PreferenceStore(path).load().new_project_style_guide.empty

    def test_the_preferences_dialog_returns_the_edited_default(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """表示 → 設定… で書いた方針が設定に戻らないと、書いても新しい作品に入らない"""
        del qt_application
        _answer_guide(monkeypatch, _write_guide)
        dialog = PreferencesDialog(Preferences())
        try:
            rows = dialog.findChildren(StyleGuideRow)
            assert len(rows) == 1
            assert rows[0].edit()
            assert dialog.preferences().new_project_style_guide == GUIDE
        finally:
            dialog.deleteLater()


def test_app_settings_do_not_carry_the_default_text(window: MainWindow) -> None:
    """アプリの設定は同意なしで AI へ送る 方針の文を載せると、文を送らない人の文が外へ出る"""
    window._remember_preferences(Preferences(new_project_style_guide=GUIDE))
    text = json.dumps(window.app_settings(), ensure_ascii=False, default=str)
    assert GUIDE.notes not in text
    assert GUIDE.tempo not in text
    preferences = window.app_settings()["preferences"]
    assert isinstance(preferences, dict)
    assert preferences["new_project_style_guide"] is True
