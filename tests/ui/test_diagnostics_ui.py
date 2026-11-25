"""ヘルプの〔診断情報を作る…〕（#516）

作った報告は本人が見て保存する 送る口は無い 壊れ方は、伏せずに見せる・作れないと
何も出ない・保存しないと決めたのに何か残る・設定の置き場が効かない
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from sashimono import __version__
from sashimono.fault_log import ERRORS_NAME, diagnostics_folder
from sashimono.ui import diagnostic_dialog
from sashimono.ui.diagnostic_dialog import DiagnosticDialog, report_folder
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(confirm_unsaved=False)
    yield created
    created.close()


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[DiagnosticDialog]:
    """窓を本当には待たずに、出した窓を覚える 何もせずに閉じた（保存しない）扱い"""
    dialogs: list[DiagnosticDialog] = []

    def fake_exec(self: DiagnosticDialog) -> int:
        dialogs.append(self)
        return int(QDialog.DialogCode.Rejected)

    monkeypatch.setattr(DiagnosticDialog, "exec", fake_exec)
    return dialogs


class TestTheMenu:
    def test_the_item_shows_the_masked_report(
        self,
        window: MainWindow,
        shown: list[DiagnosticDialog],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、エラーの文に残った素材の名前がそのまま窓に出て、そのまま保存される
        place = Path("C:/作業/社員旅行の記録.mp4")
        monkeypatch.setattr(window, "remembered_names", lambda: (str(place),))
        errors = diagnostics_folder() / ERRORS_NAME
        errors.parent.mkdir(parents=True, exist_ok=True)
        errors.write_text(
            f"=== 2026-11-24T10:00:00 版 {__version__}\nOSError: 社員旅行の記録.mp4 を読めない\n",
            encoding="utf-8",
        )
        window._actions["ヘルプ/診断情報を作る…"][0].trigger()
        (dialog,) = shown
        text = dialog.current_text()
        assert f"版: {__version__}" in text
        assert "OSError" in text
        assert "社員旅行" not in text

    def test_a_failed_report_still_shows_why(
        self,
        window: MainWindow,
        shown: list[DiagnosticDialog],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 何も出ないと、報告できないうえに理由も分からない
        def broken() -> str:
            raise RuntimeError("壊れた記録")

        monkeypatch.setattr(window, "diagnostic_text", broken)
        window.create_diagnostics()
        (dialog,) = shown
        assert "作れませんでした" in dialog.current_text()
        assert "RuntimeError" in dialog.current_text()

    def test_closing_without_saving_leaves_nothing(
        self, window: MainWindow, shown: list[DiagnosticDialog], tmp_path: Path
    ) -> None:
        # 送らない（保存しない）と決めたら、どこにも書かず、覚えた置き場も変えない
        # 保存の窓が開く所（設定の置き場）と診断の記録の置き場の両方を、作る前と比べる
        # （PR #550 の指摘 見ている所が保存先と別だと、誤って書いても通ってしまう）
        folder = tmp_path / "報告"
        folder.mkdir()
        window._remember_preferences(replace(window._preferences, report_folder=str(folder)))
        diagnostics = diagnostics_folder()
        diagnostics.mkdir(parents=True, exist_ok=True)
        before_files = {*folder.rglob("*"), *diagnostics.rglob("*.txt")}
        before = window._preferences
        window.create_diagnostics()
        assert shown[0].saved_path is None
        assert window._preferences == before
        assert {*folder.rglob("*"), *diagnostics.rglob("*.txt")} == before_files

    def test_saving_remembers_the_folder(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 保存した所を次に開く所にする 本人が直した中身を書く（直す前の物を書かない）
        target = tmp_path / "報告" / "診断.txt"
        target.parent.mkdir()

        def save_edited(self: DiagnosticDialog) -> int:
            self.text.setPlainText("本人が直した中身")
            assert self.save_to(target)
            return int(QDialog.DialogCode.Accepted)

        monkeypatch.setattr(DiagnosticDialog, "exec", save_edited)
        window.create_diagnostics()
        assert target.read_text(encoding="utf-8") == "本人が直した中身"
        assert window._preferences.report_folder == str(target.parent)
        assert PreferenceStore().stored().report_folder == str(target.parent)

    def test_a_folder_that_could_not_be_remembered_is_tried_again(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、設定を書けなかったのに覚えたことになり、書けるようになってから同じ所へ
        # 保存しても書き直さない 書けなかった知らせも保存の知らせで消える（PR #550 の指摘）
        target = tmp_path / "報告" / "診断.txt"
        target.parent.mkdir()

        def save(self: DiagnosticDialog) -> int:
            assert self.save_to(target)
            return int(QDialog.DialogCode.Accepted)

        monkeypatch.setattr(DiagnosticDialog, "exec", save)
        real_save = PreferenceStore.save

        def refuse(self: PreferenceStore, preferences: Preferences) -> None:
            raise OSError("書けない")

        monkeypatch.setattr(PreferenceStore, "save", refuse)
        before = window._preferences.report_folder
        window.create_diagnostics()
        assert target.is_file()
        assert window._preferences.report_folder == before
        message = window.statusBar().currentMessage()
        assert "診断情報を保存しました" in message and "覚えていません" in message

        monkeypatch.setattr(PreferenceStore, "save", real_save)
        window.create_diagnostics()
        assert window._preferences.report_folder == str(target.parent)
        assert PreferenceStore().stored().report_folder == str(target.parent)


class TestTheFolder:
    def test_a_missing_folder_falls_back(self, tmp_path: Path) -> None:
        # 無いフォルダで保存の窓を開くと、見当違いの所で開いてどこへ保存したかを見失う
        assert report_folder(str(tmp_path)) == tmp_path
        assert report_folder(str(tmp_path / "無い")) == diagnostic_dialog.default_report_folder()
        assert report_folder("") == diagnostic_dialog.default_report_folder()

    def test_the_settings_keep_both_new_items(self, qt_application: QApplication) -> None:
        # 設定画面で OK を押しただけで、選んだ値が既定へ戻らない
        del qt_application
        chosen = replace(Preferences(), report_folder="D:\\報告", splash_motion=False)
        dialog = PreferencesDialog(chosen)
        kept = dialog.preferences()
        assert (kept.report_folder, kept.splash_motion) == ("D:\\報告", False)
        dialog.deleteLater()


def test_a_folder_only_in_memory_is_written(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 壊れると、設定画面で変えて書けなかった置き場へ保存したときに「もう覚えている」と見て
    # 書き直さず、次の起動で前の置き場に戻る（PR #550 の指摘）
    target = tmp_path / "報告" / "診断.txt"
    target.parent.mkdir()
    window._preferences = replace(window._preferences, report_folder=str(target.parent))

    def save(self: DiagnosticDialog) -> int:
        assert self.save_to(target)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(DiagnosticDialog, "exec", save)
    assert PreferenceStore().stored().report_folder != str(target.parent)
    window.create_diagnostics()
    assert PreferenceStore().stored().report_folder == str(target.parent)


def test_a_folder_already_written_is_also_set_in_memory(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 壊れると、ファイルには A があり画面の中だけ B（書けなかった）のとき、A へ保存しても
    # 画面の中が B のまま残り、次の保存の窓が B で開く（PR #550 の指摘）
    written = tmp_path / "A"
    written.mkdir()
    PreferenceStore().save(replace(window._preferences, report_folder=str(written)))
    window._preferences = replace(window._preferences, report_folder=str(tmp_path / "B"))
    target = written / "診断.txt"

    def save(self: DiagnosticDialog) -> int:
        assert self.save_to(target)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(DiagnosticDialog, "exec", save)
    window.create_diagnostics()
    assert window._preferences.report_folder == str(written)


def test_comparing_the_folder_does_not_reset_what_save_compares_with(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 壊れると、保存の置き場を比べるために設定を読み直した所で、保存が変えた所を見分ける元が
    # ファイルの今の中身に替わる すると 2 つ目の窓が書いた設定（テーマ）を、この窓の古い値を
    # 「変えた」と見て書き戻して消す（PR #550 の指摘）
    target = tmp_path / "報告" / "診断.txt"
    target.parent.mkdir()
    assert window._preferences.theme != "light"
    # 2 つ目の窓（別のプロセス）の代わりに、設定のファイルを直に書き換える
    # 同じプロセスの読み手で書くと、見分ける元（プロセスの中で 1 つ）まで替わる
    import json

    store = PreferenceStore()
    data = json.loads(store.path.read_text(encoding="utf-8")) if store.path.exists() else {}
    data["theme"] = "light"
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(data), encoding="utf-8")

    def save(self: DiagnosticDialog) -> int:
        assert self.save_to(target)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(DiagnosticDialog, "exec", save)
    window.create_diagnostics()
    stored = PreferenceStore().stored()
    assert stored.report_folder == str(target.parent)
    assert stored.theme == "light"
