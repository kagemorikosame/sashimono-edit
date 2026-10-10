"""自動の退避と世代バックアップの設定（#271）

既定は前からの動き（30 秒ごとに退避・20 世代・``%LOCALAPPDATA%\\Sashimono``）のまま
置き場を変えた人のところで、書けない置き場のせいで退避が黙って止まらないことと、
世代数を減らしたときにバックアップを黙ってまとめて消さないことを押さえる
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWidget

from sashimono.core.commands import RenameProject
from sashimono.core.io import (
    RecoverySession,
    backup_before_save,
    backup_folder,
    default_state_root,
)
from sashimono.core.io.recovery import BACKUP_GENERATIONS
from sashimono.core.model import Project
from sashimono.ui.backup_settings import (
    StateFolderField,
    confirm_backup_changes,
    folder_caution,
    folder_refusal,
    plan_trim_all,
    state_root_for,
)
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.recovery_dialog import RecoveryDialog
from sashimono.ui.workspace import (
    AUTOSAVE_SECONDS_RANGE,
    BACKUP_GENERATIONS_RANGE,
    Preferences,
    PreferenceStore,
)


def _unwritable(tmp_path: Path) -> Path:
    """書けない置き場 親がファイルなので、中にフォルダを作れない（抜いたドライブと同じ）"""
    blocker = tmp_path / "ファイル"
    blocker.write_text("x", encoding="utf-8")
    return blocker / "置き場"


def _window(preferences: Preferences | None = None) -> MainWindow:
    if preferences is not None:
        PreferenceStore().save(preferences)
    return MainWindow(Project.create(), confirm_unsaved=False)


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """尋ねる窓・知らせる窓・フォルダを選ぶ窓が開いたらその場で落とす

    開いたまま待つと、試験は誰も押さないボタンを待って止まる 尋ねることを確かめる
    試験は、この後で自分の答えに差し替える
    """

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)
    monkeypatch.setattr(RecoveryDialog, "exec", popped)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _window()
    yield created
    created.close()


class TestTheDefaults:
    def test_they_are_what_it_did_before(self) -> None:
        # 既定を変えると、設定を開いたことのない人の退避とバックアップが黙って変わる
        plain = Preferences()
        assert plain.autosave
        assert plain.autosave_seconds == 30
        assert plain.backup
        assert plain.backup_generations == BACKUP_GENERATIONS == 20
        assert plain.state_folder == ""
        assert state_root_for(plain) == default_state_root()

    def test_an_old_file_reads_with_the_new_defaults(self, tmp_path: Path) -> None:
        # 項目を足す前の版が書いた設定に、新しい項目は無い 無いのを壊れたと見て全部を
        # 既定に戻すと、前に選んだほかの項目まで消える
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"theme": "light"}), encoding="utf-8")
        loaded = PreferenceStore(path).load()
        assert loaded.theme == "light"
        assert loaded.autosave_seconds == 30
        assert loaded.backup_generations == 20
        assert loaded.state_folder == ""


class TestSaving:
    def test_it_comes_back(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(
            autosave=False,
            autosave_seconds=10,
            backup=False,
            backup_generations=5,
            state_folder=str(tmp_path / "退避"),
        )
        store.save(chosen)
        assert store.load() == chosen

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("autosave_seconds", AUTOSAVE_SECONDS_RANGE[0] - 1),
            ("autosave_seconds", AUTOSAVE_SECONDS_RANGE[1] + 1),
            ("autosave_seconds", True),
            ("autosave_seconds", 30.5),
            ("backup_generations", 0),
            ("backup_generations", BACKUP_GENERATIONS_RANGE[1] + 1),
            ("backup_generations", "20"),
            # True は 1 と等しく範囲の中に入る 数として通すと 1 世代だけになる
            ("backup_generations", True),
        ],
    )
    def test_an_absurd_number_falls_back(self, tmp_path: Path, key: str, value: object) -> None:
        # 0 秒ごとの退避は編集を止め、0 世代は「作らない」と区別が付かない
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({key: value}), encoding="utf-8")
        assert getattr(PreferenceStore(path).load(), key) == getattr(Preferences(), key)

    def test_a_relative_folder_falls_back(self, tmp_path: Path) -> None:
        # 相対の場所は起動した作業フォルダで行き先が変わり、退避が毎回違う所へ散る
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"state_folder": "退避"}), encoding="utf-8")
        assert PreferenceStore(path).load().state_folder == ""


class TestTheDialog:
    def test_it_shows_what_is_set(self, qt_application: QApplication, tmp_path: Path) -> None:
        del qt_application
        chosen = Preferences(
            autosave=False,
            autosave_seconds=45,
            backup=False,
            backup_generations=7,
            state_folder=str(tmp_path),
        )
        dialog = PreferencesDialog(chosen)
        result = dialog.preferences()
        assert (
            result.autosave,
            result.autosave_seconds,
            result.backup,
            result.backup_generations,
            result.state_folder,
        ) == (False, 45, False, 7, str(tmp_path))
        # 切った物の数は選べない 選べると、変えたのに効かない
        assert not dialog._backups.autosave_seconds.isEnabled()
        assert not dialog._backups.backup_generations.isEnabled()

    def test_choosing_the_default_place_keeps_it_empty(self, qt_application: QApplication) -> None:
        # 既定の場所を文字で持つと、ユーザー名の違う機械へ設定を写したときにそこを指し続ける
        del qt_application
        field = StateFolderField("")
        field.line.setText(str(default_state_root()))
        assert field.folder() == ""

    def test_a_refused_folder_keeps_the_dialog_open(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        warned: list[str] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text, *_a: warned.append(text))
        dialog = PreferencesDialog(Preferences())
        dialog._backups.state_folder.line.setText(str(_unwritable(tmp_path)))
        dialog.accept()
        assert warned
        assert dialog.result() != PreferencesDialog.DialogCode.Accepted


class TestTheFolderChecks:
    def test_a_writable_folder_is_fine(self, tmp_path: Path) -> None:
        assert folder_refusal(tmp_path / "退避", install=tmp_path / "exe") is None

    def test_an_unwritable_folder_is_refused(self, tmp_path: Path) -> None:
        assert folder_refusal(_unwritable(tmp_path), install=None) is not None

    def test_the_install_folder_is_refused(self, tmp_path: Path) -> None:
        # exe の隣は zip を展開し直すと丸ごと消える（更新で消えない置き場の決まり）
        install = tmp_path / "Sashimono"
        assert folder_refusal(install / "退避", install=install) is not None

    def test_a_synced_folder_is_questioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 数十秒おきの退避がそのまま同期され、回線と相手の容量を使う
        monkeypatch.setenv("OneDrive", str(tmp_path / "od"))
        assert folder_caution(tmp_path / "od" / "退避") is not None
        assert folder_caution(tmp_path / "local") is None

    def test_a_network_place_is_questioned(self) -> None:
        assert folder_caution(Path("\\\\server\\share\\退避")) is not None


class TestReducingTheGenerations:
    def _make_backups(self, root: Path, count: int) -> Path:
        target = root.parent / "本編.sme"
        for number in range(count):
            target.write_text(str(number), encoding="utf-8")
            backup_before_save(target, root)
        return target

    def test_it_asks_with_the_number_that_will_go(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 黙って減らすと、次の保存でそのプロジェクトの控えがまとめて消える
        del qt_application
        root = tmp_path / "退避"
        self._make_backups(root, 6)
        asked: list[str] = []

        def say_no(_parent: QWidget, _title: str, text: str, *_args: object) -> object:
            asked.append(text)
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", say_no)
        before = Preferences(state_folder=str(root))
        after = Preferences(state_folder=str(root), backup_generations=2)
        assert not confirm_backup_changes(None, before, after)
        assert asked and "4 本" in asked[0]
        # 確かめるだけで、その場では消さない
        assert len(list(backup_folder(root.parent / "本編.sme", root).iterdir())) == 6

    def test_nothing_to_lose_means_no_question(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "退避"
        self._make_backups(root, 2)
        monkeypatch.setattr(QMessageBox, "question", lambda *_a: pytest.fail("尋ねた"))
        before = Preferences(state_folder=str(root))
        assert confirm_backup_changes(
            None, before, Preferences(state_folder=str(root), backup_generations=5)
        )

    def test_raising_it_asks_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(QMessageBox, "question", lambda *_a: pytest.fail("尋ねた"))
        assert confirm_backup_changes(None, Preferences(), Preferences(backup_generations=50))


class TestTheTimer:
    def test_the_default_is_every_thirty_seconds(self, window: MainWindow) -> None:
        assert window._autosave_timer.isActive()
        assert window._autosave_timer.interval() == 30_000

    def test_a_new_interval_works_without_a_restart(self, window: MainWindow) -> None:
        window._apply_preferences(Preferences(autosave_seconds=10))
        assert window._autosave_timer.interval() == 10_000
        assert window._autosave_timer.isActive()

    def test_turning_it_off_stops_the_timer(self, window: MainWindow) -> None:
        # 書き込みだけ飛ばすのでは、切ったのに起き続ける
        window._apply_preferences(Preferences(autosave=False))
        assert not window._autosave_timer.isActive()
        window._apply_preferences(Preferences(autosave=True))
        assert window._autosave_timer.isActive()

    def test_it_starts_off_when_set_off(self, qt_application: QApplication) -> None:
        del qt_application
        created = _window(Preferences(autosave=False))
        try:
            assert not created._autosave_timer.isActive()
        finally:
            created.close()


class TestTheBackups:
    def test_they_go_to_the_chosen_folder_with_the_chosen_count(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del qt_application
        root = tmp_path / "退避"
        created = _window(Preferences(state_folder=str(root), backup_generations=2))
        try:
            created._path = tmp_path / "本編.sme"
            for number in range(5):
                created.execute(RenameProject(f"{number} 回目"))
                assert created.save_project()
            kept = list(backup_folder(created._path, root).iterdir())
            assert len(kept) == 2
            assert not backup_folder(created._path).exists()
        finally:
            created.close()

    def test_turning_them_off_makes_none(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del qt_application
        created = _window(Preferences(backup=False))
        try:
            created._path = tmp_path / "本編.sme"
            for number in range(2):
                created.execute(RenameProject(f"{number} 回目"))
                assert created.save_project()
            assert not backup_folder(created._path).exists()
        finally:
            created.close()

    def test_an_unwritable_folder_backs_up_to_the_default(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 書けないまま黙ると、本人はバックアップがあると思ったまま無い
        del qt_application
        created = _window(Preferences(state_folder=str(_unwritable(tmp_path))))
        try:
            created._path = tmp_path / "本編.sme"
            for number in range(2):
                created.execute(RenameProject(f"{number} 回目"))
                assert created.save_project()
            assert len(list(backup_folder(created._path).iterdir())) == 1
            assert "既定の置き場" in created.statusBar().currentMessage()
        finally:
            created.close()

    def test_the_folder_menu_finds_the_default_copies(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 書けずに既定の側へ控えた物を、設定の置き場だけ見て「無い」と言わない
        del qt_application
        opened: list[QUrl] = []

        def fake_open(url: QUrl) -> bool:
            opened.append(url)
            return True

        monkeypatch.setattr(QDesktopServices, "openUrl", fake_open)
        created = _window(Preferences(state_folder=str(tmp_path / "退避")))
        try:
            created._path = tmp_path / "本編.sme"
            created._path.write_text("x", encoding="utf-8")
            backup_before_save(created._path)
            created.open_backup_folder()
            assert opened == [QUrl.fromLocalFile(str(backup_folder(created._path)))]
            # 選んだ置き場に控えがあれば、そちらを先に開く 既定の側だけ見ると、
            # 設定の置き場へ書いた新しい控えに辿り着けない
            chosen = tmp_path / "退避"
            backup_before_save(created._path, chosen)
            created.open_backup_folder()
            assert opened[-1] == QUrl.fromLocalFile(str(backup_folder(created._path, chosen)))
        finally:
            created.close()


class TestTheRecoveryFolder:
    def test_it_goes_to_the_chosen_folder(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del qt_application
        root = tmp_path / "退避"
        created = _window(Preferences(state_folder=str(root)))
        try:
            created.execute(RenameProject("作業中"))
            created.autosave()
            assert created._recovery.path.parent == root / "recovery"
            assert created._recovery.path.is_file()
        finally:
            created.close()

    def test_an_unwritable_folder_falls_back_and_says_so(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 書けない所を選んだまま起動しても、退避は止めない
        del qt_application
        created = _window(Preferences(state_folder=str(_unwritable(tmp_path))))
        try:
            assert created._recovery.path.parent == default_state_root() / "recovery"
            assert "書けない" in created.statusBar().currentMessage()
            created.execute(RenameProject("作業中"))
            created.autosave()
            assert created._recovery.path.is_file()
        finally:
            created.close()

    def test_losing_the_folder_midway_falls_back(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ドライブを抜いた・回線が切れた 書けない所へ書きに行き続けると退避が黙って止まる
        del qt_application
        created = _window(Preferences(state_folder=str(tmp_path / "退避")))
        try:

            def unplugged(_self: RecoverySession, *_args: object) -> None:
                raise OSError("抜いた")

            monkeypatch.setattr(created._recovery, "save", unplugged.__get__(created._recovery))
            created.execute(RenameProject("作業中"))
            created.autosave()
            assert created._recovery.path.parent == default_state_root() / "recovery"
            assert created._recovery.path.is_file()
            assert "書けない" in created.statusBar().currentMessage()
        finally:
            created.close()

    def test_changing_the_folder_moves_the_unsaved_work(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        # 前の置き場の退避を先に消すと、新しい方へ書く前に落ちたときに何も残らない
        window.execute(RenameProject("作業中"))
        window.autosave()
        old = window._recovery.path
        assert old.is_file()
        root = tmp_path / "新しい置き場"
        window._apply_preferences(Preferences(state_folder=str(root)))
        assert window._recovery.path.parent == root / "recovery"
        assert window._recovery.path.is_file()
        assert not old.exists()

    def test_a_failed_move_keeps_the_old_unsaved_work(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 新しい置き場へ書けないまま前の退避を消すと、その間に落ちたときに何も残らない
        window.execute(RenameProject("作業中"))
        window.autosave()
        old = window._recovery.path
        root = tmp_path / "新しい置き場"
        plain_save = RecoverySession.save

        def refuse_new(self: RecoverySession, project: Project, source: Path | None) -> None:
            if self.path.is_relative_to(root):
                raise OSError("書けない")
            plain_save(self, project, source)

        monkeypatch.setattr(RecoverySession, "save", refuse_new)
        window._apply_preferences(Preferences(state_folder=str(root)))
        assert old.is_file()
        assert window._recovery.path == old
        assert "書けない" in window.statusBar().currentMessage()

    def test_an_unreachable_old_folder_does_not_stop_the_move(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 前の置き場が回線の切れたネットワークだと、閉じるときに退避を消せず OSError が上がる
        # 上げたままだと設定の反映が途中で止まり、後の片付けと知らせが走らない
        window.execute(RenameProject("作業中"))
        window.autosave()
        unreachable = window._recovery
        plain_close = RecoverySession.close

        def cut_off(self: RecoverySession) -> None:
            plain_close(self)
            if self is unreachable:
                raise OSError("回線が切れた")

        monkeypatch.setattr(RecoverySession, "close", cut_off)
        target = tmp_path / "本編.sme"
        oldest = None
        for number in range(5):
            target.write_bytes(bytes([number]) * 400_000)
            copied = backup_before_save(target, tmp_path / "新しい置き場")
            oldest = oldest or copied
        assert oldest is not None
        root = tmp_path / "新しい置き場"
        chosen = Preferences(state_folder=str(root), state_limit_mb=1)
        window._apply_preferences(chosen, plan_trim_all(chosen))
        assert window._recovery.path.parent == root / "recovery"
        assert window._recovery.path.is_file()
        # 後の片付けも最後まで走る
        assert not oldest.exists()
        assert "容量の上限" in window.statusBar().currentMessage()

    def test_an_unreachable_old_folder_is_told(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 消せなかった退避が残ることを黙らない
        window.execute(RenameProject("作業中"))
        window.autosave()
        unreachable = window._recovery
        plain_close = RecoverySession.close

        def cut_off(self: RecoverySession) -> None:
            plain_close(self)
            if self is unreachable:
                raise OSError("回線が切れた")

        monkeypatch.setattr(RecoverySession, "close", cut_off)
        window._apply_preferences(Preferences(state_folder=str(tmp_path / "新しい置き場")))
        assert "消せなかった" in window.statusBar().currentMessage()

    def test_a_failed_move_with_an_unclosable_new_folder_keeps_the_old_session(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 新しい置き場へ書けず、その錠を閉じるのも失敗したとき 前のセッションへ戻す前に
        # 例外が上がると、閉じたセッションを指したまま次の退避が書けない
        window.execute(RenameProject("作業中"))
        window.autosave()
        old = window._recovery
        root = tmp_path / "新しい置き場"
        plain_save, plain_close = RecoverySession.save, RecoverySession.close

        def refuse_new(self: RecoverySession, project: Project, source: Path | None) -> None:
            if self.path.is_relative_to(root):
                raise OSError("書けない")
            plain_save(self, project, source)

        def cut_off(self: RecoverySession) -> None:
            plain_close(self)
            if self.path.is_relative_to(root):
                raise OSError("回線が切れた")

        monkeypatch.setattr(RecoverySession, "save", refuse_new)
        monkeypatch.setattr(RecoverySession, "close", cut_off)
        window._apply_preferences(Preferences(state_folder=str(root)))
        assert window._recovery is old
        assert "書けない" in window.statusBar().currentMessage()
        window.execute(RenameProject("続き"))
        window.autosave()
        assert old.path.is_file()


class TestOfferingRecovery:
    def _crash_in(self, root: Path, name: str) -> None:
        crashed = RecoverySession(root)
        crashed.save(Project.create(name=name), None)
        lock = crashed._lock
        assert lock is not None
        lock.abandon()
        crashed._lock = None

    def test_both_folders_are_searched(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選んだ置き場へ書けずに既定へ戻した起動が落ちると、退避は既定の側に残る
        # 片方しか見ないと、その作業は復元を勧められずに埋もれる
        del qt_application
        chosen = tmp_path / "退避"
        self._crash_in(chosen, "選んだ側")
        self._crash_in(default_state_root(), "既定の側")
        offered: list[str] = []

        def look(dialog: RecoveryDialog) -> int:
            offered.extend(entry.name for entry in dialog._entries)
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(RecoveryDialog, "exec", look)
        created = _window(Preferences(state_folder=str(chosen)))
        try:
            created.offer_recovery()
            assert sorted(offered) == ["既定の側", "選んだ側"]
        finally:
            created.close()
