"""Sashimono が本人の退避とバックアップを消す道の総点検（#271）

どの道でも次の 2 つを守ることを見る（一覧は docs/development.md の「更新で消えない置き場」）

- 確かめに出した物（または、はじめから消すと決めて知らせる決まりの物）しか消さない
  確かめで見せた数と、実際に消える数が一致する
- 新しい物を書けたことを確かめてから古い物を消す
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from sashimono.core.commands import RenameProject
from sashimono.core.io import (
    RecoverySession,
    backup_before_save,
    backup_folder,
    default_state_root,
    find_orphans,
)
from sashimono.core.io.recovery import TrimItem
from sashimono.core.model import Project
from sashimono.ui.backup_settings import confirm_backup_changes
from sashimono.ui.main_window import MainWindow
from sashimono.ui.recovery_dialog import RecoveryDialog
from sashimono.ui.theme import current_theme
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 開いたまま待つと、誰も押さないボタンを待って止まる"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical", "exec"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)
    monkeypatch.setattr(RecoveryDialog, "exec", popped)


@pytest.fixture(autouse=True)
def same_theme(qt_application: QApplication) -> Iterator[None]:
    """試験の前後でアプリ全体のテーマが変わらないことを見る"""
    before = (current_theme(), qt_application.styleSheet())
    yield
    assert (current_theme(), qt_application.styleSheet()) == before, "テーマを変えたまま終えた"


def _window(preferences: Preferences | None = None) -> MainWindow:
    if preferences is not None:
        PreferenceStore().save(preferences)
    return MainWindow(Project.create(), confirm_unsaved=False)


def _fill(target: Path, count: int, root: Path | None = None) -> list[Path]:
    """``target`` の控えを ``count`` 本作る 古い順 消さないよう世代数は大きくしておく"""
    made = []
    for number in range(count):
        target.write_bytes(bytes([number]) * 1000)
        copied = backup_before_save(target, root, keep=200)
        assert copied is not None
        made.append(copied)
    return made


def _answer(monkeypatch: pytest.MonkeyPatch, yes: bool) -> list[str]:
    """尋ねる窓に答える 尋ねた文を返す"""
    asked: list[str] = []

    def reply(_parent: QWidget, _title: str, text: str, *_args: object) -> object:
        asked.append(text)
        return QMessageBox.StandardButton.Yes if yes else QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", reply)
    return asked


def _shown_count(text: str) -> int:
    found = re.search(r"(\d+) 本を消します", text)
    assert found is not None, text
    return int(found.group(1))


class TestSavingRotatesOnlyOne:
    def test_a_save_over_the_limit_drops_only_one(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 置き場を変えた先や別の窓の設定で世代数を超えていても、保存のたびに黙ってまとめて消さない
        del qt_application
        target = tmp_path / "本編.sme"
        made = _fill(target, 6)
        created = _window(Preferences(backup_generations=2))
        try:
            created._path = target
            created.execute(RenameProject("作業中"))
            assert created.save_project()
            kept = sorted(backup_folder(target).iterdir())
            # 1 本作って、いちばん古い 1 本だけを消す
            assert len(kept) == 6
            assert not made[0].exists()
            assert all(path.is_file() for path in made[1:])
        finally:
            created.close()


class TestConfirmingGenerations:
    def test_switching_to_a_folder_with_many_backups_asks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 世代数は同じでも、切り替えた先の控えが多ければ確かめる
        chosen = tmp_path / "前の置き場"
        made = _fill(tmp_path / "本編.sme", 6, chosen)
        asked = _answer(monkeypatch, yes=False)
        before = Preferences(backup_generations=2)
        after = Preferences(backup_generations=2, state_folder=str(chosen))
        assert not confirm_backup_changes(None, before, after)
        assert _shown_count(asked[0]) == 4
        assert all(path.is_file() for path in made)

    def test_turning_backups_back_on_asks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fill(tmp_path / "本編.sme", 6)
        asked = _answer(monkeypatch, yes=False)
        before = Preferences(backup=False, backup_generations=2)
        after = Preferences(backup=True, backup_generations=2)
        assert not confirm_backup_changes(None, before, after)
        assert _shown_count(asked[0]) == 4

    def test_the_shown_number_is_what_goes(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 6 本ある所で 2 世代にすると 4 本と見せ、OK で 4 本を消す 次の保存は入れ替えの 1 本だけ
        del qt_application
        target = tmp_path / "本編.sme"
        made = _fill(target, 6)
        asked = _answer(monkeypatch, yes=True)
        approved: list[TrimItem] = []
        after = Preferences(backup_generations=2)
        assert confirm_backup_changes(None, Preferences(), after, approved)
        shown = _shown_count(asked[0])
        created = _window()
        try:
            created._apply_preferences(after, approved)
            gone = [path for path in made if not path.exists()]
            assert len(gone) == shown == 4
            assert made[-2].is_file() and made[-1].is_file()
            created._path = target
            created.execute(RenameProject("作業中"))
            assert created.save_project()
            assert len(list(backup_folder(target).iterdir())) == 2
            assert made[-1].is_file()
        finally:
            created.close()


class TestRecoveryIsWrittenBeforeTheOldGoes:
    def test_a_failed_fallback_keeps_the_last_good_recovery(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選んだ置き場へ書けず、既定の置き場へも書けない 先に古い方を閉じると両方消える
        del qt_application
        created = _window(Preferences(state_folder=str(tmp_path / "選んだ置き場")))
        try:
            created.execute(RenameProject("最後に書けた"))
            created.autosave()
            good = created._recovery
            assert good.path.is_file()

            def broken(_self: RecoverySession, *_args: object) -> None:
                raise OSError("書けない")

            monkeypatch.setattr(RecoverySession, "save", broken)
            created.execute(RenameProject("書けなかった"))
            created.autosave()
            assert created._recovery is good
            assert good.path.is_file()
            assert "自動退避に失敗した" in created.statusBar().currentMessage()
        finally:
            monkeypatch.undo()
            created.close()

    def test_the_new_recovery_exists_before_the_old_is_closed(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        chosen = tmp_path / "選んだ置き場"
        created = _window(Preferences(state_folder=str(chosen)))
        try:
            created.execute(RenameProject("作業中"))
            created.autosave()
            old = created._recovery
            plain_save, plain_close = RecoverySession.save, RecoverySession.close
            seen: list[bool] = []

            def chosen_fails(self: RecoverySession, project: Project, source: Path | None) -> None:
                if self.path.is_relative_to(chosen):
                    raise OSError("抜いた")
                plain_save(self, project, source)

            def watch(self: RecoverySession) -> None:
                if self is old:
                    fresh = list((default_state_root() / "recovery").glob("*.sme"))
                    seen.append(bool(fresh))
                plain_close(self)

            monkeypatch.setattr(RecoverySession, "save", chosen_fails)
            monkeypatch.setattr(RecoverySession, "close", watch)
            created.execute(RenameProject("続き"))
            created.autosave()
            assert seen == [True]
            assert created._recovery.path.is_file()
        finally:
            created.close()


class TestOtherPathsKeepTheOld:
    def test_a_failed_save_keeps_the_recovery(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 保存できなかったのに退避を消すと、落ちたときに何も残らない
        del qt_application
        warned: list[str] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, *_a: warned.append(title))
        created = _window()
        try:
            created.execute(RenameProject("作業中"))
            created.autosave()
            blocker = tmp_path / "ファイル"
            blocker.write_text("x", encoding="utf-8")
            created._path = blocker / "本編.sme"
            assert not created.save_project()
            assert warned == ["保存できない"]
            assert created._recovery.path.is_file()
        finally:
            created.close()

    def test_a_restore_that_cannot_be_copied_keeps_the_original(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 自分の退避へ写せないまま元を捨てると、復元した作業がどこにも残らない
        del qt_application
        crashed = RecoverySession()
        crashed.save(Project.create(name="落ちた作業"), None)
        lock = crashed._lock
        assert lock is not None
        lock.abandon()
        crashed._lock = None
        (entry,) = find_orphans()
        created = _window()
        try:

            def broken(_self: RecoverySession, *_args: object) -> None:
                raise OSError("書けない")

            monkeypatch.setattr(RecoverySession, "save", broken)
            assert created.restore_recovery(entry)
            assert entry.path.is_file()
        finally:
            monkeypatch.undo()
            created.close()
