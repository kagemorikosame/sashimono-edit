"""Sashimono が本人の退避とバックアップを消す道の総点検（#271）

どの道でも次の 2 つを守ることを見る（一覧は docs/development.md の「更新で消えない置き場」）

- 確かめに出した物（または、はじめから消すと決めて知らせる決まりの物）しか消さない
  確かめで見せた数と、実際に消える数が一致する
- 新しい物を書けたことを確かめてから古い物を消す
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWidget

from sashimono.core.commands import RenameProject
from sashimono.core.io import (
    RecoverySession,
    UnavailableRecovery,
    backup_before_save,
    backup_folder,
    default_state_root,
    find_orphans,
)
from sashimono.core.io.recovery import TrimItem
from sashimono.core.model import Project
from sashimono.ui import backup_settings, main_window
from sashimono.ui.backup_settings import STATE_SUBFOLDER, confirm_backup_changes
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
        made = _fill(tmp_path / "本編.sme", 6, chosen / STATE_SUBFOLDER)
        asked = _answer(monkeypatch, yes=False)
        before = Preferences(backup_generations=2)
        after = Preferences(backup_generations=2, state_folder=str(chosen))
        assert not confirm_backup_changes(None, before, after)
        assert _shown_count(asked[0]) == 4
        assert all(path.is_file() for path in made)

    def test_turning_backups_back_on_asks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 入れ直したときに確かめないと、次の保存で超えていた控えを黙ってまとめて失う
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
        # 既定へ書く前に古い退避を消すと、その間に落ちたときに何も残らない
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


def _refuse(monkeypatch: pytest.MonkeyPatch, *blocked: Path) -> None:
    """その場所だけ消せなくする（使っている最中・権限が無い など）"""
    plain = Path.unlink
    keys = {os.path.normcase(str(path)) for path in blocked}

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if os.path.normcase(str(self)) in keys:
            raise PermissionError(f"使用中: {self}")
        plain(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)


def _crash_in(root: Path, name: str, days_ago: int) -> Path:
    crashed = RecoverySession(root)
    crashed.save(Project.create(name=name), None)
    lock = crashed._lock
    assert lock is not None
    lock.abandon()
    crashed._lock = None
    meta = crashed.path.with_suffix(".json")
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["saved_at"] = (datetime.now() - timedelta(days=days_ago)).isoformat()
    meta.write_text(json.dumps(data), encoding="utf-8")
    return crashed.path


class TestAFailureMidwayIsStillTold:
    """途中の 1 件で失敗しても、消せた物を知らせ、残りも続けて片付ける"""

    def test_a_rotation_that_cannot_remove_is_told_on_save(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 入れ替えで古い控えを消せなかったことを、保存の知らせに出す 黙ると、本人は世代数を
        # 守れていると思ったまま控えが増えていく
        del qt_application
        target = tmp_path / "本編.sme"
        made = _fill(target, 2)
        created = _window(Preferences(backup_generations=2))
        try:
            created._path = target
            created.execute(RenameProject("作業中"))
            _refuse(monkeypatch, made[0])
            assert created.save_project()
            message = created.statusBar().currentMessage()
            assert "保存した" in message and "1 本は消せなかった" in message
            assert made[0].is_file()
        finally:
            monkeypatch.undo()
            created.close()

    def test_trimming_after_a_save(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 途中で 1 件消せないだけで知らせが出ないと、本人は消えた控えを知らない
        del qt_application
        target = tmp_path / "本編.sme"
        made = _fill(target, 4)
        created = _window(Preferences(state_limit_mb=1))
        try:
            for path in made:
                path.write_bytes(b"x" * 400_000)
            _refuse(monkeypatch, made[1])
            message = created._trim_state()
            assert "片付けた" in message and "1 件は消せなかった" in message
            assert not made[0].exists() and made[1].is_file()
        finally:
            monkeypatch.undo()
            created.close()

    def test_applying_what_was_confirmed(
        self, window_with_backups: tuple[MainWindow, list[Path]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 途中で 1 件消せないだけで知らせが出ないと、本人は確かめた控えがどれだけ消えたか知らない
        created, made = window_with_backups
        asked = _answer(monkeypatch, yes=True)
        approved: list[TrimItem] = []
        after = Preferences(backup_generations=2)
        assert confirm_backup_changes(None, Preferences(), after, approved)
        assert _shown_count(asked[0]) == 4
        _refuse(monkeypatch, made[1])
        created._apply_preferences(after, approved)
        message = created.statusBar().currentMessage()
        assert "確かめた物を片付けた: 3 件" in message and "1 件は消せなかった" in message
        assert [path.exists() for path in made] == [False, True, False, False, True, True]

    def test_tidying_at_start(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 途中で 1 件消せないだけで知らせが出ないと、本人は片付いた退避を知らない
        del qt_application
        paths = [_crash_in(default_state_root(), name, 10) for name in ("一", "二", "三")]
        # 前の起動で勧めてから日数を過ぎたことにする（初めて見せた起動では片付けない）
        stamp = (datetime.now() - timedelta(days=10)).timestamp()
        for path in paths:
            mark = path.with_suffix(".offered")
            mark.touch()
            os.utime(mark, (stamp, stamp))
        monkeypatch.setattr(RecoveryDialog, "exec", lambda _dialog: 0)
        created = _window(Preferences(recovery_keep_days=7))
        try:
            _refuse(monkeypatch, paths[1])
            created.offer_recovery()
            message = created.statusBar().currentMessage()
            assert "2 件片付けた" in message and "1 件は消せなかった" in message
            assert [path.exists() for path in paths] == [False, True, False]
        finally:
            monkeypatch.undo()
            created.close()

    def test_a_discard_that_fails_is_told_and_not_asked_again(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 消せない物を残したまま窓を出し直すと、同じ 1 件を何度も尋ねる
        del qt_application
        path = _crash_in(default_state_root(), "捨てる", 0)
        shown: list[int] = []

        def discard_it(dialog: RecoveryDialog) -> int:
            shown.append(1)
            dialog.choice = ("discard", dialog._entries[0])
            return 1

        monkeypatch.setattr(RecoveryDialog, "exec", discard_it)
        created = _window()
        try:
            _refuse(monkeypatch, path)
            created.offer_recovery()
            assert shown == [1]
            assert path.is_file()
            assert "捨てられなかった" in created.statusBar().currentMessage()
        finally:
            monkeypatch.undo()
            created.close()


@pytest.fixture
def window_with_backups(
    qt_application: QApplication, tmp_path: Path
) -> Iterator[tuple[MainWindow, list[Path]]]:
    del qt_application
    made = _fill(tmp_path / "本編.sme", 6)
    created = _window()
    yield created, made
    created.close()


class TestBackupsLeftInTheOldFolder:
    def test_the_folder_menu_reaches_them_after_a_move(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 置き場を A から B へ変えた 前の控えは A に残す決まりなので、A を探さないと
        # 控えがあるのに「バックアップはありません」と出て、画面からたどれない
        del qt_application
        opened: list[QUrl] = []

        def fake_open(url: QUrl) -> bool:
            opened.append(url)
            return True

        monkeypatch.setattr(QDesktopServices, "openUrl", fake_open)
        monkeypatch.setattr(RecoveryDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)
        old, new = tmp_path / "A", tmp_path / "B"
        target = tmp_path / "本編.sme"
        target.write_text("前の中身", encoding="utf-8")
        first = _window(Preferences(state_folder=str(old)))
        try:
            first._path = target
            first.execute(RenameProject("作業中"))
            assert first.save_project()
            first._apply_preferences(Preferences(state_folder=str(new)))
            first.open_backup_folder()
            expected = QUrl.fromLocalFile(str(backup_folder(target, old / STATE_SUBFOLDER)))
            assert opened == [expected]
        finally:
            first.close()
        # 次の起動でも、退避の無い前の置き場を忘れずに探す
        second = _window()
        try:
            second.offer_recovery()
            second._path = target
            second.open_backup_folder()
            assert opened[-1] == expected
        finally:
            second.close()


class TestABrokenCopyInTheChosenFolder:
    def test_the_folder_menu_opens_the_good_copy(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選んだ置き場へ写す途中で失敗し、既定の置き場へ控え直した 途中の物や空のフォルダが
        # 選んだ置き場に残ると、そちらを開いて正しい控えへたどり着けない
        del qt_application
        chosen = tmp_path / "選んだ置き場"
        plain_copy = shutil.copyfile

        def copy(source: Path, destination: Path, **kwargs: bool) -> Path:
            if Path(destination).is_relative_to(chosen):
                Path(destination).write_bytes(b"half")
                raise OSError("書いている途中でドライブを抜いた")
            return Path(plain_copy(source, destination, **kwargs))

        monkeypatch.setattr(shutil, "copyfile", copy)
        opened: list[QUrl] = []

        def fake_open(url: QUrl) -> bool:
            opened.append(url)
            return True

        monkeypatch.setattr(QDesktopServices, "openUrl", fake_open)
        target = tmp_path / "本編.sme"
        target.write_text("前の中身", encoding="utf-8")
        created = _window(Preferences(state_folder=str(chosen)))
        try:
            created._path = target
            created.execute(RenameProject("作業中"))
            assert created.save_project()
            assert not backup_folder(target, chosen).exists()
            created.open_backup_folder()
            assert opened == [QUrl.fromLocalFile(str(backup_folder(target)))]
        finally:
            created.close()


class TestAnOrdinaryFolderIsLeftAlone:
    def test_the_users_files_are_neither_read_nor_removed(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 普段使っているフォルダを置き場に選んだ その直下の recovery と backups は本人の物
        # Sashimono の物として読んだり消したりすると、本人のファイルを失う
        del qt_application
        chosen = tmp_path / "書類"
        (chosen / "recovery").mkdir(parents=True)
        (chosen / "recovery" / "重要.json").write_text("{}", encoding="utf-8")
        (chosen / "backups" / "去年").mkdir(parents=True)
        (chosen / "backups" / "去年" / "20240101-000000-000000-000.sme").write_bytes(b"x" * 2000)
        monkeypatch.setattr(RecoveryDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)
        target = tmp_path / "本編.sme"
        target.write_text("前の中身", encoding="utf-8")
        created = _window(
            Preferences(state_folder=str(chosen), state_limit_mb=1, recovery_keep_days=1)
        )
        try:
            created.offer_recovery()
            created._path = target
            created.execute(RenameProject("作業中"))
            created.autosave()
            assert created.save_project()
            assert created._recovery.path.is_relative_to(chosen / STATE_SUBFOLDER)
            assert backup_folder(target, chosen / STATE_SUBFOLDER).is_dir()
        finally:
            created.close()
        assert (chosen / "recovery" / "重要.json").is_file()
        assert (chosen / "backups" / "去年" / "20240101-000000-000000-000.sme").is_file()


class TestAnUnusableDefaultFolder:
    def test_going_back_to_an_unwritable_default_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 書けない既定の置き場へ戻すのを通すと、退避を移す所で止まり、次の起動も同じ所で止まる
        blocker = tmp_path / "ファイル"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(backup_settings, "default_state_root", lambda: blocker / "既定")
        warned: list[str] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text, *_a: warned.append(text))
        before = Preferences(state_folder=str(tmp_path / "選んだ置き場"))
        assert not confirm_backup_changes(None, before, Preferences())
        assert warned and str(blocker / "既定") in warned[0]

    def test_starting_without_a_usable_default_keeps_going_and_says_so(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 既定の置き場が使えないだけで起動が止まると、本人は作業そのものができない
        del qt_application
        blocker = tmp_path / "ファイル"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(main_window, "default_state_root", lambda: blocker / "既定")
        monkeypatch.setattr(backup_settings, "default_state_root", lambda: blocker / "既定")
        created = _window()
        try:
            assert isinstance(created._recovery, UnavailableRecovery)
            assert "退避できません" in created.statusBar().currentMessage()
            created.execute(RenameProject("作業中"))
            created.autosave()
            assert "自動退避に失敗した" in created.statusBar().currentMessage()
        finally:
            created.close()
