"""残った退避の日数での片付けと、置き場の容量の上限（#271 の 4）

どちらも本人の物を消すので、消してはいけない物（まだ勧めていない落ちた作業・開いている
作業の今の退避・各プロジェクトのいちばん新しいバックアップ）が残ることを中心に見る
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWidget

from sashimono.core.commands import RenameProject
from sashimono.core.io import (
    RecoverySession,
    backup_before_save,
    backup_folder,
    default_state_root,
)
from sashimono.core.model import Project
from sashimono.ui.backup_settings import confirm_backup_changes, plan_trim_all
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.recovery_dialog import RecoveryDialog
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


def _window(preferences: Preferences | None = None) -> MainWindow:
    if preferences is not None:
        PreferenceStore().save(preferences)
    return MainWindow(Project.create(), confirm_unsaved=False)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _window()
    yield created
    created.close()


def _crash_in(root: Path, name: str, days_ago: int = 0) -> Path:
    """落ちた作業を 1 つ残す ``days_ago`` 日前に退避したことにする"""
    crashed = RecoverySession(root)
    crashed.save(Project.create(name=name), None)
    lock = crashed._lock
    assert lock is not None
    lock.abandon()
    crashed._lock = None
    if days_ago:
        meta = crashed.path.with_suffix(".json")
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["saved_at"] = (datetime.now() - timedelta(days=days_ago)).isoformat()
        meta.write_text(json.dumps(data), encoding="utf-8")
    return crashed.path


def _decline(monkeypatch: pytest.MonkeyPatch) -> None:
    """復元の窓で何も選ばずに閉じたことにする"""
    monkeypatch.setattr(RecoveryDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)


def _fill(target: Path, count: int, size: int = 400_000) -> list[Path]:
    """既定の置き場に ``target`` の控えを ``count`` 本作る 古い順"""
    made = []
    for number in range(count):
        target.write_bytes(bytes([number]) * size)
        copied = backup_before_save(target)
        assert copied is not None
        made.append(copied)
    return made


class TestTheSettings:
    def test_the_defaults_tidy_nothing(self) -> None:
        # 既定を変えると、設定を開いたことのない人の退避と控えが黙って消え始める
        plain = Preferences()
        assert plain.recovery_keep_days == 0
        assert plain.state_limit_mb == 0

    def test_they_come_back(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(recovery_keep_days=30, state_limit_mb=500)
        store.save(chosen)
        assert store.load() == chosen

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("recovery_keep_days", -1),
            ("recovery_keep_days", 366),
            ("recovery_keep_days", True),
            ("state_limit_mb", -1),
            ("state_limit_mb", 1_000_001),
            ("state_limit_mb", "100"),
        ],
    )
    def test_an_absurd_value_falls_back(self, tmp_path: Path, key: str, value: object) -> None:
        # 壊れた値で片付けを始めない 既定（片付けない・上限なし）へ戻す
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({key: value}), encoding="utf-8")
        assert getattr(PreferenceStore(path).load(), key) == 0

    def test_the_dialog_shows_them(self, qt_application: QApplication) -> None:
        del qt_application
        dialog = PreferencesDialog(Preferences(recovery_keep_days=14, state_limit_mb=300))
        result = dialog.preferences()
        assert (result.recovery_keep_days, result.state_limit_mb) == (14, 300)


class TestTidyingAtStart:
    def test_an_old_orphan_goes_only_after_it_was_offered(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        old = _crash_in(default_state_root(), "古い", days_ago=10)
        created = _window(Preferences(recovery_keep_days=7))
        try:
            # 起動しただけでは片付けない まだ一度も勧めていない
            assert old.is_file()
            _decline(monkeypatch)
            created.offer_recovery()
            assert not old.exists()
            assert "1 件片付けた" in created.statusBar().currentMessage()
        finally:
            created.close()

    def test_a_recent_orphan_stays(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        recent = _crash_in(default_state_root(), "新しい", days_ago=2)
        _decline(monkeypatch)
        created = _window(Preferences(recovery_keep_days=7))
        try:
            created.offer_recovery()
            assert recent.is_file()
        finally:
            created.close()

    def test_turned_off_keeps_everything(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        old = _crash_in(default_state_root(), "古い", days_ago=300)
        _decline(monkeypatch)
        created = _window()
        try:
            created.offer_recovery()
            assert old.is_file()
        finally:
            created.close()

    def test_nothing_goes_before_every_orphan_was_shown(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 復元の窓で 1 件を捨てると次の窓が出る その前に片付けが走ると、
        # 残りは窓に出ないまま消える
        del qt_application
        first = _crash_in(default_state_root(), "一つ目", days_ago=10)
        second = _crash_in(default_state_root(), "二つ目", days_ago=10)
        shown: list[int] = []

        def discard_first(dialog: RecoveryDialog) -> int:
            shown.append(len(dialog._entries))
            if len(shown) == 1:
                assert first.is_file() and second.is_file()
                dialog.choice = ("discard", dialog._entries[0])
                return QDialog.DialogCode.Accepted
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(RecoveryDialog, "exec", discard_first)
        created = _window(Preferences(recovery_keep_days=7))
        try:
            created.offer_recovery()
            assert shown == [2, 1]
        finally:
            created.close()


class TestTheSizeLimit:
    def test_saving_over_the_limit_trims_the_oldest_and_says_so(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del qt_application
        target = tmp_path / "本編.sme"
        made = _fill(target, 4)
        created = _window(Preferences(state_limit_mb=1))
        try:
            created._path = target
            created.execute(RenameProject("作業中"))
            created.autosave()
            assert created.save_project()
            kept = sorted(backup_folder(target).iterdir())
            # いちばん新しい控え（今の保存の前の中身）は必ず残る
            assert kept[-1] not in made
            assert not made[0].exists()
            assert "容量の上限" in created.statusBar().currentMessage()
        finally:
            created.close()

    def test_setting_the_limit_trims_now_but_keeps_what_matters(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        made = _fill(tmp_path / "本編.sme", 5)
        unseen = _crash_in(default_state_root(), "見せていない")
        window.execute(RenameProject("作業中"))
        window.autosave()
        chosen = Preferences(state_limit_mb=1)
        window._apply_preferences(chosen, plan_trim_all(chosen))
        assert not made[0].exists()
        # 消さない物 いちばん新しい控え・開いている作業の今の退避・まだ勧めていない落ちた作業
        assert made[-1].is_file()
        assert window._recovery.path.is_file()
        assert unseen.is_file()
        assert "容量の上限" in window.statusBar().currentMessage()

    def test_no_limit_trims_nothing(self, window: MainWindow, tmp_path: Path) -> None:
        made = _fill(tmp_path / "本編.sme", 5)
        window._apply_preferences(Preferences(theme="light"))
        assert all(path.is_file() for path in made)

    def test_setting_it_asks_with_what_will_go(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made = _fill(tmp_path / "本編.sme", 5)
        asked: list[str] = []

        def say_no(_parent: QWidget, _title: str, text: str, *_args: object) -> object:
            asked.append(text)
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", say_no)
        assert not confirm_backup_changes(None, Preferences(), Preferences(state_limit_mb=1))
        assert asked and "バックアップ 本編" in asked[0]
        # 確かめるだけで、その場では消さない
        assert all(path.is_file() for path in made)

    def test_nothing_to_lose_means_no_question(self, tmp_path: Path) -> None:
        _fill(tmp_path / "本編.sme", 2)
        assert confirm_backup_changes(None, Preferences(), Preferences(state_limit_mb=100))

    def test_raising_it_asks_nothing(self, tmp_path: Path) -> None:
        # 上げても 2MB を超えるぶん（6 本で約 2.4MB）を置く 超えていないと、尋ねる決まりが
        # 壊れても尋ねる物が無く、試験が見分けられない
        _fill(tmp_path / "本編.sme", 6)
        before = Preferences(state_limit_mb=1)
        assert confirm_backup_changes(None, before, Preferences(state_limit_mb=2))

    def test_only_what_was_confirmed_goes_when_applied(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        # 確かめた後に退避を移すなどで置き場の中身が変わると、計画が変わる 計画どおりに
        # 消すと、確かめに出していない控えまで消える
        made = _fill(tmp_path / "本編.sme", 5)
        chosen = Preferences(state_limit_mb=1)
        confirmed = plan_trim_all(chosen)[:1]
        window._apply_preferences(chosen, confirmed)
        assert not made[0].exists()
        assert all(path.is_file() for path in made[1:])

    def test_nothing_confirmed_means_nothing_goes_when_applied(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        made = _fill(tmp_path / "本編.sme", 5)
        window._apply_preferences(Preferences(state_limit_mb=1))
        assert all(path.is_file() for path in made)

    def test_the_dialog_hands_over_what_was_confirmed(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 設定画面で見せた一覧が、そのまま当てる側へ渡る 渡らないと何も消えないか、
        # 見せていない物まで消える
        del qt_application
        _fill(tmp_path / "本編.sme", 5)
        shown = {p for item in plan_trim_all(Preferences(state_limit_mb=1)) for p in item.paths}
        assert shown
        monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.StandardButton.Yes)
        dialog = PreferencesDialog(Preferences())
        dialog._backups.state_limit_mb.setValue(1)
        dialog.accept()
        assert {p for item in dialog.approved_trim for p in item.paths} == shown


class TestReducingGenerationsCountsEveryFolder:
    def test_the_default_folder_is_counted_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選んだ置き場へ書けなかった保存の控えは既定の側にある 次にまた書けなければ、
        # そちらにも同じ世代数を当てて消す 数えないと、確かめに出していない控えが消える
        _fill(tmp_path / "本編.sme", 6)
        asked: list[str] = []

        def say_no(_parent: QWidget, _title: str, text: str, *_args: object) -> object:
            asked.append(text)
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", say_no)
        chosen = str(tmp_path / "選んだ置き場")
        before = Preferences(state_folder=chosen)
        after = Preferences(state_folder=chosen, backup_generations=2)
        assert not confirm_backup_changes(None, before, after)
        assert asked and "4 本" in asked[0]


class TestFindingWorkAfterAMove:
    def test_work_left_in_the_old_folder_is_found_next_time(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A から B へ変えたが B に書けず A の退避のまま続け、その後に落ちた 設定は B のまま
        # 次の起動が B と既定しか探さないと、その作業を復元に出せない
        del qt_application
        old_root, new_root = tmp_path / "A", tmp_path / "B"
        first = _window(Preferences(state_folder=str(old_root)))
        first.execute(RenameProject("移せなかった作業"))
        first.autosave()
        plain_save = RecoverySession.save

        def refuse_new(self: RecoverySession, project: Project, source: Path | None) -> None:
            if self.path.is_relative_to(new_root):
                raise OSError("書けない")
            plain_save(self, project, source)

        monkeypatch.setattr(RecoverySession, "save", refuse_new)
        first._apply_preferences(Preferences(state_folder=str(new_root)))
        assert first._recovery.path.is_relative_to(old_root)
        assert PreferenceStore().load().state_folder == str(new_root)
        # 落ちたことにする 錠を手放し、閉じても退避を消さない
        crashed = first._recovery
        lock = crashed._lock
        assert lock is not None
        lock.abandon()
        crashed._lock = None
        monkeypatch.setattr(crashed, "clear", lambda: None)
        first.close()

        offered: list[str] = []

        def look(dialog: RecoveryDialog) -> int:
            offered.extend(entry.name for entry in dialog._entries)
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(RecoveryDialog, "exec", look)
        second = _window()
        try:
            second.offer_recovery()
            assert "移せなかった作業" in offered
        finally:
            second.close()
