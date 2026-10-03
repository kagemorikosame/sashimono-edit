"""自動更新の画面の側 起動時に黙って確かめる・知らせる・尋ねる・再起動を断る

ネットワークは偽物の取り口 入れ替え係は起こさない（差し替える）
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from sashimono import __version__
from sashimono.core.io import load_project, save_project
from sashimono.core.io.serialize import FORMAT_VERSION, PRE_UPGRADE_SUFFIX
from sashimono.core.model import Project
from sashimono.ui import updates as updates_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.updates import ANSWER_NEXT_START, ANSWER_SKIP, UpdateController
from sashimono.ui.workspace import Preferences, PreferenceStore
from sashimono.update.fetch import MemoryTransport
from sashimono.update.package import APP_EXE, Layout
from sashimono.update.state import UpdateState, UpdateStateStore
from sashimono.update.swap import Launched, SwapPlan
from tests.update.helpers import release

NEWER = "9000.0.0"


class _Window(QMainWindow):
    """閉じるかどうかを選べる窓 保存を尋ねて取り消された所をまねる"""

    def __init__(self) -> None:
        super().__init__()
        self.allow_close = True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self.allow_close:
            event.accept()
        else:
            event.ignore()


class _Harness:
    """controller と、その周りの偽物"""

    def __init__(self, tmp_path: Path, preferences: Preferences, *, keys: bool = True) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.transport, self.manifest = release(self.key, NEWER)
        self.transport_made = 0
        install = tmp_path / "Programs" / "Sashimono"
        install.mkdir(parents=True)
        (install / APP_EXE).write_bytes(b"MZ")
        self.layout = Layout(install)
        self.store = UpdateStateStore(tmp_path / "state.json")
        self.preferences = preferences
        self.blockers: list[str] = []
        #: 保存の確認の答え 偽なら取り消した
        self.confirm = True
        #: 確認と入れ替え係を起こした順
        self.events: list[str] = []
        self.informed: list[tuple[str, str]] = []
        self.launched: list[SwapPlan] = []
        self.window = _Window()
        self.controller = UpdateController(
            self.window,
            preferences=lambda: self.preferences,
            blockers=lambda: list(self.blockers),
            arguments=lambda: ["作品.sme"],
            transport=self._transport,
            keys=[self.key.public_key()] if keys else [],
            layout=self.layout,
            store=self.store,
            clock=lambda: 1_000_000.0,
            threaded=False,
            launcher=self._launch,
            waiter=lambda _launched: True,
            confirm_close=self._confirm,
        )
        self.controller._inform = self._inform  # type: ignore[method-assign]
        self.sleeper: subprocess.Popen[bytes] | None = None

    def _transport(self) -> MemoryTransport:
        self.transport_made += 1
        return self.transport

    def _inform(self, title: str, text: str) -> None:
        self.informed.append((title, text))

    def _confirm(self) -> bool:
        self.events.append("confirm")
        return self.confirm

    def _launch(self, plan: SwapPlan) -> Launched:
        self.events.append("launch")
        self.launched.append(plan)
        self.sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return Launched(self.sleeper, self.layout.install.parent / "result.txt")

    def close(self) -> None:
        if self.sleeper is not None and self.sleeper.poll() is None:
            self.sleeper.kill()
            self.sleeper.wait()
        self.window.deleteLater()


@pytest.fixture
def harness(tmp_path: Path, qt_application: QApplication) -> Iterator[_Harness]:
    del qt_application
    made = _Harness(tmp_path, Preferences())
    yield made
    made.close()


class TestThePreferences:
    def test_the_defaults_are_the_safe_side(self) -> None:
        """知らない人が困らない側 確かめる・ベータは受け取らない・入れる前に尋ねる"""
        plain = Preferences()
        assert (plain.update_check, plain.update_beta, plain.update_confirm) == (True, False, True)

    def test_they_come_back(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        chosen = Preferences(update_check=False, update_beta=True, update_confirm=False)
        store.save(chosen)
        assert store.load() == chosen

    def test_broken_values_fall_back(self, tmp_path: Path) -> None:
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"update_check": "no", "update_beta": 1}), encoding="utf-8")
        loaded = PreferenceStore(path).load()
        assert (loaded.update_check, loaded.update_beta) == (True, False)

    def test_the_dialog_shows_and_returns_them(self, qt_application: QApplication) -> None:
        del qt_application
        chosen = Preferences(update_check=False, update_beta=True, update_confirm=False)
        dialog = PreferencesDialog(chosen)
        try:
            returned = dialog.preferences()
        finally:
            dialog.deleteLater()
        assert (returned.update_check, returned.update_beta, returned.update_confirm) == (
            False,
            True,
            False,
        )


class TestCheckingOnStart:
    def test_a_newer_version_is_fetched_quietly(self, harness: _Harness) -> None:
        """落とすのは黙って行う 入れる（再起動が要る）のは本人の合図を待つ"""
        harness.controller.start()
        assert harness.layout.staged_version() == NEWER
        state = harness.store.load()
        assert state.ready_version == NEWER and not state.apply_on_start
        assert not harness.controller.button.isHidden()
        assert NEWER in harness.controller.button.text()
        assert harness.informed == []
        assert harness.launched == []

    def test_turned_off_it_does_not_ask(self, harness: _Harness) -> None:
        """切ったら本当に止まる（取りにも行かない） 締め切り前の人が今の版に留まれる"""
        harness.preferences = Preferences(update_check=False)
        harness.controller.start()
        assert harness.transport_made == 0
        assert harness.transport.requested == []

    def test_not_again_within_hours(self, harness: _Harness) -> None:
        harness.store.save(UpdateState(last_checked=1_000_000.0 - 60))
        harness.controller.start()
        assert harness.transport.requested == []

    def test_a_build_without_keys_does_nothing(
        self, tmp_path: Path, qt_application: QApplication
    ) -> None:
        """公開鍵の入っていない版は確かめない（黙って何もしない）"""
        del qt_application
        made = _Harness(tmp_path, Preferences(), keys=False)
        try:
            made.controller.start()
            assert made.transport.requested == []
            assert made.informed == []
        finally:
            made.close()

    def test_failures_are_silent(self, harness: _Harness) -> None:
        """繋がらない・署名が通らないことを、起動のたびに知らせない"""
        harness.transport.pages.clear()
        harness.controller.start()
        assert harness.informed == []
        assert harness.controller.button.isHidden()
        # 失敗した回も時刻を残す 繋がらない間に起動のたびに取りに行かない
        assert harness.store.load().last_checked == 1_000_000.0

    def test_not_asking_means_next_start(self, harness: _Harness) -> None:
        """尋ねない設定では、次の起動の頭で入れる 今の編集の途中では再起動しない"""
        harness.preferences = Preferences(update_confirm=False)
        harness.controller.start()
        assert harness.store.load().apply_on_start
        assert harness.launched == []

    def test_an_unwritable_place_is_sent_to_the_page(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Layout, "writable", lambda _self: False)
        harness.controller.start()
        assert harness.layout.staged_version() is None
        assert "配布のページ" in harness.window.statusBar().currentMessage()

    def test_a_finished_update_is_told(self, harness: _Harness) -> None:
        harness.store.save(UpdateState(ready_version=__version__, last_checked=1_000_000.0))
        harness.controller.start()
        assert "更新しました" in harness.window.statusBar().currentMessage()


class TestCheckingByHand:
    def test_up_to_date_is_said(self, harness: _Harness) -> None:
        harness.transport, _ = release(harness.key, __version__)
        harness.controller.check_now()
        assert harness.informed and "最新" in harness.informed[0][1]

    def test_a_failure_is_explained(self, harness: _Harness) -> None:
        harness.transport.pages.clear()
        harness.controller.check_now()
        assert harness.informed and harness.informed[0][0] == "更新を確かめられませんでした"

    def test_without_keys_it_says_why(self, tmp_path: Path, qt_application: QApplication) -> None:
        del qt_application
        made = _Harness(tmp_path, Preferences(), keys=False)
        try:
            made.controller.check_now()
            assert made.informed and "鍵" in made.informed[0][1]
            assert made.transport.requested == []
        finally:
            made.close()


class TestOffering:
    @pytest.fixture
    def ready(self, harness: _Harness) -> _Harness:
        harness.controller.start()
        assert harness.controller.ready_version() == NEWER
        return harness

    def test_next_start_is_remembered(
        self, ready: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ready.controller, "_choose", lambda *_a, **_k: ANSWER_NEXT_START)
        ready.controller.offer()
        assert ready.store.load().apply_on_start
        assert ready.launched == []

    def test_a_skipped_version_is_dropped(
        self, ready: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ready.controller, "_choose", lambda *_a, **_k: ANSWER_SKIP)
        ready.controller.offer()
        state = ready.store.load()
        assert NEWER in state.skipped and state.ready_version == ""
        assert not ready.layout.staged.exists()
        assert ready.controller.button.isHidden()


class TestRestarting:
    @pytest.fixture
    def ready(self, harness: _Harness) -> _Harness:
        harness.controller.start()
        return harness

    def test_busy_work_refuses_the_restart(self, ready: _Harness) -> None:
        """編集の途中の作業（AI・字幕起こし）が動いている間は、勝手に再起動しない"""
        ready.blockers = ["字幕を起こしています"]
        assert not ready.controller.restart_now()
        assert ready.launched == []
        assert ready.informed and "字幕を起こしています" in ready.informed[0][1]

    def test_the_swapper_gets_the_plan(
        self, ready: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        quit_called: list[bool] = []
        monkeypatch.setattr(QApplication, "quit", lambda: quit_called.append(True))
        assert ready.controller.restart_now()
        plan = ready.launched[0]
        assert (plan.mode, tuple(plan.arguments)) == ("apply", ("作品.sme",))
        assert quit_called == [True]

    def test_the_save_question_comes_before_the_swapper(
        self, ready: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """入れ替え係を先に起こすと、保存の確認で迷っている間に待ちを諦めて今の版を起こし、
        本体が 2 つ動く 確認が済んでから起こす
        """
        monkeypatch.setattr(QApplication, "quit", lambda: None)
        assert ready.controller.restart_now()
        assert ready.events == ["confirm", "launch"]

    def test_a_cancelled_save_question_starts_nothing(self, ready: _Harness) -> None:
        """保存の確認で取り消したら、入れ替え係を起こさない（起こしてから止めるのでもない）"""
        ready.confirm = False
        assert not ready.controller.restart_now()
        assert ready.launched == []
        assert ready.store.load().ready_version == NEWER

    def test_a_cancelled_rollback_is_not_remembered(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """戻すのを取り消したのに、今の版が「飛ばす版」に残ると、次の更新を受け取れない"""
        harness.layout.previous.mkdir()
        (harness.layout.previous / APP_EXE).write_bytes(b"MZ")
        monkeypatch.setattr(
            harness.controller, "_choose", lambda *_a, **_k: updates_module.ANSWER_NOW
        )
        harness.confirm = False
        assert not harness.controller.roll_back()
        assert __version__ not in harness.store.load().skipped
        assert harness.launched == []

    def test_a_refused_close_stops_the_swapper(self, ready: _Harness) -> None:
        """確認の後でも窓が閉じられなければ、入れ替え係を止める 残すと後で閉じた所で入れ替わる"""
        ready.window.allow_close = False
        assert not ready.controller.restart_now()
        assert ready.sleeper is not None
        assert ready.sleeper.wait(timeout=10) is not None

    def test_a_helper_that_does_not_start_keeps_the_app(self, ready: _Harness) -> None:
        """台本の実行が止められた機械で、本体だけ終わって誰も起こし直さない、を防ぐ"""
        ready.controller._waiter = lambda _launched: False
        assert not ready.controller.restart_now()
        assert ready.informed and "PowerShell" in ready.informed[0][1]

    def test_rolling_back_needs_a_previous_version(self, harness: _Harness) -> None:
        assert not harness.controller.roll_back()
        assert harness.informed and "前の版" in harness.informed[0][0]

    def test_rolling_back_skips_this_version(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """戻した版を、次の確認ですぐ入れ直さない"""
        harness.layout.previous.mkdir()
        (harness.layout.previous / APP_EXE).write_bytes(b"MZ")
        monkeypatch.setattr(
            harness.controller, "_choose", lambda *_a, **_k: updates_module.ANSWER_NOW
        )
        monkeypatch.setattr(QApplication, "quit", lambda: None)
        assert harness.controller.roll_back()
        assert harness.launched[0].mode == "rollback"
        assert __version__ in harness.store.load().skipped


class TestTheEditor:
    @pytest.fixture
    def window(self, qt_application: QApplication) -> Iterator[MainWindow]:
        del qt_application
        created = MainWindow(Project.create(), confirm_unsaved=False)
        yield created
        created.close()

    def test_turning_checks_off_cancels_the_reservation(self, window: MainWindow) -> None:
        """切った時点で〔次の起動で入れる〕の予約も外す 切ったのに次の起動で入れ替わらない"""
        store = UpdateStateStore()
        store.save(UpdateState(ready_version=NEWER, apply_on_start=True))
        window._apply_preferences(Preferences(update_check=False))
        assert not store.load().apply_on_start
        assert store.load().ready_version == NEWER

    def test_the_start_reads_the_same_setting(self) -> None:
        """起動の頭（Qt を読む前）は設定のファイルを直に読む 名前や読み方が食い違うと、
        切ったのに入れ替わる
        """
        from sashimono.update.flow import PREFERENCES_FILE, updates_allowed

        store = PreferenceStore()
        assert store.path.name == PREFERENCES_FILE
        assert updates_allowed()
        store.save(Preferences(update_check=False))
        assert not updates_allowed()
        store.save(Preferences(update_check=True))
        assert updates_allowed()

    def test_the_help_menu_has_both(self, window: MainWindow) -> None:
        assert "ヘルプ/更新を確かめる…" in window._actions
        assert "ヘルプ/前の版に戻す…" in window._actions

    def test_an_idle_editor_has_no_blockers(self, window: MainWindow) -> None:
        assert window.update_blockers() == []

    def test_a_working_assistant_blocks(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(type(window._chat), "working", property(lambda _self: True))
        assert window.update_blockers()

    def test_closing_for_an_update_asks_only_once(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """保存の確認は入れ替え係を起こす前に済ませた 閉じるときにもう一度尋ねると、
        そこで取り消されたとき入れ替え係だけが残る
        """
        del qt_application
        from PySide6.QtWidgets import QMessageBox

        from sashimono.core.commands import AddTrack
        from sashimono.core.model import Track, TrackKind

        window = MainWindow(Project.create(), confirm_unsaved=True)
        window.document.execute(AddTrack(Track(TrackKind.VIDEO, "V9")))
        assert window.is_modified

        def asked(*_args: object) -> QMessageBox.StandardButton:
            raise AssertionError("閉じるときにもう一度尋ねた")

        monkeypatch.setattr(QMessageBox, "question", asked)
        assert window._close_for_update()

    def test_the_editor_asks_before_the_swapper(self, window: MainWindow) -> None:
        """編集画面は保存の確認（_confirm_discard）を入れ替え係より先に渡している"""
        assert window.updates._confirm_close == window._confirm_discard
        assert window.updates._close_window == window._close_for_update

    def test_saving_an_older_file_keeps_a_copy(
        self, tmp_path: Path, qt_application: QApplication
    ) -> None:
        """自動更新の後に前の版へ戻した人が、上げる前の作品を開けるように"""
        del qt_application
        path = tmp_path / "作品.sme"
        save_project(Project.create(), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = FORMAT_VERSION - 1
        path.write_text(json.dumps(data), encoding="utf-8")
        window = MainWindow(load_project(path), path=path, confirm_unsaved=False)
        try:
            assert window.save_project()
        finally:
            window.close()
        copy = tmp_path / f"作品.sme{PRE_UPGRADE_SUFFIX}"
        assert json.loads(copy.read_text(encoding="utf-8"))["version"] == FORMAT_VERSION - 1
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == FORMAT_VERSION
