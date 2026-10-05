"""自動更新の画面の側 起動時に黙って確かめる・知らせる・尋ねる・再起動を断る

ネットワークは偽物の取り口 入れ替え係は起こさない（差し替える）
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from sashimono import __version__
from sashimono.core.io import load_project, save_project
from sashimono.core.io.locks import try_hold
from sashimono.core.io.serialize import FORMAT_VERSION, PRE_UPGRADE_SUFFIX
from sashimono.core.model import Project
from sashimono.ui import updates as updates_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.updates import ANSWER_NEXT_START, ANSWER_SKIP, UpdateController
from sashimono.ui.workspace import Preferences, PreferenceStore
from sashimono.update.fetch import MemoryTransport
from sashimono.update.package import APP_EXE, Layout, write_build_info
from sashimono.update.state import (
    STAGE_LOCK,
    SWAP_LOCK,
    UpdateState,
    UpdateStateStore,
    lock_path,
)
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
        self.controller._notify = self._notify  # type: ignore[method-assign]
        #: 手を止めさせない窓で知らせた物（尋ねずに済ませたこと）
        self.notified: list[tuple[str, str]] = []
        self.sleeper: subprocess.Popen[bytes] | None = None

    def _transport(self) -> MemoryTransport:
        self.transport_made += 1
        return self.transport

    def _inform(self, title: str, text: str) -> None:
        self.informed.append((title, text))

    def _notify(self, title: str, text: str) -> None:
        self.notified.append((title, text))

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
        state = ready.store.load()
        # 本人が選んだ予約として覚える 後で〔入れる前に尋ねる〕を入れても外さない
        assert state.apply_on_start and state.apply_chosen
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


class TestSwitchingSettings:
    """設定を切り替えたとき、待っている版と予約を今の好みにそろえる（表は docs）"""

    def test_stop_asking_reserves_the_ready_version(self, harness: _Harness) -> None:
        """確認ありで落とした後に〔入れる前に尋ねる〕を切る 説明どおり次の起動の頭で入れる"""
        harness.controller.start()
        assert not harness.store.load().apply_on_start
        harness.controller.apply_preferences(Preferences(update_confirm=False))
        state = harness.store.load()
        assert state.apply_on_start and not state.apply_chosen

    def test_turning_beta_off_drops_a_ready_beta(self, harness: _Harness) -> None:
        """ベータを待たせたままベータを切ると、次の起動でベータが入っていた"""
        beta = "9000.0.0b1"
        harness.layout.staged.mkdir()
        (harness.layout.staged / APP_EXE).write_bytes(b"MZ")
        write_build_info(harness.layout.staged, beta, "cp314")
        harness.store.save(UpdateState(ready_version=beta, apply_on_start=True, apply_chosen=True))
        harness.controller.apply_preferences(Preferences(update_beta=True))
        assert harness.controller.ready_version() == beta

        harness.controller.apply_preferences(Preferences(update_beta=False))
        state = harness.store.load()
        assert (state.ready_version, state.apply_on_start) == ("", False)
        assert not harness.layout.staged.exists()
        assert harness.controller.button.isHidden()

    def test_turning_beta_off_keeps_a_release(self, harness: _Harness) -> None:
        harness.controller.start()
        harness.controller.apply_preferences(Preferences(update_beta=False))
        assert harness.controller.ready_version() == NEWER


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

    @pytest.mark.parametrize(
        ("lock", "title"),
        [(SWAP_LOCK, "ほかの窓で入れ替えを始めています"), (STAGE_LOCK, "今は入れられません")],
    )
    def test_another_window_at_work_starts_nothing(
        self, ready: _Harness, lock: str, title: str
    ) -> None:
        """2 つの窓で入れ替えを選ぶと、入れ替え係が 2 つ起きて .previous を消し合う 起こさない"""
        held = try_hold(lock_path(lock, ready.store.path.parent))
        assert held is not None
        try:
            assert not ready.controller.restart_now()
        finally:
            held.release()
        assert ready.launched == []
        assert ready.informed and ready.informed[0][0] == title

    def test_losing_the_race_says_so(self, ready: _Harness) -> None:
        """ほぼ同時に起こして、ほかの窓の入れ替え係が先に錠を取った この窓は閉じれば済む"""
        result = ready.layout.install.parent / "result.txt"
        result.write_text("started\nalready-running\n", encoding="utf-8")
        ready.controller._waiter = lambda _launched: False
        assert not ready.controller.restart_now()
        assert ready.informed and ready.informed[0][0] == "ほかの窓で入れ替えを始めています"

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

    def test_asking_again_drops_only_the_automatic_reservation(self, window: MainWindow) -> None:
        """〔入れる前に尋ねる〕を入れたら、尋ねない設定が自動で付けた予約は外す
        本人が〔次の起動で入れる〕を選んだ予約は、もう答えてあるので残す
        """
        store = UpdateStateStore()
        store.save(UpdateState(ready_version=NEWER, apply_on_start=True))
        window._apply_preferences(Preferences(update_confirm=True))
        assert not store.load().apply_on_start

        store.save(UpdateState(ready_version=NEWER, apply_on_start=True, apply_chosen=True))
        window._apply_preferences(Preferences(update_confirm=True))
        assert store.load().apply_on_start

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

        from sashimono.update.flow import UpdateChoices, update_choices

        store.save(Preferences(update_confirm=False))
        assert update_choices() == UpdateChoices(check=True, confirm=False)
        store.save(Preferences())
        assert update_choices() == UpdateChoices()

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


def _put_script(install: Path, relative: str, text: str = "@揺れ\n--track0:量,0,100,0\n") -> Path:
    path = install / "scripts" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _appdata_scripts() -> Path:
    from sashimono.core import userdirs

    return userdirs.config_root() / "scripts"


class TestScriptsBesideTheExe:
    """exe の隣の ``scripts`` に置いた物を ``%APPDATA%`` 側へ移す（Issue #244）

    自動更新では写すが、zip を手で展開し直してフォルダごと入れ替えると消える 既定は起動の
    ときに尋ねずに移し、何をどこへ移したかを 1 度知らせる（利用者の決定）
    """

    @pytest.fixture
    def asked(self, harness: _Harness) -> list[str]:
        """移すかを尋ねた文面 答えは「移す」"""
        answers: list[str] = []

        def ask(text: str) -> bool:
            answers.append(text)
            return True

        harness.controller._ask_move = ask  # type: ignore[method-assign]
        harness.preferences = Preferences(update_check=False)
        return answers

    def test_the_default_is_to_move(self) -> None:
        assert Preferences().scripts_move == "auto"

    def test_the_setting_comes_back_and_breaks_safely(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(scripts_move="ask"))
        assert store.load().scripts_move == "ask"
        (tmp_path / "preferences.json").write_text(
            json.dumps({"scripts_move": "ぜんぶ消す"}), encoding="utf-8"
        )
        assert store.load().scripts_move == "auto"

    def test_the_dialog_returns_it(self, qt_application: QApplication) -> None:
        del qt_application
        dialog = PreferencesDialog(Preferences(scripts_move="off"))
        try:
            assert dialog.preferences().scripts_move == "off"
        finally:
            dialog.deleteLater()

    def test_they_are_moved_on_start_and_told_once(
        self, harness: _Harness, asked: list[str]
    ) -> None:
        mine = _put_script(harness.layout.install, "自分の/効果.anm2")
        rescanned: list[bool] = []
        harness.controller._rescan_scripts = lambda: rescanned.append(True)
        harness.controller.start()
        # 尋ねない 移して、何をどこへ移したかを知らせる
        assert asked == []
        assert not mine.exists()
        assert (_appdata_scripts() / "自分の" / "効果.anm2").is_file()
        assert rescanned == [True]
        # 手を止めさせない窓で知らせる（尋ねる窓ではない）
        assert harness.informed == [] and len(harness.notified) == 1
        assert "自分の/効果.anm2" in harness.notified[0][1]
        assert str(_appdata_scripts()) in harness.notified[0][1]
        # 2 回目の起動では何も言わない
        harness.controller.start()
        assert len(harness.notified) == 1

    def test_a_name_taken_in_appdata_is_kept_and_told_once(
        self, harness: _Harness, asked: list[str]
    ) -> None:
        mine = _put_script(harness.layout.install, "効果.anm2", "古い")
        _appdata_scripts().mkdir(parents=True)
        (_appdata_scripts() / "効果.anm2").write_text("新しい", encoding="utf-8")
        harness.controller.start()
        assert mine.read_text(encoding="utf-8") == "古い"
        assert (_appdata_scripts() / "効果.anm2").read_text(encoding="utf-8") == "新しい"
        assert len(harness.notified) == 1 and "上書きせず" in harness.notified[0][1]
        harness.controller.start()
        assert len(harness.notified) == 1

    def test_a_bundle_kept_together_is_told_once(self, harness: _Harness, asked: list[str]) -> None:
        """モジュールがぶつかる配布物は効果ごと残し、そう知らせる（PR #245 の Codex の指摘）"""
        effect = _put_script(harness.layout.install, "配布物/効果.anm2")
        _put_script(harness.layout.install, "配布物/common.lua", "return { v = 1 }")
        (_appdata_scripts() / "配布物").mkdir(parents=True)
        (_appdata_scripts() / "配布物" / "common.lua").write_text("return {}", encoding="utf-8")
        harness.controller.start()
        assert effect.is_file()
        assert len(harness.notified) == 1 and "一式のまま" in harness.notified[0][1]
        harness.controller.start()
        assert len(harness.notified) == 1

    def test_listing_runs_off_the_ui_thread_too(
        self, harness: _Harness, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """scripts 全体を辿って一覧を作るのも裏のスレッド（PR #245 の Codex の指摘）
        尋ねる設定では、一覧ができてから画面の側で尋ね、答えを受けてから裏で移す
        """
        import threading
        import time

        from sashimono.update.portable import user_script_files as real_list

        threads: dict[str, str] = {}

        def listing(install: Path) -> list[Path]:
            threads["list"] = threading.current_thread().name
            return real_list(install)

        def ask(text: str) -> bool:
            threads["ask"] = threading.current_thread().name
            return True

        monkeypatch.setattr(updates_module, "user_script_files", listing)
        mine = _put_script(harness.layout.install, "自分の/効果.anm2")
        controller = UpdateController(
            harness.window,
            preferences=lambda: Preferences(update_check=False, scripts_move="ask"),
            blockers=list,
            arguments=list,
            layout=harness.layout,
            store=harness.store,
            threaded=True,
        )
        controller._ask_move = ask  # type: ignore[method-assign]
        informed: list[str] = []

        def inform(title: str, text: str) -> None:
            informed.append(text)

        controller._inform = inform  # type: ignore[method-assign]
        try:
            assert controller.offer_script_move()
            deadline = time.monotonic() + 10
            while not informed and time.monotonic() < deadline:
                qt_application.processEvents()
                time.sleep(0.02)
            main = threading.main_thread().name
            assert threads["list"] != main and threads["ask"] == main
            assert informed and not mine.exists()
        finally:
            controller.deleteLater()

    def test_a_synced_folder_is_not_moved_automatically(
        self, harness: _Harness, asked: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """exe の隣が OneDrive などの中なら自動では移さない 元を消すと、ほかの機械からも消える
        1 度だけ知らせる 手で選んだときは、そのことも添えて尋ねてから移す
        """
        monkeypatch.setenv("OneDrive", str(harness.layout.install.parent))
        mine = _put_script(harness.layout.install, "自分の/効果.anm2")
        harness.controller.start()
        harness.controller.start()
        assert mine.is_file() and asked == []
        assert len(harness.notified) == 1 and "同期フォルダ" in harness.notified[0][1]
        assert harness.controller.offer_script_move(manual=True)
        assert len(asked) == 1 and "同期フォルダ" in asked[0]
        assert not mine.exists()

    def test_startup_does_not_walk_scripts_for_a_page_to_update_by_hand(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """起動時の自動の確認では詳しい注意を作らない

        scripts を辿らない（PR #245 の Codex の指摘）
        """
        monkeypatch.setattr(Layout, "writable", lambda _self: False)
        walked: list[str] = []

        def notes(new_abi: str) -> str:
            walked.append(new_abi)
            return ""

        monkeypatch.setattr(harness.controller, "_by_hand_notes", notes)
        harness.preferences = Preferences(scripts_move="off")
        harness.controller.start()
        assert walked == []
        assert "配布のページ" in harness.window.statusBar().currentMessage()

    def test_a_page_to_update_by_hand_is_prepared_off_the_ui_thread(
        self, harness: _Harness, qt_application: QApplication
    ) -> None:
        """手で確かめたときの注意（scripts を辿る・導入先を測る）は裏のスレッドで作る"""
        import threading
        import time

        controller = UpdateController(
            harness.window,
            preferences=Preferences,
            blockers=list,
            arguments=list,
            layout=harness.layout,
            store=harness.store,
            threaded=True,
        )
        threads: list[str] = []
        shown: list[str] = []

        def notes(abi: str) -> str:
            threads.append(threading.current_thread().name)
            return "注意"

        def inform(title: str, text: str) -> None:
            shown.append(text)

        controller._inform = inform  # type: ignore[method-assign]
        setattr(controller, "_by_hand_notes", notes)  # noqa: B010 - 型の上では別の関数
        try:
            controller._tell_by_hand(True, "新しい版があります", "https://example.invalid", "cp314")
            deadline = time.monotonic() + 10
            while not shown and time.monotonic() < deadline:
                qt_application.processEvents()
                time.sleep(0.02)
            assert threads and threads[0] != threading.main_thread().name
            assert shown and "注意" in shown[0]
        finally:
            controller.deleteLater()

    def test_the_reinstall_estimate_is_measured_off_the_ui_thread(
        self, harness: _Harness, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """入れる前の確認の見積もり（導入先の大きさを最大 1.5 秒測る）は裏のスレッドで作る"""
        import threading
        import time

        harness.controller.start()
        threads: list[str] = []

        def note(abi: str) -> str:
            threads.append(threading.current_thread().name)
            return ""

        monkeypatch.setattr(updates_module, "runtime_note", note)
        controller = UpdateController(
            harness.window,
            preferences=Preferences,
            blockers=list,
            arguments=list,
            layout=harness.layout,
            store=harness.store,
            threaded=True,
        )
        asked: list[str] = []

        def choose(_title: str, html: str, **_options: object) -> str:
            asked.append(html)
            return updates_module.ANSWER_LATER

        setattr(controller, "_choose", choose)  # noqa: B010 - 型の上では別の関数
        try:
            controller.offer()
            deadline = time.monotonic() + 10
            while not asked and time.monotonic() < deadline:
                qt_application.processEvents()
                time.sleep(0.02)
            assert threads and threads[0] != threading.main_thread().name
            assert asked
        finally:
            controller.deleteLater()

    def test_closing_waits_for_the_bundle_being_moved(
        self, harness: _Harness, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """閉じるときは、裏の移しが今の束を終えるのを待ち、新しい束には入らせない
        （PR #245 の Codex の指摘 元をよけたまま止まると、次の起動で入れ替えが先に走って消える）
        """
        import threading
        import time

        from sashimono.update.portable import move_user_scripts as real_move

        first = _put_script(harness.layout.install, "甲/効果.anm2")
        second = _put_script(harness.layout.install, "乙/効果.anm2")
        inside = threading.Event()
        release_move = threading.Event()

        def slow_move(install: Path, target: Path, **options: Any) -> object:
            stop = options["should_stop"]

            def checked() -> bool:
                inside.set()
                release_move.wait(10)
                return bool(stop())

            return real_move(install, target, should_stop=checked)

        monkeypatch.setattr(updates_module, "move_user_scripts", slow_move)
        controller = UpdateController(
            harness.window,
            preferences=lambda: Preferences(update_check=False),
            blockers=list,
            arguments=list,
            layout=harness.layout,
            store=harness.store,
            threaded=True,
        )
        try:
            assert controller.offer_script_move()
            # 一覧を作り終えた知らせは画面の側で受け取って、移しを裏へ出す
            deadline = time.monotonic() + 10
            while not inside.is_set() and time.monotonic() < deadline:
                qt_application.processEvents()
                time.sleep(0.02)
            assert inside.is_set()
            threading.Timer(0.3, release_move.set).start()
            assert controller.shutdown(timeout=10)
            # 閉じると決めた後は新しい束に入らない 1 つ目の束は入る前に止まったので残る
            assert first.is_file() and second.is_file()
            assert not list(harness.layout.install.glob(".scripts-removing-*"))
        finally:
            release_move.set()
            controller.deleteLater()
            qt_application.processEvents()

    def test_a_failed_launch_clears_only_its_own_marker_so_moves_are_not_held_10_minutes(
        self, harness: _Harness
    ) -> None:
        """画面から入れ替えるとき、入れ替え係を起こせなければ自分の「入れ替え係を待っている」印を
        外す 壊れて外れないと、exe の隣のスクリプトの移しが印の期限（10 分）まで止まる
        ほかの窓が置いた印は外さない（PR #245 の CodeRabbit の指摘 外すと、その窓の入れ替え係が
        swap.lock を取るまでの隙に移しが始まる）
        """
        from sashimono.core import userdirs
        from sashimono.update.portable import SWAP_PENDING, mark_swap_pending, swap_pending

        harness.controller.start()
        mark_swap_pending(userdirs.config_root(), __version__, "ほかの窓")

        def refuse(plan: SwapPlan) -> Launched:
            raise OSError("台本の実行が止められている")

        harness.controller._launcher = refuse
        assert not harness.controller.restart_now()
        left = sorted(p.name for p in userdirs.config_root().glob(f"{SWAP_PENDING}.*"))
        assert left == [f"{SWAP_PENDING}.ほかの窓"]
        assert swap_pending(userdirs.config_root())

    def test_a_finished_swap_s_marker_is_cleared_so_moves_are_not_held_10_minutes(
        self, harness: _Harness
    ) -> None:
        """入れ替えが済んだ（印の版と今の版が違う）印は起動の後に外す 外れないと、移しが印の
        期限（10 分）まで止まる 同じ版の印はまだ入れ替えていないので残す
        """
        from sashimono.core import userdirs
        from sashimono.update.portable import mark_swap_pending, swap_pending

        mark_swap_pending(userdirs.config_root(), __version__, "待っている")
        harness.controller.start()
        assert swap_pending(userdirs.config_root())  # まだ入れ替えていない
        (userdirs.config_root() / "scripts-move.swap-pending.待っている").unlink()
        mark_swap_pending(userdirs.config_root(), "0.0.1", "済んだ")
        harness.controller.start()
        assert not swap_pending(userdirs.config_root())  # 入れ替えが済んだ

    def test_no_move_while_the_swapper_runs(self, harness: _Harness) -> None:
        """入れ替え係が今の版のフォルダを付け替えている間は移さない（swap.lock）"""
        mine = _put_script(harness.layout.install, "自分の/効果.anm2")
        held = try_hold(lock_path(SWAP_LOCK, harness.store.path.parent))
        assert held is not None
        try:
            assert not harness.controller.offer_script_move()
        finally:
            held.release()
        assert mine.is_file()

    def test_the_editor_waits_for_the_move_when_closing(self) -> None:
        """編集画面は閉じるときに裏の移しを待つ（shutdown を呼ぶ）"""
        import inspect

        source = inspect.getsource(MainWindow.closeEvent)
        assert "self._updates.shutdown()" in source

    def test_restarting_waits_for_the_move(self, harness: _Harness) -> None:
        """移している最中に入れ替えると、写す側と移す側が同じファイルを同時に触る"""
        harness.controller.start()
        harness.controller._moving = True
        assert not harness.controller.restart_now()
        assert harness.launched == []
        assert "移しています" in harness.informed[-1][1]

    def test_moving_runs_off_the_ui_thread(
        self, harness: _Harness, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """走査・写し・照らし・元を消すのは裏のスレッド 大きな配布物で編集画面を固めない
        （PR #245 の Codex の指摘） 知らせとスクリプトの読み直しだけ画面の側へ戻す
        """
        import threading
        import time

        from sashimono.update.portable import move_user_scripts as real_move

        release_move = threading.Event()
        threads: list[str] = []

        def slow_move(install: Path, target: Path, **options: Any) -> object:
            threads.append(threading.current_thread().name)
            release_move.wait(10)
            return real_move(install, target, **options)

        monkeypatch.setattr(updates_module, "move_user_scripts", slow_move)
        mine = _put_script(harness.layout.install, "自分の/効果.anm2")
        controller = UpdateController(
            harness.window,
            preferences=lambda: Preferences(update_check=False),
            blockers=list,
            arguments=list,
            layout=harness.layout,
            store=harness.store,
            threaded=True,
        )
        notified: list[str] = []
        rescanned: list[str] = []

        def notify(title: str, text: str) -> None:
            notified.append(text)

        controller._notify = notify  # type: ignore[method-assign]
        controller._rescan_scripts = lambda: rescanned.append(threading.current_thread().name)
        try:
            began = time.monotonic()
            assert controller.offer_script_move()
            # 移し終えるのを待たずに戻る（画面は動き続ける）
            assert time.monotonic() - began < 2
            assert mine.is_file() and notified == []
            # 移している間にもう一度呼んでも、2 つ目は始めない
            assert not controller.offer_script_move()
            release_move.set()
            deadline = time.monotonic() + 10
            while not notified and time.monotonic() < deadline:
                qt_application.processEvents()
                time.sleep(0.02)
            assert threads and threads[0] != threading.main_thread().name
            assert len(notified) == 1 and not mine.exists()
            assert rescanned == [threading.main_thread().name]
        finally:
            release_move.set()
            controller.deleteLater()

    def test_the_notice_does_not_block_the_editor(
        self, harness: _Harness, qt_application: QApplication
    ) -> None:
        """起動して黙って出す知らせが親の窓を塞ぐと、出ている間は編集画面を閉じられない
        （配布版の確かめ check_clean_machine.ps1 が窓へ閉じる知らせを送って待つ）
        """
        del qt_application
        controller = UpdateController(
            harness.window,
            preferences=Preferences,
            blockers=list,
            arguments=list,
            layout=harness.layout,
            store=harness.store,
            threaded=False,
        )
        controller._notify("スクリプトの置き場", "見本")
        boxes = [
            w for w in QApplication.topLevelWidgets() if w.windowTitle() == "スクリプトの置き場"
        ]
        try:
            assert boxes and all(box.isVisible() for box in boxes)
            assert QApplication.activeModalWidget() is None
        finally:
            for box in boxes:
                box.close()
            controller.deleteLater()

    def test_asking_is_once_for_the_same_files(self, harness: _Harness, asked: list[str]) -> None:
        harness.preferences = Preferences(update_check=False, scripts_move="ask")

        def decline(text: str) -> bool:
            asked.append(text)
            return False

        harness.controller._ask_move = decline  # type: ignore[method-assign]
        mine = _put_script(harness.layout.install, "効果.anm2")
        harness.controller.start()
        harness.controller.start()
        assert len(asked) == 1 and "1 個" in asked[0]
        assert mine.is_file()
        # 新しく置いた物があれば、もう 1 度だけ尋ねる
        _put_script(harness.layout.install, "次の.anm2")
        harness.controller.start()
        assert len(asked) == 2 and "2 個" in asked[1]

    def test_off_does_nothing(self, harness: _Harness, asked: list[str]) -> None:
        harness.preferences = Preferences(update_check=False, scripts_move="off")
        mine = _put_script(harness.layout.install, "効果.anm2")
        harness.controller.start()
        assert mine.is_file() and asked == [] and harness.informed == [] and not harness.notified

    def test_the_menu_asks_whatever_the_setting(self, harness: _Harness, asked: list[str]) -> None:
        harness.preferences = Preferences(scripts_move="off")
        _put_script(harness.layout.install, "効果.anm2")
        assert harness.controller.offer_script_move(manual=True)
        assert len(asked) == 1
        assert (_appdata_scripts() / "効果.anm2").is_file()

    def test_the_menu_says_when_there_is_nothing(self, harness: _Harness) -> None:
        harness.controller.offer_script_move(manual=True)
        assert harness.informed and harness.informed[0][0] == "移す物はありません"

    def test_the_bundled_readme_is_not_theirs(self, harness: _Harness, asked: list[str]) -> None:
        _put_script(harness.layout.install, "README.txt", "同梱の説明")
        harness.controller.start()
        assert harness.informed == [] and harness.notified == [] and asked == []
        assert (harness.layout.install / "scripts" / "README.txt").is_file()

    def test_not_over_another_question(
        self, harness: _Harness, asked: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """退避の復元などを尋ねている最中には動かない 次の起動へ回す"""
        monkeypatch.setattr(QApplication, "activeModalWidget", lambda: harness.window)
        mine = _put_script(harness.layout.install, "効果.anm2")
        harness.controller.start()
        assert mine.is_file() and harness.notified == [] and asked == []
        assert harness.store.load().scripts_offered == ()

    def test_the_editor_has_the_menu(self, qt_application: QApplication) -> None:
        del qt_application
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert "互換/exe の隣のスクリプトを移す…" in window._actions
            assert window.updates._rescan_scripts == window.rescan_scripts
        finally:
            window.close()

    def test_rolling_back_carries_them_to_the_version_rolled_back_to(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """戻すと今の版は previous に回り、次の更新で消える 戻る先の版へ写しておく"""
        harness.layout.previous.mkdir()
        (harness.layout.previous / APP_EXE).write_bytes(b"MZ")
        _put_script(harness.layout.install, "効果.anm2")
        monkeypatch.setattr(
            harness.controller, "_choose", lambda *_a, **_k: updates_module.ANSWER_NOW
        )
        monkeypatch.setattr(QApplication, "quit", lambda: None)
        assert harness.controller.roll_back()
        assert (harness.layout.previous / "scripts" / "効果.anm2").is_file()

    def test_files_that_cannot_be_carried_stop_the_update(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """写せないまま入れ替えると、本人の物は次の更新で消える版にだけ残る 入れ替え係を
        起こさずに止め、何を写せなかったかを知らせる（PR #245 の CodeRabbit の指摘）
        """
        harness.preferences = Preferences(scripts_move="off")
        harness.controller.start()
        assert harness.controller.ready_version() == NEWER
        _put_script(harness.layout.install, "自分の/効果.anm2")
        (harness.layout.staged / "scripts").mkdir(exist_ok=True)
        (harness.layout.staged / "scripts" / "自分の").write_text("塞ぐ", encoding="utf-8")
        monkeypatch.setattr(QApplication, "quit", lambda: pytest.fail("終わってはいけない"))
        assert not harness.controller.restart_now()
        assert harness.launched == []
        assert harness.informed and harness.informed[-1][0] == "入れ替えを止めました"
        assert "自分の/効果.anm2" in harness.informed[-1][1]
        assert harness.controller.ready_version() == NEWER

    def test_files_that_cannot_be_carried_stop_the_rollback(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.preferences = Preferences(scripts_move="off")
        harness.layout.previous.mkdir()
        (harness.layout.previous / APP_EXE).write_bytes(b"MZ")
        (harness.layout.previous / "scripts").mkdir()
        (harness.layout.previous / "scripts" / "自分の").write_text("塞ぐ", encoding="utf-8")
        _put_script(harness.layout.install, "自分の/効果.anm2")
        monkeypatch.setattr(
            harness.controller, "_choose", lambda *_a, **_k: updates_module.ANSWER_NOW
        )
        monkeypatch.setattr(QApplication, "quit", lambda: pytest.fail("終わってはいけない"))
        assert not harness.controller.roll_back()
        assert harness.launched == []
        assert harness.informed[-1][0] == "入れ替えを止めました"
        assert __version__ not in harness.store.load().skipped

    def test_a_swap_stopped_before_the_window_is_told_once(self, harness: _Harness) -> None:
        """起動の頭で止めた入れ替えは、窓を出した後に 1 度だけ、窓を塞がずに知らせる"""
        harness.preferences = Preferences(update_check=False, scripts_move="off")
        harness.store.save(UpdateState(pending_notice="写せなかったので止めました"))
        harness.controller.start()
        harness.controller.start()
        assert harness.notified == [("入れ替えを止めました", "写せなかったので止めました")]
        assert harness.informed == []

    def test_rolling_back_takes_the_edited_one(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """移さない設定で、更新した後に直した物は、戻した版でも直した中身になる"""
        harness.preferences = Preferences(scripts_move="off")
        harness.layout.previous.mkdir()
        (harness.layout.previous / APP_EXE).write_bytes(b"MZ")
        _put_script(harness.layout.previous, "効果.anm2", "直す前")
        _put_script(harness.layout.install, "効果.anm2", "直した")
        monkeypatch.setattr(
            harness.controller, "_choose", lambda *_a, **_k: updates_module.ANSWER_NOW
        )
        monkeypatch.setattr(QApplication, "quit", lambda: None)
        assert harness.controller.roll_back()
        carried = harness.layout.previous / "scripts" / "効果.anm2"
        assert carried.read_text(encoding="utf-8") == "直した"

    def test_a_page_to_update_by_hand_warns_about_them(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """自動では入れ替えられない場所 配布のページから手で入れ替える前に、消える物を言う"""
        monkeypatch.setattr(Layout, "writable", lambda _self: False)
        harness.preferences = Preferences(scripts_move="off")
        _put_script(harness.layout.install, "効果.anm2")
        harness.controller.check_now()
        assert harness.informed
        text = harness.informed[0][1]
        assert "配布のページ" in text and "1 個" in text
        assert "フォルダごと入れ替えると消えます" in text


class TestPythonChangeNote:
    """Python が上がる版では、入れ直しの機能・大きさ・時間を入れる前の確認に添える"""

    def test_the_note_says_what_and_how_much(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "runtime"
        (target / "faster_whisper-1.1.0.dist-info").mkdir(parents=True)
        (target / "faster_whisper").mkdir()
        (target / "faster_whisper" / "_ext.cp314-win_amd64.pyd").write_bytes(b"x" * 4096)
        monkeypatch.setattr(updates_module, "runtime_target_dir", lambda: target)
        note = updates_module.runtime_note("cp399")
        assert "cp399" in note and "入れ直しが要ります" in note
        assert "入れ直しが要るのは 字幕起こし" in note
        assert "今入れてある分を測った値" in note and "Mbps なら約" in note

    def test_the_same_python_says_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sashimono.runtime import python_abi

        (tmp_path / "x").mkdir()
        monkeypatch.setattr(updates_module, "runtime_target_dir", lambda: tmp_path)
        assert updates_module.runtime_note(python_abi()) == ""

    def test_the_offer_shows_every_line(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """入れる前の確認は字を HTML で出す 改行のまま渡すと 1 行に潰れる"""
        harness.controller.start()
        harness.store.save(replace(harness.store.load(), ready_python_abi="cp399"))
        monkeypatch.setattr(updates_module, "runtime_note", lambda abi: f"入れ直し {abi}\n2 行目")
        shown: list[str] = []

        def choose(_title: str, html: str, **_options: object) -> str:
            shown.append(html)
            return updates_module.ANSWER_LATER

        monkeypatch.setattr(harness.controller, "_choose", choose)
        harness.controller.offer()
        assert shown and "入れ直し cp399<br>2 行目" in shown[0]
