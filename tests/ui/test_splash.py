"""起動画面（#526）

起動画面は「起動できているのか分からない」を消すための物 壊れ方は 3 つ
重い部品を読んだ後に出る（出るのが遅れて意味が無い）・編集画面や確かめの窓を隠したまま残る・
起動に失敗したのに残って固まって見える どれも画面を出さずに、出す順と閉じる時で見る
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

import sashimono.ui.main_window as main_window_module
from sashimono import app
from sashimono.core.io import save_project
from sashimono.core.model import Project
from sashimono.ui.splash import (
    SPLASH_MOTION_DEFAULT,
    STEP_PARTS,
    STEP_PROJECT,
    STEP_WINDOW,
    STILL_MESSAGE,
    SplashChoices,
    StartupSplash,
    showing_splash,
    splash_choices,
)
from sashimono.ui.theme import THEME_DARK
from sashimono.ui.workspace import Preferences, PreferenceStore

ROOT = Path(__file__).resolve().parents[2]

#: 起動画面を出す前に読んではいけない物 どれも冷えた機械で秒単位で掛かる
#: numpy は入口（``sashimono.app``）が字幕起こしの部品を読むときにもう読んでいる 入口の
#: 読み込み全体で温まって 0.3 秒ほど（2026-11-24 に測った） ``QApplication`` までの 0.29 秒に
#: 入っている数なので、ここでは数えない
HEAVY = ("sashimono.ui.main_window", "sashimono.engine.gpu", "av", "OpenGL", "PIL")


class TestItComesUpBeforeTheHeavyParts:
    def test_what_the_splash_needs_does_not_load_the_editor(self) -> None:
        # 壊れると、起動画面を出す前に編集画面の部品を読み終え、出す意味が無くなる
        # （ui の頭が main_window を読んでいた・サーフェス形式が gpu の頭を読んでいた）
        code = (
            "import sys\n"
            "import sashimono.ui.splash, sashimono.ui.theme, sashimono.engine.gl_format\n"
            "import sashimono.app\n"
            f"print([name for name in {HEAVY!r} if name in sys.modules])\n"
        )
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            env=environment,
            timeout=300,
            check=False,
        )
        assert completed.stdout.strip() == "[]", completed.stderr


class TestTheChoicesAreReadEarly:
    @pytest.mark.parametrize(
        "content",
        [
            None,
            "{}",
            json.dumps({"theme": "light", "splash_motion": False}),
            json.dumps({"theme": "system"}),
            json.dumps({"theme": "紫", "splash_motion": "no"}),
            json.dumps([1, 2]),
            "{壊れた",
        ],
    )
    def test_the_early_read_agrees_with_the_preferences(
        self, tmp_path: Path, content: str | None
    ) -> None:
        # 壊れると、起動画面だけ別のテーマで描かれるか、設定で切った動きが起動画面で動く
        path = tmp_path / "preferences.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        early = splash_choices(path)
        full = PreferenceStore(path).load()
        assert (early.theme, early.motion) == (full.theme, full.splash_motion)

    def test_the_defaults_are_the_same_as_the_preferences(self) -> None:
        # 既定を 2 か所に持つと、片方だけ変えたときに食い違う
        assert Preferences().splash_motion is SPLASH_MOTION_DEFAULT is True
        assert SplashChoices().theme == Preferences().theme == THEME_DARK

    def test_it_reads_the_file_the_preferences_are_saved_to(self) -> None:
        # 名前が違うと、何を選んでも起動画面は既定のまま
        PreferenceStore().save(Preferences(theme="light", splash_motion=False))
        assert splash_choices() == SplashChoices(theme="light", motion=False)


class TestTheMessage:
    def test_with_motion_the_steps_change_the_message(self, qt_application: QApplication) -> None:
        # 壊れると、動きを入れていても文字が変わらず固まって見える
        del qt_application
        splash = StartupSplash(motion=True, steps=3)
        assert splash.message() == STILL_MESSAGE
        splash.step(STEP_PARTS)
        assert splash.message() == f"{STEP_PARTS}（1/3）"
        splash.step(STEP_WINDOW)
        assert splash.message() == f"{STEP_WINDOW}（2/3）"
        splash.deleteLater()

    def test_without_motion_it_stays_still(self, qt_application: QApplication) -> None:
        # 切っても動くなら、設定がある方が質が悪い
        del qt_application
        splash = StartupSplash(motion=False)
        for step in (STEP_PARTS, STEP_PROJECT, STEP_WINDOW):
            splash.step(step)
        assert splash.message() == STILL_MESSAGE
        splash.deleteLater()

    def test_a_click_does_not_hide_it(self, qt_application: QApplication) -> None:
        # 元の QSplashScreen は押されると隠れ、後の段で出し直さない 待ちかねて押した人ほど
        # 起動しているかが分からなくなる（PR #550 の指摘）
        splash = StartupSplash()
        splash.show()
        qt_application.processEvents()
        QTest.mouseClick(splash, Qt.MouseButton.LeftButton)
        assert splash.isVisible()
        splash.close()
        splash.deleteLater()

    def test_it_does_not_stay_on_top(self, qt_application: QApplication) -> None:
        # 前に出し続けると、起動の途中で出る確かめの窓やほかのソフトの窓を隠す
        del qt_application
        splash = StartupSplash()
        assert not splash.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        splash.deleteLater()


class _Recorder:
    """偽の編集画面が作られた時と出された時に、起動画面が出ていたかを覚える"""

    def __init__(self) -> None:
        self.splash: StartupSplash | None = None
        self.events: list[tuple[str, bool]] = []
        self.opened: list[tuple[Project | None, Path | None]] = []

    def window_type(self, fail: bool = False) -> type[QWidget]:
        recorder = self

        class FakeWindow(QWidget):
            def __init__(self, project: Project | None = None, path: Path | None = None) -> None:
                splash = recorder.splash
                recorder.events.append(("作る", splash is not None and splash.isVisible()))
                if fail:
                    raise RuntimeError("部品が読めない")
                super().__init__()
                recorder.opened.append((project, path))

            def show(self) -> None:
                super().show()
                splash = recorder.splash
                recorder.events.append(("出す", splash is not None and splash.isVisible()))

        return FakeWindow


class TestTheStart:
    def _open(
        self,
        application: QApplication,
        monkeypatch: pytest.MonkeyPatch,
        arguments: list[str],
        *,
        fail: bool = False,
    ) -> tuple[_Recorder, StartupSplash | None, QWidget | None]:
        recorder = _Recorder()
        monkeypatch.setattr(main_window_module, "MainWindow", recorder.window_type(fail))
        choices = SplashChoices()
        kept: StartupSplash | None = None
        window: QWidget | None = None
        with showing_splash(application, choices) as splash:
            recorder.splash = kept = splash
            window = app._open_editor(application, arguments, splash, choices)
        return recorder, kept, window

    def test_the_splash_is_up_while_the_window_is_made_and_gone_after(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、起動画面が出ないまま待たせるか、編集画面の上に残って確かめの窓
        # （復元・素材が見つからない）を隠す 確かめの窓は起動画面を閉じた後に出す
        recorder, splash, window = self._open(qt_application, monkeypatch, ["sashimono"])
        assert recorder.events == [("作る", True), ("出す", True)]
        assert splash is not None and not splash.isVisible()
        assert window is not None
        window.deleteLater()

    def test_a_given_project_is_opened_under_the_splash(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 作品を渡したときは、開き終えるまで起動画面を出しておく
        path = tmp_path / "作品.sme"
        save_project(Project.create(), path)
        recorder, splash, window = self._open(qt_application, monkeypatch, ["sashimono", str(path)])
        ((project, opened_path),) = recorder.opened
        assert project is not None and opened_path == path
        assert splash is not None and not splash.isVisible()
        assert window is not None
        window.deleteLater()

    def test_a_failed_start_does_not_leave_the_splash(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、起動に失敗した後も起動画面だけが残り、固まったように見える
        recorder = _Recorder()
        monkeypatch.setattr(main_window_module, "MainWindow", recorder.window_type(fail=True))
        choices = SplashChoices()
        kept: StartupSplash | None = None
        with (
            pytest.raises(RuntimeError, match="部品が読めない"),
            showing_splash(qt_application, choices) as splash,
        ):
            recorder.splash = kept = splash
            app._open_editor(qt_application, ["sashimono"], splash, choices)
        assert recorder.events == [("作る", True)]
        assert kept is not None and not kept.isVisible()
