"""縦動画への作り替え（#334）の設定画面とメニューと、本物の窓で AI が作り替えた作品

偽のホストの試験（tests/ai/test_reframe_tool.py）は道具の受け口を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに作り替えた作品が取り消し 1 回で
戻り、その段が AI の物として積まれること、メニューの〔縦の作品に作り替える…〕が同じ作品を
人の段として作ることを見る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, Author
from sashimono.core.model import MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.core.reframe import REFRAME_FOLLOWS, REFRAME_SIZES, STEADY, ReframeStyle
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import SceneState
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending
from tests.conftest import make_clip


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    作り替えは取り消せる編集で、確かめはブリッジの承認（ここでは自動で承認）だけ 窓が出たら
    作りが変わったということで、開いたまま待つと試験が人の操作を待って止まる
    """

    def refuse(*_arguments: object, **_options: object) -> None:
        raise AssertionError("試験の途中で窓が開いた")

    for owner, name in (
        (QDialog, "exec"),
        (QDialog, "open"),
        (QDialog, "show"),
        (QMainWindow, "show"),
        (QMessageBox, "exec"),
        (QMessageBox, "warning"),
        (QMessageBox, "information"),
        (QMessageBox, "question"),
        (QMessageBox, "critical"),
        (QFileDialog, "getSaveFileName"),
        (QFileDialog, "getOpenFileName"),
    ):
        monkeypatch.setattr(owner, name, refuse)


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    windows: list[MainWindow] = []
    yield windows
    for window in windows:
        window.close()
        window.deleteLater()


def _work(video_media: MediaItem) -> Project:
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(video_media,))
    track = Track(TrackKind.VIDEO, "V1")
    base = base.with_timeline(replace(base.timeline, tracks=(track,)))
    return AddClip(track.id, make_clip(0, 300, video_media)).apply(base)


def _call(window: MainWindow, arguments: dict[str, Any]) -> Any:
    """AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道） 変更は自動で承認する"""
    operation = find_operation("reframe")
    assert operation is not None
    bridge = allow_sending(EditorBridge(window, auto_approve=True))
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while thread.is_alive():
        bridge.pump()
        QApplication.processEvents()
        thread.join(0.01)
        if time.monotonic() > deadline:
            bridge.cancel()
            thread.join(5.0)
            pytest.fail("道具が返らなかった")
    if "error" in box:
        raise box["error"]
    return box["result"]


class TestWindow:
    def test_the_tool_uses_the_saved_preference_and_undoes_once(
        self, opened: list[MainWindow], video_media: MediaItem
    ) -> None:
        # 壊れると、設定画面で選んだ大きさが AI に効かないか、作り替えを取り消し 1 回で戻せない
        PreferenceStore().save(Preferences(reframe_size="1080x1350", reframe_follow="center"))
        work = _work(video_media)
        window = MainWindow(work, confirm_unsaved=False)
        opened.append(window)
        assert window.reframe_style == ReframeStyle(size="1080x1350", follow="center")
        result = _call(window, {})
        assert result["resolution"] == "1080x1350"
        assert window.document.project.settings.resolution == (1080, 1350)
        assert window.document.undo_by is Author.AI
        window.document.undo()
        assert window.document.project == work


class TestPreference:
    def test_the_defaults_make_a_short_and_settle_the_window(self) -> None:
        # 壊れると、何も選んでいない人の作品が思わぬ大きさになるか、枠が動き続けて見にくい
        plain = Preferences()
        assert plain.reframe_style == ReframeStyle()
        assert (plain.reframe_size, plain.reframe_follow, plain.reframe_smooth_ms) == (
            "1080x1920",
            "auto",
            1000,
        )

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        # 壊れると、手で書き換えた値（知らない大きさ・知らない動かし方・負の長さ）のまま、
        # 道具が毎回断る
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"reframe_size": "1081x1920", "reframe_follow": "zoom", "reframe_smooth_ms": -5}',
            encoding="utf-8",
        )
        assert PreferenceStore().load().reframe_style == ReframeStyle()

    def test_the_dialog_offers_and_returns_the_values(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面で縦動画への作り替えの既定を変えても保存されない
        del qt_application
        dialog = PreferencesDialog(Preferences(reframe_size="1080x1080", reframe_smooth_ms=3000))
        try:
            assert dialog.preferences().reframe_style == ReframeStyle(
                size="1080x1080", smooth_ms=3000
            )
            dialog._reframe_size.setCurrentIndex(REFRAME_SIZES.index("720x1280"))
            dialog._reframe_follow.setCurrentIndex(REFRAME_FOLLOWS.index(STEADY))
            dialog._reframe_smooth.setValue(500)
            assert dialog.preferences().reframe_style == ReframeStyle(
                size="720x1280", follow=STEADY, smooth_ms=500
            )
        finally:
            dialog.deleteLater()


def _shape(project: Project) -> list[object]:
    """解像度と、クリップごとのエフェクトの種類と値"""
    return [
        project.settings.resolution,
        *(
            [(effect.kind, dict(effect.params)) for effect in clip.effects]
            for track in project.timeline.tracks
            for clip in track.clips
        ),
    ]


class TestMenu:
    """メニューの〔縦の作品に作り替える…〕から人が使う（#334 2026-10-20 利用者の決定）

    窓は main_window の ask_reframe を差し替えて答える 差し替えずに窓が開くと no_windows が落とす
    """

    def test_it_matches_the_tool_and_undoes_once_as_the_person(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、人の作り替えと AI の作り替えで違う縦の作品ができるか、人の段が AI の段に
        # 数えられて AI の取り消しで消える
        work = _work(video_media)
        window = MainWindow(work, confirm_unsaved=False)
        opened.append(window)
        asked: list[ReframeStyle] = []

        def answer(_parent: object, style: ReframeStyle) -> ReframeStyle:
            asked.append(style)
            return ReframeStyle(size="720x1280", follow="center", smooth_ms=500)

        monkeypatch.setattr(main_window_module, "ask_reframe", answer)
        assert window.reframe_project()
        assert asked == [window.reframe_style]
        made = window.document.project
        assert made.settings.resolution == (720, 1280)
        assert window.document.undo_by is Author.PERSON
        window.document.undo()
        assert window.document.project == work
        # 同じ値で AI の道具を通すと同じ作品になる（足した配置の欄の ID は作るたびに違うので見ない）
        _call(window, {"width": 720, "height": 1280, "follow": "center"})
        assert _shape(window.document.project) == _shape(made)

    def test_the_menu_value_is_only_for_that_time(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 窓で試した値が設定に残ると、次から知らないうちに別の大きさで作られる
        window = MainWindow(_work(video_media), confirm_unsaved=False)
        opened.append(window)
        before = window.reframe_style
        monkeypatch.setattr(
            main_window_module,
            "ask_reframe",
            lambda *_args: ReframeStyle(size="1080x1080", follow="center"),
        )
        assert window.reframe_project()
        assert window.reframe_style == before

    def test_cancelling_changes_nothing(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # やめたのに変えると、窓を閉じただけで作品が縦になる
        work = _work(video_media)
        window = MainWindow(work, confirm_unsaved=False)
        opened.append(window)
        monkeypatch.setattr(main_window_module, "ask_reframe", lambda *_args: None)
        assert not window.reframe_project()
        assert window.document.project == work

    def test_a_running_analysis_changes_nothing(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 途中までの並びで作ると、読んでいない所の枠が真ん中に止まり、やり直しても直らない
        work = _work(video_media)
        window = MainWindow(work, confirm_unsaved=False)
        opened.append(window)
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="auto")
        )
        monkeypatch.setattr(
            window._analyzer, "scene_state", lambda _media: SceneState(None, running=True)
        )
        assert not window.reframe_project()
        assert window.document.project == work
        assert "解析" in window.statusBar().currentMessage()

    def test_a_locked_track_says_why(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 黙って何もしないと、人はなぜ縦にならないのか分からない
        work = _work(video_media)
        track = replace(work.timeline.tracks[0], locked=True)
        work = work.with_timeline(work.timeline.replace_track(track))
        window = MainWindow(work, confirm_unsaved=False)
        opened.append(window)
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="center")
        )
        assert not window.reframe_project()
        assert window.document.project == work
        assert "ロック" in window.statusBar().currentMessage()
