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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, Author
from sashimono.core.emphasis import fixed_transform
from sashimono.core.model import (
    AnimatedValue,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.reframe import REFRAME_FOLLOWS, REFRAME_SIZES, STEADY, ReframeStyle
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import SceneState, ScreenState
from sashimono.engine.scenes import SceneSignal
from sashimono.engine.subject import SUBJECT_BINS, SUBJECT_SECONDS, SubjectTrack
from sashimono.engine.vision import Box, Face, ScreenKind, ScreenReport, ScreenSample
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.reframe_dialog import ReframeDialog
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

    def test_the_faces_are_on_unless_turned_off(self, qt_application: QApplication) -> None:
        # 既定で切ると、部品を入れた人も知らないうちに肌色の絵へ枠が寄ったままになる
        # 設定画面で切れないと、手元を映したい人が毎回顔に枠を取られる
        del qt_application
        assert Preferences().reframe_style.faces is True
        dialog = PreferencesDialog(Preferences())
        try:
            assert dialog._reframe_faces.isChecked()
            dialog._reframe_faces.setChecked(False)
            assert dialog.preferences().reframe_style.faces is False
        finally:
            dialog.deleteLater()
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"reframe_faces": "no"}', encoding="utf-8")
        assert PreferenceStore().load().reframe_faces is True
        PreferenceStore().save(Preferences(reframe_faces=False))
        assert PreferenceStore().load().reframe_style.faces is False

    def test_the_menu_window_offers_the_faces(self, qt_application: QApplication) -> None:
        # 窓で切れないと、1 回だけ顔を使わずに作りたい人が設定を書き換えて戻すことになる
        del qt_application
        dialog = ReframeDialog(ReframeStyle(faces=False))
        try:
            assert not dialog.faces_box.isChecked()
            dialog.faces_box.setChecked(True)
            assert dialog.choice().faces is True
        finally:
            dialog.deleteLater()

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

    def test_a_failed_analysis_is_told(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 解析できなかった素材を黙って真ん中で切り出すと、人は被写体を追っていないことに気付かない
        work = _work(video_media)
        window = MainWindow(work, confirm_unsaved=False)
        opened.append(window)
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="auto")
        )
        monkeypatch.setattr(
            window._analyzer,
            "scene_state",
            lambda _media: SceneState(None, running=False, failure="読めない"),
        )
        assert window.reframe_project()
        assert "解析ができなかった素材 1 本" in window.statusBar().currentMessage()

    def test_the_faces_are_asked_for_and_used(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、部品を入れて顔を調べ終えた人の作り替えでも、人の窓だけ顔を使わずに作る
        # （AI の道具と違う縦の作品になる）か、使ったかどうかが分からない
        window = MainWindow(_work(video_media), confirm_unsaved=False)
        opened.append(window)
        media = window.document.project.media[0]
        times = np.arange(SUBJECT_SECONDS / 2, 10.0, SUBJECT_SECONDS)
        columns = np.zeros((times.shape[0], SUBJECT_BINS), np.float32)
        columns[:, 28] = 1.0
        rows = np.zeros_like(columns)
        rows[:, SUBJECT_BINS // 2] = 1.0
        subject = SubjectTrack(times, columns, rows, np.full(times.shape, 0.02, np.float32))
        frames = np.arange(1, 301, dtype=np.float64) / 30
        flat = np.full(frames.shape, 0.01, np.float32)
        signal = SceneSignal(
            frames, flat, flat, analyzed=10.0, complete=True, stream=0, subject=subject
        )
        face = Face(Box(-0.3 * 1920, 0.0, 120.0, 120.0), 0.9)
        report = ScreenReport(
            ScreenKind.FACES,
            "偽の顔",
            1920,
            1080,
            tuple(ScreenSample(moment, faces=(face,)) for moment in (2.0, 5.0, 8.0)),
            3,
            True,
            stream=0,
        )
        asked: list[tuple[MediaId, ScreenKind]] = []
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="steady")
        )
        monkeypatch.setattr(
            main_window_module,
            "VISION_PACK",
            SimpleNamespace(status=lambda: SimpleNamespace(ready=True)),
        )
        monkeypatch.setattr(
            window._analyzer, "scene_state", lambda _media: SceneState(signal, running=False)
        )
        monkeypatch.setattr(
            window._screen, "request", lambda item, kind: asked.append((item.id, kind))
        )
        monkeypatch.setattr(
            window._screen, "state", lambda _item, _kind: ScreenState(report, running=False)
        )
        assert window.reframe_project()
        assert asked == [(media.id, ScreenKind.FACES)]
        assert "顔の位置も手がかり" in window.statusBar().currentMessage()
        transform = fixed_transform(window.document.project.timeline.tracks[0].clips[0])
        assert transform is not None
        position = transform.params["pos_x"]
        assert isinstance(position, AnimatedValue)
        # 顔は左（横の 0.2） 枠を左へ寄せると、絵は右へ動く
        assert position.at(0) > 300

    def test_without_the_parts_the_person_is_told(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 黙ると、部品を入れていない人は顔を見て枠を決めたと思い込む
        window = MainWindow(_work(video_media), confirm_unsaved=False)
        opened.append(window)
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="auto")
        )
        monkeypatch.setattr(
            main_window_module,
            "VISION_PACK",
            SimpleNamespace(status=lambda: SimpleNamespace(ready=False)),
        )
        monkeypatch.setattr(
            window._analyzer,
            "scene_state",
            lambda _media: SceneState(None, running=False, failure="読めない"),
        )
        monkeypatch.setattr(
            window._screen,
            "request",
            lambda *_args: pytest.fail("部品が無いのに顔を頼んだ"),
        )
        assert window.reframe_project()
        assert "部品が入っていない" in window.statusBar().currentMessage()

    def test_the_window_tells_the_reason_apart(
        self,
        opened: list[MainWindow],
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 真ん中を切り出したのに「動きと肌色だけで見積もった」と出すと、人は被写体を追ったと
        # 思い込む 追う動画の無い作品で黙ると、追えないことに気付かない（PR #408 の CodeRabbit）
        monkeypatch.setattr(
            main_window_module,
            "VISION_PACK",
            SimpleNamespace(status=lambda: SimpleNamespace(ready=True)),
        )
        window = MainWindow(_work(video_media), confirm_unsaved=False)
        opened.append(window)
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="center")
        )
        window.statusBar().clearMessage()
        assert window.reframe_project()
        assert "動きと肌色" not in window.statusBar().currentMessage()
        empty = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(video_media,))
        bare = MainWindow(empty, confirm_unsaved=False)
        opened.append(bare)
        monkeypatch.setattr(
            main_window_module, "ask_reframe", lambda *_args: ReframeStyle(follow="auto")
        )
        assert bare.reframe_project()
        message = bare.statusBar().currentMessage()
        assert "背景の動画が無い" in message
        assert "動きと肌色だけで見積もりました" not in message

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
