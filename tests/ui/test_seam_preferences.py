"""ジャンプカットのつなぎ目（#332）の設定画面と、本物の窓で AI が付けた処理

偽のホストの試験（tests/ai/test_speed_seam_tools.py）は道具の結果を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに付けた処理と速度の変更が
取り消し 1 回で戻ることを見る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.model import TRANSITION_KIND
from sashimono.core.seams import SEAM_NONE, SEAM_VIDEOS, SEAM_ZOOM, SeamStyle
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending
from tests.core.test_seams import JumpCut


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    つなぎ目の処理と速度の変更は取り消せる編集で、確かめはブリッジの承認（ここでは自動で
    承認）だけ 窓が出たら作りが変わったということで、開いたまま待つと試験が人の操作を待って止まる
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


def _call(window: MainWindow, name: str, arguments: dict[str, Any]) -> Any:
    """AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道） 変更は自動で承認する"""
    operation = find_operation(name)
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
        self, opened: list[MainWindow]
    ) -> None:
        # 壊れると、設定画面で選んだつなぎ目の処理が AI に効かないか、付けた処理を
        # 取り消し 1 回で戻せない
        PreferenceStore().save(Preferences(seam_video=SEAM_ZOOM, seam_zoom_percent=120))
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        assert window.seam_style == SeamStyle(video=SEAM_ZOOM, zoom_percent=120)
        result = _call(window, "smooth_cuts", {})
        assert result["zoomed_clip_ids"] == [str(work.pictures[1].id)]
        window.document.undo()
        assert window.document.project == work.project

    def test_a_crossfade_goes_above_the_picture_in_the_real_window(
        self, opened: list[MainWindow]
    ) -> None:
        # 壊れると、本物の窓では場面切り替えが絵より下へ入り、つなぎ目が何も溶けない
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _call(window, "smooth_cuts", {"frames": [60]})
        tracks = window.document.project.timeline.tracks
        order = [track.id for track in tracks]
        placed = [
            track.id
            for track in tracks
            for clip in track.clips
            if clip.source is not None and clip.source.kind == TRANSITION_KIND
        ]
        assert len(placed) == 1
        assert order.index(placed[0]) > order.index(work.video.id)

    def test_speed_goes_back_in_one_undo(self, opened: list[MainWindow]) -> None:
        # 壊れると、本物の窓で速度を変えた絵と音を戻すのに取り消しが 2 回要る
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _call(window, "set_speed", {"clip_ids": [str(work.pictures[2].id)], "speed_percent": 200})
        assert window.document.undo_label is not None
        window.document.undo()
        assert window.document.project == work.project


class TestPreference:
    def test_the_defaults_do_not_crop(self) -> None:
        # 壊れると、何も選んでいない人の動画が寄って画面の端（焼き込んだ字幕や頭）が欠ける
        plain = Preferences()
        assert plain.seam_style == SeamStyle()
        assert (plain.seam_video, plain.seam_crossfade_ms, plain.seam_audio_fade_ms) == (
            "crossfade",
            100,
            20,
        )

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        # 壊れると、手で書き換えた値（知らない処理・負の長さ・大きすぎる寄り）のまま、
        # 道具が毎回断るか、画面の外まで寄る
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"seam_video": "wipe", "seam_crossfade_ms": -5,'
            ' "seam_zoom_percent": 400, "seam_audio_fade_ms": "長め"}',
            encoding="utf-8",
        )
        assert PreferenceStore().load().seam_style == SeamStyle()

    def test_the_dialog_offers_and_returns_the_values(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面でつなぎ目の処理を変えても保存されない
        del qt_application
        dialog = PreferencesDialog(Preferences(seam_video=SEAM_NONE, seam_audio_fade_ms=40))
        try:
            assert dialog.preferences().seam_style == SeamStyle(video=SEAM_NONE, audio_fade_ms=40)
            dialog._seam_video.setCurrentIndex(SEAM_VIDEOS.index(SEAM_ZOOM))
            dialog._seam_crossfade.setValue(200)
            dialog._seam_zoom.setValue(115)
            dialog._seam_audio.setValue(0)
            assert dialog.preferences().seam_style == SeamStyle(
                video=SEAM_ZOOM, crossfade_ms=200, zoom_percent=115, audio_fade_ms=0
            )
        finally:
            dialog.deleteLater()
