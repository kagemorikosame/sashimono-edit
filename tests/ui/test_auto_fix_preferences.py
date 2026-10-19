"""色と雑音の自動補正（#336）の設定画面と、本物の窓で AI が付けた補正

偽のホストの試験（tests/ai/test_auto_fix_tools.py）は道具の結果を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに付けた補正が取り消し 1 回で
戻ることを見る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import Author
from sashimono.core.model import MediaItem
from sashimono.effects.audio import DENOISE_KIND
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.auto_color import GRADE_KIND
from sashimono.engine.cache.thumbnails import Filmstrip
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending
from tests.ai.test_auto_fix_tools import _strip, _waveform
from tests.core.test_seams import JumpCut
from tests.engine.test_auto_color import _scene


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    色と雑音の補正は取り消せる編集で、確かめはブリッジの承認（ここでは自動で承認）だけ
    窓が出たら作りが変わったということで、開いたまま待つと試験が人の操作を待って止まる
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


def _prepared(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, strip: Filmstrip, waveform: Waveform
) -> None:
    """本物の窓の解析器が、サムネイルと波形を用意し終えたことにする（素材のファイルは無い）"""

    def filmstrip(media: MediaItem) -> Filmstrip:
        del media
        return strip

    def waveform_of(media: MediaItem, stream: int | None = None) -> Waveform:
        del media, stream
        return waveform

    monkeypatch.setattr(window._analyzer, "filmstrip", filmstrip)
    monkeypatch.setattr(window._analyzer, "waveform", waveform_of)


class TestWindow:
    def test_the_colour_uses_the_saved_strength_and_undoes_once(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、設定画面で選んだ強さが AI に効かないか、付けた補正を取り消し 1 回で戻せない
        PreferenceStore().save(Preferences(auto_color_strength=40))
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _prepared(window, monkeypatch, _strip(_scene(dark=0.25, light=0.7)), _waveform(0.01))
        result = _call(window, "auto_color", {"clip_ids": [str(work.pictures[0].id)]})
        assert result["strength_percent"] == 40.0
        located = window.document.project.timeline.locate_clip(work.pictures[0].id)
        assert located is not None and located[1].effects[0].kind == GRADE_KIND
        assert window.document.undo_by is Author.AI
        window.document.undo()
        assert window.document.project == work.project

    def test_the_noise_uses_the_saved_amount_and_undoes_once(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、設定画面で選んだ下げる量が AI に効かないか、取り消し 1 回で戻せない
        PreferenceStore().save(Preferences(denoise_reduction_db=18))
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _prepared(window, monkeypatch, _strip(_scene(dark=0.0, light=1.0)), _waveform(0.01))
        result = _call(window, "reduce_noise", {"track_id": str(work.audio.id)})
        assert result["reduction_db"] == 18.0
        located = window.document.project.timeline.locate_clip(work.sounds[0].id)
        assert located is not None
        assert any(effect.kind == DENOISE_KIND for effect in located[1].effects)
        window.document.undo()
        assert window.document.project == work.project


class TestPreference:
    def test_the_defaults_are_gentle(self) -> None:
        # 壊れると、何も選んでいない人の絵が測ったとおりに強く直り（わざとの夕焼けが白くなる）、
        # 声が水の中のように削れる
        plain = Preferences()
        assert (plain.auto_color_strength, plain.denoise_reduction_db) == (70, 12)

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        # 壊れると、手で書き換えた値（範囲の外・文字）のまま、道具が毎回断る
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"auto_color_strength": 500, "denoise_reduction_db": "強め"}', encoding="utf-8"
        )
        loaded = PreferenceStore().load()
        assert (loaded.auto_color_strength, loaded.denoise_reduction_db) == (70, 12)

    def test_the_dialog_offers_and_returns_the_values(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面で強さや下げる量を変えても保存されない
        del qt_application
        dialog = PreferencesDialog(Preferences(auto_color_strength=30, denoise_reduction_db=6))
        try:
            chosen = dialog.preferences()
            assert (chosen.auto_color_strength, chosen.denoise_reduction_db) == (30, 6)
            dialog._auto_color_strength.setValue(90)
            dialog._denoise_reduction.setValue(24)
            chosen = dialog.preferences()
            assert (chosen.auto_color_strength, chosen.denoise_reduction_db) == (90, 24)
        finally:
            dialog.deleteLater()


def test_the_thumbnail_is_what_the_analyzer_made(
    opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 壊れると、本物の窓が解析器のサムネイルを AI へ渡さず、色を測る道具がいつまでも待つ
    work = JumpCut()
    window = MainWindow(work.project, confirm_unsaved=False)
    opened.append(window)
    strip = _strip(np.zeros((4, 4, 4), dtype=np.uint8))
    _prepared(window, monkeypatch, strip, _waveform(0.01))
    assert window.filmstrip(work.media) is strip
