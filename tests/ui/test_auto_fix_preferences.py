"""色と雑音の自動補正（#336）の設定画面と、本物の窓で AI が付けた補正

偽のホストの試験（tests/ai/test_auto_fix_tools.py）は道具の結果を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに付けた補正が取り消し 1 回で
戻ることを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from sashimono.ai.auto_fix import ColorRequest, NoiseRequest
from sashimono.core.commands import Author, SetTrackState
from sashimono.core.model import ClipId, MediaItem, Project
from sashimono.effects.audio import DENOISE_KIND
from sashimono.effects.denoise import DEFAULT_SENSITIVITY_DB
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.auto_color import GRADE_KIND
from sashimono.engine.cache.thumbnails import Filmstrip
from sashimono.ui import main_window as main_window_module
from sashimono.ui.auto_fix_dialogs import ColorDialog, NoiseDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.test_auto_fix_tools import _strip, _waveform
from tests.core.test_seams import JumpCut
from tests.engine.test_auto_color import _scene
from tests.ui.ai_tool_calls import call_tool
from tests.ui.test_seam_preferences import _menu_action


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
    """AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道） 変更は自動で承認する

    待ちの上限は tests/ui/ai_tool_calls.py にまとめた 試験ごとに 60 秒で区切ると、
    遅い機械では返る前に落ちる（#405）
    """
    return call_tool(window, name, arguments)


def _prepared(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, strip: Filmstrip, waveform: Waveform
) -> None:
    """本物の窓の解析器が、サムネイルと波形を用意し終えたことにする（素材のファイルは無い）"""

    def filmstrip(media: MediaItem, stream: int | None = None) -> Filmstrip:
        del media, stream
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


COLOR_TEXT = "色を自動で整える…"
NOISE_TEXT = "ノイズを自動で減らす…"


def _shape(project: Project) -> list[tuple[str, int, list[tuple[str, dict[str, object]]]]]:
    """クリップの位置とエフェクトの値 足したエフェクトの ID は作るたびに違うので見ない"""
    return [
        (
            track.name,
            clip.timeline_start,
            [(effect.kind, dict(effect.params)) for effect in clip.effects],
        )
        for track in project.timeline.tracks
        for clip in sorted(track.clips, key=lambda c: c.timeline_start)
    ]


class TestRightClick:
    """タイムラインの右クリックから人が使う（#336 2026-10-20 利用者の決定）

    窓は main_window の ask_color・ask_noise を差し替えて答える 差し替えずに窓が開くと
    no_windows が落とす
    """

    @pytest.mark.parametrize(("text", "on_audio"), [(COLOR_TEXT, False), (NOISE_TEXT, True)])
    def test_the_items_are_on_a_media_clip(
        self, opened: list[MainWindow], text: str, on_audio: bool
    ) -> None:
        # 壊れると、人は AI に頼まないと色を測って整えたり、雑音を測って減らしたりできない
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        track = work.audio if on_audio else work.video
        assert _menu_action(window, track, 30, text).isEnabled()

    @pytest.mark.parametrize(("text", "on_audio"), [(COLOR_TEXT, False), (NOISE_TEXT, True)])
    def test_a_locked_track_greys_them_out(
        self, opened: list[MainWindow], text: str, on_audio: bool
    ) -> None:
        # 壊れると、人が触らせないためにロックしたトラックのクリップの色や音が右クリックで変わる
        work = JumpCut()
        track = work.audio if on_audio else work.video
        project = SetTrackState(track.id, locked=True).apply(work.project)
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        assert not _menu_action(window, track, 30, text).isEnabled()

    def test_colour_matches_the_tool_and_undoes_once_as_the_person(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が右クリックで整えた色と AI の auto_color が食い違うか、窓の既定が本人の
        # 設定と違うか、取り消し 1 回で戻らないか、人の変更が AI の段に数えられて AI に戻される
        PreferenceStore().save(Preferences(auto_color_strength=60))
        work = JumpCut()
        asked: list[int] = []

        def answer(_parent: object, strength: int, _partners: object) -> ColorRequest:
            asked.append(strength)
            return ColorRequest(strength=float(strength))

        monkeypatch.setattr(main_window_module, "ask_color", answer)
        strip = _strip(_scene(dark=0.25, light=0.7))
        by_hand = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_hand)
        _prepared(by_hand, monkeypatch, strip, _waveform(0.01))
        _menu_action(by_hand, work.video, 30, COLOR_TEXT).trigger()
        assert asked == [60]
        by_tool = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_tool)
        _prepared(by_tool, monkeypatch, strip, _waveform(0.01))
        _call(by_tool, "auto_color", {"clip_ids": [str(work.pictures[0].id)]})
        assert _shape(by_hand.document.project) == _shape(by_tool.document.project)
        assert by_hand.document.undo_by is Author.PERSON
        by_hand.document.undo()
        assert by_hand.document.project == work.project

    def test_noise_matches_the_tool_and_undoes_once_as_the_person(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が右クリックで足したノイズ除去と AI の reduce_noise が食い違うか、窓の
        # 既定が本人の設定と違うか、取り消し 1 回で戻らない
        PreferenceStore().save(Preferences(denoise_reduction_db=18))
        work = JumpCut()
        asked: list[int] = []

        def answer(_parent: object, reduction: int) -> NoiseRequest:
            asked.append(reduction)
            return NoiseRequest(reduction=float(reduction), sensitivity=DEFAULT_SENSITIVITY_DB)

        monkeypatch.setattr(main_window_module, "ask_noise", answer)
        strip = _strip(_scene(dark=0.0, light=1.0))
        by_hand = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_hand)
        _prepared(by_hand, monkeypatch, strip, _waveform(0.01))
        _menu_action(by_hand, work.audio, 30, NOISE_TEXT).trigger()
        assert asked == [18]
        by_tool = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_tool)
        _prepared(by_tool, monkeypatch, strip, _waveform(0.01))
        _call(by_tool, "reduce_noise", {"clip_ids": [str(work.sounds[0].id)]})
        assert _shape(by_hand.document.project) == _shape(by_tool.document.project)
        assert by_hand.document.undo_by is Author.PERSON
        by_hand.document.undo()
        assert by_hand.document.project == work.project

    def test_a_skipped_clip_is_told_in_the_status_bar(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、雑音が聞こえない録音で何も起きず、人はなぜ掛からないのか分からない
        work = JumpCut()
        monkeypatch.setattr(
            main_window_module,
            "ask_noise",
            lambda *_args: NoiseRequest(reduction=12.0, sensitivity=DEFAULT_SENSITIVITY_DB),
        )
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _prepared(window, monkeypatch, _strip(_scene(dark=0.0, light=1.0)), _waveform(1e-5))
        _menu_action(window, work.audio, 30, NOISE_TEXT).trigger()
        assert "聞こえない" in window.statusBar().currentMessage()
        assert window.document.project == work.project

    def test_cancelling_changes_nothing(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、窓でやめたのに色や音が変わり、取り消しの段が増える
        work = JumpCut()
        monkeypatch.setattr(main_window_module, "ask_color", lambda *_args: None)
        monkeypatch.setattr(main_window_module, "ask_noise", lambda *_args: None)
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _prepared(window, monkeypatch, _strip(_scene(dark=0.25, light=0.7)), _waveform(0.01))
        _menu_action(window, work.video, 30, COLOR_TEXT).trigger()
        _menu_action(window, work.audio, 30, NOISE_TEXT).trigger()
        assert window.document.project == work.project
        assert not window.document.can_undo


class TestDialogs:
    def test_the_colour_dialog_returns_what_was_chosen(self, qt_application: QApplication) -> None:
        # 壊れると、窓で強さや合わせる相手を選んでも、別の値で整える
        del qt_application
        dialog = ColorDialog(60, [(ClipId("相手"), "V1 00:00:01:00")])
        try:
            assert dialog.choice() == ColorRequest(strength=60.0)
            dialog.strength.setValue(90)
            dialog.white_balance.setChecked(False)
            dialog.partner.setCurrentIndex(1)
            assert dialog.choice() == ColorRequest(
                strength=90.0, white_balance=False, match=ClipId("相手")
            )
        finally:
            dialog.deleteLater()

    def test_turning_both_off_disables_ok(self, qt_application: QApplication) -> None:
        # 壊れると、何も直さない選び方で OK を押せ、押したのに何も起きない
        del qt_application
        dialog = ColorDialog(60, [])
        try:
            dialog.exposure.setChecked(False)
            dialog.white_balance.setChecked(False)
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
            assert not ok.isEnabled()
        finally:
            dialog.deleteLater()

    def test_the_noise_dialog_returns_what_was_chosen(self, qt_application: QApplication) -> None:
        # 壊れると、窓で量や「掛ける」を選んでも、別の値で掛ける
        del qt_application
        dialog = NoiseDialog(12)
        try:
            assert dialog.choice() == NoiseRequest(12.0, DEFAULT_SENSITIVITY_DB)
            dialog.reduction.setValue(20)
            dialog.sensitivity.setValue(6.5)
            dialog.force.setChecked(True)
            assert dialog.choice() == NoiseRequest(20.0, 6.5, force=True)
        finally:
            dialog.deleteLater()
