"""ジャンプカットのつなぎ目（#332）の設定画面と、本物の窓で AI が付けた処理

偽のホストの試験（tests/ai/test_speed_seam_tools.py）は道具の結果を見る ここでは、設定画面の
値が保存されて道具の既定になること、本物の窓でブリッジ越しに付けた処理と速度の変更が
取り消し 1 回で戻ることを見る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from fractions import Fraction
from typing import Any

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import Author, Command, RippleCut, SetTrackState
from sashimono.core.model import TRANSITION_KIND, Project, Track
from sashimono.core.seams import (
    SEAM_NONE,
    SEAM_VIDEOS,
    SEAM_ZOOM,
    UNFIT_SEAM,
    SeamStyle,
    SmoothCuts,
    seams_after_cuts,
)
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.seam_dialogs import SpeedChoice
from sashimono.ui.subtitle.dialogs import JetCutDialog
from sashimono.ui.subtitle.panel import SubtitlePanel
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending
from tests.core.test_seams import JumpCut, with_bystander
from tests.ui.test_subtitle_panel import StubAnalyzer, make_waveform


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


SPEED_TEXT = "速度を変える…"
SEAMS_TEXT = "つなぎ目を目立たなくする…"


def _menu_action(window: MainWindow, track: Track, frame: int, text: str) -> QAction:
    view = window._timeline
    view.resize(1200, 400)
    band = next(b for b in view.view_layout.bands(view.project.timeline) if b.track.id == track.id)
    point = QPoint(int(view.view_layout.frame_to_x(frame)), (band.top + band.bottom) // 2)
    menu = view.build_context_menu(point)
    found = next((a for a in menu.actions() if a.text().startswith(text)), None)
    assert found is not None, [a.text() for a in menu.actions()]
    return found


def _shape(project: Project) -> list[tuple[str, int, int, list[tuple[str, dict[str, object]]]]]:
    """クリップの位置・長さ・エフェクトの値 足したエフェクトの ID は作るたびに違うので見ない"""
    return [
        (
            track.name,
            clip.timeline_start,
            clip.duration,
            [(effect.kind, dict(effect.params)) for effect in clip.effects],
        )
        for track in project.timeline.tracks
        for clip in sorted(track.clips, key=lambda c: c.timeline_start)
    ]


class TestRightClick:
    """タイムラインの右クリックから人が使う（#332 2026-10-18 利用者の決定）

    窓は main_window の ask_speed・ask_seams を差し替えて答える 差し替えずに窓が開くと
    no_windows が落とす
    """

    @pytest.mark.parametrize("text", [SPEED_TEXT, SEAMS_TEXT])
    def test_the_items_are_on_a_media_clip(self, opened: list[MainWindow], text: str) -> None:
        # 壊れると、人は AI に頼まないと速度を中身ごと変えたり、つなぎ目をまとめて処理したりできない
        work = JumpCut()
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        assert _menu_action(window, work.video, 90, text).isEnabled()

    @pytest.mark.parametrize("text", [SPEED_TEXT, SEAMS_TEXT])
    def test_a_locked_track_greys_them_out(self, opened: list[MainWindow], text: str) -> None:
        # 壊れると、人が触らせないためにロックしたトラックのクリップが右クリックで変わる
        work = JumpCut()
        project = SetTrackState(work.video.id, locked=True).apply(work.project)
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        assert not _menu_action(window, work.video, 90, text).isEnabled()

    def test_speed_matches_the_tool_and_undoes_once_as_the_person(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が右クリックで変えた速度と AI が変えた速度で長さや点の位置が食い違うか、
        # 取り消し 1 回で戻らないか、人の変更が AI の段に数えられて AI に取り消される
        work = JumpCut()
        asked: list[int] = []

        def answer(_parent: object, percent: int) -> SpeedChoice:
            asked.append(percent)
            return SpeedChoice(Fraction(2), keep_length=False)

        monkeypatch.setattr(main_window_module, "ask_speed", answer)
        by_hand = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_hand)
        _menu_action(by_hand, work.video, 150, SPEED_TEXT).trigger()
        # 窓の最初の速度は今の速度
        assert asked == [100]
        by_tool = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_tool)
        _call(by_tool, "set_speed", {"clip_ids": [str(work.pictures[2].id)], "speed_percent": 200})
        assert by_hand.document.project.timeline == by_tool.document.project.timeline
        assert by_hand.document.undo_by is Author.PERSON
        by_hand.document.undo()
        assert by_hand.document.project == work.project

    def test_seams_match_the_tool_and_undo_once_as_the_person(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が右クリックで付けた処理と AI の smooth_cuts が食い違うか、窓の既定が本人の
        # 設定と違うか、取り消し 1 回で戻らない 選んだクリップの頭と終わりのつなぎ目だけに付く
        PreferenceStore().save(Preferences(seam_video=SEAM_ZOOM, seam_zoom_percent=120))
        work = JumpCut()
        asked: list[SeamStyle] = []

        def answer(_parent: object, style: SeamStyle, _note: str) -> SeamStyle:
            asked.append(style)
            return style

        monkeypatch.setattr(main_window_module, "ask_seams", answer)
        by_hand = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_hand)
        _menu_action(by_hand, work.video, 90, SEAMS_TEXT).trigger()
        assert asked == [SeamStyle(video=SEAM_ZOOM, zoom_percent=120)]
        by_tool = MainWindow(work.project, confirm_unsaved=False)
        opened.append(by_tool)
        _call(by_tool, "smooth_cuts", {"frames": [60, 120]})
        assert _shape(by_hand.document.project) == _shape(by_tool.document.project)
        assert by_hand.document.undo_by is Author.PERSON
        by_hand.document.undo()
        assert by_hand.document.project == work.project

    def test_seams_leave_unselected_clips_on_other_tracks_alone(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、選んだクリップのつなぎ目と同じフレームで区切れている別のトラックの、
        # 選んでいないクリップまで寄せる（Codex の指摘）
        work = JumpCut()
        project, _track, _before, after = with_bystander(work)
        monkeypatch.setattr(
            main_window_module,
            "ask_seams",
            lambda _parent, _style, _note: SeamStyle(video=SEAM_ZOOM, audio_fade_ms=0),
        )
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        _menu_action(window, work.video, 90, SEAMS_TEXT).trigger()
        located = window.document.project.timeline.locate_clip(after.id)
        assert located is not None and located[1] == after
        assert window.document.can_undo

    def test_a_refusal_says_why(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、寄りを選んだのに音だけのつなぎ目しか無いとき、ステータスバーに
        # 「付けられませんでした（）」と空の括弧が出て、何を変えればよいか分からない
        # （CodeRabbit の指摘）
        work = JumpCut()
        project = SetTrackState(work.video.id, locked=True).apply(work.project)
        monkeypatch.setattr(
            main_window_module,
            "ask_seams",
            lambda _parent, _style, _note: SeamStyle(video=SEAM_ZOOM, audio_fade_ms=0),
        )
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        _menu_action(window, work.audio, 90, SEAMS_TEXT).trigger()
        message = window.statusBar().currentMessage()
        assert "（）" not in message
        assert UNFIT_SEAM in message
        assert window.document.project == project

    def test_cancelling_changes_nothing(
        self, opened: list[MainWindow], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、窓でやめたのに速度やつなぎ目が変わり、取り消しの段が増える
        work = JumpCut()
        monkeypatch.setattr(main_window_module, "ask_speed", lambda *_args: None)
        monkeypatch.setattr(main_window_module, "ask_seams", lambda *_args: None)
        window = MainWindow(work.project, confirm_unsaved=False)
        opened.append(window)
        _menu_action(window, work.video, 90, SPEED_TEXT).trigger()
        _menu_action(window, work.video, 90, SEAMS_TEXT).trigger()
        assert window.document.project == work.project
        assert not window.document.can_undo


class TestJetCutDialog:
    def test_smoothing_is_off_and_starts_from_the_preference(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、今までどおり詰めるだけのつもりの人のつなぎ目に処理が勝手に付くか、
        # 窓の欄が本人の設定と違う値から始まる
        del qt_application
        dialog = JetCutDialog(has_transcript=False, seam_style=SeamStyle(audio_fade_ms=50))
        try:
            assert dialog.seams() is None
            dialog._smooth.setChecked(True)
            assert dialog.seams() == SeamStyle(audio_fade_ms=50)
        finally:
            dialog.deleteLater()

    def test_the_cut_and_smoothing_are_one_request(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、窓で「つなぎ目も処理する」を選んでも処理が付かないか、詰めるのと別の段になり
        # 取り消し 1 回で戻らない
        del qt_application
        work = JumpCut()
        # 1.6 秒鳴って 1.1 秒黙り、また鳴る 1 本目の終わりの無音が切れる
        widget = SubtitlePanel(
            work.project, StubAnalyzer(make_waveform([(0.5, 300), (0.0, 200), (0.5, 5500)]))
        )
        widget.seam_style = SeamStyle(video=SEAM_NONE, audio_fade_ms=30)
        issued: list[tuple[list[Command], str]] = []
        widget.commands_requested.connect(lambda commands, label: issued.append((commands, label)))

        def accept(dialog: JetCutDialog) -> int:
            dialog._smooth.setChecked(True)
            return 1

        monkeypatch.setattr(JetCutDialog, "exec", accept)
        try:
            widget.jet_cut()
            commands, _label = issued[-1]
            assert isinstance(commands[0], RippleCut)
            assert isinstance(commands[1], SmoothCuts)
            assert commands[1].style == SeamStyle(video=SEAM_NONE, audio_fade_ms=30)
            assert commands[1].frames == seams_after_cuts(commands[0].ranges)
        finally:
            widget.deleteLater()
