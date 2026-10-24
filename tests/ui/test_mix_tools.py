"""音量とミックスの道具（#329）を、本物の窓・本物の検査・本物のミキサで確かめる

偽のホストの試験（tests/ai/test_mix_tools.py）は道具が作る命令を見る ここでは、そろえた
音量が本当に基準の LUFS で鳴るか、ダッキングの線でミキサの音が本当に下がるか、設定画面の
値が道具の既定になるかを見る 音だけを扱うので GPU は要らない
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.core.commands import AddClip, AddScene, SetClipProperty, SetTrackState, new_scene
from sashimono.core.commands.fixed import FADE_EFFECT_KIND, VOLUME_EFFECT_KIND, fixed_effect
from sashimono.core.commands.volume import clip_volume
from sashimono.core.mixing import (
    DEFAULT_DUCK_ATTACK_MS,
    DEFAULT_DUCK_DEPTH_DB,
    DEFAULT_DUCK_RELEASE_MS,
    DuckShape,
)
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio import AudioMixer
from sashimono.engine.decode.probe import probe_media
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.mix_dialogs import DuckChoice, DuckDialog, LoudnessChoice, LoudnessDialog
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.timeline.view import TimelineView
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.check_media import tone, write_tone
from tests.ui.ai_tool_calls import call_tool, wait_for_check


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    音量の道具は取り消せる編集で、確かめはブリッジの承認（ここでは自動で承認）だけ
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


@dataclass(frozen=True)
class _Media:
    #: -20 dBFS の 1 kHz（両方のチャンネル）は -20 LUFS 配信の基準より 6 LU 小さい
    quiet: MediaItem
    #: 声の代わり 字幕の区間（1〜3 秒）を持たせる
    voice: MediaItem


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> _Media:
    folder = tmp_path_factory.mktemp("mix")
    quiet = probe_media(write_tone(folder / "小さい.wav", tone(6.0, -20.0)))
    voice = probe_media(write_tone(folder / "声.wav", tone(6.0, -20.0)))
    words = Transcript(
        segments=(TranscriptSegment(start=Fraction(1), end=Fraction(3), text="こんにちは"),),
        language="ja",
        model="test",
    )
    return _Media(quiet, voice.with_transcript(words))


def _sound(media: MediaItem, duration: int) -> Clip:
    return Clip(
        timeline_start=0,
        duration=duration,
        media_id=media.id,
        effects=(fixed_effect(VOLUME_EFFECT_KIND), fixed_effect(FADE_EFFECT_KIND)),
    )


def _project(media: _Media) -> tuple[Project, Track, Track]:
    """声とBGM（どちらも 6 秒）を別の音声トラックに置いた 30 fps の作品"""
    base = Project.create(
        ProjectSettings(frame_rate=FrameRate(30)), media=(media.quiet, media.voice)
    )
    voice = Track(kind=TrackKind.AUDIO, name="声")
    bgm = Track(kind=TrackKind.AUDIO, name="BGM")
    project = base.with_timeline(replace(base.timeline, tracks=(voice, bgm)))
    project = AddClip(voice.id, _sound(media.voice, 180)).apply(project)
    project = AddClip(bgm.id, _sound(media.quiet, 180)).apply(project)
    return project, voice, bgm


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    windows: list[MainWindow] = []
    yield windows
    for window in windows:
        window.close()
        window.deleteLater()


def _window(opened: list[MainWindow], project: Project) -> MainWindow:
    window = MainWindow(project, confirm_unsaved=False)
    opened.append(window)
    return window


def _call(window: MainWindow, name: str, arguments: dict[str, Any]) -> Any:
    """本物の AI と同じ道で道具を呼ぶ（変更の承認と待ちの上限は tests/ui/ai_tool_calls.py）"""
    return call_tool(window, name, arguments)


def _measured(
    window: MainWindow,
    name: str,
    arguments: dict[str, Any],
    finished: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """測る道具を、窓の検査が測り終えた後の答えで返す

    道具に秒を渡して待たせない 渡した秒で測り終えない遅い機械では、道具が途中を返して
    落ちる（#405） 1 回目は待たずに測り始めさせ、検査の終わりの知らせを待ってから
    同じ引数で呼び直す 測り終えた頼みは使い回されるので、2 回目は待たずに答えが出る
    1 回目のうちに測り終えていれば、それが答え（呼び直すと、そろえた後の作品を測り直す）
    """
    quick = {**arguments, "wait_seconds": 0}
    first: dict[str, Any] = _call(window, name, quick)
    if finished(first):
        return first
    wait_for_check(window)
    again: dict[str, Any] = _call(window, name, quick)
    assert finished(again), again
    return again


def _rms_db(project: Project, frame: int) -> float:
    mixer = AudioMixer(project)
    try:
        samples = mixer.render_frames(frame, 1)
    finally:
        mixer.close()
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


class TestNormalize:
    def test_the_work_ends_up_at_the_target(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、AI が「そろえた」と言っても、書き出す音は基準の LUFS にならない
        project, _voice, bgm = _project(media)
        alone = replace(
            project,
            timeline=replace(
                project.timeline,
                tracks=tuple(t for t in project.timeline.tracks if t.id == bgm.id),
            ),
        )
        window = _window(opened, alone)
        result = _measured(window, "normalize_loudness", {}, lambda done: "measuring" not in done)
        assert result["measured_lufs"] == pytest.approx(-20.0, abs=0.5)
        assert result["gain_db"] == pytest.approx(6.0, abs=0.5)
        check = _measured(
            window, "check_output", {"checks": ["loudness"]}, lambda done: done["complete"]
        )
        assert check["summary"]["integrated_lufs"] == pytest.approx(-14.0, abs=0.5)
        # 取り消し 1 回で測る前の音量へ戻る
        window.document.undo()
        assert window.document.project == alone


class TestDuck:
    def test_the_mixer_plays_the_bgm_lower_while_speaking(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、線に点は打たれても鳴る音が下がらず、声が BGM に埋もれたまま書き出される
        project, voice, bgm = _project(media)
        window = _window(opened, project)
        _call(
            window,
            "duck_audio",
            {"track_id": str(bgm.id), "voice_track_id": str(voice.id)},
        )
        ducked = window.document.project
        bgm_only = replace(
            ducked,
            timeline=replace(
                ducked.timeline,
                tracks=tuple(replace(t, muted=t.id == voice.id) for t in ducked.timeline.tracks),
            ),
        )
        # 話している 1〜3 秒の真ん中と、話していない 5 秒の所
        difference = _rms_db(bgm_only, 60) - _rms_db(bgm_only, 150)
        assert difference == pytest.approx(-DEFAULT_DUCK_DEPTH_DB, abs=0.5)

    def test_the_preference_shapes_the_dip(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、設定画面で選んだ下げ幅が AI のダッキングに効かない
        PreferenceStore().save(Preferences(duck_depth_db=20))
        project, voice, bgm = _project(media)
        window = _window(opened, project)
        assert window.duck_shape == DuckShape(
            depth_db=20, attack_ms=DEFAULT_DUCK_ATTACK_MS, release_ms=DEFAULT_DUCK_RELEASE_MS
        )
        result = _call(
            window,
            "duck_audio",
            {"track_id": str(bgm.id), "voice_track_id": str(voice.id)},
        )
        assert result["depth_db"] == 20
        (clip,) = next(t for t in window.document.project.timeline.tracks if t.id == bgm.id).clips
        assert min(k.value for k in clip_volume(clip).keyframes) == pytest.approx(10.0)


class TestPreference:
    def test_the_defaults_are_moderate(self) -> None:
        # 壊れると、何も選んでいない人の BGM が消えるほど下がるか、ほとんど下がらない
        plain = Preferences()
        assert (plain.duck_depth_db, plain.duck_attack_ms, plain.duck_release_ms) == (12, 300, 800)

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        # 壊れると、手で書き換えた値（0 dB や負の速さ）のまま、ダッキングが効かないか線が崩れる
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"duck_depth_db": 0, "duck_attack_ms": -5, "duck_release_ms": "速く"}',
            encoding="utf-8",
        )
        loaded = PreferenceStore().load()
        assert loaded.duck_shape == DuckShape()

    def test_the_dialog_offers_and_returns_the_values(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面で下げ幅や速さを変えても保存されない
        del qt_application
        dialog = PreferencesDialog(
            Preferences(duck_depth_db=8, duck_attack_ms=150, duck_release_ms=1200)
        )
        try:
            chosen = dialog.preferences()
            assert (chosen.duck_depth_db, chosen.duck_attack_ms, chosen.duck_release_ms) == (
                8,
                150,
                1200,
            )
            dialog._duck_depth.setValue(18)
            dialog._duck_attack.setValue(500)
            dialog._duck_release.setValue(2000)
            chosen = dialog.preferences()
            assert chosen.duck_shape == DuckShape(depth_db=18, attack_ms=500, release_ms=2000)
        finally:
            dialog.deleteLater()


DUCK_TEXT = "BGM を声に合わせて下げる…"
LOUDNESS_TEXT = "音量を基準にそろえる…"


def _clip_point(view: TimelineView, track: Track, frame: int) -> QPoint:
    for band in view.view_layout.bands(view.project.timeline):
        if band.track.id == track.id:
            return QPoint(int(view.view_layout.frame_to_x(frame)), (band.top + band.bottom) // 2)
    raise AssertionError(f"トラックが無い: {track.name}")


def _menu_action(window: MainWindow, track: Track, text: str) -> QAction:
    view = window._timeline
    view.resize(1200, 400)
    menu = view.build_context_menu(_clip_point(view, track, 30))
    found = next((a for a in menu.actions() if a.text() == text), None)
    assert found is not None, [a.text() for a in menu.actions()]
    return found


def _bgm_clip(window: MainWindow, bgm: Track) -> Clip:
    (clip,) = next(t for t in window.document.project.timeline.tracks if t.id == bgm.id).clips
    return clip


class TestRightClick:
    """タイムラインの右クリックから人が使う（#329 2026-10-17 利用者の決定）

    窓は main_window の ask_duck / ask_loudness を差し替えて答える 差し替えずに窓が開くと
    no_windows が落とす
    """

    def test_both_items_are_on_a_sound_clip(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、人は AI に頼まないと BGM を声に合わせて下げられない
        project, _voice, bgm = _project(media)
        window = _window(opened, project)
        assert _menu_action(window, bgm, DUCK_TEXT).isEnabled()
        assert _menu_action(window, bgm, LOUDNESS_TEXT).isEnabled()

    def test_a_locked_track_greys_them_out(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、人が触らせないためにロックしたトラックの BGM の音量が右クリックで変わる
        project, _voice, bgm = _project(media)
        project = SetTrackState(bgm.id, locked=True).apply(project)
        window = _window(opened, project)
        assert not _menu_action(window, bgm, DUCK_TEXT).isEnabled()
        assert not _menu_action(window, bgm, LOUDNESS_TEXT).isEnabled()

    def test_ducking_from_the_menu_matches_the_tool(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が右クリックで下げた BGM と AI が下げた BGM の線が食い違う
        PreferenceStore().save(Preferences(duck_depth_db=15))
        project, voice, bgm = _project(media)
        asked: list[tuple[list[str], DuckShape]] = []

        def answer(_parent: object, voices: list[Track], shape: DuckShape) -> DuckChoice:
            asked.append(([t.name for t in voices], shape))
            return DuckChoice(voice.id, shape)

        monkeypatch.setattr(main_window_module, "ask_duck", answer)
        by_hand = _window(opened, project)
        _menu_action(by_hand, bgm, DUCK_TEXT).trigger()
        # 声のトラックの候補は下げるクリップの外 下げ方の既定は本人の設定
        assert asked == [(["声"], DuckShape(depth_db=15))]
        by_tool = _window(opened, project)
        _call(by_tool, "duck_audio", {"track_id": str(bgm.id), "voice_track_id": str(voice.id)})
        assert clip_volume(_bgm_clip(by_hand, bgm)) == clip_volume(_bgm_clip(by_tool, bgm))
        assert clip_volume(_bgm_clip(by_hand, bgm)).keyframes
        # 取り消し 1 回で戻る
        by_hand.document.undo()
        assert by_hand.document.project == project

    def test_a_muted_voice_is_not_offered(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、書き出しで鳴らない声のトラックを選べて、声が無いのに BGM だけが下がる
        # 窓の差し替えの中で落とすと Qt が例外を飲むので、呼ばれたかを数えて後で見る
        asked: list[object] = []
        monkeypatch.setattr(main_window_module, "ask_duck", lambda *args: asked.append(args))
        project, voice, bgm = _project(media)
        project = SetTrackState(voice.id, muted=True).apply(project)
        window = _window(opened, project)
        _menu_action(window, bgm, DUCK_TEXT).trigger()
        assert asked == []
        assert window.document.project == project

    def test_a_disabled_clip_is_not_levelled(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、鳴らない（無効の）クリップの音量まで変え、有効に戻すとそろっていない
        monkeypatch.setattr(
            main_window_module, "ask_loudness", lambda *_args: LoudnessChoice(-14.0)
        )
        project, _voice, bgm = _project(media)
        (clip,) = next(t for t in project.timeline.tracks if t.id == bgm.id).clips
        project = SetClipProperty(clip.id, "enabled", False).apply(project)
        window = _window(opened, project)
        _menu_action(window, bgm, LOUDNESS_TEXT).trigger()
        assert window._pending_loudness is None
        assert window.document.project == project

    def test_a_scene_beside_the_narration_still_ducks(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、声のトラックに置いたシーンが 1 本あるだけで、右クリックから下げられない
        project, voice, bgm = _project(media)
        scene = new_scene(project, "場面")
        project = AddScene(scene).apply(project)
        project = AddClip(voice.id, Clip(timeline_start=180, duration=30, scene_id=scene.id)).apply(
            project
        )
        monkeypatch.setattr(
            main_window_module,
            "ask_duck",
            lambda _parent, _voices, shape: DuckChoice(voice.id, shape),
        )
        window = _window(opened, project)
        _menu_action(window, bgm, DUCK_TEXT).trigger()
        assert clip_volume(_bgm_clip(window, bgm)).keyframes

    def test_cancelling_changes_nothing(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、窓でやめたのに BGM が下がる
        monkeypatch.setattr(main_window_module, "ask_duck", lambda *_args: None)
        project, _voice, bgm = _project(media)
        window = _window(opened, project)
        _menu_action(window, bgm, DUCK_TEXT).trigger()
        assert window.document.project == project

    def test_loudness_from_the_menu_measures_then_applies(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、右クリックでそろえても基準の音量にならないか、測る間に画面が固まる
        asked: list[float] = []

        def answer(_parent: object, target: float) -> LoudnessChoice:
            asked.append(target)
            return LoudnessChoice(-14.0)

        monkeypatch.setattr(main_window_module, "ask_loudness", answer)
        project, _voice, bgm = _project(media)
        window = _window(opened, project)
        _menu_action(window, bgm, LOUDNESS_TEXT).trigger()
        # 基準の既定は本人の設定（仕上がりを調べる音量の基準）
        assert asked == [-14.0]
        # 60 秒で区切ると、遅い機械では測り終える前に落ちる（#405） 検査の終わりの知らせを待つ
        wait_for_check(window, then=lambda: window._pending_loudness is None)
        # BGM だけを測るので -20 LUFS から +6 dB 声のクリップは変えない
        assert clip_volume(_bgm_clip(window, bgm)).static == pytest.approx(200.0, rel=0.06)
        voice_clip = window.document.project.timeline.tracks[0].clips[0]
        assert clip_volume(voice_clip).static == 100.0
        window.document.undo()
        assert window.document.project == project


class TestDialogs:
    def test_the_duck_dialog_starts_from_the_given_shape(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、窓の既定が設定と違い、OK を押すだけで設定と違う下げ方になる
        del qt_application
        voice = Track(kind=TrackKind.AUDIO, name="ナレーション")
        dialog = DuckDialog([voice], DuckShape(depth_db=9, attack_ms=200, release_ms=600))
        try:
            assert dialog.choice() == DuckChoice(
                voice.id, DuckShape(depth_db=9, attack_ms=200, release_ms=600)
            )
            dialog.depth_box.setValue(20)
            assert dialog.choice().shape.depth_db == 20
        finally:
            dialog.deleteLater()

    def test_the_loudness_dialog_starts_from_the_preference(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、放送向けに -23 を選んだ人の窓が配信の -14 で開く
        del qt_application
        dialog = LoudnessDialog(-23.0)
        try:
            assert dialog.choice() == LoudnessChoice(-23.0, -1.0)
        finally:
            dialog.deleteLater()
