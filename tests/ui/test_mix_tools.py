"""音量とミックスの道具（#329）を、本物の窓・本物の検査・本物のミキサで確かめる

偽のホストの試験（tests/ai/test_mix_tools.py）は道具が作る命令を見る ここでは、そろえた
音量が本当に基準の LUFS で鳴るか、ダッキングの線でミキサの音が本当に下がるか、設定画面の
値が道具の既定になるかを見る 音だけを扱うので GPU は要らない
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip
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
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending
from tests.check_media import tone, write_tone


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
    """AI のスレッドからブリッジ越しに道具を呼ぶ（本物の AI と同じ道） 変更は自動で承認する"""
    operation = find_operation(name)
    assert operation is not None
    # 送る物の確かめ（#342）は tests/ui/test_send_consent.py が見る ここは音の道筋だけを見る
    bridge = allow_sending(EditorBridge(window, auto_approve=True))
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120.0
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
        result = _call(window, "normalize_loudness", {"wait_seconds": 60})
        assert result["measured_lufs"] == pytest.approx(-20.0, abs=0.5)
        assert result["gain_db"] == pytest.approx(6.0, abs=0.5)
        check = _call(window, "check_output", {"checks": ["loudness"], "wait_seconds": 60})
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
