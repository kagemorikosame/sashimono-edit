"""前の音を読む音の効果の動く値を、区切りの中でもつなぐかの設定（PR #231 の指摘）

速さと忠実さの釣り合いなので本人の設定に出す 既定は入（つないでも予算に収まる）
再生と書き出しで同じ音にするため、どちらのミキサにも同じ値が届く
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.encode import ExportSettings
from sashimono.ui.playback import PlaybackController
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


def test_it_is_on_unless_turned_off(tmp_path: Path) -> None:
    assert Preferences().smooth_audio_motion is True
    assert ExportSettings(path=tmp_path / "a.mp4").smooth_history is True
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(smooth_audio_motion=False))
    assert store.load().smooth_audio_motion is False


def test_the_dialog_shows_and_returns_it(qt_application: QApplication) -> None:
    del qt_application
    dialog = PreferencesDialog(Preferences(smooth_audio_motion=False))
    try:
        assert dialog.preferences().smooth_audio_motion is False
        dialog._smooth_audio_motion.setChecked(True)
        assert dialog.preferences().smooth_audio_motion is True
    finally:
        dialog.deleteLater()


def test_playback_hands_it_to_the_mixer(qt_application: QApplication) -> None:
    del qt_application
    playback = PlaybackController(Project.create(), smooth_history=False)
    try:
        assert playback._mixer.smooth_history is False
        playback.set_smooth_history(True)
        assert playback._mixer.smooth_history is True
    finally:
        playback.close()
