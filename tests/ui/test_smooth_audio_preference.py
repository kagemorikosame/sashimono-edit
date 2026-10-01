"""前の音を読む音の効果の動く値を、区切りの中でもつなぐかの設定（PR #231 の指摘）

速さと忠実さの釣り合いなので本人の設定に出す 既定は入（つないでも予算に収まる）
再生と書き出しで同じ音にするため、どちらのミキサにも同じ値が届く
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.encode import ExportSettings
from sashimono.ui.playback import PlaybackController
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


def test_a_turned_off_setting_stays_off_after_restart(tmp_path: Path) -> None:
    # 壊れると、保存した「切る」が読み戻せず、切った設定が次の起動で入りに戻る 既定（入）と
    # 書き出しの既定がずれると、設定を触っていない人でも再生と書き出しの音が食い違う
    assert Preferences().smooth_audio_motion is True
    assert ExportSettings(path=tmp_path / "a.mp4").smooth_history is True
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(smooth_audio_motion=False))
    assert store.load().smooth_audio_motion is False


def test_the_dialog_keeps_what_was_chosen(qt_application: QApplication) -> None:
    # 壊れると、設定画面を開いて〔OK〕を押しただけで、切っていた設定が入りに戻る
    # （画面に今の値が出ない） あるいは、チェックを替えても保存されない
    del qt_application
    dialog = PreferencesDialog(Preferences(smooth_audio_motion=False))
    try:
        assert dialog.preferences().smooth_audio_motion is False
        dialog._smooth_audio_motion.setChecked(True)
        assert dialog.preferences().smooth_audio_motion is True
    finally:
        dialog.deleteLater()


def test_a_changed_setting_reaches_the_playback_mixer(qt_application: QApplication) -> None:
    # 壊れると、設定を替えても再生のミキサへ値が届かず、プレビューは前の掛け方のまま鳴り、
    # 書き出し（設定から値を渡す）と音が食い違う
    del qt_application
    playback = PlaybackController(Project.create(), smooth_history=False)
    try:
        assert playback._mixer.smooth_history is False
        playback.set_smooth_history(True)
        assert playback._mixer.smooth_history is True
    finally:
        playback.close()


def test_the_export_gets_the_same_setting_as_playback(qt_application: QApplication) -> None:
    # 壊れると、プレビューでは切った掛け方で鳴るのに、書き出した動画は入りの掛け方で鳴る
    del qt_application
    from sashimono.ui.export_dialog import ExportDialog

    dialog = ExportDialog(Project.create(), smooth_history=False)
    try:
        settings = dialog._settings()
        if settings is None:
            pytest.skip("映像のコーデックが無い")
        assert settings.smooth_history is False
    finally:
        dialog.deleteLater()
