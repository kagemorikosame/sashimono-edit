"""書き出しの色変換を GPU で行うかの設定（#372）

速さと、手元で確かめていない駆動との相性の釣り合いなので本人の設定に出す 既定は入
（書き出す前に試験の絵で CPU の変換と比べ、食い違えば自分で CPU に戻る） 切った値が書き出しまで届く
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.encode import ExportSettings, build_settings
from sashimono.engine.encoders import PRIORITY_AUTO
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


def test_it_is_on_by_default_and_a_turned_off_setting_survives_a_restart(tmp_path: Path) -> None:
    # 既定がずれると、設定を触っていない人の書き出しが遅いままになる 保存した「切る」が
    # 読み戻せないと、駆動との相性で切った人の書き出しが次の起動で GPU に戻って乱れる
    assert Preferences().export_gpu_color is True
    assert ExportSettings(path=tmp_path / "a.mp4").gpu_color is True
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(export_gpu_color=False))
    assert store.load().export_gpu_color is False


def test_a_broken_value_in_the_file_falls_back_to_the_default(tmp_path: Path) -> None:
    # 手で書き換えた設定の壊れた値で起動ごと失敗すると、設定を消すまで編集できない
    path = tmp_path / "preferences.json"
    path.write_text('{"export_gpu_color": "はい"}', encoding="utf-8")
    assert PreferenceStore(path).load().export_gpu_color is True


def test_the_dialog_keeps_what_was_chosen(qt_application: QApplication) -> None:
    # 壊れると、設定画面を開いて〔OK〕を押しただけで、切っていた設定が入りに戻る
    del qt_application
    dialog = PreferencesDialog(Preferences(export_gpu_color=False))
    try:
        assert dialog.preferences().export_gpu_color is False
        dialog._export_gpu_color.setChecked(True)
        assert dialog.preferences().export_gpu_color is True
    finally:
        dialog.deleteLater()


def test_the_ai_export_takes_the_setting(tmp_path: Path) -> None:
    # AI の書き出しも同じ作り方で組む 届かないと、切った人の書き出しが AI からだけ GPU で変換される
    settings = build_settings(
        tmp_path / "a.mp4",
        codec=None,
        bitrate_mbps=12,
        frame_range=None,
        pipeline_depth=2,
        decode_threads=4,
        smooth_history=True,
        gpu_color=False,
        chapters=True,
        encoder_priority=PRIORITY_AUTO,
    )
    assert settings.gpu_color is False


def test_the_window_passes_the_setting_to_the_export(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 壊れると、設定画面で切っても書き出しの窓へ届かず、GPU で変換したまま書き出す
    del qt_application
    seen: list[bool] = []

    class Recorder:
        embeds_chapters = True
        started_folder = None

        def __init__(self, *_arguments: object, gpu_color: bool = True, **_options: object):
            seen.append(gpu_color)

        def exec(self) -> int:
            return 0

        def deleteLater(self) -> None:  # noqa: N802 - Qt の名前に合わせる
            """窓を作っていないので捨てる物も無い"""

    monkeypatch.setattr(main_window_module, "ExportDialog", Recorder)
    window = MainWindow(Project.create(), confirm_unsaved=False)
    try:
        window._preferences = Preferences(export_gpu_color=False)
        window.export()
    finally:
        window.close()
    assert seen == [False]
