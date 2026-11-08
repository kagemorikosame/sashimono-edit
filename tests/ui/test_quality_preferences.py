"""通しの確かめ（#414）で足した好みの設定 既定・保存・設定の窓・書き出しの窓"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project, ProjectSettings
from sashimono.engine.encode import DEFAULT_BITRATE_MBPS, BitrateChoice
from sashimono.ui.workspace import Preferences, PreferenceStore

#: 変えた値（既定と違う物）
CHANGED: dict[str, Any] = {
    "loudness_limiter": False,
    "ai_auto_approve": True,
    "export_bitrate_auto": False,
    "export_bitrate_mbps": 30,
}


def test_the_defaults_help_people_who_do_not_know() -> None:
    """既定を逆にすると、知らない人が基準に届かない音・大きすぎるファイル・黙って進む編集に困る"""
    plain = Preferences()
    assert plain.loudness_limiter is True
    assert plain.export_bitrate_auto is True
    assert plain.export_bitrate_mbps == DEFAULT_BITRATE_MBPS
    # 取り消せる編集でも黙って進むと困る人がいるので、任せるのは本人が入れたときだけ
    assert plain.ai_auto_approve is False
    assert plain.export_bitrate == BitrateChoice(auto=True, mbps=DEFAULT_BITRATE_MBPS)


def test_the_choices_survive_a_restart(tmp_path: Path) -> None:
    """保存し忘れると、変えた設定が次の起動で戻る"""
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(**CHANGED))
    loaded = store.load()
    for name, value in CHANGED.items():
        assert getattr(loaded, name) == value, name


def test_broken_values_fall_back(tmp_path: Path) -> None:
    """壊れた値で起動を止めず、その項目だけ既定へ戻す"""
    path = tmp_path / "preferences.json"
    path.write_text(
        json.dumps({"export_bitrate_mbps": 0, "loudness_limiter": "いいえ"}), encoding="utf-8"
    )
    loaded = PreferenceStore(path).load()
    assert loaded.export_bitrate_mbps == DEFAULT_BITRATE_MBPS
    assert loaded.loudness_limiter is True


@pytest.mark.parametrize("changed", [True, False])
def test_the_settings_dialog_keeps_them(qt_application: QApplication, changed: bool) -> None:
    """設定の窓に出し忘れると、開いて OK を押しただけで既定に戻る"""
    from sashimono.ui.preferences_dialog import PreferencesDialog

    del qt_application
    preferences = Preferences(**CHANGED) if changed else Preferences()
    dialog = PreferencesDialog(preferences)
    try:
        kept = dialog.preferences()
        for name in CHANGED:
            assert getattr(kept, name) == getattr(preferences, name), name
    finally:
        dialog.deleteLater()


def test_the_export_window_starts_from_the_choice(qt_application: QApplication) -> None:
    """壊れると、設定で決めたビットレートが書き出しの窓に出ず、決め打ちの 12 Mbps に戻る"""
    from sashimono.ui.export_dialog import ExportDialog

    del qt_application
    project = Project.create(ProjectSettings())
    dialog = ExportDialog(project, bitrate=BitrateChoice(auto=False, mbps=30))
    try:
        assert dialog._bitrate.value() == 30
    finally:
        dialog.deleteLater()
