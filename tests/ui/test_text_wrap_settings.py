"""折り返しの幅の欄と、新しい字幕を自動で折り返す設定（#249）"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import insert_generated
from sashimono.core.model import Clip, GeneratedSource, Project
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.panel import InspectorPanel
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def _show(panel: InspectorPanel, source: GeneratedSource) -> None:
    project = Project.create()
    for command in insert_generated(project, source):
        project = command.apply(project)
    clip: Clip = next(c for t in project.timeline.tracks for c in t.clips)
    panel.set_project(project)
    panel.set_clip(clip.id)


class TestInspector:
    def test_the_width_can_be_set_on_a_plain_text(self, panel: InspectorPanel) -> None:
        _show(panel, TEXT.create())
        editor = panel._editors[("source", "wrap_width")]
        assert editor.isEnabled()

    def test_the_aviutl_layout_greys_the_width_out(self, panel: InspectorPanel) -> None:
        # 動かしても絵が変わらない欄を触れるままにすると、壊れたように見える 隠すと、組み方を
        # 変えれば使えることに気付けない 灰色にして理由を出す
        _show(panel, TEXT.create(layout="aviutl"))
        editor = panel._editors[("source", "wrap_width")]
        assert not editor.isEnabled()
        assert "AviUtl2" in editor.toolTip()


class TestPreferences:
    def test_the_default_wraps_at_ninety_percent(self) -> None:
        # 既定は入 知らない人ほど、長い字幕が画面の端で切れて困る
        plain = Preferences()
        assert plain.subtitle_wrap is True
        assert plain.subtitle_wrap_share == 90

    def test_turning_it_off_stops_the_wrap(self) -> None:
        # 切ったのに幅が入ると、設定がある方が質が悪い
        assert Preferences(subtitle_wrap=False).subtitle_wrap_share == 0

    def test_a_broken_share_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "preferences.json"
        path.write_text(
            json.dumps({"subtitle_wrap": "はい", "subtitle_wrap_percent": 500}), encoding="utf-8"
        )
        loaded = PreferenceStore(path).load()
        assert loaded.subtitle_wrap is True
        assert loaded.subtitle_wrap_percent == 90

    def test_the_choice_is_saved(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(subtitle_wrap=False, subtitle_wrap_percent=75))
        loaded = store.load()
        assert (loaded.subtitle_wrap, loaded.subtitle_wrap_percent) == (False, 75)

    def test_the_dialog_shows_and_returns_it(self, qt_application: QApplication) -> None:
        # 開いて OK を押しただけで値が変わらないこと
        del qt_application
        chosen = Preferences(subtitle_wrap=False, subtitle_wrap_percent=70)
        dialog = PreferencesDialog(chosen)
        assert dialog.preferences() == chosen
        assert not dialog._subtitle_wrap_percent.isEnabled()
        dialog._subtitle_wrap.setChecked(True)
        assert dialog._subtitle_wrap_percent.isEnabled()
