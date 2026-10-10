"""着せ替えで折り返しの幅を残すかの設定と、取り消し 1 回で戻ること（#283）"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import insert_generated
from sashimono.core.model import AnimatedValue, GeneratedSource, Project
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.ui import template_dialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore

TEMPLATE = (
    "[Object]\nframe=0,179\n[Object.0]\neffect.name=テキスト\nサイズ=60.00\n"
    "フォント=Noto Sans JP\n文字装飾=標準文字\nテキスト=字幕テキスト\n"
    "[Object.1]\neffect.name=標準描画\nX=0.00\nY=400.00\n"
)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application

    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


class _Chosen:
    """テンプレートの棚の代わり 開くとすぐ〔選択中のクリップに適用〕で閉じる"""

    choice: tuple[str, list[MappedObject]] = ("restyle", [])

    def __init__(self, parent: object = None) -> None:
        del parent
        self.origin: Path | None = None

    def exec(self) -> QDialog.DialogCode:
        return QDialog.DialogCode.Accepted

    def deleteLater(self) -> None:  # noqa: N802 - 本物の棚（Qt の窓）と同じ名前
        """開いた側は閉じたあとに捨てる 代わりの物には捨てる中身が無い"""


def _restyle(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> GeneratedSource:
    styled = TEXT.create(text="字幕", font="Segoe UI", font_style="Semibold", wrap_width=640)
    for command in insert_generated(window.document.project, styled, at_frame=0):
        window.execute(command)
    clip = window.document.project.timeline.tracks[0].clips[0]
    window.select_clip(clip.id)
    objects = [
        item
        for obj in parse_exo(TEMPLATE).objects
        if (item := map_object(obj, FrameRate(30))) is not None
    ]
    monkeypatch.setattr(_Chosen, "choice", ("restyle", objects))
    monkeypatch.setattr(template_dialog, "TemplateDialog", _Chosen)
    window.show_templates()
    source = window.document.project.timeline.tracks[0].clips[0].source
    assert isinstance(source, GeneratedSource)
    return source


def _wrap(source: GeneratedSource) -> float:
    value = source.params["wrap_width"]
    assert isinstance(value, AnimatedValue)
    return value.at(0)


class TestWindow:
    def test_the_default_matches_the_template(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window._preferences = Preferences()
        source = _restyle(window, monkeypatch)
        assert (source.params["font"], source.params["font_style"]) == ("Noto Sans JP", "")
        assert _wrap(source) == 0.0

    def test_the_setting_keeps_the_width(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 設定を入れても幅が消えるなら、設定がある方が質が悪い
        window._preferences = Preferences(restyle_keep_wrap=True)
        source = _restyle(window, monkeypatch)
        assert _wrap(source) == 640.0
        assert source.params["font_style"] == ""

    def test_one_undo_brings_the_old_look_back(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 着せ替えは書き換えと効果の付け外しの何段にもなる 1 回で戻らないと、途中の
        # どちらとも違う見た目を通って戻ることになる
        window._preferences = Preferences()
        _restyle(window, monkeypatch)
        window.undo()
        source = window.document.project.timeline.tracks[0].clips[0].source
        assert isinstance(source, GeneratedSource)
        assert (source.params["font"], source.params["font_style"]) == ("Segoe UI", "Semibold")
        assert _wrap(source) == 640.0


class TestPreferences:
    def test_the_default_clears_the_width(self) -> None:
        # 既定は棚の見本と同じ見た目になる側
        assert Preferences().restyle_keep_wrap is False

    def test_the_choice_is_saved(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(restyle_keep_wrap=True))
        assert store.load().restyle_keep_wrap is True

    def test_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"restyle_keep_wrap": "はい"}), encoding="utf-8")
        assert PreferenceStore(path).load().restyle_keep_wrap is False

    def test_the_dialog_shows_and_returns_it(self, qt_application: QApplication) -> None:
        del qt_application
        chosen = Preferences(restyle_keep_wrap=True)
        dialog = PreferencesDialog(chosen)
        assert dialog.preferences() == chosen
        dialog._restyle_keep_wrap.setChecked(False)
        assert dialog.preferences().restyle_keep_wrap is False
