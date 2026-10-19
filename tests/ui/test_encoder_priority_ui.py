"""どのエンコーダを先に使うかの設定と、書き出しの窓の AMD の項目（#365）

GPU のエンコーダを使うか・どの GPU を先に使うかは人によって答えが違う
（速さと CPU の空きを取るか、同じビットレートでの画質を取るか） 表示 → 設定… から選べ、
書き出しの窓・AI の書き出し・控えの作成のすべてに効く 既定は自動（NVIDIA → AMD → Intel → CPU）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from sashimono.ai.host import ExportRequest
from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.io.atomic import expect_now
from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.encode import RANGE_ALL, ExportSettings, exporter
from sashimono.engine.encoders import (
    ENCODER_PRIORITIES,
    PRIORITY_AMD,
    PRIORITY_AUTO,
    PRIORITY_CPU,
)
from sashimono.ui.export_dialog import AUTO_CODEC, CODEC_LABELS, ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import ENCODER_PRIORITY_CHOICES, PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """尋ねる窓・知らせる窓・ファイルを選ぶ窓が開いたらその場で落とす

    開いたまま待つと、試験は誰も押さないボタンを待って止まる
    """

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


@pytest.fixture
def every_codec_opens(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """NVIDIA・AMD・Intel の GPU と CPU のすべてで書ける機械を真似る

    この開発機には AMD も Intel の GPU も無い 並びと優先を確かめるには、全部が開ける形にする
    """
    exporter._opens.cache_clear()
    monkeypatch.setattr(exporter, "_open_encoder", lambda _name: None)
    yield
    exporter._opens.cache_clear()


def _codecs_in(dialog: ExportDialog) -> list[str]:
    """書き出しの窓の「コーデック」に並んだ名前 自動の項目は除く"""
    box = dialog._codec
    return [str(box.itemData(i)) for i in range(box.count()) if box.itemData(i) != AUTO_CODEC]


def test_the_default_is_auto_and_it_survives_a_restart(tmp_path: Path) -> None:
    # 既定が自動でないと、設定を触っていない人の書き出しが遅い CPU に落ちたり、
    # 保存した選びを読み戻せず、次の起動で自動へ戻ったりする
    assert Preferences().encoder_priority == PRIORITY_AUTO
    assert ExportSettings(path=tmp_path / "a.mp4").encoder_priority == PRIORITY_AUTO
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(encoder_priority=PRIORITY_AMD))
    assert store.load().encoder_priority == PRIORITY_AMD


def test_an_unknown_value_falls_back_to_auto(tmp_path: Path) -> None:
    # 手で書き換えた設定や、後の版で足した項目を読んだときに、書き出しが断られ続ける
    path = tmp_path / "preferences.json"
    path.write_text('{"encoder_priority": "matrox"}', encoding="utf-8")
    assert PreferenceStore(path).load().encoder_priority == PRIORITY_AUTO


def test_the_preferences_dialog_lists_every_choice_and_keeps_it(
    qt_application: QApplication,
) -> None:
    # 項目が抜けると、選べない優先が残るか、〔OK〕を押しただけで選んでいた優先が自動へ戻る
    del qt_application
    assert [value for _label, value in ENCODER_PRIORITY_CHOICES] == list(ENCODER_PRIORITIES)
    dialog = PreferencesDialog(Preferences(encoder_priority=PRIORITY_CPU))
    try:
        assert dialog.preferences().encoder_priority == PRIORITY_CPU
        box = dialog._encoder_priority
        box.setCurrentIndex(box.findData(PRIORITY_AMD))
        assert dialog.preferences().encoder_priority == PRIORITY_AMD
    finally:
        dialog.deleteLater()


def test_amd_has_a_name_in_the_export_dialog() -> None:
    # 名前が無いと、書き出しの窓に h264_amf と素の名前が出て、AMD の GPU だと分からない
    assert CODEC_LABELS["h264_amf"] == "H.264 (AMD GPU)"
    assert CODEC_LABELS["hevc_amf"] == "H.265 (AMD GPU)"


def test_the_export_dialog_follows_the_priority(
    qt_application: QApplication, every_codec_opens: None
) -> None:
    # 効かないと、設定で AMD を先にと選んでも、書き出しの窓の自動が NVENC を使う
    del qt_application
    plain = ExportDialog(Project.create())
    amd = ExportDialog(Project.create(), encoder_priority=PRIORITY_AMD)
    try:
        assert _codecs_in(plain)[:2] == ["h264_nvenc", "h264_amf"]
        assert _codecs_in(amd)[0] == "h264_amf"
        assert "AMD GPU" in amd._codec.itemText(0)
        settings = amd._settings()
        assert settings is not None and settings.encoder_priority == PRIORITY_AMD
    finally:
        plain.deleteLater()
        amd.deleteLater()


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


def test_the_window_passes_the_priority_to_the_export_dialog(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 渡さないと、設定で選んだ優先が書き出しの窓に効かない
    window._preferences = replace(window._preferences, encoder_priority=PRIORITY_CPU)
    seen: list[str] = []

    def look(dialog: ExportDialog) -> int:
        seen.append(dialog._encoder_priority)
        return 0

    monkeypatch.setattr(ExportDialog, "exec", look)
    window.export()
    assert seen == [PRIORITY_CPU]


def test_the_assistant_export_follows_the_priority(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # AI の書き出し（#327）が設定を見ないと、CPU を先にと選んだ人の動画を AI だけ GPU で書く
    window._preferences = replace(window._preferences, encoder_priority=PRIORITY_CPU)
    started: list[ExportSettings] = []
    monkeypatch.setattr(
        window._assistant_export,
        "start",
        lambda _project, settings, _check, _expect=None: started.append(settings),
    )
    clip = Clip(timeline_start=0, duration=30, source=TEXT.create())
    track = Track(TrackKind.VIDEO, "V1")
    window._document.execute(AddTrack(track))
    window._document.execute(AddClip(track.id, clip))
    path = tmp_path / "完成.mp4"
    window.start_export(
        ExportRequest(
            path=path,
            scene_id=None,
            range_choice=RANGE_ALL,
            codec=None,
            bitrate_mbps=12,
            expect=expect_now(path),
        )
    )
    assert [settings.encoder_priority for settings in started] == [PRIORITY_CPU]


def test_changing_the_priority_remakes_the_proxy_builder(window: MainWindow) -> None:
    # 作り直さないと、設定で替えた優先が、次に起動し直すまで控えの作成に効かない
    assert window._proxies.encoder_priority == PRIORITY_AUTO
    window._apply_preferences(replace(window._preferences, encoder_priority=PRIORITY_CPU))
    assert window._proxies.encoder_priority == PRIORITY_CPU
