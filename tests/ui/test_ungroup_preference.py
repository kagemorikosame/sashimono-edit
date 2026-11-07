"""AI がグループを解く前に聞くかの設定（#428 ``Preferences.ai_ask_before_ungroup``）"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import MediaItem, Transcript
from sashimono.ui.chat import ChatPanel
from sashimono.ui.workspace import Preferences
from tests.ai.conftest import FakeHost, make_loaded
from tests.ui.test_chat_panel import _ready


def test_it_is_on_from_the_start() -> None:
    """既定が切だと、知らない人ほどカットを頼んだだけで束ねたクリップが黙ってばらばらになる"""
    assert Preferences().ai_ask_before_ungroup is True


def test_the_panel_hands_it_to_the_bridge(
    qt_application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    video_media: MediaItem,
    transcript: Transcript,
) -> None:
    """パネルが橋渡しへ渡し忘れると、切っても尋ね続け、入れ直しても尋ねない"""
    del qt_application
    _ready(monkeypatch)
    widget = ChatPanel(FakeHost(make_loaded(video_media, transcript)))
    try:
        widget.apply_preferences(Preferences(ai_ask_before_ungroup=False))
        assert widget._bridge.ask_first is False
        widget.apply_preferences(Preferences(ai_ask_before_ungroup=True))
        assert widget._bridge.ask_first is True
    finally:
        widget.close_session()
        widget.deleteLater()


def test_the_choice_survives_a_restart(tmp_path: Path) -> None:
    """保存し忘れると、切ったはずの問いが次の起動で戻る"""
    from sashimono.ui.workspace import PreferenceStore

    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(ai_ask_before_ungroup=False))
    assert store.load().ai_ask_before_ungroup is False


@pytest.mark.parametrize("value", [True, False])
def test_the_settings_dialog_keeps_it(qt_application: QApplication, value: bool) -> None:
    """設定の窓に出し忘れると、開いて OK を押しただけで既定に戻る"""
    from sashimono.ui.preferences_dialog import PreferencesDialog

    del qt_application
    dialog = PreferencesDialog(Preferences(ai_ask_before_ungroup=value))
    try:
        assert dialog.preferences().ai_ask_before_ungroup is value
    finally:
        dialog.deleteLater()
