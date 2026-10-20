"""ナレーション（#335）の設定画面・書き出しの知らせと、本物の窓で AI が作って置く声

偽のホストの試験（tests/ai/test_narration_tool.py）は道具の結果を見る ここでは、設定画面の
エンジンの場所と既定の声が保存されて道具の既定になること、本物の窓でブリッジ越しに作った声が
本物の読み方で読み込まれ、取り消し 1 回で戻ること、書き出しのときに表記を知らせることを見る
窓が開いたらその場で落とす（開いてよい所だけ、試験の中で差し替えて答える）
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddCredits
from sashimono.core.io.serialize import save_project
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.engine.audio.voicevox import AUTO_VOICE, DEFAULT_ENGINE_URL
from sashimono.ui.export_dialog import CREDIT_NOTE, ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import allow_sending, make_loaded
from tests.fake_clipboard import FakeClipboard
from tests.fake_voice_engine import FakeVoiceEngine, closed_engine_url, fake_voice_engine

WINDOWS_WRITES = pytest.mark.skipif(
    sys.platform != "win32", reason="AI がファイルを書くのは Windows だけ（pin_holds_the_folder）"
)


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす 開いてよい知らせは試験の中で差し替える

    開いたまま待つと、試験が人の操作を待って止まる
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


@pytest.fixture
def engine() -> Iterator[FakeVoiceEngine]:
    with fake_voice_engine() as running:
        yield running


@pytest.fixture
def told(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """知らせ（QMessageBox.information と warning）の文を受け取る 窓は開かない"""
    texts: list[str] = []

    def keep(_parent: object, _title: str, text: str, *_rest: object) -> int:
        texts.append(text)
        return int(QMessageBox.StandardButton.Ok.value)

    monkeypatch.setattr(QMessageBox, "information", keep)
    monkeypatch.setattr(QMessageBox, "warning", keep)
    return texts


class TestPreference:
    def test_the_defaults_reach_a_started_voicevox(self) -> None:
        # 壊れると、VOICEVOX を入れて起動しただけの人が、設定を開かないとつながらない
        plain = Preferences()
        assert plain.narration_engine_url == DEFAULT_ENGINE_URL
        assert plain.narration_voice == AUTO_VOICE
        assert plain.narration_settings.ai_picks_voice is True
        assert plain.narration_settings.speed_percent == 100

    def test_an_outside_place_in_the_file_falls_back(self) -> None:
        # 壊れると、設定ファイルを手で書き換えた外の場所へ、台本の文を送る
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"narration_engine_url": "http://192.168.1.10:50021", "narration_voice": -7,'
            ' "narration_speed_percent": 900, "narration_ai_picks_voice": "はい"}',
            encoding="utf-8",
        )
        loaded = PreferenceStore().load()
        assert loaded.narration_engine_url == DEFAULT_ENGINE_URL
        assert loaded.narration_voice == AUTO_VOICE
        assert loaded.narration_speed_percent == 100
        assert loaded.narration_ai_picks_voice is True

    def test_saved_values_are_read_back(self) -> None:
        # 壊れると、設定画面で選んだ AivisSpeech の場所や声が次の起動で消える
        chosen = Preferences(
            narration_engine_url="http://127.0.0.1:10101",
            narration_voice=888753760,
            narration_voice_name="まお（ノーマル）",
            narration_ai_picks_voice=False,
        )
        PreferenceStore().save(chosen)
        loaded = PreferenceStore().load()
        assert loaded.narration_settings == chosen.narration_settings


class TestDialog:
    def test_reading_the_engine_lists_voices_and_the_choice_is_returned(
        self, qt_application: QApplication, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、設定画面で声を選べないか、選んだ声が AI の既定にならない
        del qt_application
        dialog = PreferencesDialog(Preferences(narration_engine_url=engine.url))
        try:
            dialog._read_voices()
            box = dialog._narration_voice
            names = [box.itemText(index) for index in range(box.count())]
            assert names == [
                "エンジンの一覧の最初の声",
                "四国めたん（ノーマル）",
                "四国めたん（あまあま）",
                "ずんだもん（ノーマル）",
            ]
            box.setCurrentIndex(box.findData(3))
            dialog._narration_ai_picks.setChecked(False)
            dialog._narration_speed.setValue(120)
            chosen = dialog.preferences()
            assert chosen.narration_voice == 3
            assert chosen.narration_voice_name == "ずんだもん（ノーマル）"
            assert chosen.narration_ai_picks_voice is False
            assert chosen.narration_speed_percent == 120
            assert "VOICEVOX" in dialog._narration_status.text()
        finally:
            dialog.deleteLater()

    def test_a_stopped_engine_is_told_under_the_row_without_a_window(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、エンジンを起動していない人が〔声の一覧を読む〕を押すと窓が出るか、
        # 何も起きず理由が分からない
        del qt_application
        dialog = PreferencesDialog(
            Preferences(
                narration_engine_url=closed_engine_url(),
                narration_voice=3,
                narration_voice_name="ずんだもん（ノーマル）",
            )
        )
        try:
            dialog._read_voices()
            assert "起動" in dialog._narration_status.text()
            # 選んでいた声は、エンジンが動いていなくても名前のまま残る
            assert dialog._narration_voice.currentText() == "ずんだもん（ノーマル）"
            assert dialog.preferences().narration_voice == 3
        finally:
            dialog.deleteLater()

    def test_a_voice_missing_from_the_engine_is_kept_and_shown(
        self, qt_application: QApplication, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、別のエンジンへ替えて一覧を読んだだけで、選んでいた声が黙って最初の声に変わる
        del qt_application
        dialog = PreferencesDialog(Preferences(narration_engine_url=engine.url, narration_voice=42))
        try:
            dialog._read_voices()
            assert "このエンジンに無い" in dialog._narration_voice.currentText()
            assert dialog.preferences().narration_voice == 42
        finally:
            dialog.deleteLater()

    def test_the_terms_of_the_chosen_voice_are_shown_as_the_engine_says(
        self, qt_application: QApplication, engine: FakeVoiceEngine, told: list[str]
    ) -> None:
        # 壊れると、声ごとの規約（クレジットの書き方・非商用だけの声）を本人が確かめられない
        del qt_application
        dialog = PreferencesDialog(Preferences(narration_engine_url=engine.url, narration_voice=3))
        try:
            dialog._show_voice_terms()
            assert len(told) == 1
            assert "ずんだもんの規約 クレジットを書くこと" in told[0]
            assert "VOICEVOX:ずんだもん" in told[0]
        finally:
            dialog.deleteLater()

    def test_an_outside_place_keeps_the_dialog_open(
        self, qt_application: QApplication, told: list[str]
    ) -> None:
        # 壊れると、外の機械の場所を書いて OK を押した人の値が黙って前の値へ戻り、直した
        # つもりで気付かない
        del qt_application
        dialog = PreferencesDialog(Preferences())
        try:
            dialog._narration_url.setText("http://192.168.1.10:50021")
            dialog.accept()
            assert dialog.result() != QDialog.DialogCode.Accepted
            assert len(told) == 1
            assert "自分の機械の中" in told[0]
            assert dialog.preferences().narration_engine_url == DEFAULT_ENGINE_URL
        finally:
            dialog.deleteLater()


class TestExportTellsTheCredits:
    def _project(self) -> Project:
        return AddCredits(("VOICEVOX:ずんだもん", "AivisSpeech:まお")).apply(Project.create())

    def test_the_dialog_shows_and_copies_the_credits(
        self, qt_application: QApplication, fake_clipboard: FakeClipboard
    ) -> None:
        # 壊れると、声を使った作品を書き出す人に表記を知らせず、概要欄に入れ忘れる
        # （規約に反するのは本人）
        del qt_application
        dialog = ExportDialog(self._project())
        try:
            labels = [label.text() for label in dialog.findChildren(type(dialog._summary))]
            assert any(CREDIT_NOTE in text and "AivisSpeech:まお" in text for text in labels)
            dialog._copy_credits()
            assert fake_clipboard.text() == "VOICEVOX:ずんだもん\nAivisSpeech:まお"
        finally:
            dialog.deleteLater()

    def test_finishing_tells_the_credits_again(
        self, qt_application: QApplication, told: list[str]
    ) -> None:
        # 壊れると、書き出しの窓で読み流した人が、書き終えた後に表記を思い出せない
        del qt_application
        dialog = ExportDialog(self._project())
        try:
            dialog._on_finished("C:/動画/完成.mp4")
            assert len(told) == 1
            assert "VOICEVOX:ずんだもん" in told[0]
            assert CREDIT_NOTE in told[0]
        finally:
            dialog.deleteLater()

    def test_a_project_without_voices_says_nothing(
        self, qt_application: QApplication, told: list[str]
    ) -> None:
        # 壊れると、声を使っていない作品でも知らせが出て、知らせを読み流す癖が付く
        del qt_application
        dialog = ExportDialog(Project.create())
        try:
            dialog._on_finished("C:/動画/完成.mp4")
            assert CREDIT_NOTE not in told[0]
        finally:
            dialog.deleteLater()


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    windows: list[MainWindow] = []
    yield windows
    for window in windows:
        window._chat.close_session()
        window.close()
        window.deleteLater()


def _call(window: MainWindow, name: str, arguments: dict[str, Any]) -> Any:
    """AI のスレッドからブリッジ越しに道具を呼ぶ 取り消せない操作の確かめには許すで答える"""
    operation = find_operation(name)
    assert operation is not None
    bridge = allow_sending(EditorBridge(window, auto_approve=True))
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while thread.is_alive():
        bridge.pump()
        approval = bridge.take_approval()
        if approval is not None:
            approval.allow()
        QApplication.processEvents()
        thread.join(0.01)
        if time.monotonic() > deadline:
            bridge.cancel()
            thread.join(5.0)
            pytest.fail("道具が返らなかった")
    if "error" in box:
        raise box["error"]
    return box["result"]


@WINDOWS_WRITES
class TestWindow:
    def test_the_voice_is_made_read_placed_and_undone_once(
        self,
        opened: list[MainWindow],
        engine: FakeVoiceEngine,
        tmp_path: Path,
        video_media: MediaItem,
        transcript: Transcript,
    ) -> None:
        # 壊れると、本物の窓で作った WAV が素材として読めないか、置いた声と素材と表記を
        # 戻すのに取り消しが何回も要る
        PreferenceStore().save(Preferences(narration_engine_url=engine.url))
        project = replace(make_loaded(video_media, transcript), name="作品")
        path = tmp_path / "作品.sme"
        save_project(project, path)
        window = MainWindow(project, confirm_unsaved=False)
        opened.append(window)
        window._path = path
        assert window.narration_settings.url == engine.url

        _call(window, "list_voices", {})
        result = _call(
            window, "generate_narration", {"text": "こんにちは。", "voice_id": 3, "at_frame": 30}
        )
        written = Path(result["file"])
        assert written == (tmp_path / "narration" / "narration_001.wav").resolve()
        media = next(m for m in window.document.project.media if m.path == written)
        assert media.has_audio
        assert window.document.project.credits == ("VOICEVOX:ずんだもん",)
        assert str(written) in window.remembered_names()

        window.document.undo()
        assert window.document.project == project
        assert written.exists()
