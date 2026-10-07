"""導入ボタンを押したあと、そのまま字幕起こしとアシスタントが使えること（Issue #27）

配布版の置き場（専用フォルダ）で、本物の pip の代わりに、置くはずの物を書く
偽物で導入する 直す前は、どちらも導入のあとボタンが押せないままだった
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.asr import TranscriptionService
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.runtime import LEFT_RUNNING_NOTE
from tests.ai.conftest import FakeHost, make_loaded
from tests.test_runtime_after_install import (
    _fake_installer,
    _left_running_installer,
    _LeftPip,
    _slow_cancelled_installer,
    _wait_for,
    frozen,
    isolated_imports,
    write_distribution,
)

__all__ = ["frozen", "isolated_imports"]  # 試験の部品として使う（ruff に未使用と言わせない）


class _IdleService:
    """起こしの受け付け 導入の試験では起こしまで進めないので、持っているだけ"""


class TestTranscribeAfterInstall:
    def test_the_run_button_is_enabled_right_after_installing(
        self,
        frozen: Path,
        monkeypatch: pytest.MonkeyPatch,
        qt_application: QApplication,
        video_media: MediaItem,
    ) -> None:
        from sashimono.asr import runtime_status
        from sashimono.ui.subtitle import transcribe_dialog

        if runtime_status().installed:
            pytest.skip("この環境には faster-whisper が入っていて、未導入の状態を作れない")

        monkeypatch.setattr(
            transcribe_dialog,
            "install_runtime",
            _fake_installer(
                lambda: write_distribution(frozen, "faster-whisper", "faster_whisper", "1.2.0")
            ),
        )
        # 起こしの受け付けは導入が済むまで 1 度も呼ばれない 本物を作ると
        # faster-whisper を読みに行くので、何もしない代わりを型だけ合わせて渡す
        service = cast(TranscriptionService, _IdleService())
        dialog = transcribe_dialog.TranscribeDialog(video_media, service)
        assert dialog._run_button.isEnabled() is False

        dialog._start_install()
        _wait_for(lambda: dialog._install_done is None, qt_application)

        assert dialog._run_button.isEnabled() is True
        assert "再起動しなくても" in dialog._status.text()
        dialog.deleteLater()


class TestTranscribeInstallCancelled:
    def test_closing_during_the_install_does_not_report_success(
        self,
        frozen: Path,
        monkeypatch: pytest.MonkeyPatch,
        qt_application: QApplication,
        video_media: MediaItem,
    ) -> None:
        """導入中に「閉じる（中断）」を押した瞬間に、成功と読まないこと

        読むと、pip が走っている最中に「再起動しなくてもそのまま使えます」と出て、
        まだ入っていないのに起こすボタンを押せるように見える
        """
        from sashimono.ui.subtitle import transcribe_dialog

        monkeypatch.setattr(transcribe_dialog, "install_runtime", _slow_cancelled_installer())
        service = cast(TranscriptionService, _IdleService())
        dialog = transcribe_dialog.TranscribeDialog(video_media, service)

        dialog._start_install()
        dialog.reject()  # 中断を頼む
        _wait_for(lambda: dialog._install_done is None, qt_application)

        assert "使えます" not in dialog._status.text()
        assert "導入に失敗しました" in dialog._log.toPlainText()
        dialog.deleteLater()


class TestTranscribeInstallLeftRunning:
    def test_a_pip_left_running_blocks_the_button_until_it_comes_back(
        self,
        frozen: Path,
        monkeypatch: pytest.MonkeyPatch,
        qt_application: QApplication,
        video_media: MediaItem,
    ) -> None:
        """字幕起こしの導入の窓も、pip が戻らずに残っている間は導入を押せなくして再起動を頼む"""
        from sashimono.ui.subtitle import transcribe_dialog

        left = _LeftPip()
        monkeypatch.setattr(transcribe_dialog, "pip_left_running", left)
        monkeypatch.setattr(transcribe_dialog, "install_runtime", _left_running_installer(left))
        service = cast(TranscriptionService, _IdleService())
        dialog = transcribe_dialog.TranscribeDialog(video_media, service)

        dialog._start_install()
        _wait_for(lambda: dialog._install_done is None, qt_application)

        assert dialog._install_button.isEnabled() is False
        assert LEFT_RUNNING_NOTE in dialog._status.text()
        dialog._start_install()
        assert dialog._install_done is None

        left.left = False
        dialog._check_leftover()
        assert dialog._install_button.isEnabled() is True
        assert LEFT_RUNNING_NOTE not in dialog._status.text()
        dialog.deleteLater()


@pytest.fixture
def without_installed_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """開発機に入っている SDK と claude を見えなくする 配る先の機械と同じ状態"""
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if "site-packages" not in p.replace("\\", "/")]
    )
    for name in list(sys.modules):
        if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
            monkeypatch.delitem(sys.modules, name)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    for variable in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)


def write_sdk(site: Path) -> Path:
    """claude-agent-sdk を pip で入れたときの形 Claude Code 本体を同梱している"""
    write_distribution(site, "claude-agent-sdk", "claude_agent_sdk", "0.2.158")
    bundled = site / "claude_agent_sdk" / "_bundled" / "claude.exe"
    bundled.parent.mkdir(parents=True, exist_ok=True)
    bundled.write_bytes(b"")
    return bundled


class TestAssistantAfterInstall:
    def test_the_input_opens_right_after_installing(
        self,
        frozen: Path,
        without_installed_sdk: None,
        monkeypatch: pytest.MonkeyPatch,
        qt_application: QApplication,
        video_media: MediaItem,
        transcript: Transcript,
    ) -> None:
        """直す前は 2 つの理由で開かなかった

        入れた SDK が import の道に載らない（配布版）ことと、SDK に同梱された
        Claude Code を見ずに、別に入れた ``claude`` を PATH に探していたこと
        """
        del without_installed_sdk
        from sashimono.ui import setup
        from sashimono.ui.chat import ChatPanel

        monkeypatch.setattr(setup, "install_runtime", _fake_installer(lambda: write_sdk(frozen)))
        project: Project = make_loaded(video_media, transcript)
        panel = ChatPanel(FakeHost(project))
        assert panel._input.isEnabled() is False

        results: list[bool] = []
        panel._setup.finished.connect(results.append)
        panel._setup.start()
        _wait_for(lambda: bool(results), qt_application)

        assert results == [True]
        assert panel._input.isEnabled() is True
        assert panel._send_button.isEnabled() is True
        assert panel._parts_box.isHidden() is True
        # ログインがまだなので、その案内が出る
        assert panel._login_box.isHidden() is False
        # 導入欄は隠れるので、再起動の要る・要らないの案内は会話の欄に残す
        assert "再起動しなくても" in panel._view.toPlainText()
        panel.close_session()
        panel.deleteLater()
