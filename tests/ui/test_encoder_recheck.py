"""設定画面の「エンコーダを確かめ直す」（#464）

GPU の駆動の不調で使えなかったエンコーダは、起動中ずっと使えない物として覚えている
駆動が直ったときに、閉じずに GPU の書き出しへ戻せることをここで押さえる
"""

from __future__ import annotations

import threading

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.engine import encoders
from sashimono.ui.encoder_recheck import EncoderRecheck, recheck_text
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences
from tests.waiting import wait_until


def _settle(recheck: EncoderRecheck) -> None:
    """裏の確かめが終わり、窓へ返るまで画面の手を回す"""
    wait_until(lambda: not recheck.running, "確かめ直し", step=QApplication.processEvents)


class TestRecheckText:
    def test_rechecking_forgets_a_trouble_that_has_gone(
        self, probe_here: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """前に落ちた物の答えを捨てて確かめ直し、使える物を並べる

        捨てないと、駆動を入れ直して直っても閉じるまで CPU で書き出し続ける
        """
        encoders._verdicts["h264_nvenc"] = False
        encoders._troubles["h264_nvenc"] = "確かめの途中でプロセスごと落ちた"
        monkeypatch.setattr(encoders, "probe_encoder", lambda _name: None)
        text = recheck_text()
        assert encoders.probe_troubles() == {}
        assert "h264_nvenc" in text
        assert "使えなかった物" not in text

    def test_a_trouble_that_stays_is_told_with_the_way_out(
        self, probe_here: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 理由を出さないと、押しても GPU に戻らない訳が分からない
        def broken(name: str) -> None:
            if name == "h264_nvenc":
                # 確かめで落ちた形を残す 本物は子が落ちたときに _probe_in_child が書く
                encoders._troubles[name] = "確かめの途中でプロセスごと落ちた"
                raise RuntimeError("落ちた")

        monkeypatch.setattr(encoders, "probe_encoder", broken)
        text = recheck_text()
        assert "使えなかった物" in text
        assert "再起動" in text


class TestDialog:
    def test_the_button_rechecks_without_freezing_the_dialog(
        self, qt_application: QApplication
    ) -> None:
        """押すと裏で確かめ直し、終わったら結果を行の下に出す

        画面のスレッドで確かめると、固まった駆動を待ち切るまで設定の窓が固まる
        """
        del qt_application
        gate = threading.Event()

        def slow() -> str:
            gate.wait(5)
            return "使えるエンコーダ: libx264"

        dialog = PreferencesDialog(Preferences())
        try:
            dialog._encoder_recheck = EncoderRecheck(dialog, slow)
            dialog._recheck_encoders()
            assert "確かめています" in dialog._encoder_status.text()
            dialog._recheck_encoders()
            assert "前の確かめ直しがまだ終わっていません" in dialog._encoder_status.text()
            gate.set()
            _settle(dialog._encoder_recheck)
            assert dialog._encoder_status.text() == "使えるエンコーダ: libx264"
        finally:
            gate.set()
            dialog.deleteLater()
