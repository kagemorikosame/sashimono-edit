"""書き出しの窓の「コーデック」の欄が埋まるまで待つ（#464）

窓は、エンコーダの答えをまだ覚えていなければ裏で確かめ、終わってから欄を並べる
覚えた答えを試験の前に忘れる試験（probe_here）では、窓を作った直後の欄は「確かめています」
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from sashimono.ui.export_dialog import ExportDialog
from tests.waiting import wait_until


def settled(dialog: ExportDialog) -> ExportDialog:
    """裏の確かめが終わって欄が埋まるまで画面の手を回し、同じ窓を返す"""
    wait_until(
        lambda: not dialog.checking_codecs, "書き出しの窓の候補", step=QApplication.processEvents
    )
    return dialog
