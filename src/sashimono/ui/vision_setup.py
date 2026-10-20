"""画面の解析（#324）の部品を入れる窓

〔表示〕→〔設定…〕の〔画面の解析の部品…〕から開く 中身は字幕起こしと AI 連携と同じ
導入の欄（:class:`~sashimono.ui.setup.SetupSection`）で、何が入るかを先に見せ、ログを流す

導入の途中で閉じられたら止めるよう頼み、止まるまで窓を閉じない 閉じた窓の欄が消えた後に
裏の導入が終わると、終わりの知らせを受ける相手が無く、導入先の錠を持ったまま残る
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from sashimono.engine.vision import VISION_PACK
from sashimono.runtime import FeaturePack
from sashimono.ui.setup import SetupSection

__all__ = ["VisionSetupDialog"]

#: 止めるよう頼んだ後、終わったかを見る間隔（ミリ秒）
_CLOSE_POLL_MS = 150


class VisionSetupDialog(QDialog):
    """画面の解析の部品の導入の窓"""

    def __init__(self, parent: QWidget | None = None, *, pack: FeaturePack = VISION_PACK) -> None:
        super().__init__(parent)
        self.setWindowTitle("画面の解析の部品")
        intro = QLabel(
            "AI アシスタントが素材の画面の中の文字（テロップ・看板・資料）と顔の位置を調べるための"
            "部品です 文字は Windows の OCR で、顔は顔の検出のモデル（YuNet）で、どちらもこの"
            "パソコンの中で調べ、絵は外へ送りません 読めた文字は「AI へ送る物」の「字幕と文字」"
            "として、送る前に尋ねます onnxruntime は字幕起こしと同じ物を使います",
            self,
        )
        intro.setWordWrap(True)
        self.section = SetupSection(pack, self)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.section)
        layout.addStretch(1)
        layout.addWidget(self._buttons)
        self._closing = QTimer(self)
        self._closing.setInterval(_CLOSE_POLL_MS)
        self._closing.timeout.connect(self._close_when_idle)
        self.resize(560, 320)

    def reject(self) -> None:
        if self.section.busy:
            # 止めるよう頼み、止まってから閉じる（このファイルの頭の説明）
            self.section.cancel()
            self._buttons.setEnabled(False)
            self._closing.start()
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self.section.busy:
            event.ignore()
            self.reject()
            return
        super().closeEvent(event)

    def _close_when_idle(self) -> None:
        if self.section.busy:
            return
        self._closing.stop()
        self._buttons.setEnabled(True)
        super().reject()
