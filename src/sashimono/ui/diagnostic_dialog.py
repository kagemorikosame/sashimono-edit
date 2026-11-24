"""診断情報の窓（ヘルプの〔診断情報を作る…〕 #516）

作った報告の全文を、伏せた後の形で見せる 本人が読んで、気になる所を消してから保存する
自動では送らない 窓口（Discussions）へ添えるのは本人 保存しないで閉じれば何も残らない

組むのは :mod:`sashimono.diagnostic_report` ここは見せる・保存する・写すだけ
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.io.atomic import write_atomic
from sashimono.diagnostic_report import suggested_name
from sashimono.ui import system_clipboard

__all__ = ["DiagnosticDialog", "ReportFolderField", "default_report_folder", "report_folder"]

#: 窓の頭に出す説明
EXPLANATION = (
    "不具合を調べるための情報です 自動では送りません\n"
    "開いている作品の素材の名前と字幕・ファイルの場所・利用者名・API の鍵などは伏せてあります"
    " 読んで、ほかにも出したくない所があれば消してから保存してください\n"
    "保存したファイルは、報告の窓口（不具合・要望を送る）の書き込みに添えてください"
)


def default_report_folder() -> Path:
    """設定が空のときに保存の窓を開く所 「ドキュメント」 取れなければホーム"""
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(documents) if documents else Path.home()


def report_folder(chosen: str) -> Path:
    """保存の窓を開く所 選んだフォルダが無くなっていれば既定へ戻す

    無いフォルダを渡すと、窓が見当違いの所（最後に開いた所）で開き、どこへ保存したかを見失う
    """
    if chosen and Path(chosen).is_dir():
        return Path(chosen)
    return default_report_folder()


class ReportFolderField(QWidget):
    """設定画面の「診断情報を保存する場所」 空は既定 選ぶボタンと既定へ戻すボタンを添える"""

    def __init__(self, folder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.line = QLineEdit(folder, self)
        # 空を「既定」と見せる 既定の場所を文字で入れると、利用者名の違う機械へ設定を
        # 写したときに、その場所を指したまま保存できなくなる（パッケージの置き場と同じ）
        self.line.setPlaceholderText(f"既定（{default_report_folder()}）")
        self.line.setReadOnly(True)
        choose = QPushButton("選ぶ…", self)
        choose.clicked.connect(self._choose)
        reset = QPushButton("既定に戻す", self)
        reset.clicked.connect(self.line.clear)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line, 1)
        layout.addWidget(choose)
        layout.addWidget(reset)

    def folder(self) -> str:
        """選んだ場所 空は既定の場所"""
        return self.line.text().strip()

    def _choose(self) -> None:
        start = str(report_folder(self.line.text()))
        chosen = QFileDialog.getExistingDirectory(self, "診断情報を保存する場所", start)
        if chosen:
            self.line.setText(str(Path(chosen)))


class DiagnosticDialog(QDialog):
    """伏せた報告を見せて、本人が保存するかを決める窓

    ``open_report`` は〔報告の窓口を開く〕で呼ぶ（編集画面のヘルプと同じ所を開く）
    保存した場所は :attr:`saved_path` に残る 呼んだ側が、次に開く所として覚える
    """

    def __init__(
        self,
        text: str,
        *,
        folder: str = "",
        open_report: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("診断情報")
        self.setObjectName("diagnosticDialog")
        self._folder = folder
        self._open_report = open_report
        #: 保存したファイル 保存していなければ ``None``
        self.saved_path: Path | None = None

        layout = QVBoxLayout(self)
        note = QLabel(EXPLANATION, self)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.text = QPlainTextEdit(text, self)
        # 並びの字下げと表が崩れないよう、幅のそろった書体で見せる
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text, 1)

        buttons = QHBoxLayout()
        save = QPushButton("ファイルに保存…", self)
        save.setDefault(True)
        save.clicked.connect(self._choose_and_save)
        copy = QPushButton("写す", self)
        copy.setToolTip("報告の書き込みへ直に貼るときに使う 写すのはこの窓の今の中身")
        copy.clicked.connect(self._copy)
        page = QPushButton("報告の窓口を開く", self)
        page.setEnabled(open_report is not None)
        page.clicked.connect(self._open)
        close = QPushButton("閉じる（保存しない）", self)
        close.clicked.connect(self.reject)
        for button in (save, copy, page):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.resize(760, 560)

    def current_text(self) -> str:
        """本人が直した後の中身 保存と写しはこれを使う（直す前の物を出さない）"""
        return self.text.toPlainText()

    def save_to(self, path: Path) -> bool:
        """``path`` へ書く 書けなければ理由を窓で知らせて偽

        書いてから置き換える 同じ名前を選んだときに、書き損じで前の報告を壊さない
        """
        try:
            write_atomic(path, self.current_text())
        except OSError as exc:
            QMessageBox.warning(self, "診断情報", f"保存できませんでした\n{exc}")
            return False
        self.saved_path = path
        return True

    def _choose_and_save(self) -> None:
        start = report_folder(self._folder) / suggested_name()
        chosen, _filter = QFileDialog.getSaveFileName(
            self, "診断情報を保存", str(start), "テキスト (*.txt)"
        )
        if chosen and self.save_to(Path(chosen)):
            self.accept()

    def _copy(self) -> None:
        system_clipboard.copy_text(self.current_text())

    def _open(self) -> None:
        if self._open_report is not None:
            self._open_report()
