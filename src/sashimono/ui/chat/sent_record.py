"""AI へ送った物を見る窓（#425 表示 → 設定… の「送った物を見る…」）

上に、種類ごとに最後に送った物の下見（アプリを閉じるまでだけ持つ中身）を出す
「尋ねずに送る」にした後も、何が出ているかを確かめられるように
下に、送った物の記録（日時・道具・種類・件数 中身は持たない）を新しい順に並べる 記録は
ここで消せる
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai.send_log import KEEP_DAYS, SendLog
from sashimono.ui.chat.send_consent import SampleView

__all__ = ["CLEAR_TEXT", "SentRecordDialog"]

CLEAR_TEXT = "記録を消す"
#: 記録の一覧に出す件数 これより古い物はファイルに残るが、窓には出さない
#: 何千行を並べると開くのが重くなる 見直したいのは最近の物
SHOWN_ENTRIES = 500


class SentRecordDialog(QDialog):
    """最後に送った物と、送った物の記録"""

    def __init__(self, log: SendLog, parent: QWidget | None = None) -> None:
        """最後に送った物と、記録の一覧を並べる"""
        super().__init__(parent)
        self._log = log
        self.setWindowTitle("AI へ送った物")
        layout = QVBoxLayout(self)
        heading = QLabel("最後に送った物（アプリを閉じると忘れます）", self)
        layout.addWidget(heading)
        body = QWidget(self)
        column = QVBoxLayout(body)
        self.last_views: list[QWidget] = []
        last = log.last()
        for sent in last.values():
            when = QLabel(f"{sent.at:%Y-%m-%d %H:%M:%S} {sent.tool}", body)
            column.addWidget(when)
            view = SampleView(sent.sample, body)
            self.last_views += [when, view]
            column.addWidget(view)
        if not last:
            column.addWidget(QLabel("まだ何も送っていません", body))
        column.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(min(body.sizeHint().height() + 8, 320))
        layout.addWidget(scroll)

        keep = "残しています" if log.keep else "残さない設定です（前に残した分だけ出ます）"
        layout.addWidget(
            QLabel(
                f"送った物の記録（日時・道具・種類・件数 中身は残しません {KEEP_DAYS} 日まで）"
                f" 今は{keep}",
                self,
            )
        )
        self.entries = QPlainTextEdit(self)
        self.entries.setReadOnly(True)
        self.entries.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.entries)
        self._fill()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.clear = QPushButton(CLEAR_TEXT, self)
        self.clear.setToolTip("送った物の記録と、最後に送った物の下見を消す 元には戻せません")
        buttons.addButton(self.clear, QDialogButtonBox.ButtonRole.ResetRole)
        self.clear.clicked.connect(self._clear)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(640, 560)

    def _fill(self) -> None:
        """記録の一覧を新しい順に出す 多いときは古い物を省く"""
        entries = self._log.entries()
        lines = [entry.text() for entry in entries[:SHOWN_ENTRIES]]
        if len(entries) > SHOWN_ENTRIES:
            lines.append(f"…ほか {len(entries) - SHOWN_ENTRIES} 件（古い物）")
        self.entries.setPlainText("\n".join(lines) if lines else "記録はありません")

    def _clear(self) -> None:
        """確かめてから、記録と最後に送った物を消す"""
        answer = QMessageBox.question(
            self,
            CLEAR_TEXT,
            "送った物の記録と、最後に送った物の下見を消します 元には戻せません よろしいですか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self._log.clear():
            QMessageBox.warning(self, CLEAR_TEXT, "記録のファイルを消せませんでした")
        for view in self.last_views:
            view.setVisible(False)
        self._fill()
