"""互換性レポート 何が足りなくて動かないのかを見せる

AviUtl の ``obj`` API は広い 全部を一度に実装することはできないので、大事なのは
「動かない」ことではなく**何が足りないかが分かること**にした

スクリプトが未対応の関数を呼んだら記録され、ここに使用回数つきで並ぶ 次に何を
実装すべきかを、勘ではなく実際に使われた回数で決められる
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono import __version__
from sashimono.compat.aviutl.catalog import script_catalog
from sashimono.compat.aviutl.report import CompatibilityReport, global_report
from sashimono.ui.report_masking import marked_root_labels, mask_report, root_lines
from sashimono.ui.system_clipboard import clipboard
from sashimono.ui.theme import Colors, themed_style

__all__ = ["CompatibilityDialog", "report_text"]


def report_text(
    report: CompatibilityReport,
    scripts: int,
    folders: Sequence[tuple[str, str]] | None = None,
    roots: Sequence[Path | str] = (),
) -> str:
    """不具合の報告に貼る文面 版と、画面に出ている記録を全部入れる

    版を頭に入れるのは、同じ記録でも版によって直っているかが変わるため
    貼る人に版を別に調べさせると、欄が空のまま届く
    探索先は伏せた形で入れる AviUtl2 の Script を見に行っているかどうかは、
    読めない原因を追うのに要る
    """
    lines = [
        f"Sashimono Edit {__version__} 互換性レポート",
        report.summary(),
        f"読み込み済みのスクリプト {scripts} 本",
        *root_lines([str(root) for root in roots]),
        *report.lines(),
    ]
    return mask_report("\n".join(lines), roots, folders)


class CompatibilityDialog(QDialog):
    """未対応 API と読み込みの失敗を一覧で出す"""

    def __init__(
        self, report: CompatibilityReport | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AviUtl 互換性レポート")
        self.resize(560, 420)
        self._report = report if report is not None else global_report

        self._summary = QLabel(self)
        self._summary.setWordWrap(True)

        self._scripts = QLabel(self)
        self._scripts.setWordWrap(True)
        themed_style(self._scripts, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._list = QListWidget(self)

        clear = QPushButton("記録を消す", self)
        clear.clicked.connect(self._clear)
        rescan = QPushButton("スクリプトを読み直す", self)
        rescan.clicked.connect(self._rescan)
        # 一覧からは行を 1 つずつしか選べず、Ctrl+C でも写せない 不具合の報告に
        # 貼ってもらうには、全部をまとめて写す口が要る
        copy = QPushButton("内容をコピー", self)
        copy.clicked.connect(self.copy_to_clipboard)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.setText("閉じる")
        buttons.addButton(copy, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(rescan, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(clear, QDialogButtonBox.ButtonRole.ResetRole)

        layout = QVBoxLayout(self)
        layout.addWidget(self._summary)
        layout.addWidget(self._scripts)
        layout.addWidget(self._list, 1)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self) -> None:
        catalog = script_catalog()
        entries = catalog.all()
        self._summary.setText(self._report.summary())

        # 貼る文で印に置き換える探索先は、画面では印を添えて出す
        labels = marked_root_labels(catalog.roots)
        self._scripts.setText(
            "\n".join([f"スクリプト {len(entries)} 本を読み込み済み", *root_lines(labels)])
        )

        self._list.clear()
        lines = self._report.lines()
        self._list.addItems(lines if lines else ["まだ記録はありません"])

    def copy_to_clipboard(self) -> None:
        """画面の記録を、報告に貼れる形でクリップボードへ写す"""
        catalog = script_catalog()
        clipboard().setText(report_text(self._report, len(catalog.all()), roots=catalog.roots))

    def _clear(self) -> None:
        self._report.clear()
        self.refresh()

    def _rescan(self) -> None:
        catalog = script_catalog()
        catalog.scan()
        catalog.register_all()
        self.refresh()
