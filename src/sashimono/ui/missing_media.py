"""見つからない素材の知らせと、探し直しの窓（#447）

作品と素材のフォルダを別の PC へ移すと、絵は映らず音は無音になるのに、前は何も知らせなかった
開いたときに見つからない素材を並べ、1 本を選び直すと同じフォルダの残りもまとめて付け替える
窓は待たずに出す（閉じるまで止めない） 開いた直後に止めると、作品を見ながら探せない
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.model import MediaId, MediaItem, Project
from sashimono.core.relink import missing_media

__all__ = ["MissingMediaDialog", "nearest_folder"]


def nearest_folder(path: Path) -> Path:
    """``path`` の上で今もあるフォルダ 探す窓を元の場所の近くから始める"""
    probe = Path(path).parent
    while not probe.is_dir() and probe != probe.parent:
        probe = probe.parent
    return probe if probe.is_dir() else Path.home()


class MissingMediaDialog(QDialog):
    """見つからない素材の一覧 〔探し直す…〕で選んだ素材を付け替える

    付け替え（確かめ・残りの探し・コマンド）は ``relink`` に任せる（窓と AI の道具が同じ所を通る）
    ``relink`` は付け替えた後の知らせの文を返す
    """

    def __init__(
        self,
        project: Callable[[], Project],
        relink: Callable[[MediaId, Path], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("見つからない素材")
        self._project = project
        self._relink = relink
        self._summary = QLabel(self)
        self._summary.setWordWrap(True)
        self._list = QListWidget(self)
        self._list.itemDoubleClicked.connect(lambda _item: self._search())
        self._result = QLabel(self)
        self._result.setWordWrap(True)
        search = QPushButton("探し直す…", self)
        search.clicked.connect(self._search)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.addButton(search, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self._summary)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._result)
        layout.addWidget(buttons)
        self.resize(560, 360)
        self.refresh()

    def missing(self) -> tuple[MediaItem, ...]:
        """今の作品で見つからない素材"""
        return missing_media(self._project())

    def refresh(self) -> None:
        """一覧を作り直す 選んでいた行はなるべく残す"""
        current = self._list.currentItem()
        chosen = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._list.clear()
        items = self.missing()
        for item in items:
            row = QListWidgetItem(f"{item.name}\n    元の場所: {item.path}")
            row.setData(Qt.ItemDataRole.UserRole, str(item.id))
            self._list.addItem(row)
            if str(item.id) == chosen:
                self._list.setCurrentItem(row)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._summary.setText(
            f"素材のファイルが {len(items)} 本見つかりません 絵は映らず、音は鳴りません\n"
            "〔探し直す…〕で 1 本を選ぶと、同じフォルダか同じ並びのフォルダにある残りも"
            "名前で見つけて付け替えます（取り消しで戻せます）"
            if items
            else "見つからない素材はありません"
        )

    def _search(self) -> None:
        """選んだ行の素材を探す窓を出し、選んだファイルへ付け替える"""
        current = self._list.currentItem()
        if current is None:
            return
        media_id = MediaId(str(current.data(Qt.ItemDataRole.UserRole)))
        item = next((m for m in self.missing() if m.id == media_id), None)
        if item is None:
            return
        suffix = Path(item.path).suffix
        name, _ = QFileDialog.getOpenFileName(
            self,
            f"「{item.name}」を探す",
            str(nearest_folder(Path(item.path)) / Path(item.path).name),
            f"元と同じ種類 (*{suffix});;すべてのファイル (*)" if suffix else "すべてのファイル (*)",
        )
        if not name:
            return
        self._result.setText(self._relink(media_id, Path(name)))
        self.refresh()
