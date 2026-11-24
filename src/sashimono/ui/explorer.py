"""エクスプローラー 登録したフォルダの素材を探して試し、タイムラインへ置く（#520 #521）

作品に読み込んだ素材の一覧（:mod:`sashimono.ui.media_pool`）とは別のパネルにする 手持ちの
効果音や BGM を使うたびに読み込みの窓でファイルを探していた手間を、ここで登録したフォルダを
見て回るだけにする AI アシスタントが自動で使う素材の出どころも、ここで登録したフォルダ（#522）

登録の保存と見て回る所はコア（:mod:`sashimono.core.material_library`）に置き、ここは薄くする
置いた物は作品の素材に入り、読み込みと同じ命令で 1 回の取り消しで戻る（窓の ``import_media``）
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, QUrl, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.material_library import (
    AUDIO_SUFFIXES,
    CATEGORIES,
    FolderScan,
    LibraryFile,
    LibraryFolder,
    MeasureBook,
    add_folder,
    match_files,
    remove_folder,
    scan_folder,
)
from sashimono.ui.flow_layout import FlowLayout, narrow_combo
from sashimono.ui.media_icons import GRID_ICON_SIZE, thumbnail_icon

__all__ = [
    "ALL_FOLDERS",
    "DISPLAY_LIMIT",
    "ExplorerPanel",
    "ask_category",
    "ask_folder",
]

#: フォルダの選択肢の「すべて」の値
ALL_FOLDERS = ""
#: 一覧に並べる行の上限 数万の行を並べると、絞り込みの 1 文字ごとに画面が止まる
#: 越えたら絞り込むよう知らせる
DISPLAY_LIMIT = 3000

#: 行の持つファイルの場所
_PATH = Qt.ItemDataRole.UserRole


def ask_folder(parent: QWidget) -> str | None:
    """登録するフォルダを選ばせる 選ばなければ ``None``"""
    chosen = QFileDialog.getExistingDirectory(parent, "登録するフォルダを選ぶ")
    return chosen or None


def ask_category(parent: QWidget, current: str | None = None) -> str | None:
    """フォルダの分類を選ばせる やめたら ``None``

    分類はフォルダごとに 1 つ（2026-11-24 本人の決定） ファイルごとに付けさせない
    """
    names = list(CATEGORIES.values())
    keys = list(CATEGORIES)
    start = keys.index(current) if current in keys else 0
    name, ok = QInputDialog.getItem(
        parent,
        "フォルダの分類",
        "このフォルダの素材の分類（AI に使わせるかは 表示 → 設定… で分類ごとに選ぶ）",
        names,
        start,
        False,
    )
    if not ok:
        return None
    return keys[names.index(name)]


class _FileList(QListWidget):
    """ファイルの行を、ファイルの URL を載せてタイムラインへ引いていけるようにした一覧

    タイムラインはエクスプローラー（OS）から落としたファイルと同じく、読み込んで落とした所へ
    置く（取り消し 1 回で戻る） 素材一覧の行と違い、まだ作品の素材ではないので ID は載せない
    """

    def mimeTypes(self) -> list[str]:  # noqa: N802 - Qt の命名規約
        return ["text/uri-list"]

    def mimeData(self, items: Sequence[QListWidgetItem]) -> QMimeData:  # noqa: N802 - Qt の命名規約
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(item.data(_PATH))) for item in items])
        return mime

    def startDrag(self, supportedActions: Qt.DropAction) -> None:  # noqa: N802, N803 - Qt の命名規約
        # 写すことだけを許す 動かす（Move）で受け取られると、Qt は引いた元の行を消す
        del supportedActions
        super().startDrag(Qt.DropAction.CopyAction)


class ExplorerPanel(QWidget):
    """登録したフォルダの一覧・絞り込み・下見と試聴・タイムラインへ置く"""

    #: 登録を変えた 引数は新しい登録の一覧（``tuple[LibraryFolder, ...]``） 覚えるのは窓の仕事
    folders_changed = Signal(object)
    #: ファイルをタイムラインへ置くよう頼まれた 引数は場所の一覧
    place_requested = Signal(list)
    #: 知らせ（試聴できない など） 窓がステータスバーへ出す
    message = Signal(str)

    def __init__(
        self,
        folders: Sequence[LibraryFolder] = (),
        parent: QWidget | None = None,
        *,
        book: MeasureBook | None = None,
        play: Callable[[Path, float], None] | None = None,
        stop: Callable[[], None] | None = None,
    ) -> None:
        """``play`` と ``stop`` は試聴の口 窓が本物（:class:`~sashimono.engine.audio.library_sound.
        SoundPreview`）を渡す 渡さなければ試聴のボタンは知らせを出すだけ（試験で音を鳴らさない）
        """
        super().__init__(parent)
        self._folders: tuple[LibraryFolder, ...] = tuple(folders)
        self._book = book
        self._play = play
        self._stop = stop
        self._volume = 100
        #: 見て回った結果 フォルダの場所で引く 絞り込みの 1 文字ごとに見て回り直さない
        self._scans: dict[str, FolderScan] = {}
        self._playing: Path | None = None

        self._folder_box = QComboBox(self)
        narrow_combo(self._folder_box, 12)
        self._folder_box.setToolTip("見るフォルダ 登録したフォルダの分類と名前")
        self._folder_box.currentIndexChanged.connect(self._show_files)

        register = QPushButton("登録…", self)
        register.setToolTip("フォルダを分類（効果音・BGM・動画・画像など）を付けて登録する")
        register.clicked.connect(self._register)
        self._remove_button = QPushButton("外す", self)
        self._remove_button.setToolTip("選んだフォルダの登録を外す ファイルは消さない")
        self._remove_button.clicked.connect(self._remove)
        rescan = QPushButton("読み直す", self)
        rescan.setToolTip("フォルダの中を見直す 後から足したファイルを一覧へ出す")
        rescan.clicked.connect(self.rescan)

        self._filter = QLineEdit(self)
        self._filter.setPlaceholderText("名前で絞り込む（空白で区切ると全部を含む物）")

        self._filter.textChanged.connect(self._show_files)

        self._status = QLabel(self)
        self._status.setWordWrap(True)

        self._list = _FileList(self)
        self._list.setDragEnabled(True)
        self._list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self._list.setDefaultDropAction(Qt.DropAction.CopyAction)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection)
        self._list.itemDoubleClicked.connect(lambda _item: self._place_selected())

        self._preview = QLabel(self)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(0)
        self._preview.hide()

        self._listen = QPushButton("試聴", self)
        self._listen.setToolTip("選んだ音を鳴らす もう一度押すと止める")
        self._listen.clicked.connect(self._toggle_listen)
        self._place_button = QPushButton("タイムラインへ", self)
        self._place_button.setToolTip(
            "選んだファイルを読み込んで再生位置へ置く 取り消し 1 回で戻る"
        )
        self._place_button.clicked.connect(self._place_selected)

        # 狭いパネルでは折り返す（素材一覧と同じ 1 列だと左の列の最小の幅が広がる）
        top = FlowLayout()
        top.addWidget(register)
        top.addWidget(self._remove_button)
        top.addWidget(rescan)
        bottom = FlowLayout()
        bottom.addWidget(self._listen)
        bottom.addWidget(self._place_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._folder_box)
        layout.addLayout(top)
        layout.addWidget(self._filter)
        layout.addWidget(self._status)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._preview)
        layout.addLayout(bottom)

        self._fill_folders()
        self._on_selection()

    # --- 外から ---

    @property
    def folders(self) -> tuple[LibraryFolder, ...]:
        return self._folders

    def set_folders(self, folders: Sequence[LibraryFolder]) -> None:
        """登録を差し替える（設定画面で変えたとき） 選んでいたフォルダはそのまま選ぶ"""
        folders = tuple(folders)
        if folders == self._folders:
            return
        self._folders = folders
        self._fill_folders()

    def set_preview_volume(self, percent: int) -> None:
        self._volume = max(0, min(100, percent))

    def rescan(self) -> None:
        """見て回った結果を捨てて見直す"""
        self._scans.clear()
        self._show_files()

    def visible_files(self) -> list[Path]:
        """一覧に出ている行のファイル（試験と下見の確かめに使う）"""
        return [Path(str(self._list.item(row).data(_PATH))) for row in range(self._list.count())]

    def select_file(self, path: Path) -> bool:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if Path(str(item.data(_PATH))) == path:
                self._list.setCurrentItem(item)
                return True
        return False

    def selected_files(self) -> list[Path]:
        return [Path(str(item.data(_PATH))) for item in self._list.selectedItems()]

    def status_text(self) -> str:
        return self._status.text()

    def drag_data(self) -> QMimeData:
        """選んだ行をドラッグしたときの中身（試験用 本物のドラッグは OS の窓を待つ）"""
        return self._list.mimeData(self._list.selectedItems())

    # --- 登録 ---

    def _fill_folders(self) -> None:
        chosen = self._folder_box.currentData()
        self._folder_box.blockSignals(True)
        self._folder_box.clear()
        self._folder_box.addItem("すべてのフォルダ", ALL_FOLDERS)
        for folder in self._folders:
            self._folder_box.addItem(folder.label, folder.path)
            self._folder_box.setItemData(
                self._folder_box.count() - 1, folder.path, Qt.ItemDataRole.ToolTipRole
            )
        index = self._folder_box.findData(chosen) if chosen is not None else -1
        self._folder_box.setCurrentIndex(max(0, index))
        self._folder_box.blockSignals(False)
        # 外したフォルダの結果は捨てる 残すと、同じ場所を別の分類で登録し直したときに前の分類が出る
        known = {folder.path for folder in self._folders}
        self._scans = {key: scan for key, scan in self._scans.items() if key in known}
        self._show_files()

    def _register(self) -> None:
        path = ask_folder(self)
        if path is None:
            return
        category = ask_category(self)
        if category is None:
            return
        self.register(path, category)

    def register(self, path: str, category: str) -> None:
        """フォルダを分類を付けて登録する 同じフォルダなら分類を変える"""
        self._folders = add_folder(self._folders, path, category)
        self._scans.pop(path, None)
        self._fill_folders()
        index = self._folder_box.findData(path)
        if index >= 0:
            self._folder_box.setCurrentIndex(index)
        self.folders_changed.emit(self._folders)

    def _remove(self) -> None:
        path = self._folder_box.currentData()
        if not path:
            self.message.emit("外すフォルダを上の欄で選んでください")
            return
        self._folders = remove_folder(self._folders, str(path))
        self._fill_folders()
        self.folders_changed.emit(self._folders)

    # --- 一覧 ---

    def _chosen_folders(self) -> tuple[LibraryFolder, ...]:
        path = self._folder_box.currentData()
        if not path:
            return self._folders
        return tuple(folder for folder in self._folders if folder.path == path)

    def _scan(self, folder: LibraryFolder) -> FolderScan:
        known = self._scans.get(folder.path)
        if known is None or known.folder != folder:
            known = scan_folder(folder)
            self._scans[folder.path] = known
        return known

    def _show_files(self) -> None:
        folders = self._chosen_folders()
        self._remove_button.setEnabled(bool(self._folder_box.currentData()))
        selected = set(self.selected_files())
        self._list.clear()
        if not self._folders:
            self._status.setText(
                "フォルダが登録されていません 〔登録…〕で効果音や BGM のフォルダを登録すると、"
                "ここで探して置けます AI に使わせる分類は 表示 → 設定… で選びます"
            )
            self._on_selection()
            return
        notes: list[str] = []
        files: list[LibraryFile] = []
        for folder in folders:
            scan = self._scan(folder)
            if scan.missing:
                # 消えた・移った・外付けのドライブを外した 登録は残し、知らせるだけ
                notes.append(
                    f"{folder.label} が見つかりません（消えたか移った） 戻すか、〔外す〕で外せます"
                )
                continue
            if scan.truncated:
                notes.append(f"{folder.label} はファイルが多いので、頭の {len(scan.files)} 個だけ")
            if scan.error:
                notes.append(f"{folder.label}: {scan.error}")
            files.extend(scan.files)
        matched = match_files(files, self._filter.text())
        for file in matched[:DISPLAY_LIMIT]:
            item = QListWidgetItem(self._row_text(file))
            item.setData(_PATH, str(file.path))
            item.setToolTip(str(file.path))
            self._list.addItem(item)
            if file.path in selected:
                item.setSelected(True)
        count = f"{len(matched)} 個"
        if len(matched) > DISPLAY_LIMIT:
            count += f"（頭の {DISPLAY_LIMIT} 個を表示 名前で絞り込んでください）"
        self._status.setText("  ".join([count, *notes]))
        self._on_selection()

    def _row_text(self, file: LibraryFile) -> str:
        """行の文 分類と、測ってあれば長さ（AI が測った物をここでも見せる）"""
        parts = [file.relative, f"[{file.folder.category_name}]"]
        measure = self._book.get(file) if self._book is not None else None
        if measure is not None and measure.seconds > 0:
            parts.append(f"{measure.seconds:.1f} 秒")
        return "  ".join(parts)

    # --- 下見と試聴 ---

    def _on_selection(self) -> None:
        files = self.selected_files()
        self._place_button.setEnabled(bool(files))
        # 鳴らしている間は、音でない行を選んでも止められるように押せるままにする
        self._listen.setEnabled(
            self._playing is not None or (len(files) == 1 and _is_sound(files[0]))
        )
        self._show_preview(files[0] if len(files) == 1 else None)

    def _show_preview(self, path: Path | None) -> None:
        """動画と画像は見本の絵 音は絵を出さない（試聴で確かめる）"""
        if path is None or _is_sound(path):
            self._preview.clear()
            self._preview.hide()
            return
        from sashimono.engine.decode.image import read_image

        tile = read_image(path)
        if tile is None or tile.ndim != 3 or 0 in tile.shape:
            self._preview.setText("見本の絵を作れません")
        else:
            size = QSize(GRID_ICON_SIZE.width() * 2, GRID_ICON_SIZE.height() * 2)
            self._preview.setPixmap(thumbnail_icon(tile).pixmap(size))
        self._preview.show()

    def _toggle_listen(self) -> None:
        files = self.selected_files()
        was = self._playing
        if was is not None:
            self.stop_listening()
            # 鳴らしていた物と違う行を選んでいたら、止めた後にその行を鳴らす
            if len(files) != 1 or files[0] == was:
                return
        if len(files) != 1 or not _is_sound(files[0]):
            return
        if self._play is None:
            self.message.emit("この画面では試聴できません")
            return
        self._playing = files[0]
        self._play(files[0], self._volume / 100)
        self._listen.setText("止める")

    def stop_listening(self) -> None:
        if self._playing is not None and self._stop is not None:
            self._stop()
        self._playing = None
        self._listen.setText("試聴")
        self._on_selection()

    def _place_selected(self) -> None:
        files = self.selected_files()
        if files:
            self.place_requested.emit(files)


def _is_sound(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES
