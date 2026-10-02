"""メディアプール 読み込んだ素材の一覧"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.model import MediaId, MediaItem, Project
from sashimono.core.timebase import FrameRate, format_timecode, seconds_to_frame
from sashimono.ui.media_icons import (
    GRID_ICON_SIZE,
    LIST_ICON_SIZE,
    audio_icon,
    pending_icon,
    thumbnail_icon,
)
from sashimono.ui.theme import theme_signals

__all__ = [
    "MEDIA_MIME",
    "VIEW_ICONS",
    "VIEW_LIST",
    "VIEW_MODES",
    "MediaPoolWidget",
    "ThumbnailSource",
    "media_ids_in",
]

#: 行の説明（進み具合を添える前）を持たせる所 文言から添えた分を切り取る形にすると、
#: 素材の名前に同じ区切りが入っていたときに名前まで削る
_BASE_TEXT = Qt.ItemDataRole.UserRole + 1
#: アイコン表示で絵の下に出す名前 長さや大きさまで並べると、枠の幅で折り返して読めない
_NAME_TEXT = Qt.ItemDataRole.UserRole + 2

#: 一覧からタイムラインへ素材を引いていくときの中身の形式 素材の ID を行ごとに並べる
#: ファイルの URL にしないのは、一覧の上へ落としたときに同じ素材をもう一度
#: 読み込んでしまうのと、タイムラインが「読み込んで置く」と取り違えるため
MEDIA_MIME = "application/x-sashimono-media-ids"

#: 表示の切り替え 一覧（名前・長さ・大きさを 1 行ずつ）とアイコン（絵を並べる）
VIEW_LIST = "list"
VIEW_ICONS = "icons"
VIEW_MODES = (VIEW_LIST, VIEW_ICONS)

#: 素材の頭の方の 1 コマを返す関数 まだ作っていなければ ``None``
#: 描き直しのたびに呼ぶので、待たせずに返すこと（作るのは裏のスレッドの仕事）
ThumbnailSource = Callable[[MediaItem], np.ndarray | None]


def media_ids_in(mime: QMimeData) -> list[MediaId]:
    """一覧から引いてきた素材の ID 一覧から来たのでなければ空"""
    if not mime.hasFormat(MEDIA_MIME):
        return []
    try:
        raw = bytes(mime.data(MEDIA_MIME).data()).decode("utf-8")
    except UnicodeDecodeError:
        # 同じ形式の名前で、ほかのアプリが壊れた中身を引いてくることがある ここで
        # 投げると、ドラッグの知らせの途中で例外になり、落とせない印も出せない
        return []
    return [MediaId(line) for line in raw.splitlines() if line]


class _MediaList(QListWidget):
    """素材の行を、ID を載せてタイムラインへ引いていけるようにした一覧"""

    def mimeTypes(self) -> list[str]:  # noqa: N802 - Qt の命名規約
        return [MEDIA_MIME]

    def mimeData(self, items: Sequence[QListWidgetItem]) -> QMimeData:  # noqa: N802 - Qt の命名規約
        mime = QMimeData()
        ids = "\n".join(str(item.data(Qt.ItemDataRole.UserRole)) for item in items)
        mime.setData(MEDIA_MIME, ids.encode("utf-8"))
        return mime

    def startDrag(self, supportedActions: Qt.DropAction) -> None:  # noqa: N802, N803 - Qt の命名規約
        # 写すことだけを許す 動かす（Move）で受け取られると、Qt は引いた元の行を
        # 一覧から消す 素材はプロジェクトに残ったまま、一覧からだけ見えなくなる
        del supportedActions
        super().startDrag(Qt.DropAction.CopyAction)


#: 読み込みダイアログのフィルタ
MEDIA_FILTER = (
    "メディア (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wav *.mp3 *.aac *.flac *.m4a "
    "*.png *.jpg *.jpeg *.bmp *.webp);;すべてのファイル (*)"
)


class MediaPoolWidget(QWidget):
    """素材の一覧と、読み込み・タイムラインへの配置"""

    #: 読み込みが要求された 引数はパスの一覧
    import_requested = Signal(list)
    #: 素材をタイムラインへ置くよう要求された 引数は素材 ID
    insert_requested = Signal(str)
    #: 字幕を起こすよう要求された 引数は素材 ID
    transcribe_requested = Signal(str)
    #: メディアプールから外すよう要求された 引数は素材 ID
    remove_requested = Signal(str)
    #: 本人が表示（一覧・アイコン）を切り替えた 引数は :data:`VIEW_MODES` のどれか
    #: 覚えておくのは窓の仕事（好みの設定に書く）
    view_mode_changed = Signal(str)

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = project
        #: 行に添える進み具合と失敗の理由 一覧を作り直しても残すために持つ
        self._notes: dict[MediaId, tuple[str, str]] = {}
        self._view_mode = VIEW_LIST
        self._thumbnails: ThumbnailSource | None = None
        #: できたサムネイルの絵 一覧を作り直すたびに縮め直さないよう、素材ごとに持つ
        #: 作ったときのファイルも覚える 同じ ID のまま別のファイルへ差し替わったら作り直す
        self._icons: dict[MediaId, tuple[Path, QIcon]] = {}
        self._audio_icon = audio_icon()
        self._pending_icon = pending_icon()
        theme_signals().changed.connect(self._redraw_icons)

        self._list = _MediaList(self)
        # 引いていけるのは外（タイムライン）だけ 一覧の中で並べ替えられると、
        # 素材の並び（プロジェクトの中身）と画面の並びが食い違う
        self._list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self._list.setDefaultDropAction(Qt.DropAction.CopyAction)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_menu)

        import_button = QPushButton("読み込み…", self)
        import_button.clicked.connect(self._choose_files)
        self._insert_button = QPushButton("タイムラインへ", self)
        self._insert_button.clicked.connect(self._insert_selected)
        self._insert_button.setEnabled(False)
        self._list.itemSelectionChanged.connect(
            lambda: self._insert_button.setEnabled(bool(self._list.selectedItems()))
        )

        self._view_buttons = QButtonGroup(self)
        self._view_buttons.setExclusive(True)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(import_button)
        buttons.addWidget(self._insert_button)
        buttons.addStretch(1)
        for mode, text, tip in (
            (VIEW_LIST, "一覧", "名前・長さ・大きさを 1 行ずつ並べる"),
            (VIEW_ICONS, "アイコン", "サムネイルを大きく並べる"),
        ):
            button = QToolButton(self)
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked(mode == self._view_mode)
            button.setProperty("view_mode", mode)
            # 押したときだけ知らせる 窓が覚えた表示を当てたとき（set_view_mode）まで
            # 知らせると、起動のたびに好みの設定を書き直す
            button.clicked.connect(lambda _checked=False, chosen=mode: self._choose_view(chosen))
            self._view_buttons.addButton(button)
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(buttons)
        layout.addWidget(self._list)

        self.setAcceptDrops(True)
        self._apply_view_mode()
        self.set_project(project)

    # --- 表示の切り替えとサムネイル ---

    @property
    def view_mode(self) -> str:
        return self._view_mode

    def set_view_mode(self, mode: str) -> None:
        """表示を切り替える 知らない値は一覧にする（設定ファイルを手で書いた人がいる）"""
        mode = mode if mode in VIEW_MODES else VIEW_LIST
        for button in self._view_buttons.buttons():
            button.setChecked(button.property("view_mode") == mode)
        if mode == self._view_mode:
            return
        self._view_mode = mode
        self._apply_view_mode()
        for item, media in self._rows():
            self._show_note(item, media)

    def set_thumbnail_source(self, source: ThumbnailSource | None) -> None:
        """行の頭に出す絵の出どころ 窓が解析係（MediaAnalyzer）を渡す

        一覧は解析係を知らない 知っていると、試験で一覧だけを作るのにも解析の
        スレッドを立てることになる
        """
        self._thumbnails = source
        self.refresh_thumbnails()

    def refresh_thumbnails(self) -> None:
        """できたサムネイルを行へ載せる 解析が 1 本終わるたびに窓が呼ぶ

        まだ絵の無い行だけを見る 載せ終えた行まで毎回縮め直すと、
        素材を 100 本並べたときに 250ms ごとに 100 枚を縮めることになる
        """
        for item, media in self._rows():
            if media.id not in self._icons:
                item.setIcon(self._icon_for(media))

    def _rows(self) -> Iterator[tuple[QListWidgetItem, MediaItem]]:
        """一覧の行と、その行の素材 消えた素材の行は飛ばす

        素材は ID の表から引く 行ごとに :meth:`Project.find_media`（先頭から順に探す）で
        引くと、素材が N 本で N×N 回になり、解析の知らせのたびに画面が引っ掛かる
        """
        by_id = {media.id: media for media in self._project.media}
        for row in range(self._list.count()):
            item = self._list.item(row)
            media = by_id.get(MediaId(str(item.data(Qt.ItemDataRole.UserRole))))
            if media is not None:
                yield item, media

    def _choose_view(self, mode: str) -> None:
        if mode == self._view_mode:
            return
        self.set_view_mode(mode)
        self.view_mode_changed.emit(mode)

    def _apply_view_mode(self) -> None:
        icons = self._view_mode == VIEW_ICONS
        self._list.setViewMode(
            QListView.ViewMode.IconMode if icons else QListView.ViewMode.ListMode
        )
        self._list.setIconSize(GRID_ICON_SIZE if icons else LIST_ICON_SIZE)
        # 切り替えると Qt がアイコン表示の既定（自由に動かせる・落とし込みを受ける）へ
        # 戻すので、そのたびに当て直す 動かせるままだと、引いたつもりが一覧の中で
        # 絵の位置が変わるだけになる
        self._list.setMovement(QListView.Movement.Static)
        self._list.setResizeMode(QListView.ResizeMode.Adjust)
        self._list.setWordWrap(icons)
        self._list.setSpacing(6 if icons else 1)
        self._list.setGridSize(
            QSize(GRID_ICON_SIZE.width() + 24, GRID_ICON_SIZE.height() + 44) if icons else QSize()
        )
        self._list.setDragEnabled(True)
        self._list.setDragDropMode(QListWidget.DragDropMode.DragOnly)

    def _redraw_icons(self) -> None:
        """テーマが変わった 印とサムネイルの周りの地を今の色で描き直す

        絵はできた時点の色で焼き込んであり、描き直すだけでは前のテーマのまま残る
        """
        self._audio_icon = audio_icon()
        self._pending_icon = pending_icon()
        self._icons = {}
        for item, media in self._rows():
            item.setIcon(self._icon_for(media))

    def _icon_for(self, media: MediaItem) -> QIcon:
        cached = self._icons.get(media.id)
        if cached is not None and cached[0] == media.path:
            return cached[1]
        if not (media.has_video or media.is_still):
            return self._audio_icon
        tile = self._thumbnails(media) if self._thumbnails is not None else None
        if tile is None or tile.ndim not in (2, 3) or 0 in tile.shape:
            return self._pending_icon
        icon = thumbnail_icon(tile)
        self._icons[media.id] = (media.path, icon)
        return icon

    def set_project(self, project: Project) -> None:
        """一覧を作り直す

        選択は素材 ID で復元する 行番号で覚えると、素材を消したときに
        別のものが選ばれる
        """
        selected = self.selected_media_id()
        self._project = project
        # 外した素材と、中身のファイルが差し替わった素材の絵は捨てる 外した素材の分を
        # 残すと抱え続け、差し替わった分を残すと前のファイルの絵が出続ける
        present = {media.id: media.path for media in project.media}
        self._icons = {
            key: cached for key, cached in self._icons.items() if present.get(key) == cached[0]
        }

        self._list.clear()
        for media in project.media:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(media.id))
            item.setData(_BASE_TEXT, _describe(media, project.rate))
            item.setData(_NAME_TEXT, media.name)
            item.setIcon(self._icon_for(media))
            self._show_note(item, media)
            self._list.addItem(item)
            if media.id == selected:
                self._list.setCurrentItem(item)

    def set_progress(self, notes: Mapping[MediaId, tuple[str, str]]) -> None:
        """行に進み具合を添える ``notes`` は素材 → （添える文言、失敗の理由）

        変わった行だけ書き換える 一覧を作り直すと、スクロールの位置と
        選んでいる行が 250ms ごとに揺れる
        """
        if dict(notes) == self._notes:
            return
        self._notes = dict(notes)
        for item, media in self._rows():
            self._show_note(item, media)

    def row_text(self, media_id: MediaId) -> str | None:
        """その素材の行に今出ている文言 無ければ ``None``"""
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == str(media_id):
                return item.text()
        return None

    def _show_note(self, item: QListWidgetItem, media: MediaItem) -> None:
        note, reason = self._notes.get(media.id, ("", ""))
        if self._view_mode == VIEW_ICONS:
            # 枠の幅が狭いので、添える分は名前の下の行へ回す 名前の後ろへ続けると、
            # 折り返しで名前の方が切れる
            base = str(item.data(_NAME_TEXT))
            text = f"{base}\n[{note}]" if note else base
            # 一覧表示では行に出ている長さと大きさを、絵にかざしたときに読めるようにする
            tooltip = f"{item.data(_BASE_TEXT)}\n{media.path}"
        else:
            base = str(item.data(_BASE_TEXT))
            text = f"{base}   [{note}]" if note else base
            tooltip = str(media.path)
        if reason:
            tooltip = f"{tooltip}\n{reason}"
        if item.text() != text:
            item.setText(text)
        if item.toolTip() != tooltip:
            item.setToolTip(tooltip)

    def selected_media_id(self) -> MediaId | None:
        items = self._list.selectedItems()
        if not items:
            return None
        return MediaId(str(items[0].data(Qt.ItemDataRole.UserRole)))

    # --- 入力 ---

    def dragEnterEvent(self, event: object) -> None:  # noqa: N802 - Qt の命名規約
        mime = getattr(event, "mimeData", None)
        if mime is not None and mime().hasUrls():
            event.acceptProposedAction()  # type: ignore[attr-defined]

    def dropEvent(self, event: object) -> None:  # noqa: N802 - Qt の命名規約
        mime = getattr(event, "mimeData", None)
        if mime is None:
            return
        paths = [Path(url.toLocalFile()) for url in mime().urls() if url.isLocalFile()]
        if paths:
            self.import_requested.emit(paths)
            event.acceptProposedAction()  # type: ignore[attr-defined]

    def _choose_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "素材を読み込む", "", MEDIA_FILTER)
        if names:
            self.import_requested.emit([Path(name) for name in names])

    def _insert_selected(self) -> None:
        media_id = self.selected_media_id()
        if media_id is not None:
            self.insert_requested.emit(str(media_id))

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self.insert_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def _show_menu(self, position: QPoint) -> None:
        item = self._list.itemAt(position)
        if item is None:
            return
        # 右クリックした行を選び直す 選んでいた別の行が対象になると、
        # 「消したつもりのない素材が消えた」になる
        self._list.setCurrentItem(item)
        menu = self.build_menu(MediaId(str(item.data(Qt.ItemDataRole.UserRole))))
        menu.exec(self._list.viewport().mapToGlobal(position))

    def build_menu(self, media_id: MediaId) -> QMenu:
        """素材 1 つに対する右クリックメニュー 表示と中身を分けてあるのはテストのため"""
        menu = QMenu(self)
        # triggered は押されたかどうか（bool）を渡してくる PySide6 は受け取れない
        # 引数を捨てて呼ぶが、タイムライン側と同じく明示的に受けて捨てる形にそろえる
        place = menu.addAction("タイムラインへ置く")
        place.triggered.connect(lambda _checked=False: self.insert_requested.emit(str(media_id)))
        transcribe = menu.addAction("字幕を起こす…")
        transcribe.triggered.connect(
            lambda _checked=False: self.transcribe_requested.emit(str(media_id))
        )
        media = self._project.find_media(media_id)
        transcribe.setEnabled(media is not None and media.has_audio)
        reveal = menu.addAction("ファイルの場所を開く")
        reveal.triggered.connect(lambda _checked=False: self._reveal(media_id))
        menu.addSeparator()
        remove = menu.addAction("プールから外す")
        remove.triggered.connect(lambda _checked=False: self.remove_requested.emit(str(media_id)))
        return menu

    def _reveal(self, media_id: MediaId) -> None:
        media = self._project.find_media(media_id)
        if media is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(media.path.parent)))


def _describe(media: MediaItem, rate: FrameRate) -> str:
    """一覧に出す 1 行 長さと中身の種類が一目で分かるようにする"""
    parts = [media.name]
    if media.is_still:
        parts.append("静止画")
    elif media.duration > 0:
        parts.append(format_timecode(seconds_to_frame(media.duration, rate), rate))

    kinds = []
    if media.has_video:
        stream = media.video_streams[0]
        width, height = stream.display_size
        kinds.append(f"{width}x{height}")
    if media.has_audio:
        count = len(media.audio_streams)
        kinds.append(f"音声{count}本" if count > 1 else "音声")
    if kinds:
        parts.append(" / ".join(kinds))

    return "   ".join(parts)
