"""プリセットと自作のエイリアスを整理して選ぶ窓（#276 #277）

左に分類とごみ箱、右に見本の絵つきの一覧 上のタブでプリセットとエイリアスを切り替える
名前の変更・分類の変更・複製・削除（ごみ箱へ）・書き出し・読み込み・置き場を開くを、
2 つで同じ操作にする ファイルの扱いはコアの :class:`~sashimono.core.io.library.Library`

開く所は 3 つ

* 設定パネルの〔プリセット…〕→〔管理…〕 選んで〔当てる〕と、選んでいるクリップへ当てる
* タイムラインの〔追加〕→〔エイリアス〕→〔管理…〕 選んで〔置く〕と、右クリックした所へ置く
* 〔オブジェクト〕→〔プリセットとエイリアス…〕 整理だけ（当てる・置く先が無い）

テンプレートの棚（AviUtl のエイリアス・YMM4 のテンプレート）は並べない 他人が作った
配布物で、名前を変えたり消したりすると配布元と食い違う 棚は〔互換〕→〔テンプレート…〕

見本の絵は開いた瞬間に描かない 覚えている物だけをすぐ出し、足りない物は見えている物
だけを別のスレッドで描いて、できた物から埋める（:mod:`sashimono.ui.library_thumbnails`）
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.io.aliases import Alias
from sashimono.core.io.library import (
    ALIAS,
    PRESET,
    Library,
    LibraryChangedError,
    LibraryConflictError,
    LibraryEntry,
    LibraryError,
    LibraryKind,
    TrashEntry,
)
from sashimono.core.io.presets import Preset
from sashimono.ui.flow_layout import FlowLayout
from sashimono.ui.library_thumbnails import LookThumbnails, Thumbnail
from sashimono.ui.library_view import (
    LibraryOptions,
    library_options,
    shared_thumbnails,
    thumbnail_pixmap,
)
from sashimono.ui.theme import Colors, themed_style

__all__ = ["CONFLICT_BOTH", "CONFLICT_OVERWRITE", "CONFLICT_SKIP", "LibraryDialog", "open_library"]

#: 読み込みで同じ名前があったときの答え
CONFLICT_OVERWRITE = "overwrite"
CONFLICT_BOTH = "both"
CONFLICT_SKIP = "skip"

#: 一覧で見せる見本の大きさ（論理の画素） 置いておく絵の半分 画面の倍率 2 でも粗くならない
_ICON = QSize(128, 72)
#: 一覧の 1 枠の大きさ 名前を 2 行まで出す
_GRID = QSize(150, 112)
#: 読み込みで選べるファイル 旧い拡張子のプリセットも読める
_IMPORT_FILTER = "プリセットとエイリアス (*.smep *.smea *.kmkp *.nvpreset);;すべてのファイル (*)"
_KIND_LABELS = {PRESET: "プリセット", ALIAS: "エイリアス"}
#: 並べ方
_SORT_NAME = "name"
_SORT_CATEGORY = "category"
_SORT_DATE = "date"

#: 分類の一覧の印
_ALL = "all"
_TRASH = "trash"
#: 分類の印の頭 後ろに分類の名前を付ける 分類の名前が ``all`` でも「すべて」と取り違えない
#: Qt に組（tuple）を持たせると並び（list）になって戻り、比べても一致しない
_CATEGORY = "category:"

_ENTRY = Qt.ItemDataRole.UserRole
_KEY = Qt.ItemDataRole.UserRole + 1

T = TypeVar("T")


class LibraryDialog(QDialog):
    """整理の窓 ``pick`` の種類だけ〔当てる〕〔置く〕で選べる（選んだ物は :attr:`chosen`）"""

    def __init__(
        self,
        library: Library,
        *,
        kind: LibraryKind = PRESET,
        pick: LibraryKind | None = None,
        thumbnails: LookThumbnails | None = None,
        options: LibraryOptions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("プリセットとエイリアス")
        self.resize(780, 540)
        self._library = library
        self._pick = pick
        self._options = options if options is not None else library_options()
        # 描き方は設定を当てたときに係へ渡してある（``set_library_options``）
        self._thumbnails = thumbnails if thumbnails is not None else shared_thumbnails()
        self._thumbnails.ready.connect(self._on_thumbnail)
        #: 選んだ物（〔当てる〕〔置く〕） 閉じた後で呼んだ側が読む
        self.chosen: LibraryEntry | None = None
        self._entries: tuple[LibraryEntry, ...] = ()
        self._trashed: tuple[TrashEntry, ...] = ()
        #: 見本の鍵ごとの一覧の項目 絵ができたら同じ鍵の項目へ貼る
        self._waiting: dict[str, list[QListWidgetItem]] = {}
        self._backdrop_icon = QIcon(thumbnail_pixmap(None, self._options.backdrop))

        self._tabs = QTabBar(self)
        for name in (PRESET, ALIAS):
            self._tabs.addTab(_KIND_LABELS[name])
        self._tabs.setCurrentIndex(0 if kind == PRESET else 1)
        self._tabs.currentChanged.connect(lambda _index: self.refresh())

        self._search = QLineEdit(self)
        self._search.setPlaceholderText("名前で探す")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(lambda _text: self._fill_items())

        self._sort = QComboBox(self)
        self._sort.addItem("名前の順", _SORT_NAME)
        self._sort.addItem("分類の順", _SORT_CATEGORY)
        self._sort.addItem("保存した日の新しい順", _SORT_DATE)
        self._sort.currentIndexChanged.connect(lambda _index: self._fill_items())

        self._folders = QListWidget(self)
        self._folders.setMinimumWidth(130)
        self._folders.currentItemChanged.connect(lambda *_: self._fill_items())

        self._list = QListWidget(self)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._update_buttons)
        self._list.itemDoubleClicked.connect(lambda _item: self._pick_selected())
        self._list.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible())

        splitter = QSplitter(self)
        splitter.addWidget(self._folders)
        splitter.addWidget(self._list)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([160, 600])

        self._note = QLabel(self)
        self._note.setWordWrap(True)
        themed_style(self._note, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._pick_button = QPushButton("当てる" if pick == PRESET else "置く", self)
        self._pick_button.setDefault(True)
        self._pick_button.clicked.connect(self._pick_selected)
        self._rename_button = self._button("名前を変える…", self.rename_selected)
        self._category_button = self._button("分類を変える…", self.recategorize_selected)
        self._duplicate_button = self._button("複製…", self.duplicate_selected)
        self._delete_button = self._button("削除", self.delete_selected)
        self._export_button = self._button("書き出す…", self.export_selected)
        self._import_button = self._button("読み込む…", self.import_files)
        self._folder_button = self._button("フォルダを開く", self.open_folder)
        self._restore_button = self._button("元に戻す", self.restore_selected)
        self._purge_button = self._button("ごみ箱から消す…", self.purge_selected)
        self._empty_button = self._button("ごみ箱を空にする…", self.empty_trash)
        close = self._button("閉じる", self.reject)

        # 狭い画面でボタンが 1 列に並ぶと窓の幅が画面を越えるので、折り返す
        buttons = FlowLayout()
        if pick is not None:
            buttons.addWidget(self._pick_button)
        for widget in (
            self._rename_button,
            self._category_button,
            self._duplicate_button,
            self._delete_button,
            self._restore_button,
            self._purge_button,
            self._empty_button,
            self._export_button,
            self._import_button,
            self._folder_button,
            close,
        ):
            buttons.addWidget(widget)
        if pick is None:
            self._pick_button.hide()

        top = QHBoxLayout()
        top.addWidget(self._tabs)
        top.addStretch(1)
        top.addWidget(self._search, 1)
        top.addWidget(self._sort)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._note)
        layout.addLayout(buttons)

        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(0)
        self._visible_timer.timeout.connect(self._request_visible)
        self.refresh()

    # --- 見せる ---

    @property
    def kind(self) -> LibraryKind:
        return PRESET if self._tabs.currentIndex() == 0 else ALIAS

    def set_kind(self, kind: LibraryKind) -> None:
        self._tabs.setCurrentIndex(0 if kind == PRESET else 1)

    def refresh(self) -> None:
        """置き場を読み直す ほかの窓で変えた物もここで拾う"""
        kind = self.kind
        self._entries = self._library.entries(kind)
        self._trashed = self._library.trashed(kind)
        current = self._folder()
        self._folders.blockSignals(True)
        self._folders.clear()
        self._add_folder("すべて", _ALL)
        for category in sorted({entry.category for entry in self._entries}):
            self._add_folder(category, _CATEGORY + category)
        self._add_folder(f"ごみ箱（{len(self._trashed)}）", _TRASH)
        chosen = next(
            (
                row
                for row in range(self._folders.count())
                if self._folders.item(row).data(_ENTRY) == current
            ),
            0,
        )
        self._folders.setCurrentRow(chosen)
        self._folders.blockSignals(False)
        self._fill_items()

    def entries_shown(self) -> list[LibraryEntry | TrashEntry]:
        """一覧に出ている物（並びのとおり） 試験と、選び直しに使う"""
        return [self._list.item(row).data(_ENTRY) for row in range(self._list.count())]

    def select(self, *names: str) -> None:
        """名前で選ぶ 試験と、操作の後に同じ物を選び直すのに使う"""
        self._list.clearSelection()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_ENTRY).name in names:
                item.setSelected(True)
                self._list.setCurrentItem(item)

    def show_folder(self, folder: str) -> None:
        for row in range(self._folders.count()):
            if self._folders.item(row).data(_ENTRY) == folder:
                self._folders.setCurrentRow(row)
                return

    def _add_folder(self, label: str, data: str) -> None:
        item = QListWidgetItem(label)
        item.setData(_ENTRY, data)
        self._folders.addItem(item)

    def _folder(self) -> str:
        item = self._folders.currentItem()
        return str(item.data(_ENTRY)) if item is not None else _ALL

    def _in_trash(self) -> bool:
        return self._folder() == _TRASH

    def _fill_items(self) -> None:
        self._list.clear()
        self._waiting.clear()
        words = self._search.text().strip().casefold()
        trash = self._in_trash()
        thumbnails = self._thumbnails.enabled and not trash
        if thumbnails:
            self._list.setViewMode(QListView.ViewMode.IconMode)
            self._list.setIconSize(_ICON)
            self._list.setGridSize(_GRID)
            self._list.setWordWrap(True)
            self._list.setResizeMode(QListView.ResizeMode.Adjust)
            self._list.setMovement(QListView.Movement.Static)
            self._list.setUniformItemSizes(True)
        else:
            self._list.setViewMode(QListView.ViewMode.ListMode)
            self._list.setIconSize(QSize())
            self._list.setGridSize(QSize())
        if trash:
            for trashed in self._trashed:
                if words and words not in trashed.name.casefold():
                    continue
                item = QListWidgetItem(f"{trashed.name}（{trashed.category}）")
                item.setData(_ENTRY, trashed)
                item.setToolTip(f"{_stamp(trashed.deleted_at)} に消した 〔元に戻す〕で戻せる")
                self._list.addItem(item)
        else:
            folder = self._folder()
            category = folder.removeprefix(_CATEGORY) if folder.startswith(_CATEGORY) else ""
            for entry in self._sorted(self._entries):
                if category and entry.category != category:
                    continue
                if words and words not in entry.name.casefold():
                    continue
                item = QListWidgetItem(entry.name)
                item.setData(_ENTRY, entry)
                item.setToolTip(_describe(entry))
                if thumbnails:
                    item.setIcon(self._backdrop_icon)
                self._list.addItem(item)
        self._update_note()
        self._update_buttons()
        self._schedule_visible()

    def _sorted(self, entries: tuple[LibraryEntry, ...]) -> list[LibraryEntry]:
        order = self._sort.currentData()
        if order == _SORT_CATEGORY:
            return sorted(entries, key=lambda e: (e.category, e.name))
        if order == _SORT_DATE:
            return sorted(entries, key=lambda e: -e.modified)
        return list(entries)

    def _update_note(self) -> None:
        parts: list[str] = []
        if self._in_trash():
            parts.append("ごみ箱の物は〔元に戻す〕で元の分類へ戻せる 空にするまでは消えない")
        elif not self._entries:
            if self.kind == PRESET:
                parts.append("まだ無い 設定パネルの〔プリセット…〕→〔この見た目を保存…〕で作れる")
            else:
                parts.append("まだ無い タイムラインの右クリック〔エイリアスとして保存…〕で作れる")
        elif self._thumbnails.enabled and self._thumbnails.simple:
            # 出せていない物を出せているように見せない（テンプレートの棚の下絵と同じ考え）
            parts.append("見本は文字と図形だけ（縁取り・グローなどのエフェクトは出ていない）")
        self._note.setText(" ".join(parts))

    # --- 見本の絵 ---

    def _schedule_visible(self) -> None:
        if self._thumbnails.enabled:
            self._visible_timer.start()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の名前
        super().resizeEvent(event)
        # 広げると見える項目が増える その見本も頼む
        self._schedule_visible()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt の名前
        super().showEvent(event)
        self._schedule_visible()

    def _request_visible(self) -> None:
        """見えている項目の見本を出す 覚えていれば貼り、無ければ描くよう頼む"""
        if not self._thumbnails.enabled or self._in_trash():
            return
        viewport = self._list.viewport().rect()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_KEY) is not None:
                continue
            if not self._list.visualItemRect(item).intersects(viewport):
                continue
            entry: LibraryEntry = item.data(_ENTRY)
            cached = self._thumbnails.cached(entry.item)
            if cached is not None:
                item.setData(_KEY, "")
                self._show_thumbnail(item, cached)
                continue
            key = self._thumbnails.request(entry.item)
            if key is None:
                continue
            item.setData(_KEY, key)
            self._waiting.setdefault(key, []).append(item)

    def _on_thumbnail(self, key: str) -> None:
        waiting = self._waiting.pop(key, [])
        # 鍵で引く GPU が使えないと分かると、品物から引き直した鍵は簡易の描き方の物になる
        found = self._thumbnails.thumbnail(key)
        if found is None:
            return
        for item in waiting:
            self._show_thumbnail(item, found)
        if found.simple:
            # GPU が使えないと分かったら、エフェクトが出ていないことをその場で書く
            self._update_note()

    def _show_thumbnail(self, item: QListWidgetItem, thumbnail: Thumbnail) -> None:
        entry: LibraryEntry = item.data(_ENTRY)
        tip = _describe(entry)
        if thumbnail.error:
            tip = f"{tip}\n{thumbnail.error}"
        elif thumbnail.empty:
            tip = f"{tip}\n見本のコマに見える物がない（透明・画面の外・時間で出てくる物など）"
        item.setToolTip(tip)
        if thumbnail.image is not None:
            item.setIcon(QIcon(thumbnail_pixmap(thumbnail.image, self._options.backdrop)))

    # --- ボタン ---

    def _button(self, text: str, slot: Callable[[], object]) -> QPushButton:
        button = QPushButton(text, self)
        button.setAutoDefault(False)
        button.clicked.connect(lambda _checked=False: slot())
        return button

    def _selected(self) -> list[LibraryEntry]:
        return [
            item.data(_ENTRY)
            for item in self._list.selectedItems()
            if isinstance(item.data(_ENTRY), LibraryEntry)
        ]

    def _selected_trash(self) -> list[TrashEntry]:
        return [
            item.data(_ENTRY)
            for item in self._list.selectedItems()
            if isinstance(item.data(_ENTRY), TrashEntry)
        ]

    def _update_buttons(self) -> None:
        trash = self._in_trash()
        chosen = self._selected()
        for button in (
            self._rename_button,
            self._category_button,
            self._duplicate_button,
            self._delete_button,
            self._export_button,
            self._import_button,
            self._folder_button,
            self._pick_button,
        ):
            button.setVisible(not trash and (button is not self._pick_button or bool(self._pick)))
        for button in (self._restore_button, self._purge_button, self._empty_button):
            button.setVisible(trash)
        self._pick_button.setEnabled(len(chosen) == 1 and self._pick == self.kind)
        self._rename_button.setEnabled(len(chosen) == 1)
        self._duplicate_button.setEnabled(len(chosen) == 1)
        self._category_button.setEnabled(bool(chosen))
        self._delete_button.setEnabled(bool(chosen))
        self._export_button.setEnabled(bool(chosen))
        self._restore_button.setEnabled(bool(self._selected_trash()))
        self._purge_button.setEnabled(bool(self._selected_trash()))
        self._empty_button.setEnabled(bool(self._trashed))

    def _pick_selected(self) -> None:
        chosen = self._selected()
        if self._pick != self.kind or len(chosen) != 1:
            return
        self.chosen = chosen[0]
        self.accept()

    # --- 操作 ---

    def rename_selected(self) -> None:
        chosen = self._selected()
        if len(chosen) != 1:
            return
        entry = chosen[0]
        name = self.ask_text("名前を変える", "新しい名前", entry.name)
        if name is None or name == entry.name:
            return
        done = self._run(functools.partial(self._library.rename, entry, name))
        if done is not None:
            self.refresh()
            self.select(done.name)

    def recategorize_selected(self) -> None:
        chosen = self._selected()
        if not chosen:
            return
        categories = sorted({entry.category for entry in self._entries})
        category = self.ask_category(categories, chosen[0].category)
        if category is None:
            return
        moved: list[str] = []
        for entry in chosen:
            if entry.category == category:
                continue
            done = self._run(functools.partial(self._library.recategorize, entry, category))
            if done is not None:
                moved.append(done.name)
        self.refresh()
        self.show_folder(_CATEGORY + category)
        self.select(*moved)

    def duplicate_selected(self) -> None:
        chosen = self._selected()
        if len(chosen) != 1:
            return
        entry = chosen[0]
        name = self.ask_text("複製", "写しの名前", f"{entry.name} のコピー")
        if name is None:
            return
        done = self._run(functools.partial(self._library.duplicate, entry, name))
        if done is not None:
            self.refresh()
            self.select(done.name)

    def delete_selected(self) -> None:
        chosen = self._selected()
        if not chosen:
            return
        if self._options.confirm_delete:
            names = "、".join(f"「{entry.name}」" for entry in chosen[:5])
            more = f" ほか {len(chosen) - 5} 件" if len(chosen) > 5 else ""
            if not self.confirm(
                "ごみ箱へ移す",
                f"{names}{more}をごみ箱へ移す？ ごみ箱から〔元に戻す〕で戻せる",
            ):
                return
        for entry in chosen:
            self._run(functools.partial(self._library.delete, entry), overwritable=False)
        self.refresh()

    def restore_selected(self) -> None:
        for trashed in self._selected_trash():
            self._run(functools.partial(self._library.restore, trashed))
        self.refresh()

    def purge_selected(self) -> None:
        chosen = self._selected_trash()
        if not chosen:
            return
        # 戻せなくなる操作なので、確かめの設定に関係なく必ず尋ねる
        if not self.confirm(
            "ごみ箱から消す", f"{len(chosen)} 件をごみ箱から消す？ もう戻せなくなる"
        ):
            return
        self._run(
            functools.partial(self._library.empty_trash, self.kind, chosen), overwritable=False
        )
        self.refresh()

    def empty_trash(self) -> None:
        if not self._trashed:
            return
        label = _KIND_LABELS[self.kind]
        if not self.confirm(
            "ごみ箱を空にする",
            f"{label}のごみ箱の {len(self._trashed)} 件を消す？ もう戻せなくなる",
        ):
            return
        self._run(functools.partial(self._library.empty_trash, self.kind), overwritable=False)
        self.refresh()

    def export_selected(self) -> None:
        chosen = self._selected()
        if not chosen:
            return
        folder = self.choose_folder()
        if folder is None:
            return
        written = 0
        for entry in chosen:
            done = self._run(functools.partial(self._library.export, entry, folder))
            if done is not None:
                written += 1
        self.tell("書き出す", f"{written} 件を {folder} へ書き出した")

    def import_files(self) -> None:
        paths = self.choose_files()
        if not paths:
            return
        added = 0
        problems: list[str] = []
        last_kind: LibraryKind | None = None
        for path in paths:
            try:
                kind, item = self._library.read_file(path)
            except LibraryError as exc:
                problems.append(str(exc))
                continue
            overwrite = False
            if self._library.occupied(kind, item):
                answer = self.ask_conflict(kind, item)
                if answer == CONFLICT_SKIP:
                    continue
                if answer == CONFLICT_BOTH:
                    item = _renamed(item, self._library.free_name(kind, item))
                else:
                    overwrite = True
            try:
                self._library.add(kind, item, overwrite=overwrite)
            except (LibraryError, OSError) as exc:
                problems.append(f"{path.name}: {exc}")
                continue
            added += 1
            last_kind = kind
        if last_kind is not None and last_kind != self.kind:
            # 読み込んだ物の方のタブへ移る 読み込んだのに一覧に出ないと、失敗に見える
            self.set_kind(last_kind)
        else:
            self.refresh()
        message = f"{added} 件を読み込んだ"
        if problems:
            message = f"{message}\n読めなかった物:\n" + "\n".join(problems)
        self.tell("読み込む", message)

    def open_folder(self) -> None:
        root = self._library.root(self.kind)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.tell("フォルダを開く", f"置き場を作れなかった: {exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _run(self, action: Callable[..., T], *, overwritable: bool = True) -> T | None:
        """1 つの操作を行う 同じ名前があれば上書きを尋ね、ほかの窓が変えていたら読み直す

        ``overwritable`` なら ``action`` は ``overwrite`` を受け取る（名前の重なる操作）
        """
        try:
            if not overwritable:
                return action()
            try:
                return action(overwrite=False)
            except LibraryConflictError as conflict:
                if not self.confirm(
                    "上書きの確かめ", f"{conflict} 上書きする？ 前の物はごみ箱へ移る"
                ):
                    return None
                return action(overwrite=True)
        except LibraryChangedError as exc:
            self.tell("一覧が変わっていた", f"{exc}\n一覧を読み直す")
            self.refresh()
        except (LibraryError, OSError) as exc:
            self.tell("できなかった", str(exc))
        return None

    # --- 尋ねる（試験で差し替える） ---

    def ask_text(self, title: str, label: str, text: str) -> str | None:
        answer, accepted = QInputDialog.getText(self, title, label, text=text)
        return answer.strip() if accepted and answer.strip() else None

    def ask_category(self, categories: list[str], current: str) -> str | None:
        """分類を選ぶ 打てば新しい分類になる"""
        index = categories.index(current) if current in categories else 0
        answer, accepted = QInputDialog.getItem(
            self,
            "分類を変える",
            "分類（新しい名前を打てば、その分類を作る）",
            categories or [current],
            index,
            True,
        )
        return answer.strip() if accepted and answer.strip() else None

    def ask_conflict(self, kind: LibraryKind, item: Preset | Alias) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("読み込む")
        where = f"「{item.category}」に" if kind == PRESET else ""
        box.setText(f"{where}「{item.name}」がもうある どうする？")
        overwrite = box.addButton("上書き（前の物はごみ箱へ）", QMessageBox.ButtonRole.AcceptRole)
        both = box.addButton("両方残す（番号を付ける）", QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton("飛ばす", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(both)
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()
        if clicked is overwrite:
            return CONFLICT_OVERWRITE
        if clicked is skip:
            return CONFLICT_SKIP
        return CONFLICT_BOTH

    def confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def tell(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def choose_folder(self) -> Path | None:
        folder = QFileDialog.getExistingDirectory(self, "書き出す先のフォルダ")
        return Path(folder) if folder else None

    def choose_files(self) -> list[Path]:
        paths, _ = QFileDialog.getOpenFileNames(self, "読み込む", "", _IMPORT_FILTER)
        return [Path(path) for path in paths]

    def done(self, result: int) -> None:
        # 走り係は止めない（次に開いたときにコンテキストと描画係を作り直さない） 頼んだ物は
        # 描き終えて置き場へ残り、次に開いたときにすぐ出る 知らせだけ外す
        self._thumbnails.ready.disconnect(self._on_thumbnail)
        super().done(result)


def open_library(
    parent: QWidget | None,
    library: Library,
    *,
    kind: LibraryKind,
    pick: LibraryKind | None = None,
) -> LibraryEntry | None:
    """整理の窓を開く 〔当てる〕〔置く〕で選ばれた物を返す（閉じただけなら ``None``）"""
    dialog = LibraryDialog(library, kind=kind, pick=pick, parent=parent)
    answer = dialog.exec()
    chosen = dialog.chosen
    # 開くたびに作る窓 閉じたら捨てる
    dialog.deleteLater()
    return chosen if answer == QDialog.DialogCode.Accepted else None


def _describe(entry: LibraryEntry) -> str:
    lines = [f"分類: {entry.category}", f"保存した日: {_stamp(entry.modified)}"]
    if isinstance(entry.item, Preset) and entry.item.span is None:
        lines.append("前の版で保存したプリセット（足したエフェクトだけを足す）")
    if entry.legacy:
        lines.append("改名前の形式 名前や分類を変えると新しい形式で書き直す")
    lines.append(str(entry.path))
    return "\n".join(lines)


def _stamp(seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds))


def _renamed(item: Preset | Alias, name: str) -> Preset | Alias:
    if isinstance(item, Preset):
        return replace(item, name=name)
    return replace(item, name=name)
