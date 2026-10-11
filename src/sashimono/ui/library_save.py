"""プリセットとエイリアスを保存する窓（#276 の追記）

前は名前だけの入力窓で、同じ名前があると閉じた後に「上書きしますか」と尋ね、上書きすると
前の内容は戻せないまま消えた（プリセットは問いの後で書き換え、エイリアスも同じ） 最初の窓の
中で上書きを指定できるようにし、前の内容はごみ箱へ移す

* 保存済みの一覧（名前と見本）から選べる 選ぶと、その名前と分類が入り、上書きの印も入る
* 「同じ名前があれば上書きする」の印は最初から出す（既定は外れている 設定で入れられる）
  名前が既存と重なると、窓の中に警告を出し、印を入れるまで〔保存〕を押せない
* 上書きになるときは、前の見本といまの見本を並べる（一覧の見本の係と作り置きを使い回す）
* 分類は既存から選ぶか、打って新しい分類にする エイリアスの置き場は名前で決まるので、
  分類だけが違っても上書きになる 警告にそう書く
* Enter だけでは保存しない（〔保存〕を明かに押す） 上書きの印を入れた後も同じ
* 保存の時に、窓を出した後でほかの窓が保存し直していたら断り、読み直すよう知らせる
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.io.aliases import Alias, alias_refusal
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
from sashimono.core.model import Clip, Project, Track
from sashimono.ui.library_thumbnails import LookThumbnails, Thumbnail
from sashimono.ui.library_view import (
    VISIBLE_DELAY_MS,
    LibraryOptions,
    library_options,
    shared_thumbnails,
    thumbnail_pixmap,
)
from sashimono.ui.theme import Colors, themed_style

__all__ = [
    "UNDO_NOTICE_MS",
    "OverwriteUndo",
    "SaveChoice",
    "SaveDialog",
    "UndoNotice",
    "UndoOffers",
    "ask_save",
    "capture_look",
    "offer_undo",
    "save_look",
    "undo_offers",
]

Item = Preset | Alias
#: 名前と分類から、保存する物を作る手 いまのクリップを写す
Make = Callable[[str, str], Item]

_KIND_LABELS = {PRESET: "プリセット", ALIAS: "エイリアス"}
#: 一覧の見本の大きさ（論理の画素）
_ICON = QSize(96, 54)
#: 前後の見本の大きさ 一覧より大きく見せる 置いておく絵（256x144）を大きく越えるとぼやける
_PREVIEW = QSize(288, 162)


@dataclass(frozen=True, slots=True)
class SaveChoice:
    """窓で決めた保存の仕方"""

    name: str
    category: str
    #: 上書きされる物（窓が前の見本として見せた物） 上書きしないなら ``None``
    replacing: LibraryEntry | None


class SaveDialog(QDialog):
    """名前・分類・上書きの指定・前後の見本を 1 つの窓で決める"""

    def __init__(
        self,
        library: Library,
        kind: LibraryKind,
        *,
        make: Make,
        suggestion: str,
        category: str,
        thumbnails: LookThumbnails | None = None,
        options: LibraryOptions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        label = _KIND_LABELS[kind]
        self.setWindowTitle(f"{label}を保存")
        self._library = library
        self._kind = kind
        self._make = make
        self._options = options if options is not None else library_options()
        self._thumbnails = thumbnails if thumbnails is not None else shared_thumbnails()
        self._thumbnails.ready.connect(self._on_thumbnail)
        self._entries = library.entries(kind)
        self._occupant: LibraryEntry | None = None
        self._waiting: dict[str, list[tuple[QLabel | None, QListWidgetItem | None]]] = {}

        self._name = QLineEdit(suggestion, self)
        self._category = QComboBox(self)
        self._category.setEditable(True)
        categories = sorted({entry.category for entry in self._entries} | {category})
        self._category.addItems(categories)
        self._category.setCurrentText(category)
        self._overwrite = QCheckBox("同じ名前があれば上書きする（前の内容はごみ箱へ移る）", self)
        self._overwrite.setChecked(self._options.overwrite_without_asking)
        self._warning = QLabel(self)
        self._warning.setWordWrap(True)
        themed_style(self._warning, lambda: f"color: {Colors.WARNING.name()};")

        self._list = QListWidget(self)
        self._list.setViewMode(QListView.ViewMode.IconMode)
        self._list.setIconSize(_ICON)
        self._list.setGridSize(QSize(_ICON.width() + 24, _ICON.height() + 34))
        self._list.setWordWrap(True)
        self._list.setResizeMode(QListView.ResizeMode.Adjust)
        self._list.setMovement(QListView.Movement.Static)
        self._list.setMinimumHeight(_ICON.height() + 44)
        self._list.setMaximumHeight(_ICON.height() * 2 + 80)
        backdrop = QIcon(thumbnail_pixmap(None, self._options.backdrop))
        for entry in self._entries:
            item = QListWidgetItem(entry.name)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(f"分類: {entry.category}")
            item.setIcon(backdrop)
            self._list.addItem(item)
        self._list.itemClicked.connect(self._pick)

        self._before = _preview_label(self)
        self._after = _preview_label(self)
        self._before_title = QLabel("前（ごみ箱へ移る）", self)
        previews = QHBoxLayout()
        for title, picture in (
            (self._before_title, self._before),
            (QLabel("いま", self), self._after),
        ):
            column = QVBoxLayout()
            column.addWidget(title)
            column.addWidget(picture)
            previews.addLayout(column)

        self._save = QPushButton("保存", self)
        # Enter だけで保存（上書き）まで進まない 既定のボタンにせず、押したときだけ保存する
        self._save.setAutoDefault(False)
        self._save.setDefault(False)
        self._save.clicked.connect(self.accept)
        cancel = QPushButton("やめる", self)
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)

        form = QFormLayout()
        form.addRow("名前", self._name)
        form.addRow("分類", self._category)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._save)
        buttons.addWidget(cancel)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel(f"保存した{label}（選ぶとその名前で上書きする）", self))
        layout.addWidget(self._list)
        layout.addWidget(self._overwrite)
        layout.addWidget(self._warning)
        layout.addLayout(previews)
        layout.addLayout(buttons)

        self._name.textChanged.connect(lambda _text: self._update())
        self._category.currentTextChanged.connect(lambda _text: self._update())
        self._overwrite.toggled.connect(lambda _on: self._update())
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(VISIBLE_DELAY_MS)
        self._timer.timeout.connect(self._request_pictures)
        self._update()
        self._timer.start()

    # --- 外から ---

    def choice(self) -> SaveChoice:
        name, category = self._typed()
        return SaveChoice(
            name=name,
            category=category,
            replacing=self._occupant if self._overwrite.isChecked() else None,
        )

    def can_save(self) -> bool:
        return self._save.isEnabled()

    def warning(self) -> str:
        return self._warning.text() if self._warning.isVisible() else ""

    def set_name(self, name: str) -> None:
        self._name.setText(name)

    def set_category(self, category: str) -> None:
        self._category.setCurrentText(category)

    def set_overwrite(self, on: bool) -> None:
        self._overwrite.setChecked(on)

    def pick(self, name: str) -> None:
        """一覧から選ぶ（試験と、窓の中のクリック）"""
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(Qt.ItemDataRole.UserRole).name == name:
                self._pick(item)
                return

    # --- 中の手順 ---

    def _typed(self) -> tuple[str, str]:
        return self._name.text().strip(), self._category.currentText().strip() or "ユーザー"

    def _pick(self, item: QListWidgetItem) -> None:
        # 一覧から選んだ物は上書きとして扱う（選んだこと自体が上書きの指定）
        entry: LibraryEntry = item.data(Qt.ItemDataRole.UserRole)
        self._name.setText(entry.name)
        self._category.setCurrentText(entry.category)
        self._overwrite.setChecked(True)

    def _update(self) -> None:
        name, category = self._typed()
        self._occupant = None
        if name:
            self._occupant = self._library.occupant(
                self._kind, self._make(name, category), self._entries
            )
        occupied = self._occupant is not None
        lines: list[str] = []
        if occupied and self._occupant is not None:
            lines.append("この名前はもうあります 上書きすると、前の内容はごみ箱へ移ります")
            if self._kind == ALIAS and self._occupant.category != category:
                lines.append(
                    f"エイリアスの置き場は名前で決まるので、分類（いまは「{self._occupant.category}」）"
                    "が違っても上書きになります"
                )
            if not self._overwrite.isChecked():
                lines.append("上書きするなら「同じ名前があれば上書きする」に印を入れてください")
        self._warning.setText(" ".join(lines))
        self._warning.setVisible(bool(lines))
        self._before.setVisible(occupied)
        self._before_title.setVisible(occupied)
        self._save.setEnabled(bool(name) and (not occupied or self._overwrite.isChecked()))
        self._timer.start()

    def _request_pictures(self) -> None:
        """前後の見本と一覧の見本を出す 覚えている絵はすぐ、無い物は描くよう頼む"""
        if not self._thumbnails.enabled:
            return
        name, category = self._typed()
        targets: list[tuple[Item, QLabel | None, QListWidgetItem | None]] = [
            (self._make(name or "無題", category), self._after, None)
        ]
        if self._occupant is not None:
            targets.append((self._occupant.item, self._before, None))
        for row in range(self._list.count()):
            listed = self._list.item(row)
            targets.append((listed.data(Qt.ItemDataRole.UserRole).item, None, listed))
        for look, label, row_item in targets:
            found = self._thumbnails.cached(look)
            if found is not None:
                self._show(found, label, row_item)
                continue
            key = self._thumbnails.request(look)
            if key is None:
                continue
            # 前といまが同じ見た目なら鍵も同じ 1 つの鍵に貼る先を全部ためる
            self._waiting.setdefault(key, []).append((label, row_item))

    def _on_thumbnail(self, key: str) -> None:
        waiting = self._waiting.pop(key, [])
        found = self._thumbnails.thumbnail(key)
        if found is None:
            return
        for label, row_item in waiting:
            self._show(found, label, row_item)

    def _show(self, found: Thumbnail, label: QLabel | None, item: QListWidgetItem | None) -> None:
        pixmap = thumbnail_pixmap(found.image, self._options.backdrop)
        if label is not None:
            label.setPixmap(
                pixmap.scaled(
                    _PREVIEW,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        if item is not None:
            item.setIcon(QIcon(pixmap))

    def done(self, result: int) -> None:
        self._thumbnails.ready.disconnect(self._on_thumbnail)
        super().done(result)


def _preview_label(parent: QWidget) -> QLabel:
    label = QLabel(parent)
    label.setFixedSize(_PREVIEW)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setPixmap(QPixmap())
    themed_style(label, lambda: f"background: {Colors.VIEWER_BACKGROUND.name()};")
    return label


def ask_save(
    parent: QWidget | None,
    library: Library,
    kind: LibraryKind,
    *,
    make: Make,
    suggestion: str,
    category: str = "ユーザー",
) -> SaveChoice | None:
    """保存の窓を開く 〔保存〕で決めた物を返す やめたら ``None``"""
    dialog = SaveDialog(
        library, kind, make=make, suggestion=suggestion, category=category, parent=parent
    )
    answer = dialog.exec()
    choice = dialog.choice()
    dialog.deleteLater()
    return choice if answer == QDialog.DialogCode.Accepted else None


def save_look(
    parent: QWidget | None,
    library: Library,
    kind: LibraryKind,
    *,
    make: Make,
    suggestion: str,
    category: str = "ユーザー",
    ask: Callable[..., SaveChoice | None] | None = None,
    tell: Callable[[str], None] | None = None,
) -> LibraryEntry | None:
    """窓で決めて保存する 保存した物を返す（やめた・断られたら ``None``）

    ``ask`` と ``tell`` は試験で差し替える 省くと窓と知らせの小窓を出す
    """
    asking = ask if ask is not None else ask_save
    chosen = asking(parent, library, kind, make=make, suggestion=suggestion, category=category)
    if chosen is None:
        return None
    telling = tell if tell is not None else (lambda text: _inform(parent, kind, text))
    item = make(chosen.name, chosen.category)
    try:
        saved, trashed = library.save_keeping(kind, item, replacing=chosen.replacing)
        offer_undo(library, kind, saved, trashed)
        return saved
    except LibraryChangedError as exc:
        telling(
            f"{exc}\nほかで変わったので保存しなかった 一覧を読み直して、もう一度保存してください"
        )
    except LibraryConflictError as exc:
        # 窓を出した後で、ほかの窓が同じ名前で保存した 黙って上書きしない
        telling(f"{exc}\nほかの窓で同じ名前が保存された 上書きするなら、もう一度保存してください")
    except (LibraryError, OSError) as exc:
        telling(f"保存できなかった: {exc}")
    return None


#: 上書きして保存した知らせの〔元に戻す〕を出しておく長さ（ミリ秒） 押す気になるまでの間は
#: 出し、ほかの作業へ移った後まで残して押し間違えさせない 消えた後も管理の窓のごみ箱から戻せる
UNDO_NOTICE_MS = 15_000


class OverwriteUndo:
    """上書きして保存した 1 回ぶんの〔元に戻す〕 1 度だけ使える"""

    def __init__(
        self, library: Library, kind: LibraryKind, saved: LibraryEntry, trashed: TrashEntry
    ) -> None:
        self.library = library
        self.kind = kind
        self.saved = saved
        self.trashed = trashed
        self.used = False

    @property
    def label(self) -> str:
        return f"{_KIND_LABELS[self.kind]}「{self.saved.name}」を上書きした（前の内容はごみ箱）"

    def undo(self) -> str:
        """上書きの前に戻す 戻した結果の知らせの文を返す 2 度目は何もしない"""
        if self.used:
            return "もう元に戻してある"
        self.used = True
        try:
            self.library.undo_overwrite(self.saved, self.trashed)
        except LibraryChangedError as exc:
            # 保存した後でほかの窓が変えていた そちらの変更を戻すつもりで消さない
            return f"{exc} 戻さなかった（管理の窓のごみ箱から戻せる）"
        except (LibraryError, OSError) as exc:
            return f"戻せなかった: {exc}"
        return f"「{self.saved.name}」を上書きの前に戻した（上書きした方はごみ箱へ移した）"


class UndoOffers(QObject):
    """上書きして保存したことを、知らせの帯（:class:`UndoNotice`）へ伝える"""

    offered = Signal(object)


_offers = UndoOffers()


def undo_offers() -> UndoOffers:
    """保存した所（設定パネル・タイムライン・一覧の窓）と、知らせを出す窓をつなぐ"""
    return _offers


def offer_undo(
    library: Library, kind: LibraryKind, saved: LibraryEntry, trashed: TrashEntry | None
) -> None:
    """上書きしたときだけ〔元に戻す〕を出す 新しく保存しただけなら戻す前の内容が無い"""
    if trashed is not None:
        _offers.offered.emit(OverwriteUndo(library, kind, saved, trashed))


class UndoNotice(QWidget):
    """ステータスバーの知らせ 上書きした直後に〔元に戻す〕をしばらく出す"""

    #: 戻した結果の文 窓はステータスバーに出す
    finished = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)
        self.button = QPushButton("元に戻す", self)
        self.button.setAutoDefault(False)
        self.button.clicked.connect(self._undo)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self.button)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(UNDO_NOTICE_MS)
        self._timer.timeout.connect(self._expire)
        self._current: OverwriteUndo | None = None
        self.hide()

    @property
    def current(self) -> OverwriteUndo | None:
        """いま〔元に戻す〕で戻せる物 出ていなければ ``None``"""
        return self._current

    def offer(self, undo: object) -> None:
        """新しい上書きの〔元に戻す〕を出す 前の物は差し替える（戻すのは直前の 1 回）"""
        if not isinstance(undo, OverwriteUndo):
            return
        self._current = undo
        self._label.setText(undo.label)
        self.button.setEnabled(True)
        self.show()
        self._timer.start()

    def _expire(self) -> None:
        # 消えた後は押せない（ごみ箱からは戻せる）
        self._current = None
        self.hide()

    def _undo(self) -> None:
        undo = self._current
        # 押したらすぐ消す もう一度押して、戻した物をまた戻すことをさせない
        self._current = None
        self.button.setEnabled(False)
        self._timer.stop()
        self.hide()
        if undo is not None:
            self.finished.emit(undo.undo())


def _inform(parent: QWidget | None, kind: LibraryKind, text: str) -> None:
    QMessageBox.information(parent, f"{_KIND_LABELS[kind]}を保存", text)


def capture_look(
    project: Project | None,
    track: Track,
    clip: Clip,
    kind: LibraryKind,
    name: str,
    category: str,
) -> Item | None:
    """いまのクリップの見た目を、保存する形にする 保存できないクリップは ``None``"""
    if kind == ALIAS:
        if alias_refusal(clip) is not None:
            return None
        return Alias.of(name, clip, category=category)
    # 設定パネルの保存と同じ写し方（#288 の capture_preset） 画面の層の知識が要るので、そこを借りる
    from sashimono.ui.inspector.panel import capture_preset

    return replace(capture_preset(name, project, track, clip), category=category)
