"""テンプレートの棚 AviUtl のエイリアスと YMM4 のアイテムテンプレートを並べる

配布されている字幕デザインは、見本の文字が入ったテキストオブジェクトとして
配られている 使い方は 2 通りあり、両方できるようにしてある

* **タイムラインへ置く** — 見本の文字ごと置く 新しくテロップを作るとき
* **選択中のクリップに適用** — 今の文字と長さを残して、見た目だけ着せ替える
  すでに打ってある字幕にデザインを当てるとき こちらが本命

下絵は**文字と図形だけ**を描いたもの 縁取りやグラデーションは GPU のエフェクト
として積まれるので、ここには出ない 出せない部分を出せているように見せると、
選ぶときの判断を誤らせる
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sashimono import __version__
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import (
    TemplateCatalog,
    TemplateEntry,
    TemplateError,
    template_catalog,
)
from sashimono.compat.mapped import MappedObject
from sashimono.ui.library_thumbnails import LookThumbnails, Thumbnail
from sashimono.ui.library_view import (
    VISIBLE_DELAY_MS,
    LibraryOptions,
    failed_pixmap,
    library_options,
    shared_thumbnails,
    thumbnail_pixmap,
)
from sashimono.ui.report_masking import marked_root_labels, mask_report, root_lines
from sashimono.ui.shelf_looks import shelf_look
from sashimono.ui.system_clipboard import clipboard
from sashimono.ui.theme import Colors, themed_style

__all__ = ["TemplateDialog", "notes_text"]

#: 下絵の大きさ 一覧の横に置くので、縦横比だけ合わせた小さめのもの
_PREVIEW = (384, 216)

#: 下絵を描くときの実寸 配布物は 1080p を前提にしている
_CANVAS = (1920, 1080)

#: 一覧の小さな見本の大きさ（論理の画素） 名前の行の高さに収まる程度
_ICON = QSize(64, 36)
#: 一覧の項目に持たせる見本の鍵 頼んだ物は 2 度頼まない（空は覚えた絵を貼った）
_KEY = Qt.ItemDataRole.UserRole + 1


def notes_text(
    entry: TemplateEntry,
    contents: str,
    notes: Sequence[str],
    roots: Sequence[Path | str] = (),
    folders: Sequence[tuple[str, str]] | None = None,
) -> str:
    """互換の報告に貼る文面 版と、選んだテンプレートの見分けと、注意書きを全部入れる

    受けた側が同じ物を手元で開けるように、名前・種類・ファイルの場所（棚の中の
    どこか）を入れる YMM4 は 1 ファイルに何本も入っているので、原本の中の位置も入れる
    テンプレートの中身（見本の文字・フォント・JSON など）は入れない 配布物で、
    再配布の条件が作者ごとに違う 足りない物は注意書きの行に名前で出ている

    伏せ方は互換性レポートと同じ物を使う 棚の置き場は本人が決めた場所で、
    ホームの外なら利用者名を含みうるので、外にある物は ``<探索先1>`` の印にする
    """
    if entry.source == "ymm4" and entry.error:
        # 読めないファイルは本数が分からない 1 本目と書くと、残りは読めたと読まれる
        kind = "YMM4 のアイテムテンプレート（ファイルを読めず、本数は不明）"
    elif entry.source == "ymm4":
        # 原本の中の位置を書く 読める物だけの通し番号だと、空の物を飛ばした後ろや
        # エフェクトの一覧の物が、原本の別の物を指す
        kind = f"YMM4 のアイテムテンプレート（{entry.origin or 'ファイル全体で 1 本'}）"
    else:
        kind = "AviUtl のエイリアス"
    lines = [
        f"Sashimono Edit {__version__} テンプレートの注意書き",
        f"名前: {entry.name}",
        f"種類: {kind}",
    ]
    # AviUtl の見出しは置き場のフォルダ名で、置き場の直下なら置き場そのものの名前
    # （利用者名でありうる）になる 1 段の名前は伏せる側が拾わないので入れない
    # 場所はファイルの行に伏せた形で出る YMM4 の見出しはファイル名と中の分類から作る
    if entry.source == "ymm4":
        lines.append(f"分類: {entry.folder}")
    lines += [
        f"ファイル: {entry.path}",
        f"読んだ結果: {contents}",
        *root_lines([str(root) for root in roots]),
    ]
    lines += ["注意書き:", *(f"  {note}" for note in notes)] if notes else ["注意書き: （なし）"]
    return mask_report("\n".join(lines), roots, folders)


class TemplateDialog(QDialog):
    """テンプレートを選んで、置くか着せるかを決める"""

    #: 見えている項目の見本がそろった（:meth:`thumbnails_settled` が真になった）
    thumbnails_done = Signal()

    def __init__(
        self,
        catalog: TemplateCatalog | None = None,
        parent: QWidget | None = None,
        *,
        roots: tuple[Path, ...] | None = None,
        thumbnails: LookThumbnails | None = None,
        options: LibraryOptions | None = None,
    ) -> None:
        """``roots`` を渡すと、既定のフォルダではなくその置き場だけを並べる

        渡せないと「読み直す」が必ず本人のフォルダを見に行く 写真を撮る道具
        （``tools/shots.py``）は決まった置き場だけを並べたいので、その口を開ける
        ``thumbnails`` と ``options`` は一覧の小さな見本（#277） 省くとプリセットの一覧と
        同じ係と設定を使う
        """
        super().__init__(parent)
        self._options = options if options is not None else library_options()
        self._thumbnails = thumbnails if thumbnails is not None else shared_thumbnails()
        self._thumbnails.ready.connect(self._on_thumbnail)
        #: 見本の鍵ごとの一覧の項目 絵ができたら同じ鍵の項目へ貼る
        self._waiting: dict[str, list[QTreeWidgetItem]] = {}
        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(VISIBLE_DELAY_MS)
        self._visible_timer.timeout.connect(self._request_visible)
        self.setWindowTitle("テンプレート")
        self.resize(820, 520)
        self._catalog = catalog if catalog is not None else template_catalog()
        self._roots = roots
        self._report = CompatibilityReport()
        self._loaded: list[MappedObject] = []
        #: 貼る文の「中身」の行と注意書き 画面に出した物と同じ物を写す
        self._contents = ""
        self._note_lines: tuple[str, ...] = ()
        #: 最後に走査した置き場 貼る文で伏せる探索先
        self._scanned: tuple[Path, ...] = ()

        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(["名前"])
        self._tree.setColumnCount(1)
        self._tree.currentItemChanged.connect(lambda *_: self._on_selected())
        if self._shows_thumbnails():
            self._tree.setIconSize(_ICON)
        # 送ったり畳みを開いたりすると見える項目が変わる その見本を頼む
        self._tree.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible())
        self._tree.itemExpanded.connect(lambda _item: self._schedule_visible())

        self._preview = QLabel(self)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(*_PREVIEW)
        themed_style(self._preview, lambda: f"background: {Colors.VIEWER_BACKGROUND.name()};")

        self._detail = QLabel(self)
        self._detail.setWordWrap(True)
        themed_style(self._detail, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._notes = QListWidget(self)
        self._notes.setMaximumHeight(90)
        # 一覧からは行を 1 つずつしか選べず、Ctrl+C でも写せない 互換の報告に
        # 手で打ち写してもらうと、写し間違いと写し漏れがそのまま届く
        self._copy_button = QPushButton("内容をコピー", self)
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        copy_row = QHBoxLayout()
        copy_row.addStretch(1)
        copy_row.addWidget(self._copy_button)

        self._place_button = QPushButton("タイムラインへ置く", self)
        self._restyle_button = QPushButton("選択中のクリップに適用", self)
        self._place_button.clicked.connect(lambda: self._finish("place"))
        self._restyle_button.clicked.connect(lambda: self._finish("restyle"))

        rescan = QPushButton("読み直す", self)
        rescan.clicked.connect(self.refresh)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.setText("閉じる")
        buttons.addButton(rescan, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self._restyle_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self._place_button, QDialogButtonBox.ButtonRole.AcceptRole)

        side = QVBoxLayout()
        side.addWidget(self._preview)
        side.addWidget(self._detail)
        side.addWidget(self._notes)
        side.addLayout(copy_row)
        side.addStretch(1)

        columns = QHBoxLayout()
        columns.addWidget(self._tree, 1)
        columns.addLayout(side, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(columns, 1)
        layout.addWidget(buttons)

        #: 選ばれた結果 ``("place" | "restyle", 写した結果)``
        self.choice: tuple[str, list[MappedObject]] | None = None
        #: 選ばれたテンプレートの置き場 書かれた素材のパスに無いとき、ここで探す
        self.origin: Path | None = None
        self.refresh()

    # --- 一覧 ---

    def refresh(self) -> None:
        """棚を読み直して並べ直す"""
        from sashimono.compat.catalog import default_template_roots

        self._scanned = self._roots if self._roots is not None else default_template_roots()
        self._catalog.scan(self._scanned)
        # どの印がどの置き場かは画面にだけ出す 貼る文には出さない（出すと伏せた意味が無い）
        # 報告を受けた側に「<探索先1> はどこか」と聞かれたときに、本人がここで答えられる
        self._copy_button.setToolTip(
            "\n".join(
                [
                    "選んだテンプレートの注意書きを、互換の報告に貼れる形で写します",
                    "テンプレートの中身は入りません ホームと設定の置き場は伏せます",
                    *root_lines(marked_root_labels(self._scanned)),
                ]
            )
        )
        self._tree.clear()

        groups: dict[str, QTreeWidgetItem] = {}
        for entry in self._catalog.all():
            label = f"{entry.folder}（{'YMM4' if entry.source == 'ymm4' else 'AviUtl'}）"
            group = groups.get(label)
            if group is None:
                group = QTreeWidgetItem([label])
                groups[label] = group
                self._tree.addTopLevelItem(group)
            # 読めなかった物は選ぶ前から分かるようにする 選んで初めて分かると、
            # 読めない物ばかりの棚で 1 つずつ選んで確かめることになる
            node = QTreeWidgetItem([f"{entry.name}（読めません）" if entry.error else entry.name])
            node.setData(0, Qt.ItemDataRole.UserRole, entry)
            if self._shows_thumbnails():
                node.setIcon(0, self._backdrop_icon)
            group.addChild(node)

        self._waiting.clear()
        self._tree.expandAll()
        self._schedule_visible()
        if not groups:
            self._detail.setText(
                # 区切りの空白が無いと、2 つの文が 1 つにつながって読める
                "テンプレートが見つかりません "
                "AviUtl2 の Alias フォルダか、YMM4 の ItemTemplate フォルダを探します"
            )
        self._on_selected()

    # --- 一覧の見本（#277） ---

    def _shows_thumbnails(self) -> bool:
        return self._options.shelf and self._thumbnails.enabled

    @functools.cached_property
    def _backdrop_icon(self) -> QIcon:
        return QIcon(thumbnail_pixmap(None, self._options.backdrop))

    def _schedule_visible(self) -> None:
        if self._shows_thumbnails():
            self._visible_timer.start()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の名前
        super().resizeEvent(event)
        self._schedule_visible()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt の名前
        super().showEvent(event)
        self._schedule_visible()

    def _leaves(self) -> list[QTreeWidgetItem]:
        found: list[QTreeWidgetItem] = []
        for top in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(top)
            if group is None:
                continue
            for row in range(group.childCount()):
                child = group.child(row)
                if child is not None:
                    found.append(child)
        return found

    def _request_visible(self) -> None:
        """見えている項目の見本だけを出す 棚は数百本あり、全部を頼むと開くのが遅れる"""
        if not self._shows_thumbnails():
            return
        viewport = self._tree.viewport().rect()
        for node in self._leaves():
            if node.data(0, _KEY) is not None:
                continue
            if not self._tree.visualItemRect(node).intersects(viewport):
                continue
            entry = node.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(entry, TemplateEntry):
                continue
            look = shelf_look(entry)
            cached = self._thumbnails.cached(look)
            if cached is not None:
                node.setData(0, _KEY, "")
                self._show_thumbnail(node, cached)
                continue
            key = self._thumbnails.request(look)
            if key is None:
                continue
            node.setData(0, _KEY, key)
            self._waiting.setdefault(key, []).append(node)
        self._tell_if_settled()

    def _on_thumbnail(self, key: str) -> None:
        waiting = self._waiting.pop(key, [])
        found = self._thumbnails.thumbnail(key)
        if found is not None:
            for node in waiting:
                self._show_thumbnail(node, found)
        if waiting:
            self._tell_if_settled()

    def thumbnails_settled(self) -> bool:
        """見えている項目の見本がそろったか（出さない設定なら常に真） 写真を撮る道具が待つ"""
        if not self._shows_thumbnails():
            return True
        return not self._waiting and not self._visible_timer.isActive()

    def _tell_if_settled(self) -> None:
        if self.thumbnails_settled():
            self.thumbnails_done.emit()

    def _show_thumbnail(self, node: QTreeWidgetItem, thumbnail: Thumbnail) -> None:
        if thumbnail.failed or (thumbnail.image is None and thumbnail.error):
            # 読めない配布物は失敗の印 地だけにすると、描いている途中と見分けが付かない
            node.setIcon(0, QIcon(failed_pixmap()))
            node.setToolTip(0, thumbnail.error)
        elif thumbnail.image is not None:
            node.setIcon(0, QIcon(thumbnail_pixmap(thumbnail.image, self._options.backdrop)))
            if thumbnail.empty:
                # 画像を使う物・動きで後から出る物・エフェクトだけの物など 地だけの見本の理由
                node.setToolTip(0, "見本のコマに見える物がない（素材の画像は見本では読まない）")
            elif thumbnail.simple:
                node.setToolTip(0, "見本は文字と図形だけ（エフェクトは出ていない）")

    def done(self, result: int) -> None:
        # 走り係は止めない（プリセットの一覧と同じ） 知らせだけ外す
        self._thumbnails.ready.disconnect(self._on_thumbnail)
        super().done(result)

    def _selected_entry(self) -> TemplateEntry | None:
        item = self._tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.ItemDataRole.UserRole)
        return data if isinstance(data, TemplateEntry) else None

    def _on_selected(self) -> None:
        entry = self._selected_entry()
        self._loaded = []
        self._notes.clear()
        self._contents = ""
        self._note_lines = ()
        self._preview.setPixmap(QPixmap())
        self._place_button.setEnabled(entry is not None)
        self._restyle_button.setEnabled(False)
        # 読めなかったテンプレートも写せるようにする 読めない理由こそ報告に要る
        self._copy_button.setEnabled(entry is not None)
        if entry is None:
            return

        self._report.clear()
        try:
            self._loaded = entry.load(report=self._report)
        except TemplateError as exc:
            self._contents = f"読み込めません: {exc}"
            self._detail.setText(self._contents)
            self._place_button.setEnabled(False)
            return

        self._contents = self._summarize()
        self._detail.setText(self._describe(entry))
        # 絵を持たないテンプレート（YMM4 のアニメーション効果など）は置けない
        # 着せることしかできないので、そちらだけを押せるようにする
        self._place_button.setEnabled(any(item.has_picture for item in self._loaded))
        self._restyle_button.setEnabled(
            any(
                inner.clip.source and inner.clip.source.kind == "text"
                for inner in self._walk_loaded()
            )
            or self._is_effects_only()
        )
        self._note_lines = self._report.lines()
        self._notes.addItems(self._note_lines)
        self._show_preview()

    def _walk_loaded(self) -> list[MappedObject]:
        """まとめた中身（YMM4 の合成するグループ）も含めた全部 文字は中にあることがある"""
        return [inner for item in self._loaded for inner in item.walk()]

    def _is_effects_only(self) -> bool:
        return bool(self._loaded) and not any(item.has_picture for item in self._loaded)

    def _summarize(self) -> str:
        """読んだ結果の 1 行 画面と貼る文の両方に出す 別々に組むと食い違う"""
        effects = sum(len(item.clip.effects) for item in self._walk_loaded())
        if self._is_effects_only():
            return f"エフェクトだけのテンプレート（{effects} 段）"
        kinds = [item.kind or "?" for item in self._loaded]
        return f"{len(self._loaded)} オブジェクト（{'、'.join(kinds)}）／エフェクト {effects} 段"

    def _describe(self, entry: TemplateEntry) -> str:
        if self._is_effects_only():
            hint = "中身は持ちません 選んだクリップに効果を足す形で使います"
        else:
            hint = "下絵は文字と図形だけ 縁取りやグラデーションは含まれていません"
        return f"{entry.path}\n{self._summarize()}\n{hint}"

    def copy_to_clipboard(self) -> None:
        """選んだテンプレートの注意書きを、報告に貼れる形でクリップボードへ写す"""
        entry = self._selected_entry()
        if entry is None:
            return
        clipboard().setText(notes_text(entry, self._contents, self._note_lines, self._scanned))

    def _show_preview(self) -> None:
        """先頭のオブジェクトの中身だけを描く

        まとめた中身（YMM4 の合成するグループ）の中まで探す 合成するグループは
        それ自体が中身を持たないので、上だけを見ると吹き出しの字幕テンプレートの
        下絵が何も出ない
        """
        from sashimono.engine.sources import render_source

        source = next((item.clip.source for item in self._walk_loaded() if item.clip.source), None)
        if source is None:
            return
        # 1080p で描いてから縮める 文字の大きさは 1080p を前提に決められて
        # いるので、小さい画面にそのまま描くと画面からはみ出す
        array = render_source(source, *_CANVAS)
        if array is None:  # pragma: no cover - 既知の種別なら必ず描ける
            return
        image = QImage(
            array.tobytes(), _CANVAS[0], _CANVAS[1], QImage.Format.Format_RGBA8888
        ).copy()
        self._preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                *_PREVIEW,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _finish(self, action: str) -> None:
        if not self._loaded:
            return
        self.choice = (action, self._loaded)
        entry = self._selected_entry()
        self.origin = entry.path.parent if entry is not None else None
        self.accept()
