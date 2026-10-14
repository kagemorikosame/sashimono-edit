"""字幕を置くときの見た目の一覧（#308）

プリセットとエイリアスの管理の窓（#276 #277 ``ui/library_dialog.py``）と同じ見本の絵の係・
同じ大きさの枠で並べる 上のタブで出どころ（いつもの・プリセット・エイリアス・テンプレートの棚）
を切り替え、名前で探せる 同じ物が窓によって違う絵で出ると、選んだ物と置いた物が同じか迷う

* 文字を持たない物（図形・スクリプトのオブジェクトのエイリアスなど）は灰色にして選べない
  理由は項目の説明に出す 並べないと、保存したはずのエイリアスが見つからないと受け取られる
* テンプレートの棚は数百本あることがあり、読むのが重い タブを初めて開いたときに並べ、
  配布物の中身は選んだときに読む 文字を持たない物はそこで灰色にする
* 見本の絵は見えている物だけを別のスレッドで描き、できた物から埋める（管理の窓と同じ）
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import TemplateEntry, TemplateError
from sashimono.compat.subtitle_looks import (
    ALIAS,
    DEFAULT,
    PRESET,
    SELECTED,
    TEMPLATE,
    LookChoice,
    alias_choice,
    default_choice,
    preset_choice,
    selected_choice,
    template_choice,
    template_refusal,
)
from sashimono.core.io.aliases import Alias
from sashimono.core.io.library import ALIAS as LIBRARY_ALIAS
from sashimono.core.io.library import PRESET as LIBRARY_PRESET
from sashimono.core.io.library import Library
from sashimono.core.io.presets import Preset
from sashimono.core.model import Clip
from sashimono.engine.render.look_preview import SampleSource
from sashimono.ui.library_thumbnails import Item, LookThumbnails, Thumbnail
from sashimono.ui.library_view import (
    VISIBLE_DELAY_MS,
    LibraryOptions,
    failed_pixmap,
    library_options,
    shared_thumbnails,
    show_as_grid,
    thumbnail_pixmap,
)
from sashimono.ui.shelf_looks import shelf_look
from sashimono.ui.theme import Colors, themed_style

__all__ = ["SAME", "LookPicker", "LookSources", "same_choice"]

#: 話し手ごとの選択で「全員と同じ」を表す出どころ
SAME = "same"

#: タブの並び（いつもの・プリセット・エイリアス・テンプレートの棚）
_BASIC = "basic"
_TABS = (
    (_BASIC, "いつもの"),
    (PRESET, "プリセット"),
    (ALIAS, "エイリアス"),
    (TEMPLATE, "テンプレートの棚"),
)

_CHOICE = Qt.ItemDataRole.UserRole
_KEY = Qt.ItemDataRole.UserRole + 1


def same_choice() -> LookChoice:
    return LookChoice(SAME, "全員と同じ", SAME)


@dataclass(frozen=True, slots=True)
class LookSources:
    """一覧に並べる物の出どころ 窓を開くたびに作る

    ``shelf`` はテンプレートの棚を読む手 棚のタブを初めて開いたときにだけ呼ぶ
    ``sample`` は既定の見た目の見本に描くクリップ（字幕の既定の大きさと縁取り）
    """

    library: Library
    shelf: Callable[[], Sequence[TemplateEntry]]
    sample: Clip
    selected: Clip | None = None


class LookPicker(QWidget):
    """見た目を 1 つ選ぶ一覧 選び直すと :attr:`changed`"""

    changed = Signal()

    def __init__(
        self,
        sources: LookSources,
        *,
        thumbnails: LookThumbnails | None = None,
        options: LibraryOptions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sources = sources
        self._options = options if options is not None else library_options()
        self._thumbnails = thumbnails if thumbnails is not None else shared_thumbnails()
        self._thumbnails.ready.connect(self._on_thumbnail)
        self._backdrop_icon = QIcon(thumbnail_pixmap(None, self._options.backdrop))
        #: 「全員と同じ」を並べるか（話し手ごとの選択のとき）
        self._allow_same = False
        self._choices: dict[str, list[LookChoice]] = {}
        #: 棚の配布物を読んだ結果の断る理由（鍵ごと） 選び直すたびに読み直さない
        self._template_refusals: dict[str, str] = {}
        self._current: LookChoice | None = None
        self._waiting: dict[str, list[QListWidgetItem]] = {}

        self._tabs = QTabBar(self)
        for _name, label in _TABS:
            self._tabs.addTab(label)
        self._tabs.currentChanged.connect(lambda _index: self._fill())
        self._search = QLineEdit(self)
        self._search.setPlaceholderText("名前で探す")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(lambda _text: self._fill())

        self._list = QListWidget(self)
        self._list.setObjectName("subtitle_looks")
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.itemSelectionChanged.connect(self._on_selected)
        self._list.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible())
        self._list.setMinimumHeight(180)

        self._note = QLabel(self)
        self._note.setObjectName("subtitle_look_note")
        self._note.setWordWrap(True)
        themed_style(self._note, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._tabs)
        top.addStretch(1)
        top.addWidget(self._search, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._note)

        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(VISIBLE_DELAY_MS)
        self._visible_timer.timeout.connect(self._request_visible)
        self._fill()

    # --- 外から ---

    def current(self) -> LookChoice | None:
        """選んでいる見た目 棚の配布物は読んだ結果の断る理由も入れて返す"""
        return self._current

    def set_allow_same(self, allow: bool) -> None:
        if allow == self._allow_same:
            return
        self._allow_same = allow
        self._choices.pop(_BASIC, None)
        self._fill()

    def choose(self, key: str) -> bool:
        """鍵で選ぶ 無ければ偽（覚えた見た目が消えていた など） 入っているタブへ移る"""
        for index, (tab, _label) in enumerate(_TABS):
            if tab == TEMPLATE and not key.startswith(f"{TEMPLATE}:"):
                # 棚を読むのは重い 棚の鍵でなければ探しに行かない
                continue
            found = next((c for c in self._choices_of(tab) if c.key == key), None)
            if found is None:
                continue
            if self._known(found).refusal:
                # 選べない物（文字を持たない など）は覚えていても選ばない 灰色の項目を
                # 選んだことにすると、窓を開いた直後から〔焼き込む〕が押せない
                return False
            self._search.blockSignals(True)
            self._search.clear()
            self._search.blockSignals(False)
            self._tabs.blockSignals(True)
            self._tabs.setCurrentIndex(index)
            self._tabs.blockSignals(False)
            self._fill(select=key)
            return True
        return False

    def shown(self) -> list[LookChoice]:
        """一覧に出ている物（並びのとおり） 試験が見る"""
        return [self._list.item(row).data(_CHOICE) for row in range(self._list.count())]

    def show_tab(self, kind: str) -> None:
        for index, (tab, _label) in enumerate(_TABS):
            if tab == kind:
                self._tabs.setCurrentIndex(index)
                return

    def release(self) -> None:
        """窓を閉じるときに、見本の知らせを外す（走り係は止めない 管理の窓と同じ）"""
        self._thumbnails.ready.disconnect(self._on_thumbnail)

    # --- 並べる ---

    def _tab(self) -> str:
        return _TABS[max(0, self._tabs.currentIndex())][0]

    def _choices_of(self, tab: str) -> list[LookChoice]:
        found = self._choices.get(tab)
        if found is not None:
            return found
        made: list[LookChoice] = []
        if tab == _BASIC:
            if self._allow_same:
                made.append(same_choice())
            if self._sources.selected is not None:
                made.append(selected_choice(self._sources.selected))
            made.append(default_choice())
        elif tab == PRESET:
            made = [
                preset_choice(entry.item)
                for entry in self._sources.library.entries(LIBRARY_PRESET)
                if isinstance(entry.item, Preset)
            ]
        elif tab == ALIAS:
            made = [
                alias_choice(entry.item)
                for entry in self._sources.library.entries(LIBRARY_ALIAS)
                if isinstance(entry.item, Alias)
            ]
        else:
            made = [template_choice(entry) for entry in self._sources.shelf()]
        self._choices[tab] = made
        return made

    def _fill(self, select: str | None = None) -> None:
        wanted = select if select is not None else (self._current.key if self._current else "")
        self._list.blockSignals(True)
        self._list.clear()
        self._waiting.clear()
        show_as_grid(self._list, self._thumbnails.enabled)
        words = self._search.text().strip().casefold()
        chosen: QListWidgetItem | None = None
        for choice in self._choices_of(self._tab()):
            if words and words not in choice.label.casefold():
                continue
            shown = self._known(choice)
            item = QListWidgetItem(shown.label)
            item.setData(_CHOICE, shown)
            if shown.refusal:
                # 選べない物は灰色にする 選べないのに並ぶ訳を説明に書く
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(shown.refusal)
            if self._thumbnails.enabled:
                item.setIcon(self._backdrop_icon)
            self._list.addItem(item)
            if shown.key == wanted:
                chosen = item
        if chosen is not None:
            chosen.setSelected(True)
            self._list.setCurrentItem(chosen)
            self._current = chosen.data(_CHOICE)
        self._list.blockSignals(False)
        if select is not None:
            self._on_selected()
        self._update_note()
        self._schedule_visible()

    def _known(self, choice: LookChoice) -> LookChoice:
        refusal = self._template_refusals.get(choice.key)
        return replace(choice, refusal=refusal) if refusal else choice

    def _on_selected(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        item = items[0]
        choice: LookChoice = item.data(_CHOICE)
        if (
            choice.kind == TEMPLATE
            and not choice.refusal
            and choice.key not in self._template_refusals
        ):
            choice = self._checked(choice)
            item.setData(_CHOICE, choice)
            if choice.refusal:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(choice.refusal)
        self._current = choice
        self._update_note()
        self.changed.emit()

    def _checked(self, choice: LookChoice) -> LookChoice:
        """棚の配布物を読み、字幕に着せられるかを確かめる 読んだ結果は覚える"""
        entry = choice.item
        assert isinstance(entry, TemplateEntry)
        try:
            refusal = template_refusal(entry.load(report=CompatibilityReport()))
        except (*TemplateError, OSError, ValueError) as exc:
            refusal = f"読み込めない: {exc}"
        self._template_refusals[choice.key] = refusal
        return replace(choice, refusal=refusal) if refusal else choice

    def _update_note(self) -> None:
        current = self._current
        parts: list[str] = []
        if current is not None and current.refusal:
            parts.append(f"「{current.label}」は使えない: {current.refusal}")
        elif self._list.count() == 0:
            parts.append(_EMPTY.get(self._tab(), "見つからない"))
        elif self._thumbnails.enabled and self._thumbnails.simple:
            parts.append("見本は文字と図形だけ（縁取り・グローなどのエフェクトは出ていない）")
        self._note.setText(" ".join(parts))

    # --- 見本の絵（管理の窓と同じ係） ---

    def _schedule_visible(self) -> None:
        if self._thumbnails.enabled:
            self._visible_timer.start()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の名前
        super().resizeEvent(event)
        self._schedule_visible()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt の名前
        super().showEvent(event)
        self._schedule_visible()

    def _request_visible(self) -> None:
        if not self._thumbnails.enabled:
            return
        viewport = self._list.viewport().rect()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_KEY) is not None:
                continue
            if not self._list.visualItemRect(item).intersects(viewport):
                continue
            look = self._look_of(item.data(_CHOICE))
            if look is None:
                item.setData(_KEY, "")
                continue
            cached = self._thumbnails.cached(look)
            if cached is not None:
                item.setData(_KEY, "")
                self._show_thumbnail(item, cached)
                continue
            key = self._thumbnails.request(look)
            if key is None:
                continue
            item.setData(_KEY, key)
            self._waiting.setdefault(key, []).append(item)

    def thumbnails_settled(self) -> bool:
        """見えている項目の見本がそろったか 試験が待つ"""
        if not self._thumbnails.enabled:
            return True
        return not self._waiting and not self._visible_timer.isActive()

    def _look_of(self, choice: LookChoice) -> Item | None:
        """見本の係へ渡す形 既定と選んでいるテキストはクリップをエイリアスの形にして描く"""
        item = choice.item
        if choice.kind == DEFAULT:
            return Alias.of("既定の見た目", self._sources.sample)
        if choice.kind == SELECTED and isinstance(item, Clip):
            return Alias.of("選んでいるテキスト", item)
        if isinstance(item, TemplateEntry):
            look: SampleSource = shelf_look(item)
            return look
        if choice.kind in (PRESET, ALIAS) and item is not None and not isinstance(item, Clip):
            return item
        return None

    def _on_thumbnail(self, key: str) -> None:
        waiting = self._waiting.pop(key, [])
        found = self._thumbnails.thumbnail(key)
        if found is None:
            return
        for item in waiting:
            self._show_thumbnail(item, found)
        if found.simple:
            self._update_note()

    def _show_thumbnail(self, item: QListWidgetItem, thumbnail: Thumbnail) -> None:
        if thumbnail.failed or (thumbnail.image is None and thumbnail.error):
            item.setIcon(QIcon(failed_pixmap()))
        elif thumbnail.image is not None:
            item.setIcon(QIcon(thumbnail_pixmap(thumbnail.image, self._options.backdrop)))


#: 何も並ばないときの添え書き どこで作れるかを書く
_EMPTY = {
    PRESET: "まだ無い 設定パネルの〔プリセット…〕→〔この見た目を保存…〕で作れる",
    ALIAS: "まだ無い タイムラインの右クリック〔エイリアスとして保存…〕で作れる",
    TEMPLATE: (
        "テンプレートが見つからない"
        " AviUtl2 の Alias フォルダか、YMM4 の ItemTemplate フォルダを探す"
    ),
}
