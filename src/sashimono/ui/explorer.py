"""エクスプローラー 登録したフォルダの素材を探して下見し、タイムラインへ置く（#520 #521）

作品に読み込んだ素材の一覧（:mod:`sashimono.ui.media_pool`）とは別のパネルにする 手持ちの
効果音や BGM を使うたびに読み込みの窓でファイルを探していた手間を、ここで登録したフォルダを
見て回るだけにする AI アシスタントが自動で使う素材の出どころも、ここで登録したフォルダ（#522）

作りは動画編集ソフトのメディアブラウザーの考え方に合わせる（2026-11-24 本人の要望）
左に登録したフォルダを分類ごとに束ねた木、右の上に道具の帯（今いる場所・名前と種類の絞り込み・
一覧とアイコンの切り替え・アイコンの大きさ）と中身、右の下に下見のプレーヤー
選ぶと下見に出し、ダブルクリックで再生する（好みで「タイムラインへ置く」にも変えられる）

重い仕事（測る・見本の絵と波形を作る・再生のデコード）はどれも裏のスレッドで行い、画面の
スレッドを止めない 測った物は AI の道具と同じ控え（:class:`MeasureBook`）に入れ、見本の絵と
波形は本人のキャッシュの置き場へ書く 登録の保存と見て回る所はコア
（:mod:`sashimono.core.material_library`）にあり、ここは並べて見せるだけにする
"""

from __future__ import annotations

import contextlib
import io
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QMimeData,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    QPointF,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QDesktopServices,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.io.atomic import write_atomic_bytes
from sashimono.core.material_library import (
    CATEGORIES,
    CATEGORY_OTHER,
    KIND_IMAGE,
    KIND_SOUND,
    KIND_VIDEO,
    LibraryFile,
    LibraryFolder,
    Measure,
    MeasureBook,
    add_folder,
    is_favorite,
    kind_of,
    list_folder,
    match_files,
    reachable_dir,
    remove_folder,
    same_place,
    scan_folder,
    toggle_favorite,
)
from sashimono.core.model import MediaItem
from sashimono.engine.audio.library_sound import (
    PreviewPlayer,
    measure_library_file,
    probe,
    sound_peaks,
    still_of,
)
from sashimono.ui.flow_layout import FlowLayout, narrow_combo
from sashimono.ui.media_icons import GRID_ICON_SIZE, audio_icon, pending_icon, thumbnail_icon
from sashimono.ui.theme import Colors

__all__ = [
    "COLUMNS",
    "DISPLAY_LIMIT",
    "DOUBLE_CLICK_ACTIONS",
    "DOUBLE_CLICK_PREVIEW",
    "DOUBLE_CLICK_TIMELINE",
    "FAVORITES",
    "ICON_SIZE_RANGE",
    "STAR_OFF",
    "STAR_ON",
    "VIEW_ICONS",
    "VIEW_LIST",
    "VIEW_MODES",
    "ExplorerPanel",
    "ask_category",
    "ask_folder",
    "sound_peaks",
]

#: 一覧に並べる行の上限 数万の行を並べると、絞り込みの 1 文字ごとに画面が止まる
#: 越えたら絞り込むよう知らせる
DISPLAY_LIMIT = 3000
#: 表示の切り替え 一覧（列のある表）とアイコン（見本の絵を並べる）
VIEW_LIST = "list"
VIEW_ICONS = "icons"
VIEW_MODES = (VIEW_LIST, VIEW_ICONS)
#: アイコンの幅（画素）の範囲 小さいと絵が読めず、大きいとパネルに 1 つしか並ばない
ICON_SIZE_RANGE = (64, 256)
#: ダブルクリックの動き 下見で再生する（メディアブラウザーと同じ）か、タイムラインへ置くか
DOUBLE_CLICK_PREVIEW = "preview"
DOUBLE_CLICK_TIMELINE = "timeline"
DOUBLE_CLICK_ACTIONS = {
    DOUBLE_CLICK_PREVIEW: "下見で再生する",
    DOUBLE_CLICK_TIMELINE: "タイムラインへ置く",
}

#: 一覧の列 測っていない値は空で出し、裏で測れたら埋める
#: 頭の ☆ はお気に入りの印 押すと入れる・外すが切り替わる
COLUMNS = ("☆", "名前", "種類", "長さ", "大きさ", "解像度", "フレームレート", "更新日", "フォルダ")
_STAR, _NAME, _KIND, _LENGTH, _LOUDNESS, _SIZE, _FPS, _MODIFIED, _FOLDER = range(len(COLUMNS))
#: お気に入りの印 付いている・付いていない
STAR_ON = "★"
STAR_OFF = "☆"
#: 木のお気に入りの根と、その下の項目の印（登録したフォルダの場所の代わりに持つ）
FAVORITES = "お気に入り"
_FAVORITE_NODE = "★"
_KIND_NAMES = {KIND_SOUND: "音", KIND_VIDEO: "動画", KIND_IMAGE: "画像"}
#: 種類の絞り込み（空はすべて）
_KIND_FILTERS = (("", "すべて"), (KIND_SOUND, "音"), (KIND_VIDEO, "動画"), (KIND_IMAGE, "画像"))

#: 行の持つファイルの場所・並べ替えの値・ファイルの印
_PATH = Qt.ItemDataRole.UserRole
_SORT = Qt.ItemDataRole.UserRole + 1
_ID = Qt.ItemDataRole.UserRole + 2
#: アイコン表示で名前の下に出す長さ
_UNDER = Qt.ItemDataRole.UserRole + 3
#: お気に入りか（アイコン表示で名前の頭に印を出す）
_FAV = Qt.ItemDataRole.UserRole + 4

#: 木の項目の持つ物 （登録したフォルダの場所, 開いたフォルダの場所） 分類の根は分類の名前
_NODE = Qt.ItemDataRole.UserRole
#: 下のフォルダをまだ読んでいない印の子 開いたときに読む（数万のフォルダを先に読まない）
_PLACEHOLDER = "…"

#: 1 つの場所を開いたときに裏で測る数の上限 一覧を開くたびに数千を測り続けない
_MEASURE_LIMIT = 400
#: 波形の見本の点の数
_PEAK_COLUMNS = 400
#: 見本の絵を覚える大きさ（長い辺） アイコンの最大より少し大きくして、縮めて使う
_THUMB_EDGE = 320
#: 下見を描き直す間隔（ミリ秒）
_TICK_MS = 33
#: 狭いとき上下に並べ替える幅（画素） 素材一覧と重ねた既定の左の列の幅でも木と中身が欠けない
_NARROW = 560


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


def _clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes}:{secs:02}"


# --- 裏の仕事 ---


class _Worker(QObject):
    """裏のスレッドで測る・見本を作る 終わったら画面のスレッドへ知らせる

    知らせは ``finished``（仕事の名前, 結果） この物は画面のスレッドに住むので、裏の
    スレッドから送った知らせは Qt が画面のスレッドに積み直す 画面の部品はそこで触る
    """

    finished = Signal(str, object)

    def __init__(self, parent: QObject | None = None, *, workers: int = 2) -> None:
        super().__init__(parent)
        # 2 本だけ 測るのは頭の 1 分のデコードで重く、本数を増やすと作品の再生と取り合う
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="explorer")
        #: 仕事の名前 → （登録の印, 仕事） 印は同じ名前の古い仕事が終わったときに、
        #: 後から頼んだ新しい仕事の登録まで消さないため
        self._pending: dict[str, tuple[object, Future[None]]] = {}
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, key: str, job: Callable[[], object]) -> None:
        with self._lock:
            if self._closed or key in self._pending:
                return
            token = object()

            def run() -> None:
                try:
                    result = job()
                except Exception:
                    # 1 つの壊れた素材で裏の仕事を止めない その行の列と絵が空のまま残るだけ
                    result = None
                with self._lock:
                    entry = self._pending.get(key)
                    if entry is not None and entry[0] is token:
                        del self._pending[key]
                    if self._closed:
                        return
                self.finished.emit(key, result)

            self._pending[key] = (token, self._pool.submit(run))

    def forget(self) -> None:
        """待っている仕事を捨てる（別のフォルダを開いた）

        まだ始まっていない仕事は取り消す 残すと、前の場所の数千の見本を作り終えるまで、
        新しく開いた場所の見本が待たされる 走っている物は終わるまで待たない
        """
        with self._lock:
            for _token, future in self._pending.values():
                future.cancel()
            self._pending.clear()

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


class _ThumbStore:
    """見本の絵と波形の控え 本人のキャッシュの置き場へ ``.npy`` で書く

    PNG にしないのは、裏のスレッドで Qt の絵を作らずに済ませるため（numpy の配列のまま
    読み書きでき、画面のスレッドで 1 度だけ絵にする）
    """

    def __init__(self, folder: Path | None) -> None:
        self._folder = folder

    def _path(self, file: LibraryFile, kind: str) -> Path | None:
        if self._folder is None:
            return None
        return self._folder / f"{file.id}-{file.size}-{file.modified_ns}-{kind}.npy"

    def load(self, file: LibraryFile, kind: str) -> np.ndarray | None:
        path = self._path(file, kind)
        if path is None or not path.is_file():
            return None
        try:
            loaded: np.ndarray = np.load(path, allow_pickle=False)
            return loaded
        except (OSError, ValueError):
            return None

    def save(self, file: LibraryFile, kind: str, data: np.ndarray) -> None:
        path = self._path(file, kind)
        if path is None:
            return
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic_bytes(path, buffer.getvalue())
        except OSError:
            # 書けなくても作り直せば済む 見本の絵の置き場が無いだけで画面を止めない
            return


def _small_tile(tile: np.ndarray) -> np.ndarray:
    while max(tile.shape[0], tile.shape[1]) > _THUMB_EDGE * 2:
        tile = tile[::2, ::2]
    return np.ascontiguousarray(tile)


def _peaks_icon(peaks: np.ndarray) -> QIcon:
    """波形の見本の絵 音の素材のアイコン 決まった棒の印より、どんな音かが目で分かる"""
    size = GRID_ICON_SIZE
    canvas = QPixmap(size)
    canvas.fill(Colors.AUDIO_CLIP)
    painter = QPainter(canvas)
    painter.setPen(QPen(Colors.WAVEFORM, 1))
    middle = size.height() / 2
    count = len(peaks)
    for x in range(size.width()):
        level = float(peaks[min(count - 1, x * count // size.width())]) if count else 0.0
        reach = max(0.5, level * (middle - 4))
        painter.drawLine(QPointF(x, middle - reach), QPointF(x, middle + reach))
    painter.end()
    return QIcon(canvas)


# --- 中身の一覧 ---


class _FileModel(QStandardItemModel):
    """中身の行 ドラッグでファイルの URL を載せる

    タイムラインはエクスプローラー（OS）から落としたファイルと同じく、読み込んで落とした所へ
    置く（取り消し 1 回で戻る） 素材一覧の行と違い、まだ作品の素材ではないので ID は載せない
    """

    def mimeTypes(self) -> list[str]:  # noqa: N802 - Qt の命名規約
        return ["text/uri-list"]

    def mimeData(self, indexes: Sequence[QModelIndex]) -> QMimeData:  # noqa: N802 - Qt の命名規約
        mime = QMimeData()
        rows = sorted({index.row() for index in indexes})
        mime.setUrls([QUrl.fromLocalFile(str(self.item(row, _NAME).data(_PATH))) for row in rows])
        return mime

    def supportedDragActions(self) -> Qt.DropAction:  # noqa: N802 - Qt の命名規約
        # 写すことだけを許す 動かす（Move）で受け取られると、Qt は引いた元の行を消す
        return Qt.DropAction.CopyAction


class _SortProxy(QSortFilterProxyModel):
    """列の見出しで並べ替える 数の列は数で並べる（文字で並べると 10 秒が 9 秒の前に来る）"""

    def lessThan(  # noqa: N802 - Qt の命名規約
        self, left: QModelIndex | QPersistentModelIndex, right: QModelIndex | QPersistentModelIndex
    ) -> bool:
        a, b = left.data(_SORT), right.data(_SORT)
        if a is None:
            return b is not None
        if b is None:
            return False
        return bool(a < b)


class _UnderDelegate(QStyledItemDelegate):
    """アイコン表示で名前の下に長さを出す"""

    def initStyleOption(  # noqa: N802 - Qt の命名規約
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> None:
        super().initStyleOption(option, index)
        if index.data(_FAV):
            # アイコン表示には印の列が無い 名前の頭に印を出して、一覧と同じく見分けられるようにする
            option.text = f"{STAR_ON} {option.text}"
        under = index.data(_UNDER)
        if under:
            # 行の区切りの字で渡す 改行の字のままだと 1 行に詰まって名前と長さがくっつく
            option.text = f"{option.text} {under}"


# --- 下見 ---


class _WaveView(QWidget):
    """音の下見の波形と再生位置の線 押した所へ飛ぶ"""

    seek_requested = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: np.ndarray | None = None
        self._fraction = 0.0
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_peaks(self, peaks: np.ndarray | None) -> None:
        self._peaks = peaks
        self.update()

    def set_fraction(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        if abs(fraction - self._fraction) > 1e-4:
            self._fraction = fraction
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt の命名規約
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Colors.MEDIA_BACKDROP)
        width, height = self.width(), self.height()
        middle = height / 2
        peaks = self._peaks
        if peaks is not None and len(peaks):
            painter.setPen(QPen(Colors.WAVEFORM, 1))
            for x in range(width):
                level = float(peaks[min(len(peaks) - 1, x * len(peaks) // max(1, width))])
                reach = max(0.5, level * (middle - 2))
                painter.drawLine(QPointF(x, middle - reach), QPointF(x, middle + reach))
        painter.setPen(QPen(Colors.PLAYHEAD, 2))
        head = self._fraction * width
        painter.drawLine(QPointF(head, 0), QPointF(head, height))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self.width() > 0:
            self.seek_requested.emit(event.position().x() / self.width())


class _Picture(QLabel):
    """下見の絵 場所の大きさに合わせて縦横比を保って縮める"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(80, 45))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_tile(self, tile: np.ndarray | None) -> None:
        if tile is None:
            self._image = None
            self.clear()
            return
        height, width = int(tile.shape[0]), int(tile.shape[1])
        data = np.ascontiguousarray(tile[:, :, :4] if tile.ndim == 3 else tile, dtype=np.uint8)
        if data.ndim != 3 or data.shape[2] != 4:
            return
        self._image = QImage(
            data.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        ).copy()
        self._draw()

    def has_picture(self) -> bool:
        return self._image is not None

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        self._draw()

    def _draw(self) -> None:
        if self._image is None:
            return
        self.setPixmap(
            QPixmap.fromImage(
                self._image.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        )


class _PreviewPane(QWidget):
    """エクスプローラーの中の小さなプレーヤー 絵か波形・再生／一時停止・シーク・時刻・音量"""

    message = Signal(str)
    #: 裏で調べ終えた（ファイル, 素材か断りの文） 裏のスレッドから送り、画面のスレッドで受ける
    _probed = Signal(object, object)

    def __init__(
        self,
        player: PreviewPlayer,
        before_play: Callable[[], None] | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._player = player
        self._before_play = before_play
        self._shown_frame = -1
        self._seeking = False
        self._last_error: str | None = None
        #: 調べている（まだ開いていない）ファイルと、開いたらすぐ鳴らすか
        self._wanted: Path | None = None
        self._peaks_for_wanted: np.ndarray | None = None
        self._play_when_open = False
        self._probed.connect(self._on_probed)

        self._title = QLabel("ファイルを選ぶとここで下見できます", self)
        self._title.setWordWrap(True)
        self._picture = _Picture(self)
        self._wave = _WaveView(self)
        self._wave.hide()
        self._wave.seek_requested.connect(self._seek_fraction)

        self.play_button = QToolButton(self)
        self.play_button.setText("再生")
        self.play_button.setToolTip("選んだファイルを下見で再生する・止める（Space）")
        self.play_button.clicked.connect(self.toggle)
        self._seek = QSlider(Qt.Orientation.Horizontal, self)
        self._seek.setRange(0, 1000)
        self._seek.setToolTip("下見の再生位置")
        self._seek.sliderPressed.connect(self._press_seek)
        self._seek.sliderReleased.connect(self._release_seek)
        self._seek.sliderMoved.connect(self._move_seek)
        # キーで動かしたときもシークする（ドラッグを離すのを待つ作りだと、キーの分は次の
        # 描き直しで元へ戻る） 描き直しで値を入れたときは知らせが来ない（actionTriggered は
        # 人の操作だけ）
        self._seek.actionTriggered.connect(self._key_seek)
        self._time = QLabel("0:00 / 0:00", self)
        self._volume = QSlider(Qt.Orientation.Horizontal, self)
        self._volume.setRange(0, 100)
        self._volume.setValue(100)
        self._volume.setMaximumWidth(90)
        self._volume.setToolTip("下見の音量 100 は作品の再生と同じ大きさ")
        self._volume.valueChanged.connect(lambda value: self._player.set_volume(value / 100))

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.play_button)
        controls.addWidget(self._seek, 1)
        controls.addWidget(self._time)
        controls.addWidget(QLabel("音量", self))
        controls.addWidget(self._volume)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._title)
        layout.addWidget(self._picture, 1)
        layout.addWidget(self._wave, 1)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._set_enabled(False)

    # --- 外から ---

    @property
    def player(self) -> PreviewPlayer:
        return self._player

    def set_volume(self, percent: int) -> None:
        self._volume.setValue(max(0, min(100, percent)))
        self._player.set_volume(self._volume.value() / 100)

    def show_file(self, path: Path | None, peaks: np.ndarray | None = None) -> None:
        """下見に出す 同じファイルなら位置を保つ 再生はしない（ダブルクリックか〔再生〕）

        素材を調べるのは裏のスレッド 初めて選んだ動画は開いて頭のコマを読むので、遅い
        ドライブでは画面のスレッドで待つと次の素材を選べない 調べ終えたときにまだ同じ
        ファイルを選んでいれば下見に出す
        """
        if path is not None and path in (self._player.path, self._wanted):
            return
        self._player.close()
        self._shown_frame = -1
        self._picture.set_tile(None)
        self._wave.set_peaks(None)
        self._play_when_open = False
        self._wanted = path
        self._peaks_for_wanted = peaks
        if path is None:
            self._title.setText("ファイルを選ぶとここで下見できます")
            self._set_enabled(False)
            self._show_state()
            self._timer.stop()
            return
        self._title.setText(f"{path.name}（調べています）")
        self._set_enabled(False)
        threading.Thread(
            target=lambda: self._probed.emit(path, probe(path)),
            name="explorer-probe",
            daemon=True,
        ).start()

    @property
    def opening(self) -> bool:
        """調べている途中か"""
        return self._wanted is not None

    def _on_probed(self, path: Path, item: object) -> None:
        if path != self._wanted:
            # 調べている間に別のファイルを選んだ 古い結果は使わない
            return
        self._wanted = None
        peaks = self._peaks_for_wanted
        self._title.setText(path.name)
        if not isinstance(item, MediaItem):
            # 開けなかった 開き直すと画面のスレッドでまた待つので、裏で調べた訳をそのまま出す
            self._play_when_open = False
            self.message.emit(str(item) if item else "下見できません")
            self._set_enabled(False)
            return
        info = self._player.open(path, item)
        if info is None:
            self._play_when_open = False
            self.message.emit(self._player.error or "下見できません")
            self._set_enabled(False)
            return
        sound_only = info.has_audio and not info.has_video and not info.is_still
        self._wave.setVisible(sound_only)
        self._picture.setVisible(not sound_only)
        if sound_only:
            self._wave.set_peaks(peaks)
        self._set_enabled(not info.is_still)
        self._timer.start()
        self._show_state()
        if self._play_when_open:
            self._play_when_open = False
            self.play()

    def set_peaks(self, path: Path, peaks: np.ndarray | None) -> None:
        if path == self._player.path:
            self._wave.set_peaks(peaks)
        elif path == self._wanted:
            self._peaks_for_wanted = peaks

    def toggle(self) -> None:
        if self._player.info is None:
            # 調べている途中に〔再生〕やダブルクリックが来たら、開き終えてから鳴らす
            if self._wanted is not None:
                self._play_when_open = True
            return
        if not self._player.playing and self._before_play is not None:
            self._before_play()
        self._player.toggle()
        self._timer.start()
        self._show_state()

    def play(self) -> None:
        if self._wanted is not None:
            self._play_when_open = True
            return
        if not self._player.playing:
            self.toggle()

    def stop(self) -> None:
        self._player.pause()
        self._show_state()

    def close_player(self) -> None:
        self._timer.stop()
        self._player.close()

    # --- 中 ---

    def _set_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self._seek.setEnabled(enabled)

    def _press_seek(self) -> None:
        self._seeking = True

    def _move_seek(self, value: int) -> None:
        info = self._player.info
        if info is not None:
            self._time.setText(f"{_clock(info.duration * value / 1000)} / {_clock(info.duration)}")

    def _release_seek(self) -> None:
        self._seeking = False
        self._seek_fraction(self._seek.value() / 1000)

    def _key_seek(self, action: int) -> None:
        if self._seeking or action == QSlider.SliderAction.SliderMove.value:
            # ドラッグの途中は離したときにシークする
            return
        # 知らせは値が変わる前に来る 変わった後の値で飛ぶ
        QTimer.singleShot(0, lambda: self._seek_fraction(self._seek.value() / 1000))

    def _seek_fraction(self, fraction: float) -> None:
        info = self._player.info
        if info is not None:
            self._player.seek(info.duration * fraction)
            self._timer.start()
            self._show_state()

    def _tick(self) -> None:
        self._show_state()
        error = self._player.error
        if error is not None and error != self._last_error:
            self.message.emit(error)
        self._last_error = error
        player = self._player
        info = player.info
        if info is None:
            # 開いていない（選んでいない・調べている途中） 描き直す物が無い
            self._timer.stop()
            return
        if player.playing:
            return
        frame = player.frame()
        waiting = (info.has_video or info.is_still) and (
            frame is None or frame[0] != self._shown_frame
        )
        if not waiting:
            # 止まっていて絵も出し終えた（音だけの素材は絵を待たない） 何もしないのに
            # 30 回/秒起きないよう、描き直しを止める
            self._timer.stop()

    def _show_state(self) -> None:
        player = self._player
        info = player.info
        frame = player.frame()
        if frame is not None and frame[0] != self._shown_frame:
            self._shown_frame = frame[0]
            self._picture.set_tile(frame[1])
        playing = player.playing
        self.play_button.setText("一時停止" if playing else "再生")
        if info is None:
            self._time.setText("0:00 / 0:00")
            self._seek.setValue(0)
            self._wave.set_fraction(0)
            return
        position = player.position
        fraction = position / info.duration if info.duration > 0 else 0.0
        if not self._seeking:
            self._seek.setValue(int(fraction * 1000))
            self._time.setText(f"{_clock(position)} / {_clock(info.duration)}")
        self._wave.set_fraction(fraction)


# --- パネル ---


class ExplorerPanel(QWidget):
    """登録したフォルダの木・中身の一覧かアイコン・下見のプレーヤー・タイムラインへ置く"""

    #: 登録を変えた 引数は新しい登録の一覧（``tuple[LibraryFolder, ...]``） 覚えるのは窓の仕事
    folders_changed = Signal(object)
    #: ファイルをタイムラインへ置くよう頼まれた 引数は場所の一覧
    place_requested = Signal(list)
    #: ファイルを素材一覧へだけ読み込むよう頼まれた 引数は場所の一覧
    import_requested = Signal(list)
    #: 表示（一覧・アイコン）を本人が切り替えた 覚えるのは窓の仕事
    view_mode_changed = Signal(str)
    #: アイコンの大きさを本人が変えた
    icon_size_changed = Signal(int)
    #: 知らせ（下見できない など） 窓がステータスバーへ出す
    message = Signal(str)
    #: お気に入りを変えた 引数は新しい一覧（``tuple[str, ...]``） 覚えるのは窓の仕事
    favorites_changed = Signal(object)
    #: 開いている場所か選んだファイルが変わった（開いたフォルダ, 選んだファイル 無ければ空）
    #: 窓が覚え、次の起動でそこから開く
    location_changed = Signal(str, str)

    def __init__(
        self,
        folders: Sequence[LibraryFolder] = (),
        parent: QWidget | None = None,
        *,
        book: MeasureBook | None = None,
        player: PreviewPlayer | None = None,
        before_play: Callable[[], None] | None = None,
        thumbs: Path | None = None,
        measure: Callable[[Path], Measure | None] | None = measure_library_file,
    ) -> None:
        """``player`` は下見の口 渡さなければ本物（音は機械の出口から鳴る）
        試験は鳴らさない物を渡す ``before_play`` は下見を鳴らす前に呼ぶ（窓が作品の再生を
        止める 2 つの音が重なると聞き分けられない）
        ``thumbs`` は見本の絵と波形を書く置き場 ``None`` なら覚えない ``measure`` は裏で測る口
        """
        super().__init__(parent)
        self._folders: tuple[LibraryFolder, ...] = tuple(folders)
        self._favorites: tuple[str, ...] = ()
        self._book = book if book is not None else MeasureBook()
        self._measure = measure
        self._thumbs = _ThumbStore(thumbs)
        self._view_mode = VIEW_LIST
        self._icon_size = 128
        self._double_click = DOUBLE_CLICK_PREVIEW
        #: 今の場所の行のファイル 印で引く（裏の知らせが来たときに行を探す）
        self._files: dict[str, LibraryFile] = {}
        #: 作った見本の絵 印で引く
        self._icons: dict[str, QIcon] = {}
        self._peaks: dict[str, np.ndarray] = {}
        self._audio_icon = audio_icon()
        self._pending_icon = pending_icon()

        self._worker = _Worker(self)
        self._worker.finished.connect(self._on_worker)
        # 中身を読むのは別の 1 本 測る・見本を作る仕事の後ろに並ぶと、開いた場所が出るまで待つ
        self._lister = _Worker(self, workers=1)
        self._lister.finished.connect(self._on_listed)
        self._listing_number = 0
        self._listed_number = 0
        #: 前回の場所から開いたときに、読み終えたら選ぶファイル
        self._pending_select: list[Path] = []

        # --- 左 木 ---
        register = QPushButton("登録…", self)
        register.setToolTip("フォルダを分類（効果音・BGM・動画・画像など）を付けて登録する")
        register.clicked.connect(self._register)
        self._remove_button = QPushButton("外す", self)
        self._remove_button.setToolTip("選んだ登録したフォルダの登録を外す ファイルは消さない")
        self._remove_button.clicked.connect(self._remove)
        rescan = QPushButton("読み直す", self)
        rescan.setToolTip("フォルダの中を見直す 後から足したファイルを一覧へ出す")
        rescan.clicked.connect(self.rescan)
        tree_buttons = FlowLayout()
        tree_buttons.addWidget(register)
        tree_buttons.addWidget(self._remove_button)
        tree_buttons.addWidget(rescan)
        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.itemExpanded.connect(self._expand)
        self._tree.currentItemChanged.connect(lambda *_args: self._show_files())
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_tree_menu)
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(tree_buttons)
        left_layout.addWidget(self._tree, 1)

        # --- 右の上 道具の帯と中身 ---
        self._crumbs = QLabel(self)
        self._crumbs.setTextFormat(Qt.TextFormat.PlainText)
        self._filter = QLineEdit(self)
        self._filter.setPlaceholderText("名前で絞り込む（下のフォルダも探す）")
        self._filter.textChanged.connect(self._show_files)
        self._kind_box = QComboBox(self)
        for key, name in _KIND_FILTERS:
            self._kind_box.addItem(name, key)
        narrow_combo(self._kind_box, 4)
        self._kind_box.setToolTip("種類で絞り込む")
        self._kind_box.currentIndexChanged.connect(self._show_files)
        self._view_buttons = QButtonGroup(self)
        self._view_buttons.setExclusive(True)
        view_row = FlowLayout()
        view_row.addWidget(self._kind_box)
        for mode, text, tip in (
            (VIEW_LIST, "一覧", "名前・長さ・大きさなどを列に並べる 見出しで並べ替える"),
            (VIEW_ICONS, "アイコン", "見本の絵と波形を大きく並べる"),
        ):
            button = QToolButton(self)
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked(mode == self._view_mode)
            button.setProperty("view_mode", mode)
            # 押したときだけ知らせる 窓が覚えた表示を当てたとき（set_view_mode）まで知らせると、
            # 起動のたびに好みの設定を書き直す
            button.clicked.connect(lambda _checked=False, chosen=mode: self._choose_view(chosen))
            self._view_buttons.addButton(button)
            view_row.addWidget(button)
        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setRange(*ICON_SIZE_RANGE)
        self._size_slider.setValue(self._icon_size)
        self._size_slider.setMaximumWidth(120)
        self._size_slider.setToolTip("アイコンの大きさ")
        self._size_slider.valueChanged.connect(self._resize_icons)
        self._size_slider.sliderReleased.connect(
            lambda: self.icon_size_changed.emit(self._icon_size)
        )
        view_row.addWidget(self._size_slider)

        self._model = _FileModel(0, len(COLUMNS), self)
        self._model.setHorizontalHeaderLabels(list(COLUMNS))
        self._proxy = _SortProxy(self)
        self._proxy.setSourceModel(self._model)
        self._table = QTableView(self)
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(_NAME, Qt.SortOrder.AscendingOrder)
        self._table.verticalHeader().hide()
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        # 印の列は 1 字なので細くする 既定の幅だと名前の列が右へ押し出される
        self._table.setColumnWidth(_STAR, 32)
        self._icons_view = QListView(self)
        self._icons_view.setModel(self._proxy)
        self._icons_view.setModelColumn(_NAME)
        self._icons_view.setViewMode(QListView.ViewMode.IconMode)
        self._icons_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._icons_view.setMovement(QListView.Movement.Static)
        self._icons_view.setWordWrap(True)
        self._icons_view.setSpacing(6)
        self._icons_view.setItemDelegate(_UnderDelegate(self._icons_view))
        for view in (self._table, self._icons_view):
            view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            view.setDragEnabled(True)
            view.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
            view.setDefaultDropAction(Qt.DropAction.CopyAction)
            view.doubleClicked.connect(self._on_double_click)
            view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            view.customContextMenuRequested.connect(
                lambda position, shown=view: self._show_menu(shown, position)
            )
        # 2 つの表示で選んだ行をそろえる（切り替えても同じファイルを選んだまま）
        self._icons_view.setSelectionModel(self._table.selectionModel())
        self._table.selectionModel().selectionChanged.connect(lambda *_a: self._on_selection())
        self._table.clicked.connect(self._on_cell_clicked)
        self._views = QStackedWidget(self)
        self._views.addWidget(self._table)
        self._views.addWidget(self._icons_view)
        self._status = QLabel(self)
        self._status.setWordWrap(True)

        browse = QWidget(self)
        browse_layout = QVBoxLayout(browse)
        browse_layout.setContentsMargins(0, 0, 0, 0)
        browse_layout.setSpacing(4)
        browse_layout.addWidget(self._crumbs)
        browse_layout.addWidget(self._filter)
        browse_layout.addLayout(view_row)
        browse_layout.addWidget(self._views, 1)
        browse_layout.addWidget(self._status)

        # --- 右の下 下見 ---
        self._preview = _PreviewPane(
            player if player is not None else PreviewPlayer(), before_play, self
        )
        self._preview.message.connect(self.message)
        self._place_button = QPushButton("タイムラインへ", self)
        self._place_button.setToolTip(
            "選んだファイルを読み込んで再生位置へ置く 取り消し 1 回で戻る"
        )
        self._place_button.clicked.connect(self._place_selected)
        bottom = QWidget(self)
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self._preview, 1)
        bottom_layout.addWidget(self._place_button)

        self._right = QSplitter(Qt.Orientation.Vertical, self)
        self._right.addWidget(browse)
        self._right.addWidget(bottom)
        self._right.setStretchFactor(0, 3)
        self._right.setStretchFactor(1, 2)
        # 下見を初めから広く取る 伸び縮みの比だけだと、初めて出したときに中身の最小の高さが
        # 優先され、下見の絵が小さく出る
        self._right.setSizes([380, 320])
        self._outer = QSplitter(Qt.Orientation.Horizontal, self)
        self._outer.addWidget(left)
        self._outer.addWidget(self._right)
        self._outer.setStretchFactor(0, 1)
        self._outer.setStretchFactor(1, 3)
        # 木は畳める（狭いパネルで中身を広く使う）
        self._outer.setChildrenCollapsible(True)
        # 中身は巻物に入れる 木・道具の帯・中身・下見を縦に積むと最小の高さが大きく、素材一覧と
        # 重ねた左の列がそのまま窓の最小の高さになり、1280x720 のノート PC で窓が画面に収まらない
        # 低い窓では巻物で見る（設定の窓と同じ考え方）
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._outer)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(scroll)

        # Space は焦点がここにあるときだけ下見の再生と停止にする 作品の再生の Space（窓の
        # ショートカット）と取り合わないよう、焦点のある部品でショートカットより先に受け取る
        # 名前の絞り込みの欄では空白を打てるように、ここには付けない
        for widget in (
            self._tree,
            self._table,
            self._icons_view,
            self._preview,
            self._preview.play_button,
        ):
            widget.installEventFilter(self)

        self._fill_tree()
        self._apply_view_mode()
        self._on_selection()

    # --- 外から ---

    @property
    def folders(self) -> tuple[LibraryFolder, ...]:
        return self._folders

    @property
    def view_mode(self) -> str:
        return self._view_mode

    @property
    def icon_size(self) -> int:
        return self._icon_size

    @property
    def preview(self) -> _PreviewPane:
        return self._preview

    def set_folders(self, folders: Sequence[LibraryFolder]) -> None:
        """登録を差し替える（設定画面で変えたとき） 知らせは出さない"""
        folders = tuple(folders)
        if folders == self._folders:
            return
        self._folders = folders
        self._fill_tree()

    @property
    def favorites(self) -> tuple[str, ...]:
        return self._favorites

    def set_favorites(self, favorites: Sequence[str]) -> None:
        """お気に入りを差し替える（起動したとき） 知らせは出さない"""
        favorites = tuple(favorites)
        if favorites == self._favorites:
            return
        self._favorites = favorites
        self._fill_tree()

    def set_favorite(self, paths: Sequence[Path | str], wanted: bool) -> None:
        """お気に入りに入れる（``wanted`` が偽なら外す） 窓へ知らせて覚えさせる"""
        favorites = self._favorites
        for path in paths:
            favorites = toggle_favorite(favorites, path, wanted)
        if favorites == self._favorites:
            return
        self._favorites = favorites
        self._fill_tree()
        self.favorites_changed.emit(self._favorites)

    def is_favorite(self, path: Path | str) -> bool:
        return any(same_place(entry, path) for entry in self._favorites)

    def restore_location(self, folder: str, file: str) -> bool:
        """前回の場所から開く そのフォルダが消えていれば何もしない（先頭から開く）

        選んでいたファイルは中身を読み終えてから選ぶ（中身は裏で読む）
        """
        if not folder or not reachable_dir(Path(folder))[0]:
            return False
        self._pending_select = [Path(file)] if file else []
        if not self.open_folder(folder):
            self._pending_select = []
            return False
        return True

    def set_preview_volume(self, percent: int) -> None:
        self._preview.set_volume(percent)

    def set_double_click(self, action: str) -> None:
        self._double_click = action if action in DOUBLE_CLICK_ACTIONS else DOUBLE_CLICK_PREVIEW

    def set_view_mode(self, mode: str) -> None:
        """表示を切り替える 知らない値は一覧にする（設定ファイルを手で書いた人がいる）"""
        mode = mode if mode in VIEW_MODES else VIEW_LIST
        for button in self._view_buttons.buttons():
            button.setChecked(button.property("view_mode") == mode)
        if mode != self._view_mode:
            self._view_mode = mode
            self._apply_view_mode()

    def set_icon_size(self, size: int) -> None:
        size = max(ICON_SIZE_RANGE[0], min(ICON_SIZE_RANGE[1], size))
        self._size_slider.blockSignals(True)
        self._size_slider.setValue(size)
        self._size_slider.blockSignals(False)
        self._resize_icons(size)

    def rescan(self) -> None:
        """見て回った結果を捨てて見直す"""
        self._fill_tree()

    def shutdown(self) -> None:
        """窓を閉じる 下見を止め、裏の仕事を捨てる"""
        self._preview.close_player()
        self._worker.close()
        self._lister.close()

    def visible_files(self) -> list[Path]:
        """中身に出ている行のファイル（今の並びで）"""
        return [
            Path(str(self._proxy.index(row, _NAME).data(_PATH)))
            for row in range(self._proxy.rowCount())
        ]

    def cell(self, path: Path, column: int) -> str:
        """その行のその列に出ている文字（試験と写真で使う）"""
        for row in range(self._model.rowCount()):
            if Path(str(self._model.item(row, _NAME).data(_PATH))) == path:
                return str(self._model.item(row, column).text())
        return ""

    def select_file(self, path: Path) -> bool:
        return self.select_files([path]) == 1

    def select_files(self, paths: Sequence[Path]) -> int:
        """中身に出ている行のうち ``paths`` の物をまとめて選ぶ 選べた数を返す"""
        wanted = set(paths)
        model = self._table.selectionModel()
        chosen = QItemSelection()
        first: QModelIndex | None = None
        count = 0
        for row in range(self._proxy.rowCount()):
            index = self._proxy.index(row, _NAME)
            if Path(str(index.data(_PATH))) in wanted:
                chosen.select(index, self._proxy.index(row, len(COLUMNS) - 1))
                first = first or index
                count += 1
        flags = (
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows
        )
        model.blockSignals(True)
        model.select(chosen, flags)
        if first is not None:
            model.setCurrentIndex(first, QItemSelectionModel.SelectionFlag.NoUpdate)
        model.blockSignals(False)
        # 知らせを止めてまとめて選んだので、下見と〔タイムラインへ〕は最後に 1 回だけ直す
        self._table.viewport().update()
        self._icons_view.viewport().update()
        self._on_selection()
        return count

    def selected_files(self) -> list[Path]:
        rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        return [Path(str(self._proxy.index(row, _NAME).data(_PATH))) for row in rows]

    def open_folder(self, path: str) -> bool:
        """木で登録したフォルダ（か、その下のフォルダ）を選ぶ"""
        target = os.path.normcase(os.path.normpath(path))
        found = self._find_node(target)
        if found is None:
            return False
        self._tree.setCurrentItem(found)
        return True

    def status_text(self) -> str:
        return self._status.text()

    def location_text(self) -> str:
        return self._crumbs.text()

    def drag_data(self) -> QMimeData:
        """選んだ行をドラッグしたときの中身（試験用 本物のドラッグは OS の窓を待つ）"""
        rows = self._table.selectionModel().selectedRows()
        return self._model.mimeData([self._proxy.mapToSource(index) for index in rows])

    def build_menu(self, paths: Sequence[Path]) -> QMenu:
        """行の右クリックメニュー 表示と中身を分けてあるのは試験のため"""
        menu = QMenu(self)
        chosen = list(paths)
        place = menu.addAction("タイムラインへ置く")
        place.triggered.connect(lambda _checked=False: self.place_requested.emit(chosen))
        pool = menu.addAction("素材一覧へ読み込む")
        pool.triggered.connect(lambda _checked=False: self.import_requested.emit(chosen))
        menu.addSeparator()
        reveal = menu.addAction("エクスプローラーで開く")
        reveal.triggered.connect(lambda _checked=False: self._reveal(chosen))
        self._add_favorite_actions(menu, chosen)
        return menu

    def build_tree_menu(self, path: Path) -> QMenu:
        """木のフォルダの右クリックメニュー お気に入りの付け外し"""
        menu = QMenu(self)
        self._add_favorite_actions(menu, [path])
        return menu

    def _add_favorite_actions(self, menu: QMenu, chosen: list[Path]) -> None:
        """お気に入りに入れる・外す 両方をいつも並べ、今の状態で押せる方だけを押せるようにする
        （出したり消したりすると、同じ所を押したつもりで別の項目を押す）
        """
        menu.addSeparator()
        marked = [self.is_favorite(path) for path in chosen]
        add = menu.addAction("お気に入りに入れる")
        add.setEnabled(not all(marked))
        add.triggered.connect(lambda _checked=False: self.set_favorite(chosen, True))
        drop = menu.addAction("お気に入りから外す")
        drop.setEnabled(any(marked))
        drop.triggered.connect(lambda _checked=False: self.set_favorite(chosen, False))

    # --- 登録と木 ---

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
        self._fill_tree()
        self.open_folder(path)
        self.folders_changed.emit(self._folders)

    def _remove(self) -> None:
        node = self._node()
        registered = node[0] if node is not None and node[0] != _FAVORITE_NODE else None
        if not registered:
            self.message.emit("外すフォルダを左の木で選んでください")
            return
        self._folders = remove_folder(self._folders, registered)
        self._fill_tree()
        self.folders_changed.emit(self._folders)

    def _fill_tree(self) -> None:
        """木を作り直す 選んでいた所は場所で選び直す"""
        current = self._node()
        self._tree.blockSignals(True)
        self._tree.clear()
        if self._favorites:
            # 一番上に置く すぐ呼び出したい素材をまとめる所（2026-11-24 本人の要望）
            root = QTreeWidgetItem([f"{STAR_ON} {FAVORITES}"])
            root.setData(0, _NODE, (_FAVORITE_NODE, ""))
            self._tree.addTopLevelItem(root)
            for entry in self._favorites:
                place = Path(entry)
                if _is_file(place):
                    # ファイルは根を選んだときの中身に出す 木にはフォルダだけを並べる
                    continue
                missing = not reachable_dir(place)[0]
                label = place.name or entry
                item = QTreeWidgetItem([f"{label}（見つかりません）" if missing else label])
                item.setToolTip(0, entry)
                item.setData(0, _NODE, (_FAVORITE_NODE, entry))
                if missing:
                    item.setForeground(0, Colors.TEXT_MUTED)
                else:
                    item.addChild(QTreeWidgetItem([_PLACEHOLDER]))
                root.addChild(item)
            root.setExpanded(True)
        for key, name in CATEGORIES.items():
            members = [folder for folder in self._folders if folder.category == key]
            if not members:
                continue
            root = QTreeWidgetItem([name])
            root.setData(0, _NODE, ("", key))
            self._tree.addTopLevelItem(root)
            for folder in members:
                directory = Path(folder.path)
                missing = not reachable_dir(directory)[0]
                label = directory.name or folder.path
                item = QTreeWidgetItem([f"{label}（見つかりません）" if missing else label])
                item.setToolTip(0, folder.path)
                item.setData(0, _NODE, (folder.path, folder.path))
                if missing:
                    item.setForeground(0, Colors.TEXT_MUTED)
                else:
                    item.addChild(QTreeWidgetItem([_PLACEHOLDER]))
                root.addChild(item)
            root.setExpanded(True)
        self._tree.blockSignals(False)
        if current is not None and current[1]:
            found = self._find_node(os.path.normcase(os.path.normpath(current[1])))
            if found is not None:
                self._tree.setCurrentItem(found)
        self._show_files()

    def _find_node(self, target: str) -> QTreeWidgetItem | None:
        """場所の木の項目 まだ開いていない下のフォルダは、辿りながら開く

        登録したフォルダの下を先に探す お気に入りのフォルダは登録したフォルダの中にもあるので、
        先に探すと前回の場所がいつもお気に入りの下で開く
        """
        roots = [self._tree.topLevelItem(top) for top in range(self._tree.topLevelItemCount())]
        favorites = [r for r in roots if r is not None and r.data(0, _NODE) == (_FAVORITE_NODE, "")]
        for root in [r for r in roots if r not in favorites] + favorites:
            if root is None:
                continue
            for index in range(root.childCount()):
                child = root.child(index)
                found = self._descend(child, target) if child is not None else None
                if found is not None:
                    return found
        return None

    def _descend(self, item: QTreeWidgetItem, target: str) -> QTreeWidgetItem | None:
        node = item.data(0, _NODE)
        if not isinstance(node, tuple):
            return None
        here = os.path.normcase(os.path.normpath(node[1]))
        if here == target:
            return item
        if not target.startswith(here.rstrip(os.sep) + os.sep):
            return None
        self._expand(item)
        item.setExpanded(True)
        for index in range(item.childCount()):
            child = item.child(index)
            found = self._descend(child, target) if child is not None else None
            if found is not None:
                return found
        return None

    def _expand(self, item: QTreeWidgetItem) -> None:
        """開いたときに下のフォルダを 1 段だけ読む"""
        first = item.child(0) if item.childCount() == 1 else None
        if first is None or first.text(0) != _PLACEHOLDER:
            return
        item.takeChild(0)
        node = item.data(0, _NODE)
        if not isinstance(node, tuple):
            return
        folder = (
            self._owner(Path(node[1])) if node[0] == _FAVORITE_NODE else self._registered(node[0])
        )
        if folder is None:
            return
        listing = list_folder(folder, Path(node[1]), limit=0)
        for directory in listing.folders:
            child = QTreeWidgetItem([directory.name])
            child.setToolTip(0, str(directory))
            child.setData(0, _NODE, (node[0], str(directory)))
            child.addChild(QTreeWidgetItem([_PLACEHOLDER]))
            item.addChild(child)

    def _registered(self, path: str) -> LibraryFolder | None:
        return next((folder for folder in self._folders if folder.path == path), None)

    def _owner(self, path: Path) -> LibraryFolder:
        """お気に入りの物が入っている登録したフォルダ 登録の外（登録を外した後など）なら、
        その場所を「その他」として扱う（お気に入りは登録と別に覚えるので、外しても残る）
        """
        for folder in self._folders:
            if is_favorite([folder.path], path):
                return folder
        return LibraryFolder(str(path if reachable_dir(path)[0] else path.parent), CATEGORY_OTHER)

    def _show_tree_menu(self, position: QPoint) -> None:
        item = self._tree.itemAt(position)
        node = item.data(0, _NODE) if item is not None else None
        if not isinstance(node, tuple) or not node[0] or not node[1]:
            return
        menu = self.build_tree_menu(Path(node[1]))
        menu.exec(self._tree.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _node(self) -> tuple[str, str] | None:
        """選んでいる木の項目 （登録したフォルダの場所, 開いた場所） 分類の根は（空, 分類）"""
        item = self._tree.currentItem()
        node = item.data(0, _NODE) if item is not None else None
        return node if isinstance(node, tuple) else None

    # --- 中身 ---

    def _location(self) -> tuple[list[tuple[LibraryFolder, Path]], list[str]]:
        """今の場所で見る（登録したフォルダ, その中の場所）の組と、今いる場所の名前の並び"""
        node = self._node()
        if node is None:
            return [(folder, Path(folder.path)) for folder in self._folders], ["すべてのフォルダ"]
        registered, where = node
        if registered == _FAVORITE_NODE:
            if not where:
                return [], [FAVORITES]
            return [(self._owner(Path(where)), Path(where))], [FAVORITES, Path(where).name]
        if not registered:
            members = [folder for folder in self._folders if folder.category == where]
            return [(folder, Path(folder.path)) for folder in members], [CATEGORIES[where]]
        folder = self._registered(registered)
        if folder is None:
            return [], []
        directory = Path(where)
        crumbs = [folder.category_name, Path(folder.path).name or folder.path]
        with contextlib.suppress(ValueError):
            crumbs.extend(directory.relative_to(folder.path).parts)
        return [(folder, directory)], crumbs

    def _show_files(self) -> None:
        """今の場所の中身を裏で読み、読めたら並べる

        見て回るのは裏のスレッド 名前の欄を 1 文字変えるたびに下まで探すので、大きな登録先や
        応答の遅い外付けのドライブでは、画面のスレッドで待つと次の字を打てない 読み終えたときに
        場所と絞り込みが変わっていれば（頼みの番号が違えば）捨てる
        """
        self._worker.forget()
        pairs, crumbs = self._location()
        node = self._node()
        self._remove_button.setEnabled(
            node is not None and bool(node[0]) and node[0] != _FAVORITE_NODE
        )
        self._crumbs.setText(" › ".join(crumbs))
        query = self._filter.text().strip()
        kind = str(self._kind_box.currentData() or "")
        favorite_place = node is not None and node[0] == _FAVORITE_NODE
        if not self._folders and not self._favorites:
            self._listing_number += 1
            self._listed_number = self._listing_number
            self._fill_rows([])
            self._status.setText(
                "フォルダが登録されていません 〔登録…〕で効果音や BGM のフォルダを登録すると、"
                "ここで探して下見し、置けます AI に使わせる分類は 表示 → 設定… で選びます"
            )
            return
        self._listing_number += 1
        number = self._listing_number
        if node == (_FAVORITE_NODE, ""):
            job: Callable[[], object] = partial(
                _favorite_listing, self._favorites, self._owner, query, kind
            )
        else:
            deep = bool(query) if favorite_place else (bool(query) or node is None or not node[0])
            job = partial(_listing, pairs, deep, query, kind)
        self._lister.forget()
        self._lister.submit(f"list:{number}", job)

    @property
    def listing(self) -> bool:
        """中身を裏で読んでいる途中か（読み終えて並べるまで）"""
        return self._listed_number != self._listing_number

    def _on_listed(self, key: str, result: object) -> None:
        if key != f"list:{self._listing_number}":
            # 読んでいる間に別の場所を開いたか、絞り込みを変えた 古い結果は並べない
            return
        self._listed_number = self._listing_number
        if not isinstance(result, tuple):
            self._fill_rows([])
            self._status.setText("この場所を読めません")
            return
        files, status = result
        self._fill_rows(files)
        self._status.setText(status)
        self._emit_location()

    def _fill_rows(self, files: Sequence[LibraryFile]) -> None:
        selected = self.selected_files()
        if self._pending_select:
            # 前回の場所から開いたときの選び 中身を読み終えてから選ぶ
            selected, self._pending_select = list(self._pending_select), []
        self._table.setSortingEnabled(False)
        self._model.removeRows(0, self._model.rowCount())
        self._files = {}
        unmeasured: list[LibraryFile] = []
        for file in files:
            self._files[file.id] = file
            measure = self._book.get(file)
            if measure is None:
                unmeasured.append(file)
            self._model.appendRow(self._row(file, measure))
        self._table.setSortingEnabled(True)
        self._table.resizeColumnToContents(_NAME)
        # 1 つずつ選ぶと前の選びが外れ、最後の 1 つしか残らない まとめて選び直す
        self.select_files(selected)
        measurer = self._measure
        if measurer is not None:
            for file in unmeasured[:_MEASURE_LIMIT]:
                self._worker.submit(f"measure:{file.id}", _measured(measurer, file))
        if self._view_mode == VIEW_ICONS:
            self._request_icons()
        self._on_selection()

    def _row(self, file: LibraryFile, measure: Measure | None) -> list[QStandardItem]:
        kind = kind_of(file.path)
        cells = [QStandardItem() for _ in COLUMNS]
        favorite = self.is_favorite(file.path)
        cells[_STAR].setText(STAR_ON if favorite else STAR_OFF)
        cells[_STAR].setData(int(favorite), _SORT)
        cells[_STAR].setToolTip("押すとお気に入りに入れる・外す")
        cells[_STAR].setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        cells[_NAME].setData(favorite, _FAV)
        cells[_NAME].setText(file.name)
        cells[_NAME].setData(str(file.path), _PATH)
        cells[_NAME].setData(file.id, _ID)
        cells[_NAME].setData(file.name.casefold(), _SORT)
        cells[_NAME].setToolTip(str(file.path))
        cells[_KIND].setText(_KIND_NAMES[kind])
        cells[_KIND].setData(kind, _SORT)
        stamp = datetime.fromtimestamp(file.modified_ns / 1e9)
        cells[_MODIFIED].setText(stamp.strftime("%Y-%m-%d %H:%M"))
        cells[_MODIFIED].setData(file.modified_ns, _SORT)
        parent = str(Path(file.relative).parent)
        cells[_FOLDER].setText("" if parent == "." else parent)
        cells[_FOLDER].setData(parent.casefold(), _SORT)
        for cell in cells:
            cell.setEditable(False)
            cell.setDropEnabled(False)
        if kind == KIND_SOUND:
            cells[_NAME].setIcon(self._icons.get(_stamp(file), self._audio_icon))
        else:
            cells[_NAME].setIcon(self._icons.get(_stamp(file), self._pending_icon))
        self._fill_measure(cells, measure)
        return cells

    def _fill_measure(self, cells: Sequence[QStandardItem], measure: Measure | None) -> None:
        """測った値を列へ入れる 測っていなければ空（裏で測れたら埋める）"""
        if measure is None:
            return
        if measure.seconds > 0:
            cells[_LENGTH].setText(
                _clock(measure.seconds) if measure.seconds >= 60 else f"{measure.seconds:.1f} 秒"
            )
            cells[_LENGTH].setData(measure.seconds, _SORT)
            cells[_NAME].setData(cells[_LENGTH].text(), _UNDER)
        if measure.loudness is not None:
            cells[_LOUDNESS].setText(f"{measure.loudness:.1f} LUFS")
            cells[_LOUDNESS].setData(measure.loudness, _SORT)
        if measure.width and measure.height:
            cells[_SIZE].setText(f"{measure.width}x{measure.height}")
            cells[_SIZE].setData(measure.width * measure.height, _SORT)
        if measure.fps:
            cells[_FPS].setText(f"{measure.fps:g}")
            cells[_FPS].setData(measure.fps, _SORT)

    def _row_of(self, key: str) -> list[QStandardItem] | None:
        for row in range(self._model.rowCount()):
            if self._model.item(row, _NAME).data(_ID) == key:
                return [self._model.item(row, column) for column in range(len(COLUMNS))]
        return None

    def _on_worker(self, key: str, result: object) -> None:
        kind, _, file_key = key.partition(":")
        file = self._files.get(file_key)
        if file is None:
            # 別の場所を開いた後に届いた 測った物だけは控えに入れる（次に開いたときに使う）
            return
        if kind == "measure" and isinstance(result, tuple):
            asked, measure = result
            # 頼んだときのファイルと今の行のファイルが同じ（大きさと更新時刻まで）ときだけ使う
            # 差し替わった後の行に前のファイルの長さを出さない
            if asked != file or not isinstance(measure, Measure):
                return
            self._book.put(asked, measure)
            cells = self._row_of(file_key)
            if cells is not None:
                self._fill_measure(cells, measure)
        elif kind == "still" and isinstance(result, np.ndarray):
            self._icons[_stamp(file)] = thumbnail_icon(result)
            self._set_icon(file)
        elif kind == "peaks" and isinstance(result, np.ndarray):
            self._peaks[_stamp(file)] = result
            self._icons[_stamp(file)] = _peaks_icon(result)
            self._set_icon(file)
            self._preview.set_peaks(file.path, result)

    def _set_icon(self, file: LibraryFile) -> None:
        cells = self._row_of(file.id)
        if cells is not None:
            cells[_NAME].setIcon(self._icons[_stamp(file)])

    def _request_icons(self) -> None:
        """アイコン表示の見本の絵と波形を頼む 控えにあればそれを使う"""
        for file in self._files.values():
            # 同じ場所でも中身を差し替えた素材は作り直す（大きさと更新時刻まで見る）
            if _stamp(file) in self._icons:
                continue
            self._request_picture(file)

    def _request_picture(self, file: LibraryFile) -> None:
        kind = kind_of(file.path)
        store = self._thumbs
        if kind == KIND_SOUND:
            cached = store.load(file, "peaks")
            if cached is not None:
                self._on_worker(f"peaks:{file.id}", cached)
                return

            def peaks(target: LibraryFile = file) -> np.ndarray | None:
                made = sound_peaks(target.path, _PEAK_COLUMNS)
                if made is not None:
                    store.save(target, "peaks", made)
                return made

            self._worker.submit(f"peaks:{file.id}", peaks)
            return
        cached = store.load(file, "still")
        if cached is not None:
            self._on_worker(f"still:{file.id}", cached)
            return

        def still(target: LibraryFile = file) -> np.ndarray | None:
            tile = still_of(target.path)
            if tile is None:
                return None
            small = _small_tile(tile)
            store.save(target, "still", small)
            return small

        self._worker.submit(f"still:{file.id}", still)

    # --- 表示 ---

    def _choose_view(self, mode: str) -> None:
        if mode == self._view_mode:
            return
        self.set_view_mode(mode)
        self.view_mode_changed.emit(mode)

    def _apply_view_mode(self) -> None:
        icons = self._view_mode == VIEW_ICONS
        self._views.setCurrentWidget(self._icons_view if icons else self._table)
        self._size_slider.setEnabled(icons)
        self._resize_icons(self._icon_size)
        if icons:
            self._request_icons()

    def _resize_icons(self, size: int) -> None:
        self._icon_size = size
        height = size * 9 // 16
        self._icons_view.setIconSize(QSize(size, height))
        self._icons_view.setGridSize(QSize(size + 24, height + 48))
        self._table.setIconSize(QSize(32, 18))

    # --- 選ぶ・下見・置く ---

    def _on_cell_clicked(self, index: QModelIndex) -> None:
        """印の列を押した お気に入りを切り替える"""
        if index.column() != _STAR:
            return
        path = Path(str(index.siblingAtColumn(_NAME).data(_PATH)))
        self.set_favorite([path], not self.is_favorite(path))

    def _emit_location(self) -> None:
        node = self._node()
        folder = node[1] if node is not None and node[0] and node[1] else ""
        files = self.selected_files()
        self.location_changed.emit(folder, str(files[0]) if len(files) == 1 else "")

    def _on_selection(self) -> None:
        self._emit_location()
        files = self.selected_files()
        self._place_button.setEnabled(bool(files))
        if len(files) != 1:
            if not files:
                self._preview.show_file(None)
            return
        path = files[0]
        file = next((f for f in self._files.values() if f.path == path), None)
        peaks = self._peaks.get(_stamp(file)) if file is not None else None
        self._preview.show_file(path, peaks)
        if file is not None and peaks is None and kind_of(path) == KIND_SOUND:
            # 音の下見に波形を出す 一覧表示でも作る（作った物はアイコンの絵にも使う）
            self._request_picture(file)

    def _on_double_click(self, index: QModelIndex) -> None:
        del index
        if self._double_click == DOUBLE_CLICK_TIMELINE:
            self._place_selected()
            return
        self._on_selection()
        self._preview.play()

    def _place_selected(self) -> None:
        files = self.selected_files()
        if files:
            self.place_requested.emit(files)

    def _show_menu(self, view: QAbstractItemView, position: QPoint) -> None:
        index = view.indexAt(position)
        if not index.isValid():
            return
        # 右クリックした行が選ばれていなければ選び直す 選んでいた別の行が対象になると、
        # 置いたつもりのない物が置かれる
        if not view.selectionModel().isRowSelected(index.row(), QModelIndex()):
            self.select_file(Path(str(index.siblingAtColumn(_NAME).data(_PATH))))
        menu = self.build_menu(self.selected_files())
        menu.exec(view.viewport().mapToGlobal(position))
        # 右クリックのたびに作るメニュー 捨てないとパネルの子として残り続ける
        menu.deleteLater()

    def _reveal(self, paths: Sequence[Path]) -> None:
        if paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths[0].parent)))

    # --- 入力と大きさ ---

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        if (
            event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_Space
            and not event.modifiers()
        ):
            if event.type() == QEvent.Type.ShortcutOverride:
                # 受け取ると窓の Space（作品の再生）へは行かず、下の KeyPress になって届く
                event.accept()
                return True
            self._preview.toggle()
            return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        # 狭いときは木を上・中身を下へ並べ替える 横に並べたままだと、素材一覧と重ねた左の
        # 列の幅で名前の列と下見の操作が欠ける
        wanted = Qt.Orientation.Vertical if self.width() < _NARROW else Qt.Orientation.Horizontal
        if self._outer.orientation() != wanted:
            self._outer.setOrientation(wanted)


def _bound(job: Callable[[Path], object], path: Path) -> Callable[[], object]:
    """裏の仕事へ渡す、場所を決めた呼び出し（繰り返しの中の lambda は最後の場所を掴む）"""
    return lambda: job(path)


def _is_file(path: Path) -> bool:
    """ファイルか 見られない場所（アクセス権を失った）でも投げずに偽"""
    try:
        return path.is_file()
    except OSError:
        return False


def _measured(job: Callable[[Path], Measure | None], file: LibraryFile) -> Callable[[], object]:
    """裏で測る仕事 頼んだときのファイルと組にして返す

    測っている間に同じ名前のファイルが差し替わると、前のファイルの長さを新しいファイルの物
    として覚えてしまう 測り終えたときに大きさと更新時刻が変わっていれば捨てる
    """

    def run() -> object:
        measure = job(file.path)
        try:
            stat = file.path.stat()
        except OSError:
            return None
        if (stat.st_size, stat.st_mtime_ns) != (file.size, file.modified_ns):
            return None
        return (file, measure)

    return run


def _listing(
    pairs: Sequence[tuple[LibraryFolder, Path]], deep: bool, query: str, kind: str
) -> tuple[list[LibraryFile], str]:
    """場所の中身と状態の文（裏のスレッドで呼ぶ）"""
    notes: list[str] = []
    files: list[LibraryFile] = []
    for folder, directory in pairs:
        present, problem = reachable_dir(Path(folder.path))
        if not present:
            notes.append(
                f"{folder.label} が見つかりません（消えたか移った） 戻すか〔外す〕で外せます"
                if not problem
                else f"{folder.label} を{problem}"
            )
            continue
        if deep:
            # 名前で絞るときと分類の根では下まで探す 場所は登録したフォルダのまま数える
            scan = scan_folder(LibraryFolder(str(directory), folder.category))
            if scan.missing:
                notes.append(f"{directory.name} が見つかりません（消えたか移った）")
            if scan.truncated:
                notes.append(f"{folder.label} はファイルが多いので頭の {len(scan.files)} 個だけ")
            files.extend(
                LibraryFile(file.path, folder, file.size, file.modified_ns) for file in scan.files
            )
        else:
            listed = list_folder(folder, directory)
            if listed.missing:
                # 開いていた下のフォルダが消えた 空のフォルダと見分けが付くように知らせる
                notes.append(f"{directory.name} が見つかりません（消えたか移った）")
            files.extend(listed.files)
    matched = [file for file in match_files(files, query) if not kind or kind_of(file.path) == kind]
    count = f"{len(matched)} 個"
    if len(matched) > DISPLAY_LIMIT:
        count += f"（頭の {DISPLAY_LIMIT} 個を表示 名前で絞り込んでください）"
    return matched[:DISPLAY_LIMIT], "  ".join([count, *notes])


def _favorite_listing(
    favorites: Sequence[str],
    owner: Callable[[Path], LibraryFolder],
    query: str,
    kind: str,
) -> tuple[list[LibraryFile], str]:
    """お気に入りの根の中身 印を付けたファイル（フォルダは木から辿る）"""
    files: list[LibraryFile] = []
    missing = 0
    for entry in favorites:
        place = Path(entry)
        if reachable_dir(place)[0]:
            continue
        try:
            stat = place.stat()
        except OSError:
            # 消えた・移った物も外さずに残す 数だけ知らせる
            missing += 1
            continue
        files.append(LibraryFile(place, owner(place), stat.st_size, stat.st_mtime_ns))
    matched = [file for file in match_files(files, query) if not kind or kind_of(file.path) == kind]
    if not favorites:
        return [], "お気に入りはありません"
    notes = [f"{len(matched)} 個"]
    if missing:
        notes.append(f"{missing} 個が見つかりません（消えたか移った） 右クリックで外せます")
    return matched, "  ".join(notes)


def _stamp(file: LibraryFile) -> str:
    """見本の絵と波形を覚える名前 同じ場所でも中身を差し替えたら別の名前になる"""
    return f"{file.id}-{file.size}-{file.modified_ns}"
