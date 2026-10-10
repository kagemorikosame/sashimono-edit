"""プリセットとエイリアスの見本の絵を覚えて配る係（#277）

作り置きと後から埋める

* **作り置き** 描いた絵はキャッシュの置き場（``looks``）へ PNG で置く 鍵は中身の指紋
  （:func:`~sashimono.core.io.library.look_fingerprint`）と描き方と本体の版 名前や分類を
  変えても作り直さず、中身が変われば鍵が変わって作り直す 本体を上げたら描き方が
  変わっているかもしれないので、版も鍵に入れる
* **後から埋める** 一覧を開いたときは、覚えている絵だけを出してすぐ開く 足りない物は
  見えている物だけを走り係（:class:`~sashimono.engine.render.look_preview.LookWorker`）へ
  頼み、描けたものから :attr:`LookThumbnails.ready` で知らせる

壊れた絵は掴まない 読めない・大きさの違う PNG は消して描き直す
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QImage

from sashimono import __version__
from sashimono.core.io.aliases import Alias
from sashimono.core.io.library import look_fingerprint
from sashimono.core.io.presets import Preset
from sashimono.engine.cache.store import default_cache_root
from sashimono.engine.render.look_preview import LookPicture, LookWorker

__all__ = [
    "THUMBNAILS_FULL",
    "THUMBNAILS_OFF",
    "THUMBNAILS_SIMPLE",
    "THUMBNAIL_MODES",
    "THUMBNAIL_SIZE",
    "LookThumbnails",
    "Thumbnail",
]

#: 見本の描き方 :attr:`~sashimono.ui.workspace.Preferences.library_thumbnails` の値
#: エフェクトも描く（GPU）
THUMBNAILS_FULL = "full"
#: 文字と図形だけ（CPU） エフェクトは出ない
THUMBNAILS_SIMPLE = "simple"
#: 出さない（名前だけの一覧）
THUMBNAILS_OFF = "off"
THUMBNAIL_MODES = (THUMBNAILS_FULL, THUMBNAILS_SIMPLE, THUMBNAILS_OFF)

#: 置いておく絵の大きさ（画素） 一覧では 128x72 で見せ、画面の倍率 2 でもぼやけない
THUMBNAIL_SIZE = QSize(256, 144)
#: 置き場の名前空間
NAMESPACE = "looks"
#: 絵の置き方の版 置き方（大きさ・切り出し・地）を変えたら上げる 上げないと前の絵が残る
_FORMAT = 1
#: 置き場に残す枚数の上限 超えたら古く使った物から消す 消した物のプリセットの絵が
#: 残り続けて置き場が膨らむのを止める 1 枚 30KB ほどなので上限でも 60MB ほど
_DISK_LIMIT = 2000
#: 上限を超えたときに残す枚数 1 枚超えるたびに 1 枚ずつ消すと、毎回置き場を数え直す
_DISK_KEEP = 1500
#: 覚えておく絵の数 1 枚 147KB（256x144 の RGBA） 300 枚で 44MB
_MEMORY_LIMIT = 300
#: 見えた物がなかったことを PNG に書いておく印
_EMPTY_KEY = "sashimono-empty"

Item = Preset | Alias


class Thumbnail:
    """配る 1 枚 ``image`` は ``None`` なら描けなかった（``error`` に理由）"""

    __slots__ = ("empty", "error", "image", "simple")

    def __init__(
        self,
        image: QImage | None,
        *,
        simple: bool = False,
        empty: bool = False,
        error: str = "",
    ) -> None:
        self.image = image
        self.simple = simple
        self.empty = empty
        self.error = error


class LookThumbnails(QObject):
    """見本の絵の置き場と走り係をまとめる 一覧の窓が閉じたら :meth:`release` で走り係を止める"""

    #: 鍵の絵ができた（描けなかったときも知らせる 待ちの印を外すため）
    ready = Signal(str)

    def __init__(
        self,
        root: Path | None = None,
        *,
        mode: str = THUMBNAILS_FULL,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._root = root
        self._mode = mode
        self._memory: OrderedDict[str, Thumbnail] = OrderedDict()
        #: 頼んで描いている途中の鍵と、頼んだときに簡易の描き方だったか
        self._pending: dict[str, bool] = {}
        self._worker: LookWorker | None = None
        #: GPU のコンテキストを作れなかった このプロセスの間は簡易の描き方にする
        self._gpu_unavailable = False
        self._pruned = False

    # --- 外から ---

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        # 描き方が変わると鍵も変わる 走り係も作り直す（GPU を使うかが変わる）
        self.release()
        self._mode = mode

    @property
    def simple(self) -> bool:
        """エフェクトを描かない描き方か 一覧の窓が「エフェクトは出ていない」と書くのに使う"""
        return self._mode == THUMBNAILS_SIMPLE or self._gpu_unavailable

    @property
    def enabled(self) -> bool:
        return self._mode != THUMBNAILS_OFF

    def key_for(self, item: Item) -> str:
        drawing = "cpu" if self.simple else "gpu"
        text = f"{_FORMAT}|{__version__}|{drawing}|{look_fingerprint(item)}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def cached(self, item: Item) -> Thumbnail | None:
        """覚えている絵 無ければ ``None``（描かない 描くのは :meth:`request`）"""
        if not self.enabled:
            return None
        key = self.key_for(item)
        found = self._memory.get(key)
        if found is not None:
            self._memory.move_to_end(key)
            return found
        loaded = self._load(key)
        if loaded is not None:
            self._remember(key, loaded)
        return loaded

    def thumbnail(self, key: str) -> Thumbnail | None:
        """:attr:`ready` で知らせた鍵の絵 描き方が途中で変わっても、頼んだ鍵で引ける"""
        return self._memory.get(key)

    def request(self, item: Item) -> str | None:
        """描くよう頼む 覚えている・頼んである物は頼まない 頼んだ鍵を返す"""
        if not self.enabled:
            return None
        key = self.key_for(item)
        if key in self._pending or self.cached(item) is not None:
            return key
        worker = self._ensure_worker()
        self._pending[key] = self.simple
        worker.submit(key, item)
        return key

    def release(self) -> None:
        """走り係を止める 頼んだまま描いていない物は捨てる（次に開いたときに頼み直す）

        一覧の窓を閉じても止めない コンテキストと描画係を作り直すと、開くたびに
        0.7 秒ほど待つことになる 止めるのは描き方を変えたときとアプリを閉じるとき
        """
        worker = self._worker
        self._worker = None
        self._pending.clear()
        if worker is not None:
            worker.drawn.disconnect(self._on_drawn)
            worker.gpu_failed.disconnect(self._on_gpu_failed)
            worker.stop()
            worker.deleteLater()

    # --- 中の手順 ---

    def _ensure_worker(self) -> LookWorker:
        if self._worker is not None:
            return self._worker
        worker: LookWorker | None = None
        if not self.simple:
            from sashimono.engine.gpu import GLContextError

            try:
                worker = LookWorker(gpu=True)
            except GLContextError:
                # サーフェスも作れない機械 簡易の描き方に切り替え、一覧の窓はその旨を書く
                self._gpu_unavailable = True
        if worker is None:
            worker = LookWorker(gpu=False)
        worker.drawn.connect(self._on_drawn)
        worker.gpu_failed.connect(self._on_gpu_failed)
        worker.start()
        self._worker = worker
        return worker

    def _on_gpu_failed(self, _reason: str) -> None:
        # 以後の鍵は簡易の描き方の物になる 走り係も簡易の描き方で描き続ける
        self._gpu_unavailable = True

    def _on_drawn(self, key: str, result: object) -> None:
        if key not in self._pending:
            # 止めた後に届いた物 頼み直したときに描き直すので捨てる
            return
        asked_simple = self._pending.pop(key)
        if isinstance(result, LookPicture):
            thumbnail = Thumbnail(
                _to_thumbnail(result.image), simple=result.simple, empty=result.empty
            )
            # 覚えるのを先にする 置き場に書けなくても、この起動の間は見本が出る
            self._remember(key, thumbnail)
            assert thumbnail.image is not None
            if result.simple == asked_simple:
                # GPU で頼んだのに簡易で描いた物（GPU が使えなかった）は置き場へ書かない
                # 書くと、GPU の使える機械へ移ったときに、エフェクトの無い絵を掴み続ける
                self._save(key, thumbnail.image, empty=result.empty)
        else:
            self._remember(key, Thumbnail(None, error=str(result)))
        self.ready.emit(key)

    def _remember(self, key: str, thumbnail: Thumbnail) -> None:
        self._memory[key] = thumbnail
        self._memory.move_to_end(key)
        while len(self._memory) > _MEMORY_LIMIT:
            self._memory.popitem(last=False)

    def _path(self, key: str) -> Path:
        root = self._root if self._root is not None else default_cache_root()
        return root / NAMESPACE / key[:2] / f"{key}.png"

    def _load(self, key: str) -> Thumbnail | None:
        path = self._path(key)
        if not path.is_file():
            return None
        image = QImage(str(path))
        if image.isNull() or image.size() != THUMBNAIL_SIZE:
            # 壊れた絵（書きかけ・別の版の大きさ）は掴まずに描き直す
            path.unlink(missing_ok=True)
            return None
        # 使った時刻を付け直す 上限を超えて消すときに、よく使う物を残す
        with contextlib.suppress(OSError):
            os.utime(path)
        # 鍵は描き方ごとに違うので、いまの描き方の鍵で見つかった物はいまの描き方で描いた物
        return Thumbnail(image, simple=self.simple, empty=image.text(_EMPTY_KEY) == "1")

    def _save(self, key: str, image: QImage, *, empty: bool) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if empty:
                image.setText(_EMPTY_KEY, "1")
            # 書きかけを読まないよう、別の名前に書いてから差し替える
            # 形式は拡張子で決まるので、一時の名前も .png で終える
            temporary = path.with_name(f"{path.stem}.writing.png")
            if image.save(str(temporary)):
                temporary.replace(path)
            else:
                temporary.unlink(missing_ok=True)
        except OSError:
            # 置き場に書けなくても見本は出す（次に開いたときに描き直すだけ）
            return
        self._prune(path.parent.parent)

    def _prune(self, folder: Path) -> None:
        """置き場が上限を超えたら、古く使った物から消す 1 回の起動で 1 度だけ数える"""
        if self._pruned:
            return
        self._pruned = True
        try:
            files = [path for path in folder.rglob("*.png") if path.is_file()]
            if len(files) <= _DISK_LIMIT:
                return
            files.sort(key=lambda path: path.stat().st_mtime)
            for path in files[: len(files) - _DISK_KEEP]:
                path.unlink(missing_ok=True)
        except OSError:
            return


def _to_thumbnail(image: np.ndarray) -> QImage:
    """描いた RGBA を置いておく大きさの画像へ 切り出しは 16:9 なので縦横の割合は変わらない"""
    height, width = image.shape[:2]
    data = np.ascontiguousarray(image)
    source = QImage(data.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    # 縮める前に事前乗算へ直す ストレートのまま混ぜると、透明な所の色（黒）が縁へにじみ、
    # 明るい地で文字の周りに黒い輪が出る
    premultiplied = source.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    return premultiplied.scaled(
        THUMBNAIL_SIZE,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    ).copy()
