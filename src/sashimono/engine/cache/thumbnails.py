"""タイムラインに並べる動画サムネイル（フィルムストリップ）

素材 1 本につき、等間隔のサムネイルを 1 枚のシート画像にまとめて持つ
1 枚ずつファイルにすると、10 分の素材で数百個のファイルができ、読み込みだけで
遅くなる シートなら 1 回の読み込みで済み、必要な列を切り出すだけになる
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from sashimono.engine.decode import ProbeError, VideoDecoder

from .store import CacheStore, load_arrays, media_key, save_arrays

__all__ = [
    "NAMESPACE",
    "SUFFIX",
    "Filmstrip",
    "build_filmstrip",
    "filmstrip_key",
    "load_filmstrip",
    "save_filmstrip",
]

NAMESPACE = "thumbnail"
SUFFIX = ".strip.npz"
#: シートの版 2 でタグの無い HD の素材を BT.709 で読むようにした（#61）
#: 上げないと、BT.601 で読んで色のずれたシートが残り続ける
#: 3 で素材の時刻を頭から数えるようにした（#123） 頭が 0 より後ろの素材は、それまで
#: 最初の 1 枚ばかりが並ぶシートになっていた 上げないとそれが残り続ける
FORMAT_VERSION = 3

#: サムネイルの高さ（ピクセル） トラックを広げても足りる程度に取り、
#: 表示時に縮小する 低すぎると拡大時にぼやける
THUMBNAIL_HEIGHT = 72

#: サムネイルを取る間隔（秒） 細かすぎると生成に時間がかかり、
#: 粗すぎるとタイムラインを拡大したときに同じ絵が並ぶ
DEFAULT_INTERVAL = Fraction(1, 2)

#: 1 本の素材から作るサムネイルの上限 長尺でシートが巨大にならないようにする
MAX_THUMBNAILS = 600


@dataclass(frozen=True, slots=True)
class Filmstrip:
    """等間隔のサムネイルを横に並べた 1 枚のシート"""

    #: ``(高さ, 幅 * 枚数, 4)`` の RGBA 配列
    sheet: np.ndarray
    #: 1 枚あたりの幅
    tile_width: int
    #: サムネイル間の間隔（秒）
    interval: Fraction

    @property
    def count(self) -> int:
        return self.sheet.shape[1] // self.tile_width if self.tile_width else 0

    @property
    def height(self) -> int:
        return int(self.sheet.shape[0])

    def at(self, source_seconds: Fraction) -> np.ndarray | None:
        """素材内の時刻に最も近いサムネイルを返す"""
        index = self.index_at(source_seconds)
        return None if index is None else self.tile(index)

    def index_at(self, source_seconds: Fraction) -> int | None:
        """素材内の時刻に最も近いサムネイルの番号 無ければ ``None``

        タイムラインは番号ごとに画像を貯めて使い回す :meth:`at` と同じ選び方にしないと、
        貯めた絵と描くはずの絵が食い違う
        """
        if self.count == 0 or self.interval <= 0:
            return None
        index = int(max(Fraction(0), source_seconds) / self.interval)
        return min(index, self.count - 1)

    def tile(self, index: int) -> np.ndarray | None:
        if not 0 <= index < self.count:
            return None
        start = index * self.tile_width
        return self.sheet[:, start : start + self.tile_width]


def filmstrip_key(path: Path, interval: Fraction, height: int) -> str:
    return media_key(path, extra=f"fs{FORMAT_VERSION}:{interval}:{height}")


def build_filmstrip(
    path: Path,
    *,
    interval: Fraction = DEFAULT_INTERVAL,
    height: int = THUMBNAIL_HEIGHT,
    stream_index: int | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Filmstrip | None:
    """素材からサムネイルシートを作る

    重い処理なのでバックグラウンドで呼ぶこと ``should_cancel`` が真を返したら
    ``None`` を返して抜ける
    """
    try:
        decoder = VideoDecoder(path, stream_index)
    except ProbeError:
        return None

    with decoder:
        info = decoder.info
        source_width, source_height = info.display_size
        if source_height <= 0:
            return None
        tile_width = max(1, round(source_width * height / source_height))

        duration = decoder.duration
        # 静止画は長さを持たない 1 枚だけ作る
        count = 1 if duration <= 0 else min(MAX_THUMBNAILS, max(1, int(duration / interval) + 1))
        # 上限に当たったら間隔を広げる 詰めて並べると同じ絵が続くだけになる
        if duration > 0 and count == MAX_THUMBNAILS:
            interval = Fraction(duration) / MAX_THUMBNAILS

        tiles: list[np.ndarray] = []
        for index in range(count):
            if should_cancel is not None and should_cancel():
                return None
            frame = decoder.frame_at(index * interval)
            tiles.append(
                _resize_nearest(frame, tile_width, height)
                if frame is not None
                else np.zeros((height, tile_width, 4), dtype=np.uint8)
            )
            if progress is not None:
                progress((index + 1) / count)

    sheet = np.concatenate(tiles, axis=1) if tiles else np.zeros((height, 0, 4), dtype=np.uint8)
    return Filmstrip(sheet=sheet, tile_width=tile_width, interval=interval)


def save_filmstrip(store: CacheStore, key: str, filmstrip: Filmstrip) -> Path:
    # 隣り合うサムネイルは似た絵なので圧縮がよく効く 長尺素材でシートが
    # 数十 MB になるため、ここは圧縮した方がよい
    return save_arrays(
        store.prepare(NAMESPACE, key, SUFFIX),
        {
            "version": np.array([FORMAT_VERSION]),
            "sheet": filmstrip.sheet,
            "tile_width": np.array([filmstrip.tile_width]),
            "interval": np.array([filmstrip.interval.numerator, filmstrip.interval.denominator]),
        },
        compressed=True,
    )


def load_filmstrip(store: CacheStore, key: str) -> Filmstrip | None:
    data = load_arrays(store.path_for(NAMESPACE, key, SUFFIX))
    if data is None:
        return None
    try:
        if int(data["version"][0]) != FORMAT_VERSION:
            return None
        interval = data["interval"]
        return Filmstrip(
            sheet=data["sheet"],
            tile_width=int(data["tile_width"][0]),
            interval=Fraction(int(interval[0]), int(interval[1])),
        )
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        return None


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """最近傍でサムネイルサイズへ縮小する

    サムネイルは数十ピクセルなので、補間の質より速さを取る 素材 1 本で
    数百枚作るため、ここが遅いと読み込み直後の待ち時間に直結する
    """
    source_height, source_width = image.shape[:2]
    if source_height == 0 or source_width == 0:
        return np.zeros((height, width, 4), dtype=np.uint8)
    rows = (np.arange(height) * source_height // height).clip(0, source_height - 1)
    columns = (np.arange(width) * source_width // width).clip(0, source_width - 1)
    return np.ascontiguousarray(image[rows][:, columns])
