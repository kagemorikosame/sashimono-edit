"""音量の推移のグラフの中身（#338 C2）

AI は静止画しか見られない 音の推移を絵で渡すため、横に時間・縦にラウドネスを取った
1 枚を作る 何をどこへ描くか（升目・線の値・帯）はここで決め、描くのは編集画面
（:meth:`~sashimono.ai.host.EditorHost.draw_sound_graph`） ここは Qt を知らない
偽のホストの試験でも同じ中身を確かめられるように

大きさの上限は、コマを並べる 1 枚（:mod:`.sheet`）と同じ考え（長い辺 1568 画素まで）
既定の 1200x320 は 500 トークンほど
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BAND_HEIGHT",
    "FLOOR_LUFS",
    "GRAPH_DEFAULT_HEIGHT",
    "GRAPH_DEFAULT_WIDTH",
    "GRAPH_MAX_HEIGHT",
    "GRAPH_MIN_HEIGHT",
    "GRAPH_MIN_WIDTH",
    "LABEL_HEIGHT",
    "MARGIN_LEFT",
    "MAX_MARKS",
    "SUBTITLE_HEIGHT",
    "TOP_LUFS",
    "Band",
    "SoundGraph",
    "clipped_columns",
    "graph_columns",
    "plot_width",
    "spans",
    "time_step",
]

GRAPH_DEFAULT_WIDTH = 1200
GRAPH_DEFAULT_HEIGHT = 320
#: 横幅の下限 時刻の字（00:00:00:00）を 6 つほど並べられる幅
GRAPH_MIN_WIDTH = 480
GRAPH_MIN_HEIGHT = 200
GRAPH_MAX_HEIGHT = 600
#: 縦軸の下の端（LUFS） 話し声の小さな所（-40 前後）と息継ぎ（-50 前後）まで見える深さ
#: これより静かな所は下の端に張り付ける（無音は帯の灰色で分かる）
FLOOR_LUFS = -60.0
#: 縦軸の上の端（LUFS） フルスケールの正弦波でも 0 を少し超える程度
TOP_LUFS = 0.0
#: 縦線で示すフレームの上限 これより多いと番号が重なって読めない
MAX_MARKS = 32

#: 描く所の余白（画素） 左は縦軸の字（-60）、上は凡例、下は見分けの帯・字幕の帯・時刻の字
MARGIN_LEFT = 44
MARGIN_RIGHT = 12
MARGIN_TOP = 22
#: 見分けの帯の高さ 色で話し声・音楽・無音を分ける
BAND_HEIGHT = 12
#: 字幕の帯の高さ
SUBTITLE_HEIGHT = 6
#: 時刻の字の帯の高さ
LABEL_HEIGHT = 16
MARGIN_BOTTOM = BAND_HEIGHT + SUBTITLE_HEIGHT + LABEL_HEIGHT + 6

#: 時刻の目盛りの間の候補（秒） 目盛りが 4〜10 本になる物を選ぶ
_TIME_STEPS = (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)


@dataclass(frozen=True, slots=True)
class Band:
    """見分けの帯 1 つ 横の画素で ``[left, right)``"""

    left: int
    right: int
    label: str


@dataclass(frozen=True, slots=True)
class SoundGraph:
    """描く物すべて 横の位置はどれも画像の左端からの画素"""

    width: int
    height: int
    #: 描く所の 1 列ごとの、瞬時のラウドネスの最大（LUFS） 測れない（無音・未解析）は ``None``
    momentary: tuple[float | None, ...]
    #: 1 列ごとの短時間（3 秒）のラウドネス
    short_term: tuple[float | None, ...]
    #: 頭打ち（最大のサンプルがフルスケール）になった列
    clipped: tuple[int, ...]
    bands: tuple[Band, ...]
    #: 字幕のある所 ``(左, 右)``
    subtitles: tuple[tuple[int, int], ...]
    #: 音量の基準の横線（LUFS） 引かなければ ``None``
    target: float | None
    #: 時刻の目盛り ``(横の位置, 字)``
    time_ticks: tuple[tuple[int, str], ...]
    #: 頼まれたフレームの縦線 ``(横の位置, 番号の字)``
    marks: tuple[tuple[int, str], ...]
    #: 解析が途中なら、ここから右はまだ測っていない ``None`` なら全部測った
    analyzed_until: int | None = None

    @property
    def left(self) -> int:
        return MARGIN_LEFT

    @property
    def right(self) -> int:
        return self.width - MARGIN_RIGHT

    @property
    def top(self) -> int:
        return MARGIN_TOP

    @property
    def bottom(self) -> int:
        """描く所（線を引く所）の下の端 その下に見分けの帯・字幕の帯・時刻の字を並べる"""
        return self.height - MARGIN_BOTTOM

    def y(self, lufs: float) -> float:
        """ラウドネスの縦の位置 上が大きい 範囲の外は端に寄せる"""
        clamped = max(FLOOR_LUFS, min(TOP_LUFS, lufs))
        share = (clamped - FLOOR_LUFS) / (TOP_LUFS - FLOOR_LUFS)
        return self.bottom - share * (self.bottom - self.top)


def plot_width(width: int) -> int:
    """描く所の横幅（列の数）"""
    return width - MARGIN_LEFT - MARGIN_RIGHT


def graph_columns(
    values: np.ndarray, columns: int, *, reduce: str = "max", floor: float = -70.0
) -> tuple[float | None, ...]:
    """区切りごとの値を ``columns`` 列に束ねる ``floor`` 以下と束ねる物の無い列は ``None``

    最大で束ねるのは、長い作品を 1 列に何十区切りも詰めたとき、短い大きな音（音割れの手前の
    山）を平均で潰さないため 終わりの値（``last``）は短時間のラウドネスのように、既に
    ならしてある値に使う
    """
    count = len(values)
    found: list[float | None] = []
    for column in range(columns):
        low = column * count // columns
        high = max(low + 1, (column + 1) * count // columns)
        part = values[low : min(high, count)]
        if len(part) == 0:
            found.append(None)
            continue
        value = float(part.max() if reduce == "max" else part[-1])
        found.append(round(value, 1) if math.isfinite(value) and value > floor else None)
    return tuple(found)


def clipped_columns(peaks_db: np.ndarray, columns: int, level_db: float) -> tuple[int, ...]:
    """最大のサンプルが ``level_db`` 以上になった列の番号（描く所の左端から）

    丸めた値（:func:`graph_columns`）で比べると、-0.04 dBFS のような手前の山も頭打ちに数える
    """
    count = len(peaks_db)
    found: list[int] = []
    for column in range(columns):
        low = column * count // columns
        high = min(count, max(low + 1, (column + 1) * count // columns))
        if low < high and float(peaks_db[low:high].max()) >= level_db:
            found.append(column)
    return tuple(found)


def time_step(seconds: float, columns: int) -> float:
    """時刻の目盛りの間（秒） 1 本の字（約 80 画素）が重ならず、多すぎない間"""
    most = max(2, columns // 90)
    for step in _TIME_STEPS:
        if seconds / step <= most:
            return float(step)
    return float(_TIME_STEPS[-1])


def spans(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """真の続く所 ``(頭, 終わり)``"""
    found: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            found.append((start, index))
            start = None
    if start is not None:
        found.append((start, len(flags)))
    return found
