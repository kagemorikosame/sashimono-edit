"""波形表示のためのピーク解析

タイムラインは 1 ピクセルに数百〜数十万サンプルを描く 毎回それだけの音声を
読み直すのは論外なので、あらかじめ min/max のピークを段階的な解像度で作っておき、
表示倍率に応じて使い分ける

段階を持たせるのが要点 1 段階だけだと、拡大時は粗く、縮小時は読む量が多すぎる
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from sashimono.engine.decode import AudioDecoder

__all__ = ["PeakLevel", "Waveform", "analyze_waveform"]

#: 最も細かい段階で、1 ピークにまとめるサンプル数
#: 48kHz なら 1 ピーク約 5.3ms 編集で見る最大倍率でも十分細かい
BASE_SAMPLES_PER_PEAK = 256

#: 段階ごとの粗さの比 8 倍ずつ粗くする
LEVEL_RATIO = 8

#: 一度に読むサンプル数 大きすぎるとメモリを、小さすぎると呼び出し回数を食う
CHUNK_SAMPLES = 1 << 18


@dataclass(frozen=True, slots=True)
class PeakLevel:
    """1 段階分のピーク

    ``peaks`` の形は ``(ピーク数, チャンネル数, 2)`` 最後の次元が ``(最小, 最大)``
    平均や絶対値の最大ではなく min/max を持つのは、波形の非対称性（打楽器など）を
    潰さないため
    """

    samples_per_peak: int
    peaks: np.ndarray

    @property
    def count(self) -> int:
        return int(self.peaks.shape[0])

    @property
    def channels(self) -> int:
        return int(self.peaks.shape[1])


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Waveform:
    """1 本の音声ストリームのピーク一式

    弱参照を取れるようにしてある タイムラインが波形の画像を貯めるとき、解析結果を
    強く持つと、使わなくなった素材の解析を捨ててもメモリが空かない
    """

    sample_rate: int
    channels: int
    total_samples: int
    levels: tuple[PeakLevel, ...]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("段階が 1 つも無い")

    @property
    def duration(self) -> Fraction:
        return Fraction(self.total_samples, self.sample_rate)

    def level_for(self, samples_per_pixel: float) -> PeakLevel:
        """表示倍率に見合う段階を選ぶ

        1 ピクセルあたりのサンプル数を超えない中で最も粗い段階を返す 粗すぎると
        ピークが 1 個も入らないピクセルができ、波形が途切れて見える
        """
        chosen = self.levels[0]
        for level in self.levels:
            if level.samples_per_peak <= samples_per_pixel:
                chosen = level
            else:
                break
        return chosen

    def envelope(self, start_sample: int, end_sample: int, columns: int) -> np.ndarray:
        """``[start_sample, end_sample)`` を ``columns`` 本に束ねた min/max を返す

        形は ``(columns, チャンネル数, 2)`` 範囲外は 0 で埋める 描画側は
        この配列をそのまま縦線として描けばよい
        """
        if columns <= 0 or end_sample <= start_sample:
            return np.zeros((max(columns, 0), self.channels, 2), dtype=np.float32)

        span = end_sample - start_sample
        level = self.level_for(span / columns)
        out = np.zeros((columns, self.channels, 2), dtype=np.float32)

        # 各列が対応するピーク範囲を一括で求める 列ごとに Python で回すと、
        # 横 2000 ピクセルのタイムラインで描画のたびに効いてくる
        edges = start_sample + np.linspace(0, span, columns + 1)
        starts = np.floor(edges[:-1] / level.samples_per_peak).astype(np.int64)
        stops = np.ceil(edges[1:] / level.samples_per_peak).astype(np.int64)
        starts = np.clip(starts, 0, level.count)
        stops = np.clip(np.maximum(stops, starts + 1), 0, level.count)

        # 束ねる所もまとめて求める 列ごとに回すと、実素材を 100 本並べた全体表示で
        # 1 回の描画に 10ms ほど掛かった（#201） 隣の列の範囲は重なることがあるので、
        # 列の頭と終わりを交互に並べて reduceat に渡し、偶数番（頭から終わりまで）だけを使う
        filled = starts < stops
        if not filled.any():
            return out
        low = int(starts[filled].min())
        high = int(stops[filled].max())
        window = level.peaks[low:high]
        # 最後の終わりは window の長さに等しく、reduceat は範囲の外の番号を受けない
        # 末尾に 1 行足して受けられるようにする（その行から先の結果は捨てる）
        padded = np.concatenate([window, window[-1:]], axis=0)
        bounds = np.clip(np.stack([starts, stops], axis=1).ravel() - low, 0, high - low)
        minima = np.minimum.reduceat(padded[:, :, 0], bounds, axis=0)[0::2]
        maxima = np.maximum.reduceat(padded[:, :, 1], bounds, axis=0)[0::2]
        out[filled, :, 0] = minima[filled]
        out[filled, :, 1] = maxima[filled]
        return out

    def envelopes(self, segments: Sequence[tuple[int, int, int]]) -> np.ndarray:
        """``(頭, 終わり, 列の数)`` の並びを、それぞれ :meth:`envelope` と同じく束ねて縦に繋げた物

        形は ``(列の数の合計, チャンネル数, 2)`` 並びの順に置く 値は 1 本ずつ
        :meth:`envelope` を呼んだときと同じ（最小・最大は束ね方で変わらない）

        タイムラインは、拡大して初めて見る倍率で何百本ものクリップの波形を作り直す
        1 本ずつ呼ぶと numpy を 20 回ほど呼ぶ所が本数分になり、200 本で 6ms ほど掛かった
        （#260） 同じ段階を使う範囲をまとめれば、numpy を呼ぶ回数は段階の数で済む
        """
        sizes = [max(columns, 0) for _, _, columns in segments]
        offsets = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]).astype(np.int64)
        out = np.zeros((int(offsets[-1]), self.channels, 2), dtype=np.float32)
        groups: dict[int, list[int]] = {}
        for number, (start, end, columns) in enumerate(segments):
            if columns <= 0 or end <= start:
                continue
            # 段階は番号で分ける 段階どうしを == で比べると、中の配列を比べてしまう
            level = self.level_for((end - start) / columns)
            index = next(n for n, found in enumerate(self.levels) if found is level)
            groups.setdefault(index, []).append(number)
        for level_number, numbers in groups.items():
            self._envelopes_on(self.levels[level_number], segments, numbers, offsets, out)
        return out

    def _envelopes_on(
        self,
        level: PeakLevel,
        segments: Sequence[tuple[int, int, int]],
        numbers: list[int],
        offsets: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """同じ段階 ``level`` を使う範囲をまとめて束ね、``out`` のそれぞれの所へ書く"""
        # 列の区切りは 1 本ずつ envelope と同じ式（linspace）で求める まとめて作ると
        # 浮動小数の丸めが変わり、区切りの 1 ピークがずれることがある
        edges = [
            start + np.linspace(0, end - start, columns + 1)
            for start, end, columns in (segments[n] for n in numbers)
        ]
        spp = level.samples_per_peak
        starts = np.concatenate([np.floor(e[:-1] / spp) for e in edges]).astype(np.int64)
        stops = np.concatenate([np.ceil(e[1:] / spp) for e in edges]).astype(np.int64)
        count = level.count
        starts = np.clip(starts, 0, count)
        stops = np.clip(np.maximum(stops, starts + 1), 0, count)
        places = np.concatenate([np.arange(offsets[n], offsets[n + 1]) for n in numbers])
        filled = starts < stops
        if not filled.any():
            return
        first, last = starts[filled], stops[filled]
        # reduceat は配列の外の番号を受けない 終わりが末尾（count）の列は 1 つ手前で止め、
        # 末尾の 1 行を後から足す 頭が末尾の 1 行なら、reduceat はその行をそのまま返す
        ends_at_tail = last >= count
        bounds = np.stack([first, np.minimum(last, count - 1)], axis=1).ravel()
        peaks = level.peaks
        minima = np.minimum.reduceat(peaks[:, :, 0], bounds, axis=0)[0::2]
        maxima = np.maximum.reduceat(peaks[:, :, 1], bounds, axis=0)[0::2]
        if ends_at_tail.any():
            minima[ends_at_tail] = np.minimum(minima[ends_at_tail], peaks[count - 1, :, 0])
            maxima[ends_at_tail] = np.maximum(maxima[ends_at_tail], peaks[count - 1, :, 1])
        target = places[filled]
        out[target, :, 0] = minima
        out[target, :, 1] = maxima


def analyze_waveform(
    path: Path,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    stream_index: int | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Waveform | None:
    """素材を読み切ってピークを作る

    重い処理なのでバックグラウンドで呼ぶ前提 ``should_cancel`` が真を返したら
    途中で ``None`` を返して抜ける 素材を差し替えたのに前の解析が走り続ける、
    という状態を避けるため
    """
    with AudioDecoder(
        path, sample_rate=sample_rate, channels=channels, stream_index=stream_index
    ) as decoder:
        total = int(decoder.duration * sample_rate)
        base: list[np.ndarray] = []
        consumed = 0

        for chunk in _chunks(decoder, total):
            if should_cancel is not None and should_cancel():
                return None
            base.append(_reduce(chunk, BASE_SAMPLES_PER_PEAK))
            consumed += len(chunk)
            if progress is not None and total > 0:
                progress(min(1.0, consumed / total))

    peaks = np.concatenate(base, axis=0) if base else np.zeros((0, channels, 2), dtype=np.float32)
    levels = [PeakLevel(BASE_SAMPLES_PER_PEAK, peaks)]

    # 粗い段階は、細かい段階から作る 元の音声を読み直す必要は無い
    while levels[-1].count > 1:
        coarser = _coarsen(levels[-1])
        if coarser.count == levels[-1].count:
            break
        levels.append(coarser)

    if progress is not None:
        progress(1.0)
    return Waveform(
        sample_rate=sample_rate,
        channels=channels,
        total_samples=max(total, peaks.shape[0] * BASE_SAMPLES_PER_PEAK),
        levels=tuple(levels),
    )


def _chunks(decoder: AudioDecoder, total: int) -> Iterator[np.ndarray]:
    """素材を先頭から順に読み出す"""
    cursor = 0
    while cursor < total:
        count = min(CHUNK_SAMPLES, total - cursor)
        yield decoder.read(cursor, count)
        cursor += count


def _reduce(samples: np.ndarray, samples_per_peak: int) -> np.ndarray:
    """``(サンプル数, チャンネル数)`` を ``(ピーク数, チャンネル数, 2)`` へ"""
    count, channels = samples.shape
    groups = (count + samples_per_peak - 1) // samples_per_peak
    padded_length = groups * samples_per_peak
    if padded_length != count:
        # 端数は最後のサンプルで埋める 0 で埋めると、末尾に無い谷が生まれる
        pad = np.repeat(samples[-1:], padded_length - count, axis=0)
        samples = np.concatenate([samples, pad], axis=0)

    grouped = samples.reshape(groups, samples_per_peak, channels)
    out = np.empty((groups, channels, 2), dtype=np.float32)
    out[:, :, 0] = grouped.min(axis=1)
    out[:, :, 1] = grouped.max(axis=1)
    return out


def _coarsen(level: PeakLevel) -> PeakLevel:
    """1 段階粗いピークを作る"""
    count = level.count
    groups = (count + LEVEL_RATIO - 1) // LEVEL_RATIO
    padded_length = groups * LEVEL_RATIO
    peaks = level.peaks
    if padded_length != count and count > 0:
        pad = np.repeat(peaks[-1:], padded_length - count, axis=0)
        peaks = np.concatenate([peaks, pad], axis=0)

    grouped = peaks.reshape(groups, LEVEL_RATIO, level.channels, 2)
    out = np.empty((groups, level.channels, 2), dtype=np.float32)
    out[:, :, 0] = grouped[:, :, :, 0].min(axis=1)
    out[:, :, 1] = grouped[:, :, :, 1].max(axis=1)
    return PeakLevel(level.samples_per_peak * LEVEL_RATIO, out)
