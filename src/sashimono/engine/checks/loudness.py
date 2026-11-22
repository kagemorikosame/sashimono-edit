"""ラウドネス（ITU-R BS.1770-4 の積分ラウドネス）を numpy だけで測る

新しい依存（scipy など）を足さないため、K 特性の 2 段の IIR は、インパルス応答を
有限の長さで切った FIR として FFT で畳み込む 切る長さは応答が十分に消える所にしてあり
（:data:`IMPULSE_SECONDS`）、EBU Tech 3341 の試験信号で ±0.1 LU に入ることを試験で見る

測り方（BS.1770-4）

- 各チャンネルに K 特性（高域のシェルフ + 38 Hz の高域通過）を掛ける
- 400 ms の区切りを 100 ms ずつずらして（75% 重ね）平均の二乗を取り、チャンネルの重み
  （前の 3 本は 1.0、後ろの 2 本は 1.41、LFE は数えない）を掛けて足す
- -0.691 + 10 log10(足した値) が区切りのラウドネス -70 LUFS より静かな区切りを捨て
  （絶対ゲート）、残りの平均より 10 LU 静かな区切りも捨て（相対ゲート）、残った区切りの
  平均を積分ラウドネスにする

係数は 48 kHz で規格の表の値になる式（libebur128 と同じ設計の式）から、どのサンプリング
レートでも作る 48 kHz に変換してから測ると、変換の手間とその誤差が増える
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise

import numpy as np

from sashimono.effects.limiter import DEFAULT_RELEASE_MS, ceiling_before_fader, reduction

__all__ = [
    "ABSOLUTE_GATE",
    "Dynamics",
    "FineSums",
    "Headroom",
    "LoudnessLane",
    "LoudnessMeter",
    "block_loudness",
    "channel_weights",
    "gated_loudness",
    "k_weighting",
]

#: 絶対ゲート（LUFS） これより静かな区切りは測らない 無音で値が下がり続けないため
ABSOLUTE_GATE = -70.0
#: 相対ゲート（LU） 残った区切りの平均よりこれだけ静かな区切りも捨てる
RELATIVE_GATE = -10.0
#: 区切りのラウドネスの式の定数 1 kHz の正弦波で K 特性の増幅（約 +0.691 dB）を打ち消す
_OFFSET = -0.691
#: インパルス応答を切る長さ（秒） 38 Hz の高域通過の極は 48 kHz で半径 0.995 ほどで、
#: 0.2 秒（9600 サンプル）先では e^-48 まで減る 二乗のエネルギーの誤差は測れないほど小さい
#: 短くすると低い音のエネルギーが残り、低音の多い音楽で値がずれる
IMPULSE_SECONDS = 0.2
#: 溜めてからまとめて畳み込む長さ（秒） 応答（0.2 秒）より十分長くすると、FFT の手間が
#: 音の長さにほぼ比例する 長くしすぎると、途中の値（検査の途中で返すラウドネス）が遅れる
CHUNK_SECONDS = 1.0


def k_weighting(sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K 特性の 2 段の双二次フィルタの係数 ``(b1, a1, b2, a2)``

    1 段目は頭の影響を真似た高域のシェルフ（約 +4 dB）、2 段目は低い音を落とす高域通過
    48 kHz では BS.1770-4 の表の値（b1 = 1.53512485958697 など）になる
    """
    rate = float(sample_rate)
    # 1 段目 高域のシェルフ
    f0 = 1681.974450955533
    gain = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10.0 ** (gain / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b1 = np.array(
        [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    )
    a1 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    # 2 段目 高域通過
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1.0 + k / q + k * k
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return b1, a1, b2, a2


def _biquad(b: np.ndarray, a: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """双二次フィルタをそのまま回す インパルス応答を作るときだけ使う（短い信号のため）"""
    out = np.zeros_like(signal)
    x1 = x2 = y1 = y2 = 0.0
    for index, x in enumerate(signal.tolist()):
        y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        out[index] = y
        x2, x1 = x1, x
        y2, y1 = y1, y
    return out


@lru_cache(maxsize=8)
def _impulse(sample_rate: int) -> np.ndarray:
    """K 特性の 2 段を通したインパルス応答 :data:`IMPULSE_SECONDS` で切る

    Python の行の繰り返しで回すのは 1 度だけ（サンプリングレートごとに覚える） 長い音を
    同じ書き方で回すと、10 分の音で数分かかる
    """
    length = max(16, round(sample_rate * IMPULSE_SECONDS))
    impulse = np.zeros(length)
    impulse[0] = 1.0
    b1, a1, b2, a2 = k_weighting(sample_rate)
    return _biquad(b2, a2, _biquad(b1, a1, impulse))


def channel_weights(channels: int) -> np.ndarray:
    """チャンネルの重み（BS.1770-4 の表） 並びは FFmpeg の 5.1（左・右・中央・LFE・左後・右後）

    LFE は数えない 後ろの 2 本は 1.41（+1.5 dB） 表に無い並び（3・4・7 本以上）は、
    どれを後ろと見なすか決められないので、全部を 1.0 で数える
    """
    if channels == 6:
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
    return np.ones(channels)


#: 100 ms の区切りを、抑えた後の見積もりのためにいくつへ割って覚えるか（1 つ 5 ms #494）
#: 戻る速さ（150 ms）より十分に短く、3 時間の作品でも 17 MB に収まる
_FINE_PARTS = 20
#: 1 秒あたりの区切りの数
_FINE_RATE = _FINE_PARTS * 10


@dataclass(frozen=True, slots=True, eq=False)
class LoudnessLane:
    """重ならないクリップの組 1 つの手がかり（#494） リミッターはこの音へ別々に掛かる"""

    energy: np.ndarray
    peak: np.ndarray
    #: 組のクリップの頭（測り始めからの秒）
    starts: tuple[float, ...] = ()
    #: 100 ms に満たない末尾の、5 ms ごとの山 音量の窓には入らないが、書き出す音には入る
    tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: リミッターが見る山（トラックの音量と定位の前 末尾も含む） ``peak`` は掛けた後の山なので、
    #: 絞る量はこちらで読む（片側へ振ると、振った先に無い側の山で両方が絞られる）
    detected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: 組のクリップごとの、リミッターの後に掛かる倍率（dB ``starts`` と同じ並び）
    faders: tuple[float, ...] = ()

    def spans(self) -> list[tuple[int, int, float]]:
        """``detected`` の中でクリップごとの区切りと、その倍率

        クリップの頭が 5 ms の区切りの途中に来ると、その区切りは前後どちらの範囲にも入れる
        区切りの山からはどちらのクリップの山か分からない 片方へ丸めると、倍率の大きい側の頭の山を
        浅く数え、本人の深さを越えて潰す（PR #495 の 4 回目のレビュー）
        """
        count = len(self.detected)
        # 区切りちょうどの頭は割り算の誤差で前の区切りへ掛からないよう、わずかに寄せて丸める
        heads = [start * _FINE_RATE for start in self.starts[1:]]
        ends = [min(count, max(0, math.ceil(head - 1e-6))) for head in heads]
        begins = [min(count, max(0, math.floor(head + 1e-6))) for head in heads]
        found = zip([0, *begins], [*ends, count], self.faders, strict=True)
        return [(low, high, fader) for low, high, fader in found if high > low]

    def ceilings(self, ceiling_db: float) -> np.ndarray:
        """区切りごとの、リミッターの欄に入る上限（倍率） 2 つのクリップが入る区切りは低い方"""
        found = np.full(len(self.detected), math.inf)
        for low, high, fader in self.spans():
            here = 10.0 ** (ceiling_before_fader(ceiling_db, fader) / 20.0)
            found[low:high] = np.minimum(found[low:high], here)
        return np.where(np.isinf(found), 1.0, found)

    @property
    def peaks(self) -> np.ndarray:
        """末尾まで含めた山"""
        return np.concatenate([self.peak, self.tail])


class Headroom:
    """組ごとに抑えた音の和が ``top`` を越えない、リミッターの上限（#494）

    区切りごとに Σ min(上限, 山) = top となる上限を解き、一番低い物を取る（山の時刻が揃った
    最悪の形なので、越える側へは外さない） 掛ける量を変えて何度も引くので、並べた物を覚える
    """

    def __init__(self, lanes: Sequence[LoudnessLane]) -> None:
        self._peaks = np.sort(np.stack([lane.peaks.astype(np.float64) for lane in lanes]), axis=0)
        self._total = self._peaks.sum(axis=0)

    def ceiling(self, gain_db: float, top_db: float) -> float:
        """``gain_db`` 上げた和を ``top_db`` に収める上限（dBFS） どこも越えなければ ``inf``"""
        allowed = 10.0 ** ((top_db - gain_db) / 20.0)
        hot = self._total > allowed
        if not np.any(hot):
            return math.inf
        peaks = self._peaks[:, hot]
        level = np.full(peaks.shape[1], -math.inf)
        below = np.zeros(peaks.shape[1])
        for index, row in enumerate(peaks):
            level = np.maximum(level, (allowed - below) / (len(peaks) - index))
            below += row
        return gain_db + 20.0 * math.log10(max(float(level.min()), 1e-12))


@dataclass(frozen=True, slots=True, eq=False)
class Dynamics:
    """リミッターで抑えた後の音量を、混ぜ直さずに見積もる手がかり（#494）

    抑えた分の音量は掛けた dB どおりに増えない 見込まずに掛けると基準へ届かないまま書き出す
    ``eq=False`` なのは numpy の配列を持つため
    """

    #: 5 ms ごとの、K 特性と重みを掛けた二乗の和
    energy: np.ndarray
    #: 同じ区切りの最大のサンプル
    peak: np.ndarray
    #: 100 ms の区切り 1 つのサンプル数
    step: int
    #: 重なって鳴るときの、組ごとの手がかり 無ければ混ぜた音へ 1 つのリミッターとして数える
    lanes: tuple[LoudnessLane, ...] = ()
    #: 組どうしの掛け合わせ（組, 組, 5 ms ごとの K 特性を掛けた音の積の和） 相殺の具合を表す
    crosses: tuple[tuple[int, int, np.ndarray], ...] = ()

    def deepest_db(self, ceiling_db: float) -> float:
        """掛ける量 0 dB で、一番深く絞るリミッターの量（dB） どれも当たらなければ負

        混ぜた山からは読めない 逆相・定位・トラックの音量で、リミッターの見る山と止まる所が違う
        """
        deepest = -math.inf
        for lane in self.lanes:
            for low, high, fader in lane.spans():
                top = float(lane.detected[low:high].max())
                if top > 0.0:
                    local = ceiling_before_fader(ceiling_db, fader)
                    deepest = max(deepest, 20.0 * math.log10(top) - local)
        return deepest

    def limited_loudness(
        self, gain_db: float, ceiling_db: float, starts: Sequence[float] = ()
    ) -> float | None:
        """``gain_db`` 上げて ``ceiling_db`` のリミッターを通した積分ラウドネス

        区切りの中は一番強く絞った量で通して数える 見積もりが低めに出るので、届かない側へは外さない
        ``starts`` はクリップの頭（測り始めからの秒） そこで絞る量の持ち越しを切る

        組があれば、和の大きさを Σk²E + 2Σ(ki kj Cij) で数える（k は組ごとの絞り、E は組の大きさ、
        C は組どうしの掛け合わせ） 混ぜた音へ絞りの比を掛けるだけだと、組ごとに絞りが違うときの
        相殺を読めない（逆相で 2.25 LU 外した PR #495 の CodeRabbit の指摘）
        """
        ratio = 10.0 ** (gain_db / 20.0)
        ceiling = 10.0 ** (ceiling_db / 20.0) / ratio
        if self.lanes:
            count = len(self.energy)
            kept_by = [
                _kept(lane.detected[:count], lane.ceilings(ceiling_db)[:count] / ratio, lane.starts)
                for lane in self.lanes
            ]
            summed = np.zeros(len(self.energy))
            for lane, kept in zip(self.lanes, kept_by, strict=True):
                summed += lane.energy * kept * kept
            for first, second, cross in self.crosses:
                summed += 2.0 * cross * kept_by[first] * kept_by[second]
            # 打ち消し合う所は丸めで負に触れる
            energy = np.maximum(summed, 0.0) * (ratio * ratio)
        else:
            kept = _kept(self.peak, ceiling, starts)
            energy = self.energy * kept * kept * (ratio * ratio)
        ticks = energy.reshape(-1, _FINE_PARTS).sum(axis=1)
        if len(ticks) < 4:
            return None
        return gated_loudness(
            (ticks[:-3] + ticks[1:-2] + ticks[2:-1] + ticks[3:]) / (4 * self.step)
        )


def _kept(peak: np.ndarray, ceiling: float | np.ndarray, starts: Sequence[float]) -> np.ndarray:
    """区切りごとに、リミッターの後へ残る割合 ``starts`` で絞る量の持ち越しを切る"""
    wanted = 1.0 - np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    kept = np.ones(len(wanted))
    if np.any(wanted > 0.0):
        interval = 0.1 / _FINE_PARTS
        release = DEFAULT_RELEASE_MS / 1000.0
        resets = {round(start / interval) for start in starts}
        bounds = sorted({0, len(wanted), *(at for at in resets if 0 < at < len(wanted))})
        for low, high in pairwise(bounds):
            kept[low:high] -= reduction(wanted[low:high], interval, release)
    return kept


class FineSums:
    """サンプルごとの値を、:class:`Dynamics` と同じ境目の 5 ms ごとに畳んで覚える

    ``fold`` は畳み方 和（既定）か、山なら ``np.maximum``
    """

    def __init__(self, step: int, fold: np.ufunc = np.add) -> None:
        self._step = step
        self._fold = fold
        self._cuts = np.arange(_FINE_PARTS) * step // _FINE_PARTS
        self._partial = np.zeros(0)
        self._found: list[np.ndarray] = []

    def add(self, values: np.ndarray) -> None:
        joined = np.concatenate([self._partial, values])
        whole = len(joined) // self._step * self._step
        if whole:
            rows = joined[:whole].reshape(-1, self._step)
            summed = self._fold.reduceat(rows, self._cuts, axis=1)
            self._found.append(summed.ravel().astype(np.float32))
        self._partial = joined[whole:]

    def joined(self) -> np.ndarray:
        if not self._found:
            return np.zeros(0, dtype=np.float32)
        if len(self._found) > 1:
            self._found = [np.concatenate(self._found)]
        return self._found[0]

    def tail(self) -> np.ndarray:
        """100 ms に満たない末尾の分"""
        count = len(self._partial)
        if not count:
            return np.zeros(0, dtype=np.float32)
        cuts = self._cuts[self._cuts < count]
        return np.asarray(self._fold.reduceat(self._partial, cuts), dtype=np.float32)


class LoudnessMeter:
    """音を少しずつ渡して、積分ラウドネスと最大のサンプルを測る

    渡す音は ``(サンプル数, チャンネル数)`` の float フィルタの続きと区切りの端数を
    次の塊へ持ち越すので、どんな長さで渡しても同じ値になる
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        *,
        tap: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        #: K 特性を掛けた音の受け手（組どうしの掛け合わせ）
        self._tap = tap
        self._rate = int(sample_rate)
        self._channels = int(channels)
        self._impulse = _impulse(self._rate)
        self._weights = channel_weights(self._channels)
        taps = len(self._impulse)
        #: 前の塊の終わり（畳み込みの続きに要る分）
        self._tail = np.zeros((taps - 1, self._channels))
        #: 100 ms の区切り 1 つのサンプル数 400 ms の区切りはこれを 4 つ並べた物
        self._step = max(1, self._rate // 10)
        #: まだ 100 ms に満たない、重みを掛けた二乗の端数
        self._partial = np.zeros(0)
        #: 100 ms ごとの、重みを掛けた二乗の和
        self._energies: list[float] = []
        self._peak = 0.0
        #: 100 ms を細かく割る境目と、割った区切りの大きさと山（:meth:`dynamics`）
        self._cuts = np.arange(_FINE_PARTS) * self._step // _FINE_PARTS
        self._partial_peaks = np.zeros(0)
        self._fine_energy: list[np.ndarray] = []
        self._fine_peak: list[np.ndarray] = []
        self._responses: dict[int, np.ndarray] = {}
        #: まだ畳み込んでいない音（:data:`CHUNK_SECONDS` まで溜める）
        self._pending: list[np.ndarray] = []
        self._pending_count = 0

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def peak(self) -> float:
        """これまでの最大のサンプルの大きさ（1.0 がフルスケール）"""
        return self._peak

    def add(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        block = np.asarray(samples, dtype=np.float64).reshape(-1, self._channels)
        self._peak = max(self._peak, float(np.max(np.abs(block))))
        # 溜めてからまとめて畳み込む 1 コマ（1600 サンプル）ずつ畳み込むと、FFT の大きさが
        # 応答の長さ（9600）で決まり、同じ音を何倍も計算する 1 サンプルずつ渡されると何百倍
        self._pending.append(block)
        self._pending_count += len(block)
        if self._pending_count >= self._rate * CHUNK_SECONDS:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        block = np.concatenate(self._pending)
        self._pending = []
        self._pending_count = 0
        weighted = self._filter(block)
        if self._tap is not None:
            self._tap(weighted)
        energy = np.sum(weighted * weighted * self._weights, axis=1)
        joined = np.concatenate([self._partial, energy])
        peaks = np.concatenate([self._partial_peaks, np.abs(block).max(axis=1)])
        whole = len(joined) // self._step * self._step
        if whole:
            ticks = joined[:whole].reshape(-1, self._step)
            self._energies.extend(float(value) for value in ticks.sum(axis=1))
            fine = np.add.reduceat(ticks, self._cuts, axis=1)
            self._fine_energy.append(fine.ravel().astype(np.float32))
            loudest = np.maximum.reduceat(peaks[:whole].reshape(-1, self._step), self._cuts, axis=1)
            self._fine_peak.append(loudest.ravel().astype(np.float32))
        self._partial = joined[whole:]
        self._partial_peaks = peaks[whole:]

    def _filter(self, block: np.ndarray) -> np.ndarray:
        """K 特性を掛ける 前の塊の終わりを頭に付けて畳み込み、頭の分を捨てる"""
        taps = len(self._impulse)
        joined = np.concatenate([self._tail, block])
        count = len(block)
        size = 1 << (count + taps - 1 - 1).bit_length()
        response = self._responses.get(size)
        if response is None:
            # 塊の長さはたいてい毎回同じ 応答の FFT は大きさごとに 1 度だけ作る
            response = np.fft.rfft(self._impulse, size)[:, None]
            self._responses[size] = response
        spectrum = np.fft.rfft(joined, size, axis=0) * response
        filtered = np.fft.irfft(spectrum, size, axis=0)[taps - 1 : taps - 1 + count]
        self._tail = joined[len(joined) - (taps - 1) :]
        return np.asarray(filtered)

    @property
    def block_samples(self) -> int:
        """100 ms の区切り 1 つのサンプル数"""
        return self._step

    def block_energies(self) -> np.ndarray:
        """100 ms の区切りごとの、K 特性と重みを掛けた二乗の和（測り終えた区切りだけ）

        音の推移（#322）はこの並びから瞬時と短時間のラウドネスを出す 同じ K 特性を
        もう 1 度掛けると、長い作品で畳み込みの手間が 2 倍になる
        """
        self._flush()
        return np.asarray(self._energies, dtype=np.float64)

    def dynamics(self) -> Dynamics | None:
        """抑えた後の音量の見積もりに使う手がかり 測れた区切りが無ければ ``None``

        つなぐ手間が作品の長さに比例するので、測り終えてから 1 度だけ呼ぶ
        """
        self._flush()
        if not self._fine_energy:
            return None
        if len(self._fine_energy) > 1:
            # 検査の結果は読まれるたびに組み直される 毎回つなぐと長い作品で画面が引っ掛かる
            self._fine_energy = [np.concatenate(self._fine_energy)]
            self._fine_peak = [np.concatenate(self._fine_peak)]
        return Dynamics(self._fine_energy[0], self._fine_peak[0], self._step)

    def tail_peaks(self) -> np.ndarray:
        """まだ 100 ms に満たない末尾の、5 ms ごとの山 音の山は :attr:`peak` と同じく末尾も数える"""
        self._flush()
        count = len(self._partial_peaks)
        if not count:
            return np.zeros(0, dtype=np.float32)
        cuts = self._cuts[self._cuts < count]
        return np.maximum.reduceat(self._partial_peaks, cuts).astype(np.float32)

    def integrated(self) -> float | None:
        """積分ラウドネス（LUFS） 測れる区切りが無ければ（400 ms に満たない・無音）``None``"""
        energies = self.block_energies()
        if len(energies) < 4:
            return None
        # 400 ms の区切り = 100 ms の区切り 4 つ 区切りごとの平均の二乗
        blocks = (energies[:-3] + energies[1:-2] + energies[2:-1] + energies[3:]) / (4 * self._step)
        return gated_loudness(blocks)


def block_loudness(mean_squares: np.ndarray) -> np.ndarray:
    """区切りの平均の二乗（K 特性と重みを掛けた物）を LUFS にする 無音は ``-inf``"""
    with np.errstate(divide="ignore"):
        return np.asarray(_OFFSET + 10.0 * np.log10(mean_squares))


def gated_loudness(blocks: np.ndarray) -> float | None:
    """400 ms の区切りの平均の二乗から、2 つのゲートを通した積分ラウドネス（LUFS）

    検査（:class:`LoudnessMeter`）と音の推移の道具（#322）が同じ式を通る 別々に書くと、
    同じ作品で検査の値と推移の要約の値が食い違う
    """
    if len(blocks) == 0:
        return None
    levels = block_loudness(blocks)
    kept = blocks[levels > ABSOLUTE_GATE]
    if len(kept) == 0:
        return None
    relative = _OFFSET + 10.0 * math.log10(float(np.mean(kept))) + RELATIVE_GATE
    gated = blocks[(levels > ABSOLUTE_GATE) & (levels > relative)]
    if len(gated) == 0:
        return None
    return _OFFSET + 10.0 * math.log10(float(np.mean(gated)))
