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
from functools import lru_cache

import numpy as np

__all__ = [
    "ABSOLUTE_GATE",
    "LoudnessMeter",
    "channel_weights",
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


class LoudnessMeter:
    """音を少しずつ渡して、積分ラウドネスと最大のサンプルを測る

    渡す音は ``(サンプル数, チャンネル数)`` の float フィルタの続きと区切りの端数を
    次の塊へ持ち越すので、どんな長さで渡しても同じ値になる
    """

    def __init__(self, sample_rate: int, channels: int) -> None:
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
        energy = np.sum(weighted * weighted * self._weights, axis=1)
        joined = np.concatenate([self._partial, energy])
        whole = len(joined) // self._step * self._step
        if whole:
            sums = joined[:whole].reshape(-1, self._step).sum(axis=1)
            self._energies.extend(float(value) for value in sums)
        self._partial = joined[whole:]

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

    def integrated(self) -> float | None:
        """積分ラウドネス（LUFS） 測れる区切りが無ければ（400 ms に満たない・無音）``None``"""
        self._flush()
        energies = np.asarray(self._energies)
        if len(energies) < 4:
            return None
        # 400 ms の区切り = 100 ms の区切り 4 つ 区切りごとの平均の二乗
        blocks = (energies[:-3] + energies[1:-2] + energies[2:-1] + energies[3:]) / (4 * self._step)
        with np.errstate(divide="ignore"):
            levels = _OFFSET + 10.0 * np.log10(blocks)
        kept = blocks[levels > ABSOLUTE_GATE]
        if len(kept) == 0:
            return None
        relative = _OFFSET + 10.0 * math.log10(float(np.mean(kept))) + RELATIVE_GATE
        gated = blocks[(levels > ABSOLUTE_GATE) & (levels > relative)]
        if len(gated) == 0:
            return None
        return _OFFSET + 10.0 * math.log10(float(np.mean(gated)))
