"""クリップの音の雑音の大きさを、波形から見積もる（#336）

AI の道具（``reduce_noise``）が、ノイズ除去を掛けるかどうかと、掛けたときに何が起きるかを
決めるのに使う 取り込みのときに作る波形（:mod:`.waveform` 256 標本ごとの山）だけを読み、
音を読み直さない 長い素材でも一瞬で終わる

見るのは 2 つ

- 静かな所の大きさ（雑音） 21 ms ごとの山の大きさの、小さい方から 10% の所
  話す人は息継ぎや語の切れ目で必ず黙るので、そこに残る音が雑音
- 大きい所の大きさ（声） 同じ山の、大きい方から 5% の所

山（最大の振れ幅）で測るので、実効値より数 dB 大きく出る 比べるのは同じ測り方どうし
（雑音と声の差・決めた線）なので、そろっていればよい
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sashimono.engine.audio.waveform import Waveform

__all__ = [
    "CLOSE_MARGIN_DB",
    "MIN_SECONDS",
    "QUIET_NOISE_DB",
    "NoiseEstimate",
    "estimate_noise",
]

#: これより短い範囲は測らない 黙る所が 1 つも無いことが多く、声を雑音と見誤る
MIN_SECONDS = 1.0
#: 雑音の山がこれより小さければ、聞こえないとみなして下げない（dBFS）
#: 48 kHz の 16 bit の録音で、静かな部屋の何もしていないマイクがおよそ -70 から -60
QUIET_NOISE_DB = -60.0
#: 雑音と声の差がこれより小さければ、静かな所が無いとみなす（dB） 音楽・ずっと続く環境音・
#: 雑音に埋もれた声のどれか 下げると声や音楽まで削れる
CLOSE_MARGIN_DB = 10.0
#: 束ねる長さ（山の数） 256 標本の山 4 つで 48 kHz なら 21 ms ノイズ除去の窓と同じ長さ
_GROUP = 4
#: 小さすぎる値の下限（dBFS） 完全な無音を -inf にしない
_FLOOR_DB = -120.0


@dataclass(frozen=True, slots=True)
class NoiseEstimate:
    """波形から見積もった雑音と声の大きさ（どちらも山の dBFS）"""

    noise_db: float
    speech_db: float
    #: 測った長さ（秒）
    seconds: float

    @property
    def margin_db(self) -> float:
        """声が雑音よりどれだけ大きいか"""
        return self.speech_db - self.noise_db

    @property
    def audible(self) -> bool:
        """雑音が聞こえる大きさか"""
        return self.noise_db > QUIET_NOISE_DB

    @property
    def has_quiet_parts(self) -> bool:
        """黙る所がある（雑音だけの所と声の所が分かれている）か"""
        return self.margin_db >= CLOSE_MARGIN_DB


def estimate_noise(waveform: Waveform, start: Fraction, end: Fraction) -> NoiseEstimate | None:
    """素材の ``[start, end)`` 秒の雑音と声の大きさ 短すぎる・範囲の外なら ``None``"""
    level = waveform.levels[0]
    rate = waveform.sample_rate
    first = max(0, int(start * rate) // level.samples_per_peak)
    last = min(level.count, -(-int(end * rate) // level.samples_per_peak))
    seconds = (last - first) * level.samples_per_peak / max(rate, 1)
    if last <= first or seconds < MIN_SECONDS:
        return None
    peaks = level.peaks[first:last]
    # 左右のどちらかと、上下のどちらかの大きい方 片側だけに乗った雑音も拾う
    amplitude = np.abs(peaks).max(axis=(1, 2))
    groups = len(amplitude) // _GROUP
    if groups == 0:
        return None
    grouped = amplitude[: groups * _GROUP].reshape(groups, _GROUP).max(axis=1)
    noise, speech = np.percentile(grouped, [10.0, 95.0])
    return NoiseEstimate(
        noise_db=round(_decibels(float(noise)), 1),
        speech_db=round(_decibels(float(speech)), 1),
        seconds=round(seconds, 2),
    )


def _decibels(amplitude: float) -> float:
    if amplitude <= 0.0:
        return _FLOOR_DB
    return max(_FLOOR_DB, 20.0 * float(np.log10(amplitude)))
