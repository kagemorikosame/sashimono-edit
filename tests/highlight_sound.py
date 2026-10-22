"""盛り上がりの試験（#325）に使う、作った音と波形

笑い・話し声・一定の拍の音楽・叫びを、決まった乱数で作る 実際の素材は配布物なので
リポジトリに入れられず、形の分かった音で見分けの境目を押さえる
"""

from __future__ import annotations

import numpy as np

from sashimono.engine.audio.waveform import BASE_SAMPLES_PER_PEAK, PeakLevel, Waveform

RATE = 48000


def voiced(seconds: float, amp: float, pitch: float = 220.0) -> np.ndarray:
    """声に似た音（基音と 2 倍音） 端は 20 ms で立ち上げて落とす"""
    t = np.arange(round(seconds * RATE)) / RATE
    tone = np.sin(2 * np.pi * pitch * t) + 0.5 * np.sin(2 * np.pi * pitch * 2 * t)
    ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)
    out: np.ndarray = amp * tone * ramp / 1.5
    return out


def quiet(seconds: float, rng: np.random.Generator, amp: float = 0.002) -> np.ndarray:
    """部屋の雑音"""
    out: np.ndarray = amp * rng.standard_normal(round(seconds * RATE))
    return out


def laugh(
    seconds: float, rng: np.random.Generator, *, rate: float = 5.0, amp: float = 0.5
) -> np.ndarray:
    """笑いに似た音 1 秒に ``rate`` 回、同じ間隔で「は」を吐き、合間は息だけになる"""
    parts = []
    period = 1.0 / rate
    for _ in range(round(seconds * rate)):
        on = period * 0.55
        parts.append(voiced(on, amp, 260 + rng.uniform(-20, 20)))
        parts.append(quiet(period - on, rng, 0.01))
    return np.concatenate(parts)


def speech(seconds: float, rng: np.random.Generator, *, amp: float = 0.2) -> np.ndarray:
    """話し声に似た音 音節の長さ（0.08〜0.35 秒）と合間がばらつく"""
    parts = []
    total = 0.0
    while total < seconds:
        syllable = float(rng.uniform(0.08, 0.35))
        short = float(rng.uniform(0.0, 0.05))
        pause = float(rng.uniform(0.15, 0.5))
        gap = short if rng.random() < 0.75 else pause
        parts.append(
            voiced(syllable, amp * float(rng.uniform(0.5, 1.0)), 150 + rng.uniform(-30, 30))
        )
        parts.append(quiet(gap, rng, 0.004))
        total += syllable + gap
    return np.concatenate(parts)[: round(seconds * RATE)]


def beat(seconds: float, *, rate: float = 4.0, amp: float = 0.3) -> np.ndarray:
    """一定の拍の音楽 低い持続音の上に 1 秒 ``rate`` 回の打音"""
    t = np.arange(round(seconds * RATE)) / RATE
    bed = 0.05 * np.sin(2 * np.pi * 110 * t)
    hits = amp * np.exp(-((t % (1 / rate)) / 0.04)) * np.sin(2 * np.pi * 80 * t)
    out: np.ndarray = bed + hits
    return out


def waveform_of(samples: np.ndarray) -> Waveform:
    """取り込みの解析（:func:`~sashimono.engine.audio.waveform.analyze_waveform`）と同じ形の波形"""
    count = -(-samples.shape[0] // BASE_SAMPLES_PER_PEAK) * BASE_SAMPLES_PER_PEAK
    padded = np.zeros(count, np.float32)
    padded[: samples.shape[0]] = samples
    grouped = padded.reshape(-1, BASE_SAMPLES_PER_PEAK)
    peaks = np.empty((grouped.shape[0], 2, 2), np.float32)
    peaks[:, :, 0] = grouped.min(axis=1)[:, None]
    peaks[:, :, 1] = grouped.max(axis=1)[:, None]
    return Waveform(RATE, 2, samples.shape[0], (PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),))
