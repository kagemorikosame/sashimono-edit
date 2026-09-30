"""タイムラインの波形に、クリップの音の効果を大まかに映す

波形は素材を解析したピーク（:class:`~sashimono.engine.audio.Waveform`）で、音量を下げても
リバーブを掛けても形が変わらなかった（利用者の要望） 鳴らす音を列ごとに作り直すと重いので、
列ごとの最小・最大へ効き方だけを掛ける

- 音量調整・音量フェード・トラックの音量は、列の時刻の倍率を掛ける
- ディレイは、ずらした波形を小さくして重ねる
- リバーブは、ピークを残響の長さで減っていく尾として後ろへ伸ばす
- モノラル化・音程の調整は大きさをほぼ変えないので映さない 左右の振り分け（パン）も
  左右をまとめた 1 本の波形では見えないので映さない

画面に出すだけの目安 鳴る音は :mod:`sashimono.engine.audio.mixer` が決める
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np

from sashimono.core.model import AnimatedValue, Clip, Effect
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import EffectDefinition
from sashimono.engine.audio.mixer import audio_stack, effect_values

__all__ = ["shape_envelope", "shape_key"]

#: 映す効果 ほかの音の効果は大きさを変えないとみなす
_SHAPED = frozenset({"audio_volume", "audio_fade", "audio_delay", "audio_reverb"})


def shape_key(clip: Clip, track_gain: float = 1.0) -> Hashable:
    """波形の画像を貯める鍵に足す値 効き方が変われば変わる 何も映さなければ ``None``"""
    parts = tuple(
        (effect.kind, tuple(sorted((name, repr(value)) for name, value in effect.params.items())))
        for definition, effect in audio_stack(clip)
        if definition.kind in _SHAPED
    )
    if not parts and track_gain == 1.0:
        return None
    return (parts, track_gain, clip.duration)


def shape_envelope(
    minimum: np.ndarray,
    maximum: np.ndarray,
    clip: Clip,
    rate: FrameRate,
    first: float,
    last: float,
    *,
    track_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """列ごとの最小・最大に、クリップの音の効果を積んだ順に掛けた物

    ``first`` ``last`` は列の並びの両端が指す、クリップの頭から数えたフレーム
    """
    columns = len(maximum)
    if columns == 0:
        return minimum, maximum
    low = minimum.astype(np.float64)
    high = maximum.astype(np.float64)
    fps = float(rate.fps)
    span = max(last - first, 1e-9)
    frames = first + (np.arange(columns) + 0.5) * span / columns
    seconds = frames / fps
    length = clip.duration / fps
    per_second = columns / (span / fps)
    for definition, effect in audio_stack(clip):
        kind = definition.kind
        if kind not in _SHAPED:
            continue
        if kind == "audio_volume":
            gain = _animated(definition, effect, "volume", frames) / 100.0
            gain = np.maximum(gain, 0.0)
            low, high = low * gain, high * gain
            continue
        values = effect_values(definition, effect, int(first))
        if kind == "audio_fade":
            gain = np.ones(columns)
            fade_in = max(values.get("fade_in", 0.0), 0.0)
            fade_out = max(values.get("fade_out", 0.0), 0.0)
            if fade_in > 0:
                gain = np.minimum(gain, seconds / fade_in)
            if fade_out > 0:
                gain = np.minimum(gain, (length - seconds) / fade_out)
            gain = np.clip(gain, 0.0, 1.0)
            low, high = low * gain, high * gain
        elif kind == "audio_delay":
            low, high = _echoed(low, high, values, per_second)
        else:
            low, high = _tailed(low, high, values, per_second)
    if track_gain != 1.0:
        low, high = low * track_gain, high * track_gain
    return low.astype(np.float32), high.astype(np.float32)


def _animated(
    definition: EffectDefinition, effect: Effect, name: str, frames: np.ndarray
) -> np.ndarray:
    """動く値を列の時刻で解く 動かない値は 1 度だけ解く（列ごとに解くと 8000 列で重い）"""
    raw = effect.params.get(name)
    if not isinstance(raw, AnimatedValue) or not raw.keyframes:
        return np.full(len(frames), effect_values(definition, effect, 0).get(name, 100.0))
    whole = np.floor(frames).astype(np.int64)
    unique, inverse = np.unique(whole, return_inverse=True)
    solved = np.array([effect_values(definition, effect, int(f)).get(name, 100.0) for f in unique])
    return np.asarray(solved[inverse], dtype=np.float64)


def _echoed(
    low: np.ndarray, high: np.ndarray, values: dict[str, float], per_second: float
) -> tuple[np.ndarray, np.ndarray]:
    """ディレイ ずらした波形を小さくして足す（鳴る音と同じく足し合わせる）"""
    mix = float(np.clip(values.get("mix", 50.0), 0.0, 100.0)) / 100.0
    feedback = float(np.clip(values.get("feedback", 40.0), 0.0, 90.0)) / 100.0
    step = max(values.get("time", 250.0), 1.0) / 1000.0 * per_second
    if mix <= 0.0 or step < 0.5:
        return low, high
    out_low, out_high = low.copy(), high.copy()
    gain = mix
    shift = step
    while gain > 0.001 and round(shift) < len(high):
        moved = round(shift)
        out_low[moved:] += low[:-moved] * gain
        out_high[moved:] += high[:-moved] * gain
        gain *= feedback
        shift += step
    return out_low, out_high


def _tailed(
    low: np.ndarray, high: np.ndarray, values: dict[str, float], per_second: float
) -> tuple[np.ndarray, np.ndarray]:
    """リバーブ ピークを残響の長さで 60 dB 減る尾として後ろへ伸ばし、量だけ足す

    尾は「前のどこかのピークが今まで減った値」の一番大きい物 対数にして積み上げの
    最大を取れば、列を Python で回さずに求まる
    """
    mix = float(np.clip(values.get("mix", 30.0), 0.0, 100.0)) / 100.0
    decay = float(np.clip(values.get("decay", 1.5), 0.1, 5.0))
    if mix <= 0.0:
        return low, high
    fall = 6.91 / (decay * per_second)
    index = np.arange(len(high), dtype=np.float64)
    level = np.maximum(np.abs(low), np.abs(high))
    logged = np.log(np.maximum(level, 1e-9)) + index * fall
    tail = np.exp(np.maximum.accumulate(logged) - index * fall) * mix
    return low - tail, high + tail
