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
from sashimono.effects.audio import MAX_HISTORY_SECONDS
from sashimono.effects.definition import EffectDefinition
from sashimono.engine.audio.mixer import audio_stack, effect_values

__all__ = ["shape_envelope", "shape_history_frames", "shape_key"]

#: 映す効果 ほかの音の効果は大きさを変えないとみなす
_SHAPED = frozenset({"audio_volume", "audio_fade", "audio_delay", "audio_reverb"})


def shape_history_frames(clip: Clip, rate: FrameRate) -> float:
    """映す効果が前の音から作る形（ディレイのやまびこ・リバーブの尾）の届く長さ（フレーム）

    見えている範囲だけの波形を作るとき、その手前にある元の山の響きも映すのに使う
    手前を読まずに形を掛けると、見える範囲の手前で鳴った音の尾が、スクロールすると
    消えたり出たりした（PR #231 の指摘） 値が動くときはキーフレームの所と両端で
    一番長い物を取る（スクロールの位置で長さが変わると、同じ所の形が変わって見える）
    """
    seconds = 0.0
    for definition, effect in audio_stack(clip):
        if definition.kind not in ("audio_delay", "audio_reverb") or not definition.audio_history:
            continue
        frames = {0, max(clip.duration - 1, 0)}
        for value in effect.params.values():
            if isinstance(value, AnimatedValue):
                frames.update(k.frame for k in value.keyframes if 0 <= k.frame < clip.duration)
        seconds += max(
            definition.audio_history(effect_values(definition, effect, frame)) for frame in frames
        )
    return min(seconds, MAX_HISTORY_SECONDS) * float(rate.fps)


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

        # 動く値は列ごとのコマで解く（音量と同じ） 頭のコマの値だけで描くと、キーフレームで
        # フェードの長さやディレイの量を動かしても、波形が頭の値のまま変わらなかった
        # （PR #231 の指摘） 動かない値は 1 度だけ解く
        if kind == "audio_fade":
            gain = np.ones(columns)
            fade_in = np.maximum(_animated(definition, effect, "fade_in", frames), 0.0)
            fade_out = np.maximum(_animated(definition, effect, "fade_out", frames), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = np.where(fade_in > 0, np.minimum(gain, seconds / fade_in), gain)
                gain = np.where(fade_out > 0, np.minimum(gain, (length - seconds) / fade_out), gain)
            gain = np.clip(gain, 0.0, 1.0)
            low, high = low * gain, high * gain
        elif kind == "audio_delay":
            low, high = _echoed(
                low,
                high,
                _animated(definition, effect, "mix", frames),
                _animated(definition, effect, "feedback", frames),
                _animated(definition, effect, "time", frames),
                per_second,
            )
        else:
            low, high = _tailed(
                low,
                high,
                _animated(definition, effect, "mix", frames),
                _animated(definition, effect, "decay", frames),
                per_second,
            )
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
    low: np.ndarray,
    high: np.ndarray,
    mix_values: np.ndarray,
    feedback_values: np.ndarray,
    time_values: np.ndarray,
    per_second: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ディレイ ずらした波形を小さくして足す（鳴る音と同じく足し合わせる）

    値は列ごと 列に届くやまびこは、その列の値（間隔・量・繰り返し）で前の列から取る
    鳴る音もやまびこの出る所の値で掛けるので、同じ向きに合わせる
    """
    mix = np.clip(mix_values, 0.0, 100.0) / 100.0
    feedback = np.clip(feedback_values, 0.0, 90.0) / 100.0
    step = np.maximum(time_values, 1.0) / 1000.0 * per_second
    count = len(high)
    if not np.any(mix > 0.0) or np.all(step < 0.5):
        return low, high
    out_low, out_high = low.copy(), high.copy()
    index = np.arange(count)
    gain = mix.copy()
    shift = step.copy()
    usable = step >= 0.5
    while True:
        alive = usable & (gain > 0.001) & (np.round(shift) < count)
        if not np.any(alive):
            break
        source = index - np.round(shift).astype(np.int64)
        alive &= source >= 0
        picked = np.clip(source, 0, count - 1)
        out_low += np.where(alive, low[picked] * gain, 0.0)
        out_high += np.where(alive, high[picked] * gain, 0.0)
        gain = gain * feedback
        shift = shift + step
    return out_low, out_high


def _tailed(
    low: np.ndarray,
    high: np.ndarray,
    mix_values: np.ndarray,
    decay_values: np.ndarray,
    per_second: float,
) -> tuple[np.ndarray, np.ndarray]:
    """リバーブ ピークを残響の長さで 60 dB 減る尾として後ろへ伸ばし、量だけ足す

    尾は「前のどこかのピークが今まで減った値」の一番大きい物 対数にして積み上げの
    最大を取れば、列を Python で回さずに求まる 長さが列ごとに違うときは、出てくる
    長さ（0.1 秒刻み 設定の刻みと同じ）ごとに尾を作り、列ごとにその列の長さの尾を使う
    """
    mix = np.clip(mix_values, 0.0, 100.0) / 100.0
    if not np.any(mix > 0.0):
        return low, high
    decay = np.round(np.clip(decay_values, 0.1, 5.0), 1)
    index = np.arange(len(high), dtype=np.float64)
    level = np.maximum(np.abs(low), np.abs(high))
    logged_level = np.log(np.maximum(level, 1e-9))
    tail = np.zeros(len(high))
    for length in np.unique(decay):
        fall = 6.91 / (float(length) * per_second)
        logged = logged_level + index * fall
        reached = np.exp(np.maximum.accumulate(logged) - index * fall)
        tail = np.where(decay == length, reached, tail)
    tail = tail * mix
    return low - tail, high + tail
