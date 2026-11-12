"""音の山を上限で抑えるリミッター（#414）

通しの確かめで、作品の音量が配信の基準（-14 LUFS）に 3.8〜6 LU 届かなかった 全部のクリップへ
同じだけ掛ける ``normalize_loudness`` は、一瞬の山が天井（-1 dBFS）に当たった所で止まるため
山だけを抑えれば、平均の大きさ（ラウドネス）を上げられる

**先の音を読まない** ミキサは前の音しか読み直さない（:attr:`EffectDefinition.audio_history`）
先読みのあるリミッター（山の手前から絞り始める物）は、出す音を遅らせないと作れず、遅らせると
同じクリップの絵と音がずれる ここでは山の標本で即座に絞り、そこから ``release`` で戻す
上限を越える標本は 1 つも出ない（掛ける量は標本ごとに上限から決める）

戻り方は指数 ある標本の絞る量は、それまでの標本の絞りたい量を経過時間で減らした物の最大
指数の最大は対数にすると累積の最大（``np.maximum.accumulate``）で 1 度に求まるので、標本ごとの
繰り返しを Python で回さずに済む（48 kHz の 0.34 秒の区切りで 1 ms ほど）
前の音は ``release`` の :data:`_HISTORY_TIMES` 倍だけ読む それより前の山の名残は e の -10 乗
（約 0.005%）で、聞き分けられない 塊の切り方が違っても同じ音になる
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sashimono.effects.audio import AudioContext

__all__ = [
    "CEILING_DB_RANGE",
    "DEFAULT_CEILING_DB",
    "DEFAULT_RELEASE_MS",
    "RELEASE_MS_RANGE",
    "limit",
    "limiter_history",
]

#: 上限（dBFS）の範囲と既定 既定は配信の多くが求める -1 dBTP（音量をそろえる道具の天井と同じ）
CEILING_DB_RANGE = (-20.0, 0.0)
DEFAULT_CEILING_DB = -1.0
#: 戻る速さ（ミリ秒） 短いと話し声の 1 音ごとに音量が揺れて濁り（歪み）、長いと山の後の言葉まで
#: 小さく聞こえる 話し声で山の後の音節（100〜200 ms）を潰さない長さを既定にした
RELEASE_MS_RANGE = (10.0, 1000.0)
DEFAULT_RELEASE_MS = 150.0
#: 前の音を読む長さ（戻る速さの何倍） 名残が e の -10 乗まで減る所
_HISTORY_TIMES = 10.0


def _release_seconds(values: dict[str, float]) -> float:
    low, high = RELEASE_MS_RANGE
    return float(np.clip(values.get("release", DEFAULT_RELEASE_MS), low, high)) / 1000.0


def limiter_history(values: dict[str, float]) -> float:
    """前の音を読む長さ（秒）"""
    return _release_seconds(values) * _HISTORY_TIMES


def limit(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """``ceiling`` dBFS を越える山を絞る 左右は同じだけ絞る（定位を動かさない）"""
    if len(samples) == 0:
        return samples
    low, high = CEILING_DB_RANGE
    ceiling = 10.0 ** (float(np.clip(values.get("ceiling", DEFAULT_CEILING_DB), low, high)) / 20.0)
    peaks = np.abs(samples).max(axis=1).astype(np.float64)
    # 標本ごとの絞りたい量（0 は絞らない） 上限より大きい標本だけが正になる
    wanted = 1.0 - np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    if not np.any(wanted > 0.0):
        return samples
    keep = min(max(context.keep_from, 0), len(samples))
    step = -1.0 / (_release_seconds(values) * max(context.sample_rate, 1))
    index = np.arange(len(samples), dtype=np.float64)
    # 絞る量 h[n] = max(wanted[n], h[n-1] * a) a = e^step
    # = a^n * max_{k<=n}(wanted[k] * a^-k) を対数で求める 0 は -inf で最大に効かない
    with np.errstate(divide="ignore"):
        logs = np.log(wanted) - index * step
    reduction = np.exp(np.maximum.accumulate(logs) + index * step)
    gain = (1.0 - reduction).astype(np.float32)
    out = samples * gain[:, None]
    if keep:
        # 前の音として読んだだけの所は使われない 長さだけそろえる
        out[:keep] = 0.0
    return np.asarray(out, dtype=np.float32)
