"""音を加工するエフェクト AviUtl の音声フィルタに当たるもの

映像のエフェクトはシェーダで書くが、音は GPU を通さないので
:attr:`~sashimono.effects.definition.EffectDefinition.audio_process` に
関数を入れる 仕組みは :mod:`sashimono.engine.audio.mixer` が呼ぶ

項目名は AviUtl2 に音声ファイルを置いてフィルタを積み、エイリアスを作らせて
読み取った（推測していない） AviUtl2 v2.1.6a の音声フィルタは
``音量フェード`` ``モノラル化`` ``音量調整`` の **3 つだけ**
（``音声波形表示`` は音ではなく絵を描くので映像の側にある）

サンプルは ``(数, チャンネル)`` の float32 で、-1..1 に収まっているとは限らない
（混ぜた後に整える） 返す形も同じにする
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.spec import TrackSpec

__all__ = ["AudioContext", "register_audio_effects"]


@dataclass(frozen=True, slots=True)
class AudioContext:
    """音のエフェクトへ渡す、時間まわりの手がかり

    ``offset`` は**クリップ先頭からのサンプル位置** フェードのように
    位置で効き方が変わるものは、これと ``duration`` から進み具合を出す
    """

    #: クリップ先頭から数えた、この塊の先頭のサンプル位置
    offset: int
    #: 1 秒あたりのサンプル数
    sample_rate: int
    #: クリップの長さ（サンプル）
    duration: int


def _volume(samples: np.ndarray, values: dict[str, float], _context: AudioContext) -> np.ndarray:
    """音量調整 ``音量`` は %（100 で元のまま） ``左右`` は -100..100"""
    gain = max(values.get("volume", 100.0), 0.0) / 100.0
    out = samples * gain
    return _panned(out, values.get("pan", 0.0) / 100.0)


def _fade(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """音量フェード 端の ``イン`` ``アウト`` 秒で 0 から 1 へ

    位置ごとに掛ける量が変わるので、塊の中でもサンプルごとに計算する
    塊の先頭の値だけで掛けると、塊の境目で音が階段状に変わる
    """
    rate = max(context.sample_rate, 1)
    fade_in = max(values.get("fade_in", 0.0), 0.0) * rate
    fade_out = max(values.get("fade_out", 0.0), 0.0) * rate
    # 位置は整数で持つ float32 だと 1677 万（48 kHz で 6 分弱）を超えた辺りから
    # 1 サンプル単位を表せなくなり、長いクリップでフェードが階段状になる
    index = np.arange(len(samples), dtype=np.int64) + context.offset

    gain = np.ones(len(samples), dtype=np.float64)
    if fade_in > 0.0:
        gain = np.minimum(gain, index / fade_in)
    if fade_out > 0.0:
        left = context.duration - index
        gain = np.minimum(gain, left / fade_out)
    return samples * np.clip(gain, 0.0, 1.0).astype(np.float32)[:, None]


def _monaural(samples: np.ndarray, values: dict[str, float], _context: AudioContext) -> np.ndarray:
    """モノラル化 ``比率`` は左右を混ぜる量（0 で混ぜない 100 で完全にモノラル）

    AviUtl の ``比率`` は **0 が元のまま** 逆に読むと、既定のままで
    ステレオが潰れる
    """
    amount = np.clip(values.get("ratio", 0.0) / 100.0, 0.0, 1.0)
    if amount <= 0.0 or samples.shape[1] < 2:
        return samples
    middle = samples.mean(axis=1, keepdims=True)
    mixed: np.ndarray = samples * (1.0 - amount) + middle * amount
    return mixed


#: 前の音を読み直す長さの上限（秒） 長い残響や繰り返しの多いディレイでも、1 回に読む量を抑える
MAX_HISTORY_SECONDS = 10.0
#: 音程を変えるときの窓（秒） 短いほど遅れが小さく、長いほど低い音がうなりにくい
PITCH_WINDOW_SECONDS = 0.05
#: ディレイの繰り返しをここまで小さくなったら打ち切る（-60 dB）
_DELAY_FLOOR = 0.001


def _reverb_length(values: dict[str, float]) -> float:
    return float(np.clip(values.get("decay", 1.5), 0.1, 5.0))


@lru_cache(maxsize=16)
def _impulse(sample_rate: int, decay: float, channels: int) -> np.ndarray:
    """残響の響き方（長さ ``decay`` 秒で 60 dB 下がる雑音） 左右で別の乱数にして広がりを出す

    乱数の種は決めておく 書き出しとプレビュー、塊の切り方が違っても同じ響きになる
    エネルギーを 1 にそろえ、長さを変えても響きの大きさが大きく変わらないようにする
    """
    length = max(1, int(decay * sample_rate))
    rng = np.random.default_rng(20260930)
    envelope = np.exp(-6.91 * np.arange(length) / length)
    noise = rng.standard_normal((length, channels)) * envelope[:, None]
    noise /= np.sqrt(np.sum(noise**2, axis=0, keepdims=True)) + 1e-12
    return noise.astype(np.float32)


def _reverb(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """リバーブ 元の音に、響き（:func:`_impulse`）を畳み込んだ音を ``量`` だけ足す

    前の音は呼ぶ側が読み直して渡す（:attr:`EffectDefinition.audio_history`） 塊の頭より前は
    無音として数える
    """
    mix = float(np.clip(values.get("mix", 30.0), 0.0, 100.0)) / 100.0
    if mix <= 0.0 or len(samples) == 0:
        return samples
    impulse = _impulse(context.sample_rate, _reverb_length(values), samples.shape[1])
    size = len(samples) + len(impulse) - 1
    fft_size = 1 << (size - 1).bit_length()
    spectrum = np.fft.rfft(samples, fft_size, axis=0) * np.fft.rfft(impulse, fft_size, axis=0)
    wet = np.fft.irfft(spectrum, fft_size, axis=0)[: len(samples)]
    return np.asarray(samples + wet * mix, dtype=np.float32)


def _delay_repeats(values: dict[str, float]) -> int:
    feedback = float(np.clip(values.get("feedback", 40.0), 0.0, 90.0)) / 100.0
    if feedback <= 0.0:
        return 1
    return int(min(50, 1 + np.ceil(np.log(_DELAY_FLOOR) / np.log(feedback))))


def _delay_history(values: dict[str, float]) -> float:
    return max(values.get("time", 250.0), 1.0) / 1000.0 * _delay_repeats(values)


def _delay(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """ディレイ ``時間`` ごとに、``繰り返し`` の割合で小さくなるやまびこを ``量`` だけ足す"""
    mix = float(np.clip(values.get("mix", 50.0), 0.0, 100.0)) / 100.0
    step = round(max(values.get("time", 250.0), 1.0) / 1000.0 * context.sample_rate)
    if mix <= 0.0 or step <= 0:
        return samples
    feedback = float(np.clip(values.get("feedback", 40.0), 0.0, 90.0)) / 100.0
    out = samples.astype(np.float32, copy=True)
    gain = mix
    for repeat in range(1, _delay_repeats(values) + 1):
        shift = step * repeat
        if shift >= len(samples):
            break
        out[shift:] += samples[:-shift] * gain
        gain *= feedback
    return out


def _pitch(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """音程を半音で変える（長さは変えない）

    読む位置を少しずつ遅らせる 2 本の読み口を窓の半分ずらして重ねる（ドップラーのやり方）
    読む位置の遅れは**クリップの頭から数えた位置**だけで決まるので、塊の切り方が違っても
    同じ音になる 前の音は窓の長さだけ要る（呼ぶ側が読み直して渡す） 遅れは平均で窓の半分
    """
    semitones = float(np.clip(values.get("semitones", 0.0), -24.0, 24.0))
    if semitones == 0.0 or len(samples) == 0:
        return samples
    ratio = 2.0 ** (semitones / 12.0)
    window = max(8, int(PITCH_WINDOW_SECONDS * context.sample_rate))
    positions = np.arange(len(samples), dtype=np.float64)
    absolute = positions + context.offset
    phase = np.mod(absolute * (1.0 - ratio) / window, 1.0)
    out = np.zeros_like(samples, dtype=np.float32)
    for shift in (0.0, 0.5):
        tap = np.mod(phase + shift, 1.0)
        read = positions - tap * window
        left = np.floor(read).astype(np.int64)
        weight = (read - left).astype(np.float32)[:, None]
        valid = (left >= 0)[:, None]
        first = samples[np.clip(left, 0, len(samples) - 1)] * valid
        second = samples[np.clip(left + 1, 0, len(samples) - 1)] * ((left + 1) >= 0)[:, None]
        value = first * (1.0 - weight) + second * weight
        out += value * (np.sin(np.pi * tap) ** 2).astype(np.float32)[:, None]
    return out


def _panned(samples: np.ndarray, pan: float) -> np.ndarray:
    """左右の振り分け -1 で左だけ 1 で右だけ

    片側を絞るだけにする 反対側を持ち上げると、真ん中に寄せた音が大きくなる
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    if pan == 0.0 or samples.shape[1] != 2:
        return samples
    out = samples.copy()
    if pan > 0.0:
        out[:, 0] *= 1.0 - pan
    else:
        out[:, 1] *= 1.0 + pan
    return out


def register_audio_effects() -> None:
    definitions = (
        EffectDefinition(
            kind="audio_volume",
            label="音量調整",
            category="音",
            parameters=(
                TrackSpec("volume", "音量", 0, 400, 100, unit="%"),
                TrackSpec("pan", "左右", -100, 100, 0, unit="%"),
            ),
            audio_process=_volume,
        ),
        EffectDefinition(
            kind="audio_fade",
            label="音量フェード",
            category="音",
            parameters=(
                TrackSpec("fade_in", "イン", 0, 60, 0, unit="秒"),
                TrackSpec("fade_out", "アウト", 0, 60, 0, unit="秒"),
            ),
            audio_process=_fade,
        ),
        EffectDefinition(
            kind="audio_monaural",
            label="モノラル化",
            category="音",
            parameters=(TrackSpec("ratio", "比率", 0, 100, 0, unit="%"),),
            audio_process=_monaural,
        ),
        # ここから下は AviUtl2 に無い物（利用者の要望） 値は塊の頭の値で 1 塊ぶん掛ける
        EffectDefinition(
            kind="audio_reverb",
            label="リバーブ",
            category="音",
            parameters=(
                TrackSpec("decay", "残響の長さ", 0.1, 5, 1.5, step=0.1, unit="秒"),
                TrackSpec("mix", "残響の量", 0, 100, 30, unit="%"),
            ),
            audio_process=_reverb,
            audio_history=_reverb_length,
        ),
        EffectDefinition(
            kind="audio_delay",
            label="ディレイ",
            category="音",
            parameters=(
                TrackSpec("time", "間隔", 1, 2000, 250, step=1, unit="ミリ秒"),
                TrackSpec("feedback", "繰り返し", 0, 90, 40, unit="%"),
                TrackSpec("mix", "やまびこの量", 0, 100, 50, unit="%"),
            ),
            audio_process=_delay,
            audio_history=_delay_history,
        ),
        EffectDefinition(
            kind="audio_pitch",
            label="音程の調整",
            category="音",
            parameters=(TrackSpec("semitones", "音程", -24, 24, 0, step=0.1, unit="半音"),),
            audio_process=_pitch,
            audio_history=lambda _values: PITCH_WINDOW_SECONDS,
        ),
    )
    for definition in definitions:
        registry.register(definition)
