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

from dataclasses import dataclass, field
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
    #: 渡した音のうち、この位置（渡した音の頭から数える）より前の出力は使われない
    #: 前の音として読むだけの所 長い前の音を読む物（ディレイ）は、ここから先だけを
    #: 計算してよい 0 なら全部を使う
    keep_from: int = 0
    #: 同じ音へ値だけ変えて続けて掛けるとき（区切りの中で値をつなぐ）に、値に依らない
    #: 途中の結果（リバーブの元の音の周波数）を置いておく入れ物 呼ぶ側が 1 組ごとに渡す
    #: ``None`` なら置かない
    shared: dict[str, object] | None = field(default=None, compare=False)


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


#: 前の音を読み直す長さの上限（秒） 読む長さは値から決め（ディレイ 1 本なら最も長くて
#: 間隔 2 秒・繰り返し 90%・量 100% の 132 秒）、これは何本も積んだときに読む量と覚えておく
#: 量が際限なく増えないための歯止め 前は 10 秒で切っていて、2000ms・90% のやまびこが
#: 10 秒を過ぎた所で、まだ聞こえるのに途切れた（PR #231 の指摘）
MAX_HISTORY_SECONDS = 300.0
#: 音程を変えるときの窓（秒） 短いほど遅れが小さく、長いほど低い音がうなりにくい
PITCH_WINDOW_SECONDS = 0.05
#: ディレイの繰り返しをここまで小さくなったら打ち切る（-60 dB）
_DELAY_FLOOR = 0.001


def _reverb_length(values: dict[str, float]) -> float:
    """残響の長さ（秒） 0.1 秒刻み（設定の刻みと同じ）に丸める

    キーフレームで動かすと区切りごとに長さが変わり、そのたびに響き方（24 万サンプルの
    雑音）を作り直して 1 区切り 60ms 掛かった 刻みに丸めれば作り直すのは刻みが変わる
    所だけで、作った物も :func:`_impulse` の貯めに残る
    """
    return round(float(np.clip(values.get("decay", 1.5), 0.1, 5.0)), 1)


@lru_cache(maxsize=4)
def _noise(sample_rate: int, channels: int) -> np.ndarray:
    """響きの元の雑音（最も長い 5 秒ぶん） 乱数の種は決めておく

    長さごとに乱数を引き直すと、キーフレームで長さを動かしたとき 24 万サンプルの乱数を
    刻みごとに作り直すことになる 同じ種で引いた列の頭は長さに依らず同じなので、1 度だけ
    引いて頭から切り出せば、長さごとに引いたときと同じ響きになる
    """
    rng = np.random.default_rng(20260930)
    return rng.standard_normal((int(5.0 * sample_rate), channels))


@lru_cache(maxsize=16)
def _impulse(sample_rate: int, decay: float, channels: int) -> np.ndarray:
    """残響の響き方（長さ ``decay`` 秒で 60 dB 下がる雑音） 左右で別の乱数にして広がりを出す

    乱数の種は決めておく 書き出しとプレビュー、塊の切り方が違っても同じ響きになる
    エネルギーを 1 にそろえ、長さを変えても響きの大きさが大きく変わらないようにする
    """
    length = max(1, int(decay * sample_rate))
    base = _noise(sample_rate, channels)
    if length > len(base):
        base = np.random.default_rng(20260930).standard_normal((length, channels))
    envelope = np.exp(-6.91 * np.arange(length) / length)
    noise = base[:length] * envelope[:, None]
    noise /= np.sqrt(np.sum(noise**2, axis=0, keepdims=True)) + 1e-12
    return noise.astype(np.float32)


@lru_cache(maxsize=16)
def _impulse_spectrum(sample_rate: int, decay: float, channels: int, fft_size: int) -> np.ndarray:
    """響き方の周波数の形 塊ごとに求め直すと、残響 1 本で 1 塊の手間が 1.5 倍になる"""
    return np.fft.rfft(_impulse(sample_rate, decay, channels), fft_size, axis=0)


def _reverb(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """リバーブ 元の音に、響き（:func:`_impulse`）を畳み込んだ音を ``量`` だけ足す

    前の音は呼ぶ側が読み直して渡す（:attr:`EffectDefinition.audio_history`） 塊の頭より前は
    無音として数える ``keep_from`` から先だけが要るときは、そこから響きの長さぶん前からを
    輪で畳み込み（overlap-save）、回り込みの無い所だけを使う 全体を畳み込むより FFT が
    半分の長さで済む（長い残響で 1 区切りの手間が予算を越えていた）
    """
    mix = float(np.clip(values.get("mix", 30.0), 0.0, 100.0)) / 100.0
    if mix <= 0.0 or len(samples) == 0:
        return samples
    decay = _reverb_length(values)
    length = len(_impulse(context.sample_rate, decay, samples.shape[1]))
    keep = min(max(context.keep_from, 0), len(samples))
    if keep >= length - 1 and keep < len(samples):
        begin = keep - (length - 1)
        segment = samples[begin:]
        fft_size = 1 << (len(segment) - 1).bit_length()
        response = _impulse_spectrum(context.sample_rate, decay, samples.shape[1], fft_size)
        spectrum = _input_spectrum(segment, fft_size, (begin, fft_size), context) * response
        wet = np.fft.irfft(spectrum, fft_size, axis=0)[length - 1 : len(segment)]
        out = np.zeros(samples.shape, dtype=np.float32)
        out[keep:] = samples[keep:] + wet * mix
        return out
    size = len(samples) + length - 1
    fft_size = 1 << (size - 1).bit_length()
    response = _impulse_spectrum(context.sample_rate, decay, samples.shape[1], fft_size)
    spectrum = _input_spectrum(samples, fft_size, (0, fft_size), context) * response
    wet = np.fft.irfft(spectrum, fft_size, axis=0)[: len(samples)]
    return np.asarray(samples + wet * mix, dtype=np.float32)


def _input_spectrum(
    samples: np.ndarray, fft_size: int, key: tuple[int, int], context: AudioContext
) -> np.ndarray:
    """畳み込む元の音の周波数 値だけ変えて同じ音へもう 1 度掛けるときは使い回す

    区切りの中で値をつなぐとき（:attr:`AudioContext.shared`）、残響の長さが変わると
    畳み込みを 2 度する 元の音の FFT は値に依らないので 1 度で済ませる
    """
    shared = context.shared
    name = f"reverb-input-{key[0]}-{key[1]}"
    if shared is not None:
        found = shared.get(name)
        if isinstance(found, np.ndarray):
            return found
    spectrum = np.fft.rfft(samples, fft_size, axis=0)
    if shared is not None:
        shared[name] = spectrum
    return spectrum


def _delay_repeats(values: dict[str, float]) -> int:
    """聞こえる（-60 dB より大きい）やまびこの数 量も含めて数える（波形の表示と同じ決まり）

    前は 50 回で切っていて、繰り返しの多い設定ではまだ聞こえる所で途切れた
    """
    mix = float(np.clip(values.get("mix", 50.0), 0.0, 100.0)) / 100.0
    feedback = float(np.clip(values.get("feedback", 40.0), 0.0, 90.0)) / 100.0
    if feedback <= 0.0 or mix <= _DELAY_FLOOR:
        return 1
    # mix * feedback ** (r - 1) > floor を満たす最大の r
    repeats = 1 + int(np.floor(np.log(_DELAY_FLOOR / mix) / np.log(feedback) - 1e-9))
    return max(1, repeats)


def _delay_history(values: dict[str, float]) -> float:
    return max(values.get("time", 250.0), 1.0) / 1000.0 * _delay_repeats(values)


def _delay(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """ディレイ ``時間`` ごとに、``繰り返し`` の割合で小さくなるやまびこを ``量`` だけ足す"""
    mix = float(np.clip(values.get("mix", 50.0), 0.0, 100.0)) / 100.0
    step = round(max(values.get("time", 250.0), 1.0) / 1000.0 * context.sample_rate)
    if mix <= 0.0 or step <= 0:
        return samples
    feedback = float(np.clip(values.get("feedback", 40.0), 0.0, 90.0)) / 100.0
    # 使われる所（keep_from から先）だけを作って足す 前の音は 100 秒を超えることがあり、全体を
    # 写して繰り返しの数だけ足すと 1 区切りに 1 秒近く掛かって再生が間に合わない
    # 1 サンプルごとに足す順は全体へ足すときと同じなので、値は変わらない 頭は 0 のまま
    keep = min(max(context.keep_from, 0), len(samples))
    if keep == 0:
        out = samples.astype(np.float32, copy=True)
    else:
        out = np.zeros(samples.shape, dtype=np.float32)
        out[keep:] = samples[keep:]
    gain = mix
    for repeat in range(1, _delay_repeats(values) + 1):
        shift = step * repeat
        if shift >= len(samples):
            break
        low = max(shift, keep)
        if low < len(samples):
            out[low:] += samples[low - shift : len(samples) - shift] * gain
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
