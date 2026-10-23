"""音のノイズ除去（#336） 一定に鳴り続ける雑音（空調・機械の音・マイクのサー）を下げる

外の機械学習のモデルは使わない（#298 の「決めること」 配る zip と使用許諾に響く）
numpy の FFT だけで、次の 3 つをする

1. **雑音の見積もり** 周波数ごとに、直前 :data:`LEARN_SECONDS` 秒の中で最も静かだった所の
   大きさを雑音とみなす（minimum statistics） 話し声は息継ぎや語の切れ目で周波数ごとに必ず
   静かになるので、そこが雑音の高さになる 一定に鳴り続ける音（ブーン・サー）は静かな所が
   無いので雑音として残る
2. **下げ方** 雑音に近い大きさの周波数だけを、``reduction`` dB まで下げる（spectral
   subtraction の形） 声の周波数は雑音よりずっと大きいので下げない
3. **掛け方** 下げ方から最小位相のフィルタを作り、:data:`BLOCK` 標本ごとに掛け替える
   （前の塊のフィルタから少しずつ移す） 最小位相は先の音を読まずに済み、遅れが出ない

**先の音は読まない** ミキサは前の音しか読み直さない（:attr:`EffectDefinition.audio_history`）
ある時刻の出力は、その時刻までの音だけで決まる 塊を細かく切っても大きく切っても同じ音に
なり、プレビューと書き出しが 1 標本も違わない（試験で区切りを変えて確かめる）
先の音を読む作り（ふつうの短時間フーリエ変換の重ね足し）にすると、ミキサの区切り
（0.34 秒）ごとに先の音が欠けて段が出る

区切りの位置はクリップの頭から数える（``AudioContext.offset``） フレームの頭がクリップの
頭からの決まった位置に並ぶので、どこから鳴らし始めても同じ所で同じ計算になる
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    from sashimono.effects.audio import AudioContext

__all__ = [
    "AUTO_REDUCTION_DB_RANGE",
    "BLOCK",
    "DEFAULT_REDUCTION_DB",
    "DEFAULT_SENSITIVITY_DB",
    "LEARN_SECONDS",
    "REDUCTION_DB_RANGE",
    "SENSITIVITY_DB_RANGE",
    "denoise",
    "denoise_history",
]

#: 下げ方を掛け替える 1 塊（標本） 48 kHz で 5.3 ms 短いほど話し始めに早く付いていけるが、
#: 塊ごとにフィルタを作るので重くなる
BLOCK = 256
#: 周波数を見る窓（標本） 48 kHz で 21 ms・周波数の刻み 47 Hz 短いと低い音の雑音（ブーン）
#: と声を分けられず、長いと話し始めの子音まで均して下げる
FRAME = 1024
#: 掛けるフィルタの長さ（標本） 最小位相のフィルタは頭に力が集まるので、窓の半分で切っても
#: 形はほとんど変わらない 切ると畳み込みの FFT が半分の長さで済む（ミキサの区切りの頭で
#: 21 ms の予算に収めるため）
TAPS = 512
#: 雑音を見積もる長さ（秒） 長いと話し続ける人の声の切れ目を拾えるが、雑音が変わったとき
#: （空調が止まったなど）に付いていくのが遅くなる
LEARN_SECONDS = 1.0
#: クリップの頭で、見積もりが育つまで下げる量を絞る長さ（秒） 頭は見積もる音がまだ無く、
#: 鳴り始めの声を雑音とみなして下げてしまう
WARMUP_SECONDS = 0.3
#: 大きさを均すフレームの数（およそ 85 ms） 窓は 4 分の 3 ずつ重なるので、少ないと隣どうしが
#: ほとんど同じで、最も静かな所が雑音の平均より大きく下に外れる
_SMOOTH = 16
#: 最も静かな所を探すとき、まとめて見るフレームの数（直前 1 秒を、この単位で区切って調べる）
_CHUNK = 8
#: 最も静かな所の大きさから、雑音の平均の大きさへ直す倍率 白い雑音を通して測った（2.2）
#: 最も静かな所は平均より必ず小さいので、直さないと雑音を半分しか下げない
_BIAS = 2.2
#: 下げる量の既定と範囲（dB） 大きくすると雑音は消えるが、声の小さい所まで削れて水の中で
#: 話すような音（ミュージカルノイズ）になる 12 dB で雑音はおよそ 4 分の 1 の大きさ
DEFAULT_REDUCTION_DB = 12.0
REDUCTION_DB_RANGE = (0.0, 40.0)
#: AI の道具と本人の設定で選べる下げる量（dB） 0 は「下げない」で、道具で頼む意味が無い
#: 3 dB より小さいと聞いて分からない
AUTO_REDUCTION_DB_RANGE = (3, 40)
#: 雑音とみなす幅の既定と範囲（dB） 見積もった雑音よりこれだけ大きい所まで雑音として下げる
#: 大きくすると残る雑音が減るが、小さな声も雑音として下げる
DEFAULT_SENSITIVITY_DB = 4.0
SENSITIVITY_DB_RANGE = (0.0, 20.0)
#: 前の音を読む長さ（秒）のうち、見積もりの長さを越える分 均す長さ・まとめて見る単位・窓・
#: 下げ方のならしの分 8 kHz でも足りる長さにしてある（標本で決まる長さを秒へ直すため）
_HISTORY_SLACK = 0.6


def denoise_history(_values: dict[str, float]) -> float:
    """前の音を読む長さ（秒） 値に依らない（見積もる長さは決めてある）"""
    return LEARN_SECONDS + _HISTORY_SLACK


def denoise(samples: np.ndarray, values: dict[str, float], context: AudioContext) -> np.ndarray:
    """``samples`` の ``keep_from`` から先の雑音を下げる その前は元の音のまま返す"""
    count = len(samples)
    reduction = float(np.clip(values.get("reduction", DEFAULT_REDUCTION_DB), *REDUCTION_DB_RANGE))
    if reduction <= 0.0 or count == 0:
        return samples
    keep = min(max(context.keep_from, 0), count)
    if keep >= count:
        return samples
    sensitivity = float(
        np.clip(values.get("sensitivity", DEFAULT_SENSITIVITY_DB), *SENSITIVITY_DB_RANGE)
    )
    origin = context.offset
    first = (origin + keep) // BLOCK
    last = (origin + count - 1) // BLOCK
    rate = max(context.sample_rate, 1)
    gains = _block_gains(
        samples.mean(axis=1, dtype=np.float64),
        origin=origin,
        first=first,
        last=last,
        rate=rate,
        reduction=reduction,
        sensitivity=sensitivity,
    )
    filters = _minimum_phase(gains)
    out = np.array(samples, dtype=np.float32, copy=True)
    filtered = _filtered(samples, origin, first, last, filters)
    head = first * BLOCK
    begin = max(origin + keep, head)
    out[begin - origin :] = filtered[begin - head : origin + count - head]
    return out


def _frames_from(first: int, rate: int) -> tuple[int, int, int]:
    """塊 ``first`` からを掛けるのに要るフレームの頭と、直前を見る長さ（フレーム・まとめの数）"""
    learn = max(_CHUNK, round(LEARN_SECONDS * rate / BLOCK))
    chunks = max(1, learn // _CHUNK)
    # 塊 k は、フレーム k-1・k-2・k-3（k の頭までに終わる窓）の下げ方を均して使い、
    # クロスフェードの相手に前の塊 k-1 も要るので、下げ方はフレーム first-4 から要る
    earliest = first - 4
    head = ((earliest // _CHUNK - chunks) * _CHUNK - _SMOOTH) // _CHUNK * _CHUNK
    return head, chunks, earliest


def _block_gains(
    mono: np.ndarray,
    *,
    origin: int,
    first: int,
    last: int,
    rate: int,
    reduction: float,
    sensitivity: float,
) -> np.ndarray:
    """塊 ``first - 1`` から ``last`` までの、周波数ごとの下げ方（掛ける倍率）

    フレーム j はクリップの頭から ``(j + 1) * BLOCK`` で終わる窓 塊 k はフレーム k-1 までしか
    見ない（その塊の中の音も見ない） 先を見ると、ミキサの区切りの終わりで窓が欠けて区切りの
    切り方で音が変わる
    """
    head, chunks, earliest = _frames_from(first, rate)
    tail = last - 1
    total = tail - head + 1
    start = head * BLOCK - (FRAME - BLOCK)
    span = np.zeros((tail + 1) * BLOCK - start, dtype=np.float64)
    low = max(origin, start)
    high = min(origin + len(mono), (tail + 1) * BLOCK)
    if high > low:
        span[low - start : high - start] = mono[low - origin : high - origin]
    frames = sliding_window_view(span, FRAME)[::BLOCK][:total]
    spectrum = np.fft.rfft(frames * _window(), axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    index = np.arange(total)
    summed = np.concatenate([np.zeros((1, power.shape[1])), np.cumsum(power, axis=0)])
    level = (summed[index + 1] - summed[np.maximum(index + 1 - _SMOOTH, 0)]) / _SMOOTH
    # 均す窓の全部がクリップの中と、渡された音の中にあるフレームだけを見積もりに使う
    # クリップの頭より前は無音なので、使うと雑音を 0 と見積もって何も下げない
    reach = (index + head) * BLOCK - (FRAME - BLOCK) - (_SMOOTH - 1) * BLOCK
    usable = (reach >= max(0, origin)) & (index >= _SMOOTH - 1)
    level[~usable] = np.inf
    # ここから先は下げ方を使うフレーム（earliest から）だけを計算する 見積もりのための
    # 1 秒ぶんまで計算すると、ミキサの区切りの頭で手間が 4 倍になる
    since = earliest - head
    noise = _quietest(level, chunks)[since:]
    noise = np.where(np.isfinite(noise), noise, 0.0) * (_BIAS * 10.0 ** (sensitivity / 10.0))
    picked = index[since:]
    now = (summed[picked + 1] - summed[picked - 1]) / 2.0

    # クリップの頭は見積もりが育つまで量を絞る 塊の位置だけで決まるので区切りに依らない
    warm = max(1, round(WARMUP_SECONDS * rate / BLOCK))
    grown = np.clip((picked + head - (_SMOOTH - 1) - FRAME // BLOCK) / warm, 0.0, 1.0)
    floor = 10.0 ** (-reduction * grown / 20.0)
    squared = 1.0 - noise / np.maximum(now, 1e-20)
    gain = np.sqrt(np.clip(squared, (floor**2)[:, None], 1.0))
    # 隣の周波数と均す 1 本だけ下げ残すと、そこだけが鳴る細い音（ミュージカルノイズ）になる
    padded = np.concatenate([gain[:, :1], gain, gain[:, -1:]], axis=1)
    gain = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    # 塊 k はフレーム k-1・k-2・k-3 の平均 塊ごとに下げ方が跳ねると、ざらついた音になる
    smoothed: np.ndarray = (gain[2:] + gain[1:-1] + gain[:-2]) / 3.0
    return smoothed


def _quietest(level: np.ndarray, chunks: int) -> np.ndarray:
    """フレームごとに、直前 ``chunks`` まとまりと今のまとまりの中で最も静かな大きさ

    まとまり（:data:`_CHUNK` フレーム）の区切りはクリップの頭からの位置で決まる（呼ぶ側が
    頭をまとまりの区切りにそろえる） 1 フレームずつ 1 秒ぶんを見比べると、ミキサの
    区切りの頭で手間が予算を越える
    """
    total = len(level)
    blocks = -(-total // _CHUNK)
    padded = np.full((blocks * _CHUNK, level.shape[1]), np.inf)
    padded[:total] = level
    grouped = padded.reshape(blocks, _CHUNK, -1)
    running = np.minimum.accumulate(grouped, axis=1).reshape(blocks * _CHUNK, -1)[:total]
    lows = grouped.min(axis=1)
    before = np.concatenate([np.full((chunks, level.shape[1]), np.inf), lows])
    earlier = sliding_window_view(before, chunks, axis=0)[:blocks].min(axis=-1)
    quietest: np.ndarray = np.minimum(running, earlier[np.arange(total) // _CHUNK])
    return quietest


def _minimum_phase(gains: np.ndarray) -> np.ndarray:
    """下げ方（周波数ごとの倍率）を、同じ大きさの最小位相のフィルタの周波数の形へ

    ケプストラムを折り返して作る（大きさから位相を決める） 0 位相のフィルタは前後対称で
    先の音を読むことになり、ミキサの区切りで段が出る 下げない（倍率 1）所は素通しになる
    """
    cepstrum = np.fft.irfft(np.log(np.maximum(gains, 1e-6)), FRAME, axis=1)
    folded = np.zeros_like(cepstrum)
    half = FRAME // 2
    folded[:, 0] = cepstrum[:, 0]
    folded[:, 1:half] = 2.0 * cepstrum[:, 1:half]
    folded[:, half] = cepstrum[:, half]
    response = np.fft.irfft(np.exp(np.fft.rfft(folded, axis=1)), FRAME, axis=1)[:, :TAPS]
    response[:, -_FADE:] *= _fade()
    spectrum: np.ndarray = np.fft.rfft(response, _CONVOLUTION, axis=1)
    return spectrum


def _filtered(
    samples: np.ndarray, origin: int, first: int, last: int, filters: np.ndarray
) -> np.ndarray:
    """塊 ``first`` から ``last`` までに、前の塊のフィルタから今の塊のフィルタへ移しながら掛ける

    返すのは塊 ``first`` の頭からの音（標本の数は塊の数 x :data:`BLOCK`）
    """
    blocks = last - first + 1
    channels = samples.shape[1]
    start = first * BLOCK - (TAPS - 1)
    stop = (last + 1) * BLOCK
    span = np.zeros((stop - start, channels), dtype=np.float64)
    low = max(origin, start)
    high = min(origin + len(samples), stop)
    if high > low:
        span[low - start : high - start] = samples[low - origin : high - origin]
    pieces = sliding_window_view(span, TAPS - 1 + BLOCK, axis=0)[::BLOCK][:blocks]
    spectrum = np.fft.rfft(pieces, _CONVOLUTION, axis=2)
    useful = slice(TAPS - 1, TAPS - 1 + BLOCK)
    current = np.fft.irfft(spectrum * filters[1:, None, :], _CONVOLUTION, axis=2)[:, :, useful]
    previous = np.fft.irfft(spectrum * filters[:-1, None, :], _CONVOLUTION, axis=2)[:, :, useful]
    weight = (np.arange(BLOCK) + 0.5) / BLOCK
    mixed = previous + (current - previous) * weight
    joined: np.ndarray = mixed.transpose(0, 2, 1).reshape(blocks * BLOCK, channels)
    return joined


#: フィルタの終わりを 0 へ寄せる長さ 切った所で急に途切れると、周波数の形に細かい波が立つ
_FADE = 64
#: 畳み込みの FFT の長さ 掛ける音（塊 + フィルタの長さ - 1）が収まる 2 の冪
_CONVOLUTION = 1 << (TAPS - 1 + BLOCK - 1).bit_length()


def _window() -> np.ndarray:
    return np.hanning(FRAME + 1)[:FRAME]


def _fade() -> np.ndarray:
    return np.cos(np.linspace(0.0, np.pi / 2.0, _FADE)) ** 2
