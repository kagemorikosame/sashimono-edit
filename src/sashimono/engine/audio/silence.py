"""無音区間の検出 ジェットカットの土台

すでに作ってある波形ピーク（:mod:`sashimono.engine.audio.waveform`）を使う 音声を
読み直さないので、素材の長さによらず一瞬で終わる ピークは min/max なので、
区間内の最大振幅がそのまま得られる

返すのは**素材内のソース秒**の区間 タイムラインのことは知らない どのクリップの
どこに当たるかは :mod:`sashimono.core.jetcut` が決める
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from sashimono.core.model import Transcript
from sashimono.engine.audio.waveform import Waveform

__all__ = [
    "SilenceChoice",
    "SilenceOptions",
    "TimeRange",
    "choose_silence",
    "detect_silence",
    "keep_speech",
    "suggest_silence",
]

#: ソース秒の区間 ``[start, end)``
type TimeRange = tuple[Fraction, Fraction]

#: 無音判定の下限 これより静かな環境音は無いものとして扱う
_FLOOR_DB = -120.0


@dataclass(frozen=True, slots=True)
class SilenceOptions:
    """無音判定の条件"""

    #: これを下回っていれば無音とみなす（フルスケール基準の dB）
    threshold_db: float = -40.0
    #: これより短い無音は切らない 息継ぎまで切ると聞いていて忙しい
    min_silence: Fraction = field(default_factory=lambda: Fraction(1, 2))
    #: 無音の前後に残す余白 発話の立ち上がりを切り落とさないため
    padding: Fraction = field(default_factory=lambda: Fraction(1, 10))
    #: 無音に挟まれた発話がこれより短ければ、その発話ごと落とす 0 なら落とさない
    #: 舌打ちやマウスの音だけが残るのを防ぐが、既定では切らない側に倒す
    min_keep: Fraction = field(default_factory=lambda: Fraction(0))

    def __post_init__(self) -> None:
        if self.min_silence <= 0:
            raise ValueError(f"最短無音長は正でなければならない: {self.min_silence}")
        if self.padding < 0 or self.min_keep < 0:
            raise ValueError("余白と最短発話長は負にできない")

    @property
    def amplitude(self) -> float:
        """判定に使う振幅（0..1）"""
        return float(10.0 ** (max(self.threshold_db, _FLOOR_DB) / 20.0))


def _loudness(waveform: Waveform) -> np.ndarray:
    """最も細かい段階のピークごとの、チャンネルをまたいだ最大振幅

    片方だけ鳴っている区間を無音と誤らないため
    """
    peaks = waveform.levels[0].peaks
    return np.maximum(np.abs(peaks[:, :, 0]), np.abs(peaks[:, :, 1])).max(axis=1)


def detect_silence(
    waveform: Waveform, options: SilenceOptions | None = None
) -> tuple[TimeRange, ...]:
    """波形から無音区間を拾う

    最も細かい段階のピークを使う 粗い段階だと 1 ピークに数秒が入り、短い無音が
    埋もれる
    """
    resolved = options if options is not None else SilenceOptions()
    level = waveform.levels[0]
    if level.count == 0:
        return ()

    quiet = _loudness(waveform) < resolved.amplitude

    per_peak = Fraction(level.samples_per_peak, waveform.sample_rate)
    total = waveform.duration

    ranges: list[TimeRange] = []
    for begin, end in _runs(quiet):
        start = begin * per_peak + resolved.padding
        stop = min(end * per_peak, total) - resolved.padding
        if stop - start >= resolved.min_silence:
            ranges.append((start, stop))

    if resolved.min_keep > 0:
        ranges = _absorb_short_speech(ranges, resolved.min_keep)
    return tuple(ranges)


@dataclass(frozen=True, slots=True)
class SilenceChoice:
    """無音カットのしきい値 1 つと、そこで切れる見込み"""

    threshold_db: float
    cuts: int
    seconds: float


#: 勧める値を読む区切りの長さ（秒） 細かい段階のまま何十回も数えると長い素材で画面が止まる
_CHOICE_TICK_SECONDS = 0.1
#: これより静かな所は素の無音（デジタルの 0）と読む（dBFS） 間の音量を探す数に入れない
#: 入れると素の無音が多い素材で、部屋の音の間の音量まで探しに行かない
_DIGITAL_SILENCE_DB = -90.0
#: 間が素の無音だけの音へ勧める値 間と話の間のどこでも同じだけ切れる
_CLEAN_SILENCE_DB = -50.0
#: 素の無音だけの音の話し声の頭打ちの下へ取る幅（dB）
_CLEAN_MARGIN_DB = 3.0
#: しきい値を 1 dB 上げたときに増える切れる秒の、見分けの下限（秒と全体の割合の大きい方）
_KNEE_SECONDS = 1.0
_KNEE_SHARE = 0.005
#: 増える所の上へ足す余裕（dB） 間の音量ちょうどでは、間の中の小さな山で区間が割れる
_KNEE_MARGIN_DB = 1
#: 話の大きな所（上から 1 割）からこれ以上静かな所だけを間と読む（dB） 鳴り続ける曲で、
#: 曲の音量そのものを間と読まないため
_SPEECH_GAP_DB = 10.0
#: 勧める値の前後に並べる幅（dB）
_NEARBY_DB = 3


def _quiet_cuts(
    peaks: np.ndarray, times: np.ndarray, threshold_db: float, min_silence: float, padding: float
) -> tuple[int, float]:
    """音の山が ``threshold_db`` を下回り続ける所のうち、切れる物の数と秒

    :func:`detect_silence` と同じく前後の余白を除いた長さで見る ``times`` は区切りの頭の時刻の
    並び（終わりを足して 1 つ長い） 末尾の半端な区切りも実の長さで数える 丸ごと数えると、
    jet_cut が切らない短い間まで見込みに入る
    """
    edges = np.diff(np.concatenate([[0], (peaks < threshold_db).astype(np.int8), [0]]))
    kept = times[edges == -1] - times[edges == 1] - 2 * padding
    # 浮動小数の端数で、ちょうど最短の無音の長さの区間を落とさない
    kept = kept[kept >= min_silence - 1e-9]
    return int(kept.size), round(float(kept.sum()), 1)


def choose_silence(
    peaks: np.ndarray,
    tick: float,
    min_silence: float,
    padding: float,
    *,
    total: float = math.inf,
) -> tuple[SilenceChoice | None, tuple[SilenceChoice, ...]]:
    """音の山の推移（区切り ``tick`` 秒ごとの dBFS）から、無音カットの勧めるしきい値を選ぶ

    前後の値で切れる見込みも返す 勧められなければ ``(None, ())`` ``total`` は音の長さ（秒）

    間の音量は素材ごとに違う 話の間の環境音に当たる所では、しきい値を 1 dB 上げたときに
    切れる秒が大きく増える そこを越えた値を勧める 本物の素材では、AI が何度も切って
    戻して着いた値と同じになった（#413） 間が素の無音だけなら増える所が無いので、話より
    十分静かな値を返す 切れる物が増えない（曲が鳴り続けている）なら勧めない
    素の無音は間の音量を探す数に入れない 素の無音と部屋の音の間の両方がある素材で、素の無音
    だけを切る値を勧めないため
    """
    live = peaks[peaks > _DIGITAL_SILENCE_DB]
    if live.size == 0:
        return None, ()

    times = np.minimum(np.arange(peaks.size + 1) * tick, total)

    def choice(threshold: float) -> SilenceChoice:
        cuts, seconds = _quiet_cuts(peaks, times, threshold, min_silence, padding)
        return SilenceChoice(threshold, cuts, seconds)

    def around(threshold: float) -> tuple[SilenceChoice, ...]:
        return tuple(choice(threshold + step) for step in (-_NEARBY_DB, 0, _NEARBY_DB))

    # 上の端は話の大きな所から決める 中央値で止めると、間が半分を超える素材で間の音量に届かない
    quiet, loud = (float(value) for value in np.percentile(live, (5, 90)))
    candidates = range(math.floor(quiet), math.ceil(loud - _SPEECH_GAP_DB) + 1)
    rises = np.diff([choice(float(value)).seconds for value in candidates])
    if rises.size and rises.max() >= max(_KNEE_SECONDS, _KNEE_SHARE * peaks.size * tick):
        threshold = float(candidates[int(rises.argmax()) + 1] + _KNEE_MARGIN_DB)
        return choice(threshold), around(threshold)
    clean = min(_CLEAN_SILENCE_DB, quiet - _CLEAN_MARGIN_DB)
    return (choice(clean), around(clean)) if choice(clean).cuts else (None, ())


def suggest_silence(
    waveform: Waveform, min_silence: float, padding: float
) -> tuple[SilenceChoice | None, tuple[SilenceChoice, ...]]:
    """素材の波形から無音カットの勧めるしきい値を選ぶ（:func:`choose_silence`）

    :func:`detect_silence` と同じ波形の音の山で見る タイムラインの混ぜた音で決めると、BGM や
    クリップの音量の分だけ素材の音とずれ、同じ値が別の所を切る
    """
    level = waveform.levels[0]
    if level.count == 0:
        return None, ()
    per_peak = level.samples_per_peak / waveform.sample_rate
    group = max(1, round(_CHOICE_TICK_SECONDS / per_peak))
    loudest = np.maximum.reduceat(_loudness(waveform), np.arange(0, level.count, group))
    peaks = 20.0 * np.log10(np.maximum(loudest, 10.0 ** (_FLOOR_DB / 20.0)))
    total = float(waveform.duration)
    return choose_silence(peaks, group * per_peak, min_silence, padding, total=total)


def keep_speech(
    ranges: tuple[TimeRange, ...], transcript: Transcript, *, margin: Fraction = Fraction(1, 10)
) -> tuple[TimeRange, ...]:
    """起こし結果に発話がある区間を、切る候補から外す

    波形だけでは、小さな声と環境音を区別できない 認識器が文字を起こせた区間は
    確実に発話なので、そこは残す これがあると閾値を強気に振れる
    """
    protected = [(segment.start - margin, segment.end + margin) for segment in transcript.segments]
    if not protected:
        return ranges

    result: list[TimeRange] = []
    for start, end in ranges:
        pieces = [(start, end)]
        for guard_start, guard_end in protected:
            pieces = [
                piece for current in pieces for piece in _subtract(current, guard_start, guard_end)
            ]
        result.extend(piece for piece in pieces if piece[1] > piece[0])
    return tuple(result)


def _subtract(piece: TimeRange, start: Fraction, end: Fraction) -> list[TimeRange]:
    """``piece`` から ``[start, end)`` を引く 0 個・1 個・2 個の区間になる"""
    left, right = piece
    if end <= left or start >= right:
        return [piece]
    remaining: list[TimeRange] = []
    if left < start:
        remaining.append((left, start))
    if end < right:
        remaining.append((end, right))
    return remaining


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """真が続く区間を ``[begin, end)`` の一覧で返す"""
    if not mask.any():
        return []
    # 端に偽を足してから差分を取る 境界の場合分けをここで消しておく
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _absorb_short_speech(ranges: list[TimeRange], min_keep: Fraction) -> list[TimeRange]:
    """無音に挟まれた短い発話を、前後の無音ごと 1 つにまとめる"""
    if not ranges:
        return ranges
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end < min_keep:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged
