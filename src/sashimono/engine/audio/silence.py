"""無音区間の検出 ジェットカットの土台

すでに作ってある波形ピーク（:mod:`sashimono.engine.audio.waveform`）を使う 音声を
読み直さないので、素材の長さによらず一瞬で終わる ピークは min/max なので、
区間内の最大振幅がそのまま得られる

返すのは**素材内のソース秒**の区間 タイムラインのことは知らない どのクリップの
どこに当たるかは :mod:`sashimono.core.jetcut` が決める
"""

from __future__ import annotations

import math
import threading
import weakref
from collections import OrderedDict
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


#: 音の幅を読む区切りの長さ（秒） 細かいピークのままだと、話の中の一瞬の谷まで数に入る
_CHOICE_TICK_SECONDS = 0.1
#: これより静かな所は素の無音（デジタルの 0）と読む（dBFS） 間の音量を探す数に入れない
#: 入れると素の無音が多い素材で、部屋の音の間の音量まで探しに行かない
_DIGITAL_SILENCE_DB = -90.0
#: 間が素の無音だけの音へ勧める値 間と話の間のどこでも同じだけ切れる
_CLEAN_SILENCE_DB = -50.0
#: 素の無音だけの音で、話の一番静かな所の下へ取る幅（dB）
_CLEAN_MARGIN_DB = 3.0
#: 1 dB でこれだけ（全体の割合）切れる秒が増えて初めて間の音量と読む 小さな山で止まらないため
_HUMP_SHARE = 0.01
#: 話の大きな所を読むパーセント点 最大にしないのは、1 度のクリック音を話と読まないため
#: 話がこれより少ない素材は話と間を見分けない
_LOUD_PERCENTILE = 99.5
#: 話の大きな所からこれ以上静かな所だけを間と読む（dB） 鳴り続ける曲で、曲の音量そのものを
#: 間と読まないため 本物の素材の間は話の大きな所より 20 dB ほど下にある
_SPEECH_GAP_DB = 15.0
#: 勧める値の前後に並べる幅（dB）
_NEARBY_DB = 3

#: 勧める値と前後の見込み
type _Suggestion = tuple[SilenceChoice | None, tuple[SilenceChoice, ...]]

#: 覚える答えの数 同じ作品を調べ直す間は 1 本目の素材の答えが残れば足りる
_SUGGESTED_LIMIT = 32
#: 波形・最短の無音・余白ごとの答え 画面の波形の絵（``ui/timeline/painter.py``）と同じく波形は
#: 弱参照で持ち、取り出すときに同じ物かを確かめる 番号だけで引くと、捨てた波形の番号を次の
#: 素材が引き継いだときに前の素材の値を返す 中身では比べない ピークの配列は比べると重く、
#: ``Waveform`` はハッシュできない 素材を解析し直せば別の波形になるので、古い値は引かれない
#: 2 時間の素材で数えると 0.4 秒ほど掛かるので、呼ぶたびに数えない AI のスレッドで数えて
#: 画面のスレッドが読むので鍵で守る 数える間は鍵を持たない
_SUGGESTED: OrderedDict[tuple[int, float, float], tuple[weakref.ref[Waveform], _Suggestion]] = (
    OrderedDict()
)
_SUGGESTED_LOCK = threading.Lock()


def _quiet_cuts(
    peaks: np.ndarray, times: np.ndarray, threshold_db: float, min_silence: float, padding: float
) -> tuple[int, float]:
    """音の山が ``threshold_db`` を下回り続ける所のうち、切れる物の数と秒

    :func:`detect_silence` と同じ長さの決まりで見る ``times`` はピークの頭の時刻の並び
    （終わりを足して 1 つ長い 終わりは音の長さで抑えてある）
    """
    edges = np.diff(np.concatenate([[0], (peaks < threshold_db).astype(np.int8), [0]]))
    kept = times[edges == -1] - times[edges == 1] - 2 * padding
    # 浮動小数の端数で、ちょうど最短の無音の長さの区間を落とさない
    kept = kept[kept >= min_silence - 1e-9]
    return int(kept.size), float(kept.sum())


def _hump_end(rises: np.ndarray, floor: float) -> int | None:
    """切れる秒の増え方 ``rises`` の最初の山が終わる位置（値の並びの番号） 山が無ければ ``None``

    間の音量の所で増え方が山になり、越えると落ちて、話の谷へ入るとまた増え続ける 全体で一番
    増える所を選ぶと、話の谷の増え方が間の山を上回る素材で話を切る（本物の素材で -24 dB を
    勧めた） そこで最初の山を取る 山の頂上の半分より落ちた所が山の終わり
    """
    peak, top = 0.0, None
    for index, rise in enumerate(rises.tolist()):
        if rise > peak and rise >= floor:
            peak, top = rise, index
        elif top is not None and rise < peak / 2:
            return index
    if top is not None:
        return top + 1
    return int(rises.argmax()) + 1 if rises.size and rises.max() > 0 else None


def choose_silence(
    peaks: np.ndarray,
    tick: float,
    min_silence: float,
    padding: float,
    *,
    total: float = math.inf,
    group: int = 1,
) -> _Suggestion:
    """音の山の推移（``tick`` 秒ごとの dBFS）から、無音カットの勧めるしきい値を選ぶ

    前後の値で切れる見込みも返す 勧められなければ ``(None, ())`` ``total`` は音の長さ（秒）
    ``group`` は音の幅を読むときに束ねるピークの数 切れる見込みは束ねずに数える
    束ねて数えると、区切りをまたいだ間が短く読まれ、:func:`detect_silence` と数が合わない

    やることは 1 つ 鳴っている一番静かな所から話の大きな所の :data:`_SPEECH_GAP_DB` 下までの
    値を 1 dB ずつ試し、切れる秒の増え方の最初の山（:func:`_hump_end`）が終わる値を勧める
    話の間の環境音の所が山になる 本物の素材では AI が何度も切って戻して着いた値と同じになった
    試す値が 2 つに満たないか増える所が無ければ、話と間の差が :data:`_SPEECH_GAP_DB` に届かない
    か、間が素の無音だけ 素の無音で切れるときだけ、話より静かな値を返す
    素の無音は音の幅を読む数に入れない 部屋の音の間の音量を探せなくなるため

    音楽の打楽器と話は音の山では分けられない 音楽が主の素材は呼ぶ側が断る
    """
    levels = np.maximum.reduceat(peaks, np.arange(0, peaks.size, group)) if peaks.size else peaks
    live = levels[levels > _DIGITAL_SILENCE_DB]
    if live.size == 0:
        return None, ()

    times = np.minimum(np.arange(peaks.size + 1) * tick, total)

    def choice(threshold: float) -> SilenceChoice:
        cuts, seconds = _quiet_cuts(peaks, times, threshold, min_silence, padding)
        return SilenceChoice(threshold, cuts, round(seconds, 1))

    def around(threshold: float) -> tuple[SilenceChoice, ...]:
        return tuple(choice(threshold + step) for step in (-_NEARBY_DB, 0, _NEARBY_DB))

    quiet = float(live.min())
    top = float(np.percentile(live, _LOUD_PERCENTILE)) - _SPEECH_GAP_DB
    # ちょうど差のある間を切るには、間の音量の 1 つ上の値まで試す
    candidates = range(math.floor(quiet), math.ceil(top) + 2)
    gains = [_quiet_cuts(peaks, times, v, min_silence, padding)[1] for v in candidates]
    end = _hump_end(np.diff(gains), _HUMP_SHARE * peaks.size * tick)
    if end is not None:
        threshold = float(candidates[end])
        return choice(threshold), around(threshold)
    clean = min(_CLEAN_SILENCE_DB, quiet - _CLEAN_MARGIN_DB)
    return (choice(clean), around(clean)) if choice(clean).cuts else (None, ())


def suggest_silence(waveform: Waveform, min_silence: float, padding: float) -> _Suggestion:
    """素材の波形から無音カットの勧めるしきい値を選ぶ（:func:`choose_silence`）

    :func:`detect_silence` と同じ波形の音の山で見る タイムラインの混ぜた音で決めると、BGM や
    クリップの音量の分だけ素材の音とずれ、同じ値が別の所を切る
    答えは波形ごとに覚える（:data:`_SUGGESTED`）
    """
    level = waveform.levels[0]
    if level.count == 0:
        return None, ()
    key = (id(waveform), min_silence, padding)
    with _SUGGESTED_LOCK:
        entry = _SUGGESTED.get(key)
        if entry is not None and entry[0]() is waveform:
            _SUGGESTED.move_to_end(key)
            return entry[1]
    per_peak = level.samples_per_peak / waveform.sample_rate
    peaks = 20.0 * np.log10(np.maximum(_loudness(waveform), 10.0 ** (_FLOOR_DB / 20.0)))
    answer = choose_silence(
        peaks,
        per_peak,
        min_silence,
        padding,
        total=float(waveform.duration),
        group=max(1, round(_CHOICE_TICK_SECONDS / per_peak)),
    )
    with _SUGGESTED_LOCK:
        _SUGGESTED[key] = (weakref.ref(waveform), answer)
        _SUGGESTED.move_to_end(key)
        while len(_SUGGESTED) > _SUGGESTED_LIMIT:
            _SUGGESTED.popitem(last=False)
    return answer


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
