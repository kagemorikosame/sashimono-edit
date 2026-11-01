"""語の時刻を音の波形で補い、切ってよい位置を探す（#420）

語の時刻は認識の見積もりで、語の境目と実際の音の谷は 0.1 秒ほどずれる 音量だけでは語の中の
息継ぎと語の境目を見分けられない 2 つを合わせて、語の境目の近くで一番静かな所を切る位置にする

波形はピークの段階（:mod:`sashimono.engine.audio.waveform`）を読む 解析済みの波形を使うので
音を読み直さず、AI の道具の中で呼んでも編集画面が止まらない
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

import numpy as np

from sashimono.core.model import Word
from sashimono.core.words import WordMap
from sashimono.engine.audio.waveform import Waveform

__all__ = [
    "JOINED",
    "PAUSE",
    "UNCHECKED",
    "VALLEY",
    "WORD_EDGE",
    "CutPoint",
    "CutReading",
    "cut_points",
    "level_at",
    "read_cut",
]

#: 語の間に音の切れ目がある（息継ぎより長い間）
PAUSE = "pause"
#: 語の間で音が下がる（短い谷） 話が続く所でも、ここなら語は欠けない
VALLEY = "valley"
#: 語の間だが声が続く 切ると語尾がつながって聞こえることがある
JOINED = "joined"
#: 波形の解析がまだで、音を確かめていない
UNCHECKED = "unchecked"
#: 切り所が語の内側に入ったので語の端まで出した 音の谷ではなく語の時刻で決めた位置
WORD_EDGE = "word_edge"

#: 音の大きさを見る刻み（秒） 語の谷（30 ms ほど）を捉えられ、波形の一番細かい段階より粗い
_STEP = Fraction(1, 100)
#: 谷を探す幅（秒） 語の時刻のずれ（実物の素材で 9 割が 0.09 秒以内）を覆う
_SEARCH = Fraction(1, 10)
#: 話の大きさを読む幅（秒） 短いと、間の多い所で話の大きさを小さく読む
_REFERENCE_REACH = Fraction(10)
#: 話の大きさの基準 周りの音の 9 割の値 山の 1 点だと声を張った所に引きずられる
_REFERENCE_PERCENTILE = 90.0
#: 周りの音全体がこれより小さければ、話していない所として読む（dBFS）
_NEARLY_SILENT_DB = -55.0
#: 話の大きさからこれだけ下がれば静かと読む（dB） 実物の素材で語の境目の谷の中央値が -17〜-23、
#: 語の中が -7〜-8 これより浅いと語の中まで静かと読み、深いと録音の雑音の多い素材で谷を
#: 1 つも拾えない
QUIET_DB = -24.0
#: 声が続く所で同じ小ささと読む差（dB）
_FLAT_DB = 1.0
#: 静かな所を読むときに、続いて静かであることを求めるコマの数 1 つだけ低いのは波の 0 交差
_QUIET_COLUMNS = 3
#: これより長く静かなら話の間と読む（秒）
_PAUSE_SECONDS = Fraction(12, 100)
#: 間の端に残す音（秒） 間の端ちょうどで切ると、語の頭の子音が落ちる
_EDGE_PADDING = Fraction(5, 100)
#: 話の頭と終わりで音を見る幅（秒）
_EDGE_REACH = Fraction(3, 10)
#: 返す位置の数 AI が読むのは近い数個で、長く返すと会話の費用だけが増える
MAX_POINTS = 6


@dataclass(frozen=True, slots=True)
class CutPoint:
    """切ってよい位置 1 つ 時刻は素材の秒"""

    #: 音の入りに使う位置（後ろの語の手前）
    in_seconds: Fraction
    #: 音の出に使う位置（前の語の後ろ）
    out_seconds: Fraction
    before: str
    after: str
    #: :data:`PAUSE` :data:`VALLEY` :data:`JOINED` :data:`UNCHECKED`
    kind: str
    #: 静かな長さ（秒） 波形が無ければ 0
    quiet_seconds: float = 0.0
    #: 周りの話の大きさと比べた、一番静かな所の大きさ（dB） 波形が無ければ ``None``
    level_db: float | None = None

    def distance(self, seconds: Fraction) -> Fraction:
        return abs((self.in_seconds + self.out_seconds) / 2 - seconds)


def _levels(waveform: Waveform, start: Fraction, end: Fraction) -> np.ndarray:
    """``[start, end)`` の音の大きさ（dBFS）を :data:`_STEP` 刻みで"""
    rate = waveform.sample_rate
    first = int(max(start, Fraction(0)) * rate)
    last = min(int(end * rate), waveform.total_samples)
    columns = int((last - first) * _STEP.denominator / rate) if last > first else 0
    if columns <= 0:
        return np.zeros(0, dtype=np.float32)
    envelope = waveform.envelope(first, last, columns)
    peaks = np.abs(envelope).max(axis=(1, 2))
    return np.asarray(20.0 * np.log10(np.maximum(peaks, 1e-6)), dtype=np.float32)


def _reference(waveform: Waveform, seconds: Fraction) -> float:
    levels = _levels(waveform, seconds - _REFERENCE_REACH, seconds + _REFERENCE_REACH)
    if levels.size == 0:
        return 0.0
    # 話している所だけで見る 間の無音まで数えると、間の多い素材で話の大きさを小さく読む
    spoken = levels[levels > _NEARLY_SILENT_DB]
    if spoken.size == 0:
        # 周りに話が無い 全部が静かと読める基準を返す 決まった値を基準にすると、雑音だけの所を
        # 基準に近い大きさと読み、声が続くと判定する
        return float(levels.max()) - QUIET_DB + _FLAT_DB
    return float(np.percentile(spoken, _REFERENCE_PERCENTILE))


def _held(levels: np.ndarray) -> np.ndarray:
    """続く :data:`_QUIET_COLUMNS` コマの中の最大 波の 0 交差の 1 点を静かと読まない"""
    if levels.size < _QUIET_COLUMNS:
        return levels
    windows = np.lib.stride_tricks.sliding_window_view(levels, _QUIET_COLUMNS)
    return np.asarray(windows.max(axis=1))


def level_at(waveform: Waveform | None, seconds: Fraction) -> float | None:
    """``seconds`` の音が、周りの話の大きさよりどれだけ小さいか（dB） 波形が無ければ ``None``

    切った所の前後 2 コマを見る 切った瞬間の 1 点は波の 0 交差に当たることがある
    """
    if waveform is None:
        return None
    levels = _levels(waveform, seconds - 2 * _STEP, seconds + 2 * _STEP)
    if levels.size == 0:
        return None
    return float(_held(levels).min()) - _reference(waveform, seconds)


@dataclass(frozen=True, slots=True)
class CutReading:
    """素材の 1 点を切ったときの読み取り 検査と ``word_boundaries`` が同じ答えを使う"""

    #: その点を中に含む語 無ければ ``None``
    word: Word | None
    #: 語で確かめられたか 偽なら、語の途中かを言えない
    known: bool


def read_cut(words: WordMap, waveform: Waveform | None, seconds: Fraction) -> CutReading:
    """``seconds`` が語の途中かを読む

    語の無い所は、音が静かと分かるときだけ確かめられたと数える 語の間の無音はふつうにあるので一律に
    断らず、声が続く所と波形がまだの所を「言えない」にする
    """
    word = words.word_at(seconds)
    if word is not None or not words.unworded_at(seconds):
        return CutReading(word, True)
    level = level_at(waveform, seconds)
    return CutReading(None, level is not None and level <= QUIET_DB)


def _between(waveform: Waveform, before: Word, after: Word, reference: float) -> CutPoint:
    """2 つの語の間で一番よい切り所 音が下がった一番長い所を選ぶ"""
    start = max(Fraction(0), before.end - _SEARCH)
    levels = _levels(waveform, start, after.start + _SEARCH)
    if levels.size == 0:
        return CutPoint(after.start, before.end, before.text, after.text, UNCHECKED)
    held = _held(levels) - reference
    quiet = held <= QUIET_DB
    if not quiet.any():
        # 声が平らに続く所は一番小さい点に意味が無い ほぼ同じ小ささの中で語の境目に近い点を取る
        # 窓の頭を返すと、語の時刻より 0.1 秒も手前を勧める
        lowest = np.flatnonzero(held <= held.min() + _FLAT_DB)
        middle = float(((before.end + after.start) / 2 - start) / _STEP) - 1
        index = int(lowest[np.argmin(np.abs(lowest - middle))])
        at = start + (index + 1) * _STEP
        return CutPoint(at, at, before.text, after.text, JOINED, 0.0, round(float(held[index]), 1))
    # 静かなコマの続きを 1 つずつ数える 一番長い続きが、語の境目の本当の谷
    changes = np.flatnonzero(np.diff(np.concatenate(([0], quiet.astype(np.int8), [0]))))
    runs = changes.reshape(-1, 2)
    low, high = max(runs.tolist(), key=lambda run: run[1] - run[0])
    head = start + low * _STEP
    tail = start + (high + _QUIET_COLUMNS - 1) * _STEP
    length = tail - head
    level = round(float(held[low:high].min()), 1)
    kind = PAUSE if length >= _PAUSE_SECONDS else VALLEY
    return CutPoint(
        max(head, tail - _EDGE_PADDING),
        min(tail, head + _EDGE_PADDING),
        before.text,
        after.text,
        kind,
        float(length),
        level,
    )


def cut_points(
    words: Sequence[Word], waveform: Waveform | None, seconds: Fraction, window: Fraction
) -> list[CutPoint]:
    """``seconds`` の前後 ``window`` 秒の語の境目を、切ってよい順ではなく近い順に返す

    波形が無ければ語の時刻だけで返す（:data:`UNCHECKED`） AI の道具が呼ぶので、語の数が多くても
    見るのは前後の語だけ
    """
    low, high = seconds - window, seconds + window
    if not words:
        return []
    # 話の頭と終わりも切り所 語の並びの端には相手の語が無いので、空の語を置いて同じ道で見る
    # 見ないと、最初の語の途中で始めたクリップの入りを戻す先が無い
    first = max(Fraction(0), words[0].start - _EDGE_REACH)
    head = Word(first, first, "")
    tail = Word(words[-1].end + _EDGE_REACH, words[-1].end + _EDGE_REACH, "")
    pairs = [
        (before, after)
        for before, after in pairwise((head, *words, tail))
        if after.start >= low and before.end <= high
    ]
    if waveform is None:
        points = [
            CutPoint(after.start, before.end, before.text, after.text, UNCHECKED)
            for before, after in pairs
        ]
    else:
        reference = _reference(waveform, seconds)
        points = [_between(waveform, before, after, reference) for before, after in pairs]
    points.sort(key=lambda point: point.distance(seconds))
    return points[:MAX_POINTS]
