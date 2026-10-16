"""音量の線を組む計算（#329 B3 音量とミックス）

BGM を話している間だけ下げる（ダッキング）線と、音量の線を全体で同じ比だけ動かす
計算をここに置く どちらも Qt も音も知らない 話している区間（素材の秒）と、今の音量の線
（:class:`~sashimono.core.model.AnimatedValue` の %）を受け取り、新しい線を返すだけ
線を変える命令は :mod:`sashimono.core.commands.volume` が作る

音量の値はクリップの固定の音量（``audio_volume`` の ``volume`` 0〜400%）と同じ単位で持つ
人がタイムラインの音量の線で打つ点と同じ物になるので、AI が下げた所を人が線で直せる
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise

from sashimono.core.jetcut import FrameRange, SourceRange
from sashimono.core.model import AnimatedValue, Clip, Keyframe
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "DEFAULT_DUCK_ATTACK_MS",
    "DEFAULT_DUCK_DEPTH_DB",
    "DEFAULT_DUCK_RELEASE_MS",
    "DUCK_ATTACK_MS_RANGE",
    "DUCK_DEPTH_DB_RANGE",
    "DUCK_RELEASE_MS_RANGE",
    "VOLUME_LIMIT",
    "DuckShape",
    "db_to_ratio",
    "duck_curve",
    "loudest",
    "ratio_to_db",
    "scaled_volume",
    "speech_frames",
]

#: 音量の上限（%） 固定の音量の欄（``audio_volume`` の ``volume``）の定義の上限と同じ
#: 超えた値を入れると、設定パネルの欄が端に張り付いたまま、鳴る音だけが大きくなる
VOLUME_LIMIT = 400.0

#: 話している間に BGM を下げる量（dB） 既定は 12 dB 放送や配信の手引きでよく使う
#: 10〜15 dB の真ん中 声がはっきり聞こえ、BGM が消えたとは感じない量
DEFAULT_DUCK_DEPTH_DB = 12
#: 下げ始めてから下がり切るまで（ミリ秒） 話し始めの前から下げ始めるので、声の頭は
#: 下がった BGM に乗る 短すぎると BGM がぶつりと切れたように聞こえる
DEFAULT_DUCK_ATTACK_MS = 300
#: 話し終えてから元の音量へ戻るまで（ミリ秒） 下げるより長くする 早く戻すと文の間で
#: BGM が出入りして（ポンピング）落ち着かない
DEFAULT_DUCK_RELEASE_MS = 800
#: 設定画面と道具の引数で受ける範囲 下げ幅の 3 dB 未満は聞き分けにくく、40 dB を超えると
#: 0% と変わらない 速さは 1 秒の 1/50 から数秒まで
DUCK_DEPTH_DB_RANGE = (3, 40)
DUCK_ATTACK_MS_RANGE = (20, 3000)
DUCK_RELEASE_MS_RANGE = (20, 5000)


@dataclass(frozen=True, slots=True)
class DuckShape:
    """ダッキングの形 下げる量と、下げる速さ・戻る速さ"""

    depth_db: float = DEFAULT_DUCK_DEPTH_DB
    attack_ms: int = DEFAULT_DUCK_ATTACK_MS
    release_ms: int = DEFAULT_DUCK_RELEASE_MS

    def frames(self, rate: FrameRate) -> tuple[int, int]:
        """下げる長さと戻る長さ（フレーム） どちらも 1 フレーム以上

        0 フレームにすると、上の点と下の点が同じフレームに重なり、線の点として持てない
        （同じフレームに 2 つの点は置けない）
        """
        return (
            max(1, seconds_to_frame(Fraction(self.attack_ms, 1000), rate, Rounding.NEAREST)),
            max(1, seconds_to_frame(Fraction(self.release_ms, 1000), rate, Rounding.NEAREST)),
        )


def db_to_ratio(db: float) -> float:
    """dB を音量の比へ（+6 dB でおよそ 2 倍）"""
    return float(10.0 ** (db / 20.0))


def ratio_to_db(ratio: float) -> float:
    """音量の比を dB へ 0 以下は -inf"""
    return 20.0 * math.log10(ratio) if ratio > 0 else -math.inf


def loudest(value: AnimatedValue) -> float:
    """線の一番大きい値 点が無ければ値そのもの"""
    if not value.keyframes:
        return value.static
    return max(keyframe.value for keyframe in value.keyframes)


def scaled_volume(value: AnimatedValue, ratio: float) -> AnimatedValue:
    """音量の線の全部の点（点が無ければ値そのもの）に同じ比を掛ける 0〜上限で止める

    足すのではなく掛ける 下げた所と上げた所の差（dB）を保つため 全体に +3 dB を足す
    つもりで % を足すと、ダッキングで下げた所だけ上げ幅が大きくなり、声と BGM の釣り合いが
    崩れる
    """

    def clamp(level: float) -> float:
        return min(VOLUME_LIMIT, max(0.0, level * ratio))

    return AnimatedValue(
        static=clamp(value.static),
        keyframes=tuple(replace(k, value=clamp(k.value)) for k in value.keyframes),
    )


def speech_frames(clip: Clip, speech: Iterable[SourceRange], rate: FrameRate) -> list[FrameRange]:
    """素材の秒で表した話している区間を、クリップの頭からのフレームの区間へ

    外側へ丸める 内側へ丸めると、声の頭と終わりの 1 フレームが下がり切っていない BGM に乗る
    クリップに見えていない所（トリムで外した所）は落とす
    """
    found: list[FrameRange] = []
    source_in = clip.source_in
    for start, end in speech:
        begin = (start - source_in) / clip.speed
        stop = (end - source_in) / clip.speed
        first = max(0, seconds_to_frame(begin, rate, Rounding.FLOOR))
        last = min(clip.duration, seconds_to_frame(stop, rate, Rounding.CEIL))
        if last > first:
            found.append((first, last))
    return found


def _merged(spans: Iterable[FrameRange], gap: int) -> list[FrameRange]:
    """重なる区間と、間が ``gap`` フレーム以下の区間を 1 つにまとめる

    間が下げる長さと戻る長さの和より短いと、戻り切る前にまた下がり始める 点を並べると
    下りの点が上りの点より前に来て、線が折り返す
    """
    ordered = sorted((start, end) for start, end in spans if end > start)
    merged: list[FrameRange] = []
    for start, end in ordered:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _value_at(points: Sequence[tuple[int, float]], frame: int, base: float) -> float:
    """折れ線の ``frame`` での値 点の並びの外は ``base``（下げていない音量）

    ``points`` はフレームの昇順で、同じフレームの点を持たない（:func:`duck_curve` が並べる）
    """
    if not points or frame < points[0][0] or frame > points[-1][0]:
        return base
    for (left, low), (right, high) in pairwise(points):
        if left <= frame <= right:
            return low + (high - low) * (frame - left) / (right - left)
    return points[0][1]


def duck_curve(
    duration: int,
    speech: Iterable[FrameRange],
    *,
    base: float,
    shape: DuckShape,
    rate: FrameRate,
) -> AnimatedValue | None:
    """話している区間（クリップの頭からのフレーム）の間だけ下げた音量の線

    区間ごとに 4 つの点を打つ 話し始める ``attack`` 前に元の音量、話し始めで下がり切り、
    話し終えるまで下げたまま、``release`` 後に元へ戻る クリップの端で切れる所は、端の
    フレームにその時点の値の点を打つ（線の値は端の点で止まるので、打たないと端の外の
    点の値まで伸びる） 下げる区間が 1 つも無ければ ``None``（線を変えない）
    """
    attack, release = shape.frames(rate)
    last = duration - 1
    # 下げ始めと戻りがクリップに掛かる区間も数える クリップの終わりのすぐ後で話し始めるなら、
    # 終わり際から下げ始めておかないと、次の BGM のクリップの頭で急に下がる
    spans = [
        (start, end)
        for start, end in _merged(speech, attack + release)
        if end + release > 0 and start - attack <= last
    ]
    if not spans or duration <= 0:
        return None
    ducked = min(VOLUME_LIMIT, max(0.0, base * db_to_ratio(-shape.depth_db)))
    points: list[tuple[int, float]] = []
    for start, end in spans:
        points += [(start - attack, base), (start, ducked), (end, ducked), (end + release, base)]
    kept: dict[int, float] = {}
    if points[0][0] < 0:
        kept[0] = _value_at(points, 0, base)
    for frame, level in points:
        if 0 <= frame <= last:
            kept.setdefault(frame, level)
    if points[-1][0] > last:
        kept.setdefault(last, _value_at(points, last, base))
    keyframes = tuple(Keyframe(frame=frame, value=level) for frame, level in sorted(kept.items()))
    return AnimatedValue(static=base, keyframes=keyframes)
