"""音量の線を組む計算（#329 B3 音量とミックス）

BGM を話している間だけ下げる（ダッキング）線と、音量の線を全体で同じ比だけ動かす
計算と、それを命令にする所（AI の道具とタイムラインの右クリックが同じ所を通る）を
ここに置く どちらも Qt も音も知らない 話している区間（素材の秒）と、今の音量の線
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

from sashimono.core.commands.base import Command
from sashimono.core.commands.volume import clip_volume, volume_command
from sashimono.core.jetcut import FrameRange, SourceRange
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Keyframe,
    Project,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "DEFAULT_DUCK_ATTACK_MS",
    "DEFAULT_DUCK_DEPTH_DB",
    "DEFAULT_DUCK_RELEASE_MS",
    "DUCK_ATTACK_MS_RANGE",
    "DUCK_DEPTH_DB_RANGE",
    "DUCK_RELEASE_MS_RANGE",
    "VOLUME_LIMIT",
    "DuckPlan",
    "DuckShape",
    "LoudnessGain",
    "audible_clips",
    "db_to_ratio",
    "duck_curve",
    "gain_commands",
    "loudest",
    "loudness_gain",
    "measure_project",
    "plan_ducking",
    "ratio_to_db",
    "scaled_volume",
    "sounds_now",
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


@dataclass(frozen=True, slots=True)
class DuckPlan:
    """ダッキングで変える物 命令はまとめて 1 回の取り消しで戻せるよう、呼ぶ側が 1 度に当てる"""

    commands: tuple[Command, ...]
    #: 下げたクリップと、その新しい線
    curves: tuple[tuple[Clip, AnimatedValue], ...]
    #: 範囲に話している所が無く、変えなかったクリップ
    untouched: tuple[Clip, ...]


def plan_ducking(
    clips: Sequence[Clip],
    speech: Sequence[FrameRange],
    *,
    shape: DuckShape,
    rate: FrameRate,
    base: float | None = None,
) -> DuckPlan:
    """``clips``（BGM）を、話している区間 ``speech``（タイムラインのフレーム）の間だけ下げる

    下げていない音量 ``base`` を省くと、クリップの線の一番大きい値 前に下げた線を掛け直すとき、
    下げた所ではなく元の音量を基準にする 下げた所を基準にすると、掛けるたびに BGM が小さくなる
    """
    commands: list[Command] = []
    curves: list[tuple[Clip, AnimatedValue]] = []
    untouched: list[Clip] = []
    for clip in clips:
        level = loudest(clip_volume(clip)) if base is None else base
        relative = [
            (start - clip.timeline_start, end - clip.timeline_start) for start, end in speech
        ]
        curve = duck_curve(clip.duration, relative, base=level, shape=shape, rate=rate)
        if curve is None:
            untouched.append(clip)
            continue
        commands.append(volume_command(clip, curve))
        curves.append((clip, curve))
    return DuckPlan(tuple(commands), tuple(curves), tuple(untouched))


def measure_project(project: Project, clip_ids: Iterable[ClipId]) -> Project:
    """測るクリップだけを残したプロジェクト ミュートとソロは外す

    ほかのクリップを残すと、BGM だけをそろえたいのに声まで含めた音量を測る ソロは聞き比べの
    ための物で、残すとソロのトラックの外にある測りたい音が鳴らない
    """
    wanted = set(clip_ids)
    tracks = tuple(
        replace(
            track,
            clips=tuple(clip for clip in track.clips if clip.id in wanted),
            muted=False,
            solo=False,
        )
        for track in project.timeline.tracks
    )
    return replace(project, timeline=replace(project.timeline, tracks=tracks))


@dataclass(frozen=True, slots=True)
class LoudnessGain:
    """測った音量から決めた、全部のクリップへ掛ける量"""

    gain_db: float
    #: 基準の手前で止めた訳 ``peak``（音の山が天井を超える）と ``volume_limit``（欄の上限）
    limited_by: tuple[str, ...] = ()


def loudness_gain(
    clips: Sequence[Clip],
    *,
    measured: float,
    peak_db: float | None,
    target: float,
    ceiling: float,
) -> LoudnessGain:
    """測った積分ラウドネス ``measured`` を ``target`` へ寄せる量（dB）

    上げすぎると音が割れるので、山 ``peak_db`` が天井 ``ceiling`` に当たる所で止める
    どれかのクリップの線が欄の上限を超える所でも止める 超えた分は鳴らないので、そろえた
    つもりの音量にならない
    """
    gain = target - measured
    limited: list[str] = []
    if peak_db is not None and peak_db + gain > ceiling:
        gain = ceiling - peak_db
        limited.append("peak")
    top = max((loudest(clip_volume(clip)) for clip in clips), default=0.0)
    if top > 0 and top * db_to_ratio(gain) > VOLUME_LIMIT:
        gain = ratio_to_db(VOLUME_LIMIT / top)
        limited.append("volume_limit")
    return LoudnessGain(gain, tuple(limited))


def gain_commands(clips: Sequence[Clip], gain_db: float) -> list[Command]:
    """どのクリップの音量の線にも同じ dB を掛ける命令 声と BGM の釣り合いは変えない"""
    ratio = db_to_ratio(gain_db)
    return [volume_command(clip, scaled_volume(clip_volume(clip), ratio)) for clip in clips]


def _heard_tracks(project: Project) -> set[int]:
    """ミキサが混ぜるトラック（の ``id()``） 聞こえる音のトラックと、映る映像トラック

    ミキサ（:meth:`~sashimono.engine.audio.mixer.AudioMixer.render`）と同じ並びで決める
    映像トラックは置いたシーンの音だけを鳴らし、ソロとミュートは絵の側で見る
    """
    timeline = project.timeline
    heard = {id(track) for track in timeline.active_sound_tracks()}
    heard |= {id(t) for t in timeline.active_picture_tracks() if t.kind is TrackKind.VIDEO}
    return heard


def sounds_now(project: Project, track: Track, clip: Clip) -> bool:
    """書き出しで ``clip`` の音が鳴るか ミュート・ソロ・無効（:attr:`Clip.enabled`）を見る

    ミキサと同じ決まりにする 食い違うと、ソロで外れたトラックや無効にしたクリップの音まで
    測って音量をそろえ、書き出す音が基準からずれる（#375 の Codex の指摘）
    """
    return clip.enabled and id(track) in _heard_tracks(project) and project.plays_sound(track, clip)


def audible_clips(project: Project) -> list[tuple[Track, Clip]]:
    """書き出しで音が鳴るクリップすべて（:func:`sounds_now` が真の物）"""
    heard = _heard_tracks(project)
    return [
        (track, clip)
        for track in project.timeline.tracks
        if id(track) in heard
        for clip in track.clips
        if clip.enabled and project.plays_sound(track, clip)
    ]
