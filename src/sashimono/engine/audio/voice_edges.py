"""字幕の行の頭と終わりを、声の出だしと終わりへ寄せる（#426）

Whisper の時刻は声とずれる 文の単位の時刻は無音を落とした後の区切りの癖を引きずり、small では
頭が中央値で 0.22 秒早く、終わりが 0.2 秒遅かった（利用者の「字幕が声より早く出て遅く消える」と
同じ向き） 語の時刻から作った文の時刻は大きく外れる行が減る代わりに、合成の声では頭も終わりも
0.1〜0.3 秒早めに寄った どちらの癖も認識の中からは直せないので、認識の後に音の大きさだけで寄せる

寄せる幅は前後 :data:`REACH` まで 認識の時刻が大きく外れた行（別の文と取り違えた物など）は、
音で寄せても正しい所へは届かないので、遠くへは動かさない

音の大きさは 10 ms ごとのピーク（dBFS）で見る 起こすときは読んだ音から作り
（:func:`loudness_of_samples`）、起こした後に合わせ直すときは解析済みの波形から作る
（:func:`loudness_of_waveform`） どちらも同じ形なので、同じ寄せ方が効く
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from fractions import Fraction
from itertools import pairwise

import numpy as np

from sashimono.core.commands.subtitle import RetimeSegments
from sashimono.core.model import MediaId, SegmentId, Transcript, TranscriptSegment
from sashimono.core.subtitle_timing import MIN_LENGTH, SubtitleTiming, retimed, shifted
from sashimono.engine.audio.waveform import Waveform

__all__ = [
    "END_QUIET_DB",
    "REACH",
    "START_QUIET_DB",
    "STEP",
    "fit_to_voice",
    "found_edges",
    "loudness_of_samples",
    "loudness_of_waveform",
    "retime",
    "retime_command",
]

#: 音の大きさを見る刻み（秒） 語の頭の子音（20〜30 ms）を捉えられる細かさ
STEP = Fraction(1, 100)
#: 寄せる幅（秒） 合成の声で測った認識のずれ（9 割が 0.36 秒以内）を覆い、隣の文までは届かない
REACH = Fraction(1, 2)
#: 声の出だしと読む大きさ（周りの話の大きさから dB） 合成の声の 3 つの組で -24 と -30 を比べ、
#: 頭のずれの中央値はどちらも 0.01〜0.02 秒 -30 の方が「か」「た」の頭の弱い子音を拾う
START_QUIET_DB = -30.0
#: 声の終わりと読む大きさ（周りの話の大きさから dB） 頭より深く取る 文の終わりの「す」は息だけの
#: 弱い音で、-24 では落ちて終わりが 0.1 秒早くなった（-40 で 0.045 秒）
END_QUIET_DB = -40.0
#: 周りで一番静かな所（雑音）からこれだけ上を越えないと声と読まない（dB） 雑音の多い素材で、
#: 深いしきい値が雑音そのものを声と読んで文の終わりを無音の中まで延ばさないように
_FLOOR_MARGIN_DB = 6.0
#: 雑音の大きさを読む割合 周りの音の 1 割の値
_FLOOR_PERCENTILE = 10.0
#: 話の大きさを読む幅（秒） 短いと間の多い所で話の大きさを小さく読む
_REFERENCE_REACH = Fraction(10)
#: 話の大きさの基準 周りの音の 9 割の値 山の 1 点だと声を張った所に引きずられる
_REFERENCE_PERCENTILE = 90.0
#: 周りの音全体がこれより小さければ話していない所と読み、寄せない（dBFS）
_NEARLY_SILENT_DB = -55.0
#: これより短い山は声と読まない（コマ数） 机を叩いたような一瞬の音で頭を早めない
_HOLD_STEPS = 3
#: 音の大きさを作るときに 1 度に読むコマ数（60 秒分） 一時の配列を小さく保つ
_LOUDNESS_CHUNK_STEPS = 6000
#: これより短い谷は声の続きと読む（コマ数 0.12 秒） 語の間の息継ぎで文の終わりを早めない
_GAP_STEPS = 12


def loudness_of_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """1 本の音（モノラル）を :data:`STEP` ごとのピークの大きさ（dBFS）にする"""
    hop = max(1, int(sample_rate * STEP))
    count = len(samples) // hop
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    peaks = np.empty(count, dtype=np.float32)
    # 塊ごとに絶対値を取る 全長で取ると、起こした音（1 時間で 230MB）と同じ長さの配列がもう
    # 1 つ並ぶ（音を読む所が避けているのと同じ形）
    for first in range(0, count, _LOUDNESS_CHUNK_STEPS):
        last = min(count, first + _LOUDNESS_CHUNK_STEPS)
        block = np.asarray(samples[first * hop : last * hop], dtype=np.float32)
        peaks[first:last] = np.abs(block).reshape(last - first, hop).max(axis=1)
    return np.asarray(20.0 * np.log10(np.maximum(peaks, 1e-6)), dtype=np.float32)


def loudness_of_waveform(waveform: Waveform) -> np.ndarray:
    """解析済みの波形を :data:`STEP` ごとのピークの大きさ（dBFS）にする 音は読み直さない"""
    columns = int(waveform.duration / STEP)
    if columns <= 0:
        return np.zeros(0, dtype=np.float32)
    end = int(columns * STEP * waveform.sample_rate)
    envelope = waveform.envelope(0, end, columns)
    peaks = np.abs(envelope).max(axis=(1, 2))
    return np.asarray(20.0 * np.log10(np.maximum(peaks, 1e-6)), dtype=np.float32)


def retime(
    segments: Sequence[TranscriptSegment], timing: SubtitleTiming, loudness: np.ndarray | None
) -> tuple[TranscriptSegment, ...]:
    """行を ``timing`` のとおりに声へ寄せてずらす 並びは開始の昇順

    起こした直後（:mod:`sashimono.asr.timing`）と、起こした後の〔時刻を合わせる…〕・AI の
    ``retime_subtitles`` が同じここを通る 寄せた後にずらす（ずらしてから寄せると、好みで
    早めた頭がまた声の出だしへ戻る） ``loudness`` が無ければ寄せずにずらすだけ
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    if not ordered or not timing.changes:
        return tuple(ordered)
    times = [(s.start, s.end) for s in ordered]
    if timing.fit_to_voice and loudness is not None:
        times = fit_to_voice(times, loudness)
    if timing.start_offset_ms or timing.end_offset_ms:
        times = shifted(times, timing.start_offset, timing.end_offset)
    return retimed(ordered, times)


def retime_command(
    media_id: MediaId,
    transcript: Transcript,
    timing: SubtitleTiming,
    loudness: np.ndarray | None,
    *,
    stream: int | None = None,
    only: Collection[SegmentId] | None = None,
) -> RetimeSegments | None:
    """起こした後の字幕を合わせ直す命令 動く行が無ければ ``None``

    ``only`` を渡せばその行だけを動かす 寄せる窓とずらす上限は全部の行で決める（選んだ行だけで
    決めると、寄せる窓が選ばなかった隣の行の声まで届く） 選ばなかった行は動かさないので、
    動かした行は前の行の終わりより前へ出さず、次の行の頭を越えないよう止め直す
    """
    moved = {s.id: s for s in retime(transcript.segments, timing, loudness)}
    chosen = {s.id for s in transcript.segments} if only is None else set(only)
    # 動かせない行を動かさない行へ回したら、前の行の限りが変わるので頭から決め直す 回す行が
    # 増えなくなるまで繰り返す（行の数だけで止まる） 1 回で済ませると、続けて選んだ行の後ろの
    # 方を戻したときに、先に動かした前の行と重なる（PR #434 の Codex の指摘）
    while True:
        edges, stuck = _bounded(transcript.segments, moved, chosen)
        if not stuck:
            break
        chosen -= stuck
    times = [
        (segment.id, start, end)
        for segment, (start, end) in zip(transcript.segments, edges, strict=True)
        if segment.id in chosen and (start, end) != (segment.start, segment.end)
    ]
    if not times:
        return None
    return RetimeSegments(media_id, tuple(times), stream=stream)


def _bounded(
    lines: Sequence[TranscriptSegment],
    moved: dict[SegmentId, TranscriptSegment],
    chosen: set[SegmentId],
) -> tuple[list[tuple[Fraction, Fraction]], set[SegmentId]]:
    """選んだ行を前後の行の間に止めた時刻と、見える長さが残らず動かせない行

    元の行の順（字幕は頭の昇順が不変条件）のまま限りを当てる 動かした後の頭で並べ直すと、
    大きくずらした行が選ばなかった隣の行を追い越したときに前後が入れ替わり、重なりを止められない
    """
    edges = [
        (moved[s.id].start, moved[s.id].end) if s.id in chosen else (s.start, s.end) for s in lines
    ]
    stuck: set[SegmentId] = set()
    for index, segment in enumerate(lines):
        if segment.id not in chosen:
            continue
        start, end = edges[index]
        least = min(segment.end - segment.start, MIN_LENGTH)
        low = edges[index - 1][1] if index > 0 and _apart(lines[index - 1], segment) else None
        high = (
            edges[index + 1][0]
            if index + 1 < len(lines) and _apart(segment, lines[index + 1])
            else None
        )
        floor = max(Fraction(0), low) if low is not None else Fraction(0)
        if high is not None and high - floor < least:
            # 前後の行の間（と 0 秒より後）に見える長さが残らない 動かすと潰れるか重なるか負の
            # 時刻になる
            stuck.add(segment.id)
            continue
        if high is not None:
            start = min(start, high - least)
            end = min(end, high)
        start = max(start, floor)
        edges[index] = (start, max(end, start + least))
    return edges, stuck


def _apart(earlier: TranscriptSegment, later: TranscriptSegment) -> bool:
    """元の時刻で重なっていない 重なっていた行どうしは寄せた後も重ねたままでよい"""
    return earlier.end <= later.start


def fit_to_voice(
    times: Sequence[tuple[Fraction, Fraction]], loudness: np.ndarray
) -> list[tuple[Fraction, Fraction]]:
    """行ごとの ``(頭, 終わり)`` を、近くの声の出だしと終わりへ寄せる 並びは開始の昇順

    隣の行の元の時刻は越えない Whisper が 1 つの話を 2 行に割った所は声が続いているので、越えて
    よいことにすると前の行の頭まで戻って 2 行が重なる 声の大きさが読めない所（素材の外・周りが
    ほぼ無音）は元の時刻のまま返す
    """
    fitted: list[tuple[Fraction, Fraction]] = []
    for index, (start, end) in enumerate(times):
        low, high = _window(times, index)
        new_start = _start_edge(loudness, start, low, min(start + REACH, end))
        new_end = _end_edge(loudness, end, max(end - REACH, start), high)
        new_start = start if new_start is None else new_start
        new_end = end if new_end is None else new_end
        if new_end <= new_start:
            new_start, new_end = start, end
        fitted.append((new_start, new_end))
    return _untangled(times, fitted)


def found_edges(
    times: Sequence[tuple[Fraction, Fraction]], loudness: np.ndarray
) -> list[tuple[bool, bool]]:
    """行ごとの ``(頭, 終わり)`` から :func:`fit_to_voice` と同じ窓で、声の境目が見つかったか

    寄せた結果が元の時刻と同じでも、境目の上にいたのか、窓の中に境目が無くて動かなかったのかは
    分からない 字幕の時刻の確かめは、設定のずらしが足されているかどうかを、どちらの位置から
    境目が見つかるかで見分ける（PR #451）
    """
    out: list[tuple[bool, bool]] = []
    for index, (start, end) in enumerate(times):
        low, high = _window(times, index)
        head = _start_edge(loudness, start, low, min(start + REACH, end))
        tail = _end_edge(loudness, end, max(end - REACH, start), high)
        out.append((head is not None, tail is not None))
    return out


def _window(times: Sequence[tuple[Fraction, Fraction]], index: int) -> tuple[Fraction, Fraction]:
    """``index`` の行の頭を探す窓の下端と、終わりを探す窓の上端

    隣の行の元の時刻は越えない 元の行が重なっている（認識が重ねて返した）ときは、隣の行で窓を
    絞らない 絞ると窓の向きが逆になる
    """
    start, end = times[index]
    before = times[index - 1][1] if index > 0 else None
    after = times[index + 1][0] if index + 1 < len(times) else None
    low = start - REACH if before is None or before > start else max(start - REACH, before)
    high = end + REACH if after is None or after < end else min(end + REACH, after)
    return low, high


def _untangled(
    times: Sequence[tuple[Fraction, Fraction]], fitted: list[tuple[Fraction, Fraction]]
) -> list[tuple[Fraction, Fraction]]:
    """寄せた後に前の行の終わりが次の行の頭を越えた所を、次の行の頭で止める

    元から重なっていた行はそのまま 並びは開始の昇順に保つ（字幕の不変条件）
    """
    out = list(fitted)
    for index in range(1, len(out)):
        start, end = out[index]
        previous_start, previous_end = out[index - 1]
        if start < previous_start:
            start = previous_start
        if times[index - 1][1] <= times[index][0] and previous_end > start:
            previous_end = max(start, previous_start)
            out[index - 1] = (previous_start, previous_end)
        out[index] = (start, max(start, end))
    return out


def _start_edge(
    loudness: np.ndarray, at: Fraction, low: Fraction, high: Fraction
) -> Fraction | None:
    """窓の中の声の出だし 見つからなければ ``None``（呼ぶ側が元の時刻のまま残す）"""
    window = _voiced(loudness, low, high, START_QUIET_DB)
    if window is None:
        return None
    first, voiced = window
    here = min(max(_step(at) - first, 0), len(voiced) - 1)
    if voiced[here]:
        # 声の途中 頭が窓の中にあればそこまで戻す 窓の端まで声が続くなら、前の話と続いて
        # いて頭が分からないので動かさない
        head = here
        while head > 0 and voiced[head - 1]:
            head -= 1
        return _seconds(first + head) if head > 0 else None
    ahead = np.flatnonzero(voiced[here:])
    return _seconds(first + here + int(ahead[0])) if ahead.size else None


def _end_edge(loudness: np.ndarray, at: Fraction, low: Fraction, high: Fraction) -> Fraction | None:
    """窓の中の声の終わり 見つからなければ ``None``（呼ぶ側が元の時刻のまま残す）"""
    window = _voiced(loudness, low, high, END_QUIET_DB)
    if window is None:
        return None
    first, voiced = window
    # 終わりの時刻を含むコマ（終わりのちょうど前の 10 ms）
    here = min(max(_step(at) - first - 1, 0), len(voiced) - 1)
    if voiced[here]:
        tail = here
        while tail < len(voiced) - 1 and voiced[tail + 1]:
            tail += 1
        return _seconds(first + tail + 1) if tail < len(voiced) - 1 else None
    behind = np.flatnonzero(voiced[: here + 1])
    return _seconds(first + int(behind[-1]) + 1) if behind.size else None


def _voiced(
    loudness: np.ndarray, low: Fraction, high: Fraction, quiet_db: float
) -> tuple[int, np.ndarray] | None:
    """``[low, high)`` の声のあるコマ 窓の頭のコマの番号と一緒に返す 読めなければ ``None``"""
    first = max(0, _step(low))
    last = min(len(loudness), _step(high))
    if last - first < 1:
        return None
    reach = _step(_REFERENCE_REACH)
    around = loudness[max(0, first - reach) : min(len(loudness), last + reach)]
    spoken = around[around > _NEARLY_SILENT_DB]
    if spoken.size == 0:
        return None
    reference = float(np.percentile(spoken, _REFERENCE_PERCENTILE))
    floor = float(np.percentile(around, _FLOOR_PERCENTILE))
    threshold = max(reference + quiet_db, floor + _FLOOR_MARGIN_DB)
    if threshold >= reference:
        # 話と雑音の差が無い（音楽が鳴り続ける・雑音の方が大きい）所は、声の境目を読めない
        return None
    voiced = loudness[first:last] >= threshold
    return first, _closed(_held(voiced))


def _held(voiced: np.ndarray) -> np.ndarray:
    """:data:`_HOLD_STEPS` コマ続かない山を落とす"""
    if voiced.size < _HOLD_STEPS:
        return np.zeros_like(voiced)
    runs = np.lib.stride_tricks.sliding_window_view(voiced, _HOLD_STEPS).all(axis=1)
    kept = np.zeros_like(voiced)
    for offset in range(_HOLD_STEPS):
        kept[offset : offset + runs.size] |= runs
    return kept


def _closed(voiced: np.ndarray) -> np.ndarray:
    """:data:`_GAP_STEPS` コマ以下の谷を埋める"""
    filled = voiced.copy()
    marks = np.flatnonzero(voiced)
    for left, right in pairwise(marks):
        if 1 < right - left <= _GAP_STEPS:
            filled[left:right] = True
    return filled


def _step(seconds: Fraction) -> int:
    return int(seconds / STEP)


def _seconds(step: int) -> Fraction:
    return step * STEP
