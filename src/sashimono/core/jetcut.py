"""無音区間から、タイムライン上で切る範囲を決める

素材の無音は**ソース秒**で得られる（:mod:`sashimono.engine.audio.silence`） それが
タイムラインのどこに当たるかは、その素材を使っているクリップごとに違う ここは
字幕の投影（:mod:`sashimono.core.projection`）と同じ変換を、区間に対して行う

同じ変換をもう 1 度書いているように見えるが、扱う対象が違う 字幕は「表示する
1 枚」、こちらは「消す範囲」で、丸め方向が逆になる 字幕は欠けないよう外側へ、
カットは発話を削らないよう内側へ丸める
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction

from sashimono.core.model import Clip, MediaId, Project
from sashimono.core.projection import subtitle_voice
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = ["FrameRange", "SourceRange", "merge_ranges", "plan_cuts"]

#: ソース秒の区間 ``[start, end)``
type SourceRange = tuple[Fraction, Fraction]

#: タイムライン上のフレーム区間 ``[start, end)``
type FrameRange = tuple[int, int]


def plan_cuts(
    project: Project,
    media_id: MediaId,
    silences: Sequence[SourceRange],
    *,
    min_frames: int = 1,
    stream: int | None = None,
) -> tuple[FrameRange, ...]:
    """素材の無音区間を、タイムライン上で切る範囲へ落とす

    その素材を使っているクリップすべてが対象になる 同じ素材を 2 回置いていれば、
    2 か所とも切る 片方だけ切ると、切ったつもりの無い方に無音が残る

    映像と音声がリンクしている場合、両方が同じ範囲を返すので、重なりをまとめた
    時点で 1 つになる

    ``stream`` は無音を探した音声（素材の音声の番号） 渡せば、その音を鳴らすクリップの
    位置だけで落とす リンクを外して音声 2 だけを切り詰めていると、音声 1 のクリップの
    位置で音声 2 の無音を切ることになる（PR #231 の見直し） その音を鳴らすクリップが
    1 本も無ければ（映像だけ残したなど）、前と同じく素材のクリップすべてで落とす
    ``None`` は音で絞らない（1 本目の意味ではない） 無音を探した音があるなら、呼ぶ側が
    :meth:`MediaItem.transcript_stream` で番号にそろえて渡す
    """
    media = project.find_media(media_id)
    used = [
        (track, clip)
        for track in project.timeline.tracks
        for clip in track.clips
        if clip.media_id == media_id
    ]
    if stream is not None and media is not None:
        wanted = media.transcript_stream(stream)
        hearing = [(t, c) for t, c in used if subtitle_voice(project, t, c, media) == wanted]
        used = hearing or used
    ranges: list[FrameRange] = []
    for _, clip in used:
        ranges.extend(_project_clip(clip, silences, project.rate, min_frames))
    return merge_ranges(ranges)


def _project_clip(
    clip: Clip, silences: Sequence[SourceRange], rate: FrameRate, min_frames: int
) -> list[FrameRange]:
    """1 つのクリップに掛かる無音を、タイムラインのフレーム区間へ"""
    source_in = clip.source_in
    source_out = clip.source_out(rate)

    found: list[FrameRange] = []
    for start, end in silences:
        visible_start = max(start, source_in)
        visible_end = min(end, source_out)
        if visible_end <= visible_start:
            continue

        # 内側へ丸める 外側へ丸めると、無音の端にあるわずかな発話まで消える
        begin = clip.timeline_start + _to_offset(visible_start, clip, rate, Rounding.CEIL)
        stop = clip.timeline_start + _to_offset(visible_end, clip, rate, Rounding.FLOOR)

        begin = max(begin, clip.timeline_start)
        stop = min(stop, clip.timeline_end)
        if stop - begin >= min_frames:
            found.append((begin, stop))
    return found


def _to_offset(source_time: Fraction, clip: Clip, rate: FrameRate, rounding: Rounding) -> int:
    elapsed = (source_time - clip.source_in) / clip.speed
    return seconds_to_frame(elapsed, rate, rounding)


def merge_ranges(ranges: Iterable[FrameRange]) -> tuple[FrameRange, ...]:
    """重なる範囲・隣り合う範囲を 1 つにまとめ、開始位置順に並べる

    まとめておかないと、後ろから順に切っていく処理で範囲が二重に効く
    """
    ordered = sorted((start, end) for start, end in ranges if end > start)
    if not ordered:
        return ()

    merged: list[FrameRange] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)
