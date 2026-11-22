"""選んだ書体に無い字を持つテキストを拾う（#494）

絵の画素からは、どの字が代わりの書体で出たのか分からない 本文と書体から数える 描かないので
GPU が無くても調べられる
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterator

from sashimono.core.model import Clip, Project, Timeline
from sashimono.core.timebase import seconds_to_frame
from sashimono.engine.checks.scan import MISSING_GLYPH, Run
from sashimono.engine.render.renderer import MAX_SCENE_DEPTH
from sashimono.engine.sources import missing_glyphs

__all__ = ["glyph_runs"]

#: 返事に載せる字の数 読めない書体を選ぶと本文の全部が並ぶ
_SHOWN = 24


def glyph_runs(project: Project, start: int, end: int) -> list[Run]:
    """``[start, end)`` に映るテキストのうち、書体に無い字を持つ物"""
    found: list[Run] = []
    for low, high, clip, scene_clip in _texts(project, project.timeline, range(start, end)):
        assert clip.source is not None
        absent = missing_glyphs(clip.source)
        if not absent:
            continue
        values: dict[str, float | int | str | bool] = {
            "characters": absent[:_SHOWN],
            "font": str(clip.source.params.get("font", "")),
            "matched": bool(clip.source.params.get("match_fallback", False))
            and clip.source.params.get("layout") != "aviutl",
        }
        if scene_clip is not None:
            values["in_scene_clip"] = str(scene_clip.id)
        found.append(Run(MISSING_GLYPH, low, high, values, clip_ids=(str(clip.id),)))
    return found


def _texts(
    project: Project,
    timeline: Timeline,
    frames: range,
    inner_at: Callable[[int], int] = lambda frame: frame,
    depth: int = 0,
) -> Iterator[tuple[int, int, Clip, Clip | None]]:
    """``frames``（外側のコマ）に映るテキストと、映る区間

    ``inner_at`` は外側のコマから、そこに映る ``timeline`` のコマへの写し 速度は正なので戻らない
    区間の端だけを写すと、倍速で飛ばされて 1 コマも映らない字幕まで問題に数える
    """
    for track in timeline.active_picture_tracks():
        for clip in track.clips:
            if not clip.enabled or not project.draws_picture(track, clip):
                continue
            first = bisect_left(frames, clip.timeline_start, key=inner_at)
            last = bisect_left(frames, clip.timeline_end, key=inner_at)
            if first >= last:
                continue
            shown = frames[first:last]
            if clip.source is not None and clip.source.kind == "text":
                yield shown.start, shown.stop, clip, None
            if clip.scene_id is None or depth >= MAX_SCENE_DEPTH:
                continue
            scene = project.find_scene(clip.scene_id)
            if scene is None:
                continue

            def within(frame: int, placed: Clip = clip) -> int:
                seconds = placed.picture_time(inner_at(frame) - placed.timeline_start, project.rate)
                return seconds_to_frame(seconds, project.rate)

            for low, high, inner, _scene in _texts(
                project, scene.timeline, shown, within, depth + 1
            ):
                yield low, high, inner, clip
