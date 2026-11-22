"""選んだ書体に無い字を持つテキストを拾う（#494）

絵の画素からは、どの字が代わりの書体で出たのか分からない 本文と書体から数える 描かないので
GPU が無くても調べられる
"""

from __future__ import annotations

from collections.abc import Iterator

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
    for low, high, clip, scene_clip in _texts(project, project.timeline, start, end):
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
    project: Project, timeline: Timeline, start: int, end: int, depth: int = 0
) -> Iterator[tuple[int, int, Clip, Clip | None]]:
    """映るテキストと、外側のフレームでの区間 シーンの中は置いたクリップの区間で返す"""
    for track in timeline.active_picture_tracks():
        for clip in track.clips:
            if not clip.enabled or clip.timeline_end <= start or clip.timeline_start >= end:
                continue
            if not project.draws_picture(track, clip):
                continue
            low, high = max(start, clip.timeline_start), min(end, clip.timeline_end)
            if clip.source is not None and clip.source.kind == "text":
                yield low, high, clip, None
            if clip.scene_id is None or depth >= MAX_SCENE_DEPTH:
                continue
            scene = project.find_scene(clip.scene_id)
            if scene is None:
                continue
            first, last = (
                seconds_to_frame(clip.picture_time(at, project.rate), project.rate)
                for at in (low - clip.timeline_start, high - 1 - clip.timeline_start)
            )
            window = (min(first, last), max(first, last) + 1)
            for _low, _high, inner, _scene in _texts(project, scene.timeline, *window, depth + 1):
                yield low, high, inner, clip
