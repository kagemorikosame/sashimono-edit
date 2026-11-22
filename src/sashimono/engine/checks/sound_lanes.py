"""重なって鳴るクリップを、重ならない組へ分けて測る（#494）

リミッターはクリップごとに掛かる 混ぜた音だけを測ると、どのクリップがどれだけ抑えられるかも、
抑えた音どうしの和がどこまで上がるかも分からない 混ぜる途中の音を組ごとに分けて覚えれば、
2 回混ぜなくても見積もれる
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sashimono.core.model import ClipId, Project
from sashimono.engine.checks.loudness import LoudnessLane, LoudnessMeter

__all__ = ["LaneMeters"]

#: 分ける組の上限 1 組ごとに 3 時間で 17 MB と K 特性の計算が増える 越えたら分けない
MAX_LANES = 6


class LaneMeters:
    """組ごとの音量計"""

    def __init__(
        self,
        lane_of: dict[ClipId, int],
        starts: Sequence[tuple[float, ...]],
        *,
        sample_rate: int,
        channels: int,
    ) -> None:
        #: クリップがどの組か ミキサへ渡す
        self.lane_of = lane_of
        self._starts = starts
        self._meters = [LoudnessMeter(sample_rate, channels) for _ in starts]

    @classmethod
    def of(cls, project: Project, start: int, end: int) -> LaneMeters | None:
        """``[start, end)`` で鳴るクリップを、頭の早い順に空いている組へ入れる

        重なりが無いか、組が :data:`MAX_LANES` を越えれば ``None``
        """
        timeline = project.timeline
        tracks = {
            track.id: track
            for track in (*timeline.active_sound_tracks(), *timeline.active_picture_tracks())
        }
        sounding = sorted(
            (
                (clip.timeline_start, clip.timeline_end, clip.id)
                for track in tracks.values()
                for clip in track.clips
                if clip.enabled
                and clip.timeline_start < end
                and clip.timeline_end > start
                and project.plays_sound(track, clip)
            ),
            key=lambda row: row[:2],
        )
        lane_of: dict[ClipId, int] = {}
        reaches: list[int] = []
        starts: list[list[float]] = []
        frame = float(project.rate.frame_duration)
        for low, high, clip_id in sounding:
            lane = next((at for at, reach in enumerate(reaches) if reach <= low), len(reaches))
            if lane == len(reaches):
                if lane >= MAX_LANES:
                    return None
                reaches.append(high)
                starts.append([])
            reaches[lane] = high
            lane_of[clip_id] = lane
            starts[lane].append((low - start) * frame)
        if len(reaches) < 2:
            return None
        return cls(
            lane_of,
            [tuple(lane_starts) for lane_starts in starts],
            sample_rate=project.settings.sample_rate,
            channels=project.settings.channels,
        )

    @property
    def count(self) -> int:
        return len(self._meters)

    def add(self, sounds: np.ndarray) -> None:
        """``sounds`` は（組, サンプル, チャンネル）"""
        for meter, sound in zip(self._meters, sounds, strict=True):
            meter.add(sound)

    def lanes(self) -> tuple[LoudnessLane, ...]:
        found: list[LoudnessLane] = []
        for meter, starts in zip(self._meters, self._starts, strict=True):
            dynamics = meter.dynamics()
            if dynamics is None:
                return ()
            found.append(LoudnessLane(dynamics.energy, dynamics.peak, starts))
        return tuple(found)
