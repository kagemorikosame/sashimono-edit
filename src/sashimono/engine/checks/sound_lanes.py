"""鳴るクリップを重ならない組へ分けて、リミッターの見る所も一緒に測る（#494）

リミッターはクリップごとに掛かる 混ぜた音だけを測ると、どのクリップがどれだけ抑えられるかも、
抑えた音どうしの和がどこまで上がるかも分からない 混ぜる途中の音を組ごとに分けて覚えれば、
2 回混ぜなくても見積もれる
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from itertools import combinations

import numpy as np

from sashimono.core.mixing import audible_clips, track_gain_db
from sashimono.core.model import ClipId, Project
from sashimono.engine.checks.loudness import (
    FineSums,
    LoudnessLane,
    LoudnessMeter,
    channel_weights,
)

__all__ = ["LaneMeters"]

#: 分ける組の上限 組と、組どうしの掛け合わせ（組の 2 つ選び）ごとに場所が要る 3 時間・4 組で
#: 170 MB ほど 越えたら分けない（見積もらずに、届いたかを言い切らない）
MAX_LANES = 4


class LaneMeters:
    """組ごとの音量計"""

    def __init__(
        self,
        lane_of: dict[ClipId, int],
        starts: Sequence[tuple[float, ...]],
        faders: Sequence[tuple[float, ...]],
        *,
        sample_rate: int,
        channels: int,
    ) -> None:
        #: クリップがどの組か ミキサへ渡す
        self.lane_of = lane_of
        self._starts = starts
        self._faders = faders
        #: まだ掛け合わせていない、K 特性を掛けた組ごとの音
        self._weighted: list[deque[np.ndarray]] = [deque() for _ in starts]
        self._meters = [
            LoudnessMeter(sample_rate, channels, tap=waiting.append) for waiting in self._weighted
        ]
        self._weights = channel_weights(channels)
        self._crosses = {
            pair: FineSums(self._meters[0].block_samples)
            for pair in combinations(range(len(starts)), 2)
        }
        self._detected = [FineSums(self._meters[0].block_samples, np.maximum) for _ in starts]

    @classmethod
    def of(cls, project: Project, start: int, end: int) -> LaneMeters | None:
        """``[start, end)`` で鳴るクリップを、頭の早い順に空いている組へ入れる

        鳴る物が無いか、組が :data:`MAX_LANES` を越えれば ``None`` 重ならなくても 1 組で測る
        （定位とトラックの音量があると、混ぜた山はリミッターの見る山と違う）
        """
        # 鳴る音の決まりはミキサと同じ所で見る 絵の側で映る混合トラックまで足すと、音のソロで
        # 外れた音が組に入る
        sounding = sorted(
            (
                (clip.timeline_start, clip.timeline_end, clip.id, track_gain_db(track))
                for track, clip in audible_clips(project)
                if clip.timeline_start < end and clip.timeline_end > start
            ),
            key=lambda row: row[:2],
        )
        lane_of: dict[ClipId, int] = {}
        reaches: list[int] = []
        starts: list[list[float]] = []
        faders: list[list[float]] = []
        frame = float(project.rate.frame_duration)
        for low, high, clip_id, fader in sounding:
            lane = next((at for at, reach in enumerate(reaches) if reach <= low), len(reaches))
            if lane == len(reaches):
                if lane >= MAX_LANES:
                    return None
                reaches.append(high)
                starts.append([])
                faders.append([])
            reaches[lane] = high
            lane_of[clip_id] = lane
            starts[lane].append((low - start) * frame)
            faders[lane].append(fader)
        if not reaches:
            return None
        return cls(
            lane_of,
            [tuple(lane_starts) for lane_starts in starts],
            [tuple(lane_faders) for lane_faders in faders],
            sample_rate=project.settings.sample_rate,
            channels=project.settings.channels,
        )

    @property
    def count(self) -> int:
        return len(self._meters)

    def add(self, sounds: np.ndarray, detected: np.ndarray) -> None:
        """``sounds`` は（組, サンプル, チャンネル）

        ``detected`` は（組, サンプル）の、リミッターの見る山
        """
        for meter, sound in zip(self._meters, sounds, strict=True):
            meter.add(sound)
        for peaks, seen in zip(self._detected, detected, strict=True):
            peaks.add(seen)
        self._multiply()

    def _multiply(self) -> None:
        # どの音量計も同じ長さを受け取るので、畳み込んだ塊の境目もそろう
        while all(self._weighted):
            blocks = [waiting.popleft() for waiting in self._weighted]
            for (first, second), sums in self._crosses.items():
                sums.add((blocks[first] * blocks[second]) @ self._weights)

    def lanes(self) -> tuple[LoudnessLane, ...]:
        found: list[LoudnessLane] = []
        for meter, peaks, starts, faders in zip(
            self._meters, self._detected, self._starts, self._faders, strict=True
        ):
            dynamics = meter.dynamics()
            if dynamics is None:
                return ()
            seen = np.concatenate([peaks.joined(), peaks.tail()])
            found.append(
                LoudnessLane(
                    dynamics.energy, dynamics.peak, starts, meter.tail_peaks(), seen, faders
                )
            )
        return tuple(found)

    def crosses(self) -> tuple[tuple[int, int, np.ndarray], ...]:
        """組どうしの掛け合わせ :meth:`lanes` の後で呼ぶ（溜めた音を吐かせてから数える）"""
        self._multiply()
        return tuple((*pair, sums.joined()) for pair, sums in self._crosses.items())
