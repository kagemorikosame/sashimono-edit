"""起こした行の時刻を声に合わせる（#426）

認識器が返した行の時刻を、音の大きさで声の出だしと終わりへ寄せ、本人の好みの分だけずらす
バックエンドに依らない後始末なので、faster-whisper の外に置く 偽の認識器でも同じ道を通せる
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sashimono.asr.backend import TranscribeOptions
from sashimono.core.model import TranscriptSegment
from sashimono.engine.audio.voice_edges import loudness_of_samples, retime

__all__ = ["adjust_timing"]


def adjust_timing(
    segments: Sequence[TranscriptSegment],
    options: TranscribeOptions,
    samples: np.ndarray | None,
    sample_rate: int,
) -> tuple[TranscriptSegment, ...]:
    """``options.timing`` に従って行の時刻を寄せてずらす 並びは開始の昇順

    ``samples`` は起こした音（モノラル） 読めなかった（長さの分からない素材をパスのまま渡した）
    ときは ``None`` で、寄せずにずらすだけにする 音を読み直すと長い素材でもう 1 回待たせる
    """
    timing = options.timing
    loudness = None
    if timing.fit_to_voice and samples is not None:
        loudness = loudness_of_samples(samples, sample_rate)
    return retime(segments, timing, loudness)
