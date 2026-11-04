"""語の時刻の試験（#420）で使う作品と波形を組む

語は Whisper の日本語と同じく 1 字ずつ持たせる 実物の語の時刻がこの形だから
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    Word,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio.waveform import PeakLevel, Waveform

RATE = FrameRate(30)
_SAMPLE_RATE = 48000
_SAMPLES_PER_PEAK = 480
_LOUD = 0.5
_QUIET = 0.001


def spoken(start: Fraction, text: str, step: Fraction = Fraction(1, 5)) -> tuple[Word, ...]:
    """``start`` 秒から 1 字 ``step`` 秒ずつ話す語の並び"""
    return tuple(
        Word(start + index * step, start + (index + 1) * step, letter)
        for index, letter in enumerate(text)
    )


def segment(words: tuple[Word, ...], **fields: object) -> TranscriptSegment:
    text = "".join(word.text for word in words)
    return TranscriptSegment(words[0].start, words[-1].end, text, words, **fields)  # type: ignore[arg-type]


def sample_transcript() -> Transcript:
    """「カメラを」1.0〜1.8 秒「作ります」1.8〜2.6 秒 間を置いて「はい」4.0〜4.4 秒"""
    return Transcript(
        (
            segment(spoken(Fraction(1), "カメラを作ります")),
            segment(spoken(Fraction(4), "はい")),
        ),
        language="ja",
    )


def wave(
    quiet: tuple[tuple[float, float], ...] = ((0.0, 1.0), (2.6, 4.0), (4.4, 12.0)),
    seconds: float = 12.0,
    floor: float = _QUIET,
) -> Waveform:
    """話している所は大きく、``quiet`` の区間（秒）だけ静かな波形 既定は sample_transcript に合う

    ``floor`` は静かな所の振幅 雑音のある部屋の録音を作るときに上げる
    """
    count = round(seconds * _SAMPLE_RATE / _SAMPLES_PER_PEAK)
    amplitude = np.full(count, _LOUD, dtype=np.float32)
    per_second = _SAMPLE_RATE / _SAMPLES_PER_PEAK
    for low, high in quiet:
        amplitude[round(low * per_second) : round(high * per_second)] = floor
    peaks = np.stack((-amplitude, amplitude), axis=-1)[:, np.newaxis, :]
    return Waveform(
        sample_rate=_SAMPLE_RATE,
        channels=1,
        total_samples=count * _SAMPLES_PER_PEAK,
        levels=(PeakLevel(_SAMPLES_PER_PEAK, peaks),),
    )


def placed(media: MediaItem, *clips: Clip, kind: TrackKind = TrackKind.AUDIO) -> Project:
    """``media`` を持ち、1 本のトラックへ ``clips`` を置いた作品"""
    project = Project.create(ProjectSettings(frame_rate=RATE), media=(media,))
    track = Track(kind=kind, name="A1", clips=clips)
    return project.with_timeline(replace(project.timeline, tracks=(track,)))


def in_scene(project: Project, clip: Clip) -> Project:
    """``project`` のトラックの中身を 1 つのシーンに入れ、``clip`` として置いた作品"""
    scene = Scene(name="場面", timeline=project.timeline)
    placed_clip = replace(clip, scene_id=scene.id)
    outer = Track(kind=TrackKind.VIDEO, name="V1", clips=(placed_clip,))
    timeline = replace(project.timeline, tracks=(outer,))
    return replace(project, timeline=timeline, scenes=(scene,))
