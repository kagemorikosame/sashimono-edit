"""声のクリップの話している所を拾う（ダッキング #329）

字幕があれば字幕の区間を使う 認識器が文字を起こせた所は確かに話している 波形だけでは
小さな声と環境音を分けられない 字幕が無ければ、無音カットと同じ波形（:mod:`.silence`）の
無音の外側を話している所とみなす

AI の道具（``duck_audio``）とタイムラインの右クリック〔BGM を声に合わせて下げる…〕が
同じ所を通る 入口ごとに書くと、同じ作品で人と AI の下げる所が食い違う
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from sashimono.core.jetcut import FrameRange
from sashimono.core.mixing import speech_frames
from sashimono.core.model import Clip, MediaItem, Project, Track
from sashimono.core.projection import subtitle_voice
from sashimono.engine.audio.silence import SilenceOptions, TimeRange, detect_silence
from sashimono.engine.audio.waveform import Waveform

__all__ = [
    "DEFAULT_SPEECH_THRESHOLD_DB",
    "SPEECH_SOURCES",
    "SpeechError",
    "WaveformPendingError",
    "speaks",
    "speech_ranges",
    "voice_speech",
]

#: 話している所を何から拾うか ``auto`` は字幕があれば字幕、無ければ波形
SPEECH_SOURCES = ("auto", "subtitles", "waveform")
#: 波形から拾うとき、これより大きい所を声とみなす（dBFS） 無音カットの既定と同じ
DEFAULT_SPEECH_THRESHOLD_DB = -40.0
#: 波形から拾うときの無音の最短（秒） これより短い息継ぎは無音として数えない
#: ダッキングは下げる長さと戻る長さの和より短い間をつなぐので、細かく拾っても線は増えない
_SPEECH_GAP = Fraction(1, 10)

#: 素材と音声の番号から、用意できていれば波形を返す（無ければ ``None``）
type WaveformOf = Callable[[MediaItem, int], Waveform | None]


class SpeechError(ValueError):
    """話している所を拾えない 文面はそのまま人と AI へ出す"""


class WaveformPendingError(SpeechError):
    """波形がまだ用意できていない 呼ぶ側は解析を頼んでから、少し待って呼び直す"""

    def __init__(self, media: MediaItem) -> None:
        super().__init__(f"{media.name} の波形解析がまだ終わっていません 少し待ってください")
        self.media = media


def speech_ranges(
    media: MediaItem,
    stream: int,
    waveform_of: WaveformOf,
    *,
    source: str = "auto",
    threshold_db: float = DEFAULT_SPEECH_THRESHOLD_DB,
) -> list[TimeRange]:
    """素材の話している区間（素材の秒）"""
    transcript = media.transcript_for(stream)
    if source != "waveform" and transcript is not None and transcript.segments:
        return [(segment.start, segment.end) for segment in transcript.segments]
    if source == "subtitles":
        raise SpeechError(f"{media.name} の字幕がありません 先に字幕を起こしてください")
    waveform = waveform_of(media, stream)
    if waveform is None:
        raise WaveformPendingError(media)
    silences = detect_silence(
        waveform,
        SilenceOptions(threshold_db=threshold_db, min_silence=_SPEECH_GAP, padding=Fraction(0)),
    )
    speech: list[TimeRange] = []
    cursor = Fraction(0)
    for start, end in silences:
        if start > cursor:
            speech.append((cursor, start))
        cursor = max(cursor, end)
    if waveform.duration > cursor:
        speech.append((cursor, waveform.duration))
    return speech


def speaks(project: Project, clip: Clip) -> bool:
    """声として話している所を拾えるクリップか（音を持つ素材のクリップ）

    置いたシーンは音を鳴らすが、素材の字幕も波形も持たない 声のトラックにナレーションと
    並べて置いてあることがあるので、拾えない物として外す（#375 の Codex の指摘）
    """
    media = project.find_media(clip.media_id) if clip.media_id is not None else None
    return media is not None and media.has_audio


def voice_speech(
    project: Project,
    voices: Sequence[tuple[Track, Clip]],
    waveform_of: WaveformOf,
    *,
    source: str = "auto",
    threshold_db: float = DEFAULT_SPEECH_THRESHOLD_DB,
) -> list[FrameRange]:
    """声のクリップの話している区間（タイムラインのフレーム）

    鳴らしている音（音声トラックは stream_index、混合は audio_stream）の字幕と波形を使う
    別の音の字幕で下げると、2 か国語の素材で話していない所の BGM が下がる
    """
    if source not in SPEECH_SOURCES:
        raise SpeechError(f"話している所の拾い方は {'・'.join(SPEECH_SOURCES)} のどれかです")
    usable = [(track, clip) for track, clip in voices if speaks(project, clip)]
    if not usable:
        # 置いたシーンなどが混ざっていても、拾えるナレーションがあればそれで下げる
        # 1 つも無いときだけ断る 1 本でも混ざると全体を断ると、声のトラックごと選べない
        raise SpeechError("声のクリップがありません（置いたシーンからは話している所を拾えない）")
    speech: list[FrameRange] = []
    for track, clip in usable:
        media = project.find_media(clip.media_id) if clip.media_id is not None else None
        if media is None:
            continue
        stream = subtitle_voice(project, track, clip, media)
        ranges = speech_ranges(media, stream, waveform_of, source=source, threshold_db=threshold_db)
        speech.extend(
            (clip.timeline_start + start, clip.timeline_start + end)
            for start, end in speech_frames(clip, ranges, project.rate)
        )
    return speech
