"""起こした行の時刻を声に合わせる（#426）

faster-whisper は入れずに、決まった時刻を返す代わりのモデルで、起こしの道に寄せる処理が入って
いるかを見る 声の位置が分かる音（1.0〜2.0 秒だけ鳴る）を作り、0.3 秒早い行を返させる
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.asr.backend import TranscribeOptions
from sashimono.asr.whisper import FasterWhisperBackend
from sashimono.core.subtitle_timing import SubtitleTiming

RATE = 16000


class _Segment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words: list[Any] = []


class _Info:
    duration = 3.0
    language = "ja"


class _Model:
    """Whisper の癖をまねて、声より 0.3 秒早く始まり 0.3 秒遅く終わる行を返す"""

    def transcribe(self, audio: Any, **_: Any) -> tuple[list[_Segment], _Info]:
        del audio
        return [_Segment(0.7, 2.3, "あいうえお")], _Info()


def _voice(path: Path) -> Path:
    t = np.arange(3 * RATE) / RATE
    samples = np.where((t >= 1.0) & (t < 2.0), 0.5 * np.sin(2 * np.pi * 220 * t), 0.0)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((samples * 32767).astype(np.int16).tobytes())
    return path


def _times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: TranscribeOptions
) -> list[tuple[float, float]]:
    backend = FasterWhisperBackend()
    monkeypatch.setattr(backend, "_ensure_model", lambda options: _Model())
    transcript = backend.transcribe(_voice(tmp_path / "声.wav"), options)
    assert transcript is not None
    return [(float(s.start), float(s.end)) for s in transcript.segments]


def test_the_lines_are_moved_to_the_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 壊れると、字幕が声より早く出て遅く消える（利用者の報告 #426）
    options = TranscribeOptions(timing=SubtitleTiming())
    ((start, end),) = _times(tmp_path, monkeypatch, options)
    assert start == pytest.approx(1.0, abs=0.02)
    assert end == pytest.approx(2.0, abs=0.02)


def test_the_offsets_reach_the_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 壊れると、設定で選んだ「少し早く出す」が起こした字幕に効かない
    options = TranscribeOptions(timing=SubtitleTiming(start_offset_ms=-200, end_offset_ms=500))
    ((start, end),) = _times(tmp_path, monkeypatch, options)
    assert start == pytest.approx(0.8, abs=0.02)
    assert end == pytest.approx(2.5, abs=0.02)


def test_turned_off_the_lines_are_left_as_recognized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 切っても寄せると、音楽の上で話す素材で寄せ方が合わない人が逃げられない
    options = TranscribeOptions(timing=SubtitleTiming(fit_to_voice=False))
    assert _times(tmp_path, monkeypatch, options) == [(0.7, 2.3)]
