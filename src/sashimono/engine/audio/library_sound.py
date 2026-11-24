"""登録したフォルダの素材を測る・試聴する（エクスプローラー #521 #522）

測るのは長さと、頭の :data:`MEASURE_SECONDS` 秒の音の大きさ（積分ラウドネスと山） AI が
効果音と BGM を選ぶ手がかりにする（2026-11-24 本人の決定 雰囲気は見積もらない）
全部をデコードしないのは、数分の BGM を数十本測るだけで数十秒待たせるため 頭の 1 分で
曲の大きさの見当は付く（静かな前奏だけの曲は小さく出るので、返事にそう書く）

試聴は作品の再生（:class:`~sashimono.engine.audio.player.AudioPlayer`）とは別に鳴らす
作品の再生の時計を動かすと、再生位置と映像が試聴の音へ引っ張られる
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np

from sashimono.core.material_library import Measure
from sashimono.engine.checks.loudness import LoudnessMeter
from sashimono.engine.decode.audio import AudioDecoder
from sashimono.engine.decode.probe import ProbeError, probe_media

__all__ = [
    "MEASURE_SECONDS",
    "PREVIEW_SECONDS",
    "SoundPreview",
    "measure_library_file",
]

#: 音の大きさを測る長さ（頭から 秒）
MEASURE_SECONDS = 60
#: 試聴で鳴らす長さの上限（秒） 長い BGM を最後まで読んでから鳴らすと、押してから音が出るまで待つ
PREVIEW_SECONDS = 30
#: 測るときと試聴のサンプリングレート
_RATE = 48000
#: 一度にデコードする長さ（サンプル）
_CHUNK = _RATE * 5


def measure_library_file(path: Path) -> Measure | None:
    """``path`` の長さと音の大きさ 開けなければ ``None``"""
    try:
        item = probe_media(path)
    except (ProbeError, OSError):
        return None
    seconds = float(item.duration)
    if not item.has_audio:
        return Measure(seconds=seconds, has_audio=False, has_video=item.has_video)
    try:
        with AudioDecoder(path, sample_rate=_RATE, channels=2) as decoder:
            meter = LoudnessMeter(_RATE, 2)
            total = min(int(decoder.duration * _RATE), MEASURE_SECONDS * _RATE)
            cursor = 0
            while cursor < total:
                count = min(_CHUNK, total - cursor)
                meter.add(decoder.read(cursor, count).astype(np.float64))
                cursor += count
    except (ProbeError, OSError, ValueError):
        # 音の道が壊れていても長さは分かる 大きさは測れなかったとだけ返す
        return Measure(seconds=seconds, has_audio=True, has_video=item.has_video)
    loudness = meter.integrated()
    peak = meter.peak
    return Measure(
        seconds=seconds,
        has_audio=True,
        has_video=item.has_video,
        loudness=round(loudness, 1) if loudness is not None and math.isfinite(loudness) else None,
        peak=round(20 * math.log10(peak), 1) if peak > 0 else None,
    )


class SoundPreview:
    """エクスプローラーの試聴 1 度に鳴らすのは 1 つだけ

    読むのは裏のスレッド 押した後に画面を止めない 鳴らすのは ``sounddevice`` の
    ``play``（作品の再生とは別の流れ）
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: 鳴らす頼みの番号 止めた後に読み終えた古い頼みを鳴らさない
        self._generation = 0
        self._playing: Path | None = None

    @property
    def playing(self) -> Path | None:
        with self._lock:
            return self._playing

    def play(
        self, path: Path, volume: float, on_error: Callable[[str], None] | None = None
    ) -> None:
        """``path`` を頭から鳴らす ``volume`` は 0〜1 の倍率 前の音は止める"""
        self.stop()
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._playing = path
        thread = threading.Thread(
            target=self._run,
            args=(path, max(0.0, min(1.0, volume)), generation, on_error),
            name="sound-preview",
            daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            was_playing = self._playing is not None
            self._playing = None
        if was_playing:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                return

    def _run(
        self, path: Path, volume: float, generation: int, on_error: Callable[[str], None] | None
    ) -> None:
        try:
            with AudioDecoder(path, sample_rate=_RATE, channels=2) as decoder:
                count = min(int(decoder.duration * _RATE), PREVIEW_SECONDS * _RATE)
                samples = decoder.read_seconds(Fraction(0), Fraction(count, _RATE))
        except (ProbeError, OSError, ValueError) as exc:
            self._fail(generation, on_error, f"試聴できません: {exc}")
            return
        with self._lock:
            if generation != self._generation:
                return
        try:
            import sounddevice as sd

            sd.play((samples * volume).astype(np.float32), _RATE)
        except Exception as exc:
            self._fail(generation, on_error, f"音を鳴らせません: {exc}")

    def _fail(self, generation: int, on_error: Callable[[str], None] | None, message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._playing = None
        if on_error is not None:
            on_error(message)
