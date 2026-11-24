"""登録したフォルダの素材を測る・見本の絵と波形を作る・下見で再生する（エクスプローラー #521 #522）

測るのは長さ・絵の大きさとコマ数と、頭の :data:`MEASURE_SECONDS` 秒の音の大きさ（積分
ラウドネスと山） AI が効果音と BGM を選ぶ手がかりにし、エクスプローラーの列にも出す
（2026-11-24 本人の決定 雰囲気は見積もらない） 全部をデコードしないのは、数分の BGM を数十本
測るだけで数十秒待たせるため 頭の 1 分で曲の大きさの見当は付く

下見の再生（:class:`PreviewPlayer`）は作品の再生（:class:`~sashimono.engine.audio.player.
AudioPlayer`）とは別に鳴らす 作品の再生の時計を動かすと、再生位置と映像が下見の音へ引っ張られる
デコードは裏のスレッドで行い、頭から全部は読まない（今の位置から読む） 長い動画でも押してすぐ鳴る
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import numpy as np

from sashimono.core.material_library import Measure
from sashimono.engine.checks.loudness import LoudnessMeter
from sashimono.engine.decode.audio import AudioDecoder
from sashimono.engine.decode.image import read_image
from sashimono.engine.decode.probe import ProbeError, probe_media
from sashimono.engine.decode.video import VideoDecoder

__all__ = [
    "MEASURE_SECONDS",
    "PREVIEW_EDGE",
    "AudioSink",
    "PreviewPlayer",
    "measure_library_file",
    "sound_peaks",
    "still_of",
]

#: 音の大きさを測る長さ（頭から 秒）
MEASURE_SECONDS = 60
#: 測るときと下見のサンプリングレート
_RATE = 48000
#: 一度にデコードする長さ（サンプル）
_CHUNK = _RATE * 5
#: 下見で出す絵の長い辺の上限（画素） 4K の絵をそのまま毎コマ画面へ渡すと、縮める手間で
#: 画面のスレッドが詰まる 下見の場所はパネルの中なので、この大きさで足りる
PREVIEW_EDGE = 960
#: 下見の音を出す 1 回の長さ（サンプル） 短いと一時停止がすぐ効くが、書く回数が増える
_BLOCK = 2048
#: 動画の絵を読みに行く間隔（秒） 60 コマの素材でも追い付く
_FRAME_POLL = 1 / 60
#: 波形の見本を作るときのサンプリングレート 見本は数百の点なので粗くてよく、長い BGM でも速い
_PEAK_RATE = 8000


def measure_library_file(path: Path) -> Measure | None:
    """``path`` の長さ・絵の大きさとコマ数・音の大きさ 開けなければ ``None``"""
    try:
        item = probe_media(path)
    except (ProbeError, OSError):
        return None
    seconds = float(item.duration)
    width = height = 0
    fps = 0.0
    if item.has_video:
        stream = item.video_streams[0]
        width, height = stream.display_size
        rate = stream.frame_rate
        fps = round(rate.num / rate.den, 3) if rate.den else 0.0
    if not item.has_audio:
        return Measure(
            seconds=seconds,
            has_audio=False,
            has_video=item.has_video,
            width=width,
            height=height,
            fps=fps,
        )
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
        return Measure(
            seconds=seconds,
            has_audio=True,
            has_video=item.has_video,
            width=width,
            height=height,
            fps=fps,
        )
    loudness = meter.integrated()
    peak = meter.peak
    return Measure(
        seconds=seconds,
        has_audio=True,
        has_video=item.has_video,
        loudness=round(loudness, 1) if loudness is not None and math.isfinite(loudness) else None,
        peak=round(20 * math.log10(peak), 1) if peak > 0 else None,
        width=width,
        height=height,
        fps=fps,
    )


def still_of(path: Path) -> np.ndarray | None:
    """見本の絵（画像はその絵 動画は頭の 1 コマ） RGBA uint8 読めなければ ``None``"""
    return read_image(path)


def sound_peaks(path: Path, columns: int) -> np.ndarray | None:
    """波形の見本 ``columns`` 個の 0〜1 の山 読めなければ ``None``

    全体を粗いレートで読む 下見の波形は全体の形が分かればよく、再生の位置の線を引く先になる
    """
    if columns <= 0:
        return None
    try:
        with AudioDecoder(path, sample_rate=_PEAK_RATE, channels=1) as decoder:
            total = int(decoder.duration * _PEAK_RATE)
            if total <= 0:
                return None
            samples = np.abs(decoder.read(0, total)[:, 0])
    except (ProbeError, OSError, ValueError):
        return None
    edges = np.linspace(0, len(samples), columns + 1).astype(int)
    peaks = np.array(
        [samples[a:b].max() if b > a else 0.0 for a, b in pairwise(edges)],
        dtype=np.float32,
    )
    return np.clip(peaks, 0.0, 1.0)


class AudioSink(Protocol):
    """下見の音の出口 本物は ``sounddevice`` の流れ 試験は鳴らさずに受け取る"""

    def write(self, samples: np.ndarray) -> None: ...

    def close(self) -> None: ...


def _device_sink(rate: int, channels: int) -> AudioSink:
    import sounddevice as sd

    stream = sd.OutputStream(samplerate=rate, channels=channels, dtype="float32")
    stream.start()

    class _Sink:
        def write(self, samples: np.ndarray) -> None:
            stream.write(np.ascontiguousarray(samples, dtype=np.float32))

        def close(self) -> None:
            stream.stop()
            stream.close()

    return _Sink()


@dataclass(frozen=True, slots=True)
class PreviewInfo:
    duration: float
    has_video: bool
    has_audio: bool
    is_still: bool


class PreviewPlayer:
    """エクスプローラーの中の小さなプレーヤー 1 つのファイルを再生・一時停止・シークする

    時計は経った時間で数える 音は書くたびに出口が待つので、時計とほぼそろって進む
    （下見なので、作品の再生ほど口の合いを詰めない） 音と絵は別のスレッドで読み、
    頼みの番号（``_generation``）が変わったら抜ける 止めた後に読み終えた古い頼みを鳴らさない
    """

    def __init__(
        self,
        *,
        sink: Callable[[int, int], AudioSink] = _device_sink,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._lock = threading.Lock()
        self._generation = 0
        self._path: Path | None = None
        self._info: PreviewInfo | None = None
        self._volume = 1.0
        #: 止まっている間の位置 再生中は ``_anchor`` から数える
        self._position = 0.0
        #: 再生を始めた（位置, 時計の値） 止まっていれば ``None``
        self._anchor: tuple[float, float] | None = None
        self._frame: np.ndarray | None = None
        self._frame_number = 0
        self._error: str | None = None

    # --- 外から ---

    def open(self, path: Path) -> PreviewInfo | None:
        """``path`` を開く 前のファイルは止める 開けなければ ``None``"""
        self.close()
        try:
            item = probe_media(path)
        except (ProbeError, OSError) as exc:
            with self._lock:
                self._error = f"開けません: {exc}"
            return None
        info = PreviewInfo(
            duration=float(item.duration),
            has_video=item.has_video,
            has_audio=item.has_audio,
            is_still=item.is_still,
        )
        with self._lock:
            self._path = path
            self._info = info
            self._position = 0.0
            self._error = None
        if info.has_video or info.is_still:
            self._start_still(0.0)
        return info

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            self._path = None
            self._info = None
            self._anchor = None
            self._position = 0.0
            self._frame = None

    @property
    def path(self) -> Path | None:
        with self._lock:
            return self._path

    @property
    def info(self) -> PreviewInfo | None:
        with self._lock:
            return self._info

    @property
    def playing(self) -> bool:
        with self._lock:
            self._settle()
            return self._anchor is not None

    @property
    def position(self) -> float:
        with self._lock:
            self._settle()
            return self._now()

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def frame(self) -> tuple[int, np.ndarray] | None:
        """今出す絵と、その番号（変わったかを見分ける） 無ければ ``None``"""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame_number, self._frame

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, volume))

    def play(self) -> None:
        with self._lock:
            info, path = self._info, self._path
            if info is None or path is None or info.is_still or self._anchor is not None:
                return
            start = self._position
            if info.duration > 0 and start >= info.duration - 0.05:
                # 終わりで止まっていたら頭から（メディアブラウザーと同じ）
                start = 0.0
            self._generation += 1
            generation = self._generation
            self._position = start
            self._anchor = (start, self._clock())
            self._error = None
        if info.has_audio:
            self._spawn(self._play_sound, path, generation, start)
        if info.has_video:
            self._spawn(self._play_picture, path, generation, start)

    def pause(self) -> None:
        with self._lock:
            if self._anchor is None:
                return
            self._position = self._now()
            self._anchor = None
            self._generation += 1

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            info = self._info
            if info is None:
                return
            target = max(0.0, min(seconds, info.duration))
            was_playing = self._anchor is not None
            self._generation += 1
            self._anchor = None
            self._position = target
        if was_playing:
            self.play()
        elif info.has_video:
            self._start_still(target)

    # --- 中 ---

    def _now(self) -> float:
        if self._anchor is None:
            return self._position
        start, at = self._anchor
        return start + max(0.0, self._clock() - at)

    def _settle(self) -> None:
        """終わりまで来たら止まった形にする（錠の内で呼ぶ）"""
        info = self._info
        if self._anchor is None or info is None:
            return
        if self._now() >= info.duration:
            self._position = info.duration
            self._anchor = None
            self._generation += 1

    def _alive(self, generation: int) -> bool:
        with self._lock:
            self._settle()
            return generation == self._generation

    def _spawn(self, target: Callable[..., None], *arguments: object) -> None:
        threading.Thread(
            target=target, args=arguments, name="explorer-preview", daemon=True
        ).start()

    def _start_still(self, seconds: float) -> None:
        with self._lock:
            path, generation = self._path, self._generation
        if path is not None:
            self._spawn(self._decode_still, path, generation, seconds)

    def _decode_still(self, path: Path, generation: int, seconds: float) -> None:
        try:
            if seconds <= 0:
                tile = read_image(path)
            else:
                with VideoDecoder(path) as decoder:
                    tile = decoder.frame_at(Fraction(seconds).limit_denominator(100000))
        except (ProbeError, OSError, ValueError):
            tile = None
        if tile is not None:
            self._show(generation, tile)

    def _show(self, generation: int, tile: np.ndarray) -> None:
        small = _shrink(tile)
        with self._lock:
            if generation != self._generation:
                return
            self._frame = small
            self._frame_number += 1

    def _play_picture(self, path: Path, generation: int, start: float) -> None:
        del start
        try:
            with VideoDecoder(path, should_stop=lambda: not self._alive(generation)) as decoder:
                while self._alive(generation):
                    with self._lock:
                        now = self._now()
                    tile = decoder.frame_at(Fraction(now).limit_denominator(100000))
                    if tile is None:
                        break
                    self._show(generation, tile)
                    time.sleep(_FRAME_POLL)
        except (ProbeError, OSError, ValueError) as exc:
            self._fail(generation, f"絵を読めません: {exc}")

    def _play_sound(self, path: Path, generation: int, start: float) -> None:
        sink: AudioSink | None = None
        try:
            with AudioDecoder(path, sample_rate=_RATE, channels=2) as decoder:
                total = int(decoder.duration * _RATE)
                cursor = int(start * _RATE)
                sink = self._sink(_RATE, 2)
                while cursor < total and self._alive(generation):
                    count = min(_BLOCK, total - cursor)
                    with self._lock:
                        volume = self._volume
                    sink.write((decoder.read(cursor, count) * volume).astype(np.float32))
                    cursor += count
        except (ProbeError, OSError, ValueError) as exc:
            self._fail(generation, f"音を読めません: {exc}")
        except Exception as exc:
            # 出力の無い機械（PortAudio の失敗）でも落とさず、鳴らせない訳を画面へ出す
            self._fail(generation, f"音を鳴らせません: {exc}")
        finally:
            # 閉じる失敗は鳴り終えた後なので知らせない
            if sink is not None:
                with contextlib.suppress(Exception):
                    sink.close()

    def _fail(self, generation: int, message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._error = message


def _shrink(tile: np.ndarray) -> np.ndarray:
    """長い辺が :data:`PREVIEW_EDGE` を越えなくなるまで半分に間引く 縮める手間を裏で済ます"""
    while max(tile.shape[0], tile.shape[1]) > PREVIEW_EDGE:
        tile = tile[::2, ::2]
    return np.ascontiguousarray(tile)
