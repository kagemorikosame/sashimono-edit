"""完成物の検査（#337）の試験に使う素材を、その場で PyAV で焼く

ほかの人が作った素材はリポジトリに入れない どこに何の問題があるかが分かっている絵と音を作る

- :func:`write_video` 粗い模様が毎コマ動く絵 ``frozen`` の秒の間だけ同じ絵を出し続ける
- :func:`write_tone` 1 kHz の正弦波 振幅をフルスケールの dB で決める（左右同じ）
  両方のチャンネルに同じ振幅 A の正弦波を入れると、BS.1770 の積分ラウドネスは 20 log10(A)
  （1 kHz で K 特性の増幅と式の -0.691 が打ち消し合う） -14 dBFS の正弦波は -14 LUFS
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

__all__ = ["FPS", "HEIGHT", "RATE", "WIDTH", "moving_frame", "write_tone", "write_video"]

WIDTH, HEIGHT, FPS = 320, 180, 30
RATE = 48000


def moving_frame(index: int) -> np.ndarray:
    """明るさが中ほどの粗い模様 1 コマで 6 画素ずつ横へ流れる（止まりにも黒にもならない）"""
    rng = np.random.default_rng(7)
    blocks = rng.integers(60, 230, (HEIGHT // 10 + 2, WIDTH // 10 + 40, 3), dtype=np.uint8)
    wide = np.kron(blocks, np.ones((10, 10, 1), dtype=np.uint8))
    left = (index * 6) % (wide.shape[1] - WIDTH)
    return np.ascontiguousarray(wide[:HEIGHT, left : left + WIDTH])


def write_video(
    path: Path,
    seconds: float,
    *,
    frozen: tuple[float, float] | None = None,
    black: tuple[float, float] | None = None,
    tone_db: float | None = None,
    clipped: tuple[float, float] | None = None,
    silent: tuple[float, float] | None = None,
) -> Path:
    """動く絵の動画を焼く 区間（秒）を渡すと、そこだけ問題のある絵と音にする

    ``tone_db`` を渡すと音も付ける（AAC） ``clipped`` の間は振幅 2 の正弦波を ±1 で
    切った音（頭打ちが続く）、``silent`` の間は無音
    """
    frames = round(seconds * FPS)
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=FPS)
        if not isinstance(stream, av.VideoStream):
            raise TypeError(f"{codec} が映像の流れを作らなかった")
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        # 画質の揺れで止めた絵に差が出ないよう、量子化を固定する
        stream.options = {"crf": "18"} if codec == "libx264" else {"qscale": "2"}
        audio = None
        if tone_db is not None:
            made = container.add_stream("aac", rate=RATE)
            if not isinstance(made, av.AudioStream):
                raise TypeError("aac が音の流れを作らなかった")
            audio = made
            audio.layout = "stereo"
        for index in range(frames):
            second = index / FPS
            if black is not None and black[0] <= second < black[1]:
                image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            elif frozen is not None and frozen[0] <= second < frozen[1]:
                image = moving_frame(round(frozen[0] * FPS))
            else:
                image = moving_frame(index)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None and tone_db is not None:
            samples = tone(seconds, tone_db, clipped=clipped, silent=silent)
            _encode_audio(container, audio, samples)
    return path


def tone(
    seconds: float,
    db: float,
    *,
    clipped: tuple[float, float] | None = None,
    silent: tuple[float, float] | None = None,
) -> np.ndarray:
    """``(サンプル数, 2)`` の float32 の 1 kHz の正弦波"""
    count = round(seconds * RATE)
    times = np.arange(count) / RATE
    wave = 10.0 ** (db / 20.0) * np.sin(2.0 * np.pi * 1000.0 * times)
    if clipped is not None:
        part = slice(round(clipped[0] * RATE), round(clipped[1] * RATE))
        wave[part] = np.clip(2.0 * np.sin(2.0 * np.pi * 1000.0 * times[part]), -1.0, 1.0)
    if silent is not None:
        wave[round(silent[0] * RATE) : round(silent[1] * RATE)] = 0.0
    return np.stack([wave, wave], axis=1).astype(np.float32)


def write_tone(path: Path, samples: np.ndarray) -> Path:
    """``samples``（``(サンプル数, 2)``）を 16 bit の WAV に書く 圧縮しないので値がそのまま残る"""
    with av.open(str(path), "w") as container:
        made = container.add_stream("pcm_s16le", rate=RATE)
        if not isinstance(made, av.AudioStream):
            raise TypeError("pcm_s16le が音の流れを作らなかった")
        made.layout = "stereo"
        _encode_audio(container, made, samples)
    return path


def _encode_audio(
    container: av.container.OutputContainer, stream: av.AudioStream, samples: np.ndarray
) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    step = 1024
    for start in range(0, len(pcm), step):
        block = np.ascontiguousarray(pcm[start : start + step].T.astype(np.float32))
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
        frame.sample_rate = RATE
        frame.pts = start
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
