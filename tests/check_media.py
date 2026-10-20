"""完成物の検査（#337）の試験に使う素材を、その場で PyAV で焼く

ほかの人が作った素材はリポジトリに入れない どこに何の問題があるかが分かっている絵と音を作る

- :func:`write_video` 粗い模様が毎コマ動く絵 ``frozen`` の秒の間だけ同じ絵を出し続ける
- :func:`write_tone` 1 kHz の正弦波 振幅をフルスケールの dB で決める（左右同じ）
  両方のチャンネルに同じ振幅 A の正弦波を入れると、BS.1770 の積分ラウドネスは 20 log10(A)
  （1 kHz で K 特性の増幅と式の -0.691 が打ち消し合う） -14 dBFS の正弦波は -14 LUFS
- :func:`talk` と :func:`chord` 話し声と音楽の見分け（#322）に使う音 本物の声と曲は
  使えないので、見分けの手がかり（音節と息継ぎの上下・ならされた和音）だけを持つ音を作る
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

__all__ = [
    "FPS",
    "HEIGHT",
    "RATE",
    "WIDTH",
    "chord",
    "moving_frame",
    "talk",
    "write_tone",
    "write_video",
]

WIDTH, HEIGHT, FPS = 320, 180, 30
RATE = 48000


def talk(seconds: float, *, seed: int = 0, level: float = 0.2) -> np.ndarray:
    """話し声らしい音 ``(サンプル数, 2)``

    倍音の多い声（110〜180 Hz）の音節（0.12〜0.25 秒）を短い間で並べ、1〜2 秒ごとに
    0.3〜0.6 秒の息継ぎを入れる 話し声の見分けの手がかり（大きさの上下と谷）を持つ
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(round(seconds * RATE))
    at = 0.0
    while at < seconds:
        phrase = at + rng.uniform(1.0, 2.0)
        while at < min(phrase, seconds):
            length = rng.uniform(0.12, 0.25)
            begin = round(at * RATE)
            count = min(round(length * RATE), len(out) - begin)
            times = np.arange(count) / RATE
            pitch = rng.uniform(110.0, 180.0)
            voice = sum(np.sin(2 * np.pi * pitch * h * times) / h for h in range(1, 12))
            out[begin : begin + count] += level * voice * np.hanning(count) * rng.uniform(0.4, 1.0)
            at += length + rng.uniform(0.02, 0.08)
        at += rng.uniform(0.3, 0.6)
    return np.stack([out, out], axis=1).astype(np.float32)


def chord(seconds: float, *, level: float = 0.1) -> np.ndarray:
    """ならされた和音（A の長三和音と倍音） ``(サンプル数, 2)`` 最大の振幅が ``level``"""
    times = np.arange(round(seconds * RATE)) / RATE
    wave = np.zeros_like(times)
    for pitch in (220.0, 277.18, 329.63):
        for harmonic in range(1, 5):
            wave += np.sin(2 * np.pi * pitch * harmonic * times) / harmonic
    wave *= level / np.max(np.abs(wave))
    return np.stack([wave, wave], axis=1).astype(np.float32)


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
    audio_codec: str = "aac",
    held: tuple[float, float] | None = None,
    audio_delay: float = 0.0,
    audio_gap: tuple[float, float] | None = None,
    audio_seconds: float | None = None,
) -> Path:
    """動く絵の動画を焼く 区間（秒）を渡すと、そこだけ問題のある絵と音にする

    ``tone_db`` を渡すと音も付ける（既定は書き出しと同じ AAC） ``clipped`` の間は
    振幅 2 の正弦波を ±1 で切った音（頭打ちが続く）、``silent`` の間は無音

    音の位置や大きさを細かく見る試験は ``audio_codec="pcm_s16le"`` と ``.mkv`` で焼く
    AAC は頭打ちの平らな所を崩す（手元の FFmpeg では 1.05〜1.17 秒の山が 0.965 まで
    下がった） 崩れ方はエンコーダの版で変わり、CI の Windows では音割れの終わりが
    0.17 秒手前に来た 圧縮しない PCM ならどの環境でも同じ値になる

    時刻（PTS）の試験は ``.mkv`` で焼く
    - ``held`` の間はコマを書かない（VFR） 頭のコマが次のコマまで出続ける
    - ``audio_delay`` 秒だけ音の頭を遅らせる（音は映像より遅れて始まる）
    - ``audio_gap``（音の秒）の間は音のフレームを書かず、PTS をその分だけ飛ばす
    - ``audio_seconds`` を渡すと、音だけその長さにする（映像より長い音）
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
            made = container.add_stream(audio_codec, rate=RATE)
            if not isinstance(made, av.AudioStream):
                raise TypeError(f"{audio_codec} が音の流れを作らなかった")
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
            if held is not None and held[0] < second < held[1] - 1e-9:
                continue
            picture = av.VideoFrame.from_ndarray(image, format="rgb24")
            picture.pts = index
            picture.time_base = Fraction(1, FPS)
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None and tone_db is not None:
            heard = audio_seconds if audio_seconds is not None else seconds
            samples = tone(heard - audio_delay, tone_db, clipped=clipped, silent=silent)
            _encode_audio(container, audio, samples, start=round(audio_delay * RATE), gap=audio_gap)
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
    container: av.container.OutputContainer,
    stream: av.AudioStream,
    samples: np.ndarray,
    *,
    start: int = 0,
    gap: tuple[float, float] | None = None,
) -> None:
    """``start`` サンプル目から書く ``gap``（秒 音の頭から）の間は書かずに PTS を飛ばす"""
    pcm = np.clip(samples, -1.0, 1.0)
    step = 1024
    skip = (round(gap[0] * RATE), round(gap[1] * RATE)) if gap is not None else None
    for offset in range(0, len(pcm), step):
        if skip is not None and skip[0] <= offset < skip[1]:
            continue
        block = np.ascontiguousarray(pcm[offset : offset + step].T.astype(np.float32))
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
        frame.sample_rate = RATE
        frame.pts = start + offset
        frame.time_base = Fraction(1, RATE)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
