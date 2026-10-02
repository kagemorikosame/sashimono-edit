"""テスト用の実素材を ffmpeg で生成する

デコードや書き出しは実ファイルでしか検証できない バイナリをリポジトリに置くと
差分が読めなくなるので、必要なものをその場で作る 生成物はセッション内で使い回す
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "SampleMedia",
    "decode_all_frames",
    "encoder_available",
    "ffmpeg_available",
    "libx264_available",
    "make_color_tagged",
    "make_rotated",
    "make_sample",
    "make_silent_gap",
]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


#: 符号化器の一覧を待つ上限（秒）
ENCODER_LIST_TIMEOUT = 30


def libx264_available() -> bool:
    """外の ffmpeg で H.264 を焼けるか

    ffmpeg があっても libx264 が入っていない組み立て方があり、そこでは素材を
    作る所で落ちる 落ちると取り付け口（fixture）のエラーになり、「使えない
    環境では飛ばす」という他の場所の決まりと食い違うので、先に見て分ける
    """
    return encoder_available("libx264")


@functools.cache
def encoder_available(name: str) -> bool:
    """外の ffmpeg に符号化器 ``name`` が入っているか

    一覧を取るのに 100ms 前後かかるので、名前ごとに 1 度だけ見て覚える
    """
    if not ffmpeg_available():
        return False
    try:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            # 応答しない ffmpeg に当たると、待ち時間を切らないとテストの進行が
            # 止まったまま戻らない 一覧は 100ms 前後で返るので 30 秒で十分長い
            timeout=ENCODER_LIST_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if listing.returncode != 0:
        return False
    # 一覧は「 V....D libx264   libx264 H.264 ...」の形 説明文にも符号化器の
    # 名前が出るので、名前の欄（2 列目）が一致する行だけを数える
    # 手元の ffmpeg 8.1.2 は一覧を標準出力へ出すが、組み立て方によっては
    # 標準エラーへ出るという指摘があったので、両方を見る
    for line in (listing.stdout + "\n" + listing.stderr).splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[1] == name:
            return True
    return False


@dataclass(frozen=True, slots=True)
class SampleMedia:
    """生成した素材と、その素材が持っているはずの性質"""

    path: Path
    width: int
    height: int
    fps: str
    duration: float
    has_audio: bool
    sample_rate: int


def make_sample(
    directory: Path,
    name: str,
    *,
    width: int = 320,
    height: int = 240,
    fps: str = "30",
    duration: float = 2.0,
    audio: bool = True,
    sample_rate: int = 44100,
    tone_hz: int = 440,
    #: 音量の増幅（dB） lavfi の sine は振幅が 0.1 程度しかないので、
    #: 実運用に近い波形が要るときに持ち上げる
    gain_db: float = 0.0,
    pattern: str = "testsrc2",
    keyframe_interval: int | None = None,
) -> SampleMedia:
    """ffmpeg でテスト素材を作る すでにあればそれを返す

    ``testsrc2`` はフレームごとに絵が変わるので、シークが正しい位置に着地したかは
    :func:`decode_all_frames` で作った参照列との一致で確かめられる
    """
    path = directory / name
    if path.exists():
        return SampleMedia(path, width, height, fps, duration, audio, sample_rate)

    if not libx264_available():
        pytest.skip("ffmpeg に libx264 が無いので実素材のテストを飛ばす")

    directory.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    source = f"{pattern}=size={width}x{height}:rate={fps}:duration={duration}"
    command += ["-f", "lavfi", "-i", source]
    if audio:
        tone = f"sine=frequency={tone_hz}:duration={duration}:sample_rate={sample_rate}"
        if gain_db:
            tone += f",volume={gain_db}dB"
        command += ["-f", "lavfi", "-i", tone]

    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if keyframe_interval is not None:
        command += ["-g", str(keyframe_interval), "-keyint_min", str(keyframe_interval)]
    if audio:
        command += ["-c:a", "aac", "-ac", "2"]

    command.append(str(path))
    subprocess.run(command, check=True, capture_output=True)
    return SampleMedia(path, width, height, fps, duration, audio, sample_rate)


def make_rotated(directory: Path, name: str, source: Path, degrees: int) -> Path:
    """既存の素材に回転情報だけを付けた複製を作る 画素は触らない"""
    path = directory / name
    if path.exists():
        return path
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation",
            str(degrees),
            "-i",
            str(source),
            "-c",
            "copy",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def make_color_tagged(
    directory: Path, name: str, *, transfer: str, primaries: str, matrix: str = "bt2020nc"
) -> Path:
    """伝達特性・原色・行列の色の印を付けた短い動画 画素は SDR のまま

    印は ``setparams`` でフレームへ付け、符号化器にビットストリームと mp4 へ書かせる
    ffmpeg の出力の設定（``-color_trc`` など）は、ffmpeg 8 の libx264 では行列しか
    書かれなかった 中身まで HDR にしなくても、見分けるのは印だけなので試験には足りる
    """
    path = directory / name
    if path.exists():
        return path
    if not libx264_available():
        pytest.skip("ffmpeg に libx264 が無いので実素材のテストを飛ばす")
    directory.mkdir(parents=True, exist_ok=True)
    tags = f"setparams=color_primaries={primaries}:color_trc={transfer}:colorspace={matrix}"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=0.5",
            "-vf",
            tags,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def make_delayed(directory: Path, name: str, source: Path, seconds: float) -> Path:
    """先頭フレームの時刻を後ろへずらした複製を作る 画素は触らない

    タイムラインの途中から始まる素材（分割して書き出したもの等）はこの形
    時刻 0 に絵が無いので、0 だけを見て素材の良し悪しを決めると取り違える
    """
    path = directory / name
    if path.exists():
        return path
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-c",
            "copy",
            "-output_ts_offset",
            str(seconds),
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def make_silent_gap(
    directory: Path, name: str, *, duration: float = 6.0, sample_rate: int = 48000
) -> Path:
    """前半と後半に音があり、真ん中が無音の素材 ジェットカットの検証用"""
    path = directory / name
    if path.exists():
        return path
    third = duration / 3
    filter_complex = (
        f"sine=frequency=440:duration={third}:sample_rate={sample_rate}[a];"
        f"anullsrc=r={sample_rate}:cl=stereo:d={third}[b];"
        f"sine=frequency=880:duration={third}:sample_rate={sample_rate}[c];"
        f"[a][b][c]concat=n=3:v=0:a=1[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def decode_all_frames(path: Path) -> list[tuple[float, np.ndarray]]:
    """先頭から順に全フレームを復号し、``(表示時刻, RGBA 配列)`` の一覧を返す

    シークの正しさは「飛んだ結果が、順に読んだ結果と一致するか」でしか確かめられない
    その参照側を作る
    """
    import av

    frames: list[tuple[float, np.ndarray]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            pts, time_base = frame.pts, frame.time_base
            time = float(pts * time_base) if pts is not None and time_base else 0.0
            frames.append((time, frame.to_ndarray(format="rgba")))
    return frames
