"""映像が 2 本ある素材のサムネイルを流れごとに作る（#336）

色を自動で整える道具はサムネイルで色を測る 素材ごとに 1 本しか持たないと、2 本目の映像を
映すクリップの色を 1 本目のカメラの絵で決めてしまう 実際に 2 本入ったファイルを作って確かめる
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np
import pytest

from sashimono.engine.cache import CacheStore, MediaAnalyzer
from sashimono.engine.cache.analyzer import picture_stream
from sashimono.engine.decode import probe_media


def _two_cameras(path: Path) -> Path:
    """赤い映像と青い映像を 1 本ずつ持つ 2 秒の mkv"""
    with av.open(str(path), "w") as container:
        streams = []
        for _ in range(2):
            stream = container.add_stream("ffv1", rate=10)
            stream.width = 32
            stream.height = 24
            stream.pix_fmt = "yuv420p"
            streams.append(stream)
        for index in range(20):
            for stream, colour in zip(streams, ((220, 20, 20), (20, 20, 220)), strict=True):
                image = np.zeros((24, 32, 3), dtype=np.uint8)
                image[:] = colour
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
        for stream in streams:
            for packet in stream.encode():
                container.mux(packet)
    return path


def _until(predicate: Callable[[], bool], timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_each_picture_stream_gets_its_own_thumbnails(tmp_path: Path) -> None:
    # 壊れると、2 本目の映像を映すクリップの色を 1 本目のカメラの絵で測り、青い絵を
    # 赤かぶりと見て直す
    media = probe_media(_two_cameras(tmp_path / "二台.mkv"))
    if len(media.video_streams) < 2:
        pytest.skip("映像を 2 本持つファイルを作れなかった")
    first, second = (stream.index for stream in media.video_streams[:2])
    analyzer = MediaAnalyzer(CacheStore(tmp_path / "cache"))
    try:
        analyzer.request(media)
        assert _until(
            lambda: (
                analyzer.filmstrip(media, first) is not None
                and analyzer.filmstrip(media, second) is not None
            )
        )
        red = analyzer.filmstrip(media, first)
        blue = analyzer.filmstrip(media, second)
        assert red is not None and blue is not None
        red_tile, blue_tile = red.tile(0), blue.tile(0)
        assert red_tile is not None and blue_tile is not None
        assert red_tile[..., 0].mean() > red_tile[..., 2].mean() + 100
        assert blue_tile[..., 2].mean() > blue_tile[..., 0].mean() + 100
        # 流れを省いたときは 1 本目（タイムラインの前からの引き方と同じ）
        assert analyzer.filmstrip(media) is red
    finally:
        analyzer.close()


def test_an_unknown_stream_falls_back_to_the_first(tmp_path: Path) -> None:
    # 壊れると、素材に無い番号を持つクリップ（差し替えた素材など）のサムネイルがいつまでも
    # 用意されず、色を測る道具が待たせ続ける デコーダと同じく 1 本目へ寄せる
    media = probe_media(_two_cameras(tmp_path / "二台.mkv"))
    assert picture_stream(media, None) is None
    assert picture_stream(media, media.video_streams[0].index) is None
    assert picture_stream(media, 99) is None
