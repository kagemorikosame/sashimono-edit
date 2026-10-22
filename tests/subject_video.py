"""被写体の位置を確かめるための動画を、その場で作る（#334）

ほかの人が作った素材はリポジトリに入れない 被写体の位置が分かっている絵を PyAV で焼く

| 秒 | 中身 |
|---|---|
| 0〜2 | 暗い青の地の右（横の 3/4）に、肌色の丸（顔）と、点滅する口 |
| 2〜4 | 同じ顔が左（横の 1/4）へ移る |
| 4〜6 | 肌色の無い灰色の地を、白い四角が左から右へ動く |
| 6〜8 | 何も動かない灰色の地（手がかりが無い） |
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

__all__ = ["FPS", "HEIGHT", "WIDTH", "write_subject_video"]

WIDTH, HEIGHT, FPS = 320, 180, 30
#: 顔の色（RGB） YCbCr にすると Cb 105 前後・Cr 150 前後で、肌色の範囲の真ん中に入る
SKIN = (224, 172, 140)


def _face(center: int, frame: int) -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = (30, 40, 70)
    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    image[(rows - HEIGHT // 2) ** 2 + (columns - center) ** 2 < 28**2] = SKIN
    if frame % 4 < 2:
        image[(rows - HEIGHT // 2 - 12) ** 2 + (columns - center) ** 2 < 6**2] = (90, 30, 30)
    return image


def _square(frame: int) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    left = 20 + frame * 4
    image[70:110, left : left + 40] = 240
    return image


def write_subject_video(path: Path) -> Path:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for index in range(FPS * 8):
            second = index / FPS
            if second < 2:
                image = _face(WIDTH * 3 // 4, index)
            elif second < 4:
                image = _face(WIDTH // 4, index)
            elif second < 6:
                image = _square(index - FPS * 4)
            else:
                image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
