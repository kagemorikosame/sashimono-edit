"""被写体の位置を確かめるための動画を、その場で作る（#334）

ほかの人が作った素材はリポジトリに入れない 被写体の位置が分かっている絵を PyAV で焼く

| 秒 | 中身 |
|---|---|
| 0〜2 | 暗い青の地の右（横の 3/4）に、肌色の丸（顔）と、点滅する口 |
| 2〜4 | 同じ顔が左（横の 1/4）へ移る |
| 4〜6 | 肌色の無い灰色の地を、白い四角が左から右へ動く |
| 6〜8 | 何も動かない灰色の地（手がかりが無い） |

:func:`write_drawing_video` は別の動画（#407） 左に肌色の大きな絵とその上を動くペン先、右に
目の印の付いた小さな顔
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

__all__ = [
    "DRAWING",
    "EYE",
    "FACE_CENTER",
    "FACE_RADIUS",
    "FPS",
    "HEIGHT",
    "WIDTH",
    "write_drawing_video",
    "write_subject_video",
    "write_turned_video",
]

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


#: 偽の顔の部品が顔の印と見る目の色（RGB） 肌色の範囲にも、絵の地の色にも入らない
EYE = (20, 40, 230)
#: 肌色の大きな絵の左右（画素） 幅 100 は 1920x1080 から 1080x1920 を切る枠（横の 0.316）を
#: この大きさに縮めた幅とほぼ同じ 絵の全部が 1 つの枠に収まる、顔にとっていちばん不利な形
DRAWING = (20, 120)
#: 小さな顔の中心（画素）と半径
FACE_CENTER, FACE_RADIUS = (270, 90), 14


def _drawing(frame: int) -> np.ndarray:
    """肌色の大きな絵を描く画面と、右の小さな顔（#407）

    絵の上をペン先（暗い四角）がジグザグに動き続け、顔は口だけが小さく動く 動きと肌色だけで
    見積もると、絵の方が濃くて枠が絵に決まる（実素材の、肌色に近い絵を描く配信と同じ形）
    """
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = (30, 40, 70)
    image[30:150, DRAWING[0] : DRAWING[1]] = SKIN
    step = frame * 5
    pen_x = DRAWING[0] + 6 + (step % 160 if step % 320 < 160 else 160 - step % 160) // 2
    pen_y = 40 + (frame * 3) % 100
    image[pen_y : pen_y + 8, pen_x : pen_x + 8] = (60, 30, 20)
    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    x, y = FACE_CENTER
    image[(rows - y) ** 2 + (columns - x) ** 2 < FACE_RADIUS**2] = SKIN
    image[y - 6 : y - 2, x - 7 : x - 3] = EYE
    image[y - 6 : y - 2, x + 3 : x + 7] = EYE
    if frame % 4 < 2:
        image[y + 4 : y + 8, x - 3 : x + 3] = (90, 30, 30)
    return image


def write_drawing_video(path: Path, seconds: int = 4) -> Path:
    """肌色の大きな絵と小さな顔の動画（:func:`_drawing`）を焼く 場面の切り替わりは無い"""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for index in range(FPS * seconds):
            frame = av.VideoFrame.from_ndarray(_drawing(index), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def write_turned_video(path: Path, rotation: int, seconds: int = 2) -> Path:
    """回して見る素材（スマホの縦撮りの形 #407）の、回す前の絵だけを焼く

    回した後（時計回りに ``rotation`` 度）に、暗い青の地の右上（横の 3/4・上から 1/3）に目の
    印の付いた顔が見える絵を、逆に回して書く 回転の情報は ffmpeg で付ける
    （tests/media_fixtures.py の ``make_rotated``） 回す前と後で縦横が入れ替わる
    90・270 度は、回す前が縦長
    """
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        turned = rotation in (90, 270)
        stream.width = HEIGHT if turned else WIDTH
        stream.height = WIDTH if turned else HEIGHT
        stream.pix_fmt = "yuv420p"
        for index in range(FPS * seconds):
            # 顔は縦の真ん中から外す（上の 1/3） 真ん中だと、横と縦を取り違えても割合が
            # たまたま近くなり、取り違えを見落とす
            x, y = WIDTH * 3 // 4, HEIGHT // 3
            upright = np.roll(_face(x, index), y - HEIGHT // 2, axis=0)
            upright[y - 10 : y - 5, x - 12 : x - 6] = EYE
            upright[y - 10 : y - 5, x + 6 : x + 12] = EYE
            # 回した後が upright になる絵は、upright を反時計回りに同じだけ回した物
            raw = np.ascontiguousarray(np.rot90(upright, k=rotation // 90))
            frame = av.VideoFrame.from_ndarray(raw, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


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
