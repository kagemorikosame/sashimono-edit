"""場面の切り替わりを確かめるための動画を、その場で作る（#319）

ほかの人が作った素材はリポジトリに入れない 切り替わりの位置が分かっている絵を
PyAV で焼く 実写で間違えやすい所をそれぞれ 1 か所ずつ入れてある

| 秒 | 中身 | 切り替わりか |
|---|---|---|
| 0〜2 | 粗い模様の速いパン（1 コマで 12 画素） | 違う（ずっと差が大きいだけ） |
| 2〜4 | 暗い青の地に動く白い球 | 2.0 で切り替わる |
| 4〜6 | 同じくらいの明るさの暗い赤の地に動く白い球 | 4.0 で切り替わる（画素の差は小さい） |
| 6〜7 | 左が暗く右が明るい濃淡 | 6.0 で切り替わる |
| 7〜8 | 左右を返した濃淡 | 7.0 で切り替わる（ヒストグラムは同じ） |
| 8〜10 | 暗い灰色がゆっくり明るくなる | 8.0 で切り替わる その後は違う（※） |
| 10〜12 | 別の模様のゆっくりしたパン | 10.0 で切り替わる |

※ ヒストグラムの段は跳ねるが、画素はほとんど動かない
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

__all__ = ["CUTS", "FPS", "HEIGHT", "WIDTH", "write_scene_video"]

WIDTH, HEIGHT, FPS = 320, 180, 30
#: 焼いた動画の切り替わり（素材の秒） 新しい場面の最初のフレームの時刻
CUTS = (2.0, 4.0, 6.0, 7.0, 8.0, 10.0)


def _texture(seed: int, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, (height // 12 + 2, width // 12 + 2, 3), dtype=np.uint8)
    return np.kron(blocks, np.ones((12, 12, 1), dtype=np.uint8))


def _pan(seed: int, frame: int, speed: int) -> np.ndarray:
    wide = _texture(seed, WIDTH * 3, HEIGHT)
    left = (frame * speed) % (WIDTH * 2)
    return np.ascontiguousarray(wide[:HEIGHT, left : left + WIDTH])


def _ball(color: tuple[int, int, int], frame: int) -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = color
    center = 30 + (frame * 5) % (WIDTH - 60)
    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    image[(rows - HEIGHT // 2) ** 2 + (columns - center) ** 2 < 24**2] = 255
    return image


def _gradient(flip: bool) -> np.ndarray:
    row = np.linspace(0, 255, WIDTH).astype(np.uint8)
    if flip:
        row = row[::-1]
    return np.ascontiguousarray(np.broadcast_to(row[None, :, None], (HEIGHT, WIDTH, 3)))


def _flat(level: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), level, np.uint8)


def frames() -> list[np.ndarray]:
    """上の表のとおりの RGB の絵 30fps で 12 秒"""
    second = FPS
    plan: list[np.ndarray] = []
    plan += [_pan(1, index, 12) for index in range(2 * second)]
    plan += [_ball((40, 60, 140), index) for index in range(2 * second)]
    plan += [_ball((140, 60, 40), index) for index in range(2 * second)]
    plan += [_gradient(False)] * second + [_gradient(True)] * second
    plan += [_flat(12 + index * 30 // (2 * second)) for index in range(2 * second)]
    plan += [_pan(7, index, 2) for index in range(2 * second)]
    return plan


def write_scene_video(path: Path) -> Path:
    """:func:`frames` を H.264（無ければ MPEG-4）で焼く 鍵フレームは頭だけ

    鍵フレームを細かく入れると、切り替わりの所で鍵フレームの画質の跳ねまで差に乗る
    配られる動画のように鍵フレームの間を長く取る
    """
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=FPS)
        # 名前を変数で渡すと、型の上では音や字幕の流れかもしれないので確かめて絞る
        if not isinstance(stream, av.VideoStream):
            raise TypeError(f"{codec} が映像の流れを作らなかった")
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"g": "600"} if codec == "libx264" else {"g": "600", "qscale": "3"}
        for image in frames():
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
