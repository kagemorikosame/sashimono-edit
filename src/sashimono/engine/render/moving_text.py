"""位置だけが動くテキストの絵を使い回す（#295）

位置のキーフレームで動く字幕は、字の形も縁も同じなのに、位置が変わるたびに字を組み直して
塗っていた（1920x1080 の字幕 20 本が動き続けると、縁 1 つで 1 コマ 0.4 秒 層 3 つで 1.3 秒）
位置を 0 にした字を 1 枚描いて覚え、置く所だけを動かす

画素より細かい位置は 1/4 画素の格子へ丸め、端数（縦横 4 通りずつ）ごとに描いて覚える
描いた字を画素の間で補間してずらすと、端数の位置で輪郭が隣の画素へ分かれて字がにじむ
（並べて見ると縁がはっきり太く見えた） 丸めてずれるのは 1/8 画素まで 1 コマに 0.3 画素ずつ
動く字の重心の動きは、その都度描いていた今までの 0.26〜0.33 に対して 0.22〜0.53 で、
見て分かる揺れにはならない 1 本の字幕で描くのは、どれだけ長く動いても多くて 16 枚
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from sashimono.core.model import ClipId
from sashimono.engine.sources import Frame

__all__ = [
    "MAX_SPRITE_SCREENS",
    "SUBPIXEL_STEPS",
    "Spot",
    "Sprite",
    "SpriteShelf",
    "spot_of",
]

#: 1 画素をいくつに分けて置くか 4 なら 1/4 画素の格子 細かくするほど描く枚数が 2 乗で
#: 増え（8 なら 1 本で 64 枚）、粗くすると端数の位置で字が跳ぶ（1 なら 0.3 画素ずつ動く字が
#: 3 コマに 1 度だけ 1 画素跳ぶ）
SUBPIXEL_STEPS = 4

#: 覚えておく字の絵の量（画面何枚ぶん） 字の絵は字の周りだけの大きさで、1080p の字幕 1 本の
#: 1 枚は画面の 1/20 ほど 20 本がそれぞれ 16 枚の端数を持つと、ちょうどこのくらいになる
#: 溢れても古いクリップの絵から捨てて描き直すだけで、絵は変わらない
MAX_SPRITE_SCREENS = 16

#: 色の付いた範囲（画素、左・上・右・下 どれも整数）
_Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Spot:
    """字の絵を置く所"""

    #: 位置の端数（1/:data:`SUBPIXEL_STEPS` 画素の何個分か 横, 縦）
    cell: tuple[int, int]
    #: 字の絵の左上を置く所（置く先の絵の画素）
    offset: tuple[int, int]
    #: 字の絵を描くときの位置（画面の画素 Y は上が正） 端数の分だけずらして描く
    shift: tuple[float, float]


def spot_of(
    position: tuple[float, float],
    scale: tuple[float, float],
    canvas: tuple[int, int],
    sprite: tuple[int, int],
) -> Spot:
    """設定の位置 ``position``（画面の画素 Y は上が正）の字を置く所

    ``canvas`` は置く先の絵の大きさ ``sprite`` は字の絵の大きさ（偶奇は ``canvas`` と同じ
    :func:`~sashimono.engine.sources.text_sprite_canvas`） どちらも真ん中が位置 0 なので、
    整数の画素ぶんは置く所で、端数は描く位置で受け持つ
    """
    wholes: list[int] = []
    cells: list[int] = []
    for value, factor in zip(position, scale, strict=True):
        # 四捨五入を床で書く ``round`` は偶数へ丸めるので、0.125 と 0.375 の丸め方が食い違う
        steps = math.floor(value * factor * SUBPIXEL_STEPS + 0.5)
        wholes.append(steps // SUBPIXEL_STEPS)
        cells.append(steps % SUBPIXEL_STEPS)
    # 画面の Y は下が正 設定の Y は上が正なので、上へずらした字は上（小さい側）に置く
    offset = (
        (canvas[0] - sprite[0]) // 2 + wholes[0],
        (canvas[1] - sprite[1]) // 2 - wholes[1],
    )
    shift = (
        cells[0] / SUBPIXEL_STEPS / scale[0],
        cells[1] / SUBPIXEL_STEPS / scale[1],
    )
    return Spot((cells[0], cells[1]), offset, shift)


@dataclass(frozen=True, slots=True)
class Sprite:
    """位置の端数ぶんだけずらして描いた字の絵 1 枚"""

    image: np.ndarray
    #: 文字の枠（AviUtl2 の組み方のテキストだけ 字の絵の画素）
    framed: Frame | None
    #: 色の付いた範囲（字の絵の画素） 色が無ければ ``None``
    box: _Box | None

    def placed(
        self,
        offset: tuple[int, int],
        canvas: tuple[int, int],
        reuse: tuple[np.ndarray, tuple[int, int, int, int]] | None = None,
    ) -> tuple[np.ndarray, Frame | None, _Box | None, bool]:
        """``canvas`` の大きさの絵の ``offset`` へ置いた物

        返すのは（絵, 文字の枠, 色の付いた範囲, 範囲が確かか）

        ``reuse`` は前のコマに置いた絵と、そこへ字の絵を置いた所（左, 上, 幅, 高さ） 大きさが
        同じならその絵を消して置き直す 毎コマ新しく 0 で埋めた絵を作ると、GPU へ送るときに
        OS が画面 1 枚ぶんの頁を割り当て直し、送るのが 1 本で 1 ms ほど遅くなる（20 本で 1 コマ
        60 ms 置き直すと 25 ms） 前の絵は作ったコマで GPU へ送り終えているので、書き換えてよい

        範囲は字の絵が置く先に収まるときだけ確か はみ出して切れたときは、切れた後の
        絵の範囲を数え直さないと分からない（呼ぶ側が要るときに数える）
        """
        width, height = canvas
        left, top = offset
        sprite_height, sprite_width = int(self.image.shape[0]), int(self.image.shape[1])
        if reuse is not None and reuse[0].shape == (height, width, 4):
            image = reuse[0]
            old_left, old_top, old_width, old_height = reuse[1]
            image[
                max(0, old_top) : max(0, old_top + old_height),
                max(0, old_left) : max(0, old_left + old_width),
            ] = 0
        else:
            image = np.zeros((height, width, 4), dtype=np.uint8)
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(width, left + sprite_width), min(height, top + sprite_height)
        if x1 > x0 and y1 > y0:
            image[y0:y1, x0:x1] = self.image[y0 - top : y1 - top, x0 - left : x1 - left]
        framed = None
        if self.framed is not None:
            framed = (
                self.framed[0] + left,
                self.framed[1] + top,
                self.framed[2] + left,
                self.framed[3] + top,
            )
        if self.box is None:
            return image, framed, None, True
        box = (self.box[0] + left, self.box[1] + top, self.box[2] + left, self.box[3] + top)
        inside = box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
        return image, framed, box if inside else None, inside


class SpriteShelf:
    """クリップごとの字の絵（端数ごとに 1 枚）の置き場

    クリップごとに鍵（字の設定と絵の大きさ）を 1 つだけ持つ 鍵が変われば、そのクリップの
    前の絵はもう使わないので捨てる 量は画素の数で数え、古く使ったクリップから捨てる
    """

    def __init__(self) -> None:
        self._held: OrderedDict[ClipId, tuple[object, dict[tuple[int, int], Sprite]]] = (
            OrderedDict()
        )

    def get(self, clip_id: ClipId, key: object, cell: tuple[int, int]) -> Sprite | None:
        held = self._held.get(clip_id)
        if held is None or held[0] != key:
            return None
        self._held.move_to_end(clip_id)
        return held[1].get(cell)

    def put(
        self, clip_id: ClipId, key: object, cell: tuple[int, int], sprite: Sprite, budget: int
    ) -> None:
        """覚える ``budget`` は全部のクリップで持てるバイト数"""
        held = self._held.get(clip_id)
        if held is None or held[0] != key:
            held = (key, {})
            self._held[clip_id] = held
        held[1][cell] = sprite
        self._held.move_to_end(clip_id)
        used = sum(_bytes(sprites) for _, sprites in self._held.values())
        # 古く使ったクリップから捨てる 今のクリップは最後に回す（今描いている絵を捨てない）
        while used > budget and len(self._held) > 1:
            _, (_, dropped) = self._held.popitem(last=False)
            used -= _bytes(dropped)
        # 1 本だけで上限を超えるなら、そのクリップの古い端数から捨てる 今描いた 1 枚は残す
        sprites = held[1]
        for old in [found for found in sprites if found != cell]:
            if used <= budget:
                break
            used -= int(sprites.pop(old).image.nbytes)

    def clear(self) -> None:
        self._held.clear()

    def __len__(self) -> int:
        """覚えている絵の枚数"""
        return sum(len(sprites) for _, sprites in self._held.values())


def _bytes(sprites: dict[tuple[int, int], Sprite]) -> int:
    return sum(int(sprite.image.nbytes) for sprite in sprites.values())
