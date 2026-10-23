"""合成の結果を GPU の上で動画用の色（BT.709 limited の YUV 4:2:0）へ変換して読み戻す（#372）

書き出しの 1 コマの 3 割は CPU の色変換（swscale）で、読み戻しも RGBA の 4 byte/画素を
運んでいた ここで変換してから読み戻すと、色変換が CPU から消え、運ぶ量も 1.5 byte/画素
（4 割）に減る 1080p・素材 3 枚の NVENC で 16.4 ms が 10.0 ms になった（#372 の試作）

色の決まりは CPU の道（:func:`~sashimono.engine.colorspace.to_bt709`）と同じにする

- 行列は BT.709、範囲は limited（Y 16..235、色差 16..240）
- 入れる値は CPU の道と同じ RGBA8（:meth:`Compositor.resolved_texture`） 8bit に丸めた
  後の値から変換するので、丸めの前の値から変換したときのずれが出ない
- 色差の間引きも swscale に合わせる 横は隣の 2 画素の平均、縦は 1:3:3:1 の重みで 4 行
  （端は端の行を繰り返す） 4 画素の単純な平均にすると、細かい模様で色差が 30 ずれた
- 差は 8bit の丸めの 1 まで（:data:`COLOR_TOLERANCE`） 試験で押さえる

読み戻しは PBO（読み取り用のバッファ）を何枚か回して非同期にする 次のコマを合成している
間に、前のコマが GPU から運ばれる 取り出す順は入れた順のまま（先に入れた物から出る）
"""

from __future__ import annotations

import ctypes
from collections import deque

import numpy as np
from OpenGL import GL
from OpenGL.error import Error as GLLibraryError

from sashimono.engine.gpu.glutil import (
    FULL_RECT,
    VERTEX_SHADER,
    Framebuffer,
    Program,
    ScreenQuad,
    ShaderError,
)

__all__ = [
    "COLOR_TOLERANCE",
    "GPU_PIXEL_FORMATS",
    "MIN_SIZE",
    "READBACK_BUFFERS",
    "YuvReadback",
    "gpu_convertible",
    "open_readback",
    "planes_shape",
]

#: GPU で作れる画素形式 どちらも 8bit の 4:2:0 で、1 枚の R8 のバッファに並べられる
#: 10bit や 4:4:4 は CPU の道に残す 使う人が少なく、別の並べ方と別の試験が要る
GPU_PIXEL_FORMATS = frozenset({"yuv420p", "nv12"})

#: いまの CPU の変換（swscale）との画素の差の上限 8bit の丸め 1 つぶん
#: swscale は整数で、こちらは浮動小数で計算するので、丸めの境目で 1 だけずれる
COLOR_TOLERANCE = 1

#: 読み戻しに回す PBO の枚数 2 枚で、1 枚を運んでいる間にもう 1 枚へ次のコマを入れられる
#: 増やしても、合成より速く運べる間は速くならず、GPU のメモリを食うだけ
READBACK_BUFFERS = 2

#: GPU で変換する絵の縦横の下限（画素） これより小さい絵は CPU で変換する
#: （:func:`gpu_convertible`） 16x16 で swscale と 1 まで揃うことを試験で見ている
MIN_SIZE = 16

_LAYOUTS = {"yuv420p": 0, "nv12": 1}

_FRAGMENT_SHADER = """
#version 430 core
out vec4 frag_color;
// 符号化済みの sRGB の RGBA8（上下を返して、行 0 が絵の上の端）
uniform sampler2D u_texture;
uniform ivec2 u_size;
// 0 なら yuv420p（Y・U・V の 3 面） 1 なら nv12（Y と、U と V を交互に並べた 1 面）
uniform int u_layout;

const vec3 KY = vec3(0.2126, 0.7152, 0.0722);

vec3 rgb_at(int x, int y) {
    return texelFetch(u_texture, ivec2(x, clamp(y, 0, u_size.y - 1)), 0).rgb;
}

float luma(vec3 c) {
    return 16.0 + 219.0 * dot(KY, c);
}

// 色差 (Cb, Cr) BT.709 の定義のまま 範囲は limited の 16..240
vec2 chroma_of(vec3 c) {
    float y = dot(KY, c);
    return vec2(128.0 + 224.0 * (c.b - y) / 1.8556, 128.0 + 224.0 * (c.r - y) / 1.5748);
}

// 色差の 1 画素 swscale と同じ間引き方にする 横は 2 画素の平均、縦は 1:3:3:1 で 4 行
// 4 画素の単純な平均では、細かい模様で今の書き出しと色差が 30 ずれた
vec2 chroma_at(int cx, int cy) {
    int x = cx * 2;
    int y = cy * 2;
    const float weights[4] = float[4](1.0, 3.0, 3.0, 1.0);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        int row = y - 1 + i;
        sum += weights[i] * (rgb_at(x, row) + rgb_at(x + 1, row));
    }
    return chroma_of(sum / 16.0);
}

void main() {
    int width = u_size.x;
    int height = u_size.y;
    ivec2 here = ivec2(gl_FragCoord.xy);
    float value;
    if (here.y < height) {
        value = luma(rgb_at(here.x, here.y));
    } else if (u_layout == 1) {
        vec2 uv = chroma_at(here.x / 2, here.y - height);
        value = (here.x % 2 == 0) ? uv.x : uv.y;
    } else {
        // U の面と V の面は、幅 width の行をまたいで続けて並ぶ 高さが 4 の倍数でないと
        // U の面が行の途中で終わり、同じ行の残りから V が始まる
        int half_width = width / 2;
        int plane_size = half_width * (height / 2);
        int index = (here.y - height) * width + here.x;
        int plane = index / plane_size;
        int at = index - plane * plane_size;
        vec2 uv = chroma_at(at % half_width, at / half_width);
        value = plane == 0 ? uv.x : uv.y;
    }
    // 丸めてから書く 書き込みの丸め方は実装に任されていて、境目で 1 ずれる機械がある
    frag_color = vec4(floor(value + 0.5) / 255.0, 0.0, 0.0, 1.0);
}
"""


def gpu_convertible(pixel_format: str, width: int, height: int) -> bool:
    """``pixel_format`` で ``width`` x ``height`` の絵を GPU で作れるか

    4:2:0 は縦横とも 2 画素で色差を 1 つ持つので、奇数の大きさは CPU の道に残す
    :data:`MIN_SIZE` より小さい絵も CPU に残す swscale は高さが 8 に満たない絵で
    色差の縦の間引き方を変え、同じ式では 64 までずれた そこまで小さい絵は CPU でも軽い
    """
    return (
        pixel_format in GPU_PIXEL_FORMATS
        and width >= MIN_SIZE
        and height >= MIN_SIZE
        and not (width | height) & 1
    )


def planes_shape(width: int, height: int) -> tuple[int, int]:
    """読み戻した配列の形 ``(高さ x 1.5, 幅)`` の uint8

    PyAV の ``VideoFrame.from_ndarray`` が yuv420p と nv12 で受ける形と同じ
    """
    return height * 3 // 2, width


class YuvReadback[Tag]:
    """RGBA8 のテクスチャを YUV へ変換し、PBO で非同期に読み戻す

    :meth:`push` で変換して運び始め、:meth:`pop` で入れた順に受け取る 運び終わる前に
    受け取ろうとすると、そこで待つ ``Tag`` は入れた物に付けておく印（書き出しではコマの
    番号） 受け取るときに一緒に返すので、取り違えても番号でわかる

    GL コンテキストが current な状態で使う 生成・使用・解放をすべて同じコンテキストで
    行うこと 片付けは :meth:`release` 何度呼んでもよく、運んでいる途中でも捨てられる
    """

    def __init__(
        self, width: int, height: int, pixel_format: str, *, buffers: int = READBACK_BUFFERS
    ) -> None:
        if not gpu_convertible(pixel_format, width, height):
            raise ValueError(f"GPU で作れない形式か大きさ: {pixel_format} {width}x{height}")
        if buffers < 1:
            raise ValueError(f"PBO は 1 枚以上: {buffers}")
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self._rows, _ = planes_shape(width, height)
        self._size = self._rows * width
        self._buffers: list[int] = []
        self._free: deque[int] = deque()
        self._pending: deque[tuple[int, Tag]] = deque()
        self._program: Program | None = None
        self._target: Framebuffer | None = None
        self._quad: ScreenQuad | None = None
        try:
            self._program = Program(VERTEX_SHADER, _FRAGMENT_SHADER)
            self._target = Framebuffer(width, self._rows, internal_format=GL.GL_R8)
            self._quad = ScreenQuad()
            for _ in range(buffers):
                handle = int(GL.glGenBuffers(1))
                self._buffers.append(handle)
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, handle)
                GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, self._size, None, GL.GL_STREAM_READ)
                self._free.append(handle)
        except BaseException:
            # 途中まで取った資源を返してから投げる 返さないと、作れなかった書き出しの
            # たびに GPU のメモリが 1 枚ぶんずつ減っていく
            self.release()
            raise
        finally:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

    @property
    def pending(self) -> int:
        """運び始めて、まだ受け取っていない枚数"""
        return len(self._pending)

    @property
    def full(self) -> bool:
        """空いた PBO が無い 次の :meth:`push` の前に :meth:`pop` で受け取る"""
        return not self._free

    def push(self, texture: int, tag: Tag) -> None:
        """``texture``（符号化済みの RGBA8、行 0 が絵の上）を変換して運び始める

        いっぱいのときに呼ぶと ``RuntimeError`` 受け取らずに上書きすると、運び終わる
        前のコマが黙って消え、書き出した動画からコマが抜ける
        """
        if self._program is None or self._target is None or self._quad is None:
            raise RuntimeError("解放した読み戻しは使えない")
        if not self._free:
            raise RuntimeError("空いた PBO が無い 先に pop で受け取る")
        self._target.bind()
        GL.glDisable(GL.GL_BLEND)
        program = self._program
        program.use()
        program.set_vec4("u_rect", FULL_RECT)
        program.set_bool("u_flip", False)
        program.set_ivec2("u_size", (self.width, self.height))
        program.set_int("u_layout", _LAYOUTS[self.pixel_format])
        program.bind_texture("u_texture", texture)
        self._quad.draw()

        handle = self._free.popleft()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, handle)
        try:
            # PBO を結んだまま読むと、受け皿は CPU の配列ではなくバッファの中の位置になる
            # ここでは待たずに戻り、運ぶのは GPU が裏で進める
            # None を渡すと PyOpenGL が配列を作って同期の読み取りになるので、0 の位置を渡す
            GL.glReadPixels(
                0, 0, self.width, self._rows, GL.GL_RED, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0)
            )
        except BaseException:
            self._free.appendleft(handle)
            raise
        finally:
            # 結んだまま戻ると、ほかの所の glReadPixels（合成の範囲の帯など）が CPU の配列では
            # なくこの PBO へ書き、配列は空のまま返る
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._pending.append((handle, tag))

    def pop(self) -> tuple[Tag, np.ndarray]:
        """いちばん先に入れたコマを受け取る 運び終わっていなければ待つ

        返す配列は :func:`planes_shape` の形の写し PBO は次のコマに使い回すので、
        中を指したままの配列を渡すと、書き込みのスレッドが読む前に次のコマで上書きされる
        """
        if not self._pending:
            raise RuntimeError("受け取るコマが無い")
        handle, tag = self._pending.popleft()
        image = np.empty(planes_shape(self.width, self.height), dtype=np.uint8)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, handle)
        try:
            pointer = GL.glMapBufferRange(
                GL.GL_PIXEL_PACK_BUFFER, 0, self._size, GL.GL_MAP_READ_BIT
            )
            if not pointer:
                raise ShaderError("読み戻しのバッファを開けない")
            try:
                ctypes.memmove(image.ctypes.data, pointer, self._size)
            finally:
                GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
        finally:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            # 読めなかったときも PBO は返す 返さないと、次の push が空きを待てずに断られる
            self._free.append(handle)
        return tag, image

    def release(self) -> None:
        """GL の資源を返す 運んでいる途中のコマは捨てる 何度呼んでもよい

        書き出しの中止や失敗では、受け取らないまま抜ける そのときもここで PBO を返す
        返さないと、中止するたびに 1 枚 3MB（4K なら 12MB）ずつ GPU のメモリが残る
        """
        self._pending.clear()
        self._free.clear()
        if self._buffers:
            GL.glDeleteBuffers(len(self._buffers), self._buffers)
            self._buffers = []
        if self._target is not None:
            self._target.release()
            self._target = None
        if self._quad is not None:
            self._quad.release()
            self._quad = None
        if self._program is not None:
            self._program.release()
            self._program = None

    @property
    def released(self) -> bool:
        """:meth:`release` で資源を返した後か"""
        return self._program is None and not self._buffers


def open_readback[Tag](
    width: int, height: int, pixel_format: str, *, buffers: int = READBACK_BUFFERS
) -> YuvReadback[Tag] | None:
    """GPU の変換を用意する 使えなければ ``None``（呼ぶ側は CPU の道で続ける）

    使えないのは、形式や大きさが合わないときと、駆動がシェーダや PBO を断ったとき
    断られたことを理由に書き出しごと失敗させると、今まで書き出せていた機械で書けなくなる
    """
    if not gpu_convertible(pixel_format, width, height):
        return None
    try:
        return YuvReadback(width, height, pixel_format, buffers=buffers)
    except (ShaderError, GLLibraryError):
        return None
