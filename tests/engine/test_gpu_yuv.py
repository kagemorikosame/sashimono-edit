"""GPU の上で動画用の色へ変換して読み戻す所（#372）

書き出しの色は CPU の変換（swscale）と同じでなければならない ここが壊れると、
GPU のある機械で書き出した動画だけ色がずれるか、コマが抜けたり入れ替わったりする
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from OpenGL import GL

from sashimono.engine.colorspace import bt709_planes
from sashimono.engine.gpu import Compositor, GLContextError, OffscreenGLContext, Placement
from sashimono.engine.gpu.glutil import Texture
from sashimono.engine.gpu.yuv import (
    COLOR_TOLERANCE,
    YuvReadback,
    gpu_convertible,
    open_readback,
    planes_shape,
)


@pytest.fixture
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    with context:
        yield context
    context.release()


def _opaque(image: np.ndarray) -> np.ndarray:
    image[..., 3] = 255
    return image


def _gradient(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 4), np.uint8)
    image[..., 0] = np.linspace(0, 255, width).astype(np.uint8)[None, :]
    image[..., 1] = np.linspace(0, 255, height).astype(np.uint8)[:, None]
    image[..., 2] = 255 - image[..., 0]
    return _opaque(image)


def _bars(width: int, height: int) -> np.ndarray:
    """黒・白と、彩度のいちばん高い 6 色 limited の端（16 と 235・240）がここで出る"""
    colors = [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    image = np.zeros((height, width, 4), np.uint8)
    for index, color in enumerate(colors):
        image[:, index * width // 8 : (index + 1) * width // 8, :3] = color
    return _opaque(image)


def _noise(width: int, height: int) -> np.ndarray:
    """隣の画素と関係の無い色 色差の間引き方の違いがいちばん大きく出る"""
    image = np.random.default_rng(372).integers(0, 256, (height, width, 4), dtype=np.uint8)
    return _opaque(image)


def _stripes(width: int, height: int) -> np.ndarray:
    """1 行おき・1 列おきの赤と緑 縦の間引きの重みと端の扱いが違うと、ここでずれる"""
    image = np.zeros((height, width, 4), np.uint8)
    image[0::2, :, 0] = 255
    image[1::2, :, 1] = 255
    image[:, 0::2, 2] = 200
    return _opaque(image)


PATTERNS = {"gradient": _gradient, "bars": _bars, "noise": _noise, "stripes": _stripes}


def _convert(image: np.ndarray, pixel_format: str) -> np.ndarray:
    """``image`` を GPU で変換して読み戻す GL コンテキストが current な前提"""
    height, width = image.shape[:2]
    texture = Texture.from_array(image, srgb=False)
    readback: YuvReadback[int] = YuvReadback(width, height, pixel_format)
    try:
        readback.push(texture.handle, 0)
        _tag, planes = readback.pop()
    finally:
        readback.release()
        texture.release()
    return planes


def _difference(left: np.ndarray, right: np.ndarray) -> int:
    assert left.shape == right.shape
    return int(np.abs(left.astype(np.int16) - right.astype(np.int16)).max())


class TestChoosing:
    """GPU で作れる物だけを GPU へ回す GL が無くても確かめられる"""

    @pytest.mark.parametrize(
        ("pixel_format", "width", "height", "expected"),
        [
            ("yuv420p", 1920, 1080, True),
            ("nv12", 3840, 2160, True),
            ("yuv420p", 640, 362, True),
            # 奇数の大きさは色差が半端になる CPU の道に残す
            ("yuv420p", 1921, 1080, False),
            ("yuv420p", 1920, 1081, False),
            # 10bit や 4:4:4 は並べ方が違う 作れない形式を GPU へ回すと絵が崩れる
            ("yuv444p", 1920, 1080, False),
            ("yuv420p10le", 1920, 1080, False),
            ("rgb24", 1920, 1080, False),
            ("yuv420p", 0, 0, False),
            # swscale は小さい絵で色差の間引き方を変える 同じ式では合わないので CPU に残す
            ("yuv420p", 64, 6, False),
            ("yuv420p", 16, 16, True),
        ],
    )
    def test_only_even_8bit_420_goes_to_the_gpu(
        self, pixel_format: str, width: int, height: int, expected: bool
    ) -> None:
        # 作れない形式まで GPU へ回すと、書き出した動画の色や形が崩れる
        assert gpu_convertible(pixel_format, width, height) is expected

    def test_an_unsupported_format_falls_back_without_touching_gl(self) -> None:
        # GL を触る前に断る 形式が合わないだけで駆動に頼むと、GPU の無い機械で落ちうる
        assert open_readback(1920, 1080, "yuv444p") is None
        assert open_readback(1921, 1080, "yuv420p") is None

    def test_the_planes_have_the_shape_pyav_takes(self) -> None:
        # 形が違うと PyAV が受け取らず、GPU のある機械でだけ書き出しが失敗する
        assert planes_shape(1920, 1080) == (1620, 1920)


class TestColor:
    """いまの CPU の変換（swscale）と、8bit の丸めの 1 まで同じ"""

    @pytest.mark.parametrize("pixel_format", ["yuv420p", "nv12"])
    @pytest.mark.parametrize("pattern", sorted(PATTERNS))
    @pytest.mark.parametrize(("width", "height"), [(320, 240), (322, 182), (16, 16)])
    def test_matches_the_cpu_conversion(
        self,
        gl: OffscreenGLContext,
        pixel_format: str,
        pattern: str,
        width: int,
        height: int,
    ) -> None:
        # 外れると、GPU のある機械で書き出した動画だけ色がずれる（黒が浮く・赤がくすむなど）
        # 182 は高さが 4 の倍数でない U の面が行の途中で終わり、同じ行から V が始まる
        image = PATTERNS[pattern](width, height)
        planes = _convert(image, pixel_format)
        assert _difference(planes, bt709_planes(image, pixel_format)) <= COLOR_TOLERANCE

    def test_the_limited_range_ends_are_exact(self, gl: OffscreenGLContext) -> None:
        # 黒は 16、白は 235、灰の色差は 128 ちょうど ずれると黒が浮くか白が飛ぶ
        image = _bars(64, 16)
        planes = _convert(image, "yuv420p")
        assert int(planes[0, 0]) == 16
        assert int(planes[0, 8]) == 235
        u_start = 16
        assert int(planes[u_start, 0]) == 128
        assert int(planes[u_start, 4]) == 128

    def test_alpha_is_ignored_like_the_cpu(self, gl: OffscreenGLContext) -> None:
        # swscale は RGBA の A を捨てる GPU だけ A を掛けると、半透明の所の色が変わる
        image = _gradient(64, 32)
        image[..., 3] = np.arange(64, dtype=np.uint8)[None, :] * 4
        planes = _convert(image, "yuv420p")
        assert _difference(planes, bt709_planes(image, "yuv420p")) <= COLOR_TOLERANCE

    @pytest.mark.parametrize("encoded", [True, False])
    def test_the_composited_picture_matches_what_is_read_back(
        self, gl: OffscreenGLContext, encoded: bool
    ) -> None:
        # 書き出しは合成した絵を GPU の上で変換する 今の道（read してから CPU で変換）と
        # 入る値が食い違うと、半透明に重ねた所だけ色が変わる
        width, height = 96, 64
        compositor = Compositor(width, height, encoded=encoded)
        texture = Texture.from_array(_noise(32, 32))
        try:
            compositor.begin((0.0, 0.0, 0.0, 0.0))
            compositor.draw(texture, placement=Placement(10, 6, 50, 40), opacity=0.6)
            compositor.underlay((0.0, 0.0, 0.0, 1.0))
            image = compositor.read()
            readback: YuvReadback[int] = YuvReadback(width, height, "yuv420p")
            try:
                readback.push(compositor.resolved_texture(), 0)
                _tag, planes = readback.pop()
            finally:
                readback.release()
        finally:
            texture.release()
            compositor.release()
        assert _difference(planes, bt709_planes(image, "yuv420p")) <= COLOR_TOLERANCE


class TestReadback:
    """非同期の読み戻しは、入れた順に出て、片付けで資源を残さない"""

    def test_frames_come_out_in_the_order_they_went_in(self, gl: OffscreenGLContext) -> None:
        # 入れ替わると、書き出した動画のコマの並びが崩れる
        width, height = 32, 16
        textures = [
            Texture.from_array(_opaque(np.full((height, width, 4), value, np.uint8)), srgb=False)
            for value in (0, 60, 120, 180, 240)
        ]
        readback: YuvReadback[int] = YuvReadback(width, height, "yuv420p")
        seen: list[tuple[int, int]] = []
        try:
            for index, texture in enumerate(textures):
                readback.push(texture.handle, index)
                if readback.full:
                    tag, planes = readback.pop()
                    seen.append((tag, int(planes[0, 0])))
            while readback.pending:
                tag, planes = readback.pop()
                seen.append((tag, int(planes[0, 0])))
        finally:
            readback.release()
            for texture in textures:
                texture.release()
        assert [tag for tag, _luma in seen] == [0, 1, 2, 3, 4]
        lumas = [luma for _tag, luma in seen]
        assert lumas == sorted(lumas)
        assert lumas[0] == 16

    def test_pushing_into_a_full_ring_is_refused(self, gl: OffscreenGLContext) -> None:
        # 黙って上書きすると、運び終わる前のコマが消えて動画からコマが抜ける
        texture = Texture.from_array(_gradient(16, 16), srgb=False)
        readback: YuvReadback[int] = YuvReadback(16, 16, "nv12", buffers=2)
        try:
            readback.push(texture.handle, 0)
            readback.push(texture.handle, 1)
            assert readback.full
            with pytest.raises(RuntimeError, match="空いた PBO"):
                readback.push(texture.handle, 2)
            assert readback.pop()[0] == 0
            readback.push(texture.handle, 2)
        finally:
            readback.release()
            texture.release()

    def test_release_while_frames_are_in_flight_frees_the_buffers(
        self, gl: OffscreenGLContext
    ) -> None:
        # 中止や失敗で受け取らないまま抜けたときも返す 残すと中止のたびに GPU のメモリが減る
        texture = Texture.from_array(_gradient(16, 16), srgb=False)
        readback: YuvReadback[int] = YuvReadback(16, 16, "yuv420p")
        handles = list(readback._buffers)
        readback.push(texture.handle, 0)
        readback.release()
        readback.release()
        texture.release()
        assert readback.released
        assert readback.pending == 0
        assert not any(GL.glIsBuffer(handle) for handle in handles)

    def test_the_pack_buffer_is_unbound_after_use(self, gl: OffscreenGLContext) -> None:
        # 結んだまま戻ると、ほかの所の glReadPixels（合成の範囲の帯や read）が配列ではなく
        # この PBO へ書き、プレビューや書き出しの絵が空のまま返る
        texture = Texture.from_array(_gradient(16, 16), srgb=False)
        readback: YuvReadback[int] = YuvReadback(16, 16, "yuv420p")
        try:
            readback.push(texture.handle, 0)
            assert int(GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING)) == 0
            readback.pop()
            assert int(GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING)) == 0
        finally:
            readback.release()
            texture.release()

    def test_popping_with_nothing_in_flight_is_refused(self, gl: OffscreenGLContext) -> None:
        # 空の所から読むと、前のコマの古い絵がもう 1 度書かれる
        readback: YuvReadback[int] = YuvReadback(16, 16, "yuv420p")
        try:
            with pytest.raises(RuntimeError, match="受け取るコマが無い"):
                readback.pop()
        finally:
            readback.release()
