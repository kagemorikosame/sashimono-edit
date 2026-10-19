"""YUV と RGB の間の色の行列と色のタグ（#61）

色の範囲は SDR の sRGB / Rec.709 書き出しは BT.709 / limited で変換してタグを付け、
読み込みはタグに従い、タグが無ければ大きさから当てる ここが崩れると、書き出した
動画の赤が明るく緑が暗くなったり、タグの無い HD の素材がくすんで見えたりする
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from av.video.reformatter import (
    ColorPrimaries,
    ColorRange,
    Colorspace,
    ColorTrc,
    VideoReformatter,
)

from sashimono.core.model import MediaItem
from sashimono.engine.cache.proxy import ProxyStore, create_proxy, proxy_codecs
from sashimono.engine.cache.store import CacheStore, media_key
from sashimono.engine.colorspace import source_matrix, tag_bt709, to_bt709, to_rgb_array
from sashimono.engine.decode import VideoDecoder
from sashimono.engine.decode.image import read_image
from tests.color_bars import (
    AVCOL_SPC_BT470BG,
    AVCOL_SPC_BT709,
    AVCOL_SPC_UNSPECIFIED,
    BT709_YUV,
    COLORS,
    assert_close,
    bars,
    rgb_at_bars,
    skip_without_libx264,
    write_bars,
    yuv_at_bars,
)


class TestWriting:
    def test_rgb_becomes_bt709_limited(self) -> None:
        # 直す前は swscale の既定の BT.601 で、赤の Y が 63 ではなく 81 になっていた
        rgb = av.VideoFrame.from_ndarray(bars(320, 240), format="rgb24")
        converted = to_bt709(rgb, "yuv420p")
        assert_close(yuv_at_bars(converted), BT709_YUV, tolerance=1)

    def test_the_frame_carries_the_tags(self) -> None:
        # フレームのタグが無いと、エンコーダによっては行列を書かず、再生側の推測で色が変わる
        rgb = av.VideoFrame.from_ndarray(bars(320, 240), format="rgb24")
        converted = to_bt709(rgb, "yuv420p")
        assert converted.colorspace == AVCOL_SPC_BT709
        assert converted.color_range == ColorRange.MPEG
        assert converted.color_primaries == ColorPrimaries.BT709
        assert converted.color_trc == ColorTrc.BT709

    def test_a_reused_reformatter_gives_the_same_pixels(self) -> None:
        # 書き出しは swscale の表を使い回して 1 枚 10ms を削っている（#56）
        # 使い回した表が別の設定で作られていると、色だけが静かにずれる
        reformatter = VideoReformatter()
        image = bars(320, 240)
        fresh = to_bt709(av.VideoFrame.from_ndarray(image, format="rgb24"), "yuv420p")
        for _ in range(3):
            reused = to_bt709(
                av.VideoFrame.from_ndarray(image, format="rgb24"),
                "yuv420p",
                reformatter=reformatter,
            )
            assert reused.colorspace == fresh.colorspace
            assert reused.color_range == fresh.color_range
            assert reused.color_primaries == fresh.color_primaries
            assert reused.color_trc == fresh.color_trc
            for plane, expected in zip(reused.planes, fresh.planes, strict=True):
                assert bytes(plane) == bytes(expected)

    def test_rgba_and_rgb24_give_the_same_pixels(self) -> None:
        # 書き出しは合成結果を RGBA のまま渡し、rgb24 へ詰め直す手間を省いている
        # swscale が A を混ぜるようになったら、半透明の所の色が変わって出る
        image = bars(320, 240)
        with_alpha = np.dstack([image, np.full(image.shape[:2], 128, dtype=np.uint8)])
        plain = to_bt709(av.VideoFrame.from_ndarray(image, format="rgb24"), "yuv420p")
        rgba = to_bt709(
            av.VideoFrame.from_ndarray(np.ascontiguousarray(with_alpha), format="rgba"), "yuv420p"
        )
        for plane, expected in zip(rgba.planes, plain.planes, strict=True):
            assert bytes(plane) == bytes(expected)

    def test_an_rgb_target_is_left_alone(self) -> None:
        # 行列の無い書式へ行列を持ち込むと、値が YUV の式で曲げられる
        rgb = av.VideoFrame.from_ndarray(bars(320, 240), format="rgb24")
        converted = to_bt709(rgb, "bgr24")
        assert rgb_at_bars(converted.to_ndarray(format="rgb24")) == list(COLORS)

    def test_a_yuv_stream_is_tagged(self, tmp_path: Path) -> None:
        # ストリームにタグが無いと、書き出したファイルを再生ソフトやブラウザが BT.601 で
        # 読むことがあり、赤が明るく緑が暗く見える
        skip_without_libx264()
        with av.open(str(tmp_path / "out.mp4"), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.pix_fmt = "yuv420p"
            tag_bt709(stream)
            context = stream.codec_context
            assert context.color_primaries == ColorPrimaries.BT709
            assert context.color_trc == ColorTrc.BT709
            assert context.colorspace == AVCOL_SPC_BT709
            assert context.color_range == ColorRange.MPEG

    def test_an_rgb_stream_is_not_tagged(self, tmp_path: Path) -> None:
        # 行列の無い絵に行列を書くと、読む側が YUV として変換し直して色が崩れる
        with av.open(str(tmp_path / "out.mov"), mode="w") as container:
            stream = container.add_stream("png", rate=30)
            stream.pix_fmt = "rgb24"
            tag_bt709(stream)
            assert stream.codec_context.colorspace == AVCOL_SPC_UNSPECIFIED


class TestGuessingTheMatrix:
    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (1920, 1080, Colorspace.ITU709),
            (1280, 720, Colorspace.ITU709),
            # 縦に撮ったスマホの映像 幅は狭いが HD
            (720, 1280, Colorspace.ITU709),
            # PAL の SD 高さ 576 までは SD とみなす
            (720, 576, Colorspace.ITU601),
            (640, 480, Colorspace.ITU601),
        ],
    )
    def test_an_untagged_frame_is_judged_by_size(
        self, width: int, height: int, expected: Colorspace
    ) -> None:
        # 大きさの判定を誤ると、タグの無い素材が別の行列で読まれる HD を BT.601 で読むと
        # 赤がくすみ緑が黄色へ寄り、SD を BT.709 で読むと肌色が赤っぽくなる
        frame = av.VideoFrame(width, height, "yuv420p")
        assert source_matrix(frame) == expected

    def test_a_tag_wins_over_the_size(self) -> None:
        # タグより大きさを優先すると、BT.601 と明記した HD の素材まで BT.709 で読んで色が変わる
        frame = av.VideoFrame(1920, 1080, "yuv420p")
        frame.colorspace = AVCOL_SPC_BT470BG
        assert source_matrix(frame) is None

    def test_rgb_has_no_matrix(self) -> None:
        # RGB に行列を指定すると swscale が要らない変換を挟み、PNG の素材の色まで動く
        assert source_matrix(av.VideoFrame(1920, 1080, "rgb24")) is None


class TestReading:
    @pytest.mark.parametrize(
        ("name", "width", "height", "matrix", "tag"),
        [
            # 直す前はこれが BT.601 で読まれ、赤が (232, 0, 1)・緑が (19, 255, 8) になっていた
            ("hd_untagged_709.mkv", 1280, 720, Colorspace.ITU709, None),
            ("sd_untagged_601.mkv", 640, 480, Colorspace.ITU601, None),
            ("sd_tagged_709.mkv", 640, 480, Colorspace.ITU709, AVCOL_SPC_BT709),
            ("hd_tagged_601.mkv", 1280, 720, Colorspace.ITU601, AVCOL_SPC_BT470BG),
        ],
    )
    def test_colors_come_back(
        self,
        tmp_path: Path,
        name: str,
        width: int,
        height: int,
        matrix: Colorspace,
        tag: int | None,
    ) -> None:
        # 行列の当て方を誤ると、読み込んだ素材が編集画面で赤がくすみ緑が黄色へ寄るなど
        # 元と違う色で出る
        path = write_bars(tmp_path / name, width, height, matrix=matrix, tag=tag)
        with VideoDecoder(path) as decoder:
            image = decoder.frame_at(Fraction(0))
        assert image is not None
        assert_close(rgb_at_bars(image), COLORS, tolerance=3)

    def test_an_effect_image_reads_like_a_clip(self, tmp_path: Path) -> None:
        # 画像合成などのエフェクトは動画も受け、先頭のコマを使う ここだけ素の to_ndarray だと、
        # 同じタグの無い HD の動画が、素材としては正しい色、エフェクトでは BT.601 の色になる
        path = write_bars(tmp_path / "hd.mkv", 1280, 720, matrix=Colorspace.ITU709, tag=None)
        image = read_image(path)
        assert image is not None
        assert_close(rgb_at_bars(image), COLORS, tolerance=3)


def gray_steps(color_range: ColorRange) -> av.VideoFrame:
    """Y が 16 / 128 / 235 の 3 段の灰色 範囲のタグだけを変えて使う

    16 と 235 は limited なら黒と白、full なら暗い灰と明るい灰 範囲を取り違えると
    ここが 0 と 255 に潰れるので、どちらで読んだかが値で分かる
    """
    width, height = 1280, 720
    frame = av.VideoFrame(width, height, "yuv420p")
    luma = np.empty((height, width), np.uint8)
    for index, value in enumerate((16, 128, 235)):
        luma[:, index * width // 3 : (index + 1) * width // 3] = value
    frame.planes[0].update(luma.tobytes())
    for plane in frame.planes[1:]:
        plane.update(bytes([128]) * plane.buffer_size)
    frame.colorspace = AVCOL_SPC_BT709
    frame.color_range = color_range
    return frame


def gray_levels(image: np.ndarray) -> list[int]:
    width = image.shape[1]
    return [int(image[image.shape[0] // 2, x]) for x in (width // 6, width // 2, width * 5 // 6)]


class TestRange:
    """範囲（full / limited）のタグが効いていること

    PyAV の変換はフレームに付いた範囲をそのまま swscale へ渡す ここで指定し直さなくても
    効いていることを、値で押さえておく
    """

    def test_full_range_is_read_as_full(self) -> None:
        # full の素材を limited として読むと、暗い所が黒へ明るい所が白へ潰れ、影や空の階調が消える
        image = to_rgb_array(gray_steps(ColorRange.JPEG), "rgb24")
        assert gray_levels(image[:, :, 0]) == [16, 128, 235]

    def test_limited_range_is_read_as_limited(self) -> None:
        # 普通の動画（limited）を full として読むと、黒が灰色に白がくすんで、
        # 素材が全体に白っぽく霞む
        image = to_rgb_array(gray_steps(ColorRange.MPEG), "rgb24")
        assert gray_levels(image[:, :, 0]) == [0, 130, 255]

    def test_full_range_is_squeezed_into_limited(self) -> None:
        # 控えは limited で書く full の 16 は limited の 30 あたり（16 + 16 * 219 / 255）
        converted = to_bt709(gray_steps(ColorRange.JPEG), "yuv420p")
        luma = converted.to_ndarray()[: converted.height]
        assert_close(
            [(value, 0, 0) for value in gray_levels(luma)],
            [(30, 0, 0), (126, 0, 0), (218, 0, 0)],
            tolerance=1,
        )
        assert converted.color_range == ColorRange.MPEG


@pytest.mark.skipif(not proxy_codecs(), reason="控えを作れるコーデックが無い")
class TestProxy:
    def test_a_proxy_of_untagged_hd_keeps_its_colors(self, tmp_path: Path) -> None:
        # 控えは SD の大きさなので、タグが無いと BT.601 で読まれる 元の BT.709 のまま
        # 縮めただけだと、控えで見たときだけ色がずれる
        source = write_bars(tmp_path / "hd.mkv", 1280, 720, matrix=Colorspace.ITU709, tag=None)
        target = tmp_path / "proxy.mp4"
        assert create_proxy(source, target, height=240) is not None

        with av.open(str(target)) as container:
            context = container.streams.video[0].codec_context
            assert context.colorspace == AVCOL_SPC_BT709
            assert context.color_range == ColorRange.MPEG
        with VideoDecoder(target) as decoder:
            image = decoder.frame_at(Fraction(0))
        assert image is not None
        # 控えは小ささを取った非可逆なので幅を持たせる 行列を取り違えると 20 以上ずれる
        assert_close(rgb_at_bars(image), COLORS, tolerance=6)

    def test_old_proxies_are_not_reused(self, tmp_path: Path) -> None:
        # 版 1 の控えはタグが無く、今の読み方では色がずれる 鍵が同じだと掴んでしまう
        path = tmp_path / "a.mp4"
        path.write_bytes(b"x")
        media = MediaItem(path=path, duration=Fraction(2))
        store = ProxyStore(CacheStore(tmp_path / "cache"), height=540)
        assert store.key_for(media) != media_key(path, extra="proxy540")
