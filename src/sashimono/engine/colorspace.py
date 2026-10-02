"""YUV と RGB の間の色の行列と、ファイルに付ける色のタグ

色の範囲は SDR の sRGB / Rec.709 と決めてある（#32） 合成は sRGB の値のまま行うので、
ここで扱うのは YUV との出入り口だけ

- 読むとき: 素材のタグに従う タグが無ければ大きさから当てる（HD は BT.709、SD は BT.601）
- 書くとき: 必ず BT.709 の行列・limited の範囲で変換し、同じ値をタグとして付ける

swscale は何も言わないと BT.601 で変換する 行列を指定しないまま HD を書き出すと、
赤が明るく緑が暗くずれ、タグも無いので再生側は推測で読む（#61）
"""

from __future__ import annotations

import av
import av.video.frame
import av.video.stream
import numpy as np
from av.video.reformatter import (
    ColorPrimaries,
    ColorRange,
    Colorspace,
    ColorTrc,
    VideoReformatter,
)

__all__ = [
    "VideoReformatter",
    "color_tags",
    "source_matrix",
    "tag_bt709",
    "to_bt709",
    "to_rgb_array",
]

#: AVColorSpace の「指定なし」と「予約」 PyAV は列挙型を出していないので値で持つ
#: どちらも行列が分からないという意味で、大きさから当てるしかない
_AVCOL_SPC_UNSPECIFIED = 2
_AVCOL_SPC_RESERVED = 3
_UNKNOWN_MATRICES = frozenset({_AVCOL_SPC_UNSPECIFIED, _AVCOL_SPC_RESERVED})

#: AVColorSpace の BT.709 コーデックの設定へ入れる値 :class:`Colorspace` は swscale 側の
#: 番号で、たまたま同じ 1 だが意味が違うので混ぜない
_AVCOL_SPC_BT709 = 1


def _has_matrix(pixel_format: av.VideoFormat) -> bool:
    """行列が意味を持つ書式か RGB・グレー・パレットには行列が無い

    RGB の素材（PNG など）にまで行列を指定すると、swscale が要らない変換を挟もうとする
    """
    return not pixel_format.is_rgb and len(pixel_format.components) >= 3


def source_matrix(frame: av.video.frame.VideoFrame) -> Colorspace | None:
    """素材のフレームを RGB に戻すときの行列 タグに任せてよいなら ``None``

    タグの無い素材は、mpv などの再生ソフトと同じく大きさで当てる 幅 1280 以上か
    高さ 577 以上なら HD とみなして BT.709、それより小さければ SD の BT.601
    当てずに swscale の既定へ任せると、タグの無い HD の素材がすべて BT.601 で読まれ、
    赤がくすみ緑が黄色へ寄る
    """
    if not _has_matrix(frame.format) or frame.colorspace not in _UNKNOWN_MATRICES:
        return None
    if frame.width >= 1280 or frame.height > 576:
        return Colorspace.ITU709
    return Colorspace.ITU601


def _reformat(
    frame: av.video.frame.VideoFrame,
    reformatter: VideoReformatter | None,
    *,
    width: int | None = None,
    height: int | None = None,
    pixel_format: str,
    src_colorspace: Colorspace | None = None,
    dst_colorspace: Colorspace | None = None,
    dst_color_range: ColorRange | None = None,
) -> av.video.frame.VideoFrame:
    """``reformatter`` があればそれで、無ければフレーム自身の表で変換する

    フレーム自身に任せると、フレームごとに swscale の表を作り直す
    受ける項目はここで使う 6 つに絞る ``**options`` で素通しにすると、
    PyAV の引数が変わっても型検査に掛からない
    """
    if reformatter is None:
        return frame.reformat(
            width=width,
            height=height,
            format=pixel_format,
            src_colorspace=src_colorspace,
            dst_colorspace=dst_colorspace,
            dst_color_range=dst_color_range,
        )
    return reformatter.reformat(
        frame,
        width=width,
        height=height,
        format=pixel_format,
        src_colorspace=src_colorspace,
        dst_colorspace=dst_colorspace,
        dst_color_range=dst_color_range,
    )


def to_rgb_array(frame: av.video.frame.VideoFrame, pixel_format: str) -> np.ndarray:
    """素材のフレームを RGB 系の配列へ 行列は :func:`source_matrix` で決める"""
    return frame.to_ndarray(format=pixel_format, src_colorspace=source_matrix(frame))


def to_bt709(
    frame: av.video.frame.VideoFrame,
    pixel_format: str,
    *,
    width: int | None = None,
    height: int | None = None,
    reformatter: VideoReformatter | None = None,
) -> av.video.frame.VideoFrame:
    """BT.709 / limited の YUV へ変換し、そのタグを付けたフレームを返す

    ``width`` / ``height`` を渡すと、縮めるのと行列の変換を 1 回で済ませる
    RGB の書式を頼まれたら行列もタグも関係ないので、書式だけ変える

    ``reformatter`` を渡すと、swscale の変換表をフレームをまたいで使い回す
    渡さないと :class:`~av.video.frame.VideoFrame` ごとに新しい表が作られる
    毎フレーム作り直すのは 1920x1080 で 1 枚 10ms ほどで、使い回すと 1ms を切る
    （書き出しの色変換 15.6ms のうち 10ms 近くがこれだった）
    出る画素は同じなので、同じ設定で呼び続ける所だけ渡す
    """
    if not _has_matrix(av.VideoFormat(pixel_format)):
        return _reformat(frame, reformatter, width=width, height=height, pixel_format=pixel_format)
    converted = _reformat(
        frame,
        reformatter,
        width=width,
        height=height,
        pixel_format=pixel_format,
        src_colorspace=source_matrix(frame),
        dst_colorspace=Colorspace.ITU709,
        dst_color_range=ColorRange.MPEG,
    )
    # 色域と転送特性は値を変えずにタグだけ付ける reformat の dst_color_trc などに
    # 渡すと swscale が曲線の変換を試み、未対応の組み合わせでは失敗する
    converted.color_primaries = ColorPrimaries.BT709
    converted.color_trc = ColorTrc.BT709
    return converted


def color_tags(stream: av.video.stream.VideoStream) -> tuple[str, str]:
    """素材の映像の伝達特性と原色の印 FFmpeg の名前（``smpte2084`` ``bt2020`` など）

    印が無い（指定なし）・PyAV が知らない番号のときは空 HDR の素材を見分けて知らせる
    ために読む（:attr:`~sashimono.core.model.VideoStreamInfo.color_outside_sdr`）
    名前は FFmpeg（ffprobe）と同じ綴りにする 列挙型の名前の ``_`` を ``-`` に替えれば揃う
    """
    context = stream.codec_context
    return _tag_name(ColorTrc, context.color_trc), _tag_name(
        ColorPrimaries, context.color_primaries
    )


def _tag_name(kind: type[ColorTrc] | type[ColorPrimaries], value: object) -> str:
    try:
        tag = kind(int(value))  # type: ignore[call-overload]  # PyAV の型は int を返すと書いていない
    except (TypeError, ValueError):
        return ""
    if tag.name == "UNSPECIFIED":
        return ""
    return tag.name.lower().replace("_", "-")


def tag_bt709(stream: av.video.stream.VideoStream) -> None:
    """書き出すストリームに BT.709 / limited の色のタグを付ける

    タグはフレームではなくコーデックの設定に付ける エンコーダ（libx264・NVENC・QSV）が
    ビットストリームの VUI へ書くのも、MP4 の colr へ写されるのも、開く前の設定の値
    フレームにだけ付けても、ファイルには残らない

    転送特性は sRGB（iec61966-2-1）ではなく BT.709 と書く 合成が作るのは sRGB で
    符号化された値で、厳密には sRGB だが、H.264 で sRGB と書いたファイルはブラウザや
    テレビ・スマホの再生支援で無視されたり、別の曲線で読まれたりする SDR の動画では
    sRGB の値を BT.709 と書いてそのまま渡すのが一般的なやり方で、YouTube の推奨も BT.709
    再生側は BT.709 と書かれた SDR をおおむね sRGB と同じ曲線で表示するので、見た目も揃う

    RGB で書くストリームには付けない 行列の無い絵に行列を書くと、読む側が YUV として
    変換し直して色が崩れる ``pix_fmt`` を決めてから呼ぶこと
    """
    context = stream.codec_context
    if context.format is None or not _has_matrix(context.format):
        return
    context.color_primaries = ColorPrimaries.BT709
    context.color_trc = ColorTrc.BT709
    context.colorspace = _AVCOL_SPC_BT709
    context.color_range = ColorRange.MPEG
