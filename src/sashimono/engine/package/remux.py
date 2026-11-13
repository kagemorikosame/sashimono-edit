"""素材の使っている範囲だけを、再エンコードせずに写す（#423）

持ち出し用のパッケージは画質を落とさない約束なので、パケットをそのまま写す（FFmpeg の
``-c copy`` に当たる） 再エンコードしないので、映像はキーフレームでしか切れない 使う区間の
前のキーフレームから入れる

**時刻（PTS）は元のまま保つ** 作品の中の時刻（``Clip.source_in``・字幕の時刻・絵を止める
時刻）は素材の原点（:func:`~sashimono.engine.decode.probe.media_origin`）から数えてあり、
写した素材の原点が元と同じなら、作品を 1 か所も書き換えずに済む 原点は道の頭の時刻で
決まるので、どの道も頭のパケット 1 つ（映像はキーフレーム）を必ず入れる（錨） 錨と使う区間の
間は何も無い区間になるが、そこを読むクリップは無い（使う区間だけを入れたので）

写し終えたら、原点・道の並び・回転・色・区間の頭の絵を元と比べる 1 つでも違えば写した物を
捨て、素材をまるごと入れる（呼ぶ側 :mod:`sashimono.engine.package.pack`） 入れ物ごとの
癖（時刻の丸め・頭の時刻の書き方）を推測で当てにせず、写した結果で確かめるため
"""

from __future__ import annotations

import bisect
import contextlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import av.error
import numpy as np

from sashimono.core.model import MediaItem
from sashimono.engine.decode.audio import AudioDecoder
from sashimono.engine.decode.probe import ProbeError, media_origin, moving_pictures, probe_media
from sashimono.engine.decode.rational import as_fraction
from sashimono.engine.decode.video import VideoDecoder

__all__ = [
    "EDITING_CODECS",
    "REENCODE_CRF",
    "REENCODE_LEAST_PSNR",
    "Cancelled",
    "Reencoded",
    "Span",
    "cut_media",
    "editing_codec",
    "frame_psnr",
    "output_format",
    "reencode_media",
    "spans_union",
]

#: 区間の後ろに足す長さ（秒） B フレームの並べ替えで、区間の終わりの絵を出すのに
#: 後ろの数枚のパケットを待つ素材がある 数枚ぶんあれば足りる
_TAIL_PAD = Fraction(1, 2)

#: シークが目当てより後ろのキーフレームに着いたとき、さらに前へ戻す長さ（秒）
#: 索引の粗い入れ物では、目当ての手前のキーフレームを返さないことがある
_SEEK_BACKOFF = Fraction(5)

#: シークをやり直す回数の上限 戻しても見つからなければ頭から読む
_SEEK_TRIES = 4

#: 入れ物の名前（PyAV の ``format.name`` に含まれる名前）と、書き出すときの名前と拡張子
#: どれも時刻を道ごとの時間の単位で持てる（区間の間の空きを表せる）入れ物 AVI のように
#: フレームの番号で数える入れ物は空きを表せないので、まるごと入れる側へ回す
_FORMATS: tuple[tuple[str, str, str], ...] = (
    ("mov", "mov", ".mov"),
    ("mp4", "mp4", ".mp4"),
    ("matroska", "matroska", ".mkv"),
    ("webm", "matroska", ".mkv"),
    ("mpegts", "mpegts", ".ts"),
)


@dataclass(frozen=True, slots=True)
class Span:
    """素材の中の時刻の区間（秒 原点から数える） ``[start, end]``"""

    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        """終わりが頭より前の区間を断る"""
        if self.end < self.start:
            raise ValueError(f"区間の終わりが頭より前: {self.start} - {self.end}")


def spans_union(spans: Sequence[Span]) -> list[Span]:
    """重なる・接する区間をまとめ、頭の順に並べる"""
    merged: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if merged and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return merged


def output_format(source: Path) -> tuple[str, str] | None:
    """``source`` を切り出して書くときの入れ物の名前と拡張子 切り出せない入れ物は ``None``

    元と同じ種類の入れ物にする 道の符号（コーデック）が入れ物に入るかは入れ物ごとに違い、
    元の入れ物なら必ず入る mp4 と mov は同じ読み手（``mov,mp4,...``）なので拡張子で分ける
    """
    try:
        with av.open(str(source)) as container:
            names = container.format.name.split(",")
    except (av.error.FFmpegError, OSError):
        return None
    suffix = source.suffix.lower()
    if "mov" in names:
        # mov の読み手は mp4・m4a・3gp も読む 拡張子が mov 以外なら mp4 で書く
        # （mp4 の書き手は mov より入る符号が少ないので、入らなければ呼ぶ側がまるごと入れる）
        return (
            ("mov", ".mov") if suffix == ".mov" else ("mp4", ".m4a" if suffix == ".m4a" else ".mp4")
        )
    for name, muxer, extension in _FORMATS:
        if name in names:
            return muxer, extension
    return None


def _kept_streams(container: av.container.InputContainer) -> list[av.stream.Stream] | None:
    """写す道 番号を元と同じに保てなければ ``None``

    クリップは素材の道を番号で指す（``Clip.stream_index`` ``Clip.audio_stream``） 写した素材で
    番号がずれると、別の音が鳴る 映像と音声の道を写し、後ろに並ぶほかの道（位置の記録・
    時刻の符号など）は落とす 映像と音声の間にほかの道があると、落とすと番号がずれるので
    切り出さない（呼ぶ側がまるごと入れる） カバー画像の道も同じ
    """
    pictures = {stream.index for stream in moving_pictures(container)}
    audio = {stream.index for stream in container.streams.audio}
    wanted = pictures | audio
    if not wanted:
        return None
    last = max(wanted)
    streams = [stream for stream in container.streams if stream.index <= last]
    if any(stream.index not in wanted for stream in streams):
        return None
    return sorted(streams, key=lambda stream: stream.index)


def _seconds(packet: av.Packet[Any]) -> Fraction | None:
    """パケットの表示の時刻（秒 PTS の数え方） 分からなければ復号の時刻"""
    stamp = packet.pts if packet.pts is not None else packet.dts
    if stamp is None or packet.time_base is None:
        return None
    return stamp * as_fraction(packet.time_base)


def _decode_seconds(packet: av.Packet[Any]) -> Fraction | None:
    """パケットの解く順の時刻（秒） 分からなければ ``None``"""
    stamp = packet.dts if packet.dts is not None else packet.pts
    if stamp is None or packet.time_base is None:
        return None
    return stamp * as_fraction(packet.time_base)


class Cancelled(Exception):  # noqa: N818 止めたことを知らせる印で、失敗ではない
    """利用者が止めた"""


def cut_media(
    source: Path,
    target: Path,
    spans: Sequence[Span],
    *,
    muxer: str,
    keep_metadata: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> bool:
    """``source`` の ``spans`` の区間だけを ``target`` へ写す 確かめて元と同じなら真

    偽のときは ``target`` を消してある（呼ぶ側がまるごと入れる） 止められたら
    :class:`Cancelled` を上げる（``target`` は消す）

    ``keep_metadata`` が偽なら、入れ物と道の付け足しの情報（撮った場所・機器の名前など）を
    写さない 回転と色は道の符号の情報（表示の行列・色の印）なので、偽でも残る 音の道の
    言語だけは写す（多言語の音を選ぶ手がかりで、個人の情報ではない）
    """
    merged = spans_union(spans)
    try:
        written = _write(source, target, merged, muxer, keep_metadata, cancelled)
    except Cancelled:
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
        raise
    except (av.error.FFmpegError, OSError, ValueError, ProbeError):
        written = False
    try:
        same = written and _same_as_source(source, target, merged, cancelled)
    except Cancelled:
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
        raise
    if same:
        return True
    with contextlib.suppress(OSError):
        target.unlink(missing_ok=True)
    return False


def _write(
    source: Path,
    target: Path,
    spans: list[Span],
    muxer: str,
    keep_metadata: bool,
    cancelled: Callable[[], bool],
    reencode: frozenset[int] = frozenset(),
    sizes: list[tuple[int, int]] | None = None,
) -> bool:
    """使う区間のパケットを写して書く 書けなければ偽

    ``reencode`` の番号の映像の道だけは、写さずに解いて H.264 にし直す（#446 :class:`_Encoder`）
    音とほかの映像の道は写す
    """
    with av.open(str(source)) as reader:
        kept = _kept_streams(reader)
        if kept is None:
            return False
        if not reencode <= {stream.index for stream in kept}:
            return False
        if keep_metadata and len(kept) < len(reader.streams):
            # 付け足しの情報を残すと選んだのに、後ろの道（位置の記録など）を落とすと選んだ意味が
            # 無い まるごと入れる（#449 の CodeRabbit の指摘）
            return False
        origin = media_origin(reader)
        video = moving_pictures(reader)
        guide = video[0] if video else kept[0]
        with av.open(str(target), "w", format=muxer) as writer:
            mapping: dict[int, av.stream.Stream] = {}
            encoders: dict[int, _Encoder] = {}
            for stream in kept:
                if stream.index in reencode:
                    encoder = _Encoder.open(writer, stream)
                    if encoder is None:
                        return False
                    encoders[stream.index] = encoder
                    copy = encoder.stream
                else:
                    copy = writer.add_stream_from_template(stream)
                if keep_metadata:
                    copy.metadata.update(stream.metadata)
                elif "language" in stream.metadata:
                    copy.metadata["language"] = stream.metadata["language"]
                mapping[stream.index] = copy
            if keep_metadata:
                writer.metadata.update(reader.metadata)
            #: 道ごとに写したパケットの時刻 区間を前のキーフレームまで広げると隣の区間と
            #: 重なることがあり、同じパケットを 2 度写さないため 復号の時刻（dts）は matroska の
            #: B フレームで空になるので、表示の時刻で見分ける（道の中で重ならない）
            #: 並べ替えの範囲が素材ごとに違うので、写した時刻を全部持つ
            #: （1 つ数十バイトで、2 時間の素材でも数 MB）
            written: dict[int, set[int]] = {stream.index: set() for stream in kept}
            #: 道ごとに最後に写した復号の時刻（道の時間の単位）
            last_dts: dict[int, int] = {}
            picture_indices = {stream.index for stream in video}

            def put(packet: av.Packet[Any]) -> None:
                """パケットを 1 つ書く 同じ時刻の物は 2 度書かない"""
                index = packet.stream.index
                stamp = packet.pts if packet.pts is not None else packet.dts
                if stamp is None or stamp in written[index]:
                    return
                if packet.dts is None and packet.pts is not None and index in last_dts:
                    # matroska は復号の時刻を持たず、シークの直後のパケットは空で来る 空のまま
                    # 渡すと書き手が表示の時刻から推して、前の区間の終わりより前の値を作り、
                    # 順が戻ったとして断る 前の値の次を入れる（matroska は復号の時刻を書かない
                    # ので、読み直した素材には残らない）
                    packet.dts = min(packet.pts, last_dts[index] + 1)
                if packet.dts is not None:
                    if index in last_dts and packet.dts <= last_dts[index]:
                        return
                    last_dts[index] = packet.dts
                written[index].add(stamp)
                if index in encoders:
                    encoders[index].push(packet, writer)
                    return
                packet.stream = mapping[index]
                writer.mux(packet)

            # 隙間を持てない音の道（:func:`_steady`）
            steady = [stream for stream in kept if _steady(stream, muxer)]
            quiet = {stream.index for stream in steady}
            # 区間は頭の順に並び重ならない（spans_union） パケットごとに全部の区間を見ると、
            # 1 時間の音（十数万パケット）と数百の区間で書き出しが長く止まる 頭の並びから探す
            starts = [span.start for span in spans]

            def hush(packet: av.Packet[Any]) -> av.Packet[Any]:
                return _hush(packet, spans, starts, origin)

            # 錨 どの道も頭のパケットを入れて、道の頭の時刻（素材の原点）を元と同じに保つ
            reader.seek(0)
            anchored: set[int] = set()
            for packet in reader.demux(*kept):
                # 遅く始まる道があると、頭のパケットを探すのに長く読む その間も止められるように
                if cancelled():
                    raise Cancelled
                index = packet.stream.index
                if index in anchored or _seconds(packet) is None:
                    continue
                if index in picture_indices and not packet.is_keyframe:
                    continue
                put(hush(packet) if index in quiet else packet)
                anchored.add(index)
                if len(anchored) == len(kept):
                    break

            # 隙間を持てない音の道は、頭から最後の区間の終わりまで続けて写す 区間の外は無音に
            # 置き換える（時刻は保ったまま、使っていない所の声や会話を持ち出さない）
            if steady and spans:
                whole = Span(Fraction(0), max(span.end for span in spans))
                _copy_span(
                    reader,
                    steady,
                    steady[0],
                    set(),
                    origin,
                    whole,
                    lambda packet: put(hush(packet)),
                    cancelled,
                )
            rest = [stream for stream in kept if stream not in steady]
            # 残りが無いのに読むと、demux は空の指定を「全部の道」と読み、写さない道（時刻つきの
            # データの道など）まで書き手へ渡して落ちる
            for span in spans if rest else ():
                _copy_span(reader, rest, guide, picture_indices, origin, span, put, cancelled)
            for encoder in encoders.values():
                encoder.finish(writer)
                if sizes is not None:
                    sizes.append((encoder.read_bytes, encoder.wrote_bytes))
    return True


#: 無音を作れる PCM の形と、無音の 1 バイト 符号つきの整数と小数は 0 が無音 符号なしの
#: 8bit は真ん中 a-law と μ-law は 0 を表す値（0 を入れると最大の音になる）
_PCM_SILENCE = {
    **{
        f"pcm_{kind}": 0
        for kind in (
            "s16le", "s16be", "s24le", "s24be", "s32le", "s32be", "s64le", "s64be",
            "f16le", "f24le", "f32le", "f32be", "f64le", "f64be", "s8",
        )
    },
    "pcm_u8": 0x80,
    "pcm_alaw": 0xD5,
    "pcm_mulaw": 0xFF,
}  # fmt: skip


def _steady(stream: av.stream.Stream, muxer: str) -> bool:
    """区間の間の空きを書けない道か

    mov と mp4 の書き手は PCM の音を 1 標本ずつの決まった長さで並べ、パケットの時刻を書かない
    空きを空けて渡すと詰めて書くので、2 つ目の区間からの音が前へずれる（ProRes の .mov は
    たいてい PCM の音を持つ） 空けずに続けて写し、区間の外は無音にする（:func:`_hush`）
    無音を作れない形（見出しを持つ pcm_dvd など）は空けて写し、確かめに落ちて全体を入れる
    """
    return (
        stream.type == "audio"
        and muxer in ("mov", "mp4")
        and str(stream.codec_context.name) in _PCM_SILENCE
    )


#: PCM の形ごとの 1 標本のバイト数 名前の数字（s16 なら 16 ビット）から読む 数字を持たない
#: a-law と μ-law は 8 ビット
_PCM_BYTES = {name: int(re.sub(r"\D", "", name) or 8) // 8 for name in _PCM_SILENCE}


def _hush(
    packet: av.Packet[Any], spans: list[Span], starts: list[Fraction], origin: Fraction
) -> av.Packet[Any]:
    """区間の外にある PCM の標本を無音に置き換える（パケットの中でも標本ごと）

    続けて写すと区間の間の音も入る 使っていない所の声や会話を持ち出すと、範囲だけを入れると
    約束したパッケージから、渡した人の知らない音が漏れる 区間に掛かるパケットも丸ごと残すと、
    区間の終わりの直後の言葉が入る 長さと時刻はそのまま（ずらさない）
    区間には書き出しの余裕（:attr:`PackageOptions.margin`）が足してあるので、ここで足さない
    ``spans`` は頭の順に並び重ならない ``starts`` はその頭の並び
    """
    shown = _seconds(packet)
    codec = packet.stream.codec_context
    name = str(codec.name)
    rate = int(codec.sample_rate or 0)
    width = _PCM_BYTES[name] * int(codec.channels or 0)
    if shown is None or rate <= 0 or width <= 0:
        return packet
    data = bytes(packet)
    count = len(data) // width
    fill = _PCM_SILENCE[name]
    kept = bytearray([fill]) * len(data)
    head = shown - origin
    end = head + Fraction(count, rate)
    # パケットの頭より前に始まる最後の区間から、パケットの終わりより前に始まる区間までを見る
    for span in spans[max(0, bisect.bisect_right(starts, head) - 1) :]:
        if span.start > end:
            break
        # 標本 i は shown + i / rate に鳴る 区間に入る標本の番号の範囲を求めて元の音を戻す
        first = max(0, math.ceil((origin + span.start - shown) * rate))
        last = min(count, math.floor((origin + span.end - shown) * rate) + 1)
        if first < last:
            kept[first * width : last * width] = data[first * width : last * width]
    if kept == data:
        return packet
    quiet = av.Packet(bytes(kept))
    quiet.pts = packet.pts
    quiet.dts = packet.dts
    quiet.duration = packet.duration
    quiet.time_base = packet.time_base
    quiet.stream = packet.stream
    return quiet


def _copy_span(
    reader: av.container.InputContainer,
    kept: list[av.stream.Stream],
    guide: av.stream.Stream,
    picture_indices: set[int],
    origin: Fraction,
    span: Span,
    put: Callable[[av.Packet[Any]], None],
    cancelled: Callable[[], bool],
) -> None:
    """1 つの区間を写す 映像は区間の頭より前のキーフレームから、終わりの少し後ろまで"""
    head = origin + span.start
    tail = origin + span.end + _TAIL_PAD
    time_base = as_fraction(guide.time_base) if guide.time_base else Fraction(1, 1000)
    target: Fraction | None = head
    for _ in range(_SEEK_TRIES):
        assert target is not None
        reader.seek(max(0, int(target / time_base)), stream=guide, backward=True)
        first = _first_keyframe(reader, guide, cancelled)
        if first is None or first <= head or target <= 0:
            break
        # 着いたキーフレームが区間の頭より後ろ さらに前から探す
        target = max(Fraction(0), target - _SEEK_BACKOFF)
    else:
        # 戻しても見つからない入れ物は頭から読む 遅いが、取りこぼすよりよい
        target = None
    # 探すために読み進めたので、もう 1 度同じ所へ戻ってから写す
    if target is None:
        reader.seek(0)
    else:
        reader.seek(max(0, int(target / time_base)), stream=guide, backward=True)

    started: set[int] = set()
    finished: set[int] = set()
    for packet in reader.demux(*kept):
        if cancelled():
            raise Cancelled
        index = packet.stream.index
        if index in finished:
            continue
        shown = _seconds(packet)
        decoded = _decode_seconds(packet)
        if shown is None or decoded is None:
            continue
        if index in picture_indices:
            if index not in started:
                # 区間の頭以前で最後のキーフレームから入れる 頭より後ろのキーフレームしか
                # 無ければ、そこから入れる（素材の頭より前を指す区間）
                if not packet.is_keyframe:
                    continue
                started.add(index)
            if decoded > tail:
                finished.add(index)
                continue
            put(packet)
        else:
            if shown > tail:
                finished.add(index)
                continue
            if shown + _TAIL_PAD < head:
                continue
            put(packet)
        if len(finished) == len(kept):
            break


def _first_keyframe(
    reader: av.container.InputContainer, guide: av.stream.Stream, cancelled: Callable[[], bool]
) -> Fraction | None:
    """今の位置から読んで最初のキーフレームの時刻 無ければ ``None``"""
    for packet in reader.demux(guide):
        if cancelled():
            raise Cancelled
        if packet.is_keyframe and _seconds(packet) is not None:
            return _seconds(packet)
    return None


def _same_as_source(
    source: Path, target: Path, spans: list[Span], cancelled: Callable[[], bool]
) -> bool:
    """写した素材が、使う区間で元と同じ絵を出すか

    入れ物によって時刻の丸め（matroska は 1ms）や頭の時刻の書き方が違う 推測で信じず、
    素材の原点・道の並び・回転・色・区間の頭の絵（画素ごと）を元と比べる
    区間が多いと比べるのにも時間が掛かるので、区間ごとに止めるよう頼まれたかを見る
    （窓を閉じたときに、比べ終わるまで待たせない）
    """
    try:
        before = probe_media(source)
        after = probe_media(target)
        with av.open(str(source)) as a, av.open(str(target)) as b:
            if media_origin(a) != media_origin(b):
                return False
    except (ProbeError, av.error.FFmpegError, OSError):
        return False
    if [s.index for s in before.video_streams] != [s.index for s in after.video_streams]:
        return False
    if [(s.index, s.codec, s.channels) for s in before.audio_streams] != [
        (s.index, s.codec, s.channels) for s in after.audio_streams
    ]:
        return False
    for old, new in zip(before.video_streams, after.video_streams, strict=True):
        if (old.rotation, old.color_transfer, old.color_primaries, old.codec, old.pixel_format) != (
            new.rotation,
            new.color_transfer,
            new.color_primaries,
            new.codec,
            new.pixel_format,
        ):
            return False
    try:
        # 映像の道は全部見る キーフレームの位置は道ごとに違い、最初の道で切った所が 2 本目の
        # 道のキーフレームより後ろだと、2 本目の区間の頭の絵が欠ける（#449 の CodeRabbit の指摘）
        for stream in before.video_streams:
            if not _same_pictures(source, target, stream.index, spans, cancelled):
                return False
        # 音の道も区間の頭を比べる AAC は前のパケットを読まないと頭が違う音になる
        for audio in before.audio_streams:
            if not _same_sound(
                source, target, audio.index, audio.sample_rate, audio.channels, spans, cancelled
            ):
                return False
    except (ProbeError, av.error.FFmpegError, OSError, ValueError):
        return False
    return True


#: 音を比べる長さ（秒） 区間の頭の、前の音に頼る所を見るのに足りる
_SOUND_CHECK = Fraction(1, 4)
#: 音の違いを許す幅（振幅 -80 dB） AAC は解き始める所で前の窓の重なりが違い、同じパケットでも
#: 1e-5（-98 dB）ほどの差が残る（実物で測った） 聞こえない差で切り出しを捨てないよう、それより
#: 大きく、パケットの欠け（区間の頭の音が別物になる）よりずっと小さい幅にする
#: 比べる速さは素材の速さのまま 変換して比べると、変換の位相のずれ（0.002）で同じ音でも違って見える
_SOUND_TOLERANCE = 1e-4


def _same_pictures(
    source: Path, target: Path, index: int, spans: list[Span], cancelled: Callable[[], bool]
) -> bool:
    """区間の頭の絵が画素ごとに同じか 読めなかった所があれば、確かめられなかったとして偽"""
    # 読み手にも止める頼みを渡す 区間の頭はキーフレームから読み進めるので、長い GOP の
    # 高い解像度では 1 枚に時間が掛かり、窓を閉じても待たされる（#449 の CodeRabbit の指摘）
    with (
        VideoDecoder(source, index, should_stop=cancelled) as original,
        VideoDecoder(target, index, should_stop=cancelled) as copied,
    ):
        for span in spans:
            if cancelled():
                raise Cancelled
            moment = max(Fraction(0), span.start)
            old_frame = original.frame_at(moment)
            new_frame = copied.frame_at(moment)
            if cancelled():
                raise Cancelled
            # 両方とも読めずに絵が無いと同じに見える 音と同じく、読めなかったら切り出さない
            if original.decode_error is not None or copied.decode_error is not None:
                return False
            if old_frame is None and new_frame is None:
                continue
            if old_frame is None or new_frame is None:
                return False
            if not np.array_equal(old_frame, new_frame):
                return False
    return True


#: 音を比べられるチャンネルの数（読み手が並びを知っている物） ほかの数はまるごと入れる
_SOUND_CHANNELS = frozenset({1, 2, 6, 8})


def _same_sound(
    source: Path,
    target: Path,
    index: int,
    sample_rate: int,
    channels: int,
    spans: list[Span],
    cancelled: Callable[[], bool],
) -> bool:
    """区間の頭の音が素材のチャンネルのまま同じか 読めなかった所があれば偽"""
    # チャンネルは素材のまま全部比べる 2 つへ混ぜて比べると、5.1 の LFE（混ぜると消える）の
    # 欠けを見落とす 比べられない数なら確かめられないので切り出さない（#449 の CodeRabbit の指摘）
    if channels not in _SOUND_CHANNELS:
        return False
    with (
        AudioDecoder(
            source, sample_rate=sample_rate, channels=channels, stream_index=index
        ) as original,
        AudioDecoder(
            target, sample_rate=sample_rate, channels=channels, stream_index=index
        ) as copied,
    ):
        for span in spans:
            if cancelled():
                raise Cancelled
            moment = max(Fraction(0), span.start)
            length = min(_SOUND_CHECK, max(Fraction(0), span.end - moment)) or _SOUND_CHECK
            old = original.read_seconds(moment, length)
            new = copied.read_seconds(moment, length)
            if old.shape != new.shape:
                return False
            if old.size and float(np.max(np.abs(old - new))) > _SOUND_TOLERANCE:
                return False
        # 解けなかった所は読み手が無音で埋める 両方が無音なら同じに見えるので、解けなかったら
        # 確かめられなかったとして切り出さない（#449 の CodeRabbit の指摘）
        if original.decode_error is not None or copied.decode_error is not None:
            return False
    return True


# --- 編集用の大きな形式を符号にし直す（#446） --------------------------------------
#
# ProRes・DNxHR・無圧縮は 1 枚ずつ絵を持つので、使う範囲だけでも数十 GB になりうる
# 持ち出し用パッケージは品質を保つ約束（#423）なので既定にはせず、選んだときだけ H.264 へ
# 符号にし直す 時刻（PTS）と道の番号は元のまま保ち、作品を 1 か所も書き換えずに済ませる
# 書き終えたら元との差（PSNR）を測り、:data:`REENCODE_LEAST_PSNR` に届かなければ捨てて
# 写す側へ戻す

#: 編集用の大きな形式（FFmpeg のデコーダの名前） どれも 1 枚ずつ絵を持つ（前後の絵に頼らない）
#: ffv1 は可逆の圧縮でもう十分小さいので入れない
EDITING_CODECS = frozenset(
    {
        "prores",
        "dnxhd",
        "rawvideo",
        "v210",
        "v410",
        "cfhd",
        "huffyuv",
        "ffvhuff",
        "utvideo",
        "magicyuv",
    }
)

#: 符号にし直すときの x264 の画質（CRF 小さいほど元に近い） 18 で「見て分からない」と
#: 言われることが多いが、色を直す・拡大するなど後で絵をいじると差が出やすいので、余裕を取って 12
REENCODE_CRF = 12

#: 元との差がこれより大きい（PSNR が低い）ときは使わずに写す側へ戻す（dB）
#: 40 dB を下回ると、動きの多い所や暗い所で目に見える崩れが出始める
REENCODE_LEAST_PSNR = 40.0

#: 符号にし直した映像が元のこの割合より小さくならなければ使わない
_REENCODE_WORTH = Fraction(4, 5)

#: キーフレームの間隔（秒） 長いとクリップの頭へ跳ぶたびに前のキーフレームから解き直す
_REENCODE_GOP = 2

#: x264 が受け取れる画素の形 これ以外はここへ寄せる（深さと色差の間引きは保つ）
_X264_FORMATS = frozenset(
    {"yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "gray"}
)


@dataclass(frozen=True, slots=True)
class Reencoded:
    """符号にし直した結果 窓と目録が知らせる"""

    #: 元の符号の名前（``prores`` など）
    codec: str
    #: 新しい符号の名前（``h264``）
    to: str
    #: 新しい画素の形（作品の素材の情報に書き直す）
    pixel_format: str
    #: 使う区間で測った元との差のうち一番大きい所（PSNR dB）
    psnr: float
    #: 映像の元の大きさと、符号にし直した大きさ（使う区間の分 バイト）
    before: int = 0
    after: int = 0


def editing_codec(source: Path) -> str | None:
    """``source`` の映像が編集用の大きな形式なら、その符号の名前"""
    try:
        with av.open(str(source)) as container:
            for stream in moving_pictures(container):
                name = stream.codec_context.name
                if name in EDITING_CODECS:
                    return str(name)
    except (av.error.FFmpegError, OSError):
        return None
    return None


def reencodable(pixel_format: str | None) -> bool:
    """その画素の形の映像を H.264 にし直せるか（窓に対象として並べるかの判断 #446）"""
    return pixel_format is not None and _x264_format(pixel_format) is not None


def _x264_format(name: str) -> str | None:
    """元の画素の形に一番近い、x264 が受け取れる形 寄せられない形は ``None``

    透明の道を持つ形（ProRes 4444 など）と RGB の形は寄せない 透明が消える・色の変換で
    わずかに色が変わるのを黙って入れるより、写して大きいままにする
    """
    if name in _X264_FORMATS:
        return name
    try:
        fmt = av.VideoFormat(name)
    except (ValueError, av.error.FFmpegError):
        return None
    if fmt.is_rgb or fmt.has_palette or fmt.is_bayer or len(fmt.components) in (2, 4):
        return None
    bits = max(component.bits for component in fmt.components)
    if bits > 10:
        # x264 は 10bit まで 12bit 以上を 10bit へ落とすと、確かめの差も落とした形で測るので、
        # 深さを失ったことが差に出ない 写して大きいままにする
        return None
    if len(fmt.components) == 1:
        return "gray" if bits == 8 else None
    deep = bits > 8
    wide = fmt.chroma_width(16) == 16
    tall = fmt.chroma_height(16) == 16
    half_wide = fmt.chroma_width(16) == 8
    half_tall = fmt.chroma_height(16) == 8
    # 色差の間引きが x264 の 3 つ（4:4:4・4:2:2・4:2:0）と同じ物だけ寄せる 4:4:0・4:1:1 などを
    # 寄せると色差を捨てるが、確かめの差は元も寄せた形にしてから測るので、捨てた分が差に出ない
    if wide and tall:
        chroma = "444"
    elif half_wide and tall:
        chroma = "422"
    elif half_wide and half_tall:
        chroma = "420"
    else:
        return None
    return f"yuv{chroma}p" + ("10le" if deep else "")


class _Encoder:
    """映像の道 1 本を解いて H.264 にし直す 時刻（PTS）は元の道の時間の単位のまま渡す"""

    def __init__(self, stream: Any, source: Any, pixel_format: str) -> None:
        self.stream = stream
        self._source = source
        self._pixel_format = pixel_format
        self._last_pts: int | None = None
        #: 解いた元のパケットと、書いた新しいパケットの大きさの和（バイト） 小さくならなければ
        #: 符号にし直す意味が無いので、呼ぶ側が写す側へ戻す
        self.read_bytes = 0
        self.wrote_bytes = 0

    @classmethod
    def open(cls, writer: Any, source: Any) -> _Encoder | None:
        """書き手に符号の道を足す 寄せられない画素の形なら ``None``"""
        context = source.codec_context
        pixel_format = _x264_format(str(context.pix_fmt or ""))
        if pixel_format is None or not context.width or not context.height:
            return None
        rate = source.average_rate or source.guessed_rate
        stream = writer.add_stream(
            "libx264",
            rate=rate,
            options={
                "crf": str(REENCODE_CRF),
                "preset": "medium",
                # B フレームを使わない 使うと mov の書き手が頭に編集の表を足し、道の頭の時刻
                # （素材の原点）が元とずれる 原点がずれると作品の時刻が全部別の絵を指す
                "bf": "0",
            },
        )
        stream.time_base = source.time_base
        encoder = stream.codec_context
        encoder.width = context.width
        encoder.height = context.height
        encoder.pix_fmt = pixel_format
        encoder.time_base = source.time_base
        encoder.gop_size = max(1, int(float(rate or 30) * _REENCODE_GOP))
        encoder.max_b_frames = 0
        # 色の印は元のまま HDR や BT.2020 の素材が、開いた先で Rec.709 として褪せて出ないように
        encoder.color_primaries = context.color_primaries
        encoder.color_trc = context.color_trc
        encoder.colorspace = context.colorspace
        encoder.color_range = context.color_range
        return cls(stream, source, pixel_format)

    def push(self, packet: Any, writer: Any) -> None:
        """パケットを解き、出てきた絵を符号にして書く"""
        self.read_bytes += packet.size
        for frame in self._source.decode(packet):
            self._encode(frame, writer)

    def _encode(self, frame: Any, writer: Any) -> None:
        if frame.pts is None or (self._last_pts is not None and frame.pts <= self._last_pts):
            # 時刻の無い絵と、前に書いた時刻以前の絵は書かない 書き手は時刻が戻ると断る
            return
        if self._last_pts is None:
            self._take_colors(frame)
        self._last_pts = frame.pts
        picture = frame.reformat(format=self._pixel_format)
        picture.pts = frame.pts
        picture.time_base = self._source.time_base
        for out in self.stream.encode(picture):
            self.wrote_bytes += out.size
            writer.mux(out)

    def _take_colors(self, frame: Any) -> None:
        """道に書いていない色の印を、最初の絵から取る（符号の作り手は最初の絵で開く）

        ProRes は範囲（テレビの範囲）を道でなく絵ごとに持つ 書かずに開くと、開いた先の読み手が
        範囲を推し直して色が浅く・濃くずれる（試験の素材で RGB の PSNR が 36 dB に落ちた）
        """
        encoder = self.stream.codec_context
        for name in ("color_range", "colorspace", "color_primaries", "color_trc"):
            if not getattr(encoder, name, None) and getattr(frame, name, None):
                with contextlib.suppress(AttributeError, ValueError, TypeError):
                    setattr(encoder, name, getattr(frame, name))

    def finish(self, writer: Any) -> None:
        """解き手と符号の作り手に残った物を出し切る"""
        with contextlib.suppress(av.error.FFmpegError):
            for frame in self._source.decode(None):
                self._encode(frame, writer)
        for out in self.stream.encode(None):
            self.wrote_bytes += out.size
            writer.mux(out)


def reencode_media(
    source: Path,
    target: Path,
    spans: Sequence[Span],
    *,
    muxer: str,
    keep_metadata: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> Reencoded | None:
    """``source`` の ``spans`` の区間を、編集用の形式の映像だけ H.264 にし直して書く

    元との差が :data:`REENCODE_LEAST_PSNR` に届かない・時刻や道が元と違う・書けないときは
    ``target`` を消して ``None``（呼ぶ側が写す側へ戻す） 止められたら :class:`Cancelled`
    """
    merged = spans_union(spans)
    try:
        with av.open(str(source)) as container:
            chosen = {
                stream.index: str(stream.codec_context.name)
                for stream in moving_pictures(container)
                if stream.codec_context.name in EDITING_CODECS
            }
    except (av.error.FFmpegError, OSError):
        return None
    if len(chosen) != 1:
        # 知らせと目録（:class:`Reencoded`）は符号と画素の形を 1 つずつしか持たない 2 本を
        # し直すと、2 本目の素材の情報が元の符号のまま残り、中身の H.264 と食い違う
        # 編集用の形式の映像を 2 本持つ素材はまれなので、元のまま入れる
        return None
    try:
        sizes: list[tuple[int, int]] = []
        written = _write(
            source, target, merged, muxer, keep_metadata, cancelled, frozenset(chosen), sizes
        )
        before = sum(read for read, _wrote in sizes)
        after = sum(wrote for _read, wrote in sizes)
        # 小さくならない（細かい模様や雑音の多い絵は H.264 でも縮まない）なら、画質を落とすだけ
        # なので使わない
        smaller = written and after < before * _REENCODE_WORTH
        report = _close_to_source(source, target, merged, chosen, cancelled) if smaller else None
        if report is not None:
            report = replace(report, before=before, after=after)
    except Cancelled:
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
        raise
    except (av.error.FFmpegError, OSError, ValueError, ProbeError):
        report = None
    if report is None:
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
    return report


#: 区間の中で差を測る所（区間の長さの割合） 頭だけだと、時刻が 1 枚ずれていても気付けない
_PSNR_POINTS = (Fraction(0), Fraction(1, 2), Fraction(9, 10))


def frame_psnr(old: Any, new: Any) -> float:
    """2 枚の絵の差（PSNR dB） 同じなら無限大

    YUV の面ごとに素材の深さのまま比べ、画素の数で重みを付けて平均する（FFmpeg の ``psnr``
    の average と同じ数え方） RGB へ直してから比べると、色の変換の丸めと色差の広げ方で差が
    何倍にも見え（試験の素材で YUV 53 dB が RGB 36 dB）、符号そのものの差が分からない
    """
    if (old.width, old.height) != (new.width, new.height):
        return 0.0
    if old.format.name != new.format.name:
        # 詰め方だけが違う形（無圧縮の uyvy422 と yuv422p など）は、新しい形へ並べ直して比べる
        old = old.reformat(format=new.format.name)
    deep = max(component.bits for component in old.format.components) > 8
    kind = np.uint16 if deep else np.uint8
    peak = float((1 << max(component.bits for component in old.format.components)) - 1)
    squared = 0.0
    count = 0
    for one, two in zip(old.planes, new.planes, strict=True):
        size = np.dtype(kind).itemsize
        a = np.frombuffer(one, kind).reshape(one.height, one.line_size // size)[:, : one.width]
        b = np.frombuffer(two, kind).reshape(two.height, two.line_size // size)[:, : two.width]
        squared += float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        count += a.size
    if squared == 0 or count == 0:
        return math.inf
    return 10 * math.log10(peak * peak / (squared / count))


def _stamps(
    path: Path, index: int, origin: Fraction, span: Span, cancelled: Callable[[], bool]
) -> list[int]:
    """区間の中の絵の時刻（道の時間の単位） パケットだけを読む（解かないので速い）"""
    found: list[int] = []
    with av.open(str(path)) as reader:
        stream = reader.streams[index]
        base = as_fraction(stream.time_base)
        reader.seek(max(0, int((origin + span.start) / base)), stream=stream, backward=True)
        for packet in reader.demux(stream):
            if cancelled():
                raise Cancelled
            if packet.pts is None:
                continue
            seconds = packet.pts * base
            if seconds > origin + span.end:
                break
            if seconds >= origin + span.start:
                found.append(packet.pts)
    return sorted(found)


def _frame_at(
    path: Path, index: int, moment: Fraction, cancelled: Callable[[], bool]
) -> Any | None:
    """``moment``（道の頭からでなく入れ物の時刻の秒）に映る絵 その時刻以前で最後の絵"""
    with av.open(str(path)) as reader:
        stream = reader.streams[index]
        base = as_fraction(stream.time_base)
        reader.seek(max(0, int(moment / base)), stream=stream, backward=True)
        shown = None
        for frame in reader.decode(stream):
            if cancelled():
                raise Cancelled
            if frame.pts is None:
                continue
            if frame.pts * base > moment:
                break
            shown = frame
        return shown


def _picture_end(item: MediaItem, index: int) -> Fraction:
    """映像の道の終わり（秒 素材の頭から） 分からなければ素材全体の長さ"""
    for stream in item.video_streams:
        if stream.index == index and stream.end_time is not None:
            return stream.end_time
    return item.duration


def _within(spans: list[Span], end: Fraction) -> list[Span]:
    """差を測る所を決める区間 素材の終わりより後ろを切る（書く区間は変えない）

    全体を入れるときの区間は素材より長い（10 億秒） そのまま測る所を決めると、真ん中も終わりも
    素材の後ろを指して最後の絵どうしを比べ、途中の崩れに気付けない
    """
    if end <= 0:
        return spans
    return [Span(span.start, min(span.end, end)) for span in spans if span.start < end] or spans


def _close_to_source(
    source: Path,
    target: Path,
    spans: list[Span],
    chosen: dict[int, str],
    cancelled: Callable[[], bool],
) -> Reencoded | None:
    """符号にし直した素材が、使う区間で元とほぼ同じ絵を同じ時刻に出すか 出すなら結果"""
    try:
        before = probe_media(source)
        after = probe_media(target)
        with av.open(str(source)) as a, av.open(str(target)) as b:
            if media_origin(a) != media_origin(b):
                return None
    except (ProbeError, av.error.FFmpegError, OSError):
        return None
    if [s.index for s in before.video_streams] != [s.index for s in after.video_streams]:
        return None
    if [(s.index, s.codec, s.channels) for s in before.audio_streams] != [
        (s.index, s.codec, s.channels) for s in after.audio_streams
    ]:
        return None
    pixel_format = ""
    for old, new in zip(before.video_streams, after.video_streams, strict=True):
        if (old.rotation, old.color_transfer, old.color_primaries, old.width, old.height) != (
            new.rotation,
            new.color_transfer,
            new.color_primaries,
            new.width,
            new.height,
        ):
            return None
        if old.index in chosen:
            pixel_format = new.pixel_format
        elif (old.codec, old.pixel_format) != (new.codec, new.pixel_format):
            return None
        elif not _same_pictures(source, target, old.index, spans, cancelled):
            # 写しただけの映像の道も、キーフレームが区間の頭より後ろにあると頭の絵が欠ける
            # 写す側（_same_as_source）と同じ確かめを通さないと、欠けたまま「し直した」と知らせる
            return None
    worst = math.inf
    try:
        origin = Fraction(0)
        with av.open(str(source)) as reader:
            origin = media_origin(reader)
        for index in chosen:
            for span in spans:
                if _stamps(source, index, origin, span, cancelled) != _stamps(
                    target, index, origin, span, cancelled
                ):
                    # 区間の中の絵の時刻が 1 つでも違えば、作品の時刻が別の絵を指す・絵が抜ける
                    return None
            for span in _within(spans, _picture_end(before, index)):
                for point in _PSNR_POINTS:
                    moment = origin + span.start + (span.end - span.start) * point
                    shown = _frame_at(source, index, moment, cancelled)
                    made = _frame_at(target, index, moment, cancelled)
                    if shown is None and made is None:
                        continue
                    if shown is None or made is None or shown.pts != made.pts:
                        return None
                    worst = min(worst, frame_psnr(shown, made))
    except (av.error.FFmpegError, OSError, ValueError):
        return None
    if worst < REENCODE_LEAST_PSNR:
        return None
    for audio in before.audio_streams:
        if not _same_sound(
            source, target, audio.index, audio.sample_rate, audio.channels, spans, cancelled
        ):
            return None
    first = next(iter(chosen.values()))
    return Reencoded(codec=first, to="h264", pixel_format=pixel_format, psnr=worst)
