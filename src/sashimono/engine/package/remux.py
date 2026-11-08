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

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import av.error
import numpy as np

from sashimono.engine.decode.audio import AudioDecoder
from sashimono.engine.decode.probe import ProbeError, media_origin, moving_pictures, probe_media
from sashimono.engine.decode.rational import as_fraction
from sashimono.engine.decode.video import VideoDecoder

__all__ = [
    "Cancelled",
    "Span",
    "cut_media",
    "output_format",
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
) -> bool:
    with av.open(str(source)) as reader:
        kept = _kept_streams(reader)
        if kept is None:
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
            for stream in kept:
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
                packet.stream = mapping[index]
                writer.mux(packet)

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
                put(packet)
                anchored.add(index)
                if len(anchored) == len(kept):
                    break

            for span in spans:
                _copy_span(reader, kept, guide, picture_indices, origin, span, put, cancelled)
    return True


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
    with VideoDecoder(source, index) as original, VideoDecoder(target, index) as copied:
        for span in spans:
            if cancelled():
                raise Cancelled
            moment = max(Fraction(0), span.start)
            old_frame = original.frame_at(moment)
            new_frame = copied.frame_at(moment)
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
