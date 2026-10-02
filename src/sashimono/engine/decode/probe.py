"""素材を開いて :class:`~sashimono.core.model.MediaItem` を組み立てる"""

from __future__ import annotations

import hashlib
import itertools
import threading
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import av.error

from sashimono.core.model import AudioStreamInfo, MediaItem, VideoStreamInfo
from sashimono.core.timebase import FrameRate
from sashimono.engine.colorspace import color_tags
from sashimono.engine.decode.rational import as_fraction

__all__ = [
    "PROBE_CACHE_SIZE",
    "ROTATION_PACKET_LIMIT",
    "ProbeError",
    "clear_probe_cache",
    "forget_probe",
    "media_origin",
    "moving_pictures",
    "probe_media",
]

#: 静止画として扱う拡張子 長さを持たず、タイムライン上で任意に伸ばせる
STILL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})

#: フレームレートが取れなかったときの既定値 静止画や壊れたヘッダで起きる
FALLBACK_FRAME_RATE = FrameRate(30)


class ProbeError(Exception):
    """素材を開けない、または中身を解釈できない"""


#: 回転を読むために頭の 1 枚を待つ間に読むパケットの上限 B フレームやスレッドの遅れで
#: 最初の数十パケットは絵が出ないことがあるので、それより十分多く取る
ROTATION_PACKET_LIMIT = 256

#: 調べた結果を覚えておく素材の数 1 本あたり数百バイトなので、多めに持っても軽い
PROBE_CACHE_SIZE = 512

_Facts = tuple[Fraction, tuple[VideoStreamInfo, ...], tuple[AudioStreamInfo, ...]]


def probe_media(path: Path) -> MediaItem:
    """ファイルを解析して素材情報を返す

    映像・音声の各ストリームを個別に記録する 多言語音声や 5.1ch の素材では
    音声が複数本あり、読み込み時にそれぞれ別トラックへ展開できるようにするため

    同じファイル（:func:`_identity` が同じ）を 2 度目からは開かずに答える
    映像と音声のデコーダは作るたびにここを呼ぶ（再生やシークのたび、控えや字幕起こしも）
    毎回開くと、そのたびに素材を開いて頭の 1 枚を復号する 素材 ID は呼ぶたびに新しく作る
    読み込みのたびに別の素材として登録するため

    覚えた結果を使い回すのは、読み込んだ後の素材を開き直す所（デコーダ・控え・字幕起こし）
    のためと割り切る 中身の印は、同じ大きさのまま真ん中だけを書き換えて時刻を戻した物を
    見分けられない（見分けるには全体を読むことになり、覚える意味が無くなる） そこで
    素材を読み込む操作（ファイルの読み込み・ドロップ・テンプレートや .exo の素材・AI の
    読み込み）は :func:`forget_probe` で覚えた結果を捨ててから呼ぶ 読み込み直せば必ず
    開き直す

    同じ素材を同時に何本ものスレッドから頼まれたら、開くのは 1 回で、ほかは待って同じ
    結果を受け取る（読み込みは 4 本のスレッドで調べる）
    """
    path = Path(path)
    duration, video_streams, audio_streams = _facts(path, _identity(path))
    return MediaItem(
        path=path,
        duration=duration,
        video_streams=video_streams,
        audio_streams=audio_streams,
    )


def forget_probe(path: Path) -> None:
    """``path`` について覚えている結果を捨てる 素材を読み込む操作の前に呼ぶ"""
    target = Path(path)
    with _lock:
        for key in [key for key in _cache if key[0] == target]:
            del _cache[key]
        # 調べている最中の物があれば世代を進める 古い調べは終わっても覚えさせず、これから
        # 読み込む側もそれを待たない 待つと、捨てる前に始まった調べの結果を受け取り、
        # その結果が捨てた後に覚え直される 調べている物が無ければ、覚え直される物も
        # 待たされる物も無いので世代は要らない
        if any(key[0] == target for key in _pending):
            _generation[target] = _generation.get(target, 0) + 1


def clear_probe_cache() -> None:
    """覚えている調べた結果を捨てる 試験で開いた回数を数えるときのため"""
    with _lock:
        _cache.clear()


#: 覚えた結果の鍵 場所・中身の印
_Key = tuple[Path, tuple[int, ...]]
#: 調べている最中の印の鍵 場所・中身の印・世代（:func:`forget_probe` で進む）
_PendingKey = tuple[Path, tuple[int, ...], int]

#: 調べた結果 古い物から捨てる
_cache: OrderedDict[_Key, _Facts] = OrderedDict()
#: いま調べている素材 同じ素材を頼んだほかのスレッドは、これが終わるのを待つ
_pending: dict[_PendingKey, threading.Event] = {}
#: 調べている最中の物がある場所だけの世代 無い場所は 0 とみなす
#: 調べ終わってその場所に調べている物が無くなれば消すので、扱った場所の数だけ増え続けない
_generation: dict[Path, int] = {}
_lock = threading.Lock()


def _facts(path: Path, identity: tuple[int, ...]) -> _Facts:
    """素材の長さとストリーム 覚えていればそれを、無ければ開いて調べて覚える

    ``functools.lru_cache`` は同じ鍵の同時の呼び出しを待ち合わせず、読み込みの 4 本の
    スレッドが同じ素材を 2 重 3 重に開く 調べている間は印を立て、ほかは待つ
    開けなかった素材は覚えない 待っていた側は自分で開き直して、同じ理由の例外を受け取る
    調べている最中の印には世代を入れる :func:`forget_probe` の後に頼んだ側は、捨てる前に
    始まった調べを待たずに自分で開き、捨てる前に始まった調べの結果は覚えない
    """
    key = (path, identity)
    while True:
        with _lock:
            found = _cache.get(key)
            if found is not None:
                _cache.move_to_end(key)
                return found
            mine = (path, identity, _generation.get(path, 0))
            waiting = _pending.get(mine)
            if waiting is None:
                done = threading.Event()
                _pending[mine] = done
                break
        waiting.wait()
    try:
        facts = _read_facts(path)
    except BaseException:
        with _lock:
            _finish(mine)
        done.set()
        raise
    with _lock:
        # 調べている間に捨てられていたら覚えない 読み込み直した側の結果だけを残す
        if _generation.get(path, 0) == mine[2]:
            _cache[key] = facts
            while len(_cache) > PROBE_CACHE_SIZE:
                _cache.popitem(last=False)
        _finish(mine)
    done.set()
    return facts


def _finish(mine: _PendingKey) -> None:
    """調べ終わった印を外す その場所に調べている物が残っていなければ世代も消す

    ``_lock`` を持って呼ぶ 世代は、捨てる前に始まった調べを見分けるためだけの物で、
    調べている物が無くなれば見分ける相手がいない 消して 0 へ戻しても、覚えた結果の鍵に
    世代は入っていないので、覚えた結果は外れない
    """
    del _pending[mine]
    path = mine[0]
    if not any(key[0] == path for key in _pending):
        _generation.pop(path, None)


#: 覚えた結果を使ってよいかを見るために読む、ファイルの頭と尻の長さ（バイト）
#: 素材のヘッダ（mp4 の moov が頭か尻にある・wav の fmt）はたいていこの中に入る
_FINGERPRINT_BYTES = 64 * 1024


def _identity(path: Path) -> tuple[int, ...]:
    """覚えた結果を使ってよいかを決める印 中身が変わっていれば変わる

    更新時刻と大きさだけだと、同じ大きさの別の素材で上書きして更新時刻を戻した物
    （時刻を保つ写し方・展開）を見分けられず、前の長さや解像度のまま置いてしまう
    ファイルの番号（名前を替えて差し替えると変わる）・``st_ctime_ns``（POSIX では状態の
    変わった時刻で、時刻を戻しても変わる Windows の Python 3.12 と 3.13 では作った時刻で、
    上書きでは変わらない）・頭と尻の中身の要約も入れる 読むのは 128KB だけで、素材を
    開いて頭の 1 枚を復号するより軽い 真ん中だけの書き換えはこれでも見分けられないので、
    読み込む操作では覚えた結果を捨てる（:func:`probe_media`）
    """
    try:
        stat = path.stat()
        digest = hashlib.blake2b(digest_size=16)
        with path.open("rb") as handle:
            digest.update(handle.read(_FINGERPRINT_BYTES))
            if stat.st_size > _FINGERPRINT_BYTES:
                handle.seek(max(_FINGERPRINT_BYTES, stat.st_size - _FINGERPRINT_BYTES))
                digest.update(handle.read(_FINGERPRINT_BYTES))
    except OSError:
        raise ProbeError(f"ファイルが見つからない: {path}") from None
    return (
        stat.st_mtime_ns,
        stat.st_size,
        stat.st_ino,
        stat.st_ctime_ns,
        int.from_bytes(digest.digest(), "big"),
    )


def _read_facts(path: Path) -> _Facts:
    """素材を開いて、長さとストリームを調べる"""
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as exc:
        raise ProbeError(f"素材を開けない: {path} ({exc})") from exc

    with container:
        is_still = path.suffix.lower() in STILL_SUFFIXES
        origin = media_origin(container)
        pictures = moving_pictures(container)
        audio_streams = tuple(_audio_info(stream) for stream in container.streams.audio)
        if not pictures and not audio_streams:
            raise ProbeError(f"映像も音声も含まれていない: {path}")
        duration = Fraction(0) if is_still else _container_duration(container, origin)
        # 回転は最後に読む 頭の 1 枚を復号するので、ほかの値を読む前に進めない
        rotations = [0] * len(pictures) if is_still else _rotations(path, container, pictures)
        video_streams = tuple(
            _video_info(stream, rotation, origin)
            for stream, rotation in zip(pictures, rotations, strict=True)
        )

    return duration, video_streams, audio_streams


def _rotations(
    path: Path, container: av.container.InputContainer, pictures: list[av.VideoStream]
) -> list[int]:
    """映像ストリームごとの回転 表示行列はストリームごとに持つ

    1 本目の回転をほかへ写すと、向きの違う 2 本目（別のカメラの角度など）が横倒しになる
    2 本目からは開き直して頭から読む 1 本目の頭の 1 枚を読んだ所から続けると、
    2 本目の頭のパケットを読み飛ばしている
    """
    if not pictures:
        return []
    rotations = [_display_rotation(container, pictures[0])]
    for stream in pictures[1:]:
        try:
            with av.open(str(path)) as again:
                same = cast(av.VideoStream, again.streams[stream.index])
                rotations.append(_display_rotation(again, same))
        except (av.error.FFmpegError, OSError, IndexError):
            rotations.append(0)
    return rotations


def moving_pictures(container: av.container.InputContainer) -> list[av.VideoStream]:
    """映像として読む映像ストリーム カバー画像（``attached_pic``）は除く

    mp3 や m4a に付いたジャケットの絵は、ffmpeg では映像ストリームとして見える
    数えると音楽の素材が動画として扱われ、映像トラックへ置かれて描かれる
    絵は 1 枚しか無いので、デコーダが時刻でシークすると途中から PermissionError で落ちる
    静止画のファイル（png など）の絵は印が付いていないので、ここでは残る
    """
    return [
        stream
        for stream in container.streams.video
        if not stream.disposition & av.stream.Disposition.attached_pic
    ]


def media_origin(container: av.container.InputContainer) -> Fraction:
    """素材の時刻の原点（秒 PTS の数え方） 素材の中の時刻は、PTS からこれを引いて数える

    分割して書き出した物や放送の録画（MPEG-TS）は、最初のフレームの PTS が 0 より後ろにある
    PTS そのままで数えると、置いたクリップ（``source_in`` 0・長さは素材の長さ）が読む時刻が
    すべて最初のフレームより前になり、頭の絵が止まったまま音も鳴らない（Issue #123）
    クリップの ``source_in``・``hold_at``、YMM4 の ``ContentOffset`` は素材の頭から数えるので、
    こちらも頭から数える（YMM4 は本体で測った docs/development.md「素材の中の時刻は頭から数える」）

    原点は素材 1 本に 1 つ 映像と音声で別々に引くと、素材の中の映像と音の食い違い
    （頭の音が映像より早く始まる、など）が消えて音がずれる

    映像があれば最初の映像ストリームの頭にする コンテナの頭（全ストリームの最小）にしないのは、
    AAC の前置き（プライミング）が映像より 20ms ほど早く始まる素材が多いため コンテナの頭を
    原点にすると、映像のフレームの時刻がすべて前置きの分だけ後ろへずれ、フレームの境目の時刻で
    1 つ前の絵が出る 前置きは復号すると捨てる区間なので、原点より前（負の時刻）へ出してよい
    YMM4 も、音が映像より 0.52 秒早く始まる素材で ``ContentOffset`` 0 の頭に映像の最初の
    フレームを出した（映像の頭が原点）
    映像の無い素材は音声の頭の最小、どちらも分からなければコンテナの頭

    負の頭（B フレームの並べ替えで最初の PTS が負になる素材など）は 0 にする この修正の前から
    0 を原点として正しく映っていた素材で、引くと絵と音がずれ動く
    秒は分数で持つ コンテナの頭（マイクロ秒に丸めてある）から作ると、フレームの時刻が
    ちょうどの境目から 1µs 未満ずれ、境目の時刻で 1 つ前の絵が出る
    """
    for stream in moving_pictures(container)[:1]:
        start = _stream_start(stream)
        if start is not None:
            return max(Fraction(0), start)
    starts = [_stream_start(stream) for stream in container.streams.audio]
    known = [start for start in starts if start is not None]
    if known:
        return max(Fraction(0), min(known))
    if container.start_time is not None:
        return max(Fraction(0), Fraction(container.start_time, av.time_base))
    return Fraction(0)


def _stream_start(stream: av.stream.Stream) -> Fraction | None:
    """ストリームの頭の時刻（秒 PTS の数え方） 分からなければ ``None``"""
    if stream.start_time is None or stream.time_base is None:
        return None
    return stream.start_time * as_fraction(stream.time_base)


def _container_duration(container: av.container.InputContainer, origin: Fraction) -> Fraction:
    """素材全体の長さ（秒） 原点（:func:`media_origin`）から終わりまで

    コンテナの長さを優先する ストリームごとの長さは映像と音声で食い違うことがあり、
    どちらを採るかで末尾が欠けたり余ったりするため
    コンテナの長さは全ストリームの最小の頭から数えてある 映像より早く始まる音の前置きを
    含んだままにすると、置いたクリップが 1 フレーム長くなり、最後の 1 フレームは映像の
    終わりの後なので何も映らない 原点より前の区間は長さから除く 頭が 0 以下の素材は
    もとから原点が 0 なので、長さは変わらない
    """
    streams = (*moving_pictures(container), *container.streams.audio)
    if container.duration is not None:
        whole = Fraction(container.duration, av.time_base)
        if container.start_time is not None:
            start: Fraction | None = Fraction(container.start_time, av.time_base)
        else:
            # コンテナが頭を書いていなければ、道の頭の最小で代える どちらも無いときは
            # 長さの数え始めが分からないので原点を引かない 引くと長さが 0 まで縮み、
            # 素材を置いてもクリップができない
            known = [start for start in map(_stream_start, streams) if start is not None]
            start = min(known) if known else None
        return max(Fraction(0), _relative_end(start, whole, origin))

    longest = Fraction(0)
    for stream in streams:
        if stream.duration is not None and stream.time_base is not None:
            length = Fraction(stream.duration) * as_fraction(stream.time_base)
            longest = max(longest, _relative_end(_stream_start(stream), length, origin))
    return longest


def _relative_end(start: Fraction | None, length: Fraction, origin: Fraction) -> Fraction:
    """頭が ``start`` で長さ ``length`` の区間の終わりを、原点から数えた時刻にする

    頭を 0 に丸めてから原点を引くと、負の頭から始まる音（前置き）と正の映像の頭を
    持つ素材で、負の区間の分だけ長くなり、終わりに何も無い区間ができる
    原点が 0 の素材（頭が 0 以下）は、この修正の前と同じく負の頭を 0 に丸める
    丸めないと、B フレームの並べ替えで頭が負になっていた素材の長さがすべて縮む
    頭が分からないときは原点から始まるものと見る 0 から始まるものと見ると、
    原点の分だけ短くなる
    """
    if origin <= 0:
        return max(Fraction(0), start or Fraction(0)) + length
    head = start if start is not None else origin
    return head + length - origin


def _video_info(
    stream: av.video.stream.VideoStream, rotation: int, origin: Fraction
) -> VideoStreamInfo:
    rate = stream.average_rate or stream.guessed_rate or stream.base_rate
    frame_rate = FrameRate(rate.numerator, rate.denominator) if rate else FALLBACK_FRAME_RATE
    transfer, primaries = color_tags(stream)
    return VideoStreamInfo(
        color_transfer=transfer,
        color_primaries=primaries,
        index=stream.index,
        width=stream.codec_context.width,
        height=stream.codec_context.height,
        frame_rate=frame_rate,
        time_base=as_fraction(stream.time_base) if stream.time_base else Fraction(1, 1000),
        codec=stream.codec_context.name,
        pixel_format=stream.format.name if stream.format else "",
        rotation=rotation,
        end_time=_stream_end(stream, origin),
    )


def _stream_end(stream: av.video.stream.VideoStream, origin: Fraction) -> Fraction | None:
    """映像の道の終わりの時刻（秒） 道が長さを書いていなければ ``None``

    デコーダと同じく素材の原点（:func:`media_origin`）から数える 道の長さは道の頭から
    数えてあるので、道の頭を足してから原点を引く 映像より早く始まる音がある素材では、
    道の長さそのものとは道の頭と原点の差の分だけ違う（映像の原点では差は 0）

    道の頭が分からないときは、原点から始まるものと見て道の長さを終わりにする
    （:func:`_relative_end` と同じ見方） 0 から始まるものと見て原点を引くと、原点の分だけ
    終わりが早まり、最後のフレームより前から絵が出なくなる 終わりを捨てて素材の長さで
    見ると、音の方が長い素材で映像の後も音が続く間ずっと最後の絵が残る
    """
    if stream.duration is None or stream.time_base is None:
        return None
    length = Fraction(stream.duration) * as_fraction(stream.time_base)
    if stream.start_time is None:
        return length if length > 0 else None
    end = Fraction(stream.start_time) * as_fraction(stream.time_base) + length - origin
    return end if end > 0 else None


def _audio_info(stream: av.audio.stream.AudioStream) -> AudioStreamInfo:
    context = stream.codec_context
    return AudioStreamInfo(
        index=stream.index,
        sample_rate=context.sample_rate,
        channels=context.layout.nb_channels,
        time_base=as_fraction(stream.time_base) if stream.time_base else Fraction(1, 48000),
        codec=context.name,
        language=stream.metadata.get("language") or None,
    )


def _display_rotation(container: av.container.InputContainer, stream: av.VideoStream) -> int:
    """表示時に適用すべき時計回りの回転角を返す

    スマホの縦撮り素材は回転情報（表示行列）を持つのが普通で、無視すると横倒しで表示される
    PyAV 18 はストリームの表示行列を読む口を持たない（書く口だけ）が、FFmpeg はストリームの
    表示行列を復号した絵へ写すので、頭の 1 枚を復号してその絵の回転を読む

    前は ffprobe を別のプログラムとして起こして読んでいた 窓を持たない配布版では、素材を
    調べるたび（読み込み・再生・控え・字幕起こし）に黒い窓が一瞬出た（ffprobe が PATH に
    無い機械では回転を読めず、縦撮りが横倒しのままだった） 頭の 1 枚の復号は 1080p で 5ms、
    4K の H.265 で 30ms ほどで、ffprobe を起こす 35〜45ms より速い

    取得できない場合は 0 を返し、素材の読み込み自体は続行する 読むパケットには上限
    （:data:`ROTATION_PACKET_LIMIT`）を置く 絵を 1 枚も出さない映像（壊れた道・復号できない
    符号）で頭の 1 枚を待ち続けると、ファイルの終わりまで読み、大きな素材では読み込みや
    再生の始まりが止まる（前の ffprobe には 15 秒の上限があった）
    """
    frame: av.VideoFrame | None = None
    try:
        # 上限の数だけ取り出す 数えてから止めると、上限の次の 1 つまで読んでしまう
        for packet in itertools.islice(container.demux(stream), ROTATION_PACKET_LIMIT):
            frames = packet.decode()
            if frames:
                frame = frames[0]
                break
    except (av.error.FFmpegError, OSError, ValueError):
        return 0
    if frame is None:
        return 0
    # 絵の回転は ffprobe と同じく反時計回りの角度 表示時に必要なのは時計回りなので反転する
    clockwise = round(-float(frame.rotation)) % 360
    return clockwise if clockwise in (0, 90, 180, 270) else 0
