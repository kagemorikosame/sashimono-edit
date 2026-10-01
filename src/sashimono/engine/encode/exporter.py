"""タイムラインをファイルへ書き出す

プレビューと同じ :class:`~sashimono.engine.render.FrameRenderer` と
:class:`~sashimono.engine.audio.AudioMixer` を使う 別経路にすると
「プレビューでは出るのに書き出すと出ない」が起きる
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import cast

import av
import av.audio.fifo
import av.audio.frame
import av.audio.stream
import av.error
import av.video.codeccontext
import av.video.frame
import av.video.stream
import numpy as np

from sashimono.core.model import Project, TrackKind
from sashimono.engine.audio import AudioMixer
from sashimono.engine.colorspace import VideoReformatter, tag_bt709, to_bt709
from sashimono.engine.gpu import OffscreenGLContext
from sashimono.engine.render import (
    DEFAULT_DECODE_THREADS,
    FULL_QUALITY,
    MAX_DECODE_THREADS,
    FrameRenderer,
)

__all__ = [
    "COLOR_OPTIONS",
    "DEFAULT_PIPELINE_DEPTH",
    "MAX_DECODE_THREADS",
    "MAX_PIPELINE_DEPTH",
    "MEASURED_DECODE_MS",
    "MEASURED_EXPORT_MS",
    "MEASURED_EXPORT_TOTAL_MS",
    "ExportError",
    "ExportSettings",
    "available_video_codecs",
    "export_project",
]

#: 合成の先へ何枚ぶん進めてよいか（0 で直列）
#: 1 枚は画面 1 枚ぶんの RGBA（1080p で 8MB、4K で 33MB） 深くしても、
#: 合成と書き込みのどちらか遅い方より速くはならない 2 枚あれば片方の揺れを吸える
DEFAULT_PIPELINE_DEPTH = 2
#: 設定で選べる上限 これ以上はメモリを食うだけで速くならない
MAX_PIPELINE_DEPTH = 8

#: 1920x1080 を 3 枚重ねて 60 枚書き出したときの 1 枚あたりの内訳（ミリ秒）
#: 合成 / 読み戻し / 色変換 / エンコード + mux の順（NVENC・NVIDIA GPU）
#: 前の 2 つは GPU の側で、後ろの 2 つが別スレッドへ逃がせる分
#: 測り直すときは tools\bench_export.py
MEASURED_EXPORT_MS = (11.2, 2.7, 6.0, 0.7)

#: 同じ素材を書き出し切ったときの 1 枚あたりの実測（ミリ秒） 1 枚ずつ / 2 枚先まで
#: 内訳の差（6.7ms）ほど縮まないのは、色変換の前半（画素を PyAV へ写す所）が
#: GIL を握ったままで、合成の側の Python の処理と取り合うため
MEASURED_EXPORT_TOTAL_MS = (31.4, 28.5)

#: 同じ素材の合成（デコードを含む GPU 合成）1 枚あたりの実測（ミリ秒）
#: 並べない（1 本ずつ）/ 4 本まで並べる の順 1920x1080 を 3 枚重ね
#: PyAV のデコードは GIL を解放するので、別の素材どうしなら本当に重なる
#: 測り直すときは tools\bench_export.py --decode-threads 1 と付けない場合を比べる
MEASURED_DECODE_MS = (15.9, 11.2)

#: 優先順に並べた映像コーデック 前にあるものから、使えるものを選ぶ
#: NVENC は CPU をほとんど使わないので、長尺でも編集を続けながら書き出せる
VIDEO_CODEC_PREFERENCE = ("h264_nvenc", "h264_qsv", "libx264")

_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}

#: 色のタグを決めるコーデックのオプション名 :attr:`ExportSettings.options` には入れられない
COLOR_OPTIONS = frozenset({"color_primaries", "color_trc", "colorspace", "color_range"})

#: エンコーダ自身の設定をまとめて渡すオプション この中にも色の指定を書ける
_ENCODER_PARAM_OPTIONS = frozenset({"x264-params", "x264opts", "x265-params"})
#: まとめ書きの中で色のタグを決める名前（x264 と x265 で共通）
_ENCODER_COLOR_PARAMS = frozenset(
    {"colorprim", "transfer", "colormatrix", "range", "fullrange", "input-range"}
)


class ExportError(RuntimeError):
    """書き出しを開始できない、または途中で失敗した"""


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """書き出しの設定"""

    path: Path
    #: ``None`` なら :func:`available_video_codecs` の先頭を使う
    video_codec: str | None = None
    #: 映像のビットレート（bps） ``None`` ならコーデックの既定に任せる
    video_bitrate: int | None = 12_000_000
    audio_codec: str = "aac"
    audio_bitrate: int = 192_000
    pixel_format: str = "yuv420p"
    #: 書き出すフレーム範囲 ``None`` なら全体
    frame_range: tuple[int, int] | None = None
    #: コーデックへ渡す追加オプション プリセットや品質指定を通す口
    #: 色のタグ（:data:`COLOR_OPTIONS`）は BT.709 に固定しているので受け付けない
    options: dict[str, str] = field(default_factory=dict)
    #: GPU の合成を、色変換・エンコード・mux の何枚ぶん先へ進めてよいか
    #: 0 なら 1 枚ずつ直列に処理する（スレッドを使わない）
    pipeline_depth: int = DEFAULT_PIPELINE_DEPTH
    #: 重ねたレイヤーのデコードを、いくつまで同時に走らせてよいか 1 なら並べない
    decode_threads: int = DEFAULT_DECODE_THREADS
    #: 前の音を読む音の効果の動く値を、区切りの中でもつなぐ（:class:`AudioMixer`）
    #: 再生と同じ音にするため、本人の設定（プレビューと同じ値）を渡す
    smooth_history: bool = True


#: 開けるかを試すときの大きさ NVENC は小さすぎる画を断る（64x64 では開けない）ので、
#: 書き出しでよく使う大きさに近い 16:9 で試す 小さくすると、使える NVENC まで外れる
_PROBE_SIZE = (640, 360)


def available_video_codecs() -> list[str]:
    """この環境で使える映像コーデックを、優先順に返す

    入っているだけでは数えない 実際に開けたものだけを返す（#67）
    QSV は FFmpeg に組み込まれていても、Intel の GPU や駆動が無い機械では開けない
    入っているかだけで選ぶと、開けない QSV が既定になって書き出しが失敗する
    """
    return [name for name in VIDEO_CODEC_PREFERENCE if _opens(name)]


@cache
def _opens(name: str) -> bool:
    """``name`` のエンコーダを開けるか 1 つにつき 1 度だけ試して覚える

    NVENC を開くのは 1 回 100 ms ほどかかる 書き出しの画面を開くたびに試すと、
    そのたびに待たされる 機械の GPU は起動中に変わらないので、覚えておいてよい
    """
    try:
        _open_encoder(name)
    except Exception:
        # 入っていない（UnknownCodecError）・開けない（ArgumentError など）の種類は
        # 環境によって違う 名前で拾うと、知らない種類の失敗で候補の列挙ごと落ちる
        return False
    return True


def _open_encoder(name: str) -> None:
    """``name`` のエンコーダを yuv420p で開いてみる 開けなければ投げる

    画素形式は :attr:`ExportSettings.pixel_format` の既定と同じ yuv420p に固定する
    答えはコーデックごとに 1 つだけ覚えるので、書き出しごとの設定には追従できない
    別の画素形式で開けないときは、コーデックを指定しない書き出しなら始めた時点で次の候補へ移る
    """
    # create は種類の union を返す 映像の属性を触るので、ここで型を確定させる
    context = cast("av.video.codeccontext.VideoCodecContext", av.CodecContext.create(name, "w"))
    context.width, context.height = _PROBE_SIZE
    context.pix_fmt = "yuv420p"
    context.time_base = Fraction(1, 30)
    # 開いた文脈は捨てるだけでよい PyAV は参照が切れたときに閉じる
    context.open()


class _EncoderOpenError(ExportError):
    """エンコーダを開けなかった まだ 1 コマも書いていないので、別のコーデックでやり直せる"""


def _color_options_in(options: dict[str, str]) -> list[str]:
    """``options`` のうち、色のタグを変えてしまう指定の名前

    x264 / x265 はまとめ書きのオプション（``x264-params`` など）の中でも色を指定できる
    こちらはビットストリームの VUI だけを書き換え、MP4 の colr は BT.709 のまま残るので、
    コンテナとビットストリームでタグが食い違ったファイルになる（再生側によって読み方が変わる）
    """
    found = sorted(COLOR_OPTIONS & options.keys())
    for name in sorted(_ENCODER_PARAM_OPTIONS & options.keys()):
        # 項目の区切りは ``:`` 、名前と値は ``=`` x264opts だけは ``,`` で区切った書き方も通る
        for item in options[name].replace(",", ":").split(":"):
            key = item.split("=", 1)[0].strip().lower()
            if key in _ENCODER_COLOR_PARAMS:
                found.append(f"{name} の {key}")
    return found


def export_project(
    project: Project,
    settings: ExportSettings,
    *,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """プロジェクトを 1 本の動画ファイルへ書き出す

    ``should_cancel`` が真を返したら、書きかけのファイルを消して
    :class:`ExportError` を投げる 中途半端なファイルを残すと、書き出せたのか
    どうかが分からなくなる
    """
    start, end = settings.frame_range or (0, project.duration)
    total = end - start
    if total <= 0:
        raise ExportError("書き出す範囲が空")

    # 指定が無ければ、開けた候補を前から試す 試しに開けても、実際の大きさでは
    # 断られることがある（NVENC の H.264 は幅 4096 を超える画を開けない）
    # そのときに書き出しごと失敗させず、最後は libx264 まで落とす
    codecs = [settings.video_codec] if settings.video_codec else available_video_codecs()
    if not codecs:
        raise ExportError("使える映像コーデックが見つからない")

    # 色のタグは options で上書きできてしまう（コーデックを開くときに options が後から効く）
    # 画素は必ず BT.709 / limited で変換するので、別のタグを通すと中身と食い違う
    # 黙って捨てると頼んだ指定が効かない理由が分からないので、始める前に断る
    conflicting = _color_options_in(settings.options)
    if conflicting:
        raise ExportError(
            f"色のタグは BT.709 に固定している オプションでは変えられない: {', '.join(conflicting)}"
        )

    # 深さは合成済みの絵を何枚抱えるかで、4K なら 1 枚 33MB 大きい値をそのまま通すと
    # 長い書き出しの途中でメモリを使い切る 設定画面の側は範囲を見ているので、
    # ここを素通しにすると API から直に呼んだときだけ守られない形になる
    # 整数かどうかも見る 2.5 のような値は範囲の確認だけなら通ってしまい、
    # 席やキューの数として半端な値が奥まで流れる（失敗しても ExportError にならない）
    depth = settings.pipeline_depth
    if (
        not isinstance(depth, int)
        or isinstance(depth, bool)
        or not 0 <= depth <= MAX_PIPELINE_DEPTH
    ):
        raise ExportError(f"先読みの深さは 0 から {MAX_PIPELINE_DEPTH} までの整数: {depth!r}")

    # デコードのスレッド数も同じ理由で見る 0 や負の数を素通しにすると、スレッドを
    # 1 本も作らない走り係になり、最初の先読みで書き出しが止まったまま返らない
    threads = settings.decode_threads
    if (
        not isinstance(threads, int)
        or isinstance(threads, bool)
        or not 1 <= threads <= MAX_DECODE_THREADS
    ):
        raise ExportError(f"デコードの並列数は 1 から {MAX_DECODE_THREADS} までの整数: {threads!r}")

    settings.path.parent.mkdir(parents=True, exist_ok=True)
    context = OffscreenGLContext()
    renderer = FrameRenderer(project, context=context, quality=FULL_QUALITY, decode_threads=threads)
    mixer = AudioMixer(project, smooth_history=settings.smooth_history)

    try:
        for index, codec in enumerate(codecs):
            try:
                _encode(
                    project, settings, codec, start, end, renderer, mixer, progress, should_cancel
                )
                break
            except _EncoderOpenError:
                # 開けなかっただけなら、まだ何も描いておらず音も進めていない
                # 書きかけの入れ物だけ消して次へ移る 候補が尽きたら理由をそのまま伝える
                settings.path.unlink(missing_ok=True)
                if index == len(codecs) - 1:
                    raise
    except BaseException:
        # 例外でもキャンセルでも、書きかけを残さない
        settings.path.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
        mixer.close()
        context.release()

    return settings.path


def _encode(
    project: Project,
    settings: ExportSettings,
    codec: str,
    start: int,
    end: int,
    renderer: FrameRenderer,
    mixer: AudioMixer,
    progress: Callable[[float], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> None:
    rate = project.rate
    width, height = project.settings.resolution
    # 混合トラックは音を鳴らすクリップがあるときだけ数える 絵だけの混合トラックで音の
    # 道を作ると、今まで無音の道を持たなかった書き出しに黙った音声が付く 無効にした
    # クリップはミキサが飛ばすので、同じく数えない
    has_audio = any(
        bool(track.clips)
        if track.kind is TrackKind.AUDIO
        else any(clip.enabled and project.plays_sound(track, clip) for clip in track.clips)
        for track in project.timeline.active_sound_tracks()
    )

    try:
        container = av.open(str(settings.path), mode="w")
    except (av.error.FFmpegError, OSError) as exc:
        raise ExportError(f"出力ファイルを開けない: {settings.path} ({exc})") from exc

    with container:
        # 入っていない名前やコンテナが受けないコーデックは、開く前のここで断られる
        # 開けないときと同じく次の候補へ移れるようにする 素の例外のまま出すと、
        # 指定なしの書き出しでも libx264 まで落ちずに失敗する
        try:
            stream = container.add_stream(codec, rate=Fraction(rate.num, rate.den))
        except (av.error.FFmpegError, ValueError) as exc:
            raise _EncoderOpenError(f"映像のエンコーダ {codec} を使えない ({exc})") from exc
        # add_stream は種類の union を返す 以降は映像として扱うので、ここで型を確定させる
        video = cast("av.video.stream.VideoStream", stream)
        video.width = width
        video.height = height
        video.pix_fmt = settings.pixel_format
        # 付けないと再生側が行列を推測する HD なら BT.709 と当てる再生ソフトが多いが、
        # ブラウザや編集ソフトの一部は BT.601 で読み、書いた値と違う色になる
        tag_bt709(video)
        if settings.video_bitrate:
            video.bit_rate = settings.video_bitrate
        if settings.options:
            video.options = dict(settings.options)

        audio = None
        fifo = None
        if has_audio:
            audio = cast(
                "av.audio.stream.AudioStream",
                container.add_stream(settings.audio_codec, rate=mixer.sample_rate),
            )
            audio.bit_rate = settings.audio_bitrate
            fifo = av.audio.fifo.AudioFifo()

        # 最初の mux に任せず、ここで開く 任せると開けない失敗が FFmpeg の例外のまま
        # 途中から飛び出し、別のコーデックへやり直せるかどうかを呼び出し側が区別できない
        try:
            video.codec_context.open()
        except av.error.FFmpegError as exc:
            raise _EncoderOpenError(f"映像のエンコーダ {codec} を開けない ({exc})") from exc

        # フレームの時刻の刻み ストリームの time_base を毎回読み直してはいけない
        # 多重化が始まった時点でコンテナ側の値（MP4 なら 1/15360）に書き換わるので、
        # 読み直すと 2 フレーム目以降の PTS がほぼ 0 に潰れる
        frame_time_base = Fraction(rate.den, rate.num)

        total = end - start
        writer = _FrameWriter(
            container=container,
            video=video,
            audio=audio,
            fifo=fifo,
            mixer=mixer,
            frame_time_base=frame_time_base,
            pixel_format=settings.pixel_format,
            total=total,
            progress=progress,
        )

        # 書き込み（色変換・エンコード・mux）は別スレッドへ逃がす 渡す順序は
        # そのままなので、出来上がるファイルは直列のときと 1 バイトも変わらない
        with _WritePipeline(writer.write, settings.pipeline_depth) as pipeline:
            for index, frame_number in enumerate(range(start, end)):
                if should_cancel is not None and should_cancel():
                    raise ExportError("書き出しを中止した")
                # 書き込み側が落ちていたら、そこで合成をやめる 続けても捨てられるだけで、
                # 本当の失敗（閉じるときに投げ直す）が後ろへ遠ざかる
                if pipeline.failed:
                    break
                # 席を取ってから描く 描いてから待つと、キューの上限に加えて手元の 1 枚が
                # 余分に残り、選んだ枚数より多く抱える（4K なら 1 枚 33MB）
                pipeline.reserve()
                pipeline.submit(index, frame_number, renderer.render(frame_number))

        # 最後の 1 枚を渡した後、書き込みが終わるまでの間に押された中止もここで拾う
        # 見ないと、中止したのに「書き出しました」と出てファイルも残る
        if should_cancel is not None and should_cancel():
            raise ExportError("書き出しを中止した")

        # エンコーダに溜まっている分を吐き出す これを忘れると末尾が欠ける
        # ここへ来るのは書き込みスレッドが終わった後だけ（`with` が待って join する）
        if audio is not None and fifo is not None:
            _flush_audio(container, audio, fifo)
        container.mux(video.encode(None))


class _FrameWriter:
    """合成済みの 1 枚を、色変換してエンコードし mux する

    音声もここで流す 音声と映像の mux は同じコンテナへの書き込みで、
    別のスレッドから触ると FFmpeg の側で順序が壊れる 片側だけ逃がさない
    """

    def __init__(
        self,
        *,
        container: av.container.OutputContainer,
        video: av.video.stream.VideoStream,
        audio: av.audio.stream.AudioStream | None,
        fifo: av.audio.fifo.AudioFifo | None,
        mixer: AudioMixer,
        frame_time_base: Fraction,
        pixel_format: str,
        total: int,
        progress: Callable[[float], None] | None,
    ) -> None:
        self._container = container
        self._video = video
        self._audio = audio
        self._fifo = fifo
        self._mixer = mixer
        self._frame_time_base = frame_time_base
        self._pixel_format = pixel_format
        self._total = total
        self._progress = progress
        #: 音声の書き込み位置 出力レートでの通し番号で、そのまま PTS になる
        self._audio_cursor = 0
        # swscale の変換表を書き出しの間ずっと使い回す 毎フレーム作り直すと
        # 1920x1080 で 1 枚 10ms ほど増える（色変換 15.6ms のうち 10ms 近く）
        # 大きさも書式も書き出しの間は変わらないので、1 つで足りる
        self._reformatter = VideoReformatter()

    def write(self, index: int, frame_number: int, image: np.ndarray) -> None:
        # 音声を先に流す AAC は先頭にプライミングを持つため最初のパケットの
        # DTS が負になり、映像を先に入れると多重化の順序が逆転して弾かれる
        if self._audio is not None and self._fifo is not None:
            self._audio_cursor = _write_audio(
                self._container,
                self._audio,
                self._fifo,
                self._mixer,
                frame_number,
                self._audio_cursor,
            )

        # 合成結果は RGBA のまま渡す rgb24 へ詰め直すと、飛び飛びの読み出しで
        # 6MB を写す手間が増えるだけ（1920x1080 で 1 枚 3ms）
        # swscale は RGBA の A を捨てるので、出る画素は rgb24 から変換したものと同じ
        frame = av.video.frame.VideoFrame.from_ndarray(image, format="rgba")
        frame = to_bt709(frame, self._pixel_format, reformatter=self._reformatter)
        frame.pts = index
        frame.time_base = self._frame_time_base
        self._container.mux(self._video.encode(frame))

        # 進み具合は**書けた枚数**で出す 合成した枚数で出すと、パイプラインの
        # 深さのぶんだけ先へ進んで見え、100% になってから実際の書き込みを待つ
        if self._progress is not None:
            self._progress((index + 1) / self._total)


#: 書き込みスレッドへ渡す品物 ``None`` は「もう来ない」の合図
_Item = tuple[int, int, np.ndarray] | None


class _WritePipeline:
    """GPU の合成と、CPU の書き込みを別のスレッドで重ねる

    書き出しの内訳は :data:`MEASURED_EXPORT_MS` を見る 前の 2 つ（合成・読み戻し）が
    GPU を待つ所で、後ろの 2 つ（色変換・エンコード + mux）が CPU の所
    直列にすると、CPU が動いている間ずっと GPU が遊んでいる

    ``depth`` が 0 なら、スレッドを作らずその場で書く（設定で切れるようにするため）
    1 以上なら、**書き込み中の 1 枚に加えて ``depth`` 枚**まで手元に置く
    席（:meth:`reserve`）で数えるので、合成の側が描いてから渡すまでの間も含まれる

    失敗したときに**黙って取りこぼさない**のがこの作りの肝
    - 書き込みスレッドが投げた例外は覚えておき、:meth:`close` で投げ直す
    - 失敗した後も品物を受け取り続ける 受け取りをやめると、合成の側が
      いっぱいのキューへ ``put`` したまま永久に止まる
    - 席は書き終わったかどうかに関わらず返す 返さないと、失敗した後に
      :meth:`reserve` で止まる
    """

    def __init__(self, write: Callable[[int, int, np.ndarray], None], depth: int) -> None:
        self._write = write
        self._depth = max(0, depth)
        self._error: BaseException | None = None
        # 席は「書き込み中の 1 枚」ぶん多く用意する depth と同じにすると、
        # 深さ 1 が実質直列（書き終わるまで次を描けない）になる
        self._slots = threading.Semaphore(self._depth + 1) if self._depth else None
        self._queue: queue.Queue[_Item] = queue.Queue(maxsize=self._depth + 2)
        self._thread: threading.Thread | None = None

    @property
    def failed(self) -> bool:
        """書き込み側が失敗したか 真なら合成を続けても意味が無い"""
        return self._error is not None

    def __enter__(self) -> _WritePipeline:
        if self._depth > 0:
            self._thread = threading.Thread(
                target=self._run, name="sashimono-export-writer", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # 合成の側が中止や例外で抜けたときも、必ずスレッドを終わらせてから戻る
        # 終わらせないと、書きかけのファイルを消す側とスレッドが同じコンテナを
        # 取り合い、FFmpeg の中で落ちる
        self.close(reraise=kind is None)

    def reserve(self) -> None:
        """合成を始める前に、1 枚ぶんの席を取る 空くまで待つ

        描いてから :meth:`submit` で待つ作りにすると、キューに入っている分と
        書き込み中の分に加えて、手元の 1 枚が余分に残る 4K では 1 枚 33MB あり、
        設定した枚数より多く抱えることになる
        """
        if self._slots is not None:
            self._slots.acquire()

    def submit(self, index: int, frame_number: int, image: np.ndarray) -> None:
        if self._thread is None:
            self._write(index, frame_number, image)
            return
        # 席は :meth:`reserve` で取ってある ここで待つことは普通は無い
        self._queue.put((index, frame_number, image))

    def close(self, *, reraise: bool = True) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        if reraise and self._error is not None:
            error = self._error
            self._error = None
            raise error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                # すでに失敗していれば捨てるだけ 受け取りは続ける（上の説明のとおり）
                if self._error is None:
                    self._write(*item)
            except BaseException as error:
                self._error = error
            finally:
                # 席は必ず返す 失敗したときに返さないと、合成の側が reserve で止まる
                if self._slots is not None:
                    self._slots.release()


def _write_audio(
    container: av.container.OutputContainer,
    stream: av.audio.stream.AudioStream,
    fifo: av.audio.fifo.AudioFifo,
    mixer: AudioMixer,
    frame_number: int,
    cursor: int,
) -> int:
    """1 フレーム分の音声を FIFO へ流し、エンコーダが要求する粒度で切り出す

    AAC は 1024 サンプル単位でしか受け取らない 映像のフレーム境界とは
    一致しないので、FIFO を挟んで詰め替える 戻り値は次の書き込み位置
    """
    block = mixer.render_frames(frame_number, 1)
    if len(block) == 0:
        return cursor

    frame = _to_audio_frame(block, mixer.sample_rate, mixer.channels)
    # PTS を付けないと、エンコーダが時刻を持たないパケットを出して多重化に失敗する
    frame.time_base = Fraction(1, mixer.sample_rate)
    frame.pts = cursor
    fifo.write(frame)

    frame_size = stream.codec_context.frame_size or 1024
    while True:
        chunk = fifo.read(frame_size)
        if chunk is None:
            break
        container.mux(stream.encode(chunk))
    return cursor + len(block)


def _flush_audio(
    container: av.container.OutputContainer,
    stream: av.audio.stream.AudioStream,
    fifo: av.audio.fifo.AudioFifo,
) -> None:
    remainder = fifo.read()
    if remainder is not None:
        container.mux(stream.encode(remainder))
    container.mux(stream.encode(None))


def _to_audio_frame(
    samples: np.ndarray, sample_rate: int, channels: int
) -> av.audio.frame.AudioFrame:
    """``(サンプル数, チャンネル数)`` の float32 を PyAV のフレームへ

    出力段でここだけクリッピングする 合成の途中で頭打ちにすると、後段の
    調整で潰れた音しか扱えなくなる
    """
    clipped = np.clip(samples, -1.0, 1.0).astype(np.float32)
    planar = np.ascontiguousarray(clipped.T)
    frame = av.audio.frame.AudioFrame.from_ndarray(planar, format="fltp", layout=_LAYOUTS[channels])
    frame.sample_rate = sample_rate
    return frame
