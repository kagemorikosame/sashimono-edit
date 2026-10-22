"""タイムラインをファイルへ書き出す

プレビューと同じ :class:`~sashimono.engine.render.FrameRenderer` と
:class:`~sashimono.engine.audio.AudioMixer` を使う 別経路にすると
「プレビューでは出るのに書き出すと出ない」が起きる
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import av
import av.audio.fifo
import av.audio.frame
import av.audio.stream
import av.container
import av.error
import av.video.codeccontext
import av.video.frame
import av.video.stream
import numpy as np
from OpenGL.error import Error as GLLibraryError

from sashimono.core.chapters import chapters_of
from sashimono.core.model import Project, TrackKind
from sashimono.engine.audio import AudioMixer
from sashimono.engine.colorspace import (
    VideoReformatter,
    bt709_planes,
    tag_bt709,
    tag_frame_bt709,
    to_bt709,
)
from sashimono.engine.encoders import (
    PRIORITY_AUTO,
    MonotonicProgress,
    encoder_options,
    order_by_priority,
    probe_encoder,
)
from sashimono.engine.gpu import OffscreenGLContext, ShaderError, Texture
from sashimono.engine.gpu.yuv import COLOR_TOLERANCE, YuvReadback, open_readback
from sashimono.engine.render import (
    DEFAULT_DECODE_THREADS,
    FULL_QUALITY,
    MAX_DECODE_THREADS,
    FrameRenderer,
)

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    # 型の宣言にしか無い（PyAV の .pyi） 実行時に読むと ImportError になる
    from av.container.core import Chapter

__all__ = [
    "CHAPTER_SUFFIXES",
    "COLOR_OPTIONS",
    "DEFAULT_PIPELINE_DEPTH",
    "MAX_DECODE_THREADS",
    "MAX_PIPELINE_DEPTH",
    "MEASURED_DECODE_MS",
    "MEASURED_EXPORT_MS",
    "MEASURED_EXPORT_TOTAL_MS",
    "MEASURED_GPU_COLOR_MS",
    "ExportError",
    "ExportSettings",
    "available_video_codecs",
    "export_chapters",
    "export_project",
    "write_chapters",
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

#: 色変換を CPU で行うときと GPU で行うとき（#372）の、書き出し 1 枚あたりの実測（ミリ秒）
#: 1920x1080 を 3 枚重ね・NVENC・先読みの深さ 2 で 60 枚を書き出し切った値（RTX 5060 Ti・
#: 駆動 610.88 書き出す前の試験の絵の確かめを含む） GPU の側は読み戻しも 1.5 byte/画素に
#: 減り、色変換が CPU から消える 同じ条件の 4K では 60.97 が 36.68 になった
#: 測り直すときは tools\bench_export.py --codec h264_nvenc --compare（4K は
#: --width 3840 --height 2160 を足す）
MEASURED_GPU_COLOR_MS = (17.26, 13.49)

#: 同じ素材の合成（デコードを含む GPU 合成）1 枚あたりの実測（ミリ秒）
#: 並べない（1 本ずつ）/ 4 本まで並べる の順 1920x1080 を 3 枚重ね
#: PyAV のデコードは GIL を解放するので、別の素材どうしなら本当に重なる
#: 測り直すときは tools\bench_export.py --decode-threads 1 と付けない場合を比べる
MEASURED_DECODE_MS = (15.9, 11.2)

#: 優先順に並べた映像コーデック 前にあるものから、使えるものを選ぶ
#: GPU のエンコーダは CPU をほとんど使わないので、長尺でも編集を続けながら書き出せる
#: AMD（AMF）は NVENC の後ろ、Intel（QSV）の前に置く NVENC を持たない AMD の機械で、
#: CPU や内蔵の Intel より先に AMD の GPU を使うため（#365） NVIDIA と AMD の両方がある
#: 機械では NVENC を先に使う 先に使う側は本人の設定
#: （:attr:`ExportSettings.encoder_priority`）で替えられる
VIDEO_CODEC_PREFERENCE = ("h264_nvenc", "h264_amf", "h264_qsv", "libx264")

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
    #: 動画用の色（BT.709 の YUV）への変換を GPU で行う（#372） 偽なら CPU（swscale）で行う
    #: GPU で行えるのは 8bit の 4:2:0（yuv420p・nv12）で縦横が偶数のときだけで、ほかは
    #: 真でも CPU で行う 結果の差は 8bit の丸めの 1 まで 本人の設定（Preferences）から渡す
    gpu_color: bool = True
    #: 前の音を読む音の効果の動く値を、区切りの中でもつなぐ（:class:`AudioMixer`）
    #: 再生と同じ音にするため、本人の設定（プレビューと同じ値）を渡す
    smooth_history: bool = True
    #: 名前の付いた目印をチャプターとして埋める（#330 2026-10-17 利用者の決定 既定は入）
    #: 効くのは :data:`CHAPTER_SUFFIXES` の入れ物だけ 本人の設定（Preferences）から渡す
    chapters: bool = True
    #: ``video_codec`` が ``None`` のとき、どの作り手のエンコーダを先に試すか（#365）
    #: :data:`~sashimono.engine.encoders.ENCODER_PRIORITIES` のどれか
    #: 本人の設定（Preferences）から渡す
    encoder_priority: str = PRIORITY_AUTO


#: チャプターを埋める入れ物 mp4・mov は章の文字トラック、mkv は Chapters の要素で持つ
#: ほかの入れ物（webm・avi など）は章を持てないか、プレーヤーが読まない 渡すと多重化に
#: 断られて書き出しごと失敗しうるので、埋めない
CHAPTER_SUFFIXES = frozenset({".mp4", ".mov", ".mkv"})


def export_chapters(project: Project, frame_range: tuple[int, int], path: Path) -> list[Chapter]:
    """``path`` へ埋めるチャプター 時刻は書き出す範囲の頭から数えたフレーム

    入れ物が :data:`CHAPTER_SUFFIXES` に無ければ空 名前の無い目印と範囲の外の目印は入れない
    （:func:`~sashimono.core.chapters.chapters_of`）
    """
    if path.suffix.lower() not in CHAPTER_SUFFIXES:
        return []
    frame = project.rate.frame_duration
    found = chapters_of(project.timeline, frame_range)
    # 型の宣言を通して渡す PyAV 18 の宣言は time_base を Fraction、19 は AVRational と書くが、
    # どちらも実行時は Fraction を受ける（19.0.1 の CI で読み戻しの試験が通る）
    # 辞書のまま書くと、入る版によって mypy が片方で落ちる
    return [
        cast(
            "Chapter",
            {
                "id": index + 1,
                # 時刻はフレームの区切りに乗っているので、フレームの長さで割れば割り切れる
                "start": int(chapter.start / frame),
                "end": int(chapter.end / frame),
                "time_base": frame,
                "metadata": {"title": chapter.title},
            },
        )
        for index, chapter in enumerate(found.chapters)
    ]


def write_chapters(container: av.container.OutputContainer, chapters: list[Chapter]) -> None:
    """入れ物にチャプターを渡す 最初の多重化（ヘッダを書く）より前に呼ぶ

    後から渡すと、mkv は頭の Chapters の要素を書き終えていて、章が入らない
    """
    if chapters:
        container.set_chapters(chapters)


def available_video_codecs(priority: str = PRIORITY_AUTO) -> list[str]:
    """この環境で使える映像コーデックを、優先順に返す

    入っているだけでは数えない 実際に開けて 1 コマ書けたものだけを返す（#67 #365）
    QSV や AMF は FFmpeg に組み込まれていても、Intel や AMD の GPU や駆動が無い機械では開けない
    入っているかだけで選ぶと、開けない QSV や AMF が既定になって書き出しが失敗する

    ``priority`` の作り手のエンコーダを前へ寄せる
    （:func:`~sashimono.engine.encoders.order_by_priority`） 知らない値は ``ValueError``
    """
    return order_by_priority([name for name in VIDEO_CODEC_PREFERENCE if _opens(name)], priority)


@cache
def _opens(name: str) -> bool:
    """``name`` のエンコーダで書けるか 1 つにつき 1 度だけ試して覚える

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
    """``name`` のエンコーダを開いて 1 コマ書いてみる 書けなければ投げる

    試し方は控えの作成と同じ物を使う（:func:`~sashimono.engine.encoders.probe_encoder`）
    名前をここに残すのは、試験が開けないエンコーダを真似るときにここを差し替えるため
    """
    probe_encoder(name)


class _EncoderError(ExportError):
    """エンコーダが使えなかった 書きかけを捨てて、別のコーデックで最初からやり直せる"""


class _EncoderOpenError(_EncoderError):
    """エンコーダを開けなかった まだ 1 コマも書いていない"""


class _EncoderWriteError(_EncoderError):
    """開けたエンコーダが途中で書けなくなった（駆動の失敗など）

    合成や多重化（mux）の失敗とは分ける そちらはコーデックを替えても直らないので、
    やり直すと同じ所でもう 1 度落ちるまで待たされる
    """


class _GpuColorError(ExportError):
    """GPU の色変換か読み戻しが、書き出しの途中で使えなくなった（#372）

    その場から CPU の道へ切り替えては続けられない 読み戻している途中のコマは GPU の
    バッファごと捨てるしかなく、絵と音の一部はもう入れ物へ書いてある 続けるとコマが欠けるか
    順が乱れるので、同じコーデックで頭から CPU の色変換で書き直す
    """


def _fallback_notice(
    refused: list[tuple[str, _EncoderError]], used: str, gpu_failure: _GpuColorError | None
) -> str:
    """書き直して書き上げたときの知らせ 何で書いたかと、前の試みが使えなかった理由

    エンコーダを替えたことと GPU の色変換をやめたことは、1 つにまとめて知らせる
    分けると、1 回の書き出しで窓に知らせが 2 度出て、どちらが最後の状態か迷う
    """
    parts: list[str] = []
    if refused:
        reasons = " / ".join(str(error) for _codec, error in refused)
        names = "、".join(codec for codec, _error in refused)
        parts.append(
            f"{names} では書き出せなかったので、{used} で最初から書き出し直した（{reasons}）"
        )
    if gpu_failure is not None:
        parts.append(f"色変換を CPU に替えて最初から書き出し直した（{gpu_failure}）")
    return " ".join(parts)


def _video_options(codec: str, settings: ExportSettings) -> dict[str, str]:
    """エンコーダへ渡す選択肢 既定（AMF の品質と上限のビットレート #365）に本人の指定を重ねる

    本人の指定（:attr:`ExportSettings.options`）を後から重ねて勝たせる 既定を後にすると、
    品質を名指しした人の指定が黙って消える
    """
    return {**encoder_options(codec, settings.video_bitrate), **settings.options}


def _encode_frame(
    video: av.video.stream.VideoStream, frame: av.video.frame.VideoFrame | None
) -> list[av.Packet[Any]]:
    """1 コマ（``None`` なら残り）をエンコードする 失敗したら :class:`_EncoderWriteError`

    多重化（mux）はここに入れない エンコーダの失敗だけを、次の候補へ移れる失敗として拾う
    """
    try:
        return video.encode(frame)
    except av.error.FFmpegError as exc:
        name = video.codec_context.name
        raise _EncoderWriteError(f"映像のエンコーダ {name} が途中で書けなくなった ({exc})") from exc


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
    notice: Callable[[str], None] | None = None,
) -> Path:
    """プロジェクトを 1 本の動画ファイルへ書き出す

    ``should_cancel`` が真を返したら、書きかけのファイルを消して
    :class:`ExportError` を投げる 中途半端なファイルを残すと、書き出せたのか
    どうかが分からなくなる

    コーデックを名指ししない書き出しで、候補のエンコーダが開けなかったり途中で書けなく
    なったりしたら、次の候補で最初から書き出し直す 書き上げたら、何で書いたかを ``notice``
    へ知らせる（#365） 黙って替えると、GPU で書いたつもりの人が遅さや画の違いの理由を探せない
    GPU の色変換（#372）が途中で使えなくなったときは、同じコーデックで色変換を CPU にして
    頭から書き直し、同じ知らせにまとめて伝える
    """
    start, end = settings.frame_range or (0, project.duration)
    total = end - start
    if total <= 0:
        raise ExportError("書き出す範囲が空")

    # 指定が無ければ、開けた候補を前から試す 試しに開けても、実際の大きさでは
    # 断られることがある（NVENC の H.264 は幅 4096 を超える画を開けない）
    # そのときに書き出しごと失敗させず、最後は libx264 まで落とす
    try:
        codecs = (
            [settings.video_codec]
            if settings.video_codec
            else available_video_codecs(settings.encoder_priority)
        )
    except ValueError as exc:
        # 知らない優先を黙って自動にすると、選んだ物が効かない理由が分からない
        raise ExportError(str(exc)) from exc
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
    #: 書けなかったコーデックと理由 次の候補で書き上げたときに、何で書いたかと一緒に知らせる
    refused: list[tuple[str, _EncoderError]] = []
    #: 途中で使えなくなった GPU の色変換 CPU で書き直したときに、エンコーダの話と一緒に知らせる
    gpu_failure: _GpuColorError | None = None
    # 書き直しても進み具合を戻さない（#365 戻ると窓のバーや AI の export_status が下がる）
    shown = MonotonicProgress(progress) if progress is not None else None

    try:
        index = 0
        while True:
            codec = codecs[index]
            try:
                _encode(
                    project,
                    settings,
                    codec,
                    start,
                    end,
                    renderer,
                    mixer,
                    shown.report if shown is not None else None,
                    should_cancel,
                )
            except _GpuColorError as exc:
                # 同じコーデックのまま、色変換だけ CPU にして頭から書き直す（#372）
                # CPU で書き直しても同じ所で落ちる物ではないので、候補は進めない
                # gpu_color を切ってから書き直すので、2 度目はここへ来ない
                settings.path.unlink(missing_ok=True)
                if not settings.gpu_color:
                    raise
                gpu_failure = exc
                settings = replace(settings, gpu_color=False)
            except _EncoderError as exc:
                # 書きかけの入れ物を消して次へ移る 候補が尽きたら理由をそのまま伝える
                settings.path.unlink(missing_ok=True)
                if index == len(codecs) - 1:
                    raise
                refused.append((codec, exc))
                index += 1
                if not isinstance(exc, _EncoderWriteError):
                    # 開く前に断られた まだ何も描いていないので、描く側をそのまま使える
                    if shown is not None:
                        shown.start_again()
                    continue
            else:
                if (refused or gpu_failure is not None) and notice is not None:
                    notice(_fallback_notice(refused, codec, gpu_failure))
                break
            if shown is not None:
                shown.start_again()
            # 途中まで描いて音も進めた 描く側と混ぜる側は前の音や絵を覚えている
            # （音の効果の残り・軌跡など）ので、作り直して最初から書き出し直す
            # 使い回すと、書き直した頭に前の試みの途中の音や絵が混ざる
            renderer.close()
            mixer.close()
            renderer = FrameRenderer(
                project, context=context, quality=FULL_QUALITY, decode_threads=threads
            )
            mixer = AudioMixer(project, smooth_history=settings.smooth_history)
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
        options = _video_options(codec, settings)
        if options:
            video.options = dict(options)

        audio = None
        fifo = None
        if has_audio:
            audio = cast(
                "av.audio.stream.AudioStream",
                container.add_stream(settings.audio_codec, rate=mixer.sample_rate),
            )
            audio.bit_rate = settings.audio_bitrate
            fifo = av.audio.fifo.AudioFifo()

        if settings.chapters:
            write_chapters(container, export_chapters(project, (start, end), settings.path))

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

        # 色変換は GPU で行えるならそちらで行う（#372） 作れない形式や駆動では CPU に残る
        source = _FrameSource(
            renderer, settings.pixel_format if settings.gpu_color else None, width, height
        )
        try:
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
                    # 席は絵を手元へ出す直前に取る（:meth:`_FrameSource.frame` が呼ぶ）
                    # 描いてから待つと、キューの上限に加えて手元の 1 枚が余分に残り、
                    # 選んだ枚数より多く抱える（4K なら 1 枚 33MB）
                    for item in source.frame(index, frame_number, pipeline.reserve):
                        pipeline.submit(*item)
                # GPU で読み戻している途中のコマを受け取る ここを忘れると末尾のコマが欠ける
                # 書き込み側が落ちていれば受け取っても捨てられるだけなので、閉じて投げ直させる
                if not pipeline.failed:
                    for item in source.drain(pipeline.reserve):
                        pipeline.submit(*item)
        except BaseException:
            # 中止や失敗で抜けたときも、読み戻しのバッファを返す 書き込みのスレッドは
            # 上の with が終わらせてあり、もう GPU の配列を待っている者はいない
            # 返すのに駆動が断っても、元の失敗を上げる 返す側の GL のエラーに差し替わると、
            # 書き直しの合図（_GpuColorError）やエンコーダの失敗・中止が見分けられなくなり、
            # CPU でなら書ける書き出しまで丸ごと失敗する
            _release_quietly(source)
            raise
        # 書き終えた後に返すのを駆動が断ったら、CPU の色変換で頭から書き直す（_GpuColorError）
        # 失敗として上げると、書けていたかもしれない動画ごと捨てて書き出しが失敗する
        # そのまま使わないのは、片付けで断った駆動が、それまでの読み戻しでも黙って壊れた面を
        # 返していないとは言えないため 書き直しは 1 回だけで、CPU の道に読み戻しは無い
        with _gpu_color_failures():
            source.release()

        # 最後の 1 枚を渡した後、書き込みが終わるまでの間に押された中止もここで拾う
        # 見ないと、中止したのに「書き出しました」と出てファイルも残る
        if should_cancel is not None and should_cancel():
            raise ExportError("書き出しを中止した")

        # エンコーダに溜まっている分を吐き出す これを忘れると末尾が欠ける
        # ここへ来るのは書き込みスレッドが終わった後だけ（`with` が待って join する）
        if audio is not None and fifo is not None:
            _flush_audio(container, audio, fifo)
        container.mux(_encode_frame(video, None))


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

        if image.ndim == 2:
            # GPU で変換済みの面（#372） 並びは書き出す画素形式そのもので、変換は要らない
            # タグだけ CPU の道と同じに付ける 写さずに配列をそのまま包む（from_ndarray で
            # 写すと 4K で 1 枚 7ms 掛かり、GPU へ移して浮いた時間の 3 分の 1 が消える）
            # 配列は読み戻しがコマごとに新しく作るので、包んだまま後で上書きされることは無い
            frame = av.video.frame.VideoFrame.from_numpy_buffer(image, format=self._pixel_format)
            tag_frame_bt709(frame)
        else:
            # 合成結果は RGBA のまま渡す rgb24 へ詰め直すと、飛び飛びの読み出しで
            # 6MB を写す手間が増えるだけ（1920x1080 で 1 枚 3ms）
            # swscale は RGBA の A を捨てるので、出る画素は rgb24 から変換したものと同じ
            frame = av.video.frame.VideoFrame.from_ndarray(image, format="rgba")
            frame = to_bt709(frame, self._pixel_format, reformatter=self._reformatter)
        frame.pts = index
        frame.time_base = self._frame_time_base
        self._container.mux(_encode_frame(self._video, frame))

        # 進み具合は**書けた枚数**で出す 合成した枚数で出すと、パイプラインの
        # 深さのぶんだけ先へ進んで見え、100% になってから実際の書き込みを待つ
        if self._progress is not None:
            self._progress((index + 1) / self._total)


#: 書き込みへ渡す 1 コマ 通し番号・タイムラインのフレーム・絵
#: 絵は RGBA の ``(高さ, 幅, 4)`` か、GPU で変換済みの YUV の面 ``(高さ x 1.5, 幅)``
_Frame = tuple[int, int, np.ndarray]


class _FrameSource:
    """合成した絵を、書き込みへ渡す形で出す GPU の色変換（#372）の出入り口

    GPU で変換できるときは、合成した絵を GPU の上で YUV へ変換し、PBO で読み戻す
    読み戻しは 1 コマ遅れて出てくる（次のコマを合成している間に運ぶ） 出る順は合成した
    順のまま :meth:`drain` で最後の分を受け取る

    書き出しを始める前に、決まった試験の絵（:func:`_check_picture`）を GPU と CPU の両方で
    変換して比べ、差が :data:`COLOR_TOLERANCE` を超えたら GPU を使わずに CPU の道で書く
    手元で確かめられたのは NVIDIA だけで、AMD や Intel の駆動で変換が食い違ったときに、
    色のずれた動画を黙って書かないため 比べるのは 1 枚ぶんの CPU の変換（1080p で 7 ms ほど）
    合成した最初のコマで比べないのは、書き出しが暗転や単色で始まることが多いため 単色の絵では
    色差の間引きの重み・端の扱い・U と V の面の境目が違っても差が 0 になり、食い違いを見逃す

    使えないとき（形式・奇数の大きさ・駆動が断った・食い違った・本人が切った）は、今までどおり
    :meth:`FrameRenderer.render` の RGBA を渡し、書き込みの側で CPU が変換する
    """

    def __init__(
        self, renderer: FrameRenderer, pixel_format: str | None, width: int, height: int
    ) -> None:
        self._renderer = renderer
        self._readback: YuvReadback[tuple[int, int]] | None = None
        if pixel_format is not None:
            # ここは _encode の try より前で、駆動の断りが素の GL のエラーのまま出ると、書き直しの
            # 合図にもならずに書き出しが始まる前に失敗する どの GL の呼び出しも、断られたら
            # CPU の道へ倒す open_readback と _agrees は中で受ける 残りは捨てるときの release
            with renderer.context:
                readback: YuvReadback[tuple[int, int]] | None = open_readback(
                    width, height, pixel_format
                )
                if readback is not None and not _agrees(readback, pixel_format, width, height):
                    # 食い違ったか確かめる途中で断られた GPU の変換は捨てる
                    # どのコマも CPU で変換する
                    _discard_quietly(readback)
                    readback = None
                self._readback = readback

    @property
    def on_gpu(self) -> bool:
        """色変換を GPU で行っているか 始める前に比べて食い違ったら偽"""
        return self._readback is not None

    def frame(self, index: int, frame_number: int, reserve: Callable[[], None]) -> list[_Frame]:
        """``frame_number`` を合成し、渡せるようになったコマを返す（0 か 1 枚）

        ``reserve`` は書き込みの席を取る 絵を手元へ出す直前に呼ぶ
        """
        readback = self._readback
        if readback is None:
            reserve()
            return [(index, frame_number, self._renderer.render(frame_number))]
        with self._renderer.context:
            self._renderer.compose(frame_number)
            texture = self._renderer.compositor.resolved_texture()
            with _gpu_color_failures():
                readback.push(texture, (index, frame_number))
            if not readback.full:
                return []
            reserve()
            with _gpu_color_failures():
                (done, number), image = readback.pop()
        return [(done, number, image)]

    def drain(self, reserve: Callable[[], None]) -> Iterator[_Frame]:
        """読み戻している途中のコマを、合成した順に出す"""
        while self._readback is not None and self._readback.pending:
            reserve()
            with self._renderer.context, _gpu_color_failures():
                (index, frame_number), image = self._readback.pop()
            yield index, frame_number, image

    def release(self) -> None:
        """読み戻しのバッファを返す 何度呼んでもよい 返した後は CPU の道で書く"""
        readback, self._readback = self._readback, None
        if readback is not None:
            with self._renderer.context:
                readback.release()


@contextmanager
def _gpu_color_failures() -> Iterator[None]:
    """書き出しの途中で GPU の色変換や読み戻しを駆動が断ったら :class:`_GpuColorError` にする

    素の GL の例外のまま出すと、CPU でなら書ける書き出しが丸ごと失敗する
    """
    try:
        yield
    except (ShaderError, GLLibraryError) as exc:
        raise _GpuColorError(f"GPU の色変換が書き出しの途中で使えなくなった ({exc})") from exc


def _discard_quietly(readback: YuvReadback[tuple[int, int]]) -> None:
    """使わないと決めた読み戻しを返す 返すのを駆動が断っても、記録に残すだけにする

    もう CPU の道で書くと決めた後なので、返せなくても書き出しは続けられる 投げると、
    CPU でなら書けた書き出しが始まる前に素の GL のエラーで失敗する
    """
    try:
        readback.release()
    except (ShaderError, GLLibraryError):
        _log.warning("使わない GPU の読み戻しを返せなかった", exc_info=True)


def _release_quietly(source: _FrameSource) -> None:
    """失敗で抜ける途中に読み戻しを返す 返すのを駆動が断っても、記録に残すだけにする

    投げると、抜ける理由になった元の失敗がこの GL のエラーに差し替わる
    """
    try:
        source.release()
    except (ShaderError, GLLibraryError):
        _log.warning("書き出しの後片付けで GPU の読み戻しを返せなかった", exc_info=True)


#: GPU の変換を確かめるときの読み戻しの印 合成したコマの印（通し番号は 0 から）と重ならない
_CHECK_TAG = (-1, -1)


def _check_picture(width: int, height: int) -> np.ndarray:
    """GPU の変換を確かめる ``width`` x ``height`` の RGBA 毎回同じ絵

    上半分は 1 行おき・1 列おきの縞 縦の間引きの重みと端の扱いが違うと、ここでずれる
    下半分は隣と関係の無い色 色差の間引き方や係数が違うと、ここがいちばん大きくずれる
    大きさを書き出しと同じにするのは、yuv420p の U と V の面の境目が大きさで動くため
    小さな絵で確かめると、境目の位置を間違える駆動を見逃す
    """
    image = np.random.default_rng(372).integers(0, 256, (height, width, 4), dtype=np.uint8)
    top = image[: height // 2]
    top[...] = 0
    top[0::2, :, 0] = 255
    top[1::2, :, 1] = 255
    top[:, 0::2, 2] = 200
    image[..., 3] = 255
    return image


def _agrees(
    readback: YuvReadback[tuple[int, int]], pixel_format: str, width: int, height: int
) -> bool:
    """``readback`` の変換が、CPU の変換と :data:`COLOR_TOLERANCE` まで揃うか

    GL コンテキストが current な状態で呼ぶ 駆動がここで断ったときも偽を返す
    書き出しごと失敗させると、今まで CPU で書き出せていた機械で書けなくなる
    """
    picture = _check_picture(width, height)
    try:
        texture = Texture.from_array(picture, srgb=False)
        try:
            readback.push(texture.handle, _CHECK_TAG)
            tag, planes = readback.pop()
        finally:
            texture.release()
    except (ShaderError, GLLibraryError):
        return False
    expected = bt709_planes(picture, pixel_format)
    if tag != _CHECK_TAG or planes.shape != expected.shape:
        return False
    difference = np.abs(planes.astype(np.int16) - expected.astype(np.int16))
    return int(difference.max()) <= COLOR_TOLERANCE


#: 書き込みスレッドへ渡す品物 ``None`` は「もう来ない」の合図
_Item = _Frame | None


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
