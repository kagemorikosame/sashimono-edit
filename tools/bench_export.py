r"""書き出しの重さの内訳を測る（Issue #56 の 1 と 2）

    .venv\Scripts\python.exe tools\bench_export.py
    .venv\Scripts\python.exe tools\bench_export.py --layers 3 --frames 60

測るのは 3 つ

1. **1 フレームの内訳** 合成（GPU）／読み戻し（``glReadPixels``）／色変換／
   エンコード／mux を分けて測る どこを並べれば効くのかを、直す前に決めるため
2. **レイヤー数ごとの合成時間** 重ねるとデコードが重さの中心になる、という
   ``engine/cache/proxy.py`` の実測を書き出しの側でも確かめる
3. **PyAV のデコードが GIL を解放するか** 別々の素材を指すデコーダを
   スレッドで並べて、直列と比べる 解放していなければ並列化しても速くならない
   （推測で並列化すると、切り替えの分だけ遅くなる）

``--compare`` を付けると、書き出しそのもの（:func:`export_project`）を
パイプラインの深さを変えて測り、出来上がったファイルのフレームが
一致するかまで見る

``--decode-threads 1`` でレイヤーごとの並列デコードを切れる 並べた場合と
比べるとき（同じ機械・同じ素材で 2 回回す）に使う

内訳と ``--compare`` は、色変換を CPU（swscale）で行う道と GPU で行う道（#372）の
両方で測る GPU の道では「読み戻し」に GPU の変換と YUV の読み戻しが入り、
「色変換」は CPU で YUV の面を PyAV のフレームで包むだけになる

ffmpeg が無いか OpenGL のコンテキストを作れない環境では、何も測らずに終わる（終了コード 0）
GPU の色変換だけを作れない環境（形式や駆動が断る）では、色変換は CPU の道だけを測る
"""

from __future__ import annotations

import argparse
import atexit
import io
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import cast

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 開発者本人の設定やキャッシュに触らない
_base = Path(tempfile.mkdtemp(prefix="sashimono-export-bench-"))
# 終わったら捨てる 素材と書き出したファイルを置くので、回すたびに残すと溜まる
atexit.register(shutil.rmtree, _base, True)
os.environ["APPDATA"] = str(_base / "roaming")
os.environ["LOCALAPPDATA"] = str(_base / "local")

import av  # noqa: E402
import av.video.frame  # noqa: E402
import numpy as np  # noqa: E402
from OpenGL import GL  # noqa: E402

from sashimono.core.commands import AddClip, AddMedia, AddTrack  # noqa: E402
from sashimono.core.model import (  # noqa: E402
    AnimatedValue,
    Clip,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate  # noqa: E402
from sashimono.engine.colorspace import (  # noqa: E402
    VideoReformatter,
    tag_bt709,
    tag_frame_bt709,
    to_bt709,
)
from sashimono.engine.decode import VideoDecoder, probe_media  # noqa: E402
from sashimono.engine.encode import (  # noqa: E402
    ExportSettings,
    available_video_codecs,
    export_project,
)
from sashimono.engine.gpu import GLContextError, OffscreenGLContext  # noqa: E402
from sashimono.engine.gpu.yuv import YuvReadback, open_readback  # noqa: E402
from sashimono.engine.render import (  # noqa: E402
    DEFAULT_DECODE_THREADS,
    FULL_QUALITY,
    MAX_DECODE_THREADS,
    FrameRenderer,
)


def _make_source(
    directory: Path, *, width: int, height: int, seconds: float, tag: str
) -> Path | None:
    """測る用の素材を ffmpeg で作る 動きのある絵にして、圧縮で楽をさせない

    作れなければ ``None`` 使えるコーデックの判定は PyAV のエンコーダを見ているので、
    外の ffmpeg に libx264 が入っているかは別の話 落とさずに、測らずに終える
    """
    path = directory / f"{tag}-{width}x{height}.mp4"
    if path.exists():
        return path
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate=30:duration={seconds}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    # 組み立てているのは固定の文字列と argparse が受けた数値、一時フォルダの中の
    # パスだけで、外から来る文字列は混ざらない shell は通さない（list 渡し）
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if subprocess.run(command, check=False).returncode != 0:
        return None
    return path


def _project(sources: list[Path], width: int, height: int, frames: int) -> Project:
    """``sources`` を 1 枚ずつ別のトラックへ重ねたプロジェクト"""
    project = Project.create(ProjectSettings(width=width, height=height, frame_rate=FrameRate(30)))
    for index, source in enumerate(sources):
        media = probe_media(source)
        project = AddMedia(media).apply(project)
        track = Track(kind=TrackKind.VIDEO, name=f"V{index + 1}")
        project = AddTrack(track).apply(project)
        clip = Clip(
            timeline_start=0,
            duration=frames,
            media_id=media.id,
            # 重ねた下の絵も描かせる 不透明のままだと、隠れた分を飛ばす作りに
            # なったときに重ねた意味が無くなる
            # 不透明度は 0.0 から 1.0 で、合成側が範囲へ丸める 70.0 と書くと
            # 1.0 に丸められて不透明のままになり、重ねた意味が消える
            opacity=AnimatedValue(0.7),
        )
        project = AddClip(track.id, clip).apply(project)
    return project


def _summary(times: list[float]) -> str:
    return f"中央 {statistics.median(times):6.2f} ms  合計 {sum(times):7.1f} ms"


def _measure(step: Callable[[], None], frames: int) -> list[float]:
    """1 回目は温まっていないので捨てる"""
    times: list[float] = []
    for number in range(frames + 1):
        started = time.perf_counter()
        step()
        elapsed = (time.perf_counter() - started) * 1000
        if number:
            times.append(elapsed)
    return times


def breakdown(
    project: Project,
    context: OffscreenGLContext,
    codec: str,
    frames: int,
    output: Path,
    threads: int,
    *,
    gpu: bool = False,
) -> dict[str, list[float]] | None:
    """1 フレームの内訳（ミリ秒） 書き出しと同じ順序で 1 段ずつ測る

    合成のあとに ``glFinish`` を入れる 入れないと命令を投げた時間だけを測り、
    実際の待ちが読み戻しの側へ寄って見える

    ``gpu`` を立てると色変換を GPU で行う道で測る 段の分け方を揃えるため、読み戻しは
    受け取るまで待つ（書き出しは 1 コマ遅れて受け取り、合成の裏で運ぶ） この機械で
    GPU の変換を作れなければ ``None``
    """
    width, height = project.settings.resolution
    readback: YuvReadback[int] | None = None
    if gpu:
        with context:
            readback = open_readback(width, height, "yuv420p")
        if readback is None:
            return None
    renderer = FrameRenderer(project, context=context, quality=FULL_QUALITY, decode_threads=threads)
    container = av.open(str(output), mode="w")
    # 書き出しの本体（_FrameWriter）と同じく、swscale の表を使い回す
    # ここだけ毎フレーム作り直すと、直さなくてよい所が重く見える
    reformatter = VideoReformatter()
    times: dict[str, list[float]] = {
        "合成": [],
        "読み戻し": [],
        "色変換": [],
        "エンコード": [],
        "mux": [],
        # 最後に 1 回だけ 1 枚あたりの段ではないので、別の行として出す
        "吐き出し": [],
    }
    closed = False
    try:
        stream = container.add_stream(codec, rate=Fraction(30, 1))
        # 映像の符号化方式で足した流れは映像の流れ 書き出し（engine/encode/exporter.py）と同じ書き方
        video = cast("av.video.stream.VideoStream", stream)
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p"
        tag_bt709(video)
        video.bit_rate = 12_000_000
        video.codec_context.open()
        for number in range(frames + 1):
            keep = number > 0
            with context:
                started = time.perf_counter()
                renderer.compose(number % max(project.duration, 1))
                GL.glFinish()
                composed = time.perf_counter()
                if readback is None:
                    image = renderer.compositor.read()
                else:
                    readback.push(renderer.compositor.resolved_texture(), number)
                    _tag, image = readback.pop()
                read = time.perf_counter()
            if readback is None:
                frame = av.video.frame.VideoFrame.from_ndarray(image, format="rgba")
                frame = to_bt709(frame, "yuv420p", reformatter=reformatter)
            else:
                # 書き出しの本体と同じく、写さずに包む
                frame = av.video.frame.VideoFrame.from_numpy_buffer(image, format="yuv420p")
                tag_frame_bt709(frame)
            frame.pts = number
            frame.time_base = Fraction(1, 30)
            converted = time.perf_counter()
            packets = video.encode(frame)
            encoded = time.perf_counter()
            container.mux(packets)
            muxed = time.perf_counter()
            if keep:
                times["合成"].append((composed - started) * 1000)
                times["読み戻し"].append((read - composed) * 1000)
                times["色変換"].append((converted - read) * 1000)
                times["エンコード"].append((encoded - converted) * 1000)
                times["mux"].append((muxed - encoded) * 1000)
        # 溜まっている分の吐き出しも数える エンコーダが数枚遅れて出す作り
        # （NVENC など）では、最後の数枚のエンコードと mux がここに寄る
        # 数えないと「エンコード」と「mux」と「合計」が実際より短く出る
        # 1 枚ごとの段とは混ぜない 混ぜると、その段の中央値が 1 回だけの値で動く
        # 閉じる所（MP4 なら moov の書き出し）まで数える ここも書き出しの一部で、
        # 抜くと長い動画ほど合計が実際より短く出る
        started = time.perf_counter()
        packets = video.encode(None)
        container.mux(packets)
        # 旗は閉じる**前**に立てる 閉じるのに失敗したとき、後始末の側がもう 1 度
        # 閉じにいくと、本当の失敗が閉じ直しの例外で隠れる
        closed = True
        container.close()
        times["吐き出し"].append((time.perf_counter() - started) * 1000)
    finally:
        try:
            # 途中で投げたときだけ閉じる 上で閉じた後にもう 1 度閉じない
            if not closed:
                container.close()
        finally:
            # 閉じるのに失敗しても、GPU とデコーダは必ず手放す
            # ここを飛ばすと、測るたびにテクスチャとデコーダが残る
            renderer.close()
            if readback is not None:
                with context:
                    readback.release()
    return times


def decode_by_layers(
    sources: list[Path],
    width: int,
    height: int,
    frames: int,
    context: OffscreenGLContext,
    threads: int,
) -> None:
    """レイヤー数ごとの合成時間 1 枚ずつ増やして、増え方を見る"""
    print("\nレイヤー数ごとの合成（読み戻しを含まない GPU 合成 デコードはこの中）")
    previous = 0.0
    for count in range(1, len(sources) + 1):
        project = _project(sources[:count], width, height, frames + 2)
        renderer = FrameRenderer(
            project, context=context, quality=FULL_QUALITY, decode_threads=threads
        )
        number = [0]
        try:

            def step(
                renderer: FrameRenderer = renderer,
                project: Project = project,
                number: list[int] = number,
            ) -> None:
                # 毎回同じフレームを描くと、2 枚目以降がデコーダの手前の絵で済んでしまい
                # デコードの重さが消える 1 枚ずつ進めて実際の書き出しに近づける
                number[0] += 1
                with context:
                    renderer.compose(number[0] % max(project.duration, 1))
                    GL.glFinish()

            times = _measure(step, frames)
        finally:
            renderer.close()
        median = statistics.median(times)
        delta = f"  (+{median - previous:5.2f} ms)" if previous else ""
        print(f"  {count} 枚  {_summary(times)}{delta}")
        previous = median


def gil_release(sources: list[Path], frames: int) -> None:
    """PyAV のデコードが GIL を解放するか 直列と、スレッドで並べた場合を比べる

    解放していなければ並べても合計は変わらない（むしろ切り替えの分だけ増える）
    その場合、レイヤーごとの並列デコードは効かないので入れない
    """
    print("\nPyAV のデコードが GIL を解放するか（別々の素材を同じ枚数だけデコードする）")
    if len(sources) < 2:
        # 1 本だとスレッドも 1 本で、並べる相手がいない 倍率は必ず 1 前後になり、
        # 解放しているかどうかの答えにならない 出すと嘘になるので測らない
        print("  素材が 1 本なので並べる相手がいない（--layers 2 以上で測る）")
        return

    def decode_one(path: Path) -> None:
        with VideoDecoder(path) as decoder:
            for number in range(frames):
                decoder.frame_at(Fraction(number, 30))

    started = time.perf_counter()
    for path in sources:
        decode_one(path)
    serial = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    # join() はスレッドの中の例外を投げ直さない デコードに失敗しても気付かず、
    # 中途半端な時間から倍率を出して判定まで表示してしまう result() で投げ直させる
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        for future in [pool.submit(decode_one, path) for path in sources]:
            future.result()
    parallel = (time.perf_counter() - started) * 1000

    speedup = serial / parallel if parallel else 0.0
    verdict = "解放している（並列化が効く）" if speedup >= 1.3 else "解放していない（効かない）"
    print(
        f"  {len(sources)} 本 × {frames} 枚  直列 {serial:7.1f} ms  "
        f"並列 {parallel:7.1f} ms  {speedup:.2f} 倍  {verdict}"
    )


def _frames_of(path: Path, limit: int) -> list[np.ndarray]:
    """書き出したファイルのフレーム ``limit`` を 1 枚超えた所で止める

    ちょうど ``limit`` 枚で止めると、全部のファイルが同じだけ足りなくても
    同じだけ多くても「同じ絵」と出てしまう 1 枚多く読んで枚数まで比べる
    """
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) > limit:
                break
    return frames


def compare_pipeline(
    project: Project,
    codec: str,
    frames: int,
    directory: Path,
    depths: list[int],
    threads: int,
    *,
    gpu_available: bool = True,
) -> None:
    """パイプラインの深さと色変換の道（CPU・GPU）を変えて書き出し、時間と中身を比べる

    GPU の道は色が丸めの 1 まで違ってよいので、絵の一致は同じ道どうしで見る

    書き出しは GPU の変換を作れないと黙って CPU で変換する ``gpu_available`` が偽
    （内訳で GPU の変換を作れなかった機械）なら GPU の行を測らない 測ると CPU の時間が
    「色変換 GPU」の行に並び、CPU と同じ速さに見えて比べた意味が無くなる
    書き出しは始める前に試験の絵で GPU と CPU の変換を比べ、食い違った駆動でも CPU に戻る
    こちらはベンチマークの側から見えないので、内訳で GPU を作れても GPU の行が CPU で
    測った時間になることがある GPU の行が CPU の行とほぼ同じなら、それを疑う
    """
    print("\n書き出し全体（パイプラインの深さと色変換の道ごと）")
    for gpu in (False, True):
        if gpu and not gpu_available:
            print("  色変換 GPU: この機械では GPU の変換を作れない")
            continue
        _compare_depths(project, codec, frames, directory, depths, threads, gpu=gpu)


def _compare_depths(
    project: Project,
    codec: str,
    frames: int,
    directory: Path,
    depths: list[int],
    threads: int,
    *,
    gpu: bool,
) -> None:
    reference: list[np.ndarray] | None = None
    way = "GPU" if gpu else "CPU"
    for depth in depths:
        path = directory / f"out-{way}-depth{depth}.mp4"
        settings = ExportSettings(
            path=path,
            video_codec=codec,
            frame_range=(0, frames),
            pipeline_depth=depth,
            decode_threads=threads,
            gpu_color=gpu,
        )
        started = time.perf_counter()
        export_project(project, settings)
        elapsed = (time.perf_counter() - started) * 1000
        produced = _frames_of(path, frames)
        if len(produced) != frames:
            # 頼んだ枚数で出ていない 絵を比べる前にここで出す
            # 全部同じだけ欠けていると、絵の比べ方では「同じ」に見えてしまう
            same = f"**{len(produced)} 枚しか出ていない（頼んだのは {frames} 枚）**"
        elif reference is None:
            reference = produced
            same = "基準"
        else:
            same = (
                "同じ絵"
                if all(np.array_equal(a, b) for a, b in zip(produced, reference, strict=True))
                else "**絵が違う**"
            )
        label = f"色変換 {way} " + ("直列" if depth == 0 else f"深さ {depth}")
        print(f"  {label:<16} {elapsed:8.1f} ms  ({elapsed / frames:6.2f} ms/枚)  {same}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=60, help="測るフレーム数")
    parser.add_argument("--layers", type=int, default=3, help="重ねる素材の枚数")
    parser.add_argument("--codec", default=None, help="使う映像コーデック 既定は先頭の候補")
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=DEFAULT_DECODE_THREADS,
        help=f"レイヤーごとの並列デコードのスレッド数 1 から {MAX_DECODE_THREADS} まで"
        " 1 で並べない",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="書き出しそのものを、パイプラインの深さを変えて測って比べる",
    )
    arguments = parser.parse_args(argv)
    for name in ("width", "height", "frames", "layers"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name} は 1 以上にしてください")
    # 並列数だけは上限も見る レンダラは受け取った値を上限で切り詰めるので、
    # 素通しにすると「999 本で測った」と表示しながら実際は 8 本で測ることになる
    # `--compare` を付けた場合は export_project が同じ値を断り、測る前に落ちる
    if not 1 <= arguments.decode_threads <= MAX_DECODE_THREADS:
        parser.error(f"--decode-threads は 1 から {MAX_DECODE_THREADS} までにしてください")

    if shutil.which("ffmpeg") is None:
        print("ffmpeg が無いので測れない")
        return 0
    codecs = available_video_codecs()
    if not codecs:
        print("使える映像コーデックが無いので測れない")
        return 0
    codec = arguments.codec or codecs[0]
    threads = arguments.decode_threads

    seconds = max(2.0, (arguments.frames + 4) / 30)
    # 層ごとに別のファイルにする 同じファイルだとレンダラがデコーダを使い回し、
    # 重ねた分のデコードが 1 回で済んでしまう
    made = [
        _make_source(
            _base, width=arguments.width, height=arguments.height, seconds=seconds, tag=f"s{index}"
        )
        for index in range(arguments.layers)
    ]
    if any(path is None for path in made):
        # 手元の ffmpeg に libx264 が無い（または testsrc2 が無い）ときにここへ来る
        # 落とさずに終える 測れない環境で「壊れた」と読まれないように
        print("ffmpeg で素材を作れないので測れない（libx264 が入っているか見る）")
        return 0
    sources = [path for path in made if path is not None]

    try:
        context = OffscreenGLContext()
    except GLContextError as error:
        print(f"OpenGL コンテキストを作れないので測れない: {error}")
        return 0

    print(
        f"素材 {arguments.width}x{arguments.height} testsrc2 30fps を {arguments.layers} 枚重ね / "
        f"{arguments.frames} 枚 / コーデック {codec} / "
        f"デコードの並列 {threads} 本"
    )
    project = _project(sources, arguments.width, arguments.height, arguments.frames + 2)
    # 色変換の道ごとの 1 枚の時間 GPU の変換を作れなかった道は入らない（--compare が見る）
    totals: dict[str, float] = {}
    try:
        for gpu in (False, True):
            way = "GPU" if gpu else "CPU"
            times = breakdown(
                project,
                context,
                codec,
                arguments.frames,
                _base / f"breakdown-{way}.mp4",
                threads,
                gpu=gpu,
            )
            if times is None:
                print(f"\n1 フレームの内訳（色変換 {way}）: この機械では GPU の変換を作れない")
                continue
            print(f"\n1 フレームの内訳（色変換 {way}）")
            total = sum(sum(values) for values in times.values())
            for name, values in times.items():
                share = sum(values) / total * 100 if total else 0.0
                print(f"  {name:<8} {_summary(values)}  {share:5.1f} %")
            totals[way] = total / arguments.frames
            print(f"  {'合計':<8} {totals[way]:6.2f} ms/枚")
        if len(totals) == 2:
            print(f"\n色変換を GPU へ移すと {totals['CPU'] / totals['GPU']:.2f} 倍")
        decode_by_layers(
            sources, arguments.width, arguments.height, arguments.frames, context, threads
        )
    finally:
        context.release()

    gil_release(sources, min(arguments.frames, 30))
    if arguments.compare:
        compare_pipeline(
            project,
            codec,
            arguments.frames,
            _base,
            [0, 2, 4],
            threads,
            gpu_available="GPU" in totals,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
