r"""1 フレームの合成の速さを測る

    .venv\Scripts\python.exe tools\bench_render.py
    .venv\Scripts\python.exe tools\bench_render.py --width 3840 --height 2160 --depth 6

性能目標「1080p のプレビューが 30fps」（1 フレーム 33.3ms 以内）と、
深く入れ子にしたシーンや大量のクリップでどこまで持つかを見る 測らずに直さない、の測る側

測るのは**プレビューと同じ道**（合成と画面への転送） ``--readback`` を付けると
書き出しと同じ道（GPU から CPU へ読み戻す）になる 4K ではこの読み戻しだけで
10ms ほど掛かるので、混ぜるとプレビューの速さを見誤る

どの場面でも、そのうち**入れ物の走査**（絵の α から色の付いた範囲を探す
``_content_box``）に掛かった分を別に出す エフェクトを積んだクリップは描くたびに
これを求めるので、4K では 1 枚ごとに 10ms 近く掛かっていた（Issue #128）

動画の場面（``--videos``）は、ffmpeg で作った素材をエフェクト無しと有りで重ねる
ffmpeg が無ければ動画の場面だけ飛ばす

GPU が無い環境では作れないので、何も測らずに終わる（終了コード 0）
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
from pathlib import Path

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 開発者本人の設定やキャッシュに触らない
_base = Path(tempfile.mkdtemp(prefix="sashimono-bench-"))
# 終わったら捨てる 動画の場面は 4K の素材を置くので、回すたびに残すと溜まる
atexit.register(shutil.rmtree, _base, True)
os.environ["APPDATA"] = str(_base / "roaming")
os.environ["LOCALAPPDATA"] = str(_base / "local")

from OpenGL import GL  # noqa: E402

from sashimono.core.commands import (  # noqa: E402
    AddClip,
    AddEffect,
    AddMedia,
    AddScene,
    AddTrack,
    Command,
    InScene,
    insert_scene,
    new_scene,
)
from sashimono.core.model import (  # noqa: E402
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    Project,
    ProjectSettings,
    Stroke,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate  # noqa: E402
from sashimono.effects.definition import registry  # noqa: E402
from sashimono.effects.sources import TEXT  # noqa: E402
from sashimono.effects.spec import ParamInput  # noqa: E402
from sashimono.effects.strokes import STROKE  # noqa: E402
from sashimono.engine.decode import probe_media  # noqa: E402
from sashimono.engine.gpu import (  # noqa: E402
    Framebuffer,
    GLContextError,
    OffscreenGLContext,
)
from sashimono.engine.render import FrameRenderer  # noqa: E402
from sashimono.engine.render import renderer as renderer_module  # noqa: E402

#: 30fps の 1 コマ（ミリ秒） プレビューの目標
BUDGET_MS = 1000 / 30

#: 画面へ出す先の大きさ プレビューの枠は画面の実寸で、素材の大きさではない
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1920, 1080


class ScanMeter:
    """入れ物の走査（``renderer._content_box``）に掛かった時間を足し上げる

    関数を差し替えて測る 走査はクリップを描く道のあちこち（エフェクトの範囲・
    スクリプトへ渡す大きさ）から呼ばれるので、呼ぶ側で囲むと取りこぼす
    """

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        #: 差し替える前の関数 差し替えている間だけ持つ
        self._original: Callable[[np.ndarray], tuple[int, int, int, int] | None] | None = None

    def install(self) -> None:
        # 差し替えたまま包み直すと、1 回の走査を内と外のラッパーが二重に足す
        # 同じプロセスで ``main`` を 2 回呼んだときに内訳が実測より大きくなる
        if self._original is not None:
            return
        original = renderer_module._content_box
        self._original = original

        def timed(image: np.ndarray) -> tuple[int, int, int, int] | None:
            started = time.perf_counter()
            try:
                return original(image)
            finally:
                self.elapsed_ms += (time.perf_counter() - started) * 1000

        renderer_module._content_box = timed

    def uninstall(self) -> None:
        """差し替えた関数を元に戻す 測り終えたら戻して、ほかの呼び出しを測りに巻き込まない"""
        if self._original is not None:
            # 持っておいた元の関数は Callable として持つので、def の型の属性へは代入として通らない
            # 名前の表へ直に戻す（setattr は ruff が代入へ書き換える）
            vars(renderer_module)["_content_box"] = self._original
            self._original = None

    def take(self) -> float:
        """ここまでの分を返して 0 へ戻す"""
        elapsed, self.elapsed_ms = self.elapsed_ms, 0.0
        return elapsed


#: 走査の時間を数える 1 つだけ ``main`` で差し替える
SCAN = ScanMeter()


def _shape(colour: tuple[float, float, float, float], size: float) -> GeneratedSource:
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "rect",
            "color": colour,
            "width": AnimatedValue(size),
            "height": AnimatedValue(size),
        },
    )


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def nested_project(settings: ProjectSettings, depth: int, length: int) -> Project:
    """シーンを ``depth`` 段の入れ子にしたプロジェクト 一番奥に図形を 1 つ置く"""
    project = Project.create(settings)
    scenes = []
    for level in range(depth):
        scene = new_scene(project, f"階層{level + 1}")
        project = AddScene(scene).apply(project)
        track = Track(TrackKind.VIDEO, f"S{level + 1}")
        project = InScene(scene.id, AddTrack(track)).apply(project)
        scenes.append((scene.id, track.id))

    innermost = Clip(timeline_start=0, duration=length, source=_shape((1.0, 1.0, 1.0, 1.0), 200))
    scene_id, track_id = scenes[-1]
    project = InScene(scene_id, AddClip(track_id, innermost)).apply(project)
    # 奥から手前へ、1 段ずつ親のシーンへ置いていく
    for (outer_id, outer_track), (inner_id, _) in zip(
        reversed(scenes[:-1]), reversed(scenes[1:]), strict=True
    ):
        placed = Clip(timeline_start=0, duration=length, scene_id=inner_id)
        project = InScene(outer_id, AddClip(outer_track, placed)).apply(project)
    return _apply(project, insert_scene(project, scenes[0][0], at_frame=0, duration=length))


def busy_project(settings: ProjectSettings, tracks: int, effects: int, length: int) -> Project:
    """トラックを ``tracks`` 本重ね、それぞれのクリップへエフェクトを ``effects`` 個積む"""
    project = Project.create(settings)
    blur = registry.get("blur")
    glow = registry.get("glow")
    assert blur is not None and glow is not None
    for index in range(tracks):
        track = Track(TrackKind.VIDEO, f"V{index + 1}")
        project = AddTrack(track).apply(project)
        clip = Clip(
            timeline_start=0,
            duration=length,
            source=_shape((0.2 + index * 0.05, 0.4, 0.9, 1.0), 200 + index * 20),
        )
        project = AddClip(track.id, clip).apply(project)
        for number in range(effects):
            definition = blur if number % 2 == 0 else glow
            project = AddEffect(clip.id, definition.create()).apply(project)
    return project


def make_video(directory: Path, width: int, height: int, seconds: float) -> Path | None:
    """測る用の動画を ffmpeg で作る 作れなければ ``None``

    動きのある絵にして、圧縮で楽をさせない 外の ffmpeg に libx264 が
    入っているかは分からないので、落とさずに動画の場面だけ飛ばす
    """
    if shutil.which("ffmpeg") is None:
        return None
    path = directory / f"source-{width}x{height}.mp4"
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
        # yuv420p は色を縦横とも 2 画素ずつまとめるので、奇数の大きさでは焼けない
        # そのときは色を間引かない yuv444p にして、指定どおりの大きさで測る
        "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p",
        str(path),
    ]
    # 組み立てているのは固定の文字列と argparse が受けた数値、一時フォルダの中の
    # パスだけで、外から来る文字列は混ざらない shell は通さない（list 渡し）
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if subprocess.run(command, check=False).returncode != 0:
        return None
    return path


def video_project(
    settings: ProjectSettings, source: Path, count: int, effects: int, length: int
) -> Project:
    """同じ動画を ``count`` 本重ね、それぞれへエフェクトを ``effects`` 個積む

    どれも少し薄くして、下の絵も描かせる 不透明のまま重ねると、
    隠れた分を飛ばす作りになったときに重ねた意味が無くなる
    """
    project = Project.create(settings)
    media = probe_media(source)
    project = AddMedia(media).apply(project)
    blur = registry.get("blur")
    glow = registry.get("glow")
    assert blur is not None and glow is not None
    for index in range(count):
        track = Track(TrackKind.VIDEO, f"V{index + 1}")
        project = AddTrack(track).apply(project)
        clip = Clip(
            timeline_start=0,
            duration=length,
            media_id=media.id,
            opacity=AnimatedValue(0.7),
        )
        project = AddClip(track.id, clip).apply(project)
        for number in range(effects):
            definition = blur if number % 2 == 0 else glow
            project = AddEffect(clip.id, definition.create()).apply(project)
    return project


#: グラデーションの場面の止まり（金属風の 4 色） 字の塗り・影・縁取りの層のどれにも使う（#312）
_METAL = (
    *(0.0, 1.0, 0.97, 0.8, 1.0),
    *(0.45, 0.85, 0.65, 0.2, 1.0),
    *(0.55, 0.55, 0.38, 0.05, 1.0),
    *(1.0, 1.0, 0.9, 0.5, 1.0),
)


def text_project(
    settings: ProjectSettings,
    count: int,
    length: int,
    *,
    gradient: bool = False,
    moving: bool = False,
) -> Project:
    """テキストを ``count`` 本重ねたプロジェクト 影とぼかし・縁取りの層を 1 つ持つ

    ``gradient`` なら塗り・影・縁取りの層をどれもグラデーションにする（#312） 偽なら同じ影と
    層を単色で持つ（グラデーションの分だけの差を見るため）
    ``moving`` なら大きさをキーフレームで動かし、毎フレーム字を組み直して塗らせる
    （止まった字は覚えた絵を重ねるだけなので、塗りの手間はここでしか測れない）
    """
    project = Project.create(settings)
    kind = "linear" if gradient else "solid"
    look: dict[str, ParamInput] = {
        "fill_type": kind,
        "fill_gradient": _METAL,
        "shadow_x": 4.0,
        "shadow_y": -4.0,
        "shadow_blur": 3.0,
        "shadow_fill_type": kind,
        "shadow_fill_gradient": _METAL,
    }
    layer = Stroke(
        params={
            **STROKE.default_params(),
            "width": AnimatedValue(4.0),
            "fill_type": kind,
            "fill_gradient": _METAL,
            "fill_angle": AnimatedValue(180.0),
        }
    )
    for index in range(count):
        track = Track(TrackKind.VIDEO, f"T{index + 1}")
        project = AddTrack(track).apply(project)
        source = TEXT.create(text=f"字幕 {index + 1}", size=48, pos_y=index * 30 - 200, **look)
        if moving:
            size = AnimatedValue(48.0, (Keyframe(0, 48.0), Keyframe(length - 1, 64.0)))
            source = source.with_param("size", size)
        source = source.with_strokes((layer,))
        clip = Clip(timeline_start=0, duration=length, source=source)
        project = AddClip(track.id, clip).apply(project)
    return project


def measure(
    project: Project, context: OffscreenGLContext, frames: int, *, readback: bool = False
) -> tuple[list[float], list[float]]:
    """1 フレームにかかる時間と、そのうち入れ物の走査の分（どちらもミリ秒）
    1 回目は温まっていないので捨てる

    測るのは既定で**プレビューと同じ道** 合成（``compose``）と、その結果を
    画面へ出す所（``present``）を測る 出す先は自前の 1920x1080 の描画先で、
    オフスクリーンの既定（0 番）だと大きさが環境任せになる
    GL の命令は投げただけでは終わっていないので、1 枚ごとに ``glFinish`` で
    終わりを待つ 待たないと、投げるのに掛かった時間を測るだけになる

    ``readback`` を真にすると**書き出しと同じ道** 書き出し
    （:mod:`sashimono.engine.encode.exporter`）と同じく、1 枚ごとに
    ``renderer.render`` を呼ぶ（中でコンテキストを取る） ``glFinish`` は
    入れない ``glReadPixels`` が終わりを待つので、足すと二重に待つ形になり
    実態より重く出る
    """
    renderer = FrameRenderer(project, context=context)
    times: list[float] = []
    scans: list[float] = []
    try:
        if readback:
            for frame in range(frames + 1):
                SCAN.take()
                started = time.perf_counter()
                # 書き出しと同じ呼び方 コンテキストは render の中で取る
                renderer.render(frame % max(project.duration, 1))
                elapsed = (time.perf_counter() - started) * 1000
                if frame:
                    times.append(elapsed)
                    scans.append(SCAN.take())
            return times, scans

        with context:
            screen = Framebuffer(PREVIEW_WIDTH, PREVIEW_HEIGHT, internal_format=GL.GL_RGBA8)
            try:
                for frame in range(frames + 1):
                    SCAN.take()
                    started = time.perf_counter()
                    renderer.compose(frame % max(project.duration, 1))
                    renderer.compositor.present(
                        screen.handle, (0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT)
                    )
                    GL.glFinish()
                    elapsed = (time.perf_counter() - started) * 1000
                    if frame:
                        times.append(elapsed)
                        scans.append(SCAN.take())
            finally:
                screen.release()
    finally:
        renderer.close()
    return times, scans


def percentile95(times: list[float]) -> float:
    """95 パーセンタイル 標本の外側へ外挿しない（inclusive）

    既定の exclusive は、標本が少ないと一番大きい値より外へ出た数を返す
    ここは 20 回前後の測定なので、実際には出ていない値を予算と比べることになる
    """
    if len(times) < 2:
        return max(times)
    return statistics.quantiles(times, n=20, method="inclusive")[18]


def report(name: str, measured: tuple[list[float], list[float]]) -> bool:
    times, scans = measured
    worst = max(times)
    p95 = percentile95(times)
    ok = p95 <= BUDGET_MS
    print(
        f"{name:<24} 中央 {statistics.median(times):7.2f} ms  95% {p95:7.2f} ms  "
        f"最悪 {worst:7.2f} ms  うち入れ物の走査 {statistics.median(scans):6.2f} ms  "
        f"{'OK' if ok else '予算超え'}"
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--depth", type=int, default=6, help="シーンの入れ子の深さ")
    parser.add_argument("--tracks", type=int, default=20, help="重ねるトラックの数")
    parser.add_argument("--effects", type=int, default=2, help="1 クリップのエフェクト数")
    parser.add_argument("--texts", type=int, default=20, help="重ねるテキストの数")
    parser.add_argument(
        "--videos", type=int, default=3, help="重ねる動画の数 0 で動画の場面を飛ばす"
    )
    parser.add_argument("--frames", type=int, default=20, help="測るフレーム数")
    parser.add_argument(
        "--readback",
        action="store_true",
        help="書き出しと同じ道（GPU から CPU へ読み戻す）を測る 既定はプレビューと同じ道",
    )
    arguments = parser.parse_args(argv)

    for name in ("width", "height", "depth", "tracks", "effects", "texts", "frames"):
        # 0 や負の値は「測れた」と言いながら何も測らない
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name} は 1 以上にしてください")
    if arguments.videos < 0:
        parser.error("--videos は 0 以上にしてください")
    settings = ProjectSettings(
        width=arguments.width, height=arguments.height, frame_rate=FrameRate(30)
    )
    length = max(arguments.frames + 1, 30)
    try:
        context = OffscreenGLContext()
    except GLContextError as error:
        print(f"OpenGL コンテキストを作れないので測れない: {error}")
        return 0

    if arguments.readback:
        path_name = f"書き出しと同じ道（{arguments.width}x{arguments.height} を CPU へ読み戻す）"
    else:
        path_name = f"プレビューと同じ道（{PREVIEW_WIDTH}x{PREVIEW_HEIGHT} の画面へ出す）"
    print(
        f"プロジェクト {arguments.width}x{arguments.height} / 目標 {BUDGET_MS:.1f} ms / {path_name}"
    )
    if arguments.readback:
        # 予算はプレビューのもの 書き出しは実時間で動く必要が無いので、
        # ここでの「予算超え」は速さの比べ方であって、不合格ではない
        # 合否にも混ぜない 混ぜると、正常な測定で終了コードが 1 になる
        print("  （目標はプレビューの値 書き出しは実時間で動かなくてよい 比べるための表示）")
    SCAN.install()
    passed = True
    try:
        for depth in range(1, arguments.depth + 1):
            project = nested_project(settings, depth, length)
            passed &= report(
                f"シーン {depth} 段",
                measure(project, context, arguments.frames, readback=arguments.readback),
            )
        project = busy_project(settings, arguments.tracks, arguments.effects, length)
        passed &= report(
            f"{arguments.tracks} 本 × エフェクト {arguments.effects}",
            measure(project, context, arguments.frames, readback=arguments.readback),
        )
        # グラデーション（#312）の有る無しを、止まった字と毎フレーム組み直す字で並べる
        for moving in (False, True):
            for gradient in (False, True):
                project = text_project(
                    settings, arguments.texts, length, gradient=gradient, moving=moving
                )
                name = f"テキスト {arguments.texts} 本 影と縁あり"
                name += " グラデーション" if gradient else " 単色"
                if moving:
                    name += " 大きさが動く"
                passed &= report(
                    name, measure(project, context, arguments.frames, readback=arguments.readback)
                )
        if arguments.videos:
            source = make_video(_base, arguments.width, arguments.height, length / 30 + 1)
            if source is None:
                print("ffmpeg（libx264）で素材を作れないので、動画の場面は測らない")
            else:
                for effects in (0, arguments.effects):
                    project = video_project(settings, source, arguments.videos, effects, length)
                    passed &= report(
                        f"動画 {arguments.videos} 本 × エフェクト {effects}",
                        measure(project, context, arguments.frames, readback=arguments.readback),
                    )
    finally:
        SCAN.uninstall()
        context.release()
    # 読み戻しの道は比べるための表示 予算はプレビューのものなので合否に使わない
    return 0 if passed or arguments.readback else 1


if __name__ == "__main__":
    sys.exit(main())
