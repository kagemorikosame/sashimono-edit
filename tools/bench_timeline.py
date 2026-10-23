r"""タイムラインの描画の速さを測る

    .venv\Scripts\python.exe tools\bench_timeline.py
    .venv\Scripts\python.exe tools\bench_timeline.py --clips 5000 --tracks 10
    .venv\Scripts\python.exe tools\bench_timeline.py --media --clips 100

性能目標「スクロール / ズームが 60fps」（1 回の描画が 16.7ms 以内）を確かめる
測らずに最適化しない、の測る側

ウィジェットを画面に出さず、同じ大きさの画像へ描かせて時間を取る 画面への転送は
Qt と GPU の仕事で、ここで見たいのは自前の描画（クリップを並べる計算と塗り）

``--media`` を付けると、テキストの代わりに実素材（ffmpeg の ``testsrc`` の動画と
``sine`` の音）を並べ、サムネイルと波形の解析を済ませてから測る テキストだけでは
サムネイルと波形を描く道を通らない 重い所を分けるため、同じ並びでサムネイルだけ・
波形だけ・どちらも描かないときの速さも並べて出す 波形は画像にして貯めてあるので、
貯めた物を毎回捨てて作り直すとき（倍率を変えた直後の 1 回）の速さも出す
"""

from __future__ import annotations

import argparse
import io
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 開発者本人の設定やキャッシュに触らない
_base = Path(tempfile.mkdtemp(prefix="sashimono-bench-"))
os.environ["APPDATA"] = str(_base / "roaming")
os.environ["LOCALAPPDATA"] = str(_base / "local")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sashimono.core.model import Clip, MediaItem, Project, Track, TrackKind  # noqa: E402
from sashimono.effects.sources import TEXT  # noqa: E402
from sashimono.engine.audio.waveform import Waveform  # noqa: E402
from sashimono.engine.cache import MediaAnalyzer  # noqa: E402
from sashimono.engine.cache.store import CacheStore  # noqa: E402
from sashimono.engine.cache.thumbnails import Filmstrip  # noqa: E402
from sashimono.engine.decode.probe import probe_media  # noqa: E402
from sashimono.ui.timeline import TimelineView  # noqa: E402
from sashimono.ui.timeline.layout import TimelineLayout  # noqa: E402
from sashimono.ui.timeline.painter import clear_waveform_images  # noqa: E402

#: 60fps の 1 コマ（ミリ秒）
BUDGET_MS = 1000 / 60
#: 並べる素材の長さの下限（秒） クリップより短いと素材の外を指すクリップになる
MEDIA_SECONDS = 10
#: 素材の大きさ サムネイルの高さは決まっているので、大きくしても描く速さは変わらず
#: 作るのと解析に時間が掛かるだけ
MEDIA_SIZE = "1280x720"
#: 解析を待つ上限（秒） 10 秒の素材 2 本なら数秒で済む 止まったら測らずに知らせる
ANALYZE_TIMEOUT = 120.0
#: ffmpeg で素材を 1 本作るのを待つ上限（秒） 10 秒の素材なら 1 秒ほどで済む 応答しない
#: ffmpeg に当たると、上限が無ければ道具も試験もいつまでも戻らない
FFMPEG_TIMEOUT = 120.0


class PlainAnalyzer(MediaAnalyzer):
    """済んだ解析（``source``）のうち、選んだ物だけを返す解析 既定はどちらも返さない

    同じ並びを、サムネイルだけ・波形だけ・どちらも無しで測れば、どちらを描く所が
    重いのかを分けられる 自分では解析しない（頼まれても何もしない）
    """

    def __init__(
        self,
        store: CacheStore,
        source: MediaAnalyzer | None = None,
        *,
        filmstrips: bool = False,
        waveforms: bool = False,
    ) -> None:
        super().__init__(store)
        self._source = source
        self._show_filmstrips = filmstrips
        self._show_waveforms = waveforms

    def filmstrip(self, media: MediaItem, stream: int | None = None) -> Filmstrip | None:
        if self._source is None or not self._show_filmstrips:
            return None
        return self._source.filmstrip(media, stream)

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        if self._source is None or not self._show_waveforms:
            return None
        return self._source.waveform(media, stream)


def make_media(directory: Path, seconds: int = MEDIA_SECONDS) -> tuple[MediaItem, MediaItem]:
    """ffmpeg で testsrc の動画と sine の音を作って読む（動画, 音）

    配布物や手元の素材を使わないのは、誰の機械でも同じ物で測れるようにするため
    ffmpeg に libx264 が無いと ``subprocess.CalledProcessError`` になる
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg が見つからない 実素材を作れない")
    directory.mkdir(parents=True, exist_ok=True)
    video, audio = directory / "bench-video.mp4", directory / "bench-audio.m4a"
    quiet = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(
        [
            *quiet,
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={MEDIA_SIZE}:rate=30:duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT,
    )
    subprocess.run(
        [
            *quiet,
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}:sample_rate=48000",
            "-ac",
            "2",
            "-c:a",
            "aac",
            str(audio),
        ],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT,
    )
    return probe_media(video), probe_media(audio)


def analyze(
    analyzer: MediaAnalyzer, media: Iterable[MediaItem], timeout: float = ANALYZE_TIMEOUT
) -> None:
    """サムネイルと波形の解析を頼み、済むまで待つ

    済む前に測ると、中身の無いクリップの速さを測ったことになる 解析に失敗した素材は
    いつまでも揃わないので、待つ上限を超えたら ``TimeoutError``
    """
    items = list(media)
    for item in items:
        analyzer.request(item)

    def ready(item: MediaItem) -> bool:
        return (not item.has_video or analyzer.filmstrip(item) is not None) and (
            not item.has_audio or analyzer.waveform(item) is not None
        )

    deadline = time.monotonic() + timeout
    while not all(ready(item) for item in items):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{timeout:.0f} 秒待ってもサムネイルと波形が揃わない")
        time.sleep(0.05)


def build_project(
    clips: int, tracks: int, length: int, *, media: tuple[MediaItem, MediaItem] | None = None
) -> Project:
    """クリップを隙間なく並べたプロジェクト 映像と音声を半分ずつ

    ``media``（動画, 音）を渡すと、映像のトラックには動画を、音声のトラックには音を指す
    クリップを並べる 渡さなければテキスト 素材を指すクリップは頭を 1/100 秒ずつずらす
    （1 秒未満） 全部を同じ所から始めると、波形の画像がどのクリップでも同じになり、
    1 枚作れば残りは貯めた物を貼るだけになって、実際のプロジェクトより速く出る
    """
    # 0 は割り算で落ち、負はクリップを 1 本も作らないまま「測れた」と言ってしまう
    for name, value in (("clips", clips), ("tracks", tracks), ("length", length)):
        if value <= 0:
            raise ValueError(f"{name} は 1 以上にしてください: {value}")
    # 余りも配る 切り捨てると、表示した本数より少ない数で測ったことになる
    per_track, remainder = divmod(clips, tracks)
    built: list[Track] = []
    for index in range(tracks):
        kind = TrackKind.VIDEO if index % 2 == 0 else TrackKind.AUDIO
        row = tuple(
            _clip(
                kind, n * length, length, media, Fraction((index * (per_track + 1) + n) % 100, 100)
            )
            for n in range(per_track + (1 if index < remainder else 0))
        )
        built.append(Track(kind, f"{'V' if kind is TrackKind.VIDEO else 'A'}{index + 1}", row))
    base = Project.create(media=media or ())
    return base.with_timeline(replace(base.timeline, tracks=tuple(built)))


def _clip(
    kind: TrackKind,
    start: int,
    length: int,
    media: tuple[MediaItem, MediaItem] | None,
    source_in: Fraction,
) -> Clip:
    if media is None:
        return Clip(timeline_start=start, duration=length, source=TEXT.create())
    video, audio = media
    if kind is TrackKind.VIDEO:
        return Clip(
            timeline_start=start,
            duration=length,
            media_id=video.id,
            stream_index=video.video_streams[0].index,
            source_in=source_in,
        )
    return Clip(
        timeline_start=start,
        duration=length,
        media_id=audio.id,
        stream_index=audio.audio_streams[0].index,
        source_in=source_in,
    )


def measure(
    view: TimelineView, layouts: list[TimelineLayout], repeat: int, *, fresh: bool = False
) -> list[float]:
    """各表示状態で描いた時間（ミリ秒） 1 回目は温まっていないので捨てる

    ``fresh`` が真なら、描くたびに貯めた波形の画像を捨てる 倍率を変えた直後の 1 回や、
    貯める量を超えて入れ替わるときの重さは、貯めた物を貼るだけの速さからは見えない
    """
    image = QImage(view.size(), QImage.Format.Format_ARGB32_Premultiplied)
    times: list[float] = []
    for layout in layouts:
        view._layout = layout
        for attempt in range(repeat + 1):
            if fresh:
                clear_waveform_images()
            painter = QPainter(image)
            started = time.perf_counter()
            view.render(painter, QPoint())
            elapsed = (time.perf_counter() - started) * 1000
            painter.end()
            if attempt:
                times.append(elapsed)
    return times


def report(name: str, times: list[float]) -> bool:
    worst = max(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else worst
    ok = p95 <= BUDGET_MS
    print(
        f"{name:<14} 中央 {statistics.median(times):6.2f} ms  95% {p95:6.2f} ms  "
        f"最悪 {worst:6.2f} ms  {'OK' if ok else '予算超え'}"
    )
    return ok


def run_states(
    view: TimelineView, project: Project, width: int, repeat: int, *, fresh: bool = False
) -> list[bool]:
    """全体表示・スクロール・ズームを測って並べる"""
    duration = project.duration
    fit = TimelineLayout(pixels_per_frame=(width - 132) / max(1, duration))
    steps = 60
    scroll = [TimelineLayout(scroll_frame=duration * n / steps) for n in range(steps)]
    zoom = [TimelineLayout(pixels_per_frame=fit.pixels_per_frame * (1.25**n)) for n in range(40)]
    return [
        report("全体を表示", measure(view, [fit], repeat * 4, fresh=fresh)),
        report("スクロール", measure(view, scroll, repeat, fresh=fresh)),
        report("ズーム", measure(view, zoom, repeat, fresh=fresh)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--clips",
        type=int,
        default=None,
        help="クリップの総数（既定はテキストなら 3000、--media なら 100）",
    )
    parser.add_argument("--tracks", type=int, default=8, help="トラックの数")
    parser.add_argument("--length", type=int, default=90, help="1 クリップの長さ（フレーム）")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--repeat", type=int, default=5, help="1 つの表示状態で描く回数")
    parser.add_argument(
        "--media",
        action="store_true",
        help="ffmpeg で作った動画と音を並べ、サムネイルと波形を描いて測る",
    )
    arguments = parser.parse_args(argv)
    clips = arguments.clips if arguments.clips is not None else (100 if arguments.media else 3000)

    application = QApplication.instance() or QApplication(sys.argv)
    del application
    work = Path(tempfile.mkdtemp(prefix="sashimono-bench-media-", dir=_base))
    analyzer = MediaAnalyzer(CacheStore(work / "cache"), sample_rate=48000, channels=2)
    partial: list[PlainAnalyzer] = []
    try:
        media: tuple[MediaItem, MediaItem] | None = None
        if arguments.media:
            # クリップより短い素材を並べると、素材の外を指すクリップで中身が描かれない
            rate = Project.create().rate
            seconds = max(MEDIA_SECONDS, math.ceil(arguments.length * rate.frame_duration) + 1)
            try:
                media = make_media(work / "media", seconds)
            # TimeoutExpired は OSError の仲間でも CalledProcessError の仲間でもない
            # 受けないと、待つ上限に掛かったときにトレースバックで終わる
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print(f"素材を作れない（ffmpeg と libx264・aac を見てください）: {error}")
                return 2
            started = time.perf_counter()
            try:
                analyze(analyzer, media)
            except TimeoutError as error:
                print(f"測らずに止めた: {error}")
                return 2
            print(f"サムネイルと波形の解析 {time.perf_counter() - started:.1f} 秒")
        project = build_project(clips, arguments.tracks, arguments.length, media=media)
        view = TimelineView(project, analyzer)
        view.resize(arguments.width, arguments.height)

        contents = "動画と音（サムネイルと波形を描く）" if media else "テキスト"
        print(
            f"クリップ {clips} 本（{contents}） / トラック {arguments.tracks} 本 / "
            f"長さ {project.duration} フレーム / 画面 {arguments.width}x{arguments.height}"
        )
        print(f"予算 {BUDGET_MS:.1f} ms（60fps）  判定は 95 パーセンタイル")
        results = run_states(view, project, arguments.width, arguments.repeat)
        if media is not None:
            print("\n波形の画像を貯めずに毎回作るとき（同じ並び 判定には入れない）")
            run_states(view, project, arguments.width, arguments.repeat, fresh=True)
            # 同じ並びで中身の一部だけを描く 全部との差が、描かなかった物を描く分
            for title, filmstrips, waveforms in (
                ("サムネイルだけを描くとき", True, False),
                ("波形だけを描くとき", False, True),
                ("サムネイルと波形を描かないとき", False, False),
            ):
                shown = PlainAnalyzer(
                    CacheStore(work / "plain"),
                    analyzer,
                    filmstrips=filmstrips,
                    waveforms=waveforms,
                )
                partial.append(shown)
                other = TimelineView(project, shown)
                other.resize(arguments.width, arguments.height)
                print(f"\n{title}（同じ並び 判定には入れない）")
                run_states(other, project, arguments.width, arguments.repeat)
    finally:
        analyzer.close()
        for shown in partial:
            shown.close()
    return 0 if all(results) else 1


def run(argv: list[str] | None = None) -> None:
    """命令として走らせる 終了コードを返したら、残ったワーカーを待たずに終わる

    解析のワーカー（ThreadPoolExecutor）は、Python が終わるときに終わるまで待たれる
    ``close()`` の取り消しは塊を読む合間にしか効かず、FFmpeg の 1 回のデコードの中で
    止まったワーカーがいると、解析の上限で終了コード 2 を返してもプロセスが終わらない
    測る道具は後片付けで守る物を持たない（一時フォルダに書くだけ）ので、出力を流して
    すぐに終える
    """
    code = main(argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    run()
