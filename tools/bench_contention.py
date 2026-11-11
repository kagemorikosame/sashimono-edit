r"""裏の処理と画面の操作の取り合いを測る（#443 の段階 1 #440 #442）

    .venv\Scripts\python.exe tools\bench_contention.py
    .venv\Scripts\python.exe tools\bench_contention.py --jobs proxy-thread proxy-process
    .venv\Scripts\python.exe tools\bench_contention.py --size 3840x2160 --rate 30 --seconds 60
    .venv\Scripts\python.exe tools\bench_contention.py --switch-interval 0.0005

裏の処理 × 画面の操作 の表を出す 裏の処理を 1 つずつ走らせ、その間ずっと画面の操作を
繰り返す 見る値は 2 つ

- 裏の処理の速さ（素材の秒数 / 掛かった秒数）を、画面が何もしていないとき（単体）と比べる
- 画面の操作 1 回の時間（操作の後に溜まったイベントを流し終えるまで）を、裏の処理が
  何も走っていないときと比べる

素材は ffmpeg で作る 録画ソフトの既定に多い 250 コマごとのキーフレームの 1080p60 に、
音（正弦波）を付けた物 実際の録画は使わない（誰の機械でも同じ物で測れるように）

画面は出さない（offscreen） 画面の部品は実物（MainWindow）で、クリップは文字 300 本と
映像 2 本 映像のクリップが指す素材は無い場所にしてある 窓が自分で解析や控えを
走らせると、測りたい裏の処理に混ざるため ``--onscreen`` で本物の画面に出す（書き出しの
GL は offscreen では作れないので、書き出しを測るときはこちら）

``--busy-thread`` を付けると、裏の処理を走らせている間（単体のときも）、Python を回し
続けるスレッドを 1 本足す アプリの中で先読みや解析が動いている状態に近い 控えの作成が
アプリの中で単体の 8 分の 1 になった（#440）のは、この取り合いがあるとき

裏の処理

- ``proxy-thread`` 控えをアプリの中のスレッドで作る（0.2.4 までの作り）
- ``proxy-process`` 控えを別のプロセスで作る（#440）
- ``scenes`` 場面の解析 ``thumbnails`` サムネイル ``waveform`` 波形
- ``export`` 書き出し（GL を作れない機械では飛ばす）
- ``python`` Python を回し続けるだけのスレッド 先読みの走り係のように、裏で Python が
  止まらずに動いている場合の上限の目安（先読みそのものは共有した GL が要り、
  画面を出さない測りでは作れない）

文字起こしは測らない 部品とモデルを別に入れる物で、入っている機械でしか動かない

測るたびに数 ms は揺れる 前後を比べるときは同じ日に交互に測る
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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 別のプロセスの子（控えの作成 #440）はこの台本を ``__mp_main__`` として読み直す そこで
# 一時フォルダを作ったり Qt と編集画面を読んだりすると、子の起動に何秒も乗り、別のプロセスの
# 速さが実際のアプリ（子は台本を読まない）より遅く出る 子では飛ばす
_IN_CHILD = __name__ == "__mp_main__"

# 開発者本人の設定に触らない
_base = (
    Path(tempfile.gettempdir())
    if _IN_CHILD
    else Path(tempfile.mkdtemp(prefix="sashimono-contention-bench-"))
)
if not _IN_CHILD:
    atexit.register(shutil.rmtree, _base, True)
    os.environ["APPDATA"] = str(_base / "roaming")
    os.environ["LOCALAPPDATA"] = str(_base / "local")

if not _IN_CHILD:
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, AddMedia, AddTrack, MoveClip  # noqa: E402
from sashimono.core.model import (  # noqa: E402
    Clip,
    ClipId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate  # noqa: E402
from sashimono.effects.sources import TEXT  # noqa: E402
from sashimono.engine.audio.waveform import analyze_waveform  # noqa: E402
from sashimono.engine.cache.proxy import (  # noqa: E402
    create_proxy,
    proxy_codecs,
    proxy_job,
    working_path,
)
from sashimono.engine.cache.thumbnails import build_filmstrip  # noqa: E402
from sashimono.engine.decode import probe_media  # noqa: E402
from sashimono.engine.scenes import analyze_scenes  # noqa: E402
from sashimono.engine.worker import run_in_process  # noqa: E402

if not _IN_CHILD:
    from sashimono.ui.main_window import MainWindow
    from sashimono.ui.workspace import Preferences, PreferenceStore

#: 何もしない間の画面の操作を、何秒ぶん測るか
IDLE_SECONDS = 5.0

#: ``python`` の裏の処理を走らせる長さ（秒） 終わりの無い処理なので区切る
PYTHON_SECONDS = 5.0


@dataclass
class JobResult:
    """裏の処理 1 つの結果"""

    name: str
    #: 単体で走らせたときの速さ（素材の秒数 / 掛かった秒数） 測らない物は ``None``
    alone: float | None
    #: 画面を動かしながら走らせたときの速さ
    contended: float | None
    #: 操作の名前 → 1 回の時間（ms）
    ops: dict[str, list[float]] = field(default_factory=dict)


# --- 素材 ---


def make_source(directory: Path, size: str, rate: int, seconds: int, gop: int) -> Path | None:
    """キーフレームを ``gop`` コマごとにだけ置いた、音つきの素材を作る"""
    path = directory / f"source-{size}-{rate}-{gop}.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={rate}:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-b:v",
        "8M",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    try:
        # 組み立てているのは固定の文字列と argparse が受けた数値、一時フォルダの中の
        # パスだけで、外から来る文字列は混ざらない shell は通さない（list 渡し）
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"素材を作れなかった: {exc}")
        return None
    return path


# --- 裏の処理 ---

Job = Callable[[Path, Path, Callable[[], bool]], float]


def _proxy_thread(source: Path, scratch: Path, stop: Callable[[], bool]) -> float:
    target = scratch / f"proxy-thread-{time.perf_counter_ns()}.mp4"
    _made(create_proxy(source, target, should_cancel=stop))
    return float(probe_media(source).duration)


def _made(result: object) -> None:
    """控えができなかったら投げる 投げないと、使えるエンコーダが無い・変換に失敗した回まで
    素材の長さを返し、何もしなかった速さが速い控えとして表に載る
    """
    if result is None:
        raise RuntimeError("控えを作れなかった（使えるエンコーダが無いか、変換に失敗した）")


def _proxy_process(source: Path, scratch: Path, stop: Callable[[], bool]) -> float:
    target = scratch / f"proxy-process-{time.perf_counter_ns()}.mp4"
    made = run_in_process(
        proxy_job,
        {
            "source": str(source),
            "target": str(target),
            "working": str(working_path(target)),
            "height": 540,
            "stream_index": None,
            "codecs": tuple(proxy_codecs()),
        },
        should_cancel=stop,
    )
    _made(made)
    return float(probe_media(source).duration)


def _scenes(source: Path, _scratch: Path, stop: Callable[[], bool]) -> float:
    analyze_scenes(source, should_cancel=stop)
    return float(probe_media(source).duration)


def _thumbnails(source: Path, _scratch: Path, stop: Callable[[], bool]) -> float:
    build_filmstrip(source, should_cancel=stop)
    return float(probe_media(source).duration)


def _waveform(source: Path, _scratch: Path, stop: Callable[[], bool]) -> float:
    analyze_waveform(source, should_cancel=stop)
    return float(probe_media(source).duration)


def _export(source: Path, scratch: Path, stop: Callable[[], bool]) -> float:
    from sashimono.engine.encode.exporter import ExportSettings, export_project

    media = probe_media(source)
    rate = media.video_streams[0].frame_rate if media.video_streams else FrameRate(30)
    frames = int(media.duration * rate.fps)
    project = Project.create(ProjectSettings(width=1920, height=1080, frame_rate=rate))
    project = AddMedia(media).apply(project)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = AddTrack(track).apply(project)
    project = AddClip(track.id, Clip(timeline_start=0, duration=frames, media_id=media.id)).apply(
        project
    )
    export_project(
        project,
        ExportSettings(path=scratch / f"export-{time.perf_counter_ns()}.mp4"),
        should_cancel=stop,
    )
    return float(media.duration)


def _python(_source: Path, _scratch: Path, stop: Callable[[], bool]) -> float:
    """Python を回し続ける 速さは数えない（区切った長さで止まる）"""
    deadline = time.perf_counter() + PYTHON_SECONDS
    total = 0
    while time.perf_counter() < deadline and not stop():
        total += sum(range(200))
    return 0.0


JOBS: dict[str, Job] = {
    "proxy-thread": _proxy_thread,
    "proxy-process": _proxy_process,
    "scenes": _scenes,
    "thumbnails": _thumbnails,
    "waveform": _waveform,
    "export": _export,
    "python": _python,
}


class Spinner:
    """Python を回し続けるスレッド（``--busy-thread``） 入れている間だけ回る"""

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Spinner:
        if self._enabled:
            self._stop.clear()

            def spin() -> None:
                total = 0
                while not self._stop.is_set():
                    total += sum(range(200))

            self._thread = threading.Thread(target=spin, name="bench-spinner", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def run_alone(job: Job, source: Path, scratch: Path, spinner: Spinner) -> float | None:
    """画面を動かさずに走らせた速さ 失敗したら ``None``"""
    began = time.perf_counter()
    try:
        with spinner:
            seconds = job(source, scratch, lambda: False)
    except Exception as exc:  # 測れない処理（GL が無い等）は飛ばす 理由は出す
        print(f"    単体で走らせられなかった: {type(exc).__name__}: {exc}")
        return None
    elapsed = time.perf_counter() - began
    return seconds / elapsed if seconds > 0 and elapsed > 0 else None


# --- 画面の操作 ---


def _settle(app: QApplication) -> None:
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    app.processEvents()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@dataclass
class Scene:
    """測りに使う窓と、操作の相手"""

    window: MainWindow
    videos: tuple[ClipId, ClipId]
    text: ClipId
    moving: Clip
    moving_track: Track
    duration: int


def build_window(app: QApplication, source: Path) -> Scene:
    """文字 300 本と映像 2 本の窓 映像の素材は無い場所を指す"""
    project = Project.create(ProjectSettings(width=1920, height=1080, frame_rate=FrameRate(60)))
    tracks = [Track(TrackKind.VIDEO, f"V{index + 1}") for index in range(7)]
    for track in tracks:
        project = AddTrack(track).apply(project)
    for index in range(300):
        track = tracks[1 + index % 6]
        clip = Clip(
            timeline_start=(index // 6) * 72, duration=60, source=TEXT.create(text=f"{index}")
        )
        project = AddClip(track.id, clip).apply(project)
    media: MediaItem = replace(probe_media(source), path=_base / "missing.mp4")
    project = AddMedia(media).apply(project)
    # 窓を作る前に控えを切る 4K の素材なら無い場所でも控えの係が動き出し、エンコーダの
    # 確かめが温めより長引くと、何もしない間の測りに重なって比べる元が歪む
    PreferenceStore().save(replace(Preferences(), use_proxy=False))
    for index in range(2):
        clip = Clip(timeline_start=index * 600, duration=540, media_id=media.id)
        project = AddClip(tracks[0].id, clip).apply(project)
    window = MainWindow(project, confirm_unsaved=False)
    window.resize(1600, 900)
    window.show()
    window._timeline.resize(1586, 262)
    window._timeline.zoom_to_fit()
    _settle(app)
    current = window.view_project
    videos = tuple(c.id for c in current.timeline.tracks[0].clips)
    moving_track = current.timeline.tracks[2]
    return Scene(
        window=window,
        videos=(videos[0], videos[1]),
        text=current.timeline.tracks[1].clips[0].id,
        moving=moving_track.clips[-1],
        moving_track=moving_track,
        duration=current.duration,
    )


def operations(app: QApplication, scene: Scene) -> dict[str, Callable[[int], None]]:
    """操作の名前 → n 回目の操作"""
    view = scene.window._timeline
    first, second = scene.videos

    def select(n: int) -> None:
        view.set_selection(((first, second)[n % 2],))

    def from_empty(_n: int) -> None:
        # 空いた所を押す（選びを外す）→ 映像を選ぶ 組み直しが 2 回走っていた道（#442 A）
        view.set_selection(())
        _settle(app)
        view.set_selection((first,))

    def switch_kind(n: int) -> None:
        view.set_selection(((first, scene.text)[n % 2],))

    def playhead(n: int) -> None:
        scene.window._seek((n * 397) % max(1, scene.duration))

    def zoom(n: int) -> None:
        view.zoom(1.25 if n % 2 == 0 else 0.8)

    def edit(n: int) -> None:
        clip = scene.moving
        scene.window.execute(
            MoveClip(clip.id, clip.timeline_start + 1 + n % 2, scene.moving_track.id)
        )

    return {
        "選ぶ": select,
        "空から選ぶ": from_empty,
        "種類を替える": switch_kind,
        "再生位置": playhead,
        "拡大縮小": zoom,
        "編集": edit,
    }


def drive(
    app: QApplication,
    ops: dict[str, Callable[[int], None]],
    keep_going: Callable[[], bool],
    starts: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """``keep_going`` が真の間、操作を順に繰り返し、1 回ずつの時間（ms）を返す

    ``starts`` を渡すと、操作ごとに始めた時刻（``time.perf_counter``）もそこへ積む
    """
    times: dict[str, list[float]] = {name: [] for name in ops}
    n = 0
    while keep_going():
        for name, action in ops.items():
            began = time.perf_counter()
            action(n)
            _settle(app)
            times[name].append((time.perf_counter() - began) * 1000)
            if starts is not None:
                starts.setdefault(name, []).append(began)
        n += 1
    return times


def run_contended(
    app: QApplication,
    ops: dict[str, Callable[[int], None]],
    job: Job,
    source: Path,
    scratch: Path,
    spinner: Spinner,
) -> tuple[float | None, dict[str, list[float]]]:
    """裏の処理を走らせながら操作を繰り返す 速さと操作の時間を返す"""
    done = threading.Event()
    answer: list[float] = []
    #: 裏の処理が終わった時刻 速さはここまでで割り、操作もここまでに始めた物だけを数える
    #: 操作の 1 周の途中で終わると、残りの操作の分まで分母に入って速さが低く出て、
    #: 取り合いの無くなった後の操作まで「取り合っている間」に混ざる
    ended: list[float] = []

    def run() -> None:
        try:
            answer.append(job(source, scratch, lambda: False))
        except Exception as exc:  # 理由は出して、操作の側を止める
            print(f"    画面を動かしながら走らせられなかった: {type(exc).__name__}: {exc}")
        finally:
            ended.append(time.perf_counter())
            done.set()

    began = time.perf_counter()
    worker = threading.Thread(target=run, name="bench-job", daemon=True)
    starts: dict[str, list[float]] = {}
    with spinner:
        worker.start()
        times = drive(app, ops, lambda: not done.is_set(), starts)
        worker.join()
    finished = ended[0] if ended else time.perf_counter()
    elapsed = finished - began
    speed = answer[0] / elapsed if answer and answer[0] > 0 and elapsed > 0 else None
    overlapped = {
        name: [ms for ms, at in zip(values, starts.get(name, []), strict=True) if at < finished]
        for name, values in times.items()
    }
    return speed, overlapped


# --- 出す ---


def _cell(values: list[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return f"{statistics.median(ordered):.1f} / {p95:.1f} / {ordered[-1]:.1f}"


def _speed(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} 倍速"


def report(idle: dict[str, list[float]], results: list[JobResult]) -> None:
    names = list(idle)
    print()
    print("画面の操作 1 回（ms 中央 / 95 パーセンタイル / 最大）")
    print()
    print("| 裏の処理 | 単体の速さ | 操作中の速さ | 比 | " + " | ".join(names) + " |")
    print("|---|---|---|---|" + "---|" * len(names))
    print("| （無し） | - | - | - | " + " | ".join(_cell(idle[name]) for name in names) + " |")
    for result in results:
        ratio = (
            f"{result.contended / result.alone:.0%}" if result.alone and result.contended else "-"
        )
        cells = " | ".join(_cell(result.ops.get(name, [])) for name in names)
        print(
            f"| {result.name} | {_speed(result.alone)} | {_speed(result.contended)} | "
            f"{ratio} | {cells} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--jobs", nargs="+", choices=sorted(JOBS), default=list(JOBS), help="測る裏の処理"
    )
    parser.add_argument("--size", default="1920x1080", help="素材の大きさ")
    parser.add_argument("--rate", type=int, default=60, help="素材のフレームレート")
    parser.add_argument("--seconds", type=int, default=60, help="素材の長さ（秒）")
    parser.add_argument("--gop", type=int, default=250, help="キーフレームの間隔（コマ）")
    parser.add_argument(
        "--switch-interval",
        type=float,
        default=None,
        help=(
            "Python のスレッドの切り替え間隔（秒 sys.setswitchinterval） 省くと窓の設定のまま"
            "（画面の応答を優先する設定なら 0.001）"
        ),
    )
    parser.add_argument(
        "--busy-thread",
        action="store_true",
        help="裏の処理の間、Python を回し続けるスレッドを 1 本足す（先読みや解析の代わり）",
    )
    parser.add_argument(
        "--onscreen", action="store_true", help="窓を本物の画面に出す（書き出しの GL が要るとき）"
    )
    args = parser.parse_args(argv)
    if not args.onscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if shutil.which("ffmpeg") is None:
        print("ffmpeg が無いので測れない")
        return 0

    scratch = _base / "work"
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"素材を作っている: {args.size} {args.rate}fps {args.seconds} 秒 {args.gop} コマごと")
    source = make_source(scratch, args.size, args.rate, args.seconds, args.gop)
    if source is None:
        return 1

    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    scene = build_window(app, source)
    # 窓を作った後に当てる 窓は設定（画面の応答を優先する 既定は入）から間隔を 1ms へ
    # 書き換えるので、先に当てると指定した間隔で測ったことにならない
    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
    print(f"切り替え間隔 {sys.getswitchinterval() * 1000:g}ms")
    ops = operations(app, scene)
    # 1 度ずつ回して温める 初めの 1 回は部品の型を読む分が乗る
    drive(app, ops, iter([True, False]).__next__)

    print("裏の処理なしで操作を測っている")
    until = time.perf_counter() + IDLE_SECONDS
    idle = drive(app, ops, lambda: time.perf_counter() < until)

    spinner = Spinner(args.busy_thread)
    if args.busy_thread:
        print("Python を回し続けるスレッドを足して測る")
    results: list[JobResult] = []
    for name in args.jobs:
        print(f"{name}: 単体")
        job = JOBS[name]
        alone = run_alone(job, source, scratch, spinner) if name != "python" else None
        if alone is None and name != "python":
            continue
        print(f"{name}: 画面を動かしながら")
        speed, times = run_contended(app, ops, job, source, scratch, spinner)
        results.append(JobResult(name, alone, speed if name != "python" else None, times))

    report(idle, results)
    scene.window.close()
    scene.window.deleteLater()
    _settle(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
