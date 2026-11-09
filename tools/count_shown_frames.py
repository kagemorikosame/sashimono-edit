r"""プレビューに実際に映ったコマを数える（#443 #463）

    .venv\Scripts\python.exe tools\count_shown_frames.py screen --region 100 200 960 540
    .venv\Scripts\python.exe tools\count_shown_frames.py screen --region 0 0 960 540 --delay 5
    .venv\Scripts\python.exe tools\count_shown_frames.py reference 素材.mp4 --region 100 200 960 540
    .venv\Scripts\python.exe tools\count_shown_frames.py file 素材.mp4 --seconds 20
    .venv\Scripts\python.exe tools\count_shown_frames.py make-source 素材.mp4

画面のスレッドの止まりと 1 コマの描画時間が目標に入っていても、画面へ出たコマが足りない
ことがある（0.2.4 では再生の位置が約 21 ms 刻みでしか進まず、60fps の素材で 35〜41 fps
しか映らなかった #463） 描く側の記録ではそれが見えないので、画面そのものを撮って数える

- ``screen`` 画面の枠を ffmpeg で撮り、前と違う絵が出た回数と間隔を ``mpdecimate`` で数える
- ``reference`` 同じ素材を ffplay で同じ枠に映し、同じ撮り方で数える 撮り方そのものが
  落とす分を含んだ「画面での基準」になる アプリの数字はまずこれと比べる
- ``file`` 元の素材を画面を通さずに数える 止まった場面がある素材では元でも 60 に届かない
- ``make-source`` 毎コマ絵が変わる 1080p60 の素材を作る 元の数がちょうど 60 になるので、
  足りない分がそのまま再生と撮る側の落としになる

撮り方は 2 つ 既定は ``ddagrab``（画面の複製の API） 画面が書き換わるたびに 1 枚届くので、
画面の切り替わりと撮る時刻がずれて落とすことが無い ``gdigrab`` は決まった間隔（60fps）で
画面を写し、60Hz の画面と拍がずれて、ffplay で映しても 47 fps ほどしか数えられなかった
（2026-11-09 手元の RTX 5060 Ti 60Hz ddagrab では 56 fps） ddagrab が使えない環境
（古い ffmpeg・リモートデスクトップ）のためだけに残す

``gdigrab`` は撮る側が追いつかないと、撮れたコマが秒数 × 60 に届かず、映ったコマも少なく出る
撮れたコマの数を必ず並べて出し、届かない回は「低め」と印を付ける ``ddagrab`` の撮れたコマは
画面が書き換わった回数で、届かなくても撮る側の落としとは限らない

撮る枠は物理の画素の座標 ``ddagrab`` は ``--output`` の画面の左上から、``gdigrab`` は
画面全体の左上から数える 拡大率 150% の画面では Qt の座標の 1.5 倍になる
"""

from __future__ import annotations

import argparse
import io
import itertools
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 撮る速さ 目標の「60fps の 1 コマ以内」を見るので、それより粗く撮ると落としが見えない
CAPTURE_RATE = 60

#: 撮り方 先頭が既定
GRABBERS = ("ddagrab", "gdigrab")

#: 比べる前にそろえる横幅 ``mpdecimate`` の閾値は 8x8 の区画の差の数で決まるので、
#: 撮った枠と元の素材で大きさが違うと、同じ動きでも「違う絵」と見なす割合が変わる
COMPARE_WIDTH = 640

#: これを越えた間隔を「止まり」として数える（#455 で扱う止まりと同じ物差し）
STALL_SECONDS = 0.1

#: 撮れたコマが秒数 × 撮る速さのこれ未満なら、撮る側が追いついていないと見る
#: 撮る側が落とした間は映ったコマも数えられず、映ったコマの数が実際より低く出る
CAPTURE_SHORTFALL = 0.98

#: ffplay が窓を出して映し始めるまでの待ち（秒） 窓が出る前から撮ると、頭の静止を止まりと数える
REFERENCE_WARMUP = 1.5

#: 見る印 ``showinfo`` を間引きの前と後ろに置き、前で撮れたコマ・後ろで映ったコマを数える
#: 並びの番号が ``Parsed_showinfo_N`` の N になる
FILTERS = f"showinfo,scale={COMPARE_WIDTH}:-2,mpdecimate,showinfo"
_CAPTURED_INDEX = 0
_SHOWN_INDEX = 3

_SHOWINFO = re.compile(
    r"\[Parsed_showinfo_(\d+) @ [^\]]+\] n:\s*\d+ pts:\s*-?\d+ pts_time:(-?[\d.]+)"
)


@dataclass(frozen=True)
class Count:
    """1 回の数えの結果"""

    label: str
    seconds: float
    captured: int
    shown: int
    interval_median_ms: float
    interval_p95_ms: float
    stalls: int
    #: 決まった間隔で撮った（``gdigrab``・素材） 撮れたコマが秒数 × 60 に届くはずの撮り方
    fixed_rate: bool = True

    @property
    def shown_fps(self) -> float:
        return self.shown / self.seconds if self.seconds > 0 else 0.0

    @property
    def low(self) -> bool:
        """撮る側が追いつかず、映ったコマの数が低めに出ている"""
        return self.fixed_rate and self.captured < self.seconds * CAPTURE_RATE * CAPTURE_SHORTFALL


def parse_showinfo(text: str) -> dict[int, list[float]]:
    """``showinfo`` の行を、置いた位置ごとの時刻の並びにする"""
    times: dict[int, list[float]] = {}
    for match in _SHOWINFO.finditer(text):
        times.setdefault(int(match.group(1)), []).append(float(match.group(2)))
    return times


def summarize(
    label: str,
    captured: Sequence[float],
    shown: Sequence[float],
    *,
    seconds: float | None = None,
    fixed_rate: bool = True,
) -> Count:
    """撮れたコマの時刻と映ったコマの時刻から、並べる数字を出す

    ``seconds`` を渡さなければ、撮れたコマの頭から尻まで（+1 コマ）を長さにする 素材を
    数えるときに頼んだ秒数で割らないのは、素材が頼んだ長さより短いことがあるため
    画面の更新ごとに撮る（``ddagrab``）ときは、絵が止まると何も届かず頭から尻までが
    縮むので、頼んだ秒数を渡す
    """
    if seconds is None:
        seconds = captured[-1] - captured[0] + 1 / CAPTURE_RATE if captured else 0.0
    intervals = [(b - a) for a, b in itertools.pairwise(shown)]
    if intervals:
        ordered = sorted(intervals)
        median = statistics.median(ordered) * 1000
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000
    else:
        median = p95 = 0.0
    stalls = sum(1 for gap in intervals if gap > STALL_SECONDS)
    return Count(label, seconds, len(captured), len(shown), median, p95, stalls, fixed_rate)


def format_table(counts: Sequence[Count]) -> str:
    """結果を Markdown の表にする PR や Issue へそのまま貼れる形"""
    lines = [
        "| 対象 | 秒数 | 撮れたコマ | 映ったコマ | 映った fps | 間隔の中央 | 間隔の 95% "
        "| 0.1 秒を越えた止まり | 注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for count in counts:
        note = "撮る側が追いつかず低め" if count.low else ""
        lines.append(
            f"| {count.label} | {count.seconds:.1f} | {count.captured} | {count.shown} "
            f"| {count.shown_fps:.1f} | {count.interval_median_ms:.1f} ms "
            f"| {count.interval_p95_ms:.1f} ms | {count.stalls} | {note} |"
        )
    return "\n".join(lines)


def grab_arguments(
    grabber: str, region: Sequence[int], seconds: float, output: int = 0
) -> list[str]:
    """撮る側の ffmpeg の引数（入力と、書き出す前に通す絵の直し）"""
    x, y, width, height = region
    if grabber == "ddagrab":
        # ``dup_frames=0`` 画面が書き換わらない間は同じ絵を足さない 足すと決まった間隔で
        # 写すのと同じになり、画面の切り替わりとの拍のずれで落とす（gdigrab と同じ）
        # 絵は GPU の上にあるので、書き出す前に手元へ下ろす
        source = (
            f"ddagrab=output_idx={output}:framerate={CAPTURE_RATE}:draw_mouse=0:dup_frames=0"
            f":offset_x={x}:offset_y={y}:video_size={width}x{height}"
        )
        return [
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            f"{seconds:.3f}",
            "-vf",
            "hwdownload,format=bgra",
        ]
    if grabber == "gdigrab":
        return [
            "-f",
            "gdigrab",
            "-framerate",
            str(CAPTURE_RATE),
            "-draw_mouse",
            "0",
            "-offset_x",
            str(x),
            "-offset_y",
            str(y),
            "-video_size",
            f"{width}x{height}",
            "-t",
            f"{seconds:.3f}",
            "-i",
            "desktop",
        ]
    raise ValueError(f"知らない撮り方: {grabber}")


def _ffmpeg(name: str = "ffmpeg") -> str:
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"{name} が見つからない PATH に置いてから走らせる")
    return found


def _analyze(
    source: list[str], label: str, *, seconds: float | None = None, fixed_rate: bool = True
) -> Count:
    """ffmpeg に間引かせて ``showinfo`` を読む

    ``-fps_mode passthrough`` を付ける 付けないと出力の速さに合わせてコマを足したり
    捨てたりし、撮った時刻の間隔がそのまま残らない
    """
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        *source,
        "-vf",
        FILTERS,
        "-fps_mode",
        "passthrough",
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg が失敗した:\n{result.stderr[-2000:]}")
    times = parse_showinfo(result.stderr)
    return summarize(
        label,
        times.get(_CAPTURED_INDEX, []),
        times.get(_SHOWN_INDEX, []),
        seconds=seconds,
        fixed_rate=fixed_rate,
    )


def capture_screen(
    region: Sequence[int],
    seconds: float,
    label: str,
    *,
    grabber: str = GRABBERS[0],
    output: int = 0,
) -> Count:
    """画面の枠を撮ってから数える

    撮りながら間引くと、比べる手間で撮る側が遅れてコマを落とす いったん可逆で
    書き出し、撮り終えてから数える 書き出しは速さ優先（``ultrafast`` の可逆）
    4:2:0 などへ落とすと色の差が潰れて「同じ絵」と見なす回が出るので RGB のまま持つ
    """
    with tempfile.TemporaryDirectory(prefix="sashimono-frames-") as folder:
        clip = Path(folder) / "capture.mkv"
        command = [
            _ffmpeg(),
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            *grab_arguments(grabber, region, seconds, output),
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264rgb",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            str(clip),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise SystemExit(f"画面を撮れなかった:\n{result.stderr[-2000:]}")
        fixed = grabber == "gdigrab"
        return _analyze(
            ["-i", str(clip)], label, seconds=None if fixed else seconds, fixed_rate=fixed
        )


def count_reference(
    path: Path,
    region: Sequence[int],
    seconds: float,
    start: float,
    label: str,
    *,
    grabber: str = GRABBERS[0],
    output: int = 0,
) -> Count:
    """素材を ffplay で同じ枠に映し、同じ撮り方で数える

    ffplay の窓の位置は画面全体の座標なので、``ddagrab`` で 2 枚目以降の画面を撮るときは
    合わない 1 枚目の画面（``--output 0``）で使う
    """
    x, y, width, height = region
    command = [
        _ffmpeg("ffplay"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-noborder",
        "-left",
        str(x),
        "-top",
        str(y),
        "-x",
        str(width),
        "-y",
        str(height),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{seconds + REFERENCE_WARMUP + 2:.3f}",
        "-autoexit",
        "-an",
        str(path),
    ]
    player = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(REFERENCE_WARMUP)
        return capture_screen(region, seconds, label, grabber=grabber, output=output)
    finally:
        player.kill()
        player.wait()


def count_file(path: Path, seconds: float, start: float, label: str) -> Count:
    """元の素材を画面を通さずに数える"""
    return _analyze(["-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(path)], label)


def make_source(path: Path, seconds: int) -> None:
    """毎コマ絵が変わる 1080p60 の素材を作る 音も付ける（音を時計にする再生を通すため）

    ``testsrc2`` は毎コマ動く部分があるので、元の素材の数がちょうど 60 になる
    """
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1920x1080:rate=60:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=48000:duration={seconds}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "60",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, check=True)


def _add_screen_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        required=True,
        metavar=("X", "Y", "W", "H"),
        help="撮る枠 物理の画素の座標（拡大率をかけた値）",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="撮る長さ（秒）")
    parser.add_argument("--grabber", choices=GRABBERS, default=GRABBERS[0], help="撮り方")
    parser.add_argument(
        "--output", type=int, default=0, help="ddagrab で撮る画面の番号（0 が 1 枚目）"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="プレビューに実際に映ったコマを数える")
    commands = parser.add_subparsers(dest="command", required=True)

    screen = commands.add_parser("screen", help="画面の枠を撮って数える")
    _add_screen_options(screen)
    screen.add_argument(
        "--delay", type=float, default=3.0, help="撮り始めるまでの待ち（秒） この間に再生を押す"
    )
    screen.add_argument("--label", default="画面", help="表の対象の欄に書く名前")

    reference = commands.add_parser("reference", help="素材を ffplay で同じ枠に映して数える")
    reference.add_argument("path", type=Path)
    _add_screen_options(reference)
    reference.add_argument("--start", type=float, default=0.0, help="映し始める位置（秒）")
    reference.add_argument("--label", default="ffplay")

    source = commands.add_parser("file", help="元の素材を画面を通さずに数える")
    source.add_argument("path", type=Path)
    source.add_argument("--seconds", type=float, default=20.0)
    source.add_argument("--start", type=float, default=0.0, help="数え始める位置（秒）")
    source.add_argument("--label", default="元の素材")

    make = commands.add_parser("make-source", help="毎コマ絵が変わる 1080p60 の素材を作る")
    make.add_argument("path", type=Path)
    make.add_argument("--seconds", type=int, default=60)

    args = parser.parse_args(argv)
    if args.command == "make-source":
        make_source(args.path, args.seconds)
        print(f"作った: {args.path}")
        return 0
    if args.command == "screen":
        # 待ちは ffmpeg の ``-ss`` に任せない 撮りながら頭を読み捨てる形になり、読み捨てから
        # 撮り始めへ移る所でコマを落として、撮れたコマが毎回 5% ほど足りなくなった
        if args.delay > 0:
            print(f"{args.delay:.0f} 秒後から {args.seconds:.0f} 秒撮る その間に再生を始める")
            time.sleep(args.delay)
        count = capture_screen(
            args.region, args.seconds, args.label, grabber=args.grabber, output=args.output
        )
    elif args.command == "reference":
        count = count_reference(
            args.path,
            args.region,
            args.seconds,
            args.start,
            args.label,
            grabber=args.grabber,
            output=args.output,
        )
    else:
        count = count_file(args.path, args.seconds, args.start, args.label)
    print(format_table([count]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
