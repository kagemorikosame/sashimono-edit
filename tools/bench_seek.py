r"""キーフレームの間隔ごとに、飛んだ先の 1 コマを出すまでを測る（#430）

    .venv\Scripts\python.exe tools\bench_seek.py
    .venv\Scripts\python.exe tools\bench_seek.py --gaps 2 8 50 --jumps 40

1920x1080 の素材を、キーフレームの間隔だけ変えて ffmpeg で作り、ランダムな位置へ飛んで
1 コマ出すまでを測る 一番長い間隔の素材からは :func:`create_proxy` で控えも作って同じく測る
結果は :mod:`sashimono.engine.cache.proxy` の「キーフレームの間隔」の表へ写す

ffmpeg が無い環境では、何も測らずに終わる（終了コード 0）
"""

from __future__ import annotations

import argparse
import atexit
import io
import random
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sashimono.engine.cache.proxy import (
    KEYFRAME_GAP_LIMIT,
    PROXY_HEIGHT,
    create_proxy,
)
from sashimono.engine.decode import VideoDecoder, keyframe_gap

#: 測る素材の大きさとフレームレート 録画で多い 1080p 30fps
WIDTH, HEIGHT, RATE = 1920, 1080, 30


def _make_source(directory: Path, gap: int, seconds: int) -> Path | None:
    """``gap`` 秒ごとにだけキーフレームを置いた素材を作る

    場面の切り替わりで足されるキーフレームを止める（``-sc_threshold 0``） 止めないと、
    動きの多い絵では間隔が短くなり、測りたい間隔にならない
    """
    path = directory / f"gop{gap}.mp4"
    frames = gap * RATE
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}:duration={seconds}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-g",
        str(frames),
        "-keyint_min",
        str(frames),
        "-sc_threshold",
        "0",
        "-b:v",
        "8M",
        str(path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"素材を作れなかった: {exc}")
        return None
    return path


def _measure(path: Path, seconds: int, jumps: int) -> tuple[float, float]:
    """ランダムな ``jumps`` か所へ飛んで 1 コマ出すまで（中央値と 95 パーセンタイル ミリ秒）

    乱数の種は固定する 素材ごとに飛ぶ先が違うと、比べた差が間隔の差なのか分からない
    """
    rng = random.Random(0)
    times: list[float] = []
    with VideoDecoder(path) as decoder:
        decoder.frame_at(Fraction(0))
        for _ in range(jumps):
            target = Fraction(rng.randrange(0, (seconds - 2) * RATE), RATE)
            start = time.perf_counter()
            decoder.frame_at(target)
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2], times[min(len(times) - 1, int(len(times) * 0.95))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 50])
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--jumps", type=int, default=80)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg が無いので測らない")
        return 0

    base = Path(tempfile.mkdtemp(prefix="sashimono-seek-bench-"))
    # 1 本 60MB ほどの素材を何本も置くので、回すたびに残すと溜まる
    atexit.register(shutil.rmtree, base, True)
    print(f"{WIDTH}x{HEIGHT} {RATE}fps {args.seconds} 秒 {args.jumps} か所へ飛ぶ")
    print(f"控えを作る線: {float(KEYFRAME_GAP_LIMIT)} 秒より長い間隔")
    print("調べた間隔は頭から読んだ所までの値 12 秒より長い間隔は 12 秒ほどと出る")
    print(f"{'読む物':<24}{'調べた間隔':>10}{'中央値':>10}{'95%':>10}")
    longest: Path | None = None
    for gap in sorted(args.gaps):
        source = _make_source(base, gap, args.seconds)
        if source is None:
            return 0
        longest = source
        # 線で止めずに読む 止めると 8 秒も 50 秒も「4 秒より長い」としか出ない
        found = keyframe_gap(source)
        median, high = _measure(source, args.seconds, args.jumps)
        seen = f"{float(found):.1f}s" if found is not None else "?"
        print(f"{f'元 {gap} 秒間隔':<24}{seen:>10}{median:>9.1f}ms{high:>8.1f}ms")

    if longest is not None:
        proxy = create_proxy(longest, base / "proxy.mp4", height=PROXY_HEIGHT)
        if proxy is None:
            print("控えを作れなかった")
            return 0
        found = keyframe_gap(proxy)
        median, high = _measure(proxy, args.seconds, args.jumps)
        seen = f"{float(found):.1f}s" if found is not None else "?"
        print(f"{f'控え {PROXY_HEIGHT}p':<24}{seen:>10}{median:>9.1f}ms{high:>8.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
