r"""キーフレームの間隔ごとに、飛んだ先の 1 コマを出すまでを測る（#430）

    .venv\Scripts\python.exe tools\bench_seek.py
    .venv\Scripts\python.exe tools\bench_seek.py --gaps 2 8 50 --jumps 40
    .venv\Scripts\python.exe tools\bench_seek.py --json 結果.json  （版ごとの記録も書く #513）

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
# 結果の形は隣の道具の物を使う 試験が importlib で読むときは tools が探す所に無い
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_results import BenchRecord

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
        # 組み立てているのは固定の文字列と argparse が受けた数値、一時フォルダの中の
        # パスだけで、外から来る文字列は混ざらない shell は通さない（list 渡し）
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"素材を作れなかった: {exc}")
        return None
    return path


def _measure(path: Path, seconds: int, jumps: int) -> list[float]:
    """ランダムな ``jumps`` か所へ飛んで 1 コマ出すまで（ミリ秒 小さい順）

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
    return times


def _summary(times: list[float]) -> tuple[float, float]:
    """中央値と 95 パーセンタイル（画面の表の値 前からの取り方のまま）"""
    return times[len(times) // 2], times[min(len(times) - 1, int(len(times) * 0.95))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 50])
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--jumps", type=int, default=80)
    parser.add_argument(
        "--json", type=Path, default=None, help="版ごとに比べる記録の形（bench_results）でも書く"
    )
    args = parser.parse_args(argv)
    # 素材を作ってから落ちると、何分も待たせた後に IndexError で終わる 先に断る
    if args.jumps <= 0:
        parser.error("--jumps は 1 以上にする")
    if args.seconds <= 2:
        parser.error("--seconds は 2 より大きくする（終わりの 2 秒へは飛ばない）")
    if any(gap <= 0 for gap in args.gaps):
        parser.error("--gaps は 1 以上にする")
    record = BenchRecord(
        "seek", {"gaps": sorted(args.gaps), "seconds": args.seconds, "jumps": args.jumps}
    )
    if shutil.which("ffmpeg") is None:
        print("ffmpeg が無いので測らない")
        record.skip("ffmpeg が無い")
        record.write(args.json)
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
            # 終了コードは前のまま 0 にする（表だけを見る使い方を変えない） 記録には落ちたと
            # 書く 飛ばした扱いにすると、版を比べたときに ffmpeg の無い機械と見分けられない
            record.fail("素材を作れなかった")
            record.write(args.json)
            return 0
        longest = source
        # 線で止めずに読む 止めると 8 秒も 50 秒も「4 秒より長い」としか出ない
        found = keyframe_gap(source)
        times = _measure(source, args.seconds, args.jumps)
        median, high = _summary(times)
        seen = f"{float(found):.1f}s" if found is not None else "?"
        print(f"{f'元 {gap} 秒間隔':<24}{seen:>10}{median:>9.1f}ms{high:>8.1f}ms")
        record.add(f"gop{gap}", times, label=f"元 {gap} 秒間隔")

    if longest is not None:
        proxy = create_proxy(longest, base / "proxy.mp4", height=PROXY_HEIGHT)
        if proxy is None:
            print("控えを作れなかった")
            record.fail("控えを作れなかった")
            record.write(args.json)
            return 0
        found = keyframe_gap(proxy)
        times = _measure(proxy, args.seconds, args.jumps)
        median, high = _summary(times)
        seen = f"{float(found):.1f}s" if found is not None else "?"
        print(f"{f'控え {PROXY_HEIGHT}p':<24}{seen:>10}{median:>9.1f}ms{high:>8.1f}ms")
        record.add(f"proxy{PROXY_HEIGHT}p", times, label=f"控え {PROXY_HEIGHT}p")
    record.write(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
