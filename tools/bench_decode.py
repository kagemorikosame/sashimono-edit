r"""素材の読みと控えの作成の速さを、CPU と GPU（NVDEC など）で比べる（#454 #443 の段階 4）

    .venv\Scripts\python.exe tools\bench_decode.py 素材.mp4
    .venv\Scripts\python.exe tools\bench_decode.py 素材.mp4 --seconds 120 --hwaccel cuda d3d11va

測るのは 3 つ どれも素材の秒数 / 掛かった秒数（倍速）

1. 読むだけ CPU（PyAV のスレッドは自動）と、``--hwaccel`` に並べた GPU のデコード
   GPU は絵を GPU のメモリに置いたままなので、控えの作成と同じく CPU へ写す（``to_ndarray``）
   までを数える 写さない速さは控えの作成では使えない
2. 控えの作成（今の作り 別のプロセス :func:`~sashimono.engine.cache.proxy.proxy_job`）
3. 控えの作成のうち、読む分の割合（1 の CPU の速さから見積もる）

控えの作成のデコードを GPU に寄せる案（#454）は、3 の割合が大きく、1 で GPU が CPU より
十分に速いときだけ効く 読む分が小さければ、寄せても変換（縮小と符号化）の時間は残る

``--seconds`` で頭から何秒を測るかを決める（長い録画を全部読むと何分もかかる）
素材は呼ぶ人が渡す 本人の録画を渡すときは、名前を結果に書き写さない
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import av

from sashimono.engine.cache.proxy import proxy_codecs, proxy_job, working_path
from sashimono.engine.worker import WorkerUnavailableError, run_in_process


def decode_speed(path: Path, seconds: float, hwaccel: str | None) -> tuple[float, int] | None:
    """頭から ``seconds`` 秒を読んで CPU の配列へ写す速さ（倍速, 枚数） 開けなければ ``None``"""
    from av.codec.hwaccel import HWAccel

    accel = None if hwaccel is None else HWAccel(device_type=hwaccel, allow_software_fallback=False)
    try:
        container = av.open(str(path), "r", hwaccel=accel)
    except (av.error.FFmpegError, ValueError, TypeError) as exc:
        print(f"  {hwaccel or 'cpu'}: 開けない {exc}")
        return None
    with container:
        stream = container.streams.video[0]
        if hwaccel is None:
            stream.thread_type = "AUTO"
        base = stream.time_base
        assert base is not None
        start = stream.start_time or 0
        began = time.perf_counter()
        count = 0
        last = 0.0
        try:
            for frame in container.decode(stream):
                # 控えは縮めて符号化するので、CPU の画素に写すまでを数える
                frame.to_ndarray(format="nv12" if hwaccel else None)
                count += 1
                if frame.pts is not None:
                    last = float((frame.pts - start) * base)
                if last >= seconds:
                    break
        except av.error.FFmpegError as exc:
            print(f"  {hwaccel or 'cpu'}: 読む途中で失敗した {exc}")
            return None
        elapsed = time.perf_counter() - began
    return (last / elapsed if elapsed > 0 else 0.0), count


def proxy_speed(path: Path, seconds: float) -> float | None:
    """頭から ``seconds`` 秒を切り出した素材で、今の控えの作り（別のプロセス）の速さ"""
    import subprocess

    with tempfile.TemporaryDirectory(prefix="sashimono-decode-bench-") as folder:
        clip = Path(folder) / "clip.mp4"
        # 頭だけを写して切り出す（符号化し直さない） 長い録画の控えを全部作ると何十分もかかる
        try:
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-t",
                    str(seconds),
                    "-c",
                    "copy",
                    str(clip),
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            # 先に測った読む速さを捨てないよう、ここで止めずに表の残りを出す
            print(f"  素材の頭を切り出せなかった {exc}")
            return None
        # 倍速の分子は切り出せた長さ 素材が --seconds より短いと、指定の秒数で割ると速さを
        # 何倍にも大きく見積もる
        with av.open(str(clip)) as cut:
            length = float(cut.duration / av.time_base) if cut.duration else seconds
        target = Path(folder) / "proxy.mp4"
        began = time.perf_counter()
        try:
            made = run_in_process(
                proxy_job,
                {
                    "source": str(clip),
                    "target": str(target),
                    "working": str(working_path(target)),
                    "height": 540,
                    "stream_index": None,
                    "codecs": tuple(proxy_codecs()),
                },
            )
        except WorkerUnavailableError as exc:
            print(f"  別のプロセスを起こせなかった {exc}")
            return None
        elapsed = time.perf_counter() - began
    if made is None:
        print("  控えを作れなかった")
        return None
    return length / elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("source", type=Path, help="測る素材")
    parser.add_argument("--seconds", type=float, default=120.0, help="頭から何秒を測るか")
    parser.add_argument(
        "--hwaccel", nargs="*", default=["cuda", "d3d11va"], help="比べる GPU のデコード"
    )
    args = parser.parse_args(argv)
    with av.open(str(args.source)) as container:
        stream = container.streams.video[0]
        print(
            f"素材: {stream.codec_context.name} {stream.width}x{stream.height} "
            f"{float(stream.average_rate or 0):.0f}fps 頭から {args.seconds:g} 秒"
        )
    results: list[tuple[str, str]] = []
    cpu_speed: float | None = None
    for accel in [None, *args.hwaccel]:
        found = decode_speed(args.source, args.seconds, accel)
        if found is not None:
            if accel is None:
                cpu_speed = found[0]
            results.append((f"読むだけ {accel or 'cpu'}", f"{found[0]:.1f} 倍速（{found[1]} 枚）"))
    speed = proxy_speed(args.source, args.seconds)
    if speed is not None:
        results.append(("控えの作成（別のプロセス）", f"{speed:.1f} 倍速"))
        if cpu_speed:
            # 読む時間は 秒数 / cpu_speed 全体は 秒数 / speed なので、読む分の割合は速さの比
            # （控えは別のプロセスで縮めながら読むので、読みと変換が重なる分だけ大きめに出る）
            share = min(speed / cpu_speed, 1.0)
            results.append(("控えの作成のうち読む分（見積もり）", f"{share:.0%}"))
    print()
    print("| 測った物 | 速さ |")
    print("|---|---|")
    for name, value in results:
        print(f"| {name} | {value} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
