r"""画面の止まりの記録を数え、原因ごとに分ける（#455）

    .venv\Scripts\python.exe tools\stall_report.py
    .venv\Scripts\python.exe tools\stall_report.py 送ってもらった\stalls.jsonl
    .venv\Scripts\python.exe tools\stall_report.py --since 2026-11-10T09:00 --min-ms 50

記録は ``表示 → 設定…`` の「画面の止まりを記録する」（環境変数 ``SASHIMONO_STALL_WATCH``）で
``%LOCALAPPDATA%\Sashimono\diagnostics\stalls.jsonl`` に 1 行ずつ溜まる
（:mod:`sashimono.stall_watch`）
ファイルを渡さなければ、この機械のその場所（と 1 世代前の ``.1``）を読む

分け方（上から順に当てはめる）

- ごみ集め 止まりの間のごみ集めの合計が、止まりの半分以上
- 設定パネル 画面のスレッドの並びでいちばん内側の画面の部品（``ui/``）が ``ui/inspector/``
- 描画 画面のスレッドの並びに ``paintGL`` ``paintEvent`` か
  ``engine/render/`` ``engine/gpu/`` がある
- タイムライン いちばん内側の画面の部品が ``ui/timeline/``（描画でない物 選ぶ・動かすの計算）
  選んだ知らせを受けたほかのパネル（字幕パネルの一覧など）は、その場所で「その他」に数える
- 裏のスレッド 画面のスレッドが本体の処理の中にいない（イベントループで GIL を待っている）のに、
  ほかのスレッドが Python を動かしていた
- その他 上のどれでもない 画面のスレッドのいちばん深い本体の場所で数える
- 並び無し 線を少し超えただけで、見張りが写す前に動き出した

いちばん下に、止まりの間に動いていた裏のスレッドの名前と、メモリ（コミット）の動きを出す

数え方の本体は :mod:`sashimono.stall_summary`（ヘルプの〔診断情報を作る…〕も同じ物を使う）
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sashimono.stall_summary import (
    BACKGROUND,
    CATEGORIES,
    GC,
    INSPECTOR,
    OTHER,
    PAINT,
    TIMELINE,
    UNSAMPLED,
    Stall,
    Summary,
    busy_threads,
    classify,
    deepest_place,
    default_paths,
    parse,
    read,
    report,
    summarize,
)

__all__ = [
    "BACKGROUND",
    "CATEGORIES",
    "GC",
    "INSPECTOR",
    "OTHER",
    "PAINT",
    "TIMELINE",
    "UNSAMPLED",
    "Stall",
    "Summary",
    "busy_threads",
    "classify",
    "deepest_place",
    "default_paths",
    "main",
    "parse",
    "read",
    "report",
    "summarize",
]

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="*", type=Path, help="記録のファイル（省けばこの機械の物）")
    parser.add_argument("--since", default=None, help="この時刻（ISO 形式の頭）より後だけを数える")
    parser.add_argument("--min-ms", type=float, default=0.0, help="これより短い止まりは数えない")
    parser.add_argument("--top", type=int, default=10, help="場所とスレッドを何件まで出すか")
    args = parser.parse_args(argv)
    paths = args.paths or default_paths()
    unread: list[Path] = []
    records = [
        record
        for record in read(paths, unread=unread)
        if float(record["stall_ms"]) >= args.min_ms
        and (args.since is None or str(record.get("time", "")) >= args.since)
    ]
    if args.paths:
        # 渡されたファイルが読めないのを黙ると「記録を取ったか確かめる」とだけ出て、打ち間違いに
        # 気付けない 既定の場所の .1 は無いのが普通なので、そちらは黙る
        for path in unread:
            print(f"読めなかった: {path}", file=sys.stderr)
    print(report(summarize(records), top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
