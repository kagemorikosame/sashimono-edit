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
- 設定パネル 画面のスレッドの並びに ``ui/inspector/`` がある
- 描画 画面のスレッドの並びに ``paintGL`` ``paintEvent`` か
  ``engine/render/`` ``engine/gpu/`` がある
- タイムライン ``ui/timeline/`` がある（描画でない物 選ぶ・動かすの計算）
- 裏のスレッド 画面のスレッドが本体の処理の中にいない（イベントループで GIL を待っている）のに、
  ほかのスレッドが Python を動かしていた
- その他 上のどれでもない 画面のスレッドのいちばん深い本体の場所で数える
- 並び無し 線を少し超えただけで、見張りが写す前に動き出した

いちばん下に、止まりの間に動いていた裏のスレッドの名前と、メモリ（コミット）の動きを出す
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GC = "ごみ集め"
INSPECTOR = "設定パネル"
PAINT = "描画"
TIMELINE = "タイムライン"
BACKGROUND = "裏のスレッド"
OTHER = "その他"
UNSAMPLED = "並び無し"

#: 表に出す順
CATEGORIES = (GC, INSPECTOR, PAINT, TIMELINE, BACKGROUND, OTHER, UNSAMPLED)

#: 何もせず待っているスレッドのいちばん深い所 ここにいるスレッドは Python を動かしていない
#: （待ちの関数は GIL を手放す） 裏のスレッドが取り合っていたかを見るときに外す
_WAITING = (
    "threading.py",
    "queue.py",
    "selectors.py",
    "connection.py",
    "stall_watch.py",
    "thread.py",
)

#: 本体の場所の頭（:func:`sashimono.stall_watch._where` が付ける）
_OURS = "sashimono/"

#: 画面のスレッドがここにしかいなければ、処理の途中ではなく、イベントループで順番を待っていた
#: （起動の入口と、止まりの見張りを打つだけのタイマー）
_LOOP = ("sashimono/app.py", "sashimono/__main__.py", "sashimono/ui/stall_monitor.py")


@dataclass
class Stall:
    """記録 1 行を読みやすくした物"""

    time: str
    ms: float
    gc_ms: float
    gc_full: int
    #: 最初に写した並び（スレッドの名前 → 行） 写せなかったら空
    threads: dict[str, list[str]]
    private_mb: float | None


@dataclass
class Summary:
    """分けた結果"""

    stalls: list[Stall]
    #: 原因 → その原因の止まり
    by_cause: dict[str, list[Stall]] = field(default_factory=lambda: defaultdict(list))
    #: その他の内訳（いちばん深い本体の場所 → 数）
    places: Counter[str] = field(default_factory=Counter)
    #: 止まりの間に Python を動かしていた裏のスレッド → 数
    busy_threads: Counter[str] = field(default_factory=Counter)


def read(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """記録を読む 壊れた行（書いている最中に落ちた等）は飛ばす"""
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "stall_ms" in record:
                records.append(record)
    records.sort(key=lambda record: str(record.get("time", "")))
    return records


def parse(record: dict[str, Any]) -> Stall:
    pauses = record.get("gc") or []
    samples = record.get("samples") or []
    memory = record.get("memory_mb") or {}
    return Stall(
        time=str(record.get("time", "")),
        ms=float(record["stall_ms"]),
        gc_ms=sum(float(pause.get("ms", 0)) for pause in pauses),
        gc_full=sum(1 for pause in pauses if pause.get("generation") == 2),
        threads=dict(samples[0].get("threads", {})) if samples else {},
        private_mb=memory.get("private"),
    )


def _main_thread(stall: Stall) -> list[str]:
    return stall.threads.get("MainThread", [])


def _file(line: str) -> str:
    return line.split(":", 1)[0]


def _function(line: str) -> str:
    return line.rsplit(" ", 1)[-1]


def _ours(line: str) -> bool:
    """本体の場所か（:func:`sashimono.stall_watch._where` が置き場ごと書いた物）"""
    return _file(line).startswith(_OURS)


def _inside(line: str, *folders: str) -> bool:
    """本体の ``folders`` のどれかの中か（``ui/inspector/`` のように本体の頭を省いて書く）"""
    return _file(line).startswith(tuple(_OURS + folder for folder in folders))


def deepest_place(lines: Sequence[str]) -> str | None:
    """いちばん深い本体の場所（行の番号は外す 同じ関数の別の行を 1 つに数える）"""
    for line in reversed(lines):
        if _ours(line):
            return f"{_file(line).removeprefix(_OURS)} {_function(line)}"
    return None


def busy_threads(stall: Stall) -> list[str]:
    """止まりの間に Python を動かしていた裏のスレッド"""
    found = []
    for name, lines in stall.threads.items():
        if name == "MainThread" or not lines:
            continue
        if _file(lines[-1]) in _WAITING:
            continue
        found.append(name.split(" #", 1)[0])
    return found


def classify(stall: Stall) -> str:
    if stall.ms > 0 and stall.gc_ms >= stall.ms / 2:
        return GC
    if not stall.threads:
        return UNSAMPLED
    main = _main_thread(stall)
    if any(_inside(line, "ui/inspector/") for line in main):
        return INSPECTOR
    if any(
        _function(line) in ("paintGL", "paintEvent")
        or _inside(line, "engine/render/", "engine/gpu/")
        for line in main
    ):
        return PAINT
    if any(_inside(line, "ui/timeline/") for line in main):
        return TIMELINE
    inside = [line for line in main if _ours(line) and _file(line) not in _LOOP]
    if not inside and busy_threads(stall):
        return BACKGROUND
    return OTHER


def summarize(records: Iterable[dict[str, Any]]) -> Summary:
    stalls = [parse(record) for record in records]
    summary = Summary(stalls)
    for stall in stalls:
        cause = classify(stall)
        summary.by_cause[cause].append(stall)
        if cause == OTHER:
            summary.places[deepest_place(_main_thread(stall)) or "（本体の外）"] += 1
        for name in set(busy_threads(stall)):
            summary.busy_threads[name] += 1
    return summary


def _row(name: str, stalls: list[Stall], total_ms: float) -> str:
    if not stalls:
        return f"| {name} | 0 | - | - | - | - |"
    lengths = sorted(stall.ms for stall in stalls)
    share = sum(lengths) / total_ms if total_ms else 0
    return (
        f"| {name} | {len(stalls)} | {sum(lengths):.0f} | {share:.0%} | "
        f"{statistics.median(lengths):.0f} | {lengths[-1]:.0f} |"
    )


def report(summary: Summary, *, top: int = 10) -> str:
    out: list[str] = []
    stalls = summary.stalls
    if not stalls:
        return "記録が無い（設定の「画面の止まりを記録する」を入れて操作したか確かめる）"
    total = sum(stall.ms for stall in stalls)
    out.append(
        f"止まり {len(stalls)} 回 合計 {total:.0f} ms（{stalls[0].time} 〜 {stalls[-1].time}）"
    )
    out.append("")
    out.append("| 原因 | 回 | 合計 ms | 割合 | 中央 ms | 最大 ms |")
    out.append("|---|---|---|---|---|---|")
    for cause in CATEGORIES:
        out.append(_row(cause, summary.by_cause.get(cause, []), total))
    full = [stall for stall in stalls if stall.gc_full]
    out.append("")
    out.append(
        f"ごみ集めの世代 2 が重なった止まり: {len(full)} 回"
        + (f"（その止まりの合計 {sum(stall.ms for stall in full):.0f} ms）" if full else "")
    )
    if summary.places:
        out.append("")
        out.append(f"その他の場所（多い順 {top} 件まで）")
        for place, count in summary.places.most_common(top):
            out.append(f"- {count} 回 {place}")
    if summary.busy_threads:
        out.append("")
        out.append("止まりの間に Python を動かしていた裏のスレッド")
        for name, count in summary.busy_threads.most_common(top):
            out.append(f"- {count} 回 {name}")
    memory = [stall for stall in stalls if stall.private_mb is not None]
    if len(memory) >= 2:
        first, last = memory[0], memory[-1]
        peak = max(stall.private_mb or 0.0 for stall in memory)
        out.append("")
        out.append(
            f"メモリ（コミット） 初め {first.private_mb:.0f} MB → 終わり {last.private_mb:.0f} MB"
            f" 最大 {peak:.0f} MB"
        )
    return "\n".join(out)


def default_paths() -> list[Path]:
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        return []
    log = Path(root) / "Sashimono" / "diagnostics" / "stalls.jsonl"
    # 古い世代から読む 時刻で並べ直すので順は結果に響かないが、読む人が追いやすい
    return [log.with_suffix(".jsonl.1"), log]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="*", type=Path, help="記録のファイル（省けばこの機械の物）")
    parser.add_argument("--since", default=None, help="この時刻（ISO 形式の頭）より後だけを数える")
    parser.add_argument("--min-ms", type=float, default=0.0, help="これより短い止まりは数えない")
    parser.add_argument("--top", type=int, default=10, help="場所とスレッドを何件まで出すか")
    args = parser.parse_args(argv)
    paths = args.paths or default_paths()
    records = [
        record
        for record in read(paths)
        if float(record["stall_ms"]) >= args.min_ms
        and (args.since is None or str(record.get("time", "")) >= args.since)
    ]
    print(report(summarize(records), top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
