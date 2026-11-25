"""画面の止まりの記録を数え、原因ごとに分ける（#455 #516）

開発者の道具（``tools/stall_report.py``）と、利用者が作る診断情報（:mod:`sashimono.diagnostic_report`）が
同じ数え方を使う 別々に書くと、報告に載った表と開発者が手元で数えた表が食い違う
記録の形は :mod:`sashimono.stall_watch` が書く物

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

Qt を使わない
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sashimono.stall_watch import default_log_path

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
    "parse",
    "read",
    "report",
    "summarize",
]

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


def read(
    paths: Iterable[Path],
    *,
    unread: list[Path] | None = None,
    skipped: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """記録を読む 壊れた行（書いている最中に落ちた等）は飛ばす

    読めなかったファイルは ``unread`` へ足す JSON としては読めても欄の型が違う行（記録の形が
    変わった後・手で書き換えた後）も飛ばし、1 行ごとにそのファイルを ``skipped`` へ足す
    通すと数える所（:func:`parse`）で落ち、同じファイルの正しい行も、診断情報のほかの記録も
    集められなくなる（PR #550 の CodeRabbit の指摘）
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            if unread is not None:
                unread.append(path)
            continue
        # 行ごとに文字へ戻す まとめて戻すと、書いている最中に落ちて文字の途中で切れた最後の
        # 行のせいで、前の正しい記録まで読めなくなる
        for raw in data.splitlines():
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or "stall_ms" not in record:
                continue
            if _well_formed(record):
                records.append(record)
            elif skipped is not None:
                skipped.append(path)
    records.sort(key=lambda record: str(record.get("time", "")))
    return records


def _number(value: object) -> bool:
    """限りのある数か（真偽は数に数えない JSON の true が 1 ms の止まりになる）

    限りの無い数（JSON の ``1e309`` は ``inf`` に読める）と数でない数は通さない 通すと合計・割合・
    中央値が ``inf`` や ``nan%`` になり、同じ報告の正しい止まりの値まで読めなくなる
    （PR #550 の指摘）
    小数にできないほど大きな整数も通さない（数える所の ``float()`` で落ちる）
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _well_formed(record: dict[str, Any]) -> bool:
    """:func:`parse` が読む欄の型が合っているか 省いた欄（``gc`` など）は合っていると見る"""
    if not _number(record.get("stall_ms")):
        return False
    pauses = record.get("gc") or []
    if not isinstance(pauses, list) or not all(
        isinstance(pause, dict) and _number(pause.get("ms", 0)) for pause in pauses
    ):
        return False
    samples = record.get("samples") or []
    if not isinstance(samples, list) or not all(isinstance(sample, dict) for sample in samples):
        return False
    if samples:
        threads = samples[0].get("threads", {})
        if not isinstance(threads, dict) or not all(
            isinstance(name, str)
            and isinstance(lines, list)
            and all(isinstance(line, str) for line in lines)
            for name, lines in threads.items()
        ):
            return False
    memory = record.get("memory_mb") or {}
    if not isinstance(memory, dict):
        return False
    private = memory.get("private")
    return private is None or _number(private)


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
        # 名前だけで見る 本体の stall_watch.py は置き場ごと書かれ、Python の置き場の下に
        # sashimono という名前のフォルダがある機械では threading.py も置き場ごと書かれる
        if _file(lines[-1]).rsplit("/", 1)[-1] in _WAITING:
            continue
        found.append(name.split(" #", 1)[0])
    return found


def classify(stall: Stall) -> str:
    if stall.ms > 0 and stall.gc_ms >= stall.ms / 2:
        return GC
    if not stall.threads:
        return UNSAMPLED
    main = _main_thread(stall)
    # 画面の部品はいちばん内側の物で決める タイムラインで選ぶと、選択の知らせを受けた
    # 字幕パネルが一覧を作り直す 外側の選んだ所で数えると、字幕パネルの重さがタイムラインに
    # 付いて、どこを直すか読み違える（#455 の実プロジェクトで 19 回が字幕パネルだった）
    panel = next((line for line in reversed(main) if _inside(line, "ui/")), None)
    if panel is not None and _inside(panel, "ui/inspector/"):
        return INSPECTOR
    if any(
        _function(line) in ("paintGL", "paintEvent")
        or _inside(line, "engine/render/", "engine/gpu/")
        for line in main
    ):
        return PAINT
    if panel is not None and _inside(panel, "ui/timeline/"):
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
    # 記録する側と同じ置き場を使う 自分で組み立てると、LOCALAPPDATA の無い機械
    # （XDG_STATE_HOME やホームの下へ書く）で、記録があるのに「無い」と出る
    log = default_log_path()
    # 古い世代から読む 時刻で並べ直すので順は結果に響かないが、読む人が追いやすい
    return [log.with_suffix(".jsonl.1"), log]
