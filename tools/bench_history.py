r"""代表の測りを版ごとに残し、同じ機械の前の版と比べて遅くなった所を見つける（#513）

    .venv\Scripts\python.exe tools\bench_history.py run              （測って今の版の記録を残す）
    .venv\Scripts\python.exe tools\bench_history.py run --repeat 3 --only seek timeline
    .venv\Scripts\python.exe tools\bench_history.py compare          （最新と、同じ機械の 1 つ前）
    .venv\Scripts\python.exe tools\bench_history.py compare --base 0.2.6 --target 0.2.7
    .venv\Scripts\python.exe tools\bench_history.py show 0.2.6

測るのは :data:`BENCHES` の 3 本（シーク・合成・タイムライン） どれも ``--json`` で
``tools/bench_results.py`` の形の記録を書く 条件はここで決め打ちにする（版ごとに条件が
変わると、速さの差なのか条件の差なのか分からない） 変えるときは前の記録と比べられなくなる
ので、比べる側は条件の違う測りを比べずに「条件が違う」と出す

1 本ずつ別のプロセスで ``--repeat`` 回走らせ、回すたびに 3 本の順を巡らせる 1 本を続けて
回すと、その間だけ重かった（裏で更新が走ったなど）ときに、その測りの全部の回が引きずられる

記録は機械ごとの置き場（既定 ``%LOCALAPPDATA%\Sashimono\bench-history`` 環境変数
``SASHIMONO_BENCH_HISTORY`` で変えられる）に ``<版>.json`` で置く 速さは機械で決まるので、
リポジトリには入れない 比べるのは CPU・コアの数・OS（と GPU を使う測りは GPU）が同じ記録だけ

項目ごとに、回ごとの中央値の並びから次を決める

- 異常値 並びの中央から大きく外れた回（:data:`OUTLIER_RATIO` か MAD の 3 倍の大きい方）
  外して数え、報告には残す
- 安定 外した後に :data:`MIN_RUNS` 回以上残り、幅（最大 − 最小）/ 中央が
  :data:`STABLE_SPREAD` 以下 **安定な項目どうしだけを退行の判定に使う**
- 遅くなった 両方とも安定で、比が 1 + 余裕（:data:`REGRESSION_RATIO` か、両方の幅の和の
  大きい方）を超えた 揺れの大きい項目は「揺れが大きい（判定に使わない）」と分けて出す
- 落ちた・欠けた・新しい 測りが落ちた回（終了コード・時間切れ・記録が読めない）は、飛ばした
  （ffmpeg や GPU が無い）のと分けて出す

``compare`` は遅くなった項目があれば終了コード 1、比べる相手が無い（記録が無い・機械が違う）
ときは 2 落ちた・揺れが大きいだけでは 0（揺れで毎回止まる確かめは、そのうち見られなくなる）
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from bench_results import (  # noqa: E402
    FAILED,
    MACHINE_KEYS,
    OK,
    SKIPPED,
    check_run,
    machine,
    read_record,
)

#: 版ごとの記録の形 中身の意味を変えたら数を上げる
SCHEMA = "sashimono-bench-history/1"


@dataclass(frozen=True, slots=True)
class Bench:
    """代表の測り 1 本 ``arguments`` は決め打ちの条件 ``timeout`` は 1 回の上限（秒）"""

    script: str
    arguments: tuple[str, ...]
    timeout: float


#: 代表の測り デコードとシーク・GPU の合成・画面の描画（Qt）で、重さの出どころが違う物を 1 本ずつ
#: 書き出し・控え・画面との取り合い（bench_contention）は 1 回が数分かかり、窓を画面に出す物も
#: あるので入れない 素材の長さと回数は、3 本を 1 回りして 1 分ほどに収まるよう縮めてある
BENCHES: dict[str, Bench] = {
    "seek": Bench("bench_seek.py", ("--gaps", "1", "8", "--seconds", "20", "--jumps", "40"), 600.0),
    "render": Bench(
        "bench_render.py",
        (
            *("--frames", "20", "--depth", "3", "--tracks", "10"),
            *("--effects", "2", "--texts", "10", "--videos", "1"),
        ),
        600.0,
    ),
    "timeline": Bench("bench_timeline.py", ("--clips", "3000", "--repeat", "3"), 600.0),
}

#: 既定の回数 異常値を 1 回外しても :data:`MIN_RUNS` 回残る数
DEFAULT_REPEAT = 5
#: 安定と見なすのに要る、異常値を外した後の回数 2 回では幅が偶然狭いだけかを見分けられない
MIN_RUNS = 3
#: 安定と見なす幅（最大 − 最小）/ 中央 2026-11-24 の手元（RTX 5060 Ti 別の試験を並べて
#: 走らせている最中）の 5 回で、10 ms を超える項目とシークは 2〜9%、3 ms に届かない軽い項目
#: （浅いシーン・動画 1 本・全体表示）は 13〜30% だった 軽い項目が判定から外れる線にする
STABLE_SPREAD = 0.10
#: 遅くなったと言う最小の比の余裕 これより小さい差は、安定でも機械の温まり方で出る
REGRESSION_RATIO = 0.10
#: 異常値と見なす中央からの外れ（比） MAD が小さすぎる（3 回がほぼ同じ）ときの下限
OUTLIER_RATIO = 0.25
#: 失敗の理由に残す標準エラーの行数 全部残すと記録が読めなくなる
ERROR_LINES = 8

ENV_HISTORY = "SASHIMONO_BENCH_HISTORY"


def default_history() -> Path:
    """記録の置き場 環境変数が無ければ本人の LOCALAPPDATA の下（機械ごとの物なので）"""
    configured = os.environ.get(ENV_HISTORY)
    if configured:
        return Path(configured)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Sashimono" / "bench-history"


def source_version() -> str:
    """版の出どころ（``__version__``）を文字で読む 本体を import すると Qt まで読み込む"""
    text = (ROOT / "src" / "sashimono" / "__init__.py").read_text(encoding="utf-8")
    found = re.search(r'^__version__\s*=\s*"([^"]+)"', text, flags=re.M)
    return found.group(1) if found else "不明"


def source_commit() -> str:
    """測った commit 取れなければ空（git の無い所でも測れるように）"""
    try:
        # 引数は固定 shell は通さない
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


# --- 走らせる -------------------------------------------------------------------

#: 1 回走らせる口 (台本の場所, 引数, 記録を書く場所, 上限秒) → (終了コード, 標準エラー)
#: 終了コードが None なら時間切れ 試験では差し替える
Execute = Callable[[Path, Sequence[str], Path, float], tuple[int | None, str]]


def execute(
    script: Path, arguments: Sequence[str], output: Path, timeout: float
) -> tuple[int | None, str]:
    """測りを別のプロセスで 1 回走らせる

    同じプロセスで続けて回すと、前の回の GL の資源や控えが残り、2 回目から速く出る
    Windows では窓を出さない（測りは画面を持たないが、コンソールの窓が前へ割り込む）
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        # 走らせるのはこの道具が決めた台本と引数だけ shell は通さない
        result = subprocess.run(
            [sys.executable, str(script), *arguments, "--json", str(output)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=flags,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, ""
    return result.returncode, result.stderr or ""


def _tail(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-ERROR_LINES:])


def run_once(name: str, bench: Bench, index: int, work: Path, call: Execute) -> dict[str, Any]:
    """1 回走らせて、記録か落ちた理由を返す

    記録が読めればそれを信じる（予算を超えた合成は終了コード 1 でも測れている）
    記録が無い・読めない・時間切れは落ちた回 飛ばした回（``skipped``）とは分ける
    """
    output = work / f"{name}-{index}.json"
    code, error = call(TOOLS / bench.script, bench.arguments, output, bench.timeout)
    if code is None:
        return {"run": index, "status": FAILED, "reason": f"{bench.timeout:.0f} 秒で終わらない"}
    try:
        record = read_record(output)
    except ValueError as exc:
        detail = _tail(error)
        reason = f"終了コード {code} 記録が無い（{exc}）"
        return {
            "run": index,
            "status": FAILED,
            "reason": reason + (f"\n{detail}" if detail else ""),
        }
    return {"run": index, **record}


def run_benches(
    names: Sequence[str], repeat: int, *, call: Execute = execute, work: Path | None = None
) -> dict[str, Any]:
    """代表の測りを ``repeat`` 回ずつ巡らせて走らせ、版ごとの記録を組む"""
    benches: dict[str, Any] = {
        name: {
            "script": BENCHES[name].script,
            "arguments": list(BENCHES[name].arguments),
            "runs": [],
        }
        for name in names
    }
    with tempfile.TemporaryDirectory(prefix="sashimono-bench-history-") as folder:
        place = work or Path(folder)
        for index in range(1, repeat + 1):
            for name in names:
                print(f"測る: {name} {index}/{repeat}", flush=True)
                result = run_once(name, BENCHES[name], index, place, call)
                if result.get("status") != OK:
                    print(f"  {result.get('status')}: {result.get('reason', '')}", flush=True)
                benches[name]["runs"].append(result)
    return {
        "schema": SCHEMA,
        "version": source_version(),
        "commit": source_commit(),
        "taken": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": machine(),
        "repeat": repeat,
        "benches": benches,
    }


# --- 数える ---------------------------------------------------------------------


@dataclass
class ItemSummary:
    """1 つの項目の、回ごとの中央値の並びから数えた物"""

    name: str
    label: str
    values: list[float] = field(default_factory=list)
    outliers: list[float] = field(default_factory=list)
    failed_runs: int = 0

    @property
    def center(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    @property
    def spread(self) -> float:
        if not self.values or self.center <= 0:
            return 0.0
        return (max(self.values) - min(self.values)) / self.center

    @property
    def stable(self) -> bool:
        return len(self.values) >= MIN_RUNS and self.spread <= STABLE_SPREAD


def split_outliers(values: list[float]) -> tuple[list[float], list[float]]:
    """回ごとの値を、残す物と異常値に分ける

    外れの線は MAD（中央からの隔たりの中央）の 3 倍と、中央の :data:`OUTLIER_RATIO` の大きい方
    MAD だけだと、3 回がほぼ同じ値のときに線が 0 近くまで縮み、わずかな揺れを異常値と呼ぶ
    2 回以下では中央が決まらないので外さない
    """
    if len(values) < MIN_RUNS:
        return list(values), []
    middle = statistics.median(values)
    deviation = statistics.median(abs(value - middle) for value in values)
    limit = max(3 * 1.4826 * deviation, OUTLIER_RATIO * middle)
    kept = [value for value in values if abs(value - middle) <= limit]
    return kept, [value for value in values if abs(value - middle) > limit]


@dataclass
class BenchSummary:
    """1 本の測りの数え 飛ばした・落ちた回と、項目ごとの数え"""

    name: str
    conditions: dict[str, Any] | None
    gpu: str | None
    machine: dict[str, Any] | None
    skipped: str = ""
    failures: list[str] = field(default_factory=list)
    items: dict[str, ItemSummary] = field(default_factory=dict)


def summarize(name: str, entry: dict[str, Any]) -> BenchSummary:
    runs = [run for run in entry.get("runs", []) if isinstance(run, dict)]
    measured = [run for run in runs if run.get("status") in (OK, FAILED) and "items" in run]
    first = measured[0] if measured else None
    summary = BenchSummary(
        name,
        first.get("conditions") if first else None,
        (first.get("machine") or {}).get("gpu") if first else None,
        first.get("machine") if first else None,
    )
    # 測れた回があっても、飛ばした回が 1 つでもあれば理由を残す（比べる側はその測りを比べない）
    # 一部の回だけ飛ばした記録は、機械の様子が回の間で変わった印で、残りの回も信じきれない
    skipped = [run for run in runs if run.get("status") == SKIPPED]
    if skipped:
        summary.skipped = str(skipped[0].get("reason") or "飛ばした")
    for run in runs:
        if run.get("status") == FAILED:
            summary.failures.append(f"{run.get('run')} 回目: {run.get('reason', '')}")
    # 最初の回と GPU・条件の違う回は標本に混ぜない 混ぜると別の機械や条件の値が 1 つの中央に
    # まとまり、比べる側は最初の回の値しか見ないので気付けない 落ちた回として数える
    usable = []
    for run in measured:
        gpu = (run.get("machine") or {}).get("gpu")
        if run.get("conditions") != summary.conditions or gpu != summary.gpu:
            summary.failures.append(
                f"{run.get('run')} 回目: GPU か条件が最初の回と違うので数えない"
            )
            continue
        usable.append(run)
    raw: dict[str, tuple[str, list[float], int]] = {}
    for run in usable:
        for item in run.get("items", []):
            key = str(item.get("name"))
            label, values, failed = raw.get(key, (str(item.get("label") or key), [], 0))
            if item.get("status") == OK and isinstance(item.get("median"), (int, float)):
                values.append(float(item["median"]))
            else:
                failed += 1
            raw[key] = (label, values, failed)
    for key, (label, values, failed) in raw.items():
        kept, outliers = split_outliers(values)
        summary.items[key] = ItemSummary(key, label, kept, outliers, failed)
    return summary


# --- 比べる ---------------------------------------------------------------------

SLOWER = "遅くなった"
FASTER = "速くなった"
SAME = "変わらない"
NOISY = "揺れが大きい（判定に使わない）"
FAILED_ITEM = "測れなかった"
MISSING = "欠けた"
NEW = "新しい"


@dataclass
class Verdict:
    bench: str
    name: str
    label: str
    kind: str
    base: float = 0.0
    target: float = 0.0
    note: str = ""

    @property
    def ratio(self) -> float:
        return self.target / self.base if self.base > 0 else 0.0


@dataclass
class Comparison:
    """2 つの版の比べ 測りごとの言い分（比べなかった理由）と、項目ごとの判定"""

    base_label: str
    target_label: str
    notes: list[str] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def regressions(self) -> list[Verdict]:
        return [verdict for verdict in self.verdicts if verdict.kind == SLOWER]


def same_machine(base: dict[str, Any] | None, target: dict[str, Any] | None) -> bool:
    if not base or not target:
        return False
    return all(base.get(key) == target.get(key) for key in MACHINE_KEYS)


def machine_text(found: dict[str, Any] | None) -> str:
    if not found:
        return "不明"
    parts = [str(found.get("cpu")), f"{found.get('cores')} コア", str(found.get("os"))]
    if found.get("gpu"):
        parts.append(str(found.get("gpu")))
    return " / ".join(parts)


def judge(bench: str, base: ItemSummary, target: ItemSummary) -> Verdict:
    """安定な項目どうしだけを、余裕を超えた比で遅い・速いと言う"""
    verdict = Verdict(bench, target.name, target.label, SAME, base.center, target.center)
    if not target.values:
        verdict.kind = FAILED_ITEM
        return verdict
    if not (base.stable and target.stable):
        verdict.kind = NOISY
        verdict.note = f"幅 {base.spread:.0%} → {target.spread:.0%}（{len(target.values)} 回）"
        return verdict
    margin = max(REGRESSION_RATIO, base.spread + target.spread)
    if verdict.ratio > 1 + margin:
        verdict.kind = SLOWER
    elif verdict.ratio < 1 - margin:
        verdict.kind = FASTER
    verdict.note = f"余裕 {margin:.0%}"
    return verdict


def label_of(data: dict[str, Any]) -> str:
    """記録の名前 ``--label`` で付けた物 無ければ版（同じ版の開発中の測りを見分けるため）"""
    return str(data.get("label") or data.get("version"))


def state_notes(name: str, summary: BenchSummary) -> list[str]:
    """今回の記録だけで分かる事（落ちた回・飛ばした回・項目ごとの測れなかった回と異常値）

    比べられるかどうかに関わらず出す 比べない枝（機械や GPU が違う・前の記録に無い）で先に
    抜けると、``compare`` だけを見る人には、なぜ測れなかったのかが見えない
    """
    notes = [f"{name}: 落ちた {failure}" for failure in summary.failures]
    if summary.skipped:
        notes.append(f"{name}: 飛ばした回がある（{summary.skipped}）")
    for key, item in summary.items.items():
        if item.failed_runs:
            notes.append(f"{name}/{key}: 測れなかった回 {item.failed_runs}")
        if item.outliers:
            shown = "、".join(f"{value:.2f}" for value in item.outliers)
            notes.append(f"{name}/{key}: 異常値として外した回 {shown} ms")
    return notes


def compare(base: dict[str, Any], target: dict[str, Any]) -> Comparison:
    """2 つの版の記録を比べる 機械・GPU・条件の違う測りは比べずに理由を残す"""
    result = Comparison(label_of(base), label_of(target))
    current = {name: summarize(name, entry) for name, entry in target.get("benches", {}).items()}
    for name, summary in sorted(current.items()):
        result.notes += state_notes(name, summary)
    if not same_machine(base.get("machine"), target.get("machine")):
        result.notes.insert(
            0,
            "機械が違うので比べない（"
            f"{machine_text(base.get('machine'))} → {machine_text(target.get('machine'))}）",
        )
        return result
    names = sorted(set(base.get("benches", {})) | set(current))
    for name in names:
        before = base.get("benches", {}).get(name)
        now = current.get(name)
        if now is None:
            result.notes.append(f"{name}: 今回は測っていない")
            continue
        if before is None:
            result.notes.append(f"{name}: 前の記録に無い（今回から測った）")
            continue
        old = summarize(name, before)
        if now.skipped or old.skipped:
            reason = now.skipped or old.skipped
            result.notes.append(f"{name}: 飛ばした回があるので比べない（{reason}）")
            continue
        if old.gpu != now.gpu:
            result.notes.append(f"{name}: GPU が違うので比べない（{old.gpu} → {now.gpu}）")
            continue
        if old.conditions != now.conditions:
            result.notes.append(
                f"{name}: 条件が違うので比べない（{old.conditions} → {now.conditions}）"
            )
            continue
        for key in sorted(set(old.items) | set(now.items)):
            if key not in now.items:
                # 落ちた回があるなら、そこまで届かずに測れなかっただけかもしれない 消えたと
                # 言うのは、全部の回が測れたのに項目が無いときだけ
                kind = FAILED_ITEM if now.failures else MISSING
                result.verdicts.append(Verdict(name, key, old.items[key].label, kind))
                continue
            if key not in old.items:
                item = now.items[key]
                result.verdicts.append(Verdict(name, key, item.label, NEW, 0.0, item.center))
                continue
            result.verdicts.append(judge(name, old.items[key], now.items[key]))
    return result


def report(comparison: Comparison) -> str:
    lines = [f"{comparison.base_label} → {comparison.target_label}"]
    lines += [f"  {note}" for note in comparison.notes]
    order = (SLOWER, FASTER, SAME, NOISY, FAILED_ITEM, MISSING, NEW)
    for kind in order:
        chosen = [verdict for verdict in comparison.verdicts if verdict.kind == kind]
        if not chosen:
            continue
        lines.append(f"[{kind}] {len(chosen)} 項目")
        for verdict in chosen:
            numbers = ""
            if verdict.base and verdict.target:
                numbers = f" {verdict.base:.2f} → {verdict.target:.2f} ms（{verdict.ratio:.2f} 倍）"
            elif verdict.target:
                numbers = f" {verdict.target:.2f} ms"
            note = f" {verdict.note}" if verdict.note else ""
            lines.append(f"  {verdict.bench}/{verdict.name} {verdict.label}{numbers}{note}")
    return "\n".join(lines)


def describe(data: dict[str, Any]) -> str:
    """1 つの版の記録を、項目ごとの中央・幅・安定か で並べる"""
    lines = [
        f"{label_of(data)}（{data.get('version')} {str(data.get('commit', ''))[:7]}"
        f" {data.get('taken', '')}）",
        f"  機械 {machine_text(data.get('machine'))}",
    ]
    for name, entry in sorted(data.get("benches", {}).items()):
        summary = summarize(name, entry)
        head = f"{name}" + (f"（GPU {summary.gpu}）" if summary.gpu else "")
        # 飛ばした回があっても、落ちた回の理由と項目ごとの数は出す（比べないのは compare の側）
        lines.append(f"{head}: 飛ばした（{summary.skipped}）" if summary.skipped else head)
        for failure in summary.failures:
            lines.append(f"  落ちた {failure}")
        for item in summary.items.values():
            state = "安定" if item.stable else "揺れが大きい"
            outliers = f" 異常値 {len(item.outliers)} 回" if item.outliers else ""
            if item.failed_runs:
                outliers += f" 測れなかった回 {item.failed_runs}"
            lines.append(
                f"  {item.name} {item.label} 中央 {item.center:.2f} ms 幅 {item.spread:.0%}"
                f"（{len(item.values)} 回）{state}{outliers}"
            )
    return "\n".join(lines)


# --- 置き場 ---------------------------------------------------------------------


def read_history(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} を読めない: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path} は {SCHEMA} の記録ではない")
    # 中まで見る 形だけ正しい壊れた記録が最新になると、既定の show と compare がそこで落ち、
    # ほかの正しい記録まで見られなくなる
    benches = data.get("benches")
    if not isinstance(benches, dict) or not isinstance(data.get("machine"), dict):
        raise ValueError(f"{path} の中身が欠けている")
    for name, entry in benches.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("runs"), list):
            raise ValueError(f"{path} の {name} の中身が欠けている")
        for run in entry["runs"]:
            check_run(run, f"{path} の {name}")
    taken(data, path)
    return data


def taken(data: dict[str, Any], path: Path | None = None) -> datetime.datetime:
    """測った時刻 時差の付いた日時として読む

    文字のまま並べると、夏時間の終わりのように時差が変わる前後で順が逆になる
    """
    try:
        found = datetime.datetime.fromisoformat(str(data.get("taken", "")))
    except ValueError as exc:
        raise ValueError(f"{path or label_of(data)} の測った時刻を読めない（{exc}）") from exc
    if found.tzinfo is None:
        raise ValueError(f"{path or label_of(data)} の測った時刻に時差が無い")
    return found


def history_files(folder: Path) -> list[tuple[Path, dict[str, Any]]]:
    """置き場の記録を、測った時刻の順に 読めない物は飛ばして知らせる"""
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(folder.glob("*.json")):
        try:
            found.append((path, read_history(path)))
        except ValueError as exc:
            print(f"[--] {exc}")
    return sorted(found, key=lambda pair: taken(pair[1]))


def pick(
    records: list[tuple[Path, dict[str, Any]]], base: str | None, target: str | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """比べる 2 つを選ぶ 既定は最新と、それより前で同じ機械の最新"""

    def named(label: str) -> dict[str, Any] | None:
        for path, found in reversed(records):
            # 名前は記録の名前・ファイル名・版のどれでも引く（ファイル名は使えない字を _ にする）
            if label in (found.get("label"), found.get("version"), path.stem):
                chosen: dict[str, Any] = found
                return chosen
        return None

    chosen = named(target) if target else (records[-1][1] if records else None)
    if chosen is None:
        return None, None
    if base:
        return named(base), chosen
    earlier = [
        data
        for _, data in records
        if data is not chosen
        # 同じ秒に測った物も前と見る（測り直しを続けて回したとき 時刻は秒までしか残さない）
        and taken(data) <= taken(chosen)
        and same_machine(data.get("machine"), chosen.get("machine"))
    ]
    return (earlier[-1] if earlier else None), chosen


def _safe_name(label: str) -> str:
    """記録のファイル名 日本語などの字はそのまま使い、使えない字だけを _ にする

    置き換えた名前には元の名前の要約を付ける 付けないと「変更 前」と「変更_前」のように違う名前が
    同じファイルになり、後の測りが前の記録を黙って置き換える 同じ名前の測り直しは同じファイル
    """
    safe = re.sub(r"[^\w.+-]", "_", label)
    if safe == label:
        return safe
    return f"{safe}-{hashlib.sha256(label.encode('utf-8')).hexdigest()[:8]}"


def main(argv: list[str] | None = None, *, call: Execute = execute) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--history", type=Path, default=None, help="記録の置き場")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="代表の測りを走らせて今の版の記録を残す")
    run.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="1 本を何回走らせるか")
    run.add_argument("--only", nargs="+", choices=sorted(BENCHES), help="走らせる測り")
    run.add_argument("--label", default=None, help="記録の名前（既定は __version__）")
    run.add_argument("--no-compare", action="store_true", help="残した後に前の版と比べない")
    comparing = commands.add_parser("compare", help="2 つの版を比べる")
    comparing.add_argument("--base", default=None, help="前の版（既定は同じ機械の 1 つ前）")
    comparing.add_argument("--target", default=None, help="今の版（既定は最新）")
    comparing.add_argument("files", nargs="*", type=Path, help="記録のファイルを 2 つ直に渡す")
    showing = commands.add_parser("show", help="1 つの版の記録を項目ごとに出す")
    showing.add_argument("label", nargs="?", default=None, help="版かファイル（既定は最新）")
    args = parser.parse_args(argv)
    folder = args.history or default_history()

    if args.command == "run":
        if args.repeat < 1:
            parser.error("--repeat は 1 以上にする")
        if args.repeat < MIN_RUNS:
            print(f"[--] {args.repeat} 回では安定かを決められず、どの項目も退行の判定に使わない")
        names = args.only or list(BENCHES)
        data = run_benches(names, args.repeat, call=call)
        label = args.label or str(data["version"])
        data["label"] = label
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{_safe_name(label)}.json"
        if target.exists():
            print(f"[--] 前の {target.name} を置き換える（同じ版で測り直した）")
        temporary = target.with_name(target.name + ".writing")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        print(describe(data))
        print(f"残した: {target}")
        if args.no_compare:
            return 0
        base, chosen = pick(history_files(folder), None, label)
        if base is None or chosen is None:
            print("同じ機械の前の記録が無いので比べない")
            return 0
        comparison = compare(base, chosen)
        print(report(comparison))
        return 1 if comparison.regressions else 0

    if args.command == "show":
        if args.label and Path(args.label).is_file():
            data = read_history(Path(args.label))
        else:
            _, found = pick(history_files(folder), None, args.label)
            if found is None:
                print(f"{folder} に記録が無い")
                return 2
            data = found
        print(describe(data))
        return 0

    if args.files:
        if len(args.files) != 2:
            parser.error("ファイルで渡すときは 前 今 の 2 つ")
        base_data: dict[str, Any] | None = read_history(args.files[0])
        target_data: dict[str, Any] | None = read_history(args.files[1])
    else:
        base_data, target_data = pick(history_files(folder), args.base, args.target)
    if base_data is None or target_data is None:
        print(f"比べる 2 つの記録が無い（{folder}）")
        return 2
    comparison = compare(base_data, target_data)
    print(report(comparison))
    if not same_machine(base_data.get("machine"), target_data.get("machine")):
        # 比べられる相手が無いのと同じ 落ちた・揺れが大きいだけ（判定の無い比べ）は 0 のまま
        return 2
    return 1 if comparison.regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
