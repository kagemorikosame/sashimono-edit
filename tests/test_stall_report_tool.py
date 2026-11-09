"""画面の止まりの記録を数える道具（tools/stall_report.py #455）

記録は :mod:`sashimono.stall_watch` が書く形をそのまま手で作る 本物の止まりを起こすと
並びが機械ごとに違い、どの原因に分かれるかを決め打ちで確かめられない
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stall_report", ROOT / "tools" / "stall_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LOOP = ["sashimono/app.py:120 main"]
IDLE_WORKER = ["threading.py:1012 run", "queue.py:180 get"]


def _record(
    ms: float,
    main: list[str] | None,
    *,
    others: dict[str, list[str]] | None = None,
    gc: list[dict[str, Any]] | None = None,
    private: float | None = None,
    time: str = "2026-11-10T10:00:00.000",
) -> dict[str, Any]:
    threads = {} if main is None else {"MainThread": main, **(others or {})}
    record: dict[str, Any] = {
        "time": time,
        "stall_ms": ms,
        "threshold_ms": 30.0,
        "gc": gc or [],
        "samples": [{"at_ms": 31.0, "threads": threads}] if threads else [],
    }
    if private is not None:
        record["memory_mb"] = {"resident": private, "private": private}
    return record


def test_garbage_collection_wins_when_it_fills_the_stall(tool: ModuleType) -> None:
    """壊れると、全体回で止まった所が設定パネルの止まりとして数えられ、手で回す設定を入れる判断を誤る"""
    record = _record(
        60,
        [*LOOP, "sashimono/ui/inspector/panel.py:300 _rebuild"],
        gc=[{"generation": 2, "ms": 40.0}],
    )
    assert tool.classify(tool.parse(record)) == tool.GC


def test_the_inspector_is_told_apart_from_the_chat_panel(tool: ModuleType) -> None:
    """壊れると、会話の panel.py で止まった所まで設定パネルに数えられる"""
    inspector = _record(50, [*LOOP, "sashimono/ui/inspector/panel.py:300 _rebuild"])
    chat = _record(50, [*LOOP, "sashimono/ui/chat/panel.py:300 _append"])
    assert tool.classify(tool.parse(inspector)) == tool.INSPECTOR
    assert tool.classify(tool.parse(chat)) == tool.OTHER


def test_painting_and_timeline_work_are_separated(tool: ModuleType) -> None:
    """壊れると、タイムラインの描画が計算の止まりに混ざり、どちらを直すべきか読めない"""
    paint = _record(50, [*LOOP, "sashimono/ui/timeline/view.py:817 paintEvent"])
    compute = _record(50, [*LOOP, "sashimono/ui/timeline/view.py:400 set_selection"])
    assert tool.classify(tool.parse(paint)) == tool.PAINT
    assert tool.classify(tool.parse(compute)) == tool.TIMELINE


def test_waiting_in_the_event_loop_while_a_worker_runs_is_background(tool: ModuleType) -> None:
    """壊れると、裏のスレッドと GIL を取り合って止まった所が「その他」に埋もれる"""
    record = _record(
        80,
        LOOP,
        others={
            "sashimono-prefetch": [
                "threading.py:1012 run",
                "sashimono/engine/render/background.py:399 _loop",
            ],
            "sashimono-analyze_0": IDLE_WORKER,
        },
    )
    stall = tool.parse(record)
    assert tool.classify(stall) == tool.BACKGROUND
    # 待っているだけのスレッドは取り合っていない 数えると、どの止まりにも解析の係が並ぶ
    assert tool.busy_threads(stall) == ["sashimono-prefetch"]


def test_a_stall_without_samples_is_counted_on_its_own(tool: ModuleType) -> None:
    """壊れると、線を少し超えただけの止まりが原因の分かった止まりに混ざる"""
    assert tool.classify(tool.parse(_record(31, None))) == tool.UNSAMPLED


def test_the_report_counts_causes_places_and_memory(tool: ModuleType, tmp_path: Path) -> None:
    """壊れると、利用者から届いた記録を読んでも、どの原因が何回・何 ms かが出ない"""
    lines = [
        _record(
            40, [*LOOP, "sashimono/ui/inspector/panel.py:1 a"], private=900, time="2026-11-10T10:00"
        ),
        _record(
            60, [*LOOP, "sashimono/ui/inspector/panel.py:2 b"], private=950, time="2026-11-10T10:01"
        ),
        _record(
            50, [*LOOP, "sashimono/core/io/save.py:9 write"], private=1000, time="2026-11-10T10:02"
        ),
        "壊れた行",
    ]
    log = tmp_path / "stalls.jsonl"
    log.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines),
        encoding="utf-8",
    )
    summary = tool.summarize(tool.read([log]))
    assert len(summary.by_cause[tool.INSPECTOR]) == 2
    assert summary.places["core/io/save.py write"] == 1
    text = tool.report(summary)
    assert "| 設定パネル | 2 | 100 | 67% | 50 | 60 |" in text
    assert "初め 900 MB → 終わり 1000 MB" in text


def test_options_narrow_what_is_counted(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """壊れると、手順の前に溜まっていた古い記録や短い揺れまで数え、操作した間の割合が薄まる"""
    log = tmp_path / "stalls.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                _record(500, LOOP, time="2026-11-09T10:00"),
                _record(35, LOOP, time="2026-11-10T10:00"),
                _record(70, LOOP, time="2026-11-10T10:01"),
            )
        ),
        encoding="utf-8",
    )
    assert tool.main([str(log), "--since", "2026-11-10", "--min-ms", "50"]) == 0
    assert "止まり 1 回 合計 70 ms" in capsys.readouterr().out
