"""測りを版ごとに残して比べる道具（tools/bench_history.py tools/bench_results.py #513）

速さそのものは測らない（機械と負荷で変わる） 確かめるのは、揺れの大きい項目を退行の判定に
使わないこと・異常値を外すこと・落ちた回と飛ばした回を分けること・機械や GPU や条件の違う
記録を比べないこと・記録の無い回を「項目が無い」と取り違えないこと
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load("bench_history")


@pytest.fixture(scope="module")
def results() -> ModuleType:
    return _load("bench_results")


MACHINE = {
    "cpu": "試験の CPU",
    "cores": 8,
    "os": "Windows 11",
    "os_build": "10.0.1",
    "gpu": None,
    "python": "3.14.9",
}


def _record(
    medians: dict[str, float],
    *,
    status: str = "ok",
    gpu: str | None = None,
    conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """1 回の走りの記録（bench_results の形） 標本は中央値だけを 1 つ置く"""
    return {
        "schema": "sashimono-bench/1",
        "bench": "seek",
        "conditions": conditions or {"jumps": 40},
        "machine": {**MACHINE, "gpu": gpu},
        "status": status,
        "reason": "",
        "items": [
            {
                "name": name,
                "label": name,
                "status": "ok",
                "unit": "ms",
                "better": "lower",
                "samples": [value],
                "median": value,
                "p95": value,
            }
            for name, value in medians.items()
        ],
    }


def _history(
    version: str, taken: str, runs: dict[str, list[dict[str, Any]]], **machine: Any
) -> dict[str, Any]:
    return {
        "schema": "sashimono-bench-history/1",
        "version": version,
        "label": version,
        "commit": "a" * 40,
        "taken": taken,
        "machine": {**MACHINE, **machine},
        "repeat": 5,
        "benches": {
            name: {"script": f"bench_{name}.py", "arguments": [], "runs": entries}
            for name, entries in runs.items()
        },
    }


def _runs(values: Sequence[float], name: str = "gop1", **options: Any) -> list[dict[str, Any]]:
    return [
        {"run": index + 1, **_record({name: value}, **options)}
        for index, value in enumerate(values)
    ]


class TestStability:
    def test_an_outlier_run_is_left_out_and_reported(self, tool: ModuleType) -> None:
        """1 回だけ裏の作業で遅かった回を中央に混ぜると、その版だけ遅く見える"""
        kept, outliers = tool.split_outliers([10.0, 10.2, 9.9, 10.1, 30.0])
        assert outliers == [30.0]
        assert sorted(kept) == [9.9, 10.0, 10.1, 10.2]

    def test_nearly_equal_runs_are_not_called_outliers(self, tool: ModuleType) -> None:
        """MAD だけで線を引くと、ほぼ同じ 3 回のわずかな揺れまで異常値になる"""
        kept, outliers = tool.split_outliers([10.0, 10.0, 10.0, 10.3])
        assert outliers == []
        assert len(kept) == 4

    def test_two_runs_are_never_stable(self, tool: ModuleType) -> None:
        """2 回の幅が狭いのは偶然かもしれない 判定に使わない"""
        summary = tool.summarize("seek", {"runs": _runs([10.0, 10.0])})
        assert not summary.items["gop1"].stable

    def test_a_wide_item_is_not_stable(self, tool: ModuleType) -> None:
        """幅の広い項目を安定と見ると、揺れだけで退行と言い、毎回止まる確かめになる"""
        summary = tool.summarize("seek", {"runs": _runs([10.0, 11.5, 12.0, 10.5, 11.0])})
        item = summary.items["gop1"]
        assert item.spread > tool.STABLE_SPREAD
        assert not item.stable


class TestCompare:
    def _pair(
        self, before: Sequence[float], after: Sequence[float], **after_options: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base = _history("0.2.6", "2026-11-01T00:00:00+09:00", {"seek": _runs(before)})
        target = _history(
            "0.2.7", "2026-11-24T00:00:00+09:00", {"seek": _runs(after, **after_options)}
        )
        return base, target

    def test_a_stable_item_that_got_slower_is_a_regression(self, tool: ModuleType) -> None:
        """安定な項目が遅くなったのを見逃すと、版ごとに残す意味が無くなる"""
        base, target = self._pair([10.0, 10.1, 9.9, 10.0, 10.2], [13.0, 13.1, 12.9, 13.0, 13.2])
        comparison = tool.compare(base, target)
        assert [v.name for v in comparison.regressions] == ["gop1"]
        assert "遅くなった" in tool.report(comparison)

    def test_a_noisy_item_is_never_a_regression(self, tool: ModuleType) -> None:
        """揺れの大きい項目で止めると、毎回止まる確かめになり、そのうち見られなくなる"""
        base, target = self._pair([10.0, 10.1, 9.9, 10.0, 10.2], [12.0, 15.0, 13.0, 16.0, 14.0])
        comparison = tool.compare(base, target)
        assert comparison.regressions == []
        assert [v.kind for v in comparison.verdicts] == [tool.NOISY]

    def test_a_small_change_is_the_same(self, tool: ModuleType) -> None:
        """余裕より小さい差を遅いと言うと、機械の温まり方だけで止まる"""
        base, target = self._pair([10.0, 10.1, 9.9, 10.0, 10.2], [10.5, 10.6, 10.4, 10.5, 10.6])
        assert [v.kind for v in tool.compare(base, target).verdicts] == [tool.SAME]

    def test_a_faster_item_is_reported_but_does_not_fail(self, tool: ModuleType) -> None:
        """速くなったのを退行と取り違えると、直した版で止まる"""
        base, target = self._pair([10.0, 10.1, 9.9, 10.0, 10.2], [5.0, 5.1, 4.9, 5.0, 5.1])
        comparison = tool.compare(base, target)
        assert [v.kind for v in comparison.verdicts] == [tool.FASTER]
        assert comparison.regressions == []

    def test_another_machine_is_not_compared(self, tool: ModuleType) -> None:
        """別の機械の記録と比べると、機械の差を退行と言う"""
        base, target = self._pair([10.0] * 5, [20.0] * 5)
        target["machine"]["cpu"] = "別の CPU"
        comparison = tool.compare(base, target)
        assert comparison.verdicts == []
        assert any("機械が違う" in note for note in comparison.notes)

    def test_another_gpu_is_not_compared(self, tool: ModuleType) -> None:
        """同じ CPU でも GPU が違えば合成の速さは比べられない"""
        base = _history("0.2.6", "2026-11-01", {"render": _runs([10.0] * 5, gpu="GPU A")})
        target = _history("0.2.7", "2026-11-24", {"render": _runs([20.0] * 5, gpu="GPU B")})
        comparison = tool.compare(base, target)
        assert comparison.verdicts == []
        assert any("GPU が違う" in note for note in comparison.notes)

    def test_other_conditions_are_not_compared(self, tool: ModuleType) -> None:
        """条件の違う測りを比べると、条件の差を速さの差と取り違える"""
        base, target = self._pair([10.0] * 5, [20.0] * 5, conditions={"jumps": 80})
        comparison = tool.compare(base, target)
        assert comparison.verdicts == []
        assert any("条件が違う" in note for note in comparison.notes)

    def test_a_skipped_bench_is_not_a_missing_item(self, tool: ModuleType) -> None:
        """GPU の無い機械で飛ばした測りを、項目が消えた（欠けた）と言わない"""
        base, _ = self._pair([10.0] * 5, [10.0] * 5)
        skipped = [{"run": 1, **_record({}, status="skipped")}]
        skipped[0]["reason"] = "GPU が無い"
        target = _history("0.2.7", "2026-11-24", {"seek": skipped})
        comparison = tool.compare(base, target)
        assert not any(v.kind == tool.MISSING for v in comparison.verdicts)
        assert any("飛ばした" in note and "GPU が無い" in note for note in comparison.notes)

    def test_a_failed_run_is_reported_apart_from_the_items(self, tool: ModuleType) -> None:
        """落ちた回は理由と一緒に出す 残りの回で数えた項目は、回が減れば揺れの扱いになる"""
        base, _ = self._pair([10.0] * 5, [10.0] * 5)
        runs = _runs([10.0, 10.0])
        runs.append({"run": 3, "status": "failed", "reason": "終了コード 1 記録が無い"})
        target = _history("0.2.7", "2026-11-24", {"seek": runs})
        comparison = tool.compare(base, target)
        assert any("落ちた" in note and "記録が無い" in note for note in comparison.notes)
        assert [v.kind for v in comparison.verdicts] == [tool.NOISY]

    def test_a_missing_and_a_new_item_are_named(self, tool: ModuleType) -> None:
        """消えた項目と増えた項目を黙ると、測りの名前を変えたときに退行を見逃す"""
        base = _history("0.2.6", "2026-11-01", {"seek": _runs([10.0] * 5, name="gop1")})
        target = _history("0.2.7", "2026-11-24", {"seek": _runs([10.0] * 5, name="gop8")})
        kinds = {v.name: v.kind for v in tool.compare(base, target).verdicts}
        assert kinds == {"gop1": tool.MISSING, "gop8": tool.NEW}


class TestRun:
    def test_a_run_without_a_record_is_a_failure_with_the_error(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """落ちて記録を書かなかった回を、項目の無い回として数えない"""

        def crash(
            script: Path, arguments: Sequence[str], output: Path, timeout: float
        ) -> tuple[int | None, str]:
            return 1, "Traceback\nRuntimeError: 試験の失敗\n"

        result = tool.run_once("seek", tool.BENCHES["seek"], 1, tmp_path, crash)
        assert result["status"] == "failed"
        assert "終了コード 1" in result["reason"]
        assert "試験の失敗" in result["reason"]

    def test_a_timeout_is_a_failure(self, tool: ModuleType, tmp_path: Path) -> None:
        """時間切れを飛ばした扱いにすると、固まる退行を見逃す"""

        def hang(
            script: Path, arguments: Sequence[str], output: Path, timeout: float
        ) -> tuple[int | None, str]:
            return None, ""

        result = tool.run_once("seek", tool.BENCHES["seek"], 1, tmp_path, hang)
        assert result["status"] == "failed"
        assert "終わらない" in result["reason"]

    def test_a_record_counts_even_with_a_failing_exit_code(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """合成の測りは予算を超えると 1 で終わる 測れているので、記録を信じる"""

        def over_budget(
            script: Path, arguments: Sequence[str], output: Path, timeout: float
        ) -> tuple[int | None, str]:
            output.write_text(json.dumps(_record({"busy": 40.0})), encoding="utf-8")
            return 1, ""

        result = tool.run_once("render", tool.BENCHES["render"], 2, tmp_path, over_budget)
        assert result["status"] == "ok"
        assert result["run"] == 2
        assert result["items"][0]["median"] == 40.0

    def test_runs_take_turns_and_the_history_is_kept_and_compared(
        self, tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """回すたびに測りの順を巡らせ、版の名前で残し、同じ機械の前の記録と比べる"""
        order: list[str] = []

        def fake(
            script: Path, arguments: Sequence[str], output: Path, timeout: float
        ) -> tuple[int | None, str]:
            order.append(script.name)
            record = _record({"item": 10.0})
            record["machine"] = tool.machine()
            output.write_text(json.dumps(record), encoding="utf-8")
            return 0, ""

        history = tmp_path / "記録"
        argv = ["--history", str(history), "run", "--repeat", "3", "--only", "seek", "timeline"]
        assert tool.main([*argv, "--label", "0.2.6"], call=fake) == 0
        assert order == ["bench_seek.py", "bench_timeline.py"] * 3
        assert (history / "0.2.6.json").is_file()
        assert tool.main([*argv, "--label", "0.2.7"], call=fake) == 0
        out = capsys.readouterr().out
        assert "0.2.6 → 0.2.7" in out
        assert "[変わらない] 2 項目" in out


class TestRecord:
    def test_an_empty_item_is_a_failure_not_a_crash(
        self, results: ModuleType, tmp_path: Path
    ) -> None:
        """標本の無い項目で落ちると、測れた項目まで記録ごと消える"""
        record = results.BenchRecord("seek", {"jumps": 1})
        record.add("gop1", [])
        record.add("gop8", [3.0, 1.0, 2.0])
        path = tmp_path / "結果.json"
        record.write(path)
        data = results.read_record(path)
        assert [item["status"] for item in data["items"]] == ["failed", "ok"]
        assert data["items"][1]["median"] == 2.0
        assert not path.with_name(path.name + ".writing").exists()

    def test_another_schema_is_refused(self, results: ModuleType, tmp_path: Path) -> None:
        """形の違う記録を読むと、中身の意味を取り違えたまま比べる"""
        path = tmp_path / "古い.json"
        path.write_text(json.dumps({"schema": "sashimono-bench/0", "items": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="記録ではない"):
            results.read_record(path)

    def test_the_machine_has_no_personal_names(self, results: ModuleType) -> None:
        """記録は人へ渡すことがある 機械の名前や利用者の名前を入れない"""
        import getpass
        import platform

        found = json.dumps(results.machine("GPU"), ensure_ascii=False)
        assert platform.node() not in found
        assert getpass.getuser() not in found

    def test_the_seek_bench_records_a_skip_without_ffmpeg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ffmpeg の無い機械では測らない 落ちたのではなく飛ばしたと書く"""
        seek = _load("bench_seek")
        monkeypatch.setattr(seek.shutil, "which", lambda _name: None)
        path = tmp_path / "seek.json"
        assert seek.main(["--json", str(path)]) == 0
        data = _load("bench_results").read_record(path)
        assert data["status"] == "skipped"
        assert data["conditions"]["jumps"] == 80
