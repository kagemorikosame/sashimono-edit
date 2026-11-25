"""測りを版ごとに残して比べる道具（tools/bench_history.py tools/bench_results.py #513）

速さそのものは測らない（機械と負荷で変わる） 確かめるのは、揺れの大きい項目を退行の判定に
使わないこと・異常値を外すこと・落ちた回と飛ばした回を分けること・機械や GPU や条件の違う
記録を比べないこと・記録の無い回を「項目が無い」と取り違えないこと
"""

from __future__ import annotations

import argparse
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
        # 日付だけの書き方は、記録と同じ時差付きの日時にそろえる
        "taken": taken if "T" in taken else f"{taken}T00:00:00+09:00",
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


def _args() -> argparse.Namespace:
    """bench_seek の引数（素材の長さと飛ぶ数）"""
    return argparse.Namespace(seconds=10, jumps=5)


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
        """2 回の幅が狭いのは偶然かもしれない 安定と見ると、たまたまの差で退行と言って止まる"""
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
        """GPU の違う記録と比べると、同じ CPU でも合成の速さの差を退行と取り違える"""
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
        """落ちた回を黙ると、少ない回で数えた値を安定と信じて判定する 理由と一緒に出し、
        回が減った項目は揺れの扱いにする
        """
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

    def test_an_item_unreached_after_a_failure_is_not_missing(self, tool: ModuleType) -> None:
        """落ちて届かなかった項目を「欠けた」と言うと、測りの名前を消したのと取り違える"""
        base = _history("0.2.6", "2026-11-01", {"seek": _runs([10.0] * 5)})
        runs = _runs([10.0] * 3, name="gop8")
        runs.append({"run": 4, **_record({}, status="failed")})
        target = _history("0.2.7", "2026-11-24", {"seek": runs})
        kinds = {v.name: v.kind for v in tool.compare(base, target).verdicts}
        assert kinds["gop1"] == tool.FAILED_ITEM

    def test_a_partly_skipped_bench_is_not_compared(self, tool: ModuleType) -> None:
        """一部の回だけ飛ばした測りは、機械の様子が回の間で変わった印 残りの回でも比べない"""
        base = _history("0.2.6", "2026-11-01", {"seek": _runs([10.0] * 5)})
        runs = _runs([20.0] * 3)
        runs.append({"run": 4, **_record({}, status="skipped")})
        target = _history("0.2.7", "2026-11-24", {"seek": runs})
        comparison = tool.compare(base, target)
        assert comparison.verdicts == []
        assert any("飛ばした" in note for note in comparison.notes)

    def test_a_run_on_another_gpu_is_not_mixed_in(self, tool: ModuleType) -> None:
        """回の間で GPU が変わった値を 1 つの中央に混ぜると、別の機械の値で判定する"""
        runs = _runs([10.0] * 3, gpu="GPU A") + _runs([30.0] * 2, gpu="GPU B")
        summary = tool.summarize("render", {"runs": runs})
        assert summary.items["gop1"].values == [10.0, 10.0, 10.0]
        assert any("GPU か条件が最初の回と違う" in failure for failure in summary.failures)

    def test_failed_runs_of_an_item_are_reported(self, tool: ModuleType) -> None:
        """成功した回だけで安定と出しても、測れなかった回は別に見えなければならない"""
        runs = _runs([10.0] * 3)
        failed = _record({"gop1": 1.0})
        failed["items"][0]["status"] = "failed"
        runs += [{"run": 4, **failed}, {"run": 5, **failed}]
        base = _history("0.2.6", "2026-11-01", {"seek": _runs([10.0] * 5)})
        target = _history("0.2.7", "2026-11-24", {"seek": runs})
        assert "測れなかった回 2" in tool.report(tool.compare(base, target))
        assert "測れなかった回 2" in tool.describe(target)

    def test_a_skipped_bench_still_shows_its_failures(self, tool: ModuleType) -> None:
        """飛ばした回があるときに落ちた理由まで隠すと、素材を作れなかったのか見分けられない"""
        runs = _runs([10.0] * 3)
        failed = _record({"gop1": 1.0})
        failed["items"][0]["status"] = "failed"
        runs.append({"run": 4, **failed})
        runs.append({"run": 5, "status": "failed", "reason": "素材を作れない"})
        skipped = _record({}, status="skipped")
        skipped["reason"] = "ffmpeg が無い"
        runs.append({"run": 6, **skipped})
        shown = tool.describe(_history("0.2.7", "2026-11-24", {"seek": runs}))
        assert "飛ばした（ffmpeg が無い）" in shown
        assert "素材を作れない" in shown
        assert "測れなかった回 1" in shown

    def test_reasons_show_even_without_a_base(self, tool: ModuleType) -> None:
        """前の記録に無い測りで理由を黙ると、compare だけを見る人には GPU が無くて飛ばしたのか
        全部の回が落ちたのかが分からない
        """
        base = _history("0.2.6", "2026-11-01", {"seek": _runs([10.0] * 5)})
        skipped = _record({}, status="skipped")
        skipped["reason"] = "GPU が無い"
        target = _history(
            "0.2.7",
            "2026-11-24",
            {
                "seek": _runs([10.0] * 5),
                "render": [{"run": 1, **skipped}],
                "timeline": [{"run": 1, "status": "failed", "reason": "素材を作れない"}],
            },
        )
        shown = tool.report(tool.compare(base, target))
        assert "GPU が無い" in shown
        assert "素材を作れない" in shown
        target["machine"] = {**target["machine"], "cpu": "別の CPU"}
        assert "素材を作れない" in tool.report(tool.compare(base, target))

    def test_only_failures_exit_zero(
        self, tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """比べる相手があって落ちただけなら 0 揺れや失敗で毎回止まる確かめは見られなくなる"""
        base = _history("0.2.6", "2026-11-01", {"seek": _runs([10.0] * 5)})
        target = _history(
            "0.2.7", "2026-11-24", {"seek": [{"run": 1, "status": "failed", "reason": "落ちた"}]}
        )
        files = []
        for data in (base, target):
            path = tmp_path / f"{data['version']}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            files.append(str(path))
        assert tool.main(["compare", *files]) == 0
        assert "落ちた" in capsys.readouterr().out

    def test_a_label_with_unusable_characters_is_found(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """名前に使えない字はファイル名で _ になる

        付けた名前で引けないと、残した記録を比べられない
        """
        data = _history("0.2.6", "2026-11-01T00:00:00+09:00", {"seek": _runs([10.0] * 5)})
        data["label"] = "手元 測り"
        path = tmp_path / "____.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        _, found = tool.pick(tool.history_files(tmp_path), None, "手元 測り")
        assert found is not None and found["label"] == "手元 測り"

    def test_different_labels_never_share_a_file(self, tool: ModuleType) -> None:
        """日本語の名前を全部 _ にすると「変更前」と「変更後」が同じファイルになり、後が前を消す"""
        names = {tool._safe_name(label) for label in ("変更前", "変更後", "変更 前", "変更_前")}
        assert len(names) == 4
        assert tool._safe_name("0.2.7") == "0.2.7"
        assert tool._safe_name("変更 前") == tool._safe_name("変更 前")

    def test_times_are_compared_with_their_offsets(self, tool: ModuleType, tmp_path: Path) -> None:
        """夏時間の終わりのように時差が変わる前後では、文字の順と時刻の順が逆になる"""
        earlier = _history("前", "2026-11-01T01:50:00-04:00", {"seek": _runs([10.0] * 5)})
        later = _history("後", "2026-11-01T01:10:00-05:00", {"seek": _runs([10.0] * 5)})
        for data in (earlier, later):
            (tmp_path / f"{data['version']}.json").write_text(json.dumps(data), encoding="utf-8")
        base, target = tool.pick(tool.history_files(tmp_path), None, None)
        assert target is not None and target["version"] == "後"
        assert base is not None and base["version"] == "前"

    @pytest.mark.parametrize(
        "benches",
        [[], {"seek": []}, {"seek": {"runs": [None]}}, {"seek": {"runs": [{"status": "ok"}]}}],
    )
    def test_a_broken_history_is_skipped(
        self, tool: ModuleType, tmp_path: Path, benches: object
    ) -> None:
        """形だけ正しい壊れた記録が最新になると、既定の show と compare がそこで落ちる"""
        good = _history("0.2.6", "2026-11-01T00:00:00+09:00", {"seek": _runs([10.0] * 5)})
        broken = {**_history("0.2.7", "2026-11-24T00:00:00+09:00", {}), "benches": benches}
        for data in (good, broken):
            (tmp_path / f"{data['version']}.json").write_text(json.dumps(data), encoding="utf-8")
        found = tool.history_files(tmp_path)
        assert [data["version"] for _, data in found] == ["0.2.6"]
        assert "0.2.6" in tool.describe(found[-1][1])


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
        """合成の測りは予算を超えると 1 で終わる 落ちた扱いにすると、遅くなった版ほど比べられない"""

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
        """1 本を続けて回すと、重かった時間にその測りの全部の回が入って、その測りだけ遅く見える

        回すたびに順を巡らせ、版の名前で残し、同じ機械の前の記録と比べるかを見る
        """
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

    def test_the_machine_has_no_personal_names(
        self, results: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """記録は人へ渡すことがある 機械の名前や利用者の名前を読みに行かない

        文字の部分一致で見ると、利用者の名前が CPU の名前の一部（Intel など）の機械で誤って落ちる
        読む口そのものを、呼ばれたら落ちる物に替えて確かめる
        """
        import getpass
        import platform

        def refuse(*_args: object) -> str:
            raise AssertionError("機械の名前か利用者の名前を読んだ")

        monkeypatch.setattr(platform, "node", refuse)
        monkeypatch.setattr(getpass, "getuser", refuse)
        found = results.machine("GPU")
        assert set(found) == {"cpu", "cores", "os", "os_build", "gpu", "python"}
        assert found["gpu"] == "GPU"
        assert found["cpu"] and found["cores"] > 0

    def test_an_item_that_is_not_a_table_is_refused(
        self, results: ModuleType, tmp_path: Path
    ) -> None:
        """壊れた項目を通すと、比べる道具が数える所で落ち、残した記録ごと読めなくなる"""
        broken = {**_record({"gop1": 1.0}), "items": [None]}
        path = tmp_path / "壊れた.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="読めない項目"):
            results.read_record(path)
        empty = _record({"gop1": 1.0})
        empty["items"][0]["samples"] = []
        path.write_text(json.dumps(empty), encoding="utf-8")
        with pytest.raises(ValueError, match="値が無い"):
            results.read_record(path)

    @pytest.mark.parametrize("value", ["10" + "0" * 400, "NaN", "Infinity", "true"])
    def test_a_value_that_is_not_a_finite_number_is_refused(
        self, results: ModuleType, tmp_path: Path, value: str
    ) -> None:
        """桁あふれや NaN を通すと、数える所で落ちるか、中央と幅が NaN になって判定が黙って崩れる"""
        text = json.dumps(_record({"gop1": 1.0})).replace('"median": 1.0', f'"median": {value}')
        path = tmp_path / "数.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="数として読めない"):
            results.read_record(path)
        path.write_text(json.dumps(_record({"gop1": 1.5})), encoding="utf-8")
        assert results.read_record(path)["items"][0]["median"] == 1.5

    def test_a_decode_error_is_not_a_measured_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """デコーダは読めないと直前の絵を返す

        その時間を測れた値として残すと、壊れた素材が速く見える
        """
        seek = _load("bench_seek")

        class Broken:
            decode_error: str | None = None

            def __init__(self, _path: Path) -> None:
                self.calls = 0

            def __enter__(self) -> Broken:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def frame_at(self, _seconds: object) -> object:
                self.calls += 1
                if self.calls > 2:
                    self.decode_error = "試験の復号の失敗"
                return object()

        monkeypatch.setattr(seek, "VideoDecoder", Broken)
        times, error = seek._measure(tmp_path / "素材.mp4", 10, 5)
        assert times == []
        assert error == "試験の復号の失敗"
        record = _load("bench_results").BenchRecord("seek", {})
        seek._report(record, "gop1", "元 1 秒間隔", None, tmp_path / "素材.mp4", _args())
        assert record.status == "failed"
        assert record.items[0]["status"] == "failed"

    def test_the_timeline_bench_records_a_skip_without_ffmpeg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """実素材の測りも、ffmpeg の無い機械は飛ばしたと書く

        落ちたと書くと、測りの失敗と見分けられない
        """
        monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        timeline = _load("bench_timeline")
        monkeypatch.setattr(timeline.shutil, "which", lambda _name: None)
        path = tmp_path / "timeline.json"
        assert timeline.main(["--media", "--clips", "4", "--json", str(path)]) == 2
        data = _load("bench_results").read_record(path)
        assert data["status"] == "skipped"

    def test_the_seek_bench_records_a_skip_without_ffmpeg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ffmpeg の無い機械で落ちたと書くと、測りの失敗と見分けられず、毎回落ちたと報告する"""
        seek = _load("bench_seek")
        monkeypatch.setattr(seek.shutil, "which", lambda _name: None)
        path = tmp_path / "seek.json"
        assert seek.main(["--json", str(path)]) == 0
        data = _load("bench_results").read_record(path)
        assert data["status"] == "skipped"
        assert data["conditions"]["jumps"] == 80
