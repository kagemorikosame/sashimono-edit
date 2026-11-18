"""評価の台本の課題「演出」（#470 #476）

見どころの手がかりから強さごとに演出の計画を付け、付いた演出の密度・近すぎる所・
同じ手がかりで同じ計画になるかを数える 実物の素材は CI に無いので、手がかりは作った見どころで渡し、
台本を 1 周させる所だけ合成の動画を使う
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sashimono.ai import direction_plan
from sashimono.core import direction_recipes
from sashimono.core.direction_recipes import DirectionTaste, RecipeBook
from sashimono.core.model import MediaItem, Transcript
from sashimono.engine.highlights import Highlight
from tests.check_media import write_video
from tests.test_eval_tasks_tool import tool

__all__ = ["tool"]

#: 4 分の作品 派手でも寄るが 1 分に 3 回までなので、強さの差が回数に出る長さ
_SECONDS = 240


def _project(tool: ModuleType, video_media: MediaItem) -> Any:
    media = replace(video_media, duration=Fraction(_SECONDS))
    return tool._project_for(media, "direction").project


def _highlights() -> list[Highlight]:
    """笑いが 7 秒おき・叫びが 11 秒おき どちらも強さを散らす 弱い動きは手がかりにしない"""
    found = []
    for t in range(3, _SECONDS - 2, 7):
        found.append(
            Highlight(
                start=float(t),
                end=float(t + 1),
                peak=t + 0.5,
                score=0.9,
                laughter=0.5 + (t * 37 % 50) / 100,
                loud=None,
                motion=0.2,
            )
        )
    for t in range(5, _SECONDS - 2, 11):
        found.append(
            Highlight(
                start=float(t),
                end=float(t + 0.5),
                peak=t + 0.25,
                score=0.8,
                laughter=None,
                loud=0.5 + (t * 13 % 50) / 100,
                motion=None,
            )
        )
    return found


def test_cues_take_only_the_strong_signals(tool: ModuleType) -> None:
    # 弱い手がかりまで渡すと、笑っていない所へ笑いの寄りを置く
    cues = tool.direction_cues(_highlights())
    assert {cue.signal for cue in cues} == {"laughter", "loud"}
    assert all(cue.score >= 0.5 for cue in cues)


def test_stronger_settings_place_more_and_never_too_close(
    tool: ModuleType, video_media: MediaItem
) -> None:
    # 強さを上げても増えない・近すぎる所が出るなら、強さの設定もやりすぎの線も効いていない
    metrics = tool.direction_metrics(
        _project(tool, video_media), tool.direction_cues(_highlights())
    )
    assert metrics["too_close"] == 0
    assert metrics["off_guide"] == 0
    assert metrics["consistent"] == 1
    assert metrics["ordered"] == 1
    zooms = [metrics[f"zoom_per_minute_{s}"] for s in ("subtle", "normal", "bold")]
    assert zooms[0] < zooms[1] < zooms[2]
    # 控えめは 1 分に 1 回までの決まり（段 3） 数え方の誤りで上限を越えて読まない
    assert zooms[0] <= 1.0
    assert metrics["shake_per_minute_bold"] > metrics["shake_per_minute_subtle"]
    assert all(metrics[f"skipped_{s}"] == 0 for s in ("subtle", "normal", "bold"))


def test_off_guide_catches_a_plan_that_ignores_the_guide(
    tool: ModuleType, video_media: MediaItem, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 方針を読まない計画でも 0 と出るなら、方針に合うかを数えていない
    original = direction_recipes.recipe_book

    def ignore_guide(taste: DirectionTaste, allowed: Iterable[str] = ()) -> RecipeBook:
        return original(taste)

    monkeypatch.setattr(direction_recipes, "recipe_book", ignore_guide)
    metrics = tool.direction_metrics(
        _project(tool, video_media), tool.direction_cues(_highlights())
    )
    assert metrics["off_guide"] > 0


def test_items_left_out_while_building_count_as_skipped(
    tool: ModuleType, video_media: MediaItem, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 組み立てで外した項目（同じ型の動きと重なる など）を数えないと、付かなくても 0 と出る
    original = direction_plan.build_items

    def drop_all(*args: Any) -> tuple[list[Any], list[tuple[str, str]]]:
        items, dropped = original(*args)
        return [], dropped + [(item.at, "外した") for item in items]

    monkeypatch.setattr(direction_plan, "build_items", drop_all)
    metrics = tool.direction_metrics(
        _project(tool, video_media), tool.direction_cues(_highlights())
    )
    assert metrics["skipped_bold"] > 0
    assert metrics["zoom_per_minute_bold"] == 0


def test_no_cues_place_nothing(tool: ModuleType, video_media: MediaItem) -> None:
    # 盛り上がりの無い素材で何かが付くと、手がかりに無い所へ演出を置いている
    metrics = tool.direction_metrics(_project(tool, video_media), [])
    assert metrics["cues"] == 0
    assert metrics["too_close"] == 0
    assert metrics["zoom_per_minute_bold"] == 0
    assert metrics["consistent"] == 1


@pytest.mark.parametrize("words", [None, Transcript(segments=())])
def test_the_direction_task_runs_without_an_answer(
    tool: ModuleType, tmp_path: Path, words: Transcript | None
) -> None:
    """正解の無いまま台本を 1 周させ、結果の行に数と値を書く

    正解を求めると「no-answer」で飛ばされ、演出の課題は 1 度も数えられない
    字幕起こしを使ったら、手がかりが変わるのでモデルの指定も書く
    """
    media = tmp_path / "ひみつの素材置き場"
    media.mkdir()
    video = write_video(media / "じっきょう.mp4", 6.0, tone_db=-12.0, silent=(2.0, 3.0))
    root = tmp_path / "evaldir"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"E07": str(video)}, ensure_ascii=False), encoding="utf-8"
    )
    book_path = tmp_path / "tasks.json"
    book_path.write_text(
        json.dumps({"materials": [{"id": "E07", "tasks": ["direction"]}]}), encoding="utf-8"
    )
    results = tmp_path / "r.jsonl"
    [record] = tool.command_run(
        tool.EvalDir(root),
        tool.load_tasks(book_path),
        ["E07"],
        results=results,
        commit="",
        transcriber=lambda path, params: words,
    )
    assert record["status"] == "ok", record
    assert "provisional" not in record
    used = words is not None
    expected = {"genre": "lets_play"}
    if used:
        expected.update(tool.ToolParams().record(tool.SUBTITLES))
    assert record["params"] == expected
    metrics = record["metrics"]
    assert metrics["too_close"] == 0
    assert metrics["consistent"] == 1
    assert metrics["subtitles"] == int(used)
    # 素材の名前や置き場は結果の行に入らない
    written = results.read_text(encoding="utf-8")
    assert "ひみつ" not in written and "じっきょう" not in written
    assert tool.HEADLINE[tool.DIRECTION] == "too_close"


@pytest.mark.parametrize("task", ["direction"])
def test_the_direction_task_is_not_scored_against_an_answer(tool: ModuleType, task: str) -> None:
    # 人の手の演出と比べると、好みの違いが誤りに数えられる
    with pytest.raises(ValueError, match="正解と比べない"):
        tool.score(task, {}, {}, tool.load_tasks())
