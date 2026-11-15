"""AI の演出の密度を読む道具（#470 段 1 #476） measure_direction

道具で付けた演出（寄る・揺らす・色）を同じ作品で数え直せること、作品を変えないこと、
テロップの本文を返さないこと、壊れた引数を黙って捨てないことを見る 数え方そのものは
tests/core/test_direction.py
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddClip, AddEffect, AddTrack, SetStyleGuide
from sashimono.core.io.serialize import json_text
from sashimono.core.model import (
    Clip,
    Effect,
    GeneratedSource,
    StyleGuide,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run


def _clip_id(host: FakeHost) -> str:
    return str(host.document.project.timeline.tracks[0].clips[0].id)


def _kinds(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["kind"]: row for row in result["kinds"]}


def test_is_read_only_and_sends_only_the_structure() -> None:
    """読むだけでないと「変更を自動で承認」を切った人に毎回確かめが出る 字幕と文字を宣言すると
    送らないと決めた人に使えない（本文は返さないので要らない）"""
    operation = find_operation("measure_direction")
    assert operation is not None
    assert operation.safety is Safety.READ_ONLY
    assert operation.sends == {Outbound.PROJECT}


def test_counts_what_emphasize_added_without_changing_the_work(host: FakeHost) -> None:
    """付けた演出を数え直せないと、AI は付けすぎたかを確かめずに仕上げる 測るだけで作品が
    変わると、取り消しの段が増えて人の取り消しがずれる"""
    clip_id = _clip_id(host)
    run(host, "emphasize", clip_id=clip_id, kind="zoom", start_frame=30, end_frame=90)
    run(host, "emphasize", clip_id=clip_id, kind="shake", start_frame=150, end_frame=170)
    before = host.document.project
    result = run(host, "measure_direction")
    assert host.document.project is before
    kinds = _kinds(result)
    assert kinds["zoom"]["count"] == 1
    assert kinds["zoom"]["strength"]["max"] == pytest.approx(120)
    assert kinds["shake"]["count"] == 1
    assert [(row["kind"], row["start_frame"]) for row in result["events"]] == [
        ("zoom", 30),
        ("shake", 150),
    ]
    assert (result["start_frame"], result["end_frame"], result["seconds"]) == (0, 300, 10.0)


def test_color_uses_the_effects_in_the_color_category(host: FakeHost) -> None:
    """分類で拾わないと、色を整えたクリップを 0 本と読み、AI が同じ補正を何度も積む"""
    kind = next(d.kind for d in registry.all() if d.category == "色")
    track = host.document.project.timeline.tracks[0]
    clip = track.clips[0]
    host.apply_commands([AddEffect(clip.id, Effect(kind))], "色")
    color = _kinds(run(host, "measure_direction"))["color"]
    assert (color["count"], color["coverage"]) == (1, 1.0)


def test_telop_text_is_not_sent(host: FakeHost) -> None:
    """本文を返すと、字幕と文字を送らないと決めた人の文字が同意なしで外へ出る"""
    track = Track(kind=TrackKind.VIDEO, name="文字")
    source = GeneratedSource("text", params={"text": "本文の秘密ZZSECRET"})
    telop = Clip(timeline_start=60, duration=30, source=source)
    host.apply_commands([AddTrack(track), AddClip(track.id, telop)], "置く")
    result = run(host, "measure_direction")
    assert _kinds(result)["telop"]["count"] == 1
    assert "ZZSECRET" not in json_text(result)


def test_close_calls_and_the_override(host: FakeHost) -> None:
    """下限を上書きできないと、ショート動画のように詰めて付ける作品で全部が「近すぎ」と出る"""
    clip_id = _clip_id(host)
    for start in (0, 60):
        run(
            host, "emphasize", clip_id=clip_id, kind="zoom", start_frame=start, end_frame=start + 20
        )
    zoom = _kinds(run(host, "measure_direction"))["zoom"]
    assert (zoom["too_close"], zoom["limit_seconds"]) == (1, 8.0)
    assert zoom["too_close_at"] == [{"first_frame": 0, "second_frame": 60, "gap": 2.0}]
    loose = _kinds(run(host, "measure_direction", min_gaps={"zoom": 1.5}))["zoom"]
    assert loose["too_close"] == 0


def test_guide_problems_are_counted_alongside(host: FakeHost) -> None:
    """方針の外を添えないと、密度は良いのに方針の外の効果を使う作品を仕上がったと読む"""
    track = host.document.project.timeline.tracks[0]
    clip = track.clips[0]
    host.apply_commands(
        [SetStyleGuide(StyleGuide(effects=("blur",))), AddEffect(clip.id, Effect("glow"))],
        "方針",
    )
    result = run(host, "measure_direction")
    assert result["outside_guide"]["effect"] == 1


def test_bad_arguments_are_refused(host: FakeHost) -> None:
    """黙って捨てると、頼んだ下限やシーンと違う物を測った数を正しいと思って使う"""
    with pytest.raises(ToolError, match="知らない型"):
        run(host, "measure_direction", min_gaps={"bounce": 2})
    with pytest.raises(ToolError, match="0 以上"):
        run(host, "measure_direction", min_gaps={"zoom": -1})
    with pytest.raises(ToolError, match="シーンが見つかりません"):
        run(host, "measure_direction", scene_id="none")
    with pytest.raises(ToolError, match="範囲が空"):
        run(host, "measure_direction", start_frame=300)
