"""AI が編集の方針（#343）を読む道具と、方針に照らす道具

方針は本人が AI に渡す決まり AI は読んで従い、方針の外の書体・色・効果を照らして見つける
方針を変える道具は置かない（AI が自分で緩められないように）
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace

import pytest

from sashimono.ai import host as host_module
from sashimono.ai import operations, style_check
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.ledger import AI_ONLY
from sashimono.ai.operations import MAX_RESULT_CHARS, find_operation
from sashimono.ai.outbound import Outbound, SendChoice
from sashimono.ai.safety import Safety
from sashimono.ai.session import system_prompt
from sashimono.core.commands import AddClip, AddScene, AddTrack, SetStyleGuide, new_scene
from sashimono.core.io.serialize import json_text
from sashimono.core.model import (
    Clip,
    Effect,
    GeneratedSource,
    ParamValue,
    Stroke,
    StyleGuide,
    Track,
    TrackKind,
)
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run
from tests.ai.test_outbound import _run

NOTE = "方針の秘密の決まりをここに書く"

GUIDE = StyleGuide(
    fonts=("源ノ角ゴシック",),
    colors=("#ffffff", "#000000"),
    subtitles="下の中央に白の字",
    tempo="間は 0.3 秒まで",
    effects=("blur",),
    notes=NOTE,
)

WHITE = (1.0, 1.0, 1.0, 1.0)
RED = (1.0, 0.0, 0.0, 1.0)


def _text(
    font: str = "源ノ角ゴシック", color: tuple[float, ...] = WHITE, **more: ParamValue
) -> Clip:
    params: dict[str, ParamValue] = {"text": "本文の秘密", "font": font, "color": color, **more}
    return Clip(timeline_start=0, duration=30, source=GeneratedSource("text", params=params))


def _guided(host: FakeHost, guide: StyleGuide = GUIDE) -> None:
    host.apply_commands([SetStyleGuide(guide)], "方針")


def _place(host: FakeHost, *clips: Clip) -> Track:
    track = Track(kind=TrackKind.VIDEO, name="文字")
    host.apply_commands([AddTrack(track), *(AddClip(track.id, clip) for clip in clips)], "置く")
    return track


def _problems(host: FakeHost) -> list[dict[str, object]]:
    result = run(host, "check_style_guide")
    rows = result["problems"]
    assert isinstance(rows, list)
    return rows


class TestReading:
    def test_the_guide_is_read_as_written(self, host: FakeHost) -> None:
        """壊れると、本人が書いた方針を AI が読めず、決まりを知らずに見た目を選ぶ"""
        _guided(host)
        result = run(host, "get_style_guide")
        assert result["set"] is True
        assert result["fonts"] == ["源ノ角ゴシック"]
        assert result["colors"] == ["#ffffff", "#000000"]
        assert result["tempo"] == "間は 0.3 秒まで"
        assert result["notes"] == NOTE
        # 効果は種類と名前で返す 種類だけだと add_effect に渡す値は分かっても何の効果か分からない
        assert result["effects"][0]["kind"] == "blur"
        assert result["effects"][0]["label"] != "blur"

    def test_no_guide_says_so(self, host: FakeHost) -> None:
        """壊れると、方針の無い作品で AI が空の欄を「何も使ってはいけない」と読み違える"""
        result = run(host, "get_style_guide")
        assert result["set"] is False
        assert "決めていない" in result["note"]

    def test_the_project_tells_whether_there_is_a_guide(self, host: FakeHost) -> None:
        """壊れると、AI は方針があることを知らないまま編集を始める

        AI への説明は get_project の has_style_guide を見よと言う
        """
        assert run(host, "get_project")["has_style_guide"] is False
        _guided(host)
        assert run(host, "get_project")["has_style_guide"] is True

    def test_the_session_tells_the_ai_to_follow_it(self) -> None:
        """壊れると、道具はあっても AI が呼ばず、方針が無いのと同じになる"""
        prompt = system_prompt()
        assert "get_style_guide" in prompt
        assert "check_style_guide" in prompt
        assert "has_style_guide" in prompt


class TestChecking:
    def test_fonts_colours_and_effects_outside_the_guide_are_found(self, host: FakeHost) -> None:
        """壊れると、方針の外の書体・色・効果が残ったまま、AI が仕上げたと答える"""
        _guided(host)
        blurred = replace(_text(), effects=(Effect("blur"), Effect("glow")))
        _place(host, _text(font="MS ゴシック"), replace(_text(color=RED), timeline_start=40))
        _place(host, blurred)
        found = {(row["kind"], row["value"]) for row in _problems(host)}
        assert found == {("font", "MS ゴシック"), ("color", "#ff0000"), ("effect", "glow")}

    def test_the_default_font_is_named(self, host: FakeHost) -> None:
        """書体を選んでいないテキストを見逃すと、既定の書体の字幕が方針どおりと数えられる"""
        _guided(host)
        _place(host, _text(font=""))
        assert [row["value"] for row in _problems(host)] == [style_check.DEFAULT_FONT]

    def test_stroke_and_gradient_colours_are_checked(self, host: FakeHost) -> None:
        """縁取りの層とグラデーションの止まりを見ないと、赤い縁や赤へ流れる字を見逃す"""
        _guided(host)
        gradient = (0.0, *WHITE, 1.0, *RED)
        outlined = GeneratedSource(
            "text",
            params={"font": "源ノ角ゴシック", "color": WHITE},
            strokes=(Stroke(params={"color": RED}),),
        )
        flowing = replace(_text(fill_type="linear", fill_gradient=gradient), timeline_start=40)
        _place(host, Clip(timeline_start=0, duration=30, source=outlined), flowing)
        wheres = sorted(str(row["where"]) for row in _problems(host))
        assert wheres == ["字の塗り", "縁取りの層 1"]

    def test_a_nearly_equal_colour_counts_as_the_same(self, host: FakeHost) -> None:
        """色の窓と 0..1 の値の行き来で 1 ずれた白まで「方針の外」と返すと、直しようが無い"""
        _guided(host)
        _place(host, _text(color=(254 / 255, 1.0, 1.0, 1.0)))
        assert _problems(host) == []

    def test_fixed_and_switched_off_effects_are_not_counted(self, host: FakeHost) -> None:
        """全部のクリップが持つ描画の欄や、切ってある効果まで返すと、どのクリップも方針の外になる"""
        _guided(host)
        clip = replace(
            _text(),
            effects=(Effect("transform", fixed=True), Effect("glow", enabled=False)),
        )
        _place(host, clip)
        assert _problems(host) == []

    def test_scenes_are_checked_too(self, host: FakeHost) -> None:
        """メインだけを見ると、シーンの中のオープニングの書体を見逃す"""
        _guided(host)
        scene = new_scene(host.document.project, "オープニング")
        track = Track(kind=TrackKind.VIDEO, clips=(_text(font="MS ゴシック"),))
        scene = replace(scene, timeline=replace(scene.timeline, tracks=(track,)))
        host.apply_commands([AddScene(scene)], "シーン")
        rows = _problems(host)
        assert [row["scene_id"] for row in rows] == [str(scene.id)]

    def test_empty_fields_are_not_checked(self, host: FakeHost) -> None:
        """決めていない欄まで照らすと、色を決めていない人の赤い字が全部方針の外になる"""
        _guided(host, StyleGuide(fonts=("源ノ角ゴシック",)))
        glowing = replace(_text(), effects=(Effect("glow"),), timeline_start=40)
        _place(host, _text(color=RED), glowing)
        result = run(host, "check_style_guide")
        assert result["problems"] == []
        assert result["checked"] == {"font": True, "color": False, "effect": False}

    def test_the_result_holds_no_sentences(self, host: FakeHost) -> None:
        """方針の文やテキストの本文を返すと、字幕と文字を送らない人の文が尋ねずに外へ出る"""
        _guided(host)
        _place(host, _text(font="MS ゴシック"))
        text = json.dumps(run(host, "check_style_guide"), ensure_ascii=False)
        for sentence in (NOTE, GUIDE.subtitles, GUIDE.tempo, "本文の秘密"):
            assert sentence not in text

    def test_a_long_list_stays_within_the_limit(self, host: FakeHost) -> None:
        """直す先の一覧の分を見ずに切ると、返事が上限を越えて打ち切られ、続きを読めない"""
        fonts = tuple(f"とても長い名前の書体その{n:03d}" + "あ" * 40 for n in range(100))
        _guided(host, StyleGuide(fonts=fonts))
        _place(
            host, *(replace(_text(font="MS ゴシック"), timeline_start=n * 30) for n in range(300))
        )
        result = run(host, "check_style_guide")
        assert len(json_text(result)) <= MAX_RESULT_CHARS
        assert "next_offset" in result
        assert "offset=" in result["note"]


class TestConsent:
    @pytest.fixture
    def text_refused(self, host: FakeHost) -> EditorBridge:
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(
            {
                Outbound.FRAMES: SendChoice.ASK,
                Outbound.TEXT: SendChoice.DENY,
                Outbound.FILES: SendChoice.ALLOW,
            }
        )
        return bridge

    def test_reading_is_refused_without_text(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        """壊れると、字幕と文字を送らないと決めた人の方針の文が、尋ねずに外へ出る"""
        _guided(host)
        _seen, box = _run(text_refused, "get_style_guide")
        assert isinstance(box.get("error"), ToolError)
        assert NOTE not in json.dumps(box.get("result"), ensure_ascii=False, default=str)

    def test_checking_still_works_without_text(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        """照らす道具まで断ると、文を送らない人は方針の外の書体を AI に直してもらえない"""
        _guided(host)
        _place(host, _text(font="MS ゴシック"))
        _seen, box = _run(text_refused, "check_style_guide")
        assert box.get("error") is None, box.get("error")


class TestReadOnly:
    def test_both_tools_only_read(self) -> None:
        """書く道具にすると、方針を読むだけで確かめの窓が出る"""
        for name in ("get_style_guide", "check_style_guide"):
            operation = find_operation(name)
            assert operation is not None
            assert operation.safety is Safety.READ_ONLY
            assert name in AI_ONLY

    def test_no_tool_changes_the_guide(self) -> None:
        """方針を変える口を AI に渡すと、AI が自分で決まりを緩められる（本人が AI に渡す決まり）"""
        for module in (operations, host_module):
            source = inspect.getsource(module)
            assert "SetStyleGuide" not in source
            assert "style_guide=" not in source
