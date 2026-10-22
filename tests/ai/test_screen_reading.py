"""人が画面で見ている物を AI も読める（読み取りの同等性 #352）

設定パネルとグラフエディタの値（get_clip）・表示の様子（get_screen）・人の選択（get_selection）・
書き出し範囲（get_project）・互換性レポート（get_compat_report）を、偽のホストで確かめる
本物の窓で同じ口を辿る試験は ``tests/ui/test_screen_reading.py``
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import CompatReport, ToolError
from sashimono.ai.operations import MAX_RESULT_CHARS, find_operation
from sashimono.ai.outbound import (
    ASKED,
    FIELD_SENDS,
    PATH_MARK,
    TEXT_MARK,
    UNKNOWN_FIELD_SENDS,
    Outbound,
    SendChoice,
)
from sashimono.compat.aviutl import objapi
from sashimono.compat.aviutl.report import (
    AVIUTL_OBJ_NAMES,
    KNOWN_RECORDS,
    CompatibilityReport,
    is_known_record,
)
from sashimono.core.commands import AddEffect, ParamPath, SetKeyframe, SetParam, SetWorkArea
from sashimono.core.io.serialize import json_text
from sashimono.core.model import ClipId, Effect, MediaItem, Project, StrokeId, Transcript
from sashimono.core.timebase import format_timecode
from sashimono.effects import registry
from sashimono.effects.definition import EffectDefinition
from sashimono.effects.sources import source_registry
from sashimono.effects.spec import FileSpec, TextSpec
from sashimono.effects.strokes import STROKE
from tests.ai.conftest import FakeHost
from tests.ai.test_bridge import call_in_thread


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, tool
    return operation(host, arguments)


def _video(project: Project) -> ClipId:
    return project.timeline.tracks[0].clips[0].id


def _text_clip(host: FakeHost) -> ClipId:
    return next(
        clip.id
        for track in host.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == "text"
    )


class TestClipValues:
    def test_untouched_fields_are_read_at_their_defaults(self, host: FakeHost) -> None:
        """壊れると、触っていない欄が無い物に見え、AI が今のぼかしの強さを知らずに
        「少し弱く」を 0 にしたり、既定の 8 を 2 倍の 16 と取り違えたりする"""
        clip = _video(host.document.project)
        host.apply_commands([AddEffect(clip, Effect(kind="blur"))], "ぼかし")
        result = run(host, "get_clip", clip_id=str(clip))
        blur = next(row for row in result["effects"] if row["kind"] == "blur")
        assert blur["params"] == {"radius": 8.0, "fixed_size": False}
        assert blur["label"] == "ぼかし"
        assert result["opacity"] == 1.0

    def test_keyframes_come_with_their_frames_and_motion(self, host: FakeHost) -> None:
        """壊れると、グラフエディタで人が見ている点と動き方を AI が読めず、「最後をゆっくり
        止めて」と頼まれても、どの点を直せばよいか分からない"""
        clip = str(_video(host.document.project))
        run(host, "add_keyframe", clip_id=clip, target="clip", name="opacity", frame=0, value=0)
        run(
            host,
            "add_keyframe",
            clip_id=clip,
            target="clip",
            name="opacity",
            frame=60,
            value=1,
            interpolation="ease_out",
        )
        opacity = run(host, "get_clip", clip_id=clip)["opacity"]
        assert opacity["keyframes"] == [
            {"frame": 0, "value": 0.0, "interpolation": "linear"},
            {"frame": 60, "value": 1.0, "interpolation": "ease_out"},
        ]
        assert "samples" not in opacity

    def test_samples_show_the_shape_between_the_points(self, host: FakeHost) -> None:
        """壊れると、点の間の形（加速・減速）が AI に見えず、加速を付けたつもりの動きが
        直線のままでも気付けない"""
        clip = str(_video(host.document.project))
        for frame, value in ((0, 0.0), (299, 1.0)):
            run(
                host,
                "add_keyframe",
                clip_id=clip,
                target="clip",
                name="opacity",
                frame=frame,
                value=value,
                interpolation="ease_in",
            )
        samples = run(host, "get_clip", clip_id=clip, samples=3)["opacity"]["samples"]
        # 頭・真ん中・終わり 加速なので真ん中はまだ半分に届かない
        assert [frame for frame, _ in samples] == [0, 150, 299]
        assert samples[0][1] == 0.0 and samples[2][1] == 1.0
        assert samples[1][1] < 0.4

    @pytest.mark.parametrize("samples", [1, 61, "3", True])
    def test_a_wrong_number_of_samples_is_refused(self, host: FakeHost, samples: object) -> None:
        """壊れると、1 点では形にならない頼みや、返事が上限を越える頼みが黙って通り、
        曲線を読んだつもりの AI が形の無い点を読む"""
        with pytest.raises(ToolError, match="samples"):
            run(host, "get_clip", clip_id=str(_video(host.document.project)), samples=samples)

    def test_the_selected_clip_is_read_when_none_is_named(self, host: FakeHost) -> None:
        """壊れると、「この字の色は？」と聞かれた AI が、設定パネルに出ている 1 本を読めない"""
        clip = _video(host.document.project)
        host.select_clip(clip)
        assert run(host, "get_clip")["clip_id"] == str(clip)
        host.select_clip(None)
        with pytest.raises(ToolError, match="clip_id"):
            run(host, "get_clip")

    def test_text_and_strokes_are_read(self, host: FakeHost) -> None:
        """壊れると、字の大きさ・位置・縁取りの太さを AI が知らずに直し、人が整えた見た目を崩す"""
        run(host, "add_text", text="見出し", size=80)
        clip = str(_text_clip(host))
        stroke = run(host, "add_stroke", clip_id=clip)["stroke_id"]
        result = run(host, "get_clip", clip_id=clip)
        assert result["kind"] == "text"
        assert result["params"]["text"] == "見出し"
        assert result["params"]["size"] == 80.0
        assert result["strokes"][0]["stroke_id"] == stroke
        layer = run(host, "get_clip", clip_id=clip, stroke_id=stroke)
        assert "width" in layer["params"]

    def test_many_strokes_are_read_layer_by_layer(self, host: FakeHost) -> None:
        """壊れると、縁取りの層を何十も重ねた字で返事が上限を越えて切られ、切られた層の値と
        層のエフェクトを後から取る道が無いまま、AI が見えた分だけで直す（#391 の Codex の指摘）"""
        run(host, "add_text", text="見出し")
        clip = _text_clip(host)
        made = [run(host, "add_stroke", clip_id=str(clip))["stroke_id"] for _ in range(8)]
        host.apply_commands(
            [
                AddEffect(clip, Effect(kind="blur"), stroke_id=StrokeId(stroke))
                for stroke in made
                for _ in range(40)
            ],
            "層ごとにぼかしを積む",
        )
        overview = run(host, "get_clip", clip_id=str(clip), samples=60)
        assert len(json_text(overview)) <= MAX_RESULT_CHARS
        assert overview["stroke_count"] == 8
        assert [row["stroke_id"] for row in overview["strokes"]] == made
        assert [row["effect_count"] for row in overview["strokes"]] == [40] * 8
        effects: list[str] = []
        offset: int | None = 0
        while offset is not None:
            layer = run(host, "get_clip", clip_id=str(clip), stroke_id=made[-1], offset=offset)
            assert len(json_text(layer)) <= MAX_RESULT_CHARS
            assert "width" in layer["params"]
            effects += [row["effect_id"] for row in layer["effects"]]
            offset = layer.get("next_offset")
        assert len(set(effects)) == 40

    def test_the_effects_of_the_scene_after_a_transition_are_marked(self, host: FakeHost) -> None:
        """壊れると、場面切り替えの前と後の場面に積んだ同じ種類のエフェクトを AI が見分けられない"""
        run(host, "add_transition")
        project = host.document.project
        clip = next(
            c
            for track in project.timeline.tracks
            for c in track.clips
            if c.source is not None and c.source.kind == "transition"
        )
        after = Effect(kind="blur")
        host.apply_commands([AddEffect(clip.id, after, after=True)], "後の場面へ")
        result = run(host, "get_clip", clip_id=str(clip.id))
        rows = [row for row in result["effects"] if row["effect_id"] == str(after.id)]
        assert rows and rows[0]["after"] is True

    def test_many_effects_are_read_in_pages(self, host: FakeHost) -> None:
        """壊れると、エフェクトを何十も積んだクリップで返事が上限を越えて打ち切られ、
        後ろのエフェクトの値をどうしても読めない"""
        clip = _video(host.document.project)
        host.apply_commands(
            [AddEffect(clip, Effect(kind="blur")) for _ in range(300)], "ぼかしを積む"
        )
        first = run(host, "get_clip", clip_id=str(clip), samples=60)
        assert len(json_text(first)) <= MAX_RESULT_CHARS
        assert first["count"] < first["total"]
        second = run(host, "get_clip", clip_id=str(clip), offset=first["next_offset"])
        assert second["offset"] == first["next_offset"]


def _many_keys(host: FakeHost, count: int) -> ClipId:
    """不透明度に ``count`` 個のキーフレームを打った頭のクリップ（読み込んだ細かな動きの代わり）"""
    clip = _video(host.document.project)
    path = ParamPath.of_clip(clip, "opacity")
    host.apply_commands(
        [SetKeyframe(path, frame, (frame % 10) / 10) for frame in range(count)], "細かな動き"
    )
    return clip


class TestManyKeyframes:
    def test_the_overview_says_which_values_were_cut(self, host: FakeHost) -> None:
        """壊れると、キーフレームの多いクリップで点が黙って切られ、AI が欠けた曲線を全部だと
        思って直し、読み込んだ細かな動きを崩す"""
        clip = _many_keys(host, 300)
        result = run(host, "get_clip", clip_id=str(clip))
        assert len(json_text(result)) <= MAX_RESULT_CHARS
        opacity = result["opacity"]
        assert opacity["keyframe_count"] == 300
        assert len(opacity["keyframes"]) < 300
        assert opacity["more_keyframes"] == {"target": "clip", "name": "opacity"}

    def test_every_keyframe_of_one_value_can_be_read_in_pages(self, host: FakeHost) -> None:
        """壊れると、切られた点を後から取る道が無く、読み込んだ細かな動きの後ろ半分を
        AI が読めないまま、点を打ち直して形を壊す"""
        clip = _many_keys(host, 1500)
        frames: list[int] = []
        offset: int | None = 0
        while offset is not None:
            page = run(
                host, "get_clip", clip_id=str(clip), target="clip", name="opacity", offset=offset
            )
            assert len(json_text(page)) <= MAX_RESULT_CHARS
            frames += [row["frame"] for row in page["keyframes"]]
            offset = page.get("next_offset")
        assert frames == list(range(1500))

    def test_one_value_reads_its_default_and_refuses_an_unknown_name(self, host: FakeHost) -> None:
        """壊れると、触っていない欄を名前で読むと何も返らず、無い名前の打ち間違いにも気付けない"""
        clip = _video(host.document.project)
        added = run(host, "add_effect", clip_id=str(clip), kind="blur")
        one = run(host, "get_clip", clip_id=str(clip), effect_id=added["effect_id"], name="radius")
        assert one["value"] == 8.0
        with pytest.raises(ToolError, match="無い"):
            run(host, "get_clip", clip_id=str(clip), effect_id=added["effect_id"], name="radiu")


class TestAfterTheTransition:
    def test_what_was_read_can_be_passed_back_to_change_it(self, host: FakeHost) -> None:
        """壊れると、場面切り替えの後の場面のエフェクトを読んだ引数（after 付き）のまま
        set_param・add_keyframe へ渡しても前の場面を探して断られ、後の場面の値を変えられない
        （#391 の Codex の指摘）"""
        run(host, "add_transition")
        clip = next(
            c
            for track in host.document.project.timeline.tracks
            for c in track.clips
            if c.source is not None and c.source.kind == "transition"
        )
        after = Effect(kind="blur")
        host.apply_commands([AddEffect(clip.id, after, after=True)], "後の場面へ")
        read = run(host, "get_clip", clip_id=str(clip.id))
        row = next(r for r in read["effects"] if r["effect_id"] == str(after.id))
        where = {"clip_id": str(clip.id), "effect_id": row["effect_id"], "after": row["after"]}
        run(host, "set_param", **where, name="radius", value=20)
        run(host, "add_keyframe", **where, name="radius", frame=10, value=30)
        radius = run(host, "get_clip", **where, name="radius")
        assert [k["frame"] for k in radius["keyframes"]] == [10]
        moved = host.document.project.timeline.locate_clip(clip.id)
        assert moved is not None
        assert moved[1].after_effects[0].params["radius"] != 8.0

    def test_after_is_refused_without_an_effect(self, host: FakeHost) -> None:
        """壊れると、after を中身の欄に付けた頼みが黙って前の場面の欄を変える"""
        run(host, "add_text", text="見出し")
        with pytest.raises(ToolError, match="after"):
            run(
                host,
                "set_param",
                clip_id=str(_text_clip(host)),
                name="size",
                value=10,
                after=True,
            )


class TestLongValues:
    def test_a_long_text_value_is_read_in_pieces(self, host: FakeHost) -> None:
        """壊れると、長い本文や線の点を名前で読み直しても返事が上限を越えて打ち切られ、
        後ろ半分を読む道が無い（#391 の Codex の指摘）"""
        body = "".join(f"{n:05d}行目の本文\n" for n in range(3000))
        run(host, "add_text", text="見出し")
        clip = _text_clip(host)
        host.apply_commands([SetParam(ParamPath.of_source(clip, "text"), body)], "長い本文")
        pieces: list[str] = []
        offset: int | None = 0
        while offset is not None:
            page = run(host, "get_clip", clip_id=str(clip), name="text", value_offset=offset)
            assert len(json_text(page)) <= MAX_RESULT_CHARS
            assert page["value_length"] == len(body)
            pieces.append(page["value"])
            offset = page.get("next_value_offset")
        assert "".join(pieces) == body

    def test_a_short_value_comes_whole(self, host: FakeHost) -> None:
        """壊れると、短い値まで区切りの印が付き、AI が続きを探しに行く"""
        run(host, "add_text", text="見出し")
        one = run(host, "get_clip", clip_id=str(_text_clip(host)), name="text")
        assert one["value"] == "見出し"
        assert "next_value_offset" not in one


class TestKeyframePosition:
    def test_an_omitted_frame_is_counted_from_the_clip_head(
        self, loaded: Project, video_media: MediaItem
    ) -> None:
        """壊れると、頭が 0 でないクリップで「ここにキーを」と頼むと、再生ヘッドの位置を
        クリップの中の位置と取り違え、クリップの外（見えない所）へキーが入る"""
        del video_media
        host = FakeHost(loaded)
        clip = _video(loaded)
        run(host, "move_clip", clip_id=str(clip), timeline_start=100)
        host.seek(130)
        run(host, "add_keyframe", clip_id=str(clip), target="clip", name="opacity", value=0.5)
        frames = run(host, "get_clip", clip_id=str(clip))["opacity"]["keyframes"]
        assert [row["frame"] for row in frames] == [30]

    def test_a_curve_without_control_points_is_refused_plainly(self, host: FakeHost) -> None:
        """壊れると、bezier を頼んだ AI に点を作る所の中の失敗がそのまま返り、何を直せば
        よいか分からないまま同じ頼みを繰り返す"""
        with pytest.raises(ToolError, match="bezier"):
            run(
                host,
                "add_keyframe",
                clip_id=str(_video(host.document.project)),
                target="clip",
                name="opacity",
                frame=0,
                value=1,
                interpolation="bezier",
            )


class TestScreen:
    def test_the_visible_range_and_zoom_are_read(self, host: FakeHost) -> None:
        """壊れると、「見えている所の無音を詰めて」と頼まれた AI が、人の見ている範囲を
        知らずに作品全体を詰める"""
        host.screen = replace(host.screen, visible_frames=(90, 180), pixels_per_frame=4.0)
        timeline = run(host, "get_screen")["timeline"]
        assert (timeline["from_frame"], timeline["to_frame"]) == (90, 180)
        assert timeline["from_timecode"] == format_timecode(90, host.project.rate)
        assert timeline["pixels_per_second"] == 120.0

    def test_the_graph_value_can_be_passed_to_get_clip(self, host: FakeHost) -> None:
        """壊れると、「今グラフに出ている動き」と言われた AI が、どの値の曲線かを読めない"""
        clip = _video(host.document.project)
        host.screen = replace(
            host.screen, graph_shown=True, graph_path=ParamPath.of_clip(clip, "opacity")
        )
        graph = run(host, "get_screen")["graph_editor"]
        assert graph == {
            "shown": True,
            "value": {"clip_id": str(clip), "name": "opacity", "target": "clip"},
        }
        read = run(host, "get_clip", clip_id=graph["value"]["clip_id"])
        assert read["opacity"] == 1.0

    def test_playing_and_progress_are_read_without_places(self, host: FakeHost) -> None:
        """壊れると、人が書き出しの窓で書き出している最中に AI が別の書き出しを始めて待たせる
        書き出し先の場所を載せると、名前と場所の同意の外で場所を送る"""
        host.screen = replace(host.screen, playing=True, person_export=0.4567)
        host.export_state = {"state": "rendering", "progress": 0.25, "path": "C:/秘密/完成.mp4"}
        result = run(host, "get_screen")
        assert result["playing"] is True
        assert result["progress"]["person_export"] == 0.457
        assert result["progress"]["assistant_export"] == {"state": "rendering", "progress": 0.25}
        assert "秘密" not in json.dumps(result, ensure_ascii=False)


class TestSelection:
    def test_the_media_and_subtitle_rows_the_person_chose_are_read(
        self, host: FakeHost, transcript: Transcript
    ) -> None:
        """壊れると、「選んだ行を直して」「この素材を起こして」の「選んだ」「この」を、AI が
        人と同じ物として読めず、別の行や素材を直す"""
        media = host.document.project.media[0]
        rows = (transcript.segments[1].id, transcript.segments[2].id)
        host.screen = replace(
            host.screen, pool_media=media.id, subtitle_media=media.id, subtitle_rows=rows
        )
        result = run(host, "get_selection")
        assert result["media_id"] == str(media.id)
        assert result["subtitles"] == {
            "media_id": str(media.id),
            "audio": 1,
            "segment_ids": [str(row) for row in rows],
        }

    def test_nothing_chosen_reads_as_empty(self, host: FakeHost) -> None:
        """壊れると、何も選んでいないのに前の素材を選んだように読ませ、AI が違う素材を直す"""
        result = run(host, "get_selection")
        assert result["media_id"] is None
        assert result["subtitles"] == {"media_id": None, "audio": None, "segment_ids": []}


class TestProjectView:
    def test_the_work_area_is_read(self, host: FakeHost) -> None:
        """壊れると、export_video の range を省いたときにどこが書き出されるのか AI に分からず、
        人が決めた範囲だけの書き出しを全体と思い込む"""
        assert run(host, "get_project")["work_area"] is None
        host.apply_commands([SetWorkArea((30, 90))], "範囲")
        area = run(host, "get_project")["work_area"]
        assert (area["start_frame"], area["end_frame"]) == (30, 90)

    def test_track_volume_is_read(self, host: FakeHost) -> None:
        """壊れると、トラックの音量で小さく鳴っている訳を AI がクリップの音量に探しに行く"""
        row = run(host, "list_tracks")["tracks"][0]
        assert row["volume_db"] == 0.0 and row["pan"] == 0.0


#: 記録に混ざりうる利用者の文字 図形の名前・debug_print の文・画像の名前・場所
USER_RECORDS = (
    "図形の種類: 秘密の星",
    "debug_print: 秘密の挨拶",
    'obj.load("秘密.png")',
    "C:/Users/秘密の人/scripts/a.anm",
)


def _report() -> CompatReport:
    recorded = CompatibilityReport()
    for _ in range(3):
        recorded.note_missing("obj.getpixeldata")
    recorded.note_missing("obj.copybuffer")
    recorded.note_control("<#ff0000,5>")
    recorded.note_failure("震え.anm", "開けない: C:/Users/秘密の人/scripts/震え.anm")
    return CompatReport.of(recorded, scripts=12, folders=("C:/Users/秘密の人/scripts",))


class TestCompatReport:
    def test_the_records_are_read_most_used_first(self, host: FakeHost) -> None:
        """壊れると、読み込んだ作品が AviUtl と違って見える訳を AI が探せず、人が窓を開いて
        読み上げることになる 多い順でないと、先に埋めるべき穴を取り違える"""
        host.compat = _report()
        result = run(host, "get_compat_report")
        assert result["scripts"] == 12
        assert result["summary"] == host.compat.summary
        assert [row["name"] for row in result["records"]] == [
            "obj.getpixeldata",
            "obj.copybuffer",
            "<#ff0000,5>",
            "震え.anm",
        ]
        assert result["records"][0]["count"] == 3

    def test_places_are_hidden_when_names_are_refused(self, host: FakeHost) -> None:
        """壊れると、名前と場所を送らないと決めた人の、スクリプトの置き場（ユーザー名入り）が
        互換性レポートを読ませただけで外へ出る"""
        host.compat = _report()
        bridge = EditorBridge(host)
        bridge.set_send_choices(
            dict.fromkeys(ASKED, SendChoice.ALLOW) | {ASKED[2]: SendChoice.DENY}
        )
        operation = find_operation("get_compat_report")
        assert operation is not None
        done, box = call_in_thread(bridge, operation, {})
        deadline = time.monotonic() + 5
        while not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            time.sleep(0.005)
        assert done.is_set() and "error" not in box
        text = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密の人" not in text
        assert "obj.getpixeldata" in text

    @pytest.mark.parametrize("refused", [Outbound.FILES, Outbound.TEXT])
    def test_known_api_names_are_read_and_the_rest_is_hidden(
        self, host: FakeHost, refused: Outbound
    ) -> None:
        """壊れると、名前と場所か字幕と文字を送らない人の所で、どの AviUtl の API が足りないかを
        AI が読めない（決まった名前まで伏せる）か、記録に混ざった利用者の図形の名前・
        debug_print の文・画像の名前が外へ出る（#391）"""
        recorded = CompatibilityReport()
        recorded.note_missing("obj.getpixeldata")
        recorded.note_missing("obj.zoom への代入")
        for secret in USER_RECORDS:
            recorded.note_missing(secret)
        host.compat = CompatReport.of(recorded, scripts=1, folders=())
        result = _through_bridge(host, "get_compat_report", {}, {refused})
        names = [row["name"] for row in result["records"]]
        assert "obj.getpixeldata" in names
        assert "obj.zoom への代入" in names
        text = json.dumps(result, ensure_ascii=False)
        assert "秘密" not in text
        assert names.count(TEXT_MARK) + names.count(PATH_MARK) == len(USER_RECORDS)

    def test_the_known_names_are_fixed_names_only(self) -> None:
        """壊れると、決まった名前の一覧に作品やスクリプトから組み立てた文字が混ざり、利用者の
        文字がそのまま送られる 一覧が obj の受ける名前を欠くと、その API の名前が読めない"""
        shape = re.compile(r"obj\.[a-z_0-9]+(?: への代入)?")
        assert all(shape.fullmatch(record) for record in KNOWN_RECORDS)
        accepted = (
            set(objapi._READABLE)
            | set(objapi._WRITABLE)
            | set(objapi._IDENTIFIERS)
            | {name.removeprefix("lua_") for name in dir(objapi.ObjApi) if name.startswith("lua_")}
        )
        assert accepted <= AVIUTL_OBJ_NAMES
        assert not is_known_record('obj.load("秘密.png")')
        assert not is_known_record("debug_print: obj.mes")


def test_the_reading_thread_is_not_the_editing_thread(host: FakeHost) -> None:
    """壊れると、画面の状態を AI のスレッドから読み、描いている途中のパネルと取り合って落ちる"""
    seen: list[int] = []

    class _Watching(FakeHost):
        def view_state(self) -> Any:
            seen.append(threading.get_ident())
            return super().view_state()

    watching = _Watching(host.document.project)
    bridge = EditorBridge(watching)
    bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
    operation = find_operation("get_screen")
    assert operation is not None
    done, _box = call_in_thread(bridge, operation, {})
    deadline = time.monotonic() + 5
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        time.sleep(0.005)
    assert seen == [threading.get_ident()]


# --- 欄の型で伏せる（#391 の Codex の指摘 2 回）---

#: 欄の型ごとに、値を送るのに要る許し ``src/sashimono/ai/outbound.py`` の ``FIELD_SENDS`` と
#: ``docs/development.md`` の「人が画面で見ている物を AI も読む」の表と同じ物を持つ
FIELD_TABLE: dict[type, set[Outbound]] = {
    TextSpec: {Outbound.TEXT},
    FileSpec: {Outbound.FILES},
}
#: 定義の分からない種類の文字の値
UNKNOWN_TABLE = {Outbound.TEXT, Outbound.FILES}
#: 文字の欄と場所の欄に書きうる値 長さ・空白・相対の名前・ユーザー名入りの場所
#: 表に集めて照らす伏せ方では、短い値と相対の名前が漏れた
TRICKY_VALUES = ["字", " ", "模様.png", "C:/Users/秘密の人/模様.png", "ここだけの秘密の文です"]
#: 文字の欄と場所の欄を好きな名前で持つスクリプト 試験の間だけ登録する
SCRIPT_KIND = "aviutl.script:試しの欄@試験"
#: 読み込んでいない（定義の分からない）スクリプト
UNKNOWN_KIND = "aviutl.script:読み込んでいない@試験"


@pytest.fixture
def script_fields() -> Iterator[None]:
    registry.register(
        EffectDefinition(
            kind=SCRIPT_KIND,
            label="試しの欄",
            category="試し",
            parameters=(TextSpec("message", "文字", ""), FileSpec("picture", "画像")),
        )
    )
    yield
    registry.unregister(SCRIPT_KIND)


@dataclass(frozen=True)
class _Field:
    """伏せ方を見る欄 1 つ どこに置き、結果のどこに出て、何の許しが要るか"""

    label: str
    #: 値を書き込んだテキストのクリップの、get_clip の結果の中の値を返す
    place: Callable[[FakeHost, ClipId, str], Callable[[dict[str, Any]], Any]]
    needs: set[Outbound]


def _in_source(name: str) -> Callable[[FakeHost, ClipId, str], Callable[[dict[str, Any]], Any]]:
    def place(host: FakeHost, clip: ClipId, value: str) -> Callable[[dict[str, Any]], Any]:
        host.apply_commands([SetParam(ParamPath.of_source(clip, name), value)], "欄")
        return lambda result: result["params"][name]

    return place


def _in_effect(
    kind: str, name: str
) -> Callable[[FakeHost, ClipId, str], Callable[[dict[str, Any]], Any]]:
    def place(host: FakeHost, clip: ClipId, value: str) -> Callable[[dict[str, Any]], Any]:
        effect = Effect(kind=kind, params={name: value})
        host.apply_commands([AddEffect(clip, effect)], "欄")
        effect_id = str(effect.id)
        return lambda result: next(
            row["params"][name] for row in result["effects"] if row["effect_id"] == effect_id
        )

    return place


FIELDS = [
    _Field("テキストの本文（text）", _in_source("text"), FIELD_TABLE[TextSpec]),
    _Field("text 以外の名前の文字の欄", _in_source("timer_format"), FIELD_TABLE[TextSpec]),
    _Field("縁取りの模様の画像（pattern）", _in_effect("border", "pattern"), FIELD_TABLE[FileSpec]),
    _Field("スクリプトの文字の欄", _in_effect(SCRIPT_KIND, "message"), FIELD_TABLE[TextSpec]),
    _Field("スクリプトの場所の欄", _in_effect(SCRIPT_KIND, "picture"), FIELD_TABLE[FileSpec]),
    _Field("定義の分からないスクリプト", _in_effect(UNKNOWN_KIND, "anything"), UNKNOWN_TABLE),
]


#: 場所を持つ欄の名前と表示名の手掛かり 文字の欄（TextSpec）にこれが付いていたら型の誤り
_PLACE_NAME = re.compile(r"(?:^|_)(?:path|file|folder|dir|directory|image|picture|sound)s?(?:_|$)")
_PLACE_LABEL = ("ファイル", "フォルダ", "場所", "パス", "画像")
_FILE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{2,4}$")


def looks_like_a_place(spec: object) -> bool:
    """名前・表示名・既定値から、ファイルや場所を持つ欄に見えるか"""
    name = str(getattr(spec, "name", ""))
    label = str(getattr(spec, "label", ""))
    default = getattr(spec, "default", "")
    return (
        _PLACE_NAME.search(name) is not None
        or any(word in label for word in _PLACE_LABEL)
        or (isinstance(default, str) and _FILE_SUFFIX.search(default) is not None)
    )


def places_typed_as_text() -> list[str]:
    """組み込みの全種類（エフェクト・中身・縁取りの層）で、場所に見えるのに文字の欄の物"""
    owners: list[tuple[str, tuple[Any, ...]]] = [
        (definition.kind, definition.parameters) for definition in registry.all()
    ]
    owners += [(source.kind, source.parameters) for source in source_registry.all()]
    owners.append(("縁取りの層", STROKE.parameters))
    return [
        f"{kind}: {spec.name}"
        for kind, specs in owners
        for spec in specs
        if isinstance(spec, TextSpec) and looks_like_a_place(spec)
    ]


def _read_through_bridge(host: FakeHost, refused: set[Outbound]) -> dict[str, Any]:
    return _through_bridge(host, "get_clip", {}, refused)


def _through_bridge(
    host: FakeHost, name: str, arguments: dict[str, Any], refused: set[Outbound]
) -> dict[str, Any]:
    """``refused`` を送らない人の所で、変更は自動で承認にして道具を動かした結果"""
    bridge = EditorBridge(host, auto_approve=True)
    bridge.set_send_choices(
        {kind: SendChoice.DENY if kind in refused else SendChoice.ALLOW for kind in ASKED}
    )
    operation = find_operation(name)
    assert operation is not None
    done, box = call_in_thread(bridge, operation, arguments)
    deadline = time.monotonic() + 5
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        time.sleep(0.005)
    assert done.is_set() and "error" not in box, box.get("error")
    result = box["result"]
    assert isinstance(result, dict)
    return result


class TestFieldsAreHiddenByType:
    def test_the_table_is_the_one_the_bridge_uses(self) -> None:
        """壊れると、どの欄を字幕と文字・名前と場所として伏せるかが、決まりの表・説明・試験で
        食い違い、表に足した型が伏せられないか、伏せたつもりの型が送られる"""
        assert {kind: set(needs) for kind, needs in FIELD_SENDS.items()} == FIELD_TABLE
        assert set(UNKNOWN_FIELD_SENDS) == UNKNOWN_TABLE

    @pytest.mark.parametrize("field", FIELDS, ids=lambda field: field.label)
    @pytest.mark.parametrize("value", TRICKY_VALUES)
    def test_a_refused_field_is_hidden_whatever_its_value(
        self, host: FakeHost, script_fields: None, field: _Field, value: str
    ) -> None:
        """壊れると、送らないと決めた文字や場所が、短い・空白・相対の名前・好きな欄の名前という
        だけで伏せられずに get_clip から外へ出る（表に集めて照らす伏せ方の抜け）"""
        del script_fields
        run(host, "add_text", text="見出し")
        clip = _text_clip(host)
        host.select_clip(clip)
        read = field.place(host, clip, value)
        for refused in field.needs:
            shown = read(_read_through_bridge(host, {refused}))
            assert shown in (TEXT_MARK, PATH_MARK), (refused, shown)

    @pytest.mark.parametrize("field", FIELDS, ids=lambda field: field.label)
    @pytest.mark.parametrize("value", TRICKY_VALUES)
    def test_an_allowed_field_is_sent_as_it_is(
        self, host: FakeHost, script_fields: None, field: _Field, value: str
    ) -> None:
        """壊れると、送ってよいと決めた人にも値が印で返り、AI が今の文字や画像を読めずに直す"""
        del script_fields
        run(host, "add_text", text="見出し")
        clip = _text_clip(host)
        host.select_clip(clip)
        read = field.place(host, clip, value)
        assert read(_read_through_bridge(host, set())) == value

    def test_the_waveform_sound_file_is_hidden_as_a_place(self, host: FakeHost) -> None:
        """壊れると、音声波形の音声ファイル（audio_path）が文字の欄として扱われ、字幕と文字を
        許して名前と場所を断った人の所で「秘密.wav」のような相対の名前が外へ出る（#391 の Codex）"""
        run(host, "add_shape", shape="waveform")
        clip = next(
            c.id
            for track in host.document.project.timeline.tracks
            for c in track.clips
            if c.source is not None and c.source.kind == "shape"
        )
        host.apply_commands(
            [SetParam(ParamPath.of_source(clip, "audio_path"), "秘密.wav")], "音声ファイル"
        )
        host.select_clip(clip)
        assert _read_through_bridge(host, {Outbound.FILES})["params"]["audio_path"] == PATH_MARK
        assert _read_through_bridge(host, set())["params"]["audio_path"] == "秘密.wav"
        # 書く道具の結果の値（項目の名前が value で、場所の項目の名前では伏せられない）
        changed = _through_bridge(
            host,
            "set_param",
            {"clip_id": str(clip), "name": "audio_path", "value": "別の秘密.wav"},
            {Outbound.FILES},
        )
        assert changed["value"] == PATH_MARK

    def test_no_built_in_place_is_typed_as_text(self) -> None:
        """壊れると、ファイルや場所を持つ欄を文字の欄（TextSpec）で定義した人がいても見張りが
        気付かず、名前と場所を断った人の所でその場所が字幕と文字として送られる"""
        assert not places_typed_as_text()

    def test_a_place_typed_as_text_is_caught(self) -> None:
        """壊れると、上の見張りが名前や既定値を見ずに通り、audio_path のような欄を見逃す"""
        wrong = (
            TextSpec("audio_path", "音声ファイル", ""),
            TextSpec("logo", "ロゴの画像", "logo.png"),
            TextSpec("message", "文字", ""),
        )
        assert [spec.name for spec in wrong if looks_like_a_place(spec)] == ["audio_path", "logo"]

    def test_numbers_are_not_hidden(self, host: FakeHost) -> None:
        """壊れると、文字と場所を送らない人の所で大きさや位置まで印になり、何も直せない"""
        run(host, "add_text", text="見出し", size=80)
        host.select_clip(_text_clip(host))
        result = _read_through_bridge(host, {Outbound.TEXT, Outbound.FILES})
        assert result["params"]["size"] == 80.0
        assert result["params"]["text"] == TEXT_MARK
