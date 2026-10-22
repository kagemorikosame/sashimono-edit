"""人が画面で見ている物を AI も読める（読み取りの同等性 #352）

設定パネルとグラフエディタの値（get_clip）・表示の様子（get_screen）・人の選択（get_selection）・
書き出し範囲（get_project）・互換性レポート（get_compat_report）を、偽のホストで確かめる
本物の窓で同じ口を辿る試験は ``tests/ui/test_screen_reading.py``
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import CompatReport, ToolError
from sashimono.ai.operations import MAX_RESULT_CHARS, find_operation
from sashimono.ai.outbound import ASKED, SendChoice
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.commands import AddEffect, ParamPath, SetKeyframe, SetWorkArea
from sashimono.core.io.serialize import json_text
from sashimono.core.model import ClipId, Effect, MediaItem, Project, Transcript
from sashimono.core.timebase import format_timecode
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
        assert "width" in result["strokes"][0]["params"]

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
