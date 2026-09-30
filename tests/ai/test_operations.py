"""AI に見せる編集操作

ここが AI からの唯一の入口なので、読み取りが正しい形を返すことと、変更が
ちゃんと履歴に載ることを押さえる 失敗したときの文面も見る AI はエラーの
文面だけを頼りに次の手を決めるため
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import OPERATIONS, ImageResult, find_operation
from sashimono.core.model import AnimatedValue, ClipId, MediaItem, Project
from tests.ai.conftest import FakeHost


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    """ツールを 1 つ呼ぶ

    引数は位置専用にしてある ツールの引数に ``name`` があるので、普通に書くと
    この関数の引数と衝突する
    """
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


def _clip_id(project: Project) -> ClipId:
    return project.timeline.tracks[0].clips[0].id


class TestCatalogue:
    def test_every_operation_has_a_schema_and_description(self) -> None:
        for operation in OPERATIONS:
            assert operation.description.strip()
            assert operation.schema["type"] == "object"
            assert "properties" in operation.schema

    def test_names_are_unique(self) -> None:
        names = [operation.name for operation in OPERATIONS]
        assert len(set(names)) == len(names)

    def test_read_and_write_are_separated(self) -> None:
        # この区別が確認ダイアログの要否を決める 読み取りに writes が付くと、
        # 一覧を見るだけで許可を求められることになる
        assert find_operation("list_clips") is not None
        assert find_operation("list_clips").writes is False  # type: ignore[union-attr]
        assert find_operation("split_clip").writes is True  # type: ignore[union-attr]


class TestReading:
    def test_get_project(self, host: FakeHost) -> None:
        result = run(host, "get_project")
        assert result["resolution"] == "1920x1080"
        assert result["fps"] == 30.0
        assert result["duration_frames"] == 300

    def test_list_media_exposes_ids(self, host: FakeHost) -> None:
        rows = run(host, "list_media")
        assert len(rows) == 1
        assert rows[0]["subtitle_count"] == 3
        assert rows[0]["has_audio"] is True

    def test_list_clips_reports_positions(self, host: FakeHost) -> None:
        rows = run(host, "list_clips")
        assert rows[0]["start"] == 0
        assert rows[0]["duration"] == 300
        assert rows[0]["start_timecode"] == "00:00:00:00"

    def test_list_clips_can_filter_by_track(self, host: FakeHost) -> None:
        assert run(host, "list_clips", track_id="そんなものは無い") == []

    def test_get_subtitles_returns_timeline_positions(self, host: FakeHost) -> None:
        rows = run(host, "get_subtitles")
        # 素材の 1 秒は 30 フレーム目 AI が見るのは編集後の位置
        assert [row["start"] for row in rows] == [30, 120, 210]
        assert rows[0]["text"] == "今日は"

    def test_list_effects_includes_ranges(self, host: FakeHost) -> None:
        rows = run(host, "list_effects")
        blur = next(row for row in rows if row["kind"] == "blur")
        radius = next(p for p in blur["parameters"] if p["name"] == "radius")
        assert radius["type"] == "number"
        assert radius["max"] > radius["min"]

    def test_get_history_is_empty_at_first(self, host: FakeHost) -> None:
        assert run(host, "get_history")["can_undo"] is False


class TestPreviewFrame:
    def test_playback_stops_before_drawing(self, host: FakeHost) -> None:
        # 再生しながら別のフレームを描くと GL の資源を取り合う
        result = run(host, "preview_frame", frame=60)
        assert host.stopped == 1
        assert isinstance(result, ImageResult)
        assert result.png.startswith(b"\x89PNG")

    def test_it_defaults_to_the_playhead(self, host: FakeHost) -> None:
        host.frame = 90
        run(host, "preview_frame")
        assert host.rendered[-1][0] == 90

    def test_the_frame_is_clamped_to_the_timeline(self, host: FakeHost) -> None:
        run(host, "preview_frame", frame=99999)
        assert host.rendered[-1][0] == 299

    def test_the_width_is_clamped(self, host: FakeHost) -> None:
        run(host, "preview_frame", width=99999)
        assert host.rendered[-1][1] == 1280

    def test_the_caption_carries_the_timecode(self, host: FakeHost) -> None:
        result = run(host, "preview_frame", frame=30)
        assert isinstance(result, ImageResult)
        assert "00:00:01:00" in result.caption


class TestEditing:
    def test_split_uses_the_playhead_and_the_selection(self, host: FakeHost) -> None:
        host.select_clip(_clip_id(host.document.project))
        host.frame = 100
        run(host, "split_clip")
        assert len(host.document.project.timeline.tracks[0].clips) == 2

    def test_editing_lands_in_the_history(self, host: FakeHost) -> None:
        run(host, "split_clip", clip_id=str(_clip_id(host.document.project)), frame=100)
        assert host.document.can_undo is True

    def test_a_failed_edit_explains_itself(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="分割位置"):
            run(host, "split_clip", clip_id=clip, frame=9999)

    def test_an_unknown_clip_points_at_the_listing_tool(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_clips"):
            run(host, "split_clip", clip_id="ないよ", frame=10)

    def test_without_a_selection_it_asks_for_an_id(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="clip_id"):
            run(host, "split_clip")

    def test_delete_with_ripple(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        run(host, "delete_clip", clip_id=clip, ripple=True)
        assert host.document.project.duration == 0

    def test_trim_needs_a_direction(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="head_delta"):
            run(host, "trim_clip", clip_id=clip)

    def test_speed_is_kept_exact(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        run(host, "set_clip_property", clip_id=clip, name="speed", value=0.5)
        assert host.document.project.timeline.tracks[0].clips[0].speed == Fraction(1, 2)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1.5", Fraction(3, 2)), (2, Fraction(2)), (0.5, Fraction(1, 2)), (None, None)],
    )
    def test_the_hold_is_kept_as_a_fraction(
        self, host: FakeHost, value: object, expected: Fraction | None
    ) -> None:
        # 直さずに渡すと、整数や小数は保存の所で落ち、文字列はクリップを作る所で落ちる
        clip = str(_clip_id(host.document.project))
        run(host, "set_clip_property", clip_id=clip, name="hold_at", value=value)
        held = host.document.project.timeline.tracks[0].clips[0].hold_at
        assert held == expected
        assert held is None or isinstance(held, Fraction)

    @pytest.mark.parametrize("value", ["止める", -1, True, [1]])
    def test_a_hold_that_is_not_a_time_is_refused(self, host: FakeHost, value: object) -> None:
        # 通すと、読めない値で落ちるか、true が 1 秒として入り、どこで止めたのか分からない
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="hold_at"):
            run(host, "set_clip_property", clip_id=clip, name="hold_at", value=value)
        assert host.document.project.timeline.tracks[0].clips[0].hold_at is None

    def test_a_bad_start_position_names_itself(self, host: FakeHost) -> None:
        # 文面に hold_at と出すと、AI は source_in ではなく hold_at を直しに行く
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="source_in は 0 以上"):
            run(host, "set_clip_property", clip_id=clip, name="source_in", value=-1)

    def test_only_allowed_properties_can_change(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="変えられるのは"):
            run(host, "set_clip_property", clip_id=clip, name="duration", value=10)


class TestEffects:
    def test_adding_an_effect_returns_its_id(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        result = run(host, "add_effect", clip_id=clip, kind="blur", params={"radius": 12})
        assert result["kind"] == "blur"
        effects = host.document.project.timeline.tracks[0].clips[0].effects
        assert str(effects[0].id) == result["effect_id"]

    def test_an_unknown_effect_lists_the_real_ones(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="blur"):
            run(host, "add_effect", clip_id=clip, kind="いい感じにするやつ")

    def test_keyframes_can_be_placed_on_an_effect(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        added = run(host, "add_effect", clip_id=clip, kind="blur")
        run(
            host,
            "add_keyframe",
            clip_id=clip,
            effect_id=added["effect_id"],
            name="radius",
            frame=0,
            value=0,
        )
        run(
            host,
            "add_keyframe",
            clip_id=clip,
            effect_id=added["effect_id"],
            name="radius",
            frame=60,
            value=30,
        )
        effect = host.document.project.timeline.tracks[0].clips[0].effects[0]
        radius = effect.params["radius"]
        assert len(radius.keyframes) == 2  # type: ignore[union-attr]

    def test_an_unknown_interpolation_lists_the_choices(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        with pytest.raises(ToolError, match="補間"):
            run(
                host,
                "add_keyframe",
                clip_id=clip,
                target="clip",
                name="opacity",
                frame=0,
                value=1,
                interpolation="ぬるっと",
            )


class TestText:
    def test_adding_text_places_a_clip(self, host: FakeHost) -> None:
        run(host, "add_text", text="テロップ", at_frame=0, duration=60, size=72)
        placed = [
            clip
            for track in host.document.project.timeline.tracks
            for clip in track.clips
            if clip.source is not None
        ]
        assert len(placed) == 1
        assert placed[0].source is not None
        assert placed[0].source.params["text"] == "テロップ"

    def test_empty_text_is_refused(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="text"):
            run(host, "add_text", text="   ")


class TestSubtitles:
    def test_text_can_be_rewritten(self, host: FakeHost) -> None:
        media = host.document.project.media[0]
        assert media.transcript is not None
        run(
            host,
            "set_subtitle_text",
            media_id=str(media.id),
            segment_id=str(media.transcript.segments[0].id),
            text="こんばんは",
        )
        updated = host.document.project.media[0].transcript
        assert updated is not None
        assert updated.segments[0].text == "こんばんは"

    def test_cleaning_reports_what_changed(self, host: FakeHost) -> None:
        media = host.document.project.media[0]
        result = run(host, "clean_subtitles", media_id=str(media.id))
        assert result["remaining"] == 3

    def test_cleaning_without_a_transcript_says_so(
        self, host: FakeHost, video_media: MediaItem
    ) -> None:
        del video_media
        media = host.document.project.media[0]
        host.document.execute(
            _forget_transcript(host.document.project, str(media.id))  # type: ignore[arg-type]
        )
        with pytest.raises(ToolError, match="字幕がありません"):
            run(host, "clean_subtitles", media_id=str(media.id))

    def test_transcribe_hands_back_a_next_step(self, host: FakeHost) -> None:
        media = host.document.project.media[0]
        result = run(host, "transcribe", media_id=str(media.id))
        assert "transcription_status" in result["next"]
        assert run(host, "transcription_status")["status"] == "large-v3 で開始"
        # 省くと今までどおり 1 本目
        assert host.transcribed_stream is None

    def test_transcribe_can_pick_the_second_voice(self, host: FakeHost) -> None:
        # ゲームの録画のマイクの声（音声 2）を起こせないと、声ではなくゲームの音が字幕になる
        from dataclasses import replace as _replace

        from sashimono.core.commands import AddMedia
        from sashimono.core.model import MediaId

        first = host.document.project.media[0]
        stream = first.audio_streams[0]
        two = _replace(
            first,
            id=MediaId("two-voices"),
            transcript=None,
            audio_streams=(stream, _replace(stream, index=stream.index + 1)),
        )
        host.document.execute(AddMedia(two))
        listed = {m["media_id"]: m for m in run(host, "list_media")}
        assert listed["two-voices"]["audio_count"] == 2
        run(host, "transcribe", media_id="two-voices", audio=2)
        assert host.transcribed_stream == stream.index + 1
        with pytest.raises(ToolError, match="2 本"):
            run(host, "transcribe", media_id="two-voices", audio=3)


def _forget_transcript(project: Project, media_id: str) -> object:
    from sashimono.core.commands import SetTranscript
    from sashimono.core.model import MediaId

    del project
    return SetTranscript(MediaId(media_id), None)


class TestJetCut:
    def test_without_a_waveform_it_says_to_wait(self, host: FakeHost) -> None:
        media = host.document.project.media[0]
        with pytest.raises(ToolError, match="波形"):
            run(host, "jet_cut", media_id=str(media.id))

    def test_silence_becomes_a_cut(self, host: FakeHost) -> None:
        from tests.ui.test_subtitle_panel import make_waveform

        # 先頭に無音、あとは鳴っている素材
        host.stub_waveform = make_waveform([(0.0, 400), (0.5, 1500)])
        media = host.document.project.media[0]
        result = run(host, "jet_cut", media_id=str(media.id), keep_speech=False)
        assert result["cuts"] == 1
        assert host.document.project.duration < 300


class TestImport:
    def test_import_probes_and_places(self, host: FakeHost, tmp_path: Path) -> None:
        target = tmp_path / "追加.mp4"
        target.write_bytes(b"dummy")
        result = run(host, "import_media", paths=[str(target)])
        assert len(result["imported"]) == 1
        assert host.probed == [target]
        # 波形とサムネイルの用意も頼む 頼まないと無音カットがいつまでも使えない
        assert len(host.analyzed) == 1

    def test_a_missing_file_is_reported(self, host: FakeHost, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match="ファイルがありません"):
            run(host, "import_media", paths=[str(tmp_path / "無い.mp4")])


class TestNavigation:
    def test_seek_moves_the_playhead(self, host: FakeHost) -> None:
        run(host, "seek", frame=42)
        assert host.playhead == 42

    def test_select_checks_the_clip_exists(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_clips"):
            run(host, "select_clip", clip_id="ないよ")

    def test_undo_reports_what_it_undid(self, host: FakeHost) -> None:
        clip = str(_clip_id(host.document.project))
        run(host, "split_clip", clip_id=clip, frame=100)
        result = run(host, "undo")
        assert result["undone"] == ["クリップを分割"]
        assert len(host.document.project.timeline.tracks[0].clips) == 1

    def test_undo_with_nothing_to_undo(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="戻せる操作がありません"):
            run(host, "undo")


class TestParameterCoercion:
    """AI が渡した値を、パラメータ定義に従って寄せること

    ここを通さないと、色に "#FFFFFF" という文字列がそのまま入る 描画側は
    黙って無視するので、見た目が変わらないまま「やりました」と言われる
    """

    def _text_clip(self, host: FakeHost) -> str:
        run(host, "add_text", text="テロップ", at_frame=0, duration=60)
        for track in host.document.project.timeline.tracks:
            for clip in track.clips:
                if clip.source is not None:
                    return str(clip.id)
        raise AssertionError("テキストが置かれていない")

    def _source_params(self, host: FakeHost, clip_id: str) -> dict[str, object]:
        located = host.document.project.timeline.locate_clip(ClipId(clip_id))
        assert located is not None and located[1].source is not None
        return dict(located[1].source.params)

    def test_hex_colours_become_channels(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="color", value="#FF8000")
        color = self._source_params(host, clip)["color"]
        assert isinstance(color, tuple)
        assert color[0] == pytest.approx(1.0)
        assert color[1] == pytest.approx(128 / 255, abs=0.01)
        assert color[3] == pytest.approx(1.0)

    def test_hex_with_alpha(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="color", value="#00000080")
        color = self._source_params(host, clip)["color"]
        assert isinstance(color, tuple)
        assert color[3] == pytest.approx(128 / 255, abs=0.01)

    def test_a_list_also_works(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="color", value=[1.0, 0.0, 0.0, 1.0])
        assert self._source_params(host, clip)["color"] == (1.0, 0.0, 0.0, 1.0)

    def test_numbers_are_clamped_to_the_declared_range(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="size", value=9999)
        size = self._source_params(host, clip)["size"]
        # サイズの上限は 512 範囲外を渡しても定義の側に寄る
        assert isinstance(size, AnimatedValue)
        assert float(size.at(0)) == 512.0

    def test_booleans_reach_check_parameters(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="bold", value=True)
        assert bool(self._source_params(host, clip)["bold"]) is True

    def test_select_parameters_take_their_identifier(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="align", value="left")
        assert self._source_params(host, clip)["align"] == "left"

    def test_text_stays_text(self, host: FakeHost) -> None:
        clip = self._text_clip(host)
        run(host, "set_param", clip_id=clip, name="text", value="差し替えた")
        assert self._source_params(host, clip)["text"] == "差し替えた"

    def test_effect_parameters_use_the_effect_definition(self, host: FakeHost) -> None:
        target = str(_clip_id(host.document.project))
        added = run(host, "add_effect", clip_id=target, kind="border")
        run(
            host,
            "set_param",
            clip_id=target,
            effect_id=added["effect_id"],
            name="color",
            value="#00FF00",
        )
        effect = host.document.project.timeline.tracks[0].clips[0].effects[0]
        assert effect.params["color"] == (0.0, 1.0, 0.0, 1.0)

    def test_the_colour_format_is_advertised(self, host: FakeHost) -> None:
        # 形式を伝えておかないと、AI は色名や rgb() を送ってくる
        rows = run(host, "list_effects")
        text = next(row for row in rows if row["kind"] == "text")
        color = next(p for p in text["parameters"] if p["name"] == "color")
        assert "#RRGGBB" in color["format"]
