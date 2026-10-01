"""AI の道具の返事を、打ち切られない長さに収める（利用者の画面）

配布版の Claude Code は MCP の道具の返事を 25000 トークンで打ち切り、残りをファイルへ
退ける アシスタントにはファイルを読む道具が無いので、字幕 328 行の get_subtitles の
続きを読めず、誤植を直せなかった 絞り込みと続きの辿り方、「誤 → 正」をまとめて
置き換える道具で直せるようにする
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import MAX_RESULT_CHARS, find_operation
from sashimono.core.io.serialize import json_text, project_from_dict, project_to_dict
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.timebase import FrameRate
from tests.ai.conftest import FakeHost

#: 打ち切られない長さの目安（字数） 25000 トークンの上限に、日本語 1 文字 1〜2 トークンで
#: 収まる所 道具はこれより短い :data:`MAX_RESULT_CHARS` を狙う
SDK_SAFE_CHARS = 16000
LINES = 328


def _call(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, tool
    return operation(host, arguments)


def _host(video_media: MediaItem) -> FakeHost:
    """字幕 328 行（1 行 1 秒 誤植「誤字」を 3 行に混ぜる）を持つ素材を 1 本置いた作品"""
    segments = tuple(
        TranscriptSegment(
            Fraction(n),
            Fraction(n) + Fraction(9, 10),
            f"{n} 行目の字幕です これは長めの本文で、読み戻しの長さを確かめるための物"
            + ("（誤字）" if n in (10, 200, 300) else ""),
        )
        for n in range(LINES)
    )
    media = replace(video_media, duration=Fraction(LINES + 5)).with_transcript(Transcript(segments))
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
    clip = Clip(timeline_start=0, duration=(LINES + 5) * 30, media_id=media.id)
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    return FakeHost(base.with_timeline(replace(base.timeline, tracks=(track,))))


def _short(result: object) -> None:
    text = json_text(result)
    assert len(text) <= SDK_SAFE_CHARS, f"返事が {len(text)} 字（打ち切られる）"


class TestReadingSubtitles:
    def test_every_page_is_short_and_all_lines_come_back(self, video_media: MediaItem) -> None:
        # 前は 328 行を 1 回で返し、打ち切られて続きが読めなかった
        host = _host(video_media)
        texts: list[str] = []
        offset = 0
        for _ in range(100):
            page = _call(host, "get_subtitles", offset=offset)
            _short(page)
            texts += [row["text"] for row in page["subtitles"]]
            if "next_offset" not in page:
                break
            assert "offset=" in page["note"]
            offset = page["next_offset"]
        assert len(texts) == LINES
        assert texts[0].startswith("0 行目")

    def test_even_a_large_limit_stays_short(self, video_media: MediaItem) -> None:
        page = _call(_host(video_media), "get_subtitles", limit=10_000)
        _short(page)
        assert len(json_text(page["subtitles"])) <= MAX_RESULT_CHARS

    def test_the_compact_form_has_only_the_line_time_and_text(self, video_media: MediaItem) -> None:
        page = _call(_host(video_media), "get_subtitles", compact=True, limit=300)
        _short(page)
        assert set(page["subtitles"][0]) == {"n", "time", "text"}
        # 軽い形なので、同じ長さの上限でより多く読める
        full = _call(_host(video_media), "get_subtitles", limit=300)
        assert page["count"] > full["count"]

    def test_words_and_times_narrow_it(self, video_media: MediaItem) -> None:
        host = _host(video_media)
        found = _call(host, "get_subtitles", contains="誤字", compact=True)
        assert [row["text"].split()[0] for row in found["subtitles"]] == ["10", "200", "300"]
        timed = _call(host, "get_subtitles", from_seconds=100, to_seconds=104.5)
        assert [row["text"].split()[0] for row in timed["subtitles"]] == [
            "100",
            "101",
            "102",
            "103",
            "104",
        ]


class TestOneLongRow:
    """1 件だけで返事の上限を越える行も、上限に収めて返す（PR #231 の指摘）

    前は件数を 1 件まで減らしたらそのまま返し、長い字幕 1 行で返事が打ち切られた
    """

    LONG = "長い字幕" * 6000

    def _long_host(self, video_media: MediaItem) -> FakeHost:
        segments = (
            TranscriptSegment(Fraction(0), Fraction(1), "短い行"),
            TranscriptSegment(Fraction(1), Fraction(2), self.LONG),
        )
        media = replace(video_media, duration=Fraction(5)).with_transcript(Transcript(segments))
        base = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
        clip = Clip(timeline_start=0, duration=150, media_id=media.id)
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        return FakeHost(base.with_timeline(replace(base.timeline, tracks=(track,))))

    @pytest.mark.parametrize("compact", [False, True])
    def test_the_answer_stays_short_and_says_how_to_read_it_all(
        self, video_media: MediaItem, compact: bool
    ) -> None:
        host = self._long_host(video_media)
        page = _call(host, "get_subtitles", compact=compact)
        assert len(json_text(page)) <= MAX_RESULT_CHARS
        long_row = page["subtitles"][1]
        assert long_row["truncated"] is True
        assert "segment_id" in long_row["text"] or "segment_id" in page["note"]
        # 全文は segment_id と text_offset で辿れば元どおり
        whole = ""
        text_offset = 0
        for _ in range(20):
            part = _call(
                host,
                "get_subtitles",
                segment_id=long_row["segment_id"],
                text_offset=text_offset,
            )
            assert len(json_text(part)) <= MAX_RESULT_CHARS
            row = part["subtitles"][0]
            whole += row["text"]
            if "next_text_offset" not in row:
                break
            text_offset = row["next_text_offset"]
        assert whole == self.LONG

    def test_any_list_keeps_one_row_inside_the_limit(self) -> None:
        from sashimono.ai.operations import _paged

        row = {"id": "x", "name": "名前" * 9000, "items": list(range(5000))}
        page = _paged([row], {}, name="rows")
        assert len(json_text(page)) <= MAX_RESULT_CHARS
        assert page["rows"][0]["truncated"] is True
        assert page["rows"][0]["id"] == "x"


def test_subtitles_inside_a_placed_scene_name_their_media_and_voice() -> None:
    # 壊れると、置いたシーンの中の字幕は素材が null・音声が 1 で返り、AI が直す字幕を
    # 素材と音声で指せない（シーンのクリップは素材を持たない PR #231）
    from pathlib import Path

    from sashimono.core.commands import AddScene, new_scene
    from sashimono.core.model import AudioStreamInfo

    media = MediaItem(
        path=Path("C:/素材/録画.mp4"),
        duration=Fraction(10),
        audio_streams=(
            AudioStreamInfo(1, 48000, 2, Fraction(1, 48000), "aac"),
            AudioStreamInfo(2, 48000, 2, Fraction(1, 48000), "aac"),
        ),
    ).with_transcript(Transcript((TranscriptSegment(Fraction(1), Fraction(2), "声"),)), 2)
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
    scene = new_scene(base, "中")
    voice = Track(
        TrackKind.AUDIO,
        "A1",
        (Clip(timeline_start=0, duration=150, media_id=media.id, stream_index=2),),
    )
    project = AddScene(replace(scene, timeline=replace(scene.timeline, tracks=(voice,)))).apply(
        base
    )
    placed = Track(
        TrackKind.VIDEO, "V1", (Clip(timeline_start=0, duration=150, scene_id=scene.id),)
    )
    host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(placed,))))
    (row,) = _call(host, "get_subtitles")["subtitles"]
    assert row["text"] == "声"
    assert row["media_id"] == str(media.id)
    assert row["audio"] == 2
    # 返った素材と音声で絞っても同じ行が取れる
    narrowed = _call(host, "get_subtitles", media_id=str(media.id), audio=2)
    assert [r["text"] for r in narrowed["subtitles"]] == ["声"]


def test_rows_just_under_the_limit_leave_room_for_the_page() -> None:
    # 壊れると、行だけで上限ぎりぎりに収めたあと、件数・続き・案内を足した返事が上限を越え、
    # 打ち切られて続きの読み方まで読めなくなる（PR #231 の指摘）
    from sashimono.ai.operations import _paged

    size = 1
    while True:
        rows = [{"id": n, "text": "x" * size} for n in range(3)]
        if len(json_text(rows[:2])) > MAX_RESULT_CHARS - 5:
            break
        size += 50
    rows = [{"id": n, "text": "x" * (size - 50)} for n in range(3)]
    assert len(json_text(rows[:2])) <= MAX_RESULT_CHARS
    page = _paged(rows, {}, name="rows", truncated_note="全文は id で読む" * 50)
    assert len(json_text(page)) <= MAX_RESULT_CHARS
    assert page["next_offset"] == page["count"]


class TestTheSecondsWindow:
    """秒で絞るときの境目 from より後に掛かり、to より前に始まる物（説明どおり）

    前は to のコマに 1 を足していて、30fps で 1 秒ちょうどを渡すと 1 秒から始まる物まで返した
    （PR #231 の指摘）
    """

    def test_the_end_second_is_left_out(self, video_media: MediaItem) -> None:
        host = _host(video_media)
        rows = _call(host, "get_subtitles", from_seconds=0, to_seconds=1, compact=True)
        assert [row["text"].split()[0] for row in rows["subtitles"]] == ["0"]

    def test_a_little_past_the_second_takes_the_next(self, video_media: MediaItem) -> None:
        host = _host(video_media)
        rows = _call(host, "get_subtitles", from_seconds=0, to_seconds=1.01, compact=True)
        assert [row["text"].split()[0] for row in rows["subtitles"]] == ["0", "1"]

    def test_what_ends_at_the_start_second_is_left_out(self, video_media: MediaItem) -> None:
        # 0 行目は 0.9 秒で終わる 0.9 秒から絞れば外れ、0.89 秒からなら残る
        host = _host(video_media)
        at = _call(host, "get_subtitles", from_seconds=0.9, to_seconds=2, compact=True)
        assert [row["text"].split()[0] for row in at["subtitles"]] == ["1"]
        before = _call(host, "get_subtitles", from_seconds=0.89, to_seconds=2, compact=True)
        assert [row["text"].split()[0] for row in before["subtitles"]] == ["0", "1"]


class TestFixingTypos:
    def test_pairs_fix_the_subtitles_and_the_burned_text(self, video_media: MediaItem) -> None:
        host = _host(video_media)
        _call(host, "place_subtitles")
        before = host.document.project
        result = _call(
            host,
            "replace_subtitle_text",
            pairs=[{"from": "誤字", "to": "正字"}, {"from": "無い言葉", "to": "x"}],
        )
        assert result["replaced"] == 3
        assert result["unmatched"] == ["無い言葉"]
        assert result["burned_text_clips"] == 3
        project = host.document.project
        transcript = project.media[0].transcript
        assert transcript is not None
        assert sum("正字" in s.text for s in transcript.segments) == 3
        assert not any("誤字" in s.text for s in transcript.segments)
        burned = [
            str(c.source.params["text"])
            for t in project.timeline.tracks
            for c in t.clips
            if c.source is not None
        ]
        assert sum("正字" in text for text in burned) == 3
        assert not any("誤字" in text for text in burned)
        # 字幕と焼き込んだ文字の両方が 1 回の取り消しで戻る
        host.document.undo()
        assert host.document.project == before

    def test_a_hand_written_title_with_the_same_text_is_left_alone(
        self, video_media: MediaItem
    ) -> None:
        # 前は本文の一致だけで焼き込みを探し、手で書いたタイトルがたまたま直す前の字幕と
        # 同じ本文だと書き換えた（PR #231 の指摘） 印の無いテキストは直さず数だけ返す
        host = _host(video_media)
        project = host.document.project
        transcript = project.media[0].transcript
        assert transcript is not None
        same = transcript.segments[10].text.strip()
        title = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource("text", {"text": same}),
        )
        titles = Track(TrackKind.VIDEO, "タイトル", (title,))
        host = FakeHost(
            project.with_timeline(
                replace(project.timeline, tracks=(*project.timeline.tracks, titles))
            )
        )
        result = _call(host, "replace_subtitle_text", pairs=[{"from": "誤字", "to": "正字"}])
        assert result["replaced"] == 3
        assert result["burned_text_clips"] == 0
        assert result["unmarked_text_clips"] == 1
        located = host.document.project.timeline.locate_clip(title.id)
        assert located is not None
        kept = located[1]
        assert kept.source is not None
        assert kept.source.params["text"] == same

    def test_only_the_chosen_voice_moves_the_burned_text(self, video_media: MediaItem) -> None:
        # 焼き込みは出どころの字幕の行で見分ける 別の素材の同じ本文の焼き込みは動かない
        host = _host(video_media)
        _call(host, "place_subtitles")
        burned = [
            c
            for t in host.document.project.timeline.tracks
            for c in t.clips
            if c.source is not None
        ]
        assert burned and all(c.subtitle_origin is not None for c in burned)
        media = host.document.project.media[0]
        origin = burned[10].subtitle_origin
        assert origin is not None
        stranger = replace(
            burned[10],
            id=type(burned[10].id)("よその焼き込み"),
            timeline_start=burned[-1].timeline_end + 30,
            subtitle_origin=replace(origin, media_id=type(media.id)("よその素材")),
        )
        project = host.document.project
        extra = Track(TrackKind.VIDEO, "よそ", (stranger,))
        host = FakeHost(
            project.with_timeline(
                replace(project.timeline, tracks=(*project.timeline.tracks, extra))
            )
        )
        result = _call(host, "replace_subtitle_text", pairs=[{"from": "誤字", "to": "正字"}])
        assert result["burned_text_clips"] == 3
        located = host.document.project.timeline.locate_clip(stranger.id)
        assert located is not None
        kept = located[1]
        assert kept.source is not None
        assert "誤字" in str(kept.source.params["text"])

    def test_the_mark_survives_saving(self, video_media: MediaItem) -> None:
        # 保存して開き直すと印が消えると、次に誤植を直したとき焼き込みが直らない
        host = _host(video_media)
        _call(host, "place_subtitles")
        loaded = project_from_dict(project_to_dict(host.document.project))
        origins = [
            c.subtitle_origin
            for t in loaded.timeline.tracks
            for c in t.clips
            if c.source is not None
        ]
        assert origins and all(o is not None for o in origins)
        result = _call(
            FakeHost(loaded), "replace_subtitle_text", pairs=[{"from": "誤字", "to": "正字"}]
        )
        assert result["burned_text_clips"] == 3

    def test_bad_pairs_are_refused(self, video_media: MediaItem) -> None:
        with pytest.raises(ToolError, match="pairs"):
            _call(_host(video_media), "replace_subtitle_text", pairs=[{"from": "", "to": "x"}])


class TestOtherLists:
    def test_many_clips_are_paged(self, video_media: MediaItem) -> None:
        host = _host(video_media)
        project = host.document.project
        media = project.media[0]
        many = tuple(
            Clip(timeline_start=n * 10, duration=10, media_id=media.id) for n in range(400)
        )
        track = Track(TrackKind.VIDEO, "V2", many)
        host = FakeHost(
            project.with_timeline(
                replace(project.timeline, tracks=(*project.timeline.tracks, track))
            )
        )
        seen = 0
        offset = 0
        for _ in range(100):
            page = _call(host, "list_clips", offset=offset, limit=300)
            _short(page)
            seen += page["count"]
            if "next_offset" not in page:
                break
            offset = page["next_offset"]
        assert seen == 401
        # 1 秒ちょうど（30 コマ目）から始まる 4 本目は「1 秒より前」に入らない
        window = _call(host, "list_clips", track_id=str(track.id), from_seconds=0, to_seconds=1)
        assert window["total"] == 3

    def test_the_effect_list_is_short_unless_one_is_asked(self) -> None:
        host = FakeHost(Project.create())
        listed = _call(host, "list_effects", limit=300)
        _short(listed)
        assert "parameters" not in listed["effects"][0]
        blur = _call(host, "list_effects", kind="blur")["effects"][0]
        assert any(p["name"] == "radius" for p in blur["parameters"])

    def test_the_media_list_is_paged(self, video_media: MediaItem) -> None:
        many = tuple(replace(video_media, id=type(video_media.id)(f"m{n}")) for n in range(300))
        host = FakeHost(Project.create(media=many))
        page = _call(host, "list_media", limit=300)
        _short(page)
        assert page["total"] == 300
