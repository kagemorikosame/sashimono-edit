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
        window = _call(host, "list_clips", track_id=str(track.id), from_seconds=0, to_seconds=1)
        assert window["total"] == 4

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
