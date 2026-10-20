"""AI の話者の道具（#323） 分ける・名前を変える・行を付け替える・読む・話者ごとに置く

どれも人の操作（字幕パネルの〔話者を分ける…〕・右クリック・話者の列）と同じ命令を通る
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.core.model import MediaItem, Project, Transcript, TranscriptSegment
from tests.ai.conftest import FakeHost, make_loaded


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


def _host(video_media: MediaItem, speakers: list[str | None]) -> FakeHost:
    texts = ["今日は", "編集ソフトを", "作ります"]
    transcript = Transcript(
        tuple(
            TranscriptSegment(Fraction(1 + 3 * n), Fraction(3 + 3 * n), text, speaker=who)
            for n, (text, who) in enumerate(zip(texts, speakers, strict=True))
        )
    )
    return FakeHost(make_loaded(video_media, transcript))


def _speakers(host: FakeHost) -> list[str | None]:
    transcript = host.document.project.media[0].transcript
    assert transcript is not None
    return [s.speaker for s in transcript]


class TestSeparating:
    def test_the_request_reaches_the_queue_with_the_count(self, video_media: MediaItem) -> None:
        """壊れると、AI が 3 人と頼んでも人数が渡らず、自動で 2 人にまとめられる"""
        host = _host(video_media, [None, None, None])
        media = host.document.project.media[0]
        result = _operation("separate_speakers")(host, {"media_id": str(media.id), "speakers": 3})
        assert host.separated == (media.id, None, 3)
        assert isinstance(result, dict) and "transcription_status" in str(result["next"])

    def test_no_subtitles_is_refused_with_the_way_forward(self, video_media: MediaItem) -> None:
        """壊れると、字幕の無い素材で分け始め、数分待った末に付ける字幕が無い"""
        host = FakeHost(Project.create(media=(video_media,)))
        media = host.document.project.media[0]
        with pytest.raises(ToolError, match="transcribe"):
            _operation("separate_speakers")(host, {"media_id": str(media.id)})
        assert host.separated is None

    def test_existing_speakers_need_replace(self, video_media: MediaItem) -> None:
        """壊れると、人が付け替えた名前と直しを、AI が黙って分け直して消す"""
        host = _host(video_media, ["山田", "佐藤", "山田"])
        media_id = str(host.document.project.media[0].id)
        with pytest.raises(ToolError, match="replace"):
            _operation("separate_speakers")(host, {"media_id": media_id})
        _operation("separate_speakers")(host, {"media_id": media_id, "replace": True})
        assert host.separated is not None

    @pytest.mark.parametrize("count", [1, 11, "多い"])
    def test_a_strange_count_is_refused(self, video_media: MediaItem, count: Any) -> None:
        """壊れると、1 人や 11 人を部品へ渡し、意味の無い分け方で字幕の話者を上書きする"""
        host = _host(video_media, [None, None, None])
        media_id = str(host.document.project.media[0].id)
        with pytest.raises(ToolError, match="speakers"):
            _operation("separate_speakers")(host, {"media_id": media_id, "speakers": count})

    def test_a_missing_part_is_passed_on_as_the_reason(self, video_media: MediaItem) -> None:
        """壊れると、部品の無い人の所で AI が理由を受け取れず、何度も頼み直す"""
        host = _host(video_media, [None, None, None])
        host.speakers_refusal = "話者の分離の部品が入っていません"
        media_id = str(host.document.project.media[0].id)
        with pytest.raises(ToolError, match="部品"):
            _operation("separate_speakers")(host, {"media_id": media_id})

    def test_the_sound_is_not_declared_as_sent(self) -> None:
        """壊れると、音を外へ送る道具に見え（または本当に送り）、同意の区分が実物とずれる"""
        operation = _operation("separate_speakers")
        assert operation.sends == frozenset({Outbound.PROJECT, Outbound.FILES})
        assert operation.writes and not operation.irreversible


class TestFixingSpeakers:
    def test_rename_changes_all_rows_and_undo_restores(self, video_media: MediaItem) -> None:
        """壊れると、AI が名前を付けても一部の行が「話者 1」のまま残るか、取り消しで戻らない"""
        host = _host(video_media, ["話者 1", "話者 2", "話者 1"])
        media_id = str(host.document.project.media[0].id)
        _operation("rename_speaker")(host, {"media_id": media_id, "from": "話者 1", "to": "山田"})
        assert _speakers(host) == ["山田", "話者 2", "山田"]
        host.document.undo()
        assert _speakers(host) == ["話者 1", "話者 2", "話者 1"]

    def test_rename_of_an_unknown_name_is_refused(self, video_media: MediaItem) -> None:
        """壊れると、打ち間違えた名前で頼んだのに成功と返り、AI が直したつもりで進む"""
        host = _host(video_media, ["話者 1", None, None])
        media_id = str(host.document.project.media[0].id)
        with pytest.raises(ToolError, match="get_subtitles"):
            _operation("rename_speaker")(host, {"media_id": media_id, "from": "話者 9", "to": "a"})

    def test_rows_can_be_given_another_speaker(self, video_media: MediaItem) -> None:
        """壊れると、分離が取り違えた行を AI が直せず、焼き込みで別の人の色になる"""
        host = _host(video_media, ["話者 1", "話者 1", "話者 1"])
        media = host.document.project.media[0]
        assert media.transcript is not None
        second = str(media.transcript.segments[1].id)
        _operation("set_subtitle_speaker")(
            host, {"media_id": str(media.id), "segment_ids": [second], "speaker": "話者 2"}
        )
        assert _speakers(host) == ["話者 1", "話者 2", "話者 1"]
        with pytest.raises(ToolError):
            _operation("set_subtitle_speaker")(
                host, {"media_id": str(media.id), "segment_ids": ["無い"], "speaker": "話者 2"}
            )


class TestReadingAndPlacing:
    def test_rows_carry_the_speaker_and_can_be_filtered(self, video_media: MediaItem) -> None:
        """壊れると、AI が誰の字幕かを読めず、話者ごとの見た目や要約ができない"""
        host = _host(video_media, ["話者 1", "話者 2", None])
        result = _operation("get_subtitles")(host, {"speaker": "話者 2"})
        assert isinstance(result, dict)
        rows = result["subtitles"]
        assert [(r["text"], r["speaker"]) for r in rows] == [("編集ソフトを", "話者 2")]
        compact = _operation("get_subtitles")(host, {"compact": True})
        assert isinstance(compact, dict)
        assert [r.get("speaker") for r in compact["subtitles"]] == ["話者 1", "話者 2", None]

    def test_place_by_speaker_puts_each_on_its_own_layer(self, video_media: MediaItem) -> None:
        """壊れると、話者で絞っても全員の字幕が置かれるか、話者ごとのレイヤーに分かれない"""
        host = _host(video_media, ["話者 1", "話者 2", "話者 1"])
        result = _operation("place_subtitles")(host, {"speaker": "話者 1"})
        assert isinstance(result, dict) and result["placed"] == 2
        everyone = _host(video_media, ["話者 1", "話者 2", "話者 1"])
        result = _operation("place_subtitles")(everyone, {})
        assert isinstance(result, dict)
        assert sorted(result["tracks"]) == ["字幕 話者 1", "字幕 話者 2"]
        with pytest.raises(ToolError, match="話者 3"):
            _operation("place_subtitles")(everyone, {"speaker": "話者 3"})
