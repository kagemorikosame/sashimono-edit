"""話者の分離（#323）の結果を字幕へ当て、話者ごとに焼き込む

- 字幕 1 枚の途中で話し手が替わるときは、重なりのいちばん長い人を付ける（字幕は割らない）
- 名前は字幕に初めて出てきた順に「話者 1」「話者 2」
- 話者の付け替えと名前の変更は、取り消し 1 回で戻る
- 焼き込みは話者ごとに別のレイヤーへ入り、同時に話す所で前の字幕が切り詰められない
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import (
    Document,
    RenameSpeaker,
    SetSegmentSpeaker,
    Voice,
    burn_subtitles,
    subtitle_voices,
    voice_label,
)
from sashimono.core.model import MediaItem, Project, Transcript, TranscriptSegment
from sashimono.core.speakers import (
    SpeakerTurn,
    assign_speakers,
    speaker_name,
    transcript_speakers,
)
from sashimono.effects.sources import TEXT
from tests.ai.conftest import make_loaded


def _turn(start: float, end: float, who: int) -> SpeakerTurn:
    return SpeakerTurn(Fraction(start), Fraction(end), who)


def _segment(start: float, end: float, text: str, speaker: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(Fraction(start), Fraction(end), text, speaker=speaker)


class TestAssigning:
    def test_the_longest_overlap_wins_and_the_row_is_not_split(self) -> None:
        """壊れると、途中で相づちが入った字幕が相づちの人の物になるか、字幕が割れて枚数が変わる"""
        transcript = Transcript((_segment(0, 4, "長く話す"),))
        # 分離の番号 5 が 3 秒、番号 2 が 1 秒話している
        result = assign_speakers(transcript, [_turn(0, 3, 5), _turn(3, 4, 2)])
        assert [(s.text, s.speaker) for s in result] == [("長く話す", "話者 1")]
        assert result.segments[0].id == transcript.segments[0].id

    def test_names_follow_the_order_of_first_appearance(self) -> None:
        """壊れると、分離の番号がそのまま名前になり、最初に話した人が「話者 3」になる"""
        transcript = Transcript((_segment(0, 1, "一"), _segment(2, 3, "二"), _segment(4, 5, "三")))
        result = assign_speakers(transcript, [_turn(0, 1, 7), _turn(2, 3, 3), _turn(4, 5, 7)])
        assert [s.speaker for s in result] == ["話者 1", "話者 2", "話者 1"]
        assert transcript_speakers(result) == ["話者 1", "話者 2"]

    def test_a_short_reply_outside_the_turns_takes_the_nearest(self) -> None:
        """壊れると、分離が落とした短い相づちだけ話者なしになり、別のレイヤーへ 1 枚だけ出る"""
        transcript = Transcript((_segment(0, 2, "前"), _segment(2.4, 2.8, "うん")))
        result = assign_speakers(transcript, [_turn(0, 2, 0), _turn(3.2, 6, 1)])
        assert [s.speaker for s in result] == ["話者 1", "話者 1"]

    def test_far_from_any_turn_gets_no_speaker(self) -> None:
        """壊れると、無音を挟んだ遠くの字幕に誰かの名前が付き、前の回の名前も残る"""
        transcript = Transcript((_segment(10, 11, "遠い", speaker="前の名前"),))
        result = assign_speakers(transcript, [_turn(0, 2, 0)])
        assert result.segments[0].speaker is None

    def test_text_and_the_edited_mark_are_kept(self) -> None:
        """壊れると、話者を付けただけで人が直した字幕が「直していない」扱いになり、起こし直しで消える"""
        edited = replace(_segment(0, 1, "直した"), edited=True)
        result = assign_speakers(Transcript((edited,)), [_turn(0, 1, 0)])
        assert result.segments[0].edited and result.segments[0].text == "直した"
        assert speaker_name(2) == "話者 2"


def _loaded(video_media: MediaItem, speakers: list[str | None]) -> tuple[Project, MediaItem]:
    texts = ["今日は", "編集ソフトを", "作ります"]
    transcript = Transcript(
        tuple(
            _segment(1 + 3 * n, 3 + 3 * n, text, speaker)
            for n, (text, speaker) in enumerate(zip(texts, speakers, strict=True))
        )
    )
    project = make_loaded(video_media, transcript)
    return project, project.media[0]


class TestCommands:
    def test_renaming_changes_every_row_and_one_undo_restores(self, video_media: MediaItem) -> None:
        """壊れると、名前を付け替えても一部の行が「話者 1」のまま残るか、取り消しで戻らない"""
        project, media = _loaded(video_media, ["話者 1", "話者 2", "話者 1"])
        document = Document(project)
        document.execute(RenameSpeaker(media.id, "話者 1", "山田"))
        after = document.project.require_media(media.id).transcript
        assert after is not None
        assert [s.speaker for s in after] == ["山田", "話者 2", "山田"]
        document.undo()
        before = document.project.require_media(media.id).transcript
        assert before is not None and before.segments[0].speaker == "話者 1"

    def test_renaming_to_another_speaker_merges_them(self, video_media: MediaItem) -> None:
        """壊れると、分離が 1 人を 2 人に割ったときに 1 人へまとめる直し方が無い"""
        project, media = _loaded(video_media, ["話者 1", "話者 2", "話者 1"])
        renamed = RenameSpeaker(media.id, "話者 2", "話者 1").apply(project)
        transcript = renamed.require_media(media.id).transcript
        assert transcript_speakers(transcript) == ["話者 1"]

    def test_renaming_an_unknown_speaker_is_refused(self, video_media: MediaItem) -> None:
        """壊れると、打ち間違えた名前の付け替えが黙って何もせず、直したつもりで残る"""
        project, media = _loaded(video_media, ["話者 1", None, None])
        with pytest.raises(KeyError):
            RenameSpeaker(media.id, "話者 9", "山田").apply(project)
        with pytest.raises(ValueError, match="空"):
            RenameSpeaker(media.id, "話者 1", "  ").apply(project)

    def test_one_row_can_be_given_another_speaker_or_none(self, video_media: MediaItem) -> None:
        """壊れると、分離が取り違えた 1 行だけを直せず、全員の名前を変えることになる"""
        project, media = _loaded(video_media, ["話者 1", "話者 1", "話者 1"])
        transcript = media.transcript
        assert transcript is not None
        second = transcript.segments[1].id
        changed = SetSegmentSpeaker(media.id, (second,), " 話者 2 ").apply(project)
        rows = changed.require_media(media.id).transcript
        assert rows is not None and [s.speaker for s in rows] == ["話者 1", "話者 2", "話者 1"]
        cleared = SetSegmentSpeaker(media.id, (second,), "").apply(changed)
        rows = cleared.require_media(media.id).transcript
        assert rows is not None and rows.segments[1].speaker is None
        assert not rows.segments[1].edited


class TestBurningBySpeaker:
    def test_each_speaker_gets_a_layer_and_overlaps_survive(self, video_media: MediaItem) -> None:
        """壊れると、マイク 1 本の対談で 2 人が同時に話す所の前の字幕が切り詰められて欠ける"""
        transcript = Transcript(
            (
                _segment(1, 5, "長く話す", "話者 1"),
                _segment(2, 4, "かぶせる", "話者 2"),
            )
        )
        project = make_loaded(video_media, transcript)
        document = Document(project)
        for command in burn_subtitles(project, TEXT.create()):
            document.execute(command)
        tracks = {t.name: t for t in document.project.timeline.tracks}
        assert {"字幕 話者 1", "字幕 話者 2"} <= set(tracks)
        first = tracks["字幕 話者 1"].clips[0]
        # 切り詰められていなければ 4 秒（120 フレーム）のまま
        assert first.duration == 120

    def test_voices_are_ordered_by_first_appearance(self, video_media: MediaItem) -> None:
        """壊れると、焼き込みの窓に「話者 10」が「話者 2」より前に並び、選ぶ人が迷う"""
        project, media = _loaded(video_media, ["話者 2", "話者 10", None])
        voices = subtitle_voices(project)
        assert voices == [
            Voice(media.id, 1, None),
            Voice(media.id, 1, "話者 2"),
            Voice(media.id, 1, "話者 10"),
        ]
        assert voice_label(project, voices[1]) == f"{media.name} 話者 2"
        assert voice_label(project, voices[0], with_media=False) == media.name

    def test_unseparated_subtitles_burn_as_before(self, video_media: MediaItem) -> None:
        """壊れると、話者を分けていない人の焼き込みまでレイヤーの名前や分かれ方が変わる"""
        project, _media = _loaded(video_media, [None, None, None])
        commands = burn_subtitles(project, TEXT.create())
        document = Document(project)
        for command in commands:
            document.execute(command)
        names = [t.name for t in document.project.timeline.tracks if t.name.startswith("字幕")]
        assert names == ["字幕"]
