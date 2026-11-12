"""言いよどみを探して詰める（#414 ``core/fillers.py`` と AI の ``cut_fillers``）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.fillers import filler_key, filler_spans
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    Word,
    new_clip_id,
)
from tests.ai.conftest import FakeHost, make_loaded

F = Fraction


def _words(*items: tuple[str, float, float]) -> tuple[Word, ...]:
    return tuple(
        Word(F(start).limit_denominator(1000), F(end).limit_denominator(1000), text)
        for text, start, end in items
    )


def _line(start: float, end: float, *items: tuple[str, float, float]) -> TranscriptSegment:
    words = _words(*items)
    return TranscriptSegment(
        F(start).limit_denominator(1000),
        F(end).limit_denominator(1000),
        "".join(word.text for word in words),
        words=words,
    )


#: 1 字ずつの語（Whisper の日本語の形）で「えーと今日は」と「あのー編集を」
SPOKEN = Transcript(
    segments=(
        _line(
            1.0,
            3.0,
            ("え", 1.0, 1.2),
            ("ー", 1.2, 1.4),
            ("と", 1.4, 1.6),
            ("今", 1.7, 1.9),
            ("日", 1.9, 2.1),
            ("は", 2.1, 2.4),
        ),
        _line(
            4.0,
            6.0,
            ("あ", 4.0, 4.2),
            ("の", 4.2, 4.4),
            ("ー", 4.4, 4.7),
            ("編", 4.8, 5.0),
            ("集", 5.0, 5.2),
            ("を", 5.2, 5.5),
        ),
    ),
    language="ja",
)


def _with_sound(project: Project) -> Project:
    """映像のクリップと同じ所に、音声トラックの音のクリップを足す

    make_loaded は映像トラックにだけ置き、映像トラックは音を鳴らさない 言いよどみは鳴らす
    クリップが無いと断るので、話している作品の形にする
    """
    (video,) = project.timeline.tracks
    sound = Track(
        kind=TrackKind.AUDIO,
        name="A1",
        clips=tuple(replace(c, id=new_clip_id()) for c in video.clips),
    )
    return project.with_timeline(replace(project.timeline, tracks=(video, sound)))


def loaded_with_sound(video_media: MediaItem, transcript: Transcript = SPOKEN) -> Project:
    """言いよどみの語の時刻を持つ字幕の素材を、映像と音の 2 本で 10 秒置いた作品"""
    return _with_sound(make_loaded(video_media, transcript))


def filler_host(video_media: MediaItem) -> FakeHost:
    """言いよどみの語の時刻を持つ字幕の素材を、10 秒置いた作品"""
    return FakeHost(loaded_with_sound(video_media))


def _run(host: FakeHost, **arguments: Any) -> Any:
    operation = find_operation("cut_fillers")
    assert operation is not None
    return operation(host, arguments)


class TestFindingFillers:
    def test_a_filler_split_into_letters_is_found_whole(self) -> None:
        """Whisper の日本語の語は 1 字ほど 束ねて比べないと「えーと」を 1 つも見つけない"""
        spans = filler_spans(SPOKEN)
        assert [(span.text, span.start, span.end) for span in spans] == [
            ("えーと", F(1), F(8, 5)),
            ("あのー", F(4), F(47, 10)),
        ]

    def test_the_longer_filler_wins(self) -> None:
        """「えー」を先に取ると「えーと」の「と」が音に残る"""
        line = _line(1.0, 2.0, ("えー", 1.0, 1.4), ("と", 1.4, 1.6), ("本", 1.7, 2.0))
        spans = filler_spans(Transcript((line,), language="ja"))
        assert [span.text for span in spans] == ["えーと"]

    def test_meaningful_words_are_left(self) -> None:
        """「あの人」の「あの」は言いよどみではない 伸ばす記号の無い形まで切ると話が欠ける"""
        line = _line(1.0, 2.0, ("あ", 1.0, 1.2), ("の", 1.2, 1.4), ("人", 1.4, 1.8))
        assert filler_spans(Transcript((line,), language="ja")) == []

    def test_a_kanji_one_after_ano_is_kept(self) -> None:
        """「あの一つ」の「一」は数 伸ばす記号と読み違えた形まで既定で切ると、話の言葉が欠ける"""
        line = _line(1.0, 2.0, ("あの", 1.0, 1.3), ("一", 1.3, 1.5), ("つ", 1.5, 1.7))
        assert filler_spans(Transcript((line,), language="ja")) == []

    def test_a_surprised_voice_is_kept_by_default(self) -> None:
        """驚きの「えっ」は反応の声 字幕の整形とは違い、音からは既定で切らない"""
        line = _line(1.0, 2.0, ("えっ", 1.0, 1.3), ("本", 1.4, 1.8))
        assert filler_spans(Transcript((line,), language="ja")) == []

    def test_written_variants_are_the_same_filler(self) -> None:
        """認識はカタカナや伸ばす記号の重ねでも書く 揃えないと見逃す"""
        assert filler_key("エーート") == filler_key("えーと")
        assert filler_key("えー、") == "えー"

    def test_a_too_short_match_is_not_cut(self) -> None:
        """語の時刻の誤り（0.06 秒ほど）と見分けが付かない長さは返さない"""
        line = _line(1.0, 2.0, ("えー", 1.0, 1.05), ("本", 1.1, 1.8))
        assert filler_spans(Transcript((line,), language="ja")) == []


class TestTheTool:
    def test_the_fillers_are_cut_and_closed_up(self, video_media: MediaItem) -> None:
        """壊れると、言いよどみが音に残るか、切った所に隙間が残る"""
        host = filler_host(video_media)
        before = host.document.project.duration
        result = _run(host, media_id=str(host.document.project.media[0].id))
        assert result["found"] == 2
        assert result["cuts"] == 2
        assert host.document.project.duration == before - result["removed_frames"]
        # 0.6 秒と 0.7 秒 30fps で 18 と 21 フレーム
        assert result["removed_frames"] == 39
        # 取り消し 1 回で切る前へ戻る
        host.document.undo()
        assert host.document.project.duration == before

    def test_the_words_are_not_returned(self, video_media: MediaItem) -> None:
        """本人の字幕の文を、字幕と文字の同意の外で AI へ渡さない"""
        host = filler_host(video_media)
        result = _run(host, media_id=str(host.document.project.media[0].id))
        assert "えーと" not in str(result)
        assert "あのー" not in str(result)

    def test_only_the_given_words(self, video_media: MediaItem) -> None:
        """壊れると、頼んだ語だけを切るつもりで既定の言いよどみまで消える"""
        host = filler_host(video_media)
        result = _run(
            host, media_id=str(host.document.project.media[0].id), words=["あのー"], only_words=True
        )
        assert result["found"] == 1

    def test_a_transcript_without_word_times_is_refused(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """語の時刻が無いまま 0 件と返すと、言いよどみは無いと読む"""
        host = FakeHost(loaded_with_sound(video_media, transcript))
        with pytest.raises(ToolError, match="words_only"):
            _run(host, media_id=str(host.document.project.media[0].id))

    def test_nothing_found_says_why(self, video_media: MediaItem) -> None:
        """書かなかった言いよどみは探せないと言わないと、AI は無かったと伝える"""
        line = _line(1.0, 2.0, ("本", 1.0, 1.4), ("日", 1.4, 1.8))
        host = FakeHost(loaded_with_sound(video_media, Transcript((line,), language="ja")))
        with pytest.raises(ToolError, match="書かずに返す"):
            _run(host, media_id=str(host.document.project.media[0].id))


class TestReviewFixes:
    """PR #451 の CodeRabbit の指摘"""

    def test_a_small_vowel_is_a_long_mark(self) -> None:
        """「えぇと」は「えーと」と同じ音 揃えないと、この書き方の言いよどみが音に残る"""
        assert filler_key("えぇと") == filler_key("えーと")
        line = _line(1.0, 2.0, ("え", 1.0, 1.2), ("ぇ", 1.2, 1.4), ("と", 1.4, 1.6))
        assert [s.text for s in filler_spans(Transcript((line,), language="ja"))] == ["えぇと"]

    def test_a_small_vowel_making_another_sound_is_kept(self) -> None:
        """「ウィーン」の「ぃ」は伸ばす音ではない 「うーん」と読むと地名の語を切る"""
        assert filler_key("ウィーン") != filler_key("うーん")
        line = _line(1.0, 2.0, ("ウィーン", 1.0, 1.6))
        assert filler_spans(Transcript((line,), language="ja")) == []

    def test_long_marks_split_into_words_are_joined(self) -> None:
        """「え」「ー」「ー」「と」と分かれても全体を切る

        揃えずに束ねると「えー」だけを切って後半の音が残る
        """
        line = _line(
            1.0, 2.0, ("え", 1.0, 1.2), ("ー", 1.2, 1.3), ("ー", 1.3, 1.4), ("と", 1.4, 1.6)
        )
        spans = filler_spans(Transcript((line,), language="ja"))
        assert [(s.start, s.end) for s in spans] == [(F(1), F(8, 5))]

    def test_far_apart_words_are_not_joined(self) -> None:
        """離れた「え」と「ーと」を束ねると、間の別の声まで切る"""
        line = _line(1.0, 3.0, ("え", 1.0, 1.2), ("今", 1.3, 2.4), ("ーと", 2.5, 2.8))
        assert filler_spans(Transcript((line,), language="ja")) == []
        apart = _line(1.0, 3.0, ("え", 1.0, 1.2), ("ーと", 2.5, 2.8))
        assert filler_spans(Transcript((apart,), language="ja")) == []

    def test_a_gap_of_exactly_the_phrase_gap_is_a_break(self) -> None:
        """語の区切りと同じ境目 ちょうど 0.12 秒の間の音まで切らない"""
        line = _line(1.0, 2.0, ("え", 1.0, 1.2), ("ーと", 1.32, 1.6))
        assert filler_spans(Transcript((line,), language="ja")) == []

    def test_only_the_given_word_is_cut_from_the_timeline(self, video_media: MediaItem) -> None:
        """数だけ見ると、頼んでいない「えーと」まで切っても通る 切った所で確かめる"""
        host = filler_host(video_media)
        result = _run(
            host, media_id=str(host.document.project.media[0].id), words=["あのー"], only_words=True
        )
        # 「あのー」は 4.0〜4.7 秒（30fps で 120〜141） 「えーと」（30〜48）は残る
        assert [(p["start_frame"], p["end_frame"]) for p in result["places"]] == [(120, 141)]
        assert result["removed_frames"] == 21
        # 計画だけでなく、当てた後の作品からも「あのー」だけが抜けたのを見る
        project = host.document.project
        for track in project.timeline.tracks:
            kept = [(clip.source_in, clip.source_out(project.rate)) for clip in track.clips]
            assert kept == [(F(0), F(4)), (F(47, 10), F(10))]

    def test_a_locked_track_is_reported(self, video_media: MediaItem) -> None:
        """声だけが詰まってロックした字幕が後ろでずれる 黙ると同期が崩れたまま進む"""
        project = loaded_with_sound(video_media)
        text = Clip(0, 300, source=GeneratedSource(kind="text", params={"text": "題"}))
        locked = Track(kind=TrackKind.VIDEO, name="文字", clips=(text,), locked=True)
        project = project.with_timeline(
            replace(project.timeline, tracks=(*project.timeline.tracks, locked))
        )
        host = FakeHost(project)
        result = _run(host, media_id=str(project.media[0].id))
        assert result["tracks_left"] == [str(locked.id)]
        assert "tracks_left" in result["note"]

    def test_a_locked_voice_is_refused_without_cutting(self, video_media: MediaItem) -> None:
        """声を鳴らすトラックがロック

        壊れると、声は残ったまま映像と字幕だけが詰まり、切った量も大きく返す
        """
        project = loaded_with_sound(video_media)
        tracks = tuple(replace(track, locked=True) for track in project.timeline.tracks)
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        host = FakeHost(project)
        with pytest.raises(ToolError, match="どれもロックしたトラック"):
            _run(host, media_id=str(project.media[0].id))
        assert host.document.project == project

    def test_a_voice_not_on_the_timeline_is_refused(self, video_media: MediaItem) -> None:
        """字幕の音声 2 を鳴らすクリップが無い 壊れると、音声 2 の時刻で音声 1 の話を切る"""
        second = AudioStreamInfo(
            index=2, sample_rate=48000, channels=2, time_base=F(1, 48000), codec="aac"
        )
        two = replace(video_media, audio_streams=(*video_media.audio_streams, second))
        project = loaded_with_sound(two, Transcript(segments=(), language="ja"))
        media = project.media[0].with_transcript(SPOKEN, stream=2)
        project = replace(project, media=(media,))
        host = FakeHost(project)
        with pytest.raises(ToolError, match="鳴らすクリップ"):
            _run(host, media_id=str(media.id), audio=2)
        assert host.document.project == project

    def test_a_picture_left_without_its_sound_is_refused(self, video_media: MediaItem) -> None:
        """音のクリップを消して映像だけが残った作品

        壊れると、鳴らさない映像を 1 本目の音声のクリップと読み、どの音も鳴っていない映像を切る
        """
        project = make_loaded(video_media, SPOKEN)
        host = FakeHost(project)
        with pytest.raises(ToolError, match="鳴らすクリップ"):
            _run(host, media_id=str(project.media[0].id))
        assert host.document.project == project
