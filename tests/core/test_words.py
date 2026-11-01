"""語の時刻を読む（#420）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.model import (
    Clip,
    MediaItem,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    Word,
)
from sashimono.core.projection import heard_pieces
from sashimono.core.words import (
    CUT_IN,
    CUT_OUT,
    WordMap,
    attach_words,
    cut_edges,
    heard_between,
    heard_words,
    phrases,
    transcript_words,
    unheard_words,
    untimed_sounds,
)
from tests.words_helpers import RATE, in_scene, placed, sample_transcript, segment, spoken


def _clip(
    media: MediaItem, start: int, duration: int, source_in: Fraction, **fields: object
) -> Clip:
    return Clip(start, duration, media_id=media.id, source_in=source_in, **fields)  # type: ignore[arg-type]


def _texts(transcript: Transcript) -> list[str]:
    return [word.text for word in phrases(transcript)]


class TestPhrases:
    """Whisper の語は日本語では 1 字ずつで、そのままでは語の境目にならない"""

    def test_the_letters_of_a_word_are_one_phrase(self) -> None:
        """壊れると「カ|メ|ラ」の間を語の境目として勧め、言葉の途中で切る"""
        assert _texts(sample_transcript()) == ["カメラを", "作ります", "はい"]

    def test_a_phrase_keeps_the_time_of_its_letters(self) -> None:
        # 壊れると、束ねた語の時刻が字の時刻とずれ、語の端を読み違える
        first, second, _ = phrases(sample_transcript())
        assert (first.start, first.end) == (Fraction(1), Fraction(9, 5))
        assert (second.start, second.end) == (Fraction(9, 5), Fraction(13, 5))

    def test_a_pause_splits_even_the_same_letters(self) -> None:
        # 字の種類が同じでも、間があれば話は切れている
        head = spoken(Fraction(1), "あい")
        tail = spoken(Fraction(2), "うえ")
        assert _texts(Transcript((segment((*head, *tail)),), language="ja")) == ["あい", "うえ"]

    def test_a_subtitle_line_always_splits(self) -> None:
        # Whisper の行の切れ目は文の切れ目 字の種類より確か
        lines = (segment(spoken(Fraction(1), "です")), segment(spoken(Fraction(2), "よね")))
        assert _texts(Transcript(lines, language="ja")) == ["です", "よね"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # 「お」は後ろの語の頭 前の語の後ろに付けると「お|茶」の間で切れる
            ("それでお茶を", ["それで", "お茶を"]),
            ("お茶を", ["お茶を"]),
            # 文の終わりの形の後はひらがなが続いても区切る
            ("そうですはい", ["そうです", "はい"]),
            # 文の終わりの形に助詞が付く所は区切らない
            ("そうですか", ["そうですか"]),
            # 助詞の後ろは文の切れ目 束ねるとその間で切った所まで語の途中と読む
            ("そうですかはい", ["そうですか", "はい"]),
            ("そうですよね", ["そうですよね"]),
        ],
    )
    def test_kana_runs(self, text: str, expected: list[str]) -> None:
        # 壊れると、ひらがなが続く所で文の切れ目まで語の途中と読む
        words = spoken(Fraction(1), text)
        assert _texts(Transcript((segment(words),), language="ja")) == expected

    def test_a_mark_ends_the_phrase(self) -> None:
        # 壊れると、句読点のある切れ目まで語の途中と読む
        words = (*spoken(Fraction(1), "はい、"), *spoken(Fraction(16, 10), "はい"))
        assert _texts(Transcript((segment(words),), language="ja")) == ["はい、", "はい"]

    def test_other_languages_are_left_as_they_are(self) -> None:
        """壊れると英語の文が 1 語になり、文の途中がすべて語の途中と読まれる"""
        words = (
            Word(Fraction(1), Fraction(2), "hello"),
            Word(Fraction(2), Fraction(3), "world"),
        )
        assert _texts(Transcript((segment(words),), language="en")) == ["hello", "world"]

    def test_no_word_times(self) -> None:
        # 壊れると、語の時刻の無い作品で読むだけの道具が落ちる
        assert phrases(None) == ()


class TestWordAt:
    def test_inside_a_phrase(self) -> None:
        # 壊れると、語の途中で切った所を見逃す
        words = WordMap(sample_transcript())
        found = words.word_at(Fraction(14, 10))
        assert found is not None and found.text == "カメラを"

    @pytest.mark.parametrize("seconds", [Fraction(1), Fraction(105, 100), Fraction(177, 100)])
    def test_the_edges_are_not_inside(self, seconds: Fraction) -> None:
        """語の時刻は見積もり 壊れると語の境目で切った所まで言葉の途中と読む"""
        assert WordMap(sample_transcript()).word_at(seconds) is None

    def test_a_pause_is_not_inside(self) -> None:
        # 壊れると、話の間を語の途中と読み、安全な切れ目を問題にする
        assert WordMap(sample_transcript()).word_at(Fraction(3)) is None

    def test_a_long_word_behind_a_short_one_is_found(self) -> None:
        # 壊れると、重なる短い語の陰で長い語を見逃し、語の途中の切れ目を調べ済みとして返す
        lines = (segment(spoken(Fraction(1), "あ" * 10)), segment(spoken(Fraction(15, 10), "い")))
        found = WordMap(Transcript(lines, language="ja")).word_at(Fraction(2))
        assert found is not None and found.text == "あ" * 10


class TestAttachWords:
    """語の時刻のために起こし直すと、人が直した字幕が消える"""

    def test_the_text_and_the_speaker_stay(self) -> None:
        # 壊れると、語の時刻を足しただけで人が直した文言と話者が消える
        edited = segment(spoken(Fraction(1), "カメラを作ります"), speaker="話者 1", edited=True)
        before = Transcript((edited.with_text("カメラを作る"),), language="ja", model="old")
        attached = attach_words(before, sample_transcript())
        (line,) = attached.segments
        assert line.text == "カメラを作る"
        assert line.speaker == "話者 1"
        assert line.edited
        assert line.id == edited.id
        assert attached.model == "old"
        assert "".join(word.text for word in line.words) == "カメラを作ります"

    def test_a_word_goes_to_the_line_it_overlaps_most(self) -> None:
        # 壊れると、語が隣の字幕の行に入り、残した語の読みが外れる
        first = segment(spoken(Fraction(1), "あ"))
        second = segment(spoken(Fraction(12, 10), "い"))
        word = Word(Fraction(11, 10), Fraction(135, 100), "う")
        recognized = Transcript((segment((word,)),))
        attached = attach_words(Transcript((first, second)), recognized)
        assert [len(line.words) for line in attached.segments] == [0, 1]

    def test_no_recognized_words_keep_what_is_there(self) -> None:
        # 壊れると、語の時刻を足す頼みで持っていた語の時刻まで消える
        line = segment(spoken(Fraction(1), "あ"))
        before = Transcript((line,))
        assert attach_words(before, Transcript()) == before

    def test_a_long_line_behind_a_short_one_gets_its_word(self) -> None:
        # 壊れると、重なる短い行の陰で長い行の語を捨て、words_only で持っていた語の時刻まで消える
        long = TranscriptSegment(Fraction(0), Fraction(10), "長い行")
        short = TranscriptSegment(Fraction(2), Fraction(3), "短い行")
        recognized = Transcript((segment(spoken(Fraction(4), "あ", Fraction(1))),))
        attached = attach_words(Transcript((long, short)), recognized)
        assert [word.text for word in attached.segments[0].words] == ["あ"]
        assert attached.segments[1].words == ()

    def test_a_tie_goes_to_the_earlier_line(self) -> None:
        # 壊れると、同じだけ重なる語の入る行が変わり、起こし直すたびに字幕の語が動く
        first = TranscriptSegment(Fraction(1), Fraction(3), "前")
        second = TranscriptSegment(Fraction(1), Fraction(3), "後")
        recognized = Transcript((segment(spoken(Fraction(2), "あ", Fraction(1, 2))),))
        attached = attach_words(Transcript((first, second)), recognized)
        assert [len(line.words) for line in attached.segments] == [1, 0]

    def test_a_word_outside_every_line_is_dropped(self) -> None:
        # 消した行の語を残すと、人が消した言葉が字幕の語として返る
        line = segment(spoken(Fraction(1), "あ"))
        recognized = Transcript((segment(spoken(Fraction(5), "い")),))
        attached = attach_words(Transcript((line,)), recognized)
        assert attached.segments[0].words == ()


class TestHeardPieces:
    def test_speed_is_taken_back_to_the_source(self, video_media: MediaItem) -> None:
        # 壊れると、速度を変えたクリップの語を別の時刻で読む
        clip = _clip(video_media, 30, 30, Fraction(1), speed=Fraction(2))
        (piece,) = heard_pieces(placed(video_media, clip))
        assert (piece.source_start, piece.source_end) == (Fraction(1), Fraction(3))
        assert piece.seconds_at(45, RATE) == Fraction(2)

    def test_the_scene_window_cuts_the_inner_sound(self, video_media: MediaItem) -> None:
        """シーンの中のクリップの端でなく、置いたクリップの窓の端が語の途中に当たる形"""
        inner = placed(video_media, _clip(video_media, 0, 300, Fraction(0)))
        window = Clip(60, 30, source_in=Fraction(1), speed=Fraction(2))
        (piece,) = heard_pieces(in_scene(inner, window))
        # シーンの 30 コマ目（1 秒）から 2 倍速で 30 コマ
        assert (piece.source_start, piece.source_end) == (Fraction(1), Fraction(3))
        assert (piece.start, piece.end) == (Fraction(60), Fraction(90))
        assert piece.speed == 2

    def test_the_same_head_at_another_speed_stays(self, video_media: MediaItem) -> None:
        # 壊れると、速度だけ違う音の片方が消え、その語と切れ目を調べない
        slow = _clip(video_media, 0, 30, Fraction(1))
        fast = _clip(video_media, 0, 30, Fraction(1), speed=Fraction(2))
        project = placed(video_media, slow)
        other = Track(kind=TrackKind.AUDIO, name="A2", clips=(fast,))
        project = project.with_timeline(
            replace(project.timeline, tracks=(*project.timeline.tracks, other))
        )
        assert sorted(piece.speed for piece in heard_pieces(project)) == [1, 2]

    def test_a_muted_track_is_not_heard(self, video_media: MediaItem) -> None:
        # 壊れると、鳴らないクリップの切れ目を問題として返す
        clip = _clip(video_media, 0, 30, Fraction(1), enabled=False)
        assert heard_pieces(placed(video_media, clip)) == []


class TestCutEdges:
    def test_both_ends_of_a_clip(self, video_media: MediaItem) -> None:
        # 壊れると、クリップの入りか出のどちらかを調べない
        project = placed(video_media, _clip(video_media, 0, 30, Fraction(1)))
        edges = cut_edges(project, 0, 30)
        assert [(edge.side, edge.seconds) for edge in edges] == [
            (CUT_IN, Fraction(1)),
            (CUT_OUT, Fraction(2)),
        ]

    def test_a_split_is_not_a_cut(self, video_media: MediaItem) -> None:
        """壊れると、分割しただけのつなぎ目を言葉の途中の切れ目として返す"""
        project = placed(
            video_media,
            _clip(video_media, 0, 30, Fraction(1)),
            _clip(video_media, 30, 30, Fraction(2)),
        )
        assert [edge.seconds for edge in cut_edges(project, 0, 60)] == [Fraction(1), Fraction(3)]

    def test_a_one_frame_gap_is_a_cut(self, video_media: MediaItem) -> None:
        # 壊れると、1 コマ無音が出るつなぎ目を分割と読み、切れ目を見逃す
        project = placed(
            video_media,
            _clip(video_media, 0, 30, Fraction(1)),
            _clip(video_media, 31, 30, Fraction(2)),
        )
        assert len(cut_edges(project, 0, 61)) == 4

    def test_a_one_frame_skip_of_the_material_is_a_cut(self, video_media: MediaItem) -> None:
        # 壊れると、素材を 1 コマ飛ばして語が欠けたつなぎ目を分割と読み、切れ目を見逃す
        project = placed(
            video_media,
            _clip(video_media, 0, 30, Fraction(1)),
            _clip(video_media, 30, 30, Fraction(2) + Fraction(1, 30)),
        )
        assert len(cut_edges(project, 0, 60)) == 4

    def test_rounding_of_a_speed_change_is_not_a_cut(self, video_media: MediaItem) -> None:
        # 壊れると、速度を変えたクリップの長さの丸めの差で、ただの分割を切れ目として返す
        project = placed(
            video_media,
            _clip(video_media, 0, 30, Fraction(1), speed=Fraction(3, 2)),
            _clip(video_media, 30, 30, Fraction(5, 2) + Fraction(1, 60), speed=Fraction(3, 2)),
        )
        assert len(cut_edges(project, 0, 60)) == 2

    def test_a_jump_is_a_cut(self, video_media: MediaItem) -> None:
        # 壊れると、飛ばしてつないだ所を分割と読み、切れ目を見逃す
        project = placed(
            video_media,
            _clip(video_media, 0, 30, Fraction(1)),
            _clip(video_media, 30, 30, Fraction(4)),
        )
        assert len(cut_edges(project, 0, 60)) == 4

    def test_the_ends_of_the_material_are_not_cuts(self, video_media: MediaItem) -> None:
        # 語の途中で始まる録音は編集では直せない
        project = placed(video_media, _clip(video_media, 0, 300, Fraction(0)))
        assert cut_edges(project, 0, 300) == []

    def test_only_the_asked_range(self, video_media: MediaItem) -> None:
        # 壊れると、頼まれた範囲の外の切れ目まで返す
        project = placed(video_media, _clip(video_media, 30, 30, Fraction(1)))
        assert [edge.side for edge in cut_edges(project, 0, 40)] == [CUT_IN]
        assert [edge.side for edge in cut_edges(project, 40, 90)] == [CUT_OUT]


class TestWordMap:
    def _line(self) -> Transcript:
        words = spoken(Fraction(1), "カメラ")
        return Transcript((TranscriptSegment(Fraction(1), Fraction(4), "カメラと足した文", words),))

    def test_far_from_the_words(self) -> None:
        # 壊れると、語が行の一部にしか付かない行の語の無い所を、語の途中ではないと読む
        assert WordMap(self._line()).unworded_at(Fraction(3))

    def test_every_line_at_the_point_is_read(self) -> None:
        # 壊れると、重なる字幕の行の後ろの行だけを見て、前の行の語の無い所を確かめたと読む
        words = spoken(Fraction(1), "カメラ")
        long = TranscriptSegment(Fraction(1), Fraction(5), "長い行", words)
        short = segment(spoken(Fraction(29, 10), "はい"))
        assert WordMap(Transcript((long, short))).unworded_at(Fraction(3))

    def test_outside_every_line_is_unworded(self) -> None:
        # 壊れると、字幕を消した所で声が続く切れ目を、語の途中ではないと読む
        lines = (segment(spoken(Fraction(1), "あ")), segment(spoken(Fraction(4), "い")))
        assert WordMap(Transcript(lines)).unworded_at(Fraction(3))

    def test_next_to_a_word_outside_the_line_is_worded(self) -> None:
        # 壊れると、語のすぐ後で切った所まで行の外として未検査になる
        lines = (segment(spoken(Fraction(1), "あ")),)
        assert not WordMap(Transcript(lines)).unworded_at(Fraction(13, 10))

    def test_a_line_without_words_is_a_unit(self) -> None:
        # 壊れると、語の時刻の無い行の切れ目を切り所として返せない
        line = TranscriptSegment(Fraction(1), Fraction(2), "行")
        words = WordMap(Transcript((line,)))
        assert [unit.text for unit in words.units] == ["行"]
        assert words.unworded_at(Fraction(3, 2))

    def test_near_the_words(self) -> None:
        # 壊れると、語の時刻のずれだけの所まで未検査になる
        assert not WordMap(self._line()).unworded_at(Fraction(17, 10))

    def test_no_subtitles(self) -> None:
        # 壊れると、字幕の無い音の読み取りで検査が落ちる
        assert WordMap(None).unworded_at(Fraction(1))


class TestUntimedSounds:
    def test_a_heard_line_without_words(self, video_media: MediaItem) -> None:
        """壊れると、語の無い行の所を調べたと読み、切れ目が無事と返る"""
        lines = (
            segment(spoken(Fraction(1), "カメラ")),
            TranscriptSegment(Fraction(5), Fraction(6), "手で足した行"),
        )
        media = video_media.with_transcript(Transcript(lines, language="ja"))
        heard = placed(media, _clip(media, 0, 30, Fraction(5)))
        assert untimed_sounds(heard, 0, 30) == {(media.id, media.transcript_stream())}

    def test_a_line_that_is_not_used_does_not_count(self, video_media: MediaItem) -> None:
        # 壊れると、置いていない所の行 1 本で作品全体が調べられない扱いになる
        lines = (
            segment(spoken(Fraction(1), "カメラ")),
            TranscriptSegment(Fraction(5), Fraction(6), "手で足した行"),
        )
        media = video_media.with_transcript(Transcript(lines, language="ja"))
        assert untimed_sounds(placed(media, _clip(media, 0, 30, Fraction(1))), 0, 30) == set()


class TestHeardWords:
    def _media(self, video_media: MediaItem) -> MediaItem:
        return video_media.with_transcript(sample_transcript())

    def test_only_the_kept_words(self, video_media: MediaItem) -> None:
        # 壊れると、切り落とした語を残っていると読む
        media = self._media(video_media)
        # 素材の 1.0〜1.8 秒だけ残す「カメラを」
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        assert "".join(item.word.text for item in heard_words(project)) == "カメラを"

    def test_a_word_half_cut_counts_by_its_middle(self, video_media: MediaItem) -> None:
        # 壊れると、端が掛かっただけの語を残っていると読む
        media = self._media(video_media)
        # 1.0〜1.5 秒 「ラ」（1.4〜1.6）の真ん中はちょうど切れ目で、鳴らない側
        project = placed(media, _clip(media, 0, 15, Fraction(1)))
        assert "".join(item.word.text for item in heard_words(project)) == "カメ"

    def test_inside_a_scene_with_speed(self, video_media: MediaItem) -> None:
        # 壊れると、シーンの中の語が速度の分だけずれる
        media = self._media(video_media)
        inner = placed(media, _clip(media, 0, 300, Fraction(0)))
        window = Clip(0, 15, source_in=Fraction(1), speed=Fraction(2))
        project = in_scene(inner, window)
        # シーンの 1〜2 秒（素材も同じ）を 2 倍速で 0.5 秒に置いた 「作」の真ん中は 1.9 秒
        assert "".join(item.word.text for item in heard_words(project)) == "カメラを作"
        assert heard_words(project, 15) == []

    def test_the_words_the_subtitle_loses(self, video_media: MediaItem) -> None:
        """0.2.3 の通しの確かめの形 字幕の文の後半が音に入っていない"""
        media = self._media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        heard = heard_words(project)
        line = sample_transcript().segments[0]
        stream = media.transcript_stream()
        missing = unheard_words(line, heard, media.id, stream, 0, 24)
        assert "".join(word.text for word in missing) == "作ります"
        assert len(heard_between(heard, media.id, stream, 0, 12)) == 2

    def test_no_word_times(self, video_media: MediaItem, transcript: Transcript) -> None:
        # 壊れると、語の時刻の無い作品で読むだけの道具が落ちる
        media = video_media.with_transcript(transcript)
        project = placed(media, _clip(media, 0, 60, Fraction(1)), kind=TrackKind.AUDIO)
        assert transcript_words(transcript) == ()
        assert heard_words(project) == []
