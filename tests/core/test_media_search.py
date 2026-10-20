"""素材の意味検索（#326 A8）の引き方 表記のゆれを揃えた部分一致と、場面の説明

字幕は作品の中の素材すべて（タイムラインに置いていない素材も）を引く 言い換えは AI が
語を増やして何度か引く（道具は部分一致のまま） 場面の説明は AI が作って覚えた文で、
字幕と同じ引き方で引く
"""

from __future__ import annotations

import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands.insert import media_alone
from sashimono.core.model import MediaItem, Project, ProjectSettings, Transcript
from sashimono.core.model.transcript import TranscriptSegment
from sashimono.core.search import (
    SCENE,
    SUBTITLE,
    SceneNote,
    fold,
    prepared_queries,
    scene_spans,
    search_media,
)
from sashimono.core.timebase import FrameRate


def _segment(start: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=Fraction(start), end=Fraction(start + 1), text=text)


def _with_lines(media: MediaItem, *lines: str, stream: int | None = None) -> MediaItem:
    transcript = Transcript(
        segments=tuple(_segment(index * 2, text) for index, text in enumerate(lines))
    )
    return media.with_transcript(transcript, stream)


def _project(*media: MediaItem) -> Project:
    return Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=media)


def _no_notes(_media: MediaItem) -> tuple[SceneNote, ...]:
    return ()


class TestFold:
    def test_katakana_and_hiragana_are_the_same(self) -> None:
        # 壊れると「らーめん」と頼まれて「ラーメン」と起こした字幕の場面を見つけられない
        assert fold("ラーメン") == fold("らーめん")

    def test_full_and_half_width_are_the_same(self) -> None:
        # 壊れると、全角の「ＯＢＳ」や半角カナの「ﾗｰﾒﾝ」で起こされた行を引けない
        assert fold("ＯＢＳ") == fold("obs")
        assert fold("ﾗｰﾒﾝ") == fold("らーめん")

    def test_spaces_do_not_split_a_word(self) -> None:
        # 壊れると、起こしが語の間に入れた空白（全角も）のせいで同じ語が引けない
        assert fold("ラー　メン 屋") == fold("らーめん屋")


class TestQueries:
    def test_an_empty_word_is_refused(self) -> None:
        # 壊れると、空白だけの語がどの行にも当たり、全部の字幕が返って何も絞れない
        with pytest.raises(ValueError, match="空"):
            prepared_queries(["ラーメン", "  "])

    def test_the_same_word_written_twice_is_kept_once(self) -> None:
        # 壊れると、言い換えで同じ語を 2 回渡したときに当たった語の数が水増しされ、並びが狂う
        assert len(prepared_queries(["ラーメン", "らーめん", "麺"])) == 2


class TestSearch:
    def test_media_off_the_timeline_are_searched(self, video_media: MediaItem) -> None:
        # 壊れると、タイムラインに置いていない素材の「あの場面」を AI も人も見つけられない
        # （get_subtitles はタイムラインに出ている字幕だけを見る）
        media = _with_lines(video_media, "今日はラーメンを食べに来ました")
        project = _project(media)
        assert project.duration == 0
        hits = search_media(project, ["らーめん"], notes_of=_no_notes)
        assert [(hit.kind, hit.media_id, hit.text) for hit in hits] == [
            (SUBTITLE, media.id, "今日はラーメンを食べに来ました")
        ]
        assert hits[0].segment_id == media.transcripts[0][1].segments[0].id

    def test_every_audio_stream_is_searched(self, video_media: MediaItem) -> None:
        # 壊れると、ゲームの音とマイクのように音声が何本もある素材で、2 本目の声の字幕を引けない
        two = replace(
            video_media,
            audio_streams=(
                video_media.audio_streams[0],
                replace(video_media.audio_streams[0], index=2),
            ),
        )
        media = _with_lines(_with_lines(two, "一本目"), "マイクの声で乾杯", stream=2)
        hits = search_media(_project(media), ["乾杯"], notes_of=_no_notes)
        assert [hit.stream for hit in hits] == [2]

    def test_more_matched_words_come_first(self, video_media: MediaItem) -> None:
        # 壊れると、言い換えを並べて引いたときに、多くの語が当たった本命の行が下に埋もれる
        media = _with_lines(video_media, "麺がうまい", "ラーメンの麺がうまい", "スープ")
        hits = search_media(_project(media), ["ラーメン", "麺"], notes_of=_no_notes)
        assert [hit.text for hit in hits] == ["ラーメンの麺がうまい", "麺がうまい"]
        assert hits[0].matched == ("ラーメン", "麺")

    def test_scene_notes_are_searched_like_subtitles(self, video_media: MediaItem) -> None:
        # 壊れると、AI が絵を見て書き留めた場面の説明（話していない場面）を引けない
        note = SceneNote(start=12.0, end=20.5, text="夜の屋台でラーメンをすする")

        def notes_of(media: MediaItem) -> tuple[SceneNote, ...]:
            return (note,) if media.id == video_media.id else ()

        hits = search_media(_project(video_media), ["らーめん"], notes_of=notes_of)
        assert [(hit.kind, hit.start, hit.end, hit.text) for hit in hits] == [
            (SCENE, 12.0, 20.5, note.text)
        ]

    def test_kinds_and_media_narrow_the_search(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 壊れると、字幕だけ・この素材だけと絞ったのに、ほかの物が混ざって AI が取り違える
        first = _with_lines(video_media, "ラーメン")
        second = _with_lines(audio_media, "ラーメン")
        note = SceneNote(start=0.0, end=1.0, text="ラーメン")
        project = _project(first, second)
        hits = search_media(
            project, ["ラーメン"], notes_of=lambda _m: (note,), media_id=second.id, kinds=(SCENE,)
        )
        assert [(hit.kind, hit.media_id) for hit in hits] == [(SCENE, second.id)]

    def test_notes_are_not_read_when_only_subtitles_are_wanted(
        self, video_media: MediaItem
    ) -> None:
        # 壊れると、字幕だけを引くときも素材の数だけ説明の控えを読みに行き、検索が遅くなる
        read: list[MediaItem] = []

        def notes_of(media: MediaItem) -> tuple[SceneNote, ...]:
            read.append(media)
            return ()

        search_media(_project(video_media), ["a"], notes_of=notes_of, kinds=(SUBTITLE,))
        assert read == []

    def test_hours_of_subtitles_answer_at_once(self) -> None:
        # 壊れると、数時間分の素材（数万行）を引くたびに編集画面が止まる
        # （道具は画面のスレッドで動く）
        media = [
            _with_lines(
                MediaItem(path=Path(f"C:/素材/{number}.mp4"), duration=Fraction(36000)),
                *(f"{number} 本目の {line} 行目 今日はいい天気ですね" for line in range(10000)),
            )
            for number in range(5)
        ]
        project = _project(*media)
        started = time.perf_counter()
        hits = search_media(project, ["ラーメン", "天気", "あめ"], notes_of=_no_notes)
        elapsed = time.perf_counter() - started
        assert len(hits) == 50000
        # 手元では 0.1 秒ほど CI の遅い機械と並べて走らせる分を見込む
        assert elapsed < 2.0, elapsed


class TestSceneSpans:
    def test_cuts_make_the_scenes(self) -> None:
        # 壊れると、AI が説明を付ける場面の区切りが切り替わりとずれ、説明が隣の場面に付く
        assert scene_spans([4.0, 6.0], 10.0, 30.0) == [(0.0, 4.0), (4.0, 6.0), (6.0, 10.0)]

    def test_a_long_scene_is_cut_into_equal_pieces(self) -> None:
        # 壊れると、話しっぱなしの 1 時間が 1 枚の絵の説明になり、途中の出来事を引けない
        assert scene_spans([], 10.0, 4.0) == [(0.0, 3.333), (3.333, 6.667), (6.667, 10.0)]

    def test_cuts_outside_the_media_are_ignored(self) -> None:
        # 壊れると、解析した所より後ろの切り替わりで長さの無い場面や逆さの場面ができる
        assert scene_spans([0.0, 12.0, 5.0], 10.0, 30.0) == [(0.0, 5.0), (5.0, 10.0)]
        assert scene_spans([1.0], 0.0, 30.0) == []


class TestMediaAlone:
    def test_the_media_seconds_are_the_frames(self, video_media: MediaItem) -> None:
        # 壊れると、場面の説明を作る 1 枚で、頼んだ素材の秒と違う所の絵を AI に見せる
        settings = ProjectSettings(width=640, height=360, frame_rate=FrameRate(30))
        alone = media_alone(settings, video_media)
        clips = [clip for track in alone.timeline.tracks for clip in track.clips]
        picture = next(clip for clip in clips if clip.native_size)
        assert picture.timeline_start == 0
        assert picture.source_in == Fraction(0)
        assert alone.duration == 300
        assert alone.media == (video_media,)

    def test_the_picture_fills_the_frame(self, video_media: MediaItem) -> None:
        # 壊れると、作品の大きさが素材と違うとき、素材の絵が黒い縁の中に小さく（大きく）出る
        settings = ProjectSettings(width=640, height=360, frame_rate=FrameRate(30))
        portrait = replace(
            video_media,
            video_streams=(replace(video_media.video_streams[0], rotation=90),),
        )
        alone = media_alone(settings, portrait)
        assert alone.settings.resolution == (1080, 1920)
        assert alone.rate == settings.frame_rate


class TestSceneNote:
    @pytest.mark.parametrize(
        ("start", "end", "text"),
        [(-1.0, 2.0, "a"), (3.0, 3.0, "a"), (0.0, float("nan"), "a"), (0.0, 1.0, " ")],
    )
    def test_a_broken_note_cannot_be_made(self, start: float, end: float, text: str) -> None:
        # 壊れると、時刻の無い説明や空の説明を覚え、引いたときに行けない場面や空の行が出る
        with pytest.raises(ValueError):
            SceneNote(start=start, end=end, text=text)
