"""語の時刻で作品を調べる 言葉の途中の切れ目と、字幕と音の食い違い（#420）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from sashimono.core.model import (
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    SubtitleOrigin,
    Track,
    TrackKind,
    Transcript,
    new_media_id,
)
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.checks import HIGH, MEDIUM
from sashimono.engine.checks.report import sorted_problems
from sashimono.engine.checks.scan import SUBTITLE_WORDS, WORD_CUT, WORD_KINDS
from sashimono.engine.checks.words import WordReport, check_words
from tests.words_helpers import in_scene, placed, sample_transcript, wave


def _media(video_media: MediaItem) -> MediaItem:
    return video_media.with_transcript(sample_transcript())


def _clip(
    media: MediaItem, start: int, duration: int, source_in: Fraction, **fields: object
) -> Clip:
    return Clip(start, duration, media_id=media.id, source_in=source_in, **fields)  # type: ignore[arg-type]


def _check(
    project: Project,
    sound: Waveform | None = None,
    *,
    kinds: frozenset[str] = WORD_KINDS,
) -> WordReport:
    return check_words(project, 0, project.duration, kinds, lambda _media, _stream: sound)


def _cuts(report: WordReport) -> list[dict[str, object]]:
    return [dict(p.measure) for p in report.problems if p.kind == WORD_CUT]


class TestWordCut:
    def test_a_cut_inside_a_word(self, video_media: MediaItem) -> None:
        """「カメ|ラを」で出を切った形 音量の測りでは見つからない切れ目"""
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        report = _check(project, wave())
        (problem,) = [p for p in report.problems if p.kind == WORD_CUT]
        assert problem.severity == HIGH
        # 問題のコマは出の手前の残った最後のコマ preview_frame で見られる
        assert problem.start == 11
        assert problem.measure["side"] == "out"
        assert problem.measure["text"] == "カメラを"
        assert problem.measure["source_seconds"] == 1.4

    def test_the_suggested_move_keeps_the_word(self, video_media: MediaItem) -> None:
        """出は語の後ろへ伸ばす 縮めると頼まれた言葉を落とす"""
        media = _media(video_media)
        sound = wave(((0.0, 1.0), (1.8, 1.86), (2.6, 12.0)))
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        (measure,) = _cuts(_check(project, sound))
        assert measure["suggested_kind"] == "valley"
        target = measure["suggested_source_seconds"]
        assert isinstance(target, float)
        assert 1.8 <= target <= 1.87
        # 素材の 1.4 秒から谷の頭の 0.05 秒後ろまで 語を削らないよう伸ばす向きへ丸める
        assert measure["suggested_tail_delta"] == 14

    def test_an_in_edge_moves_back(self, video_media: MediaItem) -> None:
        # 壊れると、入りを語の頭へ戻せず、言葉の頭が欠けたまま残る
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 60, Fraction(14, 10)))
        (measure,) = _cuts(_check(project, wave()))
        assert measure["side"] == "in"
        # 語の頭（素材の 1.0 秒）の手前の無音まで戻す 詰める向きだと語を丸ごと落とす
        delta = measure["suggested_head_delta"]
        assert isinstance(delta, int)
        assert -14 <= delta <= -12

    def test_a_quiet_cut_is_fine(self, video_media: MediaItem) -> None:
        """語の時刻の上では途中でも、音が静かなら語は欠けない

        語の時刻は前後へずれる 壊れると語の後ろの無音を切った所まで問題になる
        """
        media = _media(video_media)
        sound = wave(((0.0, 1.0), (1.3, 1.5), (2.6, 12.0)))
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        assert _cuts(_check(project, sound)) == []

    def test_a_boundary_is_fine(self, video_media: MediaItem) -> None:
        # 壊れると、語の境目で切った所まで問題になる
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        assert _cuts(_check(project, wave())) == []

    def test_without_a_waveform_it_is_not_certain(self, video_media: MediaItem) -> None:
        # 壊れると、音を確かめていない切れ目を確かな問題として直す
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        (problem,) = [p for p in _check(project, None).problems if p.kind == WORD_CUT]
        assert problem.severity == MEDIUM
        assert "確かめていない" in problem.detail

    def test_speed_is_judged_in_the_source_time(self, video_media: MediaItem) -> None:
        """2 倍速で 6 コマ（0.2 秒）でも、素材では 0.4 秒進んで「カメ|ラ」に当たる

        壊れると、速度を変えたクリップの切れ目を別の語で判定する
        """
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 6, Fraction(1), speed=Fraction(2)))
        (measure,) = _cuts(_check(project, wave()))
        assert measure["source_seconds"] == 1.4

    def test_inside_a_scene(self, video_media: MediaItem) -> None:
        """シーンを置いたクリップの窓の端が語の途中に当たる形"""
        media = _media(video_media)
        inner = placed(media, _clip(media, 0, 300, Fraction(0)))
        project = in_scene(inner, Clip(0, 12, source_in=Fraction(1)))
        (measure,) = _cuts(_check(project, wave()))
        assert measure["source_seconds"] == 1.4
        assert "scene_clip_id" in measure
        # シーンの中のクリップは trim_clip の相手にならない 間違った相手へ量を返さない
        assert "suggested_tail_delta" not in measure

    def test_the_scene_clip_is_named(self, video_media: MediaItem) -> None:
        # 壊れると、シーンの中のクリップの ID を返し、AI が開いている画面で見つけられない
        media = _media(video_media)
        inner = placed(media, _clip(media, 0, 300, Fraction(0)))
        window = Clip(0, 12, source_in=Fraction(1))
        project = in_scene(inner, window)
        (problem,) = [p for p in _check(project, wave()).problems if p.kind == WORD_CUT]
        assert problem.clip_ids == (str(window.id),)

    def test_only_the_asked_kinds(self, video_media: MediaItem) -> None:
        # 壊れると、絵だけを頼んだ呼び出しに語の問題が混ざる
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        assert _check(project, wave(), kinds=frozenset({"black"})) == WordReport()

    def test_problems_sort_with_the_rest(self, video_media: MediaItem) -> None:
        # 壊れると、語の問題だけ一覧の後ろに固まり、時刻の順に読めない
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        problems = _check(project, wave()).problems
        assert sorted_problems(reversed(problems)) == sorted_problems(problems)


class TestNoWordTimes:
    def test_it_is_not_counted_as_checked(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """調べていないのに問題 0 件と読むと、切れ目は無事と AI に言うことになる"""
        media = video_media.with_transcript(transcript)
        project = placed(media, _clip(media, 0, 30, Fraction(1)))
        report = _check(project, wave())
        assert report.skipped == WORD_KINDS
        assert report.problems == ()
        assert "words_only" in " ".join(report.notes)

    def test_one_material_without_words_is_noted(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、語の時刻の無い音の切れ目を調べたと読む
        media = _media(video_media)
        other = replace(video_media.with_transcript(transcript), id=new_media_id())
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        project = replace(project, media=(media, other))
        track = project.timeline.tracks[0]
        changed = replace(track, clips=(*track.clips, _clip(other, 30, 30, Fraction(1))))
        project = project.with_timeline(replace(project.timeline, tracks=(changed,)))
        report = _check(project, wave())
        assert report.skipped == frozenset()
        assert "語の時刻の無い音" in " ".join(report.notes)


def test_a_work_without_material_sound_is_not_skipped(video_media: MediaItem) -> None:
    """切れ目が無いのに調べられなかったと返すと、文字だけの作品の検査のたびに案内が付く"""
    project = placed(video_media)
    assert _check(project, wave()) == WordReport()


class TestSubtitleWords:
    def _burned(self, project: Project, media: MediaItem, text: str, duration: int) -> Project:
        transcript = media.transcript_for()
        assert transcript is not None
        origin = SubtitleOrigin(media.id, media.transcript_stream(), transcript.segments[0].id)
        clip = Clip(
            0,
            duration,
            source=GeneratedSource(kind="text", params={"text": text}),
            subtitle_origin=origin,
        )
        track = Track(kind=TrackKind.VIDEO, name="字幕", clips=(clip,))
        return project.with_timeline(
            replace(project.timeline, tracks=(*project.timeline.tracks, track))
        )

    def test_the_second_half_is_not_heard(self, video_media: MediaItem) -> None:
        """0.2.3 の通しの確かめの形 焼き込んだ字幕の文の後半が音に入っていない"""
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        project = self._burned(project, media, "カメラを作ります", 24)
        (problem,) = [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS]
        assert problem.severity == MEDIUM
        assert problem.measure["heard"] == "カメラを"
        assert problem.measure["missing"] == "作ります"
        assert problem.measure["text"] == "カメラを作ります"

    def test_a_shortened_subtitle_is_fine(self, video_media: MediaItem) -> None:
        # 字幕を残した音に合わせて直した形 語は行に残っているので、文言で見ないと直しても消えない
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        project = self._burned(project, media, "カメラを", 24)
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []

    def test_an_unburned_subtitle_that_crosses_a_cut(self, video_media: MediaItem) -> None:
        # 壊れると、焼き込む前の字幕の食い違いに気付けない
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        (problem,) = [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS]
        assert problem.measure["missing"] == "作ります"

    def test_a_whole_subtitle_is_fine(self, video_media: MediaItem) -> None:
        # 壊れると、切っていない字幕まで食い違いとして返す
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 60, Fraction(1)))
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []

    def test_one_particle_is_not_reported(self, video_media: MediaItem) -> None:
        # 語の時刻のずれでも 1 字の助詞は落ちる 毎回返すと本当の食い違いが埋もれる
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 45, Fraction(1)))
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []
