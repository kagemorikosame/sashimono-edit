"""語の時刻で作品を調べる 言葉の途中の切れ目と、字幕と音の食い違い（#420）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.ai.fixes import fix_for
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    SubtitleOrigin,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    new_media_id,
)
from sashimono.core.timebase import seconds_to_frame
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.checks import HIGH, MEDIUM, Problem
from sashimono.engine.checks.scan import SUBTITLE_WORDS, WORD_CUT, WORD_KINDS
from sashimono.engine.checks.words import WordReport, check_words
from tests.words_helpers import in_scene, placed, sample_transcript, segment, spoken, wave


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
        # 置いたクリップの窓の端は trim_clip で動かせる
        assert measure["inside_scene"] is False
        assert "suggested_tail_delta" in measure

    def test_a_cut_between_clips_inside_a_scene(self, video_media: MediaItem) -> None:
        """壊れると、外側の端を動かす量を返し、切れ目は残ったまま窓の端だけが変わる"""
        media = _media(video_media)
        inner = placed(
            media,
            _clip(media, 0, 12, Fraction(1)),
            _clip(media, 12, 12, Fraction(8)),
        )
        project = in_scene(inner, Clip(0, 300, source_in=Fraction(0)))
        report = _check(project, wave(), kinds=frozenset({WORD_CUT}))
        (measure,) = [m for m in _cuts(report) if m["side"] == "out"]
        assert measure["inside_scene"] is True
        assert "suggested_tail_delta" not in measure
        assert fix_for(WORD_CUT, HIGH, measure).action == "handoff"

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
        assert report.skipped == WORD_KINDS
        assert "語の時刻が無い字幕の行" in " ".join(report.notes)

    def test_a_sound_without_subtitles_is_only_counted(self, video_media: MediaItem) -> None:
        # 壊れると、話を調べられた作品に BGM を足しただけで調べられなかった扱いになる
        media = _media(video_media)
        other = replace(video_media, id=new_media_id())
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        project = replace(project, media=(media, other))
        track = project.timeline.tracks[0]
        changed = replace(track, clips=(*track.clips, _clip(other, 30, 30, Fraction(1))))
        project = project.with_timeline(replace(project.timeline, tracks=(changed,)))
        report = _check(project, wave())
        assert report.skipped == frozenset()
        assert "字幕の無い音" in " ".join(report.notes)


def test_a_work_without_material_sound_is_not_skipped(video_media: MediaItem) -> None:
    """切れ目が無いのに調べられなかったと返すと、文字だけの作品の検査のたびに案内が付く"""
    project = placed(video_media)
    assert _check(project, wave()) == WordReport()


def test_only_sounds_without_subtitles(video_media: MediaItem) -> None:
    """字幕の無い音だけなら調べたことにせず、足す字幕の無い案内は出さない

    壊れると、切れ目を調べたように読むか、無い字幕へ語の時刻を足しに行く
    """
    project = placed(video_media, _clip(video_media, 0, 30, Fraction(1)))
    report = _check(project, wave())
    assert report.skipped == WORD_KINDS
    assert "words_only" not in " ".join(report.notes)
    assert "transcribe" in " ".join(report.notes)


def test_a_cut_on_a_line_without_words_is_not_counted_as_checked(video_media: MediaItem) -> None:
    # 壊れると、語の時刻が一部だけある音で、語の無い行の切れ目を調べたと読む
    lines = (
        segment(spoken(Fraction(1), "カメラ")),
        TranscriptSegment(Fraction(5), Fraction(6), "手で足した行"),
    )
    media = video_media.with_transcript(Transcript(lines, language="ja"))
    project = placed(media, _clip(media, 0, 15, Fraction(5)))
    assert _check(project, wave(())).skipped == WORD_KINDS


def test_a_line_without_words_away_from_the_cut_stays_checked(video_media: MediaItem) -> None:
    # 壊れると、切っていない所の 1 行で切れ目の検査全体が調べられない扱いになる
    lines = (
        segment(spoken(Fraction(1), "カメラ")),
        TranscriptSegment(Fraction(2), Fraction(3), "手で足した行"),
    )
    media = video_media.with_transcript(Transcript(lines, language="ja"))
    project = placed(media, _clip(media, 0, 90, Fraction(1)))
    assert _check(project, wave(((3.5, 12.0),))).skipped == frozenset()


def test_a_silence_cut_with_room_noise_stays_checked(video_media: MediaItem) -> None:
    """通しの確かめの形 雑音のある部屋の録音を無音カットで切った

    話の間が話の大きさから -18 dB ほどまでしか下がらない 壊れると、語の無い所の切れ目の半分を
    声が続くと読み、word_cut 全体が未検査になって言葉の途中の切れ目を返さない
    """
    media = _media(video_media)
    # 素材の 3.0〜3.5 秒 「作ります」の後と「はい」の前の話の間だけを残す
    project = placed(media, _clip(media, 0, 15, Fraction(3)))
    report = _check(project, wave(floor=0.06), kinds=frozenset({WORD_CUT}))
    assert report.skipped == frozenset()


def test_a_voiced_cut_outside_the_subtitles_is_not_checked(video_media: MediaItem) -> None:
    """語の時刻のある音で、字幕の行の外で声が続く切れ目

    壊れると、字幕を消した所や認識が拾わなかった所の切れ目を無事と返す
    """
    media = _media(video_media)
    project = placed(media, _clip(media, 0, 60, Fraction(3)))
    assert WORD_CUT in _check(project, wave(((0.0, 1.0),))).skipped
    assert WORD_CUT not in _check(project, wave()).skipped


class TestPartlyWordedLine:
    """語が行の一部にしか付かない行 語の無い所が話か無音かは波形で分ける"""

    def _project(self, video_media: MediaItem) -> Project:
        words = spoken(Fraction(1), "カメラ")
        line = TranscriptSegment(Fraction(1), Fraction(4), "カメラと足した文", words)
        media = video_media.with_transcript(Transcript((line,), language="ja"))
        return placed(media, _clip(media, 0, 60, Fraction(1)))

    def test_a_voiced_cut_is_not_counted_as_checked(self, video_media: MediaItem) -> None:
        # 壊れると、語の無い所で声の途中を切っても word_cut が問題 0 件の調査済みになる
        report = _check(self._project(video_media), wave(((0.0, 1.0), (4.0, 12.0))))
        assert WORD_CUT in report.skipped
        assert "words_only" in " ".join(report.notes)

    def test_without_a_waveform_it_is_not_checked(self, video_media: MediaItem) -> None:
        # 壊れると、波形の解析がまだの切れ目を無事と返す 語のある所の medium と同じく未確認にする
        assert WORD_CUT in _check(self._project(video_media), None).skipped

    def test_a_quiet_cut_stays_checked(self, video_media: MediaItem) -> None:
        # 壊れると、語の間の無音を切っただけで未検査になり、ほとんどの作品が調べられない
        report = _check(self._project(video_media), wave(((0.0, 1.0), (2.0, 12.0))))
        assert WORD_CUT not in report.skipped


def test_the_suggestion_keeps_the_word(video_media: MediaItem) -> None:
    """語の前半で出を切り、語の頭の近くに谷がある形

    壊れると、語の頭まで縮める提案が勝ち、high の auto の直しで語が黙って消える
    """
    line = segment(spoken(Fraction(1), "カメラ", Fraction(2, 10)))
    following = segment(spoken(Fraction(162, 100), "はい"))
    media = video_media.with_transcript(Transcript((line, following), language="ja"))
    sound = wave(((0.0, 1.0), (1.04, 1.08), (1.6, 1.64), (2.1, 12.0)))
    project = placed(media, _clip(media, 0, 3, Fraction(11, 10)))
    (measure,) = [
        found
        for found in _cuts(_check(project, sound, kinds=frozenset({WORD_CUT})))
        if found["side"] == "out"
    ]
    target = measure["suggested_source_seconds"]
    assert isinstance(target, float)
    assert target >= 1.6
    delta = measure["suggested_tail_delta"]
    assert isinstance(delta, int)
    assert delta > 0


def test_shrinking_is_the_last_resort(video_media: MediaItem) -> None:
    """語の終わりの近くの入りで、語の頭が届かない形 語の外へ出る最小の量を返す

    壊れると、動かす先が今の切れ目と同じ値になり、直す手掛かりにならない
    """
    line = segment(spoken(Fraction(1), "あ" * 10))
    following = segment(spoken(Fraction(3), "はい"))
    media = video_media.with_transcript(Transcript((line, following), language="ja"))
    sound = wave(((0.0, 1.0), (2.93, 2.97), (3.4, 12.0)))
    project = placed(media, _clip(media, 0, 60, Fraction(285, 100)))
    (measure,) = _cuts(_check(project, sound, kinds=frozenset({WORD_CUT})))
    assert measure["suggested_source_seconds"] == 3.0
    assert measure["suggested_kind"] == "word_edge"
    assert measure["suggested_head_delta"] == 5


class TestBurnedInScene:
    """#420 の出発点 ハイライトのシーンの最後の片で、字幕の文の後半が音に入っていなかった"""

    def _scene(self, media: MediaItem, window: Clip, *, inner_off: bool = False) -> Project:
        """素材の音を丸ごと持つシーンに、最初の行の字幕を素材の 1.0〜2.6 秒の所へ焼き込む"""
        transcript = media.transcript_for()
        assert transcript is not None
        origin = SubtitleOrigin(media.id, media.transcript_stream(), transcript.segments[0].id)
        text = Clip(
            30,
            48,
            source=GeneratedSource(kind="text", params={"text": "カメラを作ります"}),
            subtitle_origin=origin,
        )
        burned = Track(kind=TrackKind.VIDEO, name="字幕", clips=(text,), muted=inner_off)
        inner = placed(media, _clip(media, 0, 300, Fraction(0)))
        inner = inner.with_timeline(
            replace(inner.timeline, tracks=(*inner.timeline.tracks, burned))
        )
        return in_scene(inner, window)

    def _found(self, project: Project) -> list[Problem]:
        return [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS]

    def test_the_window_cuts_the_second_half(self, video_media: MediaItem) -> None:
        """壊れると、シーンの中の字幕の後半が音に無いことを検査で拾えない"""
        media = _media(video_media)
        window = Clip(100, 24, source_in=Fraction(1))
        (problem,) = self._found(self._scene(media, window))
        assert (problem.start, problem.end) == (100, 124)
        assert problem.measure["heard"] == "カメラを"
        assert problem.measure["missing"] == "作ります"
        assert problem.clip_ids == (str(window.id),)
        assert problem.measure["inside_scene"] is True

    def test_the_speed_is_taken_back(self, video_media: MediaItem) -> None:
        # 壊れると、2 倍速で置いたシーンの字幕を別のコマで比べ、残った語を読み違える
        media = _media(video_media)
        window = Clip(0, 12, source_in=Fraction(1), speed=Fraction(2))
        (problem,) = self._found(self._scene(media, window))
        assert (problem.start, problem.end) == (0, 12)
        assert problem.measure["missing"] == "作ります"

    def test_the_fix_goes_to_the_person(self, video_media: MediaItem) -> None:
        # 壊れると、開いているタイムラインの道具で届かない文字を AI が直しに行く
        media = _media(video_media)
        (problem,) = self._found(self._scene(media, Clip(0, 24, source_in=Fraction(1))))
        assert fix_for(SUBTITLE_WORDS, MEDIUM, problem.measure).action == "handoff"

    def test_a_whole_line_is_fine(self, video_media: MediaItem) -> None:
        # 壊れると、切っていないシーンの字幕まで問題になる
        media = _media(video_media)
        assert self._found(self._scene(media, Clip(0, 60, source_in=Fraction(1)))) == []

    def test_a_line_outside_the_window_is_not_seen(self, video_media: MediaItem) -> None:
        # 壊れると、シーンの窓の外の映らない字幕を問題にする
        media = _media(video_media)
        assert self._found(self._scene(media, Clip(0, 30, source_in=Fraction(5)))) == []

    def test_a_muted_inner_track_is_not_seen(self, video_media: MediaItem) -> None:
        # 壊れると、シーンの中で消した字幕の文言を直しに行く
        media = _media(video_media)
        window = Clip(0, 24, source_in=Fraction(1))
        # 素材の字幕の行としては切れ目をまたぐので、残るのは焼き込みを指さない問題だけ
        found = self._found(self._scene(media, window, inner_off=True))
        assert all("scene_clip_id" not in problem.measure for problem in found)

    def test_a_scene_that_is_not_placed_is_not_seen(self, video_media: MediaItem) -> None:
        # 壊れると、書き出しに出ないシーンの字幕まで問題にする word_cut の走査と同じ決まり
        media = _media(video_media)
        project = self._scene(media, Clip(0, 24, source_in=Fraction(1)))
        emptied = project.with_timeline(replace(project.timeline, tracks=()))
        assert self._found(emptied) == []

    def test_only_the_asked_range(self, video_media: MediaItem) -> None:
        # 壊れると、範囲の判定にシーンの中の時刻を使い、頼んだ範囲の字幕を見逃す
        media = _media(video_media)
        project = self._scene(media, Clip(100, 24, source_in=Fraction(1)))
        report = check_words(project, 0, 50, WORD_KINDS, lambda _media, _stream: wave())
        assert [p for p in report.problems if p.kind == SUBTITLE_WORDS] == []
        report = check_words(project, 110, 120, WORD_KINDS, lambda _media, _stream: wave())
        (problem,) = [p for p in report.problems if p.kind == SUBTITLE_WORDS]
        assert (problem.start, problem.end) == (110, 120)

    def test_an_origin_without_the_material(self, video_media: MediaItem) -> None:
        # 壊れると、素材の印の無い前の版の焼き込みを調べず、問題が無いと読む
        media = _media(video_media)
        project = self._scene(media, Clip(0, 24, source_in=Fraction(1)))
        (scene,) = project.scenes
        tracks = []
        for track in scene.timeline.tracks:
            clips = tuple(
                replace(clip, subtitle_origin=replace(clip.subtitle_origin, media_id=None))
                if clip.subtitle_origin is not None
                else clip
                for clip in track.clips
            )
            tracks.append(replace(track, clips=clips))
        changed = replace(scene, timeline=replace(scene.timeline, tracks=tuple(tracks)))
        assert len(self._found(replace(project, scenes=(changed,)))) == 1


class TestSubtitleWords:
    def test_a_burned_line_without_words_is_not_checked(self, video_media: MediaItem) -> None:
        # 壊れると、音の無い所の語の時刻の無い焼き込みを比べずに「問題なし」と返す
        line = TranscriptSegment(Fraction(1), Fraction(2), "言っていない言葉")
        media = video_media.with_transcript(Transcript((line,), language="ja"))
        report = _check(self._burned(placed(media), media, "言っていない言葉", 24), wave())
        assert report.skipped == frozenset({SUBTITLE_WORDS})
        assert "words_only" in " ".join(report.notes)

    def test_another_placing_of_a_burned_line_is_looked_at(self, video_media: MediaItem) -> None:
        # 壊れると、焼き込みが 1 つあるだけで、別の所に置いた同じ行の切れ目の食い違いを見逃す
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 60, Fraction(1)), _clip(media, 90, 24, Fraction(1)))
        project = self._burned(project, media, "カメラを作ります", 48)
        (problem,) = [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS]
        assert problem.start == 90

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

    def test_no_sound_left_still_reports_the_words(self, video_media: MediaItem) -> None:
        # 壊れると、音を消して字幕だけが残った作品で、言っていない言葉に気付けない
        media = _media(video_media)
        project = self._burned(placed(media), media, "カメラを作ります", 24)
        (problem,) = _check(project, wave()).problems
        assert problem.kind == SUBTITLE_WORDS

    def test_a_repeated_word_is_not_counted_twice(self, video_media: MediaItem) -> None:
        """同じ字が繰り返す文の後半を切り、字幕を前半に直した形

        壊れると、残した語の字を鳴らない語と数え、直した字幕を報告し続ける
        """
        line = segment(spoken(Fraction(1), "はいはい"))
        media = video_media.with_transcript(Transcript((line,), language="ja"))
        project = placed(media, _clip(media, 0, 12, Fraction(1)))
        project = self._burned(project, media, "はい", 12)
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []

    def test_a_longer_copy_is_looked_at_too(self, video_media: MediaItem) -> None:
        # 壊れると、同じ行を頭をそろえて短く置いた方が、長い方の検査を止める
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        project = self._burned(project, media, "カメラを", 12)
        project = self._burned(project, media, "カメラを作ります", 24)
        (problem,) = [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS]
        assert problem.end == 24

    @pytest.mark.parametrize("off", ["muted", "disabled"])
    def test_a_subtitle_nobody_sees_is_not_reported(self, video_media: MediaItem, off: str) -> None:
        # 壊れると、消音したトラックや無効のクリップの字幕の文言を直しに行く
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 300, Fraction(1)))
        project = self._burned(project, media, "カメラを作ります", 12)
        tracks = list(project.timeline.tracks)
        burned = tracks[-1]
        tracks[-1] = (
            replace(burned, muted=True)
            if off == "muted"
            else replace(burned, clips=(replace(burned.clips[0], enabled=False),))
        )
        project = project.with_timeline(replace(project.timeline, tracks=tuple(tracks)))
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []

    def test_the_problem_stays_inside_the_asked_range(self, video_media: MediaItem) -> None:
        # 壊れると、頼んだ範囲の外のフレームを返す
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 24, Fraction(1)))
        project = self._burned(project, media, "カメラを作ります", 24)
        report = check_words(project, 6, 12, WORD_KINDS, lambda _media, _stream: wave())
        (problem,) = [p for p in report.problems if p.kind == SUBTITLE_WORDS]
        assert (problem.start, problem.end) == (6, 12)

    def test_a_sliver_of_a_subtitle_is_not_reported(self, video_media: MediaItem) -> None:
        """通しの確かめの形 起こした字幕の行が音の残り端に 3 コマだけ掛かった

        壊れると、人にも AI にも見えない数コマの字幕が 3 回の検査で残り続けた
        """
        media = _media(video_media)
        # 素材の 2.5〜4.5 秒 「作ります」の行の終わりの 0.1 秒だけが 3 コマ掛かる
        project = placed(media, _clip(media, 0, 60, Fraction(5, 2)))
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []

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

    def test_a_line_fitted_after_its_first_words_is_fine(self, video_media: MediaItem) -> None:
        """字幕の行の頭を声へ寄せて（#426）、語の時刻より少し後ろから出す形

        壊れると、音を切っていないのに、行の頭の外に出た最初の語を残した音に無いと報告する
        """
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 240, Fraction(0)))
        project = self._burned(project, media, "カメラを作ります", 1)
        tracks = list(project.timeline.tracks)
        rate = project.rate
        start = seconds_to_frame(Fraction(3, 2), rate)
        end = seconds_to_frame(Fraction(26, 10), rate)
        burned = replace(tracks[-1].clips[0], timeline_start=start, duration=end - start)
        tracks[-1] = replace(tracks[-1], clips=(burned,))
        project = project.with_timeline(replace(project.timeline, tracks=tuple(tracks)))
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []

    def test_one_particle_is_not_reported(self, video_media: MediaItem) -> None:
        # 語の時刻のずれでも 1 字の助詞は落ちる 毎回返すと本当の食い違いが埋もれる
        media = _media(video_media)
        project = placed(media, _clip(media, 0, 45, Fraction(1)))
        assert [p for p in _check(project, wave()).problems if p.kind == SUBTITLE_WORDS] == []
