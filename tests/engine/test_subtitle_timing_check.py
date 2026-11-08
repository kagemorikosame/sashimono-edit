"""字幕の出る・消える時刻のずれを仕上がりの検査で見つける（#414 ``subtitle_timing``）

通しの確かめで、字幕の出が話と最大 2 秒ずれていたのに ``check_output`` に確かめが無かった
"""

from __future__ import annotations

from fractions import Fraction

from sashimono.ai.fixes import ASK, AUTO, fix_for
from sashimono.core.model import Clip, MediaItem, Transcript, TranscriptSegment
from sashimono.core.subtitle_timing import SubtitleTiming
from sashimono.engine.audio.voice_edges import fit_to_voice, loudness_of_waveform
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.checks import MEDIUM
from sashimono.engine.checks.scan import SUBTITLE_TIMING
from sashimono.engine.checks.words import WordReport, check_words
from tests.words_helpers import placed, wave

KINDS = frozenset({SUBTITLE_TIMING})
#: 声は 1.0〜2.6 秒と 4.0〜4.4 秒（:func:`wave` の既定）
VOICE = wave()


def _media(video_media: MediaItem, *lines: tuple[float, float], edited: bool = False) -> MediaItem:
    segments = tuple(
        TranscriptSegment(
            Fraction(start).limit_denominator(100),
            Fraction(end).limit_denominator(100),
            f"行 {index}",
            edited=edited,
        )
        for index, (start, end) in enumerate(lines)
    )
    return video_media.with_transcript(Transcript(segments, language="ja"))


def _check(
    media: MediaItem, sound: Waveform | None = VOICE, timing: SubtitleTiming | None = None
) -> WordReport:
    project = placed(media, Clip(timeline_start=0, duration=300, media_id=media.id))
    return check_words(project, 0, project.duration, KINDS, lambda _m, _s: sound, timing)


def _fitted(*lines: tuple[float, float]) -> list[tuple[Fraction, Fraction]]:
    times = [
        (Fraction(a).limit_denominator(100), Fraction(b).limit_denominator(100)) for a, b in lines
    ]
    return fit_to_voice(times, loudness_of_waveform(VOICE))


def test_an_early_line_is_found(video_media: MediaItem) -> None:
    """声より 0.4 秒早く出る字幕 壊れると、ずれたまま書き出しても検査は通る"""
    report = _check(_media(video_media, (0.6, 2.6)))
    (problem,) = report.problems
    assert problem.kind == SUBTITLE_TIMING
    assert problem.severity == MEDIUM
    assert float(problem.measure["start_offset_seconds"]) < -0.3
    assert problem.measure["burned"] is False


def test_a_line_on_the_voice_is_fine(video_media: MediaItem) -> None:
    """声へ寄せた行まで知らせると、直しても消えない問題が残り続ける"""
    ((start, end),) = _fitted((1.0, 2.6))
    report = _check(_media(video_media, (float(start), float(end))))
    assert report.problems == ()
    assert report.skipped == frozenset()


def test_the_preferred_offset_is_not_a_problem(video_media: MediaItem) -> None:
    """わざと 0.3 秒早く出している人の字幕を、全部ずれと読まない"""
    ((start, end),) = _fitted((1.0, 2.6))
    early = float(start) - 0.3
    timing = SubtitleTiming(start_offset_ms=-300)
    assert _check(_media(video_media, (early, float(end))), timing=timing).problems == ()
    # 設定のずらしを足さずに比べると、ずれと読む
    assert _check(_media(video_media, (early, float(end)))).problems != ()


def test_without_a_waveform_it_is_not_counted_as_checked(video_media: MediaItem) -> None:
    """比べられなかった行を問題 0 件と読むと、ずれは無いと AI に言うことになる"""
    report = _check(_media(video_media, (0.6, 2.6)), sound=None)
    assert report.problems == ()
    assert SUBTITLE_TIMING in report.skipped


def test_switched_off_fitting_is_not_checked(video_media: MediaItem) -> None:
    """声に合わせないと決めた人の字幕を、声とずれていると言わない"""
    report = _check(_media(video_media, (0.6, 2.6)), timing=SubtitleTiming(fit_to_voice=False))
    assert report.problems == ()
    assert SUBTITLE_TIMING in report.skipped


def test_the_fix_is_retiming_unless_held(video_media: MediaItem) -> None:
    """起こしたままの行は合わせ直してよい 人が直した行と焼き込んだ字幕は尋ねる"""
    (plain,) = _check(_media(video_media, (0.6, 2.6))).problems
    assert fix_for(plain.kind, plain.severity, plain.measure).action == AUTO
    assert fix_for(plain.kind, plain.severity, plain.measure).tools == ("retime_subtitles",)
    (edited,) = _check(_media(video_media, (0.6, 2.6), edited=True)).problems
    assert fix_for(edited.kind, edited.severity, edited.measure).action == ASK
