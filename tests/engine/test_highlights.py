"""盛り上がりの手がかり（#325 engine/highlights.py）

作った音（笑い・話し声・拍の音楽・叫び）と作った差の並びで、見分けの境目を押さえる
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sashimono.core.model import Transcript, TranscriptSegment
from sashimono.engine.highlights import (
    HOP_SECONDS,
    LAUGHTER,
    LOUD,
    MOTION,
    HighlightCurves,
    highlight_curves,
    is_laughter_text,
)
from sashimono.engine.scenes import SceneSignal
from tests.highlight_sound import RATE, beat, laugh, speech, waveform_of


def _curves(
    samples: np.ndarray,
    *,
    transcript: Transcript | None = None,
    signals: tuple[str, ...] = (LAUGHTER, LOUD, MOTION),
) -> HighlightCurves:
    return highlight_curves(
        samples.shape[0] / RATE,
        waveform=waveform_of(samples),
        transcript=transcript,
        signals=signals,
    )


def _motion_signal(
    busy: tuple[float, float], *, until: float = 20.0, fps: float = 30.0, level: float = 0.06
) -> SceneSignal:
    """据え置きの画面（差 0.005）の中に、``busy`` の秒だけ大きく動く所がある並び"""
    times = np.arange(1, round(until * fps) + 1, dtype=np.float64) / fps
    pixel = np.full(times.shape, 0.005 * 30.0 / fps, np.float32)
    inside = (times >= busy[0]) & (times < busy[1])
    pixel[inside] = level * 30.0 / fps
    histogram = np.full(times.shape, 0.01, np.float32)
    return SceneSignal(times, pixel, histogram, analyzed=until, complete=True)


class TestLaughter:
    def test_a_laugh_between_talk_is_found(self) -> None:
        """話の合間の笑いを拾う

        壊れると、見どころを探す AI が笑いの所を 1 つも出せない
        """
        rng = np.random.default_rng(3)
        samples = np.concatenate([speech(15, rng), laugh(2.5, rng), speech(15, rng)])
        found = _curves(samples).highlights()
        assert len(found) == 1
        assert found[0].start == pytest.approx(15.0, abs=0.75)
        assert found[0].end == pytest.approx(17.5, abs=0.75)
        assert found[0].reasons[0] == LAUGHTER

    @pytest.mark.parametrize("pace", [3.5, 5.0, 7.5])
    def test_slow_and_fast_laughs_are_found(self, pace: float) -> None:
        """笑いの速さ（1 秒 3.5〜7.5 回）によらず拾う

        壊れると、ゆっくり笑う人の笑いを見落とす
        """
        rng = np.random.default_rng(4)
        samples = np.concatenate([speech(10, rng), laugh(2.5, rng, rate=pace), speech(10, rng)])
        curves = _curves(samples)
        assert curves.laughter is not None
        assert curves.laughter.max() >= 0.9

    @pytest.mark.parametrize("seed", range(6))
    def test_plain_talk_is_not_a_laugh(self, seed: int) -> None:
        """ふつうの話し声を笑いと見ない

        壊れると、話している所が全部盛り上がりに並び、一覧が役に立たない
        """
        samples = speech(60, np.random.default_rng(100 + seed))
        assert _curves(samples).highlights() == ()

    def test_a_steady_beat_is_not_a_laugh(self) -> None:
        """何十秒も続く一定の拍を笑いと見ない

        壊れると、BGM の流れる所が丸ごと笑いの区間になる
        """
        rng = np.random.default_rng(5)
        samples = np.concatenate([speech(20, rng), beat(30)])
        curves = _curves(samples)
        assert curves.laughter is not None
        assert curves.laughter.max() < 0.5

    def test_a_quiet_laugh_below_the_talk_is_still_found(self) -> None:
        """話し声より少し小さい笑い（聞き手の笑い）も拾う

        壊れると、撮っている人の笑いを見落とす
        """
        rng = np.random.default_rng(6)
        samples = np.concatenate([speech(20, rng), laugh(2.5, rng, amp=0.15), speech(20, rng)])
        assert list(_curves(samples).highlights()[0].reasons) == [LAUGHTER]


class TestSubtitles:
    @pytest.mark.parametrize(
        ("text", "laughs"),
        [
            ("（笑）", True),
            ("そうなんですよ(笑)", True),
            ("[Laughter]", True),
            ("ははは", True),
            ("あははは！", True),
            ("www", True),
            ("ふふふっ", True),
            ("笑顔で話しました", False),
            ("はは", False),
            ("はい", False),
            ("母が来た", False),
            ("", False),
            ("あはは", True),
            ("ｗｗ", True),
            ("草", True),
            ("haha", True),
            ("ああ、はい", False),
            ("あっ、はい", False),
            ("あああ", False),
            ("あっ", False),
            ("w", False),
        ],
    )
    def test_laughter_marks_are_read(self, text: str, laughs: bool) -> None:
        """字幕の笑いの印を見分ける

        壊れると、「笑顔」や「ああ、はい」の相づちの字幕を笑いと見て盛り上がりに返すか、
        「（笑）」「ｗｗ」「草」を見落とす
        """
        assert is_laughter_text(text) is laughs

    def test_a_marked_subtitle_makes_a_laugh(self) -> None:
        """音で拾えない笑いも、字幕の印があれば拾う

        壊れると、BGM の下の笑いを字幕があっても見落とす
        """
        rng = np.random.default_rng(7)
        samples = speech(20, rng)
        transcript = Transcript(
            segments=(TranscriptSegment(start=Fraction(8), end=Fraction(10), text="（笑）"),)
        )
        found = _curves(samples, transcript=transcript).highlights()
        assert len(found) == 1
        assert found[0].laughter_from_subtitles
        assert found[0].start == pytest.approx(8.0)

    def test_plain_subtitles_lower_the_laugh(self) -> None:
        """ふつうの話し声の字幕が重なる所は笑いの点を下げる

        壊れると、音節のそろった話し方を笑いと読む
        """
        rng = np.random.default_rng(8)
        samples = np.concatenate([speech(10, rng), laugh(2.5, rng), speech(10, rng)])
        plain = _curves(samples)
        talk = Transcript(
            segments=(TranscriptSegment(start=Fraction(9), end=Fraction(13), text="それでですね"),)
        )
        damped = _curves(samples, transcript=talk)
        assert plain.laughter is not None and damped.laughter is not None
        assert plain.laughter.max() >= 0.9
        assert damped.laughter.max() == pytest.approx(plain.laughter.max() * 0.5, abs=0.01)


class TestLoud:
    def test_a_shout_is_a_peak(self) -> None:
        """声を張った所を音量の山と見る 壊れると、叫び・歓声の所を見落とす"""
        rng = np.random.default_rng(9)
        samples = np.concatenate([speech(20, rng), speech(3, rng, amp=0.9), speech(20, rng)])
        found = _curves(samples, signals=(LOUD,)).highlights()
        assert len(found) == 1
        assert found[0].loud == pytest.approx(1.0)
        assert found[0].laughter is None
        assert 19.0 <= found[0].start <= 21.0

    def test_an_even_level_has_no_peak(self) -> None:
        """ずっと同じ大きさの素材に山を作らない

        壊れると、普段の話し声のゆらぎを山として並べる
        """
        samples = speech(40, np.random.default_rng(10))
        assert _curves(samples, signals=(LOUD,)).highlights() == ()

    def test_talk_with_pauses_has_no_peak(self) -> None:
        """話の合間が多い素材でも、ふつうの話し声を山と見ない

        壊れると、普段の大きさを合間の小さい所で低く見積もり、話している所が全部山に並ぶ
        （実際の素材で山の 8 割が外れだった）
        """
        rng = np.random.default_rng(12)
        parts = []
        for _ in range(20):
            parts.append(speech(1.0, rng))
            parts.append(0.01 * rng.standard_normal(RATE * 2))
        samples = np.concatenate(parts)
        assert _curves(samples, signals=(LOUD,)).highlights() == ()

    def test_silence_has_no_peak(self) -> None:
        """無音だけの素材で落ちず、山も作らない 壊れると、無音の素材で道具が例外を出す"""
        samples = np.zeros(RATE * 5, np.float32)
        curves = _curves(samples)
        assert curves.highlights() == ()


class TestMotion:
    def test_a_busy_stretch_is_found(self) -> None:
        """大きく動く所を拾う 壊れると、走る・転ぶ・速いパンの所を見落とす"""
        curves = highlight_curves(20.0, scenes=_motion_signal((8.0, 12.0)))
        found = curves.highlights()
        assert len(found) == 1
        assert found[0].reasons == (MOTION,)
        assert found[0].start == pytest.approx(8.0, abs=0.75)
        assert found[0].end == pytest.approx(12.0, abs=0.75)

    def test_a_game_scene_moving_a_little_more_is_found(self) -> None:
        """ふだんから少し動く実況の画面で、動きが倍ほどになった所を拾う

        壊れると、実況の素材の動く所（差 0.03 前後）をほとんど見落とす（幅を広く取って
        いたとき、実際の素材で当たりが 18% だった）
        """
        signal = _motion_signal((8.0, 12.0), level=0.035)
        busy = SceneSignal(
            signal.times,
            np.maximum(signal.pixel, 0.015),
            signal.histogram,
            analyzed=20.0,
            complete=True,
        )
        found = highlight_curves(20.0, scenes=busy).highlights()
        assert [item.reasons for item in found] == [(MOTION,)]

    def test_a_cut_is_not_motion(self) -> None:
        """場面の切り替わりの 1 コマを動きと見ない

        壊れると、カットの所が全部盛り上がりに並ぶ
        """
        signal = _motion_signal((0.0, 0.0))
        pixel = signal.pixel.copy()
        histogram = signal.histogram.copy()
        pixel[300], histogram[300] = 0.3, 0.8
        cut = SceneSignal(signal.times, pixel, histogram, analyzed=20.0, complete=True)
        assert highlight_curves(20.0, scenes=cut).highlights() == ()

    def test_sixty_frames_count_the_same(self) -> None:
        """60 fps の素材も 30 fps と同じに数える

        壊れると、60 fps の素材で大きな動きを半分に見て見落とす
        """
        thirty = highlight_curves(20.0, scenes=_motion_signal((8.0, 12.0), fps=30.0))
        sixty = highlight_curves(20.0, scenes=_motion_signal((8.0, 12.0), fps=60.0))
        assert thirty.motion is not None and sixty.motion is not None
        assert sixty.motion.max() == pytest.approx(thirty.motion.max(), abs=0.05)

    def test_a_restless_video_needs_more(self) -> None:
        """ずっと動いている素材では、普段より大きい所だけを拾う

        壊れると、ゲームの録画が丸ごと 1 区間になる
        """
        signal = _motion_signal((5.0, 7.0), level=0.2)
        busy = SceneSignal(
            signal.times,
            np.maximum(signal.pixel, 0.05),
            signal.histogram,
            analyzed=20.0,
            complete=True,
        )
        found = highlight_curves(20.0, scenes=busy).highlights()
        assert len(found) == 1
        assert found[0].start == pytest.approx(5.0, abs=0.75)

    def test_unread_time_has_no_motion(self) -> None:
        """読み終えていない所は動き 0 のまま

        壊れると、解析の途中の素材で後ろ半分を動いたと読む
        """
        signal = _motion_signal((2.0, 4.0), until=10.0)
        partial = SceneSignal(
            signal.times, signal.pixel, signal.histogram, analyzed=10.0, complete=False
        )
        curves = highlight_curves(20.0, scenes=partial)
        assert curves.motion is not None
        assert curves.motion[round(12 / HOP_SECONDS) :].max() == 0.0
        assert len(curves.highlights()) == 1


class TestCombining:
    def test_two_medium_clues_make_a_highlight(self) -> None:
        """弱い手がかりも 2 つ重なれば区間にする

        壊れると、少し笑って少し動いた所を見落とす
        """
        count = round(10 / HOP_SECONDS)
        half = np.full(count, 0.3, np.float32)
        none = np.zeros(count, np.float32)
        both = HighlightCurves(10.0, laughter=half, loud=half, motion=none)
        one = HighlightCurves(10.0, laughter=half, loud=none, motion=none)
        assert len(both.highlights()) == 1
        assert both.highlights()[0].score == pytest.approx(0.51, abs=0.001)
        assert one.highlights() == ()

    def test_close_runs_are_merged(self) -> None:
        """息継ぎほどの切れ目で区間を割らない 壊れると、1 回の笑いが 3 つの区間に並ぶ"""
        count = round(10 / HOP_SECONDS)
        curve = np.zeros(count, np.float32)
        curve[8:12] = 1.0
        curve[14:18] = 1.0
        curve[30:34] = 1.0
        found = HighlightCurves(10.0, laughter=curve, loud=None, motion=None).highlights()
        assert [(item.start, item.end) for item in found] == [(2.0, 4.5), (7.5, 8.5)]

    def test_short_blips_are_dropped(self) -> None:
        """短すぎる区間は返さない 壊れると、物音 1 つの所まで盛り上がりに並ぶ"""
        count = round(10 / HOP_SECONDS)
        curve = np.zeros(count, np.float32)
        curve[8] = 1.0
        assert HighlightCurves(10.0, loud=curve, laughter=None, motion=None).highlights() == ()
        assert (
            len(
                HighlightCurves(10.0, loud=curve, laughter=None, motion=None).highlights(
                    min_seconds=0.0
                )
            )
            == 1
        )

    def test_unasked_clues_are_not_read(self) -> None:
        """頼まれていない手がかりは調べない

        壊れると、動きだけを頼んでも音の点が混ざり、区間が変わる
        """
        rng = np.random.default_rng(11)
        samples = np.concatenate([speech(10, rng), laugh(2.5, rng), speech(10, rng)])
        curves = highlight_curves(
            samples.shape[0] / RATE,
            waveform=waveform_of(samples),
            scenes=_motion_signal((0.0, 0.0), until=samples.shape[0] / RATE),
            signals=(MOTION,),
        )
        assert curves.laughter is None and curves.loud is None
        assert curves.highlights() == ()
