"""無音カットの勧める値（#413 の 6 番） :func:`choose_silence` と波形からの読み方"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest

from sashimono.core.mixing import db_to_ratio
from sashimono.engine.audio import silence
from sashimono.engine.audio.silence import (
    SilenceOptions,
    choose_silence,
    detect_silence,
    suggest_silence,
)
from sashimono.engine.audio.waveform import PeakLevel, Waveform

_TICK = 0.1


def _ticks(*runs: tuple[float, int]) -> np.ndarray:
    """（音の山の dBFS, 区切りの数）の並びを 1 本の推移へ"""
    return np.concatenate([np.full(count, level) for level, count in runs])


def _talk(pause_db: float, repeats: int = 20) -> np.ndarray:
    return _ticks(*[(-12.0, 10), (pause_db, 10)] * repeats)


class TestTheKnee:
    def test_the_value_covers_the_pauses(self) -> None:
        """壊れると、間の音量より下を勧め、切れる所が無いと読む"""
        best, nearby = choose_silence(_talk(-40.0), _TICK, 0.5, 0.1)
        assert best is not None
        assert -40.0 < best.threshold_db <= -38.0
        assert best.cuts == 20
        assert [row.threshold_db for row in nearby] == [
            best.threshold_db - 3,
            best.threshold_db,
            best.threshold_db + 3,
        ]

    def test_mostly_pauses_still_get_a_value(self) -> None:
        """壊れると、間が中央値になる素材で探す範囲が間の音量で止まり、値を勧められない"""
        peaks = _ticks(*[(-12.0, 10), (-40.0, 24)] * 20)
        assert float(np.median(peaks)) == -40.0
        best, _nearby = choose_silence(peaks, _TICK, 0.5, 0.1)
        assert best is not None
        assert -40.0 < best.threshold_db <= -38.0

    def test_music_with_quiet_passages_has_no_value(self) -> None:
        """壊れると、探す範囲を広げたせいで、強弱のある曲の弱い所を間と読んで切る"""
        assert choose_silence(_ticks(*[(-14.0, 40), (-20.0, 40)] * 10), _TICK, 0.5, 0.1) == (
            None,
            (),
        )

    def test_steady_sound_has_no_value(self) -> None:
        """壊れると、鳴り続ける曲の音量を間と読み、曲の途中を切る"""
        assert choose_silence(_ticks((-14.0, 300)), _TICK, 0.5, 0.1) == (None, ())

    def test_pure_silence_has_no_value(self) -> None:
        """壊れると、全部が無音の素材で全体を切れると勧める"""
        assert choose_silence(_ticks((-120.0, 300)), _TICK, 0.5, 0.1) == (None, ())

    def test_nothing_measured_has_no_value(self) -> None:
        """壊れると、音の無い作品で道具ごと落ちる"""
        assert choose_silence(np.empty(0), _TICK, 0.5, 0.1) == (None, ())


class TestTheEdges:
    def test_almost_all_pauses(self) -> None:
        """壊れると、話が 1 割に満たない素材で探す範囲が空になり、切れる間があるのに値が出ない"""
        best, _nearby = choose_silence(_ticks(*[(-12.0, 5), (-40.0, 95)] * 10), _TICK, 0.5, 0.1)
        assert best is not None
        assert -40.0 < best.threshold_db <= -38.0

    def test_a_few_pauses_in_long_talk(self) -> None:
        """壊れると、間が 5 % に満たない素材で探す範囲の下の端が話の音量になり、値が出ない"""
        best, _nearby = choose_silence(_ticks(*[(-12.0, 200), (-40.0, 10)] * 5), _TICK, 0.5, 0.1)
        assert best is not None
        assert -40.0 < best.threshold_db <= -38.0
        assert best.cuts == 5

    def test_a_short_clip_with_one_pause(self) -> None:
        """壊れると、切れる秒が 1 秒に届かない短い素材では値が出ない"""
        best, _nearby = choose_silence(
            _ticks((-12.0, 10), (-40.0, 9), (-12.0, 10)), _TICK, 0.5, 0.1
        )
        assert best is not None
        assert best.cuts == 1

    def test_a_clip_too_short_to_cut(self) -> None:
        """壊れると、切れない長さの素材にも値を勧める"""
        assert choose_silence(_ticks((-12.0, 2), (-40.0, 2)), _TICK, 0.5, 0.1) == (None, ())

    def test_one_tick(self) -> None:
        """壊れると、1 区切りだけの素材で差を取れずに落ちる"""
        assert choose_silence(_ticks((-12.0, 1)), _TICK, 0.5, 0.1) == (None, ())

    def test_all_room_tone_has_no_value(self) -> None:
        """壊れると、話の無い素材の全部を間と読み、全部を切れると勧める"""
        assert choose_silence(_ticks((-40.0, 300)), _TICK, 0.5, 0.1) == (None, ())

    def test_a_click_over_steady_music_has_no_value(self) -> None:
        """壊れると、1 度のクリック音を話と読み、鳴り続ける曲を全部間と読む"""
        peaks = _ticks((-20.0, 400), (0.0, 1), (-20.0, 400))
        assert choose_silence(peaks, _TICK, 0.5, 0.1) == (None, ())

    def test_digital_silence_with_steady_talk(self) -> None:
        """壊れると、話の音量が平らな素材で素の無音の間を勧められない"""
        peaks = _ticks(*[(-14.0, 30), (-120.0, 20)] * 5)
        best, _nearby = choose_silence(peaks, _TICK, 0.5, 0.1)
        assert best is not None
        assert best.threshold_db == -50.0

    def test_a_digital_gap_too_short_to_cut_has_no_value(self) -> None:
        """壊れると、切れない短い素の無音のために値を勧める"""
        assert choose_silence(_ticks((-14.0, 30), (-120.0, 3), (-14.0, 30)), _TICK, 0.5, 0.1) == (
            None,
            (),
        )

    def test_seconds_do_not_pass_the_end(self) -> None:
        """壊れると、波形の最後のピークの半端な分だけ見込みが長くなる"""
        peaks = _ticks((-12.0, 20), (-40.0, 20))
        best, _nearby = choose_silence(peaks, _TICK, 0.5, 0.1, total=3.5)
        assert best is not None
        assert best.seconds == pytest.approx(1.3)


class TestDigitalSilence:
    def test_clean_pauses_get_a_quiet_value(self) -> None:
        """壊れると、素の無音だけが間の素材で値を勧められない"""
        best, _nearby = choose_silence(_talk(-120.0), _TICK, 0.5, 0.1)
        assert best is not None
        assert best.threshold_db == -50.0
        assert best.cuts == 20

    def test_room_tone_beside_digital_silence_is_not_missed(self) -> None:
        """壊れると、素の無音だけを切る値を勧め、-40 dBFS の部屋の音の間が残る"""
        peaks = np.concatenate([_ticks((-120.0, 100)), _talk(-40.0)])
        best, _nearby = choose_silence(peaks, _TICK, 0.5, 0.1)
        assert best is not None
        assert -40.0 < best.threshold_db <= -38.0

    def test_short_digital_gaps_do_not_hide_the_pauses(self) -> None:
        """壊れると、短い素の無音があるだけで「切れる無音が無い」と返す"""
        peaks = np.concatenate(
            [_ticks((-120.0, 2), (-12.0, 5)) for _ in range(30)] + [_talk(-40.0)]
        )
        best, _nearby = choose_silence(peaks, _TICK, 0.5, 0.1)
        assert best is not None
        assert -40.0 < best.threshold_db <= -38.0

    def test_a_quiet_recording_keeps_its_speech(self) -> None:
        """壊れると、-50 dBFS より小さい録音の話し声まで無音と勧める"""
        quiet_talk = _ticks(*[(-60.0, 10), (-120.0, 10)] * 20)
        best, _nearby = choose_silence(quiet_talk, _TICK, 0.5, 0.1)
        assert best is not None
        assert best.threshold_db < -60.0


def _waveform(*runs: tuple[float, int]) -> Waveform:
    """（dBFS, ピークの数）の並びの波形 1 ピークは 10 ms"""
    amplitude = np.concatenate([np.full(count, db_to_ratio(level)) for level, count in runs])
    amplitude = amplitude.astype(np.float32)
    peaks = np.stack([-amplitude, amplitude], axis=-1)[:, np.newaxis, :]
    return Waveform(48000, 1, amplitude.size * 480, (PeakLevel(480, peaks),))


class TestFromTheWaveform:
    @pytest.mark.parametrize("offset", [0, 3, 5, 7, 9])
    def test_a_pause_across_the_ticks_is_not_lost(self, offset: int) -> None:
        """壊れると、0.1 秒の区切りをまたいだ 0.7 秒の間が短く読まれ、切れるのに値が出ない"""
        runs = [(-12.0, 100 + offset)]
        for _ in range(8):
            runs += [(-46.0, 70), (-12.0, 100)]
        waveform = _waveform(*runs)
        best, _nearby = suggest_silence(waveform, 0.5, 0.1)
        assert best is not None
        found = detect_silence(waveform, SilenceOptions(threshold_db=best.threshold_db))
        assert best.cuts == len(found) == 8
        assert best.seconds == pytest.approx(
            float(sum(end - start for start, end in found)), abs=0.1
        )

    def test_nothing_measured(self) -> None:
        """壊れると、長さの無い波形で落ちる"""
        peaks = np.zeros((0, 1, 2), dtype=np.float32)
        assert suggest_silence(Waveform(48000, 1, 0, (PeakLevel(480, peaks),)), 0.5, 0.1) == (
            None,
            (),
        )

    def test_a_waveform_of_pure_silence(self) -> None:
        """壊れると、全部が素の無音の素材で全体を切れると勧める"""
        assert suggest_silence(_waveform((-200.0, 500)), 0.5, 0.1) == (None, ())

    def test_a_short_last_pause_is_counted_by_its_real_length(self) -> None:
        """壊れると、末尾の半端な区切りを丸ごと数え、jet_cut が切らない間まで見込みに入る"""
        loud, quiet = db_to_ratio(-12.0), db_to_ratio(-46.0)
        levels = np.tile(np.repeat([loud, quiet], 100), 10)
        # 最後の間は 0.65 秒 余白を引くと 0.45 秒で最短の 0.5 秒に届かない
        amplitude = np.concatenate([levels, np.full(100, loud), np.full(65, quiet)])
        amplitude = amplitude.astype(np.float32)
        peaks = np.stack([-amplitude, amplitude], axis=-1)[:, np.newaxis, :]
        waveform = Waveform(48000, 1, amplitude.size * 480, (PeakLevel(480, peaks),))
        best, _nearby = suggest_silence(waveform, 0.5, 0.1)
        assert best is not None
        found = detect_silence(waveform, SilenceOptions(threshold_db=best.threshold_db))
        assert best.cuts == len(found)

    @pytest.mark.parametrize("pause_db", [-46.0, -30.0])
    def test_it_agrees_with_what_jet_cut_finds(self, pause_db: float) -> None:
        """壊れると、見込みの数と jet_cut が実際に切る数が食い違う"""
        levels = np.repeat([db_to_ratio(-12.0), db_to_ratio(pause_db)], 100)
        amplitude = np.tile(levels, 10).astype(np.float32)
        peaks = np.stack([-amplitude, amplitude], axis=-1)[:, np.newaxis, :]
        waveform = Waveform(48000, 1, 20 * 48000, (PeakLevel(480, peaks),))
        best, _nearby = suggest_silence(waveform, 0.5, 0.1)
        assert best is not None
        found = detect_silence(waveform, SilenceOptions(threshold_db=best.threshold_db))
        assert best.cuts == len(found)


class TestAskingAgain:
    """勧める値は波形ごとに覚える 画面のスレッドで動くので、呼ぶたびに数えると長い素材で止まる"""

    @pytest.fixture
    def counted(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        calls: list[float] = []
        real = silence.choose_silence

        def counting(peaks: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            calls.append(float(peaks.size))
            return real(peaks, *args, **kwargs)

        monkeypatch.setattr(silence, "choose_silence", counting)
        return calls

    def test_the_same_waveform_is_counted_once(self, counted: list[float]) -> None:
        """壊れると、analyze_audio を呼ぶたびに長い素材で画面が止まる"""
        waveform = _waveform(*[(-12.0, 100), (-46.0, 100)] * 10)
        first = suggest_silence(waveform, 0.5, 0.1)
        assert suggest_silence(waveform, 0.5, 0.1) == first
        assert len(counted) == 1

    def test_a_new_waveform_is_counted_again(self, counted: list[float]) -> None:
        """壊れると、素材が変わった後も前の素材の値を勧める"""
        loud = suggest_silence(_waveform(*[(-12.0, 100), (-30.0, 100)] * 10), 0.5, 0.1)
        quiet = suggest_silence(_waveform(*[(-12.0, 100), (-46.0, 100)] * 10), 0.5, 0.1)
        assert loud[0] is not None
        assert quiet[0] is not None
        assert loud[0].threshold_db != quiet[0].threshold_db
        assert len(counted) == 2

    def test_a_waveform_with_the_same_values_is_not_mixed_up(self, counted: list[float]) -> None:
        """壊れると、解析し直した波形を中身で比べて遅くなるか、ピークの配列で落ちる"""
        runs = [(-12.0, 100), (-46.0, 100)] * 10
        assert suggest_silence(_waveform(*runs), 0.5, 0.1) == suggest_silence(
            _waveform(*runs), 0.5, 0.1
        )
        assert len(counted) == 2

    def test_a_freed_waveform_does_not_answer_for_the_next(self, counted: list[float]) -> None:
        """壊れると、捨てた波形の番号を次の波形が引き継いだとき、前の素材の値を返す"""
        for floor in (-30.0, -46.0, -30.0, -46.0):
            waveform = _waveform(*[(-12.0, 100), (floor, 100)] * 10)
            best, _nearby = suggest_silence(waveform, 0.5, 0.1)
            assert best is not None
            assert floor < best.threshold_db <= floor + 2
            del waveform
            gc.collect()
        assert len(counted) == 4

    def test_another_length_is_counted_again(self, counted: list[float]) -> None:
        """壊れると、最短の無音や余白を変えても前の長さの見込みを返す"""
        waveform = _waveform(*[(-12.0, 100), (-46.0, 60)] * 10)
        short = suggest_silence(waveform, 0.3, 0.1)
        long = suggest_silence(waveform, 0.5, 0.1)
        padded = suggest_silence(waveform, 0.3, 0.2)
        assert short[0] is not None
        assert long[0] is None
        assert padded[0] is None or padded != short
        assert len(counted) == 3

    def test_no_value_is_remembered_too(self, counted: list[float]) -> None:
        """壊れると、勧める値の無い長い素材で呼ぶたびに数え直す"""
        waveform = _waveform((-14.0, 500))
        assert suggest_silence(waveform, 0.5, 0.1) == (None, ())
        assert suggest_silence(waveform, 0.5, 0.1) == (None, ())
        assert len(counted) == 1

    def test_the_memory_does_not_grow(self, counted: list[float]) -> None:
        """壊れると、素材を次々開いたときに覚えた値が増え続ける"""
        kept = [_waveform((-14.0, 20)) for _ in range(40)]
        for waveform in kept:
            suggest_silence(waveform, 0.5, 0.1)
        assert len(counted) == 40
        assert len(silence._SUGGESTED) <= silence._SUGGESTED_LIMIT
