"""無音カットの勧める値（#413 の 6 番） :func:`choose_silence` と波形からの読み方"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.core.mixing import db_to_ratio
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

    def test_steady_sound_has_no_value(self) -> None:
        """壊れると、鳴り続ける曲の音量を間と読み、曲の途中を切る"""
        assert choose_silence(_ticks((-14.0, 300)), _TICK, 0.5, 0.1) == (None, ())

    def test_pure_silence_has_no_value(self) -> None:
        """壊れると、全部が無音の素材で全体を切れると勧める"""
        assert choose_silence(_ticks((-120.0, 300)), _TICK, 0.5, 0.1) == (None, ())

    def test_nothing_measured_has_no_value(self) -> None:
        """壊れると、音の無い作品で道具ごと落ちる"""
        assert choose_silence(np.empty(0), _TICK, 0.5, 0.1) == (None, ())


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


class TestFromTheWaveform:
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
