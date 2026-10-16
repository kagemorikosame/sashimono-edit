"""ラウドネス（ITU-R BS.1770-4）を numpy で測る所（#337）

既知の信号で値が合うことを見る 規格の係数の表（48 kHz）と、EBU Tech 3341 の試験信号
（1 kHz の正弦波 -23 dBFS → -23.0 LUFS ±0.1）と同じ形の信号を使う
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.engine.checks.loudness import LoudnessMeter, k_weighting

RATE = 48000


def _sine(db: float, seconds: float, rate: int = RATE, channels: int = 2) -> np.ndarray:
    times = np.arange(round(seconds * rate)) / rate
    wave = 10.0 ** (db / 20.0) * np.sin(2.0 * np.pi * 1000.0 * times)
    return np.repeat(wave[:, None], channels, axis=1)


def _measure(*parts: np.ndarray, rate: int = RATE, block: int = 1600) -> float | None:
    signal = np.concatenate(parts)
    meter = LoudnessMeter(rate, signal.shape[1])
    for start in range(0, len(signal), block):
        meter.add(signal[start : start + block])
    return meter.integrated()


class TestKWeighting:
    def test_the_48k_coefficients_match_the_standard(self) -> None:
        # 壊れると、K 特性の形がずれ、高い音や低い音の多い作品で値が規格とずれる
        b1, a1, b2, a2 = k_weighting(48000)
        np.testing.assert_allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285])
        np.testing.assert_allclose(a1, [1.0, -1.69065929318241, 0.73248077421585])
        np.testing.assert_allclose(b2, [1.0, -2.0, 1.0])
        np.testing.assert_allclose(a2, [1.0, -1.99004745483398, 0.99007225036621])


class TestIntegrated:
    @pytest.mark.parametrize("rate", [48000, 44100])
    @pytest.mark.parametrize("db", [-23.0, -33.0, -14.0])
    def test_a_stereo_sine_reads_its_level(self, rate: int, db: float) -> None:
        # 壊れると、基準どおりの作品を外れていると言うか、外れた作品を基準どおりと言う
        assert _measure(_sine(db, 20.0, rate), rate=rate) == pytest.approx(db, abs=0.1)

    def test_quiet_parts_below_the_relative_gate_do_not_lower_the_value(self) -> None:
        # 壊れると、話の合間の静かな所で全体の値が下がり、実際より小さいと言う
        # （EBU Tech 3341 の 3 番 -36 / -23 / -36 dBFS を 10 / 60 / 10 秒）
        measured = _measure(_sine(-36.0, 10.0), _sine(-23.0, 60.0), _sine(-36.0, 10.0))
        assert measured == pytest.approx(-23.0, abs=0.1)

    def test_silence_below_the_absolute_gate_is_ignored(self) -> None:
        # 壊れると、無音の長い作品ほど値が下がり、無音を足しただけで基準から外れる
        # 鳴り始めと鳴り終わりをまたぐ区切りは規格どおり数えるので、少しだけ下がる
        # 鳴る所を長く取って、その分を 0.02 LU ほどに抑える
        silent = np.zeros((RATE * 30, 2))
        measured = _measure(silent, _sine(-23.0, 60.0), silent)
        assert measured == pytest.approx(-23.0, abs=0.1)

    def test_the_relative_gate_counts_parts_within_10_lu(self) -> None:
        # 壊れると、相対ゲートの幅（-10 LU）が違い、静かめの所を数えるかどうかが規格と食い違う
        # -23 を 60 秒と -32.5 を 10 秒 平均はおよそ -23.6 で、-32.5 は平均の 8〜10 LU 下にある
        # -10 LU のゲートなら数え、古い草案の -8 LU なら捨てる
        measured = _measure(_sine(-23.0, 60.0), _sine(-32.5, 10.0))
        expected = 10.0 * np.log10((60 * 10 ** (-23 / 10) + 10 * 10 ** (-32.5 / 10)) / 70)
        assert measured == pytest.approx(expected, abs=0.1)

    def test_the_value_does_not_depend_on_how_the_sound_is_handed_over(self) -> None:
        # 壊れると、塊の切れ目でフィルタの続きが途切れ、渡し方で値が変わる
        signal = _sine(-20.0, 5.0) * np.linspace(0.2, 1.0, RATE * 5)[:, None]
        values = {round(_measure(signal, block=block) or 0.0, 6) for block in (1, 997, 1600, 48000)}
        assert len(values) == 1

    def test_too_short_or_silent_sound_has_no_value(self) -> None:
        # 壊れると、400 ms に満たない音や無音で -inf や 0 を返し、AI が大きく外れたと言う
        assert _measure(_sine(-23.0, 0.3)) is None
        assert _measure(np.zeros((RATE * 2, 2))) is None

    def test_sound_below_the_absolute_gate_has_no_value(self) -> None:
        # 壊れると、-70 LUFS より静かな雑音だけの作品に値が付き、ほぼ無音なのに音量を測れたと言う
        assert _measure(_sine(-75.0, 5.0)) is None


class TestChannels:
    def test_surround_rear_channels_weigh_more_and_lfe_is_ignored(self) -> None:
        # 壊れると、5.1 の作品で後ろの音と LFE の数え方が規格と違い、値がずれる
        wave = _sine(-23.0, 10.0, channels=1)[:, 0]
        front = np.zeros((len(wave), 6))
        front[:, 0] = wave
        rear = np.zeros((len(wave), 6))
        rear[:, 4] = wave
        lfe = np.zeros((len(wave), 6))
        lfe[:, 3] = wave
        assert (_measure(rear) or 0.0) - (_measure(front) or 0.0) == pytest.approx(
            10 * np.log10(1.41), abs=0.05
        )
        assert _measure(lfe) is None

    def test_the_peak_is_the_largest_sample(self) -> None:
        # 壊れると、要約の最大のサンプルが違い、音割れの手前かが読めない
        meter = LoudnessMeter(RATE, 2)
        signal = _sine(-6.0, 1.0)
        signal[100, 1] = -0.9
        meter.add(signal)
        assert meter.peak == pytest.approx(0.9)
