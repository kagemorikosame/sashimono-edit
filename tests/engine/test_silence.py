"""無音検出

ピークから作るので実素材は要らないが、実素材で作った波形でも同じ結果が出ることを
最後に 1 つだけ確かめる
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.model import Transcript, TranscriptSegment
from sashimono.engine.audio.silence import SilenceOptions, detect_silence, keep_speech
from sashimono.engine.audio.waveform import (
    BASE_SAMPLES_PER_PEAK,
    PeakLevel,
    Waveform,
    analyze_waveform,
)
from tests.media_fixtures import make_silent_gap

SAMPLE_RATE = 48000
#: 1 ピークが何秒か 48kHz / 256 サンプルなので約 5.3ms
PEAK_SECONDS = Fraction(BASE_SAMPLES_PER_PEAK, SAMPLE_RATE)


def make_waveform(pattern: list[tuple[float, int]]) -> Waveform:
    """``(振幅, ピーク数)`` の並びから波形を作る

    秒ではなくピーク数で書く 1 ピークは 48kHz / 256 サンプルで割り切れない秒数に
    なるので、秒で書くと境界が量子化されて期待値がずれる
    """
    blocks = []
    for amplitude, count in pattern:
        block = np.zeros((count, 1, 2), dtype=np.float32)
        block[:, :, 0] = -amplitude
        block[:, :, 1] = amplitude
        blocks.append(block)
    peaks = np.concatenate(blocks, axis=0)
    return Waveform(
        sample_rate=SAMPLE_RATE,
        channels=1,
        total_samples=peaks.shape[0] * BASE_SAMPLES_PER_PEAK,
        levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),),
    )


def at(peaks: int) -> Fraction:
    """ピーク数を秒へ 200 ピークで約 1.07 秒"""
    return peaks * PEAK_SECONDS


class TestDetectSilence:
    def test_a_quiet_stretch_between_speech_is_found(self) -> None:
        # 話の間の無音を見つけ損ねると、無音を詰めても間が残り、境目がずれると話の頭や終わりが削れる
        waveform = make_waveform([(0.5, 200), (0.0, 400), (0.5, 200)])
        options = SilenceOptions(padding=Fraction(0))
        assert detect_silence(waveform, options) == ((at(200), at(600)),)

    def test_padding_shrinks_the_range_on_both_sides(self) -> None:
        # 発話の立ち上がりを削らないよう、無音の内側へ余白を取る
        waveform = make_waveform([(0.5, 200), (0.0, 400), (0.5, 200)])
        options = SilenceOptions(padding=at(50))
        assert detect_silence(waveform, options) == ((at(250), at(550)),)

    def test_short_gaps_are_ignored(self) -> None:
        # 息継ぎまで切ると聞いていて忙しい
        waveform = make_waveform([(0.5, 200), (0.0, 50), (0.5, 200)])
        options = SilenceOptions(padding=Fraction(0), min_silence=at(100))
        assert detect_silence(waveform, options) == ()

    def test_the_threshold_decides_what_counts_as_quiet(self) -> None:
        # -40dB は約 0.01 0.05 は無音ではないが、-20dB（0.1）以下ではある
        waveform = make_waveform([(0.5, 200), (0.05, 400), (0.5, 200)])
        quiet = SilenceOptions(padding=Fraction(0), threshold_db=-40.0)
        loose = SilenceOptions(padding=Fraction(0), threshold_db=-20.0)
        assert detect_silence(waveform, quiet) == ()
        assert detect_silence(waveform, loose) == ((at(200), at(600)),)

    def test_silence_at_the_head_and_tail_is_found(self) -> None:
        # 頭と終わりの無音を落とすと、録画を始めてから話し出すまでの間が詰められずに残る
        waveform = make_waveform([(0.0, 200), (0.5, 200), (0.0, 200)])
        options = SilenceOptions(padding=Fraction(0))
        assert detect_silence(waveform, options) == ((at(0), at(200)), (at(400), at(600)))

    def test_a_channel_that_is_still_sounding_prevents_silence(self) -> None:
        # 片方だけ鳴っている区間を無音と判定してはいけない
        peaks = np.zeros((400, 2, 2), dtype=np.float32)
        peaks[:, 0, 1] = 0.5
        waveform = Waveform(
            sample_rate=SAMPLE_RATE,
            channels=2,
            total_samples=400 * BASE_SAMPLES_PER_PEAK,
            levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),),
        )
        assert detect_silence(waveform, SilenceOptions(padding=Fraction(0))) == ()

    def test_short_speech_islands_can_be_absorbed(self) -> None:
        # 既定でまとめると間の短い一言まで切り落とし、指定してもまとめないと
        # 咳払いを挟んだ長い間が 2 つに割れて残る
        waveform = make_waveform([(0.0, 200), (0.5, 20), (0.0, 200)])
        # 既定では残す 舌打ち 1 つのために発話を消す方が事故になる
        keep = SilenceOptions(padding=Fraction(0))
        assert len(detect_silence(waveform, keep)) == 2
        # 明示的に指定したときだけ、間の短い音ごと 1 つにまとめる
        absorb = SilenceOptions(padding=Fraction(0), min_keep=at(40))
        assert detect_silence(waveform, absorb) == ((at(0), at(420)),)

    def test_an_empty_waveform_yields_nothing(self) -> None:
        # 長さ 0 の波形で空を返せないと、長さ 0 の音声の素材に無音カットを掛けたとき
        # 字幕パネルと AI の道具が「切る所が無い」と返せずに止まる
        empty = Waveform(
            sample_rate=SAMPLE_RATE,
            channels=1,
            total_samples=0,
            levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, np.zeros((0, 1, 2), dtype=np.float32)),),
        )
        assert detect_silence(empty) == ()

    def test_invalid_options_are_rejected(self) -> None:
        # 最短無音長 0 を黙って受けると、AI の無音カットが min_silence に 0 を渡したとき
        # （画面は 0.05 秒が下限） 言葉の間のほんの一瞬まで切り、話が細切れに詰められる
        with pytest.raises(ValueError, match="最短無音長"):
            SilenceOptions(min_silence=Fraction(0))


class TestKeepSpeech:
    def test_ranges_covered_by_speech_are_dropped(self) -> None:
        # 波形が静かでも、認識器が文字を起こせた区間は発話 切らない
        transcript = Transcript(
            (TranscriptSegment(Fraction(1), Fraction(2), "小さい声でも喋っている"),)
        )
        kept = keep_speech(((Fraction(1), Fraction(2)),), transcript, margin=Fraction(0))
        assert kept == ()

    def test_a_range_around_speech_is_split_in_two(self) -> None:
        # 発話を避けて 2 つに割らないと、無音の中の小さな一言ごと切るか、前後の無音も詰めずに残す
        transcript = Transcript((TranscriptSegment(Fraction(2), Fraction(3), "喋り"),))
        kept = keep_speech(((Fraction(0), Fraction(5)),), transcript, margin=Fraction(0))
        assert kept == ((Fraction(0), Fraction(2)), (Fraction(3), Fraction(5)))

    def test_the_margin_widens_what_is_protected(self) -> None:
        # 余白を広げないと、文字起こしの時刻ぎりぎりで切り、話の頭と終わりの音が欠ける
        transcript = Transcript((TranscriptSegment(Fraction(2), Fraction(3), "喋り"),))
        kept = keep_speech(((Fraction(0), Fraction(5)),), transcript, margin=Fraction(1, 2))
        assert kept == ((Fraction(0), Fraction(3, 2)), (Fraction(7, 2), Fraction(5)))

    def test_an_empty_transcript_changes_nothing(self) -> None:
        # 文字起こしが無いのに範囲を減らすと、まだ文字起こししていない動画では
        # 無音を詰めても何も切れない
        ranges = ((Fraction(0), Fraction(1)),)
        assert keep_speech(ranges, Transcript()) == ranges


class TestWithRealMedia:
    def test_a_generated_gap_is_detected(self, media_dir: Path) -> None:
        # 3 秒の素材で、1..2 秒が無音
        path = make_silent_gap(media_dir, "gap.wav", duration=3.0)
        waveform = analyze_waveform(path, sample_rate=48000, channels=2)
        assert waveform is not None

        found = detect_silence(
            waveform, SilenceOptions(padding=Fraction(1, 20), min_silence=Fraction(1, 4))
        )
        assert len(found) >= 1
        # 素材は 1..2 秒が無音 余白の分だけ内側に寄る
        start, end = found[0]
        assert Fraction(1) <= start <= Fraction(11, 10)
        assert Fraction(19, 10) <= end <= Fraction(2)
