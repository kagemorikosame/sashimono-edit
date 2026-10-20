"""音の推移（#322）の測り方 ラウドネスの推移・幅・無音・話し声と音楽の見分け

音は float の配列を直に渡す 見分けの音は :mod:`tests.check_media` の話し声らしい音と和音
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.engine.checks import profile as profile_module
from sashimono.engine.checks.loudness import LoudnessMeter
from sashimono.engine.checks.profile import (
    MUSIC,
    NOISE,
    SILENCE,
    SPEECH,
    SPEECH_AND_MUSIC,
    SoundProfile,
    SoundProfiler,
    classify,
)
from sashimono.engine.checks.scan import LOUDNESS, AudioScan, CheckLimits
from tests.check_media import RATE, chord, talk, tone


def _profile(samples: np.ndarray, *, block: int = 1600) -> SoundProfile:
    profiler = SoundProfiler(RATE, samples.shape[1])
    for start in range(0, len(samples), block):
        profiler.add(samples[start : start + block])
    return profiler.snapshot()


def _labels(profile: SoundProfile, subtitles: np.ndarray | None = None) -> list[str]:
    return [segment.label for segment in classify(profile, subtitles=subtitles)]


class TestLoudness:
    def test_the_summary_matches_the_check(self) -> None:
        # 壊れると、同じ作品で check_output の積分ラウドネスと analyze_audio の要約が食い違い、
        # AI がどちらを信じて音量をそろえればよいか分からなくなる
        samples = np.concatenate([tone(4.0, -20.0), tone(3.0, -30.0)])
        meter = LoudnessMeter(RATE, 2)
        meter.add(samples)
        assert _profile(samples).integrated() == pytest.approx(meter.integrated(), abs=1e-9)

    def test_any_block_size_gives_the_same_curve(self) -> None:
        # 壊れると、書き出した動画（大きな塊）とタイムライン（1 コマずつ）で推移がずれ、
        # 書き出す前と後を比べた AI が、変わっていない所を変わったと読む
        samples = np.concatenate([talk(3.0), chord(3.0)])
        small = _profile(samples, block=1600)
        large = _profile(samples, block=48000 * 2 + 77)
        np.testing.assert_allclose(small.momentary(), large.momentary(), atol=1e-9)
        np.testing.assert_allclose(small.flatness, large.flatness, atol=1e-12)

    def test_the_momentary_loudness_of_a_reference_tone(self) -> None:
        # 壊れると、推移の値が規格（EBU Tech 3341 の 1 kHz -23 dBFS は -23 LUFS）からずれ、
        # 基準どおりの作品を大きすぎる・小さすぎると返す
        profile = _profile(tone(5.0, -23.0))
        assert profile.momentary()[10:] == pytest.approx(-23.0, abs=0.1)
        assert profile.short_term()[40:] == pytest.approx(-23.0, abs=0.1)

    def test_the_loudness_range_of_two_levels(self) -> None:
        # 壊れると、静かな所と大きな所の差（LRA）を読み違え、BGM と声の差が大きすぎる作品を
        # 見逃す EBU Tech 3342 の試験 1（-20 と -30 dBFS を 20 秒ずつ）は 10 LU
        profile = _profile(np.concatenate([tone(20.0, -20.0), tone(20.0, -30.0)]))
        assert profile.loudness_range() == pytest.approx(10.0, abs=1.0)

    def test_a_short_sound_has_no_range(self) -> None:
        # 壊れると、3 秒に満たない音でも幅を出し、規格の外の値を基準どおりのように返す
        assert _profile(tone(2.0, -20.0)).loudness_range() is None


class TestSilence:
    def test_a_gap_is_found_to_the_tenth_of_a_second(self) -> None:
        # 壊れると、話の間の位置が 1 秒単位でずれ、そこで切った AI が言葉の頭を削る
        samples = tone(5.0, -20.0, silent=(1.5, 2.3))
        ((start, end, loudest),) = _profile(samples).silences(-50.0, 3)
        assert (start, end) == (15, 23)
        assert loudest == -120.0

    def test_short_gaps_are_left_out(self) -> None:
        # 壊れると、音節の切れ目まで無音として並べ、一覧が埋まって本当の間を読めない
        samples = tone(5.0, -20.0, silent=(1.5, 1.7))
        assert _profile(samples).silences(-50.0, 5) == []


class TestKinds:
    def test_talking_is_speech(self) -> None:
        # 壊れると、ナレーションの所を音楽と読み、BGM を下げる所（ダッキング）を逆に選ぶ
        assert set(_labels(_profile(talk(12.0)))) == {SPEECH}

    def test_a_steady_chord_is_music(self) -> None:
        # 壊れると、BGM だけの所を話し声と読み、字幕を起こす所や話を切る所に選ぶ
        assert set(_labels(_profile(chord(12.0)))) == {MUSIC}

    def test_a_pulsing_chord_is_still_music(self) -> None:
        # 壊れると、拍ごとに大きさが上下する曲（実際の曲で多い）を、谷の多さだけで話し声寄りに
        # 読む 谷が無音まで落ちない（大きな所との差が 8 dB ほど）ので音楽と分かるはず
        samples = chord(12.0, level=0.2)
        times = np.arange(len(samples)) / RATE
        beat = 1.0 - 0.6 * ((times % 0.5) > 0.25)
        assert set(_labels(_profile((samples * beat[:, None]).astype(np.float32)))) == {MUSIC}

    def test_talking_over_a_bed_is_both(self) -> None:
        # 壊れると、BGM を敷いたままの語りを声だけと読み、BGM が声を邪魔しているかを確かめない
        samples = talk(12.0) + chord(12.0, level=0.03)
        assert set(_labels(_profile(samples))) == {SPEECH_AND_MUSIC}

    def test_noise_is_not_music(self) -> None:
        # 壊れると、空調やホワイトノイズの所を音楽と読み、BGM を差し替える所に選ぶ
        samples = np.random.default_rng(1).normal(size=(10 * RATE, 2)) * 0.05
        assert set(_labels(_profile(samples))) == {NOISE}

    def test_quiet_is_silence_with_high_confidence(self) -> None:
        # 壊れると、何も鳴っていない所に見分けの確からしさを付けず、AI が無音を確かめ直す
        segments = classify(_profile(np.zeros((3 * RATE, 2))))
        assert [(segment.label, segment.confidence) for segment in segments] == [(SILENCE, "high")]

    def test_subtitles_make_speech_certain(self) -> None:
        # 壊れると、字幕（認識器が文字を起こせた所）があっても音の形だけで決め、確からしさが
        # 上がらない 字幕のある所は話していると言い切れる
        profile = _profile(talk(6.0))
        covered = np.ones(profile.ticks, dtype=bool)
        (segment,) = classify(profile, subtitles=covered)
        assert (segment.label, segment.confidence, segment.subtitles) == (SPEECH, "high", 1.0)

    def test_a_single_odd_second_does_not_split_a_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、話の途中の 1 秒の盛り上がりで区間が割れ、一覧が細切れになって読めない
        # 1 秒ごとの見分けを決めておき、まとめ方だけを見る（音で 1 秒だけを違う見分けにすると、
        # 前後 1 秒も見るので隣の秒まで引きずられる）
        seconds = [SPEECH, SPEECH, MUSIC, SPEECH, SPEECH, SILENCE, SPEECH, NOISE, NOISE, SPEECH]

        def decided(_profile: SoundProfile, _levels: object, start: int, *_rest: object) -> object:
            return profile_module._Window(seconds[start // 10], "medium", 0.5, 0.5, None)

        monkeypatch.setattr(profile_module, "_window", decided)
        segments = classify(_profile(np.zeros((10 * RATE, 2))))
        assert [(s.start // 10, s.end // 10, s.label) for s in segments] == [
            (0, 5, SPEECH),
            (5, 6, SILENCE),
            (6, 7, SPEECH),
            (7, 9, NOISE),
            (9, 10, SPEECH),
        ]

    @pytest.mark.parametrize(
        ("confidences", "expected"),
        [
            (["high", "medium"], "medium"),
            (["medium", "low"], "low"),
            (["high", "high", "high", "medium"], "high"),
            (["high", "low"], "medium"),
        ],
    )
    def test_a_tie_in_confidence_goes_down(
        self, confidences: list[str], expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、字幕で裏付けた秒と音の形だけの秒が半々の区間を high と返し、AI が
        # 確かめずに切ってよい所と読む（丸めが偶数寄りで、半々が上にも下にも転ぶ）
        def decided(_profile: SoundProfile, _levels: object, start: int, *_rest: object) -> object:
            return profile_module._Window(SPEECH, confidences[start // 10], 0.5, 0.5, None)

        monkeypatch.setattr(profile_module, "_window", decided)
        samples = np.zeros((len(confidences) * RATE, 2))
        ((segment),) = classify(_profile(samples))
        assert segment.confidence == expected

    def test_the_levels_are_made_once_for_a_long_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、1 秒ごとに作品全体の dB を作り直し、2 時間の作品で見分けに数秒かかって
        # analyze_audio と audio_graph のたびに編集画面が止まる（長さの二乗で重くなる）
        ticks = 2 * 3600 * 10
        loud = np.repeat(np.random.default_rng(0).integers(0, 2, ticks // 10), 10) * 0.1 + 0.01
        profile = SoundProfile(energy=loud**2, rms=loud, peak=loud, flatness=np.full(ticks, 0.1))
        made: list[int] = []
        original = SoundProfile.rms_db

        def counted(self: SoundProfile) -> np.ndarray:
            made.append(1)
            return original(self)

        monkeypatch.setattr(SoundProfile, "rms_db", counted)
        classify(profile)
        assert len(made) == 1

    def test_silence_between_sounds_is_kept(self) -> None:
        # 壊れると、話の間の 1 秒の無音を前後の話し声に飲み込み、切る所の候補から消える
        samples = np.concatenate([talk(4.0), np.zeros((RATE, 2), np.float32), talk(4.0, seed=2)])
        assert SILENCE in _labels(_profile(samples))


class TestScan:
    def test_the_check_keeps_the_curve_even_without_loudness(self) -> None:
        # 壊れると、無音だけを調べた検査（check_output の checks 絞り）の後で音の道具を呼ぶと、
        # 推移が無いので音を混ぜ直す（または推移が空のまま返る）
        scan = AudioScan(
            CheckLimits(kinds=frozenset({"silence"})), sample_rate=RATE, channels=2, fps=30
        )
        scan.add(tone(2.0, -20.0))
        assert scan.loudness() is None
        assert scan.profile().ticks == 20

    def test_the_check_loudness_still_follows_the_request(self) -> None:
        # 壊れると、推移を足したことで積分ラウドネスの値が変わり、前からの検査の答えがずれる
        scan = AudioScan(
            CheckLimits(kinds=frozenset({LOUDNESS})), sample_rate=RATE, channels=2, fps=30
        )
        scan.add(tone(3.0, -14.0))
        assert scan.loudness() == pytest.approx(-14.0, abs=0.05)
