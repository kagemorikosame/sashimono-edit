"""音のノイズ除去（#336）と、波形からの雑音の見積もり

ノイズ除去は前の音だけを読む ミキサは塊を細かく切って頼むので、切り方を変えても同じ音に
なることと、雑音を下げて音（ここでは 440 Hz の音）を残すことを、数で確かめる
"""

from __future__ import annotations

import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import AddEffect, Document, insert_media
from sashimono.core.model import Project, ProjectSettings, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.audio import DENOISE_KIND, AudioContext
from sashimono.effects.denoise import denoise, denoise_history
from sashimono.engine.audio import AudioMixer
from sashimono.engine.audio.noise import estimate_noise
from sashimono.engine.audio.waveform import BASE_SAMPLES_PER_PEAK, PeakLevel, Waveform
from sashimono.engine.decode import probe_media

RATE = 48000
#: 雑音の実効値 -34 dBFS ほど 録音のサーとしては大きめ
NOISE = 0.02


def _gated_tone(seconds: float, *, noise: float = NOISE, seed: int = 1) -> np.ndarray:
    """0.7 Hz で鳴ったり止んだりする 440 Hz の音に、白い雑音を足したモノラル

    止んでいる所が雑音だけの所 話し声の息継ぎに当たる
    """
    t = np.arange(int(seconds * RATE)) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    hiss = noise * np.random.default_rng(seed).standard_normal(len(t))
    return np.asarray(tone + hiss, dtype=np.float32)


def _windows(samples: np.ndarray, window: int, **values: float) -> np.ndarray:
    """ミキサと同じく、``window`` ずつ前の音を読み直して掛ける"""
    stereo = np.repeat(samples[:, None], 2, axis=1)
    reach = int(np.ceil(denoise_history({}) * RATE))
    out = np.zeros_like(stereo)
    for start in range(0, len(stereo), window):
        end = min(start + window, len(stereo))
        before = min(start, reach)
        context = AudioContext(start - before, RATE, len(stereo), keep_from=before)
        out[start:end] = denoise(stereo[start - before : end], dict(values), context)[before:]
    return out[:, 0]


def _level_db(samples: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2))))


def _tone_amplitude(samples: np.ndarray, start: int) -> float:
    """``start`` から 1024 標本の 440 Hz の振れ幅 位相のずれは数えない"""
    t = (np.arange(1024) + start) / RATE
    part = samples[start : start + 1024].astype(np.float64)
    sine = float(np.mean(part * np.sin(2 * np.pi * 440 * t)) * 2)
    cosine = float(np.mean(part * np.cos(2 * np.pi * 440 * t)) * 2)
    return float(np.hypot(sine, cosine))


class TestTheEffect:
    def test_the_hiss_goes_down_by_the_asked_amount(self) -> None:
        # 壊れると、ノイズ除去を足しても雑音がほとんど減らないか、頼んだより大きく削れる
        samples = _gated_tone(5.0)
        gate = np.sin(2 * np.pi * 0.7 * np.arange(len(samples)) / RATE)
        quiet = (gate < -0.2) & (np.arange(len(samples)) > RATE)
        for reduction in (6.0, 12.0, 24.0):
            out = _windows(samples, 16384, reduction=reduction)
            dropped = _level_db(out[quiet]) - _level_db(samples[quiet])
            assert dropped == pytest.approx(-reduction, abs=reduction * 0.3 + 0.5)

    def test_the_tone_keeps_its_strength(self) -> None:
        # 壊れると、雑音と一緒に声（音）まで小さくなり、ノイズ除去を掛けた所だけ声が遠のく
        samples = _gated_tone(5.0)
        out = _windows(samples, 16384, reduction=24.0)
        gate = np.sin(2 * np.pi * 0.7 * np.arange(len(samples)) / RATE)
        loud = [
            start
            for start in range(int(1.5 * RATE), len(samples) - 1024, 512)
            if (gate[start : start + 1024] > 0.2).all()
        ]
        amplitudes = [_tone_amplitude(out, start) / 0.3 for start in loud]
        # 1 dB（0.89 倍）より小さい揺れは聞き分けられない 下げる量（24 dB）よりずっと小さい
        assert min(amplitudes) > 0.89
        assert max(amplitudes) < 1.06
        assert float(np.mean(amplitudes)) == pytest.approx(1.0, abs=0.03)

    def test_cutting_the_blocks_differently_hears_the_same(self) -> None:
        # 壊れると、プレビュー（細かい塊）と書き出し（大きい塊）で音が変わり、ミキサの区切り
        # （0.34 秒）ごとに段が出る 先の音を読む作りにするとこうなる
        samples = _gated_tone(3.0)
        assert np.array_equal(_windows(samples, 16384), _windows(samples, 1000))

    def test_zero_leaves_the_sound_alone(self) -> None:
        # 壊れると、下げる量を 0 にしても音が変わり、切ったつもりで音が曇る
        samples = np.repeat(_gated_tone(1.0)[:, None], 2, axis=1)
        context = AudioContext(0, RATE, len(samples))
        assert denoise(samples, {"reduction": 0.0}, context) is samples

    def test_a_clean_sound_passes_through(self) -> None:
        # 壊れると、雑音の無い音（打ち込みの音楽・合成音声）にまで掛けて音が変わる
        samples = _gated_tone(3.0, noise=0.0)
        out = _windows(samples, 16384)
        assert np.abs(out[RATE:] - samples[RATE:]).max() < 1e-6

    def test_the_head_of_the_clip_is_not_eaten(self) -> None:
        # 壊れると、クリップの頭で雑音の見積もりがまだ無く、話し始めの声を雑音とみなして
        # 頭の一言が消える 見積もりが育つまで下げる量を絞る
        samples = _gated_tone(1.0)
        out = _windows(samples, 16384, reduction=24.0)
        head = slice(0, int(0.02 * RATE))
        assert np.abs(out[head] - samples[head]).max() < 0.02

    def test_the_mixer_hears_the_same_in_small_and_large_blocks(self, tmp_path: Path) -> None:
        # 壊れると、ミキサが前の音を読み直す長さが足りず、プレビューと書き出しで音が変わる
        project = _project(_wav(tmp_path / "hiss.wav", _gated_tone(2.0)))
        mixer = AudioMixer(project)
        try:
            whole = mixer.render(0, 2 * RATE)
        finally:
            mixer.close()
        mixer = AudioMixer(project)
        try:
            parts = [mixer.render(start, 1024) for start in range(0, 2 * RATE, 1024)]
        finally:
            mixer.close()
        assert np.allclose(np.concatenate(parts)[: 2 * RATE], whole, atol=1e-6)

    def test_it_is_listed_with_the_sound_effects(self) -> None:
        # 壊れると、人がエフェクトの一覧からノイズ除去を足せない（AI だけの物になる）
        definition = registry.require(DENOISE_KIND)
        assert definition.category == "音"
        assert definition.audio_history is not None


def _wav(path: Path, samples: np.ndarray) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return path


def _project(path: Path) -> Project:
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(path)):
        document.execute(command)
    clip = next(
        c
        for t in document.project.timeline.tracks
        if t.kind in (TrackKind.AUDIO, TrackKind.MIXED)
        for c in t.clips
    )
    document.execute(AddEffect(clip.id, registry.require(DENOISE_KIND).create()))
    return document.project


def _peaks(amplitudes: np.ndarray) -> Waveform:
    """山の大きさの並びから波形を作る 1 つが 256 標本"""
    peaks = np.zeros((len(amplitudes), 1, 2), dtype=np.float32)
    peaks[:, 0, 0] = -amplitudes
    peaks[:, 0, 1] = amplitudes
    return Waveform(
        sample_rate=RATE,
        channels=1,
        total_samples=len(amplitudes) * BASE_SAMPLES_PER_PEAK,
        levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),),
    )


class TestTheEstimate:
    def test_hiss_between_words_is_measured(self) -> None:
        # 壊れると、話の合間に聞こえるサーを測れず、AI がノイズ除去を掛けない
        steps = int(10 * RATE / BASE_SAMPLES_PER_PEAK)
        amplitudes = np.full(steps, 0.01, dtype=np.float32)  # -40 dBFS の雑音
        amplitudes[: steps // 2] = 0.5  # 半分は話している
        estimate = estimate_noise(_peaks(amplitudes), Fraction(0), Fraction(10))
        assert estimate is not None
        assert estimate.noise_db == pytest.approx(-40.0, abs=0.5)
        assert estimate.audible and estimate.has_quiet_parts

    def test_a_quiet_room_is_left_alone(self) -> None:
        # 壊れると、雑音の聞こえない録音にまで掛けて、声の小さい所を削る
        steps = int(10 * RATE / BASE_SAMPLES_PER_PEAK)
        amplitudes = np.full(steps, 10 ** (-75 / 20), dtype=np.float32)
        amplitudes[: steps // 2] = 0.5
        estimate = estimate_noise(_peaks(amplitudes), Fraction(0), Fraction(10))
        assert estimate is not None
        assert not estimate.audible

    def test_music_without_pauses_is_told_apart(self) -> None:
        # 壊れると、黙る所の無い音楽を雑音とみなし、BGM ごと削る
        steps = int(10 * RATE / BASE_SAMPLES_PER_PEAK)
        amplitudes = np.random.default_rng(0).uniform(0.3, 0.6, steps).astype(np.float32)
        estimate = estimate_noise(_peaks(amplitudes), Fraction(0), Fraction(10))
        assert estimate is not None
        assert not estimate.has_quiet_parts

    def test_a_singing_voice_alone_is_told_apart(self) -> None:
        # 壊れると、歌の声だけを抜き出した物（息継ぎでも残響が残り、静かな所と大きな所の差が
        # 14 dB ほど）を雑音の乗った声と見て、声を削る（本人の手元の実素材で起きた）
        steps = int(10 * RATE / BASE_SAMPLES_PER_PEAK)
        amplitudes = np.full(steps, 10 ** (-26 / 20), dtype=np.float32)
        amplitudes[(np.arange(steps) // 8) % 2 == 0] = 10 ** (-12 / 20)
        estimate = estimate_noise(_peaks(amplitudes), Fraction(0), Fraction(10))
        assert estimate is not None
        assert not estimate.has_quiet_parts

    def test_only_the_range_of_the_clip_is_read(self) -> None:
        # 壊れると、クリップに使っていない所（トリムで外した雑音の多い所）で決めてしまう
        steps = int(10 * RATE / BASE_SAMPLES_PER_PEAK)
        amplitudes = np.full(steps, 0.2, dtype=np.float32)
        amplitudes[: steps // 2] = 10 ** (-80 / 20)
        early = estimate_noise(_peaks(amplitudes), Fraction(0), Fraction(4))
        late = estimate_noise(_peaks(amplitudes), Fraction(6), Fraction(10))
        assert early is not None and late is not None
        assert early.noise_db < -70 < late.noise_db

    def test_too_short_is_not_measured(self) -> None:
        # 壊れると、黙る所の無い短いクリップの声を雑音と見誤る
        steps = int(10 * RATE / BASE_SAMPLES_PER_PEAK)
        amplitudes = np.full(steps, 0.2, dtype=np.float32)
        assert estimate_noise(_peaks(amplitudes), Fraction(0), Fraction(1, 2)) is None
