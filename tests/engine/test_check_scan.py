"""完成物の検査の測り方（#337 黒・止まり・無音・音割れ）と、程度の付け方

絵は輝度の配列、音は float の配列を直に渡す 描かずに決まりの境目を細かく見る
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.engine.checks.report import HIGH, LOW, MEDIUM, assemble
from sashimono.engine.checks.scan import (
    BLACK,
    CLIPPING,
    FREEZE,
    SILENCE,
    AudioScan,
    CheckLimits,
    Run,
    VideoScan,
)

FPS = 30.0
RATE = 48000


def _picture(seed: int, level: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (level + 0.2 * rng.standard_normal((18, 32))).clip(0, 1).astype(np.float32)


def _video_runs(frames: list[np.ndarray], limits: CheckLimits | None = None) -> list[Run]:
    scan = VideoScan(limits or CheckLimits(), FPS)
    for index, luma in enumerate(frames):
        scan.add(index, luma)
    return scan.runs(final=True)


def _audio(
    samples: np.ndarray, limits: CheckLimits | None = None, *, block: int = 1600
) -> AudioScan:
    scan = AudioScan(limits or CheckLimits(), sample_rate=RATE, channels=2, fps=FPS)
    for start in range(0, len(samples), block):
        scan.add(samples[start : start + block])
    scan.finish()
    return scan


def _tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    times = np.arange(round(seconds * RATE)) / RATE
    wave = amplitude * np.sin(2 * np.pi * 440 * times)
    return np.stack([wave, wave], axis=1)


class TestBlack:
    def test_a_single_black_frame_between_shots_is_found(self) -> None:
        # 壊れると、つなぎ目に 1 コマだけ挟まった黒（ちらつき）を見落とす
        frames = [_picture(i) for i in range(10)]
        frames[5] = np.zeros_like(frames[5])
        (run,) = [run for run in _video_runs(frames) if run.kind == BLACK]
        assert (run.start, run.end) == (5, 6)

    def test_a_dark_scene_with_a_few_bright_pixels_is_still_black(self) -> None:
        # 壊れると、隅にロゴが 1 つ載っただけの黒い画面を黒と見なさない（98% の決まり）
        black = np.zeros((18, 32), np.float32)
        black[0, :5] = 1.0
        (run,) = _video_runs([_picture(0), black, black, _picture(1)])
        assert (run.start, run.end) == (1, 3)

    def test_a_dark_but_visible_picture_is_not_black(self) -> None:
        # 壊れると、夜景のような暗い場面まで黒いコマとして並べる
        assert _video_runs([_picture(i, level=0.2) for i in range(5)]) == []

    def test_the_level_and_the_shortest_length_are_respected(self) -> None:
        # 壊れると、AI が渡したしきい値が効かず、頼んだ物と違う一覧が返る
        dim = np.full((18, 32), 0.15, np.float32)
        frames = [_picture(0), dim, dim, _picture(1)]
        assert _video_runs(frames) == []
        assert len(_video_runs(frames, CheckLimits(black_level=0.2))) == 1
        assert _video_runs(frames, CheckLimits(black_level=0.2, min_black_seconds=0.1)) == []


class TestFreeze:
    def test_a_long_still_is_found_and_a_short_one_is_not(self) -> None:
        # 壊れると、2 秒止まった絵を見落とすか、一瞬の止まりまで並べる
        moving = [_picture(i) for i in range(30)]
        still = [_picture(99)] * 61
        frames = moving + still + [_picture(i + 200) for i in range(30)]
        (run,) = _video_runs(frames)
        assert run.kind == FREEZE
        assert (run.start, run.end) == (30, 91)
        short = moving + [_picture(99)] * 59 + moving
        assert _video_runs(short) == []

    def test_a_slow_change_is_not_a_freeze(self) -> None:
        # 壊れると、ゆっくり明るくなる絵（隣のコマとの差が小さい）を止まりと言う
        base = _picture(3, level=0.3)
        frames = [np.clip(base + index * 0.0008, 0, 1) for index in range(120)]
        assert _video_runs(frames) == []

    def test_black_is_not_reported_as_a_freeze_too(self) -> None:
        # 壊れると、長い黒が黒と止まりの 2 つで並び、同じ所を 2 度直しに行く
        frames = [_picture(0)] + [np.zeros((18, 32), np.float32)] * 90 + [_picture(1)]
        assert [run.kind for run in _video_runs(frames)] == [BLACK]

    def test_an_ongoing_run_shows_up_before_the_end(self) -> None:
        # 壊れると、検査の途中で返す結果に、今まさに続いている止まりが出ない
        scan = VideoScan(CheckLimits(), FPS)
        for index in range(90):
            scan.add(index, _picture(5))
        (run,) = scan.runs(final=False)
        assert run.ongoing
        assert (run.start, run.end) == (0, 90)


class TestSilence:
    def test_a_long_gap_is_silence_and_a_pause_is_not(self) -> None:
        # 壊れると、置き忘れた 3 秒の無音を見落とすか、話の間（0.5 秒）まで並べる
        sound = np.concatenate(
            [_tone(1.0), np.zeros((RATE * 3, 2)), _tone(1.0), np.zeros((RATE // 2, 2)), _tone(1.0)]
        )
        runs = [run for run in _audio(sound).runs(final=True) if run.kind == SILENCE]
        (run,) = runs
        assert run.start == 30
        assert run.end == 120

    def test_room_noise_is_not_silence(self) -> None:
        # 壊れると、静かな部屋の雑音（-50 dBFS）まで無音と言い、話の無い所を全部直しに行く
        rng = np.random.default_rng(0)
        noise = 10 ** (-50 / 20) * rng.standard_normal((RATE * 3, 2))
        assert _audio(noise).runs(final=True) == []

    def test_one_channel_playing_is_not_silence(self) -> None:
        # 壊れると、左右に振った声の片側だけが鳴る所を無音と言う
        sound = _tone(3.0)
        sound[:, 1] = 0.0
        assert _audio(sound).runs(final=True) == []

    def test_silence_to_the_very_end_covers_the_last_sample(self) -> None:
        # 壊れると、終わりまで続く無音の終わりが端数の区切りの分だけ手前で切れる
        sound = np.concatenate([_tone(1.0), np.zeros((RATE * 3 + 123, 2))])
        (run,) = _audio(sound).runs(final=True)
        assert run.end == pytest.approx(len(sound) * FPS / RATE, abs=1)


class TestClipping:
    def test_three_flat_samples_are_clipping_and_two_are_not(self) -> None:
        # 壊れると、頭打ちが続く音割れを見落とすか、最大まで上げただけの正しい音を割れと言う
        sound = _tone(1.0)
        sound[1000:1003] = 1.0
        (run,) = [run for run in _audio(sound).runs(final=True) if run.kind == CLIPPING]
        assert run.values["clipped_samples"] == 6
        sound = _tone(1.0)
        sound[1000:1002] = 1.0
        assert _audio(sound).runs(final=True) == []

    def test_a_run_across_blocks_is_counted_once(self) -> None:
        # 壊れると、塊の境目をまたぐ頭打ちが 2 つに割れ、どちらも短いと見なして見落とす
        sound = _tone(1.0)
        sound[1599:1601, 0] = 1.0
        sound[1601, 0] = -1.0
        (run,) = _audio(sound, block=1600).runs(final=True)
        assert run.values["clipped_samples"] == 3

    def test_close_marks_are_merged_and_far_ones_are_not(self) -> None:
        # 壊れると、1 つの大きな音で何百か所も並ぶか、離れた 2 か所が 1 つにまとまる
        sound = _tone(3.0)
        for start in range(1000, 20000, 500):
            sound[start : start + 4] = 1.0
        sound[RATE * 2 : RATE * 2 + 4] = 1.0
        runs = _audio(sound).runs(final=True)
        assert len(runs) == 2

    def test_a_mix_over_full_scale_is_marked(self) -> None:
        # 壊れると、混ぜて大きくしすぎた音と、素材が元から割れている音を見分けられない
        sound = _tone(1.0, amplitude=1.5)
        (run,) = _audio(sound).runs(final=True)
        assert run.values["over_full_scale"] is True
        assert run.values["peak_dbfs"] == pytest.approx(20 * np.log10(1.5), abs=0.01)


class TestSeverity:
    def _assemble(self, runs: list[Run], *, loudness: float | None = None) -> list[tuple[str, str]]:
        report = assemble(
            runs,
            start=0,
            end=300,
            analyzed=300,
            fps=FPS,
            complete=True,
            limits=CheckLimits(),
            index=None,
            loudness=loudness,
            peak=None,
            length_seconds=None,
        )
        return [(problem.kind, problem.severity) for problem in report.problems]

    def test_fades_at_the_edges_are_low_and_the_middle_is_not(self) -> None:
        # 壊れると、フェードの入りと抜けまで直すべき問題として並び、AI が演出を消しに行く
        runs = [Run(BLACK, 0, 10), Run(BLACK, 100, 101), Run(SILENCE, 290, 300)]
        assert self._assemble(runs) == [(BLACK, LOW), (BLACK, MEDIUM), (SILENCE, LOW)]

    def test_silence_for_the_whole_work_is_high(self) -> None:
        # 壊れると、音がまったく無い作品を頭と終わりの無音と同じ軽さで返す
        assert self._assemble([Run(SILENCE, 0, 300)]) == [(SILENCE, HIGH)]

    def test_loudness_far_from_the_target_is_high(self) -> None:
        # 壊れると、基準から大きく外れた作品と少し外れた作品が同じ程度で並ぶ
        assert self._assemble([], loudness=-14.5) == []
        assert self._assemble([], loudness=-16.0) == [("loudness", MEDIUM)]
        assert self._assemble([], loudness=-23.0) == [("loudness", HIGH)]
