"""何本もの範囲をまとめて束ねても、1 本ずつ束ねたときと同じ値になる（#260）

タイムラインは、拡大して初めて見る倍率で何百本ものクリップの波形をまとめて作る
まとめ方で値が 1 つでも変われば、同じクリップの波形が倍率の行き来で揺れて見える
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.engine.audio.waveform import PeakLevel, Waveform, _coarsen


def _waveform(count: int, seed: int = 0) -> Waveform:
    rng = np.random.default_rng(seed)
    peaks = np.empty((count, 2, 2), dtype=np.float32)
    peaks[:, :, 1] = rng.random((count, 2), dtype=np.float32)
    peaks[:, :, 0] = -rng.random((count, 2), dtype=np.float32)
    levels = [PeakLevel(256, peaks)]
    while levels[-1].count > 1:
        levels.append(_coarsen(levels[-1]))
    return Waveform(48000, 2, count * 256, tuple(levels))


def _one_by_one(wave: Waveform, segments: list[tuple[int, int, int]]) -> np.ndarray:
    parts = [wave.envelope(s, e, c) for s, e, c in segments if c > 0]
    return np.concatenate(parts) if parts else np.zeros((0, 2, 2), dtype=np.float32)


@pytest.mark.parametrize("seed", range(12))
def test_many_ranges_match_one_by_one(seed: int) -> None:
    # 値が 1 つでもずれると、初めての倍率でまとめて作った波形と、同じ倍率へ戻って 1 本ずつ
    # 作った波形の山が食い違い、ズームの行き来で波形が揺れて見える
    # 段階の違う範囲・素材の末尾に掛かる範囲・素材の外の範囲・空の範囲を混ぜる
    rng = np.random.default_rng(seed)
    wave = _waveform(int(rng.integers(1, 4000)), seed)
    total = wave.total_samples
    segments: list[tuple[int, int, int]] = []
    for _ in range(int(rng.integers(1, 60))):
        start = int(rng.integers(-total // 4, total + total // 4))
        length = int(rng.integers(-500, total))
        columns = int(rng.integers(-1, 400))
        segments.append((start, start + length, columns))
    # 素材の終わりちょうどまでの範囲（reduceat の外に出る列）を必ず入れる
    segments.append((total // 3, total, 7))
    segments.append((total - 256, total, 1))
    assert np.array_equal(wave.envelopes(segments), _one_by_one(wave, segments))


def test_an_empty_list_gives_no_columns() -> None:
    # 形が (0, チャンネル, 2) で無くなると、呼んだ側の envelope[:, :, 0] が IndexError になり、
    # タイムラインの描画がそこで止まる（今の描く所は空の並びで呼ばないが、呼べる約束は守る）
    assert wave_shape(_waveform(10).envelopes([])) == (0, 2, 2)


def test_a_silent_material_gives_zeros() -> None:
    # 中身の無い素材（ピーク 0 個）で reduceat に番号を渡すと IndexError になり、その素材の
    # クリップが見えている間はタイムラインが描けなくなる 0 で埋めた列を返して落ちない
    empty = Waveform(48000, 2, 0, (PeakLevel(256, np.zeros((0, 2, 2), dtype=np.float32)),))
    found = empty.envelopes([(0, 48000, 5), (100, 200, 3)])
    assert np.array_equal(found, _one_by_one(empty, [(0, 48000, 5), (100, 200, 3)]))


def wave_shape(array: np.ndarray) -> tuple[int, ...]:
    return tuple(array.shape)
