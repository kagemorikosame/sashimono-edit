"""波形に映す音の効果の、キーフレームで動く値（PR #231 の指摘）

前は音量のほかの効果（フェード・ディレイ・リバーブ）の値を頭のコマでしか解かず、
キーフレームで動かしても波形が頭の値のまま変わらなかった 列ごとのコマで解く
"""

from __future__ import annotations

import numpy as np

from sashimono.core.model import AnimatedValue, Clip, Keyframe, ParamValue
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.audio.shape import shape_envelope

RATE = FrameRate(30)
#: 10 秒のクリップを 1 秒 30 列で描く
FRAMES = 300
COLUMNS = 300


def _moving(start: float, end: float) -> AnimatedValue:
    """頭で ``start``、終わりで ``end`` になる値"""
    return AnimatedValue(keyframes=(Keyframe(0, start), Keyframe(FRAMES - 1, end)))


def _shape(kind: str, peaks: np.ndarray, **values: ParamValue) -> tuple[np.ndarray, np.ndarray]:
    effect = registry.require(kind).create(**values)
    clip = Clip(timeline_start=0, duration=FRAMES, effects=(effect,))
    return shape_envelope(-peaks, peaks.copy(), clip, RATE, 0.0, float(FRAMES), track_gain=1.0)


def _click() -> np.ndarray:
    peaks = np.zeros(COLUMNS, dtype=np.float32)
    peaks[150] = 1.0
    return peaks


def test_a_delay_that_grows_louder_shows_its_echo() -> None:
    # 量は頭で 0、終わりで 100 頭の値だけで描くと、やまびこが 1 つも出ない
    _, high = _shape("audio_delay", _click(), time=500, feedback=0, mix=_moving(0, 100))
    echo = 150 + 15
    assert high[echo] > 0.3


def test_a_reverb_that_grows_longer_reaches_further() -> None:
    # 残響の長さは頭で 0.1 秒、終わりで 5 秒 頭の値だけで描くと 1 秒後には尾が無い
    _, high = _shape("audio_reverb", _click(), decay=_moving(0.1, 5.0), mix=100)
    assert high[150 + 30] > 0.05


def test_a_fade_that_grows_longer_quiets_the_end() -> None:
    # 抜けの長さは頭で 0 秒、終わりで 10 秒 頭の値だけで描くと終わりの手前まで 1 のまま
    peaks = np.ones(COLUMNS, dtype=np.float32)
    _, high = _shape("audio_fade", peaks, fade_out=_moving(0, 10))
    assert high[COLUMNS - 30] < 0.5


def test_values_that_do_not_move_draw_as_before() -> None:
    _, high = _shape("audio_delay", _click(), time=500, feedback=50, mix=100)
    assert high[165] == np.float32(1.0)
    assert high[180] == np.float32(0.5)
    assert high[160] == 0.0
