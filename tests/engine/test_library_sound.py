"""エクスプローラーの素材を測る（#522） 長さと頭の 1 分の大きさ

AI が効果音と BGM を選ぶ手がかり 測り方そのもの（BS.1770）は tests/engine/test_loudness.py など
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sashimono.engine.audio.library_sound import measure_library_file
from tests.check_media import tone, write_tone


def test_measures_length_and_loudness(tmp_path: Path) -> None:
    # 測れないと、AI は名前だけで選び、効果音の所に長い曲を置くか、小さすぎる音を選ぶ
    loud = write_tone(tmp_path / "大.wav", tone(2.0, -6.0))
    quiet = write_tone(tmp_path / "小.wav", tone(2.0, -26.0))
    big = measure_library_file(loud)
    small = measure_library_file(quiet)
    assert big is not None and small is not None
    assert big.seconds == pytest.approx(2.0, abs=0.05)
    assert big.has_audio and not big.has_video
    assert big.loudness is not None and small.loudness is not None
    # 20 dB 下げた音は 20 LU ほど小さく測れる
    assert big.loudness - small.loudness == pytest.approx(20.0, abs=0.5)
    assert big.peak == pytest.approx(-6.0, abs=0.2)


def test_a_broken_file_is_not_measured(tmp_path: Path) -> None:
    # 壊れると、壊れたファイルが 1 つあるだけで探す道具ごと落ちる
    broken = tmp_path / "壊れた.wav"
    broken.write_bytes(b"not a sound")
    assert measure_library_file(broken) is None
