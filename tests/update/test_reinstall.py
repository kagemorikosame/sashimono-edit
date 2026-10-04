"""Python が上がる更新で、入れ直しが要る機能と、その大きさと時間の目安（Issue #244）

「入れ直しが要る」だけだと、2 GB を超える落とし直しを知らずに選ぶ どれを・どれだけ・
どれだけ掛かるかを、入れる前の確認に添える
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sashimono.runtime import FeaturePack
from sashimono.update.reinstall import (
    Reinstall,
    describe,
    estimate,
    folder_size,
    installed_packs,
)

ASR = FeaturePack(
    key="asr",
    label="字幕起こし",
    required=("faster-whisper>=1.1",),
    extra=("nvidia-cublas-cu12",),
    size_mb=300,
    extra_size_mb=1700,
)
AI = FeaturePack(key="ai", label="AI 連携", required=("claude-agent-sdk>=0.2.152",), size_mb=250)


def _install(target: Path, *names: str, size: int = 0) -> None:
    """pip が ``--target`` に置くのと同じ形（dist-info と包みの中身）"""
    for name in names:
        (target / f"{name}-1.0.dist-info").mkdir(parents=True)
        package = target / name
        package.mkdir()
        (package / "__init__.py").write_bytes(b"x" * size)


class TestWhatIsInstalled:
    def test_packs_are_found_by_their_dist_info(self, tmp_path: Path) -> None:
        """import して確かめない Python が上がった後は、読むと拡張モジュールで落ちる"""
        _install(tmp_path, "faster_whisper")
        assert installed_packs(tmp_path, (ASR, AI)) == [(ASR, False)]
        _install(tmp_path, "nvidia_cublas_cu12", "claude_agent_sdk")
        assert installed_packs(tmp_path, (ASR, AI)) == [(ASR, True), (AI, False)]

    def test_nothing_installed(self, tmp_path: Path) -> None:
        assert installed_packs(tmp_path, (ASR, AI)) == []
        assert estimate(tmp_path, (ASR, AI)) is None


class TestMeasuring:
    def test_the_folder_is_measured(self, tmp_path: Path) -> None:
        _install(tmp_path, "faster_whisper", size=1000)
        (tmp_path / "faster_whisper" / "深い" / "所").mkdir(parents=True)
        (tmp_path / "faster_whisper" / "深い" / "所" / "x.pyd").write_bytes(b"y" * 500)
        assert folder_size(tmp_path) == 1500

    def test_a_slow_folder_gives_up(self, tmp_path: Path) -> None:
        """測り終えなければ None 入れる前の確認を開くときに測るので、窓を固めない"""
        _install(tmp_path, "faster_whisper", "claude_agent_sdk", size=10)
        ticks: Iterator[float] = iter(float(i) for i in range(100))
        assert folder_size(tmp_path, seconds=1.5, clock=lambda: next(ticks)) is None


class TestEstimate:
    def test_a_measured_size_is_used(self, tmp_path: Path) -> None:
        _install(tmp_path, "faster_whisper", "claude_agent_sdk", size=2048)
        found = estimate(tmp_path, (ASR, AI))
        assert found == Reinstall(("字幕起こし", "AI 連携"), 4096, measured=True)

    def test_a_guess_is_used_when_measuring_gives_up(self, tmp_path: Path) -> None:
        _install(tmp_path, "faster_whisper", "nvidia_cublas_cu12", "claude_agent_sdk")
        ticks: Iterator[float] = iter(float(i) for i in range(100))
        found = estimate(tmp_path, (ASR, AI), seconds=0.5, clock=lambda: next(ticks))
        assert found is not None and not found.measured
        assert found.size_bytes == (300 + 1700 + 250) * 1024 * 1024

    @pytest.mark.parametrize(
        ("size", "measured", "expected"),
        [
            (int(2.2 * 1024**3), True, "2.2 GB（今入れてある分を測った値）"),
            (250 * 1024**2, False, "約 250 MB（目安）"),
        ],
    )
    def test_the_line_says_what_how_much_and_how_long(
        self, size: int, measured: bool, expected: str
    ) -> None:
        text = describe(Reinstall(("字幕起こし", "AI 連携"), size, measured))
        assert "字幕起こし・AI 連携" in text
        assert expected in text
        assert "100 Mbps なら約" in text and "20 Mbps なら約" in text

    def test_the_time_grows_with_the_size(self) -> None:
        """2.2 GB は 100 Mbps で約 4 分、20 Mbps で約 16 分（切り上げ）"""
        text = describe(Reinstall(("字幕起こし",), int(2.2 * 1024**3), True))
        assert "100 Mbps なら約 4 分" in text
        assert "20 Mbps なら約 16 分" in text
