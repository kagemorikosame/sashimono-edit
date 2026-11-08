"""書き出しのビットレートの既定を素材に合わせる（#414 ``encode/choices.py`` の ``BitrateChoice``）

通しの確かめで、12 Mbps 決め打ちの既定は 1〜2 Mbps の素材を 7 倍の大きさにした
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from sashimono.core.model import MediaItem, Project, ProjectSettings, Transcript
from sashimono.engine.encode import (
    DEFAULT_BITRATE_MBPS,
    MATCH_FLOOR_MBPS,
    BitrateChoice,
    matched_bitrate_mbps,
)
from tests.ai.conftest import make_loaded

#: 10 秒の素材が 2 Mbps になる大きさ（バイト）
TWO_MBPS = 2_000_000 * 10 // 8


def _sizes(**sizes: int) -> Callable[[Path], int]:
    def size_of(path: Path) -> int:
        found = sizes.get(path.stem)
        if found is None:
            raise FileNotFoundError(path)
        return found

    return size_of


def test_a_low_bitrate_source_is_matched(video_media: MediaItem) -> None:
    """壊れると、2 Mbps の素材を 12 Mbps で書き出し、見た目の変わらないファイルが 6 倍になる"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    size_of = _sizes(low=TWO_MBPS)
    # 2 Mbps の 1.25 倍を切り上げて 3 Mbps
    assert matched_bitrate_mbps(project, size_of=size_of) == 3
    assert BitrateChoice().for_project(project, size_of=size_of) == 3


def test_the_upper_value_stays_a_cap(video_media: MediaItem) -> None:
    """壊れると、カメラの 50 Mbps の素材に合わせて、上限を決めた人の書き出しが巨大になる"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/high.mp4")), _no_words())
    size_of = _sizes(high=TWO_MBPS * 25)
    assert BitrateChoice(mbps=12).for_project(project, size_of=size_of) == 12


def test_a_bigger_output_scales_up(video_media: MediaItem) -> None:
    """作品が素材より大きければ画素の数の比で上げる 上げないと拡大した絵が崩れる"""
    loaded = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    project = replace(loaded, settings=replace(loaded.settings, width=3840, height=2160))
    assert matched_bitrate_mbps(project, size_of=_sizes(low=TWO_MBPS)) == 10


def test_a_tiny_source_keeps_the_floor(video_media: MediaItem) -> None:
    """画面の録画の 0.3 Mbps に合わせると、重ねた文字の縁が崩れる"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/tiny.mp4")), _no_words())
    assert matched_bitrate_mbps(project, size_of=_sizes(tiny=TWO_MBPS // 7)) == MATCH_FLOOR_MBPS


def test_a_missing_file_falls_back(video_media: MediaItem) -> None:
    """外したドライブの素材で止まらず、決めた値で書き出せる"""
    project = make_loaded(video_media, _no_words())
    assert matched_bitrate_mbps(project, size_of=_sizes()) is None
    assert BitrateChoice().for_project(project, size_of=_sizes()) == DEFAULT_BITRATE_MBPS


def test_switched_off_uses_the_fixed_value(video_media: MediaItem) -> None:
    """切った人は、素材に関係なく決めた値で書き出す"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    size_of = _sizes(low=TWO_MBPS)
    assert BitrateChoice(auto=False, mbps=20).for_project(project, size_of=size_of) == 20


def test_a_project_without_video_uses_the_fixed_value() -> None:
    """文字だけの作品は合わせる素材が無い 0 や下限にすると図形の縁が崩れる"""
    project = Project.create(ProjectSettings())
    assert BitrateChoice().for_project(project) == DEFAULT_BITRATE_MBPS


def _no_words() -> Transcript:
    return Transcript(segments=(), language="ja")


def test_the_duration_is_used(video_media: MediaItem) -> None:
    """長さで割らないと、長い素材ほど高いビットレートと読む"""
    longer = replace(video_media, path=Path("C:/素材/low.mp4"), duration=Fraction(20))
    project = make_loaded(longer, _no_words())
    assert matched_bitrate_mbps(project, size_of=_sizes(low=TWO_MBPS)) == 2
