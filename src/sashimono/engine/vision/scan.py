"""素材の決めた時刻の絵を取り出して調べる（#324）"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np

from sashimono.engine.decode import VideoDecoder

from .report import ScreenKind, ScreenReport, ScreenSample

__all__ = ["Detector", "ScreenScan", "scan_screen"]


class Detector(Protocol):
    """1 枚の絵を調べる部品（Windows の OCR・YuNet・試験の偽物）"""

    @property
    def kind(self) -> ScreenKind: ...

    @property
    def engine(self) -> str:
        """結果と控えに書く名前（部品と版） 変われば控えを読み直す"""
        ...

    def look(self, rgba: np.ndarray, seconds: float) -> ScreenSample:
        """絵（高さ x 幅 x 4 の RGBA）を調べた結果"""
        ...


class ScreenScan:
    """調べている途中の結果 AI の道具が、終わる前に調べた所までを読むため

    書くのは調べる係のスレッド、読むのは画面のスレッド
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._report: ScreenReport | None = None

    def update(self, report: ScreenReport) -> None:
        with self._lock:
            self._report = report

    def snapshot(self) -> ScreenReport | None:
        with self._lock:
            return self._report


def scan_screen(
    path: Path,
    times: Sequence[float],
    detector: Detector,
    *,
    stream_index: int | None = None,
    by_scene: bool = True,
    scan: ScreenScan | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ScreenReport | None:
    """``times``（素材の原点から数えた秒）の絵を調べる 止められたら ``None``

    取り出せない時刻（素材の終わりの外）は飛ばす 素材を開けなければ
    :class:`~sashimono.engine.decode.ProbeError`
    """
    samples: list[ScreenSample] = []
    with VideoDecoder(path, stream_index) as decoder:
        width, height = decoder.info.display_size

        def report(complete: bool) -> ScreenReport:
            return ScreenReport(
                kind=detector.kind,
                engine=detector.engine,
                width=width,
                height=height,
                samples=tuple(samples),
                planned=len(times),
                complete=complete,
                stream=stream_index,
                by_scene=by_scene,
            )

        for index, moment in enumerate(sorted(times)):
            if should_cancel is not None and should_cancel():
                return None
            # 浮動小数のまま分数にすると 2 進の端数まで持ち、フレームの境目の所で 1 つ前の絵を取る
            frame = decoder.frame_at(Fraction(str(round(moment, 6))))
            if frame is not None:
                samples.append(detector.look(frame, moment))
            if scan is not None:
                scan.update(report(complete=False))
            if progress is not None:
                progress((index + 1) / max(1, len(times)))
    return report(complete=True)
