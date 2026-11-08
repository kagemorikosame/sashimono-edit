"""使っていない素材の控えを「置いたときに作る」設定（#441）"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, AddMedia, AddTrack
from sashimono.core.model import (
    Clip,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import PROXY_UNUSED_ON_USE


def _media(name: str) -> MediaItem:
    stream = VideoStreamInfo(
        index=0,
        width=3840,
        height=2160,
        frame_rate=FrameRate(30),
        time_base=Fraction(1, 30),
        codec="h264",
    )
    return MediaItem(path=Path(f"{name}.mp4"), duration=Fraction(10), video_streams=(stream,))


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30))),
        confirm_unsaved=False,
    )
    created._preferences = replace(created._preferences, proxy_unused=PROXY_UNUSED_ON_USE)
    yield created
    created.close()


def test_media_used_together_are_all_requested(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、素材を 1 本だけ頼む道（テンプレートを置く等）が使っている素材を見直した後、
    同じ編集で使い始めたほかの素材の控えが、次の編集まで頼まれない（#452 の Codex の指摘）"""
    asked: list[MediaId] = []
    monkeypatch.setattr(window._proxies, "request", lambda media, **_kwargs: asked.append(media.id))
    old, new = _media("old"), _media("new")
    project = window._document.project
    for item in (old, new):
        project = AddMedia(item).apply(project)
    track = Track(TrackKind.VIDEO, "V9")
    project = AddTrack(track).apply(project)
    for start, item in enumerate((old, new)):
        project = AddClip(
            track.id, Clip(timeline_start=start * 40, duration=30, media_id=item.id)
        ).apply(project)
    window._document.reset(project)
    # 置いた道が、新しい素材だけを頼んだ（このとき使っている素材を求め直す）
    window._request_proxy(new)
    asked.clear()
    window._request_newly_used()
    assert old.id in asked


def test_unused_media_are_not_requested(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、置いたときに作る設定でも、読み込んだだけの素材の控えを作り始める"""
    asked: list[MediaId] = []
    monkeypatch.setattr(window._proxies, "request", lambda media, **_kwargs: asked.append(media.id))
    item = _media("unused")
    window._document.reset(AddMedia(item).apply(window._document.project))
    window._request_proxy(item)
    window._request_newly_used()
    assert asked == []
