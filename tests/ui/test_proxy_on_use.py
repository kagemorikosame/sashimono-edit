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
from sashimono.engine.cache.proxy import ProxyBuilder
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import PROXY_UNUSED_LATER, PROXY_UNUSED_ON_USE


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
    # 普段の見回り（ステータスバーの進み具合の更新）から頼まれる
    window._show_background_progress()
    assert old.id in asked


def _placed(window: MainWindow, *items: MediaItem) -> None:
    """素材を足し、``items`` をタイムラインへ置く"""
    project = window._document.project
    for item in items:
        project = AddMedia(item).apply(project)
    track = Track(TrackKind.VIDEO, "V9")
    project = AddTrack(track).apply(project)
    for start, item in enumerate(items):
        project = AddClip(
            track.id, Clip(timeline_start=start * 40, duration=30, media_id=item.id)
        ).apply(project)
    window._document.reset(project)


def test_a_failed_proxy_is_not_asked_again_on_every_edit(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、作れずに失敗した使っている素材を、関わりの無い編集のたびに頼み直し、
    変換と失敗の知らせを繰り返す
    """
    asked: list[MediaId] = []
    monkeypatch.setattr(window._proxies, "request", lambda media, **_kwargs: asked.append(media.id))
    item = _media("broken")
    _placed(window, item)
    window._request_newly_used()
    assert asked == [item.id]
    # 関わりの無い編集（素材をもう 1 本足しただけ）で作品が変わる
    window._document.reset(AddMedia(_media("other")).apply(window._document.project))
    window._request_newly_used()
    assert asked == [item.id]


def test_removed_and_returned_media_are_asked_again(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、作っている途中に外して取り消しで戻した素材の控えを、また使い始めても頼まない"""
    asked: list[MediaId] = []
    monkeypatch.setattr(window._proxies, "request", lambda media, **_kwargs: asked.append(media.id))
    monkeypatch.setattr(window._proxies, "forget", lambda _media_id: None)
    item = _media("returned")
    _placed(window, item)
    window._request_newly_used()
    assert asked == [item.id]
    window._forget_media([item.id])
    # 戻した作品は前と別の物 同じ物を渡すと、見直し済みとして何もしない
    window._document.reset(replace(window._document.project))
    window._request_newly_used()
    assert asked == [item.id, item.id]


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


def test_switching_to_on_use_cancels_unused_proxies(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、置いたときに作ると選んだ後も、読み込んだだけの素材の控えを作り続ける"""
    window._preferences = replace(window._preferences, proxy_unused=PROXY_UNUSED_LATER)
    forgotten: list[MediaId] = []
    monkeypatch.setattr(window._proxies, "forget", forgotten.append)
    monkeypatch.setattr(window._proxies, "request", lambda *_args, **_kwargs: None)
    used, unused = _media("used"), _media("unused")
    _placed(window, used)
    window._document.reset(AddMedia(unused).apply(window._document.project))
    window._apply_preferences(replace(window._preferences, proxy_unused=PROXY_UNUSED_ON_USE))
    assert unused.id in forgotten and used.id not in forgotten


def test_the_first_requests_follow_the_order(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、素材一覧の先頭にある使っていない長い動画から作り始め、プレビューで見ている
    素材の控えがその後まで待たされる
    """
    del qt_application
    asked: list[str] = []
    monkeypatch.setattr(
        ProxyBuilder, "request", lambda _self, media, **_kwargs: asked.append(media.path.stem)
    )
    unused, used = _media("unused"), _media("used")
    project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    for item in (unused, used):
        project = AddMedia(item).apply(project)
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(project)
    project = AddClip(track.id, Clip(timeline_start=0, duration=30, media_id=used.id)).apply(
        project
    )
    created = MainWindow(project, confirm_unsaved=False)
    try:
        assert asked[:2] == ["used", "unused"]
    finally:
        created.close()
