"""字幕起こしの窓が、選んだクリップの素材の音声を並べる

利用者の画面では、音声が 1 本の動画を選んで起こしても「起こす音声」に 4 つ並んだ
起こす（Ctrl+U）はメディア欄で選んだ素材か、字幕パネルが前に開いていた素材（音声 4 本の
動画）を起こしていて、タイムラインで選んだクリップを見ていなかった
窓を開く入口ごとに、どの素材を開いたか・音声の選びがいくつかを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, AddMedia, AddTrack
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    ClipId,
    MediaItem,
    Project,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui.main_window import MainWindow
from sashimono.ui.subtitle import panel as subtitle_panel
from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog


def _movie(name: str, voices: int) -> MediaItem:
    return MediaItem(
        path=Path(f"C:/素材/{name}.mp4"),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 15360),
                codec="h264",
                pixel_format="yuv420p",
            ),
        ),
        audio_streams=tuple(
            AudioStreamInfo(
                index=number,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="aac",
            )
            for number in range(1, voices + 1)
        ),
    )


FOUR, ONE, TWO = _movie("ゲーム録画", 4), _movie("一本", 1), _movie("二本", 2)


def _project() -> tuple[Project, dict[str, ClipId]]:
    """音声 4 本・1 本・2 本の動画を 1 本ずつ別の音声トラックへ置く（どれも音を鳴らす）"""
    project = Project.create()
    placed: dict[str, ClipId] = {}
    for media in (FOUR, ONE, TWO):
        project = AddMedia(media).apply(project)
        track = Track(TrackKind.AUDIO, media.name)
        project = AddTrack(track).apply(project)
        clip = Clip(
            timeline_start=0,
            duration=60,
            media_id=media.id,
            stream_index=media.audio_streams[-1].index,
        )
        project = AddClip(track.id, clip).apply(project)
        placed[media.name] = clip.id
    return project, placed


class _Seen:
    """開いた窓を覚える 窓は開かずにすぐ閉じる（試験を止めない）"""

    def __init__(self) -> None:
        self.opened: list[tuple[str, int, int | None, bool]] = []


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> _Seen:
    record = _Seen()

    class Recording(TranscribeDialog):
        def exec(self) -> int:
            self.show()
            QApplication.processEvents()
            record.opened.append(
                (
                    self._media.name,
                    self._stream.count(),
                    self._stream.currentData(),
                    self._stream.isVisible(),
                )
            )
            self.close()
            return 0

    monkeypatch.setattr(subtitle_panel, "TranscribeDialog", Recording)
    return record


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[tuple[MainWindow, dict[str, ClipId]]]:
    del qt_application
    project, placed = _project()
    created = MainWindow(project, confirm_unsaved=False)
    yield created, placed
    created.close()


def test_the_selected_clip_is_transcribed_after_a_four_voice_movie(
    window: tuple[MainWindow, dict[str, ClipId]], seen: _Seen
) -> None:
    # 利用者の順番 音声 4 本の動画を先に起こしてから、1 本の動画のクリップを選んで起こす
    main, placed = window
    main._transcribe_media(str(FOUR.id))
    main._timeline.select(placed[ONE.name])
    main.transcribe()
    assert seen.opened[0][:2] == (FOUR.name, 4)
    name, count, _, visible = seen.opened[1]
    assert (name, count, visible) == (ONE.name, 1, False)


def test_two_voices_give_two_and_start_at_the_heard_one(
    window: tuple[MainWindow, dict[str, ClipId]], seen: _Seen
) -> None:
    # 置いたクリップが鳴らしている音（音声 2）を初めから選んでおく
    main, placed = window
    main._timeline.select(placed[TWO.name])
    main.transcribe()
    assert seen.opened[-1] == (TWO.name, 2, TWO.audio_streams[1].index, True)


def test_the_media_list_still_opens_what_was_pressed(
    window: tuple[MainWindow, dict[str, ClipId]], seen: _Seen
) -> None:
    # メディア欄の右クリックは押した素材を開く タイムラインで選んでいる物に引っ張られない
    main, placed = window
    main._timeline.select(placed[FOUR.name])
    main._transcribe_media(str(ONE.id))
    assert seen.opened[-1][:2] == (ONE.name, 1)


def test_the_subtitle_panel_follows_the_selected_clip(
    window: tuple[MainWindow, dict[str, ClipId]], seen: _Seen
) -> None:
    # 字幕パネルの〔起こす〕も、タイムラインで選んだクリップの素材を開く
    main, placed = window
    main._transcribe_media(str(FOUR.id))
    main._timeline.select(placed[ONE.name])
    main._subtitles.transcribe()
    assert seen.opened[-1][:2] == (ONE.name, 1)


def test_the_media_lookup_does_not_mix_up_same_named_files() -> None:
    # 素材の選びは ID で引く 同じ名前のファイルでも取り違えない
    same = replace(_movie("一本", 1), path=Path("D:/別/一本.mp4"))
    assert same.name == ONE.name and same.id != ONE.id
