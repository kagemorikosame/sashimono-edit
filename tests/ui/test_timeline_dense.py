"""クリップが多いときのタイムライン

全体を表示すると 1 本が 1 画素に満たなくなる そこで 1 本ずつ名前と枠を描いていたため、
1 万本で 1 回の描画に 178ms かかっていた（60fps の予算は 16.7ms）
速くするために「見えている分だけ探す」と「細いものはまとめて塗る」を入れたので、
その 2 つが見落としや取り違えを起こしていないことを押さえる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.painter import clips_in_range


def _track(count: int, length: int = 10, gap: int = 5) -> Track:
    return Track(
        TrackKind.VIDEO,
        "V1",
        tuple(
            Clip(timeline_start=n * (length + gap), duration=length, source=TEXT.create())
            for n in range(count)
        ),
    )


class TestClipsInRange:
    @pytest.mark.parametrize(
        ("start", "end"), [(0, 0), (9, 10), (10, 15), (12, 13), (14, 30), (0, 10_000), (740, 800)]
    )
    def test_it_matches_looking_at_every_clip(self, start: int, end: int) -> None:
        # 二分探索が 1 本でも取りこぼすと、画面の端のクリップが描かれず、掴めもしない
        track = _track(50)
        expected = [c for c in track.clips if c.timeline_end > start and c.timeline_start <= end]
        assert list(clips_in_range(track, start, end)) == expected

    def test_an_empty_track_is_fine(self) -> None:
        # 壊れると、空のトラックで範囲外を読みにいき、描画ごと例外で止まる
        assert list(clips_in_range(Track(TrackKind.AUDIO), 0, 100)) == []


@pytest.fixture
def crowded(qt_application: QApplication) -> Iterator[TimelineView]:
    del qt_application
    base = Project.create()
    project = base.with_timeline(replace(base.timeline, tracks=(_track(2000, 30, 0),)))
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    view = TimelineView(project, analyzer)
    view.resize(900, 200)
    # 全体を表示 1 本が 1 画素に満たない
    view._layout = TimelineLayout(
        pixels_per_frame=(900 - Metrics.TRACK_HEADER_WIDTH) / project.duration
    )
    yield view
    analyzer.close()


def _render(view: TimelineView) -> QImage:
    image = QImage(view.size(), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    view.render(painter, QPoint())
    painter.end()
    return image


class TestDenseDrawing:
    def test_thin_clips_still_show_as_a_filled_band(self, crowded: TimelineView) -> None:
        # まとめて塗る処理が抜けると、全体表示で何も無いように見える
        band = crowded._layout.bands(crowded.project.timeline)[0]
        image = _render(crowded)
        y = band.top + band.height // 2
        colours = {image.pixelColor(x, y).name() for x in range(200, 800, 37)}
        assert colours <= {Colors.VIDEO_CLIP.name(), Colors.VIDEO_CLIP_BORDER.name()}
        assert Colors.VIDEO_CLIP.name() in colours

    def test_a_thin_clip_can_still_be_grabbed(self, crowded: TimelineView) -> None:
        # 探し方を変えたので、全体表示でもマウスの下のクリップが見つかることを確かめる
        # 見つからないと、全体表示ではクリップを 1 本も選べない
        band = crowded._layout.bands(crowded.project.timeline)[0]
        hit = crowded._clip_at(QPoint(500, band.top + band.height // 2))
        assert hit is not None
        # 1 画素が数十フレームにあたるので、その画素に掛かっていれば正しい
        first, last = crowded._layout.frame_at(499), crowded._layout.frame_at(501)
        assert hit[1].timeline_start <= last and first < hit[1].timeline_end

    def test_the_header_is_never_a_clip(self, crowded: TimelineView) -> None:
        # トラックの見出しをクリップと取ると、名前を押しただけで隣のクリップが選ばれたり動いたりする
        band = crowded._layout.bands(crowded.project.timeline)[0]
        assert crowded._clip_at(QPoint(10, band.top + 5)) is None

    def test_a_selected_thin_clip_is_outlined(self, crowded: TimelineView) -> None:
        # 細いと枠を描かない作りなので、選んだものまで見えなくならないようにしている
        clip = crowded.project.timeline.tracks[0].clips[1000]
        crowded.select(clip.id)
        band = crowded._layout.bands(crowded.project.timeline)[0]
        x = int(crowded._layout.frame_to_x(clip.timeline_start))
        image = _render(crowded)
        column = {image.pixelColor(x, y).name() for y in range(band.top + 1, band.bottom - 2)}
        assert Colors.SELECTION.name() in column
