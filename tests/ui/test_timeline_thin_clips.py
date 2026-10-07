"""細い帯（名前の入らない幅のクリップ）でも中身があると分かる（#247）

引いた表示で数秒のクリップを選ぶと、色の帯と枠しか描かれず、中身が無いように見えて
素材の読み込みに失敗したのかと迷った そこで細い帯にも軽い目安（絵の平均の色・音の
大きさ）を描き、選んだ物の枠を太くし、載せたら名前と長さを出すようにした
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Clip, MediaId, MediaItem, Project, Track, TrackKind
from sashimono.core.model.ids import new_media_id
from sashimono.effects.sources import TEXT
from sashimono.engine.audio import PeakLevel, Waveform
from sashimono.engine.cache import MediaAnalyzer
from sashimono.engine.cache.thumbnails import Filmstrip
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, Colors, Metrics, use_palette
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.layout import TimelineLayout, TrackBand
from sashimono.ui.timeline.painter import (
    DENSE_MARK_WIDTH,
    DETAIL_MIN_WIDTH,
    clear_waveform_images,
)
from sashimono.ui.workspace import DETAIL_MIN_WIDTHS, Preferences, PreferenceStore

#: 1 フレームの画素数 30 フレームのクリップが 15 画素になる（既定の 24 より細い）
SCALE = 0.5
#: 細い帯にするクリップの長さ（フレーム）と、名前が入る長いクリップの長さ
THIN = 30
WIDE = 300
#: 絵の平均の色 地の色（映像の帯の色）とどのテーマでも違う
RED = QColor(220, 30, 30)


class _Analyzer(MediaAnalyzer):
    """決めた素材に、決めたサムネイルと波形を返す 自分では解析しない"""

    def __init__(
        self, filmstrips: dict[MediaId, Filmstrip], waveforms: dict[MediaId, Waveform]
    ) -> None:
        super().__init__()
        self._fixed_filmstrips = filmstrips
        self._fixed_waveforms = waveforms

    def filmstrip(self, media: MediaItem) -> Filmstrip | None:
        return self._fixed_filmstrips.get(media.id)

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        return self._fixed_waveforms.get(media.id)


def _filmstrip(colour: QColor) -> Filmstrip:
    sheet = np.zeros((72, 16 * 20, 4), dtype=np.uint8)
    sheet[:, :, 0], sheet[:, :, 1], sheet[:, :, 2] = colour.red(), colour.green(), colour.blue()
    sheet[:, :, 3] = 255
    return Filmstrip(sheet=sheet, tile_width=16, interval=Fraction(1, 2))


def _waveform(peak: float, seconds: int = 30) -> Waveform:
    count = 48000 * seconds // 256
    peaks = np.empty((count, 2, 2), dtype=np.float32)
    peaks[:, :, 0] = -peak
    peaks[:, :, 1] = peak
    return Waveform(
        sample_rate=48000,
        channels=2,
        total_samples=48000 * seconds,
        levels=(PeakLevel(256, peaks),),
    )


@pytest.fixture(params=[THEME_DARK, THEME_LIGHT])
def theme(request: pytest.FixtureRequest) -> Iterator[str]:
    """明るいテーマと暗いテーマの両方で確かめる 枠や棒の色はテーマで変わる"""
    use_palette(request.param)
    clear_waveform_images()
    yield str(request.param)
    use_palette(THEME_DARK)
    clear_waveform_images()


class _Scene:
    def __init__(self, video: MediaItem, audio: MediaItem) -> None:
        self.video = video
        self.loud = audio
        self.quiet = replace(audio, id=new_media_id(), path=Path("C:/素材/無音.wav"))
        base = Project.create(media=(self.video, self.loud, self.quiet))
        pictures = Track(
            TrackKind.VIDEO,
            "V1",
            (
                Clip(
                    timeline_start=0,
                    duration=THIN,
                    media_id=self.video.id,
                    stream_index=0,
                ),
                Clip(timeline_start=THIN, duration=THIN, source=TEXT.create()),
                Clip(
                    timeline_start=2 * THIN,
                    duration=WIDE,
                    media_id=self.video.id,
                    stream_index=0,
                ),
            ),
        )
        sounds = Track(
            TrackKind.AUDIO,
            "A1",
            (
                Clip(timeline_start=0, duration=THIN, media_id=self.loud.id, stream_index=0),
                Clip(timeline_start=THIN, duration=THIN, media_id=self.quiet.id, stream_index=0),
            ),
        )
        self.project = base.with_timeline(replace(base.timeline, tracks=(pictures, sounds)))
        self.analyzer = _Analyzer(
            {self.video.id: _filmstrip(RED)},
            {self.loud.id: _waveform(0.9), self.quiet.id: _waveform(0.0)},
        )
        self.view = TimelineView(self.project, self.analyzer)
        self.view.resize(900, 260)
        self.view._layout = TimelineLayout(pixels_per_frame=SCALE)

    def band(self, index: int) -> TrackBand:
        return self.view._layout.bands(self.view.project.timeline)[index]

    def clip(self, track: int, index: int) -> Clip:
        return self.view.project.timeline.tracks[track].clips[index]

    def left(self, clip: Clip) -> int:
        return int(self.view._layout.frame_to_x(clip.timeline_start))

    def render(self) -> QImage:
        image = QImage(self.view.size(), QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        self.view.render(painter, QPoint())
        painter.end()
        return image


@pytest.fixture
def scene(
    qt_application: QApplication, video_media: MediaItem, audio_media: MediaItem
) -> Iterator[_Scene]:
    del qt_application
    made = _Scene(video_media, audio_media)
    yield made
    made.analyzer.close()


def _close(found: QColor, wanted: QColor, slack: int = 6) -> bool:
    return all(
        abs(a - b) <= slack
        for a, b in (
            (found.red(), wanted.red()),
            (found.green(), wanted.green()),
            (found.blue(), wanted.blue()),
        )
    )


class TestWhatAThinBandShows:
    def test_the_clips_are_thin_in_this_scene(self, scene: _Scene) -> None:
        # 前提が崩れると、下の試験は名前の入る幅のクリップを見て通ってしまう
        assert THIN * SCALE < DETAIL_MIN_WIDTH <= WIDE * SCALE

    def test_a_thin_video_clip_is_filled_with_the_colour_of_its_pictures(
        self, scene: _Scene, theme: str
    ) -> None:
        # 無地の帯のままだと、素材の絵があるクリップと何も無いクリップが同じに見える
        clip = scene.clip(0, 0)
        band = scene.band(0)
        image = scene.render()
        x = scene.left(clip) + int(THIN * SCALE) // 2
        assert _close(image.pixelColor(x, band.top + band.height // 2), RED)

    def test_a_thin_clip_without_a_material_keeps_the_plain_band(
        self, scene: _Scene, theme: str
    ) -> None:
        # テキストのように素材の無いクリップまで塗ると、隣の絵の色が漏れたように見える
        clip = scene.clip(0, 1)
        band = scene.band(0)
        image = scene.render()
        x = scene.left(clip) + int(THIN * SCALE) // 2
        assert image.pixelColor(x, band.top + band.height // 2).name() == Colors.VIDEO_CLIP.name()

    def test_a_thin_sound_clip_shows_whether_it_is_loud_or_silent(
        self, scene: _Scene, theme: str
    ) -> None:
        # 音の帯が同じ見た目だと、無音の所を切り出したのかどうかを拡大するまで分からない
        band = scene.band(1)
        image = scene.render()

        def bar(clip: Clip) -> int:
            x = scene.left(clip) + int(THIN * SCALE) // 2
            return sum(
                image.pixelColor(x, y).name() == Colors.WAVEFORM.name()
                for y in range(band.top, band.bottom)
            )

        loud, quiet = bar(scene.clip(1, 0)), bar(scene.clip(1, 1))
        assert loud > band.height // 2
        # 無音でも 1 画素は残す 解析がまだの音（何も描かない）と見分けるため
        assert quiet == 1

    def test_a_turned_down_track_reads_as_silent(self, scene: _Scene, theme: str) -> None:
        # 音量を絞ったトラックの音まで大きく描くと、鳴らないのに鳴るように見える
        tracks = scene.view.project.timeline.tracks
        quiet = replace(tracks[1], volume_db=-60.0)
        project = scene.project.with_timeline(
            replace(scene.project.timeline, tracks=(tracks[0], quiet))
        )
        scene.view.set_project(project)
        band = scene.band(1)
        image = scene.render()
        x = scene.left(scene.clip(1, 0)) + int(THIN * SCALE) // 2
        column = [image.pixelColor(x, y).name() for y in range(band.top, band.bottom)]
        assert column.count(Colors.WAVEFORM.name()) == 1


class TestChosenThinBands:
    def test_a_chosen_thin_clip_has_a_thick_frame(self, scene: _Scene, theme: str) -> None:
        # 2 画素の枠は、細い帯が並んだ所で境目の線と同じ縞に見えて、どれを選んだか分からない
        clip = scene.clip(0, 1)
        scene.view.select(clip.id)
        band = scene.band(0)
        image = scene.render()
        left = scene.left(clip)
        middle = band.top + band.height // 2
        for x in range(left, left + DENSE_MARK_WIDTH):
            assert image.pixelColor(x, middle).name() == Colors.SELECTION.name()

    def test_the_clip_in_the_settings_panel_is_told_apart_from_its_partners(
        self, scene: _Scene, theme: str
    ) -> None:
        # 枠の内側に細い色の枠を描いていたときは、数画素の帯では潰れて見えなかった
        partner, editing = scene.clip(0, 0), scene.clip(0, 1)
        scene.view.set_selection([partner.id, editing.id])
        band = scene.band(0)
        image = scene.render()
        row = band.top + 1 + DENSE_MARK_WIDTH + 1
        x = scene.left(editing) + int(THIN * SCALE) // 2
        assert image.pixelColor(x, row).name() == Colors.EDITING.name()
        assert image.pixelColor(scene.left(partner) + 7, row).name() != Colors.EDITING.name()

    def test_a_one_pixel_clip_still_gets_a_visible_frame(
        self, qt_application: QApplication, theme: str
    ) -> None:
        # 全体表示の 1 画素の帯では、枠が帯の幅に潰れて線 1 本にしか見えなかった
        del qt_application
        base = Project.create()
        track = Track(
            TrackKind.VIDEO,
            "V1",
            tuple(
                Clip(timeline_start=n * 10, duration=10, source=TEXT.create()) for n in range(200)
            ),
        )
        project = base.with_timeline(replace(base.timeline, tracks=(track,)))
        analyzer = MediaAnalyzer()
        try:
            view = TimelineView(project, analyzer)
            view.resize(900, 200)
            view._layout = TimelineLayout(
                pixels_per_frame=(900 - Metrics.TRACK_HEADER_WIDTH) / project.duration
            )
            clip = track.clips[100]
            view.select(clip.id)
            band = view._layout.bands(project.timeline)[0]
            image = QImage(view.size(), QImage.Format.Format_ARGB32)
            painter = QPainter(image)
            view.render(painter, QPoint())
            painter.end()
            x = int(view._layout.frame_to_x(clip.timeline_start))
            row = [image.pixelColor(x + dx, band.top + 1).name() for dx in range(-3, 4)]
            assert row.count(Colors.SELECTION.name()) >= 6
        finally:
            analyzer.close()


def _hover(view: TimelineView, position: QPoint) -> str:
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(position),
        QPointF(view.mapToGlobal(position)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mouseMoveEvent(event)
    return view.toolTip()


class TestHoverTip:
    def test_a_thin_clip_tells_its_name_and_length(self, scene: _Scene) -> None:
        # 細い帯には名前を描かないので、載せても何も出ないと、選んで設定パネルを見るしかない
        clip = scene.clip(0, 0)
        band = scene.band(0)
        tip = _hover(scene.view, QPoint(scene.left(clip) + 5, band.top + band.height // 2))
        assert scene.video.name in tip
        assert "長さ 1.00 秒" in tip

    def test_a_wide_clip_shows_no_tip(self, scene: _Scene) -> None:
        # 名前が見えているクリップに同じ名前を重ねると、編集の邪魔になる
        clip = scene.clip(0, 2)
        band = scene.band(0)
        tip = _hover(scene.view, QPoint(scene.left(clip) + 40, band.top + band.height // 2))
        assert tip == ""


class TestDetailWidthSetting:
    def test_lowering_the_width_draws_short_clips_in_full(
        self, scene: _Scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 細かく見たい人が下限を下げても、決め打ちの 24 のまま細い帯で描かれては意味がない
        detailed: list[Clip] = []
        original = scene.view._paint_detailed

        def record(
            painter: QPainter,
            band: TrackBand,
            clip: Clip,
            rect: QRect,
            selected: bool,
            editing: bool = False,
        ) -> None:
            detailed.append(clip)
            original(painter, band, clip, rect, selected, editing)

        monkeypatch.setattr(scene.view, "_paint_detailed", record)
        scene.render()
        assert scene.clip(0, 0) not in detailed
        scene.view.set_detail_min_width(8)
        scene.render()
        assert scene.clip(0, 0) in detailed

    def test_the_width_is_kept_within_the_range(self, scene: _Scene) -> None:
        # 0 を通すと、1 画素のクリップまで 1 本ずつ描いて全体表示が重くなる
        scene.view.set_detail_min_width(0)
        assert scene.view.detail_min_width == DETAIL_MIN_WIDTHS[0]
        scene.view.set_detail_min_width(10_000)
        assert scene.view.detail_min_width == DETAIL_MIN_WIDTHS[1]

    def test_the_default_matches_the_timeline(self) -> None:
        # 既定がずれると、設定を触っていない人の見た目が版を上げただけで変わる
        assert Preferences().detail_min_width == DETAIL_MIN_WIDTH
        assert DETAIL_MIN_WIDTHS[0] <= DETAIL_MIN_WIDTH <= DETAIL_MIN_WIDTHS[1]

    def test_the_setting_is_saved_and_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        # 壊れた値（範囲の外）で起動を止めず、既定へ戻す
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(replace(Preferences(), detail_min_width=12))
        assert store.load().detail_min_width == 12
        (tmp_path / "preferences.json").write_text('{"detail_min_width": 3}', encoding="utf-8")
        assert store.load().detail_min_width == DETAIL_MIN_WIDTH

    def test_the_settings_window_carries_the_width(self, qt_application: QApplication) -> None:
        # 設定の窓に出ていないと、好みが分かれるのに変える手段が無い
        del qt_application
        dialog = PreferencesDialog(replace(Preferences(), detail_min_width=12))
        try:
            assert dialog.preferences().detail_min_width == 12
        finally:
            dialog.deleteLater()
