"""混合レイヤーの画面（#27 YMM4 型アイテムの設計 P4b）

YMM4 のように 1 本のレイヤーに何でも置ける方式をタイムラインに出す
利用者の決定 レイヤー 1 が一番上に並び、番号が大きい（下の）レイヤーほど手前に描く
ここが崩れると、混合のプロジェクトでクリップが画面に出ない・掴めない・音付きの動画の
音が見えない・新しく作ったプロジェクトが分ける方式のまま、が起きる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QMimeData, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QDragMoveEvent, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from sashimono.core.commands import (
    AddClip,
    Command,
    MoveClip,
    MoveClips,
    MoveTrack,
)
from sashimono.core.commands.fixed import VOLUME_EFFECT_KIND, with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    AudioStreamInfo,
    Clip,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Timeline,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.audio.waveform import BASE_SAMPLES_PER_PEAK, PeakLevel, Waveform
from sashimono.engine.cache import Filmstrip, MediaAnalyzer
from sashimono.ui.media_pool import MEDIA_MIME
from sashimono.ui.theme import Colors
from sashimono.ui.timeline import TimelineArea, TimelineView
from sashimono.ui.timeline import painter as painter_module
from sashimono.ui.timeline import view as view_module
from sashimono.ui.timeline.layout import TimelineLayout, TrackBand
from sashimono.ui.timeline.painter import draw_clip
from sashimono.ui.workspace import Preferences, PreferenceStore

_LEFT = Qt.MouseButton.LeftButton
_NONE = Qt.KeyboardModifier.NoModifier

MIXED = ProjectSettings(layer_mode=LayerMode.MIXED)


def _layer(number: int, *clips: Clip) -> Track:
    return Track(TrackKind.MIXED, f"レイヤー {number}", clips)


def _text(start: int, duration: int = 100) -> Clip:
    return Clip(timeline_start=start, duration=duration, source=TEXT.create())


def _project(*tracks: Track, media: tuple[MediaItem, ...] = ()) -> Project:
    base = Project.create(MIXED, media=media)
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _with_sound(media: MediaItem, start: int = 10, duration: int = 90) -> Clip:
    """音付きの動画 レイヤーの 1 本のクリップで絵も音も持つ"""
    clip = Clip(
        timeline_start=start,
        duration=duration,
        media_id=media.id,
        stream_index=media.video_streams[0].index,
        audio_stream=media.audio_streams[0].index,
    )
    return with_fixed_items(clip, picture=True, sound=True)


def _sound_only(media: MediaItem, start: int = 10) -> Clip:
    clip = Clip(
        timeline_start=start,
        duration=90,
        media_id=media.id,
        stream_index=0,
        audio_stream=media.audio_streams[0].index,
        show_picture=False,
    )
    return with_fixed_items(clip, sound=True)


@pytest.fixture
def analyzer() -> Iterator[MediaAnalyzer]:
    created = MediaAnalyzer(sample_rate=48000, channels=2)
    yield created
    created.close()


class _Harness:
    """窓の代わり 出たコマンドを当てて、プロジェクトを差し戻す"""

    def __init__(self, view: TimelineView) -> None:
        self.view = view
        self.received: list[list[Command]] = []
        view.commands_requested.connect(self._apply)

    def _apply(self, commands: list[Command], _label: str) -> None:
        self.received.append(list(commands))
        project = self.view.project
        for command in commands:
            project = command.apply(project)
        self.view.set_project(project)


@pytest.fixture
def made(qt_application: QApplication) -> Iterator[list[TimelineArea]]:
    del qt_application
    areas: list[TimelineArea] = []
    yield areas
    for area in areas:
        area.close()
        shiboken6.delete(area)


def _open(
    areas: list[TimelineArea], analyzer: MediaAnalyzer, project: Project
) -> tuple[TimelineView, _Harness]:
    area = TimelineArea(TimelineView(project, analyzer))
    area.resize(900, 400)
    area.show()
    QApplication.processEvents()
    areas.append(area)
    harness = _Harness(area.view)
    area.view.setProperty("harness", harness)
    return area.view, harness


def _diag(view: TimelineView) -> str:
    import os

    from tests import conftest

    area = view.window()
    handle = area.windowHandle()
    return (
        f"view={view.size()} area={area.size()} visible={view.isVisible()}"
        f" mapped={view.testAttribute(Qt.WidgetAttribute.WA_Mapped)}"
        f" exposed={handle.isExposed() if handle else None}"
        f" updates={view.updatesEnabled()} active={QApplication.activeWindow()}"
        f" grab={view.mouseGrabber()} buttons={QApplication.mouseButtons()}"
        f" tops={[(type(w).__name__, w.isVisible()) for w in QApplication.topLevelWidgets()]}"
        f" bands={[(b.track.name, b.top, b.height) for b in view.view_layout.bands(view.project.timeline)]}"
        f" worker={os.environ.get('PYTEST_XDIST_WORKER')} recent={list(conftest.RECENT)}"
    )


def _band(view: TimelineView, name: str) -> tuple[int, int]:
    for band in view.view_layout.bands(view.project.timeline):
        if band.track.name == name:
            return band.top, band.bottom
    raise AssertionError(f"トラックが画面に無い: {name}")


def _point(view: TimelineView, name: str, frame: int) -> QPoint:
    top, bottom = _band(view, name)
    return QPoint(int(view.view_layout.frame_to_x(frame)), (top + bottom) // 2)


def _drag(view: TimelineView, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(view, _LEFT, _NONE, start)
    QTest.mouseMove(view, QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, _LEFT, _NONE, end)


def _names(timeline: Timeline) -> list[str]:
    return [band.track.name for band in TimelineLayout().bands(timeline)]


# --- 並べ方 ---


class TestLayerOrder:
    def test_layer_one_is_at_the_top_and_the_numbers_go_down(self) -> None:
        # 利用者の決定 前はレイヤーを 1 本も並べず、混合のプロジェクトが空に見えた
        project = _project(_layer(1), _layer(2), _layer(3))
        assert _names(project.timeline) == ["レイヤー 1", "レイヤー 2", "レイヤー 3"]

    def test_the_separated_order_does_not_change(self) -> None:
        # 分ける方式の作品は今までどおり V を逆さに、A を順に
        tracks = (
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.VIDEO, "V2"),
            Track(TrackKind.AUDIO, "A1"),
            Track(TrackKind.AUDIO, "A2"),
        )
        assert _names(_project(*tracks).timeline) == ["V2", "V1", "A1", "A2"]

    def test_layers_sit_between_the_video_and_the_audio(self) -> None:
        # 方式を切り替える途中の作品 絵だけの欄を上、音だけの欄を下、両方持つレイヤーを間に
        tracks = (
            Track(TrackKind.VIDEO, "V1"),
            _layer(1),
            Track(TrackKind.AUDIO, "A1"),
            Track(TrackKind.VIDEO, "V2"),
            _layer(2),
        )
        assert _names(_project(*tracks).timeline) == ["V2", "V1", "レイヤー 1", "レイヤー 2", "A1"]

    @pytest.mark.parametrize("family", ["", "Yu Gothic UI", "Meiryo UI", "Meiryo"])
    @pytest.mark.parametrize("points", [9.0, 10.0])
    def test_the_layer_name_is_not_cut_in_the_header(
        self, qt_application: QApplication, family: str, points: float
    ) -> None:
        # 前はヘッダが狭く、「レイヤー 1」が「レイ… 1」に切れて読めなかった
        # 空の family は画面に使う既定のフォント
        from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QFontMetrics

        from sashimono.ui.timeline.painter import shown_track_name

        del qt_application
        # 無いフォントは黙って別のフォントに替わる 替わった物を測っても、Windows の
        # 既定のフォントで切れないことは確かめられないので飛ばす
        if family and family not in QFontDatabase.families():
            pytest.skip(f"{family} が入っていない機械")
        font = QFont(family) if family else QApplication.font()
        font.setPointSizeF(points)
        # 既定のフォントも同じ Windows で QT_QPA_PLATFORM=offscreen にすると、Qt は
        # フォントを 1 つも読まず（families() が空）、既定の "Sans Serif" はどの実物にも
        # 当たらない 測る物は 1 文字をどれも 1 字幅の四角とする代わりで、「レイヤー 100」
        # が 96 px になって 84 px の欄からはみ出す 画面に出る字の幅ではないので飛ばす
        if QFontInfo(font).family() not in QFontDatabase.families():
            pytest.skip("既定のフォントがどの実物にも当たらない（offscreen など）")
        metrics = QFontMetrics(font)
        for number in (1, 10, 100):
            track = _layer(number)
            assert shown_track_name(track, metrics) == track.name
        assert shown_track_name(Track(TrackKind.MIXED), metrics) == "レイヤー"

    def test_the_name_fits_beside_the_buttons_at_the_smallest_height(self) -> None:
        # 名前とボタンは 1 行 最小の高さ（28）でも、名前の所もボタンも帯の中に収まる
        from sashimono.ui.theme import Metrics
        from sashimono.ui.timeline.painter import track_button_rects, track_name_rect

        band = TrackBand(_layer(1), 100, Metrics.MIN_TRACK_HEIGHT)
        name = track_name_rect(band)
        buttons = [rect for _, _, rect in track_button_rects(band)]
        assert name.right() < buttons[0].left()
        assert all(band.top <= r.top() and r.bottom() < band.bottom for r in (name, *buttons))
        assert buttons[-1].right() < Metrics.TRACK_HEADER_WIDTH

    def test_every_track_gets_a_band(self) -> None:
        # 帯を持たないトラックがあると、縦の長さ（content_height）と並びが食い違い、
        # 一番下のトラックまで送れない
        tracks = (_layer(1), Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1"))
        timeline = _project(*tracks).timeline
        layout = TimelineLayout()
        bands = layout.bands(timeline)
        assert len(bands) == len(timeline.tracks)
        assert bands[-1].bottom == layout.content_height(timeline)


# --- クリップの中身 ---


def _filmstrip(colour: tuple[int, int, int]) -> Filmstrip:
    sheet = np.zeros((40, 64 * 4, 4), dtype=np.uint8)
    sheet[:, :, 0], sheet[:, :, 1], sheet[:, :, 2] = colour
    sheet[:, :, 3] = 255
    return Filmstrip(sheet=sheet, tile_width=64, interval=Fraction(1))


def _loud_waveform() -> Waveform:
    count = 4000
    peaks = np.zeros((count, 1, 2), dtype=np.float32)
    peaks[:, :, 0] = -1.0
    peaks[:, :, 1] = 1.0
    return Waveform(
        sample_rate=48000,
        channels=1,
        total_samples=count * BASE_SAMPLES_PER_PEAK,
        levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),),
    )


_RED = (230, 20, 20)


def _painted(track: Track, clip: Clip, media: MediaItem | None) -> QImage:
    """クリップ 1 本を 400 × 100 の絵に描く サムネイルは赤一色、波形は振り切った音"""
    image = QImage(400, 100, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0))
    layout = TimelineLayout(pixels_per_frame=4.0)
    band = TrackBand(track=track, top=0, height=100)
    rect = QRect(
        int(layout.frame_to_x(clip.timeline_start)),
        1,
        int(clip.duration * layout.pixels_per_frame),
        97,
    )
    painter = QPainter(image)
    try:
        draw_clip(
            painter,
            clip,
            band,
            layout,
            FrameRate(30),
            media=media,
            filmstrip=_filmstrip(_RED),
            waveform=_loud_waveform(),
            selected=False,
            clip_rect=rect.intersected(QRect(0, 0, 400, 100)),
        )
    finally:
        painter.end()
    return image


def _rows_with(image: QImage, x: int, match: object) -> list[int]:
    assert callable(match)
    return [y for y in range(image.height()) if match(image.pixelColor(x, y))]


def _is_red(colour: QColor) -> bool:
    return colour.red() > 180 and colour.green() < 60 and colour.blue() < 60


def _is_wave(colour: QColor) -> bool:
    return colour.rgb() == Colors.WAVEFORM.rgb()


class TestClipContent:
    def test_what_each_clip_draws(self, video_media: MediaItem, audio_media: MediaItem) -> None:
        # 種類だけで決めると、レイヤーの音付き動画に波形が出ず、BGM にサムネイルを探しに行く
        from sashimono.ui.timeline.painter import clip_content

        layer = _layer(1)
        assert clip_content(layer, _with_sound(video_media), video_media) == (True, True)
        assert clip_content(layer, _sound_only(audio_media), audio_media) == (False, True)
        assert clip_content(layer, _text(0), None) == (True, False)
        assert clip_content(Track(TrackKind.VIDEO), _text(0), None) == (True, False)
        audio_clip = Clip(timeline_start=0, duration=30, media_id=audio_media.id)
        assert clip_content(Track(TrackKind.AUDIO), audio_clip, audio_media) == (False, True)

    def test_a_video_with_sound_shows_the_picture_above_and_the_wave_below(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 前はレイヤーのクリップを音声として描き、サムネイルを出さずに波形だけを敷いていた
        del qt_application
        clip = _with_sound(video_media, start=0, duration=90)
        image = _painted(_layer(1), clip, video_media)
        x = 200
        red = _rows_with(image, x, _is_red)
        wave = _rows_with(image, x, _is_wave)
        assert red, "サムネイルが描かれていない"
        assert wave, "波形が描かれていない"
        assert max(red) < min(wave), "絵が上、波形が下に分かれていない"

    def test_a_sound_on_a_layer_is_a_wave_in_the_audio_colour(
        self, qt_application: QApplication, audio_media: MediaItem
    ) -> None:
        # 音だけの物を映像の色で塗ると、どれが BGM なのかを名前を読むまで見分けられない
        del qt_application
        clip = _sound_only(audio_media, start=0)
        image = _painted(_layer(1), clip, audio_media)
        assert not _rows_with(image, 200, _is_red)
        assert _rows_with(image, 200, _is_wave)
        # 名前の帯（上の 18 画素）と枠を外した、波形の無い所の色
        body = image.pixelColor(200, 99 - 3)
        assert body.rgb() in (Colors.AUDIO_CLIP.rgb(), Colors.WAVEFORM.rgb())

    def test_a_picture_only_clip_has_no_wave(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 音を鳴らさないクリップに波形を出すと、鳴っているように見え、無音の理由を探して
        # 音量を触ることになる
        del qt_application
        clip = replace(_with_sound(video_media, start=0), audio_stream=None)
        image = _painted(_layer(1), clip, video_media)
        assert _rows_with(image, 200, _is_red)
        assert not _rows_with(image, 200, _is_wave)

    def test_a_layer_header_is_not_greyed_out(
        self,
        made: list[TimelineArea],
        analyzer: MediaAnalyzer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 種類ごとに「出ているか」を集めると、レイヤーはどちらにも入らず、ミュートもソロも
        # していないのに名前が薄く出る
        seen: dict[str, bool] = {}
        original = painter_module.draw_track_header

        def record(painter: QPainter, band: TrackBand, *, active: bool = True) -> None:
            seen[band.track.name] = active
            original(painter, band, active=active)

        monkeypatch.setattr(view_module, "draw_track_header", record)
        muted = replace(_layer(2), muted=True)
        view, _ = _open(made, analyzer, _project(_layer(1), muted))
        view.repaint()
        assert seen == {"レイヤー 1": True, "レイヤー 2": False}, _diag(view)


# --- 動かす ---


class TestMovingOnLayers:
    def test_dragging_a_clip_down_moves_it_to_the_next_layer(
        self, made: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # レイヤー 2 はレイヤー 1 の下 下へ運んだら、番号の大きい（手前の）レイヤーへ入る
        clip = _text(20)
        view, harness = _open(made, analyzer, _project(_layer(1, clip), _layer(2)))
        second = view.project.timeline.tracks[1]
        _drag(view, _point(view, "レイヤー 1", 60), _point(view, "レイヤー 2", 60))
        ((command,),) = harness.received
        assert isinstance(command, MoveClip)
        assert command.track_id == second.id
        assert [len(t.clips) for t in view.project.timeline.tracks] == [0, 1]

    def test_moving_several_clips_shifts_them_by_layers(
        self,
        made: list[TimelineArea],
        analyzer: MediaAnalyzer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # まとめて動かしたときも同じ数え方 途中の枠は行き先のレイヤーに出す
        # （前は元のレイヤーに枠を出し、離すと別のレイヤーへ移っていた）
        first, second = _text(0, 50), _text(100, 50)
        view, harness = _open(made, analyzer, _project(_layer(1, first, second), _layer(2)))
        view.set_selection((first.id, second.id))
        target = view.project.timeline.tracks[1]
        dashed: list[str] = []
        original = view._dash_rect

        def record(painter: QPainter, band: TrackBand, start: int, end: int) -> None:
            dashed.append(band.track.name)
            original(painter, band, start, end)

        monkeypatch.setattr(view, "_dash_rect", record)
        start = _point(view, "レイヤー 1", 25)
        end = _point(view, "レイヤー 2", 25)
        QTest.mousePress(view, _LEFT, _NONE, start)
        QTest.mouseMove(view, QPoint(start.x(), (start.y() + end.y()) // 2))
        QTest.mouseMove(view, end)
        view.repaint()
        assert dashed and set(dashed) == {"レイヤー 2"}, _diag(view)
        QTest.mouseRelease(view, _LEFT, _NONE, end)
        ((command,),) = harness.received
        assert isinstance(command, MoveClips)
        assert command.track_delta == 1
        landed = view.project.timeline.find_track(target.id)
        assert landed is not None
        assert {c.id for c in landed.clips} == {first.id, second.id}

    @pytest.mark.parametrize("blocked", ["edge", "locked"])
    def test_when_one_clip_cannot_cross_all_move_in_time_only(
        self,
        made: list[TimelineArea],
        analyzer: MediaAnalyzer,
        monkeypatch: pytest.MonkeyPatch,
        blocked: str,
    ) -> None:
        # 1 本でも並びの外（edge）かロックしたレイヤー（locked）へ入るなら MoveClips は全員を断る
        # 前は入れる物の枠だけを行き先に出し、離すと何も動かず断られた
        # 枠も命令も、レイヤーを跨がずに時間だけ動かす形にそろえる
        first, second = _text(0, 50), _text(0, 50)
        third = replace(_layer(3), locked=blocked == "locked")
        layers: tuple[Track, ...] = (_layer(1, first), _layer(2, second))
        if blocked == "locked":
            layers = (*layers, third)
        view, harness = _open(made, analyzer, _project(*layers))
        view.set_selection((first.id, second.id))
        dashed: list[str] = []
        original = view._dash_rect

        def record(painter: QPainter, band: TrackBand, start: int, end: int) -> None:
            dashed.append(band.track.name)
            original(painter, band, start, end)

        monkeypatch.setattr(view, "_dash_rect", record)
        start = _point(view, "レイヤー 1", 25)
        end = _point(view, "レイヤー 2", 65)
        QTest.mousePress(view, _LEFT, _NONE, start)
        QTest.mouseMove(view, QPoint(start.x(), (start.y() + end.y()) // 2))
        QTest.mouseMove(view, end)
        view.repaint()
        assert sorted(dashed) == ["レイヤー 1", "レイヤー 2"], "枠が元のレイヤーに出ていない"
        QTest.mouseRelease(view, _LEFT, _NONE, end)
        ((command,),) = harness.received
        assert isinstance(command, MoveClips)
        assert command.track_delta == 0
        assert command.delta == 40
        tracks = view.project.timeline.tracks
        assert [[c.timeline_start for c in t.clips] for t in tracks[:2]] == [[40], [40]]

    def test_a_layer_header_drags_to_a_new_place(
        self, made: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # レイヤー 1 を一番下へ運ぶと、一番手前に重なる（並びの末尾）
        view, harness = _open(made, analyzer, _project(_layer(1), _layer(2), _layer(3)))
        top, bottom = _band(view, "レイヤー 1")
        _, last = _band(view, "レイヤー 3")
        start = QPoint(20, (top + bottom) // 2)
        end = QPoint(20, last - 2)
        _drag(view, start, end)
        ((command,),) = harness.received
        assert isinstance(command, MoveTrack)
        assert [t.name for t in view.project.timeline.tracks] == [
            "レイヤー 2",
            "レイヤー 3",
            "レイヤー 1",
        ]
        assert _names(view.project.timeline) == ["レイヤー 2", "レイヤー 3", "レイヤー 1"]


# --- 落とし込み ---


def _media_mime(*media: MediaItem) -> QMimeData:
    mime = QMimeData()
    mime.setData(MEDIA_MIME, "\n".join(str(item.id) for item in media).encode("utf-8"))
    return mime


class TestDroppingOnLayers:
    def test_the_guide_shows_the_clip_on_the_layer_under_the_pointer(
        self, made: list[TimelineArea], analyzer: MediaAnalyzer, video_media: MediaItem
    ) -> None:
        # 前は帯が無く、どこへ引いても「トラックの無い所」になって新しいレイヤーを作っていた
        project = _project(_layer(1), _layer(2), media=(video_media,))
        view, _ = _open(made, analyzer, project)
        # 映像と音声を 1 本にまとめる設定で、引いたレイヤーへそのまま入ることを見る
        # （分ける設定は、下の試験と tests/ui/test_multi_audio_entrances.py）
        view.set_split_audio(False)
        second = project.timeline.tracks[1]
        point = _point(view, "レイヤー 2", 40)
        # 中身は変数に持っておく 渡しただけだと、イベントより先に捨てられて落ちる
        mime = _media_mime(video_media)
        event = QDragMoveEvent(point, Qt.DropAction.CopyAction, mime, _LEFT, _NONE)
        view.dragMoveEvent(event)
        preview = view.drop_preview
        assert preview is not None
        assert preview.guide.spot.track_id == second.id
        (added,) = [c for c in preview.commands if isinstance(c, AddClip)]
        assert added.track_id == second.id
        assert not preview.new_tracks
        assert view.drop_spot_at(QPointF(point)).track_id == second.id

    def test_the_guide_shows_the_picture_under_the_pointer_and_the_sound_after_it(
        self, made: list[TimelineArea], analyzer: MediaAnalyzer, video_media: MediaItem
    ) -> None:
        # 既定（分ける）では、絵が引いたレイヤーへ、音がその次のレイヤーへ入る目安を出す
        # 目安が 1 本だけだと、落とした後に音のレイヤーが目安に無い所へ増える
        project = _project(_layer(1), _layer(2), media=(video_media,))
        view, _ = _open(made, analyzer, project)
        first, second = project.timeline.tracks
        mime = _media_mime(video_media)
        point = _point(view, "レイヤー 1", 40)
        view.dragMoveEvent(QDragMoveEvent(point, Qt.DropAction.CopyAction, mime, _LEFT, _NONE))
        preview = view.drop_preview
        assert preview is not None
        picture, sound = [c for c in preview.commands if isinstance(c, AddClip)]
        assert picture.track_id == first.id and picture.clip.show_picture
        assert sound.track_id == second.id and not sound.clip.show_picture


# --- 値の線 ---


def _clip_on(view: TimelineView, clip: Clip) -> Clip:
    located = view.project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[1]


def _volume(clip: Clip) -> float:
    effect = next(e for e in clip.effects if e.fixed and e.kind == VOLUME_EFFECT_KIND)
    value = effect.params["volume"]
    assert isinstance(value, AnimatedValue)
    return value.static


class TestValueLineOnALayer:
    def test_the_right_click_switches_a_video_with_sound_to_its_volume(
        self, made: list[TimelineArea], analyzer: MediaAnalyzer, video_media: MediaItem
    ) -> None:
        # P8 の残り 音付きの動画の線を右クリックから音量へ切り替え、線を動かして音量が変わる
        # レイヤーが並ぶ前は、線の受け持ちを直に確かめるしかなかった
        from sashimono.ui.timeline.painter import clip_rect_for
        from sashimono.ui.timeline.value_line import (
            SHOW_OPACITY_TEXT,
            SHOW_VOLUME_TEXT,
            line_area,
        )

        clip = _with_sound(video_media)
        view, harness = _open(made, analyzer, _project(_layer(1, clip), media=(video_media,)))
        menu = view.build_context_menu(_point(view, "レイヤー 1", 50))
        actions = {a.text(): a for a in menu.actions()}
        assert actions[SHOW_OPACITY_TEXT].isChecked()
        actions[SHOW_VOLUME_TEXT].trigger()
        menu.deleteLater()

        band = view.view_layout.bands(view.project.timeline)[0]
        rect = clip_rect_for(_clip_on(view, clip), band, view.view_layout, view.width())
        assert rect is not None
        area = line_area(rect)
        assert area is not None
        x = int(view.view_layout.frame_to_x(50))
        # 100% は下から 4 分の 1 の高さ（上限 400%） 半分の高さまで上げると 200%
        start = QPoint(x, round(area.bottom() - area.height() / 4))
        end = QPoint(x, round(area.bottom() - area.height() / 2))
        _drag(view, start, end)
        assert harness.received, "線を掴めていない"
        moved = _clip_on(view, clip)
        assert _volume(moved) == pytest.approx(200.0, abs=15)
        assert moved.opacity == clip.opacity, "不透明度の線が動いた"


# --- 設定パネルの見出し ---


class TestInspectorOnALayer:
    def test_the_heading_says_what_the_layer_clip_is(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 種類で決めると、レイヤーの BGM が「映像」、名前の無いレイヤーが「音声トラック」と出る
        from sashimono.ui.inspector.header import identify_clip

        movie = _with_sound(video_media)
        music = _sound_only(audio_media, start=200)
        project = _project(
            _layer(1, movie),
            Track(TrackKind.MIXED, "", (music,)),
            media=(video_media, audio_media),
        )
        first = identify_clip(project, movie.id)
        assert first is not None
        assert (first.kind, first.name, first.track) == ("映像（音付き）", "本編.mp4", "レイヤー 1")
        assert first.color == Colors.VIDEO_CLIP_BORDER
        second = identify_clip(project, music.id)
        assert second is not None
        assert (second.kind, second.track) == ("音声", "レイヤー 2")
        assert second.color == Colors.AUDIO_CLIP_BORDER

    def test_a_sound_on_a_layer_shows_only_the_sound_group(
        self, qt_application: QApplication, audio_media: MediaItem
    ) -> None:
        # 音だけの物に描画の組を出すと、動かしても何も変わらない欄を触らせる
        from sashimono.ui.inspector.panel import InspectorPanel, _Section

        del qt_application
        music = _sound_only(audio_media)
        project = _project(_layer(1, music), media=(audio_media,))
        panel = InspectorPanel()
        try:
            panel.set_project(project)
            panel.set_clip(music.id)
            headings = [s.heading for s in panel._body.findChildren(_Section)]
            assert headings == ["音声"]
        finally:
            panel.close()
            shiboken6.delete(panel)


# --- 新しいプロジェクトの方式 ---


class TestNewProjectLayers:
    def test_the_preference_defaults_to_mixed_and_is_kept(self, tmp_path: Path) -> None:
        # 利用者の要望 既定は混合 分ける方式を選んだ人は次の起動でもそのまま
        assert Preferences().new_project_layers == LayerMode.MIXED
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(new_project_layers=LayerMode.SEPARATED))
        assert store.load().new_project_layers == LayerMode.SEPARATED

    def test_an_unknown_value_falls_back_to_mixed(self, tmp_path: Path) -> None:
        # 知らない値のまま持つと、起動した直後の空のプロジェクトを作る所で ProjectSettings が
        # ValueError を出し、窓が開かない（新しい版で足した値を古い版で読んだときなど）
        path = tmp_path / "preferences.json"
        path.write_text('{"new_project_layers": "layered"}', encoding="utf-8")
        assert PreferenceStore(path).load().new_project_layers == LayerMode.MIXED

    def test_the_model_default_stays_separated(self) -> None:
        # 古いファイルと、既定の設定で組み立てる試験の動きを変えない（設計の決め）
        assert ProjectSettings().layer_mode == LayerMode.SEPARATED

    def test_the_preferences_dialog_carries_it(self, qt_application: QApplication) -> None:
        # 画面が値を返さないと、設定を開いて OK を押しただけで分ける方式の好みが混合へ戻る
        from sashimono.ui.preferences_dialog import PreferencesDialog

        del qt_application
        dialog = PreferencesDialog(Preferences(new_project_layers=LayerMode.SEPARATED))
        try:
            assert dialog.preferences().new_project_layers == LayerMode.SEPARATED
        finally:
            dialog.deleteLater()

    def test_the_new_project_dialog_starts_from_the_given_mode(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 作るときに選べないと、分けたい作品のたびに設定を切り替えることになる
        from PySide6.QtWidgets import QComboBox

        from sashimono.ui.project_presets import ProjectPresetStore
        from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

        del qt_application
        presets = ProjectPresetStore(tmp_path / "presets.json")
        dialog = ProjectSettingsDialog(MIXED, new=True, presets=presets)
        try:
            assert dialog.settings().layer_mode == LayerMode.MIXED
            boxes = [
                box
                for box in dialog.findChildren(QComboBox)
                if box.findData(LayerMode.SEPARATED) >= 0
            ]
            (box,) = boxes
            box.setCurrentIndex(box.findData(LayerMode.SEPARATED))
            assert dialog.settings().layer_mode == LayerMode.SEPARATED
        finally:
            dialog.deleteLater()

    def test_the_settings_of_an_open_project_do_not_switch_the_mode(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 作ったあとに方式だけを黙って変えると、置いてあるトラックはそのままで置き方だけが変わる
        from sashimono.ui.project_presets import ProjectPresetStore
        from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

        del qt_application
        presets = ProjectPresetStore(tmp_path / "presets.json")
        dialog = ProjectSettingsDialog(MIXED, presets=presets)
        try:
            assert dialog.settings().layer_mode == LayerMode.MIXED
        finally:
            dialog.deleteLater()


class TestTheWindowFollowsThePreference:
    def test_the_first_project_is_mixed(self, qt_application: QApplication) -> None:
        # 起動した直後の空のプロジェクトも新規作成と同じ 前はモデルの既定（分ける）だった
        from sashimono.ui.main_window import MainWindow

        del qt_application
        window = MainWindow(confirm_unsaved=False)
        try:
            assert window.document.project.settings.layer_mode == LayerMode.MIXED
        finally:
            window.close()

    def test_the_first_project_follows_a_separated_preference(
        self, qt_application: QApplication
    ) -> None:
        # 好みを見ずに混合で作ると、分ける方式を選んだ人も起動するたびに混合の空の
        # プロジェクトから始まり、素材を置くと 1 本のレイヤーにまとまってしまう
        from sashimono.ui.main_window import MainWindow

        del qt_application
        PreferenceStore().save(Preferences(new_project_layers=LayerMode.SEPARATED))
        window = MainWindow(confirm_unsaved=False)
        try:
            assert window.document.project.settings.layer_mode == LayerMode.SEPARATED
        finally:
            window.close()

    def test_new_starts_from_the_preference(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 〔新規〕の窓の初めの値が好み OK だけで押せば、その方式で作る
        from sashimono.ui.main_window import MainWindow
        from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

        del qt_application
        monkeypatch.setattr(
            ProjectSettingsDialog, "exec", lambda _self: QDialog.DialogCode.Accepted.value
        )
        window = MainWindow(_project(Track(TrackKind.VIDEO, "V1")), confirm_unsaved=False)
        try:
            window._preferences = Preferences(new_project_layers=LayerMode.MIXED)
            window.new_project()
            assert window.document.project.settings.layer_mode == LayerMode.MIXED
            window._preferences = Preferences(new_project_layers=LayerMode.SEPARATED)
            window.new_project()
            assert window.document.project.settings.layer_mode == LayerMode.SEPARATED
        finally:
            window.close()


# --- 置いた物を選ぶ ---


class TestPlacingOnALayerSelectsIt:
    def test_a_text_added_from_the_menu_is_selected(self, qt_application: QApplication) -> None:
        # 置いた物を映像トラックの中から探していたので、レイヤーに置いたテキストが選ばれず、
        # 追加したのに設定パネルが開かなかった
        from sashimono.ui.main_window import MainWindow

        del qt_application
        window = MainWindow(Project.create(MIXED), confirm_unsaved=False)
        try:
            window._insert_generated(TEXT.create(), "テキストを追加")
            placed = [c for t in window.document.project.timeline.tracks for c in t.clips]
            (clip,) = placed
            assert window._timeline.selected_clips == (clip.id,)
        finally:
            window.close()


# --- プレビューの外枠 ---


def _movie(name: str) -> MediaItem:
    """160 × 90 の音付きの動画"""
    return MediaItem(
        path=Path(f"C:/素材/{name}"),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=160,
                height=90,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 30),
                codec="h264",
            ),
        ),
        audio_streams=(
            AudioStreamInfo(
                index=1,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="aac",
            ),
        ),
    )


class TestPreviewOutlineOnALayer:
    def test_a_video_with_sound_on_a_layer_moves_in_the_preview(
        self, qt_application: QApplication
    ) -> None:
        # #162 の外枠 レイヤーの音付きの動画も、絵を描くので掴んで動かせる
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent

        from sashimono.ui.preview import PreviewWidget

        del qt_application
        media = _movie("本編.mp4")
        clip = replace(_with_sound(media, start=0, duration=60), native_size=True)
        settings = ProjectSettings(width=320, height=180, layer_mode=LayerMode.MIXED)
        base = Project.create(settings, media=(media,))
        project = base.with_timeline(replace(base.timeline, tracks=(_layer(1, clip),)))
        widget = PreviewWidget(project, prefetch_bytes=0, prefetch_thread=False)
        try:
            widget.resize(320, 180)
            widget.set_selection(clip.id)
            committed: list[list[Command]] = []
            widget.commands_requested.connect(lambda commands, _: committed.append(commands))
            for kind, point, buttons in (
                (QEvent.Type.MouseButtonPress, (160, 90), _LEFT),
                (QEvent.Type.MouseMove, (175, 85), _LEFT),
                (QEvent.Type.MouseMove, (190, 70), _LEFT),
                (QEvent.Type.MouseButtonRelease, (190, 70), Qt.MouseButton.NoButton),
            ):
                button = Qt.MouseButton.NoButton if kind is QEvent.Type.MouseMove else _LEFT
                event = QMouseEvent(kind, QPointF(*point), QPointF(*point), button, buttons, _NONE)
                QApplication.sendEvent(widget, event)
            assert len(committed) == 1, "レイヤーのクリップの外枠を掴めない"
        finally:
            widget.shutdown()
            widget.deleteLater()
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
