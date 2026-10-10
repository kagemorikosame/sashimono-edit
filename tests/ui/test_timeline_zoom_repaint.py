"""細い帯との境を越えるズームで、1 回の描画が 50ms かかった（#260）

短いクリップ 2000 本・8 トラック・1920 幅で、細い帯（名前を描かない幅）との境を越えて
拡大すると、名前・サムネイル・波形・値の線まで描くクリップが 1 回で 400 本を超える
測ると、波形の画像を作り直す所は初めての倍率のときの一部で、残りはクリップごとに
同じ物を毎回作り直していた所だった（名前の字を並べる・サムネイルの配列を画像にする・
波形の範囲と効き方の鍵を求める・素材の一覧をなめる）

速さは CI の機械で揺れるので、時間ではなく「作り直した回数」「素材の一覧をなめた回数」で
見る 描く絵が変わらないことは、前の描き方で描いた物と画素まで比べて押さえる
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Clip, MediaId, MediaItem, Project, Track, TrackKind
from sashimono.core.model.ids import new_media_id
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio import PeakLevel, Waveform
from sashimono.engine.cache import MediaAnalyzer
from sashimono.engine.cache.thumbnails import Filmstrip
from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, Colors, Metrics, use_palette
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline import painter as painter_module
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.painter import (
    DETAIL_MIN_WIDTH,
    _draw_clip_label,
    _draw_filmstrip,
    _draw_waveform,
    clear_waveform_images,
    to_qimage,
)

RATE = FrameRate(30)
#: 1 本の長さ（フレーム） 30 フレームを境の手前と先の倍率で描く
LENGTH = 30
#: 境の手前（細い帯）・境の先（名前まで描く）・さらに先 の 1 フレームの画素数
SCALES = (0.5, 1.0, 1.25, 1.5625)


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    # 前の試験が貯めた物を使うと、作り直した回数を 0 と数えてしまう
    _forget()
    yield
    use_palette(THEME_DARK)
    _forget()


def _forget() -> None:
    clear_waveform_images()
    painter_module._LABEL_TEXTS.clear()
    painter_module._FILMSTRIP_TILES.clear()


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


def _filmstrip(count: int = 40) -> Filmstrip:
    # 1 枚ずつ色の違うサムネイル 同じ色だと、違う番号の絵を描いても画素で見分けられない
    tile = 16
    sheet = np.zeros((72, tile * count, 4), dtype=np.uint8)
    for index in range(count):
        part = sheet[:, index * tile : (index + 1) * tile]
        part[:, :, 0] = (index * 37) % 256
        part[:, :, 1] = (index * 91) % 256
        part[:, :, 2] = np.arange(72, dtype=np.uint8)[:, np.newaxis] * 3
        part[:, :, 3] = 255
    return Filmstrip(sheet=sheet, tile_width=tile, interval=Fraction(1, 2))


def _waveform(seconds: int = 30) -> Waveform:
    # 大きさの揺れる波形 平らだと、形が変わったかを画素で見られない
    count = 48000 * seconds // 256
    swing = np.abs(np.sin(np.arange(count, dtype=np.float32) / 23.0)).astype(np.float32)
    peaks = np.empty((count, 2, 2), dtype=np.float32)
    peaks[:, :, 0] = -swing[:, np.newaxis] * 0.8
    peaks[:, :, 1] = swing[:, np.newaxis]
    return Waveform(
        sample_rate=48000,
        channels=2,
        total_samples=48000 * seconds,
        levels=(PeakLevel(256, peaks),),
    )


class _Scene:
    """映像と音声のトラックに、素材の頭を少しずつずらした短いクリップを並べる"""

    def __init__(self, video: MediaItem, audio: MediaItem, clips: int = 24) -> None:
        # 使わない素材も混ぜる 素材の一覧をなめる所は、数が多いほど重くなる
        spare = tuple(
            replace(audio, id=new_media_id(), path=Path(f"C:/素材/{n}.wav")) for n in range(20)
        )
        base = Project.create(media=(*spare, video, audio))
        self.names = {video.name, audio.name}

        def row(kind: TrackKind, media: MediaItem) -> Track:
            return Track(
                kind,
                "V1" if kind is TrackKind.VIDEO else "A1",
                tuple(
                    Clip(
                        timeline_start=n * LENGTH,
                        duration=LENGTH,
                        media_id=media.id,
                        stream_index=media.video_streams[0].index
                        if kind is TrackKind.VIDEO
                        else media.audio_streams[0].index,
                        source_in=Fraction(n % 7, 10),
                    )
                    for n in range(clips)
                ),
            )

        tracks = (row(TrackKind.VIDEO, video), row(TrackKind.AUDIO, audio))
        self.project = base.with_timeline(replace(base.timeline, tracks=tracks))
        self.analyzer = _Analyzer({video.id: _filmstrip()}, {audio.id: _waveform()})
        self.view = TimelineView(self.project, self.analyzer)
        self.view.resize(900, 260)

    def render(self, scale: float) -> QImage:
        self.view._layout = TimelineLayout(pixels_per_frame=scale)
        image = QImage(self.view.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
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


def _count(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> list[int]:
    original: Callable[..., Any] = getattr(owner, name)
    calls = [0]

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counting)
    return calls


def test_the_scene_crosses_the_thin_band_limit() -> None:
    # 前提が崩れると、下の試験は名前を描かないクリップだけを数えて通ってしまう
    assert LENGTH * SCALES[0] < DETAIL_MIN_WIDTH <= LENGTH * SCALES[1]


class TestZoomingDoesNotRedoTheSameWork:
    def test_names_are_laid_out_once_per_name(self, scene: _Scene) -> None:
        # drawText は呼ぶたびに字を並べ直す 400 本の名前で 17ms かかり、境を越えるズームの
        # 1 回の描画が予算を超えた 同じ名前は倍率が変わっても並べ直さない
        # 数は増えた分で見る ほかの試験が先に描いた分も数に入っている
        before = painter_module._LABEL_TEXTS.prepared
        for scale in SCALES * 2:
            scene.render(scale)
        assert painter_module._LABEL_TEXTS.prepared - before == len(scene.names)

    def test_thumbnails_become_images_once(self, scene: _Scene) -> None:
        # 描くたびに配列から画像を作ると、写しを 2 度取る 倍率を変えても同じ絵は作り直さない
        scene.render(SCALES[-1])
        first = painter_module._FILMSTRIP_TILES.converted
        assert first > 0
        for scale in SCALES * 2:
            scene.render(scale)
        scene.render(SCALES[-1])
        assert painter_module._FILMSTRIP_TILES.converted == first

    def test_the_wave_range_and_its_shaping_are_worked_out_once_per_clip(
        self, scene: _Scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 範囲は分数で数え、効き方の鍵はエフェクトを全部なめる 200 本の波形で毎回求めると
        # 1ms を超えた クリップが同じ物なら、倍率を変えても求め直さない
        calls = _count(monkeypatch, painter_module, "shape_key")
        for scale in SCALES[1:] * 3:
            scene.render(scale)
        sounds = len(scene.project.timeline.tracks[1].clips)
        assert 0 < calls[0] <= sounds

    def test_painting_does_not_search_the_media_list_per_clip(
        self, scene: _Scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # クリップごとに素材の一覧をなめると、素材の多い作品ほど 1 回の描画が重くなる
        # （描く所の引き表と、値の線の種類を決める所の 2 か所でなめていた）
        calls = _count(monkeypatch, Project, "find_media")
        scene.render(SCALES[-1])
        assert calls[0] == 0

    def test_revisiting_a_zoom_does_not_rebuild_any_waveform(
        self, scene: _Scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 行き来するズームで同じ倍率へ戻ったら、波形の画像は 1 枚も作り直さない
        built = _count(monkeypatch, painter_module, "waveform_image")
        for scale in SCALES:
            scene.render(scale)
        first = built[0]
        assert first > 0
        for scale in reversed(SCALES):
            scene.render(scale)
        assert built[0] == first


def _old_label(painter: QPainter, rect: QRect, name: str, *, editing: bool) -> None:
    """直す前の名前の帯（drawText で毎回並べる） 新しい描き方と画素まで比べる相手"""
    label_rect = QRect(rect.left(), rect.top(), rect.width(), Metrics.CLIP_LABEL_HEIGHT)
    shade = QColor(Colors.EDITING) if editing else QColor(Colors.CLIP_LABEL_SHADE)
    if editing:
        shade.setAlpha(170)
    painter.fillRect(label_rect, shade)
    painter.setPen(QPen(Colors.CLIP_LABEL, 1))
    font = QFont(painter.font())
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.drawText(
        label_rect.adjusted(4, 0, -4, 0),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        name,
    )


_BACKGROUND = QColor(40, 60, 90)


def _canvas(draw: Callable[[QPainter], None], size: tuple[int, int] = (260, 60)) -> QImage:
    image = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(_BACKGROUND)
    painter = QPainter(image)
    draw(painter)
    painter.end()
    return image


class TestTheLookStaysTheSame:
    @pytest.mark.parametrize("theme", [THEME_DARK, THEME_LIGHT])
    @pytest.mark.parametrize("editing", [False, True])
    @pytest.mark.parametrize("width", [3, 9, 24, 41, 230])
    @pytest.mark.parametrize(
        "name",
        ["bench-video.mp4", "本編の素材 とても長い名前のクリップ.mp4", "<b>太字</b>.wav", "g"],
    )
    def test_a_name_is_drawn_like_draw_text(
        self,
        qt_application: QApplication,
        audio_media: MediaItem,
        theme: str,
        editing: bool,
        width: int,
        name: str,
    ) -> None:
        # 並べた字を使い回すと、置く位置や切り落とす所が 1 画素でもずれれば名前が揺れて見える
        # 「<」で始まる名前は書式を決め打ちしないと HTML として読まれ、字が消える
        del qt_application
        use_palette(theme)
        media = replace(audio_media, path=Path(f"C:/素材/{name}"))
        clip = Clip(timeline_start=0, duration=LENGTH, media_id=media.id, stream_index=0)
        rect = QRect(7, 5, width, 50)

        def new(painter: QPainter) -> None:
            painter.save()
            painter.setClipRect(rect)
            _draw_clip_label(painter, rect, clip, media, editing=editing)
            painter.restore()

        def old(painter: QPainter) -> None:
            painter.save()
            painter.setClipRect(rect)
            _old_label(painter, rect, media.name, editing=editing)
            painter.restore()

        assert _canvas(new) == _canvas(old)

    def test_a_thumbnail_is_drawn_like_the_array_it_came_from(
        self, qt_application: QApplication
    ) -> None:
        # 貯めた画像と、描くたびに配列から作った画像が違えば、拡大の前後で絵が入れ替わる
        del qt_application
        strip = _filmstrip()
        clip = Clip(timeline_start=0, duration=600, media_id=None, source_in=Fraction(3, 2))
        layout = TimelineLayout(pixels_per_frame=0.7)
        rect = QRect(Metrics.TRACK_HEADER_WIDTH, 4, 400, 37)

        def old(painter: QPainter) -> None:
            scale = rect.height() / strip.height
            tile_width = max(1, int(strip.tile_width * scale))
            x = rect.left()
            while x < rect.right():
                frame = layout.frame_at(x)
                tile = strip.at(clip.picture_time(frame - clip.timeline_start, RATE))
                if tile is None:
                    break
                painter.drawImage(QRectF(x, rect.top(), tile_width, rect.height()), to_qimage(tile))
                x += tile_width

        size = (rect.right() + 10, 50)
        drawn = [
            _canvas(lambda p: _draw_filmstrip(p, rect, clip, layout, RATE, strip), size)
            for _ in range(2)
        ]
        assert drawn[0] == _canvas(old, size)
        # 2 回目は貯めた画像を貼る それでも同じ絵
        assert drawn[1] == drawn[0]

    def test_a_turned_down_track_redraws_the_same_clip_smaller(
        self, qt_application: QApplication
    ) -> None:
        # 範囲と効き方の鍵をクリップごとに貯めると、トラックの音量だけを変えたときに前の
        # 大きさの波形が残りかねない 鍵にはトラックの音量も入れる
        del qt_application
        wave = _waveform()
        clip = Clip(timeline_start=0, duration=90, media_id=None)
        layout = TimelineLayout(pixels_per_frame=2.0)
        rect = QRect(Metrics.TRACK_HEADER_WIDTH, 0, 180, 40)
        size = (rect.right() + 10, 50)

        def lit(image: QImage) -> int:
            ground = _BACKGROUND.rgba()
            return sum(
                image.pixel(x, y) != ground
                for x in range(rect.left(), rect.right())
                for y in range(rect.height())
            )

        loud = _canvas(
            lambda p: _draw_waveform(p, rect, clip, layout, RATE, wave, track_gain=1.0), size
        )
        quiet = _canvas(
            lambda p: _draw_waveform(p, rect, clip, layout, RATE, wave, track_gain=0.25), size
        )
        again = _canvas(
            lambda p: _draw_waveform(p, rect, clip, layout, RATE, wave, track_gain=1.0), size
        )
        assert lit(quiet) < lit(loud) / 2
        assert again == loud
