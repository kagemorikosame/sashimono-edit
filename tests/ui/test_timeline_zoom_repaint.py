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

from collections.abc import Callable, Hashable, Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

from sashimono.core.commands.fixed import VOLUME_EFFECT_KIND, fixed_effect
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    GroupId,
    Keyframe,
    MediaId,
    MediaItem,
    Project,
    Track,
    TrackKind,
    heard_stream,
)
from sashimono.core.model.ids import new_media_id
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.audio import PeakLevel, Waveform
from sashimono.engine.audio.shape import shape_key
from sashimono.engine.cache import MediaAnalyzer
from sashimono.engine.cache.thumbnails import Filmstrip
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, Colors, Metrics, use_palette
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline import keyframes as keyframes_module
from sashimono.ui.timeline import painter as painter_module
from sashimono.ui.timeline.keyframes import draw_keyframes
from sashimono.ui.timeline.layout import TimelineLayout, TrackBand
from sashimono.ui.timeline.painter import (
    DETAIL_MIN_WIDTH,
    _draw_clip_label,
    _draw_filmstrip,
    _draw_waveform,
    clear_waveform_images,
    draw_clip,
    to_qimage,
)
from sashimono.ui.workspace import Preferences, PreferenceStore

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
        self, mixed: _Mixed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 範囲は分数で数え、効き方の鍵はエフェクトを全部なめる 200 本の波形で毎回求めると
        # 1ms を超えた クリップが同じ物なら、倍率を変えても求め直さない
        # 音量の違うクリップが並ぶので、1 本分を全部に使い回すと 1 回しか求めずに済んでしまう
        # クリップごとにちょうど 1 回ずつ求めたかを見る
        asked: list[ClipId] = []
        # 描く所が引く名前を差し替えるので、元の関数は形を求める所から取る
        original = shape_key

        def recording(clip: Clip, track_gain: float = 1.0) -> Hashable:
            asked.append(clip.id)
            return original(clip, track_gain)

        monkeypatch.setattr(painter_module, "shape_key", recording)
        for scale in SCALES[1:] * 3:
            mixed.render(scale)
        volumes = {
            repr(clip.effects[0].params["volume"])
            for clip in mixed.project.timeline.tracks[1].clips
            if clip.id in asked
        }
        assert len(volumes) > 1
        assert len(asked) == len(set(asked))

    def test_each_clip_keeps_its_own_wave_shape(
        self, mixed: _Mixed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 範囲と効き方をクリップごとに貯める所を取り違えると、音量の違うクリップに別の
        # クリップの大きさの波形が出る 貯めずに毎回求めたときの絵と比べる
        for scale in SCALES[1:]:
            mixed.render(scale)
        drawn = mixed.render(SCALES[-1])

        class Fresh(painter_module._WaveSpans):
            def get(
                self, clip: Clip, rate: FrameRate, sample_rate: int, track_gain: float
            ) -> tuple[int, int, painter_module._Shaping]:
                self.clear()
                return super().get(clip, rate, sample_rate, track_gain)

        monkeypatch.setattr(painter_module, "_WAVE_SPANS", Fresh())
        clear_waveform_images()
        assert mixed.render(SCALES[-1]) == drawn

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
        # 作り直すと、行き来するズームで同じ倍率へ戻るたびに境の先の 1 段で 200 枚を作り直し、
        # そのたびに数 ms 引っかかる 同じ倍率へ戻ったら、波形の画像は 1 枚も作り直さない
        del monkeypatch
        images = painter_module._WAVEFORM_IMAGES
        before = images.built
        for scale in SCALES:
            scene.render(scale)
        first = images.built
        assert first > before
        for scale in reversed(SCALES):
            scene.render(scale)
        assert images.built == first

    def test_new_waves_are_painted_together(
        self, scene: _Scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1 本ずつ束ねて塗ると、初めて見る倍率の 1 段で numpy を本数分呼び、200 本で 6ms
        # ほど掛かった 同じ解析・同じ高さの波形は、1 回の描画で 1 度にまとめて塗る
        painted = _count(monkeypatch, painter_module, "waveform_image")
        bundled = _count(monkeypatch, Waveform, "envelopes")
        single = _count(monkeypatch, Waveform, "envelope")
        before = painter_module._WAVEFORM_IMAGES.built
        scene.render(SCALES[-1])
        made = painter_module._WAVEFORM_IMAGES.built - before
        assert made > 3
        assert painted[0] == 1
        assert bundled[0] == 1
        assert single[0] == 0


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


def _old_paint_detailed(
    view: TimelineView,
    painter: QPainter,
    band: TrackBand,
    clips: list[tuple[Clip, QRect]],
    selected: frozenset[ClipId],
    editing: ClipId | None,
    table: dict[MediaId, MediaItem],
    *,
    stretch: bool = False,
) -> int:
    """直す前の描き方（1 本ずつ draw_clip・ひし形・値の線） まとめて描く物と画素まで比べる相手"""
    del stretch
    project = view.project
    for clip, rect in clips:
        media = table.get(clip.media_id) if clip.media_id is not None else None
        scene = project.find_scene(clip.scene_id) if clip.scene_id is not None else None
        draw_clip(
            painter,
            clip,
            band,
            view._layout,
            project.rate,
            media=media,
            filmstrip=view._analyzer.filmstrip(media) if media is not None else None,
            waveform=view._analyzer.waveform(media, heard_stream(band.track, clip))
            if media is not None
            else None,
            selected=clip.id in selected,
            clip_rect=rect,
            scene_name=scene.name if scene is not None else None,
            editing=clip.id == editing,
        )
        draw_keyframes(painter, clip, view._layout, rect, selected=clip.id in selected)
        view._value_lines.paint(
            painter,
            project,
            band.track,
            clip,
            view._layout,
            rect,
            band.height,
            selected=clip.id in selected,
        )
    return 0


def _volume(percent: float) -> Effect:
    volume = fixed_effect(VOLUME_EFFECT_KIND)
    return replace(volume, params={**volume.params, "volume": AnimatedValue(percent)})


class _Mixed:
    """色々なクリップを並べた 3 本のトラック（映像・音声・レイヤー）

    無効・グループ・速さ・不透明度の違い・キーフレーム・素材の無いクリップ・音量の違い・
    隙間・絵と音の両方を描くレイヤーを混ぜる まとめて描く所が 1 つでも順や色を
    取り違えれば、1 本ずつ描いた物と画素が変わる
    """

    def __init__(self, video: MediaItem, audio: MediaItem) -> None:
        base = Project.create(media=(video, audio))
        fade = AnimatedValue(1.0, (Keyframe(0, 0.2), Keyframe(20, 1.0)))
        pictures = []
        sounds = []
        layers = []
        for n in range(30):
            # 9 本ごとに半分の長さの隙間を空ける（隙間の前後はまとめて描かない）
            start = n * LENGTH + (n // 9) * (LENGTH // 2)
            clip = Clip(
                timeline_start=start,
                duration=LENGTH,
                media_id=video.id,
                stream_index=0,
                source_in=Fraction(n % 5, 10),
                enabled=n % 7 != 3,
                group_id=GroupId(f"{n:06x}") if n % 6 == 1 else None,
                speed=Fraction(2) if n % 8 == 2 else Fraction(1),
                opacity=fade if n % 10 == 4 else AnimatedValue(0.4 + (n % 3) * 0.3),
            )
            if n % 11 == 5:
                clip = Clip(timeline_start=start, duration=LENGTH, source=TEXT.create())
            pictures.append(clip)
            sounds.append(
                Clip(
                    timeline_start=start,
                    duration=LENGTH,
                    media_id=audio.id,
                    stream_index=0,
                    source_in=Fraction(n % 4, 10),
                    effects=(_volume(40.0 + (n % 4) * 37.5),),
                )
            )
            layers.append(
                Clip(
                    timeline_start=start,
                    duration=LENGTH,
                    media_id=video.id,
                    stream_index=0,
                    audio_stream=video.audio_streams[0].index,
                    source_in=Fraction(n % 3, 10),
                )
            )
        tracks = (
            Track(TrackKind.VIDEO, "V1", tuple(pictures)),
            replace(Track(TrackKind.AUDIO, "A1", tuple(sounds)), volume_db=-6.0),
            Track(TrackKind.MIXED, "L1", tuple(layers)),
        )
        self.project = base.with_timeline(replace(base.timeline, tracks=tracks))
        self.analyzer = _Analyzer(
            {video.id: _filmstrip()}, {video.id: _waveform(), audio.id: _waveform()}
        )
        self.view = TimelineView(self.project, self.analyzer)
        self.view.resize(1200, 300)
        chosen = [clip.id for track in tracks for clip in track.clips][::5]
        self.view._selection = tuple(chosen)

    def render(self, scale: float) -> QImage:
        self.view._layout = TimelineLayout(pixels_per_frame=scale, scroll_frame=17.0)
        image = QImage(self.view.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        self.view.render(painter, QPoint())
        painter.end()
        return image


@pytest.fixture
def mixed(
    qt_application: QApplication, video_media: MediaItem, audio_media: MediaItem
) -> Iterator[_Mixed]:
    del qt_application
    made = _Mixed(video_media, audio_media)
    yield made
    made.analyzer.close()


class TestDrawingABandAtOnce:
    @pytest.mark.parametrize("theme", [THEME_DARK, THEME_LIGHT])
    def test_the_band_looks_like_drawing_one_by_one(
        self, mixed: _Mixed, theme: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 段ごとにまとめて描くと、1 本の中の順（地・サムネイル・波形・名前・線・枠・ひし形・
        # 値の線）や色・切り落とす範囲を 1 つでも取り違えれば、画素が変わる
        use_palette(theme)
        together = []
        for scale in SCALES[1:]:
            _forget()
            together.append(mixed.render(scale))
        view = mixed.view

        def old(*args: Any, **kwargs: Any) -> int:
            return _old_paint_detailed(view, *args, **kwargs)

        monkeypatch.setattr(view, "_paint_detailed", old)
        for scale, drawn in zip(SCALES[1:], together, strict=True):
            _forget()
            assert drawn == mixed.render(scale), scale

    def test_flat_lines_of_neighbours_are_drawn_as_one(
        self, scene: _Scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 隣り合う同じ高さの線を 1 本ずつ引くと、400 本で線だけで数 ms 掛かる 平らな線は
        # 続く所をまとめて引く（影と線で 2 回） 並んだクリップの線の高さはどれも同じ
        lines = _count(monkeypatch, QPainter, "drawPolyline")
        scene.render(SCALES[-1])
        tracks = len(scene.project.timeline.tracks)
        assert lines[0] == 2 * tracks


class TestStretchingWavesWhileZooming:
    def test_by_default_a_new_zoom_draws_the_exact_wave_at_once(self, mixed: _Mixed) -> None:
        # 既定は今の見た目 伸ばして仮に描くと、細かい山がずれた波形が一瞬出る
        assert Preferences().stretch_waves is False
        assert mixed.view.stretch_waves is False
        images = painter_module._WAVEFORM_IMAGES
        mixed.render(SCALES[1])
        before = images.built
        mixed.render(SCALES[2])
        assert images.built > before
        assert mixed.view._stretched == 0
        assert not mixed.view._wave_settle.isActive()

    def test_while_zooming_a_nearby_wave_is_stretched_then_rebuilt(
        self, mixed: _Mixed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 入れても作ってしまえば設定の意味が無く、止まった後に描き直さなければ、伸ばした
        # ずれた波形がいつまでも残る ズームの途中は作らず、止まったら正しい波形に描き直す
        exact = mixed.render(SCALES[2])
        _forget()
        mixed.view.set_stretch_waves(True)
        images = painter_module._WAVEFORM_IMAGES
        mixed.render(SCALES[1])
        before = images.built
        rough = mixed.render(SCALES[2])
        # 前の倍率で見えていなかったクリップ（近い倍率の絵が無い）だけは作る
        assert mixed.view._stretched > 0
        assert images.built - before < mixed.view._stretched
        assert mixed.view._wave_settle.isActive()
        assert rough != exact
        # 止まった（倍率が変わらないまま時間が過ぎた）ら描き直しを頼み、作り直す
        updates = _count(monkeypatch, mixed.view, "update")
        mixed.view._wave_settle.stop()
        mixed.view._settle_waves()
        assert updates[0] == 1
        settled = mixed.render(SCALES[2])
        assert images.built > before
        assert mixed.view._stretched == 0
        assert settled == exact

    def test_turning_it_off_stops_stretching(self, mixed: _Mixed) -> None:
        # 切ったのに伸ばしたままだと、設定がある方が質が悪い
        mixed.view.set_stretch_waves(True)
        mixed.render(SCALES[1])
        mixed.view.set_stretch_waves(False)
        before = painter_module._WAVEFORM_IMAGES.built
        mixed.render(SCALES[2])
        assert mixed.view._stretched == 0
        assert painter_module._WAVEFORM_IMAGES.built > before

    def test_the_setting_is_saved_and_shown(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 設定の窓に出ていないと、好みが分かれるのに変える手段が無い 壊れた値は既定へ戻す
        del qt_application
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(replace(Preferences(), stretch_waves=True))
        assert store.load().stretch_waves is True
        (tmp_path / "preferences.json").write_text('{"stretch_waves": "yes"}', encoding="utf-8")
        assert store.load().stretch_waves is False
        dialog = PreferencesDialog(replace(Preferences(), stretch_waves=True))
        try:
            assert dialog.preferences().stretch_waves is True
        finally:
            dialog.deleteLater()


def _held_clips(view: TimelineView) -> set[int]:
    """クリップごとの描き方の控えが持っているクリップ（id）"""
    held: set[int] = set()
    held.update(id(entry[0]) for entry in painter_module._CLIP_LOOKS._entries.values())
    held.update(id(entry[0]) for entry in painter_module._TILE_STEPS._entries.values())
    for entry in painter_module._WAVE_SPANS._entries.values():
        held.update((id(entry[0]), id(entry[6].clip)))
    held.update(id(entry[0]) for entry in keyframes_module._FRAMES.values())
    held.update(id(entry[0]) for entry in view._value_lines._shapes.values())
    held.update(id(entry[0]) for entry in view._glance_cache.values())
    return held


class TestSwitchingProjects:
    def test_the_old_project_is_let_go(
        self, mixed: _Mixed, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 控えはクリップを強く持つ 別のプロジェクトを開いても捨てないと、新しいクリップで
        # 上限（8192 本）まで埋まるまで、前のプロジェクトがまるごとメモリに残る
        for scale in (SCALES[0], SCALES[-1]):
            mixed.render(scale)
        old = {id(clip) for track in mixed.project.timeline.tracks for clip in track.clips}
        assert old & _held_clips(mixed.view)
        other = _Mixed(replace(video_media, id=new_media_id()), audio_media)
        try:
            mixed.view.forget_drawing()
            mixed.view.set_project(other.project)
            for scale in (SCALES[0], SCALES[-1]):
                mixed.render(scale)
            assert _held_clips(mixed.view)
            assert not old & _held_clips(mixed.view)
        finally:
            other.analyzer.close()

    def test_the_window_forgets_the_drawing_when_it_swaps_the_project(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 新規・開く・復元の差し替え（_leave_project）で捨てないと、上と同じく前の
        # プロジェクトが残る
        del qt_application
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            calls = _count(monkeypatch, window._timeline, "forget_drawing")
            window._leave_project(window.document.project)
            assert calls[0] == 1
        finally:
            window.close()
