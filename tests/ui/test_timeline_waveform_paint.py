"""タイムラインのクリップに波形を描く所の速さ（#201 #205）

実素材を 100 本並べて測ると、1 回の描画が 50ms 近くかかり、60fps の予算（16.7ms）を
大きく超えた 重いのは波形で、見えている列ごとに Python で値を切り詰めて drawLine を
呼んでいた（#201） まとめて drawLines に渡すようにしても、列ごとに QLineF を作る所が
8ms ほど残った（#205） 今は線を numpy で画像に塗り、同じ倍率の間は作った画像を貯めて貼る
描く絵は線で描いていたときと同じまま、作り直す回数が減ったことを押さえる
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator

import numpy as np
import pytest
from PySide6.QtCore import QLineF, QRect
from PySide6.QtGui import QImage, QPainter, QPen

from sashimono.core.model import Clip, Effect, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.engine.audio import PeakLevel, Waveform
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline.layout import TimelineLayout, TrackBand
from sashimono.ui.timeline.painter import (
    WAVEFORM_IMAGE_MAX_COLUMNS,
    _draw_waveform,
    _WaveformImages,
    clear_waveform_images,
    clip_rect_for,
    waveform_image,
)

RATE = FrameRate(30)


@pytest.fixture(autouse=True)
def _fresh_images() -> Iterator[None]:
    # 前の試験が貯めた画像を使うと、作り直す回数を数える試験が 0 回と数える
    clear_waveform_images()
    yield
    clear_waveform_images()


def _waveform(low: float, high: float, seconds: int = 4) -> Waveform:
    count = 48000 * seconds // 256
    peaks = np.empty((count, 2, 2), dtype=np.float32)
    peaks[:, :, 0] = low
    peaks[:, :, 1] = high
    return Waveform(
        sample_rate=48000,
        channels=2,
        total_samples=48000 * seconds,
        levels=(PeakLevel(256, peaks),),
    )


def _pixels(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    view = np.frombuffer(image.constBits(), dtype=np.uint32, count=image.width() * image.height())
    return view.reshape(image.height(), image.width()).copy()


def _paint(
    waveform: Waveform, rect: QRect, layout: TimelineLayout, clip: Clip | None = None
) -> np.ndarray:
    """``rect`` へ描いた画面を返す"""
    canvas = QImage(
        rect.right() + 10, rect.bottom() + 10, QImage.Format.Format_ARGB32_Premultiplied
    )
    canvas.fill(0)
    painter = QPainter(canvas)
    clip = clip or Clip(timeline_start=0, duration=90, source=TEXT.create())
    _draw_waveform(painter, rect, clip, layout, RATE, waveform)
    painter.end()
    return _pixels(canvas)


def _count_envelopes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """``Waveform.envelope`` が呼ばれるたびに列の数を書き留める"""
    calls: list[int] = []
    original = Waveform.envelope

    def counting(self: Waveform, start: int, end: int, columns: int) -> np.ndarray:
        calls.append(columns)
        return original(self, start, end, columns)

    monkeypatch.setattr(Waveform, "envelope", counting)
    return calls


def test_the_image_paints_the_same_pixels_as_the_lines_it_replaced() -> None:
    """線を引いていたときと同じ画素を塗る 1 画素ずれると、切り替えた前後で波形が太って見える"""
    rng = np.random.default_rng(1)
    for height in (5, 20, 41, 42, 77):
        maximum = rng.uniform(-1.5, 1.5, 300).astype(np.float32)
        minimum = np.minimum(maximum, rng.uniform(-1.5, 1.5, 300).astype(np.float32))
        maximum[:20] = minimum[:20] = 0.0
        expected = QImage(300, height, QImage.Format.Format_ARGB32_Premultiplied)
        expected.fill(0)
        painter = QPainter(expected)
        centre, half = height / 2.0, height / 2.0 - 1.0
        tops = centre - np.clip(maximum, -1.0, 1.0).astype(np.float64) * half
        bottoms = np.maximum(
            centre - np.clip(minimum, -1.0, 1.0).astype(np.float64) * half, tops + 1.0
        )
        painter.setPen(QPen(Colors.WAVEFORM, 1))
        painter.drawLines(
            [
                QLineF(column, top, column, bottom)
                for column, (top, bottom) in enumerate(
                    zip(tops.tolist(), bottoms.tolist(), strict=True)
                )
            ]
        )
        painter.end()
        drawn = _pixels(waveform_image(minimum, maximum, height))
        assert np.array_equal(drawn, _pixels(expected)), height


def test_silence_still_leaves_a_thin_line() -> None:
    """無音でも細い線を残す 何も描かないと、音のクリップなのか見分けられない

    長さ 1 の線は、線で描いていたときも端の 2 行を塗っていた
    """
    silent = np.zeros(400, dtype=np.float32)
    painted = (_pixels(waveform_image(silent, silent, 42)) != 0).sum(axis=0)
    assert ((painted >= 1) & (painted <= 2)).all()


def test_repainting_at_the_same_zoom_does_not_rebuild_the_waveform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """再生ヘッドが動くたびの描き直しとスクロールで、束ねる所から作り直さない

    作り直すと、実素材を 100 本並べた全体表示で波形だけで 5ms を超え、ほかと合わせて
    60fps の予算を超えた（#205）
    """
    calls = _count_envelopes(monkeypatch)
    waveform = _waveform(-0.5, 0.5)
    rect = QRect(Metrics.TRACK_HEADER_WIDTH, 10, 400, 42)
    first = _paint(waveform, rect, TimelineLayout(pixels_per_frame=10.0))
    again = _paint(waveform, rect, TimelineLayout(pixels_per_frame=10.0))
    _paint(waveform, rect, TimelineLayout(pixels_per_frame=10.0, scroll_frame=12.0))
    assert len(calls) == 1
    assert np.array_equal(first, again)
    _paint(waveform, rect, TimelineLayout(pixels_per_frame=20.0))
    assert len(calls) == 2


def test_a_clip_scrolled_past_the_left_edge_shows_its_later_part() -> None:
    """頭が画面の左へ出たクリップは、見えている所の波形を出す

    クリップ全体の画像を作って貼るので、貼る所を間違えると頭の波形が左端に出る
    """
    peaks = np.zeros((48000 * 3 // 256, 2, 2), dtype=np.float32)
    half = peaks.shape[0] // 2
    # 前半は無音、後半は大きな音
    peaks[half:, :, 0] = -0.9
    peaks[half:, :, 1] = 0.9
    waveform = Waveform(48000, 2, 48000 * 3, (PeakLevel(256, peaks),))
    layout = TimelineLayout(pixels_per_frame=10.0, scroll_frame=60.0)
    rect = QRect(Metrics.TRACK_HEADER_WIDTH, 10, 300, 42)
    pixels = _paint(waveform, rect, layout)
    painted = (pixels[rect.top() : rect.bottom() + 1, rect.left() : rect.right() + 1] != 0).sum(
        axis=0
    )
    # 60 フレーム目から先（2 秒目から先）は後半なので、どの列も縦に長い線
    assert (painted > 30).all()


def test_the_end_of_the_sound_is_drawn_when_the_clip_starts_between_pixels() -> None:
    """左端が画素の途中にあっても、音の終わりまで描く（#217 の指摘）

    列の数を 長さ × 倍率 の切り上げにすると、左端の端数の分だけ最後の列が矩形の外へ出て、
    音の終わりのピークが消えた
    """
    peaks = np.zeros((48000 * 3 // 256, 2, 2), dtype=np.float32)
    # 最後の 2% だけ大きな音 前は 1 列にも入らなかった
    tail = int(peaks.shape[0] * 0.98)
    peaks[tail:, :, 0] = -0.9
    peaks[tail:, :, 1] = 0.9
    waveform = Waveform(48000, 2, 48000 * 3, (PeakLevel(256, peaks),))
    layout = TimelineLayout(pixels_per_frame=10.2 / 90)
    clip = Clip(timeline_start=6, duration=90, source=TEXT.create())
    band = TrackBand(Track(TrackKind.AUDIO, "A1", (clip,)), 10, 46)
    rect = clip_rect_for(clip, band, layout, 1920)
    assert rect is not None
    pixels = _paint(waveform, rect, layout, clip)
    last = pixels[rect.top() : rect.bottom() + 1, rect.right()]
    assert (last != 0).sum() > 30


def test_a_clip_wider_than_the_limit_builds_only_the_visible_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """長尺素材を大きく拡大したときは、クリップ全体を画像にしない メモリと時間を食う"""
    calls = _count_envelopes(monkeypatch)
    waveform = _waveform(-0.5, 0.5, seconds=60)
    clip = Clip(timeline_start=0, duration=1800, source=TEXT.create())
    layout = TimelineLayout(pixels_per_frame=WAVEFORM_IMAGE_MAX_COLUMNS / 1000)
    rect = QRect(Metrics.TRACK_HEADER_WIDTH, 10, 500, 42)
    pixels = _paint(waveform, rect, layout, clip)
    assert calls == [500]
    assert (pixels[rect.top() : rect.bottom() + 1, rect.left() : rect.right() + 1] != 0).any()


def test_old_images_are_dropped_past_the_budget() -> None:
    """貯める量には上限がある 無いと、倍率を変えるたびに作った画像が残り続ける"""
    waveform = _waveform(-0.5, 0.5)
    images = _WaveformImages(budget=3 * 100 * 40 * 4)
    for columns in range(100, 110):
        assert images.get(waveform, 0, 48000, columns, 40) is not None
    assert images._used <= 3 * 100 * 40 * 4
    assert len(images._entries) <= 3


def test_an_image_larger_than_the_budget_is_drawn_but_not_kept() -> None:
    """1 枚で上限を超える画像は貯めない 貯めると上限を超えたまま残る（#217 の指摘）"""
    waveform = _waveform(-0.5, 0.5)
    images = _WaveformImages(budget=100 * 40 * 4 - 1)
    assert images.get(waveform, 0, 48000, 100, 40) is not None
    assert images._used <= 100 * 40 * 4 - 1
    assert len(images._entries) == 0


def test_the_cache_does_not_keep_a_discarded_waveform_alive() -> None:
    """使わなくなった素材の解析を捨てたら、貯めた画像が解析を抱えたままにしない"""
    waveform = _waveform(-0.5, 0.5)
    _paint(
        waveform,
        QRect(Metrics.TRACK_HEADER_WIDTH, 10, 400, 42),
        TimelineLayout(pixels_per_frame=10.0),
    )
    alive = weakref.ref(waveform)
    del waveform
    gc.collect()
    assert alive() is None


def _tall(painted: np.ndarray) -> np.ndarray:
    """列ごとに塗った行の数"""
    return np.asarray((painted != 0).sum(axis=0))


_RECT = QRect(Metrics.TRACK_HEADER_WIDTH, 10, 300, 82)
_LAYOUT = TimelineLayout(pixels_per_frame=300 / 90)


def _sound_clip(*effects: Effect) -> Clip:
    return Clip(timeline_start=0, duration=90, effects=effects)


class TestTheSoundShapesTheWave:
    """音量やリバーブを変えたら、タイムラインの波形にも映す（利用者の要望）

    前は素材を解析したピークをそのまま描き、音量を 0 にしても波形が同じ大きさだった
    """

    def test_half_the_volume_draws_half_the_wave(self) -> None:
        waveform = _waveform(-0.8, 0.8)
        full = _tall(_paint(waveform, _RECT, _LAYOUT, _sound_clip()))
        quiet = _sound_clip(registry.require("audio_volume").create(volume=50.0))
        half = _tall(_paint(waveform, _RECT, _LAYOUT, quiet))
        middle = full.shape[0] // 2
        column = _RECT.left() + 150
        assert half[column] == pytest.approx(full[column] / 2, abs=2)
        assert middle > 0

    def test_the_track_volume_also_counts(self) -> None:
        waveform = _waveform(-0.8, 0.8)
        canvas = QImage(_RECT.right() + 10, _RECT.bottom() + 10, QImage.Format.Format_ARGB32)
        canvas.fill(0)
        painter = QPainter(canvas)
        _draw_waveform(painter, _RECT, _sound_clip(), _LAYOUT, RATE, waveform, track_gain=0.25)
        painter.end()
        quarter = _tall(_pixels(canvas))[_RECT.left() + 150]
        full = _tall(_paint(waveform, _RECT, _LAYOUT, _sound_clip()))[_RECT.left() + 150]
        assert quarter == pytest.approx(full / 4, abs=2)

    def test_a_fade_thins_the_start(self) -> None:
        waveform = _waveform(-0.8, 0.8)
        fade = registry.require("audio_fade").create(fade_in=1.0)
        drawn = _tall(_paint(waveform, _RECT, _LAYOUT, _sound_clip(fade)))
        assert drawn[_RECT.left() + 2] < drawn[_RECT.left() + 250] / 4

    def test_a_reverb_leaves_a_tail(self) -> None:
        # 前半だけ鳴る素材 リバーブを掛けると、鳴り終えた後にも尾が残る
        peaks = np.zeros((48000 * 4 // 256, 2, 2), dtype=np.float32)
        peaks[: len(peaks) // 4, :, 0] = -0.8
        peaks[: len(peaks) // 4, :, 1] = 0.8
        waveform = Waveform(
            sample_rate=48000, channels=2, total_samples=48000 * 4, levels=(PeakLevel(256, peaks),)
        )
        dry = _tall(_paint(waveform, _RECT, _LAYOUT, _sound_clip()))
        reverb = registry.require("audio_reverb").create(decay=2.0, mix=50.0)
        wet = _tall(_paint(waveform, _RECT, _LAYOUT, _sound_clip(reverb)))
        after = _RECT.left() + 110
        assert dry[after] <= 2
        assert wet[after] > 6

    def test_changing_the_volume_rebuilds_the_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 鍵に効き方を入れないと、音量を変えても前の大きさの画像が貼られる
        waveform = _waveform(-0.8, 0.8)
        calls = _count_envelopes(monkeypatch)
        _paint(waveform, _RECT, _LAYOUT, _sound_clip())
        _paint(waveform, _RECT, _LAYOUT, _sound_clip())
        assert len(calls) == 1
        quiet = _sound_clip(registry.require("audio_volume").create(volume=40.0))
        _paint(waveform, _RECT, _LAYOUT, quiet)
        assert len(calls) == 2
