"""エクスプローラーの素材を測る・見本を作る・下見で再生する（#521 #522）

測り方そのもの（BS.1770）は tests/engine/test_loudness.py など 画面は tests/ui/test_explorer.py
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from sashimono.engine.audio.library_sound import (
    PreviewPlayer,
    measure_library_file,
    sound_peaks,
    still_of,
)
from tests.check_media import tone, write_tone, write_video
from tests.waiting import wait_until


class _Sink:
    """鳴らさずに受け取る出口 機械の出口を待たないので、時計より速く読み進む"""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.samples = 0
        self.closed = threading.Event()
        self.drained: bool | None = None
        #: 渡すと、最初の 1 回を書いた後はこれが立つまで次を書かせない（止める間合いを決める）
        self._gate = gate

    def write(self, samples: np.ndarray) -> None:
        self.samples += len(samples)
        if self._gate is not None:
            self._gate.wait()

    def close(self, drain: bool) -> None:
        self.drained = drain
        self.closed.set()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_measures_length_and_loudness(tmp_path: Path) -> None:
    # 測れないと、AI は名前だけで選び、効果音の所に長い曲を置くか、小さすぎる音を選ぶ
    loud = write_tone(tmp_path / "大.wav", tone(2.0, -6.0))
    quiet = write_tone(tmp_path / "小.wav", tone(2.0, -26.0))
    big = measure_library_file(loud)
    small = measure_library_file(quiet)
    assert big is not None and small is not None
    assert big.seconds == pytest.approx(2.0, abs=0.05)
    assert big.has_audio and not big.has_video
    assert big.loudness is not None and small.loudness is not None
    # 20 dB 下げた音は 20 LU ほど小さく測れる
    assert big.loudness - small.loudness == pytest.approx(20.0, abs=0.5)
    assert big.peak == pytest.approx(-6.0, abs=0.2)


def test_measures_the_picture_of_a_video(tmp_path: Path) -> None:
    # 壊れると、エクスプローラーの解像度とフレームレートの列が空のまま並べ替えられない
    movie = write_video(tmp_path / "動画.mp4", 1.0, tone_db=-20.0)
    measure = measure_library_file(movie)
    assert measure is not None and measure.has_video
    assert measure.width > 0 and measure.height > 0 and measure.fps > 0


def test_a_broken_file_is_not_measured(tmp_path: Path) -> None:
    # 壊れると、壊れたファイルが 1 つあるだけで探す道具と一覧が落ちる
    broken = tmp_path / "壊れた.wav"
    broken.write_bytes(b"not a sound")
    assert measure_library_file(broken) is None
    assert sound_peaks(broken, 10) is None
    assert still_of(broken) is None


def test_peaks_follow_the_loudness(tmp_path: Path) -> None:
    # 壊れると、下見とアイコンの波形が音の形と合わず、どこが大きいかが見えない
    samples = np.concatenate([tone(1.0, -40.0), tone(1.0, -6.0)])
    path = write_tone(tmp_path / "後ろが大きい.wav", samples)
    peaks = sound_peaks(path, 20)
    assert peaks is not None and len(peaks) == 20
    assert peaks[:9].max() < 0.05 < 0.3 < peaks[11:].min()


class TestPreviewPlayer:
    def test_plays_pauses_and_seeks_without_the_device(self, tmp_path: Path) -> None:
        # 壊れると、下見の〔再生〕で音が出ないか、止めても鳴り続け、シークが効かない
        path = write_tone(tmp_path / "音.wav", tone(3.0, -12.0))
        sinks: list[_Sink] = []
        clock = _Clock()

        def make(rate: int, channels: int) -> _Sink:
            assert (rate, channels) == (48000, 2)
            sinks.append(_Sink())
            return sinks[-1]

        player = PreviewPlayer(sink=make, clock=clock)
        info = player.open(path)
        assert info is not None and info.has_audio and not info.has_video
        player.set_volume(0.5)
        player.play()
        assert player.playing
        wait_until(lambda: bool(sinks) and sinks[0].samples > 0, "下見の音")
        clock.now = 1.0
        assert player.position == pytest.approx(1.0)
        player.pause()
        assert not player.playing
        wait_until(sinks[0].closed.is_set, "止めた後の出口の片付け")
        assert player.position == pytest.approx(1.0)
        player.seek(2.5)
        assert player.position == pytest.approx(2.5)
        # 終わりまで来たら止まった形になり、次の〔再生〕は頭から
        player.play()
        # 出口を開き終えて時計が進み始めてから時計を進める（開く間は時計を進めない作り）
        wait_until(lambda: len(sinks) == 2 and sinks[1].samples > 0, "2 回目の下見の音")
        clock.now = 10.0
        assert not player.playing
        assert player.position == pytest.approx(3.0, abs=0.01)
        player.play()
        assert player.position == pytest.approx(0.0)
        player.close()

    def test_a_video_shows_its_first_picture_and_moves(self, tmp_path: Path) -> None:
        # 壊れると、動画を選んでも下見が黒いままか、再生しても絵が動かない
        path = write_video(tmp_path / "動画.mp4", 2.0, tone_db=-20.0)
        clock = _Clock()
        sinks: list[_Sink] = []

        def make(_rate: int, _channels: int) -> _Sink:
            sinks.append(_Sink())
            return sinks[-1]

        player = PreviewPlayer(sink=make, clock=clock)
        info = player.open(path)
        assert info is not None and info.has_video
        wait_until(lambda: player.frame() is not None, "頭の絵")
        first = player.frame()
        assert first is not None
        player.play()
        # 音の出口を開き終えてから時計を進める 開くまでの時間は再生に数えない作りなので、
        # 先に進めると開いたときに頭へ取り直される
        wait_until(lambda: bool(sinks) and sinks[0].samples > 0, "下見の音")
        clock.now = 1.0
        # 番号でなく絵そのものが変わったかを見る 同じ頭の絵を出し直すだけでも番号は進む
        wait_until(
            lambda: (f := player.frame()) is not None and not np.array_equal(f[1], first[1]),
            "動いた絵",
        )
        player.close()
        assert player.frame() is None

    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        # 壊れると、消えたファイルを選んだだけで下見が落ちるか、黙って何も出ない
        player = PreviewPlayer(sink=lambda _rate, _channels: _Sink())
        assert player.open(tmp_path / "無い.wav") is None
        assert player.error is not None and "開けません" in player.error
        player.play()
        assert not player.playing

    def test_seeking_near_the_end_keeps_the_place(self, tmp_path: Path) -> None:
        # 壊れると、再生中に終わりの近くへ飛ぶと頭へ戻り、飛んだ所が聞けない
        path = write_tone(tmp_path / "音.wav", tone(3.0, -12.0))
        clock = _Clock()
        player = PreviewPlayer(sink=lambda _rate, _channels: _Sink(), clock=clock)
        player.open(path)
        player.play()
        player.seek(2.98)
        assert player.position == pytest.approx(2.98)
        # 止めている間に終わりの近くへ飛んでから〔再生〕を押しても頭へ戻さない
        player.pause()
        player.seek(2.98)
        player.play()
        assert player.position == pytest.approx(2.98)
        player.close()

    def test_a_short_sound_waits_for_the_output(self, tmp_path: Path) -> None:
        # 壊れると、出口を開くのが音の長さより遅い短い効果音は 1 つも書かないまま終わる
        path = write_tone(tmp_path / "短い.wav", tone(0.05, -12.0))
        clock = _Clock()
        opened = threading.Event()
        sinks: list[_Sink] = []

        def slow(_rate: int, _channels: int) -> _Sink:
            opened.wait()
            sinks.append(_Sink())
            return sinks[-1]

        player = PreviewPlayer(sink=slow, clock=clock)
        player.open(path)
        player.play()
        # 開いている間に時計が音の長さを越えても、まだ鳴っている側（開き終えてから数える）
        clock.now = 1.0
        assert player.playing
        assert player.position == pytest.approx(0.0)
        opened.set()
        wait_until(lambda: bool(sinks) and sinks[0].closed.is_set(), "短い音の書き終わり")
        assert sinks[0].samples > 0

    def test_a_sound_that_cannot_play_stops(self, tmp_path: Path) -> None:
        # 壊れると、出口を開けない機械で時計だけが進み、〔一時停止〕のまま押し直しても試せない
        path = write_tone(tmp_path / "音.wav", tone(3.0, -12.0))

        def broken(_rate: int, _channels: int) -> _Sink:
            raise RuntimeError("出口がありません")

        player = PreviewPlayer(sink=broken)
        player.open(path)
        player.play()
        wait_until(lambda: player.error is not None, "鳴らせない知らせ")
        assert not player.playing
        assert "鳴らせません" in (player.error or "")

    def test_stopping_throws_away_what_was_written(self, tmp_path: Path) -> None:
        # 壊れると、止めた後も出口に書いた分が鳴り続け、古い音が聞こえる
        path = write_tone(tmp_path / "音.wav", tone(3.0, -12.0))
        sinks: list[_Sink] = []
        gate = threading.Event()

        def make(_rate: int, _channels: int) -> _Sink:
            sinks.append(_Sink(gate if not sinks else None))
            return sinks[-1]

        clock = _Clock()
        player = PreviewPlayer(sink=make, clock=clock)
        player.open(path)
        player.play()
        wait_until(lambda: bool(sinks) and sinks[0].samples > 0, "下見の音")
        player.pause()
        gate.set()
        wait_until(sinks[0].closed.is_set, "止めた後の片付け")
        assert sinks[0].drained is False
        # 終わりまで鳴らしたときだけ鳴らし切る
        player.seek(0.0)
        player.play()
        wait_until(lambda: len(sinks) == 2 and sinks[1].closed.is_set(), "鳴らし切り")
        assert sinks[1].drained is True
