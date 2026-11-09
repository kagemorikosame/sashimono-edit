"""再生中の映像が 1 コマずつ進むこと（#463）

音の時計はかたまり（1024 サンプル 48kHz で約 21 ms）を書くたびにしか進まず、
60fps の素材では 1 コマ飛ばしが規則的に起きた かたまりを書いた時刻からの経過で
位置を補う その補いが、口と声のずれを増やさず、止めたときや位置を変えたときに
戻りすぎないことも見る

実際のデバイスは使わない 書き込みの時刻と時計を試験から決め、毎回同じ並びで確かめる
"""

from __future__ import annotations

import itertools
import random
import threading
from dataclasses import replace

import numpy as np
import pytest
import sounddevice

from sashimono.core.model import Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.audio import player as player_module
from sashimono.engine.audio.player import BLOCK_SAMPLES, EXTRAPOLATE_LIMIT, AudioPlayer
from sashimono.ui import playback as playback_module
from sashimono.ui.playback import POLL_INTERVAL_MS, FramePhase, PlaybackController

RATE = 48000
#: デバイスのバッファに残る量 MME の low で 1 かたまりの数倍
BUFFERED = 4 * BLOCK_SAMPLES
BLOCK_SECONDS = BLOCK_SAMPLES / RATE


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _SilentMixer:
    sample_rate = RATE
    channels = 2

    def render(self, start: int, count: int) -> np.ndarray:
        del start
        return np.zeros((count, 2), dtype=np.float32)


class _SteppingPlayer(AudioPlayer):
    """書き込みを試験から進める出口 デバイスもスレッドも使わない"""

    clock = _Clock()

    def __init__(self, mixer: object) -> None:
        del mixer
        super().__init__(_SilentMixer(), clock=_SteppingPlayer.clock)  # type: ignore[arg-type]
        self.running = False

    @property
    def is_playing(self) -> bool:
        return self.running

    def start(self, from_sample: int, *, end_sample: int | None = None) -> None:
        self._stop.clear()
        self._reset(from_sample, end_sample)
        self.running = True

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        with self._lock:
            self._anchors.clear()


def _controller(monkeypatch: pytest.MonkeyPatch) -> tuple[PlaybackController, _SteppingPlayer]:
    _SteppingPlayer.clock = _Clock()
    monkeypatch.setattr(playback_module, "AudioPlayer", _SteppingPlayer)
    settings = ProjectSettings(width=64, height=48, frame_rate=FrameRate(60))
    base = Project.create(settings)
    clip = Clip(timeline_start=0, duration=60 * 10, source=TEXT.create(text="x"))
    track = Track(TrackKind.VIDEO, "V1", clips=(clip,))
    controller = PlaybackController(base.with_timeline(replace(base.timeline, tracks=(track,))))
    player = controller._player
    assert isinstance(player, _SteppingPlayer)
    return controller, player


def _play(
    controller: PlaybackController,
    player: _SteppingPlayer,
    seconds: float,
    *,
    late: float = 0.0,
) -> list[int]:
    """1ms ずつ時計を進め、かたまりの間隔ごとに書き、8ms ごとに再生の側が位置を読む

    ``late`` 秒までの遅れを書き込みが戻る時刻へ乱数で足す デバイスは決まった速さで
    鳴らすが、書き込みが戻るのは空きができた後で、遅れることがある
    """
    frames: list[int] = []
    controller.frame_changed.connect(frames.append)
    controller.play()
    clock = player.clock
    jitter = random.Random(463)
    block = 0
    next_write = 0.0
    next_tick = 0.0
    for step in range(int(seconds * 1000)):
        clock.now = step / 1000
        while clock.now >= next_write:
            player._advance(BLOCK_SAMPLES, BUFFERED)
            block += 1
            next_write = block * BLOCK_SECONDS + jitter.uniform(0.0, late)
        if clock.now >= next_tick:
            controller._tick()
            next_tick += POLL_INTERVAL_MS / 1000
    controller.close()
    return frames


class TestFramesAdvanceOneByOne:
    def test_each_frame_is_shown_at_60fps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 補わないと、位置が 21 ms 刻みで進み 60fps（16.7 ms）のコマが規則的に飛ぶ
        controller, player = _controller(monkeypatch)
        frames = _play(controller, player, 2.0)
        assert len(frames) > 100
        steps = {b - a for a, b in itertools.pairwise(frames)}
        assert steps == {1}

    def test_late_writes_do_not_skip_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 手元の MME では書き込みが戻る間隔の 95% が 30 ms（本来は 21 ms） 最後のかたまり
        # だけから補うと上限で止まって次で 2 コマ進むので、最近のかたまりの一番先を使う
        controller, player = _controller(monkeypatch)
        frames = _play(controller, player, 2.0, late=0.010)
        steps = {b - a for a, b in itertools.pairwise(frames)}
        assert steps == {1}

    def test_without_the_fill_frames_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 上の試験が補いで通っていることの裏付け 補いを止めると 2 コマ飛ばしが出る
        monkeypatch.setattr(player_module, "EXTRAPOLATE_LIMIT", 0)
        controller, player = _controller(monkeypatch)
        frames = _play(controller, player, 2.0)
        steps = [b - a for a, b in itertools.pairwise(frames)]
        assert 2 in steps


def _player() -> tuple[AudioPlayer, _Clock]:
    clock = _Clock()
    player = AudioPlayer(_SilentMixer(), clock=clock)  # type: ignore[arg-type]
    player._reset(0, None)
    return player, clock


class TestLipSync:
    def test_the_fill_does_not_lead_the_sound(self) -> None:
        # 書き込みは空きができてから戻るので、戻る時刻は本当の時刻より遅れることはあっても
        # 早まることはない 遅れた分だけ補いが遅れるが、音より先へは出ない
        # ずれの最大は補わない時計（最大 1 かたまり遅れ）より小さいこと
        player, clock = _player()
        jitter = random.Random(463)
        writes = [k * BLOCK_SECONDS + jitter.uniform(0.0, 0.005) for k in range(1, 200)]
        filled_errors: list[float] = []
        stepped_errors: list[float] = []
        index = 0
        for step in range(int(writes[-1] * 1000)):
            clock.now = step / 1000
            while index < len(writes) and writes[index] <= clock.now:
                player._advance(BLOCK_SAMPLES, BUFFERED)
                index += 1
            # 比べるのは鳴り始めてから その前は時計が頭で待っていて、補いの話ではない
            if index <= BUFFERED // BLOCK_SAMPLES:
                continue
            heard = clock.now * RATE - BUFFERED
            filled_errors.append(player.position_sample - heard)
            stepped_errors.append(player._position - heard)
        # 先へ出ない（切り捨ての 1 サンプルは許す）
        assert max(filled_errors) <= 1
        # 遅れは書き込みの揺れ（5 ms）と、この試験が時計を進める刻み（1 ms）の和に収まり、
        # 補わない時計の遅れより小さい
        assert min(filled_errors) >= -(0.006 * RATE + 1)
        assert min(filled_errors) > min(stepped_errors)
        mean_filled = sum(abs(e) for e in filled_errors) / len(filled_errors)
        mean_stepped = sum(abs(e) for e in stepped_errors) / len(stepped_errors)
        assert mean_filled < mean_stepped / 2


class TestFillLimits:
    def test_nothing_is_filled_before_the_sound_starts(self) -> None:
        # バッファに積んでいるだけの間に経過で進めると、音より先に映像が動き出す
        player, clock = _player()
        player._advance(BLOCK_SAMPLES, BUFFERED)
        clock.now = 0.5
        assert player.position_sample == 0

    def test_a_stalled_device_stops_the_fill_at_one_block(self) -> None:
        # デバイスが詰まって書けない間に経過だけで進めると、映像だけが先へ走る
        player, clock = _player()
        for _ in range(5):
            player._advance(BLOCK_SAMPLES, BUFFERED)
        base = player._position
        clock.now = 0.3
        assert player.position_sample == base + EXTRAPOLATE_LIMIT

    def test_the_fill_never_passes_what_was_written(self) -> None:
        # 書いていない所はまだ鳴らない バッファが 1 かたまりより小さい出口でも越えない
        player, clock = _player()
        for _ in range(3):
            player._advance(BLOCK_SAMPLES, BLOCK_SAMPLES // 2)
        clock.now = 0.3
        assert player.position_sample == 3 * BLOCK_SAMPLES

    def test_the_clock_never_goes_back_while_playing(self) -> None:
        # 遅れた書き込みの後に早い書き込みが続いても、答えた位置より手前は答えない
        # 戻ると映像が 1 コマ戻って見える
        player, clock = _player()
        jitter = random.Random(455)
        last = -1
        moment = 0.0
        for _ in range(300):
            moment += BLOCK_SECONDS + jitter.uniform(-0.008, 0.012)
            for fraction in (0.25, 0.5, 0.75):
                clock.now = moment - BLOCK_SECONDS * (1 - fraction)
                position = player.position_sample
                assert position >= last
                last = position
            clock.now = moment
            player._advance(BLOCK_SAMPLES, BUFFERED)

    def test_stopping_freezes_the_clock(self) -> None:
        # 止めた後も経過で進めると、止めた所より先から再生し直すことになる
        player, clock = _player()
        for _ in range(6):
            player._advance(BLOCK_SAMPLES, BUFFERED)
        clock.now = 0.01
        stopped = player.position_sample
        player.stop()
        clock.now = 1.0
        assert player.position_sample == stopped

    def test_moving_back_resets_the_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 手前へ飛んだら前の回に答えた位置を捨てる 残すと、飛んだ先ではなく前の所を答え続ける
        written = threading.Event()

        class _BlockingStream:
            latency = 0.05

            def __init__(self, **options: object) -> None:
                del options

            def start(self) -> None:
                pass

            def write(self, block: np.ndarray) -> None:
                del block
                written.set()
                player._stop.wait(2.0)

            def stop(self) -> None:
                pass

            def close(self) -> None:
                pass

        monkeypatch.setattr(sounddevice, "OutputStream", _BlockingStream)
        player, clock = _player()
        for _ in range(10):
            player._advance(BLOCK_SAMPLES, BUFFERED)
        clock.now = 0.01
        assert player.position_sample > 4800
        player.start(4800)
        assert written.wait(2.0)
        assert player.position_sample == 4800
        player.stop()


def _play_on_vsync(
    controller: PlaybackController,
    player: _SteppingPlayer,
    seconds: float,
    *,
    period: float,
    offset: float,
    wobble: float,
) -> list[tuple[float, int]]:
    """描く所が画面の切り替わりを待つときの見回り ``period`` ごとに ``wobble`` の揺れで読む

    音の時計は揺れずに進むものとし（補いが効いた後の姿）、見回りの時刻だけを動かす
    """
    shown: list[tuple[float, int]] = []
    controller.frame_changed.connect(lambda frame: shown.append((player.clock.now, frame)))
    controller.play()
    jitter = random.Random(60)
    tick = 1
    while tick * period < seconds:
        player.clock.now = tick * period + offset + jitter.uniform(-wobble, wobble)
        # 位置を時計どおりに置く 書き込みの揺れは上の試験で見た
        with player._lock:
            player._position = int(player.clock.now * RATE)
            player._written = player._position + BUFFERED
            player._anchors.clear()
        controller._tick()
        tick += 1
    controller.close()
    return shown


class TestTicksOnTheDisplayRate:
    def test_ticks_near_the_frame_boundary_still_show_each_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 60Hz の画面で 60fps の素材 読む時刻がコマの境目（offset 0）にそろい、±3 ms
        # 揺れる 補正しないと同じコマと 2 コマ先が交互に出る
        controller, player = _controller(monkeypatch)
        shown = _play_on_vsync(controller, player, 6.0, period=1 / 60, offset=0.0, wobble=0.003)
        settled = [frame for moment, frame in shown if moment > 1.0]
        steps = {b - a for a, b in itertools.pairwise(settled)}
        assert steps == {1}

    def test_without_the_phase_lock_frames_flicker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 上の試験が補正で通っていることの裏付け
        monkeypatch.setattr(playback_module, "PHASE_LOCK", 2.0)
        controller, player = _controller(monkeypatch)
        shown = _play_on_vsync(controller, player, 6.0, period=1 / 60, offset=0.0, wobble=0.003)
        settled = [frame for moment, frame in shown if moment > 1.0]
        steps = [b - a for a, b in itertools.pairwise(settled)]
        assert 2 in steps

    def test_the_shift_stays_within_a_frame_and_a_quarter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 口と声のずれ 補正で出すコマを音の位置から 1.25 コマより先へ出さない
        # 補正しない読み方（切り捨て）でも、出すコマは音より最大 1 コマ遅れる
        controller, player = _controller(monkeypatch)
        shown = _play_on_vsync(controller, player, 6.0, period=1 / 60, offset=0.0, wobble=0.003)
        for moment, frame in shown:
            exact = moment * 60
            assert -1.25 <= frame - exact <= 1.25

    def test_the_clock_and_display_drifting_apart_skips_rarely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 画面が 59.94Hz で素材が 60fps 位相は 16 秒ほどで 1 周する 避けられない飛びは
        # 1 周に 1 回で、境目の近くを通る間に何度も行き来しない
        controller, player = _controller(monkeypatch)
        shown = _play_on_vsync(
            controller, player, 9.5, period=1001 / 60000, offset=0.0, wobble=0.003
        )
        settled = [frame for moment, frame in shown if moment > 1.0]
        steps = [b - a for a, b in itertools.pairwise(settled)]
        assert steps.count(2) <= 1
        assert all(step in (1, 2) for step in steps)

    @pytest.mark.parametrize(
        ("interval", "fps"),
        [(0.008, 60), (1 / 60, 30), (1 / 60, 24), (1 / 144, 60)],
        ids=["8ms で 60fps", "60Hz で 30fps", "60Hz で 24fps", "144Hz で 60fps"],
    )
    def test_ticks_out_of_step_with_frames_are_not_shifted(self, interval: float, fps: int) -> None:
        # 読む間隔とコマの間隔が合っていないと位相がそろわないので、ずらさない
        # ずらすと、良くならないのに口と声がずれるだけになる
        phase = FramePhase()
        for tick in range(600):
            phase.frame(tick * interval * fps)
        assert abs(phase.bias) < 0.05
