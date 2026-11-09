"""再生の制御

音を時計にする 映像は「今どこを再生しているか」を音に尋ねて、その位置の
フレームを出す 映像を時計にすると、映像が遅れたときに音を飛ばすことになり、
途切れが即座に耳につく 音は途切れさせず、映像側がコマを落として追いつく
"""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QTimer, Signal

from sashimono.core.model import Project
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio import AudioMixer, AudioPlayer, PlaybackError

__all__ = ["FramePhase", "PlaybackController"]

#: 再生位置を見に行く間隔（ミリ秒） フレーム間隔より細かくしておかないと、
#: 表示するフレームが飛ぶ
POLL_INTERVAL_MS = 8

#: 位相を均す重み（1 回読むごとに今の値をこれだけ混ぜる） 0.05 で約 20 回 60Hz で 0.3 秒
PHASE_SMOOTHING = 0.05

#: 読む時刻の位相がこれ以上そろっていたら、境目を避ける補正をかける（0〜1 の揃い具合）
#: 揃っていないとき（読む間隔とコマの間隔が合っていない）は、どこで読んでも
#: 境目の近くに来る回は散らばっていて、補正しても何も良くならない
#: 60Hz で読む時刻が ±3 ms 揺れると揃い具合は 0.78 ほど 30fps を 60Hz で読む・8 ms ごとに
#: 読む・24fps を 60Hz で読むときは 0 近くまで下がるので、間の 0.5 で分ける
PHASE_LOCK = 0.5


class FramePhase:
    """読む時刻をコマの境目から離す補正

    描く所が画面の切り替わりを待つので、再生の見回りは 8 ms ではなく画面の切り替わり
    （60Hz で 16.7 ms）ごとにしか回らない（2026-11-09 手元で中央 16.5 ms） 60fps の素材と
    60Hz の画面では、読む時刻がコマのどこに来るか（位相）がほとんど動かず、それが境目の
    近くにあると、数 ms の揺れで「前と同じコマ」と「2 コマ先」が交互に出る（#463 で
    出したコマの 13%）

    読んだ位置の位相を均し、そろっていれば、読む時刻がコマの真ん中に来るように位置を
    ずらしてからコマを決める ずらす量は 1 コマ未満が基本で、境目の向こうへ回り込む
    ときだけ 1/4 コマまで余分を許す（毎回行き来しないため） 回り込んだ所で 1 回だけ
    同じコマが続くか 1 コマ飛ぶ 音と画面の時計の速さの差でゆっくり起きる分で、避けられない
    """

    def __init__(self) -> None:
        self._x = 0.0
        self._y = 0.0
        self.bias = 0.0

    def reset(self) -> None:
        """位置を変えたら均した位相を捨てる 前の回の位相は新しい位置の読み方と関係が無い"""
        self._x = self._y = 0.0
        self.bias = 0.0

    def frame(self, exact: float) -> int:
        """``exact``（コマの単位の位置 端数つき）を出すコマにする"""
        angle = 2 * math.pi * (exact - math.floor(exact))
        self._x += PHASE_SMOOTHING * (math.cos(angle) - self._x)
        self._y += PHASE_SMOOTHING * (math.sin(angle) - self._y)
        if math.hypot(self._x, self._y) >= PHASE_LOCK:
            # 円の上で平均を取る 0.02 と 0.98 は境目をはさんだ同じ所で、ただ足すと 0.5 になる
            mean = math.atan2(self._y, self._x) / (2 * math.pi)
            target = 0.5 - mean
            # 今のずれから近い向きへ寄せる 遠い向きへ回ると 1 コマ飛ぶ
            self.bias += (target - self.bias + 0.5) % 1.0 - 0.5
            if self.bias > 1.25:
                self.bias -= 1.0
            elif self.bias < -0.25:
                self.bias += 1.0
        else:
            self.bias *= 1 - PHASE_SMOOTHING
        return math.floor(exact + self.bias)


class PlaybackController(QObject):
    """再生・停止と、再生位置の通知"""

    #: 再生位置が進んだ 引数はフレーム番号
    frame_changed = Signal(int)
    #: 再生状態が変わった
    state_changed = Signal(bool)
    #: 音声出力を開けなかった等 引数はメッセージ
    failed = Signal(str)

    def __init__(
        self, project: Project, parent: QObject | None = None, *, smooth_history: bool = True
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._mixer = AudioMixer(project, smooth_history=smooth_history)
        self._player = AudioPlayer(self._mixer)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._playing = False
        self._frame = 0
        self._start_frame = 0
        self._phase = FramePhase()

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def frame(self) -> int:
        return self._frame

    def set_project(self, project: Project) -> None:
        was_playing = self._playing
        if was_playing:
            self.stop()
        self._project = project
        self._mixer.set_project(project)

    def set_smooth_history(self, smooth: bool) -> None:
        """前の音を読む音の効果の動く値を区切りの中でもつなぐか（本人の設定）

        鳴らしている間にミキサを触ると、再生のスレッドと取り合う 鳴っていれば止めて
        替え、同じ所から鳴らし直す
        """
        if smooth == self._mixer.smooth_history:
            return
        was_playing = self._playing
        if was_playing:
            self.stop()
        self._mixer.set_smooth_history(smooth)
        if was_playing:
            self.play()

    def set_frame(self, frame: int) -> None:
        """再生ヘッドを動かす 再生中なら、その位置から再生し直す"""
        self._frame = max(0, frame)
        if self._playing:
            self._restart_from(self._frame)

    def toggle(self) -> None:
        self.stop() if self._playing else self.play()

    def play(self) -> None:
        if self._playing:
            return
        end = self._project.duration
        if self._frame >= end:
            # 末尾で押されたら頭から 止まったままより意図に近い
            self._frame = 0

        self._start_frame = self._frame
        self._phase.reset()
        try:
            self._player.start(
                _frame_to_sample(self._frame, self._project.rate, self._mixer.sample_rate),
                end_sample=_frame_to_sample(end, self._project.rate, self._mixer.sample_rate),
            )
        except PlaybackError as exc:
            self.failed.emit(str(exc))
            return

        self._playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def stop(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self._player.stop()
        self.state_changed.emit(False)

    def close(self) -> None:
        self._timer.stop()
        self._player.close()
        self._mixer.close()

    def _restart_from(self, frame: int) -> None:
        self._player.stop()
        self._start_frame = frame
        self._phase.reset()
        try:
            self._player.start(
                _frame_to_sample(frame, self._project.rate, self._mixer.sample_rate),
                end_sample=_frame_to_sample(
                    self._project.duration, self._project.rate, self._mixer.sample_rate
                ),
            )
        except PlaybackError as exc:
            self._playing = False
            self._timer.stop()
            self.state_changed.emit(False)
            self.failed.emit(str(exc))

    def _reached_end(self, position: int) -> bool:
        """音の位置がプロジェクトの終わりに届いたか

        フレームへ直してから比べない 29.97fps のように割り切れない速さでは、
        終わりのサンプル（切り捨て）をフレームへ戻すと（これも切り捨て）1 つ手前になり、
        最後まで鳴らしても終わりに届かないまま止まる
        """
        end = self._project.duration
        rate, sample_rate = self._project.rate, self._mixer.sample_rate
        if position >= _frame_to_sample(end, rate, sample_rate):
            return True
        return _sample_to_frame(position, rate, sample_rate) >= end

    def _tick(self) -> None:
        if not self._playing:
            return

        position = self._player.position_sample
        if not self._player.is_playing:
            # 止まったのを見てから位置を読み直す 読んだ直後に鳴り終えていると、
            # 終わりの手前の位置のまま「デバイスが止まった」側へ入ってしまう
            position = self._player.position_sample
            if not self._reached_end(position):
                # デバイスが抜かれた等 止まった所で止める 最後に読んだ位置も知らせる
                # 知らせないと再生ヘッドが前の回の所に残り、そこから再生し直すことになる
                frame = _sample_to_frame(position, self._project.rate, self._mixer.sample_rate)
                if frame != self._frame:
                    self._frame = frame
                    self.frame_changed.emit(frame)
                self.stop()
                return

        if self._reached_end(position):
            self._frame = self._project.duration
            self.frame_changed.emit(self._frame)
            self.stop()
            return

        rate = self._project.rate
        exact = position * rate.num / (rate.den * self._mixer.sample_rate)
        frame = min(self._phase.frame(exact), self._project.duration - 1)

        # 戻さない 補正が境目の向こうへ回り込んだときに 1 つ手前になるが、そこは同じコマを
        # 続けて出す 戻すと絵が 1 コマ戻って見える
        if frame > self._frame:
            self._frame = frame
            self.frame_changed.emit(frame)


def _frame_to_sample(frame: int, rate: FrameRate, sample_rate: int) -> int:
    return frame * rate.den * sample_rate // rate.num


def _sample_to_frame(sample: int, rate: FrameRate, sample_rate: int) -> int:
    return sample * rate.num // (rate.den * sample_rate)
