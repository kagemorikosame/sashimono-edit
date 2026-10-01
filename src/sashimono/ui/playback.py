"""再生の制御

音を時計にする 映像は「今どこを再生しているか」を音に尋ねて、その位置の
フレームを出す 映像を時計にすると、映像が遅れたときに音を飛ばすことになり、
途切れが即座に耳につく 音は途切れさせず、映像側がコマを落として追いつく
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from sashimono.core.model import Project
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio import AudioMixer, AudioPlayer, PlaybackError

__all__ = ["PlaybackController"]

#: 再生位置を見に行く間隔（ミリ秒） フレーム間隔より細かくしておかないと、
#: 表示するフレームが飛ぶ
POLL_INTERVAL_MS = 8


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

        frame = _sample_to_frame(position, self._project.rate, self._mixer.sample_rate)

        if frame != self._frame:
            self._frame = frame
            self.frame_changed.emit(frame)


def _frame_to_sample(frame: int, rate: FrameRate, sample_rate: int) -> int:
    return frame * rate.den * sample_rate // rate.num


def _sample_to_frame(sample: int, rate: FrameRate, sample_rate: int) -> int:
    return sample * rate.num // (rate.den * sample_rate)
