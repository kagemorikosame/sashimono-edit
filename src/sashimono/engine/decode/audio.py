"""音声のデコードとリサンプル

呼び出し側はプロジェクトのサンプリングレート・チャンネル数で欲しがるので、素材が
何であってもここで揃えて返す 返すのは常に float32 の ``(サンプル数, チャンネル数)``
編集中の音量計算やミックスを float でやる方が、クリッピングの扱いが素直になる
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from types import TracebackType

import av
import av.audio.resampler
import av.error
import numpy as np

from sashimono.core.model import AudioStreamInfo
from sashimono.core.timebase import Rounding, seconds_to_pts
from sashimono.engine.decode.probe import ProbeError, media_origin, probe_media
from sashimono.engine.decode.rational import as_fraction

__all__ = ["AudioDecoder"]

#: この秒数までなら、シークせず順方向にデコードして目的位置まで進む
FORWARD_DECODE_WINDOW = Fraction(1, 2)

#: シーク時に目的位置より手前へ余分に戻る秒数
#: AAC などはフレーム先頭にデコーダのプライミング（無効サンプル）を含むため、
#: 目的位置ちょうどへ飛ぶとその無音混じりの領域を掴んでしまう 助走させて捨てる
SEEK_PREROLL = Fraction(1, 4)

_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


class AudioDecoder:
    """1 本の音声ストリームから、指定した範囲のサンプルを取り出す

    スレッドセーフではない 素材ごと・再生系統ごとに 1 つずつ持つこと
    """

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int,
        channels: int = 2,
        stream_index: int | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"サンプリングレートが不正: {sample_rate}")
        if channels not in _LAYOUTS:
            raise ValueError(f"未対応のチャンネル数: {channels}")

        self._path = Path(path)
        self._sample_rate = sample_rate
        self._channels = channels

        try:
            self._container = av.open(str(self._path))
        except (av.error.FFmpegError, OSError) as exc:
            raise ProbeError(f"素材を開けない: {self._path} ({exc})") from exc

        streams = self._container.streams.audio
        if not streams:
            self._container.close()
            raise ProbeError(f"音声ストリームが無い: {self._path}")

        self._stream = (
            streams[0]
            if stream_index is None
            else next((s for s in streams if s.index == stream_index), streams[0])
        )
        # 音声は 1 本のストリームを順に読むだけなので、映像ほど並列化の効果は無い
        # それでも AAC の復号はスレッドが効くので有効にしておく
        self._stream.codec_context.thread_type = "AUTO"

        item = probe_media(self._path)
        self._info = next(
            (s for s in item.audio_streams if s.index == self._stream.index),
            item.audio_streams[0],
        )
        self._duration = item.duration
        #: 素材の時刻の原点（秒 PTS の数え方） 映像のデコーダと同じ値を引く 音だけ別の
        #: 原点（音の道の頭）から数えると、素材の中で映像と音がずれている分が消えて口が合わない
        self._origin = media_origin(self._container)

        self._resampler = self._new_resampler()
        self._frames = self._container.decode(self._stream)
        #: 読み込み済みだがまだ返していないサンプル ``(サンプル数, チャンネル数)``
        self._buffer = _empty(self._channels)
        #: ``_buffer`` の先頭に対応する出力サンプル番号
        self._buffer_start = 0
        self._position_known = False
        self._exhausted = False
        #: 途中で復号に失敗したときの理由 失敗した所から先は無音で返すので、呼ぶ側には
        #: 終わりと見分けが付かない 再生は無音で続けてよいが、字幕起こしのように
        #: 読んだ音を丸ごと使う側は、欠けたまま成功と見なさないためにこれを見る
        #: シークしても消さない 一度でも壊れた所を読んだことを覚えておく
        self._decode_error: str | None = None
        #: 失敗した位置（出力のサンプル番号） 読めた所の終わり 位置が分からなければ ``None``
        self._decode_error_at: int | None = None

    @property
    def info(self) -> AudioStreamInfo:
        return self._info

    @property
    def decode_error(self) -> str | None:
        """途中で復号に失敗したときの理由 失敗していなければ ``None``"""
        return self._decode_error

    @property
    def decode_error_at(self) -> int | None:
        """失敗した位置（出力のサンプル番号） 読めた所の終わりで、ここから先が無音になる"""
        return self._decode_error_at

    @property
    def duration(self) -> Fraction:
        return self._duration

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def __enter__(self) -> AudioDecoder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._container.close()

    def read(self, start_sample: int, count: int) -> np.ndarray:
        """出力レートでの ``start_sample`` から ``count`` サンプルを返す

        素材の範囲外は無音で埋める 長さが足りないからといって短い配列を返すと、
        呼び出し側が毎回長さを揃える羽目になり、そこでずれが生まれる
        """
        if count <= 0:
            return _empty(self._channels)

        out = np.zeros((count, self._channels), dtype=np.float32)
        if start_sample + count <= 0:
            return out

        # 素材の先頭より前を要求された分は無音のまま残す
        offset = max(0, -start_sample)
        cursor = max(0, start_sample)
        remaining = count - offset

        if self._needs_seek(cursor):
            self._seek(cursor)

        while remaining > 0:
            if len(self._buffer) == 0:
                if not self._fill():
                    break
                continue

            buffer_end = self._buffer_start + len(self._buffer)
            if buffer_end <= cursor:
                # まるごと目的位置より手前 捨てて次を読む
                self._buffer = _empty(self._channels)
                self._buffer_start = buffer_end
                continue

            if self._buffer_start > cursor:
                # シークが行き過ぎた等で穴が空いている その分は無音で埋める
                gap = min(remaining, self._buffer_start - cursor)
                offset += gap
                cursor += gap
                remaining -= gap
                continue

            skip = cursor - self._buffer_start
            take = min(remaining, len(self._buffer) - skip)
            out[offset : offset + take] = self._buffer[skip : skip + take]
            self._buffer = self._buffer[skip + take :]
            self._buffer_start = cursor + take
            offset += take
            cursor += take
            remaining -= take

        return out

    def read_seconds(self, start: Fraction, duration: Fraction) -> np.ndarray:
        """秒で指定して読む フレーム境界を扱わない呼び出し側のための入口"""
        start_sample = int(start * self._sample_rate)
        count = int(duration * self._sample_rate)
        return self.read(start_sample, count)

    def _new_resampler(self) -> av.audio.resampler.AudioResampler:
        return av.audio.resampler.AudioResampler(
            format="fltp",
            layout=_LAYOUTS[self._channels],
            rate=self._sample_rate,
        )

    def _needs_seek(self, cursor: int) -> bool:
        window = int(FORWARD_DECODE_WINDOW * self._sample_rate)
        if not self._position_known:
            return cursor > window
        if cursor < self._buffer_start:
            return True
        return cursor - (self._buffer_start + len(self._buffer)) > window

    def _seek(self, cursor: int) -> None:
        # 手前へ余らせる分は原点を足してから引く 素材の中の時刻で 0 に丸めてから原点を
        # 足すと、頭の近く（余らせる分より手前）へ飛ぶときに原点より前へ戻れず、原点を
        # またぐ復号の単位から読めないコンテナでは頭の音が欠ける
        seconds = max(
            Fraction(0), self._origin + Fraction(cursor, self._sample_rate) - SEEK_PREROLL
        )
        time_base = self._stream.time_base or Fraction(1, self._sample_rate)
        pts = seconds_to_pts(seconds, as_fraction(time_base), Rounding.FLOOR)
        try:
            self._container.seek(pts, stream=self._stream, backward=True)
        except av.error.FFmpegError:
            self._container.seek(0, stream=self._stream, backward=True)

        # リサンプラは内部に前のフレームの続きを持っているので、シークしたら作り直す
        # 使い回すと、飛んだ先の音に前の位置の尻尾が混ざる
        self._resampler = self._new_resampler()
        self._frames = self._container.decode(self._stream)
        self._buffer = _empty(self._channels)
        self._exhausted = False
        # 着地点は要求位置より手前のキーフレームになる 実際の位置は最初に読めた
        # フレームの PTS から決めるので、ここでは未確定にしておく
        self._buffer_start = 0
        self._position_known = False

    def _fill(self) -> bool:
        """サンプルが得られるまでデコードを進める 1 つでも足せたら ``True``

        リサンプラは 1 フレーム入れても出力を返さないことがある（内部に溜める）
        1 回で諦めると、呼び出し側は「もう読めない」と誤解して途中で打ち切る
        """
        while not self._exhausted:
            try:
                frame = next(self._frames)
            except StopIteration:
                self._exhausted = True
                return self._flush_resampler()
            except av.error.FFmpegError as exc:
                self._exhausted = True
                self._decode_error = str(exc)
                if self._position_known:
                    self._decode_error_at = self._buffer_start + len(self._buffer)
                return False

            if self._resample_into_buffer(frame):
                return True
        return False

    def _flush_resampler(self) -> bool:
        """リサンプラに残っているサンプルを吐き出す"""
        return self._resample_into_buffer(None)

    def _resample_into_buffer(self, frame: av.AudioFrame | None) -> bool:
        """フレームをリサンプルしてバッファへ足す 何か得られたら ``True``

        リサンプル後のフレームは出力レートのタイムベースで PTS を持つので、
        PTS がそのまま出力サンプル番号になる シーク直後の実際の着地位置は
        これでしか分からない
        """
        produced = False
        for resampled in self._resampler.resample(frame):
            chunk = _planar_to_interleaved(resampled)
            if not len(chunk):
                continue
            self._append(chunk, _output_sample_index(resampled, self._sample_rate, self._origin))
            produced = True
        return produced

    def _append(self, chunk: np.ndarray, position: int | None) -> None:
        if len(self._buffer) == 0:
            self._buffer = chunk
            if position is not None:
                self._buffer_start = position
                self._position_known = True
            return
        self._buffer = np.concatenate([self._buffer, chunk], axis=0)


def _planar_to_interleaved(frame: av.AudioFrame) -> np.ndarray:
    """``fltp`` のフレームを ``(サンプル数, チャンネル数)`` へ"""
    array = frame.to_ndarray()
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.ascontiguousarray(array.T.astype(np.float32, copy=False))


def _empty(channels: int) -> np.ndarray:
    return np.zeros((0, channels), dtype=np.float32)


def _output_sample_index(frame: av.AudioFrame, sample_rate: int, origin: Fraction) -> int | None:
    """リサンプル後のフレームの先頭が、出力の何サンプル目にあたるか（素材の原点から）

    原点より前（映像より早く始まる音の前置き）は負の番号になる 切り捨ては負の側へ取る
    0 の側へ丸めると、原点をまたぐフレームが 1 サンプル後ろへずれる
    """
    if frame.pts is None:
        return None
    time_base = as_fraction(frame.time_base) if frame.time_base else Fraction(1, sample_rate)
    return math.floor((frame.pts * time_base - origin) * sample_rate)
