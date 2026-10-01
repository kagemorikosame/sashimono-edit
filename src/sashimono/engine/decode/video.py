"""映像のデコードとシーク

編集ソフトの再生は「順方向に読み続ける」のと「任意の位置へ飛ぶ」が交互に来る
順方向は前のフレームから続けて読むのが圧倒的に速く、飛ぶときはキーフレームまで
戻ってから読み直すしかない この 2 つを使い分けるのがこのクラスの仕事
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import TracebackType

import av
import av.error
import numpy as np

from sashimono.core.model import VideoStreamInfo
from sashimono.core.timebase import Rounding, seconds_to_pts
from sashimono.engine.colorspace import to_rgb_array
from sashimono.engine.decode.probe import ProbeError, media_origin, moving_pictures, probe_media
from sashimono.engine.decode.rational import as_fraction

__all__ = ["VideoDecoder"]

#: この秒数までなら、シークせず順方向にデコードして目的位置まで進む
#: シークはキーフレームまで戻るので、GOP 1 つ分を読み直すより前進の方が速いことが多い
FORWARD_DECODE_WINDOW = Fraction(1)


class VideoDecoder:
    """1 本の映像ストリームからフレームを取り出す

    スレッドセーフではない 素材ごと・再生系統ごとに 1 つずつ持つこと
    """

    def __init__(self, path: Path, stream_index: int | None = None) -> None:
        self._path = Path(path)
        try:
            self._container = av.open(str(self._path))
        except (av.error.FFmpegError, OSError) as exc:
            raise ProbeError(f"素材を開けない: {self._path} ({exc})") from exc

        # 素材の解析と同じくカバー画像は映像に数えない 数えると音楽のファイルを開けてしまい、
        # 1 枚しか無い絵の中をシークして落ちる
        streams = moving_pictures(self._container)
        if not streams:
            self._container.close()
            raise ProbeError(f"映像ストリームが無い: {self._path}")

        self._stream = (
            streams[0]
            if stream_index is None
            else next((s for s in streams if s.index == stream_index), streams[0])
        )
        # スレッド並列デコードは 4K 素材で目に見えて効く
        self._stream.thread_type = "AUTO"

        item = probe_media(self._path)
        self._info = next(
            (s for s in item.video_streams if s.index == self._stream.index),
            item.video_streams[0],
        )
        self._duration = item.duration
        #: 素材の時刻の原点（秒 PTS の数え方） 時刻はフレームの PTS からこれを引いて数える
        #: 音声のデコーダも同じ原点を引くので、素材の中の映像と音の食い違いはそのまま残る
        self._origin = media_origin(self._container)
        # 絵を出すのは映像の道の終わりまで 道の終わりも同じ原点から数えて取ってある
        # コンテナの終わりで切ると、音の方が長い素材では音だけの区間にも直前の絵が残る
        # 最後の絵を出し続けたいクリップは ``Clip.hold_at`` で明示して止める
        # 道の長さが分からない素材は素材全体の長さで切る（これも原点から数えてある）
        self._end = self._info.end_time if self._info.end_time is not None else self._duration

        self._frames = self._container.decode(self._stream)
        #: 今「表示されている」フレーム 最後に返したもの
        self._current: av.VideoFrame | None = None
        #: 1 つ先読みしたフレーム これの時刻が来るまで ``_current`` が表示され続ける
        self._pending: av.VideoFrame | None = None

    @property
    def info(self) -> VideoStreamInfo:
        return self._info

    @property
    def duration(self) -> Fraction:
        return self._duration

    def __enter__(self) -> VideoDecoder:
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

    def frame_at(self, seconds: Fraction) -> np.ndarray | None:
        """``seconds`` の時点で表示されているフレームを RGBA uint8 で返す

        戻り値は ``(高さ, 幅, 4)`` 素材の終端を越えた場合は ``None``
        回転情報を持つ素材では、表示すべき向きに直してから返す
        """
        target = max(Fraction(0), Fraction(seconds))
        if self._duration > 0 and target >= self._end:
            return None

        frame = self._decode_at(target)
        if frame is None:
            return None
        return _to_rgba(frame, self._info.rotation)

    def _decode_at(self, target: Fraction) -> av.VideoFrame | None:
        """``target`` を超えない最後のフレームを返す それが表示中のフレーム

        あるフレームがいつまで表示されるかは、次のフレームを読むまで分からない
        そこで常に 1 つ先読みし、その時刻が来るまで手前のフレームを返し続ける
        """
        if self._needs_seek(target):
            self._seek(target)

        while True:
            if self._pending is None:
                self._pending = self._read_next()
                if self._pending is None:
                    # 終端 最後に読めたフレームがそのまま表示され続ける
                    return self._current

            if self._current is None or self._frame_time(self._pending) <= target:
                self._current = self._pending
                self._pending = None
                continue

            return self._current

    def _read_next(self) -> av.VideoFrame | None:
        try:
            return next(self._frames)
        except StopIteration:
            return None
        except av.error.FFmpegError:
            return None

    def _needs_seek(self, target: Fraction) -> bool:
        """順方向デコードで届かない位置ならシークが必要"""
        if self._current is None:
            return target > FORWARD_DECODE_WINDOW
        position = self._frame_time(self._current)
        if target < position:
            return True
        return target - position > FORWARD_DECODE_WINDOW

    def _seek(self, target: Fraction) -> None:
        """``target`` 以前のキーフレームへ飛ぶ

        ``backward=True`` で必ず手前のキーフレームに着地させる 行き過ぎると
        目的フレームを飛び越してしまい、もう一度シークし直すことになる
        """
        time_base = self._stream.time_base or Fraction(1, 1000)
        pts = seconds_to_pts(target + self._origin, as_fraction(time_base), Rounding.FLOOR)
        try:
            self._container.seek(pts, stream=self._stream, backward=True)
        except av.error.FFmpegError:
            self._container.seek(0, stream=self._stream, backward=True)

        self._frames = self._container.decode(self._stream)
        self._current = None
        self._pending = None

    def _frame_time(self, frame: av.VideoFrame) -> Fraction:
        """フレームの表示時刻（秒 素材の原点から） ``time`` は float なので PTS から作り直す

        原点を引かないと、頭が 0 より後ろの素材でクリップの読む時刻がすべて最初の
        フレームより前になり、頭の絵が止まったまま動かない（Issue #123）
        """
        if frame.pts is None or frame.time_base is None:
            return Fraction(0)
        return frame.pts * as_fraction(frame.time_base) - self._origin


def _to_rgba(frame: av.VideoFrame, rotation: int) -> np.ndarray:
    """デコード済みフレームを RGBA の配列へ 回転があれば適用する

    行列は :func:`~sashimono.engine.colorspace.to_rgb_array` に決めさせる 素の
    ``to_ndarray`` はタグの無い素材を大きさに関係なく BT.601 で読む タグの無い HD の
    素材はほとんど BT.709 で作られているので、赤がくすみ緑が黄色へ寄った絵になる
    """
    image = to_rgb_array(frame, "rgba")
    if rotation == 0:
        return image
    # np.rot90 は反時計回りなので、時計回り 90 度は k=-1 にあたる
    return np.ascontiguousarray(np.rot90(image, k=-rotation // 90))
