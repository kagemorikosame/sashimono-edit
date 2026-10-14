"""場面の切り替わりを、フレーム間の差から見つける（#319）

外の機械学習のモデルは使わない 配る zip が大きくなり、使用許諾（配布は GPL の条件）にも
響くため まず軽い計算で足りるかを試す（#298 の「決めること」） 手元の 4K の実写 154 秒
（はっきりした切り替わり 20 か所）では、切り替わりの点数はどれも 0.2 以上、切り替わりでない
所は 0.1 以下に分かれた（:data:`DEFAULT_THRESHOLD` の説明）

見るのは輝度だけ 素材を全フレーム先頭から読み、1 枚ごとに

- 縮めた輝度の、前のフレームとの差の平均（画素の差）
- 輝度のヒストグラムの、前のフレームとの差（形の差）

を並べて覚える（:class:`SceneSignal`） 感度は覚えた後で掛ける（:meth:`SceneSignal.cuts`）
感度ごとに素材を読み直さずに済み、AI が感度を変えて何度聞いても一瞬で返せる

点数は「前後の普段の差からどれだけ跳ねたか」で数える 速いパンや揺れる手持ちのように、
ずっと差が大きい所を切り替わりと見ないため 形の差は、画素の差も跳ねたときだけ足す
暗い所がゆっくり明るくなる（フェード）だけで形の差は大きく動くが、画素はほとんど動かない
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import av.error
import numpy as np

from sashimono.engine.decode.probe import media_origin, moving_pictures
from sashimono.engine.decode.rational import as_fraction

__all__ = [
    "DEFAULT_MIN_SCENE",
    "DEFAULT_THRESHOLD",
    "MEASURED_SCENE_SECONDS",
    "SAMPLE_WIDTH",
    "SceneCut",
    "SceneScan",
    "SceneSignal",
    "analyze_scenes",
]

#: 輝度を縮める横幅（画素） 縦は素材の縦横比に合わせる
#: 32・64・128・256 で手元の 4K の実写を読ませ、どれも同じ 20 か所を拾い、かかった時間も
#: 変わらなかった（読むのはデコードの時間がほとんど） 細かくしても得が無いので決め打ちにする
SAMPLE_WIDTH = 64

#: ヒストグラムの段の数 細かくすると、ノイズで隣の段へ移っただけの画素まで差に数える
HISTOGRAM_BINS = 32

#: 普段の差を測る範囲（前後それぞれの秒） 短いと、パンの始まりのような動きの変わり目を
#: 跳ねたと見る 長いと、短い間に続く切り替わりどうしが普段の差を押し上げる
BASELINE_SECONDS = 0.5

#: 画素の差に掛ける重み 画素の差（0..1）は切り替わりでも 0.1〜0.3 ほどにしかならず、
#: 形の差（0.2〜1）とそろえるため
PIXEL_WEIGHT = 2.0

#: 既定の感度（点数のしきい値） 手元の 4K の実写 154 秒で、切り替わり 20 か所の点数は
#: 0.21〜0.49、切り替わりでない所の最大は 0.09 だった その間に置く
DEFAULT_THRESHOLD = 0.15

#: 既定の最短の場面（秒） これより短い間に 2 つ見つかったら、点数の高い方だけを残す
#: 一瞬のフラッシュは入りと戻りで 2 回跳ねるので、片方に寄せる
DEFAULT_MIN_SCENE = 0.5

#: 解析にかかった時間（秒）の実測 デコードのスレッド数 → 秒 0 は「すべて」（PyAV に任せる）
#: 2026-10-14 に手元（Windows 11 論理 20 コア）で、4K 30fps の H.264 の実写 154 秒を読ませた
#: 設定画面（解析に使う CPU）に出す 縮める幅や間引きでは時間が変わらなかった（読む時間の
#: ほとんどがデコード）ので、速さの釣り合いはスレッドの数で取る
MEASURED_SCENE_SECONDS: dict[int, float] = {1: 72.0, 2: 38.0, 4: 22.0, 0: 7.0}

#: 輝度の面を、そのまま 1 バイト 1 画素で読める形式 これ以外（10bit や RGB）は縮めながら
#: 灰色へ変換させる 面を直に読むと、縮めるのに swscale を通すより 4K で 1 枚 4ms ほど速い
_EIGHT_BIT_LUMA = frozenset(
    {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "nv21", "gray"}
)


@dataclass(frozen=True, slots=True)
class SceneCut:
    """場面の切り替わり 1 つ 時刻は素材の原点から数えた秒"""

    #: 新しい場面の最初のフレームの時刻
    seconds: float
    #: その 1 つ前のフレームの時刻 切り替わりはこの間にある
    before: float
    #: 点数（:meth:`SceneSignal.scores`）
    score: float


@dataclass(frozen=True, slots=True)
class SceneSignal:
    """素材のフレーム間の差の並び 感度を掛ける前の、覚えておく物

    ``times[i]`` のフレームと、その 1 つ前のフレームの差が ``pixel[i]`` と ``histogram[i]``
    最初のフレームは比べる相手が無いので並びに入らない
    """

    times: np.ndarray
    pixel: np.ndarray
    histogram: np.ndarray
    #: どこまで読んだか（秒） 読み終えていれば素材の終わり
    analyzed: float
    #: 素材の終わりまで読んだか 偽なら途中まで（:class:`SceneScan` の写し）
    complete: bool
    #: 調べた映像ストリームの番号（素材の中の ``index``） ``None`` は分からない（先頭の映像）
    #: 映像が何本もある素材では先頭だけを調べる 番号を持たないと、2 本目を出すクリップにも
    #: 先頭の切り替わりを当てはめてしまう（PR #358 の Codex の指摘）
    stream: int | None = None
    #: 素材が途中で壊れていて、読めた所（:attr:`analyzed`）までで終えた 読み終えては
    #: いないので :attr:`complete` は偽 読み直しても同じ所で止まるので、覚えて使い回す
    #: 完了として扱うと、後ろの切り替わりの無い一覧を全部だと思って使う（PR #358 の
    #: CodeRabbit の指摘）
    broken: bool = False

    @classmethod
    def empty(cls, *, complete: bool = True, stream: int | None = None) -> SceneSignal:
        return cls(
            times=np.zeros(0, np.float64),
            pixel=np.zeros(0, np.float32),
            histogram=np.zeros(0, np.float32),
            analyzed=0.0,
            complete=complete,
            stream=stream,
        )

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def scores(self) -> np.ndarray:
        """1 枚ごとの点数 前後の普段の差（中央値）からどれだけ跳ねたか

        画素の差の跳ねを主に見て、形の差の跳ねは画素の跳ねの 2 倍までしか数えない
        形の差だけで数えると、暗い所のフェードで切り替わりを見つける（手元の実写の
        終わりのフェードで 0.29 になり、しきい値を越えた） 画素の差だけで数えると、
        同じ明るさで絵の違う場面の切り替わりを点数が低く出る
        """
        count = len(self)
        if count == 0:
            return np.zeros(0, np.float32)
        half = _baseline_half(self.times)
        pixel = np.maximum(0.0, self.pixel - _sliding_median(self.pixel, half)) * PIXEL_WEIGHT
        shape = np.maximum(0.0, self.histogram - _sliding_median(self.histogram, half))
        scores: np.ndarray = np.maximum(pixel, np.minimum(shape, 2.0 * pixel)).astype(np.float32)
        return scores

    def cuts(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_scene: float = DEFAULT_MIN_SCENE,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[SceneCut, ...]:
        """しきい値を越えた切り替わり 時刻の順

        ``min_scene`` 秒より近い 2 つは、点数の高い方だけを残す 範囲（``start`` 以上
        ``end`` 未満）で絞るのは、寄せた後に行う 先に絞ると、範囲の端の外にある強い
        切り替わりに寄せられて消えるはずの弱い物が残り、範囲の取り方で答えが変わる
        """
        scores = self.scores()
        candidates = np.flatnonzero(scores >= threshold)
        # 点数の高い順に取り、近くに取った物があれば捨てる 並びが同じ点数なら早い方を先に
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        kept: list[int] = []
        for index in order:
            moment = float(self.times[index])
            if all(abs(moment - float(self.times[other])) >= min_scene for other in kept):
                kept.append(int(index))
        kept.sort()
        found = []
        for index in kept:
            moment = float(self.times[index])
            if start is not None and moment < start:
                continue
            if end is not None and moment >= end:
                continue
            before = float(self.times[index - 1]) if index > 0 else self._first_before(index)
            found.append(SceneCut(seconds=moment, before=before, score=float(scores[index])))
        return tuple(found)

    def _first_before(self, index: int) -> float:
        """最初の差の 1 つ前（最初のフレーム）の時刻 覚えていないので、間の幅から見積もる"""
        if len(self) >= 2:
            step = float(self.times[1] - self.times[0])
            return max(0.0, float(self.times[index]) - step)
        return 0.0


class SceneScan:
    """読んでいる途中の差の並び 裏のスレッドが足し、AI の道具が写しを取る

    途中までの写しを返せるようにするのは、長い素材の解析が終わる前に AI が頼んだとき、
    読み終えた所までの切り替わりだけでも先に渡すため（#321）
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._times: list[float] = []
        self._pixel: list[float] = []
        self._histogram: list[float] = []
        self._analyzed = 0.0
        #: 読んでいる映像ストリームの番号（:attr:`SceneSignal.stream`）
        self.stream: int | None = None

    def add(self, moment: float, pixel: float, histogram: float) -> None:
        with self._lock:
            self._times.append(moment)
            self._pixel.append(pixel)
            self._histogram.append(histogram)
            self._analyzed = max(self._analyzed, moment)

    def reach(self, moment: float) -> None:
        """差を足さずに読んだ所だけを進める（最初のフレーム）"""
        with self._lock:
            self._analyzed = max(self._analyzed, moment)

    def snapshot(
        self, *, complete: bool = False, analyzed: float | None = None, broken: bool = False
    ) -> SceneSignal:
        with self._lock:
            return SceneSignal(
                times=np.asarray(self._times, np.float64),
                pixel=np.asarray(self._pixel, np.float32),
                histogram=np.asarray(self._histogram, np.float32),
                analyzed=self._analyzed if analyzed is None else analyzed,
                complete=complete,
                stream=self.stream,
                broken=broken,
            )


def analyze_scenes(
    path: Path,
    *,
    threads: int = 2,
    scan: SceneScan | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SceneSignal | None:
    """素材を先頭から読み切って、フレーム間の差を並べる

    重い処理なので裏のスレッドで呼ぶ ``should_cancel`` が真を返したら ``None`` で抜ける
    ``threads`` はデコードのスレッドの数 0 は PyAV に任せる（機械のコアをすべて使う）
    映像の無い素材は ``None`` 静止画（長さ 0）は空の並びを返す
    ``scan`` を渡すと、読んだ所から順に足していく（途中までを見せるため）
    """
    scan = scan if scan is not None else SceneScan()
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError):
        return None
    with container:
        streams = moving_pictures(container)
        if not streams:
            return None
        stream = streams[0]
        scan.stream = stream.index
        stream.thread_type = "AUTO"
        if threads > 0:
            stream.codec_context.thread_count = threads
        origin = media_origin(container)
        duration = _duration(container, stream)
        if duration <= 0:
            return SceneSignal.empty(stream=stream.index)

        #: 1 つ前のフレームの（縮めた輝度, ヒストグラム）
        previous: tuple[np.ndarray, np.ndarray] | None = None
        last = 0.0
        try:
            for frame in container.decode(stream):
                if should_cancel is not None and should_cancel():
                    return None
                moment = _frame_seconds(frame, origin)
                if moment is None:
                    continue
                luma = _luma(frame)
                shape = _histogram(luma)
                if previous is not None and previous[0].shape == luma.shape:
                    difference = np.abs(luma.astype(np.int16) - previous[0].astype(np.int16))
                    scan.add(
                        moment,
                        float(difference.mean()) / 255.0,
                        float(np.abs(shape - previous[1]).sum()) / 2.0,
                    )
                else:
                    # 途中で大きさが変わる素材（配信の録画など）は比べずに数え直す
                    # 大きさの違う絵を比べる意味は無く、形の違いで落ちる
                    scan.reach(moment)
                previous, last = (luma, shape), moment
                if progress is not None:
                    progress(min(1.0, moment / duration))
        except av.error.FFmpegError:
            # 途中で壊れた素材は、読めた所までを結果にする 全部を捨てると、末尾が欠けた
            # だけの録画で場面を 1 つも返せない ただし読み終えたとは言わない（読めたのは
            # ``last`` まで） 言うと、後ろの切り替わりの無い一覧を全部だと思って使う
            if progress is not None:
                progress(1.0)
            return scan.snapshot(analyzed=last, broken=True)
    if progress is not None:
        progress(1.0)
    return scan.snapshot(complete=True, analyzed=max(last, duration))


def _duration(container: av.container.InputContainer, stream: av.VideoStream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * as_fraction(stream.time_base))
    if container.duration is not None:
        return container.duration / 1_000_000
    return 0.0


def _frame_seconds(frame: av.VideoFrame, origin: Fraction) -> float | None:
    """フレームの時刻（素材の原点からの秒） デコーダと同じく PTS から原点を引く"""
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * as_fraction(frame.time_base) - origin)


def _luma(frame: av.VideoFrame) -> np.ndarray:
    """縮めた輝度 ``(高さ, 幅)`` の uint8

    8bit の YUV は輝度の面から等間隔に拾う（平均しない） 差を比べるだけなので、
    1 画素ずつのぼけより速さを取る それ以外は swscale で縮めながら灰色にする
    """
    width = SAMPLE_WIDTH
    height = max(2, round(width * frame.height / max(frame.width, 1)))
    if frame.format.name in _EIGHT_BIT_LUMA and frame.width >= width and frame.height >= height:
        plane = frame.planes[0]
        rows = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)
        rows = rows[: frame.height, : frame.width]
        step_y = frame.height // height
        step_x = frame.width // width
        return np.ascontiguousarray(
            rows[step_y // 2 :: step_y, step_x // 2 :: step_x][:height, :width]
        )
    return frame.reformat(width=width, height=height, format="gray").to_ndarray()


def _histogram(luma: np.ndarray) -> np.ndarray:
    """輝度の段ごとの割合（合計 1）"""
    counts = np.bincount((luma >> 3).ravel(), minlength=HISTOGRAM_BINS)
    return counts.astype(np.float32) / max(1, luma.size)


def _baseline_half(times: np.ndarray) -> int:
    """普段の差を測る片側の枚数 フレームの間隔の中央値から求める"""
    if times.shape[0] < 2:
        return 1
    step = float(np.median(np.diff(times)))
    if step <= 0:
        return 1
    return max(2, round(BASELINE_SECONDS / step))


def _sliding_median(values: np.ndarray, half: int) -> np.ndarray:
    """前後 ``half`` 枚ずつ（自分も含む）の中央値

    自分を含めても、跳ねた 1 枚は中央値を動かさない 端は折り返して埋める
    長い素材（1 時間 60fps で 21 万枚）でも窓を作る配列が大きくならないよう、区切って求める
    """
    count = values.shape[0]
    if count == 0:
        return values
    width = 2 * half + 1
    # 1 枚だけは折り返せない（numpy が断る） 同じ値で埋めれば中央値はその値
    source = values.astype(np.float32)
    if count > 1:
        padded = np.pad(source, half, mode="reflect")
    else:
        padded = np.pad(source, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    result = np.empty(count, np.float32)
    chunk = 20000
    for begin in range(0, count, chunk):
        result[begin : begin + chunk] = np.median(windows[begin : begin + chunk], axis=1)
    return result
