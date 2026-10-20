"""被写体のいそうな所を、動きと肌色から見積もる（#334 縦動画への再構図）

外の機械学習のモデル（顔や人の検出）は使わない 配る zip が大きくなり、使用許諾（配布は
GPL の条件）にも響くため（#298 の「決めること」） 場面の解析（:mod:`sashimono.engine.scenes`）が
全フレームを読むついでに、縮めた絵から次の 2 つを数える 同じ所で数えるのは、素材をもう 1 度
読み直すと 4K の 2 分半で 20〜70 秒かかるため
（:data:`~sashimono.engine.scenes.MEASURED_SCENE_SECONDS`）

- 動き 前のフレームとの輝度の差（話す口・手振り・歩く人）
- 肌色 色差（Cb・Cr）が肌の範囲に入る画素（止まって話す人の顔）

:data:`SUBJECT_SECONDS` ごとに 1 行にまとめ、横と縦の濃さの分布（:attr:`SubjectTrack.columns`
:attr:`SubjectTrack.rows`）として覚える 切り出す枠の幅は作る縦の作品で決まるので、枠の位置は
覚えた分布から後で選ぶ（:func:`best_center`） 枠の幅ごとに読み直さずに済む

精度の限り（道具の説明にも書く）

- 顔かどうかは見ていない 肌色に近い物（木の床・ベージュの壁・赤みのある照明）も人と数える
  画面の多くを肌色が占める絵では肌色を数えない（:data:`SKIN_CROWD` :data:`SKIN_FLOOD`）が、
  見分けはしない
- 動く物が 2 つあれば、多く動く方を選ぶ 話していない人が大きく動けばそちらへ寄る
- 素材に焼き込まれた字幕やテロップは、出入りする瞬間だけ動きとして数える
- 縮めた絵は横 64 画素 枠の位置の細かさは素材の幅の 1/32 ほど
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import av
import numpy as np

__all__ = [
    "SKIN_CB",
    "SKIN_CR",
    "SKIN_CROWD",
    "SKIN_FLOOD",
    "SKIN_WEIGHT",
    "SUBJECT_BINS",
    "SUBJECT_SECONDS",
    "SubjectScan",
    "SubjectTrack",
    "best_center",
    "gathering",
    "skin_mask",
]

#: 1 行にまとめる長さ（秒） 短いと覚える量が増え、長いと速い動きに遅れる 枠は人が見て
#: 落ち着く速さでしか動かさない（:mod:`sashimono.core.reframe` のならし）ので 0.25 秒で足りる
SUBJECT_SECONDS = 0.25

#: 横と縦の分布の段の数 縮めた絵の横 64 画素を 2 画素ずつ 枠の位置の細かさがここで決まる
SUBJECT_BINS = 32

#: 動きと見なす輝度の差の下限（0〜255） 圧縮のざらつきとセンサーのノイズは 10 までに収まる
MOTION_NOISE = 10
#: 動きの濃さが 1 になる差 これより大きな差は同じ濃さに数える（明暗の差の大きい物だけが勝たない）
MOTION_FULL = 64

#: 肌色の色差の範囲（0〜255） 顔の検出で広く使われる YCbCr の範囲（Chai と Ngan 1999）
SKIN_CB = (77, 127)
SKIN_CR = (133, 173)
#: 肌色と見なす輝度の下限 暗い所は色差が揺れ、黒い服や影が肌色の範囲に入る
SKIN_LUMA = 40
#: 肌色の重み（動きの濃さ 1 に対して） 動いている所の方を強く見る 止まって話す人の顔は
#: 動きが小さいので、肌色が無いと画面の中央へ戻ってしまう
SKIN_WEIGHT = 0.5
#: 画面の中の肌色の割合がこれを超えたら肌色の重みを下げ始め、:data:`SKIN_FLOOD` で 0 にする
#: 肌色の壁や床が画面の多くを占める絵では、肌色は人ではなく地 割合で薄めるだけでは、地が
#: 画面じゅうに同じ濃さで残り、人の動きが埋もれる（実物のベージュの地の前で話す人で、枠が
#: 人ではなく真ん中に決まっていた）
SKIN_CROWD = 0.25
SKIN_FLOOD = 0.45

#: 色差の面を直に読める形式 → （横の間引き, 縦の間引き, 色差が交互に並ぶ面か, Cb が先か）
#: これ以外は縮めながら yuv444p へ変換させる 4K で 1 枚 4ms ほどかかるので、よくある形式は直に読む
_DIRECT: dict[str, tuple[int, int, bool, bool]] = {
    "yuv420p": (1, 1, False, True),
    "yuvj420p": (1, 1, False, True),
    "yuv422p": (1, 0, False, True),
    "yuvj422p": (1, 0, False, True),
    "yuv444p": (0, 0, False, True),
    "yuvj444p": (0, 0, False, True),
    "nv12": (1, 1, True, True),
    "nv21": (1, 1, True, False),
}


@dataclass(frozen=True, slots=True)
class SubjectTrack:
    """被写体のいそうな所の並び :data:`SUBJECT_SECONDS` ごとに 1 行

    ``columns[i]`` は横の分布（左から） ``rows[i]`` は縦の分布（上から） どちらも合計 1 で、
    動きも肌色も無い行は全部 0 ``strength[i]`` は絵の中の濃さの平均（0〜）で、小さい行は
    手がかりが無いので枠の位置を決めるのに使わない
    """

    times: np.ndarray
    columns: np.ndarray
    rows: np.ndarray
    strength: np.ndarray

    @classmethod
    def empty(cls) -> SubjectTrack:
        return cls(
            times=np.zeros(0, np.float64),
            columns=np.zeros((0, SUBJECT_BINS), np.float32),
            rows=np.zeros((0, SUBJECT_BINS), np.float32),
            strength=np.zeros(0, np.float32),
        )

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def centers(self, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """行ごとの枠の中心（横は左から、縦は上から 素材の幅と高さに対する割合）

        ``window`` は枠の幅と高さ（同じく割合） 枠に入る濃さがいちばん多い所を選ぶ
        """
        width, height = window
        across = np.array([best_center(row, width) for row in self.columns], np.float64)
        down = np.array([best_center(row, height) for row in self.rows], np.float64)
        return across, down

    def gathered(
        self, window: tuple[float, float], centers: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """行ごとの、枠に集まっている度合い（0〜1 :func:`gathering`）

        枠が素材いっぱいの向きは数えない
        """
        result = np.ones(len(self), np.float64)
        for profiles, share, middle in (
            (self.columns, window[0], centers[0]),
            (self.rows, window[1], centers[1]),
        ):
            if share >= 1.0:
                continue
            result = np.minimum(
                result,
                [gathering(row, share, float(c)) for row, c in zip(profiles, middle, strict=True)],
            )
        return result


def best_center(profile: np.ndarray, window: float) -> float:
    """分布 ``profile`` に幅 ``window``（0〜1）の枠を置くとき、枠に入る量がいちばん多い中心

    同じくらい入る所が続けば（被写体が枠より小さい）、その真ん中を取る 端を取ると、
    被写体が枠の片側に寄って見える 分布が空なら真ん中（0.5）
    """
    if window >= 1.0:
        return 0.5
    total = float(profile.sum())
    if total <= 0.0:
        return 0.5
    bins = profile.shape[0]
    cumulative = np.concatenate(([0.0], np.cumsum(profile, dtype=np.float64)))
    edges = np.arange(bins + 1, dtype=np.float64)
    half = window / 2.0
    candidates = np.linspace(half, 1.0 - half, 129)
    inside = np.interp((candidates + half) * bins, edges, cumulative) - np.interp(
        (candidates - half) * bins, edges, cumulative
    )
    best = float(inside.max())
    # 量の差が全体の 2% までの所は同じと見る 1 段ずれただけの差で、枠が片側へ跳ねない
    plateau = candidates[inside >= best - 0.02 * total]
    chosen = float((plateau.min() + plateau.max()) / 2.0)
    # 枠に入った分の重心へ寄せる 同じ量の入る所の真ん中だけを取ると、画面の端の近くの
    # 小さな被写体で、枠の端までしか動けない側に引かれて被写体が枠の片側に寄る
    centers = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    low, high = chosen - half, chosen + half
    overlap = (
        np.clip(np.minimum(edges[1:] / bins, high) - np.maximum(edges[:-1] / bins, low), 0.0, None)
        * bins
    )
    weights = profile.astype(np.float64) * overlap
    if float(weights.sum()) <= 0.0:
        return chosen
    centroid = float((weights * centers).sum() / weights.sum())
    return min(max(centroid, half), 1.0 - half)


def gathering(profile: np.ndarray, window: float, center: float) -> float:
    """幅 ``window`` の枠を ``center`` に置いたとき、分布がどれだけ枠に集まっているか（0〜1）

    0 は画面じゅうに同じだけ散っている（枠に入るのは枠の幅の割合だけ） 1 は全部が枠に入る
    ゲームの画面の録画や、画面じゅうが動く手持ちの絵は 0 に近い そういう行の「いちばん多い所」は
    たまたまの偏りで、追うと枠が左右に跳ね回る（実物の画面の録画で毎秒 1000 画素を超えた）
    """
    total = float(profile.sum())
    if total <= 0.0 or window >= 1.0:
        return 0.0 if total <= 0.0 else 1.0
    bins = profile.shape[0]
    cumulative = np.concatenate(([0.0], np.cumsum(profile, dtype=np.float64)))
    edges = np.arange(bins + 1, dtype=np.float64)
    half = window / 2.0
    inside = float(
        np.interp((center + half) * bins, edges, cumulative)
        - np.interp((center - half) * bins, edges, cumulative)
    )
    return min(1.0, max(0.0, (inside / total - window) / (1.0 - window)))


def _profile(values: np.ndarray, bins: int) -> np.ndarray:
    """1 次元の量を ``bins`` 段に配り直す（合計を保つ） 縦の画素の数は素材の縦横比で変わる"""
    count = values.shape[0]
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    at = np.interp(np.linspace(0.0, count, bins + 1), np.arange(count + 1), cumulative)
    return np.diff(at)


def _normalized(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return np.zeros(values.shape, np.float32)
    normalized: np.ndarray = (values / total).astype(np.float32)
    return normalized


def skin_mask(frame: av.VideoFrame, luma: np.ndarray) -> np.ndarray | None:
    """縮めた輝度 ``luma`` と同じ升の肌色の印 色を持たない素材（灰色）は ``None``

    升の取り方は :func:`sashimono.engine.scenes._luma` と同じ（等間隔に拾う） ずれると、
    動きと肌色が別々の所を指す
    """
    height, width = luma.shape
    layout = _DIRECT.get(frame.format.name)
    if layout is not None and frame.width >= width and frame.height >= height:
        step_y = frame.height // height
        step_x = frame.width // width
        ys = step_y // 2 + np.arange(height) * step_y
        xs = step_x // 2 + np.arange(width) * step_x
        sub_x, sub_y, interleaved, cb_first = layout
        if interleaved:
            plane = frame.planes[1]
            chroma = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)
            columns = (xs >> sub_x) * 2
            first = chroma[np.ix_(ys >> sub_y, columns)]
            second = chroma[np.ix_(ys >> sub_y, columns + 1)]
            cb, cr = (first, second) if cb_first else (second, first)
        else:
            cb_plane, cr_plane = frame.planes[1], frame.planes[2]
            cb = np.frombuffer(cb_plane, np.uint8).reshape(-1, cb_plane.line_size)[
                np.ix_(ys >> sub_y, xs >> sub_x)
            ]
            cr = np.frombuffer(cr_plane, np.uint8).reshape(-1, cr_plane.line_size)[
                np.ix_(ys >> sub_y, xs >> sub_x)
            ]
    elif frame.format.name in ("gray", "gray8", "gray10le", "gray12le", "gray16le"):
        return None
    else:
        small = frame.reformat(width=width, height=height, format="yuv444p")
        planes = [
            np.frombuffer(p, np.uint8).reshape(-1, p.line_size)[:height, :width]
            for p in small.planes[1:3]
        ]
        cb, cr = planes[0], planes[1]
    mask: np.ndarray = (
        (cb >= SKIN_CB[0])
        & (cb <= SKIN_CB[1])
        & (cr >= SKIN_CR[0])
        & (cr <= SKIN_CR[1])
        & (luma >= SKIN_LUMA)
    )
    return mask


@dataclass(slots=True)
class _Bin:
    """まとめている途中の 1 行"""

    index: int
    first: float
    last: float
    motion: np.ndarray | None = None
    frames: int = 0
    skin: np.ndarray | None = None


@dataclass(slots=True)
class SubjectScan:
    """読んでいる途中の被写体の並び 解析のスレッドが足し、AI の道具が写しを取る"""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _times: list[float] = field(default_factory=list)
    _columns: list[np.ndarray] = field(default_factory=list)
    _rows: list[np.ndarray] = field(default_factory=list)
    _strength: list[float] = field(default_factory=list)
    _open: _Bin | None = None

    def feed(
        self,
        moment: float,
        difference: np.ndarray | None,
        skin: Callable[[], np.ndarray | None],
    ) -> None:
        """1 フレームを足す ``difference`` は前のフレームとの輝度の差（無ければ ``None``）

        ``skin`` は行の最初のフレームでだけ呼ぶ 色差を読むのは 1 行に 1 度で足りる
        （止まった顔の位置は 0.25 秒では動かない）
        """
        index = math.floor(moment / SUBJECT_SECONDS)
        current = self._open
        if current is not None and current.index != index:
            self._close(current)
            current = None
        if current is None:
            current = _Bin(index=index, first=moment, last=moment, skin=skin())
            self._open = current
        current.last = moment
        if difference is None:
            return
        motion = np.clip(
            (difference.astype(np.float32) - MOTION_NOISE) / (MOTION_FULL - MOTION_NOISE),
            0.0,
            1.0,
        )
        if current.motion is None or current.motion.shape != motion.shape:
            # 途中で大きさの変わる素材は、その行の動きを数え直す
            current.motion, current.frames = motion, 1
        else:
            current.motion += motion
            current.frames += 1

    def finish(self) -> None:
        """まとめている途中の行を閉じる 読み終えたとき・壊れて止まったときに呼ぶ"""
        if self._open is not None:
            self._close(self._open)

    def _close(self, done: _Bin) -> None:
        self._open = None
        motion = done.motion / max(1, done.frames) if done.motion is not None else None
        skin = done.skin
        if motion is not None and skin is not None and skin.shape != motion.shape:
            skin = None
        layers: list[np.ndarray] = []
        if motion is not None:
            layers.append(motion.astype(np.float32))
        if skin is not None:
            share = float(skin.mean())
            fade = (SKIN_FLOOD - share) / (SKIN_FLOOD - SKIN_CROWD)
            weight = SKIN_WEIGHT * min(1.0, max(0.0, fade))
            layers.append(skin.astype(np.float32) * weight)
        interest = sum(layers[1:], layers[0]) if layers else None
        if interest is None:
            columns = np.zeros(SUBJECT_BINS, np.float32)
            rows = np.zeros(SUBJECT_BINS, np.float32)
            strength = 0.0
        else:
            columns = _normalized(_profile(interest.sum(axis=0), SUBJECT_BINS))
            rows = _normalized(_profile(interest.sum(axis=1), SUBJECT_BINS))
            strength = float(interest.mean())
        with self._lock:
            self._times.append((done.first + done.last) / 2.0)
            self._columns.append(columns)
            self._rows.append(rows)
            self._strength.append(strength)

    def snapshot(self) -> SubjectTrack:
        with self._lock:
            if not self._times:
                return SubjectTrack.empty()
            return SubjectTrack(
                times=np.asarray(self._times, np.float64),
                columns=np.stack(self._columns).astype(np.float32),
                rows=np.stack(self._rows).astype(np.float32),
                strength=np.asarray(self._strength, np.float32),
            )
