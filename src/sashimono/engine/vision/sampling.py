"""画面の中身を調べる時刻を選ぶ（#324）

全フレームは調べない 文字の読み取りと顔の検出は 1 枚ずつ数十 ms かかり、4K の素材では
絵を取り出す（シークしてデコードする）方がさらに重い 場面の切り替わり（#319）で素材を
区切り、場面ごとに数枚だけ調べる 縦への作り替え（#334）も枠を場面ごとに決めるので、
場面の中の数枚で足りる

場面の頭と終わりのすぐ近くは外す 切り替わりの前後のフレームは、溶ける場面切り替えの途中や
動きのぶれで、文字も顔も読めないことが多い
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

__all__ = [
    "EDGE_SECONDS",
    "FALLBACK_STEP",
    "MAX_SAMPLES",
    "MEASURED_SCREEN_SAMPLE",
    "MIN_STEP",
    "fallback_cuts",
    "sample_times",
    "scene_bounds",
]

#: 場面の頭と終わりから外す秒 短い場面では場面の長さの 1/4 まで縮める
EDGE_SECONDS = 0.2
#: 同じ場面で調べる 2 枚の間の最短の秒 短い場面に何枚も取ると、ほぼ同じ絵を調べるだけになる
MIN_STEP = 0.5
#: 1 本の素材で調べる枚数の上限 切り替わりの多い長い素材（数千の場面）で、1 枚ずつ取り出す
#: 時間が何十分にもならないため 越えるときは場面ごとに 1 枚（場面の真ん中）へ減らし、
#: それでも越えれば場面を間引く
MAX_SAMPLES = 600
#: 場面が分からない（場面の解析に失敗した）ときに等間隔で調べる間（秒）
FALLBACK_STEP = 2.0

#: 1 枚を調べるのに掛かった時間（2026-10-20 手元の機械 論理 20 コア 30 秒・2 秒ごとに
#: キーフレームの H.264 で 40 枚の平均） 絵の取り出しと文字と顔の両方を合わせた物
#: 1920x1080 は取り出し 10 ms・文字 36 ms・顔 50 ms、3840x2160 は 31 ms・223 ms・127 ms
#: 設定画面の説明に出す（どの枚数を選ぶと、どれだけ待つかを見て選べるように）
MEASURED_SCREEN_SAMPLE = "1920x1080 で 0.1 秒・3840x2160 で 0.4 秒ほど（文字と顔の両方 手元の機械）"


def scene_bounds(cuts: Sequence[float], duration: float) -> list[tuple[float, float]]:
    """切り替わりの時刻から場面の（頭, 終わり）を並べる 素材の外の切り替わりは捨てる"""
    if duration <= 0:
        return [(0.0, 0.0)]
    inside = sorted({float(cut) for cut in cuts if 0.0 < float(cut) < duration})
    edges = [0.0, *inside, float(duration)]
    return list(itertools.pairwise(edges))


def _scene_times(start: float, end: float, per_scene: int) -> list[float]:
    length = end - start
    if length <= 0:
        return [start]
    edge = min(EDGE_SECONDS, length / 4)
    inner_start, inner_end = start + edge, end - edge
    count = max(1, min(per_scene, math.ceil((inner_end - inner_start) / MIN_STEP)))
    step = (inner_end - inner_start) / count
    # 区間を count 等分した真ん中 頭と終わりの寄りを避けたうえで、場面の中をまんべんなく見る
    return [inner_start + step * (index + 0.5) for index in range(count)]


def sample_times(
    cuts: Sequence[float],
    duration: float,
    per_scene: int,
    *,
    limit: int = MAX_SAMPLES,
) -> tuple[float, ...]:
    """調べる時刻（素材の原点から数えた秒）を時刻の順に返す

    ``cuts`` は場面の切り替わりの時刻（:meth:`~sashimono.engine.scenes.SceneSignal.cuts` の
    ``seconds``） 空なら素材全体を 1 つの場面と見る 長さの無い素材（静止画）は頭の 1 枚だけ
    """
    if duration <= 0:
        return (0.0,)
    per_scene = max(1, per_scene)
    scenes = scene_bounds(cuts, duration)
    times = [moment for start, end in scenes for moment in _scene_times(start, end, per_scene)]
    if len(times) > limit:
        # 場面ごとに 1 枚へ減らす 枚数を均等に削ると、短い場面が丸ごと抜け落ちる
        times = [(start + end) / 2 for start, end in scenes]
    if len(times) > limit:
        # それでも多ければ場面を等間隔に間引く 並びの頭だけを残すと、素材の後半を調べない
        picked = sorted({round(index * (len(times) - 1) / (limit - 1)) for index in range(limit)})
        times = [times[index] for index in picked] if limit > 1 else [times[len(times) // 2]]
    return tuple(times)


def fallback_cuts(duration: float) -> tuple[float, ...]:
    """場面が分からないときの区切り :data:`FALLBACK_STEP` 秒ごと

    区切りを置かずに 1 つの場面と見ると、長い素材でも場面 1 つ分（数枚）しか調べない
    区切りごとに 1 枚（``per_scene=1``）で使う 場面ごとの枚数で使うと、2 秒ごとに数枚になる
    """
    if duration <= FALLBACK_STEP:
        return ()
    count = math.ceil(duration / FALLBACK_STEP)
    return tuple(FALLBACK_STEP * index for index in range(1, count))
