"""測った音から、掛ける量とリミッターの上限を決める（#494）

AI の ``normalize_loudness`` と人の〔音量を基準にそろえる…〕が同じここを通る

重なって鳴る音（声と BGM など）は、抑えて下がる分を見込むと基準へ届くが、抑えた音どうしの和が
0 dBFS を越えて割れる（実物の話し声に合成の BGM で 1.097） そこで組ごとに測った手がかり
（:class:`~sashimono.engine.checks.loudness.LoudnessLane`）から、和の越えない所までリミッターの
上限を一緒に下げる 下げた分も、本人の設定の抑える深さに数える
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from sashimono.ai.auto_fix import clips_overlap, has_limiter, limited_peak, loudness_predictor
from sashimono.core.mixing import LoudnessGain, loudness_gain
from sashimono.core.model import Clip, Track
from sashimono.engine.checks.loudness import Dynamics, Headroom
from sashimono.engine.checks.report import CheckReport

__all__ = ["LoudnessPlan", "plan_loudness"]

#: 上限に収めると基準へ届かないときに、代わりに和を収める所（dBFS） 割れだけは作らない
_NO_CLIP_DB = -0.1
#: 届いたと読む差（LU）
_REACHED_LU = 0.05
#: 深さに収まる一番大きい量を探す二分の回数
_SEARCH_STEPS = 12


@dataclass(frozen=True, slots=True)
class LoudnessPlan:
    """掛ける量と、リミッターの置き方"""

    decided: LoudnessGain
    #: 重なる分の余裕として、リミッターの上限を下げる量（dB）
    lowered: float = 0.0
    #: 重なって鳴る音の和が越えないと見積もった所（dBFS） 見積もれなければ ``None``
    held_db: float | None = None
    #: 重なって鳴るのに組に分けて測れなかったか リミッターの効いた後の音量を言い切れない
    unchecked: bool = False


def plan_loudness(
    report: CheckReport,
    targets: Sequence[tuple[Track, Clip]],
    *,
    target: float,
    ceiling: float,
    squeeze: float,
    raise_ceiling: float | None,
) -> LoudnessPlan:
    """測り終えた ``report`` から決める

    重なって鳴るときは、上限に収めて届く形、割れずに届く形、割れずに一番大きい形の順に取る
    どれも上限を下げずに掛けた音より小さくなるなら下げない（届かない分と割れを知らせるに留める）
    """
    assert report.loudness is not None
    measured, peak = report.loudness, report.peak_db
    clips = [clip for _, clip in targets]
    plain = loudness_gain(
        clips,
        measured=measured,
        peak_db=peak,
        target=target,
        ceiling=ceiling,
        squeeze=squeeze,
        raise_ceiling=raise_ceiling,
        predict=loudness_predictor(report, ceiling, clips),
    )
    if not clips_overlap(clips) or peak is None or squeeze <= 0:
        return LoudnessPlan(plain)
    dynamics = report.dynamics
    if dynamics is None or not dynamics.lanes:
        return LoudnessPlan(plain, unchecked=True)
    loudest_lane = dynamics.loudest_lane_db()
    if loudest_lane > peak and (plain.squeeze_db > 0 or any(has_limiter(c) for c in clips)):
        # リミッターに当たるのは組の山 逆相で重なると混ぜた山より高く、混ぜた山で深さを数えると
        # 本人の設定より深く潰す
        peak = loudest_lane
        plain = loudness_gain(
            clips,
            measured=measured,
            peak_db=peak,
            target=target,
            ceiling=ceiling,
            squeeze=squeeze,
            raise_ceiling=raise_ceiling,
        )
    if (
        plain.squeeze_db <= 0
        or raise_ceiling is not None
        # トラックの音量でリミッターの上限の範囲を越えると、組の山が止まる所を言えない
        or limited_peak(targets, ceiling) > ceiling + _REACHED_LU
    ):
        return LoudnessPlan(plain)
    loudest = dynamics.limited_loudness(plain.gain_db, ceiling)
    if loudest is None:
        return LoudnessPlan(plain)
    plain = replace(plain, limited_lufs=loudest)
    if loudest < target - _REACHED_LU and "peak" not in plain.limited_by:
        plain = replace(plain, limited_by=(*plain.limited_by, "peak"))
    chosen = LoudnessPlan(plain)
    headroom = Headroom(dynamics.lanes)
    for top in dict.fromkeys((min(ceiling, _NO_CLIP_DB), _NO_CLIP_DB)):
        found = _within(
            dynamics,
            headroom,
            targets,
            measured=measured,
            peak=peak,
            target=target,
            ceiling=ceiling,
            squeeze=squeeze,
            top=top,
        )
        if found is None or found.decided.limited_lufs is None:
            continue
        after = found.decided.limited_lufs
        if after >= target - _REACHED_LU:
            return found
        if after >= loudest:
            chosen, loudest = found, after
    return chosen


def _within(
    dynamics: Dynamics,
    headroom: Headroom,
    targets: Sequence[tuple[Track, Clip]],
    *,
    measured: float,
    peak: float,
    target: float,
    ceiling: float,
    squeeze: float,
    top: float,
) -> LoudnessPlan | None:
    """和を ``top`` に収めたまま、深さの中で基準へ寄せる形"""
    known: dict[float, float] = {}

    def lowered(gain: float) -> float:
        if gain not in known:
            # 0.01 dB へ切り上げる 丸めで上へ外すと和が越える
            short = ceiling - headroom.ceiling(gain, top)
            known[gain] = math.ceil(short * 100.0) / 100.0 if short > 0 else 0.0
        return known[gain]

    def depth(gain: float) -> float:
        return peak + gain - ceiling + lowered(gain)

    most = ceiling + squeeze - peak
    if depth(most) > squeeze:
        low, high = min(target - measured, most), most
        if depth(low) > squeeze:
            return None
        for _ in range(_SEARCH_STEPS):
            middle = (low + high) / 2.0
            if depth(middle) > squeeze:
                high = middle
            else:
                low = middle
        most = low
    room = peak + most - ceiling
    if room <= 0:
        return None
    decided = loudness_gain(
        [clip for _, clip in targets],
        measured=measured,
        peak_db=peak,
        target=target,
        ceiling=ceiling,
        squeeze=room,
        predict=lambda gain: dynamics.limited_loudness(gain, ceiling - lowered(gain)),
    )
    drop = lowered(decided.gain_db)
    if limited_peak(targets, ceiling - drop) > ceiling - drop + _REACHED_LU:
        # 下げた上限がリミッターの欄の範囲に入らない
        return None
    after = dynamics.limited_loudness(decided.gain_db, ceiling - drop)
    limited = decided.limited_by
    if after is not None and after < target - _REACHED_LU and "peak" not in limited:
        limited = (*limited, "peak")
    return LoudnessPlan(
        LoudnessGain(decided.gain_db, limited, max(0.0, depth(decided.gain_db)), after), drop, top
    )
