"""測った音から、掛ける量とリミッターの上限を決める（#494）

AI の ``normalize_loudness`` と人の〔音量を基準にそろえる…〕が同じここを通る

重なって鳴る音（声と BGM など）は、抑えて下がる分を見込むと基準へ届くが、抑えた音どうしの和が
0 dBFS を越えて割れる（実物の話し声に合成の BGM で 1.097） そこで組ごとに測った手がかり
（:class:`~sashimono.engine.checks.loudness.LoudnessLane`）から、和の越えない所までリミッターの
上限を一緒に下げる 下げた分も、本人の設定の抑える深さに数える

計算は作品の長さに比例して重い（3 時間分で 1.6 秒、重なると 3.6 秒） 画面のスレッドで回すと
その間は描き直しも中断も効かないので、:class:`PlanJob` で裏のスレッドへ出す（#497）

リミッターはトラックの音量と定位の前に掛かる 深さと抑えた後の音量は、その所の山と、クリップごとに
実際に置ける上限で数える 混ぜた後の音で数えると、音量 -20 dB のトラックが -20 dBFS で潰れているのに
届いたと答える
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from sashimono.ai.auto_fix import (
    clips_overlap,
    has_limiter,
    limited_peak,
    loudness_predictor,
    measures_apart,
)
from sashimono.core.mixing import LoudnessGain, loudness_gain
from sashimono.core.model import Clip, Track
from sashimono.engine.checks.loudness import Dynamics, Headroom
from sashimono.engine.checks.report import CheckReport

__all__ = ["LoudnessPlan", "PlanJob", "plan_loudness"]

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
    #: 分けて測る形（重なり・定位・トラックの音量）なのに測れなかったか 抑えた後の音量を言い切れない
    unchecked: bool = False


class PlanJob:
    """:func:`plan_loudness` を裏のスレッドで回す（#497）

    重い所は numpy が GIL を離すので、回している間も画面のスレッドは 10 ms おきの合図を
    3 ms の遅れで受ける ``done`` は終わったとき（失敗も）に裏のスレッドから呼ぶ 画面へは
    Qt の知らせで渡す（ここは画面の部品を知らない）
    """

    def __init__(
        self, work: Callable[[], LoudnessPlan], done: Callable[[], None] | None = None
    ) -> None:
        self._finished = threading.Event()
        self._plan: LoudnessPlan | None = None
        self._error: Exception | None = None
        self._done = done
        thread = threading.Thread(
            target=self._run, args=(work,), name="sashimono-loudness-plan", daemon=True
        )
        thread.start()

    def _run(self, work: Callable[[], LoudnessPlan]) -> None:
        try:
            self._plan = work()
        except Exception as error:
            # ここで落とすと、待つ側は終わった印を受けられず「見積もっています」のまま残る
            self._error = error
        finally:
            self._finished.set()
            if self._done is not None:
                self._done()

    def wait(self, seconds: float) -> bool:
        """終わるまで最長 ``seconds`` 秒待ち、終わっていれば真 どのスレッドから呼んでもよい"""
        return self._finished.wait(seconds)

    def take(self) -> LoudnessPlan | None:
        """終わっていれば計画を返す（失敗はここで投げ直す） まだなら ``None``"""
        if not self._finished.is_set():
            return None
        if self._error is not None:
            raise self._error
        assert self._plan is not None
        return self._plan


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
    抑えて掛けた音が、抑えずに掛けた音より小さくなるなら抑えない
    """
    assert report.loudness is not None
    measured, peak = report.loudness, report.peak_db
    clips = [clip for _, clip in targets]
    apart = measures_apart(targets, ceiling)
    plain = loudness_gain(
        clips,
        measured=measured,
        peak_db=peak,
        target=target,
        ceiling=ceiling,
        squeeze=squeeze,
        raise_ceiling=raise_ceiling,
        predict=None if apart else loudness_predictor(report, ceiling, clips),
    )
    if not apart or peak is None or squeeze <= 0:
        return LoudnessPlan(plain)
    dynamics = report.dynamics
    if dynamics is None or not dynamics.lanes:
        return LoudnessPlan(plain, unchecked=True)
    installed = any(has_limiter(clip) for clip in clips)
    if plain.squeeze_db <= 0 and not installed:
        # 混ぜた山が上限に当たらない リミッターを積まないので、掛けた量どおりに動く
        return LoudnessPlan(plain)
    plan = _limited(
        dynamics,
        targets,
        measured=measured,
        # 深さは、混ぜた山と、一番深く絞るリミッターの高い方で数える
        peak=max(peak, ceiling + dynamics.deepest_db(ceiling)),
        target=target,
        ceiling=ceiling,
        squeeze=squeeze,
        raise_ceiling=raise_ceiling,
    )
    if installed:
        return plan
    free = loudness_gain(
        clips,
        measured=measured,
        peak_db=peak,
        target=target,
        ceiling=ceiling,
        raise_ceiling=raise_ceiling,
    )
    after = plan.decided.limited_lufs
    if after is None:
        after = measured + plan.decided.gain_db
    # 低いトラックの上限は 0 dBFS で頭打ちになり、深さの中では上げられないことがある
    # 抑えて下げるより、抑えずに山の手前まで上げる
    return LoudnessPlan(free) if after < measured + free.gain_db else plan


def _limited(
    dynamics: Dynamics,
    targets: Sequence[tuple[Track, Clip]],
    *,
    measured: float,
    peak: float,
    target: float,
    ceiling: float,
    squeeze: float,
    raise_ceiling: float | None,
) -> LoudnessPlan:
    """リミッターを積む（か、もう積んである）ときの形 ``peak`` は深さを数える山"""
    clips = [clip for _, clip in targets]
    overlap = clips_overlap(clips)
    plain = loudness_gain(
        clips,
        measured=measured,
        peak_db=peak,
        target=target,
        ceiling=ceiling,
        squeeze=squeeze,
        raise_ceiling=raise_ceiling,
        # 重ならなければ上限を下げる要が無いので、ここで見込んで掛ける
        predict=None if overlap else (lambda gain: dynamics.limited_loudness(gain, ceiling)),
    )
    if plain.squeeze_db <= 0 or raise_ceiling is not None or not overlap:
        return LoudnessPlan(plain)
    loudest = dynamics.limited_loudness(plain.gain_db, ceiling)
    if loudest is None:
        return LoudnessPlan(plain)
    plain = replace(plain, limited_lufs=loudest)
    if loudest < target - _REACHED_LU and "peak" not in plain.limited_by:
        plain = replace(plain, limited_by=(*plain.limited_by, "peak"))
    chosen = LoudnessPlan(plain)
    if limited_peak(targets, ceiling) > ceiling + _REACHED_LU:
        # トラックの音量でリミッターの上限の範囲を越えると、組の山が止まる所を言えない
        return chosen
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
