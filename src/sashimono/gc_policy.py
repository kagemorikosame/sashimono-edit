"""ごみ集めの全体回（世代 2）を、手が止まったときに回す（#455 #442 B の後半）

Python は世代 1 の回が決まった数（既定 10 回）たまると、全体回を自分で走らせる 実プロジェクト
（1322 本・エフェクト 4018 個）では全体回が 1 回 36ms ほどあり、選ぶ・動かすの最中に当たると
1〜2 コマ止まる 開いたプロジェクトは :func:`gc.freeze` で外してある（#452）が、編集で
作り直したモデルは外れたまま積もる

手で回す形にすると、自動の全体回を止め（世代 2 の数を 0 にする）、手が止まったときに
:meth:`FullCollectPolicy.collect_if_due` で回す ずっと手が止まらない人のために、
:data:`MAX_DEFER_S` を過ぎたら操作の最中でも回す 回さないと、循環した物が溜まり続けて
メモリが増える

Qt に依存しない 手が止まったことを知らせるのは画面の側（:mod:`sashimono.ui.idle_collect`）
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["IDLE_S", "MAX_DEFER_S", "CollectRecord", "FullCollectPolicy"]

#: 最後の操作からこの長さ（秒）たったら手が止まったと見る 先読みの止めの解き（300ms）より
#: 長くする 短いと、次のクリックまでの短い間に全体回が当たり、手で回す意味が無くなる
IDLE_S = 1.5

#: 全体回をこれより長く（秒）は延ばさない 操作し続けている人でも、この間隔で 1 回は回す
#: 自動の全体回は編集の続く間なら数秒〜数十秒に 1 回なので、それより十分に長く取る
MAX_DEFER_S = 120.0


@dataclass(frozen=True, slots=True)
class CollectRecord:
    """手で回した全体回 1 回"""

    #: 回した長さ（ミリ秒）
    ms: float
    #: 集めた物の数
    collected: int
    #: 手が止まって回したか（偽なら延ばす上限を過ぎて回した）
    idle: bool


class FullCollectPolicy:
    """全体回を自動にするか手で回すかを持つ

    ``collect`` と ``clock`` は試験で差し替える 本物の全体回は試験の中の物まで集める
    """

    def __init__(
        self,
        *,
        collect: Callable[[int], int] = gc.collect,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._collect = collect
        self._clock = clock
        self._manual = False
        #: 手で回す前の世代 2 の数 自動へ戻すときにこれへ戻す
        self._saved_threshold: int | None = None
        self._last_full = clock()
        #: 手で回した記録 診断の欄と試験が読む 長く動かしても増えないよう直近だけ持つ
        self.history: list[CollectRecord] = []

    @property
    def manual(self) -> bool:
        return self._manual

    def set_manual(self, manual: bool) -> None:
        """手で回すかを切り替える 切り替えたときだけ gc の設定を書き換える"""
        if manual == self._manual:
            return
        first, second, third = gc.get_threshold()
        if manual:
            self._saved_threshold = third
            # 世代 2 の数を 0 にすると、自動の全体回だけが止まる 世代 0 と 1 は今まで通り回る
            # （短い命の物は今まで通りすぐ集まり、メモリの増え方は変わらない）
            gc.set_threshold(first, second, 0)
            self._last_full = self._clock()
        else:
            gc.set_threshold(first, second, self._saved_threshold or third or 10)
            self._saved_threshold = None
        self._manual = manual

    def due(self, *, idle: bool) -> bool:
        """全体回を回す時か 手が止まっていれば、世代 1 の回が 1 度でもあれば回す"""
        if not self._manual:
            return False
        if gc.get_count()[2] == 0 and self._clock() - self._last_full < MAX_DEFER_S:
            # 前の全体回から世代 1 の回が無い 集める物がほとんど無いので回さない
            return False
        return idle or self._clock() - self._last_full >= MAX_DEFER_S

    def collect_if_due(self, *, idle: bool) -> CollectRecord | None:
        """回す時なら全体回を回す 回したら記録を返す"""
        if not self.due(idle=idle):
            return None
        began = time.perf_counter()
        collected = self._collect(2)
        record = CollectRecord(
            ms=(time.perf_counter() - began) * 1000, collected=collected, idle=idle
        )
        self._last_full = self._clock()
        self.history = [*self.history[-15:], record]
        return record

    def close(self) -> None:
        """自動へ戻す 窓を閉じるときに呼ぶ 戻さないと、同じプロセスで次に開く窓も止まったまま"""
        self.set_manual(False)
