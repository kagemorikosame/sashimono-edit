"""裏で走る処理（控え・波形・サムネイル）の進み具合を数える

数えるのはワーカースレッド、読むのは画面のスレッドのタイマー 画面の部品は
ワーカースレッドから触ると Qt が落ちるので、ここでは数を持つだけにして、
画面の側が決まった間隔で :meth:`JobBoard.poll` で読みに来る ひと続きが終わったと
決めるのも画面の側（:meth:`JobBoard.settle`） 控えと解析の 2 つの数え板を並べて
出すので、片方だけが自分の都合で数え直すと、もう片方が走っている間に失敗の数が消える

Qt に依存しない 裏の処理の係（:class:`~sashimono.engine.cache.proxy.ProxyBuilder`
と :class:`~sashimono.engine.cache.analyzer.MediaAnalyzer`）が持つ
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field

from sashimono.core.model import MediaId

__all__ = ["MIN_MEASURE_S", "JobBoard", "ProgressSnapshot"]

#: 残り時間の見込みを出し始めるまでに測る長さ（秒） 走り始めの数秒は、素材を開く分と
#: エンコーダを起こす分が混ざって速さが低く出る そのまま出すと、見込みが何倍にも長く出る
MIN_MEASURE_S = 3.0


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """ある時点の進み具合 画面へ出すためだけに使う"""

    #: ひと続きの仕事として頼まれた数 前のひと続きを画面が見終えると 0 から数え直す
    total: int = 0
    #: そのうち終わった数（失敗も数える）
    finished: int = 0
    #: そのうち失敗した数
    failed: int = 0
    #: 数え板が変わるたびに増える印 :meth:`JobBoard.settle` が、読んだ後に
    #: 何も変わっていないかを確かめるのに使う
    stamp: int = 0
    #: 走っている仕事の途中の分の合計 解析は 1 つの素材に 2 つの仕事があるので、
    #: 素材ごとの平均（``running``）から足し直すと全体の割合がずれる
    partial: float = 0.0
    #: 走っている（か待っている）素材ごとの進み具合 0..1
    running: Mapping[MediaId, float] = field(default_factory=dict)
    #: 失敗した素材と理由 作り直しに成功するか素材を外すまで残す
    #: 行ごとの表示は、ひと続きが終わった後も失敗を出し続けたいので数え直しで消さない
    failures: Mapping[MediaId, str] = field(default_factory=dict)
    #: このひと続きで失敗した理由 終わった順 数え直しで消える
    #: 終わりの知らせはこちらを使う ``failures`` から取ると、前のひと続きで失敗した
    #: 別の素材の理由が、今の失敗の数と並んで出ることがある
    recent_failures: tuple[str, ...] = ()
    #: 残りを終えるまでのおよその秒数（#441） 重さ（控えなら素材の長さ）を添えて頼んだ
    #: 仕事について、ここまでの速さから出す 速さがまだ分からないときは ``None``
    remaining_seconds: float | None = None

    @property
    def remaining(self) -> int:
        """まだ終わっていない本数（走っている物と待っている物）"""
        return max(0, self.total - self.finished)

    @property
    def busy(self) -> bool:
        return self.finished < self.total

    @property
    def fraction(self) -> float:
        """全体の進み具合 0..1 走っている物の途中の分も足す

        終わった本数だけで数えると、長い素材 1 本の控えを作る間ずっと 0% のまま動かない
        """
        if self.total <= 0:
            return 1.0
        return min(1.0, (self.finished + self.partial) / self.total)


class JobBoard:
    """仕事ごとの進み具合と、ひと続きの本数を数える

    仕事は鍵（``Hashable``）で区別し、どの素材の仕事かも一緒に持つ 解析は 1 つの
    素材に波形とサムネイルの 2 つの仕事があるので、素材と鍵を分けておかないと
    行ごとの表示で片方が片方を上書きする

    本数は数を足し引きするのではなく、走っている仕事と終わった仕事の表から数える
    素材を外したときに、その素材が足した分だけをきれいに引けるようにするため
    （数だけを持つと、外した素材の失敗が理由の無いまま数に残る）
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._active: dict[Hashable, tuple[MediaId, float]] = {}
        #: 仕事の重さ（控えなら素材の秒数） 残り時間を出すのに使う 添えなかった仕事は持たない
        self._weights: dict[Hashable, float] = {}
        #: 走り始めた時刻（最初に進み具合が来た時刻） 待っている間は速さに数えない
        self._began: dict[Hashable, float] = {}
        #: ここまでに終えた重さと、それに掛かった秒数 ひと続きを数え直しても残す
        #: 速さはこの機械とこの種類の仕事の物で、次のひと続きの見込みにもそのまま使える
        self._done_weight = 0.0
        self._done_seconds = 0.0
        #: このひと続きで終わった仕事 鍵 → （素材、失敗の理由 成功なら ``None``）
        self._done: dict[Hashable, tuple[MediaId, str | None]] = {}
        #: 失敗は仕事の鍵ごとに持つ 素材ごとに持つと、波形が失敗した素材で
        #: サムネイルを頼み直しただけで波形の失敗まで消える
        #: ひと続きを数え直しても残す 行の表示は、終わった後も失敗を出し続けたい
        self._failures: dict[Hashable, tuple[MediaId, str]] = {}
        self._stamp = 0

    def start(self, key: Hashable, media_id: MediaId, *, weight: float | None = None) -> None:
        """仕事を 1 つ数え始める 同じ鍵が走っていれば何もしない

        ``weight`` は仕事の重さ（控えなら素材の秒数） 添えると残り時間の見込みに入る
        """
        with self._lock:
            if key in self._active:
                return
            self._active[key] = (media_id, 0.0)
            if weight is not None and weight > 0:
                self._weights[key] = weight
            # 同じ仕事を頼み直したら、前の回の終わりは数えない 両方数えると
            # 「1 本のうち 2 本」になる
            self._done.pop(key, None)
            # 作り直しを頼んだ仕事の前の失敗は消す 残すと、作り直している最中も
            # 「作れなかった」と出続ける
            self._failures.pop(key, None)
            self._stamp += 1

    def begin(self, key: Hashable) -> None:
        """待ちから出て、実際に取り掛かった 速さはここから測る

        最初の進み具合から測ると、子のプロセスの起動とエンコーダの用意の分が抜ける
        短い素材が多いと、その分だけ残り時間を短く見積もる 呼ばない仕事は最初の
        :meth:`report` から測る
        """
        with self._lock:
            if key in self._active:
                self._began.setdefault(key, self._clock())

    def report(self, key: Hashable, value: float) -> None:
        with self._lock:
            entry = self._active.get(key)
            if entry is not None:
                self._active[key] = (entry[0], max(0.0, min(1.0, value)))
                self._began.setdefault(key, self._clock())

    def finish(self, key: Hashable, failure: str | None = None) -> None:
        """仕事が終わった ``failure`` があれば失敗として数える

        数え板に無い仕事（外した素材の仕事）は何もしない
        """
        with self._lock:
            entry = self._active.pop(key, None)
            if entry is None:
                return
            weight = self._weights.pop(key, None)
            began = self._began.pop(key, None)
            if failure is None and weight is not None and began is not None:
                # 失敗した仕事は速さに数えない すぐ落ちた物を数えると、見込みが速すぎる
                self._done_weight += weight
                self._done_seconds += max(0.0, self._clock() - began)
            self._done[key] = (entry[0], failure)
            if failure is not None:
                self._failures[key] = (entry[0], failure)
            self._stamp += 1

    def drop(self, key: Hashable) -> None:
        """止めた仕事を数から外す

        終わった数に入れると、素材を外しただけで「3 本のうち 3 本」と
        仕事をしたように見え、失敗に入れると外しただけで失敗が出る
        """
        with self._lock:
            self._weights.pop(key, None)
            self._began.pop(key, None)
            if self._active.pop(key, None) is not None:
                self._stamp += 1

    def forget(self, media_id: MediaId) -> None:
        """外した素材の分を、走っている物・終わった物・失敗のすべてから引く

        理由だけを消して数を残すと、「失敗 1 件」と出るのに、どの素材の
        何が失敗したのかがどこにも出ない 走っている物も外す 止まるまで
        待つと、その間は一覧に無い素材の割合が全体に混ざる
        """
        with self._lock:
            for table in (self._active, self._done, self._failures):
                owned = [key for key, entry in table.items() if entry[0] == media_id]
                for key in owned:
                    del table[key]
                    self._weights.pop(key, None)
                    self._began.pop(key, None)
            self._stamp += 1

    def poll(self) -> ProgressSnapshot:
        """今の進み具合を返す 数え直さない（数え直すのは :meth:`settle`）"""
        with self._lock:
            running: dict[MediaId, list[float]] = {}
            for media_id, value in self._active.values():
                running.setdefault(media_id, []).append(value)
            failures: dict[MediaId, list[str]] = {}
            for media_id, reason in self._failures.values():
                failures.setdefault(media_id, []).append(reason)
            return ProgressSnapshot(
                total=len(self._active) + len(self._done),
                finished=len(self._done),
                failed=sum(reason is not None for _media, reason in self._done.values()),
                stamp=self._stamp,
                partial=sum(value for _media, value in self._active.values()),
                running={key: sum(values) / len(values) for key, values in running.items()},
                recent_failures=tuple(
                    reason for _media, reason in self._done.values() if reason is not None
                ),
                failures={key: "\n".join(reasons) for key, reasons in failures.items()},
                remaining_seconds=self._remaining_seconds(),
            )

    def _remaining_seconds(self) -> float | None:
        """残りの重さを、ここまでの速さで割った秒数 ``_lock`` を持って呼ぶ

        速さは終えた仕事と走っている仕事の両方から出す 1 本目の長い素材の途中でも見込みが
        出るように、走っている分も数える 測った長さが :data:`MIN_MEASURE_S` に満たないうちは
        速さがまだ揺れるので出さない
        """
        if not self._weights:
            return None
        now = self._clock()
        weight = self._done_weight
        seconds = self._done_seconds
        left = 0.0
        for key, total in self._weights.items():
            entry = self._active.get(key)
            value = entry[1] if entry is not None else 0.0
            began = self._began.get(key)
            if began is not None:
                weight += total * value
                seconds += max(0.0, now - began)
            left += total * (1.0 - value)
        if seconds < MIN_MEASURE_S or weight <= 0:
            return None
        return left / (weight / seconds)

    def settle(self, seen: ProgressSnapshot) -> bool:
        """画面が ``seen`` で「終わった」を見届けたので、ひと続きの数を 0 へ戻す

        ``seen`` を読んだ後に何か動いていたら戻さない（偽を返す） 間に始まって
        終わった仕事を、画面が 1 度も見ないまま消さないため
        画面の側は、控えと解析の両方が止まっているときだけ呼ぶ 片方ずつ戻すと、
        もう片方が走っている間に失敗の数と全体の割合が巻き戻る
        """
        with self._lock:
            if self._active or self._stamp != seen.stamp:
                return False
            self._done.clear()
            self._stamp += 1
            return True
