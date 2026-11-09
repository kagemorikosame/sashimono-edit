"""手が止まったら、ごみ集めの全体回を回す（#455 :mod:`sashimono.gc_policy` の画面の側）

操作（:class:`~sashimono.ui.main_window._InputHold` が見る物）のたびに :meth:`IdleCollect.poke`
を呼び、:data:`~sashimono.gc_policy.IDLE_S` 操作が無ければ全体回を回す 再生中は回さない
（再生は手が止まっていても画面のスレッドがコマを出し続けるので、回すとそのコマが遅れる）
延ばす上限（:data:`~sashimono.gc_policy.MAX_DEFER_S`）は別のタイマーで見る
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer

from sashimono.gc_policy import IDLE_S, MAX_DEFER_S, FullCollectPolicy

__all__ = ["IdleCollect"]

#: 延ばす上限を過ぎていないかを見る間隔（ミリ秒） 上限より十分に短ければよい
BACKSTOP_MS = 15_000


class IdleCollect(QObject):
    """手で回す設定のときだけ動く 切っている間はタイマーも止める"""

    def __init__(
        self,
        busy: Callable[[], bool],
        parent: QObject | None = None,
        *,
        policy: FullCollectPolicy | None = None,
    ) -> None:
        """``busy`` が真の間（再生中）は、手が止まっていても回さない"""
        super().__init__(parent)
        self._busy = busy
        self.policy = policy if policy is not None else FullCollectPolicy()
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(round(IDLE_S * 1000))
        self._idle.timeout.connect(self._on_idle)
        self._backstop = QTimer(self)
        self._backstop.setInterval(min(BACKSTOP_MS, round(MAX_DEFER_S * 1000)))
        self._backstop.timeout.connect(self._on_backstop)

    def set_manual(self, manual: bool) -> None:
        self.policy.set_manual(manual)
        if manual:
            self._backstop.start()
            self._idle.start()
        else:
            self._idle.stop()
            self._backstop.stop()

    def poke(self) -> None:
        """操作が来た 手が止まるまで待ち直す"""
        if self.policy.manual:
            self._idle.start()

    def close(self) -> None:
        self._idle.stop()
        self._backstop.stop()
        self.policy.close()

    def _on_idle(self) -> None:
        if self._busy():
            # 再生中 止まるまで同じ間隔で見直す
            self._idle.start()
            return
        self.policy.collect_if_due(idle=True)

    def _on_backstop(self) -> None:
        self.policy.collect_if_due(idle=False)
