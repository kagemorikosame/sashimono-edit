"""設定画面の「エンコーダを確かめ直す」（#464）

GPU の駆動が不調な間に確かめたエンコーダは、起動中ずっと使えない物として覚えている
駆動を入れ直して直っても、アプリを閉じるまで CPU で書き出し続けることになるので、
利用者が確かめ直せる口を設定画面に置く

確かめは別のプロセスで走り、子を起こすのに 0.2 秒ほど、壊れた駆動で固まれば
:data:`~sashimono.engine.encoders.PROBE_TIMEOUT_S` 秒まで待つ 画面のスレッドで待つと
その間設定の窓が固まるので、裏のスレッドで確かめてタイマーで拾う（声の一覧の読み込みと同じ形）
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer

from sashimono.engine.encode import available_video_codecs
from sashimono.engine.encoders import forget_probes, probe_troubles

__all__ = ["EncoderRecheck", "recheck_text"]

_LOG = logging.getLogger(__name__)

#: 終わったかを見る間（ミリ秒） 人が待ちを感じない短さで、画面の手を食わない長さ
_POLL_MS = 50


def recheck_text() -> str:
    """覚えている答えを捨てて確かめ直し、窓に出す文を返す 画面のスレッドの外で呼ぶ"""
    forget_probes()
    codecs = available_video_codecs()
    troubles = probe_troubles()
    parts = [f"使えるエンコーダ: {'、'.join(codecs) if codecs else 'なし'}"]
    if troubles:
        reasons = " / ".join(f"{name} は{reason}" for name, reason in troubles.items())
        parts.append(f"使えなかった物: {reasons} PC を再起動すると直ることがある")
    return " ".join(parts)


class EncoderRecheck(QObject):
    """1 度に 1 つだけ裏で確かめ直す 終わったら ``done`` を画面のスレッドで呼ぶ"""

    def __init__(self, parent: QObject, work: Callable[[], str] = recheck_text) -> None:
        super().__init__(parent)
        self._work = work
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._finished = threading.Event()
        self._result = ""
        self._done: Callable[[str], None] | None = None

    @property
    def running(self) -> bool:
        return self._done is not None

    def start(self, done: Callable[[str], None]) -> bool:
        """裏で確かめ直す 走っている間は新しく始めない（偽を返す）"""
        if self.running:
            return False
        self._finished.clear()
        self._done = done

        def run() -> None:
            try:
                self._result = self._work()
            except Exception:
                # 拾わないと「確かめています」のまま窓に残り、押し直しても始まらない
                _LOG.exception("エンコーダの確かめ直しで思っていない失敗が起きた")
                self._result = "エンコーダの確かめ直しで思っていない失敗が起きました"
            finally:
                self._finished.set()

        threading.Thread(target=run, name="sashimono-encoder-recheck", daemon=True).start()
        self._timer.start()
        return True

    def _poll(self) -> None:
        if not self._finished.is_set():
            return
        self._timer.stop()
        done, self._done = self._done, None
        if done is not None:
            done(self._result)
