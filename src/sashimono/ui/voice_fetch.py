"""ナレーションのエンジンへの問い合わせを、画面のスレッドの外で進める（#335）

設定の窓とナレーションの窓が、声の一覧と規約をエンジンから読む エンジンが返事を止めると、
画面のスレッドで読んだ間は通信の待ちの上限まで窓が固まる 裏のスレッドで読み、終わったら
タイマーで拾って画面のスレッドで窓を直す（字幕起こしや解析と同じ「積んでタイマーで拾う」形）
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer

from sashimono.engine.audio.voicevox import Catalog, EngineError, Speaker, Style, VoiceEngine

__all__ = ["BUSY_TEXT", "Fetched", "VoiceFetch", "fetch_voices", "terms_text"]

#: 前の問い合わせの途中で押されたときに窓へ出す文 黙って捨てると押したことが効いたように見える
BUSY_TEXT = "前の問い合わせが終わっていません 終わってからもう 1 度押してください"

_LOG = logging.getLogger(__name__)

#: 終わったかを見る間（ミリ秒） 人が待ちを感じない短さで、画面の手を食わない長さ
_POLL_MS = 50


@dataclass(frozen=True, slots=True)
class Fetched:
    """読んだ物 失敗なら ``error`` に断りの文"""

    catalog: Catalog | None = None
    #: 頼んだ声の規約（頼まなければ ``None``）
    policy: str | None = None
    error: str = ""


def fetch_voices(url: str, known: Catalog | None, voice: int | None) -> Callable[[], Fetched]:
    """裏で動かす問い合わせ ``voice`` を渡すとその声の規約も読む

    ``known`` があれば一覧は読み直さない

    場所の確かめ（自分の機械の中か）はここで済ませる 受け付けない場所は :class:`EngineError`
    """
    engine = VoiceEngine(url)

    def work() -> Fetched:
        catalog = known if known is not None else engine.catalog()
        if voice is None:
            return Fetched(catalog=catalog)
        found = catalog.find(voice)
        policy = ""
        if found is not None:
            try:
                policy = engine.policy(found[0])
            except EngineError:
                # 声の規約が読めなくても、エンジンの規約と表記は見せる（どちらも本人が確かめる物）
                policy = ""
        return Fetched(catalog=catalog, policy=policy)

    return work


def terms_text(catalog: Catalog, speaker: Speaker, style: Style, policy: str) -> str:
    """規約を見せる文 声の規約とエンジンの規約の両方を、エンジンが返したまま並べる

    声の規約が空でもエンジンの規約は見せる（VOICEVOX はエンジンの利用規約も表記を求める）
    どちらも無いときだけ、エンジンのアプリで確かめるよう案内する アプリは中身を判断しない
    """
    parts = [f"{speaker.name}（{style.name}） 表記: {catalog.credit(speaker)}"]
    if policy:
        parts.append(f"声の規約:\n{policy}")
    if catalog.terms:
        parts.append(f"エンジン（{catalog.engine_name or catalog.brand}）の規約:\n{catalog.terms}")
    if not policy and not catalog.terms:
        parts.append("エンジンが規約の文を返しませんでした エンジンのアプリで確かめてください")
    return "\n\n".join(parts)


class VoiceFetch(QObject):
    """1 度に 1 つだけ裏で問い合わせる 終わったら ``done`` を画面のスレッドで呼ぶ"""

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._finished = threading.Event()
        self._result = Fetched()
        self._done: Callable[[Fetched], None] | None = None

    @property
    def running(self) -> bool:
        return self._done is not None

    def start(self, work: Callable[[], Fetched], done: Callable[[Fetched], None]) -> bool:
        """``work`` を裏で動かす 走っている間は新しく始めない（偽を返す）"""
        if self.running:
            return False
        self._finished.clear()
        self._done = done

        def run() -> None:
            try:
                self._result = work()
            except EngineError as exc:
                self._result = Fetched(error=str(exc))
            except Exception:
                # 思っていない失敗も窓に理由を出して終える 拾わないと「読んでいます」のまま残る
                _LOG.exception("ナレーションのエンジンへの問い合わせで思っていない失敗が起きた")
                self._result = Fetched(error="エンジンへの問い合わせで思っていない失敗が起きました")
            finally:
                self._finished.set()

        threading.Thread(target=run, name="sashimono-voice-fetch", daemon=True).start()
        self._timer.start()
        return True

    def _poll(self) -> None:
        if not self._finished.is_set():
            return
        self._timer.stop()
        done, self._done = self._done, None
        if done is not None:
            done(self._result)
