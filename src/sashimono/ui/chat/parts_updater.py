"""AI の部品の自動の更新を、画面を固めずに回す

確かめるのも入れるのも裏のスレッド 結果はこの部品の時計で拾い、:attr:`PartsUpdater.finished`
で知らせる 入れ替えは AI が応えていない間だけにする（応えている途中で Claude Code を
入れ替えると、その応答が途中で切れる） いつ暇かはパネルが :meth:`PartsUpdater.tick` で伝える

確かめ・入れ方・入れ替え方は :mod:`sashimono.ai.parts_update` を見る
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from sashimono.ai.parts_update import (
    FAILURE_NOTICE_COUNT,
    Lookup,
    PartsStateStore,
    find_update,
    install_staged,
    is_due,
)
from sashimono.package_index import latest_release
from sashimono.runtime import FeaturePack, refresh_runtime, runtime_target_dir

__all__ = ["PartsUpdater", "Phase"]

#: 起動からこの長さだけ待ってから確かめる（ミリ秒） 起動の直後は素材の読み込みなどで
#: 忙しく、同じ時に裏で pip を走らせると起動が遅く見える
STARTUP_DELAY_MS = 60_000

#: 裏の作業の終わりを見る間隔（ミリ秒）
POLL_MS = 200

#: 入れ替えたら読み込み直す部品 古い方が読み込まれたままだと、新しい Claude Code に
#: 古い SDK の決まりで話しかける 拡張モジュールを持つ部品（pydantic-core など）は
#: 読み込み直せないので含めない（版が変わったら再起動まで古いまま動く）
_RELOADABLE = ("claude_agent_sdk",)

#: 入れる物 試験で偽の pip に差し替える
Installer = Callable[..., bool]


class Phase(Enum):
    IDLE = "idle"
    #: PyPI に尋ねている
    CHECKING = "checking"
    #: 新しい版が見つかり、AI が応え終わるのを待っている
    PENDING = "pending"
    #: 入れている
    INSTALLING = "installing"


class PartsUpdater(QObject):
    """AI の部品を裏で新しくする 配布版でだけ動く（開発の環境は触らない）"""

    #: 入れ替えが終わった（または頼まれた更新が要らない・できないと分かった）
    #: 引数は成功したかと、入れた物（``claude-agent-sdk==0.2.164`` など 無ければ空）
    finished = Signal(bool, str)

    def __init__(
        self,
        pack: FeaturePack,
        parent: QObject | None = None,
        *,
        store: PartsStateStore | None = None,
        clock: Callable[[], float] = time.time,
        lookup: Lookup = latest_release,
        installer: Installer = install_staged,
        target: Callable[[], Path | None] = runtime_target_dir,
    ) -> None:
        super().__init__(parent)
        self._pack = pack
        self._store = store if store is not None else PartsStateStore()
        self._clock = clock
        self._lookup = lookup
        self._installer = installer
        self._target = target
        self._enabled = True
        self.phase = Phase.IDLE
        #: 頼まれた更新（版が足りないと断られた）か 頼まれたときは、要らない・できないと
        #: 分かった時点でも知らせる（待っている指示を送り直すか、案内を出すかを決めるため）
        self._asked = False
        self._pins: tuple[str, ...] = ()
        self._work: threading.Event | None = None
        #: 裏の作業の答え 作業ごとに新しい入れ物を渡し、前の作業の答えと混ざらないようにする
        self._results: list[tuple[str, ...]] = []
        self._install_result: list[bool] = []
        self._failures = self._store.load().failures
        #: 入れ替える直前にパネルが走らせる物 会話を畳み、畳み終わるのを待つ物を返す
        self.prepare: Callable[[], Callable[[], None]] | None = None

        self._start_timer = QTimer(self)
        self._start_timer.setSingleShot(True)
        self._start_timer.setInterval(STARTUP_DELAY_MS)
        self._start_timer.timeout.connect(lambda: self._check(asked=False))
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._poll)

    # --- パネルから ---

    @property
    def available(self) -> bool:
        """自動で入れられるか 設定で切っていない配布版だけ"""
        return self._enabled and self._target() is not None

    @property
    def installing(self) -> bool:
        return self.phase is Phase.INSTALLING

    @property
    def struggling(self) -> bool:
        """続けて失敗している 状態の行に小さく出す"""
        return self._failures >= FAILURE_NOTICE_COUNT

    def set_enabled(self, enabled: bool) -> None:
        """設定の入／切 切ったら確かめも入れ替えもしない（入れている最中の物は最後まで）"""
        self._enabled = enabled
        if not enabled:
            self._start_timer.stop()
            if self.phase is Phase.PENDING:
                self._abandon()

    def schedule(self) -> None:
        """起動から少し後に確かめる"""
        if self.available:
            self._start_timer.start()

    def request_now(self) -> bool:
        """版が足りないと断られた 今すぐ確かめて入れる 始められなければ偽"""
        if not self.available:
            return False
        self._asked = True
        if self.phase is Phase.IDLE:
            self._check(asked=True)
        return True

    def tick(self, idle: bool) -> None:
        """パネルの見張りから呼ぶ ``idle`` は AI が応えていないか"""
        if self.phase is Phase.PENDING and idle:
            if not self.available:
                self._abandon()
                return
            self._install()

    def _abandon(self) -> None:
        """入れずにやめる 頼まれた更新なら、できなかったと知らせる

        知らせずにやめると、パネルは送り直しを待ったまま、持っている指示をいつまでも
        会話へ渡さず、状態の行も「更新しています…」のまま残る
        """
        self.phase = Phase.IDLE
        if self._asked:
            self._asked = False
            self.finished.emit(False, "")

    def stop(self) -> None:
        """窓を閉じる 時計を止める（裏の pip は daemon なので、プロセスと一緒に終わる）"""
        self._start_timer.stop()
        self._poll_timer.stop()

    # --- 裏の作業 ---

    def _check(self, *, asked: bool) -> None:
        if not self.available or self.phase is not Phase.IDLE:
            if asked and self.phase is Phase.IDLE:
                self._abandon()
            return
        state = self._store.load()
        now = self._clock()
        if not asked and not is_due(state.last_checked, now):
            return
        self._store.save(replace(state, last_checked=now))
        done = threading.Event()
        found: list[tuple[str, ...]] = []

        def run() -> None:
            # 終わった印は finally で立てる 立たないと、確かめている途中のまま止まる
            try:
                found.append(find_update(self._pack, self._lookup))
            except Exception:  # 確かめの失敗で画面へ例外を出さない 次の機会に確かめる
                found.append(())
            finally:
                done.set()

        self.phase = Phase.CHECKING
        self._work = done
        self._results = found
        threading.Thread(target=run, name="sashimono-ai-parts-check", daemon=True).start()
        self._poll_timer.start()

    def _install(self) -> None:
        target = self._target()
        if target is None:
            self._abandon()
            return
        wait = self.prepare() if self.prepare is not None else None
        pins = self._pins
        done = threading.Event()
        result: list[bool] = []

        def run() -> None:
            try:
                result.append(
                    self._installer(pins, target=target, key=self._pack.key, before_swap=wait)
                )
            except Exception:  # 落ちても今の版は残っている 次の機会に試す
                result.append(False)
            finally:
                done.set()

        self.phase = Phase.INSTALLING
        self._work = done
        self._install_result = result
        threading.Thread(target=run, name="sashimono-ai-parts-install", daemon=True).start()
        self._poll_timer.start()

    def _poll(self) -> None:
        work = self._work
        if work is None or not work.is_set():
            return
        self._work = None
        self._poll_timer.stop()
        if self.phase is Phase.CHECKING:
            pins = self._results[0] if self._results else ()
            if pins:
                self._pins = pins
                self.phase = Phase.PENDING
                return
            self.phase = Phase.IDLE
            if self._asked:
                self._asked = False
                self.finished.emit(False, "")
            return
        if self.phase is Phase.INSTALLING:
            succeeded = bool(self._install_result and self._install_result[0])
            self.phase = Phase.IDLE
            self._asked = False
            self._failures = 0 if succeeded else self._failures + 1
            self._store.save(replace(self._store.load(), failures=self._failures))
            if succeeded:
                refresh_runtime()
                _forget_modules(_RELOADABLE)
            self.finished.emit(succeeded, " ".join(self._pins))


def _forget_modules(names: tuple[str, ...]) -> None:
    """読み込み済みの部品を忘れ、次の import で入れ替えた方を読ませる"""
    for loaded in list(sys.modules):
        if loaded.split(".", 1)[0] in names:
            del sys.modules[loaded]
