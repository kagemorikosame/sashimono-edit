"""自動更新の画面の側 確かめる・落とす・知らせる・入れる・戻す

決めごと（計画書の F-12）

- 起動のたびに裏で確かめる 起動を待たせない 確かめるのは数時間に 1 回まで
  繋がらない・署名が通らないときは黙る（手で確かめたときだけ理由を出す）
- 落とすのは黙って行う 入れる（再起動が要る）のは本人の合図で行う 尋ねない設定なら、
  次の起動の頭で入れる **編集・書き出し・AI の作業の途中で勝手に再起動しない**
- 入れた版が起動できなければ、入れ替え係が前の版へ戻す 戻したことは次の起動で知らせる
- 前の版へはメニューからも戻せる

ネットワークと入れ替え係は差し替えられる（試験は偽物を渡し、本物の GitHub へは出ない）
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton

from sashimono import __version__
from sashimono.core import userdirs
from sashimono.links import RELEASES_URL
from sashimono.runtime import FeaturePack, python_abi, runtime_target_dir
from sashimono.ui.workspace import SCRIPTS_MOVE_ASK, SCRIPTS_MOVE_OFF, Preferences
from sashimono.update.check import CheckResult, Outcome, check_for_update, is_due
from sashimono.update.fetch import FetchError, Transport, UrllibTransport
from sashimono.update.flow import UpdateBusyError, UpdateChoices, prepare, reconcile, settle
from sashimono.update.package import (
    CarryError,
    Layout,
    PackageError,
    carry_user_files,
    current_layout,
)
from sashimono.update.portable import (
    PORTABLE_SCRIPTS_DIR,
    ScriptMove,
    clear_finished_swaps,
    clear_swap_pending,
    in_synced_folder,
    is_link,
    move_user_scripts,
    new_swap_token,
    unoffered,
    user_script_files,
)
from sashimono.update.reinstall import describe, estimate
from sashimono.update.signing import trusted_keys
from sashimono.update.state import UpdateStateStore, busy_with
from sashimono.update.swap import Launched, SwapPlan, launch, wait_started

__all__ = ["STARTUP_DELAY_MS", "UpdateController"]

#: 起動してから確かめ始めるまで 窓が出て、最初の描画と退避の確認が済んでから
STARTUP_DELAY_MS = 3000

#: 裏の仕事を見に行く間隔
_POLL_MS = 200

#: 閉じるときに、裏の移しが今の束を終えるのを待つ長さ（秒） 束は普通は数 MB で 1 秒もかからない
#: 超えたら待たずに閉じる（よけた元は次の移しか引き継ぎが戻す）
SHUTDOWN_WAIT = 10.0

#: 選べる答え
ANSWER_NOW = "now"
ANSWER_NEXT_START = "next-start"
ANSWER_SKIP = "skip"
ANSWER_LATER = "later"


@dataclass(frozen=True, slots=True)
class _Finished:
    """裏の仕事の結果"""

    manual: bool
    result: CheckResult
    #: 入れ替えを待つ所へ置けたか
    staged: bool = False
    #: 置き場に書けないので、自動では入れ替えられない
    not_replaceable: bool = False
    #: 落とす・確かめる・展開するで止まったわけ
    error: str = ""


class UpdateController(QObject):
    """編集画面に 1 つ置く"""

    def __init__(
        self,
        window: QMainWindow,
        *,
        preferences: Callable[[], Preferences],
        blockers: Callable[[], list[str]],
        arguments: Callable[[], list[str]],
        transport: Callable[[], Transport] = UrllibTransport,
        keys: Sequence[Ed25519PublicKey] | None = None,
        layout: Layout | None = None,
        frozen_layout: bool = True,
        store: UpdateStateStore | None = None,
        clock: Callable[[], float] = time.time,
        threaded: bool = True,
        launcher: Callable[[SwapPlan], Launched] = launch,
        waiter: Callable[[Launched], bool] = wait_started,
        confirm_close: Callable[[], bool] | None = None,
        close_window: Callable[[], bool] | None = None,
        rescan_scripts: Callable[[], None] | None = None,
    ) -> None:
        """``frozen_layout`` が真で ``layout`` が無ければ、配布版の置き場を自分で求める

        ``confirm_close`` は閉じてよいか（保存していない変更を尋ね、保存も済ませる）
        ``close_window`` は尋ねずに閉じる 既定は窓の ``close``（尋ねる窓ならそこで尋ねる）
        ``rescan_scripts`` は exe の隣のスクリプトを移した後に、スクリプトを読み直す
        """
        super().__init__(window)
        self._rescan_scripts = rescan_scripts
        self._window = window
        self._preferences = preferences
        self._blockers = blockers
        self._arguments = arguments
        self._transport = transport
        self._keys = tuple(keys) if keys is not None else trusted_keys()
        self._layout = layout if layout is not None or not frozen_layout else current_layout()
        self._store = store if store is not None else UpdateStateStore()
        self._clock = clock
        self._threaded = threaded
        self._launcher = launcher
        self._waiter = waiter
        self._confirm_close = confirm_close if confirm_close is not None else (lambda: True)
        self._close_window = close_window if close_window is not None else window.close
        self._results: queue.SimpleQueue[_Finished] = queue.SimpleQueue()
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        #: exe の隣のスクリプトを裏で移している間は真 終わった後に画面で行うことを受け取る
        self._moving = False
        #: 走っている裏の仕事の数 0 になったら見に行く時計を止める
        self._background_jobs = 0
        #: 裏の仕事のスレッド 閉じるときに今の束を終えるのを待つ（:meth:`shutdown`）
        self._threads: list[threading.Thread] = []
        #: 閉じる 裏の移しは新しい束に入らない
        self._stopping = threading.Event()
        self._move_results: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._move_timer = QTimer(self)
        self._move_timer.setInterval(_POLL_MS)
        self._move_timer.timeout.connect(self._poll_moves)
        #: 入れ替えを待っている版があるときだけ、ステータスバーに出すボタン
        self.button = QPushButton(window)
        self.button.setFlat(True)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.offer)
        self.button.hide()

    # --- 状態 ---

    @property
    def busy(self) -> bool:
        """確かめている・落としている最中"""
        return self._busy

    @property
    def configured(self) -> bool:
        """公開鍵が入っている版か 入っていない版は更新を確かめない"""
        return bool(self._keys)

    def ready_version(self) -> str | None:
        """落として確かめ、入れ替えを待っている版"""
        if self._layout is None:
            return None
        state = self._store.load()
        if state.ready_version and self._layout.staged_version() == state.ready_version:
            return state.ready_version
        return None

    def apply_preferences(self, preferences: Preferences) -> None:
        """設定を変えたときに、待っている版と次の起動の予約を今の好みにそろえる

        決まりは起動の頭と同じ :func:`~sashimono.update.flow.reconcile`（表は docs の
        「自動更新の仕組み」） ここで独自に決めると、画面で切り替えた後に起動の頭が
        別の答えを出す
        """
        state = self._store.load()
        settled = reconcile(state, choices_of(preferences))
        if settled != state:
            with contextlib.suppress(OSError):
                self._store.save(settled)
        if not settled.ready_version and state.ready_version:
            # 先行版をやめた 待っていた版の置き場を片付け、入れるかを尋ねるボタンも下げる
            settle(layout=self._layout, store=self._store, results=())
        self._refresh_button()

    def can_roll_back(self) -> bool:
        return self._layout is not None and self._layout.has_previous()

    # --- 起動したとき ---

    def start(self) -> None:
        """前の入れ替えの結果を知らせ、exe の隣のスクリプトを移すかを尋ね、確かめる頃合いなら
        裏で確かめる
        """
        notice = settle(layout=self._layout, store=self._store)
        if notice:
            self._say(notice, 20000)
        # 入れ替え係を待っている印は、入れ替えが済んだ（印の版と今の版が違う）なら外す 同じ版なら
        # まだ入れ替えていないか失敗して戻った 失敗した印は時間が過ぎれば効かなくなる
        clear_finished_swaps(userdirs.config_root(), __version__)
        stopped = self._store.load()
        if stopped.pending_notice:
            # 起動の頭で入れ替えを止めた（まだ窓が無かった） 1 度だけ知らせて消す 窓を塞がない
            # 知らせにする 起動して黙って出る窓が編集画面を塞ぐと閉じられない
            with contextlib.suppress(OSError):
                self._store.save(replace(stopped, pending_notice=""))
            self._notify("入れ替えを止めました", stopped.pending_notice)
        self._refresh_button()
        # 確かめるのを切っている人にも勧める 手で入れ替える人ほど消える側にいる
        self.offer_script_move()
        if not self._preferences().update_check:
            return
        # 開発の環境（置き場が無い）は git で新しくするので、起動時には確かめない
        if self._layout is None or not self.configured:
            return
        if not is_due(self._store.load().last_checked, self._clock()):
            return
        self._run(manual=False)

    def check_now(self) -> None:
        """ヘルプの〔更新を確かめる…〕 間隔を気にせず確かめ、結果を必ず見せる"""
        if self._busy:
            self._say("新しい版を確かめています…", 5000)
            return
        if not self.configured:
            self._inform(
                "更新を確かめられません",
                "この版には更新を確かめる鍵が入っていないので、自動では確かめません\n"
                f"新しい版は配布のページで確かめられます\n{RELEASES_URL}",
            )
            return
        if self.ready_version() is not None:
            self.offer()
            return
        self._say("新しい版を確かめています…", 0)
        self._run(manual=True)

    # --- exe の隣のスクリプト ---

    def offer_script_move(self, *, manual: bool = False) -> bool:
        """exe の隣の ``scripts`` に本人の物があれば、``%APPDATA%`` 側へ移す 移し始めたら真

        移すのは裏のスレッド（:meth:`_moved` が終わった後に画面の側で知らせる）

        起動のときは設定（``scripts_move``）に従う 既定は尋ねずに移し、何をどこへ移したかを
        1 度知らせる 尋ねる設定では、同じ物については 1 度だけ尋ねる 移し先に同じ名前があって
        残した物も、知らせるのは 1 度だけ（知らせた物を覚える 覚えないと起動のたびに出る）
        〔互換〕→〔exe の隣のスクリプトを移す…〕（``manual``）からは、設定にかかわらず尋ねる
        """
        if self._layout is None:
            if manual:
                self._inform(
                    "移す物はありません", "開発の環境では exe の隣のスクリプト置き場を使いません"
                )
            return False
        if self._moving:
            if manual:
                self._inform("スクリプトの置き場", "今 exe の隣のスクリプトを移しています")
            return False
        mode = SCRIPTS_MOVE_ASK if manual else self._preferences().scripts_move
        if mode == SCRIPTS_MOVE_OFF:
            return False
        if busy_with(self._store.path.parent) == "swap":
            # 入れ替え係が今の版のフォルダを付け替えている 触ると付け替えを邪魔する 次の起動で移す
            return False
        install = self._layout.install
        # 一覧を作る（scripts 全体を辿る）ところから裏のスレッドで行う 大きな配布物で編集画面を
        # 固めない 尋ねる必要があるときだけ、一覧を画面の側へ返して尋ねる
        self._moving = True
        self._in_background(
            lambda: (user_script_files(install), in_synced_folder(install)),
            lambda listed: self._listed(install, *listed, manual=manual, mode=mode),
        )
        return True

    def _listed(
        self, install: Path, mine: list[Path], synced: bool, *, manual: bool, mode: str
    ) -> None:
        """一覧ができた後に画面の側で決めること 移すなら裏のスレッドで移す"""
        if manual and not mine:
            self._moving = False
            self._inform(
                "移す物はありません",
                f"{install / PORTABLE_SCRIPTS_DIR} に、自分で置いた物はありません",
            )
            return
        if not mine:
            self._moving = False
            return
        state = self._store.load()
        fresh = {path.as_posix() for path in unoffered(mine, state.scripts_offered)}
        if not manual:
            if (mode == SCRIPTS_MOVE_ASK or synced) and not fresh:
                self._moving = False
                return  # 尋ねた・知らせた物だけが残っている 答えはもう聞いた
            if QApplication.activeModalWidget() is not None:
                # 退避の復元などを尋ねている最中に重ねない 何もせずに次の起動へ回す
                self._moving = False
                return
            offered = {*state.scripts_offered, *(path.as_posix() for path in mine)}
            with contextlib.suppress(OSError):
                self._store.save(replace(state, scripts_offered=tuple(sorted(offered))))
            if synced:
                # 同期フォルダの中では自動で移さない 元を消すと、ほかの機械の exe の隣からも消え、
                # その機械の %APPDATA% には写っていない 1 度だけ知らせ、移すのは本人に任せる
                self._moving = False
                self._notify("スクリプトの置き場", synced_scripts_text(len(mine)))
                return
        text = portable_scripts_text(len(mine))
        if synced:
            text += "\n\n" + synced_scripts_text(len(mine))
        if mode == SCRIPTS_MOVE_ASK and not self._ask_move(text):
            self._moving = False
            if not manual:
                self._say("後からでも〔互換〕→〔exe の隣のスクリプトを移す…〕で移せます", 15000)
            return
        target = userdirs.config_root() / PORTABLE_SCRIPTS_DIR
        loud = manual or mode == SCRIPTS_MOVE_ASK
        # 途中でアプリを閉じても、元は確定し終えてから消すので失われない（daemon で待たせない）
        self._in_background(
            lambda: move_user_scripts(install, target, should_stop=self._stopping.is_set),
            lambda result: self._moved(result, target, loud=loud, fresh=fresh),
        )

    def _in_background(
        self,
        work: Callable[[], Any],
        then: Callable[[Any], None],
        failed: Callable[[Exception], None] | None = None,
    ) -> None:
        """``work`` を裏のスレッドで行い、終わったら ``then`` を画面の側で呼ぶ

        裏で触るのはファイルだけ（scripts を辿る・写す・導入先の大きさを測る） 画面の部品には
        触らない 試験（``threaded`` が偽）ではその場で呼ぶ ``failed`` を渡さなければ、
        exe の隣のスクリプトを移す仕事の失敗として知らせる
        """
        on_error = failed if failed is not None else self._moved_failure
        if not self._threaded:
            try:
                value = work()
            except Exception as exc:  # 裏で行う物と同じく、画面の側へ例外を出さない
                on_error(exc)
                return
            then(value)
            return

        def run() -> None:
            try:
                value = work()
            except Exception as exc:  # 裏のスレッドで落ちると、移している印が立ったまま残る
                error = exc
                self._move_results.put(lambda: on_error(error))
                return
            self._move_results.put(lambda: then(value))

        self._background_jobs += 1
        thread = threading.Thread(target=run, name="sashimono-background", daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(thread)
        thread.start()
        self._move_timer.start()

    def shutdown(self, timeout: float = SHUTDOWN_WAIT) -> bool:
        """窓を閉じるときに呼ぶ 裏の仕事が今の束を終えるのを待つ 終われば真

        移しは束の途中で止めない（元をよけたまま止まると、次の起動で更新の入れ替えが先に走った
        ときに戻らない PR #245 の Codex の指摘） 新しい束には入らないよう知らせ、``timeout`` 秒まで
        待つ 超えたら待たずに閉じる よけた元は、次に錠を取った移しか引き継ぎ（自動更新の入れ替えの
        前に必ず走る）が元の場所へ戻す
        """
        self._stopping.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def _moved_failure(self, exc: Exception) -> None:
        self._moving = False
        failure = ScriptMove(failed=((Path(), f"{type(exc).__name__}: {exc}"),))
        self._notify("スクリプトの置き場", move_summary(failure, Path()))

    def _poll_moves(self) -> None:
        try:
            done = self._move_results.get_nowait()
        except queue.Empty:
            return
        self._background_jobs -= 1
        if self._background_jobs <= 0:
            # 裏の仕事が 2 つ重なっていれば（移しと入れ直しの見積もり）、両方が終わるまで回す
            self._background_jobs = 0
            self._move_timer.stop()
        # 続きの仕事（一覧の後の移し）は done の中で、また裏のスレッドへ出して時計を回し直す
        done()

    def _moved(self, result: ScriptMove, target: Path, *, loud: bool, fresh: set[str]) -> None:
        """移し終えた後に画面の側で行うこと スクリプトの読み直しと知らせ"""
        self._moving = False
        if result.busy:
            # ほかの Sashimono が移している 何もしていないので、起動のときは黙って次へ回す
            if loud:
                self._inform(
                    "スクリプトの置き場",
                    "ほかの Sashimono の窓が移しています 終わってからもう一度選んでください",
                )
            return
        if result.linked:
            # exe の隣の scripts そのものがリンク 何も移していない 1 度だけ知らせる
            text = (
                "Sashimono.exe の隣の scripts フォルダは、別の場所を指すリンク（ジャンクション・"
                "シンボリックリンク）なので、自動では移しません（リンクの先は AviUtl と分け合う"
                "フォルダや同期先のことがあり、移すと先の物を消してしまうため） リンクの先の物は"
                "これまでどおり読み込みます 自動更新でもリンクとして引き継ぎます"
            )
            if loud:
                self._inform("スクリプトの置き場", text)
            elif fresh:
                self._notify("スクリプトの置き場", text)
            return
        if (result.moved or result.left) and self._rescan_scripts is not None:
            self._rescan_scripts()
        # 自動で移すときは、移せた物があったときと、移せずに残った物を初めて見たときだけ知らせる
        # 残った物（同じ名前があった・写せなかった）は次の起動でも残るので、毎回は出さない
        stayed = {
            path.as_posix()
            for path in (*result.kept, *result.held, *(p for p, _r in result.failed))
        }
        if loud:
            self._inform("スクリプトの置き場", move_summary(result, target))
        elif result.moved or result.left or stayed & fresh:
            self._notify("スクリプトの置き場", move_summary(result, target))

    def _notify(self, title: str, text: str) -> None:
        """尋ねずに済ませたことを知らせる 試験では差し替える

        手を止めさせない窓（modeless）で出す 起動して黙って出る窓が親の窓を塞ぐと、出ている
        間は編集画面を閉じられない（Windows の閉じる知らせも、塞がれた窓には届かない）
        """
        box = QMessageBox(QMessageBox.Icon.Information, title, text, parent=self._window)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()

    def _ask_move(self, text: str) -> bool:
        """移すかを尋ねる 試験では差し替える 既定の答えは移さない側（Enter で移さない）"""
        box = QMessageBox(self._window)
        box.setWindowTitle("スクリプトの置き場")
        box.setText(text)
        move = box.addButton("移す", QMessageBox.ButtonRole.AcceptRole)
        later = box.addButton("今はしない", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(later)
        box.exec()
        return box.clickedButton() is move

    # --- 裏の仕事 ---

    def _run(self, *, manual: bool) -> None:
        self._busy = True
        preferences = self._preferences()

        def work() -> None:
            self._results.put(self._work(manual, preferences))

        if self._threaded:
            threading.Thread(target=work, name="sashimono-update", daemon=True).start()
            self._timer.start()
        else:
            work()
            self._poll()

    def _work(self, manual: bool, preferences: Preferences) -> _Finished:
        """確かめて、入れられる版なら落として置く 裏のスレッドで走る 例外は外へ出さない"""
        state = self._store.load()
        # 失敗した回も時刻を残す 繋がらない間に、起動のたびに取りに行かない
        with contextlib.suppress(OSError):
            self._store.save(replace(state, last_checked=self._clock()))
        try:
            result = check_for_update(
                __version__,
                transport=self._transport(),
                keys=self._keys,
                beta=preferences.update_beta,
                # 手で確かめたときは飛ばした版も見せる 飛ばしたのを取り消す道になる
                skipped=() if manual else state.skipped,
            )
        except Exception as exc:  # 確かめる所の不具合で起動中の画面を落とさない
            return _Finished(manual, CheckResult(Outcome.FAILED, reason=str(exc)))
        manifest = result.manifest
        if result.outcome is not Outcome.AVAILABLE or manifest is None or self._layout is None:
            return _Finished(manual, result)
        if not self._layout.writable():
            return _Finished(manual, result, not_replaceable=True)
        try:
            if self._layout.staged_version() == manifest.version:
                # 前の起動で落とし終えている 覚え書きだけ直す 同じ版に本人が選んだ予約は残す
                current = self._store.load()
                same = current.ready_version == manifest.version
                self._store.save(
                    replace(
                        current,
                        ready_version=manifest.version,
                        ready_notes_url=manifest.notes_url,
                        ready_python_abi=manifest.python_abi,
                        apply_on_start=current.apply_on_start and same,
                        apply_chosen=current.apply_chosen and same,
                    )
                )
            else:
                prepare(
                    manifest, transport=self._transport(), layout=self._layout, store=self._store
                )
            # 予約は設定を変えたときと同じ決まりで付ける（尋ねない設定なら自動で予約する）
            self._store.save(reconcile(self._store.load(), choices_of(preferences)))
        except UpdateBusyError as exc:
            # ほかの窓が落としている・入れ替えている 待てば済むので起動時は黙る
            return _Finished(manual, result, error=f"ほかの窓で更新を進めています（{exc}）")
        except (FetchError, PackageError, OSError) as exc:
            return _Finished(manual, result, error=str(exc))
        except Exception as exc:  # 裏のスレッドで落ちると、確かめている印が立ったまま残る
            return _Finished(manual, result, error=f"{type(exc).__name__}: {exc}")
        return _Finished(manual, result, staged=True)

    def _poll(self) -> None:
        try:
            finished = self._results.get_nowait()
        except queue.Empty:
            return
        self._timer.stop()
        self._busy = False
        self._finish(finished)

    def _finish(self, finished: _Finished) -> None:
        result = finished.result
        manifest = result.manifest
        self._refresh_button()
        if finished.manual:
            self._window.statusBar().clearMessage()
        if result.outcome is Outcome.AVAILABLE and manifest is not None:
            if finished.staged:
                if self._preferences().update_confirm or finished.manual:
                    if finished.manual:
                        self.offer()
                    else:
                        self._say(f"新しい版 {manifest.version} を入れる準備ができました", 15000)
                else:
                    self._say(f"次の起動で {manifest.version} に更新します", 15000)
                return
            if finished.not_replaceable or self._layout is None:
                text = (
                    f"新しい版 {manifest.version} があります この場所に置いた Sashimono は"
                    " 自動では入れ替えられないので、配布のページから入れてください"
                    if self._layout is not None
                    else f"新しい版 {manifest.version} があります（開発の環境では入れ替えません）"
                )
                self._tell_by_hand(finished.manual, text, manifest.notes_url, manifest.python_abi)
                return
            if finished.manual:
                self._inform("新しい版を落とせませんでした", finished.error or "理由が分からない")
            return
        if result.outcome is Outcome.MANUAL and manifest is not None:
            text = (
                f"新しい版 {manifest.version} があります この版からは自動では入れられないので、"
                "配布のページから入れ直してください"
            )
            self._tell_by_hand(finished.manual, text, manifest.notes_url, manifest.python_abi)
            return
        if not finished.manual:
            return  # 最新・繋がらない・署名が通らない 起動時は黙る
        if result.outcome is Outcome.UP_TO_DATE:
            self._inform("更新を確かめた", f"今の版 {__version__} が最新です")
        elif result.outcome is Outcome.FAILED:
            self._inform(
                "更新を確かめられませんでした",
                f"{result.reason}\n繋がっていれば、しばらくしてからもう一度試してください",
            )

    # --- 入れる ---

    def _tell_by_hand(self, manual: bool, text: str, url: str, new_abi: str) -> None:
        """配布のページから手で入れ替えるよう言う 起動時の確認はステータスバーに 1 行で、
        詳しい注意（exe の隣の scripts・入れ直し）は作らない 手で確かめたときは、注意を
        作るのに scripts を辿り導入先の大きさを測るので、裏のスレッドで作ってから窓で出す
        """
        if not manual:
            self._say(text, 20000)
            return
        if self._layout is None:
            self._say_or_inform(True, text, url)
            return
        self._in_background(
            lambda: self._by_hand_notes(new_abi),
            lambda details: self._say_or_inform(True, text, url, details),
            lambda _exc: self._say_or_inform(True, text, url),
        )

    def offer(self) -> None:
        """入れ替えを待つ版を、入れるかどうか尋ねる

        Python が変わる版なら、入れ直しの見積もり（導入先の大きさを測る 最大 1.5 秒）を裏の
        スレッドで作ってから尋ねる 測る間、編集画面を固めない
        """
        version = self.ready_version()
        if version is None or self._layout is None:
            self._refresh_button()
            return
        abi = self._store.load().ready_python_abi
        self._in_background(
            lambda: runtime_note(abi),
            lambda note: self._offer_with(version, note),
            lambda _exc: self._offer_with(version, ""),
        )

    def _offer_with(self, version: str, note: str) -> None:
        if self.ready_version() != version:
            self._refresh_button()
            return  # 見積もっている間に片付いた（ほかの窓が入れた・ベータを切った）
        state = self._store.load()
        lines = [f"新しい版 {escape(version)} を入れる準備ができました（今は {__version__}）"]
        if state.ready_notes_url:
            url = escape(state.ready_notes_url, quote=True)
            lines.append(f'変わった所: <a href="{url}">{url}</a>')
        if note:
            lines.extend(escape(part) for part in note.split("\n"))
        lines.append("入れるには再起動が要ります 保存していない変更があれば、閉じる前に尋ねます")
        answer = self._choose("新しい版", "<br>".join(lines))
        if answer == ANSWER_NOW:
            self.restart_now()
        elif answer == ANSWER_NEXT_START:
            # 本人が選んだ予約 後で〔入れる前に尋ねる〕を入れても外さない（もう答えてある）
            self._store.save(replace(self._store.load(), apply_on_start=True, apply_chosen=True))
            self._say(f"次の起動で {version} に更新します", 10000)
        elif answer == ANSWER_SKIP:
            self._store.save(
                replace(
                    self._store.load(),
                    skipped=(*state.skipped, version),
                    ready_version="",
                    ready_notes_url="",
                    ready_python_abi="",
                    apply_on_start=False,
                )
            )
            settle(layout=self._layout, store=self._store, results=())
            self._refresh_button()

    def restart_now(self) -> bool:
        """今すぐ入れる 編集の途中で止めてはいけない作業が動いていれば断る"""
        if self._layout is None or self.ready_version() is None:
            return False
        return self._restart("apply")

    def roll_back(self) -> bool:
        """ヘルプの〔前の版に戻す…〕"""
        if self._layout is None or not self._layout.has_previous():
            self._inform("前の版に戻せません", "戻せる前の版が残っていません")
            return False
        from sashimono.update.package import read_build_info

        info = read_build_info(self._layout.previous)
        previous = info.version if info is not None else "前の版"
        answer = self._choose(
            "前の版に戻す",
            escape(f"今の版 {__version__} をよけて、{previous} に戻します 再起動が要ります")
            + "<br>"
            + escape("戻した後は、この版を自動では入れません（手で確かめれば入れられます）"),
            confirm_only=True,
        )
        if answer != ANSWER_NOW or not self._restart("rollback"):
            return False
        # 戻すと決まってから覚える 取り消したのに、今の版が「飛ばす版」に残らないように
        state = self._store.load()
        if __version__ not in state.skipped:
            self._store.save(replace(state, skipped=(*state.skipped, __version__)))
        return True

    def _restart(self, mode: str) -> bool:
        """保存の確認が済んでから入れ替え係を起こし、窓を閉じて終わる

        入れ替え係を先に起こすと、保存の確認で取り消したり迷って待ちの 120 秒を過ぎたりしたとき、
        入れ替え係が待つのを諦めて今の版を起こし、本体が 2 つ動く 確認で取り消されたら起こさない
        """
        assert self._layout is not None
        blockers = [*self._blockers()]
        if self._moving:
            # exe の隣のスクリプトを移している最中に入れ替えると、写す側（carry_user_files）と
            # 移す側が同じファイルを同時に触る 入れ替え係もフォルダを付け替えられない
            blockers.append("exe の隣のスクリプトを %APPDATA% へ移しています")
        if blockers:
            self._inform(
                "今は再起動できません",
                "次の作業が終わってからもう一度選んでください\n" + "\n".join(blockers),
            )
            return False
        if self._busy_elsewhere():
            return False
        if not self._confirm_close():
            return False
        # 確認で迷っている間に、ほかの窓が入れ替えを始めたかもしれない もう一度見る
        if self._busy_elsewhere():
            return False
        # 写すのはこの場で行う（裏へ出さない） 保存の確認はもう済んでいて、裏へ出すと写している
        # 間に編集でき、閉じるときにその変更を尋ねずに捨てる 入れ替え係が始まるのを待つのと同じく、
        # 待つ印の形の矢印を出して待たせる（すぐ後に窓を閉じて終わる）
        stopped: CarryError | None = None
        # この引き継ぎの「入れ替え係を待っている」印の識別子 起こせなければ自分の印だけを外す
        token = new_swap_token()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if mode == "apply":
                carry_user_files(
                    self._layout.install,
                    self._layout.staged,
                    aside=userdirs.config_root() / PORTABLE_SCRIPTS_DIR,
                    swap_mark=(__version__, token),
                )
            elif mode == "rollback":
                # 戻すと今の版は previous へ回り、次に新しい版を入れたときに消える 今の版の
                # exe の隣へ後から置いた物・直した物を、戻る先の版へ写す（今の側が正 上書きする）
                carry_user_files(
                    self._layout.install,
                    self._layout.previous,
                    overwrite=True,
                    swap_mark=(__version__, token),
                )
        except CarryError as exc:
            stopped = exc
        finally:
            QApplication.restoreOverrideCursor()
        if stopped is not None:
            # 写せないまま入れ替えると、本人の物は次の更新で消える版にだけ残る 入れ替え係を
            # 起こさずに止める 窓はまだ閉じていないので、今の版のまま続けられる
            self._inform("入れ替えを止めました", stopped.explain())
            return False
        if self._launch_swap(mode):
            return True
        # 入れ替え係を起こせなかった・閉じられなかった 引き継ぎが置いた「入れ替え係を待っている」
        # 印を外す（残すと、移しが印の切れるまで止まる）
        clear_swap_pending(userdirs.config_root(), token)
        return False

    def _launch_swap(self, mode: str) -> bool:
        """入れ替え係を起こし、走り出したら窓を閉じて終わる 起こせなければ知らせて偽"""
        assert self._layout is not None
        # 開き直す作品は確認の後で決める 確認で名前を付けて保存したら、その作品を開き直す
        plan = SwapPlan(mode, self._layout, pid=os.getpid(), arguments=self._arguments())
        try:
            launched = self._launcher(plan)
        except OSError as exc:
            self._inform("入れ替えを始められません", str(exc))
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            started = self._waiter(launched)
        finally:
            QApplication.restoreOverrideCursor()
        if not started and launched.lost_to_another():
            # ほぼ同時にほかの窓も入れ替え係を起こし、そちらが先に錠を取った
            self._inform_other_swap()
            return False
        if not started:
            self._inform(
                "入れ替えを始められません",
                "入れ替えを受け持つ PowerShell が動きませんでした（会社の機械などで止められている"
                f"ことがあります） 配布のページから入れ直してください\n{RELEASES_URL}",
            )
            return False
        # 保存の確認はもう済んでいるので、閉じるときにもう一度は尋ねない
        if not self._close_window():
            # 閉じられなかった（窓の側が断った） 入れ替え係を残すと、後で閉じた所で入れ替わる
            launched.process.kill()
            return False
        QApplication.quit()
        return True

    def _busy_elsewhere(self) -> bool:
        """ほかの窓か入れ替え係が更新を進めていれば、知らせて真

        2 つの窓で〔今すぐ再起動して入れる〕を選ぶと入れ替え係が 2 つ起き、片方が作った
        ``.previous`` をもう片方が消して戻す先まで失う 入れ替え係の側も錠で 1 つに絞るが、
        起こす前に断る方が、本人に何が起きているかを言える
        """
        busy = busy_with(self._store.path.parent)
        if busy == "swap":
            self._inform_other_swap()
            return True
        if busy == "stage":
            self._inform(
                "今は入れられません",
                "ほかの Sashimono の窓が新しい版を落としています"
                " 終わってからもう一度選んでください",
            )
            return True
        return False

    def _inform_other_swap(self) -> None:
        self._inform(
            "ほかの窓で入れ替えを始めています",
            "ほかの Sashimono の窓が新しい版への入れ替えを始めています"
            " この窓を閉じると入れ替わり、新しい版が開きます",
        )

    # --- 見せ方 ---

    def _refresh_button(self) -> None:
        version = self.ready_version()
        if version is None:
            self.button.hide()
            return
        self.button.setText(f"新しい版 {version} を入れる…")
        self.button.show()

    def _say(self, text: str, timeout: int) -> None:
        self._window.statusBar().showMessage(text, timeout)

    def _say_or_inform(self, manual: bool, text: str, url: str, details: str = "") -> None:
        """手で確かめたときは窓で、起動時はステータスバーに 1 行で出す

        ``details`` は窓のときだけ添える ステータスバーに何行も並べても読めない
        """
        if manual:
            self._inform("新しい版", "\n".join(part for part in (text, url, details) if part))
        else:
            self._say(text, 20000)

    def _by_hand_notes(self, new_abi: str) -> str:
        """配布のページから手で入れ替える人への注意 消える物と、入れ直しが要る物"""
        notes = []
        # exe の隣の scripts そのものがリンクなら、フォルダごと入れ替えても消えるのはリンクだけで、
        # 先の物は残る 「消えます」とは言わない
        if self._layout is not None and not is_link(self._layout.install / PORTABLE_SCRIPTS_DIR):
            mine = user_script_files(self._layout.install)
            if mine:
                notes.append(
                    portable_scripts_text(len(mine))
                    + "\n手で入れ替える前に〔互換〕→〔exe の隣のスクリプトを移す…〕で移してください"
                )
        note = runtime_note(new_abi)
        if note:
            notes.append(note)
        return "\n\n".join(notes)

    def _inform(self, title: str, text: str) -> None:
        """本人に読ませる 試験では差し替える"""
        QMessageBox.information(self._window, title, text)

    def _choose(self, title: str, html: str, *, confirm_only: bool = False) -> str:
        """どうするかを尋ねる 試験では差し替える"""
        box = QMessageBox(self._window)
        box.setWindowTitle(title)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        # 変わった所の URL は押せば既定のブラウザで開く（QMessageBox の文字は外のリンクを開く）
        box.setText(html)
        answers: dict[object, str] = {}
        if confirm_only:
            answers[box.addButton("戻して再起動する", QMessageBox.ButtonRole.AcceptRole)] = (
                ANSWER_NOW
            )
            later = box.addButton("やめる", QMessageBox.ButtonRole.RejectRole)
        else:
            answers[box.addButton("今すぐ再起動して入れる", QMessageBox.ButtonRole.AcceptRole)] = (
                ANSWER_NOW
            )
            answers[box.addButton("次の起動で入れる", QMessageBox.ButtonRole.ActionRole)] = (
                ANSWER_NEXT_START
            )
            answers[box.addButton("この版を飛ばす", QMessageBox.ButtonRole.DestructiveRole)] = (
                ANSWER_SKIP
            )
            later = box.addButton("後で", QMessageBox.ButtonRole.RejectRole)
        answers[later] = ANSWER_LATER
        box.setDefaultButton(later)
        box.exec()
        return answers.get(box.clickedButton(), ANSWER_LATER)


def choices_of(preferences: Preferences) -> UpdateChoices:
    """画面の設定から、更新の決まりが見る 3 項目を取り出す"""
    return UpdateChoices(
        check=preferences.update_check,
        confirm=preferences.update_confirm,
        beta=preferences.update_beta,
    )


def runtime_note(new_abi: str, packs: Sequence[FeaturePack] | None = None) -> str:
    """新しい版で Python が変わり、入れてある実行環境が読めなくなるなら、その案内

    どの機能を入れ直すのか・どれだけ落とすのか・どれだけ掛かるのかまで添える
    （:mod:`sashimono.update.reinstall`） 「入れ直しが要る」だけだと、2 GB を超える
    落とし直しを知らずに、締め切り前に選んでしまう
    """
    target = runtime_target_dir()
    if not new_abi or new_abi == python_abi() or target is None:
        return ""
    try:
        installed = target.is_dir() and any(target.iterdir())
    except OSError:
        installed = False
    if not installed:
        return ""
    lead = (
        f"この版では Python が変わる（{python_abi()} → {new_abi}）ので、入れてある字幕起こし・"
        "AI 連携の環境は入れ直しが要ります（消しはしません 入れ直すまでその機能は使えません）"
    )
    found = estimate(target, packs if packs is not None else _feature_packs())
    return lead if found is None else f"{lead}\n{describe(found)}"


def _feature_packs() -> tuple[FeaturePack, ...]:
    # 字幕起こしと AI 連携の定義は、それぞれの機能の側が持つ ここで名前を書き直すと、
    # 包みを足したときに見積もりだけ古くなる
    from sashimono.ai.environment import AI_PACK
    from sashimono.asr.environment import ASR_PACK

    return (ASR_PACK, AI_PACK)


#: 移した物を名前で並べる数 多すぎると知らせの窓が画面からはみ出す
_LISTED = 10


def move_summary(result: ScriptMove, target: Path) -> str:
    """移した結果 残した物・消せなかった物・写せなかった物は数と、どう読まれるかを言う"""
    lines = [
        (
            f"Sashimono.exe の隣の scripts から、{len(result.moved)} 個を {target} へ移しました"
            "（zip を手で展開し直しても消えない置き場です 読み込みはこれまでどおり）"
        )
        if result.moved
        else "移した物はありません"
    ]
    lines.extend(f"  {path.as_posix()}" for path in result.moved[:_LISTED])
    if len(result.moved) > _LISTED:
        lines.append(f"  ほか {len(result.moved) - _LISTED} 個")
    if result.kept:
        lines.append(
            f"移し先に同じ名前で中身の違う物があった {len(result.kept)} 個は、上書きせずに"
            " exe の隣へ残しました 同じ名前では、前から移し先の物が読まれています（これまでと同じ）"
            " 見比べて、要らない方を消してください"
        )
        lines.extend(f"  {path.as_posix()}" for path in result.kept[:_LISTED])
    if result.held:
        lines.append(
            f"{len(result.held)} 個は、同じフォルダ（一式）に移せない物がある、または同じ名前の"
            "モジュールが移し先にあるので、一式のまま exe の隣へ残しました"
            "（一部だけ移すと、移した先で別のモジュールが読まれて描画が変わるため）"
        )
        lines.extend(f"  {path.as_posix()}" for path in result.held[:_LISTED])
    if result.left:
        lines.append(
            f"{len(result.left)} 個は写せましたが、元を消せませんでした"
            "（両方に在ります 読み込むのは移した方です）"
        )
    if result.failed:
        first, reason = result.failed[0]
        lines.append(
            f"{len(result.failed)} 個は写せませんでした 元のまま残っています"
            f"（{first.as_posix()}: {reason}）"
        )
    return "\n".join(lines)


def synced_scripts_text(count: int) -> str:
    """exe の隣が OneDrive などの同期フォルダの中にあるときの案内 自動では移さない理由"""
    return (
        f"Sashimono.exe の隣の scripts フォルダに、自分で置いた物が {count} 個あります\n"
        "このフォルダは OneDrive などの同期フォルダの中なので、自動では移しません"
        "（元を消すと、同期しているほかの機械からも消えます"
        " ほかの機械の %APPDATA% には写っていません）\n"
        "移すときは、それぞれの機械で〔互換〕→〔スクリプトフォルダを開く〕で開く所へ写してから、"
        "exe の隣の物を消してください"
    )


def portable_scripts_text(count: int) -> str:
    """exe の隣の ``scripts`` に本人の物があるときの案内 移すかを尋ねる文面と、手で入れ替える
    案内（配布のページへ送るとき）で同じ言い方にする
    """
    return (
        f"Sashimono.exe の隣の scripts フォルダに、自分で置いた物が {count} 個あります\n"
        "自動更新では新しい版へ写しますが、zip を手で展開し直してフォルダごと入れ替えると消えます\n"
        f"{userdirs.config_root() / 'scripts'} へ移すと、どちらの更新でも消えません"
        "（同じ名前の物があれば上書きせずに残します 移した後もこれまでどおり読み込みます）"
    )
