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

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton

from sashimono import __version__
from sashimono.links import RELEASES_URL
from sashimono.runtime import python_abi, runtime_target_dir
from sashimono.ui.workspace import Preferences
from sashimono.update.check import CheckResult, Outcome, check_for_update, is_due
from sashimono.update.fetch import FetchError, Transport, UrllibTransport
from sashimono.update.flow import UpdateBusyError, UpdateChoices, prepare, reconcile, settle
from sashimono.update.package import Layout, PackageError, carry_user_files, current_layout
from sashimono.update.signing import trusted_keys
from sashimono.update.state import UpdateStateStore, busy_with
from sashimono.update.swap import Launched, SwapPlan, launch, wait_started

__all__ = ["STARTUP_DELAY_MS", "UpdateController"]

#: 起動してから確かめ始めるまで 窓が出て、最初の描画と退避の確認が済んでから
STARTUP_DELAY_MS = 3000

#: 裏の仕事を見に行く間隔
_POLL_MS = 200

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
    ) -> None:
        """``frozen_layout`` が真で ``layout`` が無ければ、配布版の置き場を自分で求める

        ``confirm_close`` は閉じてよいか（保存していない変更を尋ね、保存も済ませる）
        ``close_window`` は尋ねずに閉じる 既定は窓の ``close``（尋ねる窓ならそこで尋ねる）
        """
        super().__init__(window)
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
        """前の入れ替えの結果を知らせ、確かめる頃合いなら裏で確かめる"""
        notice = settle(layout=self._layout, store=self._store)
        if notice:
            self._say(notice, 20000)
        self._refresh_button()
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
                self._say_or_inform(finished.manual, text, manifest.notes_url)
                return
            if finished.manual:
                self._inform("新しい版を落とせませんでした", finished.error or "理由が分からない")
            return
        if result.outcome is Outcome.MANUAL and manifest is not None:
            text = (
                f"新しい版 {manifest.version} があります この版からは自動では入れられないので、"
                "配布のページから入れ直してください"
            )
            self._say_or_inform(finished.manual, text, manifest.notes_url)
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

    def offer(self) -> None:
        """入れ替えを待つ版を、入れるかどうか尋ねる"""
        version = self.ready_version()
        if version is None or self._layout is None:
            self._refresh_button()
            return
        state = self._store.load()
        lines = [f"新しい版 {escape(version)} を入れる準備ができました（今は {__version__}）"]
        if state.ready_notes_url:
            url = escape(state.ready_notes_url, quote=True)
            lines.append(f'変わった所: <a href="{url}">{url}</a>')
        note = runtime_note(state.ready_python_abi)
        if note:
            lines.append(escape(note))
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
        blockers = self._blockers()
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
        if mode == "apply":
            carry_user_files(self._layout.install, self._layout.staged)
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

    def _say_or_inform(self, manual: bool, text: str, url: str) -> None:
        if manual:
            self._inform("新しい版", f"{text}\n{url}")
        else:
            self._say(text, 20000)

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


def runtime_note(new_abi: str) -> str:
    """新しい版で Python が変わり、入れてある実行環境が読めなくなるなら、その案内"""
    target = runtime_target_dir()
    if not new_abi or new_abi == python_abi() or target is None:
        return ""
    try:
        installed = target.is_dir() and any(target.iterdir())
    except OSError:
        installed = False
    if not installed:
        return ""
    return (
        "この版では Python が変わるので、入れてある字幕起こし・AI 連携の環境は入れ直しが要ります"
        "（消しはしません 入れ直すまでその機能は使えません）"
    )
