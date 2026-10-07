"""未導入の機能を、その場で用意するための部品

字幕起こしと AI 連携で同じものを使う 導入の見せ方は機能が変わっても同じ
（何が入るかを出す → 実行する → ログを流す → 状態を出し直す）なので、
1 か所にまとめてある

導入は子プロセスなので、ログはキューで受けてタイマーで拾う ワーカースレッドから
ウィジェットを触ると Qt が落ちる
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence

from packaging.requirements import Requirement
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.package_index import Latest, latest_release
from sashimono.runtime import (
    LEFT_RUNNING_NOTE,
    FeaturePack,
    PackageStatus,
    PackStatus,
    install_command,
    install_result_text,
    install_runtime,
    is_newer_version,
    pip_left_running,
    refresh_runtime,
    restart_note,
    snapshot_runtime_modules,
)
from sashimono.ui.theme import Colors, themed_style

__all__ = ["SetupSection", "describe_updates"]

#: 導入ログを拾う間隔（ミリ秒）
POLL_MS = 120
#: 戻らずに残った pip が戻ったかを見る間隔（ミリ秒）
LEFTOVER_POLL_MS = 500


class SetupSection(QWidget):
    """機能の導入状況を出し、その場で入れられるようにする"""

    #: 導入が終わった 引数は成功したか
    finished = Signal(bool)
    #: 状態を見直した 引数は「いま使えるか」
    changed = Signal(bool)

    def __init__(self, pack: FeaturePack, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pack = pack
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._done: threading.Event | None = None
        #: 導入の中断の頼み 終わった知らせ（_done）とは分けて持つ
        self._cancel = threading.Event()
        self._code = 0
        #: 最後の導入のあとに出した案内 使える状態になると導入欄ごと隠す画面
        #: （アシスタント）があるので、そちらが自分の見える所へ写せるように持つ
        self.note = ""
        #: この導入を始める前に、専用フォルダから読み込み済みだった物
        #: 導入ごとに持つ 字幕起こしの導入と重なっても、控えが混ざらない
        self._before: dict[str, int] = {}
        #: 入っている版のままでも入れ替える 更新を頼まれたときに立てる
        #: pip は ``--target`` に同じ名前が在ると ``--upgrade`` 無しでは入れ替えないので、
        #: 立てないと〔環境を更新〕を押しても何も変わらない
        self._force_upgrade = False
        #: 最新の版を尋ねている最中の知らせと、その答え（部品ごとの版 尋ねられなければ None）
        self._checking: threading.Event | None = None
        self._latest: dict[str, Latest | None] = {}
        #: 尋ねた結果を言う文 状態を出し直しても消えないように持つ
        self._update_note = ""

        self._status = QLabel(self)
        self._status.setWordWrap(True)
        themed_style(self._status, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._extra = QCheckBox(pack.extra_label or "追加分も入れる", self)
        self._extra.setChecked(True)
        self._extra.setVisible(bool(pack.extra))
        self._extra.toggled.connect(self.refresh)

        self._button = QPushButton("環境を導入", self)
        # clicked は押した状態（bool）を渡してくるので、そのまま start へ繋がない
        self._button.clicked.connect(lambda: self.start())

        # 尋ねるのは押したときだけ 開くたびに尋ねると、繋がっていない機械で待たされる
        self._check_button = QPushButton("更新を確かめる", self)
        self._check_button.setToolTip("PyPI に新しい版があるかを尋ねる（押したときだけ通信する）")
        self._check_button.clicked.connect(self.check_updates)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        self._log.setMaximumBlockCount(2000)
        self._log.setMaximumHeight(160)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._extra)
        row.addStretch(1)
        row.addWidget(self._check_button)
        row.addWidget(self._button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._status)
        layout.addLayout(row)
        layout.addWidget(self._progress)
        layout.addWidget(self._log)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        #: 前の導入の pip が残っている間だけ回し、戻ったら導入のボタンを押せるように戻す
        self._leftover_timer = QTimer(self)
        self._leftover_timer.setInterval(LEFTOVER_POLL_MS)
        self._leftover_timer.timeout.connect(self._check_leftover)
        self.refresh()

    # --- 状態 ---

    @property
    def status(self) -> PackStatus:
        return self._pack.status()

    @property
    def extra(self) -> bool:
        """追加分（CUDA など）を入れる／使う選択"""
        return self._extra.isChecked()

    @property
    def busy(self) -> bool:
        return self._done is not None

    def refresh(self) -> None:
        """導入状況を見直して表示を作り直す"""
        status = self.status
        self._button.setText("環境を更新" if status.installed else "環境を導入")
        self._extra.setEnabled(not status.extra_installed and not self.busy)

        lines = [status.summary()]
        if not status.installed:
            packages = "、".join(status.missing(extra=self.extra))
            lines.append(f"入れるもの: {packages}")
            size = status.download_mb(extra=self.extra)
            if size:
                lines.append(f"ダウンロードは {_readable(size)} ほどです")
        if self._update_note:
            lines.append(self._update_note)
        self._status.setText("\n".join(lines))
        # 入っていない物の新しい版を尋ねても仕方がない 入れればいちばん新しい版が入る
        self._check_button.setVisible(status.installed)
        self._check_button.setEnabled(self._checking is None and not self.busy)
        self._block_while_pip_is_left()
        self.changed.emit(status.ready)

    def _block_while_pip_is_left(self) -> bool:
        """前の導入の pip が残っていれば、導入のボタンを押せなくして再起動を頼む

        押せると、錠を待つだけの導入が始まり、pip が書き換えたままの標準出力やログの上で
        次の pip が走る 戻れば :meth:`_check_leftover` が押せる状態へ戻す
        """
        if not pip_left_running():
            return False
        self._button.setEnabled(False)
        self._extra.setEnabled(False)
        if LEFT_RUNNING_NOTE not in self._status.text():
            self._status.setText(f"{self._status.text()}\n{LEFT_RUNNING_NOTE}".strip())
        self._leftover_timer.start()
        return True

    def _check_leftover(self) -> None:
        if pip_left_running() or self.busy:
            return
        self._leftover_timer.stop()
        self._button.setEnabled(True)
        self.refresh()

    def command_text(self) -> str:
        """これから実行するコマンド 画面に見せるため"""
        return " ".join(self._command())

    # --- 導入 ---

    def check_updates(self) -> None:
        """入れてある部品に新しい版があるかを PyPI に尋ねる 答えは見張りの時計で拾う

        尋ねるのは裏のスレッド 繋がらない機械では最長で数十秒待つので、その間に
        画面が固まらないようにする
        """
        if self._checking is not None:
            return
        done = threading.Event()
        requirements = self._pack.required
        found: dict[str, Latest | None] = {}
        self._checking = done
        self._latest = found
        self._check_button.setEnabled(False)
        self._update_note = "新しい版を確かめています…"
        self.refresh()

        def run() -> None:
            for requirement in requirements:
                found[requirement] = latest_release(requirement)
            done.set()

        threading.Thread(target=run, name=f"sashimono-check-{self._pack.key}", daemon=True).start()
        self._timer.start()

    def _finish_check(self) -> None:
        self._checking = None
        note, upgradable = describe_updates(self.status.packages, self._latest)
        self._update_note = note
        if upgradable:
            # 次に押す〔環境を更新〕で入れ替える 立てないと、名前が在るだけで飛ばされる
            self._force_upgrade = True
        self.refresh()

    def start(self, *, upgrade: bool = False) -> None:
        """導入を始める ``upgrade`` を立てると、入っている版も新しい版へ入れ替える"""
        if self.busy or self._block_while_pip_is_left():
            return
        if upgrade:
            self._force_upgrade = True
        argv = self._command()
        # pip が上書きする前の読み込み済みの物を、この導入の分として控える
        self._before = snapshot_runtime_modules()
        self._log.setVisible(True)
        self._log.clear()
        self._progress.setVisible(True)
        self._button.setEnabled(False)
        self._check_button.setEnabled(False)
        self._extra.setEnabled(False)
        self._status.setText("導入しています 数分かかることがあります")

        # 中断の頼みと、終わった知らせを分ける 同じ旗にすると、中断した瞬間に
        # 「終わった」と読まれ、pip が走っている最中に成功の案内が出る
        done = threading.Event()
        cancel = threading.Event()
        self._done = done
        self._cancel = cancel
        # 終わるまでは成功ではない 前の導入の 0 が残っていると成功に見える
        self._code = -1

        def run() -> None:
            code = install_runtime(
                pack=self._pack,
                command=argv,
                on_output=self._log_queue.put,
                should_cancel=cancel.is_set,
            )
            self._code = code
            self._log_queue.put(install_result_text(code))
            done.set()

        threading.Thread(
            target=run, name=f"sashimono-install-{self._pack.key}", daemon=True
        ).start()
        self._timer.start()

    def cancel(self) -> None:
        """導入を止めるよう頼む 終わったかどうかはワーカーが知らせる"""
        if self._done is not None:
            self._cancel.set()

    def _poll(self) -> None:
        while True:
            try:
                self._log.appendPlainText(self._log_queue.get_nowait())
            except queue.Empty:
                break

        checking = self._checking
        if checking is not None and checking.is_set():
            self._finish_check()

        done = self._done
        if done is None or not done.is_set():
            if done is None and self._checking is None:
                self._timer.stop()
            return
        self._done = None
        if self._checking is None:
            self._timer.stop()
        self._progress.setVisible(False)
        self._button.setEnabled(True)
        succeeded = self._code == 0
        if succeeded:
            # 入れ替え終えた 前に尋ねた「更新があります」は古い話になる
            self._force_upgrade = False
            self._update_note = ""
        # 状態を見直す前に import の道を作り直す 先に見直すと、配布版では
        # 入れたばかりのものが見えず「未導入」のまま止まる
        loaded = refresh_runtime(self._before) if succeeded else ()
        self.refresh()
        self.note = ""
        if succeeded:
            status = self.status
            if status.ready:
                self.note = restart_note(loaded)
            elif not status.installed:
                self.note = restart_note(loaded, visible=False)
            # 入ったが外部コマンドが足りないときは「使えます」と言わない
            # 足りない物は refresh が出した summary に書いてある
            if self.note:
                self._status.setText(f"{self._status.text()}\n{self.note}")
        self.finished.emit(succeeded)

    def _command(self) -> list[str]:
        upgrade = self.status.needs_upgrade or self._force_upgrade
        return install_command(self._pack, extra=self.extra, upgrade=upgrade)


def describe_updates(
    packages: Sequence[PackageStatus], found: Mapping[str, Latest | None]
) -> tuple[str, bool]:
    """尋ねた答えを 1 行の文にする 戻り値の 2 つ目は、範囲の中に入れ替えられる版があるか

    範囲の外の新しい版（試していない大きな版上げ）は知らせるだけで、入れ替えない
    pip は指定の範囲を守るので、押しても入らない物を「〔環境を更新〕で入れ替えます」と言わない
    """
    newer: list[str] = []
    outside: list[str] = []
    current: list[str] = []
    unknown = False
    for package in packages:
        latest = found.get(package.name)
        if latest is None or latest.allowed is None:
            unknown = True
            continue
        name = Requirement(package.name).name
        installed = package.version
        if installed is None or is_newer_version(latest.allowed, installed):
            newer.append(f"{name} {installed or '未導入'} → {latest.allowed}")
        else:
            current.append(f"{name} {installed}")
        if latest.newest is not None and is_newer_version(latest.newest, latest.allowed):
            outside.append(f"{name} {latest.newest}")
    parts: list[str] = []
    if newer:
        parts.append(f"更新があります（{'、'.join(newer)}） 〔環境を更新〕で入れ替えます")
    elif unknown:
        parts.append("新しい版を確かめられませんでした（繋がっていないかもしれません）")
    else:
        parts.append(f"いちばん新しい版です（{'、'.join(current)}）")
    if outside:
        parts.append(
            f"さらに新しい版（{'、'.join(outside)}）もありますが、このソフトの版では試していない"
            "ため入れません ソフトの更新で入るようになります"
        )
    return "\n".join(parts), bool(newer)


def _readable(megabytes: int) -> str:
    return f"{megabytes / 1000:.1f} GB" if megabytes >= 1000 else f"{megabytes} MB"
