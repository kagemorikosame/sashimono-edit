"""話者の分離の窓と、部品の導入（#323）

部品（sherpa-onnx 約 25 MB とモデル 28 MB）は使う人だけが入れる（2026-10-20 本人の決定）
入っていなければ分けるボタンの代わりに「部品を入れる」を出し、入れる物と大きさを先に見せる
字幕起こしの窓（:mod:`sashimono.ui.subtitle.transcribe_dialog`）と同じ作り

分けた結果（話者の区間）は :attr:`SpeakerDialog.turns` に入れて閉じる 字幕へ当てるのは
呼ぶ側（字幕パネル）で、窓はプロジェクトを書き換えない（UI と AI が同じ入口を通るため）
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.asr import JobKind, TranscriptionService
from sashimono.asr.service import Job
from sashimono.asr.speakers import (
    MAX_SPEAKERS,
    MODEL_SIZE_MB,
    SPEAKER_PACK,
    SpeakerOptions,
    install_command,
    runtime_status,
)
from sashimono.core.model import MediaItem
from sashimono.core.speakers import SpeakerTurn
from sashimono.runtime import (
    LEFT_RUNNING_NOTE,
    PackStatus,
    install_result_text,
    install_runtime,
    pip_left_running,
    refresh_runtime,
    restart_note,
    snapshot_runtime_modules,
)
from sashimono.ui.theme import Colors, themed_style

__all__ = ["SpeakerDialog"]

#: ワーカーからの知らせを拾う間隔（ミリ秒）
POLL_MS = 100


class SpeakerDialog(QDialog):
    """1 つの素材の 1 本の音声を、話者ごとに分ける"""

    def __init__(
        self,
        media: MediaItem,
        service: TranscriptionService,
        parent: QWidget | None = None,
        *,
        stream: int | None = None,
        status: Callable[[], PackStatus] | None = None,
    ) -> None:
        """``stream`` は初めに選んでおく音声ストリームの番号（字幕パネルで見ている音）

        ``status`` は部品の導入状況を返す関数 省けば今の機械を調べる（試験が差し替える）
        """
        super().__init__(parent)
        self.setWindowTitle(f"話者を分ける — {media.name}")
        self.resize(520, 360)
        self._media = media
        self._first_stream = stream
        self._runtime_status: Callable[[], PackStatus] = (
            status if status is not None else lambda: runtime_status()
        )
        self._service = service
        self._job: Job | None = None
        #: 分けた結果 呼ぶ側がこれを字幕へ当てる
        self.turns: tuple[SpeakerTurn, ...] | None = None
        #: 分けた音声ストリームの番号 呼ぶ側はこの音の字幕へ当てる
        self.chosen_stream: int | None = None

        self._install_log: queue.Queue[str] = queue.Queue()
        self._install_done: threading.Event | None = None
        self._install_cancel = threading.Event()
        self._install_code = 0
        self._before: dict[str, int] = {}

        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._refresh_availability()

    # --- 組み立て ---

    def _build(self) -> None:
        self._count = QComboBox(self)
        self._count.setObjectName("speaker_count")
        # 既定は自動 人数を知っているなら渡した方が分かれ方が安定する
        self._count.addItem("自動で決める", None)
        for number in range(2, MAX_SPEAKERS + 1):
            self._count.addItem(f"{number} 人", number)

        self._stream = QComboBox(self)
        for number, info in enumerate(self._media.audio_streams, start=1):
            self._stream.addItem(f"音声 {number}", info.index)
        chosen = self._stream.findData(self._media.transcript_stream(self._first_stream))
        if chosen >= 0:
            self._stream.setCurrentIndex(chosen)

        form = QFormLayout()
        form.addRow("話す人の数", self._count)
        if len(self._media.audio_streams) > 1:
            form.addRow("分ける音声", self._stream)
        else:
            # 行に置かない選びも窓の子なので、隠さないと窓の左上に浮く
            self._stream.hide()

        note = QLabel(
            "音は外へ送らず、この機械で分けます 字幕 1 枚ごとに、重なりのいちばん長い人を"
            "付けます（字幕は割りません） 名前は字幕パネルの右クリックで付け替えられます",
            self,
        )
        note.setWordWrap(True)

        self._status = QLabel(self)
        self._status.setWordWrap(True)
        themed_style(self._status, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        self._log.setMaximumBlockCount(2000)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1000)
        self._progress.setVisible(False)

        self._install_button = QPushButton("部品を入れる", self)
        self._install_button.clicked.connect(self._start_install)
        self._run_button = QPushButton("分ける", self)
        self._run_button.setDefault(True)
        self._run_button.clicked.connect(self._start)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        buttons.rejected.connect(self.reject)
        self._cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if self._cancel_button is not None:
            self._cancel_button.setText("閉じる")

        actions = QHBoxLayout()
        actions.addWidget(self._install_button)
        actions.addStretch(1)
        actions.addWidget(self._run_button)
        actions.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        layout.addWidget(self._log, 1)
        layout.addLayout(actions)

    # --- 状態 ---

    def _refresh_availability(self) -> None:
        status = self._runtime_status()
        # installed ではなく ready を見る 別の Python 向けに入った物は読めずに落ちる
        self._run_button.setEnabled(status.ready)
        self._install_button.setEnabled(not pip_left_running())
        if status.installed:
            self._install_button.setText("部品を更新")
            self._status.setText(status.summary())
        else:
            self._install_button.setText("部品を入れる")
            packages = "、".join(status.missing(extra=False))
            self._status.setText(
                f"{status.summary()}\n入れるもの: {packages}\n"
                f"部品は約 {SPEAKER_PACK.size_mb} MB 初めて分けるときにモデルを"
                f"約 {MODEL_SIZE_MB} MB 落とします"
            )
        if pip_left_running():
            self._install_button.setEnabled(False)
            self._status.setText(f"{self._status.text()}\n{LEFT_RUNNING_NOTE}")

    def _set_busy(self, busy: bool, *, message: str = "") -> None:
        self._run_button.setEnabled(not busy and self._runtime_status().ready)
        self._install_button.setEnabled(not busy and not pip_left_running())
        self._count.setEnabled(not busy)
        self._stream.setEnabled(not busy)
        self._progress.setVisible(busy)
        if self._cancel_button is not None:
            self._cancel_button.setText("中断" if busy else "閉じる")
        if message:
            self._status.setText(message)

    # --- 導入 ---

    def _start_install(self) -> None:
        if pip_left_running():
            self._refresh_availability()
            return
        command = install_command(upgrade=self._runtime_status().needs_upgrade)
        self._before = snapshot_runtime_modules()
        self._log.setVisible(True)
        self._log.clear()
        self._set_busy(True, message="部品を入れています")
        self._progress.setRange(0, 0)
        done = threading.Event()
        cancel = threading.Event()
        self._install_done = done
        self._install_cancel = cancel
        self._install_code = -1

        def run() -> None:
            # 終わった印は finally で立てる 思わぬ例外で印が立たないと、窓が導入中のまま閉じられない
            code = 1
            try:
                code = install_runtime(
                    pack=SPEAKER_PACK,
                    command=command,
                    on_output=self._install_log.put,
                    should_cancel=cancel.is_set,
                )
            except Exception as exc:  # 失敗として出す 画面のスレッドへは出さない
                self._install_log.put(f"導入の途中で落ちました: {type(exc).__name__}: {exc}")
            finally:
                self._install_code = code
                self._install_log.put(install_result_text(code))
                done.set()

        threading.Thread(target=run, name="sashimono-speaker-install", daemon=True).start()
        self._timer.start()

    # --- 分ける ---

    def _start(self) -> None:
        stream = self._media.transcript_stream(self._stream.currentData())
        if self._service.find(self._media.id, stream, separating=True) is not None:
            self._status.setText(
                "その音声はもう分けています（順番待ちを含む） 終わるのを待ってください"
            )
            return
        self.chosen_stream = stream
        options = SpeakerOptions(speakers=self._count.currentData(), audio_stream=stream)
        waiting = self._service.busy
        self._job = self._service.start_speakers(self._media.id, self._media.path, options)
        self._progress.setRange(0, 1000)
        self._set_busy(
            True,
            message=(
                "順番待ちです 前の起こし・分離が終わると始まります"
                if waiting
                else "話者を分けています"
            ),
        )
        self._timer.start()

    # --- ワーカーの見張り ---

    def _poll(self) -> None:
        self._drain_install_log()
        self._drain_job()

    def _drain_install_log(self) -> None:
        while True:
            try:
                self._log.appendPlainText(self._install_log.get_nowait())
            except queue.Empty:
                break
        done = self._install_done
        if done is None or not done.is_set():
            return
        self._install_done = None
        self._timer.stop()
        self._progress.setRange(0, 1000)
        # 押せるボタンを決め直す前に import の道を作り直す 先に決めると、配布版では入れた
        # ばかりの部品が見えず「分ける」が押せないまま残る
        loaded = refresh_runtime(self._before) if self._install_code == 0 else ()
        self._set_busy(False)
        self._refresh_availability()
        if self._install_code == 0:
            self._status.setText(restart_note(loaded, visible=self._runtime_status().installed))

    def _drain_job(self) -> None:
        job = self._job
        if job is None:
            return
        for event in job.poll():
            if event.kind is JobKind.PROGRESS:
                self._progress.setValue(int(event.ratio * 1000))
                self._status.setText(event.message)
                continue
            self._timer.stop()
            self._job = None
            self._set_busy(False)
            if event.kind is JobKind.DONE and event.speakers is not None:
                self.turns = event.speakers
                self.accept()
            else:
                self._status.setText(event.message or "終了した")
            return

    # --- 終了 ---

    def reject(self) -> None:
        """中断 走っている物があれば止めてから閉じる"""
        if self._job is not None and self._job.waiting:
            self._job.cancel()
            self._job = None
            self._timer.stop()
            super().reject()
            return
        if self._job is not None:
            self._job.cancel()
            self._status.setText("中断しています")
            return
        if self._install_done is not None:
            self._install_cancel.set()
            self._status.setText("中断しています")
            return
        self._timer.stop()
        super().reject()
