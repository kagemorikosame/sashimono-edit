"""起こしの実行ダイアログと、実行環境の導入

字幕起こしの依存は合計で 2 GB を超えるので、**初期状態では入っていない** この
ダイアログが未導入を検出したときは、起こしのボタンの代わりに「環境を導入」を出す
入れるものと実行するコマンドをそのまま画面に見せてから始める 何が入るのか
分からないまま数分のダウンロードが走るのは、それ自体が不具合に見える

導入も起こしもワーカースレッドで動く ウィジェットに触るのはタイマーで拾った
メインスレッド側だけにしてある
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.asr import (
    MODELS,
    JobKind,
    TranscribeOptions,
    TranscriptionService,
    install_command,
    install_runtime,
    runtime_status,
)
from sashimono.asr.service import Job
from sashimono.core.model import MediaItem, Transcript
from sashimono.runtime import refresh_runtime, restart_note, snapshot_runtime_modules
from sashimono.ui.theme import Colors

__all__ = ["TranscribeDialog"]

#: ワーカーからの知らせを拾う間隔（ミリ秒）
POLL_MS = 100

#: 選べる言語 自動判定は精度が落ちるので、既定は日本語にしておく
LANGUAGES: tuple[tuple[str | None, str], ...] = (
    ("ja", "日本語"),
    ("en", "英語"),
    (None, "自動判定"),
)


class TranscribeDialog(QDialog):
    """1 つの素材を起こす

    結果は :attr:`transcript` に入る 呼び出し側がそれをコマンドにして履歴へ載せる
    ここでプロジェクトを書き換えないのは、UI と AI が同じ入口を通るという方針を
    崩さないため
    """

    def __init__(
        self,
        media: MediaItem,
        service: TranscriptionService,
        parent: QWidget | None = None,
        *,
        stream: int | None = None,
    ) -> None:
        """``stream`` は初めに選んでおく音声ストリームの番号（選んだクリップが鳴らす音）"""
        super().__init__(parent)
        self.setWindowTitle(f"字幕起こし — {media.name}")
        self.resize(560, 420)

        self._media = media
        self._first_stream = stream
        self._service = service
        self._job: Job | None = None
        self.transcript: Transcript | None = None
        #: 起こした音声ストリームの番号 呼ぶ側はこの音の字幕へ取り込む
        self.chosen_stream: int | None = None
        #: 既に字幕がある音を起こし直すときに、置き換えてよいかを尋ねる 試験で差し替える
        self.confirm_replace: Callable[[str], bool] = self._ask_replace
        #: 起こせたが知らせておくこと（GPU の道具が読めず CPU で起こした など） 窓を閉じた
        #: 後に呼ぶ側が出す
        self.notice = ""

        #: 導入ワーカーからのログ スレッドをまたぐのでキューで受ける
        self._install_log: queue.Queue[str] = queue.Queue()
        self._install_done: threading.Event | None = None
        #: 導入の中断の頼み 終わった知らせ（_install_done）とは分けて持つ
        self._install_cancel = threading.Event()
        self._install_code = 0
        #: この導入を始める前に、専用フォルダから読み込み済みだった物
        self._before: dict[str, int] = {}

        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._refresh_availability()

    # --- 組み立て ---

    def _build(self) -> None:
        self._model = QComboBox(self)
        for info in MODELS:
            self._model.addItem(info.describe(), info.name)

        self._language = QComboBox(self)
        for code, label in LANGUAGES:
            self._language.addItem(label, code)

        self._prompt = QLineEdit(self)
        self._prompt.setPlaceholderText("固有名詞など（任意）")

        self._words = QCheckBox("単語ごとの時刻も取る（分割の精度が上がる・遅くなる）", self)
        self._gpu = QCheckBox("GPU を使う", self)
        self._gpu.setChecked(True)
        self._gpu.toggled.connect(self._describe_install)

        # 音声が何本もある素材（ゲームの音とマイクの声など）は、どれを起こすかを選ぶ
        # 番号はタイムラインの札（音声 N）と同じく音声ストリームの並びで 1 から数える
        self._stream = QComboBox(self)
        stream_choice = self._first_stream
        for number, stream in enumerate(self._media.audio_streams, start=1):
            detail = f"{stream.channels}ch {stream.sample_rate // 1000}kHz"
            if stream.language:
                detail += f" {stream.language}"
            self._stream.addItem(f"音声 {number}（{detail}）", stream.index)
        chosen = self._stream.findData(stream_choice) if stream_choice is not None else -1
        if chosen >= 0:
            self._stream.setCurrentIndex(chosen)

        form = QFormLayout()
        form.addRow("モデル", self._model)
        if len(self._media.audio_streams) > 1:
            form.addRow("起こす音声", self._stream)
        else:
            # 行に置かない選びも窓の子なので、隠さないと窓の左上（0, 0）に浮いて
            # 「モデル」の行を潰す（利用者の画面） 選ぶ物が 1 本なら要らない
            self._stream.hide()
        form.addRow("言語", self._language)
        form.addRow("ヒント", self._prompt)
        form.addRow("", self._words)
        form.addRow("", self._gpu)

        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()};")

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        self._log.setMaximumBlockCount(2000)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1000)
        self._progress.setVisible(False)

        self._install_button = QPushButton("環境を導入", self)
        self._install_button.clicked.connect(self._start_install)
        self._run_button = QPushButton("起こす", self)
        self._run_button.setDefault(True)
        self._run_button.clicked.connect(self._start_transcribe)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        buttons.rejected.connect(self.reject)
        self._cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if self._cancel_button is not None:
            # 既定の文言は環境の言語に従うので、ここで日本語に固定する
            self._cancel_button.setText("閉じる")

        actions = QHBoxLayout()
        actions.addWidget(self._install_button)
        actions.addStretch(1)
        actions.addWidget(self._run_button)
        actions.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        layout.addWidget(self._log, 1)
        layout.addLayout(actions)

    # --- 状態 ---

    def _refresh_availability(self) -> None:
        """導入状況を見て、押せるボタンを決める"""
        status = runtime_status()
        self._run_button.setEnabled(status.installed)
        self._install_button.setEnabled(True)
        # いつも触れるようにする ここが「GPU 版を入れるか」の選択を兼ねていて、切れば
        # CUDA ランタイム（2 GB 弱）を落とさずに済む 導入済みで CUDA ランタイムが無いときも
        # 印を付けて「環境を更新」を押せば足せる（前は押せず、後から GPU 版にする道が無かった）

        if status.installed:
            self._install_button.setText("環境を更新")
            if not status.extra_installed:
                self._gpu.setChecked(False)
            self._describe_install()
            return

        self._install_button.setText("環境を導入")
        self._describe_install()

    def _describe_install(self) -> None:
        """これから入るものを出す 何が落ちてくるのか分かってから始められるように"""
        status = runtime_status()
        cuda = self._gpu.isChecked()
        if status.installed:
            if cuda and status.extras and not status.extra_installed:
                gigabytes = status.pack.extra_size_mb / 1000
                self._status.setText(
                    f"{status.summary()}\nGPU で起こすには「環境を更新」で"
                    f" {status.pack.extra_label}（約 {gigabytes:.1f} GB）を入れてください"
                    " 入れずに起こすと CPU で起こします"
                )
            else:
                self._status.setText(status.summary())
            return
        packages = "、".join(status.missing(extra=cuda))
        size = "2 GB" if cuda else "300 MB"
        self._status.setText(
            f"{status.summary()}\n入れるもの: {packages}\n"
            f"初回は {size} ほどのダウンロードがあります"
        )

    def _set_busy(self, busy: bool, *, message: str = "") -> None:
        self._run_button.setEnabled(not busy and runtime_status().ready)
        self._install_button.setEnabled(not busy)
        self._model.setEnabled(not busy)
        self._language.setEnabled(not busy)
        self._progress.setVisible(busy)
        if self._cancel_button is not None:
            self._cancel_button.setText("中断" if busy else "閉じる")
        if message:
            self._status.setText(message)

    # --- 導入 ---

    def _start_install(self) -> None:
        command = install_command(
            cuda=self._gpu.isChecked(), upgrade=runtime_status().needs_upgrade
        )
        # pip が上書きする前の読み込み済みの物を、この導入の分として控える
        # 導入ごとに持つので、アシスタントの導入と重なっても控えが混ざらない
        self._before = snapshot_runtime_modules()
        self._log.setVisible(True)
        self._log.clear()
        self._set_busy(True, message="導入しています 数分かかります")
        self._progress.setRange(0, 0)  # 進み具合が分からないので流れる表示にする

        # 中断の頼みと、終わった知らせを分ける 同じ旗にすると、中断した瞬間に
        # 「終わった」と読まれ、pip が走っている最中に成功の案内が出る
        done = threading.Event()
        cancel = threading.Event()
        self._install_done = done
        self._install_cancel = cancel
        # 終わるまでは成功ではない 前の導入の 0 が残っていると成功に見える
        self._install_code = -1

        def run() -> None:
            code = install_runtime(
                command=command, on_output=self._install_log.put, should_cancel=cancel.is_set
            )
            self._install_code = code
            self._install_log.put(
                "導入が完了しました" if code == 0 else f"導入に失敗しました（コード {code}）"
            )
            done.set()

        threading.Thread(target=run, name="sashimono-asr-install", daemon=True).start()
        self._timer.start()

    # --- 起こし ---

    def _ask_replace(self, name: str) -> bool:
        answer = QMessageBox.question(
            self,
            "字幕起こし",
            f"{name} には字幕があります 起こした結果で置き換えますか"
            "（直した字幕も置き換わります 取り消しで戻せます）",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _start_transcribe(self) -> None:
        stream = self._stream.currentData()
        if self._media.transcript_for(stream) is not None:
            number = max(0, self._stream.currentIndex()) + 1
            name = (
                f"{self._media.name} の音声 {number}"
                if len(self._media.audio_streams) > 1
                else self._media.name
            )
            if not self.confirm_replace(name):
                return
        # AI から頼んだ起こしと同じ音なら重ねない（同じ列で番号をそろえて見る）
        key = self._media.transcript_stream(stream)
        if self._service.find(self._media.id, key) is not None:
            self._status.setText(
                "その音声はもう起こしています（順番待ちを含む） 終わるのを待ってください"
            )
            return
        stream = key
        self.chosen_stream = stream
        options = TranscribeOptions(
            model=str(self._model.currentData()),
            language=self._language.currentData(),
            device="cuda" if self._gpu.isChecked() else "cpu",
            compute_type="float16" if self._gpu.isChecked() else "int8",
            word_timestamps=self._words.isChecked(),
            initial_prompt=self._prompt.text().strip(),
            audio_stream=stream,
        )
        # 走っている起こし（AI から頼んだ物など）があれば順番待ちに入る 同時には走らせない
        waiting = self._service.busy
        self._job = self._service.start(self._media.id, self._media.path, options)

        self._progress.setRange(0, 1000)
        self._set_busy(
            True,
            message=(
                "順番待ちです 前の起こしが終わると始まります"
                if waiting
                else "起こしています 初回はモデルの取得に時間がかかります"
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
        # ボタンの有効・無効を決め直す前に import の道を作り直す 先に決めると、
        # 配布版では入れたばかりの faster-whisper が見えず「起こす」が押せないまま残る
        loaded = refresh_runtime(self._before) if self._install_code == 0 else ()
        self._set_busy(False)
        self._refresh_availability()
        if self._install_code == 0:
            self._status.setText(restart_note(loaded, visible=runtime_status().installed))

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
            if event.kind is JobKind.DONE and event.transcript is not None:
                self.transcript = event.transcript
                self.notice = event.notice
                self.accept()
            else:
                self._status.setText(event.message or "終了した")
            return

    # --- 終了 ---

    def reject(self) -> None:
        """中断 走っているものがあれば止めてから閉じる

        起こしは GPU を占有する 閉じたのに裏で回り続けると、次の操作が刺さる
        """
        if self._job is not None and self._job.waiting:
            # 順番待ちのまま止めた物は走らせない（列が飛ばす） 知らせを待つと、前の起こしが
            # 終わるまで（数分）窓を閉じられず、その間は編集もできない（PR #231 の指摘）
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
            # 止めるよう頼むだけ 終わったかどうかはワーカーが知らせる
            self._install_cancel.set()
            self._status.setText("中断しています")
            return
        self._timer.stop()
        super().reject()
