"""書き出しダイアログと、その進捗

書き出しは別スレッドで走らせる メインスレッドで回すと、数分間 UI が固まって
中止すらできなくなる
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.commands import export_range
from sashimono.core.model import Project, Scene, SceneId
from sashimono.core.timebase import format_timecode
from sashimono.engine.encode import (
    DEFAULT_PIPELINE_DEPTH,
    ExportError,
    ExportSettings,
    available_video_codecs,
)
from sashimono.engine.encode import export_project as run_export
from sashimono.engine.render import DEFAULT_DECODE_THREADS

__all__ = ["RANGE_ALL", "RANGE_WORK_AREA", "ExportDialog"]

#: コーデック名と、画面に出す説明
CODEC_LABELS = {
    "h264_nvenc": "H.264 (NVIDIA GPU)",
    "hevc_nvenc": "H.265 (NVIDIA GPU)",
    "av1_nvenc": "AV1 (NVIDIA GPU)",
    "h264_qsv": "H.264 (Intel GPU)",
    "libx264": "H.264 (CPU)",
    "libx265": "H.265 (CPU)",
}

#: 「自動」の項目に持たせる値 ``None`` は「コーデックが 1 つも無く書き出せない」項目が
#: 使っているので分ける 同じにすると、自動を選んで書き出しを押しても何も起きない
AUTO_CODEC = ""

#: 「書き出す範囲」の選びに持たせる値
RANGE_ALL = "all"
RANGE_WORK_AREA = "work_area"


class _ExportWorker(QObject):
    """別スレッドで書き出しを回す"""

    progressed = Signal(float)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, project: Project, settings: ExportSettings) -> None:
        super().__init__()
        self._project = project
        self._settings = settings
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            path = run_export(
                self._project,
                self._settings,
                progress=self.progressed.emit,
                should_cancel=self._cancel.is_set,
            )
        except ExportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"書き出しに失敗した: {exc}")
        else:
            self.finished.emit(str(path))


class ExportDialog(QDialog):
    """書き出しの設定と実行"""

    def __init__(
        self,
        project: Project,
        parent: QWidget | None = None,
        *,
        pipeline_depth: int = DEFAULT_PIPELINE_DEPTH,
        decode_threads: int = DEFAULT_DECODE_THREADS,
        scene_name: str | None = None,
    ) -> None:
        """``project`` はメインを持つプロジェクト全体

        ``scene_name`` はシーンを開いているときのその名前 書き出す物の既定はメインのまま
        （開いている物で出る物が変わると、書き出した物を取り違える） 選びにシーンを並べ、
        どのシーンでも書き出せる
        """
        super().__init__(parent)
        self.setWindowTitle("書き出し")
        self.setModal(True)
        self.resize(480, 260)

        #: メインを持つプロジェクト全体 選んだシーンはここからタイムラインを差し替えて作る
        self._whole = project
        #: 書き出すプロジェクト 選んだシーンのタイムラインをメインの所へ差し込んだ物
        self._project = project
        self._chosen: Scene | None = None
        # 画面には出さない 書き出しごとに変える物ではなく、その機械の持ち物なので
        # 本人の設定（表示 → 設定…）から来る
        self._pipeline_depth = pipeline_depth
        self._decode_threads = decode_threads
        self._thread: QThread | None = None
        self._worker: _ExportWorker | None = None

        default_name = f"{project.name}.mp4"
        self._path = QLineEdit(str(Path.home() / "Videos" / default_name), self)
        browse = QPushButton("参照…", self)
        browse.clicked.connect(self._choose_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path)
        path_row.addWidget(browse)

        self._codec = QComboBox(self)
        codecs = available_video_codecs()
        if codecs:
            # 既定は名指しせず書き出し側に選ばせる 先頭を名指しで渡すと、試しには開けても
            # 作品の大きさで断られたとき（NVENC は幅 4096 まで）に次の候補へ落ちられない
            first = CODEC_LABELS.get(codecs[0], codecs[0])
            self._codec.addItem(f"自動（{first}、使えなければ次の候補）", AUTO_CODEC)
        for name in codecs:
            self._codec.addItem(CODEC_LABELS.get(name, name), name)
        if not codecs:
            self._codec.addItem("利用できるコーデックが無い", None)
            self._codec.setEnabled(False)

        self._bitrate = QSpinBox(self)
        self._bitrate.setRange(1, 200)
        self._bitrate.setValue(12)
        self._bitrate.setSuffix(" Mbps")

        self._scene = QComboBox(self)
        self._scene.addItem("メイン", None)
        for scene in project.scenes:
            self._scene.addItem(f"シーン「{scene.name}」", scene.id)
        if not project.scenes:
            self._scene.setToolTip("シーンを作ると、ここでシーンだけを書き出せる")

        self._summary = QLabel(self)
        self._range = QComboBox(self)
        self._empty = QLabel("タイムラインが空なので書き出せない", self)
        self._codecs_ready = bool(codecs)

        form = QFormLayout()
        form.addRow("出力先", path_row)
        form.addRow("書き出す物", self._scene)
        form.addRow("コーデック", self._codec)
        form.addRow("ビットレート", self._bitrate)
        form.addRow("内容", self._summary)
        form.addRow("書き出す範囲", self._range)
        if scene_name is not None:
            # タイムラインに見えているのはシーンの範囲 既定で書き出すのはメインなので、
            # 見えている帯と違う所が出ても驚かないように、ここで言う
            note = QLabel(
                f"シーン「{scene_name}」を編集中 既定で書き出すのはメインのタイムライン"
                " シーンを書き出すときは「書き出す物」で選ぶ",
                self,
            )
            note.setWordWrap(True)
            form.addRow("", note)
        form.addRow("", self._empty)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1000)
        self._progress.setVisible(False)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("書き出し")
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("閉じる")
        self._buttons.accepted.connect(self._start)
        self._buttons.rejected.connect(self._cancel_or_close)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._progress)
        layout.addStretch(1)
        layout.addWidget(self._buttons)

        self._scene.currentIndexChanged.connect(self._on_scene_changed)
        self._on_scene_changed()

    def select_scene(self, scene_id: SceneId | None) -> None:
        """書き出す物を選ぶ ``None`` ならメイン 無いシーンなら何もしない"""
        index = self._scene.findData(scene_id)
        if index >= 0:
            self._scene.setCurrentIndex(index)

    @property
    def target(self) -> Project:
        """書き出すプロジェクト 選んだシーンのタイムラインをメインの所へ差し込んだ物"""
        return self._project

    def _on_scene_changed(self) -> None:
        """選んだシーンに合わせて、長さ・範囲・出力先の名前・押せるかを作り直す

        範囲はシーンごとに持つ（シーンのタイムラインで指定した範囲） メインの範囲を
        シーンへ当てると、シーンの長さの外を指して何も映らない
        """
        scene_id = self._scene.currentData()
        scene = self._whole.find_scene(scene_id) if scene_id is not None else None
        before = self._default_name(self._chosen)
        self._chosen = scene
        self._project = (
            self._whole if scene is None else replace(self._whole, timeline=scene.timeline)
        )
        project = self._project
        width, height = project.settings.resolution
        self._summary.setText(
            f"{width}x{height} / {project.settings.frame_rate} fps / {project.duration} フレーム"
        )
        self._fill_range(project)
        # 出力先の名前を書き換えるのは、前の既定の名前のままのときだけ 手で決めた名前は残す
        path = Path(self._path.text())
        if path.name == before:
            self._path.setText(str(path.with_name(self._default_name(scene))))
        empty = project.duration <= 0
        self._empty.setVisible(empty)
        # 押せるままにすると、押しても何も起きず理由が分からない 理由はコーデックの欄と
        # 空の断りに出ている
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._codecs_ready and not empty
        )

    def _default_name(self, scene: Scene | None) -> str:
        """出力先の既定の名前 シーンならシーンの名前を後ろに付ける（メインと上書きし合わない）"""
        if scene is None:
            return f"{self._whole.name}.mp4"
        # シーンの名前は自由に付けられる ファイル名に使えない文字を残すと、書き出しを
        # 押してから開けないと断られる
        safe = "".join("_" if c in '\\/:*?"<>|' else c for c in scene.name)
        return f"{self._whole.name}_{safe}.mp4"

    def _fill_range(self, project: Project) -> None:
        """全体か、タイムラインで指定した範囲か

        範囲があれば既定は範囲の側 目盛りに帯を引いた人は、その所を出したくて引いている
        全体を既定にすると、範囲を決めてから書き出しを開いた人が毎回選び直すことになり、
        選び忘れると長い全体を書き出して待たされる 決めたまま忘れていた人のために、
        範囲の位置と長さを選びの中に書いて、全体へ戻せるようにしておく
        """
        rate = project.settings.frame_rate
        choice = self._range
        choice.clear()
        choice.setToolTip("")
        choice.addItem(f"全体（{project.duration} フレーム）", RANGE_ALL)
        area = export_range(project.timeline)
        if area is not None:
            start, end = area
            choice.addItem(
                f"指定した範囲（{format_timecode(start, rate)} 〜 {format_timecode(end, rate)}、"
                f"{end - start} フレーム）",
                RANGE_WORK_AREA,
            )
            choice.setCurrentIndex(1)
        elif project.timeline.work_area is not None:
            # 範囲が丸ごとタイムラインの終わりより後ろにある 選べる形で出すと、
            # 何も映らない所を書き出すことになるので、理由だけ見せて選ばせない
            choice.addItem("指定した範囲（タイムラインの終わりより後ろなので使えない）", None)
            model = choice.model()
            item = model.item(1) if isinstance(model, QStandardItemModel) else None
            if item is not None:
                item.setEnabled(False)
        else:
            choice.setToolTip("タイムラインの目盛りを Shift+ドラッグすると範囲を指定できる")

    def _frame_range(self) -> tuple[int, int] | None:
        """選んだ範囲 全体なら ``None``（書き出し側がタイムラインの長さを使う）"""
        if self._range.currentData() == RANGE_WORK_AREA:
            return export_range(self._project.timeline)
        return None

    def _choose_path(self) -> None:
        name, _ = QFileDialog.getSaveFileName(
            self, "書き出し先", self._path.text(), "MP4 (*.mp4);;すべてのファイル (*)"
        )
        if name:
            self._path.setText(name)

    def _settings(self) -> ExportSettings | None:
        """画面の選択から書き出しの設定を作る 使えるコーデックが無ければ ``None``"""
        codec = self._codec.currentData()
        if codec is None:
            return None
        return ExportSettings(
            path=Path(self._path.text()),
            video_codec=str(codec) or None,
            video_bitrate=self._bitrate.value() * 1_000_000,
            frame_range=self._frame_range(),
            pipeline_depth=self._pipeline_depth,
            decode_threads=self._decode_threads,
        )

    def _start(self) -> None:
        if self._thread is not None:
            return
        settings = self._settings()
        if settings is None:
            return

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("中止")

        self._worker = _ExportWorker(self._project, settings)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _cancel_or_close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            return
        self.reject()

    def _on_progress(self, value: float) -> None:
        self._progress.setValue(int(value * 1000))

    def _on_finished(self, path: str) -> None:
        self._teardown()
        QMessageBox.information(self, "書き出し", f"書き出しました\n{path}")
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._teardown()
        QMessageBox.warning(self, "書き出し", message)

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None
        self._progress.setVisible(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("閉じる")

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt の命名規約
        # 書き出し中に閉じられたら、スレッドを畳んでから終わる
        # 放置すると Qt がスレッドの生存中に破棄されたと言って落ちる
        if self._worker is not None:
            self._worker.cancel()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(5000)
        super().closeEvent(event)  # type: ignore[arg-type]
