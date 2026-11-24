"""書き出しダイアログと、その進捗

書き出しは別スレッドで走らせる メインスレッドで回すと、数分間 UI が固まって
中止すらできなくなる
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
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
    AUDIO_KBPS_CHOICES,
    BITRATE_RANGE_MBPS,
    DEFAULT_AUDIO_KBPS,
    DEFAULT_FORMAT,
    DEFAULT_PIPELINE_DEPTH,
    FORMATS,
    RANGE_ALL,
    RANGE_WORK_AREA,
    WEB_AUTOMATIC_CODECS,
    BitrateChoice,
    ExportError,
    ExportFormat,
    ExportSettings,
    available_video_codecs,
    available_video_codecs_now,
    available_web_codecs,
    available_web_codecs_now,
    build_settings,
    chosen_range,
    export_target,
    format_named,
    format_of,
    has_sound,
    with_format_suffix,
)

# 試験が差し替える名前は前のまま（書き出しの偽物を ``run_export`` へ当てる）
from sashimono.engine.encode.export_job import export_anywhere as run_export
from sashimono.engine.encode.formats import VIDEO_WEB
from sashimono.engine.encoders import PRIORITY_AUTO
from sashimono.engine.render import DEFAULT_DECODE_THREADS
from sashimono.ui.system_clipboard import clipboard

__all__ = [
    "CHECKING_CODECS",
    "CODEC_LABELS",
    "CREDIT_NOTE",
    "RANGE_ALL",
    "RANGE_WORK_AREA",
    "ExportDialog",
    "credit_text",
]


#: 作った声の表記（#335）を知らせる文 書き出しの窓と、書き終えた知らせで同じ文を使う
CREDIT_NOTE = "この作品は音声合成の声を使っています 動画の概要欄などに次の表記を入れてください"


def credit_text(credits: tuple[str, ...], separator: str = "、") -> str:
    """作った声の表記（#335）を 1 つの文にする 概要欄へ貼るときは改行で区切る"""
    return separator.join(credits)


#: コーデック名と、画面に出す説明 AI の書き出し（#327）の確かめにも同じ名前を出す
CODEC_LABELS = {
    "h264_nvenc": "H.264 (NVIDIA GPU)",
    "hevc_nvenc": "H.265 (NVIDIA GPU)",
    "av1_nvenc": "AV1 (NVIDIA GPU)",
    "h264_amf": "H.264 (AMD GPU)",
    "hevc_amf": "H.265 (AMD GPU)",
    "h264_qsv": "H.264 (Intel GPU)",
    "libx264": "H.264 (CPU)",
    "libx265": "H.265 (CPU)",
    "libvpx-vp9": "VP9 (CPU)",
    "libsvtav1": "AV1 (CPU)",
}

#: 「自動」の項目に持たせる値 ``None`` は「コーデックが 1 つも無く書き出せない」項目が
#: 使っているので分ける 同じにすると、自動を選んで書き出しを押しても何も起きない
AUTO_CODEC = ""

#: ほかの確かめが走っている間、「コーデック」の欄に出す文（#464）
CHECKING_CODECS = "使えるエンコーダを確かめています…"

#: 音声だけの形式で鳴る物が無いときに出す断り（#527） 押せるままだと、押してから断られる
NO_SOUND = "鳴る物が無いので、音声だけの形式では書き出せない 音のクリップを置くか、映像の形式を選ぶ"

_log = logging.getLogger(__name__)


class _ExportWorker(QObject):
    """別スレッドで書き出しを回す"""

    progressed = Signal(float)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self, project: Project, settings: ExportSettings, *, separate_process: bool = False
    ) -> None:
        super().__init__()
        self._project = project
        self._settings = settings
        self._separate_process = separate_process
        self._cancel = threading.Event()
        #: 書き出し側からの知らせ（次の候補のエンコーダで書き直したなど #365）
        #: 書き出しのスレッドが足し、終わりの知らせ（finished）が届いてから画面のスレッドが読む
        self.notes: list[str] = []

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            path = run_export(
                self._project,
                self._settings,
                separate_process=self._separate_process,
                progress=self.progressed.emit,
                should_cancel=self._cancel.is_set,
                notice=self.notes.append,
            )
        except ExportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"書き出しに失敗した: {exc}")
        else:
            self.finished.emit(str(path))


#: 素材に合わせるビットレートを同時に求める数の上限（どの窓からも合わせて） 応えないネットワークの
#: 共有で止まった計算が、シーンを替えたり窓を開き直したりするたびに増えて残らないようにする
_BITRATE_SLOTS = threading.BoundedSemaphore(2)


class ExportDialog(QDialog):
    """書き出しの設定と実行"""

    #: 素材に合わせたビットレートが出た（頼んだ印、Mbps） 読むのはスレッドの外
    _bitrate_found = Signal(object, int)
    #: 裏で確かめ終えた候補を、窓のスレッドで並べる（形式の映像の作り方, 候補）
    _codecs_found = Signal(object, object)

    def __init__(
        self,
        project: Project,
        parent: QWidget | None = None,
        *,
        pipeline_depth: int = DEFAULT_PIPELINE_DEPTH,
        decode_threads: int = DEFAULT_DECODE_THREADS,
        scene_name: str | None = None,
        smooth_history: bool = True,
        gpu_color: bool = True,
        folder: Path | None = None,
        chapters: bool = True,
        encoder_priority: str = PRIORITY_AUTO,
        bitrate: BitrateChoice | None = None,
        separate_process: bool = False,
        export_format: str = DEFAULT_FORMAT,
    ) -> None:
        """``project`` はメインを持つプロジェクト全体

        ``export_format`` は前にこの窓で選んだ形式（#527 ``Preferences.export_format``）
        知らない名前なら MP4 呼んだ側は :attr:`chosen_format` を覚えて次に渡す

        ``bitrate`` はビットレートの欄の初めの値の決め方（本人の設定 #414） 省くと前からの決め打ち
        ``separate_process`` は別のプロセスで書き出すか（本人の設定 ``export_process`` #453）

        ``encoder_priority`` はどの作り手のエンコーダを先に試すか（本人の設定 #365）
        コーデックの並びと「自動」の中身がこれに従う

        ``folder`` は前に書き出したフォルダ（本人の設定 ``export_folder``） 無ければ
        ビデオのフォルダから始める

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
        self._smooth_history = smooth_history
        #: 色変換を GPU で行うか（#372） 機械の駆動との相性で決まる物なので、ここも本人の設定から
        self._gpu_color = gpu_color
        self._separate_process = separate_process
        self._thread: QThread | None = None
        self._worker: _ExportWorker | None = None
        #: 書き出し中に閉じるよう頼まれた 止まった知らせが来たら閉じる
        self._closing = False
        #: 書き出しを始めた先のフォルダ 人がこの窓で選んで書き出した所なので、呼んだ側が
        #: 覚えて、AI アシスタントが書ける場所に入れる（#350） 始めなければ ``None``
        self.started_folder: Path | None = None

        try:
            chosen = format_named(export_format)
        except ValueError:
            # 新しい版で足した形式を古い版で開いた 何も選ばれていない窓にしない
            chosen = format_named(DEFAULT_FORMAT)
        #: 今の形式 名前の拡張子・コーデックの並び・出す欄がこれに従う
        self._chosen_format = chosen
        #: 形式の欄 窓に出す順は形式の表のまま（映像を先に、音声だけを後ろに）
        self._format = QComboBox(self)
        for item in FORMATS:
            self._format.addItem(item.label, item.key)
        self._format.setCurrentIndex(self._format.findData(chosen.key))
        self._format.setToolTip(
            "書き出す入れ物 音声だけの形式は映像を描かずに音だけを書くので、映像付きより速い"
        )

        default_name = f"{project.name}{chosen.suffix}"
        start = folder if folder is not None else Path.home() / "Videos"
        self._path = QLineEdit(str(start / default_name), self)
        # 名前を打ち替えて拡張子を変えたら形式も合わせる 窓の形式のまま書き出すと、
        # 打った拡張子が黙って付け替わる setText（窓の側の書き換え）では動かない
        self._path.textEdited.connect(self._on_path_edited)
        browse = QPushButton("参照…", self)
        browse.clicked.connect(self._choose_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path)
        path_row.addWidget(browse)

        self._codec = QComboBox(self)
        self._codecs_ready = False
        self._encoder_priority = encoder_priority
        self._codecs_found.connect(self._on_codecs_found)
        self._load_codecs()

        #: 音声だけの形式の音質（MP3・M4A） 窓の中で選ぶ物で、本人の設定には覚えない
        #: 既定は知らない人が困らない 192 kbps
        self._audio_quality = QComboBox(self)
        for kbps in AUDIO_KBPS_CHOICES:
            self._audio_quality.addItem(f"{kbps} kbps", kbps)
        self._audio_quality.setCurrentIndex(self._audio_quality.findData(DEFAULT_AUDIO_KBPS))
        self._audio_quality.setToolTip("高いほど音が細かく、ファイルが大きい 迷ったら 192 kbps")

        self._bitrate = QSpinBox(self)
        self._bitrate.setRange(*BITRATE_RANGE_MBPS)
        choice = bitrate if bitrate is not None else BitrateChoice(auto=False)
        self._bitrate_choice = choice
        #: 人が欄を替えたか 替えた後に届いた素材に合わせた値で上書きしない
        self._bitrate_touched = False
        #: 欄を道具が書いている間 人が替えたと読まない
        self._bitrate_writing = False
        #: 最後に頼んだ素材に合わせる計算の印 シーンを替えた後に届いた前の結果を捨てる
        self._bitrate_token: object | None = None
        # 素材に合わせる値はファイルの大きさを読むので、出るまでは上限を出しておく
        self._set_bitrate(choice.mbps)
        self._bitrate.setSuffix(" Mbps")
        self._bitrate.valueChanged.connect(self._on_bitrate_edited)
        self._bitrate_found.connect(self._on_bitrate_found)
        if choice.auto:
            # 決め打ちでない訳が分からないと、毎回違う初めの値を不具合と読む
            self._bitrate.setToolTip(
                f"置いた素材に合わせた値（上限 {choice.mbps} Mbps） 表示 → 設定… で決め打ちにできる"
            )

        self._scene = QComboBox(self)
        self._scene.addItem("メイン", None)
        for scene in project.scenes:
            self._scene.addItem(f"シーン「{scene.name}」", scene.id)
        if not project.scenes:
            self._scene.setToolTip("シーンを作ると、ここでシーンだけを書き出せる")

        # 既定は本人の設定（#330 既定は入） 窓で替えた値は呼んだ側が設定へ覚える
        self._chapters = QCheckBox("マーカーをチャプターとして埋める", self)
        self._chapters.setChecked(chapters)
        self._chapters.setToolTip(
            "名前の付いた目印（目盛りの旗）を動画の章にする 効くのは mp4・mov・mkv だけ"
        )

        self._summary = QLabel(self)
        self._range = QComboBox(self)
        self._empty = QLabel("タイムラインが空なので書き出せない", self)
        self._silent = QLabel(NO_SOUND, self)
        self._silent.setWordWrap(True)

        form = QFormLayout()
        self._form = form
        form.addRow("出力先", path_row)
        form.addRow("形式", self._format)
        form.addRow("書き出す物", self._scene)
        form.addRow("コーデック", self._codec)
        form.addRow("ビットレート", self._bitrate)
        form.addRow("音質", self._audio_quality)
        form.addRow("内容", self._summary)
        form.addRow("書き出す範囲", self._range)
        form.addRow("", self._chapters)
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
        #: 作った声の表記（#335） 声の規約が求める物 書き出す前に見せ、書き終えたときにも知らせる
        #: 忘れて規約に反するのは本人なので、知らせる所まではアプリが受け持つ
        self._credits = project.credits
        if self._credits:
            credit_note = QLabel(f"{CREDIT_NOTE}\n{credit_text(self._credits, chr(10))}", self)
            credit_note.setWordWrap(True)
            credit_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            copy = QPushButton("表記を写す", self)
            copy.setToolTip("表記をクリップボードへ写す 概要欄にそのまま貼れる")
            copy.clicked.connect(self._copy_credits)
            credit_row = QHBoxLayout()
            credit_row.addWidget(credit_note, 1)
            credit_row.addWidget(copy)
            form.addRow("表記", credit_row)
        form.addRow("", self._empty)
        form.addRow("", self._silent)

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
        self._format.currentIndexChanged.connect(self._on_format_changed)
        self._on_scene_changed()
        self._show_format_rows()

    def _set_bitrate(self, value: int) -> None:
        """窓の側でビットレートの欄を書き換える

        本人が書き換えた印を付けないため 付くと、素材に合わせて求めた値で書き直さなくなる
        """
        self._bitrate_writing = True
        try:
            self._bitrate.setValue(value)
        finally:
            self._bitrate_writing = False

    def _on_bitrate_edited(self) -> None:
        """本人がビットレートを変えたら、後から求めた値で書き換えない

        書き換えると本人の選んだ値が消える
        """
        if not self._bitrate_writing:
            self._bitrate_touched = True

    @property
    def checking_codecs(self) -> bool:
        """使えるエンコーダを裏で確かめていて、「コーデック」の欄がまだ埋まっていない"""
        return self._codec.count() == 1 and self._codec.itemText(0) == CHECKING_CODECS

    def _load_codecs(self) -> None:
        """今の形式で選べる候補を「コーデック」の欄へ並べる 確かめが要れば裏で待つ

        「自動」に出す名前と書き出し側が実際に選ぶ物をそろえるため、同じ優先で並べる
        ほかの確かめ（設定画面の確かめ直し・起動の後の確かめ）が走っている間は待たない
        待つと、固まった駆動を待ち切るまで（候補ごとに 15 秒）窓が開かない（#464）
        音声だけの形式は映像のコーデックを使わないので並べない（欄ごと隠す）
        """
        chosen = self._chosen_format
        if chosen.audio_only:
            self._codec.clear()
            self._codecs_ready = False
            return
        if chosen.video == VIDEO_WEB:
            codecs = available_web_codecs_now(self._encoder_priority)
        else:
            codecs = available_video_codecs_now(self._encoder_priority)
        if codecs is None:
            self._codec.clear()
            self._codec.addItem(CHECKING_CODECS, None)
            self._codec.setEnabled(False)
            self._codecs_ready = False
            self._wait_for_codecs(chosen.video, self._encoder_priority)
        else:
            self._fill_codecs(codecs)

    def _fill_codecs(self, codecs: list[str]) -> None:
        """確かめた候補を「コーデック」の欄へ並べる"""
        self._codec.clear()
        # 「自動」は書き出し側が名指し無しで試す物だけから作る WebM の自動は VP9 だけで、
        # AV1 は選べるだけ 先頭の AV1 を自動と出すと、VP9 が使えない機械で自動のまま押して
        # 失敗する（PR #547 の指摘） 自動の候補が無ければ自動を出さず、名指しだけを並べる
        automatic = (
            [name for name in codecs if name in WEB_AUTOMATIC_CODECS]
            if self._chosen_format.video == VIDEO_WEB
            else codecs
        )
        if automatic:
            # 既定は名指しせず書き出し側に選ばせる 先頭を名指しで渡すと、試しには開けても
            # 作品の大きさで断られたとき（NVENC は幅 4096 まで）に次の候補へ落ちられない
            first = CODEC_LABELS.get(automatic[0], automatic[0])
            self._codec.addItem(f"自動（{first}、使えなければ次の候補）", AUTO_CODEC)
        for name in codecs:
            self._codec.addItem(CODEC_LABELS.get(name, name), name)
        if not codecs:
            self._codec.addItem("利用できるコーデックが無い", None)
        self._codec.setEnabled(bool(codecs))
        self._codecs_ready = bool(codecs)

    def _wait_for_codecs(self, video: str, encoder_priority: str) -> None:
        """走っている確かめが終わるのを裏で待ち、終わったら候補を並べる

        ``video`` は頼んだときの形式の映像の作り方 待つ間に形式を替えたら、届いた候補を捨てる
        """

        def work() -> None:
            try:
                if video == VIDEO_WEB:
                    codecs = available_web_codecs(encoder_priority)
                else:
                    codecs = available_video_codecs(encoder_priority)
            except Exception:
                # 拾わないと「確かめています」のまま残り、書き出しを押せない理由が分からない
                _log.exception("書き出しの窓でエンコーダを確かめられなかった")
                codecs = []
            try:
                self._codecs_found.emit(video, codecs)
            except RuntimeError:
                # 先に窓を閉じた 届け先が無い
                return

        threading.Thread(target=work, name="sashimono-export-codecs", daemon=True).start()

    def _on_codecs_found(self, video: str, codecs: list[str]) -> None:
        """確かめ終えた候補を並べ、書き出しを押せるかを直す

        待つ間に別の作り方の形式へ替えていたら捨てる 並べると、WebM の欄に H.264 が出る
        """
        if video != self._chosen_format.video:
            return
        self._fill_codecs(codecs)
        self._refresh_ok()

    def _refresh_ok(self) -> None:
        """〔書き出し〕を押せるかを直す 押せない理由は欄と断りの文に出ている

        押せるままにすると、押しても何も起きず理由が分からない 書き出している間は触らない
        """
        if self._thread is not None:
            return
        chosen = self._chosen_format
        empty = self._project.duration <= 0
        silent = chosen.audio_only and not has_sound(self._project)
        self._silent.setVisible(silent and not empty)
        ready = chosen.audio_only or self._codecs_ready
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            ready and not empty and not silent
        )

    @property
    def chosen_format(self) -> str:
        """今の形式の名前（#527） 呼んだ側が覚え、次に開く窓の初めの形式にする"""
        return self._chosen_format.key

    def select_format(self, key: str) -> None:
        """形式を選ぶ 知らない名前なら何もしない"""
        index = self._format.findData(key)
        if index >= 0:
            self._format.setCurrentIndex(index)

    def _on_format_changed(self) -> None:
        """形式に合わせて、名前の拡張子・コーデックの並び・出す欄・押せるかを直す"""
        chosen = format_named(str(self._format.currentData()))
        before = self._chosen_format
        self._chosen_format = chosen
        path = Path(self._path.text())
        if path.suffix.lower() != chosen.suffix:
            # 打った名前の拡張子が今の形式と同じなら書き換えない 打っている途中のカーソルが飛ぶ
            self._path.setText(str(with_format_suffix(path, chosen)))
        if chosen.video != before.video:
            self._load_codecs()
        self._show_format_rows()
        self._refresh_ok()

    def _show_format_rows(self) -> None:
        """形式で使わない欄を隠す 音声だけなら映像のコーデックとビットレート、音質は MP3 と M4A だけ

        出したままにすると、効かない欄を替えて書き出し、変わらない理由を探すことになる
        章を埋められない形式では、章の欄を押せなくする（値は本人の設定なので残す）
        """
        chosen = self._chosen_format
        self._form.setRowVisible(self._codec, not chosen.audio_only)
        self._form.setRowVisible(self._bitrate, not chosen.audio_only)
        self._form.setRowVisible(self._audio_quality, chosen.audio_quality)
        self._chapters.setEnabled(chosen.chapters)

    def _on_path_edited(self, text: str) -> None:
        """打った名前の拡張子が知っている形式なら、形式の欄をそれに合わせる"""
        found = format_of(Path(text))
        if found is not None and found.key != self._chosen_format.key:
            self.select_format(found.key)

    def _match_bitrate(self, project: Project) -> None:
        """書き出す物の素材に合わせたビットレートを、スレッドの外で求める（#414）

        素材のファイルの大きさを読む 応えないネットワークの共有を窓のスレッドで読むと、その間は
        窓が出ず編集画面も固まる（PR #451 の CodeRabbit の指摘）
        """
        if not self._bitrate_choice.auto or self._bitrate_touched:
            return
        token = object()
        self._bitrate_token = token
        choice = self._bitrate_choice
        # 求め終えるまでは上限を出す 前のシーンの低い値のまま押すと、そのまま低く書き出す
        self._set_bitrate(choice.mbps)
        if not _BITRATE_SLOTS.acquire(blocking=False):
            # 前の計算が応えない共有で止まっている 窓を開くたびに待つスレッドを増やさず上限で出す
            return

        def work() -> None:
            """素材の大きさを読んでビットレートを求める

            応えない共有で止まっても窓が固まらないように、窓とは別のスレッドで走らせる
            """
            try:
                value = choice.for_project(project)
            except (OSError, ValueError):
                # 求められなければ上限のまま 窓は落とさない
                return
            finally:
                _BITRATE_SLOTS.release()
            try:
                self._bitrate_found.emit(token, value)
            except RuntimeError:
                # 先に窓を閉じた 届け先が無い
                return

        threading.Thread(target=work, name="sashimono-export-bitrate", daemon=True).start()

    def _on_bitrate_found(self, token: object, value: int) -> None:
        """求め終えた値を欄へ出す

        前に選んでいたシーンの値と、本人が書き換えた後の値は捨てる 出すと、今の選択と合わない値で
        書き出す
        """
        if token is not self._bitrate_token or self._bitrate_touched:
            return
        self._set_bitrate(value)

    def select_scene(self, scene_id: SceneId | None) -> None:
        """書き出す物を選ぶ ``None`` ならメイン 無いシーンなら何もしない"""
        index = self._scene.findData(scene_id)
        if index >= 0:
            self._scene.setCurrentIndex(index)

    @property
    def embeds_chapters(self) -> bool:
        """「マーカーをチャプターとして埋める」の今の値 呼んだ側が本人の設定に覚える"""
        return self._chapters.isChecked()

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
        self._project = export_target(self._whole, scene.id if scene is not None else None)
        project = self._project
        width, height = project.settings.resolution
        self._summary.setText(
            f"{width}x{height} / {project.settings.frame_rate} fps / {project.duration} フレーム"
        )
        self._fill_range(project)
        self._match_bitrate(project)
        # 出力先の名前を書き換えるのは、前の既定の名前のままのときだけ 手で決めた名前は残す
        path = Path(self._path.text())
        if path.name == before:
            self._path.setText(str(path.with_name(self._default_name(scene))))
        self._empty.setVisible(project.duration <= 0)
        # シーンを替えると鳴る物の有る無しも変わる（音声だけの形式の断り）
        self._refresh_ok()

    def _default_name(self, scene: Scene | None) -> str:
        """出力先の既定の名前 シーンならシーンの名前を後ろに付ける（メインと上書きし合わない）"""
        suffix = self._chosen_format.suffix
        if scene is None:
            return f"{self._whole.name}{suffix}"
        # シーンの名前は自由に付けられる ファイル名に使えない文字を残すと、書き出しを
        # 押してから開けないと断られる
        safe = "".join("_" if c in '\\/:*?"<>|' else c for c in scene.name)
        return f"{self._whole.name}_{safe}{suffix}"

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
            return chosen_range(self._project, RANGE_WORK_AREA)
        return None

    def _choose_path(self) -> None:
        """書き出し先をファイルの窓で選ぶ 選んだ名前の拡張子が知っている形式なら形式も合わせる

        絞り込みは今の形式を先にして、ほかの形式も並べる 1 つだけだと、別の形式を書けることが
        窓から分からない
        """
        chosen = self._chosen_format
        others = [item for item in FORMATS if item is not chosen]
        filters = ";;".join(_filter(item) for item in [chosen, *others])
        name, _ = QFileDialog.getSaveFileName(
            self, "書き出し先", self._path.text(), f"{filters};;すべてのファイル (*)"
        )
        if name:
            found = format_of(Path(name))
            if found is not None:
                self.select_format(found.key)
            self._path.setText(str(with_format_suffix(Path(name), self._chosen_format)))

    def _output_path(self) -> Path:
        """書き出す名前 拡張子は今の形式に合わせる（知らない拡張子なら後ろへ足す）"""
        return with_format_suffix(Path(self._path.text()), self._chosen_format)

    def _settings(self) -> ExportSettings | None:
        """画面の選択から書き出しの設定を作る 使えるコーデックが無ければ ``None``"""
        chosen = self._chosen_format
        if chosen.audio_only:
            codec: object = AUTO_CODEC
        else:
            codec = self._codec.currentData()
            if codec is None:
                return None
        # AI の書き出しと同じ作り方で組む（#327） 画面だけ別に組むと、同じ範囲・画質を
        # 頼んでも、AI が書き出した物だけ先読みの深さや範囲が違う
        return build_settings(
            self._output_path(),
            codec=str(codec) or None,
            bitrate_mbps=self._bitrate.value(),
            frame_range=self._frame_range(),
            pipeline_depth=self._pipeline_depth,
            decode_threads=self._decode_threads,
            smooth_history=self._smooth_history,
            gpu_color=self._gpu_color,
            chapters=self._chapters.isChecked(),
            encoder_priority=self._encoder_priority,
            audio_kbps=int(self._audio_quality.currentData()),
        )

    def _start(self) -> None:
        if self._thread is not None:
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            # 名前が空・範囲が消えたなど 押しても黙って何も起きないより、理由を出す
            QMessageBox.warning(self, "書き出し", str(exc))
            return
        if settings is None:
            return
        # 書き出す名前を欄にも出す 拡張子を形式に合わせて付け替えたことが見える
        self._path.setText(str(settings.path))
        self.started_folder = settings.path.parent

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("中止")

        self._worker = _ExportWorker(
            self._project, settings, separate_process=self._separate_process
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @property
    def export_progress(self) -> float | None:
        """書き出しの進み具合（0〜1） 走っていなければ ``None`` AI が人の書き出しを読む口（#352）"""
        if self._worker is None:
            return None
        return self._progress.value() / self._progress.maximum()

    def _cancel_or_close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            return
        self.reject()

    def _on_progress(self, value: float) -> None:
        self._progress.setValue(int(value * 1000))

    def _on_finished(self, path: str) -> None:
        notes = self._worker.notes if self._worker is not None else []
        self._teardown()
        if self._closing:
            # 閉じるよう頼まれて止めている間に書き終わった 頼まれたとおり閉じる
            super().reject()
            return
        # 次の候補のエンコーダで書き直したときは、何で書いたかを添える 黙っていると、
        # GPU で書いたつもりの人が遅さや画の違いの理由を探せない
        text = "\n".join([f"書き出しました\n{path}", *notes])
        if self._credits:
            # 作った声の表記（#335） 書き終えたときにも知らせる
            text += f"\n\n{CREDIT_NOTE}\n{credit_text(self._credits, chr(10))}"
        QMessageBox.information(self, "書き出し", text)
        self.accept()

    def _copy_credits(self) -> None:
        clipboard().setText(credit_text(self._credits, "\n"))

    def _on_failed(self, message: str) -> None:
        self._teardown()
        if self._closing:
            # 止めた知らせ（中止）が届いた もうスレッドは走っていないので閉じてよい
            super().reject()
            return
        QMessageBox.warning(self, "書き出し", message)

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            # 時間を区切らずに待つ ここへ来るのはワーカーが書き出しを終えて知らせた後で、
            # スレッドはイベントループを抜けるだけ 区切って諦めると、走っている
            # スレッドを持ったまま窓を捨てることになる
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._progress.setVisible(False)
        self._refresh_ok()
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("閉じる")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt の命名規約
        # 書き出し中に閉じられたら、止めるよう頼むだけで今は閉じない 止まった知らせが
        # 来てから閉じる 走っているスレッドを持ったまま窓が壊れると、Qt はスレッドの
        # 生存中に破棄されたと言って落ちる
        if self._worker is not None:
            event.ignore()
            self._close_when_stopped()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        """〔閉じる〕と Esc 窓の × と同じく、書き出し中なら止まってから閉じる

        reject は closeEvent を通らない ここでも止まったと確かめるまで閉じない
        """
        if self._worker is not None:
            self._close_when_stopped()
            return
        super().reject()

    def _close_when_stopped(self) -> None:
        """書き出しを止めるよう頼み、止まった知らせ（``_on_failed``）が来たら閉じる

        その場で待たない 待つと止まるまで窓が固まり、時間を区切って諦めると走っている
        スレッドを残したまま閉じることになる 止まるまでは〔中止〕を押せなくして、
        止めている最中だと見せる
        """
        if self._worker is None:
            return
        self._closing = True
        self._worker.cancel()
        cancel = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setText("止めています…")
        cancel.setEnabled(False)


def _filter(item: ExportFormat) -> str:
    """ファイルの窓の絞り込みの 1 つ"""
    return f"{item.label} (*{item.suffix})"
