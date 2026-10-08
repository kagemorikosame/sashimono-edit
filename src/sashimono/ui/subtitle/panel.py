"""字幕パネル

素材を選び、その起こし結果を一覧で編集する 時刻の列に出るのは**タイムライン上の
位置**で、素材内の時刻ではない 編集中に見たいのは「動画の何分何秒に出るか」で、
素材のどこかは分かっても仕方がない その変換は投影
（:mod:`sashimono.core.projection`）が引き受ける

自分ではプロジェクトを書き換えない 操作はすべてコマンドとして外へ出す
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sashimono.asr import JobKind, TranscribeOptions, TranscriptionService, default_backend
from sashimono.asr.service import Job
from sashimono.core.commands import (
    Command,
    MergeWithNext,
    RemoveSegment,
    RippleCut,
    SetSegmentText,
    SetTranscript,
    SplitSegment,
    Voice,
    burn_defaults,
    burn_subtitles,
    export_range,
    subtitle_voices,
    subtitle_wrap_width,
    voice_label,
)
from sashimono.core.io import SUBTITLE_FILTER, save_subtitles
from sashimono.core.jetcut import plan_cuts
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    MediaId,
    MediaItem,
    Project,
    SegmentId,
    TranscriptSegment,
)
from sashimono.core.projection import project_clip, subtitle_stream, subtitle_voice
from sashimono.core.timebase import format_timecode, seconds_to_frame
from sashimono.effects.sources import TEXT
from sashimono.engine.audio.silence import SilenceOptions, detect_silence, keep_speech
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.export_dialog import RANGE_ALL, RANGE_WORK_AREA
from sashimono.ui.flow_layout import flow_of, narrow_combo
from sashimono.ui.subtitle.dialogs import BurnDialog, CleanupDialog, JetCutDialog
from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog
from sashimono.ui.system_clipboard import clipboard
from sashimono.ui.theme import Colors, theme_signals, themed_style

__all__ = ["SubtitlePanel", "ask_subtitle_range"]


#: 時刻の列に足す余白（画素） 文字の幅ぴったりだと読みにくい
TIME_COLUMN_PADDING = 24

#: 時刻の列の幅を決める見本の長さ（秒） 1 時間
#: これより短い動画でも、この幅は空けておく 桁が増えるたびに列が動くと目が疲れる
#: フレーム数ではなく秒で持つ フレーム数だと、フレームレートによって
#: 表す長さが変わってしまう（30fps の 10 万フレームは約 1 時間だが 60fps では 30 分）
MIN_TIME_SAMPLE_SECONDS = 3600


class SubtitlePanel(QWidget):
    """素材ごとの字幕の一覧と編集"""

    #: 編集操作 引数はコマンドの一覧と、履歴に出す操作名
    commands_requested = Signal(list, str)
    #: 字幕をクリックしたときの移動先（フレーム）
    seek_requested = Signal(int)
    #: ステータスバーへ出す文言
    status_message = Signal(str)

    def __init__(
        self, project: Project, analyzer: MediaAnalyzer, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._analyzer = analyzer
        self._media_id: MediaId | None = None
        #: 見ている字幕の音声ストリームの番号 ``None`` なら 1 本目 字幕は音声ごとに別
        self._stream: int | None = None
        self._frame = 0
        #: 表示中の行に対応する字幕 行番号から引く
        self._rows: list[tuple[SegmentId, int, int]] = []
        #: 一覧を作ったときの中身の印 同じなら作り直さない
        self._signature: tuple[object, ...] | None = None
        self._updating = False
        #: 起こしの実行係 バックエンドの読み込みは重いので、初めて使うときに作る
        self._service: TranscriptionService | None = None
        #: AI から頼まれた起こし 窓を開かずに走らせる 1 本ずつ順番に走り
        #: （:class:`TranscriptionService`）、
        #: 終わった物はどれもその素材と音声の字幕へ取り込む
        self._jobs: list[Job] = []
        #: 終わった起こしの知らせ 様子を聞かれたときに並べる
        self._job_notes: list[str] = []
        #: 焼き込みのひな形にするクリップを返す（タイムラインで選んでいるテキスト）
        #: 窓が差し込む 差し込まなければひな形は無い（既定の見た目）
        self.template_provider: Callable[[], Clip | None] = lambda: None
        #: 既定の見た目で焼き込む字幕を折り返す幅（画面の幅の % 0 は折り返さない #249）
        #: 窓が本人の設定（``Preferences.subtitle_wrap_share``）から入れる 既定は設定の既定と同じ
        self.wrap_share = 90
        #: 焼き込む話し手を尋ねる 試験で差し替える
        self.ask_burn: Callable[[list[tuple[Voice, str]], str], list[Voice] | None] = self._ask_burn
        #: 起こしている物の最後の進み具合（依頼ごと）
        self._last_progress: dict[int, str] = {}

        self._build()
        self.set_project(project)

    # --- 組み立て ---

    def _build(self) -> None:
        self._media = QComboBox(self)
        # 素材の名前の長さでパネルの最小の幅を決めない 長い名前の素材を読み込むたびに
        # 窓の最小の幅が伸び、1366 の画面から窓がはみ出す 名前の全部は一覧を開けば読める
        narrow_combo(self._media, 8)
        self._media.currentIndexChanged.connect(self._on_media_changed)
        # 音声が何本もある素材（ゲームの音とマイクの声など）は、どの音の字幕を見るかを選ぶ
        # 1 本の素材では出さない（今までどおり素材だけ）
        self._stream_box = QComboBox(self)
        self._stream_box.setObjectName("subtitle_stream")
        self._stream_box.currentIndexChanged.connect(self._on_stream_changed)
        self._stream_box.hide()

        self._transcribe_button = self._button("起こす…", self.transcribe)
        self._clean_button = self._button("整形…", self.clean)
        self._cut_button = self._button("無音カット…", self.jet_cut)
        self._burn_button = self._button("焼き込み", self.burn)
        self._place_rows_button = self._button("選んだ行を置く", self.place_selected_rows)
        self._place_rows_button.setToolTip(
            "選んだ字幕の行をテキストとしてタイムラインへ置く 選んでいなければ再生位置の 1 行"
        )
        self._export_button = self._button("書き出し…", self.export_file)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._media, 1)
        top.addWidget(self._stream_box)
        top.addWidget(self._transcribe_button)

        # 狭いパネルでは折り返す 1 列に並べると 5 つのボタンの幅の和がパネルの最小の幅になり、
        # 字幕のパネルだけで 1366 の画面の 3 分の 1 を取った
        actions = flow_of(
            self._clean_button,
            self._cut_button,
            self._burn_button,
            self._place_rows_button,
            self._export_button,
        )

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["時刻", "本文"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # 何行もまとめて選べる（選んだ行をテキストとして置くため）
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_menu)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.setWordWrap(True)

        header = self._table.horizontalHeader()
        # 時刻の幅は**固定**にする 中身に合わせると、1 行足すたびに Qt が
        # 全部の行を測り直すので、字幕 2000 本では作り直しに 6 秒かかっていた
        # 時刻は桁数の決まった文字列なので、見本の幅を 1 度測れば足りる
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, self._time_column_width())
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # 幅が変われば折り返しの行数も変わる 高さを取り直さないと、2 行目が
        # 隠れて本文の末尾が読めなくなる どの列でも取り直す（理由は
        # :meth:`_on_section_resized` に書いた）
        header.sectionResized.connect(self._on_section_resized)

        self._empty = QLabel("音声を持つ素材を選んでください", self)
        themed_style(self._empty, lambda: f"color: {Colors.TEXT_MUTED.name()}; padding: 8px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(actions)
        layout.addWidget(self._empty)
        layout.addWidget(self._table, 1)
        theme_signals().changed.connect(self._recolor_rows)

    def _recolor_rows(self) -> None:
        """テーマが変わった タイムラインに出ていない字幕の時刻を今の薄い色で塗り直す

        行の文字の色は行を作った時点の色で持つので、塗り直さないと前のテーマの色が残る
        """
        self._updating = True
        try:
            for row, (_, start, _end) in enumerate(self._rows):
                item = self._table.item(row, 0)
                if item is not None and start < 0:
                    item.setForeground(Colors.TEXT_MUTED)
        finally:
            self._updating = False

    def _button(self, text: str, slot: object) -> QPushButton:
        button = QPushButton(text, self)
        button.clicked.connect(slot)
        return button

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        self._table.resizeRowsToContents()

    # --- 外から差し替えるもの ---

    def set_project(self, project: Project) -> None:
        self._project = project
        self._reload_media()
        self._reload_rows()

    def set_frame(self, frame: int) -> None:
        """再生ヘッドの位置 いま出ている字幕を強調する"""
        if frame == self._frame:
            return
        self._frame = frame
        self._highlight()

    @property
    def media_id(self) -> MediaId | None:
        return self._media_id

    def select_media(self, media_id: MediaId) -> None:
        index = self._media.findData(str(media_id))
        if index >= 0:
            self._media.setCurrentIndex(index)

    def follow_clip(self, media_id: MediaId, stream: int | None) -> None:
        """タイムラインで選んだクリップの素材と、そのクリップが鳴らす音の字幕を出す

        前は字幕パネルが前に開いた素材（音声 4 本の動画など）のまま残り、音声 1 本の
        クリップを選んで起こしても、前の素材の 4 つの音声が並んだ（利用者の画面）
        """
        self.select_media(media_id)
        self.select_stream(stream)

    @property
    def stream(self) -> int | None:
        """見ている字幕の音声ストリームの番号 ``None`` なら 1 本目"""
        return self._stream

    def select_stream(self, stream: int | None) -> None:
        """見る字幕の音声を選ぶ 素材に無い番号は 1 本目"""
        media = self._current_media()
        if media is None:
            return
        index = self._stream_box.findData(media.transcript_stream(stream))
        if index >= 0:
            self._stream_box.setCurrentIndex(index)

    def _fill_streams(self) -> None:
        """音声の選びを今の素材に合わせて作り直す 前に見ていた番号があれば残す"""
        media = self._current_media()
        previous = self._stream
        self._updating = True
        self._stream_box.clear()
        streams = media.audio_streams if media is not None else ()
        for number, info in enumerate(streams, start=1):
            has = media is not None and media.transcript_for(info.index) is not None
            self._stream_box.addItem(f"音声 {number}" + ("" if has else "（字幕なし）"), info.index)
        self._updating = False
        self._stream_box.setVisible(len(streams) > 1)
        if media is None:
            self._stream = None
            return
        chosen = media.transcript_stream(previous)
        self._stream_box.setCurrentIndex(max(0, self._stream_box.findData(chosen)))
        self._stream = chosen

    def _on_stream_changed(self) -> None:
        if self._updating:
            return
        data = self._stream_box.currentData()
        self._stream = int(data) if data is not None else None
        self._reload_rows()

    # --- 一覧 ---

    def _time_column_width(self) -> int:
        """時刻の列の幅 一番長くなるタイムコードを測って決める

        見本を決め打ちにすると、100 時間を超えるタイムラインで時が 3 桁になり、
        2 桁ぶんの幅で切れる 実際の長さから決める（短いときは見本の方を使う
        短い動画で時刻の列が細くなりすぎると、桁が増えたときに毎回揺れる）
        """
        least = seconds_to_frame(MIN_TIME_SAMPLE_SECONDS, self._project.rate)
        longest = max(self._project.duration, least)
        sample = format_timecode(longest, self._project.rate)
        return self._table.fontMetrics().horizontalAdvance(sample) + TIME_COLUMN_PADDING

    def _on_section_resized(self, _index: int, _old: int, _new: int) -> None:
        """列の幅が変わったら、折り返しに合わせて行の高さを取り直す

        どの列でも取り直す 本文の列は残りを埋める作りなので、時刻の列が
        広がればそのぶん狭くなり、折り返しの行数が変わる 本文の列の合図が
        いつも飛ぶとは限らない（画面に出ていないときは飛ばない）ので、
        時刻の列の合図も受ける

        作り直しの最中は走らせない 1 行入れるたびに全部の行を測り直すと、
        本数の 2 乗で遅くなる（入れ終えてから 1 度だけ取り直す）
        """
        if not self._updating:
            self._table.resizeRowsToContents()

    def _reload_media(self) -> None:
        """素材の選択欄を作り直す 選択は ID で復元する"""
        candidates = [item for item in self._project.media if item.has_audio]
        previous = self._media_id

        self._updating = True
        self._media.clear()
        for item in candidates:
            self._media.addItem(item.name, str(item.id))
        self._updating = False

        if not candidates:
            self._media_id = None
        elif previous is not None and any(item.id == previous for item in candidates):
            self.select_media(previous)
        else:
            self._media_id = candidates[0].id
            self._media.setCurrentIndex(0)

        # 字幕の有無の印（「字幕なし」）も素材の中身に合わせて付け直す
        self._fill_streams()
        has_media = bool(candidates)
        self._transcribe_button.setEnabled(has_media)
        self._empty.setVisible(not has_media)
        self._table.setVisible(has_media)

    def _on_media_changed(self) -> None:
        if self._updating:
            return
        data = self._media.currentData()
        self._media_id = MediaId(str(data)) if data else None
        self._fill_streams()
        self._reload_rows()

    def _current_media(self) -> MediaItem | None:
        if self._media_id is None:
            return None
        return self._project.find_media(self._media_id)

    def _segments(self) -> tuple[TranscriptSegment, ...]:
        media = self._current_media()
        transcript = media.transcript_for(self._stream) if media is not None else None
        if transcript is None:
            return ()
        return transcript.segments

    def _rows_signature(self) -> tuple[object, ...]:
        """一覧の中身が変わったかを見るための印

        字幕そのものと、それがタイムラインのどこに出るかで決まる
        モデルは作り替えでしか変わらないので、字幕の一覧は同じものかどうかを
        ``id`` で見れば足りる（中身を突き合わせると本数ぶんの手間が掛かる）
        """
        media = self._current_media()
        if media is None:
            return (None,)
        # クリップに出す字幕の音（:func:`subtitle_voice`）も入れる クリップの番号だけでなく
        # トラックの種類（音声・映像・混合）でも聞こえる音が変わる 入れないと、クリップの音を
        # 替えたり別の種類のトラックへ移したりしても、一覧が前の音の時刻のまま残る（PR #231）
        clips = tuple(
            (
                str(clip.id),
                clip.timeline_start,
                clip.duration,
                clip.source_in,
                clip.speed,
                subtitle_voice(self._project, track, clip, media),
            )
            for track in self._project.timeline.tracks
            for clip in track.clips
            if clip.media_id == media.id
        )
        return (
            str(media.id),
            self._stream,
            id(media.transcript_for(self._stream)),
            self._project.rate,
            clips,
        )

    def _reload_rows(self) -> None:
        """一覧を作り直す

        時刻はタイムライン上の位置 同じ素材を 2 回置いていれば最初の 1 回の
        位置を出す 編集の入口としてはそれで足り、2 か所の時刻を並べても迷う

        中身が前と同じなら作り直さない 編集のたびに呼ばれる所で、字幕が
        2000 本あると作り直しだけで 70ms 掛かる 字幕に関係のない編集
        （クリップの色を変えるなど）でそれを払うのは無駄
        """
        signature = self._rows_signature()
        if signature == self._signature and self._table.rowCount() == len(self._segments()):
            # 中身は同じ 幅だけ入れ直す 長さが変わって桁が増えていれば、
            # ここで合図が飛んで折り返しを取り直す（変わっていなければ何も起きない）
            self._table.horizontalHeader().resizeSection(0, self._time_column_width())
            return
        # 印は**作り終えてから**立てる 途中で落ちたのに立てると、同じ
        # プロジェクトを開き直しても作り直さず、半端な表が残ったままになる
        self._signature = None
        media = self._current_media()
        segments = self._segments()
        placement = self._placement(media) if media is not None else {}

        self._updating = True
        # 描き直しを止めてから中身を入れ替える 途中の状態を描くと、
        # 1 行ごとに並べ直しが走って本数の 2 乗で遅くなる
        # 途中で落ちても必ず戻す 戻し損ねると、表が固まったまま何も映らない
        self._table.setUpdatesEnabled(False)
        try:
            # 幅は中身を入れ替える**前**に決める ここで変えても、作り直しの
            # 最中なので合図は無視される 最後に 1 度だけ取り直せば足りる
            # （あとから変えると、入れ替え直後の取り直しと合わせて 2 度測る）
            self._table.horizontalHeader().resizeSection(0, self._time_column_width())
            # いったん空にしてから伸ばす 置き換えると、古い中身を捨てる手間が
            # 1 行ずつ掛かる
            self._table.setRowCount(0)
            self._table.setRowCount(len(segments))
            self._rows = []
            for row, segment in enumerate(segments):
                start, end = placement.get(segment.id, (-1, -1))
                self._rows.append((segment.id, start, end))

                when = format_timecode(start, self._project.rate) if start >= 0 else "—"
                time_item = QTableWidgetItem(when)
                time_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if start < 0:
                    # タイムラインに出ていない字幕 素材を切った先で使われていない範囲
                    time_item.setForeground(Colors.TEXT_MUTED)
                    time_item.setToolTip("いまのタイムラインには出ていません")
                self._table.setItem(row, 0, time_item)
                self._table.setItem(row, 1, QTableWidgetItem(segment.text))
        finally:
            self._table.setUpdatesEnabled(True)
            self._updating = False
        self._signature = signature

        self._table.resizeRowsToContents()
        self._update_actions()
        self._highlight()

    def _placement(self, media: MediaItem) -> dict[SegmentId, tuple[int, int]]:
        """字幕がタイムラインのどこに出るか 最初に現れた位置だけを持つ"""
        found: dict[SegmentId, tuple[int, int]] = {}
        wanted = media.transcript_stream(self._stream)
        for track in self._project.timeline.tracks:
            for clip in track.clips:
                if clip.media_id != media.id:
                    continue
                # 見ている音の字幕を出すクリップだけ 音声 2 のクリップの位置を、音声 1 の
                # 字幕の位置として出さない
                stream = subtitle_stream(self._project, track, clip)
                if subtitle_voice(self._project, track, clip, media) != wanted:
                    continue
                for projected in project_clip(clip, media, self._project.rate, track.id, stream):
                    key = projected.segment.id
                    span = (projected.start_frame, projected.end_frame)
                    if key not in found or span < found[key]:
                        found[key] = span
        return found

    def _update_actions(self) -> None:
        has_segments = bool(self._segments())
        self._clean_button.setEnabled(has_segments)
        self._burn_button.setEnabled(has_segments)
        self._place_rows_button.setEnabled(has_segments)
        self._export_button.setEnabled(has_segments)
        media = self._current_media()
        self._cut_button.setEnabled(media is not None)

    def _highlight(self) -> None:
        """再生ヘッドの位置にある字幕を選ぶ"""
        for row, (_, start, end) in enumerate(self._rows):
            if start <= self._frame < end:
                if self._table.currentRow() != row:
                    self._updating = True
                    self._table.selectRow(row)
                    item = self._table.item(row, 1)
                    if item is not None:
                        self._table.scrollToItem(item)
                    self._updating = False
                return

    # --- 操作 ---

    def _on_row_selected(self) -> None:
        if self._updating:
            return
        row = self._table.currentRow()
        if 0 <= row < len(self._rows):
            _, start, _ = self._rows[row]
            if start >= 0:
                self.seek_requested.emit(start)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 1 or self._media_id is None:
            return
        row = item.row()
        if not (0 <= row < len(self._rows)):
            return
        segment_id = self._rows[row][0]
        self.commands_requested.emit(
            [SetSegmentText(self._media_id, segment_id, item.text(), stream=self._stream)],
            "字幕を編集",
        )

    def _show_menu(self, position: QPoint) -> None:
        row = self._table.currentRow()
        if not (0 <= row < len(self._rows)) or self._media_id is None:
            return
        segment_id = self._rows[row][0]

        menu = QMenu(self)
        jump = menu.addAction("ここへ移動")
        copy = menu.addAction("文字をコピー")
        menu.addSeparator()
        place = menu.addAction("テキストとして置く")
        menu.addSeparator()
        split = menu.addAction("再生ヘッドで分割")
        merge = menu.addAction("次と結合")
        remove = menu.addAction("削除")
        chosen = menu.exec(self._table.viewport().mapToGlobal(position))
        # 右クリックのたびに作るメニュー 選んだ項目はこの後で比べるので、後で捨てる
        menu.deleteLater()
        if chosen is place:
            self.place_selected_rows()
            return

        if chosen is jump:
            start = self._rows[row][1]
            if start >= 0:
                self.seek_requested.emit(start)
        elif chosen is copy:
            item = self._table.item(row, 1)
            if item is not None:
                clipboard().setText(item.text())
        elif chosen is remove:
            self.commands_requested.emit(
                [RemoveSegment(self._media_id, segment_id, stream=self._stream)], "字幕を削除"
            )
        elif chosen is merge:
            self.commands_requested.emit(
                [MergeWithNext(self._media_id, segment_id, stream=self._stream)], "字幕を結合"
            )
        elif chosen is split:
            self._split_at_playhead(segment_id)

    def _split_at_playhead(self, segment_id: SegmentId) -> None:
        """再生ヘッドのソース時刻で字幕を割る"""
        media = self._current_media()
        if media is None or self._media_id is None:
            return
        at = self._source_time(media, self._frame)
        if at is None:
            self.status_message.emit("再生ヘッドがこの素材の上にありません")
            return
        self.commands_requested.emit(
            [SplitSegment(self._media_id, segment_id, at, stream=self._stream)], "字幕を分割"
        )

    def _source_time(self, media: MediaItem, frame: int) -> Fraction | None:
        """タイムラインのフレームを、この素材の見ている音のソース秒へ

        見ている音の字幕を出すクリップから取る 最初に当たるクリップを使うと、リンクを外して
        音声 2 だけを切り詰めたとき、音声 1 のクリップの時刻で音声 2 の字幕を割る（PR #231）
        """
        wanted = media.transcript_stream(self._stream)
        for track in self._project.timeline.tracks:
            for clip in track.clips:
                if clip.media_id != media.id or not clip.contains(frame):
                    continue
                if subtitle_voice(self._project, track, clip, media) != wanted:
                    continue
                elapsed = (frame - clip.timeline_start) * self._project.rate.frame_duration
                return clip.source_in + elapsed * clip.speed
        return None

    # --- 起こし・整形・カット ---

    def transcribe(self) -> None:
        media = self._current_media()
        if media is None:
            return
        if self._service is None:
            # バックエンドの生成はここで初めて行う 実行環境が未導入でも
            # パネルは開けるようにしておく
            self._service = TranscriptionService(default_backend())

        dialog = TranscribeDialog(media, self._service, self, stream=self._stream)
        answer = dialog.exec()
        # 開くたびに作る窓 閉じたら捨てる 消えるのは呼んだイベントループへ戻ったときなので、
        # この後で結果を読む間は残る
        dialog.deleteLater()
        if answer and dialog.transcript is not None:
            self.commands_requested.emit(
                [SetTranscript(media.id, dialog.transcript, stream=dialog.chosen_stream)],
                f"字幕を起こす: {media.name}",
            )
            # 起こした音の字幕を見せる
            self.select_stream(dialog.chosen_stream)
            if dialog.notice:
                # 窓は起こし終えたら閉じるので、GPU から CPU へ落とした理由などはここで出す
                self.status_message.emit(dialog.notice)

    def start_transcription(
        self, media_id: MediaId, model: str, *, audio_stream: int | None = None
    ) -> str:
        """起こしを頼む AI からの依頼を受ける入口 走っている物があれば順番待ちに入る

        同じ素材の同じ音をすでに頼んでいれば断る（2 回起こしても同じ結果を 2 回取り込むだけ）
        """
        """起こしを始める AI からの依頼を受ける入口

        ダイアログを開かずに走らせる 数分かかるので、終わったかどうかは
        :meth:`transcription_status` で見る
        """
        media = self._project.find_media(media_id)
        if media is None:
            raise KeyError(f"素材が見つからない: {media_id}")
        if self._service is None:
            self._service = TranscriptionService(default_backend())
        if not self._service.backend.is_available():
            raise RuntimeError("起こしの実行環境が入っていません 字幕パネルから導入できます")

        # 番号をそろえてから比べる 1 本目は「省く（None）」と番号（AI の audio=1・窓）の
        # 2 通りで届き、そのまま比べると同じ音の起こしが 2 回走る（PR #231 の指摘）
        stream = media.transcript_stream(audio_stream)
        if self._service.find(media.id, stream) is not None:
            raise RuntimeError(f"{media.name} のその音声はもう起こしています（順番待ちを含む）")
        options = TranscribeOptions(model=model, audio_stream=stream)
        waiting = self._service.busy
        self._jobs.append(self._service.start(media.id, media.path, options))
        self.select_media(media.id)
        if waiting:
            return f"{media.name} の起こしを順番待ちに入れました 前の起こしが終わると始まります"
        return f"{media.name} の起こしを始めました"

    @property
    def transcribing(self) -> bool:
        """起こしが走っているか・順番を待っているか 更新のための再起動を断るのに使う"""
        return bool(self._jobs) or (self._service is not None and self._service.busy)

    def transcription_status(self) -> str:
        """走っている起こしの様子"""
        return self.poll_transcription()

    def poll_transcription(self) -> str:
        """AI から始めた起こしの様子を拾い、終わっていれば結果を取り込む

        定期的に呼ばれる AI が結果を聞きに来なかった場合でも、起こした内容が
        捨てられないようにするため
        """
        for job in list(self._jobs):
            for event in job.poll():
                if event.kind is JobKind.PROGRESS:
                    self._last_progress[id(job)] = f"{int(event.ratio * 100)}% — {event.message}"
                    continue
                self._jobs.remove(job)
                self._last_progress.pop(id(job), None)
                name = self._describe_job(job)
                if event.kind is JobKind.DONE and event.transcript is not None:
                    # 頼んだ素材と音へ取り込む 表で見ている素材や音には依らない
                    self.commands_requested.emit(
                        [SetTranscript(job.media_id, event.transcript, stream=job.stream)],
                        f"字幕を起こす: {name}",
                    )
                self._job_notes.append(f"{name}: {event.message or '終了した'}")
                break

        lines = []
        if self._service is not None:
            for job in self._service.jobs():
                name = self._describe_job(job)
                if job.waiting:
                    lines.append(f"順番待ち: {name}")
                else:
                    detail = self._last_progress.get(id(job), "")
                    lines.append(f"起こしている: {name}" + (f" {detail}" if detail else ""))
        lines.extend(self._job_notes[-5:])
        return "\n".join(lines) if lines else "起こしは走っていません"

    def _describe_job(self, job: Job) -> str:
        """起こしの依頼の名前 音声が何本もある素材は何本目の音かも添える"""
        media = self._project.find_media(job.media_id)
        if media is None:
            return str(job.media_id)
        if len(media.audio_streams) < 2:
            return media.name
        known = [s.index for s in media.audio_streams]
        key = media.transcript_stream(job.stream)
        return f"{media.name} 音声 {known.index(key) + 1}"

    def clean(self) -> None:
        media = self._current_media()
        transcript = media.transcript_for(self._stream) if media is not None else None
        if media is None or transcript is None:
            return
        dialog = CleanupDialog(transcript, self)
        answer = dialog.exec()
        # 開くたびに作る窓 閉じたら捨てる 消えるのは呼んだイベントループへ戻ったときなので、
        # この後で結果を読む間は残る
        dialog.deleteLater()
        if answer:
            self.commands_requested.emit(
                [SetTranscript(media.id, dialog.result_transcript(), stream=self._stream)],
                "字幕を整形",
            )

    def jet_cut(self) -> None:
        media = self._current_media()
        if media is None:
            return
        waveform = self._analyzer.waveform(media, self._stream)
        if waveform is None:
            self.status_message.emit("波形の解析がまだ終わっていません")
            return

        def estimate(options: SilenceOptions, protect: bool) -> tuple[int, float]:
            ranges = self._plan(media, options, protect)
            frames = sum(end - start for start, end in ranges)
            return len(ranges), float(frames * self._project.rate.frame_duration)

        dialog = JetCutDialog(
            has_transcript=media.transcript_for(self._stream) is not None,
            estimate=estimate,
            parent=self,
        )
        answer = dialog.exec()
        # 開くたびに作る窓 閉じたら捨てる 消えるのは呼んだイベントループへ戻ったときなので、
        # この後で結果を読む間は残る
        dialog.deleteLater()
        if not answer:
            return

        ranges = self._plan(media, dialog.options(), dialog.keep_speech)
        if not ranges:
            self.status_message.emit("切る場所が見つかりませんでした")
            return
        self.commands_requested.emit([RippleCut(ranges)], f"無音カット: {len(ranges)} か所")

    def _plan(
        self, media: MediaItem, options: SilenceOptions, protect: bool
    ) -> tuple[tuple[int, int], ...]:
        """無音の検出からタイムライン上の切る範囲までを一続きに"""
        # 見ている音の波形と字幕で決める マイクの声の無音で切りたいときに、ゲームの音で
        # 決めると切れない
        waveform = self._analyzer.waveform(media, self._stream)
        if waveform is None:
            return ()
        silences = detect_silence(waveform, options)
        transcript = media.transcript_for(self._stream)
        if protect and transcript is not None:
            silences = keep_speech(silences, transcript)
        # 見ている音が 1 本目（None）でも番号にそろえて渡す None のままだと音で絞らない
        return plan_cuts(
            self._project, media.id, silences, stream=media.transcript_stream(self._stream)
        )

    def _template(self) -> tuple[GeneratedSource | Clip, str]:
        """焼き込みのひな形と、窓に出す説明 タイムラインで選んでいるテキストがあればそれ"""
        clip = self.template_provider()
        if clip is not None and clip.source is not None and clip.source.kind == "text":
            text = str(clip.source.params.get("text", "")).splitlines()
            head = text[0][:12] if text else ""
            return clip, f"見た目: 選んでいるテキスト「{head}」を写します（本文だけ差し替え）"
        # 既定の大きさと位置は作品の高さで縮める 画素のまま置くと、720p では画面の外に出る
        settings = self._project.settings
        look = burn_defaults(settings.height, subtitle_wrap_width(settings.width, self.wrap_share))
        wrap = f"・幅 {self.wrap_share}% で折り返し" if self.wrap_share > 0 else ""
        return TEXT.create(**look), (
            f"見た目: 既定（大きさ {look['size']:.3g}・下寄せ・縁取り {look['border_width']:.3g}"
            f"{wrap}）"
            " タイムラインでテキストを選んでから焼き込むと、そのテキストの見た目を写します"
        )

    def burn(self) -> None:
        voices = subtitle_voices(self._project)
        if not voices:
            self.status_message.emit("焼き込む字幕がありません")
            return
        template, note = self._template()
        labels = [(voice, voice_label(self._project, voice)) for voice in voices]
        chosen = self.ask_burn(labels, note)
        if chosen is None:
            return
        commands: list[Command] = burn_subtitles(
            self._project, template, voices=[v for v in voices if v in chosen]
        )
        if not commands:
            self.status_message.emit("焼き込む字幕がありません")
            return
        self.commands_requested.emit(commands, "字幕を焼き込み")

    def _ask_burn(self, voices: list[tuple[Voice, str]], note: str) -> list[Voice] | None:
        dialog = BurnDialog([(voice, label) for voice, label in voices], note, self)
        answer = dialog.exec()
        # 開くたびに作る窓 閉じたら捨てる 消えるのは呼んだイベントループへ戻ったときなので、
        # この後で結果を読む間は残る
        dialog.deleteLater()
        if not answer:
            return None
        return [voice for voice in dialog.chosen() if isinstance(voice, tuple)]

    def place_selected_rows(self) -> None:
        """選んだ行（無ければ再生位置の 1 行）をテキストとしてタイムラインへ置く"""
        media = self._current_media()
        if media is None:
            return
        rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not rows:
            rows = [
                row for row, (_, start, end) in enumerate(self._rows) if start <= self._frame < end
            ]
        segments = {self._rows[row][0] for row in rows if 0 <= row < len(self._rows)}
        if not segments:
            self.status_message.emit("置く字幕の行を選んでください")
            return
        template, _note = self._template()
        voice: Voice = (media.id, media.transcript_stream(self._stream))
        commands: list[Command] = burn_subtitles(
            self._project, template, voices=[voice], segments=segments
        )
        if not commands:
            self.status_message.emit("選んだ行はタイムラインに出ていません")
            return
        self.commands_requested.emit(commands, f"字幕を置く: {len(segments)} 行")

    def export_file(self) -> None:
        frame_range = export_range(self._project.timeline)
        if frame_range is not None:
            choice = ask_subtitle_range(self, self._project)
            if choice is None:
                return
            if choice != RANGE_WORK_AREA:
                frame_range = None
        name, _ = QFileDialog.getSaveFileName(self, "字幕を書き出す", "字幕.srt", SUBTITLE_FILTER)
        if not name:
            return
        written = save_subtitles(self._project, Path(name), frame_range=frame_range)
        self.status_message.emit(f"書き出した: {written}")


def ask_subtitle_range(parent: QWidget | None, project: Project) -> str | None:
    """書き出し範囲があるときに、字幕を範囲だけにするか全体にするかを尋ねる

    返すのは :data:`RANGE_WORK_AREA` か :data:`RANGE_ALL` 取り消したら ``None``
    既定は範囲の側 書き出しダイアログと同じく、範囲を決めた人はその所を出したくて
    決めている 全体を既定にすると、動画は範囲・字幕は全体で出して、範囲の頭の分だけ
    ずれていることに再生するまで気付かない
    """
    area = export_range(project.timeline)
    if area is None:
        return RANGE_ALL
    rate = project.settings.frame_rate
    start, end = area
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("字幕を書き出す")
    box.setText(
        f"書き出し範囲（{format_timecode(start, rate)} 〜 {format_timecode(end, rate)}）が"
        "指定されている"
    )
    box.setInformativeText(
        "範囲だけを書くと、範囲の頭を 0 秒にずらし、範囲の端にかかる字幕は端で切る"
        " 範囲で書き出した動画に合う"
    )
    in_range = box.addButton("範囲だけ", QMessageBox.ButtonRole.AcceptRole)
    whole = box.addButton("全体", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(in_range)
    box.exec()
    # 開くたびに作る窓 閉じたら捨てる 消えるのは呼んだイベントループへ戻ったときなので、
    # この後で結果を読む間は残る
    box.deleteLater()
    clicked = box.clickedButton()
    if clicked is in_range:
        return RANGE_WORK_AREA
    if clicked is whole:
        return RANGE_ALL
    return None
