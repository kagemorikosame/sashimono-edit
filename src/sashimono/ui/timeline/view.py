"""タイムラインのウィジェット

自分でプロジェクトを書き換えない 操作の結果はすべて :class:`Command` として
:attr:`TimelineView.command_requested` から外へ出す UI と AI が同じ入口を通る、
という設計をここでも守るため
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum, auto

from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QGridLayout, QMenu, QWidget

from sashimono.core.clipboard import ClipboardContent, copy_clips, cut_commands, paste_commands
from sashimono.core.commands import (
    AddClip,
    Command,
    GroupClips,
    MoveClip,
    MoveClips,
    RemoveClip,
    RemoveClips,
    RenameTrack,
    SetTrackHeights,
    SetTrackState,
    SplitClip,
    TrimClip,
    TrimClips,
    UngroupClips,
)
from sashimono.core.commands.edit import (
    DEFAULT_TRACK_HEIGHT,
    MAX_TRACK_HEIGHT,
    MIN_TRACK_HEIGHT,
    shifted_track,
)
from sashimono.core.commands.layers import places_mixed
from sashimono.core.model import (
    Clip,
    ClipId,
    GroupId,
    MediaId,
    MediaItem,
    Project,
    SceneId,
    Timeline,
    Track,
    TrackId,
    TrackKind,
    heard_stream,
    new_group_id,
)
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.media_pool import media_ids_in
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline.add_menu import AddSources, TimelineAddMenus
from sashimono.ui.timeline.auto_scroll import EdgeScroller
from sashimono.ui.timeline.drop import (
    DropGuide,
    DropPreview,
    DropSpot,
    accepts,
    local_paths,
    paint_drop_guide,
    preview_drop,
    spot_at,
)
from sashimono.ui.timeline.group_reach import draw_group_reach
from sashimono.ui.timeline.keyframes import draw_keyframes, keyframe_at
from sashimono.ui.timeline.layout import TimelineLayout, TrackBand
from sashimono.ui.timeline.painter import (
    ADD_TRACK_BUTTON_SPACE,
    ADD_TRACK_BUTTON_TEXT,
    DETAIL_MIN_WIDTH,
    clip_content,
    clip_rect_for,
    clips_in_range,
    draw_dense_clips,
    draw_playhead,
    draw_ruler,
    draw_track_add_button,
    draw_track_background,
    draw_track_header,
    track_add_button_rect,
    track_button_rects,
    track_name_rect,
)
from sashimono.ui.timeline.painter import draw_clip as paint_clip
from sashimono.ui.timeline.snap import DEFAULT_SNAP_DISTANCE, nearest_snap, snap_targets
from sashimono.ui.timeline.track_drag import TrackDragger
from sashimono.ui.timeline.track_name import TrackNameEditor
from sashimono.ui.timeline.value_line import ValueGrab, ValueLineEditor
from sashimono.ui.timeline.work_area import WorkAreaEditor
from sashimono.ui.timeline.zoom_scrollbar import ZoomScrollBar

__all__ = ["TimelineArea", "TimelineView"]

#: ホイール 1 段で拡大する倍率
ZOOM_STEP = 1.25

#: 全トラックの高さを 1 段で変える量（画素） 細かいと何度も回すことになり、
#: 粗いと最小（28）から最大（240）までが数段で終わって合わせにくい
HEIGHT_STEP = 12

#: 押している間だけ磁石を切るキー 動かしている途中で押す（押してから掴むと、範囲で選ぶ
#: 操作になる） Alt は離したときに窓のメニューへ移ってしまい、Ctrl は選び足しに使っている
SNAP_OFF_MODIFIER = Qt.KeyboardModifier.ShiftModifier

#: 吸い付いた所の縦の線を出しておく長さ（ミリ秒） 一瞬だけ出して、吸い付いたと分かれば足りる
SNAP_LINE_MS = 600

#: 境目を掴める幅（上下それぞれ、画素） 狭いと掴めず、広いと名前の行の
#: ボタンに食い込む（最小の高さ 28 のトラックでもボタンが押せる幅にしてある）
RESIZE_GRAB = 3

#: 高さを続けて変えたとみなす間隔（秒） この間隔より短く続けたホイールやキーは
#: 取り消しの 1 段にまとめる 長いと、少し間を置いて別の気持ちで変えたぶんまで
#: 一緒に戻る
HEIGHT_MERGE_SECONDS = 1.0

#: 囲んで選ぶと決めるまでに動かす距離（画素） クリックのつもりの手ぶれで
#: 選択が消えないようにする
MARQUEE_THRESHOLD = 4

#: ヘッダのボタン（TRACK_BUTTONS）の説明は長いので、メニュー用の短い名前を別に持つ
_TRACK_TOGGLES = (("muted", "ミュート"), ("solo", "ソロ"), ("locked", "ロック"))


class DragKind(Enum):
    NONE = auto()
    PLAYHEAD = auto()
    MOVE_CLIP = auto()
    TRIM_HEAD = auto()
    TRIM_TAIL = auto()
    RESIZE_TRACK = auto()
    MARQUEE = auto()
    #: 値の線（不透明度・音量）を上下に動かす 中身は value_line.py
    VALUE_LINE = auto()
    #: 値の線の点を動かす
    VALUE_KEY = auto()


@dataclass(slots=True)
class DragState:
    """ドラッグ中の状態

    ドラッグ中はプロジェクトを書き換えず、確定した時点で 1 つのコマンドを出す
    途中経過をコマンドにすると、Undo 履歴が中間状態で埋まる
    """

    kind: DragKind = DragKind.NONE
    clip_id: ClipId | None = None
    origin_track: TrackId | None = None
    #: 掴んだ位置と、クリップ先頭とのフレーム差 掴んだ場所を保ったまま動かすため
    grab_offset: int = 0
    #: 現在のプレビュー位置 描画にだけ使う
    preview_start: int = 0
    preview_track: TrackId | None = None
    preview_head_delta: int = 0
    preview_tail_delta: int = 0
    moved: bool = False
    #: 高さは掴んだ位置からの差で決める マウスの位置そのもので決めると、境目から
    #: 少し離れて掴んだだけで高さが跳ぶ
    resize_track: TrackId | None = None
    grab_y: int = 0
    origin_height: int = 0
    #: 選んでいる何本かをまとめて動かしている 行き先のトラックは変えない
    #: （何本もが別々のトラックにいるとき、どこへ移すのかが決まらない）
    group: bool = False
    #: 囲んで選ぶときの始点と今の位置、始める前の選択（Ctrl を押していれば足す）
    marquee_from: QPoint | None = None
    marquee_to: QPoint | None = None
    marquee_base: tuple[ClipId, ...] = ()


class TimelineView(QWidget):
    """クリップを並べて見せ、編集操作を受け付ける"""

    #: 再生ヘッドが動いた 引数はフレーム番号
    playhead_moved = Signal(int)
    #: 選択が変わった 引数はクリップ ID、または空文字列
    selection_changed = Signal(str)
    #: 編集操作が発生した 引数はコマンドの一覧と、履歴に出す操作名
    #:
    #: 常に一覧で渡す 1 回の操作が複数のコマンドになることがあり（分割など）、
    #: それを 1 回の取り消しで戻せるようにするため
    commands_requested = Signal(list, str)
    #: :attr:`commands_requested` と同じだが、直前の同じ操作の続き 取り消しの段を
    #: 増やさずに直前の段へまとめてもらう（ホイールで高さを変え続けるときなど）
    commands_continued = Signal(list, str)
    #: シーンを置いたクリップをダブルクリックした 引数はシーンの ID
    #: 中を開くのは窓の仕事（どのシーンを編集中かは窓が持つ）
    scene_open_requested = Signal(str)
    #: ビューは窓を知らない（テストで単体で作れるように） 知らせは信号で外へ出し、
    #: ステータスバーに出すのは窓の仕事にする
    status_message = Signal(str)
    #: 右クリックの〔追加〕→〔エイリアス〕でテンプレートの棚の物を選んだ 引数は
    #: :class:`~sashimono.compat.catalog.TemplateEntry`・置くフレーム・トラック（無ければ空）
    #: 素材の読み込みと登録が要るので、置くのは窓の仕事（:meth:`MainWindow.show_templates` と同じ）
    template_requested = Signal(object, int, str)

    #: エクスプローラーからファイルを落とした 引数はパスの一覧・フレーム・トラックの ID
    #: （トラックの無い所なら空文字列） 調べて置くのは窓の読み込みの流れ
    files_dropped = Signal(list, int, str)
    #: 素材一覧から素材を落とした 引数は素材 ID（文字列）の一覧・フレーム・トラックの ID
    media_dropped = Signal(list, int, str)
    #: 値の線をドラッグしている途中の値 履歴に残さずプレビューだけ更新する
    #: （設定パネルの :attr:`InspectorPanel.preview_requested` と同じ受け口へ繋ぐ）
    preview_requested = Signal(object)

    def __init__(
        self, project: Project, analyzer: MediaAnalyzer, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._analyzer = analyzer
        self._playhead = 0
        #: 選んでいるクリップ 最後の 1 本が「主」で、設定パネルと AI の既定の
        #: 対象になる 何本選んでも、設定パネルに出せるのは 1 本だけのため
        self._selection: tuple[ClipId, ...] = ()
        #: 選択のうち、グループの仲間として引き込まれただけのクリップ 自分で押したり
        #: 囲んだりした物は入らない 設定パネルの値はこれを除いた物へ当てる
        #: （AviUtl のグループ化と同じ 束ねるのは動かす・選ぶ所だけで、拡大率を 1 本
        #: 変えただけで仲間の拡大率まで変わると、束ねた物ごとに大きさを合わせられない）
        self._pulled: frozenset[ClipId] = frozenset()
        #: Shift+クリックで範囲を決めるときの起点 最後に選んだクリップ（選び方は
        #: 問わない AI が選んだものも含む） Shift での範囲選択そのものは起点を
        #: 動かさない 動かすと、Shift を押したまま範囲を広げ直せない
        self._anchor: ClipId | None = None
        self._last_height_change = -HEIGHT_MERGE_SECONDS
        self._drag = DragState()
        self._follow_playhead = True
        self._clipboard: ClipboardContent | None = None
        #: 高さのドラッグ中だけ持つ、掴む前のプロジェクト 途中の高さは描画のため
        #: だけに当て、離したときにこれへ戻してからコマンドを出す
        self._resize_base: Project | None = None
        #: 枠を描くクリップの控え ``(プロジェクト, 選択, 結果)`` :meth:`_highlighted` を見る
        self._highlight_cache: tuple[Project, tuple[ClipId, ...], frozenset[ClipId]] | None = None

        #: スクロールバーはビューの外（下と右）に並べる 置くのは :class:`TimelineArea` で、
        #: そこで親が付け替わる それまではビューの子として隠しておく 親を持たせずに作ると、
        #: Python だけが持つ窓になり、ビューと別々の順でごみ集めに壊されて落ちることがある
        #: つまみの端を掴むと、横は拡大率、縦はトラックの高さが変わる（:meth:`_on_span_dragged`）
        self._hbar = ZoomScrollBar(Qt.Orientation.Horizontal, self)
        self._vbar = ZoomScrollBar(Qt.Orientation.Vertical, self, overscan=True)
        self._hbar.hide()
        self._vbar.hide()
        self._hbar.setAccessibleName("タイムラインの横スクロール")
        self._vbar.setAccessibleName("タイムラインの縦スクロール")
        self._hbar.valueChanged.connect(self._on_hbar)
        self._vbar.valueChanged.connect(self._on_vbar)
        self._syncing_bars = False
        #: つまみの端を掴んだときの表示 伸び縮みは掴んだ時点の目盛りで数える
        self._span_base: TimelineLayout | None = None
        for bar in (self._hbar, self._vbar):
            bar.span_started.connect(functools.partial(self._on_span_started, bar))
            bar.span_dragged.connect(functools.partial(self._on_span_dragged, bar))
            bar.span_finished.connect(functools.partial(self._on_span_finished, bar))
        self._view_layout = TimelineLayout()
        #: 右クリックの〔追加〕に並べる物の出どころ 試験で差し替える
        self.add_sources = AddSources()
        self._add_menus = TimelineAddMenus(self)
        #: 開いているシーン（メインなら ``None``） 〔追加〕→〔シーン〕から自分自身を外す
        self._open_scene: SceneId | None = None
        self._add_button_hovered = False
        #: ヘッダで書き換えている名前の入力欄（:meth:`begin_rename`） 無ければ ``None``
        self._name_editor: TrackNameEditor | None = None
        #: 磁石（吸着 :meth:`set_snap`） 既定は入（設定の既定と同じ）
        self._snap_enabled = True
        self._snap_distance = DEFAULT_SNAP_DISTANCE
        #: 動かしている途中に押しているキー（マウスとドラッグの知らせから）
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier
        #: 吸い付く先の覚え（プロジェクト・動かしている物・再生位置, フレームの並び）
        #: マウスが動くたびに全クリップを舐めないため
        self._snap_cache: tuple[object, list[int]] | None = None
        #: 吸い付いた所に出す縦の線 少しして消す（:data:`SNAP_LINE_MS`）
        self._snap_line: int | None = None
        self._snap_timer = QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.setInterval(SNAP_LINE_MS)
        self._snap_timer.timeout.connect(self._hide_snap)
        #: 書き出し範囲の Shift+ドラッグと、その帯 ほかのドラッグとは別に持つ
        self._work_area = WorkAreaEditor(self._request)
        #: ファイルや素材を引いてきている間の、落ちる所の目安 引いていなければ ``None``
        self._drop_preview: DropPreview | None = None
        #: 動画の映像と音声を分けて置くか（:meth:`set_split_audio`）
        #: 既定は設定の既定と同じ 窓が渡す前に落とされても、落とした後と同じ目安を出す
        self._split_audio = True
        #: ヘッダを掴んでトラックの順を入れ替えるドラッグ
        self._track_mover = TrackDragger(self._request)
        #: ドラッグ中に端へ寄ったら表示を送る
        self._edge_scroll = EdgeScroller(self._on_edge_scroll, self)
        #: クリップの上の不透明度・音量の線
        self._value_lines = ValueLineEditor(
            self._request, self.preview_requested.emit, self._show_project, self.update
        )

        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(160)
        # 中身は自前で描いているので、読み上げソフトにはこの名前しか伝わらない
        self.setAccessibleName("タイムライン")
        self.setAccessibleDescription(
            "選んだクリップのトラックは Shift+M でミュート、Shift+S でソロ、Shift+L でロック"
        )

    # --- 外から差し替えるもの ---

    @property
    def project(self) -> Project:
        return self._project

    def set_project(self, project: Project) -> None:
        self._project = project
        # 消えたクリップは選択から外す 存在しない ID を持ち続けると、次の操作で
        # 「見つからない」例外になる
        remaining = tuple(c for c in self._selection if project.timeline.locate_clip(c))
        if remaining != self._selection:
            self.set_selection(remaining)
        # 長さやトラックの数が変わると、スクロールできる幅と高さも変わる
        self._sync_scroll_bars()
        self.update()

    def set_split_audio(self, split: bool) -> None:
        """動画の映像と音声を分けて置くか 設定（:attr:`Preferences.media_split`）から

        落とす前の目安を、窓が実際に置くのと同じ置き方で求めるため 目安だけ設定を
        見ないと、目安に無いレイヤーが落とした後に増える
        """
        self._split_audio = split

    def set_value_lines(self, shown: bool) -> None:
        """クリップの上に不透明度・音量の線を出すか 設定（:attr:`Preferences.value_lines`）から"""
        self._value_lines.enabled = shown
        self.update()

    def _show_project(self, project: Project) -> None:
        """値の線のドラッグの途中と終わりに、描くプロジェクトを差し替える 履歴には載せない"""
        self._project = project
        self.update()

    # --- スクロール ---

    @property
    def _layout(self) -> TimelineLayout:
        return self._view_layout

    @_layout.setter
    def _layout(self, layout: TimelineLayout) -> None:
        # 拡大・スクロール・再生ヘッドの追従と、変える所があちこちにあるので、
        # 変わったらここで必ずスクロールバーを合わせる 呼び忘れると、バーの位置と
        # 見えている所がずれる
        self._view_layout = layout
        self._sync_scroll_bars()

    @property
    def horizontal_scroll_bar(self) -> ZoomScrollBar:
        return self._hbar

    @property
    def vertical_scroll_bar(self) -> ZoomScrollBar:
        return self._vbar

    @property
    def view_layout(self) -> TimelineLayout:
        """いまの拡大率とスクロール位置"""
        return self._view_layout

    def _scrollable_frames(self) -> float:
        """横にスクロールできる長さ（フレーム） プロジェクトの長さに余白を足す

        余白が無いと、最後のクリップの後ろへ置く場所を右端でしか見られない
        画面の半分を足して、末尾の後ろにも置き場が見えるようにする

        再生ヘッドが末尾より先にあれば、そこまでを長さに含める 矢印キーで末尾の
        先へ進めたとき、含めないと再生ヘッドが画面の外へ出たまま追えない
        """
        span = self._view_layout.frames_in(self.width())
        return max(self._project.duration, self._playhead) + span * 0.5

    def _scrollable_height(self) -> int:
        """縦にスクロールできる高さ（画素、目盛りの下から） 縦の範囲はここだけで決める

        トラックの帯の下に何かを描き足すときは、その高さをここに足すこと 足さないと、
        いちばん下まで送ってもその部品が画面の外に残る いまは「＋ トラック追加」だけ

        並びは描いているもの（:meth:`_painted_timeline`）で数える ファイルを引いている間は
        仮の行が末尾に増え、ボタンもその下へずれる
        """
        tracks = self._view_layout.content_height(self._painted_timeline()) - Metrics.RULER_HEIGHT
        return tracks + ADD_TRACK_BUTTON_SPACE

    def _sync_scroll_bars(self) -> None:
        """スクロールバーの範囲と位置を、いまの表示に合わせる

        横は画素で数える 1 フレームが 1 画素に満たない拡大率でも、つまみを
        滑らかに動かせるように 縦の範囲はトラックを並べた高さから

        範囲は中身の大きさだけで決め、表示の位置はその範囲へ丸める 今の位置を
        範囲に含めると、末尾で Shift+ホイールを回すたびに範囲が伸びて空白へどこまでも
        進め、トラックを消したあとも消えたトラックの高さぶん下を見たまま戻らなかった
        """
        layout = self._view_layout
        scale = layout.pixels_per_frame
        visible = max(1, self.width() - Metrics.TRACK_HEADER_WIDTH)
        content = round(self._scrollable_frames() * scale)
        maximum = max(content - visible, 0)

        rows = max(1, self.height() - Metrics.RULER_HEIGHT)
        vertical_max = max(self._scrollable_height() - rows, 0)

        # 位置を範囲へ丸めてビューにも当てる バーは範囲を縮めると値を自分で丸めるが、
        # その知らせは下で止めているので、ビューの側は自分で合わせないとずれたまま残る
        clamped = layout
        if layout.scroll_frame * scale > maximum:
            clamped = clamped.scrolled_to(maximum / scale)
        if layout.scroll_y > vertical_max:
            clamped = clamped.scrolled_vertically(vertical_max)
        if clamped != layout:
            self._view_layout = layout = clamped
            self.update()
        value = min(round(layout.scroll_frame * scale), maximum)

        self._syncing_bars = True
        try:
            self._hbar.setRange(0, maximum)
            self._hbar.setPageStep(visible)
            self._hbar.setSingleStep(max(1, visible // 20))
            self._hbar.setValue(value)
            self._vbar.setRange(0, vertical_max)
            self._vbar.setPageStep(rows)
            self._vbar.setSingleStep(Metrics.MIN_TRACK_HEIGHT)
            self._vbar.setValue(layout.scroll_y)
        finally:
            self._syncing_bars = False
        # 縦のバーは、トラックが全部見えているときも出す（つまみが全体を占める）
        # #145 では「収まれば隠す」にしていた そのころのバーは送るだけで、収まっていれば
        # 使い道が無かった 今はつまみの端でトラックの高さを変えられるので（Issue #27）、
        # 隠すと、いちばん使う「収まっている状態から高くする・低くする」ができない
        # 並べる前（まだビューの子のとき）は出さない 出すとビューの絵の上に重なる
        self._vbar.setVisible(self._vbar.parentWidget() is not self)

    def _on_hbar(self, value: int) -> None:
        if self._syncing_bars:
            return
        layout = self._view_layout
        self._view_layout = layout.scrolled_to(value / layout.pixels_per_frame)
        self.update()

    def _on_vbar(self, value: int) -> None:
        if self._syncing_bars:
            return
        self._view_layout = self._view_layout.scrolled_vertically(value)
        self.update()

    # --- つまみの端で表示の大きさを変える（中身は zoom_scrollbar.py） ---

    def _on_span_started(self, bar: ZoomScrollBar) -> None:
        self._span_base = self._view_layout
        if bar is self._vbar:
            # 高さは離すまで描画にだけ当てる（境目のドラッグと同じ） 途中をコマンドに
            # すると、取り消しの履歴が伸び縮みの途中で埋まる
            self._resize_base = self._project

    def _on_span_dragged(
        self, bar: ZoomScrollBar, start: float, end: float, moving_start: bool
    ) -> None:
        """つまみの見えている範囲が ``start``〜``end`` になるよう、表示の大きさを変える

        値は掴んだ時点の目盛り（横は画素、縦はトラックの帯の画素）で来る 動かしていない
        側の端は、見ていた場所をそのまま保つ
        """
        base = self._span_base
        if base is None or end <= start:
            return
        if bar is self._hbar:
            scale = base.pixels_per_frame
            first, last = start / scale, end / scale
            visible = max(1, self.width() - Metrics.TRACK_HEADER_WIDTH)
            zoomed = TimelineLayout(
                pixels_per_frame=visible / (last - first), scroll_y=base.scroll_y
            )
            left = last - zoomed.frames_in(self.width()) if moving_start else first
            self._layout = zoomed.scrolled_to(left)
            self.update()
            return
        project = self._resize_base
        if project is None:
            return
        rows = max(1, self.height() - Metrics.RULER_HEIGHT)
        factor = rows / (end - start)
        heights = tuple((t.id, round(t.height * factor)) for t in project.timeline.tracks)
        self._project = SetTrackHeights(heights).apply(project)
        top = end * factor - rows if moving_start else start * factor
        self._layout = self._view_layout.scrolled_vertically(round(top))
        self.update()

    def _on_span_finished(self, bar: ZoomScrollBar) -> None:
        self._span_base = None
        if bar is not self._vbar:
            return
        base, self._resize_base = self._resize_base, None
        if base is None:
            return
        preview, self._project = self._project, base
        if preview.timeline.tracks != base.timeline.tracks:
            command = SetTrackHeights(tuple((t.id, t.height) for t in preview.timeline.tracks))
            self._request([command], command.label)
        self._sync_scroll_bars()
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        # 幅が変わると 1 画面に入るフレーム数（バーのつまみの大きさ）が変わる
        self._sync_scroll_bars()

    @property
    def playhead(self) -> int:
        return self._playhead

    def set_playhead(self, frame: int, *, follow: bool = True) -> None:
        frame = max(0, frame)
        if frame == self._playhead:
            return
        self._playhead = frame
        if follow and self._follow_playhead:
            self._layout = self._layout.ensure_visible(frame, self.width())
        else:
            # 追わないとき（目盛りを掴んで動かしているときなど）も、末尾の先へ出た
            # 再生ヘッドのぶん横の長さが変わる 合わせないと、バーの端まで寄せても
            # 再生ヘッドまで届かない
            self._sync_scroll_bars()
        self.update()

    @property
    def selected_clip(self) -> ClipId | None:
        """主に選んでいるクリップ（最後に選んだもの）"""
        return self._selection[-1] if self._selection else None

    @property
    def selected_clips(self) -> tuple[ClipId, ...]:
        """選んでいるクリップすべて 選んだ順"""
        return self._selection

    def select(self, clip_id: ClipId | None) -> None:
        """1 本だけを選ぶ ``None`` なら選択を解く"""
        self.set_selection((clip_id,) if clip_id is not None else ())
        self._anchor = clip_id

    @property
    def edit_targets(self) -> tuple[ClipId, ...]:
        """設定パネルの値を当てるクリップ 選んだ順 グループの仲間として引き込まれただけの物を除く

        主のクリップは必ず入る（押した 1 本なので引き込まれた物ではない）
        """
        primary = self.selected_clip
        return tuple(c for c in self._selection if c not in self._pulled or c == primary)

    def set_selection(self, clip_ids: Iterable[ClipId], *, pulled: Iterable[ClipId] = ()) -> None:
        """選択を丸ごと入れ替える 重なった ID は 1 つにする（最後の位置を残す）

        ``pulled`` はグループの仲間として引き込んだだけのクリップ（:attr:`edit_targets`）
        """
        ordered = tuple(reversed(dict.fromkeys(reversed(tuple(clip_ids)))))
        kept = frozenset(pulled) & frozenset(ordered)
        if ordered == self._selection and kept == self._pulled:
            return
        self._selection = ordered
        self._pulled = kept
        self._anchor = self.selected_clip
        # 主のクリップが同じでも知らせる 選択から外したクリップへ、設定パネルの
        # まとめ当てが届いてしまう
        self.selection_changed.emit(self.selected_clip or "")
        self.update()

    def select_all(self) -> None:
        """全トラックのクリップを選ぶ ロックしたトラックも含める（見るだけなら困らない）"""
        self.set_selection(c.id for t in self._project.timeline.tracks for c in t.clips)
        if len(self._selection) > 1:
            self.status_message.emit(f"{len(self._selection)} 本を選択")

    def zoom(self, factor: float) -> None:
        """ウィジェットの中央を基準に拡大・縮小する メニューやボタンから"""
        self._layout = self._layout.zoomed(factor, anchor_x=self.width() / 2.0)
        self.update()

    def zoom_to_fit(self) -> None:
        """タイムライン全体が収まる倍率にする"""
        duration = max(1, self._project.duration)
        usable = max(1, self.width() - Metrics.TRACK_HEADER_WIDTH)
        self._layout = TimelineLayout(pixels_per_frame=usable / duration * 0.98)
        self.update()

    # --- 描画 ---

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt の命名規約
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Colors.TIMELINE_BACKGROUND)

        timeline = self._painted_timeline()
        width = self.width()

        for band in self._layout.bands(timeline):
            if band.bottom <= Metrics.RULER_HEIGHT or band.top >= self.height():
                continue
            draw_track_background(painter, band, width)

        start_frame, end_frame = self._layout.visible_range(width)
        scale = self._layout.pixels_per_frame
        selected = self._highlighted()
        # 設定パネルが出している 1 本 選んだ仲間と見分けて描く（主の選択は最後に選んだ物）
        editing = self.selected_clip
        media: dict[MediaId, MediaItem] | None = None
        for band in self._layout.bands(timeline):
            if band.bottom <= Metrics.RULER_HEIGHT or band.top >= self.height():
                continue
            # 名前が入らない幅のクリップは、まとめて色の帯にする 1 本ずつ描くと
            # 全体表示で数千本を描くことになり、60fps の予算に収まらない
            dense: list[Clip] = []
            for clip in clips_in_range(band.track, start_frame, end_frame):
                if clip.duration * scale < DETAIL_MIN_WIDTH:
                    dense.append(clip)
                    continue
                rect = clip_rect_for(clip, band, self._layout, width)
                if rect is not None:
                    self._paint_detailed(
                        painter, band, clip, rect, clip.id in selected, clip.id == editing
                    )
            sound_only = None
            if dense and band.track.kind is TrackKind.MIXED:
                # 素材の引き表は、細い帯のあるレイヤーが出たときに 1 度だけ作って使い回す
                # レイヤーごとに作ると、素材とレイヤーが多い作品で描くたびに掛け算で重くなる
                if media is None:
                    media = {item.id: item for item in self._project.media}
                sound_only = self._sound_only(band.track, media)
            draw_dense_clips(
                painter, band, dense, self._layout, width, selected, sound_only, editing
            )

        # グループ制御の受け持ちは、クリップの上に薄く重ねる（中身が隠れない濃さ）
        draw_group_reach(
            painter, self._layout, timeline, (width, self.height()), set(self._selection)
        )
        self._draw_drag_preview(painter)
        self._paint_snap(painter)
        self._work_area.paint_tracks(
            painter, self._layout, width, self.height(), timeline.work_area
        )

        # 役割（絵と音）で見る 種類で見ると、レイヤーはどちらにも入らず、ミュートもソロも
        # していないのに名前が薄く出る レイヤーは絵か音のどちらかが出ていれば出ている側
        active = {
            track.id
            for track in (*timeline.active_picture_tracks(), *timeline.active_sound_tracks())
        }
        for band in self._layout.bands(timeline):
            if band.bottom <= Metrics.RULER_HEIGHT or band.top >= self.height():
                continue
            draw_track_header(painter, band, active=band.track.id in active)
        self._paint_add_button(painter)
        self._track_mover.paint(painter, self._layout, self._project.timeline, width)

        draw_ruler(painter, self._layout, width, self._project.rate)
        self._work_area.paint_ruler(painter, self._layout, width, timeline.work_area)
        draw_playhead(painter, self._layout, self._playhead, self.height())
        self._paint_drop_guide(painter)

    def _sound_only(self, track: Track, media: dict[MediaId, MediaItem]) -> Callable[[Clip], bool]:
        """レイヤーのクリップが音だけか（細い帯を音声の色で塗るか）

        素材は ``media`` の引き表から引く 帯になるクリップは数千本あり、1 本ごとに素材の
        一覧をなめると全体表示の描画が 60fps の予算を超える
        """

        def judge(clip: Clip) -> bool:
            item = media.get(clip.media_id) if clip.media_id is not None else None
            picture, sound = clip_content(track, clip, item)
            return sound and not picture

        return judge

    def _highlighted(self) -> frozenset[ClipId]:
        """選んだ枠を描くクリップ 選んだものと、リンクした相手（映像と音声の組）

        リンクした相手は選択には入れない 相手は動かすのも消すのも分割するのも
        コマンドの側が一緒に扱うので、選択に入れると同じ組へ 2 度当てることになる
        ただ枠を映像の側にしか描かないと、音声も一緒に動くことが見えない（Issue #27）
        ので、描くときだけ相手にも同じ枠を付ける

        全クリップを舐めるのは 2 度だけ 選んだ 1 本ごとに相手を探すと、全部を選んだとき
        （1 万本）に描くたびに 1 万 × 1 万回回る 選択とプロジェクトが同じ間は覚えておく
        """
        cached = self._highlight_cache
        if cached is not None and cached[0] is self._project and cached[1] == self._selection:
            return cached[2]
        timeline = self._project.timeline
        chosen = set(self._selection)
        # リンクも 1 度の走査で集める 1 本ずつ locate_clip で探すと、それ自体が
        # 全クリップを舐めるので、全部を選んだときに 1 万 × 1 万回になる
        links = (
            {
                clip.link_group
                for track in timeline.tracks
                for clip in track.clips
                if clip.id in chosen and clip.link_group is not None
            }
            if chosen
            else set()
        )
        found = set(chosen)
        if links:
            # ロックしたトラックの相手には付けない 移動もトリムもその相手を動かさない
            # （:meth:`_linked_partners` と同じ決まり） 枠を付けると一緒に動くように見える
            # 選んだクリップそのものは、ロックしていても選んだ印として残す
            found.update(
                clip.id
                for track in timeline.tracks
                if not track.locked
                for clip in track.clips
                if clip.link_group in links
            )
        result = frozenset(found)
        self._highlight_cache = (self._project, self._selection, result)
        return result

    def _linked_partners(self, clip: Clip) -> list[tuple[Track, Clip]]:
        """リンクした相手（自分を除く） 相手のトラックがロックしていれば外す

        :class:`MoveClip` と :class:`TrimClip` は、ロックしたトラックの相手を動かさない
        動かない相手に落下先の枠を出すと、離したときの結果と食い違う
        """
        if clip.link_group is None:
            return []
        return [
            (track, member)
            for track, member in self._project.timeline.linked_clips(clip.link_group)
            if member.id != clip.id and not track.locked
        ]

    def _paint_detailed(
        self,
        painter: QPainter,
        band: TrackBand,
        clip: Clip,
        rect: QRect,
        selected: bool,
        editing: bool = False,
    ) -> None:
        media = self._project.find_media(clip.media_id) if clip.media_id is not None else None
        scene = self._project.find_scene(clip.scene_id) if clip.scene_id is not None else None
        paint_clip(
            painter,
            clip,
            band,
            self._layout,
            self._project.rate,
            media=media,
            filmstrip=self._analyzer.filmstrip(media) if media is not None else None,
            # 鳴らす音の波形を出す 素材だけで引くと、音声が何本もある動画を音ごとに分けて
            # 置いたとき、どのレイヤーにも 1 本目の波形が出る
            waveform=self._analyzer.waveform(media, heard_stream(band.track, clip))
            if media is not None
            else None,
            selected=selected,
            clip_rect=rect,
            scene_name=scene.name
            if scene is not None
            else ("（消えたシーン）" if clip.scene_id else None),
            editing=editing,
        )
        draw_keyframes(painter, clip, self._layout, rect, selected=selected)
        self._value_lines.paint(
            painter,
            self._project,
            band.track,
            clip,
            self._layout,
            rect,
            band.height,
            selected=selected,
        )

    def _draw_drag_preview(self, painter: QPainter) -> None:
        """ドラッグ中の落下先を枠線で示す

        実際のクリップを動かさずに枠だけ出すことで、途中経過が Undo 履歴に
        残らず、かつ落ちる位置は分かる
        """
        if self._drag.kind is DragKind.MARQUEE:
            self._draw_marquee(painter)
            return
        if self._drag.kind not in (DragKind.MOVE_CLIP, DragKind.TRIM_HEAD, DragKind.TRIM_TAIL):
            return
        if self._drag.clip_id is None:
            return
        located = self._project.timeline.locate_clip(self._drag.clip_id)
        if located is None:
            return
        _, clip = located
        bands = {b.track.id: b for b in self._layout.bands(self._project.timeline)}
        painter.setPen(QPen(Colors.SELECTION, 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self._drag.group:
            # 何本かをまとめて動かすときは、動く全員の落下先を出す 掴んだ 1 本の枠
            # だけだと、ほかのクリップがどこへ落ちるか分からない
            # トラックを跨いだぶんも :class:`MoveClips` と同じ決まり（同じ種類の並びで数える）で
            # ずらして出す 元のトラックに出すと、離した後に別のトラックへ移って驚く
            delta = self._drag.preview_start - clip.timeline_start
            _, landings = self._group_landings(self._drag)
            for landing, member in landings:
                if landing in bands:
                    self._dash_rect(
                        painter,
                        bands[landing],
                        member.timeline_start + delta,
                        member.timeline_end + delta,
                    )
            return

        target_track = self._drag.preview_track or self._drag.origin_track
        band = bands.get(target_track) if target_track is not None else None
        if band is None:
            return
        if self._drag.kind is DragKind.MOVE_CLIP:
            start, end = self._drag.preview_start, self._drag.preview_start + clip.duration
            head = tail = self._drag.preview_start - clip.timeline_start
        else:
            head, tail = self._drag.preview_head_delta, self._drag.preview_tail_delta
            start, end = clip.timeline_start + head, clip.timeline_end + tail
        self._dash_rect(painter, band, start, end)
        # リンクした相手（映像と音声の組）にも同じ枠を出す 相手は自分のトラックに
        # 残ったまま、同じだけ動く・削れる（:class:`MoveClip` :class:`TrimClip` の決まり）
        for partner_track, partner in self._linked_partners(clip):
            partner_band = bands.get(partner_track.id)
            if partner_band is not None:
                self._dash_rect(
                    painter,
                    partner_band,
                    partner.timeline_start + head,
                    partner.timeline_end + tail,
                )

    def _dash_rect(self, painter: QPainter, band: TrackBand, start: int, end: int) -> None:
        left = self._layout.frame_to_x(start)
        right = self._layout.frame_to_x(end)
        painter.drawRect(int(left), band.top + 1, max(2, int(right - left)), band.height - 3)

    def _draw_marquee(self, painter: QPainter) -> None:
        origin, current = self._drag.marquee_from, self._drag.marquee_to
        if origin is None or current is None or origin == current:
            return
        fill = QColor(Colors.SELECTION)
        fill.setAlpha(40)
        painter.setPen(QPen(Colors.SELECTION, 1, Qt.PenStyle.DashLine))
        painter.setBrush(fill)
        painter.drawRect(QRect(origin, current).normalized())

    def _trimmable_selection(self) -> tuple[ClipId, ...]:
        """選んだうち、端を動かせるもの ロックしたトラックのものは外す

        動かすときと違い、リンクした相手のトラックは見ない トリムは相手のトラックが
        ロックされていれば、その相手だけが元の長さで残る（:class:`TrimClip` の決まり）
        """
        timeline = self._project.timeline
        return tuple(
            clip_id
            for clip_id in self._selection
            if (located := timeline.locate_clip(clip_id)) is not None and not located[0].locked
        )

    def _movable_selection(self) -> tuple[ClipId, ...]:
        """選んだうち、ロックしていないトラックのもの

        Ctrl+A はロックしたトラックのクリップも選ぶ（見るだけなら困らない） それを
        そのまま :class:`MoveClips` へ渡すと、ほかのクリップまで動かせなくなる
        動かすときは、ロックしたトラックのものを最初から外す
        """
        timeline = self._project.timeline
        movable: list[ClipId] = []
        for clip_id in self._selection:
            located = timeline.locate_clip(clip_id)
            if located is None or located[0].locked:
                continue
            # リンクした相手がロックしたトラックにいても外す :class:`MoveClips` は
            # そういう組を断るので、残すと枠では動いて見えたのに離すと何も動かない
            link = located[1].link_group
            if link is not None and any(t.locked for t, _ in timeline.linked_clips(link)):
                continue
            movable.append(clip_id)
        return tuple(movable)

    def _moving_members(self) -> list[tuple[TrackId, Clip]]:
        """まとめて動かすときに動くクリップ 動かせる選択とリンクした相手

        :class:`MoveClips` と同じ決まりで集める
        """
        timeline = self._project.timeline
        found: dict[ClipId, tuple[TrackId, Clip]] = {}
        for clip_id in self._movable_selection():
            located = timeline.locate_clip(clip_id)
            if located is None:
                continue
            track, clip = located
            members = (
                list(timeline.linked_clips(clip.link_group))
                if clip.link_group is not None
                else [(track, clip)]
            )
            for member_track, member in members:
                found.setdefault(member.id, (member_track.id, member))
        return list(found.values())

    def _group_floor(self) -> int:
        """まとめて動かすとき、掴んだクリップを置ける最も前の位置

        掴んだ 1 本だけで 0 に止めると、それより前にいるほかのクリップが先頭より前へ
        出る 枠では動かせたように見えるのに、離すと断られる
        """
        if self._drag.clip_id is None:
            return 0
        located = self._project.timeline.locate_clip(self._drag.clip_id)
        members = self._moving_members()
        if located is None or not members:
            return 0
        earliest = min(member.timeline_start for _, member in members)
        return located[1].timeline_start - earliest

    # --- 入力 ---

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt の命名規約
        delta = event.angleDelta().y()
        sideways = event.angleDelta().x()
        modifiers = event.modifiers()
        zooming = modifiers & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        )
        if sideways and not zooming:
            # タッチパッドの 2 本指や、横に倒すホイールは横の量で来る 斜めに
            # なぞると縦と横が同じ 1 回に乗るので、横を当ててから縦も続けて見る
            # 量は目盛り 1 段（120）で Shift+ホイールの 1 段と同じだけ動かす
            # タッチパッドは細かい量で何度も来るので、比例させないと飛び飛びになる
            frames = self._layout.frames_in(self.width()) * 0.15 * sideways / 120.0
            self._layout = self._layout.scrolled_to(self._layout.scroll_frame - frames)
            if delta == 0 or modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.update()
                event.accept()
                return
        if delta == 0:
            return

        over_header = event.position().x() < Metrics.TRACK_HEADER_WIDTH
        if modifiers & Qt.KeyboardModifier.ControlModifier and over_header:
            # ヘッダの上では全トラックの高さを変える タイムラインの上の Ctrl+ホイールは
            # 横の拡大なので、どちらを変えたいかをマウスの位置で分ける
            self.adjust_track_heights(HEIGHT_STEP if delta > 0 else -HEIGHT_STEP)
            event.accept()
            return
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            # マウス位置を基準に拡大する 中央基準だと、拡大するたびに
            # 見ていた場所が画面外へ逃げる
            factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
            self._layout = self._layout.zoomed(factor, anchor_x=event.position().x())
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            frames = self._layout.frames_in(self.width()) * 0.15
            self._layout = self._layout.scrolled_to(
                self._layout.scroll_frame - (frames if delta > 0 else -frames)
            )
        else:
            self._layout = self._layout.scrolled_vertically(self._layout.scroll_y - (delta // 4))
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position().toPoint()
        if self._work_area.press(self._layout, position, event.modifiers()):
            return

        resizing = self._resize_band_at(position)
        if resizing is not None:
            self._resize_base = self._project
            self._drag = DragState(
                kind=DragKind.RESIZE_TRACK,
                resize_track=resizing.track.id,
                grab_y=position.y(),
                origin_height=resizing.track.height,
            )
            return

        if self._toggle_track_button(position):
            return
        if self._press_add_button(position):
            return
        if self._track_mover.press(self._layout, self._project.timeline, position):
            return

        if position.x() < Metrics.TRACK_HEADER_WIDTH:
            # ヘッダ（トラック名とボタンの列）は時間の軸の外 ここを再生ヘッドの
            # ドラッグにすると、ボタンを押し損ねたり名前を押したりしただけで、
            # 左端より前（負のフレーム）が 0 に丸められて再生ヘッドが先頭へ飛んでいた
            # （Issue #27） 目盛りの左の角も同じ ヘッダの上に時間は無い
            return

        if position.y() < Metrics.RULER_HEIGHT:
            self._drag = DragState(kind=DragKind.PLAYHEAD)
            self._scrub(position)
            return

        modifiers = event.modifiers()
        if not modifiers and self._press_keyframe(position):
            return
        # Shift+クリックは範囲選択 線の上でも線を掴まない（掴むと選び直しになって範囲が取れない）
        shifted = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if not shifted and self._press_value_line(position, modifiers):
            return
        adding = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        ranged = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        hit = self._clip_at(position)
        if hit is None:
            # 空いた所は、再生ヘッドをそこへ動かし、ドラッグすれば囲んで選ぶ
            # Ctrl を押していれば今の選択に足す（押していなければ選び直し）
            base = self._selection if adding else ()
            if not adding:
                self.select(None)
            self._drag = DragState(
                kind=DragKind.MARQUEE,
                marquee_from=position,
                marquee_to=position,
                marquee_base=base,
            )
            self._scrub(position)
            return

        track_id, clip = hit
        if adding:
            # 足したクリップはそのまま掴んで動かせる 外したクリップは掴まない
            # （選んでいないものを動かすことになる） グループはまとめて足し引きする
            self._toggle(clip.id)
            if clip.id not in self._selection:
                return
        elif ranged and self._anchor is not None:
            self._select_range(self._anchor, clip.id)
            return
        elif clip.id in self._selection:
            # 選んだ何本かのうちの 1 本を掴んだ 選び直すと、まとめて動かせない
            self.set_selection((*self._selection, clip.id), pulled=self._pulled - {clip.id})
            self._anchor = clip.id
        else:
            # グループに入っていれば、仲間ごと選ぶ 掴んだ 1 本が主
            self._select_with_group(clip.id)
            self._anchor = clip.id

        edge = self._edge_at(position, clip)
        frame = self._layout.frame_at(position.x())
        self._drag = DragState(
            kind=edge,
            clip_id=clip.id,
            origin_track=track_id,
            grab_offset=frame - clip.timeline_start,
            preview_start=clip.timeline_start,
            preview_track=track_id,
            # 何本も選んでいれば、動かすのもトリムもまとめて当てる
            group=edge in (DragKind.MOVE_CLIP, DragKind.TRIM_HEAD, DragKind.TRIM_TAIL)
            and len(self._selection) > 1,
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        position = event.position().toPoint()
        # 磁石を一時的に切るキーは動かしている途中で見る 端で送り直すときも同じ物を使う
        self._drag_modifiers = event.modifiers()
        if self._drag_to(position):
            horizontal, vertical = self._edge_scroll_axes()
            self._edge_scroll.follow(
                position, self._drag_bounds(), horizontal=horizontal, vertical=vertical
            )
            return
        self._update_cursor(position, event.modifiers())
        button = self._track_button_at(position)
        self.setToolTip(button[1] if button is not None else "")
        self._hover_add_button(position)

    def _drag_to(self, position: QPoint) -> bool:
        """ドラッグ中なら ``position`` まで進めて真を返す 端で表示を送った後にも呼ぶ"""
        if self._work_area.dragging:
            self._work_area.move(self._layout, position.x())
            self.update()
            return True
        if self._track_mover.pressed:
            self._track_mover.move(self._layout, self._project.timeline, position)
            self.update()
            return True

        if self._drag.kind is DragKind.NONE:
            return False

        if self._drag.kind is DragKind.PLAYHEAD:
            # 表示の外へ出た分は端で止める 送るのは :class:`EdgeScroller` の仕事で、
            # 再生ヘッドだけが先に画面の外へ飛ぶと、どこまで進んだのか見えない
            inside = min(max(position.x(), Metrics.TRACK_HEADER_WIDTH), self.width() - 1)
            self._scrub(QPoint(inside, position.y()))
            return True

        if self._drag.kind is DragKind.RESIZE_TRACK:
            self._preview_height(position.y())
            return True

        if self._drag.kind is DragKind.MARQUEE:
            self._update_marquee(position)
            return True

        if self._drag.kind in (DragKind.VALUE_LINE, DragKind.VALUE_KEY):
            self._value_lines.move(self._layout, position)
            return True

        self._drag_clip_to(position)
        return True

    def _edge_scroll_axes(self) -> tuple[bool, bool]:
        """いまのドラッグで、端へ寄ったときに送る向き ``(横, 縦)``

        高さの変更では送らない 境目が画面の端へ来るたびに送ると、伸ばしている帯が
        指から逃げる トラックの並べ替えは縦だけ（時間の軸は関係ない）
        """
        if self._work_area.dragging:
            return True, False
        if self._track_mover.pressed:
            return False, True
        kind = self._drag.kind
        if kind in (DragKind.MOVE_CLIP, DragKind.MARQUEE):
            return True, True
        if kind in (DragKind.PLAYHEAD, DragKind.TRIM_HEAD, DragKind.TRIM_TAIL):
            return True, False
        return False, False

    def _drag_bounds(self) -> QRect:
        """送らずに動ける範囲 ヘッダと目盛りの外側（時間の軸とトラックの帯が見えている所）"""
        return QRect(
            Metrics.TRACK_HEADER_WIDTH,
            Metrics.RULER_HEIGHT,
            max(1, self.width() - Metrics.TRACK_HEADER_WIDTH),
            max(1, self.height() - Metrics.RULER_HEIGHT),
        )

    def _on_edge_scroll(self, dx: float, dy: float, position: QPoint) -> None:
        """端で表示を送り、同じマウスの位置でドラッグを進め直す

        送っただけではマウスの下のフレームが変わったことをドラッグが知らない 進め直さないと、
        表示だけが先へ行き、再生ヘッドやクリップが置いていかれる
        """
        layout = self._view_layout
        if dx:
            layout = layout.scrolled_to(layout.scroll_frame + dx / layout.pixels_per_frame)
        if dy:
            layout = layout.scrolled_vertically(layout.scroll_y + round(dy))
        self._layout = layout
        self._drag_to(position)
        self.update()

    def _drag_clip_to(self, position: QPoint) -> None:
        """クリップの移動とトリムを ``position`` まで進める 途中は枠を描くだけ"""
        # 押しただけ（選ぶだけ）では断らない 動かし始めた所で断り、ドラッグをやめる
        if not self._drag.moved and self._refuse_locked_group(self._selection, "動かす・伸び縮み"):
            self._drag = DragState()
            return
        self._drag.moved = True
        frame = self._layout.frame_at(position.x())

        located = (
            self._project.timeline.locate_clip(self._drag.clip_id)
            if self._drag.clip_id is not None
            else None
        )
        if self._drag.kind is DragKind.MOVE_CLIP:
            floor = self._group_floor() if self._drag.group else 0
            start = max(floor, frame - self._drag.grab_offset)
            if located is not None:
                # 頭と終わりのどちらかが近くの位置へ吸い付く
                start += self._snap_shift((start, start + located[1].duration))
            self._drag.preview_start = max(floor, start)
            band = self._layout.band_at(self._project.timeline, position.y())
            if band is not None and not band.track.locked:
                self._drag.preview_track = band.track.id
        elif located is not None:
            _, clip = located
            if self._drag.kind is DragKind.TRIM_HEAD:
                frame += self._snap_shift((frame,))
                self._drag.preview_head_delta = min(frame - clip.timeline_start, clip.duration - 1)
            else:
                frame += self._snap_shift((frame,))
                self._drag.preview_tail_delta = max(frame - clip.timeline_end, -(clip.duration - 1))
        self.update()

    # --- 磁石（吸着） ---

    def set_snap(self, enabled: bool, distance: int = DEFAULT_SNAP_DISTANCE) -> None:
        """近くの位置へ吸い付くか・吸い付く距離（画面の画素） 設定とツールバーのボタンから"""
        self._snap_enabled = enabled
        self._snap_distance = max(1, distance)
        self._snap_cache = None

    @property
    def snap_enabled(self) -> bool:
        return self._snap_enabled

    @property
    def snap_line(self) -> int | None:
        """いま吸い付いた所を示している線のフレーム 出していなければ ``None``"""
        return self._snap_line

    def _snap_shift(self, edges: tuple[int, ...]) -> int:
        """``edges`` を近くの吸い付く先へずらす量 吸い付かなければ 0

        Shift を押している間は吸い付かない（一時的に切る） 動かしている途中で押しても効く
        押す前から押していると囲んで選ぶ・範囲で選ぶ操作になるので、掴んでから押す
        """
        held = self._drag_modifiers | QApplication.keyboardModifiers()
        if not self._snap_enabled or held & SNAP_OFF_MODIFIER:
            return 0
        found = nearest_snap(
            edges, self._snap_targets(), self._snap_distance / self._layout.pixels_per_frame
        )
        self._show_snap(found[1] if found is not None else None)
        return found[0] if found is not None else 0

    def _snap_targets(self) -> list[int]:
        """吸い付く先 動かしている物（選んだ物とリンクした相手）は除く 同じ中身の間は覚える"""
        moving = frozenset(member.id for _, member in self._moving_members()) | frozenset(
            self._selection
        )
        if self._drag.clip_id is not None:
            located = self._project.timeline.locate_clip(self._drag.clip_id)
            if located is not None:
                moving |= {located[1].id}
                moving |= {member.id for _, member in self._linked_partners(located[1])}
        key = (self._project, moving, self._playhead)
        if self._snap_cache is None or self._snap_cache[0] != key:
            targets = snap_targets(self._project, self._playhead, exclude=moving)
            self._snap_cache = (key, targets)
        return self._snap_cache[1]

    def _show_snap(self, frame: int | None) -> None:
        """吸い付いた所に縦の線を出す 離した後も一瞬残して、吸い付いたことを見せる"""
        if frame is None:
            return
        self._snap_line = frame
        self._snap_timer.start()
        self.update()

    def _hide_snap(self) -> None:
        self._snap_line = None
        self.update()

    def _paint_snap(self, painter: QPainter) -> None:
        if self._snap_line is None:
            return
        x = int(self._layout.frame_to_x(self._snap_line))
        if x < Metrics.TRACK_HEADER_WIDTH or x > self.width():
            return
        painter.save()
        painter.setPen(QPen(Colors.SNAP_LINE, 1))
        painter.drawLine(x, Metrics.RULER_HEIGHT, x, self.height())
        painter.restore()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        del event
        self._edge_scroll.stop()
        if self._work_area.dragging:
            self._work_area.release(self._project.timeline.work_area)
            self.update()
            return
        if self._track_mover.pressed:
            self._track_mover.release(self._project.timeline)
            self.update()
            return
        drag, self._drag = self._drag, DragState()
        if drag.kind is DragKind.RESIZE_TRACK:
            self._finish_resize(drag)
            return
        if drag.kind in (DragKind.VALUE_LINE, DragKind.VALUE_KEY):
            self._value_lines.release()
            return
        if drag.kind is DragKind.MARQUEE:
            # 選択は動かしている間に決まっている 枠を消すだけ
            if len(self._selection) > 1:
                self.status_message.emit(f"{len(self._selection)} 本を選択")
            self.update()
            return
        if drag.kind in (DragKind.NONE, DragKind.PLAYHEAD) or not drag.moved:
            self.update()
            return
        if drag.clip_id is None:
            return

        command = self._command_for(drag)
        if command is not None:
            self._request([command], command.label)
        self.update()

    def _command_for(self, drag: DragState) -> Command | None:
        """ドラッグの結果を 1 つのコマンドにまとめる"""
        if drag.clip_id is None:
            return None
        located = self._project.timeline.locate_clip(drag.clip_id)
        if located is None:
            return None
        _, clip = located

        if drag.kind is DragKind.MOVE_CLIP and drag.group:
            delta = drag.preview_start - clip.timeline_start
            movable = self._movable_selection()
            # 掴んだクリップ自身が動かせない（ロックしている）なら何もしない 動かすと、
            # 掴んだものはその場に残り、選んだほかのクリップだけが動く
            if drag.clip_id not in movable:
                return None
            tracks, _ = self._group_landings(drag)
            if not delta and not tracks:
                return None
            return MoveClips(movable, delta, track_delta=tracks)

        if drag.kind is DragKind.MOVE_CLIP:
            unchanged = (
                drag.preview_start == clip.timeline_start
                and drag.preview_track == drag.origin_track
            )
            if unchanged:
                return None
            return MoveClip(
                drag.clip_id,
                drag.preview_start,
                drag.preview_track if drag.preview_track != drag.origin_track else None,
            )

        if drag.kind is DragKind.TRIM_HEAD and drag.preview_head_delta:
            if drag.group:
                return TrimClips(self._trimmable_selection(), head_delta=drag.preview_head_delta)
            return TrimClip(drag.clip_id, head_delta=drag.preview_head_delta)
        if drag.kind is DragKind.TRIM_TAIL and drag.preview_tail_delta:
            if drag.group:
                return TrimClips(self._trimmable_selection(), tail_delta=drag.preview_tail_delta)
            return TrimClip(drag.clip_id, tail_delta=drag.preview_tail_delta)
        return None

    def _track_delta(self, drag: DragState) -> int:
        """掴んだクリップが何本ぶんトラックを跨いだか 同じ種類の並びで数える"""
        if drag.preview_track is None or drag.preview_track == drag.origin_track:
            return 0
        timeline = self._project.timeline
        origin = timeline.find_track(drag.origin_track) if drag.origin_track else None
        target = timeline.find_track(drag.preview_track)
        if origin is None or target is None or origin.kind is not target.kind:
            return 0
        same = [t.id for t in timeline.tracks if t.kind is origin.kind]
        return same.index(target.id) - same.index(origin.id)

    def _group_landings(self, drag: DragState) -> tuple[int, list[tuple[TrackId, Clip]]]:
        """まとめて動かすときに跨ぐ本数と、動く全員の行き先のトラック

        行き先は :class:`MoveClips` と同じ :func:`shifted_track` で求める 1 本でも並びの外へ
        出るか、ロックしたトラックへ入るなら、:class:`MoveClips` は全員を断る そのときは
        トラックを跨がずに時間だけ動かす（跨ぐ本数を 0 にする） 枠も離したときの命令も
        この答えを使うので、枠を出した所と実際に入る所が食い違わない 1 本だけ元の所に
        枠を残すと、ほかの枠は動いて見えるのに、離すと全員が断られる
        """
        members = self._moving_members()
        shift = self._track_delta(drag)
        if shift:
            project = self._project
            try:
                landings = [
                    (shifted_track(project, track_id, shift), member)
                    for track_id, member in members
                ]
            except ValueError:
                landings = []
            timeline = project.timeline
            blocked = not landings or any(
                (track := timeline.find_track(track_id)) is None or track.locked
                for track_id, _ in landings
            )
            if not blocked:
                return shift, landings
        return 0, members

    def _preview_height(self, y: int) -> None:
        """ドラッグ中の高さを描画にだけ当てる 履歴には載せない"""
        base, track_id = self._resize_base, self._drag.resize_track
        if base is None or track_id is None:
            return
        height = self._drag.origin_height + (y - self._drag.grab_y)
        self._drag.moved = True
        self._project = SetTrackHeights(((track_id, height),)).apply(base)
        self.update()

    def _finish_resize(self, drag: DragState) -> None:
        base, self._resize_base = self._resize_base, None
        if base is None or drag.resize_track is None:
            return
        preview, self._project = self._project, base
        track = preview.timeline.find_track(drag.resize_track)
        if drag.moved and track is not None and track.height != drag.origin_height:
            command = SetTrackHeights(((drag.resize_track, track.height),))
            self._request([command], command.label)
        self.update()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._remove_value_key(event.pos()):
            return
        self.build_context_menu(event.pos()).exec(event.globalPos())

    def _remove_value_key(self, position: QPoint) -> bool:
        """値の線の点の上の右クリックは、メニューを出さずにその点を消す"""
        clip = self._value_line_clip(position)
        return clip is not None and self._value_lines.remove_at(
            self._project, self._layout, self.width(), clip, position
        )

    def build_context_menu(self, position: QPoint) -> QMenu:
        """右クリックメニュー 表示と中身を分けてあるのはテストのため

        クリップの上ならそのクリップを選び直してから出す 選んでいた別のクリップが
        対象になると、見ていないものを消すことになる
        """
        menu = QMenu(self)
        hit = self._clip_at(position)
        count = ""
        if hit is not None:
            # 選んでいる何本かの上なら、その何本かが対象 選んでいない所なら、
            # そのクリップだけを選び直す
            if hit[1].id not in self._selection:
                # 左クリックと同じくグループは仲間ごと 1 本だけだと、削除や切り取りで束が裂ける
                self._select_with_group(hit[1].id)
            if len(self._selection) > 1:
                count = f"（{len(self._selection)} 本）"
            _action(menu, "再生ヘッドで分割", self.split_at_playhead)
            menu.addSeparator()
            _action(menu, f"コピー{count}", self.copy_selected)
            _action(menu, f"切り取り{count}", self.cut_selected)
        else:
            self._add_empty_items(menu, position)
        paste = _action(menu, "貼り付け（再生ヘッドの位置）", self.paste_at_playhead)
        paste.setEnabled(self._clipboard is not None)
        if hit is not None:
            menu.addSeparator()
            _action(menu, f"削除{count}", self.delete_selected)
            _action(menu, f"削除して詰める{count}", lambda: self.delete_selected(ripple=True))
            menu.addSeparator()
            group = _action(menu, "グループ化", self.group_selected)
            group.setEnabled(len(self._selection) > 1)
            ungroup = _action(menu, "グループ解除", self.ungroup_selected)
            ungroup.setEnabled(self._selection_has_group())
            if hit[1].scene_id is not None:
                scene_id = hit[1].scene_id
                _action(
                    menu,
                    "シーンを開く",
                    functools.partial(self.scene_open_requested.emit, str(scene_id)),
                )
            menu.addSeparator()
            located = self._project.timeline.locate_clip(hit[1].id)
            if located is not None:
                self._add_menus.add_clip_items(menu, located[0], hit[1])
                self._value_lines.add_menu_items(
                    menu, self._project, located[0], hit[1], self._selection
                )

        band = (
            self._layout.band_at(self._project.timeline, position.y())
            if position.y() >= Metrics.RULER_HEIGHT
            else None
        )
        if band is not None:
            menu.addSeparator()
            track = band.track
            name = track.name or "トラック"
            for attribute, label in _TRACK_TOGGLES:
                toggle = _action(
                    menu, f"{name} を{label}", functools.partial(self._flip, track.id, attribute)
                )
                toggle.setCheckable(True)
                toggle.setChecked(bool(getattr(track, attribute)))
            _action(menu, f"{name} の高さを戻す", functools.partial(self._reset_height, track.id))
            _action(menu, f"{name} の名前を変更…", functools.partial(self.begin_rename, track.id))
        self._work_area.add_menu_actions(
            menu, self._layout, position, self._project.timeline.work_area
        )
        # トラックを足す・消すは最後に置く 目盛りの上はトラックの欄ではないので出さない
        if position.y() >= Metrics.RULER_HEIGHT:
            self._add_menus.add_track_items(menu, band.track if band is not None else None)
        return menu

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        position = event.position().toPoint()
        if self._rename_at(position):
            return
        hit = self._clip_at(position)
        if hit is not None and hit[1].scene_id is not None:
            self.scene_open_requested.emit(str(hit[1].scene_id))
            return
        super().mouseDoubleClickEvent(event)

    # --- トラックの名前 ---

    def _rename_at(self, position: QPoint) -> bool:
        """ヘッダの名前の所のダブルクリックなら、名前の入力欄を出して真を返す

        M・S・L のボタンの上は除く 続けて押しただけで名前の入力になると、ミュートを
        2 回切り替えたつもりの操作が名前の変更に化ける
        """
        if position.x() >= Metrics.TRACK_HEADER_WIDTH or position.y() < Metrics.RULER_HEIGHT:
            return False
        if self._track_button_at(position) is not None:
            return False
        if self._resize_band_at(position) is not None:
            return False
        band = self._layout.band_at(self._project.timeline, position.y())
        if band is None:
            return False
        return self.begin_rename(band.track.id) is not None

    def begin_rename(self, track_id: TrackId) -> TrackNameEditor | None:
        """ヘッダの名前の所に入力欄を重ねて出す 見えていないトラックなら ``None``

        決めた名前は :meth:`rename_track` へ渡る 入力欄はビューの子にして、ヘッダの
        名前と同じ所へ置く 別の窓で尋ねると、どのトラックの名前なのかが隠れる
        """
        band = next(
            (b for b in self._layout.bands(self._project.timeline) if b.track.id == track_id),
            None,
        )
        if band is None or band.bottom <= Metrics.RULER_HEIGHT or band.top >= self.height():
            return None
        if self._name_editor is not None and not self._name_editor.done:
            self._name_editor.commit()
        editor = TrackNameEditor(self, track_id, band.track.name)
        editor.setGeometry(track_name_rect(band).adjusted(-4, -3, 4, 3))
        editor.committed.connect(lambda track, name: self.rename_track(TrackId(track), name))
        editor.show()
        editor.setFocus()
        editor.selectAll()
        self._name_editor = editor
        return editor

    def rename_track(self, track_id: TrackId, name: str) -> bool:
        """トラックの名前を変える（取り消せる） 変わらなければ何も出さずに偽"""
        if self._project.timeline.find_track(track_id) is None:
            return False
        command = RenameTrack(track_id, name)
        if command.apply(self._project) is self._project:
            return False
        self._request([command], command.label)
        return True

    def group_selected(self) -> bool:
        """選んでいるクリップを束ねる 2 本以上要る"""
        if len(self._selection) < 2:
            self.status_message.emit("グループ化するには 2 本以上選んでください")
            return False
        command = GroupClips(self._selection)
        self._request([command], command.label)
        return True

    def ungroup_selected(self) -> bool:
        """グループを解く 置いた動画の映像と音のつながり（リンク）も一緒に外す

        利用者から見ると、置いた映像と音は一緒に動く「グループ」で、解けば別々に動かせて
        ほしい（利用者の要望） リンクだけ残すと、グループ解除を押しても映像と音が離れない
        """
        if not self._selection_has_group():
            self.status_message.emit("グループかリンクに入っているクリップを選んでください")
            return False
        command = UngroupClips(self._selection)
        self._request([command], command.label)
        return True

    def _selection_has_group(self) -> bool:
        timeline = self._project.timeline
        for clip_id in self._selection:
            located = timeline.locate_clip(clip_id)
            if located is not None and (
                located[1].group_id is not None or located[1].link_group is not None
            ):
                return True
        return False

    def _reset_height(self, track_id: TrackId) -> None:
        command = SetTrackHeights(((track_id, DEFAULT_TRACK_HEIGHT),))
        self._request([command], command.label)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt の命名規約
        key = event.key()
        rippled = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if key == Qt.Key.Key_S:
            self.split_at_playhead()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected(ripple=rippled)
        elif key == Qt.Key.Key_Escape:
            self.select(None)
        elif key == Qt.Key.Key_Left:
            self.set_playhead(self._playhead - 1)
            self.playhead_moved.emit(self._playhead)
        elif key == Qt.Key.Key_Right:
            self.set_playhead(self._playhead + 1)
            self.playhead_moved.emit(self._playhead)
        elif key == Qt.Key.Key_Home:
            self.set_playhead(0)
            self.playhead_moved.emit(self._playhead)
        elif key == Qt.Key.Key_End:
            self.set_playhead(self._project.duration)
            self.playhead_moved.emit(self._playhead)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # --- 操作 ---

    def toggle_selected_track(self, attribute: str) -> bool:
        """選んでいるクリップのトラックの ``muted`` ``solo`` ``locked`` を切り替える

        ヘッダのボタンは描いた矩形でフォーカスが来ないので、キーボードからはこちらを
        使う 選択が無ければ何もしない（どのトラックか分からないまま切り替えない）
        """
        primary = self.selected_clip
        if primary is None:
            return False
        located = self._project.timeline.locate_clip(primary)
        if located is None:
            return False
        track = located[0]
        command = SetTrackState(track.id, **{attribute: not getattr(track, attribute)})
        self._request([command], command.label)
        return True

    def split_at_playhead(self) -> None:
        """再生ヘッドの位置でクリップを分割する S キー・編集メニュー・右クリックの共通の入口

        選んでいるクリップがあれば、選んだものだけを切る（リンクした映像・音声の相手は
        :class:`SplitClip` が一緒に割る） 何も選んでいなければ、再生ヘッドの下にある
        全部を切る どちらもロックしたトラックは切らない
        選択を見ずに全部を切っていたので、1 本だけ切るつもりで、ほかのレイヤーの
        字幕や BGM まで切れていた（Issue #27）

        リンクされた映像・音声は 1 つのコマンドで一緒に割れるので、グループごとに
        1 回だけ発行する 両方に出すと、2 回目は「すでに割れている」失敗になり、
        意味のないエラーがステータスバーに出る
        """
        frame = self._playhead
        chosen = set(self._selection)
        targets: list[Clip] = []
        seen: set[GroupId] = set()
        #: 再生ヘッドの下にあったのに、ロックで切らなかったものがあるか 通知の理由を分ける
        locked_hit = False
        for track in self._project.timeline.tracks:
            for clip in track.clips:
                if chosen and clip.id not in chosen:
                    continue
                if not (clip.timeline_start < frame < clip.timeline_end):
                    continue
                if track.locked:
                    locked_hit = True
                    continue
                if clip.link_group is not None:
                    if clip.link_group in seen:
                        continue
                    seen.add(clip.link_group)
                targets.append(clip)

        blocked = self._locked_groups(clip.id for clip in targets)
        if blocked:
            if chosen:
                self._refuse_locked_group((clip.id for clip in targets), "分割")
                return
            # 何も選ばずに切るときは、ほかのクリップは切る 断ったグループだけを知らせる
            targets = [clip for clip in targets if clip.group_id not in blocked]
            self._refuse_locked_group(
                (
                    c.id
                    for t in self._project.timeline.tracks
                    for c in t.clips
                    if c.group_id in blocked
                ),
                "そのグループは分割",
            )
            if not targets:
                return
        if chosen and not targets:
            # 黙って何もしないと、キーが効いていないのか選び方が違うのか分からない
            # ロックが理由なのに「位置にない」と出すと、再生ヘッドを動かし直すだけで終わる
            self.status_message.emit(
                "選んだクリップのトラックはロックされています"
                if locked_hit
                else "選んだクリップは再生ヘッドの位置にありません"
            )
            return
        # グループの 2 本以上を割るときは、後ろの片割れどうしを新しいグループにまとめる
        # 全部の命令へ同じ組を渡す（取り消しは 1 回で全部戻る）
        split_ids = {clip.id for clip in targets}
        for clip in targets:
            if clip.link_group is not None:
                split_ids.update(
                    member.id
                    for _, member in self._project.timeline.linked_clips(clip.link_group)
                    if member.contains(frame)
                )
        counts: dict[GroupId, int] = {}
        for track in self._project.timeline.tracks:
            for clip in track.clips:
                if clip.id in split_ids and clip.group_id is not None:
                    counts[clip.group_id] = counts.get(clip.group_id, 0) + 1
        new_groups = tuple(
            (group, new_group_id()) for group, count in sorted(counts.items()) if count >= 2
        )
        self._request(
            [SplitClip(clip.id, frame, new_groups=new_groups) for clip in targets],
            "再生ヘッドで分割",
        )

    def delete_selected(self, *, ripple: bool = False) -> None:
        if not self._selection:
            return
        if self._refuse_locked_group(self._selection, "削除"):
            return
        command: Command = (
            RemoveClips(self._selection, ripple=ripple)
            if len(self._selection) > 1
            else RemoveClip(self._selection[0], ripple=ripple)
        )
        self._request([command], command.label)

    @property
    def has_clipboard(self) -> bool:
        return self._clipboard is not None

    def copy_selected(self) -> bool:
        """選んでいるクリップをコピーする リンクした相手も一緒に入る"""
        if not self._selection:
            return False
        content = copy_clips(self._project, self._selection)
        if not content.clips:
            return False
        self._clipboard = content
        self.status_message.emit(f"{len(content.clips)} 本をコピーした")
        return True

    def cut_selected(self) -> bool:
        """コピーしてから消す 隙間は詰めない（詰めたければ「削除して詰める」）"""
        if self._refuse_locked_group(self._selection, "切り取り"):
            return False
        if not self.copy_selected() or self._clipboard is None:
            return False
        self._request(cut_commands(self._project, self._clipboard), "切り取り")
        return True

    def paste_at_playhead(self) -> bool:
        """再生ヘッドの位置へ貼り付けて、貼ったクリップを選ぶ"""
        if self._clipboard is None:
            self.status_message.emit("コピーしたクリップがありません")
            return False
        try:
            commands = paste_commands(self._project, self._clipboard, self._playhead)
        except ValueError as exc:
            self.status_message.emit(str(exc))
            return False
        self._request(commands, "貼り付け")
        # 実行は受け取った側で済んでいる 貼ったものを全部選んでおくと、そのまま
        # まとめて動かせる 1 本だけ選ぶと、残りを探して選び直すことになる
        pasted = [c.clip.id for c in commands if isinstance(c, AddClip)]
        landed = [c for c in pasted if self._project.timeline.locate_clip(c) is not None]
        if landed:
            self.set_selection(landed)
            self._anchor = landed[-1]
        return True

    def adjust_track_heights(self, delta: int) -> None:
        """全トラックの高さを ``delta`` 画素ずつ変える

        続けて変えたぶん（:data:`HEIGHT_MERGE_SECONDS` 以内）は取り消しの 1 段に
        まとめてもらう ホイールを 10 段回して、戻すのに 10 回取り消すのは重い
        """
        tracks = self._project.timeline.tracks
        wanted = [min(max(t.height + delta, MIN_TRACK_HEIGHT), MAX_TRACK_HEIGHT) for t in tracks]
        if wanted == [t.height for t in tracks]:
            # 上限や下限に張り付いて変わらないときは、続けた操作をそこで区切る
            # 区切らないと、張り付いたまま回したあとすぐ反対へ回したぶんが、
            # 張り付く前の段へまとまり、1 回の取り消しでそこまで戻る
            self._last_height_change = -HEIGHT_MERGE_SECONDS
            return
        now = time.monotonic()
        continued = now - self._last_height_change < HEIGHT_MERGE_SECONDS
        self._last_height_change = now
        command = SetTrackHeights(tuple((t.id, t.height + delta) for t in tracks))
        signal = self.commands_continued if continued else self.commands_requested
        signal.emit([command], command.label)

    def reset_track_heights(self) -> None:
        tracks = self._project.timeline.tracks
        if tracks:
            command = SetTrackHeights(tuple((t.id, DEFAULT_TRACK_HEIGHT) for t in tracks))
            self._request([command], command.label)

    def _flip(self, track_id: TrackId, attribute: str) -> None:
        track = self._project.timeline.find_track(track_id)
        if track is not None:
            command = SetTrackState(track_id, **{attribute: not getattr(track, attribute)})
            self._request([command], command.label)

    def clear_work_area(self) -> bool:
        """書き出し範囲を解除する 編集メニューから 範囲が無ければ知らせて偽を返す"""
        if self._work_area.clear(self._project.timeline.work_area):
            return True
        self.status_message.emit("書き出し範囲は指定されていません")
        return False

    def _request(self, commands: list[Command], label: str) -> None:
        if commands:
            self.commands_requested.emit(commands, label)

    # --- 補助 ---

    def _toggle(self, clip_id: ClipId) -> None:
        """Ctrl+クリック 選んでいれば外し、いなければ足す

        起点は :meth:`set_selection` が残った最後の 1 本へ移す 外したクリップを
        起点にすると、次の Shift+クリックが選んでいないクリップから範囲を取る
        """
        # 足す順はタイムラインの並びを保つ 集合から並べると選んだ順が毎回変わる
        members = self._group_of(clip_id)
        if clip_id in self._selection:
            removed = set(members)
            self.set_selection(
                (c for c in self._selection if c not in removed), pulled=self._pulled
            )
        else:
            joined = [c for c in members if c != clip_id]
            self.set_selection(
                (*self._selection, *joined, clip_id),
                pulled=(self._pulled | set(joined)) - {clip_id},
            )

    def _select_range(self, anchor: ClipId, target: ClipId) -> None:
        """Shift+クリック 起点と今のクリップを両隅にした範囲をまとめて選ぶ

        縦は 2 本のトラックの間、横は 2 本の端から端まで 同じトラックなら、その間に
        並ぶクリップが全部入る 起点は動かさないので、Shift を押したまま別の
        クリップを押せば範囲を広げ直せる

        「間」は画面に見えている並びで決める 映像トラックは下から積むので、
        モデルの並び（``timeline.tracks``）とは順が違う
        """
        timeline = self._project.timeline
        first, last = timeline.locate_clip(anchor), timeline.locate_clip(target)
        if first is None or last is None:
            self.select(target)
            return
        shown = [band.track for band in self._layout.bands(timeline)]
        order = [track.id for track in shown]
        top, bottom = sorted((order.index(first[0].id), order.index(last[0].id)))
        start = min(first[1].timeline_start, last[1].timeline_start)
        end = max(first[1].timeline_end, last[1].timeline_end)
        # グループは仲間ごと入れる 一部だけ選ぶと、そのまま動かしたときに束が裂ける
        inside = [
            clip.id
            for track in shown[top : bottom + 1]
            for clip in track.clips
            if clip.overlaps(start, end)
        ]
        chosen = [member for clip_id in inside for member in self._group_of(clip_id)]
        members = (*chosen, *self._group_of(target), target)
        picked = {*inside, target}
        self.set_selection(members, pulled=(c for c in members if c not in picked))
        self._anchor = anchor

    def _update_marquee(self, position: QPoint) -> None:
        origin = self._drag.marquee_from
        if origin is None:
            return
        if not self._drag.moved:
            if (position - origin).manhattanLength() < MARQUEE_THRESHOLD:
                return
            self._drag.moved = True
        self._drag.marquee_to = position
        rect = QRect(origin, position).normalized()
        inside = self._clips_in_rect(rect)
        # 枠に掛かった物は自分で選んだ物 掛かっていない仲間だけを引き込んだ物にする
        # 掛かった物を後ろに並べる 主（最後の 1 本）が引き込んだ仲間になると、設定パネルが
        # 枠に掛けていないクリップを出す
        caught = [
            *(m for clip_id in inside for m in self._group_of(clip_id) if m not in inside),
            *inside,
        ]
        self.set_selection(
            (*self._drag.marquee_base, *caught),
            pulled=(self._pulled | set(caught)) - set(inside),
        )
        self.update()

    def _locked_groups(self, clip_ids: Iterable[ClipId]) -> set[GroupId]:
        """渡したクリップのグループのうち、ロックしたトラックのクリップを含む物

        グループはまとめて割る・消す・動かす組 一部がロックで動かないまま残りだけに
        当てると、束ねた物の頭や長さが食い違い、束ねた意味が崩れる（利用者の要望）
        """
        timeline = self._project.timeline
        groups: set[GroupId] = set()
        for clip_id in clip_ids:
            located = timeline.locate_clip(clip_id)
            if located is not None and located[1].group_id is not None:
                groups.add(located[1].group_id)
        return {
            group
            for group in groups
            if any(track.locked for track, _ in timeline.grouped_clips(group))
        }

    def _refuse_locked_group(self, clip_ids: Iterable[ClipId], action: str) -> bool:
        """ロックしたクリップを含むグループがあれば、理由を出して真を返す（何もしない）"""
        if not self._locked_groups(clip_ids):
            return False
        self.status_message.emit(
            f"グループの中にロックしたレイヤーのクリップがあるので{action}できません"
            " ロックを外すか、グループを解除してください"
        )
        return True

    def _select_with_group(self, clip_id: ClipId) -> None:
        """押した 1 本をグループの仲間ごと選ぶ 押した 1 本が主 仲間は引き込んだ物として覚える"""
        members = self._group_of(clip_id)
        self.set_selection((*members, clip_id), pulled=(c for c in members if c != clip_id))

    def _group_of(self, clip_id: ClipId) -> tuple[ClipId, ...]:
        """グループの仲間（自分を含む） グループに入っていなければ自分だけ"""
        timeline = self._project.timeline
        located = timeline.locate_clip(clip_id)
        if located is None or located[1].group_id is None:
            return (clip_id,)
        return tuple(clip.id for _, clip in timeline.grouped_clips(located[1].group_id))

    def _clips_in_rect(self, rect: QRect) -> list[ClipId]:
        """枠に少しでも掛かったクリップ 全部を収めなくても選べる方が囲みやすい

        ここの上下は画面の画素（下が正、Qt の座標） 枠もトラックの帯も同じ画面の
        座標で持っているので、そのまま比べてよい 映像の中の位置（上が正）とは
        別のもので、混ぜて計算しない
        """
        first = self._layout.frame_at(max(rect.left(), Metrics.TRACK_HEADER_WIDTH))
        last = self._layout.frame_at(rect.right())
        return [
            clip.id
            for band in self._layout.bands(self._project.timeline)
            if band.bottom > rect.top() and band.top < rect.bottom()
            for clip in clips_in_range(band.track, first, last)
        ]

    def _scrub(self, position: QPoint) -> None:
        # スクラブ中は追従を切る 追従したままだと、掴んだ位置が画面中央へ
        # 逃げ続けて操作にならない
        self._follow_playhead = False
        self.set_playhead(self._layout.frame_at(position.x()), follow=False)
        self._follow_playhead = True
        self.playhead_moved.emit(self._playhead)

    def _track_button_at(self, position: QPoint) -> tuple[TrackId, str, str] | None:
        """ヘッダの切り替えボタンの上なら ``(トラック, 説明, 属性名)``"""
        if position.x() >= Metrics.TRACK_HEADER_WIDTH or position.y() < Metrics.RULER_HEIGHT:
            return None
        band = self._layout.band_at(self._project.timeline, position.y())
        if band is None:
            return None
        for attribute, tip, rect in track_button_rects(band):
            if rect.contains(position):
                return band.track.id, tip, attribute
        return None

    def _toggle_track_button(self, position: QPoint) -> bool:
        """ボタンの上なら切り替えのコマンドを出して真を返す

        再生ヘッドは動かさない ミュートを押すたびに見ていた場所が飛ぶと、
        聞き比べのたびに位置を戻すことになる
        """
        hit = self._track_button_at(position)
        if hit is None:
            return False
        track_id, _, attribute = hit
        track = self._project.timeline.find_track(track_id)
        if track is None:
            return False
        command = SetTrackState(track_id, **{attribute: not getattr(track, attribute)})
        self._request([command], command.label)
        return True

    def _clip_at(self, position: QPoint) -> tuple[TrackId, Clip] | None:
        """マウスの下のクリップ マウスが動くたびに呼ばれる

        トラックを縦位置で決めてから、そのトラックの中を二分探索で探す 見えている
        クリップを全部舐めると、全体表示の 1 万本でマウスを動かすだけで重くなる

        探す範囲は前後 1 画素ぶんのフレーム 全体表示では 1 画素が数十フレームに
        あたるので、フレームの前後 1 つだけを見ると画素の中のクリップを取りこぼす
        1 画素に満たないクリップは矩形が丸めで隣の画素へずれるので、矩形で当たらな
        ければ、その画素の真ん中のフレームを含むクリップを選ぶ
        """
        if position.x() < Metrics.TRACK_HEADER_WIDTH:
            return None
        band = self._layout.band_at(self._project.timeline, position.y())
        if band is None or not (band.top < position.y() < band.bottom - 2):
            return None
        first = self._layout.frame_at(position.x() - 1)
        last = self._layout.frame_at(position.x() + 1)
        candidates = clips_in_range(band.track, first, last)
        for clip in candidates:
            rect = clip_rect_for(clip, band, self._layout, self.width())
            if rect is not None and rect.contains(position):
                return band.track.id, clip
        middle = int(self._layout.x_to_frame(position.x() + 0.5))
        for clip in candidates:
            if clip.contains(middle):
                return band.track.id, clip
        return None

    def _press_keyframe(self, position: QPoint) -> bool:
        """キーフレームのひし形の上なら、再生ヘッドをそこへ動かして真を返す

        そのクリップも選ぶ 設定パネルに出るのは選んだクリップなので、選ばないと
        動かした先で、どのキーフレームの値なのかを見られない ひし形を描くのは
        名前が入る幅のクリップだけなので（:meth:`paintEvent`）、押せるのもそれだけにする
        """
        hit = self._clip_at(position)
        if hit is None:
            return False
        _, clip = hit
        if clip.duration * self._layout.pixels_per_frame < DETAIL_MIN_WIDTH:
            return False
        band = self._layout.band_at(self._project.timeline, position.y())
        rect = clip_rect_for(clip, band, self._layout, self.width()) if band else None
        frame = keyframe_at(clip, self._layout, rect, position) if rect is not None else None
        if frame is None:
            return False
        if clip.id not in self._selection:
            self._select_with_group(clip.id)
            self._anchor = clip.id
        self.set_playhead(frame, follow=False)
        self.playhead_moved.emit(self._playhead)
        return True

    def _value_line_clip(self, position: QPoint) -> Clip | None:
        """値の線を掴める所のクリップ トリムの端の近くは ``None``

        端を先に見る 線は端まで引いてあるので、線を先に取ると短いクリップや
        値が名前の帯の近くにあるクリップの端を掴めなくなる
        """
        hit = self._clip_at(position)
        if hit is None or self._edge_at(position, hit[1]) is not DragKind.MOVE_CLIP:
            return None
        return hit[1]

    def _press_value_line(self, position: QPoint, modifiers: Qt.KeyboardModifier) -> bool:
        """値の線か点を押したなら、その操作を始めて真を返す

        掴んだ 1 本だけに当てる 選んだほかのクリップは線の形も点の数も違い、同じだけ動かすと
        見ていない線まで変わる（設定パネルの一括の変更は、値の欄を見て数を決めるときに使う）
        """
        clip = self._value_line_clip(position)
        if clip is None:
            return False
        grab = self._value_lines.press(
            self._project, self._layout, self.width(), clip, position, modifiers
        )
        if grab is None:
            return False
        if clip.id not in self._selection:
            self._select_with_group(clip.id)
            self._anchor = clip.id
        if grab is not ValueGrab.DONE:
            kind = DragKind.VALUE_KEY if grab is ValueGrab.KEY else DragKind.VALUE_LINE
            self._drag = DragState(kind=kind, clip_id=clip.id)
        return True

    def _edge_at(self, position: QPoint, clip: Clip) -> DragKind:
        """クリップの端を掴んでいるならトリム、そうでなければ移動"""
        left = self._layout.frame_to_x(clip.timeline_start)
        right = self._layout.frame_to_x(clip.timeline_end)
        if abs(position.x() - left) <= Metrics.TRIM_HANDLE_WIDTH:
            return DragKind.TRIM_HEAD
        if abs(position.x() - right) <= Metrics.TRIM_HANDLE_WIDTH:
            return DragKind.TRIM_TAIL
        return DragKind.MOVE_CLIP

    def _resize_band_at(self, position: QPoint) -> TrackBand | None:
        """ヘッダの中で、トラックの下の境目の上にいればそのトラック

        ヘッダの中に限る タイムラインの側まで広げると、クリップの下端を掴んだ
        つもりが高さの変更になる
        """
        if position.x() >= Metrics.TRACK_HEADER_WIDTH or position.y() < Metrics.RULER_HEIGHT:
            return None
        for band in self._layout.bands(self._project.timeline):
            if abs(position.y() - band.bottom) <= RESIZE_GRAB:
                return band
        return None

    def _update_cursor(
        self,
        position: QPoint,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        if self._resize_band_at(position) is not None:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            return
        if position.x() < Metrics.TRACK_HEADER_WIDTH and self._track_button_at(position) is None:
            # 名前の所は掴んで並べ替えられる 形を変えないと、掴めることに気付けない
            band = self._layout.band_at(self._project.timeline, position.y())
            grabbable = band is not None and position.y() >= Metrics.RULER_HEIGHT
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if grabbable and band is not None and not band.track.locked
                else Qt.CursorShape.ArrowCursor
            )
            return
        hit = self._clip_at(position)
        if hit is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        edge = self._edge_at(position, hit[1])
        grab = (
            self._value_lines.grab_at(self._project, self._layout, self.width(), hit[1], position)
            if edge is DragKind.MOVE_CLIP and not modifiers & Qt.KeyboardModifier.ShiftModifier
            else None
        )
        if grab is not None:
            # 線は上下にしか動かないので縦の矢印 点は時刻も動くので四方の矢印
            vertical = grab is ValueGrab.LINE
            self.setCursor(
                Qt.CursorShape.SizeVerCursor if vertical else Qt.CursorShape.SizeAllCursor
            )
            return
        self.setCursor(
            Qt.CursorShape.SizeHorCursor
            if edge in (DragKind.TRIM_HEAD, DragKind.TRIM_TAIL)
            else Qt.CursorShape.OpenHandCursor
        )

    # --- 足す（右クリックの〔追加〕と「＋ トラック追加」 中身は add_menu.py） ---

    @property
    def open_scene(self) -> SceneId | None:
        return self._open_scene

    def set_open_scene(self, scene_id: SceneId | None) -> None:
        """開いているシーンを知らせる 〔追加〕→〔シーン〕から自分自身を外すのに使う"""
        self._open_scene = scene_id

    def request(self, commands: list[Command], label: str) -> None:
        """コマンドを外へ出す（:attr:`commands_requested`） 空なら何もしない"""
        self._request(commands, label)

    def place(self, commands: list[Command], label: str) -> tuple[ClipId, ...]:
        """クリップを置くコマンドを出し、置けたクリップを選ぶ

        置いたものを選ばないと設定パネルが開かず、足したのに何も起きていないように見える
        断られたら選ばない（受け取った側が戻しているので、プロジェクトに見つからない）
        """
        self._request(commands, label)
        added = [c.clip.id for c in commands if isinstance(c, AddClip)]
        landed = tuple(c for c in added if self._project.timeline.locate_clip(c) is not None)
        if landed:
            self.set_selection(landed)
        return landed

    def track_add_button(self) -> QRect | None:
        """「＋ トラック追加」の矩形 見えていなければ ``None``

        並びは描いているもの（:meth:`_painted_timeline`）を使う ファイルを引いている間は
        新しく作るトラックの仮の行が末尾に並ぶので、本物の並びで置くとその行に重なる
        """
        return track_add_button_rect(self._layout, self._painted_timeline())

    def build_track_add_menu(self) -> QMenu:
        """「＋ トラック追加」を押したときのメニュー 表示と中身を分けてあるのはテストのため"""
        return self._add_menus.track_add_menu(self)

    def _add_empty_items(self, menu: QMenu, position: QPoint) -> None:
        """空いた所の右クリックに〔追加〕を足す ヘッダと目盛りの上では足さない"""
        if position.x() < Metrics.TRACK_HEADER_WIDTH or position.y() < Metrics.RULER_HEIGHT:
            return
        band = self._layout.band_at(self._project.timeline, position.y())
        self._add_menus.add_empty_items(
            menu,
            self._layout.frame_at(position.x()),
            band.track if band is not None else None,
        )
        menu.addSeparator()

    def _paint_add_button(self, painter: QPainter) -> None:
        rect = self.track_add_button()
        if rect is not None and rect.top() < self.height():
            draw_track_add_button(painter, rect, hovered=self._add_button_hovered)

    def _press_add_button(self, position: QPoint) -> bool:
        """「＋ トラック追加」の上なら、足す種類のメニューをボタンの下に出して真を返す

        混合の方式ではメニューを出さずにレイヤーを 1 本足す 足せるのはレイヤーだけで、
        選ぶ物が 1 つしか無いメニューは押す手間が 1 回増えるだけ（利用者の要望）
        """
        rect = self.track_add_button()
        if rect is None or not rect.contains(position):
            return False
        if places_mixed(self._project):
            self._add_menus.add_track(TrackKind.MIXED)
            return True
        self.build_track_add_menu().exec(self.mapToGlobal(rect.bottomLeft()))
        return True

    def _hover_add_button(self, position: QPoint) -> None:
        rect = self.track_add_button()
        hovered = rect is not None and rect.contains(position)
        if hovered:
            kinds = (
                "レイヤーを 1 本足す" if places_mixed(self._project) else "映像・音声・エフェクト"
            )
            self.setToolTip(f"{ADD_TRACK_BUTTON_TEXT}（{kinds}）")
        if hovered != self._add_button_hovered:
            self._add_button_hovered = hovered
            self.update()

    # --- 落とし込み（エクスプローラーのファイル・素材一覧の素材） ---

    def drop_spot_at(self, position: QPointF) -> DropSpot:
        """その位置へ落としたときに置く先（フレームとトラック）

        ドラッグ中は画面に出している並び（新しく作るトラックを並べたもの）で見る
        本物の並びで見ると、仮の行の分だけずれた所のトラックへ落ちる
        """
        return spot_at(
            self._layout, self._painted_timeline(), position, real=self._project.timeline
        )

    def _snapped_spot(self, position: QPointF) -> DropSpot:
        """落とす先 置く頭を近くの位置へ吸い付かせる（磁石） 目安と落とした所で同じ物を使う"""
        spot = self.drop_spot_at(position)
        shift = self._snap_shift((spot.frame,))
        return replace(spot, frame=max(0, spot.frame + shift)) if shift else spot

    @property
    def drop_guide(self) -> DropGuide | None:
        """ドラッグ中に出している目安 引いていなければ ``None``"""
        return self._drop_preview.guide if self._drop_preview is not None else None

    @property
    def drop_preview(self) -> DropPreview | None:
        """ドラッグ中に見せている、落としたときの姿 引いていなければ ``None``"""
        return self._drop_preview

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt の命名規約
        self._track_drag(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802 - Qt の命名規約
        self._track_drag(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802 - Qt の命名規約
        del event
        self._set_drop_guide(None)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt の命名規約
        mime = event.mimeData()
        # 目安を消す前に位置を求める 消してから求めると、見えていた並び（仮の行の入った
        # もの）と違う並びで読み、見ていたのと違うトラックへ落ちる
        spot = self._snapped_spot(event.position())
        self._set_drop_guide(None)
        track = str(spot.track_id) if spot.track_id is not None else ""
        # 素材一覧から来た物を先に見る 一覧の行にファイルの URL が付いていても、
        # 読み込み直さずに、もう入っている素材として置く
        media_ids = media_ids_in(mime)
        paths = local_paths(mime)
        if not media_ids and not paths:
            event.ignore()
            return
        # 写す（Copy）として受け取る 動かす（Move）で受け取ると、引いてきた側
        # （素材一覧やエクスプローラー）が元を消しに行く
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        if media_ids:
            self.media_dropped.emit([str(media_id) for media_id in media_ids], spot.frame, track)
        else:
            self.files_dropped.emit(paths, spot.frame, track)

    def _track_drag(self, event: QDragMoveEvent) -> None:
        self._drag_modifiers = event.modifiers()
        mime = event.mimeData()
        if not accepts(mime):
            event.ignore()
            self._set_drop_guide(None)
            return
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        spot = self._snapped_spot(event.position())
        self._set_drop_guide(DropGuide(spot, tuple(media_ids_in(mime))))

    def _set_drop_guide(self, guide: DropGuide | None) -> None:
        current = self.drop_guide
        if guide == current:
            return
        # 置く先を求めるのは目安が変わったときだけ 描くたびに求めると、素材を何本も
        # 引いているときにマウスを動かすだけで重くなる
        self._drop_preview = (
            preview_drop(self._project, guide, split_audio=self._split_audio)
            if guide is not None
            else None
        )
        self.update()

    def _painted_timeline(self) -> Timeline:
        """画面に出すトラックの並び ドラッグ中は、新しく作るトラックを空のまま並べる"""
        preview = self._drop_preview
        return preview.timeline if preview is not None else self._project.timeline

    def _paint_drop_guide(self, painter: QPainter) -> None:
        if self._drop_preview is not None:
            paint_drop_guide(
                painter,
                self._layout,
                self._project,
                self._drop_preview,
                (self.width(), self.height()),
            )


class TimelineArea(QWidget):
    """タイムラインのビューと、その下と右のスクロールバーを並べる

    横のバーはヘッダの幅だけ右から始める バーが動かすのは時間の軸で、
    ヘッダは動かないので、同じ幅に並べると何を動かすバーなのかが分かりにくい
    """

    def __init__(self, view: TimelineView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = view
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(view, 0, 0, 1, 2)
        grid.addWidget(view.vertical_scroll_bar, 0, 2)
        grid.setColumnMinimumWidth(0, Metrics.TRACK_HEADER_WIDTH)
        grid.addWidget(view.horizontal_scroll_bar, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        # ビューの子として隠してあったので、並べたら出す 縦は要るときだけ出る
        view.horizontal_scroll_bar.show()
        view.set_project(view.project)

    @property
    def view(self) -> TimelineView:
        return self._view


def _action(menu: QMenu, text: str, slot: Callable[[], object]) -> QAction:
    """メニューに項目を足す ``triggered`` の引数（押されたかどうか）は捨てる"""
    action = menu.addAction(text)
    action.triggered.connect(lambda _checked=False: slot())
    return action
