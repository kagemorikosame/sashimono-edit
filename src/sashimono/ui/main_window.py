"""メインウィンドウ 各パネルを組み立て、コマンドの実行を一手に引き受ける

UI のどこから来た操作も、必ず :meth:`MainWindow.execute` を通って
:class:`~sashimono.core.commands.Document` に入る AI エージェントも同じ入口を
使う予定なので、ここが増えないようにしておく
"""

from __future__ import annotations

import contextlib
import functools
import platform
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QPoint, Qt, QTimer, QUrl, Signal, qVersion
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QImage,
    QImageWriter,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from sashimono import __version__
from sashimono.ai.host import ToolError
from sashimono.compat.aviutl import native, plugin
from sashimono.compat.aviutl.exo import ExoFile
from sashimono.core import userdirs
from sashimono.core.commands import (
    DEFAULT_GENERATED_FRAMES,
    AddClip,
    AddMedia,
    AddScene,
    Command,
    Document,
    InScene,
    ParamPath,
    RemoveMedia,
    RemoveScene,
    RenameScene,
    SetBlending,
    SetResolution,
    convert_layers,
    insert_filter,
    insert_generated,
    insert_media,
    insert_scene,
    new_scene,
    place_media,
    switch_layer_mode,
)
from sashimono.core.commands.project_format import retime_frame
from sashimono.core.io import (
    LEGACY_SUFFIXES,
    SUFFIX,
    HeldLock,
    ProjectFileError,
    RecoveryEntry,
    RecoverySession,
    backup_before_save,
    backup_folder,
    discard,
    find_orphans,
    hold_new,
    load_project,
    others_holding,
    project_presence_dir,
    save_project,
)
from sashimono.core.model import (
    Clip,
    ClipId,
    EffectId,
    GeneratedSource,
    LayerMode,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    SceneId,
    TrackId,
    TrackKind,
    heard_stream,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry as effect_registry
from sashimono.effects.sources import SHAPE, TEXT, TRANSITION
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.cache import MediaAnalyzer
from sashimono.engine.cache.proxy import ProxyBuilder, ProxyStore
from sashimono.engine.decode import ProbeError, forget_probe, probe_media
from sashimono.engine.decode.batch import ProbeBatch
from sashimono.engine.gpu import opengl_usable
from sashimono.engine.render import FrameRenderer, RenderQuality
from sashimono.links import MANUAL_URL, REPORT_URL
from sashimono.ui import media_match
from sashimono.ui.chat import ChatPanel
from sashimono.ui.export_dialog import ExportDialog
from sashimono.ui.graph_editor import GraphEditor
from sashimono.ui.inspector import InspectorPanel
from sashimono.ui.media_pool import MediaPoolWidget
from sashimono.ui.playback import PlaybackController
from sashimono.ui.preview import PreviewWidget
from sashimono.ui.progress_display import (
    ProgressIndicator,
    describe_background,
    describe_import,
    finished_message,
    overall_fraction,
    row_notes,
)
from sashimono.ui.scene_bar import SceneBar
from sashimono.ui.subtitle import SubtitlePanel
from sashimono.ui.theme import Colors
from sashimono.ui.timeline import TimelineArea, TimelineView
from sashimono.ui.timeline.drop import DropSpot
from sashimono.ui.timeline.view import HEIGHT_STEP
from sashimono.ui.transport import TransportBar
from sashimono.ui.workspace import (
    LAYOUT_VERSION,
    Preferences,
    PreferenceStore,
    ShortcutStore,
    Workspace,
    apply_dock_tabs,
)

if TYPE_CHECKING:
    from sashimono.compat.mapped import MappedObject

__all__ = ["MainWindow", "about_text"]

#: 解析の完了を画面へ反映する間隔（ミリ秒）
#: 解析はワーカースレッドで終わるので、その通知を待って毎回描き直すのではなく、
#: まとめて一定間隔で描き直す 素材を 100 本入れたときに描画で埋もれないように
ANALYSIS_REFRESH_MS = 250

#: 読み込みで素材を調べている間、進み具合を見に行く間隔（ミリ秒）
#: 解析の間隔（250ms）に合わせると、1 本だけ読み込んだときに置かれるまで
#: 最大 250ms 待たされる 前は 55ms ほどで置かれていたので、遅く感じる
#: 走っている間しか回さないので、短くしても手の空いた時の重さは変わらない
IMPORT_POLL_MS = 30

#: 保存していない変更を退避する間隔（ミリ秒）
#: 落ちたときに失うのは最大でこの長さの作業 短くするほど書き込みが増えるが、
#: 1 回は数百 KB の JSON なので 30 秒なら気にならない
AUTOSAVE_MS = 30_000

_PORTABLE = QKeySequence.SequenceFormat.PortableText

#: 素材一覧のサムネイルに使う時刻（秒） 頭の 1 コマではなく少し先を使う
#: 頭は黒からのフェードや、カメラを向ける前の揺れで、中身の分からない絵が多い
#: 短い素材では最後の 1 枚に止まる（:meth:`Filmstrip.at`）
POOL_THUMBNAIL_SECONDS = Fraction(1)

#: AviUtl のオブジェクトファイル
EXO_FILTER = "AviUtl オブジェクト (*.exo *.exa *.exo2 *.exa2);;すべてのファイル (*)"


def about_text() -> str:
    """バージョン情報の文面 不具合の報告の「版」と「環境」の欄にそのまま写せる形

    置き場は環境変数の形ではなく実際の場所で出す 開発版と配った zip、Windows と
    それ以外で場所が違い、``%APPDATA%`` と書くだけでは本人の機械でどこなのかが
    分からない
    """
    return "\n".join(
        (
            f"Sashimono Edit {__version__}",
            f"Python {platform.python_version()} / Qt {qVersion()} / {platform.platform()}",
            "",
            f"設定・スクリプト・テンプレート: {userdirs.config_root()}",
            f"退避・バックアップ: {userdirs.state_root()}",
            f"キャッシュ: {userdirs.cache_root()}",
            "",
            f"使い方: {MANUAL_URL}",
            f"不具合・要望: {REPORT_URL}",
        )
    )


def _probe_or_none(path: Path) -> MediaItem | None:
    """テンプレートの素材を開く 開けなければ ``None``

    開けない素材が 1 つあるだけで配置全体を止めない 見つからない素材と同じく
    数えて知らせ、ほかのアイテムは置く
    """
    # 読み込む操作なので覚えた結果を使わない 同じ場所へ差し替えた素材を前の中身で置かないため
    forget_probe(path)
    try:
        return probe_media(path)
    except ProbeError:
        return None


@dataclass(frozen=True, slots=True)
class _DropTarget:
    """タイムラインへ落とされた読み込みの置き先 位置と、落としたときに開いていたシーン

    シーンも覚えるのは、調べ終えるまでの間にシーンを切り替えられることがあるため
    その時点で開いているシーンへ置くと、落としていないシーンに素材が入る
    """

    spot: DropSpot
    #: ``None`` ならメインのタイムライン
    scene: SceneId | None
    #: 落としたときのフレームレート ``spot`` のフレームはこれで数えてある 調べ終えるまでに
    #: 最初の動画へ合わせてレートが変わると（:meth:`MainWindow._match_project_to`）、
    #: 数のまま使うと落とした時刻からずれる（30fps の 1 秒が 60fps の 0.5 秒になる）
    rate: FrameRate


@dataclass(frozen=True, slots=True)
class _ExoMedia:
    """``.exo`` が参照している素材を読んだ結果 登録はまだしていない

    登録をクリップの配置と同じ 1 回の :meth:`MainWindow.execute_all` で行うため、
    ここではコマンドを作るだけにする 先に登録すると、配置を断られたときに
    使われない素材と、その解析・控えの重い処理だけが残る
    （テンプレートの棚の :func:`~sashimono.compat.catalog.gather_media` と同じ作り）
    """

    #: 書かれていたパス → 結ぶ素材の id（:func:`map_exo` がそのまま受ける形）
    ids: dict[str, MediaId]
    #: 見つからないか開けなかったパス
    missing: tuple[str, ...]
    #: 素材一覧へ入れるコマンド
    commands: tuple[Command, ...]
    #: 入ったあとに解析と控えを頼む素材
    items: tuple[MediaItem, ...]


class MainWindow(QMainWindow):
    """編集画面"""

    project_changed = Signal(object)

    def __init__(
        self,
        project: Project | None = None,
        *,
        path: Path | None = None,
        confirm_unsaved: bool = True,
    ) -> None:
        """``confirm_unsaved`` を偽にすると、閉じるときに保存を尋ねない テスト用"""
        super().__init__()
        self.setWindowTitle("Sashimono Edit")
        self.resize(1440, 900)

        #: 本人の好みの設定 プロジェクトではなく本人に付く
        #: 最初の空のプロジェクトを作る前に読む 起動した直後のプロジェクトも、新規作成と
        #: 同じくレイヤーの方式の好み（:attr:`Preferences.new_project_layers`）に従わせるため
        self._preferences = PreferenceStore().load()
        self._document = Document(project if project is not None else self._blank_project())
        #: 再生ヘッドのフレームを数えているレート 変わったら数え直す（:meth:`_retime_playhead`）
        self._playhead_rate = self._document.project.rate
        self._path: Path | None = path
        #: 最後に保存した（または開いた）時点のプロジェクト 同じオブジェクトなら
        #: 変更なし モデルは frozen なので、取り消して保存した状態へ戻れば
        #: 「変更なし」に戻る 数を数える方式だとここがずれる
        self._saved: Project | None = self._document.project
        self._autosaved: Project | None = self._document.project
        self._confirm_unsaved = confirm_unsaved
        self._recovery = RecoverySession()
        #: 開いているプロジェクトの錠 同じファイルを別の窓で開いたことに気付くため
        self._project_lock: HeldLock | None = None
        if path is not None and not self._claim(path):
            # 別の窓で開いていて、開くのをやめると選ばれた 中身だけ見せて保存先を
            # 持たないと、錠を持たないまま同じファイルへ保存できてしまう 空で始める
            self._path = None
            self._document.reset(self._blank_project())
            self._saved = self._autosaved = self._document.project
        #: 操作の名前 → （QAction、既定のキー） ショートカットの設定が使う
        self._actions: dict[str, tuple[QAction, str]] = {}
        self._analyzer = MediaAnalyzer(
            sample_rate=self._document.project.settings.sample_rate,
            channels=self._document.project.settings.channels,
        )
        # 描画と書き出しの両方が見るので、窓を組み立てる前に決めておく
        native.set_enabled(self._preferences.native_modules)
        plugin.set_scan_all(self._preferences.all_aviutl_plugins)
        #: プレビュー用の控えを作る係 **書き出しには渡さない**
        #: 渡すと、画面では気付かないまま低解像度の絵が最終出力に入る
        self._proxies = ProxyBuilder(ProxyStore(height=self._preferences.proxy_height))
        self._analysis_dirty = False
        #: 控えができた素材 次の間隔でこのぶんだけ開き直す
        #: ワーカースレッドが足し、画面のスレッドが取り出すので錠で守る
        #: 守らないと、取り出した直後に足されたぶんが次の回にも残らず、
        #: その素材だけ元のファイルを読み続ける
        self._proxied: set[MediaId] = set()
        self._proxied_lock = threading.Lock()
        #: 使えない控えを捨てて作り直しを頼んだ控えの鍵 2 度目は頼まない
        #: 素材ではなく**鍵**で覚える 鍵には控えの大きさと素材の更新時刻が
        #: 入っているので、設定を変えたり素材を差し替えたりすれば作り直せる
        self._rebuilt: set[str] = set()
        #: AI が結果を確認するための描画係 初めて求められたときに作る
        self._ai_renderer: FrameRenderer | None = None
        #: 編集しているシーン ``None`` ならメイン モデルではなく画面の状態なので
        #: 窓が持つ（保存しない 開き直したらメインから始まる）
        self._active_scene: SceneId | None = None
        #: 裏で調べている読み込み 1 度に 1 回分だけ走らせ、後から頼まれた分は順に待たせる
        #: 並べて走らせると、後から頼んだ方が先に置かれることがあり、置く順が
        #: 頼んだ順と食い違う
        self._import: ProbeBatch | None = None
        #: 待っている読み込み パスと、タイムラインへ落とされた位置（落とされていなければ ``None``）
        self._import_queue: list[tuple[list[Path], _DropTarget | None]] = []
        #: タイムラインへ落とされた読み込みの、置く位置 走っている分の結果と一緒に引く
        #: 読み込みに引っ掛けて持つのは、取り消しやプロジェクトの切り替えで読み込みを
        #: 捨てたときに、位置だけが残って次の読み込みに当たらないようにするため
        self._import_spots: weakref.WeakKeyDictionary[ProbeBatch, _DropTarget] = (
            weakref.WeakKeyDictionary()
        )
        #: 控えと解析の進み具合を出していたか 終わったことを 1 度だけ知らせるため
        self._background_shown = False

        # タブの向きはパネルを重ねる前に決める Qt は重ねたときに使わないタブの並びを
        # 1 つ作って残し、それは作った時の向きのまま変わらない（ほかの部品の下に隠れて
        # 見えはしない） 先に決めておけば、残った物まで同じ向きでそろう
        apply_dock_tabs(self, self._preferences.dock_tabs)
        self._build_widgets()
        self._build_menus()
        self._connect()
        self._connect_drops()
        self._connect_preview_menu()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(ANALYSIS_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._flush_analysis)
        self._refresh_timer.start()

        # 読み込みの間だけ回す（_start_import） 画面のスレッドから読みに行く形にするのは、
        # 裏のスレッドから部品に触ると Qt が落ちるため（解析の知らせと同じ作り）
        self._import_timer = QTimer(self)
        self._import_timer.setInterval(IMPORT_POLL_MS)
        self._import_timer.timeout.connect(self._poll_import)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_MS)
        self._autosave_timer.timeout.connect(self.autosave)
        self._autosave_timer.start()

        # 既定の並びを覚えてから、前回の並びを当てる 逆にすると「初期に戻す」が
        # 前回の並びに戻るだけになる
        self._default_layout = self.saveState(LAYOUT_VERSION)
        self._workspace = Workspace()
        self._workspace.restore(self)
        self._apply_shortcuts(ShortcutStore().load())

        # 渡されたプロジェクトの素材にも効かせる コマンドラインや関連付けから
        # 開く道はここを通るだけで、_on_project_changed を通らない
        # 抜けると、4K のプロジェクトを開いても最初の 1 回だけ等倍のまま重い
        for media in self.view_project.media:
            self._request_proxy(media)
        self._apply_auto_quality()

        self._update_title()

    # --- 組み立て ---

    def _build_widgets(self) -> None:
        project = self._document.project

        self._preview = PreviewWidget(
            project,
            self,
            proxies=self._proxies.store if self._preferences.use_proxy else None,
            prefetch_bytes=self._preferences.prefetch_bytes(),
            decode_threads=self._preferences.decode_threads,
            prefetch_thread=self._preferences.prefetch_thread,
        )
        self._preview.set_handles_enabled(self._preferences.preview_handles)
        self._preview.set_keyframe_drag(self._preferences.keyframe_drag)
        self._preview.set_snap(self._preferences.preview_snap, self._preferences.snap_distance)
        self._transport = TransportBar(project.rate, self)
        self._timeline = TimelineView(project, self._analyzer, self)
        self._timeline.set_value_lines(self._preferences.value_lines)
        self._timeline.set_split_audio(self._preferences.splits_media)
        self._timeline.set_snap(self._preferences.timeline_snap, self._preferences.snap_distance)
        self._media_pool = MediaPoolWidget(project, self)
        self._inspector = InspectorPanel(self)
        # 設定パネルは選んだクリップを引くためにプロジェクトを持つ 起動直後にも渡す
        # （渡さないと、最初の編集まで何本も選んだときのまとめ当てが効かない）
        self._inspector.set_project(self.view_project)
        self._inspector.set_double_click_reset(self._preferences.double_click_reset)
        self._graph = GraphEditor(self)
        # グラフエディタも起動直後にプロジェクトを持たせる 持たせないと、開いた作品で最初の
        # 編集をするまで、キーフレームのあるクリップを選んでも曲線を引けない
        self._graph.set_project(self.view_project)
        self._subtitles = SubtitlePanel(project, self._analyzer, self)
        # 焼き込みのひな形は、タイムラインで選んでいるテキストのクリップ
        self._subtitles.template_provider = self._selected_text_clip
        self._chat = ChatPanel(self, self)
        self._playback = PlaybackController(
            project, self, smooth_history=self._preferences.smooth_audio_motion
        )

        viewer = QWidget(self)
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        if opengl_usable():
            viewer_layout.addWidget(self._preview, 1)
        else:
            # GL を使えない機械では、プレビューを窓に入れない 入れると窓ごと GL で
            # 描くようになり、閉じた後の片付け（ごみ集めか Python の終わり）でプロセス
            # ごと落ちる（#149） 片付けの順を変えても直らなかった 隠すだけでは足りない
            # Qt は GL の部品が子にいるだけで、隠れていても窓を GL で描く 部品そのものは
            # 窓の外に作っておく 窓のほかの所がプレビューへ話しかけるのを、全部で分けずに済む
            self._preview.setParent(None)
            self._preview.hide()
            notice = QLabel(
                "OpenGL 4.3 を使えないため、プレビューを出せません\n"
                "GPU のドライバを確かめてください 編集と保存はできます",
                viewer,
            )
            notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
            notice.setWordWrap(True)
            notice.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()};")
            viewer_layout.addWidget(notice, 1)
        viewer_layout.addWidget(self._transport)
        viewer.setStyleSheet(f"background-color: {Colors.VIEWER_BACKGROUND.name()};")
        self.setCentralWidget(viewer)

        pool_dock = self._dock("メディア", "media")
        pool_dock.setWidget(self._media_pool)
        pool_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, pool_dock)

        inspector_dock = self._dock("オブジェクト設定", "inspector")
        inspector_dock.setWidget(self._inspector)
        inspector_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, inspector_dock)
        self.resizeDocks([inspector_dock], [320], Qt.Orientation.Horizontal)

        graph_dock = self._dock("グラフエディタ", "graph")
        graph_dock.setWidget(self._graph)
        graph_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, graph_dock)
        # 既定では畳んでおく 曲線を触るのは慣れてからで、最初から出ていると
        # 画面が狭くなるだけになる
        graph_dock.hide()
        self._graph_dock = graph_dock

        subtitle_dock = self._dock("字幕", "subtitles")
        subtitle_dock.setWidget(self._subtitles)
        subtitle_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, subtitle_dock)
        # メディアプールと同じ場所にタブで重ねる どちらも「素材を選ぶ」ための
        # パネルで、同時に見る場面が少ない
        self.tabifyDockWidget(pool_dock, subtitle_dock)
        pool_dock.raise_()
        self._subtitle_dock = subtitle_dock

        chat_dock = self._dock("AI アシスタント", "chat")
        chat_dock.setWidget(self._chat)
        chat_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, chat_dock)
        self.tabifyDockWidget(inspector_dock, chat_dock)
        inspector_dock.raise_()
        self._chat_dock = chat_dock

        timeline_dock = self._dock("タイムライン", "timeline")
        self._scene_bar = SceneBar()
        self._scene_bar.set_snap(self._preferences.timeline_snap)
        self._scene_bar.snap_toggled.connect(self._set_snap)
        timeline_panel = QWidget()
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(0)
        timeline_layout.addWidget(self._scene_bar)
        timeline_layout.addWidget(TimelineArea(self._timeline), 1)
        timeline_dock.setWidget(timeline_panel)
        timeline_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, timeline_dock)
        self.resizeDocks([timeline_dock], [320], Qt.Orientation.Vertical)

        self.statusBar().showMessage("素材を読み込んでください")
        # 進み具合は右端に常駐させる 左の一時的な文言に出すと、ほかの操作の
        # 知らせに上書きされて消える
        self._background_indicator = ProgressIndicator(self)
        self._import_indicator = ProgressIndicator(self, cancel_text="取り消す")
        cancel = self._import_indicator.cancel_button
        if cancel is not None:
            cancel.clicked.connect(self.cancel_import)
        self.statusBar().addPermanentWidget(self._background_indicator)
        self.statusBar().addPermanentWidget(self._import_indicator)

    def _dock(self, title: str, name: str) -> QDockWidget:
        """パネルを 1 つ作る

        ``objectName`` が無いと、Qt は画面配置を保存も復元もしない（黙って飛ばす）
        表示名は訳や言い回しで変わりうるので、名前は別に固定の英字で付ける
        """
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        return dock

    def _build_menus(self) -> None:
        file_menu = self._menu("ファイル")
        self._add(file_menu, "新規", QKeySequence.StandardKey.New, self.new_project)
        self._add(file_menu, "開く…", QKeySequence.StandardKey.Open, self.open_project)
        self._add(file_menu, "保存", QKeySequence.StandardKey.Save, self.save_project)
        self._add(file_menu, "名前を付けて保存…", QKeySequence("Ctrl+Shift+S"), self.save_as)
        self._add(
            file_menu, "バックアップのフォルダを開く", QKeySequence(), self.open_backup_folder
        )
        file_menu.addSeparator()
        self._add(file_menu, "プロジェクト設定…", QKeySequence("Ctrl+Shift+P"), self.edit_settings)
        self._add(file_menu, "置き方の方式を切り替える…", QKeySequence(), self.switch_layer_mode)
        file_menu.addSeparator()
        self._add(file_menu, "素材を読み込む…", QKeySequence("Ctrl+I"), self._import_dialog)
        self._add(file_menu, "書き出し…", QKeySequence("Ctrl+E"), self.export)
        self._snapshot_save_action = self._add(
            # triggered の checked（偽）が保存先として渡らないよう、引数なしで呼ぶ
            file_menu,
            "静止画を保存…",
            QKeySequence("Ctrl+Alt+S"),
            lambda: self.save_snapshot(),
        )
        file_menu.addSeparator()
        self._add(file_menu, "終了", QKeySequence.StandardKey.Quit, self.close)

        edit_menu = self._menu("編集")
        self._undo_action = self._add(
            edit_menu, "元に戻す", QKeySequence.StandardKey.Undo, self.undo
        )
        self._redo_action = self._add(
            edit_menu, "やり直す", QKeySequence.StandardKey.Redo, self.redo
        )
        edit_menu.addSeparator()
        self._add(
            edit_menu, "再生ヘッドで分割", QKeySequence("S"), self._timeline.split_at_playhead
        )
        self._add(edit_menu, "削除", QKeySequence("Del"), lambda: self._timeline.delete_selected())
        self._add(
            edit_menu,
            "削除して詰める",
            QKeySequence("Shift+Del"),
            lambda: self._timeline.delete_selected(ripple=True),
        )
        edit_menu.addSeparator()
        # 字幕パネルの表の中で文字を編集しているあいだは、Qt が入力欄のほうへ
        # Ctrl+C を渡す（入力欄が標準のキーを先に取る） 文字のコピーと取り合わない
        self._add(edit_menu, "コピー", QKeySequence.StandardKey.Copy, self._timeline.copy_selected)
        self._add(edit_menu, "切り取り", QKeySequence.StandardKey.Cut, self._timeline.cut_selected)
        self._add(
            edit_menu,
            "貼り付け（再生ヘッドの位置）",
            QKeySequence.StandardKey.Paste,
            self._timeline.paste_at_playhead,
        )
        self._add(
            edit_menu, "すべて選択", QKeySequence.StandardKey.SelectAll, self._timeline.select_all
        )
        # Ctrl+C はクリップのコピーが使っている 絵のコピーは Alt を足して分ける
        self._snapshot_copy_action = self._add(
            edit_menu, "静止画をコピー", QKeySequence("Ctrl+Alt+C"), self.copy_snapshot
        )
        # 範囲を決めるのは目盛りの Shift+ドラッグ 解除は右クリックのほかにここにも置く
        # 範囲が横へスクロールして見えていなくても、書き出す前に消せるようにするため
        self._add(edit_menu, "書き出し範囲を解除", QKeySequence(), self._timeline.clear_work_area)
        edit_menu.addSeparator()
        # Ctrl+G はグラフエディタが先に使っている 今ある割り当ては変えない
        self._add(
            edit_menu, "グループ化", QKeySequence("Ctrl+Alt+G"), self._timeline.group_selected
        )
        self._add(
            edit_menu,
            "グループ解除",
            QKeySequence("Ctrl+Alt+Shift+G"),
            self._timeline.ungroup_selected,
        )
        edit_menu.addSeparator()
        # ヘッダのボタンと同じ切り替えをメニューにも置く キーボードだけで操作する人の
        # 入口で、ショートカットの設定にも載る
        for text, key, attribute in (
            ("トラックをミュート", "Shift+M", "muted"),
            ("トラックをソロ", "Shift+S", "solo"),
            ("トラックをロック", "Shift+L", "locked"),
        ):
            self._add(
                edit_menu,
                text,
                QKeySequence(key),
                functools.partial(self._toggle_track, attribute),
            )

        object_menu = self._menu("オブジェクト")
        self._add(object_menu, "テキストを追加", QKeySequence("Ctrl+T"), self.add_text)
        self._add(object_menu, "図形を追加", QKeySequence("Ctrl+Shift+T"), self.add_shape)
        self._add(object_menu, "場面切り替えを追加", QKeySequence(), self.add_transition)
        self._add(object_menu, "フィルタを追加", QKeySequence(), self.add_filter)

        scene_menu = self._menu("シーン")
        self._add(scene_menu, "新しいシーン…", QKeySequence("Ctrl+Alt+N"), self._ask_new_scene)
        self._add(scene_menu, "シーンを置く…", QKeySequence("Ctrl+Alt+P"), self._ask_place_scene)
        scene_menu.addSeparator()
        self._add(scene_menu, "シーンの名前を変更…", QKeySequence(), self._ask_rename_scene)
        self._add(scene_menu, "シーンを削除", QKeySequence(), self.remove_active_scene)
        scene_menu.addSeparator()
        self._add(
            scene_menu, "メインに戻る", QKeySequence("Ctrl+Alt+M"), lambda: self.open_scene(None)
        )

        subtitle_menu = self._menu("字幕")
        self._add(subtitle_menu, "字幕パネル", QKeySequence("Ctrl+Shift+U"), self.show_subtitles)
        subtitle_menu.addSeparator()
        self._add(subtitle_menu, "起こす…", QKeySequence("Ctrl+U"), self.transcribe)
        self._add(subtitle_menu, "整形…", QKeySequence("Ctrl+Shift+F"), self._subtitles.clean)
        self._add(
            subtitle_menu,
            "無音カット…",
            QKeySequence("Ctrl+Shift+J"),
            self._subtitles.jet_cut,
        )
        subtitle_menu.addSeparator()
        self._add(subtitle_menu, "焼き込み", QKeySequence(), self._subtitles.burn)
        self._add(subtitle_menu, "書き出し…", QKeySequence(), self._subtitles.export_file)

        compat_menu = self._menu("互換")
        self._add(
            compat_menu,
            "オブジェクトを読み込む…",
            QKeySequence("Ctrl+Shift+O"),
            self.import_exo,
        )
        self._add(
            compat_menu,
            "テンプレート…",
            QKeySequence("Ctrl+Shift+D"),
            self.show_templates,
        )
        compat_menu.addSeparator()
        self._add(compat_menu, "スクリプトを読み直す", QKeySequence(), self.rescan_scripts)
        self._add(compat_menu, "スクリプトフォルダを開く", QKeySequence(), self.open_script_folder)
        self._add(compat_menu, "互換性レポート…", QKeySequence(), self.show_compatibility)

        ai_menu = self._menu("AI")
        self._add(ai_menu, "アシスタント", QKeySequence("Ctrl+Shift+A"), self.show_chat)

        view_menu = self._menu("表示")
        self._add(
            view_menu, "拡大", QKeySequence.StandardKey.ZoomIn, lambda: self._timeline.zoom(1.25)
        )
        self._add(
            view_menu, "縮小", QKeySequence.StandardKey.ZoomOut, lambda: self._timeline.zoom(0.8)
        )
        self._add(view_menu, "全体を表示", QKeySequence("Shift+Z"), self._timeline.zoom_to_fit)
        self._add(
            view_menu,
            "グラフエディタ",
            QKeySequence("Ctrl+G"),
            lambda: self._graph_dock.setVisible(not self._graph_dock.isVisible()),
        )
        view_menu.addSeparator()
        self._add(
            view_menu,
            "トラックを高く",
            QKeySequence("Ctrl+Shift+Up"),
            lambda: self._timeline.adjust_track_heights(HEIGHT_STEP),
        )
        self._add(
            view_menu,
            "トラックを低く",
            QKeySequence("Ctrl+Shift+Down"),
            lambda: self._timeline.adjust_track_heights(-HEIGHT_STEP),
        )
        self._add(
            view_menu, "トラックの高さを戻す", QKeySequence(), self._timeline.reset_track_heights
        )
        view_menu.addSeparator()
        self._add(view_menu, "画面配置を初期に戻す", QKeySequence(), self.reset_layout)
        self._add(view_menu, "ショートカットの設定…", QKeySequence(), self.customize_shortcuts)
        self._add(view_menu, "設定…", QKeySequence(), self.edit_preferences)

        playback_menu = self._menu("再生")
        self._add(playback_menu, "再生 / 停止", QKeySequence("Space"), self._playback.toggle)

        # ソフトの中から受け口へ辿れるようにする 辿れないと、困った人は検索で
        # 別の場所（古い配布先や無関係の掲示板）に書き、こちらには届かない
        help_menu = self._menu("ヘルプ")
        self._add(help_menu, "使い方", QKeySequence("F1"), self.open_manual)
        self._add(help_menu, "不具合・要望を送る", QKeySequence(), self.open_report_page)
        help_menu.addSeparator()
        self._add(help_menu, "バージョン情報…", QKeySequence(), self.show_about)

        self._update_history_actions()

    def _add(
        self,
        menu: QMenu,
        text: str,
        shortcut: QKeySequence | QKeySequence.StandardKey,
        slot: Callable[[], object],
    ) -> QAction:
        action = QAction(text, self)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        # 名前は作った時点の表示で決める 「元に戻す: 分割」のように表示が
        # あとから変わる項目があり、そちらで引くと保存した割り当てが外れる
        self._actions[f"{menu.title()}/{text}"] = (action, action.shortcut().toString(_PORTABLE))
        return action

    def _apply_shortcuts(self, bindings: dict[str, str]) -> None:
        """割り当てを当てる 知らない名前は飛ばす（版が変わって消えた項目など）"""
        for name, key in bindings.items():
            entry = self._actions.get(name)
            if entry is not None:
                entry[0].setShortcut(QKeySequence(key, _PORTABLE))

    def customize_shortcuts(self) -> None:
        from sashimono.ui.shortcut_dialog import ShortcutDialog, ShortcutRow

        rows = [
            ShortcutRow(name, action.shortcut().toString(_PORTABLE), default)
            for name, (action, default) in self._actions.items()
        ]
        dialog = ShortcutDialog(rows, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        bindings = dialog.bindings()
        self._apply_shortcuts(bindings)
        overrides = {name: key for name, key in bindings.items() if key != self._actions[name][1]}
        try:
            ShortcutStore().save(overrides)
        except OSError as exc:
            self.statusBar().showMessage(f"ショートカットを保存できなかった: {exc}", 5000)

    def edit_preferences(self) -> None:
        """本人の好みの設定を変える"""
        from sashimono.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(self._preferences, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_preferences(dialog.preferences())

    def _apply_preferences(self, preferences: Preferences) -> None:
        """設定を今の画面へ反映して保存する

        控えの大きさを変えたら別の鍵になるので、作り直しを頼む
        古い控えは残るが、掴むことはない（鍵に大きさを混ぜてある）
        """
        # 作り直すのは大きさが変わったときと、控えを切ったとき
        # 大きさは置き場の鍵が変わるため 切ったときは**走っている変換を止める**ため
        # （切ったのに裏で変換が続くなら、切った意味が無い）
        # 何が変わっても作り直すと、画質の設定を触っただけで進行中の変換が止まる
        resized = preferences.proxy_height != self._preferences.proxy_height
        stopped = self._preferences.use_proxy and not preferences.use_proxy
        self._preferences = preferences
        try:
            PreferenceStore().save(preferences)
        except OSError as exc:
            self.statusBar().showMessage(f"設定を保存できなかった: {exc}", 5000)

        if resized or stopped:
            self._proxies.close()
            self._proxies = ProxyBuilder(ProxyStore(height=preferences.proxy_height))
            # 止めた変換は終わっていない 「終わった」と知らせない
            self._background_shown = False
        if not preferences.pool_progress:
            # 次の間隔を待たずに外す 切ったのに行の後ろに古い割合が残って見える
            self._media_pool.set_progress({})
        self._media_pool.set_view_mode(preferences.media_view)
        self._chat.apply_preferences(preferences)
        self._timeline.set_value_lines(preferences.value_lines)
        self._playback.set_smooth_history(preferences.smooth_audio_motion)
        self._timeline.set_split_audio(preferences.splits_media)
        self._timeline.set_snap(preferences.timeline_snap, preferences.snap_distance)
        self._scene_bar.set_snap(preferences.timeline_snap)
        self._inspector.set_double_click_reset(preferences.double_click_reset)
        apply_dock_tabs(self, preferences.dock_tabs)
        self._preview.set_proxies(self._proxies.store if preferences.use_proxy else None)
        self._preview.set_prefetch_bytes(preferences.prefetch_bytes())
        self._preview.set_prefetch_thread(preferences.prefetch_thread)
        self._preview.set_decode_threads(preferences.decode_threads)
        self._preview.set_handles_enabled(preferences.preview_handles)
        self._preview.set_keyframe_drag(preferences.keyframe_drag)
        self._preview.set_snap(preferences.preview_snap, preferences.snap_distance)
        if native.enabled() != preferences.native_modules:
            native.set_enabled(preferences.native_modules)
            # 汎用プラグインも同じ設定で入り切りする 切ったときに覚えている
            # モジュールを捨てないと、切ったあとも同じ表が返り続ける
            plugin.forget()
            # 読む・読まないで絵が変わる 先読みした絵は全部使えない
            self._preview.refresh_all()
        if plugin.scan_all() != preferences.all_aviutl_plugins:
            # 探す範囲が変わると、見つかるモジュールが変わって絵が変わる
            plugin.set_scan_all(preferences.all_aviutl_plugins)
            self._preview.refresh_all()
        for media in self.view_project.media:
            self._request_proxy(media)
        self._apply_auto_quality()

    def _apply_auto_quality(self) -> None:
        """置いてある素材の大きさに合わせて、プレビューの画質を決める

        1 枚だけなら元の素材でも入る それでも大きさで決めるのは、重ねた時点で
        入らなくなるため 測った値は :mod:`sashimono.engine.cache.proxy` の表を見る
        """
        tallest = max(
            (
                stream.display_size[1]
                for media in self.view_project.media
                for stream in media.video_streams
            ),
            default=0,
        )
        self._transport.set_quality(self._preferences.quality_for(tallest))

    def reset_layout(self) -> None:
        self.restoreState(self._default_layout, LAYOUT_VERSION)

    def _menu(self, title: str) -> QMenu:
        """メニューを 1 つ作る ``addMenu`` は None を返しうるので、ここで確定させる"""
        menu = self.menuBar().addMenu(title)
        if menu is None:  # pragma: no cover - Qt が None を返すのは異常系のみ
            raise RuntimeError(f"メニューを作れない: {title}")
        return menu

    def _connect(self) -> None:
        self._timeline.commands_requested.connect(self.execute_all)
        self._timeline.commands_continued.connect(
            lambda commands, label: self.execute_all(commands, label, merge=True)
        )
        self._timeline.playhead_moved.connect(self._on_playhead_moved)
        self._timeline.template_requested.connect(self.place_template_entry)
        self._timeline.preview_requested.connect(self._preview_command)

        self._media_pool.import_requested.connect(self.import_media)
        self._media_pool.insert_requested.connect(self._insert_media_by_id)
        self._media_pool.transcribe_requested.connect(self._transcribe_media)
        self._media_pool.remove_requested.connect(self._remove_media)
        self._timeline.status_message.connect(
            lambda message: self.statusBar().showMessage(message, 4000)
        )

        self._timeline.selection_changed.connect(self._on_selection_changed)
        self._timeline.scene_open_requested.connect(
            lambda scene_id: self.open_scene(SceneId(scene_id))
        )
        self._scene_bar.scene_selected.connect(
            lambda scene_id: self.open_scene(SceneId(scene_id) if scene_id else None)
        )
        self._scene_bar.add_requested.connect(self._ask_new_scene)
        self._scene_bar.rename_requested.connect(self._ask_rename_scene)
        self._scene_bar.remove_requested.connect(self.remove_active_scene)
        self._scene_bar.place_requested.connect(self._ask_place_scene)
        self._inspector.commands_requested.connect(self.execute_all)
        self._inspector.preview_requested.connect(self._preview_command)
        self._inspector.curve_selected.connect(self._show_curve)
        # ◆ や ◀ ▶ を押した値を、グラフエディタにも出す（開いていなければ開かない）
        self._inspector.param_focused.connect(self._graph.set_path)
        self._inspector.seek_requested.connect(self._seek)
        self._inspector.align_requested.connect(self._align_selected)
        # 触ったのが部分フィルタなら、プレビューにその範囲の枠を出す
        self._inspector.effect_focused.connect(
            lambda effect_id: self._preview.set_region_effect(EffectId(effect_id))
        )
        self._graph.commands_requested.connect(self.execute_all)
        self._graph.seek_requested.connect(self._seek)

        self._subtitles.commands_requested.connect(self.execute_all)
        self._subtitles.seek_requested.connect(self._seek)
        self._subtitles.status_message.connect(
            lambda message: self.statusBar().showMessage(message, 5000)
        )

        self._chat.status_message.connect(
            lambda message: self.statusBar().showMessage(message, 5000)
        )
        self._chat.apply_preferences(self._preferences)
        self._chat.choices_changed.connect(
            lambda model, effort: self._remember_preferences(
                replace(self._preferences, ai_model=model, ai_effort=effort)
            )
        )

        self._transport.play_toggled.connect(self._playback.toggle)
        self._transport.step_requested.connect(
            lambda delta: self._seek(self._timeline.playhead + delta)
        )
        self._transport.jump_requested.connect(self._seek)
        self._transport.quality_changed.connect(self._preview.set_quality)

        self._playback.frame_changed.connect(self._on_playback_frame)
        self._playback.state_changed.connect(self._transport.set_playing)
        # 再生中は先読みを止める 同じ GPU を奪い合うと、いま出すべきコマが遅れる
        self._playback.state_changed.connect(self._preview.set_playing)
        # 先読みを止めたら伝える 黙って効かない状態にしない
        self._preview.prefetch_stopped.connect(
            lambda message: self.statusBar().showMessage(message, 5000)
        )
        self._playback.failed.connect(lambda message: self.statusBar().showMessage(message, 5000))
        # プレビューで直接動かす 選択はタイムラインと 1 つにそろえる
        self._preview.clip_picked.connect(lambda clip_id: self._timeline.select(ClipId(clip_id)))
        self._preview.commands_requested.connect(self.execute_all)
        self._preview.preview_requested.connect(self._preview_commands)

    # --- コマンドの実行 ---

    def execute(self, command: Command) -> bool:
        """コマンドを 1 つ実行して、画面を更新する

        失敗しても落とさず、状況をステータスバーへ出す 編集操作は思いどおりに
        いかないことが普通にあり、そのたびにダイアログが出ると邪魔になる

        断られたときは偽を返す :meth:`execute_all` と同じで、成功した前提で
        続きを進めると、入っていない素材を参照したり「置いた」と出したりする
        """
        try:
            self._document.execute(self._in_active_scene(command))
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return False
        self._on_project_changed()
        return True

    def execute_all(self, commands: list[Command], label: str, *, merge: bool = False) -> bool:
        """複数のコマンドを 1 回の Undo で戻せるようにまとめて実行する

        ``merge`` が真なら、直前の同じ操作の段へまとめる（:meth:`Document.checkpoint`）
        断られてまとめて戻したときは偽を返す 呼び出し側が成功した前提で続きを
        進めると、戻した素材の解析を頼んだり、置けていないのに「置いた」と出したりする
        """
        if not commands:
            return True
        try:
            with self._document.checkpoint(label, merge=merge):
                for command in commands:
                    self._document.execute(self._in_active_scene(command))
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
            self._on_project_changed()
            return False
        self._on_project_changed()
        return True

    def _in_active_scene(self, command: Command) -> Command:
        """開いているシーンの中で実行するよう包む メインなら包まない

        画面のパネルも AI も、見ているタイムライン（:attr:`view_project`）を相手に
        コマンドを作る 包み忘れると、シーンを開いて足したクリップがメインに入る
        """
        if self._active_scene is None or isinstance(command, InScene):
            return command
        return InScene(self._active_scene, command)

    @property
    def view_project(self) -> Project:
        """いま編集しているタイムラインを ``timeline`` に差し込んだプロジェクト

        タイムライン・プレビュー・設定パネルはこれを見る 保存と書き出しは
        いつもメイン（:attr:`document` のプロジェクト）
        """
        project = self._document.project
        if self._active_scene is None:
            return project
        scene = project.find_scene(self._active_scene)
        return project if scene is None else replace(project, timeline=scene.timeline)

    @property
    def active_scene(self) -> SceneId | None:
        return self._active_scene

    def open_scene(self, scene_id: SceneId | None) -> None:
        """編集するシーンを切り替える ``None`` ならメイン"""
        if scene_id is not None and self._document.project.find_scene(scene_id) is None:
            self.statusBar().showMessage("そのシーンは見つかりません", 4000)
            return
        if scene_id == self._active_scene:
            return
        self._playback.stop()
        self._active_scene = scene_id
        # 選んでいたクリップは別のタイムラインのもの 残すと、開いた先で
        # 「見つからない」になる
        self._timeline.select(None)
        self._on_project_changed()
        self._seek(0)

    def create_scene(self, name: str) -> SceneId | None:
        """空のシーンを作って開く"""
        name = name.strip()
        if not name:
            return None
        scene = new_scene(self._document.project, name)
        try:
            self._document.execute(AddScene(scene))
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return None
        self._on_project_changed()
        self.open_scene(scene.id)
        return scene.id

    def rename_active_scene(self, name: str) -> None:
        if self._active_scene is not None:
            self.execute(RenameScene(self._active_scene, name))

    def remove_active_scene(self) -> None:
        """開いているシーンを消してメインへ戻る どこかに置かれていれば断られる"""
        target = self._active_scene
        if target is None:
            self.statusBar().showMessage("メインは消せません", 4000)
            return
        try:
            self._document.execute(RemoveScene(target))
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        self._active_scene = None
        self._on_project_changed()

    def place_scene(self, scene_id: SceneId) -> None:
        """シーンを、いま編集しているタイムラインの再生ヘッドの位置へ置く"""
        try:
            commands = insert_scene(self.view_project, scene_id, at_frame=self._timeline.playhead)
        except KeyError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        scene = self._document.project.require_scene(scene_id)
        self.execute_all(commands, f"シーンを置く: {scene.name}")

    def _ask_new_scene(self) -> None:
        count = len(self._document.project.scenes) + 1
        name, accepted = QInputDialog.getText(
            self, "新しいシーン", "シーンの名前", text=f"シーン {count}"
        )
        if accepted:
            self.create_scene(name)

    def _ask_rename_scene(self) -> None:
        if self._active_scene is None:
            self.statusBar().showMessage("メインの名前は変えられません", 4000)
            return
        scene = self._document.project.require_scene(self._active_scene)
        name, accepted = QInputDialog.getText(self, "シーンの名前", "新しい名前", text=scene.name)
        if accepted:
            self.rename_active_scene(name)

    def _ask_place_scene(self) -> None:
        # 開いているシーン自身は置けない（入れ子が自分へ戻る） 選択肢から外す
        choices = [s for s in self._document.project.scenes if s.id != self._active_scene]
        if not choices:
            self.statusBar().showMessage(
                "置けるシーンがありません（先にシーンを作ってください）", 5000
            )
            return
        # 同じ名前のシーンがあると、名前から引き直したときに先頭のものを選んでしまう
        # 番号を付けて、選んだ行の位置でシーンを決める
        names = [f"{index}. {scene.name}" for index, scene in enumerate(choices, start=1)]
        name, accepted = QInputDialog.getItem(self, "シーンを置く", "置くシーン", names, 0, False)
        if accepted and name in names:
            self.place_scene(choices[names.index(name)].id)

    def undo(self) -> None:
        self._document.undo()
        self._on_project_changed()

    def redo(self) -> None:
        self._document.redo()
        self._on_project_changed()

    def _on_project_changed(self) -> None:
        root = self._document.project
        if self._active_scene is not None and root.find_scene(self._active_scene) is None:
            # 取り消しでシーンが消えたら、メインへ戻る
            self._active_scene = None
        project = self.view_project
        self._scene_bar.set_project(root, self._active_scene)
        self._timeline.set_open_scene(self._active_scene)
        self._timeline.set_project(project)
        self._media_pool.set_project(root)
        self._inspector.set_project(project)
        self._graph.set_project(project)
        self._subtitles.set_project(project)
        self._preview.set_project(project)
        self._playback.set_project(project)
        self._transport.set_rate(project.rate)
        self._transport.set_duration(project.duration)
        self._retime_playhead(project.rate)
        # 素材が増えたら画質を見直す 4K を 1 本置いた時点で重くなるので、
        # 置いたあとに自分で下げてもらうのでは遅い
        self._apply_auto_quality()
        self._update_history_actions()
        self._update_title()
        self.project_changed.emit(project)

    def _retime_playhead(self, rate: FrameRate) -> None:
        """フレームレートが変わったら、再生ヘッドを同じ時刻（秒）のまま数え直す

        最初の動画に合わせたとき（とその取り消し）に起きる 数のまま残すと、30fps の
        1 秒（フレーム 30）にあった再生ヘッドが 60fps では 0.5 秒を指し、表示の時刻も
        再生を始める位置も変わる 長さで丸めない（:meth:`_seek` と違う） 合わせるのは
        空のプロジェクトなので、丸めると置く前に先頭へ飛ぶ
        """
        before, self._playhead_rate = self._playhead_rate, rate
        if before == rate:
            return
        frame = retime_frame(self._timeline.playhead, before, rate)
        self._timeline.set_playhead(frame)
        self._show_frame(frame)
        self._playback.set_frame(frame)

    def _update_history_actions(self) -> None:
        self._undo_action.setEnabled(self._document.can_undo)
        self._redo_action.setEnabled(self._document.can_redo)
        undo_label = self._document.undo_label
        self._undo_action.setText(f"元に戻す: {undo_label}" if undo_label else "元に戻す")
        redo_label = self._document.redo_label
        self._redo_action.setText(f"やり直す: {redo_label}" if redo_label else "やり直す")

    @property
    def is_modified(self) -> bool:
        """同一性で比べる 中身の等しさで比べると、履歴 1 段ごとにツリー全体を
        比較することになり、大きなプロジェクトでタイトルの更新が重くなる
        """
        return self._document.project is not self._saved

    def _toggle_track(self, attribute: str) -> None:
        if not self._timeline.toggle_selected_track(attribute):
            self.statusBar().showMessage("先にクリップを選んでください（そのトラックが対象）", 4000)

    def _update_title(self) -> None:
        name = self._path.name if self._path is not None else self._document.project.name
        mark = " *" if self.is_modified else ""
        self.setWindowTitle(f"{name}{mark} — Sashimono Edit")

    # --- 素材 ---

    def _import_dialog(self) -> None:
        from sashimono.ui.media_pool import MEDIA_FILTER

        names, _ = QFileDialog.getOpenFileNames(self, "素材を読み込む", "", MEDIA_FILTER)
        if names:
            self.import_media([Path(name) for name in names])

    def import_media(self, paths: list[Path], *, at: DropSpot | None = None) -> None:
        """素材を読み込んでタイムラインへ置く 調べるのは裏のスレッドで、待たずに返る

        複数選ばれた場合もまとめて 1 回の Undo で戻せるようにする
        10 本読み込んで 10 回取り消す、という操作は誰も望まない
        置くのは全部を調べ終えてから 1 回の操作で行う（:meth:`_apply_import`）
        読み込みの最中にもう一度頼まれたら、前の分が終わってから順に調べる
        ダイアログ・一覧のボタン・一覧やタイムラインへの落とし込みは、どれもここへ来る
        ``at`` はタイムラインへ落とされた位置 無ければ末尾へ並べる 落とされた位置は、
        そのとき開いているシーンの位置として覚える（:meth:`_apply_import`）
        """
        if not paths:
            return
        target = (
            _DropTarget(at, self._active_scene, self._document.project.rate)
            if at is not None
            else None
        )
        if self._import is not None:
            self._import_queue.append((list(paths), target))
            self._show_import_progress()
            return
        self._start_import(list(paths), target)

    @property
    def importing(self) -> bool:
        """素材を調べている最中か、調べるのを待っている読み込みがある"""
        return self._import is not None or bool(self._import_queue)

    def cancel_import(self) -> None:
        """読み込みを取り消す 待っている分も捨て、何も置かない

        調べ終えた分だけ置く形にしないのは、取り消したのに一部が入ると、
        どこまで入ったのかを一覧で確かめ直すことになるため
        """
        if self._drop_imports():
            self.statusBar().showMessage("素材の読み込みを取り消した", 4000)

    def _drop_imports(self) -> bool:
        """走っている読み込みと待っている読み込みを捨てる 捨てた物があれば真

        調べている最中の 1 本は止まらないが、ここで外すので終わっても置きに来ない
        （置くのは :meth:`_poll_import` が ``self._import`` を見たときだけ）
        """
        batch = self._import
        if batch is None and not self._import_queue:
            return False
        self._import = None
        self._import_queue.clear()
        if batch is not None:
            batch.cancel()
        self._import_timer.stop()
        self._import_indicator.hide()
        return True

    def _leave_project(self, previous: Project) -> None:
        """プロジェクトを差し替えた（新規・開く・復元）ときに、前のプロジェクトの裏の仕事を片付ける

        読み込みは始めたときのプロジェクトへ置く約束 捨てずに置くと、調べ終わった
        素材が差し替えた先のプロジェクトに入り、その取り消しの履歴にまで載る
        控えと解析も、前のプロジェクトにしか無い素材の分は外す 残すと、新しい
        プロジェクトのステータスバーに、前の素材の本数と失敗が出続ける
        """
        if self._drop_imports():
            self.statusBar().showMessage(
                "プロジェクトを切り替えたので、素材の読み込みを取り消した", 5000
            )
        kept = {media.id for media in self._document.project.media}
        for media in previous.media:
            if media.id not in kept:
                self._analyzer.forget(media.id)
                self._proxies.forget(media.id)
        # 前のプロジェクトで出していた進み具合の続きとして「終わった」と出さない
        self._background_shown = False
        self._background_indicator.hide()

    def wait_for_imports(self, timeout: float = 30.0) -> bool:
        """読み込みが置き終わるまで待つ 間に合えば真

        写真を撮る道具と試験のためのもの 画面の操作からは呼ばない（呼ぶと、
        裏へ移した意味が無くなり、また調べ終わるまで画面が固まる）
        """
        deadline = time.monotonic() + timeout
        while self.importing:
            if time.monotonic() > deadline:
                return False
            QApplication.processEvents()
            # タイマーを待たずに見に行く 試験の中ではタイマーが回るとは限らない
            self._poll_import()
            time.sleep(0.005)
        return True

    def _start_import(self, paths: list[Path], at: _DropTarget | None = None) -> None:
        # probe_media はこのモジュールの名前から引く 試験がここを差し替えて、
        # 開けない素材や断られる読み込みを作る
        # 読み込み直しでは必ず開き直す 覚えた結果は再生中のデコーダの作り直しのための物
        for path in paths:
            forget_probe(path)
        self._import = ProbeBatch(paths, probe_media)
        if at is not None:
            self._import_spots[self._import] = at
        self._show_import_progress()
        self._import_timer.start()

    def _show_import_progress(self) -> None:
        batch = self._import
        if batch is None:
            self._import_indicator.hide()
            return
        done = batch.progress()
        self._import_indicator.show_progress(
            describe_import(done, batch.total, len(self._import_queue)),
            done / batch.total if batch.total else 1.0,
            tooltip="\n".join(str(path) for path in batch.paths),
        )

    def _poll_import(self) -> None:
        batch = self._import
        if batch is None:
            self._import_timer.stop()
            self._import_indicator.hide()
            return
        if not batch.finished:
            self._show_import_progress()
            return
        # 置く前に外す 置く途中で例外が出ても、同じ読み込みを 30ms ごとに
        # 置き直そうとし続けない
        self._import = None
        try:
            self._apply_import(batch)
        finally:
            if self._import_queue:
                self._start_import(*self._import_queue.pop(0))
            else:
                self._import_timer.stop()
                self._import_indicator.hide()

    def _apply_import(self, batch: ProbeBatch) -> None:
        """調べ終えた素材を 1 回の操作で置く

        置く位置は調べ終えた時点のタイムラインから決める 調べている間に
        編集していても、その後ろへ並ぶ タイムラインへ落とされた読み込みは、
        落とされた位置から順に並べる（:meth:`_place_dropped`）
        落とされた読み込みは、落としたときに開いていたシーンへ置く 調べている間に
        別のシーンへ切り替えても、切り替えた先へは入れない そのシーンが消えていたら
        素材を一覧へ入れるだけにする
        """
        commands: list[Command] = []
        failures: list[str] = []
        loaded: list[MediaItem] = []
        target = self._import_spots.pop(batch, None)
        spot = target.spot if target is not None else None
        scene = target.scene if target is not None else self._active_scene
        outcomes = batch.results()
        if self._scene_project(scene) is not None:
            self._match_project_to([item for item in outcomes if isinstance(item, MediaItem)])
        if spot is not None and target is not None:
            now = self._document.project.rate
            spot = replace(spot, frame=retime_frame(spot.frame, target.rate, now))
        found = self._scene_project(scene)
        lost = found is None
        if found is None:
            project, spot, scene = self._document.project, None, None
        else:
            project = found

        for outcome in outcomes:
            if isinstance(outcome, ProbeError):
                failures.append(str(outcome))
                continue
            placed = [AddMedia(outcome)] if lost else self._place_dropped(project, outcome, spot)
            if spot is not None:
                spot = spot.after(placed)
            for command in placed:
                project = command.apply(project)
            commands.extend(placed)
            loaded.append(outcome)

        if commands and not self._execute_in_scene(
            commands, f"素材を読み込み: {batch.total} 件", scene
        ):
            # 断られるとまとめて戻る 一覧に無い素材の解析と控えを頼まないために、
            # 頼むのは通ってからにする 理由は execute_all が出しているので上書きしない
            return
        # 解析と控えは、取り消しても止められない裏の処理 入ったことを確かめてから頼む
        for media in loaded:
            self._analyzer.request(media, on_ready=self._on_analysis_ready)
            self._request_proxy(media)
        if failures:
            self.statusBar().showMessage(failures[0], 5000)
        elif lost and commands:
            self.statusBar().showMessage(
                "落とした先のシーンが無くなったので、素材を一覧へ入れるだけにした", 5000
            )
        elif commands:
            self.statusBar().showMessage(f"{batch.total} 件を読み込んだ", 3000)

    def add_text(self) -> None:
        """再生ヘッドの位置にテキストを置く"""
        self._insert_generated(TEXT.create(), "テキストを追加")

    def add_shape(self) -> None:
        self._insert_generated(SHAPE.create(), "図形を追加")

    def add_transition(self) -> None:
        """再生ヘッドの位置に場面切り替えを置く 下のトラックの切れ目に重ねて使う"""
        self._insert_generated(TRANSITION.create(), "場面切り替えを追加")

    def add_filter(self) -> None:
        """再生ヘッドの位置に、下のトラックの絵全体へ掛かるフィルタを置く

        エフェクトは積まずに置く 何を掛けたいかは人によるので、選んだ後に設定パネルで足す
        """
        self._place_generated(
            insert_filter(self.view_project, at_frame=self._timeline.playhead), "フィルタを追加"
        )

    def _insert_generated(self, source: GeneratedSource, label: str) -> None:
        self._place_generated(
            insert_generated(self.view_project, source, at_frame=self._timeline.playhead), label
        )

    def _place_generated(self, commands: list[Command], label: str) -> None:
        if not self.execute_all(commands, label):
            # 断られたら選ばない 選ぶと再生ヘッドの位置に元からあったクリップが
            # 選ばれ、設定パネルが開いて、追加できたように見える
            return
        # 置いたものをすぐ選ぶ 設定パネルが開いていないと、
        # 追加したのに何も起きていないように見える
        placed = self._last_added_clip(commands)
        if placed is not None:
            self._timeline.select(placed)

    def _last_added_clip(self, commands: list[Command]) -> ClipId | None:
        """置いた生成オブジェクトのクリップ

        置いた命令から取る 再生ヘッドの所を映像トラックから探すと、レイヤー（混合）に
        置いた物が見つからず、追加したのに設定パネルが開かない
        """
        timeline = self.view_project.timeline
        for command in reversed(commands):
            if isinstance(command, AddClip) and timeline.locate_clip(command.clip.id) is not None:
                return command.clip.id
        return None

    def show_subtitles(self) -> None:
        """字幕パネルを前へ出す"""
        self._subtitle_dock.show()
        self._subtitle_dock.raise_()

    def transcribe(self) -> None:
        """選択中の素材を起こす パネルを出してから始める

        起こしの実行環境は既定では入っていない 未導入なら、そのダイアログが
        導入のボタンを出す（:mod:`sashimono.asr.environment` を参照）
        """
        self.show_subtitles()
        # タイムラインで選んだクリップの素材を先に見る 字幕パネルは選んだクリップに付いて
        # いくので、選んでいればもう出ている メディア欄で選んだ素材は、クリップを選んで
        # いないときだけ使う（前はこちらを先に見て、別の素材を起こしていた）
        sound = self._selected_sound()
        if sound is not None:
            # 選びが変わらないまま、メディア欄の右クリックで別の素材を起こしていると、字幕
            # パネルはその素材のまま 選んだクリップへ合わせ直す（PR #231 の指摘）
            self._subtitles.follow_clip(*sound)
        else:
            selected = self._media_pool.selected_media_id()
            if selected is not None:
                self._subtitles.select_media(selected)
        self._subtitles.transcribe()

    def _selected_text_clip(self) -> Clip | None:
        """タイムラインで選んでいるテキストのクリップ 字幕の焼き込みの見た目のひな形にする"""
        primary = self._timeline.selected_clip
        located = self.view_project.timeline.locate_clip(primary) if primary else None
        if located is None:
            return None
        clip = located[1]
        return clip if clip.source is not None and clip.source.kind == "text" else None

    def _selected_sound(self) -> tuple[MediaId, int | None] | None:
        """タイムラインで選んだクリップの、音のある素材と鳴らす音声ストリーム"""
        primary = self._timeline.selected_clip
        project = self.view_project
        located = project.timeline.locate_clip(primary) if primary is not None else None
        if located is None or located[1].media_id is None:
            return None
        track, clip = located
        media = project.find_media(located[1].media_id)
        if media is None or not media.has_audio:
            return None
        # 映像のクリップ（音を鳴らさない）は、素材の 1 本目の音を初めに選んでおく
        stream = heard_stream(track, clip) if project.plays_sound(track, clip) else None
        return media.id, stream

    def _transcribe_media(self, media_id: str) -> None:
        """メディアプールの右クリックから起こす その素材を字幕パネルで選んでから始める"""
        self.show_subtitles()
        self._subtitles.select_media(MediaId(media_id))
        self._subtitles.transcribe()

    def _remove_media(self, media_id: str) -> None:
        """プールから外す タイムラインで使っていれば、理由がステータスバーに出て止まる

        外せたときだけ解析の結果（波形・サムネイル・走っている解析）も捨てる 残すと、
        もう使わない素材の波形をメモリに抱え続ける 外せなかったときは使い続けるので残す
        """
        target = MediaId(media_id)
        self.execute(RemoveMedia(target))
        if self._document.project.find_media(target) is None:
            self._analyzer.forget(target)
            self._proxies.forget(target)

    # --- タイムラインへの落とし込みと、素材一覧の表示 ---

    def _connect_drops(self) -> None:
        self._timeline.files_dropped.connect(self._on_files_dropped)
        self._timeline.media_dropped.connect(self._on_media_dropped)
        self._media_pool.set_thumbnail_source(self._pool_thumbnail)
        self._media_pool.set_view_mode(self._preferences.media_view)
        self._media_pool.view_mode_changed.connect(self._on_pool_view_changed)

    def _on_files_dropped(self, paths: list[Path], frame: int, track_id: str) -> None:
        """エクスプローラーからタイムラインへ落とされた 一覧への落とし込みと同じく裏で調べる

        プロジェクトを新しく作らなくても落とせる 起動した時点で空のプロジェクトが
        開いていて（:class:`Project` の既定）、トラックが無ければ置くときに作る
        """
        spot = DropSpot(frame, TrackId(track_id) if track_id else None)
        self.import_media(list(paths), at=spot)

    def _on_media_dropped(self, media_ids: list[str], frame: int, track_id: str) -> None:
        """素材一覧からタイムラインへ落とされた 落とした所へ置き、1 回の取り消しで戻す"""
        project = self.view_project
        media = [
            item for key in media_ids if (item := project.find_media(MediaId(key))) is not None
        ]
        if not media:
            return
        # 落とした位置は合わせる前のレートで数えてある 同じ時刻へ置くよう数え直す
        before = project.rate
        self._match_project_to(media)
        project = self.view_project
        commands = place_media(
            project,
            media,
            at_frame=retime_frame(frame, before, project.rate),
            track_id=TrackId(track_id) if track_id else None,
            split_audio=self._preferences.splits_media,
        )
        label = f"配置: {media[0].name}" if len(media) == 1 else f"配置: {len(media)} 件"
        self.execute_all(commands, label)

    def _scene_project(self, scene: SceneId | None) -> Project | None:
        """そのシーン（``None`` ならメイン）のタイムラインを差し込んだプロジェクト

        :attr:`view_project` と違い、いま開いているシーンではなく指定のシーンを見る
        シーンが消えていれば ``None``
        """
        root = self._document.project
        if scene is None:
            return root
        found = root.find_scene(scene)
        return None if found is None else replace(root, timeline=found.timeline)

    def _execute_in_scene(self, commands: list[Command], label: str, scene: SceneId | None) -> bool:
        """:meth:`execute_all` と同じだが、開いているシーンではなく ``scene`` の中で実行する

        裏で調べ終えてから置く読み込みのためのもの 調べている間にシーンを切り替えられると、
        開いているシーンで包む :meth:`execute_all` では切り替えた先へ入ってしまう

        包めるときは :meth:`execute_all` を通す 入口を 1 つに保つため（シーンで包んだ
        コマンドは :meth:`_in_active_scene` が包み直さない） 自前で実行するのは、
        シーンを開いている間にメインへ置くときだけ（メインへ出る包みが無い）
        """
        if scene == self._active_scene:
            return self.execute_all(commands, label)
        if scene is not None:
            return self.execute_all(
                [c if isinstance(c, InScene) else InScene(scene, c) for c in commands], label
            )
        if not commands:
            return True
        try:
            with self._document.checkpoint(label):
                for command in commands:
                    self._document.execute(command)
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
            self._on_project_changed()
            return False
        self._on_project_changed()
        return True

    def _place_dropped(
        self, project: Project, media: MediaItem, spot: DropSpot | None
    ) -> list[Command]:
        """読み込んだ素材 1 本を置くコマンド 落とされた位置が無ければ末尾へ並べる"""
        split = self._preferences.splits_media
        if spot is None:
            return insert_media(project, media, at_frame=None, split_audio=split)
        return place_media(
            project, [media], at_frame=spot.frame, track_id=spot.track_id, split_audio=split
        )

    def _pool_thumbnail(self, media: MediaItem) -> np.ndarray | None:
        """素材一覧の行の頭に出す 1 コマ タイムラインの絵の並びから借りる

        素材一覧のために別に素材を開かない 絵の並びは読み込んだときに裏で作っていて、
        同じ素材をもう一度デコードすると、読み込み直後の裏の仕事が倍になる
        """
        strip = self._analyzer.filmstrip(media)
        return strip.at(POOL_THUMBNAIL_SECONDS) if strip is not None else None

    def _on_pool_view_changed(self, mode: str) -> None:
        """一覧の上のボタンで表示を切り替えた 好みの設定に書いて、次に開いたときも同じにする"""
        self._remember_preferences(replace(self._preferences, media_view=mode))

    def _set_snap(self, enabled: bool) -> None:
        """タイムラインの上の〔磁石〕 次に開いたときも同じにする"""
        self._timeline.set_snap(enabled, self._preferences.snap_distance)
        self._remember_preferences(replace(self._preferences, timeline_snap=enabled))

    def _remember_preferences(self, preferences: Preferences) -> None:
        """設定画面の外で選んだ好みを覚える 次に開いたときも同じにする"""
        self._preferences = preferences
        try:
            PreferenceStore().save(self._preferences)
        except OSError as exc:
            self.statusBar().showMessage(f"設定を保存できなかった: {exc}", 5000)

    def _insert_media_by_id(self, media_id: str) -> None:
        project = self.view_project
        media = project.find_media(MediaId(media_id))
        if media is None:
            return
        self._match_project_to([media])
        self.execute_all(
            insert_media(self.view_project, media, split_audio=self._preferences.splits_media),
            f"配置: {media.name}",
        )

    def _match_project_to(self, media: list[MediaItem]) -> None:
        """空のプロジェクトへ最初の動画を置く前に、プロジェクトを動画の形へ合わせる

        合わせるかは設定（:attr:`Preferences.match_video`）で決まる（:mod:`sashimono.ui.media_match`）
        置くのとは別の取り消しの段にする 置いた後に戻したくなるのは、たいてい置き方の方で、
        合わせた形まで一緒に戻ると、次に置いたときにまた尋ねられる

        開いているシーンで包まずにプロジェクト全体へ当てる フレームレートはメインと
        全部のシーンで同じでなければならず、シーンの中だけを変えることはできない
        置く位置のフレームは、合わせた後のプロジェクトで数え直す（呼び出し側）
        """
        commands = media_match.commands_to_match(
            self, self._preferences.match_video, self._document.project, media
        )
        if not commands:
            return
        try:
            with self._document.checkpoint("プロジェクトを動画に合わせる"):
                for command in commands:
                    self._document.execute(command)
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
        self._on_project_changed()

    def _on_analysis_ready(self, media_id: MediaId) -> None:
        # ワーカースレッドから呼ばれる ここでウィジェットに触ると Qt が落ちるので、
        # 印だけ付けてメインスレッドのタイマーに描き直させる
        del media_id
        self._analysis_dirty = True

    def _request_proxy(self, media: MediaItem) -> None:
        """控えを作るよう頼む 設定で切っていれば何もしない

        素材を読み込む道は何本もある（普通に開く・復元する・AviUtl から
        取り込む・AI から） ここを 1 つにまとめておかないと、道ごとに
        書き分けることになる
        """
        if self._preferences.use_proxy:
            self._proxies.request(media, on_ready=self._on_proxy_ready)

    def _on_proxy_ready(self, media_id: MediaId) -> None:
        """控えができた ワーカースレッドから呼ばれる

        ウィジェットには触らず、どの素材かだけを覚える 次の間隔で、
        その素材のデコーダを開き直させる（開いたままだと元のファイルを
        掴み続けるので、描き直すだけでは控えに変わらない）
        """
        with self._proxied_lock:
            self._proxied.add(media_id)

    def _flush_analysis(self) -> None:
        # AI から始めた起こしの様子も、ついでにここで拾う 専用のタイマーを
        # もう 1 本増やすほどの頻度ではない
        self._subtitles.poll_transcription()
        self._show_background_progress()
        # 控えができた 開きっぱなしのデコーダは元のファイルを掴んだままなので、
        # 開き直させる（描き直すだけでは切り替わらない）
        # できた素材のぶんだけにする 全部開き直すと、別の素材の控えが
        # できるたびに再生中のクリップまでシークし直すことになる
        with self._proxied_lock:
            ready, self._proxied = self._proxied, set()
        if ready:
            self._preview.reload_sources(ready)
        # 使えない控えを捨てた素材は、作り直しを頼む 頼まないと、その回だけでなく
        # そのあとずっと元の素材を読み続ける（置き場には何も無いままなので、
        # 次に開いたときも作られない）
        for media_id in self._preview.take_discarded():
            media = self.view_project.find_media(media_id)
            if media is None:
                continue
            key = self._proxies.store.key_for(media)
            if key in self._rebuilt:
                # 作り直した控えがまた使えなかった 頼み続けると、250ms ごとに
                # 変換が走り続けて編集そのものが重くなる あきらめて元の素材で映す
                continue
            self._rebuilt.add(key)
            self._request_proxy(media)
        if not self._analysis_dirty:
            return
        self._analysis_dirty = False
        self._media_pool.refresh_thumbnails()
        self._timeline.update()

    def _show_background_progress(self) -> None:
        """控えと解析の進み具合を、ステータスバーと素材一覧の行へ出す

        ひと続きの数を 0 へ戻すのもここ 控えと解析の**両方**が止まってから戻す
        片方ずつ戻すと、解析が走っている間に控えの失敗の数が消え、全体の割合も巻き戻る
        """
        proxy = self._proxies.poll()
        analysis = self._analyzer.poll()
        self._media_pool.set_progress(
            row_notes(proxy, analysis) if self._preferences.pool_progress else {}
        )
        if proxy.busy or analysis.busy:
            self._background_indicator.show_progress(
                describe_background(proxy, analysis),
                overall_fraction(proxy, analysis),
                tooltip="\n".join([*proxy.failures.values(), *analysis.failures.values()]),
            )
            self._background_shown = True
            return
        self._background_indicator.hide()
        # 終わったことを知らせるのは、進み具合を出していたときと失敗したときだけ
        # キャッシュから一瞬で済んだ分まで知らせると、開くたびに文言が出る
        failed = proxy.failed or analysis.failed
        if self._background_shown or failed:
            self.statusBar().showMessage(
                finished_message(proxy, analysis), 8000 if failed else 4000
            )
        self._background_shown = False
        # 知らせた分を数え直す 戻さないと、次のひと続きが前の本数と失敗を抱えたまま始まる
        self._proxies.settle(proxy)
        self._analyzer.settle(analysis)

    # --- 再生とシーク ---

    def _seek(self, frame: int) -> None:
        frame = max(0, min(frame, self.view_project.duration))
        self._timeline.set_playhead(frame)
        self._show_frame(frame)
        self._playback.set_frame(frame)

    def _show_frame(self, frame: int) -> None:
        self._preview.set_frame(frame)
        self._transport.set_frame(frame)
        self._inspector.set_frame(frame)
        self._graph.set_frame(frame)
        self._subtitles.set_frame(frame)

    def _on_selection_changed(self, clip_id: str) -> None:
        selected = ClipId(clip_id) if clip_id else None
        # 何本も選んでいれば、設定パネルは主の 1 本を出しつつ、触った設定を全部へ当てる
        # グループの仲間として引き込まれただけの物には当てない（AviUtl のグループ化と同じ
        # 1 本の拡大率を変えただけで束ねた全部の拡大率が変わっていた）
        chosen = self._timeline.edit_targets
        ordered = (selected, *(c for c in chosen if c != selected)) if selected else ()
        self._inspector.set_selection(tuple(c for c in ordered if c is not None))
        self._preview.set_selection(selected)
        # グラフエディタも選んだクリップに付いていく 付いていかないと、キーフレームを入れた
        # クリップを選んでもグラフエディタが何も出さず、◆ の右クリックの奥からしか開けない
        self._graph.set_clip(selected)
        # 字幕パネルも選んだクリップの素材に付いていく（起こすときに別の素材を開かない）
        sound = self._selected_sound()
        if sound is not None:
            self._subtitles.follow_clip(*sound)

    def _align_selected(self, anchor: str) -> None:
        """設定パネルの配置のテンプレート 選んだクリップを画面のその所へ寄せる

        大きさは描く側の枠から取るので、今のコマに映っていない物は寄せられない
        黙って何もしないと、押しても効かない理由が分からない
        """
        if not self._preview.align_selected(anchor):
            self.statusBar().showMessage(
                "再生位置にこのクリップが映っていないので揃えられません", 4000
            )

    def _show_curve(self, path: ParamPath) -> None:
        self._graph.set_path(path)
        self._graph_dock.show()
        self._graph_dock.raise_()

    def _preview_command(self, command: Command) -> None:
        """履歴に残さず、プレビューだけ更新する

        スライダーのドラッグ中に呼ばれる 1 回のドラッグで数十の取り消し段を
        作らないための逃げ道で、指を離した時点で本来のコマンドが飛んでくる
        """
        self._preview_commands([command])

    def _preview_commands(self, commands: list[Command]) -> None:
        """いくつかの命令を順に当てた絵を、履歴に残さずプレビューへ出す 空なら元へ戻す

        プレビューで掴んでいる途中に呼ばれる 配置を持たない古いクリップでは、配置を
        足す命令と値を入れる命令が組で届く
        """
        preview = self.view_project
        try:
            for command in commands:
                preview = command.apply(preview)
        except (ValueError, KeyError):
            return
        self._preview.set_project(preview)
        self._preview.update()

    def _on_playhead_moved(self, frame: int) -> None:
        self._show_frame(frame)
        self._playback.set_frame(frame)

    def _on_playback_frame(self, frame: int) -> None:
        self._timeline.set_playhead(frame)
        self._show_frame(frame)

    # --- ファイル ---

    def new_project(self) -> None:
        """解像度とフレームレートを尋ねてから作る フレームレートはあとで変えられない"""
        from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

        if not self._confirm_discard():
            return
        dialog = ProjectSettingsDialog(self._new_settings(), self, new=True)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._playback.stop()
        previous = self._document.project
        self._document.reset(Project.create(dialog.settings()))
        self._leave_project(previous)
        self._path = None
        self._release_lock()
        self._mark_saved()
        self._on_project_changed()
        self._seek(0)

    def _new_settings(self) -> ProjectSettings:
        """新しく作るプロジェクトの初めの設定 方式は本人の好み

        モデルの既定（分ける）は変えない 古いファイルを開いたときと、既定の設定で
        組み立てる試験の動きを保つため 画面から新しく作るときだけ好みを当てる
        """
        return ProjectSettings(layer_mode=self._preferences.new_project_layers)

    def _blank_project(self) -> Project:
        """起動した直後や、開くのをやめたときの空のプロジェクト"""
        return Project.create(self._new_settings())

    def _mark_saved(self) -> None:
        """いまの状態を「保存済み」とする 守るものが無くなるので退避も消す"""
        self._saved = self._document.project
        self._autosaved = self._saved
        self._recovery.clear()

    def _confirm_discard(self) -> bool:
        """変更を捨ててよいか 保存を選べば保存してから真を返す"""
        if not self._confirm_unsaved or not self.is_modified:
            return True
        answer = QMessageBox.question(
            self,
            "保存していない変更",
            "変更を保存しますか",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        # 改名前の版で保存したもの（旧い拡張子）も開けるようにしておく
        patterns = " ".join(f"*{s}" for s in (SUFFIX, *LEGACY_SUFFIXES))
        name, _ = QFileDialog.getOpenFileName(
            self, "プロジェクトを開く", "", f"Sashimono Edit プロジェクト ({patterns})"
        )
        if not name:
            return
        try:
            project = load_project(Path(name))
        except ProjectFileError as exc:
            QMessageBox.warning(self, "開けない", str(exc))
            return
        if not self._claim(Path(name)):
            return

        self._playback.stop()
        previous = self._document.project
        self._document.reset(project)
        self._leave_project(previous)
        self._path = Path(name)
        self._mark_saved()
        self._on_project_changed()
        self._seek(0)
        for media in project.media:
            self._analyzer.request(media, on_ready=self._on_analysis_ready)
            self._request_proxy(media)

    def save_project(self) -> bool:
        """保存する 保存できたら真 名前がまだ無ければ尋ねる"""
        if self._path is None:
            return self.save_as()
        project = self._document.project
        note = ""
        try:
            backup_before_save(self._path)
        except OSError as exc:
            # 控えが取れなくても保存は止めない 止めると、控えのために
            # いまの作業のほうを失う
            note = f"（バックアップは作れなかった: {exc}）"
        try:
            save_project(project, self._path)
        except OSError as exc:
            QMessageBox.warning(self, "保存できない", f"{self._path}\n{exc}")
            return False
        self._mark_saved()
        self._update_title()
        self.statusBar().showMessage(f"保存した: {self._path}{note}", 5000 if note else 3000)
        return True

    def save_as(self) -> bool:
        suggested = self._path or Path(f"{self._document.project.name}{SUFFIX}")
        name, _ = QFileDialog.getSaveFileName(
            self, "名前を付けて保存", str(suggested), f"Sashimono Edit プロジェクト (*{SUFFIX})"
        )
        if not name:
            return False
        # 保存できたときだけ新しい名前に切り替える 先に切り替えると、失敗しても
        # タイトル・次の保存先・退避のメモが、書けなかった場所を指したままになる
        previous = self._path
        if not self._claim(Path(name)):
            return False
        self._path = Path(name)
        saved = self.save_project()
        if not saved:
            # 元の名前へ戻すのは、元の錠を取り直せたときだけ 取れないまま戻すと、
            # 錠は新しい名前、保存先は元の名前、と食い違う
            if previous is None:
                self._release_lock()
                self._path = None
            elif self._claim(previous):
                self._path = previous
        self._update_title()
        return saved

    def _claim(self, path: Path) -> bool:
        """このファイルを開いている窓はここ、と錠で示す 別の窓が開いていれば尋ねる

        止めはしない 読み返すだけのこともあるので、知らせたうえで本人に選ばせる
        知らせずに開けると、両方で保存したとき後から保存した方が黙って勝つ
        """
        folder = project_presence_dir(path)
        if self._project_lock is not None and self._project_lock.path.parent == folder:
            return True
        if others_holding(folder) and self._confirm_unsaved:
            answer = QMessageBox.warning(
                self,
                "別の窓で開いています",
                f"{path.name} は別の Sashimono の窓で開かれています\n"
                "両方で保存すると、あとから保存した方の内容だけが残ります",
                QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Open:
                return False
        # 「それでも開く」でも自分の錠は置く 置かないと、先の窓が閉じたあとに
        # 開いた窓からこの窓が見えない
        self._release_lock()
        self._project_lock = hold_new(folder)
        return True

    def _release_lock(self) -> None:
        if self._project_lock is not None:
            self._project_lock.release()
            self._project_lock = None

    def open_backup_folder(self) -> None:
        """控えは %LOCALAPPDATA% の奥にあり、場所を知らないと辿り着けない"""
        if self._path is None:
            self.statusBar().showMessage("まだ保存していないので、バックアップはありません", 5000)
            return
        folder = backup_folder(self._path)
        if not folder.is_dir():
            self.statusBar().showMessage("バックアップは上書き保存したときに作られます", 5000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def edit_settings(self) -> None:
        """プロジェクト設定を開く 解像度と重ね合わせの方法を変えられる"""
        from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

        settings = self._document.project.settings
        dialog = ProjectSettingsDialog(settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.settings()
        commands: list[Command] = []
        if chosen.resolution != settings.resolution:
            commands.append(SetResolution(chosen.width, chosen.height))
        if chosen.blending != settings.blending:
            commands.append(SetBlending(chosen.blending))
        # 1 回の OK で変えたものは 1 回の取り消しで戻す 分けると、取り消しの途中で
        # 解像度だけ戻った、見たことのない組み合わせを通る
        self.execute_all(commands, "プロジェクト設定を変更")

    def switch_layer_mode(self) -> bool:
        """置き方の方式を、今と逆の方式へ切り替える 置いてあるトラックも変換するかを毎回尋ねる

        変換はメインとすべてのシーンに掛かるので、開いているシーンの中へ包まずに実行する
        （:meth:`execute_all` は包む） 包むと、開いているシーンだけが変わり、メインは
        前の方式のまま残る 方式と変換は 1 回の取り消しで戻す

        変えたら真 やめたとき・断られたときは偽
        """
        from sashimono.ui.layer_mode_dialog import LayerModeDialog

        project = self._document.project
        target = (
            LayerMode.SEPARATED
            if project.settings.layer_mode == LayerMode.MIXED
            else LayerMode.MIXED
        )
        sound_kinds = effect_registry.sound_kinds()
        conversion = convert_layers(project, target, sound_kinds)
        dialog = LayerModeDialog(
            target, conversion.notices, self, convertible=conversion.project is not project
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.convert is None:
            return False
        commands = switch_layer_mode(
            project, target, convert=dialog.convert, sound_kinds=sound_kinds
        )
        label = "置き方の方式を切り替えて変換" if dialog.convert else "置き方の方式を切り替え"
        try:
            with self._document.checkpoint(label):
                for command in commands:
                    self._document.execute(command)
        except (ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
            self._on_project_changed()
            return False
        self._on_project_changed()
        return True

    # --- 退避と復元 ---

    def autosave(self) -> None:
        """保存していない変更を退避する タイマーから呼ばれる

        前回から変わっていなければ書かない 放置しているあいだ 30 秒ごとに
        同じ中身を書き直すのは、ディスクを傷めるだけで何も守らない
        """
        project = self._document.project
        if project is self._autosaved:
            return
        try:
            if self.is_modified:
                self._recovery.save(project, self._path)
            else:
                self._recovery.clear()
        except OSError as exc:
            self.statusBar().showMessage(f"自動退避に失敗した: {exc}", 5000)
            return
        self._autosaved = project

    def offer_recovery(self) -> None:
        """前回落ちた作業が残っていれば、復元するか尋ねる 起動の直後に呼ぶ"""
        from sashimono.ui.recovery_dialog import RecoveryDialog

        while entries := find_orphans():
            dialog = RecoveryDialog(entries, self)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.choice is None:
                return
            action, entry = dialog.choice
            if action == "discard":
                discard(entry)
                continue
            self.restore_recovery(entry)
            return

    def restore_recovery(self, entry: RecoveryEntry) -> bool:
        """退避を開く 保存はしないので、開いた直後は「変更あり」になる

        元の退避は、この起動の退避へ書き写してから捨てる 先に捨てると、
        書き写す前に落ちたときに何も残らない
        """
        try:
            project = load_project(entry.path)
        except ProjectFileError as exc:
            QMessageBox.warning(self, "復元できない", str(exc))
            return False
        # load_project は「無題」をファイル名で置き換える 退避のファイル名は
        # 意味の無い英数字なので、退避したときの名前へ戻す
        project = project.renamed(entry.name)
        if entry.source is not None and not self._claim(entry.source):
            return False

        self._playback.stop()
        previous = self._document.project
        self._document.reset(project)
        self._leave_project(previous)
        self._path = entry.source
        self._saved = None
        self._on_project_changed()
        self._seek(0)
        for media in project.media:
            self._analyzer.request(media, on_ready=self._on_analysis_ready)
            self._request_proxy(media)

        self.autosave()
        if self._autosaved is project:
            discard(entry)
        self.statusBar().showMessage("前回の作業を復元した まだ保存していません", 6000)
        return True

    def export(self) -> None:
        self._playback.stop()
        ExportDialog(
            self._document.project,
            self,
            pipeline_depth=self._preferences.export_pipeline_depth,
            decode_threads=self._preferences.decode_threads,
            scene_name=self._active_scene_name(),
            smooth_history=self._preferences.smooth_audio_motion,
        ).exec()

    def _active_scene_name(self) -> str | None:
        """開いているシーンの名前 メインなら ``None``"""
        if self._active_scene is None:
            return None
        scene = self._document.project.find_scene(self._active_scene)
        return scene.name if scene is not None else None

    # --- AviUtl 互換 ---

    def import_exo(self) -> None:
        """``.exo`` / ``.exa`` をタイムラインへ読み込む

        参照している素材は先に読み込んでから対応付ける 素材が見つからなくても
        止めない テキストや図形だけでも入る方が使い出がある
        """
        from sashimono.compat.aviutl.exo import ExoParseError, load_exo
        from sashimono.compat.aviutl.mapping import map_exo

        name, _ = QFileDialog.getOpenFileName(
            self, "AviUtl のオブジェクトを読み込む", "", EXO_FILTER
        )
        if not name:
            return

        source = Path(name)
        try:
            exo = load_exo(source)
        except ExoParseError as exc:
            QMessageBox.warning(self, "読み込めない", str(exc))
            return

        found = self._resolve_exo_media(exo, source)
        commands = map_exo(exo, self.view_project, media=found.ids, items=found.items)
        if not commands:
            self.statusBar().showMessage("読み込めるオブジェクトがありませんでした", 5000)
            return

        # 素材の登録と配置を 1 回の Undo にまとめる 分けると、配置を断られたときに
        # 使われていない素材だけが一覧に残る
        if not self.execute_all(
            [*found.commands, *commands], f"AviUtl から読み込み: {source.name}"
        ):
            # 断られた理由は execute_all がステータスバーに出している 上書きしない
            return
        # 解析と控えは取り消しても止まらない 入ったことを確かめてから頼む
        for media in found.items:
            self._analyzer.request(media, on_ready=self._on_analysis_ready)
            self._request_proxy(media)
        note = f"{source.name} から {len(exo.objects)} 個を読み込んだ"
        if found.missing:
            note += f"（素材 {len(found.missing)} 件が見つかりません）"
        self.statusBar().showMessage(note, 6000)

    def _resolve_exo_media(self, exo: ExoFile, source: Path) -> _ExoMedia:
        """``.exo`` が参照している素材を読む 一覧へ入れるのは呼んだ側

        相対パスは ``.exo`` のある場所からも探す AviUtl のファイルは素材と
        一緒に配られることがある
        """
        from sashimono.compat.aviutl.mapping import media_paths

        ids: dict[str, MediaId] = {}
        missing: list[str] = []
        items: list[MediaItem] = []
        for raw in media_paths(exo):
            candidates = [Path(raw), source.parent / Path(raw).name]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                missing.append(raw)
                continue
            forget_probe(path)
            try:
                media = probe_media(path)
            except ProbeError:
                missing.append(raw)
                continue
            items.append(media)
            ids[raw] = media.id
        return _ExoMedia(
            ids=ids,
            missing=tuple(missing),
            commands=tuple(AddMedia(media) for media in items),
            items=tuple(items),
        )

    def show_templates(self) -> None:
        """テンプレートの棚を開いて、選ばれたものを反映する

        「置く」と「着せる」で行き先が違うだけで、どちらも 1 回の Undo で戻る
        """
        from sashimono.compat.catalog import restyle
        from sashimono.ui.template_dialog import TemplateDialog

        dialog = TemplateDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.choice is None:
            return

        action, objects = dialog.choice
        if action == "restyle":
            clip_id = self.selected_clip
            located = (
                self.view_project.timeline.locate_clip(clip_id) if clip_id is not None else None
            )
            if located is None:
                self.statusBar().showMessage("先にテキストのクリップを選んでください", 5000)
                return
            commands = restyle(objects, located[1])
            if not commands:
                self.statusBar().showMessage("テキストのクリップにしか適用できません", 5000)
                return
            if not self.execute_all(commands, "テンプレートを適用"):
                # 断られた理由は execute_all がステータスバーに出している 上書きしない
                return
            self.statusBar().showMessage("テンプレートを適用した（文字と長さはそのまま）", 5000)
            return

        self._place_template(objects, dialog.origin, self._timeline.playhead)

    def place_template_entry(self, entry: object, frame: int, track_id: str = "") -> bool:
        """棚のテンプレートを ``frame`` へ置く タイムラインの右クリックの〔追加〕から

        ``track_id`` は右クリックしたトラック 置く物が 1 つでそこが空いていればそこへ、
        そうでなければ棚のダイアログから置くときと同じく元のレイヤーの並びで置く
        何個もの物を 1 本のトラックへ置くと、重なった所で断られる
        """
        from sashimono.compat.catalog import TemplateEntry, TemplateError

        if not isinstance(entry, TemplateEntry):
            return False
        try:
            objects = entry.load()
        except (*TemplateError, OSError) as exc:
            self.statusBar().showMessage(f"{entry.label} を読み込めません: {exc}", 6000)
            return False
        return self._place_template(
            objects, entry.path.parent, frame, TrackId(track_id) if track_id else None
        )

    def _place_template(
        self,
        objects: list[MappedObject],
        near: Path | None,
        frame: int,
        track_id: TrackId | None = None,
    ) -> bool:
        from sashimono.compat.catalog import gather_media, place

        # 画像・音声のアイテムは素材として登録してからクリップに結ぶ 結ばないと、
        # 置いたクリップは描かれず鳴らない（素材の無いクリップになる）
        project = self.view_project
        plan = gather_media(objects, project, _probe_or_none, near=near)
        pictures = [item for item in objects if item.has_picture]
        target = None
        if track_id is not None and len(pictures) == 1 and not pictures[0].children:
            track = project.timeline.find_track(track_id)
            clip = pictures[0].clip
            end = frame + (clip.duration if pictures[0].has_span else DEFAULT_GENERATED_FRAMES)
            free = track is not None and not any(c.overlaps(frame, end) for c in track.clips)
            # レイヤー（混合トラック）も受ける 映像トラックに限ると、混合の方式では
            # 右クリックしたレイヤーを無視して元のレイヤー番号の所へ入る
            placeable = (TrackKind.VIDEO, TrackKind.MIXED)
            if free and track is not None and track.kind in placeable and not track.locked:
                target = track_id
        commands = place(objects, project, at_frame=frame, track_id=target, media=plan.media)
        if not commands:
            self.statusBar().showMessage("置けるオブジェクトがありませんでした", 5000)
            return False
        # 素材の登録と配置を 1 回の Undo にまとめる 分けると、戻したときに
        # 使われていない素材だけが一覧に残る
        if not self.execute_all([*plan.commands, *commands], "テンプレートを配置"):
            return False
        # 置いた物を選ぶ 右クリックのほかの〔追加〕と同じく、設定パネルがすぐ開く
        added = [c.clip.id for c in commands if isinstance(c, AddClip)]
        landed = [c for c in added if self.view_project.timeline.locate_clip(c) is not None]
        if landed:
            self._timeline.set_selection(landed)
        for media in plan.added:
            self._analyzer.request(media, on_ready=self._on_analysis_ready)
            self._request_proxy(media)
        # 数えるのは一番上に置いたクリップだけ トラックやシーンを足すコマンドまで
        # 数えると、画像 1 つでも「2 個を置いた」と出る
        placed = sum(isinstance(command, AddClip) for command in commands)
        note = f"{placed} 個を置いた"
        if plan.missing:
            # 見つからないものと、見つかっても開けなかったものの両方を数えている
            note += f"（素材 {len(plan.missing)} 件が見つからないか開けません）"
        self.statusBar().showMessage(note, 6000)
        return True

    def rescan_scripts(self) -> None:
        """スクリプトのフォルダを読み直す"""
        from sashimono.compat.aviutl.catalog import script_catalog

        catalog = script_catalog()
        catalog.scan()
        count = catalog.register_all()
        self.statusBar().showMessage(f"スクリプトを {count} 本読み込んだ", 4000)

    def open_script_folder(self) -> None:
        """スクリプトを置く場所をエクスプローラで開く

        開くのは ``%APPDATA%\\Sashimono\\scripts``（:func:`userdirs.config_root` の下） 配布版の
        ``Sashimono.exe`` の隣の ``scripts`` も読むが、新しい版の zip でフォルダごと入れ替えると
        中身が消える（Issue #138） 置き場として案内するのは、入れ替えても残る側にする
        ``%APPDATA%`` は隠しフォルダで見つけにくいが、ここから開けばその心配は無い
        """
        target = userdirs.config_root() / "scripts"
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.statusBar().showMessage(f"スクリプトフォルダを作れなかった: {exc}", 5000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def show_compatibility(self) -> None:
        """互換性レポートを出す"""
        from sashimono.ui.compat_dialog import CompatibilityDialog

        CompatibilityDialog(parent=self).exec()

    # --- ヘルプ ---

    def open_manual(self) -> None:
        """使い方をブラウザで開く"""
        self._open_link(MANUAL_URL)

    def open_report_page(self) -> None:
        """不具合・要望の受け口をブラウザで開く"""
        self._open_link(REPORT_URL)

    def _open_link(self, url: str) -> None:
        # 既定のブラウザが決まっていない機械では開けない 黙ると押しても何も
        # 起きないように見えるので、打ち込めるように URL を出しておく
        if not QDesktopServices.openUrl(QUrl(url)):
            self.statusBar().showMessage(f"ブラウザを開けませんでした: {url}", 10000)

    def show_about(self) -> None:
        """版と置き場を出す 不具合の報告で最初に聞くことを 1 か所で見られるようにする"""
        QMessageBox.about(self, "バージョン情報", about_text())

    # --- AI 連携（EditorHost の実装）---
    #
    # AI からの操作も UI と同じ入口を通す ここが増えないようにしておけば、
    # 「UI ではできるが AI ではできない」も、その逆も生まれない

    @property
    def document(self) -> Document:
        return self._document

    @property
    def playhead(self) -> int:
        return self._timeline.playhead

    def seek(self, frame: int) -> None:
        self._seek(frame)

    @property
    def selected_clip(self) -> ClipId | None:
        return self._timeline.selected_clip

    def select_clip(self, clip_id: ClipId | None) -> None:
        self._timeline.select(clip_id)

    @property
    def selected_clips(self) -> tuple[ClipId, ...]:
        return self._timeline.selected_clips

    def select_clips(self, clip_ids: list[ClipId]) -> None:
        self._timeline.set_selection(clip_ids)

    def apply_commands(self, commands: list[Command], label: str) -> None:
        """AI からのコマンドを実行する

        UI 経由の :meth:`execute_all` と違い、失敗を握り潰さず例外にする
        AI はエラーの文面を読んで次の手を決めるので、黙って何も起きないのが
        いちばん困る
        """
        if not commands:
            return
        try:
            with self._document.checkpoint(label):
                for command in commands:
                    self._document.execute(self._in_active_scene(command))
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc)) from exc
        finally:
            self._on_project_changed()

    @property
    def project(self) -> Project:
        """AI が読むプロジェクト 画面と同じく、開いているシーンを見る"""
        return self.view_project

    @property
    def splits_media(self) -> bool:
        """AI が素材を置くときも、画面から置くときと同じ設定に従う"""
        return self._preferences.splits_media

    def set_active_scene(self, scene_id: SceneId | None) -> None:
        self.open_scene(scene_id)

    def stop_playback(self) -> None:
        self._playback.stop()

    def render_png(self, frame: int, *, width: int) -> bytes:
        """そのフレームを合成して PNG にする

        プレビューのウィジェットとは別のコンテキストで描く 再生用の資源を
        取り合わないようにするためで、代わりに 1 つ余分にコンテキストを持つ

        AI が編集の結果を目で確かめるためのもので、書き出しではない 開いているシーンを
        描く AI の読み取り（``list_clips`` など）もそのシーンが相手なので、メインを描くと
        AI が見ているクリップと絵が食い違う
        """
        project = self.view_project
        full_width = project.settings.width
        divisor = max(1, round(full_width / max(width, 1)))

        renderer = self._ai_renderer
        if renderer is None:
            renderer = FrameRenderer(project, quality=RenderQuality(divisor))
            self._ai_renderer = renderer
        else:
            renderer.set_project(project)
            renderer.set_quality(RenderQuality(divisor))

        image = renderer.render(frame)
        height, image_width = image.shape[0], image.shape[1]
        picture = QImage(
            image.tobytes(), image_width, height, image_width * 4, QImage.Format.Format_RGBA8888
        )
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        # QImage.save の書式引数は、この PySide6 では str しか受け取らない
        # （型情報は bytes だと言う） 食い違いを避けるため QImageWriter を使う
        if not QImageWriter(buffer, b"PNG").write(picture):
            raise ToolError("プレビュー画像を作れませんでした")
        return bytes(buffer.data().data())

    # --- 静止画 ---

    def _connect_preview_menu(self) -> None:
        """プレビューの右クリックに静止画の保存とコピーを出す

        撮りたい絵を見ているのはプレビューなので、メニューバーまで行かずに撮れるようにする
        """
        # メニューは 1 つだけ作って使い回す 開くたびに作ると、閉じても窓の子として残り、
        # 右クリックの回数だけ部品が増えていく
        self._preview_menu = QMenu(self)
        self._preview_menu.addAction(self._snapshot_save_action)
        self._preview_menu.addAction(self._snapshot_copy_action)
        self._preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._preview.customContextMenuRequested.connect(self._show_preview_menu)

    def _show_preview_menu(self, position: QPoint) -> None:
        self._preview_menu.popup(self._preview.mapToGlobal(position))

    def _snapshot_frame(self) -> int:
        """静止画にするフレーム 再生を止めてから再生ヘッドを 1 度だけ読む

        止めずに読むと、保存先を尋ねている間も再生ヘッドが進み、名前に入れた
        タイムコードと描いた絵のフレームが食い違う
        """
        from sashimono.ui.snapshot import snapshot_frame

        self._playback.stop()
        return snapshot_frame(self.view_project, self._timeline.playhead)

    def snapshot_image(self, frame: int | None = None) -> QImage:
        """``frame``（省けば再生ヘッドの位置）を、書き出しと同じ描き方でプロジェクトの解像度のまま描く

        プレビューと同じく開いているシーンを描く 見ている絵と撮れた絵が食い違わないように
        """
        from sashimono.ui.snapshot import render_snapshot

        if frame is None:
            frame = self._snapshot_frame()
        return render_snapshot(self.view_project, frame)

    def save_snapshot(self, path: Path | None = None) -> Path | None:
        """静止画を PNG で保存する ``path`` を省くと保存先を尋ねる 保存した場所を返す"""
        from sashimono.ui.snapshot import SNAPSHOT_FILTER, snapshot_name, write_png

        # 名前と絵の両方にこのフレームを使う 尋ねる前に決めておく
        frame = self._snapshot_frame()
        project = self.view_project
        if path is None:
            # 名前はメインのプロジェクト名で付ける シーンの中にいても、どの作品の絵かが分かる
            name = snapshot_name(replace(project, name=self._document.project.name), frame)
            folder = self._path.parent if self._path is not None else Path.home()
            chosen, _ = QFileDialog.getSaveFileName(
                self, "静止画を保存", str(folder / name), SNAPSHOT_FILTER
            )
            if not chosen:
                return None
            path = Path(chosen)
            if path.suffix.lower() != ".png":
                path = path.with_name(path.name + ".png")
        if not write_png(self.snapshot_image(frame), path):
            QMessageBox.warning(self, "静止画を保存", f"保存できなかった: {path}")
            return None
        self.statusBar().showMessage(f"静止画を保存した: {path}", 5000)
        return path

    def copy_snapshot(self) -> None:
        """静止画をクリップボードへ置く"""
        from sashimono.ui.snapshot import copy_to_clipboard

        image = self.snapshot_image()
        copy_to_clipboard(image)
        self.statusBar().showMessage(
            f"静止画をクリップボードへコピーした（{image.width()}x{image.height()}）", 5000
        )

    def probe(self, path: Path) -> MediaItem:
        forget_probe(path)
        try:
            return probe_media(path)
        except ProbeError as exc:
            raise ToolError(str(exc)) from exc

    def analyze(self, media: MediaItem) -> None:
        self._analyzer.request(media, on_ready=self._on_analysis_ready)
        self._request_proxy(media)

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        return self._analyzer.waveform(media, stream)

    def start_transcription(
        self, media_id: MediaId, model: str, *, audio_stream: int | None = None
    ) -> str:
        return self._subtitles.start_transcription(media_id, model, audio_stream=audio_stream)

    def transcription_status(self) -> str:
        return self._subtitles.transcription_status()

    def show_chat(self) -> None:
        """AI パネルを前へ出す"""
        self._chat_dock.show()
        self._chat_dock.raise_()

    # --- 終了 ---

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if not self._confirm_discard():
            event.ignore()
            return
        # 並びを保存できなくても終了は止めない 次の起動が既定の並びになるだけ
        with contextlib.suppress(OSError):
            self._workspace.save(self)
        # ここまで来たら変更は保存したか、捨てると決めたもの 退避は要らない
        self._autosave_timer.stop()
        self._recovery.close()
        self._release_lock()

        # 解放の順番が大事 GL 資源はコンテキストが生きているうちに、
        # 再生スレッドはウィジェットが消える前に畳む
        self._refresh_timer.stop()
        # 調べている最中の読み込みは捨てる 閉じた窓へ置きに来させない
        self._drop_imports()
        self._chat.close_session()
        self._playback.close()
        if self._ai_renderer is not None:
            self._ai_renderer.close()
            self._ai_renderer = None
        self._analyzer.close()
        self._proxies.close()
        self._preview.shutdown()
        super().closeEvent(event)
