"""画面配置とショートカットの保存先

どちらも「その人の使い方」で、プロジェクトではなく本人に付く ``%APPDATA%\\Sashimono``
に置くのはそのため（キャッシュや退避と違い、消えると作り直せない）

Qt の既定の保存先（Windows ではレジストリ）は使わない 自動更新で本体を入れ替えても
残ること、手で消したり他の機械へ持っていったりできることを優先して、ファイルにする
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtWidgets import QMainWindow, QTabWidget

from sashimono.ai.budget import (
    DEFAULT_DAY_IMAGES,
    DEFAULT_DAY_TOKENS,
    DEFAULT_IMAGE_EDGE,
    DEFAULT_TURN_IMAGES,
    DEFAULT_TURN_MINUTES,
    DEFAULT_TURN_TOKENS,
    IMAGE_EDGES,
    LimitAction,
    Limits,
)
from sashimono.ai.models import DEFAULT_EFFORT as DEFAULT_AI_EFFORT
from sashimono.ai.models import DEFAULT_MODEL as DEFAULT_AI_MODEL
from sashimono.ai.models import EFFORTS as AI_EFFORTS
from sashimono.ai.models import MODELS as AI_MODELS
from sashimono.ai.outbound import Outbound, SendChoice
from sashimono.compat.subtitle_looks import LookRules
from sashimono.core import userdirs
from sashimono.core.commands.preset import PresetOptions
from sashimono.core.emphasis import (
    DEFAULT_EMPHASIS_EASING,
    DEFAULT_EMPHASIS_PAN_PX,
    DEFAULT_EMPHASIS_POP_PERCENT,
    DEFAULT_EMPHASIS_RAMP_MS,
    DEFAULT_EMPHASIS_SHAKE_PX,
    DEFAULT_EMPHASIS_ZOOM_PERCENT,
    EMPHASIS_EASINGS,
    EMPHASIS_PAN_PX_RANGE,
    EMPHASIS_POP_PERCENT_RANGE,
    EMPHASIS_RAMP_MS_RANGE,
    EMPHASIS_SHAKE_PX_RANGE,
    EMPHASIS_ZOOM_PERCENT_RANGE,
    EmphasisStyle,
)
from sashimono.core.io.atomic import held_for_update, write_atomic
from sashimono.core.io.serialize import ProjectFileError, style_guide_from_json
from sashimono.core.mixing import (
    DEFAULT_DUCK_ATTACK_MS,
    DEFAULT_DUCK_DEPTH_DB,
    DEFAULT_DUCK_RELEASE_MS,
    DUCK_ATTACK_MS_RANGE,
    DUCK_DEPTH_DB_RANGE,
    DUCK_RELEASE_MS_RANGE,
    DuckShape,
)
from sashimono.core.model import LayerMode, StyleGuide
from sashimono.core.seams import (
    DEFAULT_SEAM_AUDIO_FADE_MS,
    DEFAULT_SEAM_CROSSFADE_MS,
    DEFAULT_SEAM_VIDEO,
    DEFAULT_SEAM_ZOOM_PERCENT,
    SEAM_AUDIO_FADE_MS_RANGE,
    SEAM_CROSSFADE_MS_RANGE,
    SEAM_VIDEOS,
    SEAM_ZOOM_PERCENT_RANGE,
    SeamStyle,
)
from sashimono.engine.cache.analyzer import DEFAULT_SCENE_THREADS
from sashimono.engine.checks import LOUDNESS_TARGETS
from sashimono.engine.encode import DEFAULT_PIPELINE_DEPTH, MAX_PIPELINE_DEPTH
from sashimono.engine.render import DEFAULT_DECODE_THREADS, MAX_DECODE_THREADS
from sashimono.ui.library_thumbnails import THUMBNAIL_MODES, THUMBNAILS_FULL
from sashimono.ui.library_view import BACKDROP_CHECKER, BACKDROP_MODES, LibraryOptions
from sashimono.ui.media_match import MATCH_ASK, MATCH_MODES
from sashimono.ui.media_pool import VIEW_LIST, VIEW_MODES
from sashimono.ui.preview_handles import KEYFRAME_DRAG_AT_PLAYHEAD, KEYFRAME_DRAG_MODES
from sashimono.ui.theme import THEME_DARK, THEME_MODES

__all__ = [
    "AI_IMAGE_LIMITS",
    "AI_MINUTE_LIMITS",
    "AI_TOKEN_LIMITS",
    "AUTO_QUALITY_HEIGHT",
    "DETAIL_MIN_WIDTHS",
    "DOCK_TABS_BOTTOM",
    "DOCK_TABS_TOP",
    "DOCK_TAB_POSITIONS",
    "LAYOUT_VERSION",
    "LIMIT_ACTIONS",
    "MAX_PREFETCH_MB",
    "MEDIA_SPLIT",
    "MEDIA_SPLIT_MODES",
    "MEDIA_TOGETHER",
    "MIN_PREFETCH_MB",
    "SCENES_ALWAYS",
    "SCENES_ON_DEMAND",
    "SCENES_WITH_AI",
    "SCENE_ANALYSIS_MODES",
    "SCENE_THREAD_CHOICES",
    "SCRIPTS_MOVE_ASK",
    "SCRIPTS_MOVE_AUTO",
    "SCRIPTS_MOVE_MODES",
    "SCRIPTS_MOVE_OFF",
    "SEND_ASK",
    "SEND_MODES",
    "SNAP_DISTANCES",
    "PreferenceStore",
    "Preferences",
    "ShortcutStore",
    "Workspace",
    "apply_dock_tabs",
    "config_root",
    "find_conflicts",
]

#: 画面配置の版 パネルを足したり名前を変えたりしたら上げる
#: 古い配置をそのまま当てると、新しいパネルがどこにも出てこない
LAYOUT_VERSION = 1


def config_root() -> Path:
    """本人の設定を置く場所"""
    return userdirs.config_root()


def apply_dock_tabs(window: QMainWindow, position: str) -> None:
    """重ねたパネルのタブを、どの辺に出すかを窓の全部の置き場へ当てる

    置き場（左右上下）ごとに決まるので 4 つとも当てる 1 つだけだと、パネルを
    別の辺へ動かしたときに下のタブへ戻る 画面配置の保存（``saveState``）には入らないので、
    起動のたびに当て直す
    """
    tab = (
        QTabWidget.TabPosition.South
        if position == DOCK_TABS_BOTTOM
        else QTabWidget.TabPosition.North
    )
    for area in (
        Qt.DockWidgetArea.LeftDockWidgetArea,
        Qt.DockWidgetArea.RightDockWidgetArea,
        Qt.DockWidgetArea.TopDockWidgetArea,
        Qt.DockWidgetArea.BottomDockWidgetArea,
    ):
        window.setTabPosition(area, tab)


class Workspace:
    """ウィンドウの大きさと、パネルの並び"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_root() / "workspace.ini"

    def save(self, window: QMainWindow) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        settings = QSettings(str(self.path), QSettings.Format.IniFormat)
        settings.setValue("geometry", window.saveGeometry())
        settings.setValue("state", window.saveState(LAYOUT_VERSION))
        settings.sync()

    def restore(self, window: QMainWindow) -> bool:
        """保存した並びに戻す 無い・版が違うときは何もせず偽を返す"""
        if not self.path.is_file():
            return False
        settings = QSettings(str(self.path), QSettings.Format.IniFormat)
        geometry = settings.value("geometry")
        state = settings.value("state")
        if isinstance(geometry, QByteArray):
            window.restoreGeometry(geometry)
        return isinstance(state, QByteArray) and window.restoreState(state, LAYOUT_VERSION)


class ShortcutStore:
    """既定から変えたショートカットだけを持つ

    全部を書き出さないのは、新しい版で既定を変えたときに、本人が触っていない
    ものまで古い既定のまま固まってしまうため 値が空文字なら「割り当てなし」
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_root() / "shortcuts.json"

    def load(self) -> dict[str, str]:
        """壊れていても起動は止めない 既定のまま使えれば困らない"""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): value for key, value in data.items() if isinstance(value, str)}

    def save(self, overrides: dict[str, str]) -> None:
        """表を丸ごと書く ほかの窓が先に書いた表には重ねない

        設定（:meth:`PreferenceStore.save`）と違い、変えた所だけを重ねることはしない
        本人はショートカットの画面で表の全体を見て確かめている 2 つの窓の表を重ねると、
        どちらの画面にも出ていなかった組み合わせ（同じキーに 2 つの操作）ができ、
        Qt はそのキーでどちらも動かさない 最後に確かめた表がそのまま残る方が分かりやすい
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(
            self.path, json.dumps(dict(sorted(overrides.items())), ensure_ascii=False, indent=2)
        )


def find_conflicts(bindings: dict[str, str]) -> dict[str, list[str]]:
    """同じキーに 2 つ以上の操作が割り当たっているもの キー → 操作の一覧

    Qt は重なったショートカットを、どちらも実行しない（黙って何も起きない）
    押しても反応しないのは壊れたように見えるので、決める時点で止める
    """
    owners: dict[str, list[str]] = {}
    for action, key in bindings.items():
        if key:
            owners.setdefault(key, []).append(action)
    return {key: actions for key, actions in owners.items() if len(actions) > 1}


#: 重ねたパネルのタブの置き場 :attr:`Preferences.dock_tabs` の値
DOCK_TABS_TOP = "top"
DOCK_TABS_BOTTOM = "bottom"
DOCK_TAB_POSITIONS = (DOCK_TABS_TOP, DOCK_TABS_BOTTOM)

#: 動画の映像と音声の置き方 :attr:`Preferences.media_split` の値
MEDIA_SPLIT = "split"
MEDIA_TOGETHER = "together"
MEDIA_SPLIT_MODES = (MEDIA_SPLIT, MEDIA_TOGETHER)

#: 挿入貼り付け（Ctrl+Shift+V）で押し出すトラック :attr:`Preferences.insert_paste` の値
#: 全トラック・貼り先と一緒に動く相手のトラックだけ
INSERT_ALL_TRACKS = "all"
INSERT_TARGET_TRACKS = "target"
INSERT_PASTE_MODES = (INSERT_ALL_TRACKS, INSERT_TARGET_TRACKS)

#: exe の隣の ``scripts`` に自分で置いた物の扱い :attr:`Preferences.scripts_move` の値
#: 自動で移す・移すかを尋ねる・何もしない
SCRIPTS_MOVE_AUTO = "auto"
SCRIPTS_MOVE_ASK = "ask"
SCRIPTS_MOVE_OFF = "off"
SCRIPTS_MOVE_MODES = (SCRIPTS_MOVE_AUTO, SCRIPTS_MOVE_ASK, SCRIPTS_MOVE_OFF)

#: AI へ送る物ごとの選択（:attr:`Preferences.ai_send_frames` など #342）
#: 値は :class:`~sashimono.ai.outbound.SendChoice` と同じ文字
SEND_ASK = SendChoice.ASK.value
SEND_MODES = tuple(choice.value for choice in SendChoice)
#: AI の上限に当たったときの動き（:attr:`Preferences.ai_limit_action` #344）
LIMIT_ACTIONS = tuple(action.value for action in LimitAction)
#: 上限として受け付ける範囲 0 は上限なし 上は手で書き換えた桁違いの値を弾くための目安
#: トークンは 10 億、絵は 1 万枚、時間は 1 日（分）
AI_TOKEN_LIMITS = (0, 1_000_000_000)
AI_IMAGE_LIMITS = (0, 10_000)
AI_MINUTE_LIMITS = (0, 1440)
#: 素材を読み込んだときに場面の一覧を裏で作るか（#321） :attr:`Preferences.scene_analysis` の値
#: AI アシスタントが入っているときだけ・いつも・作らない（AI が頼んだときだけ読む）
SCENES_WITH_AI = "with_ai"
SCENES_ALWAYS = "always"
SCENES_ON_DEMAND = "on_demand"
SCENE_ANALYSIS_MODES = (SCENES_WITH_AI, SCENES_ALWAYS, SCENES_ON_DEMAND)

#: 場面の解析に使うデコードのスレッドの数として選べる物 0 はすべて（PyAV に任せる）
SCENE_THREAD_CHOICES = (1, 2, 4, 0)

#: 前の版の設定の名前（``multi_audio`` 音声が 2 本以上ある動画だけの置き方）の値と、
#: 今の値の対応 分けない（``first``）を選んでいた人は、音声が 1 本の動画も分けない側へ写す
#: 読まないと、分けないと決めていた人の置き方が黙って分ける側へ変わる
_LEGACY_MULTI_AUDIO = {"split": MEDIA_SPLIT, "first": MEDIA_TOGETHER}

#: 0.2.1 まで（#278）の再生ヘッドの吸い付き方（3 択の言葉）と、今の入切の対応（#313）
#: 「常に」は入、「吸い付かない」は切 「Shift の間だけ」は入切のどちらでもないので
#: ここに書かず、今の既定へ戻す 読まないと、切っていた人のところで黙って吸い付き始める
_LEGACY_PLAYHEAD_SNAP = {"always": True, "off": False}

#: プレビューの画質を自動で落とし始める縦の画素数
#: 1080p までは等倍で 60fps に入るので、落とす値打ちが無い
AUTO_QUALITY_HEIGHT = 1081

#: 先読みに使えるメモリの下限（MB） 1080p の 1 枚が 8MB なので、
#: これより小さいと数えるほどしか置けない
MIN_PREFETCH_MB = 128

#: 上限（MB） 家庭用の GPU の載っているメモリを超えない所で止める
MAX_PREFETCH_MB = 8192


@dataclass(frozen=True, slots=True)
class Preferences:
    """本人の好みで変わる設定

    プロジェクトではなく本人に付く 同じプロジェクトを別の機械で開いたときに、
    その機械の速さに合った設定で開きたい（速い機械では等倍で見たい）

    既定は「自動」 4K を置いた人が、なぜ重いのか分からないまま使うのを避ける
    自動で画質が変わるのを嫌う人は、設定で止められる
    """

    #: プレビューで控え（プロキシ）を使う
    use_proxy: bool = True
    #: 控えの縦の画素数
    proxy_height: int = 540
    #: 高さが :data:`AUTO_QUALITY_HEIGHT` 以上の素材を読み込んだら、プレビューの画質を自動で落とす
    #: 画面の大きさは見ない（設定画面の文言は ``preferences_dialog.AUTO_QUALITY_TEXT``）
    auto_quality: bool = True
    #: 自動で落とすときの分母
    auto_quality_divisor: int = 2
    #: 手が止まっている間に、再生ヘッドの先を描いて取っておく
    #: 既定は入 効果を積んだ所で再生が飛ぶのは、なぜ飛ぶのか分からない側の人ほど
    #: 困る 貯めるのが重すぎる所は画面の側（ui/preview.py）で自分から止める
    prefetch: bool = True
    #: 先読みに使うメモリ（メガバイト）
    #: 上限を置くのは、デコードと効果の側が使う GPU のメモリを残すため
    #: 使い切ると、先読みではなくプレビューそのものが描けなくなる
    prefetch_budget_mb: int = 1024
    #: 先読みを別のスレッドで描く 切ると画面のスレッドで 1 コマずつ描く
    #: 既定は入 画面のスレッドで描くと、1 コマ描く間は操作を受け付けない
    #: （4K を 3 枚重ねて効果を積むと 1 コマ 60ms） 共有した GL コンテキストを
    #: 作れない機械では、入れたままでも画面のスレッドへ自分で戻る 切れるように
    #: するのは、ドライバとの相性で絵が乱れたときに逃げられるようにするため
    #: 実測は :data:`sashimono.engine.render.background.MEASURED_PREFETCH_STALL_MS`
    prefetch_thread: bool = True
    #: 書き出しで、GPU の合成を書き込み（色変換・エンコード・mux）の何枚ぶん先へ進めるか
    #: 0 で直列（1 枚ずつ、スレッドを使わない）
    #: 既定は 2 1 枚ぶんは画面 1 枚の RGBA（1080p で 8MB、4K で 33MB）なので、
    #: メモリの少ない機械では減らせるようにする 実測は
    #: :data:`sashimono.engine.encode.MEASURED_EXPORT_MS`
    export_pipeline_depth: int = DEFAULT_PIPELINE_DEPTH
    #: 書き出す動画に、名前の付いた目印（マーカー）をチャプターとして埋めるか（#330）
    #: 2026-10-17 に利用者が決めた 既定は入 埋めても再生には響かず、プレーヤーで章送りが
    #: できる 要らない人だけ切ればよい 書き出しの窓で替えた値もここへ覚える
    export_chapters: bool = True
    #: 重ねたレイヤーの映像デコードを、同時にいくつまで走らせるか 1 で並べない
    #: PyAV のデコードは GIL を解放するので、別の素材どうしなら本当に重なる
    #: 既定は 4 デコード中の絵を素材の数だけ抱えるので（4K で 1 枚 33MB）、
    #: メモリの少ない機械では減らせるようにする 実測は
    #: :data:`sashimono.engine.encode.MEASURED_DECODE_MS`
    decode_threads: int = DEFAULT_DECODE_THREADS
    #: リバーブ・ディレイ・音程の調整のキーフレームを、0.34 秒の区切りの中でもつなぐ
    #: 切ると区切りの頭の値で掛け、動きが最大 0.34 秒遅れて段になる（音程を動かすと階段に
    #: 聞こえる） 既定は入 つないでも 1 塊の手間は予算（21ms）の中に収まり、知らない人ほど
    #: 段を「壊れた音」と受け取る つなぐと値の動く区切りだけ 2 度掛けるので、遅い機械で
    #: 再生が途切れる人は切れるようにする 再生と書き出しの両方に効く（同じ音にするため）
    smooth_audio_motion: bool = True
    #: AviUtl2 のスクリプトモジュール（``.mod2`` の中身が DLL の物）を読む
    #: 既定は入 テレビ字幕のように、DLL が無いと絵が出ない配布スクリプトがある
    #: 読んだ DLL は Sashimono と同じ権限で動く（Lua の閉じ込めの外） 読むのは
    #: 本人がスクリプトフォルダへ置いた物だけだが、気になる人は切れるようにする
    native_modules: bool = True
    #: 控えと解析の進み具合を、素材一覧の行にも添える（ステータスバーには必ず出す）
    #: 既定は入 読み込んだ直後にプレビューが重い理由が、どの素材の控えを
    #: 作っている最中だからなのかを、知らない人ほど見て分かる必要がある
    #: 行の文字が 250ms ごとに変わるのが目障りな人は切れるようにする
    pool_progress: bool = True
    #: HDR（PQ・HLG）や広色域（BT.2020）の素材を読み込んだときに、SDR として扱うので
    #: 白っぽく出ると知らせる窓を出す（素材一覧の行の印はこの設定に関係なく出す）
    #: 既定は入 HDR を知らない人ほど、褪せた絵をソフトの不具合だと受け取る
    #: 知っていて HDR の素材を何度も読み込む人は、窓が邪魔なので切れるようにする
    hdr_notice: bool = True
    #: AviUtl2 の汎用プラグイン（``.aux2``）を全部読んで、スクリプトが引くモジュールを探す
    #: 既定は切 切っている間は、名前を出すと確かめたプラグイン（合成フォントの
    #: ``comfont.aux2``）だけを読む 全部を読むと、関係の無いプラグインが初期化で
    #: Python を起動したりウィンドウを作ったり、本人の AviUtl2 の置き場へ書いたりする
    #: （Issue #135） 表に無いプラグインのモジュールを使いたい人だけが入れる
    all_aviutl_plugins: bool = False
    #: 素材一覧の表示 一覧（``list``）かアイコン（``icons``）か
    #: 既定は一覧 名前と長さと大きさが 1 行で読めて、素材が何本あっても見渡せる
    #: 絵で選びたい人は一覧の上のボタンで切り替え、次に開いたときもそのままにする
    media_view: str = VIEW_LIST
    #: アシスタントが使うモデル 空は Claude Code の既定に任せる
    #: 既定を空にするのは、選べるようになる前と同じ動きにするため（アカウントに
    #: よって使えるモデルが違い、決め打ちすると使えない人が出る）
    ai_model: str = DEFAULT_AI_MODEL
    #: アシスタントの考える深さ 空は Claude Code の既定
    ai_effort: str = DEFAULT_AI_EFFORT
    #: アシスタントの入力欄で Enter だけで送る 切ると Ctrl+Enter で送り、Enter は改行
    #: 既定は入 チャットの多くが Enter で送る形で、知らない人はまずそう押す
    #: 長い指示を何行も書く人が、うっかり途中で送らないように切れるようにする
    chat_enter_sends: bool = True
    #: アシスタントの実行中、状態の行に回る印を添える（#250）
    #: 既定は入 文字だけだと、応答を待つ数十秒の間に固まったのかと迷う 動く物が
    #: 気になる人（OS の「視覚効果を減らす」を入れている人など）は切れる 切っても文字の状態は残る
    ai_busy_animation: bool = True
    #: アシスタントが指示をすべて終えたとき、別の窓を触っていたらタスクバーで知らせる
    #: 既定は切 頼んでいないのにタスクバーが光ると驚く 待つ間に別の作業をする人だけが入れる
    ai_done_alert: bool = False
    #: AI の部品（claude-agent-sdk と同梱の Claude Code）を自動で新しくする（配布版だけ）
    #: 既定は入 新しいモデルは新しい Claude Code でないと断られ、知らない人はエラーの
    #: 意味が分からず困る 1 日に 1 回まで PyPI を確かめ、試した範囲の中の版だけを裏で入れる
    #: 通信を増やしたくない人は切れる 切っても導入の欄の〔更新を確かめる〕〔環境を更新〕は使える
    ai_auto_update: bool = True
    #: 空のプロジェクトへ最初の動画を置いたとき、プロジェクトの解像度とフレームレートを
    #: 動画に合わせるか 尋ねる（``ask``）・常に合わせる（``always``）・合わせない（``never``）
    #: 既定は尋ねる 黙って合わせると決まった形で作る人が困り、黙って合わせないと
    #: 60fps の動画が 30fps で書き出されたことに、書き出すまで気付けない
    match_video: str = MATCH_ASK
    #: 重ねたパネル（オブジェクト設定と AI アシスタント、メディアと字幕など）の
    #: タブを上（``top``）に出すか下（``bottom``）に出すか
    #: 既定は上 Qt の既定の下だと、パネルの名前を探して窓の一番下まで目を動かすことになり、
    #: タブがあること自体に気付かない人がいた（Issue #27） 下の方が見慣れた人は戻せる
    dock_tabs: str = DOCK_TABS_TOP
    #: 選んだクリップの外枠をプレビューに出し、掴んで位置・拡大率・回転を変える
    #: 既定は入 数を打つより早く、枠が無いと絵がどこまであるのか分からない 枠が絵の
    #: 確認の邪魔になる人は切れるようにする
    preview_handles: bool = True
    #: キーフレームのある値をプレビューで動かしたとき 再生ヘッドの所へ点を打つ（既定
    #: 利用者の決定 その時刻の絵だけが変わる）か、全部の点を同じだけずらすか
    keyframe_drag: str = KEYFRAME_DRAG_AT_PLAYHEAD
    #: タイムラインのクリップの上に、不透明度（絵）と音量（音）の線を出して直接動かせるようにする
    #: 既定は出す 設定パネルを開かずにフェードや音量を決められることを、知らない人ほど線を
    #: 見て気付く サムネイルや波形に線が重なるのが目障りな人、クリップの真ん中を掴んで動かす
    #: つもりで線を掴んでしまう人は切れるようにする
    value_lines: bool = True
    #: タイムラインで、これより細いクリップは名前・サムネイル・波形を描かず細い帯にする（画素）
    #: 細い帯には絵の平均の色と音の大きさだけを描く 既定は 24（名前の頭の 1 字が入る幅
    #: 今までの値） 細かく見たい人は 8 まで下げ、短いクリップが数千本あって描くのが重い
    #: 機械では上げる 狭くすると読めない切れ端が並び、広くすると拡大しても中身が出ない
    detail_min_width: int = 24
    #: ズームの途中で、初めて見る倍率の波形を作らず、近い倍率で作った波形を伸ばして仮に描く
    #: ズームが止まってから作り直す 既定は切（今の見た目） 伸ばした波形は列の束ね方が違うので
    #: 細かい山が少しずれて見える 境を越えた先の初めての倍率の 1 段で、波形を作る分（短い
    #: クリップ 2000 本の手元の計測で 3〜6ms 遅い機械ではその数倍）を省ける ズームの
    #: 引っかかりが気になる遅い機械の人は入れる
    stretch_waves: bool = False
    #: 設定パネルで、行の名前（数はスライダーも）のダブルクリックで値を初期値へ戻す
    #: 既定は入（利用者の要望） 初期値を覚えていなくても戻せ、戻しても取り消せる
    #: 行の名前を続けて押しがちで、うっかり戻るのが嫌な人は切れるようにする
    double_click_reset: bool = True
    #: 設定パネルの〔プリセット…〕で当てるとき、文字そのもの（タイマーの書式なども）も
    #: 当てる 既定は切（利用者の決定） 見た目だけを当て、打った文字は残す（テンプレートの
    #: 棚の着せ替えと同じ） 決まった文言の見出しを丸ごと当てたい人は入れる
    preset_with_text: bool = False
    #: 同じく、画面の中の位置（X と Y）とグループ制御の対象レイヤー数も当てる 既定は切
    #: どちらも置いた場所の事情で、
    #: 字幕の見た目を当てたら全部が保存した所へ寄ってしまう 決まった所に出す物を作る人は入れる
    preset_with_position: bool = False
    #: 同じく、当てる先に足してあるエフェクトを残してプリセットの物を足す 既定は切（入れ替える）
    #: 入れ替えると保存したクリップと同じ見た目になり、試しに当て比べても前のエフェクトが
    #: 重ならない グローと影のように別々に作ったプリセットを重ねて使う人は入れる
    preset_keep_effects: bool = False
    #: プリセットとエイリアスの一覧（管理の窓 #276 #277）の見本の絵の描き方
    #: 既定はエフェクトも描く（GPU 別のスレッドで描くので一覧を開く速さには響かない）
    #: GPU を使わせたくない人・古い機械の人は文字と図形だけ（軽い）か、出さない（名前だけ）
    library_thumbnails: str = THUMBNAILS_FULL
    #: 見本の絵の地 既定は市松模様 白い文字も黒い文字も見え、透けている所も分かる
    #: 実際に重ねる映像に近い地で見たい人は暗い地か明るい地にする
    library_backdrop: str = BACKDROP_CHECKER
    #: 一覧から消すときに確かめる 既定は入 消した物はごみ箱から戻せるが、戻せることを
    #: 知らない人には黙って消えたように見える 何件も続けて整理する人は切れる
    library_confirm_delete: bool = True
    #: テンプレートの棚の一覧にも見本の絵を出す 既定は入（利用者の決定） 描き方は上の
    #: 「プリセットの見本」と同じ 棚は数が多く、GPU や置き場を使わせたくない人は切れる
    #: 切っても選んだ 1 本の大きな下絵は出る
    shelf_thumbnails: bool = True
    #: プリセットやエイリアスを保存するとき、同じ名前なら確認せず上書きする 既定は切
    #: （保存の窓の「同じ名前があれば上書きする」の印は外れていて、入れるまで保存できない）
    #: 同じ物を何度も作り直す人は入れる 入れても前の内容はごみ箱へ移る
    save_overwrite_without_asking: bool = False
    #: 設定パネルで、焦点の無い欄（クリックしていない選択の欄・数値の欄・スライダー）でも
    #: ホイールで値を変える 既定は切 切っていると、焦点の無い欄の上のホイールはパネルを送る
    #: 入れていると、パネルを送る途中で通った欄の値が変わり、気付かずに取り消しの段が積まれる
    #: 欄にカーソルを載せて回すだけで値を変えたい人は入れられるようにする
    wheel_unfocused: bool = False
    #: タイムラインの磁石 クリップを動かす・端を伸び縮みさせる・置くときに、ほかのクリップの
    #: 端・再生位置・キーフレーム・書き出し範囲の端へ吸い付く 既定は入（利用者の決定）
    #: 1 コマずつ自由に置きたい人は、タイムラインの上の〔磁石〕で切れる（Shift で一時的にも）
    timeline_snap: bool = True
    #: 吸い付く距離（画面の画素） 画面の倍率で変わらない指の感覚で決める
    snap_distance: int = 8
    #: プレビューの磁石 位置を動かすときに画面の中央・端やほかの物の端と中央へ吸い付く
    #: タイムラインの磁石とは別に切れる（利用者の要望） 既定は入 知らない人ほど中央へ
    #: 揃えにくい 1 画素ずつ自由に置きたい人は切る（Shift で一時的にも）
    preview_snap: bool = True
    #: 目盛りで再生ヘッドを動かすときに、近くの先へ吸い付くか（#278 #313） 距離は上の
    #: ``snap_distance`` タイムラインの上の〔磁石〕の隣のボタンと同じ物で、〔磁石〕とは別に切れる
    #: 既定は入（利用者の決定 「磁石のように」） 前は Shift を押している間だけ吸い付く既定に
    #: していたが、Shift を押してから目盛りを押すと書き出し範囲のドラッグになるので、知らない
    #: 人には吸い付かせ方が分からなかった 入でも掴んだ後に Shift を押している間は外れ、
    #: 吸い付く距離（8 画素）の外では自由に動く 1 コマずつ合わせたい人はボタンで切る
    playhead_snap: bool = True
    #: 再生ヘッドが吸い付く先 クリップの頭と終わり・キーフレームのコマ・書き出し範囲の端・
    #: 目印（マーカー） 既定はクリップの磁石と同じ先で、目印は切 0.2.2 までは目印を描かず、
    #: 見えない所で引っ掛かったように見えたため 目盛りに旗を描くようになっても（#330）既定は
    #: 変えない 変えると、使い慣れた人の再生ヘッドが今まで止まらなかった所で止まる
    playhead_snap_clips: bool = True
    playhead_snap_keyframes: bool = True
    playhead_snap_work_area: bool = True
    playhead_snap_markers: bool = False
    #: 挿入貼り付け（Ctrl+Shift+V）で、再生ヘッドから後ろを押し出すトラック 全トラック
    #: （``all``）か、貼り先と、そこで押すクリップのリンクの相手・グループの仲間・焼き込んだ
    #: 字幕のトラックだけ（``target``）か 既定は全トラック（Premiere Pro の既定 全トラックの
    #: 同期ロックが入っている） 知らない人ほど、別のトラックに置いた字幕や BGM が貼った
    #: 長さぶんずれたことに後で気付く ほかのトラックを動かしたくない人は切り替えられる
    insert_paste: str = INSERT_ALL_TRACKS
    #: 新しく作るプロジェクトのトラックの方式（:class:`~sashimono.core.model.LayerMode`）
    #: 新規作成の窓の初期値と、起動した直後の空のプロジェクトに使う
    #: 既定は混合（YMM4・AviUtl と同じ 1 本のレイヤーに何でも置く 利用者の決定）
    #: 映像と音声を別のトラックに分けて並べる方が慣れている人は切り替えられる
    #: モデルの既定（分ける）とは別に持つ 古いファイルと試験の動きを変えないため
    new_project_layers: str = LayerMode.MIXED
    #: 新しく作る作品の編集の方針（#343） 新規作成の窓の初期値と、起動した直後の空の作品に使う
    #: 動画の作り手は作品をまたいで同じ書体と色を使うことが多く、作るたびに書き直させない
    #: 既定は空（決めていない） 開いた作品の方針は変えない（方針は作品に付く）
    new_project_style_guide: StyleGuide = field(default_factory=StyleGuide)
    #: 動画の映像と音声の置き方 映像と音声を別のトラックへ分けて置く（``split``）か、
    #: 1 本のクリップにまとめる（``together``）か 分けると、混合の方式では置いたレイヤーに
    #: 映像、その次のレイヤーから音声を 1 本ずつ並べ、どれも一緒に動く（リンク）
    #: まとめると、音声が 2 本以上ある動画（ゲームの録画のマイクの声など）は 1 本目だけを置く
    #: 既定は分ける（利用者の要望 音声が 1 本でも映像と音声を別のレイヤーに置き、
    #: 音声が複数あればレイヤー 1 に映像、レイヤー 2 以降に音 Issue #27）
    #: 1 本にまとめると 2 本目以降の音がタイムラインのどこにも無く、鳴らす手段に気付けない
    #: レイヤーが増えるのを嫌う人・YMM4 のように 1 本で持ちたい人は切り替えられる
    media_split: str = MEDIA_SPLIT
    #: 画面の色 暗い（``dark``）・明るい（``light``）・Windows の設定に合わせる（``system``）
    #: 既定は暗い 明るいテーマを足す前からの見た目で、映像の色を見る作業では周りが暗い方が
    #: 目が明るさに慣れない 明るい部屋で使う人・暗い画面の文字が読みにくい人は切り替えられる
    #: 「合わせる」を既定にしないのは、Windows を明るくしている人の画面が、版を上げた
    #: だけで黙って明るくなるため
    theme: str = THEME_DARK
    #: 起動したときに新しい版を確かめる（数時間に 1 回まで） 見つけたら裏で落として確かめておく
    #: 既定は入 古い版のまま直った不具合に困り続ける人を作らない 締め切り前など、編集環境を
    #: 変えたくない人は切る（切っている間は今の版に留まる ヘルプの〔更新を確かめる…〕で
    #: 手で確かめられる）
    update_check: bool = True
    #: 正式版より先に出すベータ版も受け取る 既定は切 作りが大きく変わることがあり、
    #: 知らない人がいきなり受け取ると困る
    update_beta: bool = False
    #: 落として確かめた新しい版を入れる前に尋ねる 既定は入（入れるには再起動が要り、
    #: 編集の途中で勝手に再起動しないため） 切ると、尋ねずに次の起動の頭で入れる
    update_confirm: bool = True
    #: 配布版の exe の隣の ``scripts`` に自分で置いた物を、起動のときに ``%APPDATA%`` の側へ
    #: 移すか 自動で移す（``auto``）・移すかを尋ねる（``ask`` 同じ物については 1 度だけ）・
    #: 何もしない（``off``） 既定は自動で移す（利用者の決定） zip を手で展開し直すと消える所に
    #: 置いたままだと、知らない人ほど失くす 読む順は変わらない（exe の隣より ``%APPDATA%`` が
    #: 後に読まれて勝つので、移しても同じ名前の勝ち負けは同じ） exe の隣にまとめておきたい
    #: 人は尋ねる・何もしないへ切り替える（〔互換〕→〔exe の隣のスクリプトを移す…〕からは
    #: いつでも移せる）
    scripts_move: str = SCRIPTS_MOVE_AUTO
    #: 新しく作る字幕（焼き込み）のテキストを、画面の幅に合わせて自動で折り返す（#249）
    #: 既定は入 知らない人ほど、長い字幕が画面からはみ出して切れるのに困る 改行を手で
    #: 決めたい人・字幕の整形の文字数だけで分けたい人は切れる 既にあるテキストには触らない
    subtitle_wrap: bool = True
    #: 折り返す幅（画面の幅の何 %） 既定 90 は左右に 5% ずつの余白で、縁取りと影が画面の
    #: 端に掛からない 端まで使いたい人は 100、狭くまとめたい人は下げる
    subtitle_wrap_percent: int = 90
    #: 字幕を置く窓（焼き込み・選んだ行を置く #308）で、最後に使った見た目を次も選んでおく
    #: 既定は切（今まで通り、タイムラインで選んでいるテキストがあればそれ、無ければ既定の見た目）
    #: 選ぶテキストを毎回替えて写す人が多く、覚えると前のプリセットが選ばれていて気付かない
    #: いつも同じエイリアスで字幕を置く人は入れる
    subtitle_look_remember: bool = False
    #: 覚えた見た目の鍵（``compat.subtitle_looks.LookChoice.key``） 設定画面には出さない
    #: 覚える設定が入っているときだけ、置くたびに書く
    subtitle_look_last: str = ""
    #: 保存していない変更を一定の間隔で退避し、落ちた後の起動で復元を勧める（#271）
    #: 既定は入（前からの動き） 切ると、落ちたときに最後に保存した所までしか戻らない
    #: 書き込みを減らしたい人・自分でこまめに保存する人は切れる
    autosave: bool = True
    #: 退避の間隔（秒） 既定 30 は前からの値 落ちたときに失うのは最大でこの長さの作業
    #: 短くするほど書き込みが増える（1 回は数百 KB の JSON）
    autosave_seconds: int = 30
    #: 上書き保存の前に、前の中身をバックアップとして残す（#271）
    #: 既定は入（前からの動き） 切ると「さっきの保存で壊した」を戻せない
    backup: bool = True
    #: 1 つのプロジェクトについて残すバックアップの数 既定 20 は前からの値
    #: 減らしたとき（置き場を変えた・入れ直したときも）超えた分は、設定画面で消える物を
    #: 見せて確かめ、OK でその場で消す 上書き保存のときは 1 本作って 1 本消すだけ
    backup_generations: int = 20
    #: 退避とバックアップの置き場 空は既定（``%LOCALAPPDATA%\Sashimono``）
    #: 既定を空で持つのは、ユーザー名やドライブが変わった機械でも既定の置き場を指し続けるため
    #: 書けない所を選んでいたら、その起動は既定の置き場へ戻して知らせる（黙って退避を止めない）
    state_folder: str = ""
    #: 復元を勧めたのに残っている落ちた作業の退避を、何日たったら片付けるか 0 は片付けない
    #: 既定は 0（前からの動き） 片付けるのは起動して復元を尋ねた後だけで、まだ一度も
    #: 勧めていない退避は古くても消さない
    recovery_keep_days: int = 0
    #: 退避とバックアップの置き場の容量の上限（MB） 0 は上限なし（既定 前からの動き）
    #: 超えたら古い物から片付ける 世代数の上限とは別に効き、先に当たった方で消える
    #: 開いている作業の今の退避・まだ勧めていない落ちた作業・各プロジェクトのいちばん
    #: 新しいバックアップは、上限を超えていても消さない
    state_limit_mb: int = 0
    #: テンプレートの棚の〔選択中のクリップに適用〕で、今のテキストの折り返しの幅を残す（#283）
    #: 既定は切（テンプレートと同じ見た目になる側） AviUtl と YMM4 のテンプレートの多くは
    #: 折り返しの幅を持たないので、残すと棚の見本と違う行の分かれ方になり、知らない人ほど
    #: 着せ替えが効いていないと受け取る 字幕の枠の幅を先に決めてから着せ替える人は入れる
    #: 入れると、幅を使っているテキストは組み方も残す（AviUtl2 の組み方は折り返さないので、
    #: テンプレートの組み方を取ると幅を残しても 1 行のままになる）
    restyle_keep_wrap: bool = False
    #: 指示があいまいなときは、AI アシスタントが作業の前に質問する（#296）
    #: 既定は入 知らない人ほど、短い指示で意図と違うクリップや範囲を変えられて困る
    #: 聞かれるのが手間な人・指示を細かく書く人は切れる 切ると質問の道具を渡さず、
    #: 指示の文からも聞き返す基準を外す
    ai_ask_questions: bool = True
    #: 削除して詰めたあと（Shift+Delete と右クリックの「削除して詰める」）、再生ヘッドを
    #: 消した所の頭（つなぎ目）へ動かす（Issue #299） 既定は入 詰めると後ろが前へ寄り、
    #: 元の位置はつなぎ目と関係の無い所を指す つなぎ目を確かめたい人が多い
    #: 再生中に消したときは、止めずに頭から続ける（末尾に着いたら続きが無いので止める）
    #: 動いてほしくない人は切れば今までどおり
    ripple_playhead: bool = True
    #: AI アシスタントの完成物の検査（#337）が比べる音量の基準（LUFS） 選べるのは
    #: :data:`~sashimono.engine.checks.LOUDNESS_TARGETS` 配信か放送かで答えが 9 LU も違う
    #: 既定は配信の -14 個人の作品はほとんどが配信へ出す 放送へ出す人は -23 / -24 を選ぶ
    check_loudness_target: int = -14
    #: AI アシスタントが BGM を話している間だけ下げる（ダッキング #329）ときの既定の下げ幅（dB）
    #: 声を聞かせたい量は作品と BGM で違う AI に値を言わなければこれを使う 既定は放送や
    #: 配信の手引きでよく使う 10〜15 dB の真ん中の 12（:mod:`sashimono.core.mixing`）
    duck_depth_db: int = DEFAULT_DUCK_DEPTH_DB
    #: ダッキングで下げ始めてから下がり切るまでの長さ（ミリ秒） 既定は 300 短いほど BGM が
    #: ぶつりと切れたように聞こえ、長いほど声の頭に BGM が残る
    duck_attack_ms: int = DEFAULT_DUCK_ATTACK_MS
    #: ダッキングで話し終えてから元の音量へ戻るまでの長さ（ミリ秒） 既定は 800 短いと文の間で
    #: BGM が出入りして落ち着かない（ポンピング）
    duck_release_ms: int = DEFAULT_DUCK_RELEASE_MS
    #: AI アシスタントが付ける強調の動き（#331）の既定 どれだけ寄るか・揺らすか・どれだけ速く
    #: 動くかは人と作品で違う AI に値を言わなければこれを使う 既定は寄ったと分かり、画面の端が
    #: 大きく欠けず、字幕が読めなくならない側（:mod:`sashimono.core.emphasis`）
    #: ズームの拡大率（今の大きさからの %）
    emphasis_zoom_percent: int = DEFAULT_EMPHASIS_ZOOM_PERCENT
    #: ポップ（一瞬ふくらんで戻る）の拡大率（%）
    emphasis_pop_percent: int = DEFAULT_EMPHASIS_POP_PERCENT
    #: 揺れの幅（画素 上下左右それぞれ）
    emphasis_shake_px: int = DEFAULT_EMPHASIS_SHAKE_PX
    #: パンで向きだけを言われたときにずらす長さ（画素）
    emphasis_pan_px: int = DEFAULT_EMPHASIS_PAN_PX
    #: 動き始めてから動き切るまで（と戻るまで）の長さ（ミリ秒）
    emphasis_ramp_ms: int = DEFAULT_EMPHASIS_RAMP_MS
    #: 動き方（:data:`~sashimono.core.emphasis.EMPHASIS_EASINGS`） 既定はなめらかに動き出して止まる
    emphasis_easing: str = DEFAULT_EMPHASIS_EASING
    #: AI アシスタントがジャンプカットのつなぎ目を目立たなくする（#332）ときの既定
    #: 溶かすか寄るか・どれだけの長さかは人と作品で違う AI に値を言わなければこれを使う
    #: 既定は短いクロスフェードと 20 ms の音のフェード（絵の中身が欠けない側
    #: :mod:`sashimono.core.seams`）
    #: 映像の処理（:data:`~sashimono.core.seams.SEAM_VIDEOS`）
    seam_video: str = DEFAULT_SEAM_VIDEO
    #: クロスフェードの長さ（ミリ秒）
    seam_crossfade_ms: int = DEFAULT_SEAM_CROSSFADE_MS
    #: 1 本おきに寄るときの拡大率（%）
    seam_zoom_percent: int = DEFAULT_SEAM_ZOOM_PERCENT
    #: 音の継ぎ目のフェード（ミリ秒） 0 で音は触らない
    seam_audio_fade_ms: int = DEFAULT_SEAM_AUDIO_FADE_MS
    #: 素材を読み込んだとき（とプロジェクトを開いたとき）に、場面の切り替わりを裏で調べて
    #: 覚えるか（#321） AI アシスタントの道具（detect_scenes）が使う 既定は AI アシスタントが
    #: 入っているときだけ 場面の解析は素材を全フレーム読むので、AI を使わない人の機械で
    #: 毎回 CPU を使わせない（4K 154 秒を 2 本のスレッドで 38 秒） AI を使う人は入れて
    #: おけば、頼んだときに待たずに返る いつも作る・作らない（AI が頼んだときだけ読む）も選べる
    scene_analysis: str = SCENES_WITH_AI
    #: 場面の解析に使うデコードのスレッドの数 0 はすべて 既定は 2（裏で読む間も編集と再生に
    #: CPU を残す） 速い機械で早く終えたい人は増やし、遅い機械では 1 にする
    #: 縮める大きさや間引きでは速さが変わらなかった
    #: （:data:`sashimono.engine.scenes.MEASURED_SCENE_SECONDS`）ので、
    #: 速さと重さの釣り合いはこれ 1 つで取る
    scene_threads: int = DEFAULT_SCENE_THREADS
    #: 書き出しの窓で前に書き出したフォルダ 空はまだ書き出していない
    #: 好みではなく覚えている物なので、設定画面には出さない 次に開いた書き出しの窓の
    #: 初めの場所になり、AI アシスタントが書ける場所にも入る（作品のフォルダと、人が
    #: 窓で選んで書き出した所だけ 2026-10-14 利用者の決定 #350）
    export_folder: str = ""
    #: AI アシスタントが外（Anthropic）へ送る物ごとに、送る前に尋ねる（``ask``）・尋ねずに
    #: 送る（``allow``）・送らない（``deny``）（#342 :mod:`sashimono.ai.outbound`）
    #: 既定は尋ねる 何が外へ出るかを知らないまま、映像の絵や字幕の文が送られるのは困る
    #: 尋ねる窓で「送る」と答えると ``allow`` を覚える 設定で ``ask`` へ戻すとまた尋ねる
    #: 映像の絵（プレビューを描いた画像）
    ai_send_frames: str = SEND_ASK
    #: 字幕と文字（字幕起こしの結果・テキストのクリップの文）
    ai_send_text: str = SEND_ASK
    #: 素材の名前と場所 送らないと決めたら、伏せ字にして送る
    ai_send_files: str = SEND_ASK
    #: AI アシスタントが使う量の上限（#344 :mod:`sashimono.ai.budget`） 0 は上限なし
    #: どこまで使ってよいかは契約と懐で人ごとに違う 既定は、ふつうの指示では当たらず、同じ道具を
    #: 繰り返して止まらなくなった指示は止まる量 当たっても、既定では続けるかを尋ねるだけ
    #: 1 回の指示のトークン（入力 + キャッシュの書き込み + 出力）
    ai_limit_turn_tokens: int = DEFAULT_TURN_TOKENS
    #: 1 回の指示で送る絵の枚数
    ai_limit_turn_images: int = DEFAULT_TURN_IMAGES
    #: 1 回の指示の時間（分）
    ai_limit_turn_minutes: int = DEFAULT_TURN_MINUTES
    #: 1 日のトークン
    ai_limit_day_tokens: int = DEFAULT_DAY_TOKENS
    #: 1 日に送る絵の枚数
    ai_limit_day_images: int = DEFAULT_DAY_IMAGES
    #: 上限に当たったとき 止めて続けるかを尋ねる（``ask``）・尋ねずに止めて知らせる（``stop``）
    #: 既定は尋ねる 知らない人が黙って止められると、AI が途中で投げ出したように見える
    #: 席を外して任せる人は、尋ねる箱で待たされないよう「尋ねずに止める」を選べる
    ai_limit_action: str = LimitAction.ASK.value
    #: AI に見せる絵 1 枚の長い辺の上限（画素 :data:`~sashimono.ai.budget.IMAGE_EDGES`）
    #: 既定は #361 で決めた 1568（モデルが縮めずに読める大きさ） 費用を抑えたい人は小さくする
    ai_image_edge: int = DEFAULT_IMAGE_EDGE

    def budget_limits(self) -> Limits:
        """AI の上限 ブリッジの数え（:class:`~sashimono.ai.budget.UsageMeter`）へ渡す"""
        return Limits(
            turn_tokens=self.ai_limit_turn_tokens,
            turn_images=self.ai_limit_turn_images,
            turn_seconds=self.ai_limit_turn_minutes * 60,
            day_tokens=self.ai_limit_day_tokens,
            day_images=self.ai_limit_day_images,
            action=LimitAction(self.ai_limit_action),
            image_edge=self.ai_image_edge,
        )

    def send_choices(self) -> dict[Outbound, SendChoice]:
        """AI へ送る物ごとの選択 ブリッジ（:class:`~sashimono.ai.bridge.EditorBridge`）へ渡す"""
        return {
            Outbound.FRAMES: SendChoice(self.ai_send_frames),
            Outbound.TEXT: SendChoice(self.ai_send_text),
            Outbound.FILES: SendChoice(self.ai_send_files),
        }

    def with_send_choice(self, kind: Outbound, choice: SendChoice) -> Preferences:
        """1 つだけ変えた設定 確かめの窓で「送る」と答えた物を覚えるのに使う"""
        if kind is Outbound.FRAMES:
            return replace(self, ai_send_frames=choice.value)
        if kind is Outbound.TEXT:
            return replace(self, ai_send_text=choice.value)
        if kind is Outbound.FILES:
            return replace(self, ai_send_files=choice.value)
        # 同意の要らない物（指示・編集の構成）は、いつも送るので覚える所が無い
        raise ValueError(f"{kind} は送る前に尋ねる物ではありません")

    @property
    def subtitle_wrap_share(self) -> int:
        """字幕の折り返しの幅（画面の幅の %） 切ってあれば 0（折り返さない）"""
        return self.subtitle_wrap_percent if self.subtitle_wrap else 0

    @property
    def duck_shape(self) -> DuckShape:
        """AI アシスタントのダッキング（#329）の既定の形 AI が値を言わなければこれを使う"""
        return DuckShape(
            depth_db=float(self.duck_depth_db),
            attack_ms=self.duck_attack_ms,
            release_ms=self.duck_release_ms,
        )

    @property
    def emphasis_style(self) -> EmphasisStyle:
        """AI アシスタントの強調の動き（#331）の既定 AI が値を言わなければこれを使う"""
        return EmphasisStyle(
            zoom_percent=self.emphasis_zoom_percent,
            pop_percent=self.emphasis_pop_percent,
            shake_px=self.emphasis_shake_px,
            pan_px=self.emphasis_pan_px,
            ramp_ms=self.emphasis_ramp_ms,
            easing=self.emphasis_easing,
        )

    @property
    def seam_style(self) -> SeamStyle:
        """AI アシスタントのジャンプカットのつなぎ目の処理（#332）の既定"""
        return SeamStyle(
            video=self.seam_video,
            crossfade_ms=self.seam_crossfade_ms,
            zoom_percent=self.seam_zoom_percent,
            audio_fade_ms=self.seam_audio_fade_ms,
        )

    @property
    def inserts_on_all_tracks(self) -> bool:
        """挿入貼り付け（:func:`~sashimono.core.clipboard.insert_paste_commands`）へ渡す値"""
        return self.insert_paste == INSERT_ALL_TRACKS

    @property
    def splits_media(self) -> bool:
        """素材を置く所（:func:`~sashimono.core.commands.insert_media`）へ渡す値"""
        return self.media_split == MEDIA_SPLIT

    @property
    def look_rules(self) -> LookRules:
        """字幕に見た目を着せる決まり（#308） 設定パネルや棚から着せ替えるときと同じにする"""
        return LookRules(preset_options=self.preset_options, keep_wrap=self.restyle_keep_wrap)

    @property
    def preset_options(self) -> PresetOptions:
        """プリセットの当て方 設定パネル（:meth:`InspectorPanel.set_preset_options`）へ渡す"""
        return PresetOptions(
            with_text=self.preset_with_text,
            with_position=self.preset_with_position,
            keep_effects=self.preset_keep_effects,
        )

    @property
    def library_options(self) -> LibraryOptions:
        """プリセットとエイリアスの一覧の見せ方 一覧の窓（``ui/library_dialog.py``）へ渡す"""
        return LibraryOptions(
            thumbnails=self.library_thumbnails,
            backdrop=self.library_backdrop,
            confirm_delete=self.library_confirm_delete,
            shelf=self.shelf_thumbnails,
            overwrite_without_asking=self.save_overwrite_without_asking,
        )

    def prefetch_bytes(self) -> int:
        """先読みに使えるバイト数 切ってあれば 0

        0 を渡された側は「1 枚も置けない」と読んで、先読みそのものをやめる
        入り切りの旗を下まで配らずに済む
        """
        if not self.prefetch:
            return 0
        return self.prefetch_budget_mb * 1024 * 1024

    def quality_for(self, height: int) -> int:
        """その高さの素材に対して、プレビューに使う分母

        4K を 1 枚置いただけなら元の素材でも入るが、重ねた時点で外れる
        効果を積むと余裕が減り、ぼかしと発光まで積むとどの組でも入らない
        測った値は :mod:`sashimono.engine.cache.proxy` の表を見る
        （同じ数を何か所にも書くと、測り直したときに片方だけ古くなる）
        """
        if not self.auto_quality or height < AUTO_QUALITY_HEIGHT:
            return 1
        return self.auto_quality_divisor


class PreferenceStore:
    """:class:`Preferences` の読み書き

    壊れていても起動は止めない 既定のまま使えれば困らない
    （ショートカットの保存と同じ考え方）
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_root() / "preferences.json"

    def load(self) -> Preferences:
        preferences = self._parse(self._read())
        _PREFERENCES_SEEN[_path_key(self.path)] = preferences
        return preferences

    def _read(self) -> dict[str, Any] | None:
        """書いてある中身 無い・壊れているなら ``None``"""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _parse(data: dict[str, Any] | None) -> Preferences:
        if data is None:
            return Preferences()
        plain = Preferences()
        return Preferences(
            use_proxy=_flag(data.get("use_proxy"), plain.use_proxy),
            proxy_height=_size(data.get("proxy_height"), plain.proxy_height),
            auto_quality=_flag(data.get("auto_quality"), plain.auto_quality),
            auto_quality_divisor=_divisor(
                data.get("auto_quality_divisor"), plain.auto_quality_divisor
            ),
            prefetch=_flag(data.get("prefetch"), plain.prefetch),
            prefetch_budget_mb=_budget(data.get("prefetch_budget_mb"), plain.prefetch_budget_mb),
            prefetch_thread=_flag(data.get("prefetch_thread"), plain.prefetch_thread),
            export_pipeline_depth=_depth(
                data.get("export_pipeline_depth"), plain.export_pipeline_depth
            ),
            export_chapters=_flag(data.get("export_chapters"), plain.export_chapters),
            decode_threads=_threads(data.get("decode_threads"), plain.decode_threads),
            native_modules=_flag(data.get("native_modules"), plain.native_modules),
            pool_progress=_flag(data.get("pool_progress"), plain.pool_progress),
            hdr_notice=_flag(data.get("hdr_notice"), plain.hdr_notice),
            all_aviutl_plugins=_flag(data.get("all_aviutl_plugins"), plain.all_aviutl_plugins),
            media_view=_choice(data.get("media_view"), VIEW_MODES, plain.media_view),
            ai_model=_choice(data.get("ai_model"), tuple(m.id for m in AI_MODELS), plain.ai_model),
            ai_effort=_choice(
                data.get("ai_effort"), tuple(e.value for e in AI_EFFORTS), plain.ai_effort
            ),
            chat_enter_sends=_flag(data.get("chat_enter_sends"), plain.chat_enter_sends),
            ai_busy_animation=_flag(data.get("ai_busy_animation"), plain.ai_busy_animation),
            ai_done_alert=_flag(data.get("ai_done_alert"), plain.ai_done_alert),
            ai_auto_update=_flag(data.get("ai_auto_update"), plain.ai_auto_update),
            match_video=_choice(data.get("match_video"), MATCH_MODES, plain.match_video),
            dock_tabs=_choice(data.get("dock_tabs"), DOCK_TAB_POSITIONS, plain.dock_tabs),
            preview_handles=_flag(data.get("preview_handles"), plain.preview_handles),
            keyframe_drag=_choice(
                data.get("keyframe_drag"), KEYFRAME_DRAG_MODES, plain.keyframe_drag
            ),
            value_lines=_flag(data.get("value_lines"), plain.value_lines),
            detail_min_width=_detail_min_width(
                data.get("detail_min_width"), plain.detail_min_width
            ),
            stretch_waves=_flag(data.get("stretch_waves"), plain.stretch_waves),
            smooth_audio_motion=_flag(data.get("smooth_audio_motion"), plain.smooth_audio_motion),
            double_click_reset=_flag(data.get("double_click_reset"), plain.double_click_reset),
            preset_with_text=_flag(data.get("preset_with_text"), plain.preset_with_text),
            preset_with_position=_flag(
                data.get("preset_with_position"), plain.preset_with_position
            ),
            preset_keep_effects=_flag(data.get("preset_keep_effects"), plain.preset_keep_effects),
            library_thumbnails=_choice(
                data.get("library_thumbnails"), THUMBNAIL_MODES, plain.library_thumbnails
            ),
            library_backdrop=_choice(
                data.get("library_backdrop"), BACKDROP_MODES, plain.library_backdrop
            ),
            library_confirm_delete=_flag(
                data.get("library_confirm_delete"), plain.library_confirm_delete
            ),
            shelf_thumbnails=_flag(data.get("shelf_thumbnails"), plain.shelf_thumbnails),
            save_overwrite_without_asking=_flag(
                data.get("save_overwrite_without_asking"), plain.save_overwrite_without_asking
            ),
            wheel_unfocused=_flag(data.get("wheel_unfocused"), plain.wheel_unfocused),
            timeline_snap=_flag(data.get("timeline_snap"), plain.timeline_snap),
            snap_distance=_snap_distance(data.get("snap_distance"), plain.snap_distance),
            preview_snap=_flag(data.get("preview_snap"), plain.preview_snap),
            playhead_snap=_playhead_snap(data.get("playhead_snap"), plain.playhead_snap),
            playhead_snap_clips=_flag(data.get("playhead_snap_clips"), plain.playhead_snap_clips),
            playhead_snap_keyframes=_flag(
                data.get("playhead_snap_keyframes"), plain.playhead_snap_keyframes
            ),
            playhead_snap_work_area=_flag(
                data.get("playhead_snap_work_area"), plain.playhead_snap_work_area
            ),
            playhead_snap_markers=_flag(
                data.get("playhead_snap_markers"), plain.playhead_snap_markers
            ),
            insert_paste=_choice(data.get("insert_paste"), INSERT_PASTE_MODES, plain.insert_paste),
            new_project_layers=_choice(
                data.get("new_project_layers"), LayerMode.ALL, plain.new_project_layers
            ),
            new_project_style_guide=_style_guide(
                data.get("new_project_style_guide"), plain.new_project_style_guide
            ),
            media_split=_choice(
                data.get("media_split"),
                MEDIA_SPLIT_MODES,
                _LEGACY_MULTI_AUDIO.get(str(data.get("multi_audio")), plain.media_split),
            ),
            theme=_choice(data.get("theme"), THEME_MODES, plain.theme),
            update_check=_flag(data.get("update_check"), plain.update_check),
            update_beta=_flag(data.get("update_beta"), plain.update_beta),
            update_confirm=_flag(data.get("update_confirm"), plain.update_confirm),
            scripts_move=_choice(data.get("scripts_move"), SCRIPTS_MOVE_MODES, plain.scripts_move),
            subtitle_wrap=_flag(data.get("subtitle_wrap"), plain.subtitle_wrap),
            subtitle_wrap_percent=_wrap_percent(
                data.get("subtitle_wrap_percent"), plain.subtitle_wrap_percent
            ),
            subtitle_look_remember=_flag(
                data.get("subtitle_look_remember"), plain.subtitle_look_remember
            ),
            subtitle_look_last=_text(data.get("subtitle_look_last"), plain.subtitle_look_last),
            autosave=_flag(data.get("autosave"), plain.autosave),
            autosave_seconds=_within(
                data.get("autosave_seconds"), AUTOSAVE_SECONDS_RANGE, plain.autosave_seconds
            ),
            backup=_flag(data.get("backup"), plain.backup),
            backup_generations=_within(
                data.get("backup_generations"), BACKUP_GENERATIONS_RANGE, plain.backup_generations
            ),
            state_folder=_folder(data.get("state_folder"), plain.state_folder),
            recovery_keep_days=_within(
                data.get("recovery_keep_days"), RECOVERY_KEEP_DAYS_RANGE, plain.recovery_keep_days
            ),
            state_limit_mb=_within(
                data.get("state_limit_mb"), STATE_LIMIT_MB_RANGE, plain.state_limit_mb
            ),
            restyle_keep_wrap=_flag(data.get("restyle_keep_wrap"), plain.restyle_keep_wrap),
            ai_ask_questions=_flag(data.get("ai_ask_questions"), plain.ai_ask_questions),
            ripple_playhead=_flag(data.get("ripple_playhead"), plain.ripple_playhead),
            check_loudness_target=_loudness_target(
                data.get("check_loudness_target"), plain.check_loudness_target
            ),
            duck_depth_db=_within(
                data.get("duck_depth_db"), DUCK_DEPTH_DB_RANGE, plain.duck_depth_db
            ),
            duck_attack_ms=_within(
                data.get("duck_attack_ms"), DUCK_ATTACK_MS_RANGE, plain.duck_attack_ms
            ),
            duck_release_ms=_within(
                data.get("duck_release_ms"), DUCK_RELEASE_MS_RANGE, plain.duck_release_ms
            ),
            emphasis_zoom_percent=_within(
                data.get("emphasis_zoom_percent"),
                EMPHASIS_ZOOM_PERCENT_RANGE,
                plain.emphasis_zoom_percent,
            ),
            emphasis_pop_percent=_within(
                data.get("emphasis_pop_percent"),
                EMPHASIS_POP_PERCENT_RANGE,
                plain.emphasis_pop_percent,
            ),
            emphasis_shake_px=_within(
                data.get("emphasis_shake_px"), EMPHASIS_SHAKE_PX_RANGE, plain.emphasis_shake_px
            ),
            emphasis_pan_px=_within(
                data.get("emphasis_pan_px"), EMPHASIS_PAN_PX_RANGE, plain.emphasis_pan_px
            ),
            emphasis_ramp_ms=_within(
                data.get("emphasis_ramp_ms"), EMPHASIS_RAMP_MS_RANGE, plain.emphasis_ramp_ms
            ),
            emphasis_easing=_choice(
                data.get("emphasis_easing"), EMPHASIS_EASINGS, plain.emphasis_easing
            ),
            seam_video=_choice(data.get("seam_video"), SEAM_VIDEOS, plain.seam_video),
            seam_crossfade_ms=_within(
                data.get("seam_crossfade_ms"), SEAM_CROSSFADE_MS_RANGE, plain.seam_crossfade_ms
            ),
            seam_zoom_percent=_within(
                data.get("seam_zoom_percent"), SEAM_ZOOM_PERCENT_RANGE, plain.seam_zoom_percent
            ),
            seam_audio_fade_ms=_within(
                data.get("seam_audio_fade_ms"), SEAM_AUDIO_FADE_MS_RANGE, plain.seam_audio_fade_ms
            ),
            scene_analysis=_choice(
                data.get("scene_analysis"), SCENE_ANALYSIS_MODES, plain.scene_analysis
            ),
            scene_threads=_member(
                data.get("scene_threads"), SCENE_THREAD_CHOICES, plain.scene_threads
            ),
            export_folder=_folder(data.get("export_folder"), plain.export_folder),
            ai_send_frames=_choice(data.get("ai_send_frames"), SEND_MODES, plain.ai_send_frames),
            ai_send_text=_choice(data.get("ai_send_text"), SEND_MODES, plain.ai_send_text),
            ai_send_files=_choice(data.get("ai_send_files"), SEND_MODES, plain.ai_send_files),
            ai_limit_turn_tokens=_within(
                data.get("ai_limit_turn_tokens"), AI_TOKEN_LIMITS, plain.ai_limit_turn_tokens
            ),
            ai_limit_turn_images=_within(
                data.get("ai_limit_turn_images"), AI_IMAGE_LIMITS, plain.ai_limit_turn_images
            ),
            ai_limit_turn_minutes=_within(
                data.get("ai_limit_turn_minutes"), AI_MINUTE_LIMITS, plain.ai_limit_turn_minutes
            ),
            ai_limit_day_tokens=_within(
                data.get("ai_limit_day_tokens"), AI_TOKEN_LIMITS, plain.ai_limit_day_tokens
            ),
            ai_limit_day_images=_within(
                data.get("ai_limit_day_images"), AI_IMAGE_LIMITS, plain.ai_limit_day_images
            ),
            ai_limit_action=_choice(
                data.get("ai_limit_action"), LIMIT_ACTIONS, plain.ai_limit_action
            ),
            ai_image_edge=_member(data.get("ai_image_edge"), IMAGE_EDGES, plain.ai_image_edge),
        )

    def save(self, preferences: Preferences) -> None:
        """この窓で変えた項目だけを、書いてある中身へ重ねて書く

        窓はそれぞれ起動したときに読んだ設定を持っている 丸ごと書き戻すと、ほかの窓が
        その後に変えた項目（テーマなど）を、この窓が別の項目（磁石など）を変えただけで
        黙って元へ戻す 変えたかどうかは、この窓が最後に読んだか書いた値（``_PREFERENCES_SEEN``）
        と比べて決める 同じ項目を両方の窓で変えたら、後から書いた方が残る

        読んでから書き戻すまでを錠で守る 守らないと、2 つの窓が同時に書いたときに、
        先に書いた方の変更を後の方が読まずに消す 書いてある中身が無いか壊れているなら、
        この窓の値を丸ごと書く（重ねる先が無い） 知らない項目（新しい版が足した物）は残す
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = _path_key(self.path)
        ours = asdict(preferences)
        seen = _PREFERENCES_SEEN.get(key)
        before = asdict(seen) if seen is not None else {}
        with held_for_update(self.path.with_name(self.path.name + ".lock")):
            data = self._read()
            if data is None:
                data = ours
            else:
                data.update(
                    {
                        name: value
                        for name, value in ours.items()
                        if name not in before or before[name] != value
                    }
                )
            write_atomic(self.path, json.dumps(data, ensure_ascii=False, indent=2))
        _PREFERENCES_SEEN[key] = preferences


#: この窓（プロセス）が最後に読んだか書いた設定 置き場ごと :meth:`PreferenceStore.save` が
#: 「この窓で変えた項目」を見分けるのに使う 窓は 1 つのプロセスに 1 つなので、プロセスに 1 つ持つ
_PREFERENCES_SEEN: dict[str, Preferences] = {}


def _path_key(path: Path) -> str:
    """置き場の比べ方 大文字小文字や ``..`` の違いで別の置き場と見なさない"""
    return os.path.normcase(str(Path(path).resolve()))


#: 吸い付く距離として受け付ける範囲（画面の画素）
SNAP_DISTANCES = (2, 40)


def _snap_distance(value: object, default: int) -> int:
    """吸い付く距離 範囲の外や壊れた値は既定へ戻す（0 では吸い付かず、大きすぎると動かせない）"""
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if SNAP_DISTANCES[0] <= value <= SNAP_DISTANCES[1] else default


#: 字幕の折り返しの幅として受け付ける範囲（画面の幅の %）
SUBTITLE_WRAP_PERCENTS = (20, 100)


def _wrap_percent(value: object, default: int) -> int:
    """字幕の折り返しの幅（%） 範囲の外は既定へ戻す

    狭すぎると 1 行に数文字しか入らず字幕が縦に積み上がり、100 を超えると画面からはみ出す
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    low, high = SUBTITLE_WRAP_PERCENTS
    return value if low <= value <= high else default


#: 細い帯にする幅として受け付ける範囲（画素） タイムラインもこの範囲へ丸める
#: 下は線 1 本ぶんの枠と名前の頭の 1 字が入る幅 それより細くすると、名前も絵も切れ端
#: だけが縞のように並んで読めない 上は 1 本ずつ描く本数を減らして軽くしたい人のため
#: 広すぎると拡大しても中身が出ない
DETAIL_MIN_WIDTHS = (8, 96)


def _detail_min_width(value: object, default: int) -> int:
    """細い帯にする幅 範囲の外や壊れた値は既定へ戻す

    0 を通すと、1 画素のクリップまで 1 本ずつ名前と絵を描いて重い
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if DETAIL_MIN_WIDTHS[0] <= value <= DETAIL_MIN_WIDTHS[1] else default


def _flag(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _style_guide(value: object, default: StyleGuide) -> StyleGuide:
    """新しい作品の編集の方針 形が違えば既定へ戻し、持てない所（手で書き換えた色・上限を越えた分）
    だけを落とす

    持てない所を残すと、新しい作品を作るたびに方針の窓で断られて OK を押せない 丸ごと既定へ
    戻すと、足しすぎた 1 つのために書いた方針が全部消える
    """
    if value is None:
        return default
    try:
        return style_guide_from_json(value).fitted()
    except ProjectFileError:
        return default


def _playhead_snap(value: object, default: bool) -> bool:
    """再生ヘッドの吸い付きの入切 前の版の 3 択の言葉も読む（:data:`_LEGACY_PLAYHEAD_SNAP`）"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _LEGACY_PLAYHEAD_SNAP.get(value, default)
    return default


def _choice(value: object, choices: tuple[str, ...], default: str) -> str:
    """決まった言葉のどれか 知らない言葉は既定へ戻す

    新しい版で足した表示を古い版で開いたときに、分からない値のまま当てると
    何も選ばれていない画面になる
    """
    return value if isinstance(value, str) and value in choices else default


def _loudness_target(value: object, default: int) -> int:
    """音量の基準 選択肢に無い値（手で書き換えた値・壊れた値）は既定へ戻す

    選択肢の外の値を通すと、設定画面のどれも選ばれていない形で開き、OK で黙って
    先頭の基準に書き換わる
    """
    choices = {target for target, _name in LOUDNESS_TARGETS}
    if isinstance(value, int) and not isinstance(value, bool) and value in choices:
        return value
    return default


def _size(value: object, default: int) -> int:
    """控えの高さ 極端な値は既定へ戻す

    0 や負だと控えが作れず、大きすぎると元の素材より重くなる
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if 120 <= value <= 2160 else default


def _budget(value: object, default: int) -> int:
    """先読みに使うメモリ（MB） 極端な値は既定へ戻す

    小さすぎると 1 枚も置けず、設定を入れたのに何も起きない
    大きすぎると GPU のメモリを使い切り、デコードや効果の側が確保に失敗する
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if MIN_PREFETCH_MB <= value <= MAX_PREFETCH_MB else default


def _depth(value: object, default: int) -> int:
    """書き出しのパイプラインの深さ 範囲の外は既定へ戻す

    負だと ``queue`` の作り方が変わって意味が通らず、大きすぎると合成済みの絵を
    溜め込むだけでメモリを食う（4K なら 1 枚 33MB） 0 は「直列」なので許す
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if 0 <= value <= MAX_PIPELINE_DEPTH else default


def _member(value: object, choices: tuple[int, ...], default: int) -> int:
    """決まった数のどれか 知らない数（手で書いた 3 など）と数でない物は既定へ戻す"""
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if value in choices else default


def _threads(value: object, default: int) -> int:
    """デコードの並列数 範囲の外は既定へ戻す

    0 や負だと走り係がスレッドを 1 本も作れず、先読みを頼んだ所で返らなくなる
    大きすぎてもデコーダの本数（:data:`MAX_DECODE_THREADS`）より相手がいない
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if 1 <= value <= MAX_DECODE_THREADS else default


def _divisor(value: object, default: int) -> int:
    """画面の分母 1・2・4 だけ 半端な値は合成の大きさが端数になる

    型も見る JSON は ``2.0`` と書けてしまい、``2.0 in (1, 2, 4)`` は真になる
    小数のまま通すと、描画先の大きさが小数になって型の食い違いで落ちる
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if value in (1, 2, 4) else default


#: 退避の間隔として受け付ける範囲（秒） 10 秒より短いと、編集の手を止めるたびに書き込みが走る
#: 10 分を超えると、落ちたときに失う作業が退避の無いのとほとんど変わらない
AUTOSAVE_SECONDS_RANGE = (10, 600)

#: バックアップの世代数として受け付ける範囲 0 は「作らない」と同じなので入り切りの側で持つ
#: 200 を超えると 1 本数百 KB でもプロジェクトごとに数十 MB になり、置き場を食う
BACKUP_GENERATIONS_RANGE = (1, 200)

#: 残った退避を片付けるまでの日数として受け付ける範囲 0 は片付けない
RECOVERY_KEEP_DAYS_RANGE = (0, 365)

#: 置き場の容量の上限として受け付ける範囲（MB） 0 は上限なし 上は 1TB
#: 小さくしても大事な物（今の退避と各プロジェクトのいちばん新しい控え）は消さないので、
#: 下は 1MB まで許す
STATE_LIMIT_MB_RANGE = (0, 1_000_000)


def _within(value: object, bounds: tuple[int, int], default: int) -> int:
    """範囲の中の整数 範囲の外や壊れた値は既定へ戻す（bool も整数として通さない）"""
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if bounds[0] <= value <= bounds[1] else default


def _text(value: object, default: str) -> str:
    """文字の値 文字でなければ既定へ戻す（壊れた値で起動を止めない）"""
    return value if isinstance(value, str) else default


def _folder(value: object, default: str) -> str:
    """置き場のフォルダ 絶対の場所だけ通す

    相対の場所は起動したときの作業フォルダで行き先が変わり、退避が毎回違う所へ散る
    """
    if not isinstance(value, str):
        return default
    if value == "":
        return value
    return value if Path(value).is_absolute() else default
