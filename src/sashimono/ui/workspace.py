"""画面配置とショートカットの保存先

どちらも「その人の使い方」で、プロジェクトではなく本人に付く ``%APPDATA%\\Sashimono``
に置くのはそのため（キャッシュや退避と違い、消えると作り直せない）

Qt の既定の保存先（Windows ではレジストリ）は使わない 自動更新で本体を入れ替えても
残ること、手で消したり他の機械へ持っていったりできることを優先して、ファイルにする
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtWidgets import QMainWindow, QTabWidget

from sashimono.ai.models import DEFAULT_EFFORT as DEFAULT_AI_EFFORT
from sashimono.ai.models import DEFAULT_MODEL as DEFAULT_AI_MODEL
from sashimono.ai.models import EFFORTS as AI_EFFORTS
from sashimono.ai.models import MODELS as AI_MODELS
from sashimono.core import userdirs
from sashimono.core.model import LayerMode
from sashimono.engine.encode import DEFAULT_PIPELINE_DEPTH, MAX_PIPELINE_DEPTH
from sashimono.engine.render import DEFAULT_DECODE_THREADS, MAX_DECODE_THREADS
from sashimono.ui.media_match import MATCH_ASK, MATCH_MODES
from sashimono.ui.media_pool import VIEW_LIST, VIEW_MODES
from sashimono.ui.preview_handles import KEYFRAME_DRAG_AT_PLAYHEAD, KEYFRAME_DRAG_MODES
from sashimono.ui.theme import THEME_DARK, THEME_MODES

__all__ = [
    "AUTO_QUALITY_HEIGHT",
    "DETAIL_MIN_WIDTHS",
    "DOCK_TABS_BOTTOM",
    "DOCK_TABS_TOP",
    "DOCK_TAB_POSITIONS",
    "LAYOUT_VERSION",
    "MAX_PREFETCH_MB",
    "MEDIA_SPLIT",
    "MEDIA_SPLIT_MODES",
    "MEDIA_TOGETHER",
    "MIN_PREFETCH_MB",
    "SCRIPTS_MOVE_ASK",
    "SCRIPTS_MOVE_AUTO",
    "SCRIPTS_MOVE_MODES",
    "SCRIPTS_MOVE_OFF",
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".writing")
        temporary.write_text(
            json.dumps(dict(sorted(overrides.items())), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


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

#: 前の版の設定の名前（``multi_audio`` 音声が 2 本以上ある動画だけの置き方）の値と、
#: 今の値の対応 分けない（``first``）を選んでいた人は、音声が 1 本の動画も分けない側へ写す
#: 読まないと、分けないと決めていた人の置き方が黙って分ける側へ変わる
_LEGACY_MULTI_AUDIO = {"split": MEDIA_SPLIT, "first": MEDIA_TOGETHER}

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
    #: 設定パネルで、行の名前（数はスライダーも）のダブルクリックで値を初期値へ戻す
    #: 既定は入（利用者の要望） 初期値を覚えていなくても戻せ、戻しても取り消せる
    #: 行の名前を続けて押しがちで、うっかり戻るのが嫌な人は切れるようにする
    double_click_reset: bool = True
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
    #: 減らすと、次にそのプロジェクトを上書き保存したときに古い物から消える
    #: （設定画面で消える数を見せて確かめる）
    backup_generations: int = 20
    #: 退避とバックアップの置き場 空は既定（``%LOCALAPPDATA%\Sashimono``）
    #: 既定を空で持つのは、ユーザー名やドライブが変わった機械でも既定の置き場を指し続けるため
    #: 書けない所を選んでいたら、その起動は既定の置き場へ戻して知らせる（黙って退避を止めない）
    state_folder: str = ""

    @property
    def subtitle_wrap_share(self) -> int:
        """字幕の折り返しの幅（画面の幅の %） 切ってあれば 0（折り返さない）"""
        return self.subtitle_wrap_percent if self.subtitle_wrap else 0

    @property
    def inserts_on_all_tracks(self) -> bool:
        """挿入貼り付け（:func:`~sashimono.core.clipboard.insert_paste_commands`）へ渡す値"""
        return self.insert_paste == INSERT_ALL_TRACKS

    @property
    def splits_media(self) -> bool:
        """素材を置く所（:func:`~sashimono.core.commands.insert_media`）へ渡す値"""
        return self.media_split == MEDIA_SPLIT

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
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Preferences()
        if not isinstance(data, dict):
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
            smooth_audio_motion=_flag(data.get("smooth_audio_motion"), plain.smooth_audio_motion),
            double_click_reset=_flag(data.get("double_click_reset"), plain.double_click_reset),
            wheel_unfocused=_flag(data.get("wheel_unfocused"), plain.wheel_unfocused),
            timeline_snap=_flag(data.get("timeline_snap"), plain.timeline_snap),
            snap_distance=_snap_distance(data.get("snap_distance"), plain.snap_distance),
            preview_snap=_flag(data.get("preview_snap"), plain.preview_snap),
            insert_paste=_choice(data.get("insert_paste"), INSERT_PASTE_MODES, plain.insert_paste),
            new_project_layers=_choice(
                data.get("new_project_layers"), LayerMode.ALL, plain.new_project_layers
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
            autosave=_flag(data.get("autosave"), plain.autosave),
            autosave_seconds=_within(
                data.get("autosave_seconds"), AUTOSAVE_SECONDS_RANGE, plain.autosave_seconds
            ),
            backup=_flag(data.get("backup"), plain.backup),
            backup_generations=_within(
                data.get("backup_generations"), BACKUP_GENERATIONS_RANGE, plain.backup_generations
            ),
            state_folder=_folder(data.get("state_folder"), plain.state_folder),
        )

    def save(self, preferences: Preferences) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".writing")
        temporary.write_text(
            json.dumps(asdict(preferences), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)


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


def _choice(value: object, choices: tuple[str, ...], default: str) -> str:
    """決まった言葉のどれか 知らない言葉は既定へ戻す

    新しい版で足した表示を古い版で開いたときに、分からない値のまま当てると
    何も選ばれていない画面になる
    """
    return value if isinstance(value, str) and value in choices else default


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


def _within(value: object, bounds: tuple[int, int], default: int) -> int:
    """範囲の中の整数 範囲の外や壊れた値は既定へ戻す（bool も整数として通さない）"""
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value if bounds[0] <= value <= bounds[1] else default


def _folder(value: object, default: str) -> str:
    """置き場のフォルダ 絶対の場所だけ通す

    相対の場所は起動したときの作業フォルダで行き先が変わり、退避が毎回違う所へ散る
    """
    if not isinstance(value, str):
        return default
    if value == "":
        return value
    return value if Path(value).is_absolute() else default
