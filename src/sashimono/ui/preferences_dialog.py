"""本人の好みで変わる設定

プロジェクトの設定（解像度やフレームレート）とは分ける あちらは作品の持ち物で、
こちらは**その人とその機械**の持ち物 同じプロジェクトを速い機械で開いたら、
等倍で見たいことがある
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai.models import EFFORTS as AI_EFFORTS
from sashimono.ai.models import MODELS as AI_MODELS
from sashimono.ai.models import find_model
from sashimono.engine.cache.proxy import (
    BUDGET_MS,
    MEASURED_ONE_LAYER_MS,
    MEASURED_PREFETCH_MS,
    MEASURED_THREE_LAYERS_MS,
)
from sashimono.engine.encode import (
    MEASURED_DECODE_MS,
    MEASURED_EXPORT_MS,
    MEASURED_EXPORT_TOTAL_MS,
)
from sashimono.engine.render.background import MEASURED_PREFETCH_STALL_MS
from sashimono.engine.render.prefetch import BYTES_PER_FRAME_PIXEL
from sashimono.ui.media_match import MATCH_CHOICES
from sashimono.ui.media_pool import VIEW_ICONS, VIEW_LIST
from sashimono.ui.preview_handles import KEYFRAME_DRAG_CHOICES
from sashimono.ui.project_settings_dialog import LAYER_MODE_CHOICES
from sashimono.ui.theme import THEME_CHOICES
from sashimono.ui.workspace import (
    AUTO_QUALITY_HEIGHT,
    DETAIL_MIN_WIDTHS,
    DOCK_TABS_BOTTOM,
    DOCK_TABS_TOP,
    INSERT_ALL_TRACKS,
    INSERT_TARGET_TRACKS,
    MEDIA_SPLIT,
    MEDIA_TOGETHER,
    SCRIPTS_MOVE_ASK,
    SCRIPTS_MOVE_AUTO,
    SCRIPTS_MOVE_OFF,
    SNAP_DISTANCES,
    SUBTITLE_WRAP_PERCENTS,
    Preferences,
)

__all__ = [
    "AUTO_QUALITY_TEXT",
    "DECODE_THREADS",
    "PIPELINE_DEPTHS",
    "PREFETCH_BUDGETS",
    "PROXY_HEIGHTS",
    "QUALITY_DIVISORS",
    "PreferencesDialog",
]

#: 控えの大きさ 小さいほど軽いが、文字の読みやすさが落ちる
PROXY_HEIGHTS: tuple[tuple[str, int], ...] = (
    ("360p（一番軽い）", 360),
    ("540p（既定）", 540),
    ("720p（きれい）", 720),
)

#: 自動で画質を落とす項目の文言 数は判定（:meth:`Preferences.quality_for`）の境目から作る
#: 前は「画面より大きい素材では」と書いていたが、実際に見ているのは画面の大きさではなく
#: 読み込んだ素材の高さで、1080 を超える物が 1 つでもあれば落とす 文言と動きが食い違うと、
#: 1080p の画面で 1080p の素材を置いた人が「画面と同じなのに下がらない」と迷う
AUTO_QUALITY_TEXT = (
    f"高さが {AUTO_QUALITY_HEIGHT - 1} を超える素材があれば、プレビューの画質を下げる"
)

#: 自動で落とすときの分母
QUALITY_DIVISORS: tuple[tuple[str, int], ...] = (
    ("1/2 画質", 2),
    ("1/4 画質", 4),
)

#: 先読みに使うメモリ（MB） 1080p の 1 枚が 8MB
PREFETCH_BUDGETS: tuple[tuple[str, int], ...] = (
    ("512MB（控えめ）", 512),
    ("1GB（既定）", 1024),
    ("2GB", 2048),
    ("4GB（たくさん貯める）", 4096),
)

#: 書き出しで、合成を書き込みの何枚ぶん先へ進めるか
#: 1 枚は画面 1 枚の RGBA（1080p で 8MB、4K で 33MB）
PIPELINE_DEPTHS: tuple[tuple[str, int], ...] = (
    ("重ねない（1 枚ずつ）", 0),
    ("2 枚先まで（既定）", 2),
    ("4 枚先まで", 4),
)

#: 重ねたレイヤーの映像デコードを、同時にいくつまで走らせるか
#: 1 なら並べない 相手になるのは**別の素材**なので、重ねた枚数より多くしても効かない
DECODE_THREADS: tuple[tuple[str, int], ...] = (
    ("並べない（1 本ずつ）", 1),
    ("2 本まで", 2),
    ("4 本まで（既定）", 4),
    ("8 本まで", 8),
)


#: 目安を出すときの物差し **プロジェクトの解像度ではなく 1920x1080 で固定する**
#: 実際の 1 枚は画質の設定でも変わるが、設定を開く前から見当が付く数でないと
#: 「どれを選べばいいか」の助けにならない
FULL_HD_FRAME_BYTES = 1920 * 1080 * BYTES_PER_FRAME_PIXEL

#: 1GB に入る枚数と、30fps での秒数 設定の画面で目安として出す
PREFETCH_FRAMES_PER_GB = 1024 * 1024 * 1024 // FULL_HD_FRAME_BYTES
PREFETCH_SECONDS_PER_GB = PREFETCH_FRAMES_PER_GB // 30


def _megabytes(size: int) -> float:
    """バイトを MB へ 小数で返す 切り捨てると 7.9MB が 7MB になり、
    1GB に何枚入るかの見当が合わなくなる
    """
    return size / (1024 * 1024)


class PreferencesDialog(QDialog):
    """プレビューの重さに関わる設定を変える"""

    def _fit_to_screen(self, body: QWidget) -> None:
        """中身が収まる大きさで開く ただし画面の 9 割より大きくはしない（残りは巻物で見る）

        巻物にすると中身の大きさを窓が知らず、小さな窓で開く 横は項目が切れない幅にする
        """
        wanted = body.sizeHint()
        width = wanted.width() + self._scroll.verticalScrollBar().sizeHint().width() + 24
        height = wanted.height() + 64
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.9))
            height = min(height, int(available.height() * 0.9))
        self.resize(width, height)

    def __init__(self, preferences: Preferences, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("設定")

        form = QFormLayout()

        self._use_proxy = QCheckBox("プレビューに低解像度の控えを使う", self)
        self._use_proxy.setChecked(preferences.use_proxy)
        self._use_proxy.setToolTip(
            "大きい素材を読み込んだときに、裏で低解像度の控えを作って"
            "プレビューだけ差し替える 書き出しは必ず元の素材から行う"
        )
        form.addRow(self._use_proxy)

        self._proxy_height = QComboBox(self)
        for label, height in PROXY_HEIGHTS:
            self._proxy_height.addItem(label, height)
        self._select(self._proxy_height, preferences.proxy_height)
        form.addRow("控えの大きさ", self._proxy_height)

        self._auto_quality = QCheckBox(AUTO_QUALITY_TEXT, self)
        self._auto_quality.setChecked(preferences.auto_quality)
        self._auto_quality.setToolTip(
            f"読み込んだ素材のどれかの高さ（縦の画素数 縦撮りは回した後の高さ）が "
            f"{AUTO_QUALITY_HEIGHT - 1} を超えると、プレビューを「下げたときの画質」で描く "
            "書き出しは変わらない\n"
            "切ると、4K の素材でも等倍で描く 画質は上がるが、再生が追いつかなくなる"
        )
        form.addRow(self._auto_quality)

        self._auto_divisor = QComboBox(self)
        for label, divisor in QUALITY_DIVISORS:
            self._auto_divisor.addItem(label, divisor)
        self._select(self._auto_divisor, preferences.auto_quality_divisor)
        form.addRow("下げたときの画質", self._auto_divisor)

        self._prefetch = QCheckBox("手が止まっている間に、先のコマを描いておく", self)
        self._prefetch.setChecked(preferences.prefetch)
        self._prefetch.setToolTip(
            "編集していない間に、再生ヘッドの先を描いて取っておく 貯まった所は"
            "出すだけで済むので、重い所でも再生が止まらない 描いている間は"
            "GPU を使うので、ほかの作業が重くなるなら切る"
        )
        form.addRow(self._prefetch)

        self._prefetch_budget = QComboBox(self)
        for label, budget in PREFETCH_BUDGETS:
            self._prefetch_budget.addItem(label, budget)
        self._select(self._prefetch_budget, preferences.prefetch_budget_mb)
        form.addRow("先読みに使うメモリ", self._prefetch_budget)

        same_thread_ms, other_thread_ms = MEASURED_PREFETCH_STALL_MS
        self._prefetch_thread = QCheckBox("先読みを別のスレッドで描く（操作を止めない）", self)
        self._prefetch_thread.setChecked(preferences.prefetch_thread)
        self._prefetch_thread.setToolTip(
            "画面と同じスレッドで先読みすると、1 コマ描く間は操作を受け付けない "
            f"4K を 3 枚重ねて効果を積んだ所で、操作が {same_thread_ms}ms 待たされる所が、"
            f"別のスレッドなら {other_thread_ms}ms で済む（どちらも 95 パーセンタイル） "
            "GPU のドライバとの相性で先読みした絵が乱れるときは切る "
            "共有した GL を作れない機械では、入れたままでも画面のスレッドで先読みする"
        )
        form.addRow(self._prefetch_thread)

        self._pipeline_depth = QComboBox(self)
        for label, depth in PIPELINE_DEPTHS:
            self._pipeline_depth.addItem(label, depth)
        self._select(self._pipeline_depth, preferences.export_pipeline_depth)
        self._pipeline_depth.setToolTip(
            "書き出しで、GPU の合成と、CPU の色変換・エンコード・多重化を重ねて進める "
            "合成済みの絵を貯めるぶんメモリを使う（1080p で 1 枚 8MB、4K で 33MB）ので、"
            "深くすれば速いとは限らない 実測では 4K で 4 枚先まで貯めると 2 枚より遅くなった "
            "メモリが足りない機械では「重ねない」にする"
        )
        form.addRow("書き出しの先読み", self._pipeline_depth)

        self._decode_threads = QComboBox(self)
        for label, threads in DECODE_THREADS:
            self._decode_threads.addItem(label, threads)
        self._select(self._decode_threads, preferences.decode_threads)
        self._decode_threads.setToolTip(
            "重ねたクリップの映像デコードを、素材ごとに分けて同時に走らせる "
            "並べられるのは別の素材どうしだけなので、重ねた枚数より多くしても増えない "
            "デコード中の絵を素材の数だけ抱える（4K で 1 枚 33MB）ので、"
            "メモリの少ない機械では減らす 書き出しとプレビューの両方に効く"
        )
        form.addRow("レイヤーの並列デコード", self._decode_threads)

        self._smooth_audio_motion = QCheckBox(
            "リバーブ・ディレイ・音程のキーフレームをなめらかに動かす", self
        )
        self._smooth_audio_motion.setChecked(preferences.smooth_audio_motion)
        self._smooth_audio_motion.setToolTip(
            "前の音を読む音の効果は 0.34 秒ごとの区切りで掛ける 入れると区切りの中でも値を"
            "少しずつ移す 切ると区切りの頭の値のまま掛け、動きが最大 0.34 秒遅れて段になる "
            "値の動く所だけ手間が倍になるので、遅い機械で再生が途切れるときは切る "
            "書き出しとプレビューの両方に効く"
        )
        form.addRow(self._smooth_audio_motion)

        self._native_modules = QCheckBox("AviUtl2 のスクリプトモジュール（DLL）を読み込む", self)
        self._native_modules.setChecked(preferences.native_modules)
        self._native_modules.setToolTip(
            "テレビ字幕のように、処理を DLL に切り出した配布スクリプトを動かす "
            "合成フォントのように、AviUtl2 の汎用プラグイン（.aux2）が出す "
            "モジュールもここで読む 読み込むのはスクリプトフォルダに自分で置いた物と、"
            "AviUtl2 の Plugin フォルダの物だけ DLL は Sashimono と"
            "同じ権限で動くので、信頼できない物は置かない"
        )
        form.addRow(self._native_modules)

        self._pool_progress = QCheckBox("控えと解析の進み具合を、素材一覧の行にも出す", self)
        self._pool_progress.setChecked(preferences.pool_progress)
        self._pool_progress.setToolTip(
            "素材ごとに、控え（プロキシ）と波形・サムネイルを作っている途中の割合と、"
            "作れなかったことを行の後ろに添える 切ってもステータスバーには全体の"
            "進み具合が出て、作れなかった理由も終わったときに出る"
        )
        form.addRow(self._pool_progress)

        self._hdr_notice = QCheckBox("HDR の素材を読み込んだら知らせる", self)
        self._hdr_notice.setChecked(preferences.hdr_notice)
        self._hdr_notice.setToolTip(
            "HDR（PQ・HLG）や広い色域（BT.2020）の印が付いた素材を読み込んだときに、"
            "SDR（Rec.709）として扱うので白っぽく表示・書き出しされることを知らせる "
            "切っても素材一覧の行には印が付く"
        )
        form.addRow(self._hdr_notice)

        # 一覧の上のボタンでも切り替えられる ここにも置くのは、設定を開いて OK を
        # 押したときに、ボタンで選んだ表示を黙って既定へ戻さないため
        self._media_view = QComboBox(self)
        self._media_view.addItem("一覧（名前・長さ・大きさを 1 行ずつ）", VIEW_LIST)
        self._media_view.addItem("アイコン（サムネイルを並べる）", VIEW_ICONS)
        self._media_view.setCurrentIndex(max(0, self._media_view.findData(preferences.media_view)))
        self._media_view.setToolTip("素材一覧の上の「一覧」「アイコン」のボタンと同じ")
        form.addRow("素材一覧の表示", self._media_view)

        self._match_video = QComboBox(self)
        for value, text in MATCH_CHOICES:
            self._match_video.addItem(text, value)
        self._match_video.setCurrentIndex(
            max(0, self._match_video.findData(preferences.match_video))
        )
        self._match_video.setToolTip(
            "空のプロジェクトへ最初の動画を置いたとき、プロジェクトの解像度とフレームレートが"
            "動画と違えばどうするか フレームレートを変えられるのはタイムラインが空のときだけ"
        )
        form.addRow("最初の動画に合わせる", self._match_video)
        self._new_project_layers = QComboBox(self)
        for value, text in LAYER_MODE_CHOICES:
            self._new_project_layers.addItem(text, value)
        self._new_project_layers.setCurrentIndex(
            max(0, self._new_project_layers.findData(preferences.new_project_layers))
        )
        self._new_project_layers.setToolTip(
            "新しく作るプロジェクトと、起動した直後の空のプロジェクトの方式 "
            "新規作成の窓でもプロジェクトごとに選べる 開いたプロジェクトの方式は変えない"
        )
        form.addRow("新しいプロジェクトの置き方", self._new_project_layers)
        self._media_split = QComboBox(self)
        self._media_split.addItem("映像と音声を別のレイヤーに分ける（既定）", MEDIA_SPLIT)
        self._media_split.addItem("1 本のクリップにまとめる", MEDIA_TOGETHER)
        self._media_split.setCurrentIndex(
            max(0, self._media_split.findData(preferences.media_split))
        )
        self._media_split.setToolTip(
            "音声のある動画を置いたとき 分けると、置いたレイヤーに映像、その次のレイヤーから"
            "音声を 1 本ずつ並べ（ゲームの録画のマイクの声のように音声が何本あっても）、"
            "足りなければレイヤーを足す どれも一緒に動く（映像と音声のトラックを分ける方式では"
            "音声トラックを音声ごとに使う） まとめると映像と音声を 1 本のクリップで持ち、"
            "音声が 2 本以上あれば 1 本目だけを置く"
        )
        form.addRow("動画の映像と音声", self._media_split)
        self._dock_tabs = QComboBox(self)
        self._dock_tabs.addItem("上（既定）", DOCK_TABS_TOP)
        self._dock_tabs.addItem("下", DOCK_TABS_BOTTOM)
        self._dock_tabs.setCurrentIndex(max(0, self._dock_tabs.findData(preferences.dock_tabs)))
        self._dock_tabs.setToolTip(
            "オブジェクト設定と AI アシスタント、メディアと字幕のように、"
            "重ねたパネルを切り替えるタブをどちらの辺に出すか"
        )
        form.addRow("重ねたパネルのタブ", self._dock_tabs)
        self._theme = QComboBox(self)
        for value, text in THEME_CHOICES:
            self._theme.addItem(text, value)
        self._theme.setCurrentIndex(max(0, self._theme.findData(preferences.theme)))
        self._theme.setToolTip(
            "画面の色 OK を押すとその場で切り替わる（再起動は要らない） "
            "「Windows の設定に合わせる」は、Windows の 設定 → 個人用設定 → 色 の"
            "アプリの明るい・暗い（アプリ モード）に合わせ、変えたときも付いていく "
            "プレビューの絵そのもの（書き出す色）はテーマで変わらない"
        )
        form.addRow("画面の色（テーマ）", self._theme)
        self._preview_handles = QCheckBox("プレビューで外枠を出して直接動かす", self)
        self._preview_handles.setChecked(preferences.preview_handles)
        self._preview_handles.setToolTip(
            "選んだクリップの外枠をプレビューに出す 中をドラッグで位置、角で拡大率"
            "（Alt で縦横別々）、角の外で回転 Shift で縦か横の一方だけ・15 度刻み"
        )
        form.addRow(self._preview_handles)
        self._keyframe_drag = QComboBox(self)
        for value, text in KEYFRAME_DRAG_CHOICES:
            self._keyframe_drag.addItem(text, value)
        self._keyframe_drag.setCurrentIndex(
            max(0, self._keyframe_drag.findData(preferences.keyframe_drag))
        )
        form.addRow("キーフレームのある値を動かしたとき", self._keyframe_drag)
        self._preview_handles.toggled.connect(self._keyframe_drag.setEnabled)
        self._keyframe_drag.setEnabled(preferences.preview_handles)

        self._value_lines = QCheckBox("タイムラインのクリップに不透明度・音量の線を出す", self)
        self._value_lines.setChecked(preferences.value_lines)
        self._value_lines.setToolTip(
            "絵のクリップには不透明度、音のクリップには音量の線を引く 線を上下にドラッグで値、"
            "Ctrl+クリックでキーフレーム、点をドラッグで移動、点の右クリックで削除 "
            "音付きの動画は右クリックでどちらの線を出すか切り替える"
        )
        form.addRow(self._value_lines)
        self._detail_min_width = QSpinBox(self)
        self._detail_min_width.setRange(*DETAIL_MIN_WIDTHS)
        self._detail_min_width.setSuffix(" px")
        self._detail_min_width.setValue(preferences.detail_min_width)
        self._detail_min_width.setToolTip(
            "タイムラインでこれより細いクリップは、名前・サムネイル・波形を描かず細い帯にする "
            "細い帯には絵の平均の色と音の大きさだけを描き、マウスを載せると名前と長さが出る "
            "下げると引いた表示でも中身を細かく見られ、上げると短いクリップが多いときに描くのが軽い"
        )
        form.addRow("中身を描くクリップの最小の幅", self._detail_min_width)

        self._double_click_reset = QCheckBox("設定パネルの名前のダブルクリックで初期値に戻す", self)
        self._double_click_reset.setChecked(preferences.double_click_reset)
        self._double_click_reset.setToolTip(
            "オブジェクト設定の行の名前（数の値はスライダーも）をダブルクリックすると、"
            "その値を初期値に戻す キーフレームのある値は再生位置のキーだけを戻す "
            "戻しても取り消せる"
        )
        form.addRow(self._double_click_reset)

        self._timeline_snap = QCheckBox("タイムラインで近くの位置へ吸い付く（磁石）", self)
        self._timeline_snap.setChecked(preferences.timeline_snap)
        self._timeline_snap.setToolTip(
            "クリップを動かす・端を伸び縮みさせる・置くときに、ほかのクリップの頭と終わり・"
            "再生位置・キーフレーム・書き出し範囲の端へ吸い付く タイムラインの上の〔磁石〕と同じ "
            "動かしている途中で Shift を押している間は吸い付かない"
        )
        form.addRow(self._timeline_snap)
        self._snap_distance = QSpinBox(self)
        self._snap_distance.setRange(*SNAP_DISTANCES)
        self._snap_distance.setSuffix(" px")
        self._snap_distance.setValue(preferences.snap_distance)
        self._snap_distance.setToolTip("画面の画素で数える 拡大しても縮小しても同じ近さで吸い付く")
        form.addRow("吸い付く距離", self._snap_distance)
        self._preview_snap = QCheckBox("プレビューで位置を動かすときに吸い付く（磁石）", self)
        self._preview_snap.setChecked(preferences.preview_snap)
        self._preview_snap.setToolTip(
            "プレビューで絵を動かすときに、画面の端と中央・ほかの物の端と中央へ吸い付く"
            " タイムラインの磁石とは別に切れる 距離は上の「吸い付く距離」 "
            "動かしている途中で Shift を押している間は吸い付かない"
        )
        form.addRow(self._preview_snap)
        self._insert_paste = QComboBox(self)
        self._insert_paste.addItem("全トラック（既定）", INSERT_ALL_TRACKS)
        self._insert_paste.addItem("貼り先と、一緒に動く相手のトラックだけ", INSERT_TARGET_TRACKS)
        self._insert_paste.setCurrentIndex(
            max(0, self._insert_paste.findData(preferences.insert_paste))
        )
        self._insert_paste.setToolTip(
            "貼り付け（挿入 Ctrl+Shift+V）で、再生ヘッドから後ろを貼る長さぶん押し出すトラック "
            "全トラックなら、ほかのトラックの字幕や BGM とマーカーも一緒に押す "
            "貼り先だけなら、コピー元のトラックと、そこで押すクリップのリンクの相手・"
            "グループの仲間・焼き込んだ字幕のトラックだけを押す どちらもロックしたトラックに"
            "押す物があれば貼らずに止める"
        )
        form.addRow("挿入貼り付けで押し出す", self._insert_paste)

        self._all_plugins = QCheckBox("AviUtl2 の汎用プラグインを全部読んで探す", self)
        self._all_plugins.setChecked(preferences.all_aviutl_plugins)
        self._all_plugins.setToolTip(
            "切っている間は、スクリプトが引くモジュールを出すと分かっている汎用プラグイン"
            "（合成フォントの comfont.aux2）だけを読む "
            "入れると AviUtl2 の Plugin フォルダの .aux2 を全部読み、ほかのプラグインが"
            "出すモジュールも探す 読んだプラグインは初期化で自分の処理を走らせる"
            "（ウィンドウを作る、Python を起動して AviUtl2 の置き場へ書く、など）ので、"
            "要る物が無いときだけ入れる"
        )
        form.addRow(self._all_plugins)

        # アシスタントの欄の上でも選べる ここにも置くのは、設定を開いて OK を
        # 押したときに、欄の上で選んだモデルを黙って既定へ戻さないため
        self._ai_model = QComboBox(self)
        for model in AI_MODELS:
            self._ai_model.addItem(model.label, model.id)
        self._ai_model.setCurrentIndex(max(0, self._ai_model.findData(preferences.ai_model)))
        self._ai_model.setToolTip("「既定」は Claude Code がアカウントに合わせて選ぶモデル")
        form.addRow("アシスタントのモデル", self._ai_model)

        self._ai_effort = QComboBox(self)
        for effort in AI_EFFORTS:
            self._ai_effort.addItem(effort.label, effort.value)
        self._ai_effort.setCurrentIndex(max(0, self._ai_effort.findData(preferences.ai_effort)))
        self._ai_effort.setToolTip(
            "高くするほどよく考えてから答える代わりに、遅く、使う量も増える "
            "Claude Haiku 4.5 はこの指定を受け付けないので、選んでも渡さない"
        )
        form.addRow("アシスタントの考える深さ", self._ai_effort)
        # 受け付けないモデルでは選べなくする アシスタント欄の上と同じ振る舞い
        # 選べたままだと、変えても応答に何も効かない
        self._ai_model.currentIndexChanged.connect(self._update_effort_enabled)
        self._update_effort_enabled()

        self._chat_enter_sends = QCheckBox(
            "アシスタントの入力欄で Enter だけで送る（改行は Shift+Enter）", self
        )
        self._chat_enter_sends.setChecked(preferences.chat_enter_sends)
        self._chat_enter_sends.setToolTip(
            "切ると Ctrl+Enter で送り、Enter は改行になる 日本語の変換を確定する Enter では送らない"
        )
        form.addRow(self._chat_enter_sends)

        self._update_check = QCheckBox("起動したときに新しい版を確かめる", self)
        self._update_check.setChecked(preferences.update_check)
        self._update_check.setToolTip(
            "確かめるのは数時間に 1 回まで 見つけたら裏で落とし、署名と中身を確かめてから"
            "知らせる 繋がらないときは何も言わない 切ると今の版に留まる"
            "（ヘルプの〔更新を確かめる…〕で手で確かめられる）"
        )
        form.addRow(self._update_check)

        self._update_beta = QCheckBox("ベータ版も受け取る", self)
        self._update_beta.setChecked(preferences.update_beta)
        self._update_beta.setToolTip(
            "正式版より先に出す版 新しい機能を早く使える代わりに、作りが大きく変わることがある"
        )
        form.addRow(self._update_beta)

        self._update_confirm = QCheckBox("新しい版を入れる前に尋ねる", self)
        self._update_confirm.setChecked(preferences.update_confirm)
        self._update_confirm.setToolTip(
            "入れるには再起動が要る 切ると、落として確かめた版を、尋ねずに次の起動の頭で入れる"
            " どちらでも、編集や書き出しの途中で勝手に再起動はしない"
        )
        form.addRow(self._update_confirm)

        self._scripts_move = QComboBox(self)
        for value, text in (
            (SCRIPTS_MOVE_AUTO, "起動したときに自動で移す"),
            (SCRIPTS_MOVE_ASK, "移すかを尋ねる（同じ物は 1 度だけ）"),
            (SCRIPTS_MOVE_OFF, "何もしない"),
        ):
            self._scripts_move.addItem(text, value)
        self._scripts_move.setCurrentIndex(
            max(0, self._scripts_move.findData(preferences.scripts_move))
        )
        self._scripts_move.setToolTip(
            "配布版の Sashimono.exe の隣の scripts に自分で置いた物を、"
            "%APPDATA%\\Sashimono\\scripts へ移すか 自動更新では新しい版へ写すが、"
            "zip を手で展開し直してフォルダごと入れ替えると消える 移し先に同じ名前の物があれば"
            "上書きせずに残し、移した物と残した物を 1 度知らせる"
            " どれを選んでも〔互換〕→〔exe の隣のスクリプトを移す…〕から移せる"
        )
        form.addRow("exe の隣の scripts に置いた物", self._scripts_move)

        self._subtitle_wrap = QCheckBox("新しい字幕を画面の幅で自動で折り返す", self)
        self._subtitle_wrap.setChecked(preferences.subtitle_wrap)
        self._subtitle_wrap.setToolTip(
            "字幕の〔焼き込み〕で作るテキストに、画面の幅に合わせた折り返しの幅を入れる"
            " 長い字幕が画面の端で切れず、字の大きさや書体を変えても折り返し直す"
            " 句読点や閉じ括弧を行の頭に置かず、英単語は途中で切らない"
            " 既にあるテキストは変えない（テキストの設定の「折り返しの幅」で 1 本ずつ変えられる）"
            " 字幕の整形で入れた改行はそのまま残し、それでも幅を超える行だけをさらに折り返す"
        )
        form.addRow(self._subtitle_wrap)
        self._subtitle_wrap_percent = QSpinBox(self)
        self._subtitle_wrap_percent.setRange(*SUBTITLE_WRAP_PERCENTS)
        self._subtitle_wrap_percent.setSuffix(" %")
        self._subtitle_wrap_percent.setValue(preferences.subtitle_wrap_percent)
        self._subtitle_wrap_percent.setToolTip(
            "画面の幅の何 % で折り返すか 90 で左右に 5% ずつ余白が残り、縁取りや影が端に掛からない"
        )
        form.addRow("字幕を折り返す幅", self._subtitle_wrap_percent)
        self._subtitle_wrap.toggled.connect(self._subtitle_wrap_percent.setEnabled)
        self._subtitle_wrap_percent.setEnabled(preferences.subtitle_wrap)

        # 測った値をそのまま置く 「なんとなく軽くなる」ではなく、
        # どの組が 60fps に入るのかを見て選べるようにする
        # 数は控えの側（sashimono.engine.cache.proxy）から取る ここへ直に書くと、
        # 測り直したときに画面の側だけ古くなる
        plain, proxied, both = MEASURED_THREE_LAYERS_MS
        filling, showing = MEASURED_PREFETCH_MS
        compose, readback, convert, muxing = MEASURED_EXPORT_MS
        serial_ms, pipelined_ms = MEASURED_EXPORT_TOTAL_MS
        one_thread, many_threads = MEASURED_DECODE_MS
        note = QLabel(
            f"4K を 3 枚重ねたときの実測（1 コマ {BUDGET_MS:.1f}ms が 60fps の目安）\n"
            f"元のまま {plain}ms ／ 控えを使う {proxied}ms ／ さらに画質を下げる {both}ms\n"
            "効果を積むと余裕が減る ぼかしまでなら控えのままでぎりぎり入り、"
            "画質も下げると余裕が出る ぼかしと発光まで積むとどの組でも入らない"
            f"（4K を 1 枚置いただけなら、元のままでも {MEASURED_ONE_LAYER_MS}ms で収まる）\n"
            "この機械で測るには tools\\bench_proxy.py\n"
            f"先読みは 1 枚 {_megabytes(FULL_HD_FRAME_BYTES):.1f}MB（1920x1080）"
            f" 1GB でおよそ {PREFETCH_FRAMES_PER_GB} 枚＝{PREFETCH_SECONDS_PER_GB} 秒ぶん\n"
            f"貯めるのに 1 枚 {filling}ms 掛かる代わりに、貯まった所は {showing}ms で出せる"
            "（上と同じ 4K 3 枚 + blur）\n"
            f"書き出しの内訳は 1 枚あたり 合成 {compose}ms ／ 読み戻し {readback}ms ／"
            f" 色変換 {convert}ms ／ エンコード + 多重化 {muxing}ms"
            "（1920x1080 を 3 枚重ね、NVIDIA GPU）\n"
            f"後ろの 2 つを重ねると、書き出し全体で 1 枚 {serial_ms:.1f}ms の所が"
            f" {pipelined_ms:.1f}ms になる\n"
            f"レイヤーを並べてデコードすると、同じ素材の合成が 1 枚 {one_thread:.1f}ms の所が"
            f" {many_threads:.1f}ms になる この機械で測るには tools\\bench_export.py",
            self,
        )
        note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # 項目は縦に長く、ノートの画面では下の項目と OK が画面の外へはみ出していた
        # 項目と説明だけを巻物にし、OK と取り消しはいつも見える下に置く
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.addLayout(form)
        body_layout.addWidget(note)
        body_layout.addStretch(1)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(body)

        layout = QVBoxLayout(self)
        layout.addWidget(self._scroll, 1)
        layout.addWidget(buttons)
        self._fit_to_screen(body)

        self._use_proxy.toggled.connect(self._proxy_height.setEnabled)
        self._prefetch.toggled.connect(self._prefetch_budget.setEnabled)
        self._prefetch_budget.setEnabled(preferences.prefetch)
        self._prefetch.toggled.connect(self._prefetch_thread.setEnabled)
        self._prefetch_thread.setEnabled(preferences.prefetch)
        self._auto_quality.toggled.connect(self._auto_divisor.setEnabled)
        self._proxy_height.setEnabled(preferences.use_proxy)
        self._auto_divisor.setEnabled(preferences.auto_quality)
        # DLL を読まない設定では汎用プラグインも読まないので、選んでも効かない
        self._native_modules.toggled.connect(self._all_plugins.setEnabled)
        self._all_plugins.setEnabled(preferences.native_modules)

    def _update_effort_enabled(self, _index: int = -1) -> None:
        choice = find_model(str(self._ai_model.currentData()))
        self._ai_effort.setEnabled(choice is None or choice.effort)

    @staticmethod
    def _select(box: QComboBox, value: int) -> None:
        """その値の項目を選ぶ 一覧に無ければ、その値の項目を足してから選ぶ

        設定ファイルを手で書き換えた人が、一覧に無い値を入れていることがある
        先頭のままにすると、設定を開いて OK を押しただけで**黙って別の値に
        置き換わる** 触っていない項目が変わるのは、壊したのと同じ
        """
        index = box.findData(value)
        if index < 0:
            box.addItem(f"{value}（設定ファイルの値）", value)
            index = box.findData(value)
        box.setCurrentIndex(index)

    def preferences(self) -> Preferences:
        """画面で選ばれた設定"""
        return Preferences(
            use_proxy=self._use_proxy.isChecked(),
            proxy_height=int(self._proxy_height.currentData()),
            auto_quality=self._auto_quality.isChecked(),
            auto_quality_divisor=int(self._auto_divisor.currentData()),
            prefetch=self._prefetch.isChecked(),
            prefetch_budget_mb=int(self._prefetch_budget.currentData()),
            prefetch_thread=self._prefetch_thread.isChecked(),
            export_pipeline_depth=int(self._pipeline_depth.currentData()),
            decode_threads=int(self._decode_threads.currentData()),
            smooth_audio_motion=self._smooth_audio_motion.isChecked(),
            native_modules=self._native_modules.isChecked(),
            pool_progress=self._pool_progress.isChecked(),
            hdr_notice=self._hdr_notice.isChecked(),
            all_aviutl_plugins=self._all_plugins.isChecked(),
            media_view=str(self._media_view.currentData()),
            ai_model=str(self._ai_model.currentData()),
            ai_effort=str(self._ai_effort.currentData()),
            chat_enter_sends=self._chat_enter_sends.isChecked(),
            match_video=str(self._match_video.currentData()),
            dock_tabs=str(self._dock_tabs.currentData()),
            preview_handles=self._preview_handles.isChecked(),
            keyframe_drag=str(self._keyframe_drag.currentData()),
            value_lines=self._value_lines.isChecked(),
            detail_min_width=self._detail_min_width.value(),
            double_click_reset=self._double_click_reset.isChecked(),
            timeline_snap=self._timeline_snap.isChecked(),
            snap_distance=self._snap_distance.value(),
            preview_snap=self._preview_snap.isChecked(),
            insert_paste=str(self._insert_paste.currentData()),
            new_project_layers=str(self._new_project_layers.currentData()),
            media_split=str(self._media_split.currentData()),
            theme=str(self._theme.currentData()),
            update_check=self._update_check.isChecked(),
            update_beta=self._update_beta.isChecked(),
            update_confirm=self._update_confirm.isChecked(),
            scripts_move=str(self._scripts_move.currentData()),
            subtitle_wrap=self._subtitle_wrap.isChecked(),
            subtitle_wrap_percent=self._subtitle_wrap_percent.value(),
        )
