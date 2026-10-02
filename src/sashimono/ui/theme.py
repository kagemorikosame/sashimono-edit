"""UI の色と寸法

既定は暗色 編集ソフトは長時間見続けるものなので、明るい背景だと映像の色を
判断するときに目が順応してしまい、プレビューの見え方が変わる
明るい部屋で使う人・暗い画面が読みにくい人もいるので、明るいテーマも選べる
（:attr:`~sashimono.ui.workspace.Preferences.theme`）

色は :class:`Colors` の 1 か所にまとめ、テーマを切り替えると**同じ QColor の中身を
書き換える** 新しい QColor に差し替えると、どこかで先に受け取った色（トラックの
ボタンの表や、設定パネルの帯の色）だけが前のテーマのまま残る 描く所は描くたびに
``Colors`` を読むので、切り替えたあとに描き直せば再起動なしで新しい色になる
文字列へ焼き込む物（部品ごとのスタイルシート）は :func:`themed_style` で当て、
切り替えのたびに作り直す
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import shiboken6
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.core.commands.edit import DEFAULT_TRACK_HEIGHT, MAX_TRACK_HEIGHT, MIN_TRACK_HEIGHT
from sashimono.resources import (
    MAGNET_ICONS,
    MAGNET_ICONS_LIGHT,
    SPIN_ARROWS,
    SPIN_ARROWS_LIGHT,
    path_to,
)

__all__ = [
    "PALETTES",
    "THEME_CHOICES",
    "THEME_DARK",
    "THEME_LIGHT",
    "THEME_MODES",
    "THEME_SYSTEM",
    "Colors",
    "Metrics",
    "apply_theme",
    "current_theme",
    "follow_system",
    "magnet_icons",
    "resolve_theme",
    "style_sheet",
    "theme_signals",
    "themed_style",
    "use_palette",
]

#: テーマの選び方 :attr:`~sashimono.ui.workspace.Preferences.theme` の値
THEME_DARK = "dark"
THEME_LIGHT = "light"
#: Windows の「アプリ モード」（明るい・暗い）に合わせる 分からない機械では暗い側
THEME_SYSTEM = "system"
THEME_MODES = (THEME_DARK, THEME_LIGHT, THEME_SYSTEM)

#: 設定画面に出す言葉 既定を先頭に置く
THEME_CHOICES: tuple[tuple[str, str], ...] = (
    (THEME_DARK, "暗い（既定）"),
    (THEME_LIGHT, "明るい"),
    (THEME_SYSTEM, "Windows の設定に合わせる"),
)


class Colors:
    """配色 値はすべて sRGB ここに書いてあるのは暗いテーマの値

    明るいテーマの値は :data:`_LIGHT` 色を足すときは両方に書く
    （片方だけだと試験が落ちる 足し忘れた色だけ、切り替えても前のテーマのまま残る）
    """

    WINDOW = QColor("#1b1b1e")
    PANEL = QColor("#232327")
    PANEL_ALT = QColor("#26262b")
    BORDER = QColor("#3a3a40")
    TEXT = QColor("#d7d7db")
    TEXT_MUTED = QColor("#8b8b93")
    ACCENT = QColor("#7f8cf0")
    #: アクセント色の地に載せる文字（一覧の選んだ行・押している間のボタン・メニューの選んだ項目）
    #: 普通の文字の色のままだと、暗いテーマではアクセントの地との差が 2:1 ほどしかなかった
    ACCENT_TEXT = QColor("#12121a")
    #: 選んでいるタブの地 選んでいないタブ（窓の地）より一段明るく、上の文字が読める暗さ
    TAB_SELECTED = QColor("#34343c")
    #: 補足（ツールチップ）の地 窓の地より明るくして、下の部品と重なっても浮いて見せる
    TOOL_TIP = QColor("#3a3a44")
    #: 注意の文言（プロジェクト設定の「変えると崩れる」など） 赤みのある橙
    WARNING = QColor("#e07a5f")

    #: プレビューの周囲 映像の明るさを判断しやすいよう、真っ黒より少し上げる
    VIEWER_BACKGROUND = QColor("#0f0f11")
    #: 素材一覧のサムネイルの周りの地 映像の黒と見分けが付くよう、真っ黒より少し上げる
    MEDIA_BACKDROP = QColor("#111114")

    TIMELINE_BACKGROUND = QColor("#191a1d")
    TIMELINE_RULER = QColor("#1f1f23")
    TRACK_HEADER = QColor("#202024")
    TRACK_SEPARATOR = QColor("#2c2c33")
    #: 素材を引いてきた間に出す、新しく作るトラックの仮の行 本物のトラックと見分けが
    #: 付くよう、地の明るさを少しだけ変える（暗いテーマは明るく、明るいテーマは暗く）
    GHOST_ROW = QColor(255, 255, 255, 14)

    #: 再生ヘッド 素材の色と被らない色にする
    PLAYHEAD = QColor("#ff5c5c")

    VIDEO_CLIP = QColor("#33445f")
    VIDEO_CLIP_BORDER = QColor("#5b7bb0")
    AUDIO_CLIP = QColor("#26443a")
    AUDIO_CLIP_BORDER = QColor("#4c8a68")
    #: フィルタのクリップ（下のトラックの絵全体に掛かる） 映像のクリップと同じ色だと、
    #: 絵を持つクリップと取り違えて「消しても何も減らない」「動かしたら下の色が変わった」になる
    #: 映像（青）とも音声（緑）とも再生ヘッド（赤）とも離れた紫にする
    FILTER_CLIP = QColor("#4a3560")
    FILTER_CLIP_BORDER = QColor("#9a72c8")
    WAVEFORM = QColor("#7fd6ab")
    CLIP_LABEL = QColor("#e8eefc")
    #: クリップの名前の帯 クリップの地を少し沈めて、名前を地の模様（波形・絵）から浮かせる
    CLIP_LABEL_SHADE = QColor(0, 0, 0, 90)
    SELECTION = QColor("#ffffff")
    #: 磁石で吸い付いた所に一瞬出す縦の線 再生ヘッド（赤）・選択（白）・書き出し範囲（黄緑）・
    #: 設定パネルの印（紫がかった青）のどれとも取り違えない明るい黄色
    SNAP_LINE = QColor("#ffe45c")
    #: 選んだうち、オブジェクト設定が今出しているクリップ 選んだ印（白）の内側に引く
    #: 窓の中で「いま触っている所」を示す色（選んだタブ・一覧の選んだ行）と同じにする
    #: 別の色にすると、同じ意味の印が画面の場所ごとに違って見える
    #: **同じ QColor を指す** テーマを切り替えたときに、アクセントと一緒に変わる
    EDITING = ACCENT

    #: クリップの上のキーフレームのひし形 選んでいないクリップの印は控えめにし、選んだ
    #: クリップの印だけ明るくする 全部が明るいと、選んだクリップの印が埋もれる
    #: 縁は暗くし、どの色のクリップの上でも形が読めるようにする
    KEYFRAME = QColor("#c9a640")
    KEYFRAME_SELECTED = QColor("#ffd84a")
    KEYFRAME_OUTLINE = QColor("#141417")

    #: クリップの上に引く値の線 不透明度（絵）と音量（音） 下に影を敷いて、
    #: サムネイルや波形の上でも線が途切れて見えないようにする
    OPACITY_LINE = QColor("#f0f0f4")
    VOLUME_LINE = QColor("#7de39c")
    VALUE_LINE_SHADOW = QColor(0, 0, 0, 150)

    #: トラックヘッダの切り替えボタン（押している間の色）
    #: 3 つとも違う色にする 同じ色だと、どれが効いているかを文字で読むことになる
    TRACK_MUTE = QColor("#c9563f")
    TRACK_SOLO = QColor("#d8b23a")
    TRACK_LOCK = QColor("#6c7a91")
    #: 押している間のボタンの文字 上の 3 色の地に載る
    TRACK_TOGGLE_TEXT = QColor("#1b1b1e")

    #: 書き出し範囲 目盛りの上の帯と、トラックに重ねる薄い色
    #: 再生ヘッド（赤）・クリップ（青と緑）・フィルタ（紫）・選択（白）のどれとも
    #: 取り違えない黄緑にする
    WORK_AREA = QColor(150, 200, 90, 150)
    WORK_AREA_TINT = QColor(150, 200, 90, 28)
    WORK_AREA_EDGE = QColor("#a6d65a")


def _tokens() -> dict[str, QColor]:
    """``Colors`` の色の名前と QColor 同じ QColor を指す別名（``EDITING``）は 1 度だけ数える"""
    found: dict[str, QColor] = {}
    seen: set[int] = set()
    for name, value in vars(Colors).items():
        if isinstance(value, QColor) and id(value) not in seen:
            seen.add(id(value))
            found[name] = value
    return found


#: 暗いテーマの値 起動したときの ``Colors`` の写し 明るいテーマから戻すときに使う
_DARK: dict[str, QColor] = {name: QColor(value) for name, value in _tokens().items()}

#: 明るいテーマの値
#:
#: 暗いテーマの色をそのまま反転させない 意味ごとに色相をそろえたまま（映像は青・音声は緑・
#: フィルタは紫・再生ヘッドは赤・磁石は黄・書き出し範囲は黄緑）、明るい地の上で読める濃さへ
#: 下げる 文字と地の比は試験（tests/ui/test_theme.py）が WCAG の式で見る
#:
#: 白い印（選択の枠）は明るい地に溶けるので黒にする プレビューの周りは真っ白にせず中くらいの
#: 灰色にする 映像の周りが明るすぎると、目が明るさに慣れて映像が暗く見える
_LIGHT: dict[str, QColor] = {
    "WINDOW": QColor("#f2f2f5"),
    "PANEL": QColor("#e6e6eb"),
    "PANEL_ALT": QColor("#fbfbfd"),
    "BORDER": QColor("#b9b9c3"),
    "TEXT": QColor("#1c1c21"),
    "TEXT_MUTED": QColor("#5a5a64"),
    "ACCENT": QColor("#4453c9"),
    "ACCENT_TEXT": QColor("#ffffff"),
    "TAB_SELECTED": QColor("#ffffff"),
    "TOOL_TIP": QColor("#ffffff"),
    "WARNING": QColor("#b23c1e"),
    "VIEWER_BACKGROUND": QColor("#d6d6db"),
    "MEDIA_BACKDROP": QColor("#d4d4da"),
    "TIMELINE_BACKGROUND": QColor("#e9e9ee"),
    "TIMELINE_RULER": QColor("#dedee4"),
    "TRACK_HEADER": QColor("#e2e2e8"),
    "TRACK_SEPARATOR": QColor("#cbcbd3"),
    "GHOST_ROW": QColor(0, 0, 0, 16),
    "PLAYHEAD": QColor("#e0303a"),
    "VIDEO_CLIP": QColor("#a8bce4"),
    "VIDEO_CLIP_BORDER": QColor("#3e5f9e"),
    "AUDIO_CLIP": QColor("#a6d8c0"),
    "AUDIO_CLIP_BORDER": QColor("#2f7d57"),
    "FILTER_CLIP": QColor("#cdb6ea"),
    "FILTER_CLIP_BORDER": QColor("#7247a8"),
    "WAVEFORM": QColor("#1d6b47"),
    "CLIP_LABEL": QColor("#14182a"),
    "CLIP_LABEL_SHADE": QColor(255, 255, 255, 110),
    "SELECTION": QColor("#111114"),
    "SNAP_LINE": QColor("#a87000"),
    "KEYFRAME": QColor("#a8800f"),
    "KEYFRAME_SELECTED": QColor("#ffc400"),
    "KEYFRAME_OUTLINE": QColor("#141417"),
    "OPACITY_LINE": QColor("#1c1c21"),
    "VOLUME_LINE": QColor("#0d5a32"),
    "VALUE_LINE_SHADOW": QColor(255, 255, 255, 170),
    "TRACK_MUTE": QColor("#b5432d"),
    "TRACK_SOLO": QColor("#9a7408"),
    "TRACK_LOCK": QColor("#55637a"),
    "TRACK_TOGGLE_TEXT": QColor("#ffffff"),
    "WORK_AREA": QColor(70, 140, 20, 150),
    "WORK_AREA_TINT": QColor(70, 140, 20, 30),
    "WORK_AREA_EDGE": QColor("#4f8f17"),
}

#: テーマ → 色 試験も読む
PALETTES: dict[str, dict[str, QColor]] = {THEME_DARK: _DARK, THEME_LIGHT: _LIGHT}

#: いま当たっているテーマ（``dark`` か ``light``） 選び方（``system`` を含む）とは別に持つ
_current = THEME_DARK
#: 本人が選んだ選び方 Windows の設定が変わったときに、合わせ直すかを決めるのに使う
_mode = THEME_DARK


def current_theme() -> str:
    """いま当たっているテーマ ``dark`` か ``light``"""
    return _current


def resolve_theme(mode: str, scheme: Qt.ColorScheme | None = None) -> str:
    """選び方から実際のテーマへ ``system`` は Windows の設定（``scheme``）を見る

    ``scheme`` を渡さなければアプリから読む 読めない（``Unknown`` 画面の無い試験や
    古い Windows）ときは暗い側にする 何も設定しない人が見てきた見た目のままにするため
    """
    if mode == THEME_LIGHT:
        return THEME_LIGHT
    if mode != THEME_SYSTEM:
        return THEME_DARK
    if scheme is None:
        application = QGuiApplication.instance()
        if not isinstance(application, QGuiApplication):
            return THEME_DARK
        scheme = application.styleHints().colorScheme()
    return THEME_LIGHT if scheme == Qt.ColorScheme.Light else THEME_DARK


def use_palette(theme: str) -> None:
    """``Colors`` の中身をそのテーマの値へ書き換える 画面には何もしない

    QColor を差し替えず中身を書き換えるのは、先に受け取って持っている所
    （:data:`~sashimono.ui.timeline.painter.TRACK_BUTTONS` など）まで一緒に変えるため
    """
    global _current
    palette = PALETTES.get(theme, _DARK)
    tokens = _tokens()
    for name, value in palette.items():
        tokens[name].setRgba(value.rgba())
    _current = theme if theme in PALETTES else THEME_DARK


class Metrics:
    """寸法"""

    #: トラックの名前と M S L のボタンを 1 行に並べる幅 名前に使えるのは 84 画素
    #: 「レイヤー 100」が 10pt の Yu Gothic UI・メイリオで 77 画素 132 のときは 56 画素しか無く、
    #: 「レイヤー 1」が「レイ… 1」に切れて何のトラックか読めなかった（#27 P4b）
    #: 名前とボタンを 2 行に分けないのは、最小の高さ（28）でボタンが帯からはみ出すため
    TRACK_HEADER_WIDTH = 160
    RULER_HEIGHT = 22
    DEFAULT_TRACK_HEIGHT = DEFAULT_TRACK_HEIGHT
    #: トラックの高さの範囲は、変えるコマンドと同じ値を使う 別々に持つと、
    #: 描画では収まっているのに保存すると高さが変わる、が起きる
    MIN_TRACK_HEIGHT = MIN_TRACK_HEIGHT
    MAX_TRACK_HEIGHT = MAX_TRACK_HEIGHT
    CLIP_LABEL_HEIGHT = 14
    CLIP_RADIUS = 3

    #: クリップ端を掴んでトリムできる幅（ピクセル）
    TRIM_HANDLE_WIDTH = 6

    #: スナップが効く距離（ピクセル）
    SNAP_DISTANCE = 8

    #: 数値欄の増減ボタンの幅 ボタンの置き場をスタイルシートで決め打ちにするのに使う
    SPIN_BUTTON_WIDTH = 16


def _url(name: str) -> str:
    """同梱素材をスタイルシートの ``url()`` へ書ける形にする

    区切りは ``/`` にする ``\\`` のままだと、スタイルシートの字句で逃がし文字として
    読まれ、パスが壊れて絵が出ない
    """
    return f'url("{path_to(name).as_posix()}")'


def magnet_icons() -> tuple[str, str]:
    """いまのテーマの磁石の印 ``(入, 切)`` の同梱素材の名前"""
    return MAGNET_ICONS_LIGHT if _current == THEME_LIGHT else MAGNET_ICONS


def _spin_box() -> str:
    """数値欄（整数も小数も）の増減ボタン

    ボタンの置き場を右端の上下に決め打ちする 決めずに元の見た目（Windows 11）に
    任せると、ボタンが横に 2 つ並ぶのに、数字の欄はボタン 1 つぶんの幅しか空けずに
    広がる 上のボタンの大半が数字の欄の下に隠れ、押しても数字の欄が受け取って
    数が変わらなかった（Issue #27） 置き場を決めると、描く所と押せる所と数字の欄の
    幅が、どの見た目でも同じ計算から出る

    置き場を決めると元の見た目の矢印は描かれなくなるので、矢印の絵も自前で持つ
    テーマごとに別の絵にする 暗いテーマの白に近い矢印は、明るい地では見えない
    """
    arrows = SPIN_ARROWS_LIGHT if _current == THEME_LIGHT else SPIN_ARROWS
    up, down, up_off, down_off = (_url(name) for name in arrows)
    return f"""
QAbstractSpinBox {{ padding-right: {Metrics.SPIN_BUTTON_WIDTH + 4}px; }}
QAbstractSpinBox QLineEdit {{
    background: transparent;
    border: none;
    padding: 0;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: padding;
    width: {Metrics.SPIN_BUTTON_WIDTH}px;
    background-color: {Colors.PANEL.name()};
    border-left: 1px solid {Colors.BORDER.name()};
}}
QAbstractSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: 2px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: 2px;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background-color: {Colors.BORDER.name()};
}}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{
    background-color: {Colors.ACCENT.name()};
}}
QAbstractSpinBox::up-arrow {{ image: {up}; width: 8px; height: 6px; }}
QAbstractSpinBox::down-arrow {{ image: {down}; width: 8px; height: 6px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: {up_off}; }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: {down_off};
}}
"""


def _tabs() -> str:
    """タブ（重ねたドックの「メディア」「字幕」など、設定の窓のタブ）

    指定が無いと元の見た目のまま、選んだタブが明るい灰色になり、上から当てた
    白い文字が読めなかった（Issue #27） 選んだタブは地を少し明るくして文字を
    白に近くし、アクセント色の線を引く 選んでいないタブは文字を薄くするだけにして、
    どれを見ているかが色の差と線の 2 つで分かるようにする
    線は画面の中身の側に引く ドックのタブは設定で上にも下にも付くので、
    上に付くタブは下に、下に付くタブは上に引く
    """
    return f"""
QTabWidget::pane {{
    border: 1px solid {Colors.BORDER.name()};
    top: -1px;
}}
QTabBar {{
    /* タブの並びの右に残る枠（元の見た目の下敷き）を描かない 空の入力欄に見える */
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background-color: {Colors.WINDOW.name()};
    color: {Colors.TEXT_MUTED.name()};
    border: 1px solid {Colors.BORDER.name()};
    padding: 4px 12px;
}}
QTabBar::tab:top {{ border-bottom: 2px solid transparent; margin-right: 1px; }}
QTabBar::tab:bottom {{ border-top: 2px solid transparent; margin-right: 1px; }}
QTabBar::tab:hover {{
    background-color: {Colors.PANEL_ALT.name()};
    color: {Colors.TEXT.name()};
}}
QTabBar::tab:selected {{
    background-color: {Colors.TAB_SELECTED.name()};
    color: {Colors.CLIP_LABEL.name()};
}}
QTabBar::tab:top:selected {{ border-bottom-color: {Colors.ACCENT.name()}; }}
QTabBar::tab:bottom:selected {{ border-top-color: {Colors.ACCENT.name()}; }}
"""


def _tool_tip() -> str:
    """部品に載せたときに出る補足（ツールチップ）

    指定が無いと、補足の地と文字は OS の配色（パレットの ToolTipBase / ToolTipText）から
    取られる Windows の暗い配色では地も文字も暗くなり、再生ボタンなどの補足が
    読めなかった（Issue #27） スタイルシートで地と文字の両方を決め打ちにして、
    OS の配色に左右されないようにする 枠を書かないと地の色が効かない（Qt の決まり）
    """
    return f"""
QToolTip {{
    background-color: {Colors.TOOL_TIP.name()};
    color: {Colors.CLIP_LABEL.name()};
    border: 1px solid {Colors.TEXT_MUTED.name()};
    padding: 3px 6px;
}}
"""


def style_sheet() -> str:
    """アプリ全体のスタイル いまのテーマの色で作る

    ウィジェットごとに色を書くと、変えたいときに全部を探し回ることになるので
    1 箇所にまとめる 定数にしないのは、テーマを切り替えたら作り直すため
    """
    return f"""
QWidget {{
    background-color: {Colors.WINDOW.name()};
    color: {Colors.TEXT.name()};
    font-size: 12px;
}}
QMainWindow::separator {{
    background-color: {Colors.BORDER.name()};
    width: 1px;
    height: 1px;
}}
QDockWidget {{
    titlebar-close-icon: none;
    font-size: 12px;
}}
QDockWidget::title {{
    background-color: {Colors.PANEL.name()};
    padding: 4px 8px;
    border-bottom: 1px solid {Colors.BORDER.name()};
}}
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {Colors.PANEL_ALT.name()};
    border: 1px solid {Colors.BORDER.name()};
    outline: none;
}}
QListWidget::item, QTreeWidget::item, QTableWidget::item {{ padding: 3px 6px; }}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background-color: {Colors.ACCENT.name()};
    color: {Colors.ACCENT_TEXT.name()};
}}
QHeaderView::section {{
    background-color: {Colors.PANEL.name()};
    border: none;
    border-bottom: 1px solid {Colors.BORDER.name()};
    padding: 3px 6px;
}}
QPushButton, QToolButton {{
    background-color: {Colors.PANEL.name()};
    border: 1px solid {Colors.BORDER.name()};
    border-radius: 3px;
    padding: 4px 10px;
}}
QPushButton:hover, QToolButton:hover {{ background-color: {Colors.PANEL_ALT.name()}; }}
QPushButton:pressed, QToolButton:pressed {{
    background-color: {Colors.ACCENT.name()};
    color: {Colors.ACCENT_TEXT.name()};
}}
QPushButton:disabled {{ color: {Colors.TEXT_MUTED.name()}; }}
QMenuBar {{ background-color: {Colors.PANEL.name()}; }}
QMenuBar::item:selected {{
    background-color: {Colors.ACCENT.name()};
    color: {Colors.ACCENT_TEXT.name()};
}}
QMenu {{
    background-color: {Colors.PANEL.name()};
    border: 1px solid {Colors.BORDER.name()};
}}
/* 項目の余白を決めておく 決めないと、文字の大きさをここで決めているせいで項目の幅が
   文言とショートカットの和に足りず、長い項目では文言の終わりにショートカットが重なる */
QMenu::item {{ padding: 4px 32px 4px 20px; }}
QMenu::item:selected {{
    background-color: {Colors.ACCENT.name()};
    color: {Colors.ACCENT_TEXT.name()};
}}
QMenu::item:disabled {{ color: {Colors.TEXT_MUTED.name()}; }}
QStatusBar {{ background-color: {Colors.PANEL.name()}; }}
QScrollBar:horizontal, QScrollBar:vertical {{
    background: {Colors.PANEL.name()};
    border: none;
}}
QScrollBar:horizontal {{ height: 10px; }}
QScrollBar:vertical {{ width: 10px; }}
QScrollBar::handle {{
    background: {Colors.BORDER.name()};
    border-radius: 5px;
    min-width: 24px;
    min-height: 24px;
}}
QScrollBar::handle:hover {{ background: {Colors.TEXT_MUTED.name()}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QComboBox, QAbstractSpinBox, QLineEdit, QPlainTextEdit, QTextEdit {{
    background-color: {Colors.PANEL_ALT.name()};
    border: 1px solid {Colors.BORDER.name()};
    border-radius: 3px;
    padding: 3px 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {Colors.PANEL_ALT.name()};
    selection-background-color: {Colors.ACCENT.name()};
    selection-color: {Colors.ACCENT_TEXT.name()};
}}
{_spin_box()}
{_tabs()}
{_tool_tip()}
QProgressBar {{
    background-color: {Colors.PANEL_ALT.name()};
    border: 1px solid {Colors.BORDER.name()};
    border-radius: 3px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {Colors.ACCENT.name()}; }}
"""


class ThemeSignals(QObject):
    """テーマが変わったことを知らせる

    文字列や絵に色を焼き込んで持っている所（会話の欄の HTML・ボタンの印・素材一覧の
    印）は、描き直すだけでは変わらない ここへつないで作り直す
    """

    changed = Signal()


_signals: ThemeSignals | None = None


def theme_signals() -> ThemeSignals:
    """知らせの口 最初に求められたときに作る"""
    global _signals
    if _signals is None:
        _signals = ThemeSignals()
    return _signals


#: 色を焼き込んだスタイルシートを当てた部品と、その作り方 部品が消えたら一緒に消える
_styled: weakref.WeakKeyDictionary[QWidget, Callable[[], str]] = weakref.WeakKeyDictionary()


def themed_style(widget: QWidget, make: Callable[[], str]) -> None:
    """部品にスタイルシートを当て、テーマが変わったら ``make`` で作り直す

    ``widget.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()};")`` と書くと、
    その時のテーマの色が文字列に残り、切り替えても部品だけ前の色のまま残る
    同じ部品に 2 度当てたら、あとの作り方だけを覚える
    """
    _styled[widget] = make
    widget.setStyleSheet(make())


def _restyle() -> None:
    for widget, make in list(_styled.items()):
        if shiboken6.isValid(widget):
            widget.setStyleSheet(make())


#: Windows の設定の変化をつないだか 2 度つなぐと 1 回の変化で 2 度切り替える
_following = False
#: 切り替えの最中 色の決まり（colorScheme）を当てた知らせで、もう 1 度切り替えに入らない
_applying = False


def apply_theme(application: QApplication, mode: str) -> str:
    """選び方 ``mode`` のテーマを窓全体へ当てる 当たったテーマ（``dark`` か ``light``）を返す

    再起動は要らない アプリ全体のスタイルシートを作り直し、色を焼き込んだ部品を当て直し、
    全部の部品を描き直させる ``system`` のときは、Windows の設定が変わった時点でも合わせ直す

    Qt の色の決まり（``QStyleHints.colorScheme``）は、明るいテーマのときだけ明るい側を頼む
    チェックボックスの印など、スタイルシートで描かずに元の見た目（Windows 11）が描く所を
    明るい地に合わせるため 暗いテーマと ``system`` では頼まず Windows の設定のままにする
    （明るいテーマを足す前と同じ動き）
    """
    global _mode, _applying, _following
    _mode = mode if mode in THEME_MODES else THEME_DARK
    hints = application.styleHints()
    _applying = True
    try:
        hints.setColorScheme(
            Qt.ColorScheme.Light if _mode == THEME_LIGHT else Qt.ColorScheme.Unknown
        )
    finally:
        _applying = False
    if not _following:
        hints.colorSchemeChanged.connect(lambda scheme: follow_system(application, scheme))
        _following = True
    theme = resolve_theme(_mode)
    _activate(application, theme)
    return theme


def follow_system(application: QApplication, scheme: Qt.ColorScheme) -> None:
    """Windows の明るい・暗いが ``scheme`` に変わった ``system`` を選んでいるときだけ合わせ直す

    知らせに載ってきた値を使う 読み直すと、知らせを受けた時点ではまだ前の値が返る
    ことがある
    """
    if _applying or _mode != THEME_SYSTEM:
        return
    theme = resolve_theme(THEME_SYSTEM, scheme)
    if theme != _current:
        _activate(application, theme)


def _activate(application: QApplication, theme: str) -> None:
    with _no_collection():
        use_palette(theme)
        application.setStyleSheet(style_sheet())
        _restyle()
        theme_signals().changed.emit()
        # 自前で描く部品（タイムライン・グラフ・波形）は、描くたびに Colors を読む
        # 描き直しを頼まないと、次に何かが動くまで前のテーマの絵が残る
        for widget in application.allWidgets():
            widget.update()


@contextmanager
def _no_collection() -> Iterator[None]:
    """全部の部品へ配っている間は、ごみ集めを止める

    スタイルシートを当て直すと Qt は全部の部品へ知らせを配り、その途中で Python の受け手
    （設定パネルの数値欄の eventFilter・テーマの知らせにつないだ関数）が動く そこで閾値を
    越えてごみ集めが走ると、輪になって捨てられた Python 持ちの部品がその場で壊れ、Qt は
    壊れた部品へ配り続けて access violation で落ちる（PR #236 の CI） 下の ``update`` の
    繰り返しも、先に作った一覧の中の部品が途中で壊れると同じことになる
    止めるのは配る間だけ 終わったら元に戻し、溜まった分は次の閾値で片付く
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
