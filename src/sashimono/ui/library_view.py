"""プリセットとエイリアスの一覧の見せ方（#276 #277）

設定（``表示 → 設定…``）で選ぶ見せ方と、窓をまたいで使い回す見本の絵の係を持つ
一覧の窓は設定パネルの〔プリセット…〕と、タイムラインの〔追加〕→〔エイリアス〕と、
〔オブジェクト〕のメニューの 3 か所から開く どこから開いても同じ見せ方と同じ
覚えた絵を使うよう、ここに 1 つだけ置く（窓の設定を 3 か所へ配ると、1 か所だけ
前の設定のまま残る）
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass

from PySide6.QtCore import QCoreApplication, QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap

from sashimono.ui.library_thumbnails import THUMBNAIL_SIZE, THUMBNAILS_FULL, LookThumbnails

__all__ = [
    "BACKDROP_CHECKER",
    "BACKDROP_DARK",
    "BACKDROP_LIGHT",
    "BACKDROP_MODES",
    "VISIBLE_DELAY_MS",
    "LibraryOptions",
    "failed_pixmap",
    "library_options",
    "set_library_options",
    "shared_thumbnails",
    "thumbnail_pixmap",
]

#: 見本の絵の地 :attr:`~sashimono.ui.workspace.Preferences.library_backdrop` の値
BACKDROP_CHECKER = "checker"
BACKDROP_DARK = "dark"
BACKDROP_LIGHT = "light"
BACKDROP_MODES = (BACKDROP_CHECKER, BACKDROP_DARK, BACKDROP_LIGHT)

#: 地の色 画面のテーマでは変えない 見本は映像の上に重ねる物の絵で、地は映像の代わり
#: テーマに合わせると、明るいテーマで白い字幕のプリセットが見えなくなる
_DARK = QColor("#202020")
_LIGHT = QColor("#f2f2f2")
_CHECKER = (QColor("#9a9a9a"), QColor("#6e6e6e"))
#: 読めなかった配布物の印の色 暗い地の上で目立つ赤
_FAILED = QColor("#e05a5a")
#: 市松の 1 目の大きさ（置いておく絵の画素）
_CHECKER_CELL = 16


@dataclass(frozen=True, slots=True)
class LibraryOptions:
    """一覧の見せ方 どれも設定から変える"""

    #: 見本の描き方（:data:`~sashimono.ui.library_thumbnails.THUMBNAIL_MODES`）
    thumbnails: str = THUMBNAILS_FULL
    #: 見本の地（:data:`BACKDROP_MODES`）
    backdrop: str = BACKDROP_CHECKER
    #: 消すときに確かめる
    confirm_delete: bool = True
    #: テンプレートの棚の一覧にも見本の絵を出す（描き方は :attr:`thumbnails` と同じ）
    shelf: bool = True


#: 見えている項目の見本を頼むまで待つ長さ（ミリ秒）
#: 0 にすると、窓を出したのと同じイベントの回りで走り係が動き出し、GIL を取り合って窓が
#: 出るのが遅れる（手元の棚 266 本で 0.16 秒が 0.47 秒） 送り続けている間に通り過ぎた
#: 項目まで頼まないためにもまとめる
VISIBLE_DELAY_MS = 40

_options = LibraryOptions()
_thumbnails: LookThumbnails | None = None


def library_options() -> LibraryOptions:
    return _options


def set_library_options(options: LibraryOptions) -> None:
    """設定を当てる 窓を開いていなくても、次に開いたときにこの見せ方になる"""
    global _options
    _options = options
    if _thumbnails is not None:
        _thumbnails.set_mode(options.thumbnails)


def shared_thumbnails() -> LookThumbnails:
    """見本の絵の係 窓を開くたびに作ると、覚えた絵を毎回置き場から読み直す"""
    global _thumbnails
    if _thumbnails is None:
        _thumbnails = LookThumbnails(mode=_options.thumbnails)
        application = QCoreApplication.instance()
        if application is not None:
            # 走り係が動いたままプロセスを閉じると、スレッドごと壊れて落ちる
            application.aboutToQuit.connect(_thumbnails.release)
        # イベントループを回さずに終わるとき（試験・道具の台本）は aboutToQuit が来ない
        atexit.register(_thumbnails.release)
    return _thumbnails


def failed_pixmap() -> QPixmap:
    """読めなかった配布物の印 暗い地に赤い × 地だけにすると、描いている途中と見分けが付かない"""
    pixmap = QPixmap(THUMBNAIL_SIZE)
    painter = QPainter(pixmap)
    try:
        painter.fillRect(QRect(QPoint(0, 0), THUMBNAIL_SIZE), _DARK)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(_FAILED, 10))
        middle_x, middle_y = THUMBNAIL_SIZE.width() // 2, THUMBNAIL_SIZE.height() // 2
        arm = THUMBNAIL_SIZE.height() // 4
        painter.drawLine(middle_x - arm, middle_y - arm, middle_x + arm, middle_y + arm)
        painter.drawLine(middle_x - arm, middle_y + arm, middle_x + arm, middle_y - arm)
    finally:
        painter.end()
    return pixmap


def thumbnail_pixmap(image: QImage | None, backdrop: str) -> QPixmap:
    """地を敷いた見本 ``image`` が無ければ地だけ（描いている途中・描けなかった物）"""
    pixmap = QPixmap(THUMBNAIL_SIZE)
    painter = QPainter(pixmap)
    try:
        area = QRect(QPoint(0, 0), THUMBNAIL_SIZE)
        if backdrop == BACKDROP_DARK:
            painter.fillRect(area, _DARK)
        elif backdrop == BACKDROP_LIGHT:
            painter.fillRect(area, _LIGHT)
        else:
            for row in range(0, THUMBNAIL_SIZE.height(), _CHECKER_CELL):
                for column in range(0, THUMBNAIL_SIZE.width(), _CHECKER_CELL):
                    shade = _CHECKER[(row // _CHECKER_CELL + column // _CHECKER_CELL) % 2]
                    painter.fillRect(column, row, _CHECKER_CELL, _CHECKER_CELL, shade)
        if image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(area, image)
    finally:
        painter.end()
    return pixmap
