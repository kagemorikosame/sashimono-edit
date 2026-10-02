"""素材一覧に出す絵（サムネイルと、絵の無い素材の印）

サムネイルそのものは作らない 作るのはタイムラインの絵の並び（フィルムストリップ）を
裏で作る :class:`~sashimono.engine.cache.MediaAnalyzer` で、ここはできた 1 枚を
画面の部品に載る形へ直すだけ 画面のスレッドで素材を開いて 1 コマ取り出すと、
素材を何本も並べたときに一覧を作るたびに固まる
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap

from sashimono.ui.theme import Colors

__all__ = ["GRID_ICON_SIZE", "LIST_ICON_SIZE", "audio_icon", "pending_icon", "thumbnail_icon"]

#: 一覧表示の行の頭に出す絵の大きさ 16:9 にそろえる 素材ごとに縦横比が違う絵を
#: そのまま出すと、行ごとに文字の始まる位置がずれて読みにくい
LIST_ICON_SIZE = QSize(64, 36)

#: アイコン表示の絵の大きさ サムネイルの元の高さ（72）と同じにして、拡大でぼかさない
GRID_ICON_SIZE = QSize(128, 72)


def thumbnail_icon(tile: np.ndarray) -> QIcon:
    """RGBA の 1 コマ（``(高さ, 幅, 4)``）を、縦横比を保って 16:9 の枠へ収めた絵にする

    大きい方（:data:`GRID_ICON_SIZE`）で作り、一覧表示では Qt に縮めさせる
    両方の大きさで作ると、切り替えるたびに作り直すか 2 枚抱えることになる
    """
    height, width = int(tile.shape[0]), int(tile.shape[1])
    # 行の詰まった連続した 4 色の配列にしてから渡す QImage は 1 行を幅×4 バイトと
    # 読むので、3 色や灰色 1 色のまま渡すと行の外まで読んで絵が崩れる（落ちることもある）
    # 飛び飛びの切り出し（シートの 1 列）のままだと、隣の絵まで読む
    data = np.ascontiguousarray(_rgba(tile), dtype=np.uint8)
    image = QImage(data.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
    canvas = _canvas(GRID_ICON_SIZE)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.fillRect(canvas.rect(), Colors.MEDIA_BACKDROP)
    scaled = image.scaled(
        GRID_ICON_SIZE,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    left = (GRID_ICON_SIZE.width() - scaled.width()) // 2
    top = (GRID_ICON_SIZE.height() - scaled.height()) // 2
    painter.drawImage(left, top, scaled)
    painter.end()
    return QIcon(canvas)


def audio_icon() -> QIcon:
    """音声だけの素材の印 タイムラインの音声のクリップと同じ色で、波の形を描く"""
    canvas = _canvas(GRID_ICON_SIZE)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _panel(painter, Colors.AUDIO_CLIP, Colors.AUDIO_CLIP_BORDER)
    painter.setPen(QPen(Colors.WAVEFORM, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    middle = GRID_ICON_SIZE.height() / 2
    # 決まった高さの棒を並べる 素材ごとの波形にしないのは、波形を作るのが
    # 解析の仕事で、ここで読むと画面のスレッドで素材を開くことになるため
    for index, level in enumerate((0.25, 0.55, 0.9, 0.6, 0.35, 0.7, 0.45, 0.2)):
        x = 30 + index * 10
        reach = level * 22
        painter.drawLine(QPointF(x, middle - reach), QPointF(x, middle + reach))
    painter.end()
    return QIcon(canvas)


def pending_icon() -> QIcon:
    """絵をまだ作っていない（作れなかった）映像・画像の印 再生の三角を描く"""
    canvas = _canvas(GRID_ICON_SIZE)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _panel(painter, Colors.VIDEO_CLIP, Colors.VIDEO_CLIP_BORDER)
    center_x, center_y = GRID_ICON_SIZE.width() / 2, GRID_ICON_SIZE.height() / 2
    triangle = QPainterPath(QPointF(center_x - 10, center_y - 14))
    triangle.lineTo(QPointF(center_x + 16, center_y))
    triangle.lineTo(QPointF(center_x - 10, center_y + 14))
    triangle.closeSubpath()
    painter.fillPath(triangle, Colors.CLIP_LABEL)
    painter.end()
    return QIcon(canvas)


def _rgba(tile: np.ndarray) -> np.ndarray:
    """``(高さ, 幅, 色の数)`` を 4 色（RGBA）へそろえる 色の数が 1・3・4 以上のどれでもよい"""
    channels = int(tile.shape[2]) if tile.ndim == 3 else 1
    pixels = tile if tile.ndim == 3 else tile[:, :, None]
    if channels >= 4:
        return pixels[:, :, :4]
    color = np.repeat(pixels[:, :, :1], 3, axis=2) if channels < 3 else pixels[:, :, :3]
    opaque = np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([color.astype(np.uint8), opaque], axis=2)


def _canvas(size: QSize) -> QPixmap:
    canvas = QPixmap(size)
    canvas.fill(Qt.GlobalColor.transparent)
    return canvas


def _panel(painter: QPainter, fill: QColor, border: QColor) -> None:
    rect = QRectF(QRect(0, 0, GRID_ICON_SIZE.width(), GRID_ICON_SIZE.height())).adjusted(
        1, 1, -1, -1
    )
    painter.setPen(QPen(border, 2))
    painter.setBrush(fill)
    painter.drawRoundedRect(rect, 6, 6)
