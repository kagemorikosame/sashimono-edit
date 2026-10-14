"""AI に見せる、コマを並べた 1 枚を描く（#320 A2 :class:`~sashimono.ai.host.SheetCanvas`）

並べ方（:class:`~sashimono.ai.sheet.SheetLayout`）は AI 層が決め、ここは描くだけ
1 コマずつ描くのは、間で編集画面へ手を返すため（:class:`~sashimono.ai.operations.Continue`）
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QRect, Qt
from PySide6.QtGui import QColor, QImage, QImageWriter, QPainter

from sashimono.ai.host import ToolError
from sashimono.ai.sheet import LABEL_HEIGHT, SheetLayout

__all__ = ["AiSheet"]

#: 升の間と時刻の帯の地 画面のテーマに合わせない（この絵は AI が読む物で、画面に出さない）
#: 明るいテーマの白い地にすると、白い字幕の升と隙間の境が見えなくなる
SHEET_BACKGROUND = QColor(32, 32, 32)
#: 時刻の字 暗い地の上で読みやすい明るい灰
LABEL_COLOR = QColor(235, 235, 235)
#: 帯の高さに対する字の高さ（画素） 帯いっぱいにすると上下の行の字と見分けにくい
LABEL_PIXEL_SIZE = LABEL_HEIGHT - 4

#: 1 コマを描く関数 フレームと升の幅を受け取り、RGBA の画素（縦 x 横 x 4）を返す
Render = Callable[[int, int], np.ndarray]


class AiSheet:
    """コマを並べて描く 1 枚"""

    def __init__(self, layout: SheetLayout, render: Render) -> None:
        self._layout = layout
        self._render = render
        self._image = QImage(layout.width, layout.height, QImage.Format.Format_RGB32)
        self._image.fill(SHEET_BACKGROUND)

    @property
    def image(self) -> QImage:
        """描いている途中の絵（試験で中身を見るため）"""
        return self._image

    def draw(self, index: int, frame: int, label: str) -> None:
        layout = self._layout
        x, y = layout.cell(index)
        pixels = self._render(frame, layout.cell_width)
        height, width = pixels.shape[0], pixels.shape[1]
        picture = QImage(pixels.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
        target = QRect(x, y, layout.cell_width, layout.cell_height)
        band = QRect(x, y + layout.cell_height, layout.cell_width, LABEL_HEIGHT)
        painter = QPainter(self._image)
        try:
            # 大きく描いてから縮める 滑らかにしないと、字幕の細い線が途切れて読めない
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.fillRect(target, Qt.GlobalColor.black)
            painter.drawImage(target, picture)
            font = painter.font()
            font.setPixelSize(LABEL_PIXEL_SIZE)
            painter.setFont(font)
            painter.setPen(LABEL_COLOR)
            # 升より長い字は末尾を省く はみ出すと隣の升の字と重なり、どちらの時刻か読めない
            room = band.adjusted(2, 0, -2, 0)
            text = painter.fontMetrics().elidedText(
                label, Qt.TextElideMode.ElideRight, room.width()
            )
            painter.drawText(room, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        finally:
            painter.end()

    def png(self) -> bytes:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        # QImage.save の書式引数は str しか受け取らない（型情報は bytes） render_png と同じく
        # QImageWriter を使う
        if not QImageWriter(buffer, b"PNG").write(self._image):
            raise ToolError("コマを並べた画像を作れませんでした")
        return bytes(buffer.data().data())
