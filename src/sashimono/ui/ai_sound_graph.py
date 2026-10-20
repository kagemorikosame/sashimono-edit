"""AI に見せる音量の推移のグラフを描く（#338 C2）

:meth:`~sashimono.ai.host.EditorHost.draw_sound_graph` の中身

何をどこへ描くか（:class:`~sashimono.ai.sound_graph.SoundGraph`）は AI 層が決め、ここは描くだけ
色は画面のテーマに合わせない この絵は AI が読む物で画面に出さず、テーマで色が変わると
返事の説明（帯の色の意味）と合わなくなる
"""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QIODevice, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QImage, QImageWriter, QPainter, QPen, QPolygonF

from sashimono.ai.host import ToolError
from sashimono.ai.sound_graph import (
    BAND_HEIGHT,
    FLOOR_LUFS,
    LABEL_HEIGHT,
    SUBTITLE_HEIGHT,
    TOP_LUFS,
    SoundGraph,
)
from sashimono.engine.checks.profile import (
    MUSIC,
    NOISE,
    SILENCE,
    SPEECH,
    SPEECH_AND_MUSIC,
    UNCLEAR,
)

__all__ = ["BAND_COLORS", "draw_sound_graph"]

#: 地は白 線と字を黒に近い色で描くと、縮めて読んでも潰れにくい
BACKGROUND = QColor(255, 255, 255)
GRID = QColor(225, 225, 225)
AXIS_TEXT = QColor(60, 60, 60)
#: 短時間（3 秒）のラウドネス 太い線
SHORT_TERM = QColor(20, 20, 20)
#: 瞬時（400 ms）の最大 細い線 太い線と見分けられる薄い色
MOMENTARY = QColor(120, 160, 220)
TARGET = QColor(0, 140, 70)
CLIPPED = QColor(220, 30, 30)
MARK = QColor(200, 60, 160)
SUBTITLE = QColor(90, 90, 90)
UNANALYZED = QColor(200, 200, 200)
#: 見分けの帯の色 返事の説明（operations の _BAND_LEGEND）と同じ順・同じ名前
BAND_COLORS = {
    SPEECH: QColor(47, 111, 223),
    MUSIC: QColor(232, 138, 30),
    SPEECH_AND_MUSIC: QColor(138, 79, 208),
    SILENCE: QColor(170, 170, 170),
    NOISE: QColor(141, 110, 99),
    UNCLEAR: QColor(255, 255, 255),
}
#: 縦軸の目盛りの間（LU）
GRID_STEP = 10
#: 字の大きさ（画素）
TEXT_PIXELS = 11


def draw_sound_graph(graph: SoundGraph) -> bytes:
    """``graph`` を PNG にする"""
    image = QImage(graph.width, graph.height, QImage.Format.Format_RGB32)
    image.fill(BACKGROUND)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = painter.font()
        font.setPixelSize(TEXT_PIXELS)
        painter.setFont(font)
        _grid(painter, graph)
        _unanalyzed(painter, graph)
        _lines(painter, graph)
        _bands(painter, graph)
        _time(painter, graph)
        _marks(painter, graph)
        _legend(painter, graph)
    finally:
        painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    # QImage.save の書式引数は str しか受け取らない（型情報は bytes） コマを並べる 1 枚と同じく
    # QImageWriter を使う
    if not QImageWriter(buffer, b"PNG").write(image):
        raise ToolError("音量のグラフを作れませんでした")
    return bytes(buffer.data().data())


def _grid(painter: QPainter, graph: SoundGraph) -> None:
    level = int(TOP_LUFS)
    while level >= FLOOR_LUFS:
        y = round(graph.y(level))
        painter.setPen(GRID)
        painter.drawLine(graph.left, y, graph.right, y)
        painter.setPen(AXIS_TEXT)
        painter.drawText(
            QRect(0, y - 8, graph.left - 4, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            str(level),
        )
        level -= GRID_STEP
    painter.setPen(AXIS_TEXT)
    painter.drawRect(graph.left, graph.top, graph.right - graph.left, graph.bottom - graph.top)


def _unanalyzed(painter: QPainter, graph: SoundGraph) -> None:
    if graph.analyzed_until is None:
        return
    # 測っていない所を空けたままにすると、無音と読み違える 灰色で塗って分ける
    painter.fillRect(
        QRect(
            graph.analyzed_until,
            graph.top,
            graph.right - graph.analyzed_until,
            graph.bottom - graph.top,
        ),
        UNANALYZED,
    )


def _polyline(painter: QPainter, graph: SoundGraph, values: tuple[float | None, ...]) -> None:
    """値の続く所ごとに折れ線を引く 測れない列（無音）で線を切る"""
    points: list[QPointF] = []
    for column, value in enumerate(values):
        if value is None:
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
            points = []
            continue
        points.append(QPointF(graph.left + column + 0.5, graph.y(value)))
    if len(points) > 1:
        painter.drawPolyline(QPolygonF(points))


def _lines(painter: QPainter, graph: SoundGraph) -> None:
    if graph.target is not None:
        pen = QPen(TARGET, 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        y = round(graph.y(graph.target))
        painter.drawLine(graph.left, y, graph.right, y)
    painter.setPen(QPen(MOMENTARY, 1.0))
    _polyline(painter, graph, graph.momentary)
    painter.setPen(QPen(SHORT_TERM, 2.0))
    _polyline(painter, graph, graph.short_term)
    painter.setPen(QPen(CLIPPED, 2.0))
    for x in graph.clipped:
        painter.drawLine(x, graph.top, x, graph.top + 8)


def _bands(painter: QPainter, graph: SoundGraph) -> None:
    top = graph.bottom + 2
    for band in graph.bands:
        painter.fillRect(
            QRect(band.left, top, max(1, band.right - band.left), BAND_HEIGHT),
            BAND_COLORS[band.label],
        )
    painter.setPen(AXIS_TEXT)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(graph.left, top, graph.right - graph.left, BAND_HEIGHT)
    lane = top + BAND_HEIGHT + 2
    for left, right in graph.subtitles:
        painter.fillRect(QRect(left, lane, max(1, right - left), SUBTITLE_HEIGHT), SUBTITLE)


def _time(painter: QPainter, graph: SoundGraph) -> None:
    lane = graph.bottom + 2 + BAND_HEIGHT + 2 + SUBTITLE_HEIGHT
    painter.setPen(AXIS_TEXT)
    metrics = painter.fontMetrics()
    for x, label in graph.time_ticks:
        painter.drawLine(x, graph.bottom, x, graph.bottom + 3)
        width = metrics.horizontalAdvance(label)
        # 右の端の字は内へ寄せる はみ出すと切れて時刻が読めない
        left = min(max(0, x - width // 2), graph.width - width - 2)
        painter.drawText(
            QRect(left, lane, width + 2, LABEL_HEIGHT),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )


def _marks(painter: QPainter, graph: SoundGraph) -> None:
    painter.setPen(QPen(MARK, 1.0, Qt.PenStyle.DotLine))
    for x, _label in graph.marks:
        painter.drawLine(x, graph.top, x, graph.bottom)
    painter.setPen(MARK)
    for x, label in graph.marks:
        # 頭打ちの赤い印（上の端から 8 画素）の下に書く 重ねると番号が読めない
        painter.drawText(
            QRect(x + 2, graph.top + 10, 30, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            label,
        )


def _legend(painter: QPainter, graph: SoundGraph) -> None:
    painter.setPen(AXIS_TEXT)
    painter.drawText(
        QRect(4, 2, graph.width - 8, graph.top - 4),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        "LUFS  太線=短時間(3s)  細線=瞬時(400ms)  点線=基準  赤=頭打ち",
    )
