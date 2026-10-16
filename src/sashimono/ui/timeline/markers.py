"""タイムラインの目印（マーカー） 目盛りの上に旗で見せ、右クリックで置く・名前を変える・消す（#330）

AI アシスタントが置いた目印（チャプターの元）を、人が見て確かめ、手直しできるようにする
画面に出さないと、AI が置いた目印は見えないまま書き出しの目次にだけ効く

:class:`~sashimono.ui.timeline.view.TimelineView` からは入口を数行呼ぶだけにして、見せ方と
右クリックの項目はここにまとめる（書き出し範囲の :mod:`.work_area` と同じ分け方）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QInputDialog, QMenu, QWidget

from sashimono.core.commands import AddMarkers, Command, RemoveMarkers, UpdateMarker
from sashimono.core.model import Marker, Timeline
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline.layout import TimelineLayout

__all__ = ["DELETE_TEXT", "PLACE_TEXT", "RENAME_TEXT", "MarkerEditor"]

#: 右クリックに出す名前 テストもこの名前で探す
PLACE_TEXT = "ここにマーカーを置く"
RENAME_TEXT = "マーカーの名前を変更…"
DELETE_TEXT = "マーカーを削除"

#: 旗（下向きの三角）の幅の半分と高さ（画素） 目盛りの字の上の余白に収まる大きさ
_FLAG_HALF = 5
_FLAG_HEIGHT = 8
#: 右クリックで目印に当たったと見なす距離（画素） 旗の幅と同じにして、見えている旗の
#: どこを押しても当たるようにする
_HIT = _FLAG_HALF + 1
#: 名前を出す幅の上限（画素） 長い名前で隣の目印の旗を隠さない
_LABEL_WIDTH = 140


class MarkerEditor:
    """目印の見せ方と、右クリックからの操作 どれも 1 つのコマンドを出すだけ"""

    def __init__(self, request: Callable[[list[Command], str], None], parent: QWidget) -> None:
        self._request = request
        self._parent = parent

    def marker_near(self, layout: TimelineLayout, timeline: Timeline, x: float) -> Marker | None:
        """x に一番近い、当たりの距離の中の目印"""
        best: tuple[float, Marker] | None = None
        for marker in timeline.markers:
            distance = abs(layout.frame_to_x(marker.frame) - x)
            if distance <= _HIT and (best is None or distance < best[0]):
                best = (distance, marker)
        return best[1] if best is not None else None

    # --- 描画 ---

    def paint_ruler(
        self, painter: QPainter, layout: TimelineLayout, width: int, timeline: Timeline
    ) -> None:
        """目盛りの上の端に、目印の色の旗と縦の線、その右に名前を描く"""
        if not timeline.markers:
            return
        # 字の大きさとペンを戻す 戻さないと、後で描く物（再生ヘッドや落ちる所の目安）が
        # 自分でペンを決め忘れたとき、目印の色と小さい字を黙って引き継ぐ
        painter.save()
        try:
            self._paint_flags(painter, layout, width, timeline)
        finally:
            painter.restore()

    def _paint_flags(
        self, painter: QPainter, layout: TimelineLayout, width: int, timeline: Timeline
    ) -> None:
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        for marker in timeline.markers:
            x = layout.frame_to_x(marker.frame)
            # ヘッダの下へ潜る旗は描かない トラックの名前の上に色が乗り、名前が読めない
            if x < Metrics.TRACK_HEADER_WIDTH or x > width + _FLAG_HALF:
                continue
            color = QColor(marker.color)
            if not color.isValid():
                # 手で書いた作品などで色が読めなくても、目印そのものは見せる
                color = QColor(Colors.TEXT_MUTED)
            painter.setPen(QPen(color, 1))
            painter.drawLine(QPointF(x, _FLAG_HEIGHT), QPointF(x, Metrics.RULER_HEIGHT))
            flag = QPainterPath()
            flag.moveTo(x - _FLAG_HALF, 0)
            flag.lineTo(x + _FLAG_HALF, 0)
            flag.lineTo(x, _FLAG_HEIGHT)
            flag.closeSubpath()
            painter.fillPath(flag, color)
            if marker.label:
                text = metrics.elidedText(marker.label, Qt.TextElideMode.ElideRight, _LABEL_WIDTH)
                painter.setPen(QPen(Colors.TEXT, 1))
                painter.drawText(
                    QRectF(x + _FLAG_HALF + 2, 0, _LABEL_WIDTH, _FLAG_HEIGHT + 4), text
                )

    # --- 右クリック ---

    def add_menu_actions(
        self, menu: QMenu, layout: TimelineLayout, position: QPoint, timeline: Timeline
    ) -> None:
        """目盛りの上なら、目印の上では名前の変更と削除、それ以外では置く項目を足す

        トラックの上には出さない クリップの操作が並ぶ中に目印の項目が毎回混ざると、
        目当ての項目を探しにくい
        """
        if position.y() >= Metrics.RULER_HEIGHT or position.x() < Metrics.TRACK_HEADER_WIDTH:
            return
        menu.addSeparator()
        marker = self.marker_near(layout, timeline, position.x())
        if marker is None:
            frame = max(0, round(layout.x_to_frame(position.x())))
            place = menu.addAction(PLACE_TEXT)
            place.triggered.connect(lambda _checked=False: self.place(frame))
            return
        rename = menu.addAction(RENAME_TEXT)
        rename.triggered.connect(lambda _checked=False: self.ask_rename(marker))
        delete = menu.addAction(DELETE_TEXT)
        delete.triggered.connect(lambda _checked=False: self.delete(marker))

    def place(self, frame: int) -> None:
        self._request([AddMarkers((Marker(frame),))], "マーカーを置く")

    def delete(self, marker: Marker) -> None:
        self._request([RemoveMarkers((marker.frame,))], "マーカーを削除")

    def ask_rename(self, marker: Marker) -> None:
        name, accepted = QInputDialog.getText(
            self._parent, "マーカーの名前", "名前（チャプターの見出しになる）", text=marker.label
        )
        if accepted and name.strip() != marker.label:
            self._request(
                [UpdateMarker(marker.frame, replace(marker, label=name))], "マーカーの名前を変更"
            )
