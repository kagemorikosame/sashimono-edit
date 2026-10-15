"""グラデーションのバー（#312 Premiere Pro のテキストの塗りのグラデーションのエディタに倣う）

バーの上に今のグラデーションの見本を、下に色の止まりのつまみを並べる

- バーをクリックすると、その位置にその位置の色で止まりを足す
- つまみをドラッグすると位置が動く バーの下へ大きく引き出して離すと消える
- 選んだ止まりの色・不透明度・位置は下の欄で数で直す 〔＋〕〔−〕でも足し引きできる
- キーボード ←→ で選んだ止まりを 1%（Shift で 10%）動かす Ctrl+←→ で選ぶ止まりを替える
  Insert か + で足す Delete で消す Enter か Space で色を選ぶ

止まりは 2 つより減らさない 1 つだと単色と同じで、バーで動かす意味が無い
見本は字を塗る筆（:func:`sashimono.engine.fills.gradient_brush`）と同じ物で描く 別に描くと、
止まりの間の混ぜ方が字と食い違い、見本を見て決めた色が字では違って出る
"""

from __future__ import annotations

import math
from itertools import pairwise

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.model import ParamValue
from sashimono.effects.spec import (
    MAX_GRADIENT_STOPS,
    MIN_GRADIENT_STOPS,
    GradientSpec,
    GradientStop,
)
from sashimono.engine.fills import Paint, gradient_brush
from sashimono.ui.flow_layout import FlowLayout
from sashimono.ui.inspector.widgets import ParameterEditor
from sashimono.ui.theme import Colors, themed_style

__all__ = ["GradientBar", "GradientEditor", "colour_at"]

#: つまみの幅と高さ（画素） 端の止まり（位置 0 と 1）のつまみが欠けないよう、バーの左右を半分空ける
_HANDLE_WIDTH = 12
_HANDLE_HEIGHT = 14
#: バーの見本の高さ
_BAR_HEIGHT = 22
#: つまみをバーの下へこれだけ引き出したら、離したときに消す（Premiere・Photoshop と同じ操作）
_REMOVE_DISTANCE = 28
#: 押してからこれだけ動かすまではドラッグにしない 押しただけで位置が少し動くと、選ぶつもりの
#: 操作が履歴に積まれる
_DRAG_START = 3
#: 市松の 1 マス（画素） 透明な所が分かるように敷く 色はテーマで変えない（地は絵の代わり）
_CHECKER = 6
_CHECKER_LIGHT = QColor(204, 204, 204)
_CHECKER_DARK = QColor(153, 153, 153)


def colour_at(
    stops: list[GradientStop], position: float, *, smooth: bool = False
) -> tuple[float, float, float, float]:
    """``position`` の色 字を塗る筆と同じく、色と不透明度を別々に sRGB の値のまま混ぜる

    止まりを足すときの色に使う 足した所の色が前と変わらないので、足しただけで止まりの色が
    跳ねない ``smooth`` なら止まりの間を S 字で混ぜる（「グラデーション」エフェクト #364）
    """
    if position <= stops[0][0]:
        return stops[0][1]
    for (left, low), (right, high) in pairwise(stops):
        if position <= right:
            share = 0.0 if right <= left else (position - left) / (right - left)
            if smooth:
                share = share * share * (3.0 - 2.0 * share)
            mixed = (a + (b - a) * share for a, b in zip(low, high, strict=True))
            red, green, blue, alpha = mixed
            return (red, green, blue, alpha)
    return stops[-1][1]


class GradientBar(QWidget):
    """見本とつまみ 値は持ち主（:class:`GradientEditor`）が持ち、ここは描いて操作を伝える"""

    #: 止まりの並びが変わった 2 つ目が真なら確定（履歴へ積む）、偽ならドラッグの途中
    stops_changed = Signal(object, bool)
    #: 選んだ止まりが替わった（番号）
    selected = Signal(int)
    #: 色を選びたい（Enter・Space・つまみのダブルクリック）
    colour_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(_BAR_HEIGHT + _HANDLE_HEIGHT + 6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("グラデーションのバー")
        self.setToolTip(
            "クリックで色の止まりを足す ドラッグで動かし、下へ引き出して離すと消す\n"
            "←→ で動かす（Shift で大きく） Ctrl+←→ で選ぶ止まりを替える\n"
            "Insert で足す Delete で消す Enter で色を選ぶ"
        )
        self._stops: list[GradientStop] = []
        self._current = 0
        #: ドラッグ中の止まりの番号と、押した所 押していなければ ``None``
        self._dragging: int | None = None
        self._pressed_at: QPointF | None = None
        self._moved = False
        #: 下へ引き出して、離せば消える状態か
        self._removing = False
        #: 止まりの間を S 字で混ぜるか（:meth:`GradientSpec.smooth_in`）
        self._smooth = False

    def set_smooth(self, smooth: bool) -> None:
        """止まりの間を S 字で混ぜて見せるか 描く所の混ぜ方に合わせないと、見本と絵が食い違う"""
        self._smooth = smooth
        self.update()

    # --- 値 ---

    def set_stops(self, stops: list[GradientStop], current: int) -> None:
        self._stops = list(stops)
        self._current = min(max(current, 0), len(self._stops) - 1)
        self.update()

    @property
    def stops(self) -> list[GradientStop]:
        return list(self._stops)

    @property
    def current(self) -> int:
        return self._current

    def is_dragging(self) -> bool:
        return self._dragging is not None

    # --- 座標 ---

    def _track(self) -> QRectF:
        """位置 0 から 1 が並ぶ横の範囲（バーの見本の四角）"""
        margin = _HANDLE_WIDTH / 2.0 + 1.0
        return QRectF(margin, 2.0, max(1.0, self.width() - 2.0 * margin), _BAR_HEIGHT)

    def x_of(self, position: float) -> float:
        track = self._track()
        return track.left() + position * track.width()

    def position_of(self, x: float) -> float:
        track = self._track()
        return min(max((x - track.left()) / track.width(), 0.0), 1.0)

    def _handle_rect(self, index: int) -> QRectF:
        # 枠の線が画素の真ん中を通るよう半画素にそろえる 画素の境を通ると、1 画素の線が
        # 2 画素の薄い線に割れて、明るいテーマでも暗いテーマでも地に溶ける
        x = math.floor(self.x_of(self._stops[index][0])) + 0.5
        top = math.floor(self._track().bottom() + 2.0) + 0.5
        return QRectF(x - _HANDLE_WIDTH / 2.0, top, _HANDLE_WIDTH, _HANDLE_HEIGHT)

    def _handle_at(self, point: QPointF) -> int | None:
        """``point`` にあるつまみ 重なっていれば選んでいる物、次に後ろに描いた（手前の）物"""
        hits = [i for i in range(len(self._stops)) if self._handle_rect(i).contains(point)]
        if not hits:
            return None
        return self._current if self._current in hits else hits[-1]

    # --- 描く ---

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt の命名規約
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        track = self._track()
        self._paint_checker(painter, track)
        if len(self._stops) >= MIN_GRADIENT_STOPS:
            # 字を塗るのと同じ筆 角度 -90 で左から右へ、長さ 100% で四角の端から端まで
            # S 字で混ぜるエフェクト（#364）は S 字を細かい止まりで刻んで同じ筆で描く
            stops = GradientSpec.smoothed(self._stops) if self._smooth else self._stops
            paint = Paint(kind="linear", stops=GradientSpec.pack(stops), angle=-90.0)
            painter.fillRect(track, gradient_brush(paint, track))
        painter.setPen(QPen(Colors.BORDER, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(track.adjusted(-0.5, -0.5, 0.5, 0.5))
        order = [i for i in range(len(self._stops)) if i != self._current]
        for index in [*order, self._current] if self._stops else []:
            self._paint_handle(painter, index)
        painter.end()

    def _paint_checker(self, painter: QPainter, track: QRectF) -> None:
        painter.save()
        painter.setClipRect(track)
        painter.fillRect(track, _CHECKER_LIGHT)
        row = 0
        y = track.top()
        while y < track.bottom():
            x = track.left() + (_CHECKER if row % 2 else 0)
            while x < track.right():
                painter.fillRect(QRectF(x, y, _CHECKER, _CHECKER), _CHECKER_DARK)
                x += 2 * _CHECKER
            y += _CHECKER
            row += 1
        painter.restore()

    def _paint_handle(self, painter: QPainter, index: int) -> None:
        """つまみ 上の三角で位置を指し、下の四角にその止まりの色（不透明にして）を出す

        選んでいるつまみは強調の色の太い枠 引き出して消える途中のつまみは薄く描く
        """
        rect = self._handle_rect(index)
        _position, (red, green, blue, _alpha) = self._stops[index]
        chosen = index == self._current
        painter.save()
        if chosen and self._removing:
            painter.setOpacity(0.35)
        outline = Colors.ACCENT if chosen else Colors.TEXT
        painter.setPen(QPen(outline, 2.0 if chosen else 1.0))
        tip = QPainterPath()
        middle = rect.center().x()
        tip.moveTo(middle, rect.top())
        tip.lineTo(rect.right(), rect.top() + 5.0)
        tip.lineTo(rect.right(), rect.bottom())
        tip.lineTo(rect.left(), rect.bottom())
        tip.lineTo(rect.left(), rect.top() + 5.0)
        tip.closeSubpath()
        painter.setBrush(Colors.PANEL)
        painter.drawPath(tip)
        swatch = QRectF(
            rect.left() + 2.5, rect.top() + 6.5, rect.width() - 5.0, rect.height() - 9.0
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(round(red * 255), round(green * 255), round(blue * 255)))
        painter.drawRect(swatch)
        painter.restore()

    # --- マウス ---

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if event.button() != Qt.MouseButton.LeftButton or not self._stops:
            super().mousePressEvent(event)
            return
        point = event.position()
        hit = self._handle_at(point)
        if hit is None:
            if self._track().adjusted(0, 0, 0, _HANDLE_HEIGHT).contains(point):
                self._add_at(self.position_of(point.x()))
            return
        self._current = hit
        self.selected.emit(hit)
        self._dragging = hit
        self._pressed_at = point
        self._moved = False
        self._removing = False
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._dragging is None or self._pressed_at is None:
            super().mouseMoveEvent(event)
            return
        point = event.position()
        if not self._moved and (point - self._pressed_at).manhattanLength() < _DRAG_START:
            return
        self._moved = True
        removing = (
            point.y() > self.height() + _REMOVE_DISTANCE and len(self._stops) > MIN_GRADIENT_STOPS
        )
        if removing != self._removing:
            self._removing = removing
            self.update()
        if removing:
            # 消える途中は、消した後の並びで見せる（離すとそうなる）
            shown = [stop for i, stop in enumerate(self._stops) if i != self._dragging]
            self.stops_changed.emit(shown, False)
            return
        self.move_current(self.position_of(point.x()), commit=False)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._dragging is None:
            super().mouseReleaseEvent(event)
            return
        moved, removing = self._moved, self._removing
        self._dragging = None
        self._pressed_at = None
        self._moved = False
        self._removing = False
        if removing:
            self.remove_current()
        elif moved:
            self.stops_changed.emit(self.stops, True)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._handle_at(event.position()) is not None:
            self.colour_requested.emit()
            return
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt の命名規約
        self.update()
        super().focusOutEvent(event)

    # --- キーボード ---

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt の命名規約
        if not self._stops:
            super().keyPressEvent(event)
            return
        key = event.key()
        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = -1 if key == Qt.Key.Key_Left else 1
            if control:
                self.select((self._current + step) % len(self._stops))
            else:
                amount = 0.1 if shift else 0.01
                self.move_current(self._stops[self._current][0] + step * amount, commit=True)
            return
        if key in (Qt.Key.Key_Insert, Qt.Key.Key_Plus):
            self.add_after_current()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_current()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.colour_requested.emit()
            return
        super().keyPressEvent(event)

    # --- 操作 ---

    def select(self, index: int) -> None:
        self._current = min(max(index, 0), len(self._stops) - 1)
        self.selected.emit(self._current)
        self.update()

    def move_current(self, position: float, *, commit: bool) -> None:
        """選んだ止まりを ``position`` へ動かす 並びの順が入れ替わっても同じ止まりを選んだまま"""
        moving = self._stops[self._current]
        moved = (min(max(position, 0.0), 1.0), moving[1])
        rest = [stop for i, stop in enumerate(self._stops) if i != self._current]
        # 同じ位置の止まりの後ろへ入れる 前へ入れると、重ねただけで色の段の順が入れ替わる
        index = sum(1 for stop in rest if stop[0] <= moved[0])
        self._stops = [*rest[:index], moved, *rest[index:]]
        if self._dragging is not None:
            self._dragging = index
        if index != self._current:
            self._current = index
            self.selected.emit(index)
        self.update()
        self.stops_changed.emit(self.stops, commit)

    def _add_at(self, position: float) -> None:
        if len(self._stops) >= MAX_GRADIENT_STOPS:
            return
        stop = (position, colour_at(self._stops, position, smooth=self._smooth))
        index = sum(1 for existing in self._stops if existing[0] <= position)
        self._stops.insert(index, stop)
        self._current = index
        self.selected.emit(index)
        self.update()
        self.stops_changed.emit(self.stops, True)

    def add_after_current(self) -> None:
        """選んだ止まりと次の止まりの真ん中に足す 最後の止まりなら前の止まりとの真ん中"""
        if len(self._stops) >= MAX_GRADIENT_STOPS:
            return
        here = self._stops[self._current][0]
        if self._current + 1 < len(self._stops):
            there = self._stops[self._current + 1][0]
        else:
            there = self._stops[self._current - 1][0]
        self._add_at((here + there) / 2.0)

    def remove_current(self) -> None:
        if len(self._stops) <= MIN_GRADIENT_STOPS:
            return
        del self._stops[self._current]
        self._current = min(self._current, len(self._stops) - 1)
        self.selected.emit(self._current)
        self.update()
        self.stops_changed.emit(self.stops, True)


class GradientEditor(ParameterEditor):
    """グラデーションの欄 バー（:class:`GradientBar`）と、選んだ止まりの色・不透明度・位置の欄"""

    def __init__(self, spec: GradientSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._bar = GradientBar(self)
        # 混ぜ方の欄を持つ物は、項目の無いとき（既定）の S 字で始める 設定パネルが今の値で直す
        self._bar.set_smooth(spec.smooth_in({}))
        self._bar.stops_changed.connect(self._on_stops)
        self._bar.selected.connect(lambda _index: self._show_current())
        self._bar.colour_requested.connect(self._choose_colour)

        self._colour = QPushButton(self)
        self._colour.setFixedHeight(24)
        self._colour.setToolTip("選んだ止まりの色")
        self._colour.setAccessibleName("止まりの色")
        self._colour.clicked.connect(self._choose_colour)
        self._opacity = QSpinBox(self)
        self._opacity.setRange(0, 100)
        self._opacity.setSuffix(" %")
        self._opacity.setKeyboardTracking(False)
        self._opacity.setToolTip("選んだ止まりの不透明度")
        self._opacity.setAccessibleName("止まりの不透明度")
        self._opacity.valueChanged.connect(self._on_opacity)
        self._position = QDoubleSpinBox(self)
        self._position.setRange(0.0, 100.0)
        self._position.setDecimals(1)
        self._position.setSingleStep(1.0)
        self._position.setSuffix(" %")
        self._position.setKeyboardTracking(False)
        self._position.setToolTip("選んだ止まりの位置")
        self._position.setAccessibleName("止まりの位置")
        self._position.valueChanged.connect(self._on_position)
        self._add = QToolButton(self)
        self._add.setText("＋")
        self._add.setToolTip("選んだ止まりの隣に止まりを足す")
        self._add.clicked.connect(self._bar.add_after_current)
        self._remove = QToolButton(self)
        self._remove.setText("−")
        self._remove.setToolTip(f"選んだ止まりを消す（{MIN_GRADIENT_STOPS} つより減らせない）")
        self._remove.clicked.connect(self._bar.remove_current)

        controls = QWidget(self)
        # 狭いパネルでは折り返す 横 1 列に並べると、色と 2 つの数の欄でパネルの最小の幅が広がる
        flow = FlowLayout(controls)
        flow.addWidget(self._colour)
        for label, box in (("不透明度", self._opacity), ("位置", self._position)):
            pair = QWidget(controls)
            row = QHBoxLayout(pair)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(label, pair))
            row.addWidget(box)
            flow.addWidget(pair)
        flow.addWidget(self._add)
        flow.addWidget(self._remove)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._bar)
        layout.addWidget(controls)
        self.set_value(spec.default_value())

    @property
    def bar(self) -> GradientBar:
        return self._bar

    def set_smooth(self, smooth: bool) -> None:
        """止まりの間を S 字で混ぜて見せるか（選んだ色の混ぜ方 #364）"""
        self._bar.set_smooth(smooth)

    def is_busy(self) -> bool:
        return self._bar.is_dragging()

    def set_value(self, value: ParamValue | None) -> None:
        stops = GradientSpec.stops(self._spec.coerce(value))
        if stops == self._bar.stops:
            return
        self._bar.set_stops(stops, self._bar.current)
        self._show_current()

    def _value(self) -> tuple[float, ...]:
        return self._spec.coerce(GradientSpec.pack(self._bar.stops))

    def _on_stops(self, stops: list[GradientStop], commit: bool) -> None:
        value = self._spec.coerce(GradientSpec.pack(stops))
        self._show_current()
        if commit:
            self._emit(value)
            self.interaction_finished.emit()
        else:
            self._preview(value)

    def _show_current(self) -> None:
        """選んだ止まりの値を下の欄へ 欄は信号を出さずに受け取る"""
        stops = self._bar.stops
        if not stops:
            return
        position, (red, green, blue, alpha) = stops[self._bar.current]
        was = self._updating
        self._updating = True
        try:
            self._opacity.setValue(round(alpha * 100))
            self._position.setValue(round(position * 1000) / 10)
            shown = (round(red * 255), round(green * 255), round(blue * 255))
            luminance = 0.2126 * shown[0] + 0.7152 * shown[1] + 0.0722 * shown[2]
            # 色の見本のボタンの文字 地は選んだ色で、明るさで黒か白を選ぶ（色の欄と同じ）
            text = "#000000" if luminance > 140 else "#ffffff"
            swatch = f"background-color: rgb{shown}; color: {text};"
            themed_style(
                self._colour, lambda: f"{swatch} border: 1px solid {Colors.BORDER.name()};"
            )
            self._colour.setText("#{:02X}{:02X}{:02X}".format(*shown))
            self._remove.setEnabled(len(stops) > MIN_GRADIENT_STOPS)
            self._add.setEnabled(len(stops) < MAX_GRADIENT_STOPS)
        finally:
            self._updating = was

    def _replace_current(self, stop: GradientStop) -> None:
        stops = self._bar.stops
        stops[self._bar.current] = stop
        self._bar.set_stops(stops, self._bar.current)
        self._show_current()
        self._emit(self._value())

    def _on_opacity(self, percent: int) -> None:
        if self._updating:
            return
        position, (red, green, blue, _alpha) = self._bar.stops[self._bar.current]
        self._replace_current((position, (red, green, blue, percent / 100.0)))

    def _on_position(self, percent: float) -> None:
        if self._updating:
            return
        self._bar.select(self._bar.current)
        self._bar.move_current(percent / 100.0, commit=True)

    def _choose_colour(self) -> None:
        position, (red, green, blue, alpha) = self._bar.stops[self._bar.current]
        chosen = QColorDialog.getColor(
            QColor(round(red * 255), round(green * 255), round(blue * 255), round(alpha * 255)),
            self,
            "止まりの色を選ぶ",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not chosen.isValid():
            return
        self._replace_current(
            (
                position,
                (
                    chosen.red() / 255.0,
                    chosen.green() / 255.0,
                    chosen.blue() / 255.0,
                    chosen.alpha() / 255.0,
                ),
            )
        )
