"""再生コントロールとタイムコード表示"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap, QPolygonF, QResizeEvent
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from sashimono.core.timebase import FrameRate, format_timecode
from sashimono.engine.render import RenderQuality
from sashimono.ui.theme import Colors, theme_signals, themed_style

__all__ = ["TransportBar", "transport_icon"]

#: 再生品質の選択肢 分母が大きいほど軽くなる
QUALITY_CHOICES: tuple[tuple[str, int], ...] = (
    ("フル画質", 1),
    ("1/2 画質", 2),
    ("1/4 画質", 4),
)


class TransportBar(QWidget):
    """再生・停止・コマ送りと、現在位置の表示"""

    play_toggled = Signal()
    step_requested = Signal(int)
    jump_requested = Signal(int)
    quality_changed = Signal(object)

    def __init__(self, rate: FrameRate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rate = rate
        self._frame = 0
        self._duration = 0

        self._to_start = _tool_button("to_start", "先頭へ (Home)", "先頭へ")
        self._back = _tool_button("back", "1 フレーム戻る (←)", "1 フレーム戻る")
        self._play = _tool_button("play", "再生 / 停止 (Space)", "再生")
        self._forward = _tool_button("forward", "1 フレーム進む (→)", "1 フレーム進む")
        self._to_end = _tool_button("to_end", "末尾へ (End)", "末尾へ")
        self._play_icon = transport_icon("play")
        self._pause_icon = transport_icon("pause")
        self._playing = False
        # 印は文字の色で描いた絵 テーマが変わったら描き直す 描き直さないと、暗いテーマの
        # 白に近い印が明るい地の上で見えなくなる
        theme_signals().changed.connect(self._redraw_icons)

        self._to_start.clicked.connect(lambda: self.jump_requested.emit(0))
        self._back.clicked.connect(lambda: self.step_requested.emit(-1))
        self._play.clicked.connect(self.play_toggled.emit)
        self._forward.clicked.connect(lambda: self.step_requested.emit(1))
        self._to_end.clicked.connect(lambda: self.jump_requested.emit(self._duration))

        self._timecode = QLabel(self)
        monospace = QFont("Consolas")
        monospace.setStyleHint(QFont.StyleHint.Monospace)
        monospace.setPointSizeF(11)
        self._timecode.setFont(monospace)
        themed_style(self._timecode, lambda: f"color: {Colors.TEXT.name()};")

        self._duration_label = QLabel(self)
        self._duration_label.setFont(monospace)
        themed_style(self._duration_label, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._quality = QComboBox(self)
        for label, divisor in QUALITY_CHOICES:
            self._quality.addItem(label, divisor)
        self._quality.currentIndexChanged.connect(self._on_quality_changed)

        self._slash = QLabel("/", self)
        self._quality_label = QLabel("再生品質", self)
        self._quality.setToolTip("再生品質 下げるとプレビューだけ軽くなる 書き出しは変わらない")
        self._quality.setAccessibleName("再生品質")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        for button in (self._to_start, self._back, self._play, self._forward, self._to_end):
            layout.addWidget(button)
        layout.addSpacing(12)
        layout.addWidget(self._timecode)
        layout.addWidget(self._slash)
        layout.addWidget(self._duration_label)
        layout.addStretch(1)
        layout.addWidget(self._quality_label)
        layout.addWidget(self._quality)
        #: 幅が足りないときに隠す物 先に並べた組ほど残す 再生の操作・今の位置・画質の欄は
        #: 隠さない 隠すと操作できなくなる 全体の長さと欄の名前は、無くても操作はできる
        #: 隠さずに縮めると、文字が欠けた時刻や名前が出る
        self._optional: tuple[tuple[QWidget, ...], ...] = (
            (self._slash, self._duration_label),
            (self._quality_label,),
        )

        self._refresh()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt の命名規約
        """隠せる物を全部隠したときの幅 プレビューの列の最小の幅はここで決まる

        並べた全部の幅にすると、窓の最小の幅が 1366 の画面を超える
        """
        hint = super().minimumSizeHint()
        shown = [widget for group in self._optional for widget in group if not widget.isHidden()]
        return QSize(hint.width() - self._width_of(shown), hint.height())

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        self._fit(event.size().width())

    def _fit(self, width: int) -> None:
        """入るだけ出す 入らない組は丸ごと隠す（組の片方だけ出すと「/」だけが残る）"""
        room = width - self.minimumSizeHint().width()
        for group in self._optional:
            need = self._width_of(group)
            fits = room >= need
            for widget in group:
                widget.setHidden(not fits)
            if fits:
                room -= need
            else:
                # 先の組を隠したのに後の組だけ出すと、隠す順の約束が崩れる
                room = -1

    def _width_of(self, widgets: Sequence[QWidget]) -> int:
        layout = self.layout()
        spacing = layout.spacing() if layout is not None else 0
        return sum(widget.sizeHint().width() + spacing for widget in widgets)

    def set_rate(self, rate: FrameRate) -> None:
        self._rate = rate
        self._refresh()

    def set_frame(self, frame: int) -> None:
        self._frame = max(0, frame)
        self._refresh()

    def set_duration(self, frames: int) -> None:
        self._duration = max(0, frames)
        self._refresh()

    @property
    def playing(self) -> bool:
        return self._playing

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._play.setIcon(self._pause_icon if playing else self._play_icon)
        self._play.setAccessibleName("一時停止" if playing else "再生")

    def _redraw_icons(self) -> None:
        self._play_icon = transport_icon("play")
        self._pause_icon = transport_icon("pause")
        for button, glyph in (
            (self._to_start, "to_start"),
            (self._back, "back"),
            (self._forward, "forward"),
            (self._to_end, "to_end"),
        ):
            button.setIcon(transport_icon(glyph))
        self.set_playing(self._playing)

    def set_quality(self, divisor: int) -> None:
        """画質の選びを外から合わせる 一覧に無い分母なら何もしない

        選び直したときと同じ合図を出す 出さないと、表示だけ変わって
        実際の描画が前の画質のままになる
        """
        index = self._quality.findData(divisor)
        if index >= 0:
            self._quality.setCurrentIndex(index)

    def quality(self) -> int:
        return int(self._quality.currentData())

    def _on_quality_changed(self, index: int) -> None:
        self.quality_changed.emit(RenderQuality(int(self._quality.itemData(index))))

    def _refresh(self) -> None:
        self._timecode.setText(format_timecode(self._frame, self._rate))
        self._duration_label.setText(format_timecode(self._duration, self._rate))
        # 時刻の桁が変わる（60fps の素材を置いて :59 になる・1 時間を超える）と要る幅も変わる
        self._fit(self.width())


#: ボタンの印の形 16 × 16 の枠の中の、縦棒 ``(左, 上, 幅, 高さ)`` と三角（3 点）
#:
#: 文字（``▶`` ``⏸``）で出すと、Windows では ``⏸`` が絵文字の書体で描かれ、青い四角の
#: 絵になってほかのボタンと揃わなかった（Issue #27） 書体に頼らず、全部を同じ
#: 線の太さと色で自前で描く
_GLYPHS: dict[str, tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int], ...]]] = {
    "to_start": (((2, 3, 2, 10),), ((13, 3), (13, 13), (5, 8))),
    "back": (((12, 3, 2, 10),), ((11, 3), (11, 13), (3, 8))),
    "play": ((), ((4, 2), (4, 14), (13, 8))),
    "pause": (((4, 3, 3, 10), (9, 3, 3, 10)), ()),
    "forward": (((2, 3, 2, 10),), ((5, 3), (5, 13), (13, 8))),
    "to_end": (((12, 3, 2, 10),), ((3, 3), (3, 13), (11, 8))),
}

_GLYPH_SIZE = 16

#: 描く細かさ 画面の拡大率（125% や 150%）で引き伸ばしても角がぼけないよう、
#: 大きめに描いて縮めて見せる
_GLYPH_SCALE = 4


def transport_icon(name: str, color: QColor | None = None) -> QIcon:
    """再生ボタンの印 ``name`` は :data:`_GLYPHS` の鍵"""
    bars, triangle = _GLYPHS[name]
    pixmap = QPixmap(_GLYPH_SIZE * _GLYPH_SCALE, _GLYPH_SIZE * _GLYPH_SCALE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(_GLYPH_SCALE, _GLYPH_SCALE)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color if color is not None else Colors.TEXT)
    for left, top, width, height in bars:
        painter.drawRect(QRectF(left, top, width, height))
    if triangle:
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in triangle]))
    painter.end()
    icon = QIcon()
    icon.addPixmap(pixmap)
    return icon


def _tool_button(glyph: str, tooltip: str, name: str) -> QPushButton:
    button = QPushButton()
    button.setIcon(transport_icon(glyph))
    button.setIconSize(QSize(_GLYPH_SIZE, _GLYPH_SIZE))
    button.setToolTip(tooltip)
    # 印は絵なので、読み上げソフトには名前を別に渡す
    button.setAccessibleName(name)
    button.setFixedWidth(38)
    # ボタンにフォーカスが入ると、Space が再生ではなくボタンの押下になる
    # 再生ソフトで一番使うキーなので、そこは奪わせない
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button
