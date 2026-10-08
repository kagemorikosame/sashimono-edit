"""焦点の無い入力欄へのホイールを、欄ではなく周りの送り（スクロール）へ渡す

Qt の選択の欄・数値の欄・スライダーは、焦点が無くてもカーソルの下でホイールを回すと
値を変える 設定パネルをホイールで送っている途中に欄の上を通ると、触ったつもりの無い
値（フォントのスタイルや不透明度）が変わり、取り消しの段まで積まれる

焦点はクリックか Tab で取る（:attr:`Qt.FocusPolicy.StrongFocus`） 焦点のある欄は
今までどおりホイールで値を変える 焦点が無くても変えたい人は設定で切れる
（:attr:`sashimono.ui.workspace.Preferences.wheel_unfocused`）
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QWidget,
)

__all__ = ["WheelGuard"]

#: ホイールで値の変わる部品 スクロールバーは送りそのものなので含めない
_VALUE_WIDGETS: tuple[type[QWidget], ...] = (QComboBox, QAbstractSpinBox, QAbstractSlider)


class WheelGuard(QObject):
    """``scroll`` の中の入力欄が、焦点の無いときに受けたホイールを ``scroll`` へ回す"""

    def __init__(self, scroll: QAbstractScrollArea, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._scroll = scroll
        #: 偽なら何もしない（焦点の無い欄でもホイールで値を変える 設定から）
        self.enabled = True

    def watch(self, root: QWidget) -> None:
        """``root`` の下の入力欄を見張る 作り直した欄にも掛かるよう、作り直すたびに呼ぶ

        同じ部品へ何度掛けても Qt は 1 つにまとめるので、前から見張っている欄があってよい
        """
        for kind in _VALUE_WIDGETS:
            for widget in root.findChildren(kind):
                if isinstance(widget, QAbstractSlider) and widget is self._scroll_bar(widget):
                    continue
                # ホイールで焦点を取る（WheelFocus）と、送りの途中で通った欄が焦点を奪い、
                # 次の 1 刻みから値を変えてしまう クリックと Tab だけで取る
                if widget.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                widget.installEventFilter(self)

    def _scroll_bar(self, widget: QAbstractSlider) -> QAbstractSlider | None:
        """``widget`` が送りの縦か横のスクロールバーなら、それを返す"""
        if widget in (self._scroll.verticalScrollBar(), self._scroll.horizontalScrollBar()):
            return widget
        return None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        if (
            not self.enabled
            or event.type() != QEvent.Type.Wheel
            or not isinstance(watched, QWidget)
            or watched.hasFocus()
        ):
            return super().eventFilter(watched, event)
        # 欄で止めるだけだと、パネルの送りも止まる（欄の上でホイールが効かなくなる）
        # 送りの見える所へ回して、欄の無い所と同じに送る
        QApplication.sendEvent(self._scroll.viewport(), event)
        return True
