"""数値欄の文字の欄を、縁の内側と増減ボタンの手前に収める

Qt 6.12（PySide6 6.12.0）のスタイルシートの見た目は、増減ボタンを 2 つとも右に置いたとき、
数値欄の文字の欄の左端を縁と余白を見ずに 0 にし、右端をボタンの左端と同じ列にする
（``QStyleSheetStyle::subControlRect`` の ``SC_SpinBoxEditField`` 6.11.2 までは縁と余白の
内側で、ボタンの 1 つ手前まで） このソフトはボタンを右の上下に決め打ちしている（Issue #27
:func:`sashimono.ui.theme.style_sheet`）ので、どの数値欄も次のようになる

- 数字が縁に掛かって描かれる（左の余白 6 画素が無くなる）
- 文字の欄がボタンの左端の列に重なり、その列を押しても増えも減りもしない

CI が PySide6 6.12.0 を入れた日から、`tests/ui/test_theme.py` の数値欄の試験が落ちた
（PR #262 のマージと同じ頃で、当初はその変更を疑った） 配る版も次に組むときに 6.12 になる

数値欄が文字の欄を置き直すたび（大きさが変わる・見た目が変わる・出る）に、置かれた枠を
縁と余白の内側・ボタンの手前まで縮める 縮めるだけなので、正しく置く版（6.11.2 まで）では
何も変わらない Qt が直したら外してよい
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QStyle,
    QStyleOptionSpinBox,
)

from sashimono.ui.theme import Metrics

__all__ = ["SpinFieldFit", "fitted_edit_field", "install_spin_field_fit"]

#: 入れた見張りを QApplication に覚えさせる名前 テーマを何度当て直しても 1 つだけ入れる
_INSTALLED = "_sashimono_spin_field_fit"


def fitted_edit_field(spin: QAbstractSpinBox, placed: QRect) -> QRect:
    """Qt が置いた文字の欄の枠 ``placed`` を、縁と余白の内側・増減ボタンの手前に縮めた枠

    縮める要の無いとき（スタイルシートを当てていない・ボタンが無い・ボタンが右に無い）は
    ``placed`` のまま返す 元の見た目（Windows 11）は自分で正しく置くので触らない
    """
    style = spin.style()
    if style is None or style.metaObject().className() != "QStyleSheetStyle":
        return placed
    if spin.buttonSymbols() is QAbstractSpinBox.ButtonSymbols.NoButtons:
        return placed
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    control = QStyle.ComplexControl.CC_SpinBox
    up = style.subControlRect(control, option, QStyle.SubControl.SC_SpinBoxUp, spin)
    down = style.subControlRect(control, option, QStyle.SubControl.SC_SpinBoxDown, spin)
    frame = style.subControlRect(control, option, QStyle.SubControl.SC_SpinBoxFrame, spin)
    buttons = min(up.left(), down.left())
    if buttons <= placed.left():
        # ボタンが文字の欄の左にある（右から左へ書く言葉の並び） このソフトでは置かない
        return placed
    left = max(placed.left(), frame.left() + Metrics.FIELD_BORDER + Metrics.FIELD_PADDING_X)
    right = min(placed.right(), buttons - 1)
    if right < left:
        return placed
    return QRect(QPoint(left, placed.top()), QPoint(right, placed.bottom()))


class SpinFieldFit(QObject):
    """数値欄の文字の欄が置かれるたびに、:func:`fitted_edit_field` の枠へ置き直す見張り"""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        # 種類を先に見る アプリ全体の出来事が全部ここを通るので、数値欄の物以外はすぐ返す
        if event.type() not in (QEvent.Type.Move, QEvent.Type.Resize):
            return False
        if not isinstance(watched, QLineEdit):
            return False
        spin = watched.parentWidget()
        if not isinstance(spin, QAbstractSpinBox):
            return False
        placed = watched.geometry()
        fitted = fitted_edit_field(spin, placed)
        if fitted != placed:
            # 置き直すとまた Move と Resize が来るが、そのときは枠が同じなので止まる
            watched.setGeometry(fitted)
        return False


def install_spin_field_fit(app: QApplication) -> SpinFieldFit:
    """アプリ全体へ見張りを入れる 何度呼んでも 1 つだけ

    部品ごとではなくアプリへ入れるのは、設定の窓や設定パネルのように後から作る数値欄も
    同じに扱うため 数値欄を作る所すべてに手を入れると、足し忘れた所だけ崩れる
    """
    existing = app.property(_INSTALLED)
    if isinstance(existing, SpinFieldFit):
        return existing
    fit = SpinFieldFit(app)
    app.installEventFilter(fit)
    app.setProperty(_INSTALLED, fit)
    return fit
