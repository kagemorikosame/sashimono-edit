"""見た目の決まり（スタイルシート）で、読めない・押せないが起きていないか

スタイルシートは書き方 1 つで、元の見た目（Windows 11 など）との組み合わせが崩れる
崩れても例外にはならず、画面を見て初めて気付く 描いた絵と、押したときの結果で確かめる

スタイルシートはアプリ全体ではなく、試す窓にだけ当てる アプリ全体に当てると、
ほかの試験の窓まで見た目が変わる
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QDoubleSpinBox,
    QLabel,
    QMainWindow,
    QMenu,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.model import ProjectSettings
from sashimono.ui.project_settings_dialog import ProjectSettingsDialog
from sashimono.ui.theme import Colors, style_sheet


def _contrast(first: QColor, second: QColor) -> float:
    """2 色の明るさの比（WCAG の式） 4.5 以上あれば小さい字でも読める"""

    def luminance(color: QColor) -> float:
        def channel(value: float) -> float:
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

        return (
            0.2126 * channel(color.redF())
            + 0.7152 * channel(color.greenF())
            + 0.0722 * channel(color.blueF())
        )

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _count(
    image: QImage, rect_left: int, rect_top: int, width: int, height: int, color: QColor
) -> int:
    """矩形の中で ``color`` とほぼ同じ色の画素の数"""
    found = 0
    for y in range(rect_top, rect_top + height):
        for x in range(rect_left, rect_left + width):
            pixel = image.pixelColor(x, y)
            if (
                abs(pixel.red() - color.red()) <= 8
                and abs(pixel.green() - color.green()) <= 8
                and abs(pixel.blue() - color.blue()) <= 8
            ):
                found += 1
    return found


class TestTabs:
    @pytest.fixture
    def tabs(self, qt_application: QApplication) -> Iterator[QTabWidget]:
        del qt_application
        host = QWidget()
        host.setStyleSheet(style_sheet())
        widget = QTabWidget(host)
        widget.addTab(QLabel("中身"), "メディア")
        widget.addTab(QLabel("中身"), "字幕")
        widget.resize(300, 120)
        host.resize(300, 120)
        host.show()
        QApplication.processEvents()
        yield widget
        _dispose(host)

    def test_the_chosen_tab_is_readable(self) -> None:
        # 選んだタブの地が明るい灰色だと、白に近い文字が読めなかった（Issue #27）
        assert _contrast(Colors.TAB_SELECTED, Colors.CLIP_LABEL) >= 4.5
        assert _contrast(Colors.WINDOW, Colors.TEXT_MUTED) >= 4.5

    def test_the_chosen_tab_stands_out(self, tabs: QTabWidget) -> None:
        # 選んだタブと選んでいないタブが同じ見た目だと、どれを見ているのか分からない
        # 地の色が変わり、アクセント色の線が選んだタブにだけ付く
        image = tabs.grab().toImage()
        bar = tabs.tabBar()
        offset = bar.mapTo(tabs, QPoint(0, 0))
        chosen, other = bar.tabRect(0), bar.tabRect(1)

        def accents(index: int) -> int:
            rect = bar.tabRect(index).translated(offset)
            return _count(
                image, rect.left(), rect.top(), rect.width(), rect.height(), Colors.ACCENT
            )

        assert accents(0) >= chosen.width()
        assert accents(1) == 0
        middle = QPoint(chosen.left() + 4, chosen.center().y()) + offset
        assert image.pixelColor(middle).name() == Colors.TAB_SELECTED.name()
        middle = QPoint(other.left() + 4, other.center().y()) + offset
        assert image.pixelColor(middle).name() == Colors.WINDOW.name()

    def test_choosing_moves_the_mark(self, tabs: QTabWidget) -> None:
        # 選び直したときに線が付いてこないと、最初のタブを見ているように見える
        tabs.setCurrentIndex(1)
        QApplication.processEvents()
        image = tabs.grab().toImage()
        bar = tabs.tabBar()
        rect = bar.tabRect(1).translated(bar.mapTo(tabs, QPoint(0, 0)))
        assert _count(image, rect.left(), rect.top(), rect.width(), rect.height(), Colors.ACCENT)

    def test_docked_tabs_follow_the_same_rule(self, qt_application: QApplication) -> None:
        # 重ねたドック（メディアと字幕）のタブは窓が自分で作る 設定の窓のタブにだけ
        # 当てても、利用者が最初に見るこちらが読めないまま残る
        # アプリの窓（MainWindow）ではなく、ドックを重ねただけの窓で見る アプリの窓に
        # スタイルシートを当てて閉じただけで残すと、CI（GPU の無い Windows）ではあとの
        # ごみ集めで壊すときにプロセスごと落ちた 重ねたドックのタブを作るのは
        # QMainWindow なので、ここで確かめたい決まりは同じ
        del qt_application
        window = QMainWindow()
        window.setStyleSheet(style_sheet())
        window.setCentralWidget(QLabel("中央"))
        docks = [QDockWidget(title, window) for title in ("メディア", "字幕")]
        for dock in docks:
            dock.setWidget(QLabel(dock.windowTitle()))
            window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        window.tabifyDockWidget(docks[0], docks[1])
        docks[0].raise_()
        window.resize(640, 400)
        window.show()
        QApplication.processEvents()
        try:
            bars = [bar for bar in window.findChildren(QTabBar) if bar.count() > 1]
            assert bars
            for bar in bars:
                image = bar.grab().toImage()
                chosen = bar.tabRect(bar.currentIndex())
                assert _count(
                    image,
                    chosen.left(),
                    chosen.top(),
                    chosen.width(),
                    chosen.height(),
                    Colors.ACCENT,
                ), [bar.tabText(i) for i in range(bar.count())]
        finally:
            _dispose(window)


def _button_rects(spin: QSpinBox | QDoubleSpinBox) -> tuple[QPoint, QPoint]:
    """上と下のボタンの真ん中 描く所と同じ計算（見た目の部品の位置）から出す"""
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    style = spin.style()
    up = style.subControlRect(
        QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxUp, spin
    )
    down = style.subControlRect(
        QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxDown, spin
    )
    return up.center(), down.center()


def _click_like_a_mouse(spin: QWidget, point: QPoint) -> None:
    """本物のマウスと同じく、その点の一番手前の部品へ押下を届ける

    ``QTest.mouseClick(spin, ...)`` は数値欄そのものへ直に送るので、上に数字の欄
    （QLineEdit）が重なっていても押せてしまい、壊れていても通る
    """
    target = spin.childAt(point)
    if target is None:
        QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=point)
    else:
        QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=target.mapFrom(spin, point))


def _dispose(root: QWidget) -> None:
    """窓をその場で壊す

    閉じただけで残すと、いつかのごみ集めで壊され、そのとき走っている別の試験の
    途中で落ちたように見える どの試験の後始末なのかが分からなくなる
    """
    root.close()
    shiboken6.delete(root)


class TestSpinButtons:
    """元の見た目はアプリのものをそのまま使う（Windows 11 の手元では windows11）

    試験の中で見た目を差し替えない アプリ全体を切り替えても、部品ごとに当てても、
    差し替えた見た目と部品の壊れる順がずれ、あとのごみ集めで消えた見た目を触って
    CI がプロセスごと落ちた（access violation）
    """

    def test_the_up_button_of_the_resolution_steps_up(self) -> None:
        # Windows 11 の見た目では上下のボタンが横に並ぶのに、数字の欄がボタン 1 つぶん
        # しか空けずに広がり、上のボタンが数字の欄の下に隠れていた 押しても数字の欄が
        # 受け取るので、上だけ数が変わらなかった（Issue #27）
        dialog = ProjectSettingsDialog(ProjectSettings(), new=True)
        dialog.setStyleSheet(style_sheet())
        dialog.show()
        QApplication.processEvents()
        try:
            spin = dialog._width
            start = spin.value()
            up, down = _button_rects(spin)
            _click_like_a_mouse(spin, up)
            assert spin.value() == start + spin.singleStep()
            _click_like_a_mouse(spin, down)
            _click_like_a_mouse(spin, down)
            assert spin.value() == start - spin.singleStep()
        finally:
            _dispose(dialog)

    def test_the_edit_field_leaves_the_buttons_free(self) -> None:
        # 数字の欄がボタンに掛かると、掛かった所を押しても増えも減りもしない
        # 小数の数値欄（設定パネルの音量など）も同じ決まりで並ぶ
        host = QWidget()
        host.setStyleSheet(style_sheet())
        layout = QVBoxLayout(host)
        spins: list[QSpinBox | QDoubleSpinBox] = [QSpinBox(host), QDoubleSpinBox(host)]
        for spin in spins:
            spin.setRange(0, 400)
            spin.setValue(100)
            layout.addWidget(spin)
        spins[1].setSuffix(" %")
        spins[1].setFixedWidth(96)
        host.show()
        QApplication.processEvents()
        try:
            for spin in spins:
                option = QStyleOptionSpinBox()
                spin.initStyleOption(option)
                for control in (QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown):
                    button = spin.style().subControlRect(
                        QStyle.ComplexControl.CC_SpinBox, option, control, spin
                    )
                    assert not spin.lineEdit().geometry().intersects(button), type(spin).__name__
                up, _ = _button_rects(spin)
                _click_like_a_mouse(spin, up)
                assert spin.value() == 101, type(spin).__name__
        finally:
            _dispose(host)


class TestMenus:
    def test_the_shortcut_does_not_run_into_the_text(self, qt_application: QApplication) -> None:
        """メニューの項目の文言と、右に出るショートカットの間が空いていること

        アプリ全体の文字の大きさをスタイルシートで決めていると、項目の幅が文言と
        ショートカットの和に足りず、長い項目（〔互換〕→〔オブジェクトを読み込む…〕）では
        文言の終わりに Ctrl+Shift+O が重なって読めなくなっていた（手順書の写真で見つけた）
        描いた絵の 1 行で、字の無い列がいちばん長く続く所を測る
        """
        del qt_application
        menu = QMenu()
        menu.setStyleSheet(style_sheet())
        menu.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        action = menu.addAction("オブジェクトを読み込む…")
        action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        menu.popup(QPoint(0, 0))
        QApplication.processEvents()
        try:
            image = menu.grab().toImage()
            ratio = image.width() / max(1, menu.width())
            rect = menu.actionGeometry(action)
            top, bottom = int(rect.top() * ratio), int(rect.bottom() * ratio)
            background = image.pixelColor(int((rect.left() + 2) * ratio), top + 1).lightness()
            inked = [
                x
                for x in range(int(rect.left() * ratio), int(rect.right() * ratio))
                if any(
                    abs(image.pixelColor(x, y).lightness() - background) > 40
                    for y in range(top, bottom)
                )
            ]
            gaps = [after - before for before, after in itertools.pairwise(inked)]
            assert gaps
            assert max(gaps) / ratio >= 12
        finally:
            menu.hide()
            menu.deleteLater()
