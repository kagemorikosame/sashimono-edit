"""設定パネルで、焦点の無い欄の上のホイールは値を変えずにパネルを送る

前は選択の欄・数値の欄・スライダーがカーソルの下にあるだけでホイールを受け、パネルを
ホイールで送る途中に通った欄の値（フォントのスタイルや不透明度）が変わっていた
焦点の無い欄でも変えたい人は、設定（``Preferences.wheel_unfocused``）で入れられる
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QPointF, QStringListModel, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFontComboBox,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollBar,
    QSlider,
    QWidget,
)

from sashimono.core.commands import insert_generated
from sashimono.core.model import Clip, Project
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.panel import InspectorPanel
from sashimono.ui.workspace import Preferences


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    # 送れるだけ低くする 窓は前に出さない（活性にしない）
    created.resize(420, 260)
    created.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    project = Project.create()
    for command in insert_generated(project, TEXT.create(align="center")):
        project = command.apply(project)
    clip: Clip = next(c for t in project.timeline.tracks for c in t.clips)
    created.set_project(project)
    created.set_clip(clip.id)
    created.show()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()
    yield created
    created.close()
    shiboken6.delete(created)


def _wheel(widget: QWidget, steps: int = -1) -> None:
    """``widget`` の真ん中で、ホイールを ``steps`` 刻み回す（負は手前へ＝下へ送る）"""
    centre = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(
        widget,
        QWheelEvent(
            centre,
            widget.mapToGlobal(centre),
            QPoint(0, 0),
            QPoint(0, 120 * steps),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )


def _choice_box(panel: InspectorPanel) -> QComboBox:
    """行揃えの選択の欄 選択肢（左・中央・右）が書体に頼らないので、どの機械でも試せる

    スタイルの欄で試すと、そのフォントのスタイルが入っていない機械では次の選択肢が無い
    """
    box = panel._editors[("source", "align")].findChild(QComboBox)
    assert box is not None
    return box


def _size_slider(panel: InspectorPanel) -> QSlider:
    slider = panel._editors[("source", "size")].findChild(QSlider)
    assert slider is not None
    return slider


def _opacity_box(panel: InspectorPanel) -> QDoubleSpinBox:
    box = panel._editors[("clip", "opacity")].findChild(QDoubleSpinBox)
    assert box is not None
    return box


def _font_box(panel: InspectorPanel) -> QFontComboBox:
    """フォントの欄 画面を出さない環境には書体が無いので、一覧へ名前を 200 件入れる

    Windows のように書体が多いと一覧にスクロールバーが出る その形を、書体の数に頼らず作る
    """
    box = panel._editors[("source", "font")].findChild(QFontComboBox)
    assert box is not None
    model = box.model()
    assert isinstance(model, QStringListModel), "フォントの欄の一覧の作りが変わった"
    # 欄の値が変わったと知らせると、パネルがクリップのフォントを書き換える 一覧だけを差し替える
    box.blockSignals(True)
    model.setStringList([f"書体 {number:03d}" for number in range(200)])
    box.blockSignals(False)
    return box


def _popup_bar(box: QComboBox) -> QScrollBar:
    """``box`` の一覧を開き、一覧の縦のスクロールバーを返す 送れる長さになっていることも見る"""
    box.setMaxVisibleItems(5)
    box.showPopup()
    QApplication.processEvents()
    bar = box.view().verticalScrollBar()
    assert bar.maximum() > 0, "一覧が送れる長さになっていない"
    return bar


def _list_scrolls(panel: InspectorPanel, box: QComboBox) -> None:
    """開いた一覧の上でホイールを回すと、一覧が送られ、パネルは動かない"""
    bar = _popup_bar(box)
    panel_bar = panel._scroll.verticalScrollBar()
    before, panel_before = bar.value(), panel_bar.value()
    try:
        for _ in range(3):
            _wheel(box.view().viewport())
        assert bar.value() > before
        assert panel_bar.value() == panel_before
    finally:
        box.hidePopup()


class TestUnfocusedFields:
    def test_a_choice_keeps_its_value_and_the_panel_scrolls(self, panel: InspectorPanel) -> None:
        # 壊れると、送る途中で通った行揃えが「中央」から「右」へ変わる（スタイルの欄なども同じ）
        box = _choice_box(panel)
        assert not box.hasFocus() and box.count() > 1
        bar = panel._scroll.verticalScrollBar()
        assert bar.maximum() > 0, "パネルが送れる高さになっていない"
        before = bar.value()
        _wheel(box)
        assert box.currentData() == "center"
        assert bar.value() > before

    def test_a_slider_keeps_its_value(self, panel: InspectorPanel) -> None:
        # 壊れると、送る途中で通った文字の大きさのスライダーが動く
        slider = _size_slider(panel)
        before = slider.value()
        _wheel(slider, steps=1)
        assert slider.value() == before

    def test_a_number_box_keeps_its_value_and_the_panel_scrolls(
        self, panel: InspectorPanel
    ) -> None:
        # 壊れると、送る途中で通った不透明度の欄の値が変わる
        box = _opacity_box(panel)
        assert not box.hasFocus()
        bar = panel._scroll.verticalScrollBar()
        before, value = bar.value(), box.value()
        _wheel(box)
        assert box.value() == value
        assert bar.value() > before

    def test_wheel_does_not_take_the_focus(self, panel: InspectorPanel) -> None:
        # ホイールで焦点を取ると、次の 1 刻みから値が変わる 焦点はクリックと Tab で取る
        assert _choice_box(panel).focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_the_setting_lets_the_wheel_change_values(self, panel: InspectorPanel) -> None:
        # 焦点の無い欄でも回して変えたい人のための設定 入れたら今までの動きに戻る
        panel.set_wheel_unfocused(True)
        box = _choice_box(panel)
        _wheel(box)
        assert box.currentData() == "right"
        slider = _size_slider(panel)
        before = slider.value()
        _wheel(slider, steps=1)
        assert slider.value() != before

    def test_the_default_setting_is_off(self) -> None:
        # 既定は知らない人が誤って値を変えない側
        assert Preferences().wheel_unfocused is False


class TestFocusedFields:
    def test_a_focused_choice_still_follows_the_wheel(self, panel: InspectorPanel) -> None:
        # 焦点を置いた選び欄までホイールを止めると、選んだ欄をホイールで切り替えられなくなる
        box = _choice_box(panel)
        panel.activateWindow()
        box.setFocus(Qt.FocusReason.MouseFocusReason)
        QApplication.processEvents()
        if not box.hasFocus():
            pytest.skip("窓が前に出られず、欄に焦点を置けない（CI の実行機など）")
        _wheel(box)
        assert box.currentData() == "right"


class TestScrollingLists:
    """欄の中の一覧や文字の欄は、ホイールで自分を送る 見張りがパネルへ回さない（#300）"""

    def test_the_font_list_scrolls_with_the_wheel(self, panel: InspectorPanel) -> None:
        # 壊れると、フォントの欄を開いても一覧がホイールで送れず、下の書体を選べない
        _list_scrolls(panel, _font_box(panel))

    def test_a_long_choice_list_scrolls_with_the_wheel(self, panel: InspectorPanel) -> None:
        # 壊れると、項目の多い選択の欄（スタイルや種類）の一覧がホイールで送れない
        box = _choice_box(panel)
        box.blockSignals(True)
        box.addItems([f"項目 {number:03d}" for number in range(100)])
        box.blockSignals(False)
        _list_scrolls(panel, box)
        # 一覧を送っても、欄の値は変わらない
        assert box.currentData() == "center"

    def test_the_guard_still_holds_the_fields_after_a_list_scrolls(
        self, panel: InspectorPanel
    ) -> None:
        # 壊れると、一覧を送れるようにした代わりに、パネルを送る途中で通った欄の値が変わる
        _list_scrolls(panel, _font_box(panel))
        box = _choice_box(panel)
        bar = panel._scroll.verticalScrollBar()
        before = bar.value()
        _wheel(box)
        assert box.currentData() == "center"
        assert bar.value() > before

    def test_a_long_text_scrolls_inside_its_box(self, panel: InspectorPanel) -> None:
        # 壊れると、長い文の欄の上でホイールを回しても文が送れず、パネルが動く
        area = panel._editors[("source", "text")].findChild(QPlainTextEdit)
        assert area is not None and not area.hasFocus()
        area.blockSignals(True)
        area.setPlainText("\n".join(f"{number} 行目" for number in range(60)))
        area.blockSignals(False)
        QApplication.processEvents()
        bar = area.verticalScrollBar()
        assert bar.maximum() > 0, "文の欄が送れる長さになっていない"
        panel_bar = panel._scroll.verticalScrollBar()
        before, panel_before = bar.value(), panel_bar.value()
        _wheel(area.viewport())
        assert bar.value() > before
        assert panel_bar.value() == panel_before

    def test_a_field_inside_a_list_is_left_to_the_list(self, panel: InspectorPanel) -> None:
        # 壊れると、一覧の中に置いた欄の上で回したホイールが、一覧の外のパネルを送る
        listing = QListWidget(panel._body)
        for number in range(50):
            listing.addItem(QListWidgetItem(f"行 {number}"))
        slider = QSlider(Qt.Orientation.Horizontal)
        listing.setItemWidget(listing.item(0), slider)
        panel._body_layout.addWidget(listing)
        panel._wheel_guard.watch(panel._body)
        QApplication.processEvents()
        holder = slider.parentWidget()
        assert holder is not None and isinstance(holder.parentWidget(), QAbstractItemView)
        panel_bar = panel._scroll.verticalScrollBar()
        before = panel_bar.value()
        _wheel(slider)
        assert panel_bar.value() == before
