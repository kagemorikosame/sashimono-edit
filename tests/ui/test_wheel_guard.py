"""設定パネルで、焦点の無い欄の上のホイールは値を変えずにパネルを送る

前は選択の欄・数値の欄・スライダーがカーソルの下にあるだけでホイールを受け、パネルを
ホイールで送る途中に通った欄の値（フォントのスタイルや不透明度）が変わっていた
焦点の無い欄でも変えたい人は、設定（``Preferences.wheel_unfocused``）で入れられる
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox, QSlider, QWidget

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
    for command in insert_generated(project, TEXT.create(font_style="")):
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


def _style_box(panel: InspectorPanel) -> QComboBox:
    box = panel._editors[("source", "font_style")].findChild(QComboBox)
    assert box is not None
    return box


def _size_slider(panel: InspectorPanel) -> QSlider:
    slider = panel._editors[("source", "size")].findChild(QSlider)
    assert slider is not None
    return slider


class TestUnfocusedFields:
    def test_a_choice_keeps_its_value_and_the_panel_scrolls(self, panel: InspectorPanel) -> None:
        # 壊れると、送る途中で通ったスタイルの欄が「既定」から別のスタイルへ変わる
        box = _style_box(panel)
        assert not box.hasFocus() and box.count() > 1
        bar = panel._scroll.verticalScrollBar()
        assert bar.maximum() > 0, "パネルが送れる高さになっていない"
        before = bar.value()
        _wheel(box)
        assert box.currentIndex() == 0
        assert bar.value() > before

    def test_a_slider_keeps_its_value(self, panel: InspectorPanel) -> None:
        slider = _size_slider(panel)
        before = slider.value()
        _wheel(slider, steps=1)
        assert slider.value() == before

    def test_wheel_does_not_take_the_focus(self, panel: InspectorPanel) -> None:
        # ホイールで焦点を取ると、次の 1 刻みから値が変わる 焦点はクリックと Tab で取る
        assert _style_box(panel).focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_the_setting_lets_the_wheel_change_values(self, panel: InspectorPanel) -> None:
        # 焦点の無い欄でも回して変えたい人のための設定 入れたら今までの動きに戻る
        panel.set_wheel_unfocused(True)
        box = _style_box(panel)
        _wheel(box)
        assert box.currentIndex() == 1
        slider = _size_slider(panel)
        before = slider.value()
        _wheel(slider, steps=1)
        assert slider.value() != before

    def test_the_default_setting_is_off(self) -> None:
        # 既定は知らない人が誤って値を変えない側
        assert Preferences().wheel_unfocused is False


class TestFocusedFields:
    def test_a_focused_choice_still_follows_the_wheel(self, panel: InspectorPanel) -> None:
        box = _style_box(panel)
        panel.activateWindow()
        box.setFocus(Qt.FocusReason.MouseFocusReason)
        QApplication.processEvents()
        if not box.hasFocus():
            pytest.skip("窓が前に出られず、欄に焦点を置けない（CI の実行機など）")
        _wheel(box)
        assert box.currentIndex() == 1
