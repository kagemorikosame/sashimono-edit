"""フィルタのクリップを画面でどう見せるか（Issue #27）

タイムラインで映像のクリップと同じ色だと、絵を持つクリップと取り違える
設定パネルでは、ほかのクリップと同じ作りでエフェクトを積めないと、置いても何も掛けられない
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects import registry
from sashimono.effects.sources import FILTER
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.inspector import InspectorPanel
from sashimono.ui.theme import Colors
from sashimono.ui.timeline import TimelineView


def _project() -> tuple[Project, Clip]:
    invert = registry.get("invert")
    assert invert is not None
    clip = Clip(timeline_start=0, duration=60, source=FILTER.create(), effects=(invert.create(),))
    base = Project.create()
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    return base.with_timeline(replace(base.timeline, tracks=(track,))), clip


@pytest.fixture
def analyzer(qt_application: QApplication) -> Iterator[MediaAnalyzer]:
    del qt_application
    created = MediaAnalyzer(sample_rate=48000, channels=2)
    yield created
    created.close()


def test_the_timeline_paints_a_filter_in_its_own_colour(analyzer: MediaAnalyzer) -> None:
    # フィルタのクリップを映像と同じ色で塗ると、タイムラインでどれが画面全体に効く物か見分けられない
    project, _ = _project()
    view = TimelineView(project, analyzer)
    view.resize(600, 160)
    view.zoom_to_fit()
    image = QImage(view.size(), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    view.render(painter, QPoint())
    painter.end()
    colours = {
        image.pixelColor(x, y).name()
        for x in range(0, image.width(), 3)
        for y in range(0, image.height(), 3)
    }
    assert Colors.FILTER_CLIP.name() in colours
    # 映像のクリップの色で塗っていたら見分けが付かない
    assert Colors.VIDEO_CLIP.name() not in colours


def test_the_inspector_edits_the_filter_effects(qt_application: QApplication) -> None:
    # フィルタのクリップでエフェクトの欄が出ないと、置いても中身を足したり変えたりできない
    del qt_application
    project, clip = _project()
    panel = InspectorPanel()
    panel.set_project(project)
    panel.set_clip(clip.id)
    texts = [label.text() for label in panel.findChildren(QLabel)]
    assert panel._add_button.isEnabled()
    assert "色の反転" in texts
    # 合成方法は使わないので出さない 出すと、選んでも何も変わらない欄になる
    assert "合成方法" not in texts
    assert not panel.findChildren(QComboBox)
    assert "不透明度" in texts
