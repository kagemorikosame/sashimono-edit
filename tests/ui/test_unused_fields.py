"""設定パネルに、今の値で使わない欄を出さない

図形は種類ごとに読む項目が違い、全部を並べると 60 近い欄が並んで、どれを動かせば
変わるのかが分からなかった 場面切り替えは不透明度・合成モード・クリッピングを読まず、
切り替え方によっては向きやイージングも読まない 隠すだけで値は消さない
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.core.commands import insert_generated
from sashimono.core.model import Clip, GeneratedSource, Project
from sashimono.effects.sources import SHAPE, TRANSITION
from sashimono.ui.inspector.panel import InspectorPanel


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def _shown(panel: InspectorPanel, source: GeneratedSource) -> set[str]:
    project = Project.create()
    for command in insert_generated(project, source):
        project = command.apply(project)
    clip: Clip = next(c for t in project.timeline.tracks for c in t.clips)
    panel.set_project(project)
    panel.set_clip(clip.id)
    # 中身（source）とクリップ自身（clip）の欄だけを見る 描画の欄の X や回転は別の持ち主
    own = {name for owner, name in panel._editors if owner in ("source", "clip")}
    checks = {widget.objectName() for widget in panel.findChildren(QWidget)}
    return own | checks


class TestShape:
    def test_a_rectangle_hides_the_other_kinds(self, panel: InspectorPanel) -> None:
        # 壊れると、四角形に星空・集中線・音声波形の欄まで並ぶ
        shown = _shown(panel, SHAPE.create(shape="rect"))
        assert {"width", "height", "color", "line_width", "rotation"} <= shown
        assert not {"star_count", "density", "wave_volume", "corner_radius", "points"} & shown

    def test_a_starfield_shows_its_own(self, panel: InspectorPanel) -> None:
        shown = _shown(panel, SHAPE.create(shape="starfield"))
        assert {"star_count", "star_speed", "color"} <= shown
        # 星空は画面の座標で描き、大きさ・位置・回転を読まない
        assert not {"width", "height", "pos_x", "rotation", "corner_radius"} & shown

    def test_hidden_values_are_kept(self) -> None:
        # 隠した欄の値を捨てると、種類を戻したときに前の値が消えている
        source = SHAPE.create(shape="starfield", corner_radius=80)
        assert source.params["corner_radius"].static == 80  # type: ignore[union-attr]
        assert "corner_radius" in SHAPE.unused_names(source.params)


class TestTransition:
    def test_it_shows_no_picture_items(self, panel: InspectorPanel) -> None:
        # 場面切り替えは不透明度・合成モード・クリッピングを読まない
        shown = _shown(panel, TRANSITION.create())
        assert "opacity" not in shown
        assert "clip_clip_to_below" not in shown
        assert "style" in shown

    def test_a_crossfade_hides_the_direction(self, panel: InspectorPanel) -> None:
        shown = _shown(panel, TRANSITION.create(style="fade"))
        assert "angle" not in shown and "target" not in shown
        assert "easing" in shown

    def test_a_slide_shows_the_direction(self, panel: InspectorPanel) -> None:
        shown = _shown(panel, TRANSITION.create(style="slide"))
        assert {"angle", "target", "easing"} <= shown

    def test_a_cut_does_not_ease(self, panel: InspectorPanel) -> None:
        # 切り替えは真ん中で入れ替わるだけで、進み具合を読まない
        shown = _shown(panel, TRANSITION.create(style="switch"))
        assert not {"easing", "easing_mode", "angle"} & shown
