"""設定パネルの縁取りの層の組（#272 #273）

層を持たない字の縁は仮の層として見せ、触ったときに本物の層へ移す（1 回の取り消しで戻る）
層を足す・消す・並べ替える・隠す、層だけにエフェクトを掛ける操作が、どれも命令として出ること
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QToolButton

from sashimono.core.commands import (
    AddEffect,
    AddStroke,
    AdoptLegacyStroke,
    Command,
    MoveStroke,
    RemoveStroke,
    SetParam,
    SetStrokeEnabled,
    insert_generated,
)
from sashimono.core.model import (
    MAX_STROKES,
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    Stroke,
    StrokeId,
)
from sashimono.effects.sources import TEXT
from sashimono.effects.strokes import STROKE_EFFECT_KINDS
from sashimono.ui.inspector.panel import InspectorPanel
from sashimono.ui.timeline.keyframes import keyframe_frames


class _Harness:
    """設定パネルが出した命令を当てて、パネルへ戻す（本体の窓の代わり）"""

    def __init__(self, panel: InspectorPanel, source: GeneratedSource) -> None:
        self.panel = panel
        project = Project.create()
        for command in insert_generated(project, source):
            project = command.apply(project)
        self.project = project
        self.clip_id = next(c for t in project.timeline.tracks for c in t.clips).id
        self.sent: list[tuple[list[Command], str]] = []
        panel.commands_requested.connect(self._apply)
        panel.set_project(project)
        panel.set_clip(self.clip_id)

    def _apply(self, commands: list[Command], label: str) -> None:
        self.sent.append((list(commands), label))
        for command in commands:
            self.project = command.apply(self.project)
        self.panel.set_project(self.project)

    @property
    def clip(self) -> Clip:
        located = self.project.timeline.locate_clip(self.clip_id)
        assert located is not None
        return located[1]

    @property
    def strokes(self) -> tuple[Stroke, ...]:
        assert self.clip.source is not None
        return self.clip.source.strokes

    def headings(self) -> list[str]:
        return [section.heading for section in self.panel._sections()]


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def _button(panel: InspectorPanel, heading: str, text: str) -> QToolButton:
    section = next(s for s in panel._sections() if s.heading == heading)
    return next(b for b in section.findChildren(QToolButton) if b.text() == text)


def _choices(menu: QMenu) -> list[QAction]:
    """メニューの分類の下に並んだ、選べるエフェクト"""
    return [
        action
        for submenu in menu.findChildren(QMenu)
        for action in submenu.actions()
        if action.data() is not None
    ]


def _stroke_editor_owner(panel: InspectorPanel) -> list[str]:
    return sorted({owner for owner, _ in panel._editors if owner.startswith("stroke:")})


class TestALegacyBorder:
    def test_it_shows_as_a_layer_without_changing_the_clip(self, panel: InspectorPanel) -> None:
        # 前からの縁を、開いただけで層へ移すと、見ただけの作品に変更が入る
        harness = _Harness(panel, TEXT.create(border_width=6))
        assert "縁取り 1" in harness.headings()
        assert harness.strokes == ()
        assert harness.sent == []
        # 中身の組には前からの欄を出さない（2 か所に出すと、どちらを動かすか分からない）
        assert ("source", "border_width") not in panel._editors

    def test_touching_it_moves_it_into_a_layer_in_one_step(self, panel: InspectorPanel) -> None:
        harness = _Harness(panel, TEXT.create(border_width=6))
        (owner,) = _stroke_editor_owner(panel)
        editor = panel._editors[(owner, "width")]
        editor.value_changed.emit(AnimatedValue(9.0))
        (commands, _), *_ = harness.sent
        # 移す命令と値を入れる命令が 1 回で出る（1 回の取り消しで両方戻る）
        assert [type(command) for command in commands] == [AdoptLegacyStroke, SetParam]
        assert [stroke.params["width"] for stroke in harness.strokes] == [AnimatedValue(9.0)]
        assert harness.clip.source is not None
        assert harness.clip.source.params["border_width"] == AnimatedValue(0.0)

    def test_the_layer_shows_the_old_values(self, panel: InspectorPanel) -> None:
        _Harness(panel, TEXT.create(border_width=6, border_color=(1.0, 0.0, 0.0, 1.0)))
        (owner,) = _stroke_editor_owner(panel)
        clip = panel._clip()
        assert clip is not None
        assert panel._lookup(clip, owner, "width") == AnimatedValue(6.0)
        assert panel._lookup(clip, owner, "color") == (1.0, 0.0, 0.0, 1.0)


class TestLayers:
    def test_adding_a_layer(self, panel: InspectorPanel) -> None:
        harness = _Harness(panel, TEXT.create(border_width=6))
        add = panel.findChild(QPushButton, "add_stroke")
        assert add is not None and add.isEnabled()
        add.click()
        (commands, _), *_ = harness.sent
        assert [type(command) for command in commands] == [AddStroke]
        widths = [stroke.params["width"] for stroke in harness.strokes]
        # 前の縁が 1 つ目になり、足した層はその外に見える太さ
        assert widths[0] == AnimatedValue(6.0)
        assert isinstance(widths[1], AnimatedValue) and widths[1].static > 6.0
        assert harness.headings().count("縁取り 1") == 1 and "縁取り 2" in harness.headings()

    def test_moving_hiding_and_removing(self, panel: InspectorPanel) -> None:
        harness = _Harness(panel, TEXT.create())
        add = panel.findChild(QPushButton, "add_stroke")
        assert add is not None
        add.click()
        add = panel.findChild(QPushButton, "add_stroke")
        assert add is not None
        add.click()
        first, second = (stroke.id for stroke in harness.strokes)
        _button(panel, "縁取り 2", "▲").click()
        assert [stroke.id for stroke in harness.strokes] == [second, first]
        assert isinstance(harness.sent[-1][0][0], MoveStroke)
        toggle = _button(panel, "縁取り 1", "有効")
        toggle.click()
        assert isinstance(harness.sent[-1][0][0], SetStrokeEnabled)
        assert [stroke.enabled for stroke in harness.strokes] == [False, True]
        _button(panel, "縁取り 1", "✕").click()
        assert isinstance(harness.sent[-1][0][0], RemoveStroke)
        assert [stroke.id for stroke in harness.strokes] == [first]

    def test_the_limit_disables_the_button(self, panel: InspectorPanel) -> None:
        strokes = tuple(
            Stroke(params={"width": AnimatedValue(1.0 + n)}, id=StrokeId(f"s{n}"))
            for n in range(MAX_STROKES)
        )
        _Harness(panel, TEXT.create().with_strokes(strokes))
        add = panel.findChild(QPushButton, "add_stroke")
        assert add is not None and not add.isEnabled()

    def test_layer_values_stay_with_the_main_clip(self, panel: InspectorPanel) -> None:
        # 層の数と並びはクリップごとに違う 選んだほかのクリップへ同じ番号の層の値を当てると、
        # 別の太さの層まで同じ値になる
        source = TEXT.create().with_strokes((Stroke(params={"width": AnimatedValue(3.0)}),))
        harness = _Harness(panel, source)
        project = harness.project
        for command in insert_generated(project, source, at_frame=400):
            project = command.apply(project)
        harness.project = project
        other = next(
            c.id for t in project.timeline.tracks for c in t.clips if c.id != harness.clip_id
        )
        panel.set_project(project)
        panel.set_selection((harness.clip_id, other))
        owner = _stroke_editor_owner(panel)[0]
        panel._editors[(owner, "width")].value_changed.emit(AnimatedValue(5.0))
        (commands, _), *_ = harness.sent
        assert len(commands) == 1


class TestLayerEffects:
    def test_the_menu_offers_only_layer_effects(self, panel: InspectorPanel) -> None:
        harness = _Harness(panel, TEXT.create(border_width=6))
        menu = panel.stroke_effect_menu(harness.clip_id, StrokeId("どれでも"))
        kinds = [action.data()[2] for action in _choices(menu)]
        menu.deleteLater()
        assert kinds and set(kinds) <= set(STROKE_EFFECT_KINDS)
        assert "noise" not in kinds and "transform" not in kinds

    def test_adding_an_effect_to_the_old_border(self, panel: InspectorPanel) -> None:
        # 仮の層へエフェクトを足すと、先に層へ移してから足す
        harness = _Harness(panel, TEXT.create(border_width=6))
        menu = panel.stroke_effect_menu(
            harness.clip_id, StrokeId(_stroke_editor_owner(panel)[0][7:])
        )
        blur = next(action for action in _choices(menu) if action.data()[2] == "blur")
        panel._add_stroke_effect(blur)
        menu.deleteLater()
        (commands, _), *_ = harness.sent
        assert [type(command) for command in commands] == [AdoptLegacyStroke, AddEffect]
        (stroke,) = harness.strokes
        assert [effect.kind for effect in stroke.effects] == ["blur"]
        assert harness.clip.effects == () or all(e.fixed for e in harness.clip.effects)
        assert "ぼかし（縁取り 1）" in harness.headings()
        # 層のエフェクトの値も設定パネルから変えられる
        effect_id = stroke.effects[0].id
        panel._editors[(str(effect_id), "radius")].value_changed.emit(AnimatedValue(12.0))
        assert harness.strokes[0].effects[0].params["radius"] == AnimatedValue(12.0)


class TestTimelineMarks:
    def test_layer_keyframes_show_on_the_timeline(self) -> None:
        # 縁取りの層の値と層のエフェクトの値のキーも、タイムラインの印に数える
        # 数え漏らすと、縁の太さや色にキーを打っても印が出ない
        def moving(*frames: int) -> AnimatedValue:
            return AnimatedValue(1.0, tuple(Keyframe(frame, 1.0) for frame in frames))

        stroke = Stroke(
            params={"width": moving(10), "opacity": moving(30)},
            effects=(Effect("blur", {"radius": moving(50)}),),
        )
        clip = Clip(timeline_start=0, duration=60, source=TEXT.create().with_strokes((stroke,)))
        assert keyframe_frames(clip) == (10, 30, 50)
