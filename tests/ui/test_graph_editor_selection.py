"""キーフレームを入れたクリップを選ぶと、グラフエディタで曲線を触れること（利用者の報告）

再現の手順 1) 不透明度にキーフレームを入れたクリップを置く 2) タイムラインで選ぶ
3) グラフエディタを開く（Ctrl+G） → 「パラメータを選んでください」のまま何も出なかった
値を渡す口が設定パネルの ◆ の右クリック（グラフエディタで開く）しか無く、選んだクリップに
付いていかなかったため 窓は表示しない（オフスクリーン）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import ParamPath, RemoveEffect
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    Keyframe,
    LayerMode,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects import TrackSpec, registry
from sashimono.effects.sources import TEXT, TRANSITION
from sashimono.ui.graph_editor import GraphEditor
from sashimono.ui.inspector.panel import KeyframeControls
from sashimono.ui.main_window import MainWindow

START = 30


def _keyed(*keys: tuple[int, float]) -> AnimatedValue:
    return AnimatedValue(static=1.0, keyframes=tuple(Keyframe(frame=f, value=v) for f, v in keys))


def _picture_effect() -> tuple[str, str]:
    """数のスライダーを持つ映像のエフェクト（種類, 値の名前）"""
    for definition in registry.all():
        if definition.audio_process is not None:
            continue
        spec = next((s for s in definition.parameters if isinstance(s, TrackSpec)), None)
        if spec is not None:
            return definition.kind, spec.name
    raise AssertionError("数の値を持つエフェクトが無い")


def _project(*clips: Clip) -> Project:
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    tracks = tuple(Track(TrackKind.MIXED, f"レイヤー {n + 1}", (c,)) for n, c in enumerate(clips))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _faded() -> Clip:
    return Clip(
        timeline_start=START,
        duration=90,
        source=TEXT.create(),
        opacity=_keyed((0, 0.0), (30, 1.0)),
    )


@pytest.fixture
def graph(qt_application: QApplication) -> Iterator[GraphEditor]:
    del qt_application
    created = GraphEditor()
    yield created
    created.close()
    shiboken6.delete(created)


class TestFollowingTheSelection:
    def test_a_keyed_clip_shows_its_curve(self, graph: GraphEditor) -> None:
        # キーのあるクリップを選んでも曲線が出ないと、グラフで動きの緩急を直せない
        clip = _faded()
        graph.set_project(_project(clip))
        graph.set_clip(clip.id)
        assert graph.path == ParamPath.of_clip(clip.id, "opacity")
        assert "◆ クリップ: 不透明度" in graph.choices()
        # 曲線を描く値が渡っていないと、点をつまめない
        assert graph._canvas._value is not None
        assert [k.frame for k in graph._canvas._value.keyframes] == [0, 30]

    def test_a_clip_without_keys_says_how_to_make_them(self, graph: GraphEditor) -> None:
        # キーが無いときに空のままだと、グラフが壊れたのか、何をすれば曲線が出るのかが分からない
        clip = replace(_faded(), opacity=AnimatedValue(static=1.0))
        graph.set_project(_project(clip))
        graph.set_clip(clip.id)
        assert graph.path is None
        assert "◆" in graph._title.text()
        # キーが無くても値は選べる（曲線は平らな線）
        assert graph.choices()

    def test_another_value_can_be_chosen_and_is_kept(self, graph: GraphEditor) -> None:
        # 選び直すたびに先頭へ戻ると、2 つ目の値を直している途中でほかを押しただけで飛ぶ
        clip = _faded()
        graph.set_project(_project(clip))
        graph.set_clip(clip.id)
        size = graph.choices().index("テキスト: サイズ")
        graph.choose(size)
        assert graph.path == ParamPath.of_source(clip.id, "size")
        graph.set_clip(clip.id)
        assert graph.path == ParamPath.of_source(clip.id, "size")

    def test_a_removed_effect_moves_to_another_curve(self, graph: GraphEditor) -> None:
        # 外したエフェクトの値を出したままにすると、空の曲線から動けない
        kind, name = _picture_effect()
        definition = registry.require(kind)
        effect = definition.create()
        effect = Effect(
            kind=effect.kind,
            params={**effect.params, name: _keyed((0, 1.0), (10, 2.0))},
        )
        clip = replace(_faded(), effects=(effect,))
        project = _project(clip)
        graph.set_project(project)
        graph.set_clip(clip.id)
        graph.set_path(ParamPath.of_effect(clip.id, effect.id, name))
        graph.set_project(RemoveEffect(clip.id, effect.id).apply(project))
        assert graph.path == ParamPath.of_clip(clip.id, "opacity")

    def test_the_second_scene_of_a_transition_has_a_curve(self, graph: GraphEditor) -> None:
        # 後の場面のエフェクトを前の列から探していて、選んでも曲線が出なかった
        kind, name = _picture_effect()
        effect = registry.require(kind).create()
        effect = replace(effect, params={**effect.params, name: _keyed((0, 1.0), (10, 2.0))})
        clip = Clip(
            timeline_start=0, duration=60, source=TRANSITION.create(), after_effects=(effect,)
        )
        graph.set_project(_project(clip))
        graph.set_clip(clip.id)
        assert graph.path == ParamPath.of_effect(clip.id, effect.id, name, after=True)
        assert graph._spec() is not None


class TestInTheWindow:
    @pytest.fixture
    def window(self, qt_application: QApplication) -> Iterator[MainWindow]:
        del qt_application
        created = MainWindow(_project(_faded()), confirm_unsaved=False)
        yield created
        created.close()

    def test_selecting_the_clip_on_the_timeline_opens_its_curve(self, window: MainWindow) -> None:
        # 報告の手順そのもの 選んだだけで曲線が出ないと、グラフエディタを使う道が見えない
        clip = window.document.project.timeline.tracks[0].clips[0]
        window._timeline.select(clip.id)
        assert window._graph._path == ParamPath.of_clip(clip.id, "opacity")
        window._timeline.select(None)
        assert window._graph._path is None

    def test_the_key_button_brings_its_value_to_the_graph(self, window: MainWindow) -> None:
        # ◆ で打った値の曲線がグラフに出ないと、打った後にグラフで値を選び直すことになる
        clip = window.document.project.timeline.tracks[0].clips[0]
        window._timeline.select(clip.id)
        path = ParamPath.of_source(clip.id, "size")
        window._seek(START + 5)
        controls = next(
            c for c in window._inspector.findChildren(KeyframeControls) if c.path == path
        )
        controls.toggle.click()
        assert window._graph._path == path
        located = window.document.project.timeline.locate_clip(clip.id)
        assert located is not None and located[1].source is not None
        size = located[1].source.params["size"]
        assert isinstance(size, AnimatedValue)
        assert [k.frame for k in size.keyframes] == [5]
