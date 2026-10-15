"""設定パネルからテキストの塗りをグラデーションにして、描いた絵と保存の行き先を見る（#312）

本体の窓（MainWindow）を組み、設定パネルの欄から塗りの種類とバーを動かして、描いた絵の色を
読む プリセット・エイリアス・テンプレートの着せ替えでグラデーションが運ばれることも見る
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QColorDialog, QDialog, QMessageBox

from sashimono.compat.catalog import restyle
from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import (
    AdoptLegacyStroke,
    Command,
    SetParam,
    SetSource,
    insert_generated,
)
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.preset import PresetOptions
from sashimono.core.io.aliases import Alias
from sashimono.core.io.presets import Preset
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Project,
    ProjectSettings,
    Stroke,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.effects.strokes import STROKE
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.render.renderer import RenderQuality
from sashimono.ui.inspector.gradient import GradientEditor
from sashimono.ui.inspector.panel import InspectorPanel, apply_preset_commands, capture_preset
from sashimono.ui.inspector.widgets import ColorEditor, ParameterEditor
from sashimono.ui.main_window import MainWindow

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
RED_TO_BLUE = (0.0, *RED, 1.0, *BLUE)


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """試験の途中で窓（色の選択・尋ねる・知らせる）が開いたら、その場で落とす"""

    def opened(*_args: object, **_kwargs: object) -> None:
        pytest.fail("試験の途中で窓が開いた")

    monkeypatch.setattr(QDialog, "exec", opened)
    monkeypatch.setattr(QMessageBox, "exec", opened)
    for name in ("information", "warning", "question", "critical"):
        monkeypatch.setattr(QMessageBox, name, opened)
    monkeypatch.setattr(QColorDialog, "getColor", opened)


def _flush() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


def _project() -> Project:
    """640x360 の画面の真ん中に、白い「■■」の字が 1 本"""
    base = Project.create(ProjectSettings(width=640, height=360, frame_rate=FrameRate(30)))
    clip = with_fixed_items(
        Clip(
            timeline_start=0,
            duration=60,
            source=TEXT.create(text="■■", size=160, color=(1.0, 1.0, 1.0, 1.0)),
        ),
        picture=True,
    )
    tracks = (Track(TrackKind.VIDEO, "V1", (clip,)), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    yield created
    created.close()


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _text_clip(window: MainWindow) -> Clip:
    return window.document.project.timeline.tracks[0].clips[0]


def _editor(panel: InspectorPanel, name: str) -> ParameterEditor:
    found = [editor for (_owner, field), editor in panel._editors.items() if field == name]
    assert len(found) == 1, sorted(panel._editors)
    return found[0]


def _column(image: np.ndarray) -> list[tuple[int, int, int]]:
    """字の 1 文字目の真ん中の列の、塗れた所の色を上から（地は黒）"""
    lit = image[..., :3].astype(int).sum(axis=2) > 30
    ys, xs = np.nonzero(lit)
    x = int(xs.min()) + (int(xs.max()) - int(xs.min())) // 8
    rows = [y for y in range(int(ys.min()), int(ys.max()) + 1) if lit[y, x]]
    return [(int(image[y, x, 0]), int(image[y, x, 1]), int(image[y, x, 2])) for y in rows]


class TestThroughTheWindow:
    def test_the_panel_turns_the_fill_into_a_gradient(
        self, window: MainWindow, gl_context: OffscreenGLContext
    ) -> None:
        # 設定パネルから選んでも字の色が変わらなければ、作ったグラデーションが書き出しに出ない
        clip = _text_clip(window)
        window._timeline.set_selection((clip.id,))
        _flush()
        panel = window._inspector
        # 単色の間はバーが出ず、色の欄が出る
        assert isinstance(_editor(panel, "color"), ColorEditor)
        assert not any(field == "fill_gradient" for _owner, field in panel._editors)

        _editor(panel, "fill_type").value_changed.emit("linear")
        _flush()
        # グラデーションにすると色の欄が隠れ、バーが出る
        assert not any(field == "color" for _owner, field in panel._editors)
        bar = _editor(panel, "fill_gradient")
        assert isinstance(bar, GradientEditor)
        bar.value_changed.emit(RED_TO_BLUE)
        _flush()

        source = _text_clip(window).source
        assert source is not None
        assert source.params["fill_type"] == "linear"
        assert source.params["fill_gradient"] == RED_TO_BLUE

        renderer = FrameRenderer(window.document.project, context=gl_context)
        try:
            image = renderer.render(10)
        finally:
            renderer.close()
        column = _column(image)
        top, bottom = column[2], column[-3]
        # 上は赤、下は青 字の真ん中は紫（赤と青の間）
        assert top[0] > 200 and top[2] < 55, top
        assert bottom[2] > 200 and bottom[0] < 55, bottom
        middle = column[len(column) // 2]
        assert abs(middle[0] - middle[2]) < 50 and middle[1] < 30, middle

        # 画質 1/2 のプレビューでも同じ所に同じ色が出る
        half = FrameRenderer(window.document.project, context=gl_context, quality=RenderQuality(2))
        try:
            small = half.render(10)
        finally:
            half.close()
        small_column = _column(small)
        for share in (0.1, 0.5, 0.9):
            expected = column[round((len(column) - 1) * share)]
            found = small_column[round((len(small_column) - 1) * share)]
            assert all(abs(a - b) <= 30 for a, b in zip(found, expected, strict=True)), (
                share,
                found,
                expected,
            )

        # 取り消し 2 回で単色の白に戻る（種類とバーの 2 段）
        window.undo()
        window.undo()
        _flush()
        back = _text_clip(window).source
        assert back is not None
        assert back.params["fill_type"] == "solid"


class _Harness:
    """設定パネルが出した命令を当てて、パネルへ戻す（本体の窓の代わり）"""

    def __init__(self, panel: InspectorPanel, clip_source: Clip) -> None:
        self.panel = panel
        project = Project.create()
        assert clip_source.source is not None
        for command in insert_generated(project, clip_source.source):
            project = command.apply(project)
        self.project = project
        self.clip_id: ClipId = next(c for t in project.timeline.tracks for c in t.clips).id
        self.sent: list[list[Command]] = []
        panel.commands_requested.connect(self._apply)
        panel.set_project(project)
        panel.set_clip(self.clip_id)

    def _apply(self, commands: list[Command], _label: str) -> None:
        self.sent.append(list(commands))
        for command in commands:
            self.project = command.apply(self.project)
        self.panel.set_project(self.project)


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


class TestALegacyBorder:
    def test_making_its_colour_a_gradient_moves_it_into_a_layer(
        self, panel: InspectorPanel
    ) -> None:
        # 前からの縁の仮の層で種類を選んでも層へ移らないと、選んだのに縁が単色のまま残る
        harness = _Harness(
            panel, Clip(timeline_start=0, duration=30, source=TEXT.create(border_width=6))
        )
        owners = [
            owner
            for owner, field in panel._editors
            if field == "fill_type" and owner.startswith("stroke:")
        ]
        (owner,) = owners
        panel._editors[(owner, "fill_type")].value_changed.emit("linear")
        (commands,) = harness.sent
        assert [type(command) for command in commands] == [AdoptLegacyStroke, SetParam]
        clip = harness.project.timeline.locate_clip(harness.clip_id)
        assert clip is not None and clip[1].source is not None
        (stroke,) = clip[1].source.strokes
        assert stroke.params["fill_type"] == "linear"
        # 層の組にバーが出る（作り直しの指紋に層の欄の構成が入っている）
        assert any(
            field == "fill_gradient" and key.startswith("stroke:") for key, field in panel._editors
        )


class TestALayer:
    def test_choosing_a_gradient_shows_the_bar(self, panel: InspectorPanel) -> None:
        # 層の欄を作り直さないと、層の塗りをグラデーションにしてもバーが出ず色の欄が残る
        layer = Stroke(params={**STROKE.default_params(), "width": AnimatedValue(6.0)})
        source = TEXT.create().with_strokes((layer,))
        # 受け手を持っておく 捨てると命令が当たらず、パネルが作り直されない
        harness = _Harness(panel, Clip(timeline_start=0, duration=30, source=source))
        owner = next(key for key, field in panel._editors if key.startswith("stroke:"))
        assert (owner, "color") in panel._editors
        assert (owner, "fill_gradient") not in panel._editors
        panel._editors[(owner, "fill_type")].value_changed.emit("linear")
        assert isinstance(panel._editors[(owner, "fill_gradient")], GradientEditor)
        assert (owner, "color") not in panel._editors
        assert len(harness.sent) == 1


class TestTheOrder:
    def test_the_fill_sits_just_above_the_colour(self, panel: InspectorPanel) -> None:
        # 塗りの欄が定義の末尾（底）に出ると、色を探す所から遠くて見つけられない
        _Harness(panel, Clip(timeline_start=0, duration=30, source=TEXT.create(shadow_x=4)))
        names = [field for key, field in panel._editors if not key.startswith("stroke:")]
        assert names[names.index("size") + 1] == "fill_type"
        assert names[names.index("fill_type") + 1] == "color"
        assert names[names.index("shadow_blur") + 1] == "shadow_fill_type"


class TestCarriedAlong:
    def _gradient_clip(self) -> Clip:
        return with_fixed_items(
            Clip(
                timeline_start=0,
                duration=60,
                source=TEXT.create(
                    text="元", fill_type="linear", fill_gradient=RED_TO_BLUE, fill_angle=30.0
                ),
            ),
            picture=True,
        )

    def test_a_preset_takes_the_gradient(self) -> None:
        # プリセットで運べないと、作ったグラデーションの字幕をほかの字幕へ着せられない
        saved = self._gradient_clip()
        target = with_fixed_items(
            Clip(timeline_start=0, duration=90, source=TEXT.create(text="先")), picture=True
        )
        track = Track(TrackKind.VIDEO, "V1", (saved, replace(target, timeline_start=100)))
        project = Project.create().with_timeline(
            replace(Project.create().timeline, tracks=(track,))
        )
        preset = capture_preset("グラデ", project, track, saved)
        preset = Preset.from_dict(json.loads(json.dumps(preset.to_dict())))
        placed = track.clips[1]
        for command in apply_preset_commands(
            project, track, placed, preset, options=PresetOptions()
        ):
            project = command.apply(project)
        located = project.timeline.locate_clip(placed.id)
        assert located is not None and located[1].source is not None
        params = located[1].source.params
        assert params["fill_type"] == "linear"
        assert params["fill_gradient"] == RED_TO_BLUE
        assert params["text"] == "先"

    def test_an_alias_keeps_the_gradient(self) -> None:
        # エイリアスで置き直したときに単色へ戻ると、保存した見た目と違う字が置かれる
        alias = Alias.of("グラデ", self._gradient_clip())
        back = Alias.from_dict(json.loads(json.dumps(alias.to_dict())))
        assert back.clip.source is not None
        assert back.clip.source.params["fill_gradient"] == RED_TO_BLUE

    def test_a_template_restyle_takes_and_clears_the_gradient(self) -> None:
        # 着せ替えでテンプレートのグラデーションが当たらないと、棚の見本と違う字幕になる
        # グラデーションの無いテンプレートを着せたら単色へ戻る 戻らないと前の塗りが残る
        template = MappedObject(clip=self._gradient_clip(), layer=1)
        current = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=TEXT.create(text="今")), picture=True
        )
        set_source = restyle([template], current)[0]
        assert isinstance(set_source, SetSource)
        dressed = set_source.source
        assert dressed is not None
        assert dressed.params["fill_gradient"] == RED_TO_BLUE
        assert dressed.params["text"] == "今"
        plain_clip = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=TEXT.create()), picture=True
        )
        undressed = restyle(
            [MappedObject(clip=plain_clip, layer=1)], replace(current, source=dressed)
        )[0]
        assert isinstance(undressed, SetSource) and undressed.source is not None
        assert undressed.source.params["fill_type"] == "solid"
