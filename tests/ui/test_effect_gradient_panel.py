"""図形の塗りとエフェクトの色のバーを本体の窓から操作して、描いた絵と運ばれ方を見る（#364）

MainWindow を組み、設定パネルの欄から図形の塗りの種類とバー、「グラデーション」「模様で塗る」の
バーを動かして、描いた絵（等倍と画質 1/2）の色を読む プリセット・エイリアス・テンプレートの
着せ替えで止まりの並びが運ばれることも見る
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QColorDialog, QDialog, QMessageBox

from sashimono.compat.catalog import restyle
from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import AddEffect, SetSource
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.preset import PresetOptions
from sashimono.core.io.aliases import Alias
from sashimono.core.io.presets import FORMAT_VERSION, Preset
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import ParamInput, registry
from sashimono.effects.sources import SHAPE, TEXT
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.render.renderer import RenderQuality
from sashimono.ui.inspector.gradient import GradientEditor
from sashimono.ui.inspector.panel import InspectorPanel, apply_preset_commands, capture_preset
from sashimono.ui.inspector.widgets import ColorEditor, ParameterEditor
from sashimono.ui.main_window import MainWindow

RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
RED_TO_BLUE = (0.0, *RED, 1.0, *BLUE)
THREE = (0.0, *RED, 0.5, *GREEN, 1.0, *BLUE)
SIZE = 160


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


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _flush() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


def _square(**params: ParamInput) -> GeneratedSource:
    base: dict[str, ParamInput] = {"shape": "rect", "width": SIZE, "height": SIZE, "color": WHITE}
    base.update(params)
    return SHAPE.create(**base)


def _project(source: GeneratedSource, effects: tuple[Effect, ...] = ()) -> Project:
    """640x360 の画面の真ん中に、1 辺 160 の図形が 1 つ"""
    base = Project.create(ProjectSettings(width=640, height=360, frame_rate=FrameRate(30)))
    clip = with_fixed_items(
        Clip(timeline_start=0, duration=60, source=source, effects=effects), picture=True
    )
    tracks = (Track(TrackKind.VIDEO, "V1", (clip,)), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _window(project: Project) -> MainWindow:
    window = MainWindow(project, confirm_unsaved=False)
    clip = _clip(window)
    window._timeline.set_selection((clip.id,))
    _flush()
    return window


def _clip(window: MainWindow) -> Clip:
    return window.document.project.timeline.tracks[0].clips[0]


def _editor(panel: InspectorPanel, name: str) -> ParameterEditor:
    found = [editor for (_owner, field), editor in panel._editors.items() if field == name]
    assert len(found) == 1, sorted(panel._editors)
    return found[0]


def _fields(panel: InspectorPanel, owner: str = "source") -> list[str]:
    """``owner``（中身なら ``source``、エフェクトならその ID）の欄の名前を出ている順に"""
    return [field for key, field in panel._editors if key == owner]


def _owner(window: MainWindow, kind: str) -> str:
    (effect,) = [e for e in _clip(window).effects if e.kind == kind]
    return str(effect.id)


def _render(window: MainWindow, context: OffscreenGLContext, quality: int = 1) -> np.ndarray:
    renderer = FrameRenderer(
        window.document.project, context=context, quality=RenderQuality(quality)
    )
    try:
        return renderer.render(10).astype(np.int16)
    finally:
        renderer.close()


def _down(image: np.ndarray, share: float) -> tuple[int, int, int]:
    """図形の上端から ``share`` の所の、真ん中の列の色"""
    width = image.shape[1]
    scale = width / 640
    y = round((180 - SIZE / 2 + share * SIZE) * scale)
    red, green, blue = (int(v) for v in image[y, width // 2, :3])
    return red, green, blue


def _across(image: np.ndarray, share: float) -> tuple[int, int, int]:
    """図形の左端から ``share`` の所の、真ん中の行の色"""
    height, width = image.shape[:2]
    scale = width / 640
    x = round((320 - SIZE / 2 + share * SIZE) * scale)
    red, green, blue = (int(v) for v in image[height // 2, x, :3])
    return red, green, blue


def _near(found: tuple[int, int, int], expected: tuple[int, int, int], tolerance: int) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(found, expected, strict=True))


def _same_at_half(window: MainWindow, context: OffscreenGLContext, read: object) -> None:
    """画質 1/2 のプレビューでも同じ所に同じ色が出る"""
    assert callable(read)
    full = _render(window, context)
    half = _render(window, context, quality=2)
    for share in (0.1, 0.3, 0.5, 0.7, 0.9):
        expected, found = read(full, share), read(half, share)
        assert _near(found, expected, 30), (share, found, expected)


class TestAShapeThroughTheWindow:
    def test_the_panel_turns_the_shape_fill_into_a_gradient(
        self, qt_application: QApplication, gl_context: OffscreenGLContext
    ) -> None:
        # 設定パネルから選んでも図形の色が変わらなければ、作ったグラデーションが書き出しに出ない
        del qt_application
        window = _window(_project(_square()))
        try:
            panel = window._inspector
            # 単色の間はバーが出ず、色の欄が出る 塗りの種類は色の欄のすぐ上
            assert isinstance(_editor(panel, "color"), ColorEditor)
            names = _fields(panel)
            assert names[names.index("fill_type") + 1] == "color"
            assert "fill_gradient" not in names
            # 図形には行も字も無いので、グラデーションの基準の欄は出さない
            _editor(panel, "fill_type").value_changed.emit("linear")
            _flush()
            names = _fields(panel)
            assert "color" not in names and "fill_basis" not in names
            bar = _editor(panel, "fill_gradient")
            assert isinstance(bar, GradientEditor)
            bar.value_changed.emit(RED_TO_BLUE)
            _flush()

            source = _clip(window).source
            assert source is not None
            assert source.params["fill_type"] == "linear"
            assert source.params["fill_gradient"] == RED_TO_BLUE

            image = _render(window, gl_context)
            assert _near(_down(image, 0.03), (255, 0, 0), 25)
            assert _near(_down(image, 0.97), (0, 0, 255), 25)
            middle = _down(image, 0.5)
            assert abs(middle[0] - middle[2]) < 40 and middle[1] < 30, middle
            _same_at_half(window, gl_context, _down)

            # 取り消し 2 回で単色の白に戻る（種類とバーの 2 段）
            window.undo()
            window.undo()
            _flush()
            back = _clip(window).source
            assert back is not None
            assert back.params["fill_type"] == "solid"
            assert _near(_down(_render(window, gl_context), 0.5), (255, 255, 255), 2)
        finally:
            window.close()

    def test_a_background_keeps_the_fill_near_the_top(self, qt_application: QApplication) -> None:
        # 背景は高さの欄を隠す 高さの後ろに置いた塗りの欄が、欄の並びの底に出ていた
        del qt_application
        window = _window(_project(_square(shape="background")))
        try:
            names = _fields(window._inspector)
            assert names.index("fill_type") < names.index("pos_x")
            assert names[names.index("fill_type") + 1] == "color"
        finally:
            window.close()


def _gradient_effect(**params: ParamInput) -> Effect:
    base: dict[str, ParamInput] = {
        "start_color": RED,
        "end_color": BLUE,
        "angle": 0,
        "span": SIZE,
    }
    base.update(params)
    return registry.require("gradient").create(**base)


def _brush(**params: ParamInput) -> Effect:
    base: dict[str, ParamInput] = {
        "pattern": "linear",
        "size": SIZE,
        "stops": 2,
        "color0": RED,
        "offset0": 0.0,
        "color1": BLUE,
        "offset1": 1.0,
    }
    base.update(params)
    return registry.require("brush_fill").create(**base)


class TestTheGradientEffectThroughTheWindow:
    def test_the_bar_replaces_the_two_colours(
        self, qt_application: QApplication, gl_context: OffscreenGLContext
    ) -> None:
        # 開始色と終了色の欄が残ると、止まりを足した後で開始色を直しても絵が変わらない
        # バーが前からの 2 色を見せないと、触っただけで既定の白から黒へ置き換わる
        del qt_application
        window = _window(_project(_square(), (_gradient_effect(),)))
        try:
            panel = window._inspector
            names = _fields(panel, _owner(window, "gradient"))
            assert "start_color" not in names and "end_color" not in names
            bar = _editor(panel, "gradient")
            assert isinstance(bar, GradientEditor)
            assert bar._value() == RED_TO_BLUE
            # 見本は描く所と同じ混ぜ方（既定の S 字）で混ぜる
            assert bar.bar._smooth
            # 混ぜ方は決めの欄（キーフレームを打てない）
            assert (_owner(window, "gradient"), "mixing") not in panel._key_controls
            _editor(panel, "mixing").value_changed.emit("linear")
            _flush()
            (effect,) = [e for e in _clip(window).effects if e.kind == "gradient"]
            assert effect.params["mixing"] == "linear"
            bar = _editor(panel, "gradient")
            assert isinstance(bar, GradientEditor)
            # 直線を選んだら見本も直線 S 字のままだと、見本と絵の中ほどの色が食い違う
            assert not bar.bar._smooth
            straight = _render(window, gl_context)
            assert abs(_down(straight, 0.25)[2] - 64) <= 8, _down(straight, 0.25)
            _editor(panel, "mixing").value_changed.emit("smooth")
            _flush()
            bar = _editor(panel, "gradient")
            assert isinstance(bar, GradientEditor)
            assert bar.bar._smooth

            bar.value_changed.emit(THREE)
            _flush()
            (effect,) = [e for e in _clip(window).effects if e.kind == "gradient"]
            assert effect.params["gradient"] == THREE
            image = _render(window, gl_context)
            assert _near(_down(image, 0.5), (0, 255, 0), 16)
            assert _near(_down(image, 0.03), (255, 0, 0), 25)
            assert _near(_down(image, 0.97), (0, 0, 255), 25)
            _same_at_half(window, gl_context, _down)
        finally:
            window.close()


class TestPaintWithAPatternThroughTheWindow:
    def test_the_bar_replaces_the_numbered_fields(
        self, qt_application: QApplication, gl_context: OffscreenGLContext
    ) -> None:
        # 前は「色 1」「位置 1」… の 33 の欄が縦に並ぶだけで、色の並びが見えなかった
        del qt_application
        window = _window(_project(_square(), (_brush(),)))
        try:
            panel = window._inspector
            names = _fields(panel, _owner(window, "brush_fill"))
            numbered = [name for name in names if re.fullmatch(r"(color|offset)\d+", name)]
            assert not numbered, names
            assert "stops" not in names
            bar = _editor(panel, "gradient")
            assert isinstance(bar, GradientEditor)
            assert bar._value() == RED_TO_BLUE
            assert not bar.bar._smooth

            bar.value_changed.emit(THREE)
            _flush()
            (effect,) = [e for e in _clip(window).effects if e.kind == "brush_fill"]
            assert effect.params["gradient"] == THREE
            # 前からの欄は書き換えない（並びを空に戻せば前の色で塗れる）
            assert effect.params["color1"] == BLUE
            image = _render(window, gl_context)
            assert _near(_across(image, 0.5), (0, 255, 0), 16)
            _same_at_half(window, gl_context, _across)
        finally:
            window.close()

    def test_a_solid_pattern_shows_its_colour_and_switching_shows_the_bar(
        self, qt_application: QApplication
    ) -> None:
        # 単色でバーを出すと、バーを動かしても絵が変わらない 模様を替えたら欄を作り直さないと
        # 線形にしてもバーが出ない
        del qt_application
        window = _window(_project(_square(), (_brush(pattern="solid", stops=1),)))
        try:
            panel = window._inspector
            names = _fields(panel, _owner(window, "brush_fill"))
            assert "color0" in names and "gradient" not in names
            _editor(panel, "pattern").value_changed.emit("linear")
            _flush()
            names = _fields(panel, _owner(window, "brush_fill"))
            assert "gradient" in names and "color0" not in names
        finally:
            window.close()

    def test_moving_offsets_keep_their_own_fields(self, qt_application: QApplication) -> None:
        # 動く位置をバーに出すと動きが見えず、触ると動きが黙って消える
        del qt_application
        moving = AnimatedValue(0.0, (Keyframe(0, 0.2), Keyframe(30, 0.8)))
        window = _window(_project(_square(), (_brush(offset1=moving),)))
        try:
            names = _fields(window._inspector, _owner(window, "brush_fill"))
            assert "gradient" not in names
            assert {"stops", "color0", "offset0", "color1", "offset1"} <= set(names)
            # 使っていない 3 つ目から先の欄は出さない
            assert "color2" not in names
        finally:
            window.close()


class TestCarriedAlong:
    def _clip(self) -> Clip:
        return with_fixed_items(
            Clip(
                timeline_start=0,
                duration=60,
                source=_square(fill_type="linear", fill_gradient=RED_TO_BLUE, fill_angle=30.0),
                effects=(_gradient_effect(gradient=THREE), _brush(gradient=THREE)),
            ),
            picture=True,
        )

    def test_a_preset_takes_the_shape_fill_and_the_stops(self) -> None:
        # プリセットで運べないと、作ったグラデーションの図形や塗りをほかのクリップへ着せられない
        # 前の版で読めない形（版 3）で書かないと、前の版が黙って単色や前からの 2 色で当てる
        saved = self._clip()
        target = with_fixed_items(
            Clip(timeline_start=100, duration=90, source=_square(color=BLUE)), picture=True
        )
        track = Track(TrackKind.VIDEO, "V1", (saved, target))
        project = Project.create().with_timeline(
            replace(Project.create().timeline, tracks=(track,))
        )
        preset = capture_preset("グラデ", project, track, saved)
        written = preset.to_dict()
        assert written["version"] == FORMAT_VERSION == 3
        preset = Preset.from_dict(json.loads(json.dumps(written)))
        for command in apply_preset_commands(
            project, track, track.clips[1], preset, options=PresetOptions()
        ):
            project = command.apply(project)
        located = project.timeline.locate_clip(track.clips[1].id)
        assert located is not None and located[1].source is not None
        params = located[1].source.params
        assert params["fill_type"] == "linear" and params["fill_gradient"] == RED_TO_BLUE
        stops = {e.kind: e.params["gradient"] for e in located[1].effects if not e.fixed}
        assert stops == {"gradient": THREE, "brush_fill": THREE}

    @pytest.mark.parametrize(
        ("effects", "version"),
        [
            # 並びが空（前からの色）のエフェクトまで版 3 で書くと、0.2.2 の一覧から消える
            ((_brush(),), 1),
            ((_gradient_effect(),), 1),
            # 単色の図形でも、エフェクトの並びがあれば版 3 前の版が黙って前からの色で当てる
            ((_brush(gradient=THREE),), 3),
            ((_gradient_effect(gradient=THREE),), 3),
        ],
    )
    def test_the_preset_version_follows_the_stops(
        self, effects: tuple[Effect, ...], version: int
    ) -> None:
        saved = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=_square(), effects=effects),
            picture=True,
        )
        track = Track(TrackKind.VIDEO, "V1", (saved,))
        project = Project.create().with_timeline(
            replace(Project.create().timeline, tracks=(track,))
        )
        assert capture_preset("版", project, track, saved).to_dict()["version"] == version

    def test_a_gradient_shape_alone_is_written_as_version_3(self) -> None:
        # 図形の塗りだけを版 1 で書くと、前の版の本体が黙って単色の図形を当てる
        saved = with_fixed_items(
            Clip(
                timeline_start=0,
                duration=30,
                source=_square(fill_type="radial", fill_gradient=RED_TO_BLUE),
            ),
            picture=True,
        )
        track = Track(TrackKind.VIDEO, "V1", (saved,))
        project = Project.create().with_timeline(
            replace(Project.create().timeline, tracks=(track,))
        )
        assert capture_preset("図形", project, track, saved).to_dict()["version"] == 3

    def test_an_alias_keeps_the_shape_fill_and_the_stops(self) -> None:
        # エイリアスで置き直したときに単色や前からの色へ戻ると、保存した見た目と違う物が置かれる
        back = Alias.from_dict(json.loads(json.dumps(Alias.of("グラデ", self._clip()).to_dict())))
        assert back.clip.source is not None
        assert back.clip.source.params["fill_gradient"] == RED_TO_BLUE
        stops = {e.kind: e.params["gradient"] for e in back.clip.effects if not e.fixed}
        assert stops == {"gradient": THREE, "brush_fill": THREE}

    def test_a_template_restyle_takes_the_stops(self) -> None:
        # 着せ替えでテンプレートのエフェクトの並びが落ちると、棚の見本と違う色の字幕になる
        template = with_fixed_items(
            Clip(
                timeline_start=0,
                duration=30,
                source=TEXT.create(text="型"),
                effects=(_gradient_effect(gradient=THREE),),
            ),
            picture=True,
        )
        current = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=TEXT.create(text="今")), picture=True
        )
        commands = restyle([MappedObject(clip=template, layer=1)], current)
        assert isinstance(commands[0], SetSource)
        added = [c.effect for c in commands if isinstance(c, AddEffect)]
        assert [e.params["gradient"] for e in added if e.kind == "gradient"] == [THREE]
        # エフェクトだけのテンプレート（中身の無い物）も並びごと足す
        only = Clip(timeline_start=0, duration=30, effects=(_brush(gradient=THREE),))
        (added_only,) = restyle([MappedObject(clip=only, layer=1)], current)
        assert isinstance(added_only, AddEffect)
        assert added_only.effect.params["gradient"] == THREE
