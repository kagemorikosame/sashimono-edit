"""半透明の重ね合わせを、プロジェクトの設定（sRGB / リニア）どおりに混ぜること（Issue #65）

AviUtl2 と YMM4 は sRGB で符号化した値のまま混ぜる 黒の上に不透明度 50% の白を
重ねると 128、30% で 74〜76 になる（AviUtl2 の書き出しで測った） リニアで混ぜると
188 と約 150 になる

混ぜ方の取り違えは「少し明るい」にしか見えず、配布物と並べるまで気付けない
どの道（そのまま・エフェクト越し・合成モード・入れ子・画面の写し取り）を通っても
同じ値になることを数で押さえる 1 つでも符号化を 2 回掛けたり掛け忘れたりすると、
その道の絵だけ明るさが変わる

白（1.0）は符号化しても値が変わらないので、掛け忘れや掛けすぎが見えない道がある
道ごとの試験は中間の灰色（0.5）で確かめる
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddScene,
    AddTrack,
    Command,
    InScene,
    SetBlending,
    new_scene,
)
from sashimono.core.model import (
    AnimatedValue,
    Blending,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.gpu import BlendMode, GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SIZE = 32
WHITE = 1.0
GRAY = 0.5


def _decode(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else float(((c + 0.055) / 1.055) ** 2.4)


def _encode(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else float(1.055 * c ** (1 / 2.4) - 0.055)


def _expected(blending: str, level: float, opacity: float) -> int:
    """黒の上に明るさ ``level``（sRGB）の絵を ``opacity`` で重ねたときの値（0〜255）"""
    color = round(level * 255) / 255
    if blending == Blending.SRGB:
        return round(color * opacity * 255)
    return round(_encode(_decode(color) * opacity) * 255)


def test_the_expected_values_are_the_measured_ones() -> None:
    # 期待値の式そのものを、AviUtl2 の書き出しで測った値と Issue の表に合わせておく
    assert _expected(Blending.SRGB, WHITE, 0.5) == 128
    assert 74 <= _expected(Blending.SRGB, WHITE, 0.3) <= 77
    assert _expected(Blending.LINEAR, WHITE, 0.5) == 188
    assert abs(_expected(Blending.LINEAR, WHITE, 0.3) - 150) <= 2


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _fill(level: float) -> GeneratedSource:
    return GeneratedSource(
        kind="shape", params={"shape": "background", "color": (level, level, level, 1.0)}
    )


def _project(blending: str) -> Project:
    settings = ProjectSettings(width=SIZE, height=SIZE, frame_rate=FrameRate(30), blending=blending)
    return Project.create(settings)


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _single(
    blending: str, level: float, opacity: float, blend_mode: str = BlendMode.NORMAL
) -> tuple[Project, Clip]:
    """黒の上に 1 色の絵を 1 枚 ``opacity`` で重ねたプロジェクト"""
    project = _project(blending)
    track = Track(TrackKind.VIDEO, "V1")
    clip = Clip(
        timeline_start=0,
        duration=10,
        source=_fill(level),
        opacity=AnimatedValue(opacity),
        blend_mode=blend_mode,
    )
    return _apply(project, [AddTrack(track), AddClip(track.id, clip)]), clip


def _in_scene(project: Project, level: float, *, inner: float, outer: float) -> Project:
    """1 色の絵を不透明度 ``inner`` で置いたシーンを、不透明度 ``outer`` でメインへ置く"""
    scene = new_scene(project, "中")
    project = AddScene(scene).apply(project)
    inside = Track(TrackKind.VIDEO, "S1")
    picture = Clip(timeline_start=0, duration=10, source=_fill(level), opacity=AnimatedValue(inner))
    main = Track(TrackKind.VIDEO, "V1")
    placed = Clip(timeline_start=0, duration=10, scene_id=scene.id, opacity=AnimatedValue(outer))
    return _apply(
        project,
        [
            InScene(scene.id, AddTrack(inside)),
            InScene(scene.id, AddClip(inside.id, picture)),
            AddTrack(main),
            AddClip(main.id, placed),
        ],
    )


def _center(project: Project, gl_context: OffscreenGLContext) -> int:
    renderer = FrameRenderer(project, context=gl_context)
    try:
        image = renderer.render(0)
    finally:
        renderer.close()
    return int(image[SIZE // 2, SIZE // 2, 0])


@pytest.mark.parametrize("blending", Blending.ALL)
@pytest.mark.parametrize("opacity", [0.5, 0.3])
class TestOpacityOnBlack:
    def test_white_on_black_matches_the_measured_values(
        self, gl_context: OffscreenGLContext, blending: str, opacity: float
    ) -> None:
        # sRGB を選んでもリニアで混ぜると、AviUtl の素材の半透明が 60 近く明るく浮く
        project, _ = _single(blending, WHITE, opacity)
        assert abs(_center(project, gl_context) - _expected(blending, WHITE, opacity)) <= 1

    def test_a_plain_clip(
        self, gl_context: OffscreenGLContext, blending: str, opacity: float
    ) -> None:
        # 白だけでは見えない符号化の掛け忘れや掛けすぎで、半透明の灰色の素材が設定と違う明るさになる
        project, _ = _single(blending, GRAY, opacity)
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, opacity)) <= 1

    def test_through_an_effect(
        self, gl_context: OffscreenGLContext, blending: str, opacity: float
    ) -> None:
        # エフェクトの結果はリニアで届く 符号化し忘れると、sRGB の設定でも暗く沈む
        project, clip = _single(blending, GRAY, opacity)
        blur = registry.require("blur").create(radius=2.0)
        project = AddEffect(clip.id, blur).apply(project)
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, opacity)) <= 1

    def test_through_a_blend_mode(
        self, gl_context: OffscreenGLContext, blending: str, opacity: float
    ) -> None:
        # 黒の上のスクリーンは上の色そのもの シェーダで混ぜる道でも、不透明度の効き方が
        # 通常と同じでなければならない（こちらだけ違うと合成モードで明るさが跳ぶ）
        project, _ = _single(blending, GRAY, opacity, BlendMode.SCREEN)
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, opacity)) <= 1

    def test_inside_a_scene(
        self, gl_context: OffscreenGLContext, blending: str, opacity: float
    ) -> None:
        # 入れ子のキャンバスはもう符号化されている 重ねるときにもう 1 度符号化すると
        # シーンに入れただけで半透明の所が明るくなる
        project = _in_scene(_project(blending), GRAY, inner=opacity, outer=1.0)
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, opacity)) <= 1

    def test_a_scene_placed_half_transparent(
        self, gl_context: OffscreenGLContext, blending: str, opacity: float
    ) -> None:
        # 不透明なシーンを半透明で置く 入れ子の不透明度も同じ混ぜ方で効く
        project = _in_scene(_project(blending), GRAY, inner=1.0, outer=opacity)
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, opacity)) <= 1


@pytest.mark.parametrize("blending", Blending.ALL)
class TestFramebuffer:
    def _grabbed(self, blending: str, effect: str | None) -> Project:
        """半透明の灰色を重ねた画面を写し取り、そのまま（かエフェクトを掛けて）上に重ねる"""
        project, _ = _single(blending, GRAY, 0.5)
        track = Track(TrackKind.VIDEO, "V2")
        grab = Clip(timeline_start=0, duration=10, source=GeneratedSource(kind="framebuffer"))
        project = _apply(project, [AddTrack(track), AddClip(track.id, grab)])
        if effect is not None:
            project = AddEffect(grab.id, registry.require(effect).create(radius=2.0)).apply(project)
        return project

    def test_the_grab_keeps_the_brightness(
        self, gl_context: OffscreenGLContext, blending: str
    ) -> None:
        # 写した画面を不透明で重ねるだけなら、下の絵と同じ値のまま
        project = self._grabbed(blending, None)
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, 0.5)) <= 1

    def test_an_effect_on_the_grab_keeps_the_brightness(
        self, gl_context: OffscreenGLContext, blending: str
    ) -> None:
        # エフェクトへ渡すときに符号化を戻し忘れると、sRGB の設定だけ写した画面が明るく浮く
        project = self._grabbed(blending, "blur")
        assert abs(_center(project, gl_context) - _expected(blending, GRAY, 0.5)) <= 1


class TestSwitching:
    def test_switching_redraws_with_the_new_blending(self, gl_context: OffscreenGLContext) -> None:
        """同じレンダラで設定を切り替えると、次の絵から新しい混ぜ方になる

        レンダラはプレビューの間ずっと使い回す 合成先に古い設定が残ると、設定画面で
        切り替えても書き出し直すまで見た目が変わらない
        """
        project, _ = _single(Blending.SRGB, GRAY, 0.5)
        renderer = FrameRenderer(project, context=gl_context)
        try:
            first = int(renderer.render(0)[SIZE // 2, SIZE // 2, 0])
            renderer.set_project(SetBlending(Blending.LINEAR).apply(project))
            second = int(renderer.render(0)[SIZE // 2, SIZE // 2, 0])
            renderer.set_project(project)
            third = int(renderer.render(0)[SIZE // 2, SIZE // 2, 0])
        finally:
            renderer.close()
        assert abs(first - _expected(Blending.SRGB, GRAY, 0.5)) <= 1
        assert abs(second - _expected(Blending.LINEAR, GRAY, 0.5)) <= 1
        assert abs(third - _expected(Blending.SRGB, GRAY, 0.5)) <= 1

    def test_switching_reaches_nested_scenes(self, gl_context: OffscreenGLContext) -> None:
        # 入れ子の合成先は使い回す 切り替えを伝え忘れると、シーンの中だけ前の混ぜ方で
        # 溜まり、外で違う意味の値として読まれる（リニアの値を符号化済みとして読むと暗く沈む）
        project = _in_scene(_project(Blending.LINEAR), GRAY, inner=0.5, outer=1.0)
        renderer = FrameRenderer(project, context=gl_context)
        try:
            before = int(renderer.render(0)[SIZE // 2, SIZE // 2, 0])
            renderer.set_project(SetBlending(Blending.SRGB).apply(project))
            after = int(renderer.render(0)[SIZE // 2, SIZE // 2, 0])
        finally:
            renderer.close()
        assert abs(before - _expected(Blending.LINEAR, GRAY, 0.5)) <= 1
        assert abs(after - _expected(Blending.SRGB, GRAY, 0.5)) <= 1


def test_an_opaque_picture_looks_the_same_either_way(gl_context: OffscreenGLContext) -> None:
    """不透明な絵は、どちらの混ぜ方でも同じ値で出る

    混ぜ方が変えるのは半透明の所だけ 不透明な所まで変わるなら、符号化の往復が
    どこかで合っていない（暗い色ほど大きくずれる）
    """
    values = []
    for blending in Blending.ALL:
        project = _project(blending)
        track = Track(TrackKind.VIDEO, "V1")
        color = GeneratedSource(
            kind="shape", params={"shape": "background", "color": (0.2, 0.5, 0.8, 1.0)}
        )
        clip = Clip(timeline_start=0, duration=10, source=color)
        project = _apply(project, [AddTrack(track), AddClip(track.id, clip)])
        renderer = FrameRenderer(project, context=gl_context)
        try:
            values.append(renderer.render(0)[SIZE // 2, SIZE // 2, :3].astype(int))
        finally:
            renderer.close()
    assert np.abs(values[0] - values[1]).max() <= 1
