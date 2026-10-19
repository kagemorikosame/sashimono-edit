"""場面切り替え（YMM4 の ``TransitionItem``）

決まりは YMM4 に描かせた試験と、配布されている場面切り替えの書き出しから読んだ
前の場面は切り替えの頭より前に始まったクリップだけで、範囲の中の切れ目の直前で止まる
後の場面はいまの時刻の絵 進み具合は範囲の頭から終わりまで
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import place
from sashimono.compat.ymm4.template import load_template, map_template
from sashimono.core.commands import (
    AddEffect,
    ParamPath,
    RemoveEffect,
    SetParam,
    resolve_param,
)
from sashimono.core.io import project_from_dict, project_to_dict
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.easing import ease
from sashimono.effects.sources import TRANSITION
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SETTINGS = ProjectSettings(width=64, height=36, frame_rate=FrameRate(30))
RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _fill(color: tuple[float, float, float, float]) -> GeneratedSource:
    return GeneratedSource(kind="shape", params={"shape": "background", "color": color})


def _project(style: str, **params: object) -> Project:
    """赤（0〜30）から青（30〜60）へ 切り替えは 20〜40"""
    base = Project.create(SETTINGS)
    scenes = Track(
        TrackKind.VIDEO,
        "V1",
        (
            Clip(timeline_start=0, duration=30, source=_fill(RED)),
            Clip(timeline_start=30, duration=30, source=_fill(BLUE)),
        ),
    )
    transition = Clip(
        timeline_start=20,
        duration=20,
        source=TRANSITION.create(style=style, **params),  # type: ignore[arg-type]
    )
    tracks = (scenes, Track(TrackKind.VIDEO, "V2", (transition,)))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _render(project: Project, context: OffscreenGLContext, frame: int) -> np.ndarray:
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(frame)
    finally:
        renderer.close()


def test_switch_when_the_cut_is_in_the_middle(gl_context: OffscreenGLContext) -> None:
    # 切れ目と真ん中が同じ 30 のとき、1 フレームもずれずに赤から青へ変わる
    project = _project("switch")
    assert _render(project, gl_context, 29)[18, 32, 0] > 200
    assert _render(project, gl_context, 30)[18, 32, 2] > 200


def _tracks(project: Project, *tracks: Track) -> Project:
    return project.with_timeline(replace(project.timeline, tracks=tracks))


def _transition_track(start: int, duration: int, style: str, **params: object) -> Track:
    clip = Clip(
        timeline_start=start,
        duration=duration,
        source=TRANSITION.create(style=style, **params),  # type: ignore[arg-type]
    )
    return Track(TrackKind.VIDEO, "切り替え", (clip,))


def test_switch_changes_in_the_middle_not_at_the_cut(gl_context: OffscreenGLContext) -> None:
    # 赤は 30 で終わるが、切り替えは 20〜60 の真ん中の 40 で入れ替わる
    # 切れ目で入れ替えると、YMM4 では前の場面が映っている 30〜39 に青が出る
    # （じわっと抽象化切り替えは長さ 40、切れ目 30 で、YMM4 は 20 から後の場面を出す）
    base = Project.create(SETTINGS)
    scenes = Track(
        TrackKind.VIDEO,
        "V1",
        (
            Clip(timeline_start=0, duration=30, source=_fill(RED)),
            Clip(timeline_start=30, duration=30, source=_fill(BLUE)),
        ),
    )
    project = _tracks(base, scenes, _transition_track(20, 40, "switch"))
    assert _render(project, gl_context, 39)[18, 32, 0] > 200
    assert _render(project, gl_context, 40)[18, 32, 2] > 200


def test_a_clip_starting_with_the_transition_is_not_the_old_scene(
    gl_context: OffscreenGLContext,
) -> None:
    # 切り替えと同時に始まる青は後の場面にだけ入る 前の場面は何も無い（黒）
    # 前の場面にも青を入れると、黒から出てくるはずのクロスフェードが最初から青い
    # （YMM4 のペイントトランジションで差が 249 あった）
    base = Project.create(SETTINGS)
    scenes = Track(
        TrackKind.VIDEO, "V1", (Clip(timeline_start=20, duration=40, source=_fill(BLUE)),)
    )
    project = _tracks(base, scenes, _transition_track(20, 20, "fade"))
    assert _render(project, gl_context, 20)[18, 32, 2] < 5
    assert abs(int(_render(project, gl_context, 30)[18, 32, 2]) - 128) <= 3


def test_the_old_scene_stops_where_its_own_clip_ends(gl_context: OffscreenGLContext) -> None:
    # 前の場面の赤は 30 で終わる 後から始まった緑が 35 まで続いても、前の場面は
    # 赤の終わりの直前で止める 緑の終わりで止めると、32 の前の場面は空になる
    base = Project.create(SETTINGS)
    red = Track(TrackKind.VIDEO, "V1", (Clip(timeline_start=0, duration=30, source=_fill(RED)),))
    green = Track(
        TrackKind.VIDEO,
        "V2",
        (Clip(timeline_start=22, duration=13, source=_fill((0.0, 1.0, 0.0, 1.0))),),
    )
    project = _tracks(base, red, green, _transition_track(20, 20, "overlay", target="before"))
    pixel = _render(project, gl_context, 32)[18, 32]
    assert pixel[0] > 200 and pixel[1] < 5


def test_fade_mixes_the_encoded_values(gl_context: OffscreenGLContext) -> None:
    # 真ん中（30）で前と後が半分ずつ YMM4 は sRGB の値のまま混ぜるので 127 前後
    pixel = _render(_project("fade"), gl_context, 30)[18, 32]
    assert abs(int(pixel[0]) - 128) <= 3
    assert abs(int(pixel[2]) - 128) <= 3


def test_before_the_cut_the_fade_shows_the_old_scene(gl_context: OffscreenGLContext) -> None:
    # 切れ目より前は前も後も同じ絵なので、混ぜても変わらない
    pixel = _render(_project("fade"), gl_context, 25)[18, 32]
    assert pixel[0] > 250 and pixel[2] < 5


def test_push_moves_the_old_scene_out_to_the_right(gl_context: OffscreenGLContext) -> None:
    # 3/4 まで進むと、前の場面は画面の 3/4 だけ右へ 左の 3/4 には後の場面が入ってくる
    image = _render(_project("push", angle=0.0), gl_context, 35)
    assert image[18, 60, 0] > 200
    assert image[18, 10, 2] > 200


def test_the_old_scene_lasts_until_the_end(gl_context: OffscreenGLContext) -> None:
    # 赤のクリップは 30 で終わるが、重ねるだけの切り替えでは終わりまで手前に残る
    image = _render(_project("overlay", target="before"), gl_context, 38)
    assert image[18, 32, 0] > 200


def test_effects_go_to_their_own_scene(gl_context: OffscreenGLContext) -> None:
    # 前の場面のエフェクトが後の場面にも掛かると、前の場面を透明にしたつもりで切り替えの先まで消える
    opacity = registry.get("opacity")
    assert opacity is not None
    project = _project("overlay", target="before")
    track = project.timeline.tracks[1]
    hidden = replace(track.clips[0], effects=(opacity.create(amount=0.0),))
    tracks = (project.timeline.tracks[0], replace(track, clips=(hidden,)))
    project = project.with_timeline(replace(project.timeline, tracks=tracks))
    # 前の場面を消したので、後の場面（青）が見える
    assert _render(project, gl_context, 35)[18, 32, 2] > 200


#: 場面の中身が画面の一部だけにあるときの試験 図形を画面の中央から外して置く
WIDE = ProjectSettings(width=200, height=100, frame_rate=FrameRate(30))


def _scene_with(
    after_effects: tuple[Effect, ...], *, x: float, width: float, height: float
) -> Project:
    """幅 ``width`` 高さ ``height`` の白い四角を画面の中央から ``x`` 右へ置き、
    同じ時刻に始まる切り替え（重ねるだけ）の後の場面へエフェクトを積む
    切り替えと同じ時刻に始まるので前の場面は空 見えるのは後の場面だけ
    """
    base = Project.create(WIDE)
    shape = Clip(
        timeline_start=0,
        duration=30,
        source=GeneratedSource(
            kind="shape",
            params={
                "shape": "rect",
                "width": AnimatedValue(width),
                "height": AnimatedValue(height),
                "color": (1.0, 1.0, 1.0, 1.0),
            },
        ),
        effects=(registry.require("transform").create(pos_x=x),),
    )
    transition = Clip(
        timeline_start=0,
        duration=30,
        source=TRANSITION.create(style="overlay", target="after"),
        after_effects=after_effects,
    )
    tracks = (
        Track(TrackKind.VIDEO, "V1", (shape,)),
        Track(TrackKind.VIDEO, "V2", (transition,)),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def test_scene_effects_turn_around_the_contents_not_the_screen(
    gl_context: OffscreenGLContext,
) -> None:
    """場面に掛ける変形の「下端」は、場面の中身（図形）の下端

    画面の下端で回すと、ローテンショントランジション（中心点の下端で 180 度回す）の
    図形が画面の外へ回って消えた YMM4 の書き出しでは図形の下端で回り、真下へ返った
    """
    turn = registry.require("transform").create(rotation=180, pivot_h="center", pivot_v="bottom")
    # 図形は 30〜70 列、40〜60 行 下端（60 行）で 180 度回すと 60〜80 行へ返る
    image = _render(_scene_with((turn,), x=-50, width=40, height=20), gl_context, 5)
    assert image[70, 50, 0] > 200, "図形の下端で回っていない"
    assert image[50, 50, 0] < 20


def test_scene_tiles_are_the_size_of_the_contents_and_masks_sit_on_the_origin(
    gl_context: OffscreenGLContext,
) -> None:
    """タイルは中身の大きさで並び、図形のマスクは原点（画面の中央）に置かれる

    タイルを画面の大きさで並べると、リール回転風の縦に連なる帯が 1 枚だけになり隙間が
    空いた マスクを中身の中央に置くと、円形端から暗転の円が図形の真ん中へずれた
    YMM4 の書き出しでは、帯は図形の高さごとに並び、円は画面の中央から開いた
    """
    tile = registry.require("tile").create(count_x=1, count_y=3)
    mask = registry.require("shape_mask").create(shape="ellipse", width=20, height=200)
    # 図形は 80〜180 列、40〜60 行 タイルで 20〜80 行に 3 枚 マスクは 90〜110 列だけ残す
    image = _render(_scene_with((tile, mask), x=30, width=100, height=20), gl_context, 5)
    assert image[30, 100, 0] > 200, "タイルが中身の高さで並んでいないか、マスクが原点にない"
    assert image[30, 130, 0] < 20, "マスクが中身の中央へずれている"
    assert image[10, 100, 0] < 20, "タイルが 3 枚より多く並んでいる"


def test_after_effects_survive_saving() -> None:
    # 保存で後の場面のエフェクトが落ちると、開き直した場面切り替えから後の場面の加工が消える
    opacity = registry.get("opacity")
    assert opacity is not None
    project = _project("fade")
    track = project.timeline.tracks[1]
    clip = replace(track.clips[0], after_effects=(opacity.create(amount=50.0),))
    tracks = (project.timeline.tracks[0], replace(track, clips=(clip,)))
    project = project.with_timeline(replace(project.timeline, tracks=tracks))
    loaded = project_from_dict(project_to_dict(project))
    restored = loaded.timeline.tracks[1].clips[0]
    assert [e.kind for e in restored.after_effects] == ["opacity"]


@pytest.mark.parametrize("mode", ["in", "out", "inout"])
@pytest.mark.parametrize(
    "kind",
    ["linear", "sine", "quad", "cubic", "quart", "quint", "expo", "circ", "back", "bounce"],
)
def test_easing_starts_at_zero_and_ends_at_one(kind: str, mode: str) -> None:
    # 両端が 0 と 1 から外れると、場面切り替えの頭か終わりで前の場面が残るか後の場面へ跳ぶ
    assert ease(0.0, kind, mode) == pytest.approx(0.0, abs=1e-3)
    assert ease(1.0, kind, mode) == pytest.approx(1.0, abs=1e-3)


class TestEditingTheScenes:
    """前の場面と後の場面のエフェクトを、別々に積んで触れること"""

    def _clip(self) -> tuple[Project, ClipId]:
        project = _project("fade")
        clip = project.timeline.tracks[1].clips[0]
        return project, clip.id

    def test_effects_go_to_the_stack_you_asked_for(self) -> None:
        # 後の場面へ積むよう頼んだエフェクトが前に入ると、
        # 切り替えの先に掛けた効果が手前の場面に掛かる
        project, clip_id = self._clip()
        opacity = registry.get("opacity")
        assert opacity is not None
        project = AddEffect(clip_id, opacity.create(amount=10.0)).apply(project)
        project = AddEffect(clip_id, opacity.create(amount=90.0), after=True).apply(project)
        clip = project.timeline.locate_clip(clip_id)[1]  # type: ignore[index]
        assert [e.kind for e in clip.effects] == ["opacity"]
        assert [e.kind for e in clip.after_effects] == ["opacity"]
        assert clip.after_effects[0].params["amount"].static == 90.0  # type: ignore[union-attr]

    def test_a_parameter_change_finds_the_after_stack(self) -> None:
        # 後の場面のエフェクトの値を探せないと、設定パネルで後の場面の不透明度を
        # 動かしても絵が変わらない
        project, clip_id = self._clip()
        opacity = registry.get("opacity")
        assert opacity is not None
        effect = opacity.create(amount=90.0)
        project = AddEffect(clip_id, effect, after=True).apply(project)
        path = ParamPath.of_effect(clip_id, effect.id, "amount", after=True)
        project = SetParam(path, AnimatedValue(40.0)).apply(project)
        assert resolve_param(project, path) == AnimatedValue(40.0)

    def test_removing_from_the_after_stack_leaves_the_front_alone(self) -> None:
        # 後の場面から消すときに前の場面の物まで消えると、
        # 片方を外しただけで切り替え全体の加工が無くなる
        project, clip_id = self._clip()
        opacity = registry.get("opacity")
        assert opacity is not None
        front = opacity.create(amount=10.0)
        back = opacity.create(amount=90.0)
        project = AddEffect(clip_id, front).apply(project)
        project = AddEffect(clip_id, back, after=True).apply(project)
        project = RemoveEffect(clip_id, back.id, after=True).apply(project)
        clip = project.timeline.locate_clip(clip_id)[1]  # type: ignore[index]
        assert [e.id for e in clip.effects] == [front.id]
        assert clip.after_effects == ()


#: 配布されている場面切り替え（あおもや式テンプレート） 配布物なのでリポジトリには入れていない
PAINT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "ymm4"
    / "aomoya"
    / "場面切り替え_ペイントトランジション.ymmt"
)


@pytest.mark.skipif(not PAINT.is_file(), reason="ペイントトランジションの .ymmt が置かれていない")
def test_the_paint_transition_starts_from_black(gl_context: OffscreenGLContext) -> None:
    # 図形も切り替えも同じ時刻に始まるので、前の場面は黒 YMM4 の書き出しは頭の
    # 8 フレームがほぼ真っ黒（平均 1 未満） いまのフレームを前の場面にすると、
    # 頭から明るい灰色が出る（差 249）
    templates = load_template(PAINT)
    objects = map_template(list(templates[0].items), report=CompatibilityReport())
    # 写せず何も置けないと、空のタイムラインは黒なので下の検査が素通りする
    assert objects, "テンプレートから何も写せなかった"
    project = Project.create(SETTINGS)
    commands = place(objects, project, at_frame=0)
    assert commands, "置くものが無かった"
    for command in commands:
        project = command.apply(project)
    for frame in (0, 2, 5):
        image = _render(project, gl_context, frame)
        assert float(image[..., :3].mean()) < 10, f"フレーム {frame} が明るい"
    # 黒いのは頭だけ YMM4 の書き出しは 40 フレーム目で平均 190 前後まで明るくなる
    # ここまで黒なら、前の場面ではなく絵そのものが描けていない
    assert float(_render(project, gl_context, 40)[..., :3].mean()) > 100
