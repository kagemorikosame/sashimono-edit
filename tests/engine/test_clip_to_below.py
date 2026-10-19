"""すぐ下のクリップの形で切り抜く（YMM4 の「上のオブジェクトでクリッピング」）

吹き出しの形だけに模様を見せる配布テンプレートが使っている 切り抜けないと、模様の
背景が画面全体を覆う
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.io import project_from_dict, project_to_dict
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SETTINGS = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30))


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _shape(color: tuple[float, float, float, float], size: float | None) -> GeneratedSource:
    if size is None:
        return GeneratedSource(kind="shape", params={"shape": "background", "color": color})
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "rect",
            "color": color,
            "width": AnimatedValue(size),
            "height": AnimatedValue(size),
        },
    )


def _project(*, clipped: bool, below_opacity: float = 1.0) -> Project:
    base = Project.create(SETTINGS)
    square = Clip(
        timeline_start=0,
        duration=10,
        source=_shape((0.0, 0.0, 1.0, 1.0), 20.0),
        opacity=AnimatedValue(below_opacity),
    )
    cover = Clip(
        timeline_start=0,
        duration=10,
        source=_shape((1.0, 0.0, 0.0, 1.0), None),
        clip_to_below=clipped,
    )
    tracks = (Track(TrackKind.VIDEO, "V1", (square,)), Track(TrackKind.VIDEO, "V2", (cover,)))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _render(project: Project, context: OffscreenGLContext) -> np.ndarray:
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(0)
    finally:
        renderer.close()


def test_the_cover_shows_only_inside_the_shape_below(gl_context: OffscreenGLContext) -> None:
    # 切り抜きが効かないと、吹き出しの形だけに見せるはずの模様の背景が画面全体を覆う
    image = _render(_project(clipped=True), gl_context)
    # 形の中は上の赤、外は下地の黒のまま（覆われない）
    assert image[32, 32, 0] > 200 and image[32, 32, 2] < 50
    assert image[2, 2, 0] < 30


def test_a_faint_shape_below_still_clips_at_full_strength(gl_context: OffscreenGLContext) -> None:
    # 下の形の薄さまで写すと、看板テロップの板が半分透けて下の絵と混ざった色で出る
    image = _render(_project(clipped=True, below_opacity=0.5), gl_context)
    # 切り抜く形に下のクリップの不透明度まで当てると、上の赤が半分しか出ず、
    # 下の青が透けて紫になる（木製看板テロップの板が 56.4% しか出なかった不具合）
    assert image[32, 32, 0] > 200, "切り抜かれた側が形の薄さぶん薄くなっている"
    assert image[32, 32, 2] < 50, "下の絵が透けている"


def test_the_shape_below_keeps_its_own_faintness(gl_context: OffscreenGLContext) -> None:
    # 不透明度は形としては使わないが、その絵自身を描くときには当たる
    # 当たらないと、薄く重ねたつもりの絵が濃いまま出る
    base = Project.create(SETTINGS)
    square = Clip(
        timeline_start=0,
        duration=10,
        source=_shape((0.0, 0.0, 1.0, 1.0), 20.0),
        opacity=AnimatedValue(0.5),
    )
    tracks = (Track(TrackKind.VIDEO, "V1", (square,)),)
    image = _render(base.with_timeline(replace(base.timeline, tracks=tracks)), gl_context)
    assert 90 < image[32, 32, 2] < 165


def test_without_clipping_the_cover_fills_the_screen(gl_context: OffscreenGLContext) -> None:
    # 切り抜きを付けていないクリップまで切り抜くと、全面の背景が下のクリップの形にしか出ない
    image = _render(_project(clipped=False), gl_context)
    assert image[2, 2, 0] > 200


def test_a_clipped_script_reads_the_screen_below(gl_context: OffscreenGLContext) -> None:
    """切り抜かれるスクリプトの ``obj.copybuffer("obj", "frm")`` は下に重ねた絵を写す（#170）

    切り抜く側は空の合成先へ描いてから形で切る その空の方を画面として写すと、
    アクリル矩形が下の絵ではなく黒をぼかした板になる 形の中は下の青が見えるはず
    """
    from sashimono.compat.aviutl import catalog as catalog_module
    from sashimono.compat.aviutl.catalog import ScriptCatalog, set_script_catalog
    from sashimono.compat.aviutl.custom_object import custom_object_clip
    from sashimono.effects import registry

    saved = catalog_module._catalog
    catalog = ScriptCatalog(roots=())
    catalog.add_text("aviutl:試験/@写す.obj:画面", 'obj.copybuffer("obj", "frm")', kind="obj")
    set_script_catalog(catalog)
    try:
        definition = registry.get("aviutl:試験/@写す.obj:画面")
        assert definition is not None
        base = Project.create(SETTINGS)
        square = Clip(timeline_start=0, duration=10, source=_shape((0.0, 0.0, 1.0, 1.0), 20.0))
        copied = replace(custom_object_clip(definition.create(), duration=10), clip_to_below=True)
        tracks = (
            Track(TrackKind.VIDEO, "V1", (square,)),
            Track(TrackKind.VIDEO, "V2", (copied,)),
        )
        image = _render(base.with_timeline(replace(base.timeline, tracks=tracks)), gl_context)
    finally:
        catalog_module._catalog = saved
        if saved is not None:
            saved.register_all()
    assert image[32, 32, 2] > 200, image[32, 32]


def test_the_setting_is_saved() -> None:
    # 保存で切り抜きが落ちると、開き直した途端に吹き出しの模様が画面全体に広がる
    project = _project(clipped=True)
    loaded = project_from_dict(project_to_dict(project))
    assert loaded.timeline.tracks[1].clips[0].clip_to_below is True
