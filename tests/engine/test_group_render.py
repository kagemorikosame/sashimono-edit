"""グループ制御を描くと、受け持つクリップ 1 本ずつが動くこと（GPU で描いて画素で比べる）

グループ制御の配置は、クリップ自身の配置の後ろへ掛かる（画面の中央を支点に拡大・回転して、
グループの位置だけずらす） AviUtl のグループ制御の座標・拡大率・回転と同じ 不透明度は掛け合わせ、
エフェクトはクリップ 1 本ずつに掛かる 比べる相手は、同じ動きをクリップ自身に書いた絵
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddMedia, AddTrack
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.model import (
    GROUP_AS_ONE,
    GROUP_KIND,
    GROUP_LAYERS,
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SETTINGS = ProjectSettings(width=320, height=180, frame_rate=FrameRate(30))


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture
def picture(tmp_path: Path) -> MediaItem:
    """64 × 32 の画像 左半分が赤、右半分が青（回したときの向きが分かるように）"""
    from PySide6.QtGui import QImage

    pixels = np.zeros((32, 64, 4), dtype=np.uint8)
    pixels[:, :32] = (255, 40, 40, 255)
    pixels[:, 32:] = (40, 80, 255, 255)
    data = np.ascontiguousarray(pixels)
    image = QImage(data.tobytes(), 64, 32, 64 * 4, QImage.Format.Format_RGBA8888)
    path = tmp_path / "赤と青.png"
    assert image.save(str(path))
    return probe_media(path)


def _placed(clip: Clip, **values: float) -> Clip:
    clip = with_fixed_items(clip, picture=True)
    effects = []
    for effect in clip.effects:
        if effect.fixed and effect.kind == TRANSFORM_EFFECT_KIND:
            for name, value in values.items():
                effect = effect.with_param(name, AnimatedValue(value))
        effects.append(effect)
    return replace(clip, effects=tuple(effects))


def _object(media: MediaItem, **values: float) -> Clip:
    return _placed(
        Clip(timeline_start=0, duration=10, media_id=media.id, native_size=True), **values
    )


def _group(layers: int = 1, effects: tuple[Effect, ...] = (), **values: float) -> Clip:
    clip = Clip(
        timeline_start=0,
        duration=10,
        source=GeneratedSource(kind=GROUP_KIND, params={GROUP_LAYERS: layers}),
        effects=effects,
    )
    return _placed(clip, **values)


def _render(
    context: OffscreenGLContext, media: MediaItem, *clips: Clip, frame: int = 0
) -> np.ndarray:
    project = AddMedia(media).apply(Project.create(SETTINGS))
    for index, clip in enumerate(clips):
        track = Track(kind=TrackKind.VIDEO, name=f"V{index + 1}")
        project = AddClip(track.id, clip).apply(AddTrack(track).apply(project))
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(frame).astype(np.int16)
    finally:
        renderer.close()


def _same(first: np.ndarray, second: np.ndarray) -> None:
    difference = np.abs(first - second)
    assert (difference > 8).mean() < 0.002, "グループで動かした絵と、同じ動きを書いた絵が違う"
    assert first[:, :, :3].max() > 100, "何も描かれていない"


def test_the_group_moves_and_scales_around_its_position(
    gl_context: OffscreenGLContext, picture: MediaItem
) -> None:
    # クリップの X 10 を 2 倍してグループの X 40 を足す 右へ 60 ずれて 2 倍
    grouped = _render(
        gl_context, picture, _group(pos_x=40.0, scale=200.0), _object(picture, pos_x=10.0)
    )
    direct = _render(gl_context, picture, _object(picture, pos_x=60.0, scale=200.0))
    _same(grouped, direct)


def test_the_group_turns_each_object_around_its_position(
    gl_context: OffscreenGLContext, picture: MediaItem
) -> None:
    # 画面の中央から右へ 50 の物を時計回りに 90 度 中央から下へ 50 の所で縦向きになる
    grouped = _render(gl_context, picture, _group(rotation=90.0), _object(picture, pos_x=50.0))
    direct = _render(gl_context, picture, _object(picture, pos_y=-50.0, rotation=90.0))
    _same(grouped, direct)


def test_the_opacity_multiplies(gl_context: OffscreenGLContext, picture: MediaItem) -> None:
    # グループの不透明度が掛からないと、AviUtl でグループごと薄めた場面が読み込むと濃いまま出る
    group = replace(_group(), opacity=AnimatedValue(0.5))
    grouped = _render(gl_context, picture, group, _object(picture))
    direct = _render(gl_context, picture, replace(_object(picture), opacity=AnimatedValue(0.5)))
    _same(grouped, direct)


def test_the_group_effects_go_on_each_object(
    gl_context: OffscreenGLContext, picture: MediaItem
) -> None:
    # グループのエフェクトはクリップ 1 本ずつに掛かる（AviUtl のグループ制御のフィルタ効果）
    invert = registry.require("invert").create()
    grouped = _render(gl_context, picture, _group(effects=(invert,)), _object(picture))
    own = _object(picture)
    direct = _render(gl_context, picture, replace(own, effects=(invert, *own.effects)))
    _same(grouped, direct)


def test_layers_past_the_reach_stay_put(gl_context: OffscreenGLContext, picture: MediaItem) -> None:
    # 対象レイヤー数 1 で 2 本手前の物まで動くと、グループに入れていない物が一緒に動く
    grouped = _render(
        gl_context,
        picture,
        _group(pos_x=80.0),
        _object(picture, pos_y=40.0),
        _object(picture, pos_y=-40.0),
    )
    direct = _render(
        gl_context,
        picture,
        _object(picture, pos_x=80.0, pos_y=40.0),
        _object(picture, pos_y=-40.0),
    )
    _same(grouped, direct)


def test_the_group_itself_draws_nothing(gl_context: OffscreenGLContext, picture: MediaItem) -> None:
    # グループ制御そのものが何かを描くと、受け持つ物が無い所に余計な絵や枠が映り込む
    image = _render(gl_context, picture, _group(pos_x=40.0))
    assert image[:, :, :3].max() == 0


def _as_one(group: Clip) -> Clip:
    assert group.source is not None
    return replace(group, source=group.source.with_param(GROUP_AS_ONE, True))


def test_as_one_picture_hides_the_lower_object_under_the_upper(
    gl_context: OffscreenGLContext, picture: MediaItem
) -> None:
    # 1 枚の絵として扱うと、重ねた 2 つを 1 枚にしてから半分に薄める 重なった所は上の物だけが
    # 薄く出る（下の物が透けない） 1 本ずつだと、上の物越しに下の物が見える
    group = replace(_group(layers=2), opacity=AnimatedValue(0.5))
    lower, upper = _object(picture, pos_x=-12.0), _object(picture, pos_x=12.0)
    reference = _render(gl_context, picture, lower, upper)
    together = _render(gl_context, picture, _as_one(group), lower, upper)
    separate = _render(gl_context, picture, group, lower, upper)
    expected = (reference[:, :, :3] * 0.5).astype(np.int16)
    assert np.abs(together[:, :, :3] - expected).max() <= 3
    # 1 本ずつなら重なった所で下の物が透けて、1 枚にした絵と違う
    assert np.abs(separate[:, :, :3] - together[:, :, :3]).max() > 20


def test_as_one_moves_the_whole_picture(gl_context: OffscreenGLContext, picture: MediaItem) -> None:
    # 配置は 1 枚の絵に掛かる 位置の動きは 1 本ずつに掛けたときと同じ所へ出る
    together = _render(
        gl_context, picture, _as_one(_group(pos_x=40.0, scale=200.0)), _object(picture, pos_x=10.0)
    )
    direct = _render(gl_context, picture, _object(picture, pos_x=60.0, scale=200.0))
    _same(together, direct)


def _moving_x(clip: Clip) -> Clip:
    motion = AnimatedValue(
        keyframes=(Keyframe(frame=0, value=-80.0), Keyframe(frame=9, value=80.0))
    )
    effects = tuple(
        effect.with_param("pos_x", motion)
        if effect.fixed and effect.kind == TRANSFORM_EFFECT_KIND
        else effect
        for effect in clip.effects
    )
    return replace(clip, effects=effects)


def test_the_trail_reads_the_group_at_each_sample(
    gl_context: OffscreenGLContext, picture: MediaItem
) -> None:
    # 現在位置でグループを止めてから残像を描くと、過去の絵まで現在位置に重なる
    trail = registry.require("after_image").create(strength=90.0, samples=10)
    own = _object(picture)
    own = replace(own, effects=(trail, *own.effects))
    grouped = _render(gl_context, picture, _moving_x(_group()), own, frame=9)
    direct = _render(gl_context, picture, _moving_x(own), frame=9)
    _same(grouped, direct)


def test_an_outer_group_moves_an_as_one_group(
    gl_context: OffscreenGLContext, picture: MediaItem
) -> None:
    # 内側だけを別の合成先へ描いても、その合成結果へ外側の制御を掛ける
    nested = _render(
        gl_context,
        picture,
        _group(pos_x=40.0),
        _as_one(_group()),
        _object(picture, pos_x=10.0),
    )
    direct = _render(gl_context, picture, _object(picture, pos_x=50.0))
    _same(nested, direct)
