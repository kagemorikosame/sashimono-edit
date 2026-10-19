"""プレビューで掴む外枠（:func:`clip_outline`）が、描いた絵と重なること

枠の計算は描く側（置き方・配置のシェーダ・反転）を前向きに辿り直した物で、描く側を
直したときに片方だけ変わりうる 食い違うと、掴んだ所と動く物が合わない
GPU で本当に描いて、色の付いた範囲と枠を画素で比べる
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddMedia, AddTrack
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    MediaItem,
    ParamValue,
    Project,
    ProjectSettings,
    SceneId,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.spec import SelectSpec
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, RenderQuality
from sashimono.engine.render.outline import Outline, clip_outline, has_outline

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
    """64 × 32 の真っ白な画像 画面（320 × 180）とは縦横比も違う"""
    from PySide6.QtGui import QImage

    image = np.full((32, 64, 4), 255, dtype=np.uint8)
    data = np.ascontiguousarray(image)
    qimage = QImage(data.tobytes(), 64, 32, 64 * 4, QImage.Format.Format_RGBA8888)
    path = tmp_path / "白.png"
    assert qimage.save(str(path))
    return probe_media(path)


def _project(clip: Clip, media: MediaItem | None = None) -> Project:
    project = Project.create(SETTINGS)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    commands = [AddTrack(track), AddClip(track.id, clip)]
    if media is not None:
        commands.insert(0, AddMedia(media))
    for command in commands:
        project = command.apply(project)
    return project


def _param(value: float | str | bool) -> ParamValue:
    # 数は動かせる値として持つ（定義の TrackSpec と同じ形） 真偽と選択肢はそのまま
    if isinstance(value, bool | str):
        return value
    return AnimatedValue(float(value))


def _placed(clip: Clip, **values: float | str | bool) -> Clip:
    """固定の配置に値を入れたクリップ"""
    clip = with_fixed_items(clip, picture=True)
    effects = []
    for effect in clip.effects:
        if effect.fixed and effect.kind == TRANSFORM_EFFECT_KIND:
            for name, value in values.items():
                effect = effect.with_param(name, _param(value))
        effects.append(effect)
    return replace(clip, effects=tuple(effects))


def _picture_clip(media: MediaItem, *, native: bool = True, **values: float | str | bool) -> Clip:
    base = Clip(timeline_start=0, duration=10, media_id=media.id, native_size=native)
    return _placed(base, **values)


def _draw(
    project: Project, context: OffscreenGLContext, quality: RenderQuality | None = None
) -> tuple[np.ndarray, FrameRenderer]:
    renderer = (
        FrameRenderer(project, context=context)
        if quality is None
        else FrameRenderer(project, context=context, quality=quality)
    )
    return renderer.render(0), renderer


def _outline_of(
    project: Project, context: OffscreenGLContext, quality: RenderQuality | None = None
) -> tuple[np.ndarray, Outline]:
    image, renderer = _draw(project, context, quality)
    try:
        clip = project.timeline.tracks[0].clips[0]
        with context:
            extent = renderer.object_extent(clip, 0)
        canvas = (image.shape[1], image.shape[0])
        outline = clip_outline(project, clip, 0, canvas=canvas, extent=extent)
    finally:
        renderer.close()
    assert outline is not None
    return image, outline


def _painted(image: np.ndarray) -> tuple[int, int, int, int]:
    """色の付いた範囲 左・上・右・下（右と下は含まない）"""
    ys, xs = np.nonzero(image[..., :3].max(axis=2) > 128)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bounds(outline: Outline) -> tuple[float, float, float, float]:
    xs = [x for x, _ in outline.corners]
    ys = [y for _, y in outline.corners]
    return min(xs), min(ys), max(xs), max(ys)


def _assert_matches(image: np.ndarray, outline: Outline, tolerance: float = 1.5) -> None:
    """色の付いた範囲の外接矩形と、枠の外接矩形が画素の丸めの幅で重なる"""
    painted = _painted(image)
    bounds = _bounds(outline)
    for got, want in zip(painted, bounds, strict=True):
        assert abs(got - want) <= tolerance, (painted, bounds)


def _inset(outline: Outline, amount: float) -> list[tuple[float, float]]:
    """四隅を中心へ ``amount`` 画素寄せた（負なら外へ出した）点"""
    cx = sum(x for x, _ in outline.corners) / 4.0
    cy = sum(y for _, y in outline.corners) / 4.0
    points = []
    for x, y in outline.corners:
        length = math.hypot(cx - x, cy - y)
        points.append((x + (cx - x) / length * amount, y + (cy - y) / length * amount))
    return points


def _lit(image: np.ndarray, point: tuple[float, float]) -> bool:
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return False
    return bool(image[y, x, :3].max() > 128)


class TestTheOutlineIsWhereThePictureIs:
    def test_a_fitted_old_clip(self, gl_context: OffscreenGLContext, picture: MediaItem) -> None:
        # 前の版のクリップは画面に収める 素材の画素で数えると枠が真ん中に小さく出る
        clip = Clip(timeline_start=0, duration=10, media_id=picture.id)
        image, outline = _outline_of(_project(clip, picture), gl_context)
        assert _bounds(outline) == (0.0, 10.0, 320.0, 170.0)
        _assert_matches(image, outline)

    def test_a_picture_at_its_own_pixels(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 素材の画素で置いた絵に画面いっぱいの枠が出ると、絵の無い所を掴んでも絵が動いてしまう
        image, outline = _outline_of(_project(_picture_clip(picture), picture), gl_context)
        assert _bounds(outline) == (128.0, 74.0, 192.0, 106.0)
        _assert_matches(image, outline)

    def test_a_moved_and_enlarged_picture(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # Y は上が正 枠が下へ動くと、Y を上げたのに枠だけ逆へ行く
        clip = _picture_clip(picture, pos_x=30, pos_y=20, scale=150, scale_y=50)
        image, outline = _outline_of(_project(clip, picture), gl_context)
        _assert_matches(image, outline)
        left, top, right, bottom = _bounds(outline)
        assert (right - left, bottom - top) == pytest.approx((96.0, 24.0))
        assert (left + right) / 2.0 == pytest.approx(160.0 + 30.0)
        assert (top + bottom) / 2.0 == pytest.approx(90.0 - 20.0)

    def test_a_rotated_picture(self, gl_context: OffscreenGLContext, picture: MediaItem) -> None:
        # 回転は時計回りが正 逆向きに回すと外接矩形は同じでも角の位置が食い違う
        clip = _picture_clip(picture, rotation=30, pos_x=-20)
        image, outline = _outline_of(_project(clip, picture), gl_context)
        _assert_matches(image, outline)
        for point in _inset(outline, 3.0):
            assert _lit(image, point)
        for point in _inset(outline, -3.0):
            assert not _lit(image, point)
        # 左上の角は、時計回りなら元の左上より上（画面の Y は小さく）へ行かない
        top_left, top_right = outline.corners[0], outline.corners[1]
        assert top_right[1] > top_left[1]

    def test_a_picture_turned_around_its_own_corner(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 支点を絵の左上へ 画面の中央を支点に数えると、枠が絵から離れた所で回る
        clip = _picture_clip(
            picture, pivot_h="left", pivot_v="top", rotation=90, scale=50, anchor_x=4
        )
        image, outline = _outline_of(_project(clip, picture), gl_context)
        _assert_matches(image, outline)
        for point in _inset(outline, 2.0):
            assert _lit(image, point)

    def test_a_picture_moved_to_its_pivot(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 支点へ寄せる動きを枠が辿り損ねると、枠が絵から離れた所に出て、掴んだ所と動く物が合わない
        clip = _picture_clip(
            picture, pivot_h="right", pivot_v="bottom", move_to_pivot=True, rotation=45
        )
        image, outline = _outline_of(_project(clip, picture), gl_context)
        _assert_matches(image, outline)
        for point in _inset(outline, 3.0):
            assert _lit(image, point)

    def test_a_lighter_preview(self, gl_context: OffscreenGLContext, picture: MediaItem) -> None:
        # 画質を落としたプレビューでは合成が小さい 枠もその画素で数える
        clip = _picture_clip(picture, pos_x=10, scale=200)
        image, outline = _outline_of(_project(clip, picture), gl_context, RenderQuality(2))
        assert image.shape[:2] == (90, 160)
        _assert_matches(image, outline)

    @pytest.mark.parametrize("divisor", [2, 4])
    def test_a_lighter_preview_moved_and_turned(
        self, gl_context: OffscreenGLContext, picture: MediaItem, divisor: int
    ) -> None:
        # X・Y・中心は画面の画素 描く側は画質の分だけ縮めて当てる（Issue #151） 枠だけ
        # 縮めずに当てると、1/4 では絵から 3 倍ずれた所に枠が出て、掴んでも絵が付いて来ない
        clip = _picture_clip(picture, pos_x=48, pos_y=-24, anchor_x=16, rotation=30)
        image, outline = _outline_of(_project(clip, picture), gl_context, RenderQuality(divisor))
        assert image.shape[:2] == (180 // divisor, 320 // divisor)
        _assert_matches(image, outline)

    @pytest.mark.parametrize("divisor", [2, 4])
    def test_a_lighter_preview_of_a_shape(
        self, gl_context: OffscreenGLContext, divisor: int
    ) -> None:
        # 図形は自分の設定（幅・位置）も画面の画素 描く絵ごと縮めるので、入れ物も縮んで届く
        shape = Clip(
            timeline_start=0,
            duration=10,
            source=GeneratedSource(
                kind="shape",
                params={
                    "shape": "rect",
                    "width": AnimatedValue(80.0),
                    "height": AnimatedValue(40.0),
                    "pos_x": AnimatedValue(-40.0),
                },
            ),
        )
        project = _project(_placed(shape, pos_y=20, rotation=15))
        image, outline = _outline_of(project, gl_context, RenderQuality(divisor))
        _assert_matches(image, outline)

    def test_a_shape_uses_its_painted_extent(self, gl_context: OffscreenGLContext) -> None:
        # 生成オブジェクトの絵は画面いっぱいの大きさ 絵の大きさで枠を出すと画面全体が枠になる
        shape = Clip(
            timeline_start=0,
            duration=10,
            source=GeneratedSource(
                kind="shape",
                params={
                    "shape": "rect",
                    "width": AnimatedValue(80.0),
                    "height": AnimatedValue(40.0),
                    "pos_x": AnimatedValue(30.0),
                },
            ),
        )
        image, outline = _outline_of(_project(_placed(shape, rotation=20, pos_y=15)), gl_context)
        _assert_matches(image, outline)
        for point in _inset(outline, 3.0):
            assert _lit(image, point)

    def test_a_shape_larger_than_the_screen(self, gl_context: OffscreenGLContext) -> None:
        # 画面より大きい絵は別の道（絵の大きさのバッファ）で配置を掛ける
        shape = Clip(
            timeline_start=0,
            duration=10,
            source=GeneratedSource(
                kind="shape",
                params={
                    "shape": "rect",
                    "width": AnimatedValue(400.0),
                    "height": AnimatedValue(30.0),
                },
            ),
        )
        project = _project(_placed(shape, scale=50, pos_x=12, rotation=10))
        image, outline = _outline_of(project, gl_context)
        _assert_matches(image, outline)
        left, _, right, _ = _bounds(outline)
        # 半分に縮めた 400 の帯 画面に収めて数えると 320 を基に縮めた幅になる
        assert right - left > 190.0


class TestApproximate:
    def test_an_exact_outline_is_solid(self, picture: MediaItem) -> None:
        # 配置だけのふつうのクリップまで点線にすると、点線が「ずれているかもしれない」の
        # 印として働かなくなり、本当にずれる所を見分けられない
        clip = _picture_clip(picture, rotation=10)
        outline = clip_outline(_project(clip, picture), clip, 0)
        assert outline is not None and not outline.approximate

    def test_an_effect_after_the_placement_makes_it_approximate(self, picture: MediaItem) -> None:
        # 置いた後の絵に掛かる物は形を変えうる ぴったりの線で描くと、枠を信じて外す
        clip = _picture_clip(picture)
        wave = registry.require("blur").create(radius=3)
        clip = replace(clip, effects=(*clip.effects, wave))
        outline = clip_outline(_project(clip, picture), clip, 0)
        assert outline is not None and outline.approximate

    def test_an_effect_before_the_placement_keeps_it_exact(self, picture: MediaItem) -> None:
        # 足したぼかし（配置の前に入る）だけで枠が点線になると、ほとんどの絵が点線になり
        # 本当にずれている物を見分けられない
        clip = _picture_clip(picture)
        blur = registry.require("blur").create(radius=3)
        clip = replace(clip, effects=(blur, *clip.effects))
        outline = clip_outline(_project(clip, picture), clip, 0)
        assert outline is not None and not outline.approximate

    @pytest.mark.parametrize(
        ("kind", "values"),
        [("crop", {"left": 10.0}), ("expand_area", {"right": 20.0})],
    )
    def test_a_shaping_effect_before_the_placement_makes_it_approximate(
        self, picture: MediaItem, kind: str, values: dict[str, float]
    ) -> None:
        # 切り抜きは見える範囲を狭め、片側だけの領域拡張は絵をずらす 置いた矩形の枠を
        # 実線で出すと、見えていない所まで絵だと思わせる
        effect = registry.require(kind).create(**values)
        clip = _picture_clip(picture)
        clip = replace(clip, effects=(effect, *clip.effects))
        outline = clip_outline(_project(clip, picture), clip, 0)
        assert outline is not None and outline.approximate

    def test_a_broken_expansion_keeps_the_outline_finite(self, picture: MediaItem) -> None:
        # 描く側は壊れた数を既定へ戻す 枠だけが無限大の座標になると描けも掴めもしない
        expand = registry.require("expand_area").create()
        expand = expand.with_param("left", AnimatedValue(math.inf))
        clip = _picture_clip(picture, pivot_h="left", rotation=30)
        clip = replace(clip, effects=(expand, *clip.effects))
        outline = clip_outline(_project(clip, picture), clip, 0)
        assert outline is not None
        assert all(math.isfinite(v) for corner in outline.corners for v in corner)

    def test_a_tilt_makes_it_approximate(self, picture: MediaItem) -> None:
        # 奥行きの回転は遠近で台形に歪むが、枠は平らな矩形のまま 実線で出すと、絵の無い
        # 角の外を掴めると思わせる
        clip = _picture_clip(picture, rotation_x=30)
        outline = clip_outline(_project(clip, picture), clip, 0)
        assert outline is not None and outline.approximate


def test_the_pivot_choices_match_the_definition() -> None:
    # 外枠の計算は支点の選択肢を名前で見る 定義の名前を変えると、枠だけが黙って
    # 画面の中央を支点にしてずれる
    definition = registry.require(TRANSFORM_EFFECT_KIND)
    horizontal = definition.spec("pivot_h")
    vertical = definition.spec("pivot_v")
    assert isinstance(horizontal, SelectSpec) and isinstance(vertical, SelectSpec)
    assert {key for key, _ in horizontal.choices} == {"screen", "left", "right", "center", "origin"}
    assert {key for key, _ in vertical.choices} == {"screen", "top", "bottom", "middle", "origin"}


def test_scenes_and_filters_have_no_outline() -> None:
    # 下の絵や入れ子の合成をそのまま使うクリップに枠を出すと、掴んでも枠のとおりには動かない
    project = Project.create(SETTINGS)
    filtered = Clip(timeline_start=0, duration=10, source=GeneratedSource(kind="filter", params={}))
    assert clip_outline(project, filtered, 0) is None
    nested = Clip(timeline_start=0, duration=10, scene_id=SceneId("入れ子"))
    assert not has_outline(nested)
    assert clip_outline(project, nested, 0) is None


def test_the_previous_object_has_no_outline() -> None:
    # 直前オブジェクトの絵は下のクリップ次第で、自分の中身からは大きさが分からない
    # 枠を出すと、中身の無い生成オブジェクトとして掴んでも絵と合わない枠が出る
    copy = Clip(
        timeline_start=0, duration=10, source=GeneratedSource(kind="previous_object", params={})
    )
    assert not has_outline(copy)
