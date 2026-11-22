"""拡大して画面の端に掛かった絵の縁

変形は透明な地に置いた絵を地ごと拡大する 補間が地の透明を拾うので、画面いっぱいへ拡大すると
外周の 1 画素が黒と混ざって暗くなる 画面の端に掛かった縁はそれを拾わず、画面の中に見える縁は
今までどおり滑らか（YMM4 の拡大 200% も 2 画素にまたがる）なことを、描いた画素で見る
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddScene,
    AddTrack,
    InScene,
    insert_media,
    new_scene,
)
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
    ParamValue,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.edges import EdgeStep, PictureEdges, placed_edges, transformed_edges
from sashimono.engine.decode import VideoDecoder, probe_media
from sashimono.engine.encode import ExportSettings, export_project
from sashimono.engine.gpu import (
    OFF_SCREEN,
    Compositor,
    EffectProcessor,
    GLContextError,
    OffscreenGLContext,
    Placement,
    Texture,
)
from sashimono.engine.gpu.glutil import FULL_RECT
from sashimono.engine.render import FrameRenderer, RenderQuality
from tests.media_fixtures import libx264_available

WIDTH, HEIGHT = 480, 270
SETTINGS = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
GREY = (180, 180, 180, 255)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _png(directory: Path, pixels: np.ndarray) -> MediaItem:
    from PySide6.QtGui import QImage

    height, width = pixels.shape[:2]
    data = np.ascontiguousarray(pixels, dtype=np.uint8)
    image = QImage(data.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    path = directory / f"素材-{width}x{height}.png"
    assert image.save(str(path))
    return probe_media(path)


def _plain(
    width: int, height: int, colour: tuple[int, int, int, int] = GREY, margin: int = 0
) -> np.ndarray:
    pixels = np.zeros((height + 2 * margin, width + 2 * margin, 4), dtype=np.uint8)
    pixels[margin : margin + height, margin : margin + width] = colour
    return pixels


def _placed(clip: Clip, *before: Effect, **values: float | str | AnimatedValue) -> Clip:
    """配置の欄に ``values`` を入れ、その手前へ ``before`` を積んだクリップ"""
    clip = with_fixed_items(clip, picture=True)
    effects = list(before)
    for effect in clip.effects:
        if effect.kind == TRANSFORM_EFFECT_KIND:
            for name, value in values.items():
                effect = effect.with_param(
                    name, AnimatedValue(value) if isinstance(value, int | float) else value
                )
        effects.append(effect)
    return replace(clip, effects=tuple(effects))


def _media_clip(media: MediaItem, *, native: bool = True) -> Clip:
    return Clip(timeline_start=0, duration=30, media_id=media.id, native_size=native)


def _project(clip: Clip, media: MediaItem | None = None, group: Clip | None = None) -> Project:
    project = Project.create(SETTINGS)
    if media is not None:
        project = AddMedia(media).apply(project)
    if group is not None:
        above = Track(kind=TrackKind.VIDEO, name="G")
        project = AddTrack(above).apply(project)
        project = AddClip(above.id, group).apply(project)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = AddTrack(track).apply(project)
    return AddClip(track.id, clip).apply(project)


def _render(
    project: Project, context: OffscreenGLContext, frame: int = 0, divisor: int = 1
) -> np.ndarray:
    """透明の地へ描いた絵（ストレート） 縁が透明と混ざると不透明度に出る"""
    renderer = FrameRenderer(project, context=context, quality=RenderQuality(divisor))
    try:
        return renderer.render(frame, transparent=True)
    finally:
        renderer.close()


def _border(image: np.ndarray) -> np.ndarray:
    """外周の 1 画素"""
    return np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))


def _assert_full(image: np.ndarray, alpha: int = 255, level: int = 180) -> None:
    border = _border(image).astype(int)
    assert border[:, 3].min() >= alpha - 2, f"外周が透明と混ざった: {border[:, 3].min()}"
    assert np.abs(border[:, 0] - level).max() <= 2, f"外周の色が変わった: {border[:, 0].min()}"


@pytest.mark.parametrize(
    ("size", "values"),
    [
        ((320, 180), {"scale": 150.0}),
        ((160, 90), {"scale": 300.0}),
        ((48, 27), {"scale": 1000.0}),
        ((240, 54), {"scale_x": 200.0, "scale_y": 500.0}),
        ((640, 200), {"scale_x": 75.0, "scale_y": 135.0}),
        ((90, 160), {"scale": 300.0, "rotation": 90.0}),
        ((90, 160), {"scale": 300.0, "rotation": -90.0}),
        ((160, 90), {"scale": 300.0, "rotation": 180.0}),
        ((160, 90), {"scale": 300.5}),
        ((160, 90), {"scale": 300.0, "pivot_h": "left", "pos_x": -160.0}),
        ((161, 91), {"scale_x": 48000.0 / 161, "scale_y": 27000.0 / 91}),
    ],
    ids=[
        "150",
        "300",
        "1000",
        "axes",
        "narrower_and_taller",
        "turned_90",
        "turned_-90",
        "turned_180",
        "a_hair_over",
        "left_pivot",
        "odd_size",
    ],
)
def test_a_picture_scaled_to_the_screen_has_no_dark_rim(
    gl_context: OffscreenGLContext,
    tmp_path: Path,
    size: tuple[int, int],
    values: dict[str, float | str],
) -> None:
    """小さい素材を画面いっぱいへ拡大すると、外周の 1 画素だけ 3〜4 割暗く書き出される"""
    media = _png(tmp_path, _plain(*size))
    _assert_full(_render(_project(_placed(_media_clip(media), **values), media), gl_context))


@pytest.mark.parametrize("divisor", [2, 4])
@pytest.mark.parametrize(("width", "height"), [(160, 90), (162, 90)], ids=["even", "half_pixel"])
def test_the_preview_has_no_dark_rim_either(
    gl_context: OffscreenGLContext, tmp_path: Path, divisor: int, width: int, height: int
) -> None:
    """画質を落とすと素材が半端な画素へ置かれる そこで縁を読み違えると、書き出しには無い
    暗い縁がプレビューにだけ出て、利用者が要らない拡大で逃がしてしまう
    """
    media = _png(tmp_path, _plain(width, height))
    clip = _placed(_media_clip(media), scale=300.0, scale_x=100.0 * 160 / width)
    _assert_full(_render(_project(clip, media), gl_context, divisor=divisor))


def test_a_proxy_sized_picture_has_no_dark_rim(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """控え（プロキシ）は縮めた絵が届く 届いた絵の大きさで縁を測ると、控えのある素材だけ暗い"""
    small = _png(tmp_path, _plain(80, 45))
    (stream,) = small.video_streams
    media = replace(small, video_streams=(replace(stream, width=160, height=90),))
    _assert_full(_render(_project(_placed(_media_clip(media), scale=300.0), media), gl_context))


def test_the_visible_part_of_a_transparent_picture_fills_cleanly(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """透明な余白のある PNG は、素材の枠ではなく見えている所が画面の端に来る"""
    media = _png(tmp_path, _plain(160, 90, margin=12))
    _assert_full(_render(_project(_placed(_media_clip(media), scale=300.0), media), gl_context))


def test_a_translucent_picture_keeps_its_opacity_at_the_rim(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """半透明の絵の外周だけが薄くなると、色を掛ける板を拡大したときに縁取りのように見える"""
    media = _png(tmp_path, _plain(160, 90, (180, 180, 180, 128)))
    image = _render(_project(_placed(_media_clip(media), scale=300.0), media), gl_context)
    border = _border(image).astype(int)
    assert np.abs(border[:, 3] - 128).max() <= 2
    # 縁を不透明へ引き上げてもいけない
    assert np.abs(border[:, 0] - 180).max() <= 2


@pytest.mark.parametrize("divisor", [1, 2])
@pytest.mark.parametrize("cut", [40, 41])
def test_a_cropped_picture_fills_cleanly(
    gl_context: OffscreenGLContext, tmp_path: Path, divisor: int, cut: int
) -> None:
    """黒帯を切り抜いてから拡大するのが普通の使い方 素材の枠しか見ないと切った辺が暗く残る"""
    media = _png(tmp_path, _plain(160 + 2 * cut, 90))
    crop = registry.require("crop").create(left=cut, right=cut)
    clip = _placed(_media_clip(media), crop, scale=300.0)
    _assert_full(_render(_project(clip, media), gl_context, divisor=divisor))


def test_a_colour_change_before_the_scale_keeps_the_rim(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """拡大の手前に反転などを積むと直らない、では色や向きを直した素材だけ縁が暗い"""
    media = _png(tmp_path, _plain(160, 90))
    flip = registry.require("flip").create(horizontal=True)
    _assert_full(
        _render(_project(_placed(_media_clip(media), flip, scale=300.0), media), gl_context)
    )


@pytest.mark.parametrize("divisor", [1, 2, 4])
def test_a_picture_cropped_on_one_side_and_flipped_fills_cleanly(
    gl_context: OffscreenGLContext, tmp_path: Path, divisor: int
) -> None:
    """片側だけ切って裏返すと縁が反対側へ移る 移った先を追わないと、その辺だけ暗く残る"""
    media = _png(tmp_path, _plain(202, 90))
    crop = registry.require("crop").create(left=42)
    flip = registry.require("flip").create(horizontal=True)
    # 裏返した後の絵（幅 160）の真ん中は、画面の真ん中より 21 左
    clip = _placed(_media_clip(media), crop, flip, scale=300.0, pos_x=63.0)
    _assert_full(_render(_project(clip, media), gl_context, divisor=divisor))


def test_a_shape_scaled_to_the_screen_fills_cleanly(gl_context: OffscreenGLContext) -> None:
    """図形と文字は画面と同じ大きさの絵で届く 絵の端ではなく中身の端で見ないと背景の板が暗い"""
    source = GeneratedSource(
        kind="shape",
        params={
            "shape": "rect",
            "color": (180 / 255, 180 / 255, 180 / 255, 1.0),
            "width": AnimatedValue(160.0),
            "height": AnimatedValue(90.0),
        },
    )
    clip = _placed(Clip(timeline_start=0, duration=30, source=source), scale=300.0)
    _assert_full(_render(_project(clip), gl_context))


def test_a_fitted_picture_zoomed_to_the_screen_fills_cleanly(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """画面に収めた 4:3 の絵を左右の帯が消えるまで拡大すると、帯の跡の 1 列が暗く残る"""
    media = _png(tmp_path, _plain(120, 90))
    clip = _placed(_media_clip(media, native=False), scale=100.0 * 480 / 360)
    _assert_full(_render(_project(clip, media), gl_context))


@pytest.mark.parametrize(("own", "group"), [(100.0, 300.0), (150.0, 200.0), (300.0, 110.0)])
def test_a_group_zoom_to_the_screen_fills_cleanly(
    gl_context: OffscreenGLContext, tmp_path: Path, own: float, group: float
) -> None:
    """グループ制御の拡大はクリップの配置の後ろに掛かる 2 回目の拡大で暗い縁が戻ってはいけない"""
    media = _png(tmp_path, _plain(160, 90))
    controller = _placed(
        Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource(kind=GROUP_KIND, params={GROUP_LAYERS: AnimatedValue(1.0)}),
        ),
        scale=group,
    )
    clip = _placed(_media_clip(media), scale=own)
    _assert_full(_render(_project(clip, media, controller), gl_context))


@pytest.mark.parametrize(
    "first",
    [
        {"scale": 150.0, "rotation": 90.0},
        {"scale": 150.0, "pivot_h": "left", "pos_x": -40.0},
        {"scale": 150.0, "pivot_h": "left", "move_to_pivot": True, "pos_x": -120.0},
    ],
    ids=["turned", "pivot", "moved_to_pivot"],
)
def test_a_second_scale_finds_the_edge_the_first_one_left(
    gl_context: OffscreenGLContext, tmp_path: Path, first: dict[str, float | str | bool]
) -> None:
    """縁を追う計算がシェーダの動かし方とずれると、2 段に拡大した辺だけ暗い縁が戻る"""
    turned = "rotation" in first
    media = _png(tmp_path, _plain(90, 160) if turned else _plain(160, 90))
    inner = registry.require("transform").create(**first)
    clip = _placed(_media_clip(media), inner, scale=200.0)
    _assert_full(_render(_project(clip, media), gl_context))


@pytest.mark.parametrize(("margin", "outside"), [(0, 0), (200, 85)], ids=["cut", "spilling"])
def test_an_edge_moved_in_by_a_later_effect_stays_smooth(
    gl_context: OffscreenGLContext, tmp_path: Path, margin: int, outside: int
) -> None:
    """画面いっぱいへ拡大した後でグループ制御や変形が絵を動かすと、硬くした縁や画面の外に
    伸びた帯が画面の中へ入って見える（PR #496 のレビュー）
    """
    media = _png(tmp_path, _plain(160, 90, margin=margin))
    first = registry.require("transform").create(scale=300.0)
    moved = _placed(_media_clip(media), first, pos_x=100.0)
    seen = _render(_project(moved, media), gl_context)[135, 90:106, 3].astype(int)
    # 画面の大きさのバッファでは外の半分が切れ、はみ出しのバッファ（透明な余白の大きい絵）では残る
    expected = np.array([0] * 9 + [outside, 170] + [255] * 5)
    assert np.abs(seen - expected).max() <= 2, f"動かした先の縁が違う: {seen}"


@pytest.mark.parametrize("after", ["flip", "colour", "move_along"])
def test_what_follows_the_scale_does_not_bring_the_rim_back(
    gl_context: OffscreenGLContext, tmp_path: Path, after: str
) -> None:
    """拡大の後に縁を画面の端へ残す物（反転・色・端に沿った動き）を積んでも暗い縁は戻らない"""
    media = _png(tmp_path, _plain(160, 100))
    grown = registry.require("transform").create(scale=300.0)
    later = {
        "flip": registry.require("flip").create(horizontal=True),
        "colour": registry.require("color_grade").create(),
        "move_along": registry.require("transform").create(pos_y=9.0),
    }[after]
    image = _render(_project(_placed(_media_clip(media), grown, later), media), gl_context)
    # 動かすと上下の 9 画素は前の変形で画面に切られた所が入るので、その外を見る
    sides = np.concatenate((image[12:-12, 0, 3], image[12:-12, -1, 3]))
    assert sides.min() >= 253, f"外周が透明と混ざった: {sides.min()}"


def test_a_crop_after_a_scale_is_measured_from_the_object(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """切り抜きは絵の元の範囲の端から切る 拡大した後の縁から数えると、何も切らない切り抜きを
    挟んだだけで、その後に画面いっぱいへ拡大した辺が暗い（PR #496 のレビュー）
    """
    media = _png(tmp_path, _plain(160, 90))
    grown = registry.require("transform").create(scale=200.0)
    # 元の範囲（横 160..320）の左から 10 拡大した縁（80）より内なので、切るのは 170 より左
    crop = registry.require("crop").create(left=10)
    clip = _placed(
        _media_clip(media), grown, crop, scale=150.0, scale_x=48000.0 / 230 / 1.5, pos_x=-93.9130435
    )
    image = _render(_project(clip, media), gl_context)
    assert image[135, 0, 3] >= 253, "切った辺が暗い"
    uncut = registry.require("crop").create(left=1)
    moved = registry.require("transform").create(scale=150.0, pos_x=80.0)
    # 横 200..440 縁より外の 161 より左を切るだけなので、縁は 200 のまま
    clip = _placed(_media_clip(media), moved, uncut, scale=200.0, pos_x=-160.0)
    assert _render(_project(clip, media), gl_context)[135, 0, 3] >= 253, "切っていない辺が暗い"


@pytest.mark.parametrize("flipped", [False, True])
def test_a_cut_side_reads_the_column_at_the_cut(
    gl_context: OffscreenGLContext, tmp_path: Path, flipped: bool
) -> None:
    """切った辺に前の拡大のぼけ幅が残ると、外周が切り口の列でなく 1 画素内の列の色になる
    単色の素材では見えない（PR #496 のレビュー）
    """
    pixels = _plain(160, 90, (0, 0, 0, 255))
    # 2 倍に拡大すると、切り口（170）の列は黒、その 1 画素内の列は白が混ざる
    pixels[:, 46:, :3] = 255
    media = _png(tmp_path, pixels)
    before = [
        registry.require("transform").create(scale=200.0),
        registry.require("crop").create(left=10),
    ]
    shift = -93.9130435
    if flipped:
        before.append(registry.require("flip").create(horizontal=True))
        shift = -shift
    clip = _placed(
        _media_clip(media), *before, scale=150.0, scale_x=48000.0 / 230 / 1.5, pos_x=shift
    )
    seen = _render(_project(clip, media), gl_context)[135, -1 if flipped else 0]
    assert seen[3] >= 253, "切った辺が暗い"
    assert seen[0] <= 8, f"切り口より内の列の色が出た: {seen[0]}"


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0.0, (0.5, 1.5, 2.5, 3.5)),
        (90.0, (1.5, 2.5, 3.5, 0.5)),
        (180.0, (2.5, 3.5, 0.5, 1.5)),
        (-90.0, (3.5, 0.5, 1.5, 2.5)),
    ],
)
def test_the_width_of_each_side_turns_with_it(
    rotation: float, expected: tuple[float, float, float, float]
) -> None:
    """回した後で辺ごとの幅を取り違えると、切った辺と切っていない辺の外周の読み方が入れ替わる"""
    values = {"rotation": rotation, "scale": 100.0, "scale_x": 100.0, "scale_y": 100.0}
    step = EdgeStep(
        number=lambda name: values.get(name, 0.0),
        checked=lambda name: False,
        pivot=lambda: (240.0, 135.0),
        placed=(160.0, 90.0, 320.0, 180.0),
    )
    edges = PictureEdges((160.0, 90.0, 320.0, 180.0), (0.5, 1.5, 2.5, 3.5))
    moved = transformed_edges(edges, step)
    assert moved is not None
    assert moved.soft == expected


@pytest.mark.parametrize("outer", [100.0, 50.0])
@pytest.mark.parametrize("nesting", ["scene", "group"])
def test_a_filled_scene_is_the_same_as_one_with_a_full_size_picture(
    gl_context: OffscreenGLContext, tmp_path: Path, nesting: str, outer: float
) -> None:
    """入れ子のシーンと 1 枚扱いのグループは、それ自体が画面の大きさの絵 中で補正をやめると等倍で
    置いたシーンに暗い外周が戻り、縮めて置いても等倍の絵を入れたときに無い暗い枠が付く
    """
    seen = []
    for size, inner in (((160, 90), 300.0), ((WIDTH, HEIGHT), 100.0)):
        media = _png(tmp_path, _plain(*size))
        clip = _placed(_media_clip(media), scale=inner)
        if nesting == "group":
            params: dict[str, ParamValue] = {GROUP_LAYERS: AnimatedValue(1.0), GROUP_AS_ONE: True}
            source = GeneratedSource(kind=GROUP_KIND, params=params)
            group = _placed(Clip(timeline_start=0, duration=30, source=source), scale=outer)
            project = _project(clip, media, group)
        else:
            project = AddMedia(media).apply(Project.create(SETTINGS))
            scene = new_scene(project, "中")
            project = AddScene(scene).apply(project)
            inside, track = Track(TrackKind.VIDEO, "S"), Track(TrackKind.VIDEO, "V")
            project = InScene(scene.id, AddTrack(inside)).apply(project)
            project = InScene(scene.id, AddClip(inside.id, clip)).apply(project)
            project = AddTrack(track).apply(project)
            placed = _placed(Clip(timeline_start=0, duration=30, scene_id=scene.id), scale=outer)
            project = AddClip(track.id, placed).apply(project)
        seen.append(_render(project, gl_context).astype(int))
    assert seen[1][..., 3].max() >= 253, "何も描かれていない"
    assert np.abs(seen[0] - seen[1]).max() <= 2, "拡大で埋めた入れ子だけ縁が違う"


def test_a_keyframed_zoom_reaches_the_edge_without_a_jump(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """画面いっぱいになった瞬間だけ直すと、ズームの終わりで外周が 1 段光って見える"""
    media = _png(tmp_path, _plain(160, 90))
    # 左端が画面の端から 0.6 画素内側の所から、ちょうど端へ来るまで
    zoom = AnimatedValue(
        static=300.0,
        keyframes=(Keyframe(0, 100.0 * (480 - 1.2) / 160), Keyframe(24, 300.0)),
    )
    project = _project(_placed(_media_clip(media), scale=zoom), media)
    seen = [int(_render(project, gl_context, frame)[135, 0, 3]) for frame in range(25)]
    steps = np.diff(seen)
    assert seen[-1] >= 253, f"着いた所で暗い: {seen}"
    assert steps.min() >= -2, f"ズームの途中で濃さが戻る: {seen}"
    assert steps.max() <= 20, f"濃さが跳ぶ: {seen}"


@pytest.mark.parametrize(
    ("inset", "expected"), [(0.0, 255), (0.25, 200), (0.5, 128)], ids=["on", "quarter", "half"]
)
def test_a_fraction_of_a_pixel_still_shows_at_the_edge(
    gl_context: OffscreenGLContext, tmp_path: Path, inset: float, expected: int
) -> None:
    """位置の端数まで画面の端へ吸い付けると、ゆっくり動かした絵の端が 1 画素ずつ跳ぶ"""
    media = _png(tmp_path, _plain(100, 60))
    # 幅 300 の絵の左端を画面の左端から inset だけ内側へ
    clip = _placed(_media_clip(media), scale=300.0, pos_x=-90.0 + inset)
    alpha = int(_render(_project(clip, media), gl_context)[135, 0, 3])
    assert abs(alpha - expected) <= 8


@pytest.mark.parametrize(
    ("scale", "outside", "inside"), [(150.0, 43, 213), (200.0, 64, 191), (300.0, 85, 170)]
)
def test_an_edge_inside_the_screen_stays_smooth(
    gl_context: OffscreenGLContext, tmp_path: Path, scale: float, outside: int, inside: int
) -> None:
    """画面の中に見える縁までくっきりさせると、拡大した絵や回した絵の縁にぎざぎざが出て、
    YMM4 の拡大（200% で外 0.22・内 0.79）とも合わなくなる
    """
    media = _png(tmp_path, _plain(80, 48))
    image = _render(_project(_placed(_media_clip(media), scale=scale), media), gl_context)
    left = round(WIDTH / 2 - 80 * scale / 200)
    assert abs(int(image[135, left - 1, 3]) - outside) <= 2
    assert abs(int(image[135, left, 3]) - inside) <= 2


def test_a_turned_edge_stays_smooth(gl_context: OffscreenGLContext, tmp_path: Path) -> None:
    """斜めに画面を横切る縁に手を入れると、画面の端の近くだけ縁が階段になる"""
    media = _png(tmp_path, _plain(160, 90))
    clip = _placed(_media_clip(media), scale=300.0, rotation=30.0)
    top = _render(_project(clip, media), gl_context)[0, :, 3].astype(int)
    assert ((top > 20) & (top < 235)).sum() >= 2, "端の近くで斜めの縁が 2 値になった"
    assert np.abs(np.diff(top)).max() < 200


def test_zooming_past_the_edge_does_not_move_the_picture(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    """絵の続きがある所で読む位置を内へ寄せると、拡大した動画の外周だけ隣の画素へずれる"""
    pixels = _plain(WIDTH, HEIGHT)
    pixels[:, :160, :3] = 0
    pixels[:, 160:, :3] = 255
    media = _png(tmp_path, pixels)
    image = _render(_project(_placed(_media_clip(media), scale=300.0), media), gl_context)
    # 左の列は黒 1/3 と白 2/3 をリニアで混ぜた値 寄せると白だけになる
    linear = 2.0 / 3.0
    expected = 255 * (1.055 * linear ** (1 / 2.4) - 0.055)
    assert abs(int(image[135, 0, 0]) - expected) <= 2


@pytest.mark.usefixtures("gpu")
@pytest.mark.skipif(not libx264_available(), reason="libx264 を使える ffmpeg が無い")
def test_the_exported_video_has_no_dark_rim(gl_context: OffscreenGLContext, tmp_path: Path) -> None:
    """報告された形 H.264 の小さい動画を拡大して書き出すと外周が暗く、プレビューとも食い違う"""
    source = tmp_path / "単色.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi"),
            *("-i", "color=c=0xb4b4b4:size=160x90:rate=30:duration=0.2"),
            *("-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    media = probe_media(source)
    # 4K・1080p と同じ 16:9 で、縦が 4 で割れる大きさ
    project = Project.create(replace(SETTINGS, width=640, height=360))
    for command in insert_media(project, media):
        project = command.apply(project)
    (track, clip), *_ = [
        (track, clip)
        for track in project.timeline.tracks
        for clip in track.clips
        if track.kind is TrackKind.VIDEO
    ]
    zoomed = replace(
        track,
        clips=tuple(_placed(clip, scale=400.0) if c is clip else c for c in track.clips),
    )
    tracks = tuple(zoomed if t is track else t for t in project.timeline.tracks)
    project = project.with_timeline(replace(project.timeline, tracks=tracks))

    preview = _render(project, gl_context)
    path = export_project(project, ExportSettings(path=tmp_path / "書き出し.mp4"))
    # 色の範囲（limited）を戻して読むため、本体の読み手で読む
    with VideoDecoder(path) as decoder:
        decoded = decoder.frame_at(Fraction(0))
    assert decoded is not None
    exported = decoded.astype(int)

    _assert_full(preview, level=int(preview[180, 320, 0]))
    for name, line, inner in (
        ("上", exported[0], exported[4]),
        ("下", exported[-1], exported[-5]),
        ("左", exported[:, 0], exported[:, 4]),
        ("右", exported[:, -1], exported[:, -5]),
    ):
        assert abs(line[..., 1].mean() - inner[..., 1].mean()) <= 3, f"{name}の外周が暗い"
    assert abs(exported[0, :, 1].mean() - preview[0, :, 1].mean()) <= 4


@pytest.mark.parametrize(
    "broken",
    [
        {"rotation": float("inf")},
        {"scale": float("nan")},
        {"pos_x": float("inf")},
        {"scale": 1e308},
        {"scale": 1e200, "pos_x": -1e200, "scale_x": 1e200},
        {"anchor_y": float("-inf")},
    ],
)
def test_a_broken_number_does_not_stop_the_drawing(
    gl_context: OffscreenGLContext, tmp_path: Path, broken: dict[str, float]
) -> None:
    """壊れた数のファイルで縁の計算が例外を投げると、プレビューも書き出しもそのコマで止まる"""
    media = _png(tmp_path, _plain(160, 90))
    first = registry.require("transform").create()
    for name, value in broken.items():
        # 項目の範囲で切られないよう、ファイルから読んだ形（AnimatedValue）のまま入れる
        first = first.with_param(name, AnimatedValue(value))
    clip = _placed(_media_clip(media), first, scale=300.0)
    assert _render(_project(clip, media), gl_context).shape == (HEIGHT, WIDTH, 4)


@pytest.mark.parametrize(
    ("inset", "expected"), [(0.2, 204), (0.4, 153)], ids=["fifth", "two_fifths"]
)
def test_a_move_alone_keeps_the_exact_fraction(
    gl_context: OffscreenGLContext, tmp_path: Path, inset: float, expected: int
) -> None:
    """拡大率 100 で端数だけ動かした縁は元から正しい濃さ 寄せると、読み込んだ動きの端が濃くなる"""
    media = _png(tmp_path, _plain(100, 60))
    clip = _placed(_media_clip(media), pos_x=-190.0 + inset)
    assert abs(int(_render(_project(clip, media), gl_context)[135, 0, 3]) - expected) <= 2


@pytest.mark.parametrize("scale", [100.0, 120.0, 170.0, 300.0])
@pytest.mark.parametrize("shift", [0.0, -0.8, 0.3])
def test_a_soft_tail_at_the_edge_is_not_taken_for_an_edge(
    gl_context: OffscreenGLContext, scale: float, shift: float
) -> None:
    """ぼかしや粒の裾を縁と読むと、画面の端を横切る粒の端の 1 画素が濃くなり、YMM4 の書き出しから
    離れる（実物のテンプレートの雪で、違った画素の差が 31.75 から 36.75 へ開いた）
    """
    with gl_context:
        compositor = Compositor(WIDTH, HEIGHT)
        processor = EffectProcessor(WIDTH, HEIGHT, compositor.quad)
        # 画面の左端が入力のどこに当たるか そこから内へ濃くなる裾を置く
        border = round(WIDTH / 2 + (-WIDTH / 2 - shift) * 100.0 / scale)
        pixels = _plain(WIDTH, HEIGHT, (255, 255, 255, 0))
        for step, alpha in enumerate((12, 60, 130, 190, 230, 250)):
            pixels[:, border + step, 3] = alpha
        pixels[:, border + 6 : border + 40, 3] = 255
        texture = Texture.from_array(pixels)
        effects = (
            # 絵を動かしうる物の後 置いたときの縁はもう当てにならない（粒・ぼかし）
            registry.require("blur").create(radius=0.0),
            registry.require("transform").create(scale=scale, pos_x=shift),
        )
        seen = []
        for screen in (None, OFF_SCREEN):
            result = processor.apply(
                texture, effects, frame=0, fps=30.0, source_rect=FULL_RECT, screen=screen
            )
            compositor.begin((0.0, 0.0, 0.0, 0.0))
            compositor.draw_handle(
                result.color, Placement(0.0, 0.0, float(WIDTH), float(HEIGHT)), flip=False
            )
            seen.append(compositor.read(straight=True))
        texture.release()
        processor.release()
        compositor.release()
    assert np.array_equal(seen[0], seen[1]), "ぼけた裾の読む位置が動いた"


def test_the_rim_of_the_placed_frame_is_not_blurred() -> None:
    """縮めて置いた枠の縁までなまったと見積もると、画面の端から少しはみ出した絵（301%）の
    外周の中身が、画質を落としたプレビューでだけ隣の画素へずれる
    """
    rect = (52.0, 29.0, 68.0, 38.0)
    assert placed_edges(rect, rect, (64, 36)).soft == (0.5,) * 4
    inner = (56.0, 31.0, 64.0, 36.0)
    assert placed_edges(inner, rect, (64, 36)).soft == (0.625,) * 4


def test_a_work_area_is_not_taken_for_the_screen(gl_context: OffscreenGLContext) -> None:
    """焼き込みの作業場や、後で回して貼る絵の端を画面の端と見ると、画面の中に見える縁が硬くなる"""
    with gl_context:
        compositor = Compositor(WIDTH, HEIGHT)
        processor = EffectProcessor(WIDTH, HEIGHT, compositor.quad)
        pixels = _plain(WIDTH, HEIGHT, (0, 0, 0, 0))
        pixels[90:180, 160:320] = GREY
        texture = Texture.from_array(pixels)
        scaled = (registry.require("transform").create(scale=300.0),)
        alphas = []
        for screen in (None, OFF_SCREEN):
            result = processor.apply(
                texture,
                scaled,
                frame=0,
                fps=30.0,
                source_rect=FULL_RECT,
                bounds=(160.0, 90.0, 320.0, 180.0),
                screen=screen,
            )
            compositor.begin((0.0, 0.0, 0.0, 0.0))
            compositor.draw_handle(
                result.color, Placement(0.0, 0.0, float(WIDTH), float(HEIGHT)), flip=False
            )
            alphas.append(int(compositor.read(straight=True)[135, 0, 3]))
        texture.release()
        processor.release()
        compositor.release()
    assert alphas[0] >= 253
    assert abs(alphas[1] - 170) <= 2
