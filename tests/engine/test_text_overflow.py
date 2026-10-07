"""画面からはみ出すテキストを、はみ出す分まで広げた絵に描く（#256）

前はテキストを画面と同じ大きさの絵にしか描かなかった 画面の幅を超える 1 行、
画面の高さを超える行数、位置をずらして画面の外へ寄せた字、縁取りや影が画面の端に
掛かる字は、絵の端で切れていた 切れた字は、配置で動かしても切れたまま出る

広げた絵は図形と同じく、画面の中心に等倍で置く道（レンダラの ``_draw_oversized``）を通る
収まる字の絵は今までと同じ大きさで、見た目も位置も変わらないことを押さえる
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    ParamValue,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine import sources as engine_sources
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, RenderQuality
from sashimono.engine.render import renderer as renderer_module
from sashimono.engine.render.renderer import MAX_GENERATED_CACHE, _Made, _make_room
from sashimono.engine.sources import MAX_CANVAS, render_source, source_canvas

SCREEN = (1920, 1080)
#: 既定の書体の大きさ 64 で全角 1 文字およそ 52 画素 40 文字で 1920 を超える
WIDE = "あ" * 40


def text(**params: ParamValue | float | str | bool) -> GeneratedSource:
    base: dict[str, ParamValue] = {"text": "あ" * 20, "size": AnimatedValue(64.0)}
    for name, value in params.items():
        base[name] = (
            AnimatedValue(float(value))
            if isinstance(value, int | float) and not isinstance(value, bool)
            else value
        )
    return GeneratedSource(kind="text", params=base)


def ink(image: np.ndarray) -> tuple[int, int, int, int]:
    """色の付いた範囲（左・上・右・下 右と下は含まない）"""
    ys, xs = np.nonzero(image[..., 3] > 0)
    assert len(xs), "何も描かれていない"
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def drawn(source: GeneratedSource, scale: float = 1.0) -> np.ndarray:
    """描く道と同じく、:func:`source_canvas` の大きさの絵に描く"""
    width, height = round(SCREEN[0] * scale), round(SCREEN[1] * scale)
    canvas = source_canvas(source, width, height, scale=(scale, scale))
    image = engine_sources.render_source_framed(
        source, *canvas, scale=(scale, scale), screen=(width, height)
    )[0]
    assert image is not None
    return image


def assert_not_cut(image: np.ndarray) -> None:
    """色の付いた範囲が絵の端に届いていない（どこも切れていない）"""
    left, top, right, bottom = ink(image)
    height, width = image.shape[:2]
    assert left > 0 and top > 0, (left, top)
    assert right < width and bottom < height, (right, bottom, width, height)


def assert_same(after: np.ndarray, before: np.ndarray) -> None:
    """字の形と位置が同じ 輪郭の滑らかにした端の濃さだけ 4/255 まで違ってよい

    Qt は絵からはみ出す輪郭を、収まる輪郭と別の手順で塗る 切れていた前の絵の方が
    字の端の数十〜数百画素で 3〜4/255 だけ濃さが違った 位置が 1 画素ずれると 100 を超えて違う
    """
    assert int(np.abs(after - before).max()) <= 4


#: Issue の表で切れていた条件
OVERFLOWING = {
    "全角 40 文字": text(text=WIDE),
    "全角 70 文字": text(text="あ" * 70),
    "10 文字 x 20 行": text(text="\n".join(["あ" * 10] * 20)),
    "右へ 700 ずらす": text(pos_x=700),
    "上へ 600 ずらす": text(pos_y=600),
    "AviUtl2 の組み方": text(text=WIDE, layout="aviutl"),
    "縦書き": text(text=WIDE, vertical=True),
    "縁取りと影": text(text="あ" * 36, border_width=12, shadow_x=20, shadow_y=-20, shadow_blur=8),
    "大きさ 128": text(size=128),
    "左揃えで左へずらす": text(text=WIDE, anchor="left", pos_x=-900),
}


class TestCanvas:
    @pytest.mark.parametrize("name", list(OVERFLOWING))
    def test_overflowing_text_is_not_cut(self, name: str) -> None:
        # 画面の大きさの絵に描くと、はみ出した字が端で切れて、配置で動かしても戻らない
        # 端に届いていないだけでは、字と字の間で切れたときに見逃すので、十分に大きな絵へ
        # 描いた字と同じ大きさで出ているかも見る
        source = OVERFLOWING[name]
        image = drawn(source)
        assert_not_cut(image)
        reference = render_source(source, image.shape[1] + 400, image.shape[0] + 400)
        assert reference is not None
        left, top, right, bottom = ink(image)
        whole = ink(reference)
        assert right - left == whole[2] - whole[0]
        assert bottom - top == whole[3] - whole[1]

    @pytest.mark.parametrize("name", list(OVERFLOWING))
    def test_the_grown_picture_keeps_the_screen_part(self, name: str) -> None:
        # 広げた絵の画面の部分は、画面の大きさで描いた絵と同じ 広げたことで字の位置や形が
        # 変わると、収まっている部分まで見た目が動く
        source = OVERFLOWING[name]
        grown = drawn(source)
        height, width = grown.shape[:2]
        assert (width, height) != SCREEN
        left, top = (width - SCREEN[0]) // 2, (height - SCREEN[1]) // 2
        assert (width - SCREEN[0]) % 2 == 0 and (height - SCREEN[1]) % 2 == 0
        screen = render_source(source, *SCREEN)
        assert screen is not None
        part = grown[top : top + SCREEN[1], left : left + SCREEN[0]].astype(np.int16)
        # ぼかした影は、画面の端の外にあった影がにじんで入ってくるので、端から離れた所を比べる
        blurred = source.params.get("shadow_blur")
        edge = 40 if isinstance(blurred, AnimatedValue) and blurred.static else 0
        inner = (slice(edge, SCREEN[1] - edge), slice(edge, SCREEN[0] - edge))
        assert_same(part[inner], screen.astype(np.int16)[inner])

    @pytest.mark.parametrize(
        "source",
        [
            text(),
            text(text="\n".join(["あ" * 25] * 8)),
            text(layout="aviutl"),
            text(text="あ" * 10, vertical=True),
            text(border_width=8, shadow_x=10, shadow_y=-10, shadow_blur=4),
        ],
    )
    def test_fitting_text_keeps_the_screen_size(self, source: GeneratedSource) -> None:
        # 収まる字は今までと同じ大きさの絵 広げると、画面に収まる字幕まで別の道を通り、
        # 効果の掛かり方やメモリが変わる
        assert source_canvas(source, *SCREEN) == SCREEN

    def test_the_preview_grows_by_the_same_share(self) -> None:
        # 画質を落としたプレビューでも書き出しと同じ範囲が出る 設定は画面の画素なので、
        # 縮めずに見積もると広げ過ぎ、縮め過ぎると切れる
        full = source_canvas(text(text=WIDE), *SCREEN)
        half = source_canvas(text(text=WIDE), SCREEN[0] // 2, SCREEN[1] // 2, scale=(0.5, 0.5))
        assert abs(half[0] - full[0] / 2) <= 2
        assert half[1] == SCREEN[1] // 2
        assert_not_cut(drawn(text(text=WIDE), scale=0.5))

    def test_the_canvas_stops_at_the_limit(self) -> None:
        # 一辺は MAX_CANVAS まで 無限に大きな絵は作れない（1 枚で 256MB を超える）
        huge = text(text="あ" * 400, size=512)
        width, height = source_canvas(huge, *SCREEN)
        assert width <= MAX_CANVAS and height <= MAX_CANVAS
        assert width > SCREEN[0]

    def test_revealing_text_keeps_one_size(self) -> None:
        # 文字送りの途中で絵の大きさが変わると、フレームごとに大きさの違う絵を作り直す
        # 出ている字は全体の頭なので、全部を出した大きさに収まる
        sizes = {source_canvas(text(text=WIDE, reveal=share), *SCREEN) for share in (10, 50, 100)}
        assert len(sizes) == 1

    def test_moving_text_grows_with_the_position(self) -> None:
        # 流れるテロップは位置のキーフレームで画面の外から入る 今のフレームの位置で広げる
        moving = text(pos_x=AnimatedValue(0.0, keyframes=(Keyframe(0, 0.0), Keyframe(30, 1500.0))))
        assert source_canvas(moving, *SCREEN, frame=0) == SCREEN
        assert source_canvas(moving, *SCREEN, frame=30)[0] > SCREEN[0]

    def test_a_long_timer_grows(self) -> None:
        # タイマーは時刻から文字を作る 文字の長さで広げないと、長い書式が切れる
        timer = text(text="", timer_format="\\" + "\\".join("あ" * 40), size=64)
        assert source_canvas(timer, *SCREEN, fps=30.0)[0] > SCREEN[0]


class TestCache:
    def _made(self, width: int, height: int) -> _Made:
        return _Made(None, np.zeros((height, width, 4), dtype=np.uint8), None)

    def test_screen_sized_pictures_keep_the_old_count(self) -> None:
        # 画面の大きさの絵は今までどおり 48 枚覚える 減らすと字幕の多い作品で作り直しが増える
        cache: OrderedDict[object, _Made] = OrderedDict()
        screen_bytes = 64 * 36 * 4
        for index in range(MAX_GENERATED_CACHE + 5):
            made = self._made(64, 36)
            _make_room(cache, made.image, screen_bytes)  # type: ignore[arg-type]
            cache[index] = made
        assert len(cache) == MAX_GENERATED_CACHE
        assert next(iter(cache)) == 5

    def test_a_large_picture_counts_as_several(self) -> None:
        # 一辺 8192 まで広げた絵は画面の何十倍にもなる 1 枚 1 つで数えると、長いテロップを
        # 並べただけで数 GB を抱える
        cache: OrderedDict[object, _Made] = OrderedDict()
        screen_bytes = 64 * 36 * 4
        for index in range(MAX_GENERATED_CACHE):
            made = self._made(64, 36)
            _make_room(cache, made.image, screen_bytes)  # type: ignore[arg-type]
            cache[index] = made
        big = self._made(64 * 4, 36 * 2)
        _make_room(cache, big.image, screen_bytes)  # type: ignore[arg-type]
        cache["big"] = big
        assert len(cache) == MAX_GENERATED_CACHE - 8 + 1
        used = sum(max(1, -(-made.image.nbytes // screen_bytes)) for made in cache.values())
        assert used <= MAX_GENERATED_CACHE


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


#: GL の試験は小さい画面で描く 大きさ 32 の全角 40 文字で 1000 画素を超え、640 に収まらない
SMALL = (640, 360)


def _placed(clip: Clip, **values: float) -> Clip:
    clip = with_fixed_items(clip, picture=True)
    effects: list[Effect] = []
    for effect in clip.effects:
        if effect.fixed and effect.kind == TRANSFORM_EFFECT_KIND:
            for name, value in values.items():
                effect = effect.with_param(name, AnimatedValue(float(value)))
        effects.append(effect)
    return replace(clip, effects=tuple(effects))


def _project(source: GeneratedSource, *effects: Effect, **placement: float) -> Project:
    project = Project.create(
        ProjectSettings(width=SMALL[0], height=SMALL[1], frame_rate=FrameRate(30))
    )
    track = Track(kind=TrackKind.VIDEO, name="V1")
    clip = _placed(Clip(timeline_start=0, duration=10, source=source), **placement)
    clip = replace(clip, effects=(*effects, *clip.effects))
    for command in (AddTrack(track), AddClip(track.id, clip)):
        project = command.apply(project)
    return project


def _render(project: Project, context: OffscreenGLContext, divisor: int = 1) -> np.ndarray:
    renderer = FrameRenderer(project, context=context, quality=RenderQuality(divisor))
    try:
        return renderer.render(0)
    finally:
        renderer.close()


def _lit(image: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(image[..., :3].max(axis=2) > 100)
    assert len(xs), "何も描かれていない"
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def small(**params: ParamValue | float | str | bool) -> GeneratedSource:
    return text(**{"text": WIDE, "size": 32.0, **params})


class TestRenderer:
    def test_moving_a_long_line_shows_the_rest(self, gl_context: OffscreenGLContext) -> None:
        # 約 1040 画素の行（右端は画面の右端より約 200 外）を左へ 400 ずらす 画面の大きさで
        # 描いていたころは、画面の右端で切れた字がそのまま左へ動き、右端が 240 で止まっていた
        image = _render(_project(small(), pos_x=-400), gl_context)
        assert _lit(image)[2] > SMALL[0] - 400 + 160

    def test_moving_text_back_from_outside(self, gl_context: OffscreenGLContext) -> None:
        # 位置で画面の外へ置いた字を、配置で画面へ戻す 画面の大きさの絵には描かれず何も出なかった
        image = _render(_project(small(text="あ" * 4, pos_x=600), pos_x=-600), gl_context)
        left, _top, right, _bottom = _lit(image)
        assert abs((left + right) / 2 - SMALL[0] / 2) < 8

    @pytest.mark.parametrize(
        "effects",
        [
            (),
            ("blur",),
        ],
    )
    def test_the_screen_looks_the_same_without_moving(
        self,
        gl_context: OffscreenGLContext,
        monkeypatch: pytest.MonkeyPatch,
        effects: tuple[str, ...],
    ) -> None:
        # 動かさなければ、画面に出る絵は広げる前と同じ（画面の外は元々見えない）
        # ぼかしは画面の端の外にあった字がにじんで入ってくるので、端から離れた所だけを比べる
        added = tuple(registry.require(kind).create() for kind in effects)
        project = _project(small(border_width=3, shadow_x=4, shadow_y=-4, shadow_blur=2), *added)
        grown = _render(project, gl_context).astype(np.int16)
        monkeypatch.setattr(
            renderer_module, "source_canvas", lambda source, width, height, **_: (width, height)
        )
        before = _render(project, gl_context).astype(np.int16)
        inner = (slice(40, -40), slice(40, -40)) if effects else (slice(None), slice(None))
        assert_same(grown[inner], before[inner])

    def test_the_preview_shows_the_same_range(self, gl_context: OffscreenGLContext) -> None:
        # 画質を落としたプレビューでも、動かして見えてくる範囲は書き出しと同じ
        project = _project(small(), pos_x=-400)
        full = _lit(_render(project, gl_context))
        half = _lit(_render(project, gl_context, divisor=2))
        assert abs(half[2] * 2 - full[2]) <= 4
        assert abs(half[0] * 2 - full[0]) <= 4

    def test_the_outline_covers_the_whole_line(self, gl_context: OffscreenGLContext) -> None:
        # プレビューで掴む外枠も、はみ出した字まで入れる 画面の大きさの絵の範囲で出すと、
        # 外枠が画面の幅で止まり、掴んだ所と字が合わない
        project = _project(small())
        clip = project.timeline.tracks[0].clips[0]
        renderer = FrameRenderer(project, context=gl_context)
        try:
            extent = renderer.object_extent(clip, 0)
        finally:
            renderer.close()
        assert extent is not None
        (left, _top, right, _bottom), (width, _height) = extent
        assert width > SMALL[0]
        assert right - left > SMALL[0] * 1.5

    def test_an_aviutl_frame_stays_the_container(self, gl_context: OffscreenGLContext) -> None:
        # AviUtl2 の組み方の入れ物は文字の枠（#64） 画面より大きい絵の道で色の付いた範囲だけに
        # すると、はみ出す長さの字だけ効果の基準が字の形に変わる
        project = _project(small(layout="aviutl"))
        clip = project.timeline.tracks[0].clips[0]
        renderer = FrameRenderer(project, context=gl_context)
        try:
            extent = renderer.object_extent(clip, 0)
            image = renderer._generate(clip, 0, project.rate)
        finally:
            renderer.close()
        assert extent is not None and image is not None
        box = extent[0]
        left, top, right, bottom = ink(image)
        # 枠は字の形より左右にも上下にも広い（字の左右の余白とベースラインの下）
        assert box[0] < left and box[2] >= right - 1
        assert box[1] <= top and box[3] >= bottom
