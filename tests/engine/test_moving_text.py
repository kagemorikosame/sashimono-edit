"""位置だけが動くテキストの絵を使い回す（#295）

位置のキーフレームで動く字幕は、字の形も縁も同じなのに、位置が変わるたびに字を組み直して
塗っていた 速さは時間ではなく**字の絵を作った回数**で見る 時間は機械と並びで揺れるが、
回数は作りが同じなら同じになる

絵は、同じ位置（1/4 画素の格子へ丸めた位置）に止めた字を今までの道で描いた物と比べる
置き直した字が、その位置で描き直した字と同じなら、端数の位置でもにじまない
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.core.commands import AddClip, AddTrack, ParamPath, SetParam
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    ParamValue,
    Project,
    ProjectSettings,
    Stroke,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, RenderQuality
from sashimono.engine.render import renderer as renderer_module
from sashimono.engine.render.moving_text import SUBPIXEL_STEPS, Sprite, SpriteShelf, spot_of
from sashimono.engine.sources import Frame, render_source_framed, text_sprite_canvas

SIZE = (640, 360)
LENGTH = 40
WHITE = (1.0, 1.0, 1.0, 1.0)
RED = (1.0, 0.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0, 1.0)


class _WindowWatch(QObject):
    """試験の間に開いた窓を覚えて、すぐ閉じる"""

    def __init__(self) -> None:
        super().__init__()
        self.opened: list[str] = []

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 Qt の名前
        if event.type() != QEvent.Type.Show:
            return False
        if isinstance(watched, QWindow) or (isinstance(watched, QWidget) and watched.isWindow()):
            self.opened.append(type(watched).__name__)
            watched.close()
        return False


@pytest.fixture(autouse=True)
def no_windows(qt_application: QApplication) -> Iterator[None]:
    # 描く試験は画面を出さない 窓が開くと、試験を走らせている人の画面に割り込み、
    # 閉じる人がいないと試験が止まる 開いたらその場で閉じ、試験を落とす
    watch = _WindowWatch()
    qt_application.installEventFilter(watch)
    try:
        yield
    finally:
        qt_application.removeEventFilter(watch)
    if watch.opened:
        pytest.fail(f"試験の間に窓が開いた: {', '.join(watch.opened)}")


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """レンダラが字の絵を作った記録（作った絵の大きさ） 回数で速さを見る"""
    calls: list[tuple[int, int]] = []

    def counted(
        source: GeneratedSource, width: int, height: int, **options: Any
    ) -> tuple[np.ndarray | None, Frame | None]:
        calls.append((width, height))
        return render_source_framed(source, width, height, **options)

    monkeypatch.setattr(renderer_module, "render_source_framed", counted)
    return calls


def moving(start: float, step: float) -> AnimatedValue:
    """1 コマに ``step`` 画素ずつ進む値"""
    return AnimatedValue(
        0.0, (Keyframe(0, start), Keyframe(LENGTH - 1, start + step * (LENGTH - 1)))
    )


def text(*strokes: Stroke, **params: ParamValue) -> GeneratedSource:
    base: dict[str, ParamValue] = {
        "text": "字あL",
        "size": AnimatedValue(64.0),
        "color": YELLOW,
        "border_width": AnimatedValue(5.0),
    }
    base.update(params)
    return GeneratedSource(kind="text", params=base, strokes=strokes)


def layer(width: float, colour: tuple[float, ...], **extra: ParamValue) -> Stroke:
    return Stroke(params={"width": AnimatedValue(width), "color": colour, **extra})


def project_of(source: GeneratedSource) -> tuple[Project, Clip]:
    project = Project.create(
        ProjectSettings(width=SIZE[0], height=SIZE[1], frame_rate=FrameRate(30))
    )
    track = Track(kind=TrackKind.VIDEO, name="V1")
    clip = Clip(0, LENGTH, source=source)
    for command in (AddTrack(track), AddClip(track.id, clip)):
        project = command.apply(project)
    return project, clip


def renderer_of(project: Project, context: OffscreenGLContext, divisor: int = 1) -> FrameRenderer:
    return FrameRenderer(project, context=context, quality=RenderQuality(divisor))


def snapped(value: float, scale: float) -> float:
    """``value``（画面の画素）を合成の画素の 1/4 の格子へ丸めた位置（画面の画素）"""
    return math.floor(value * scale * SUBPIXEL_STEPS + 0.5) / SUBPIXEL_STEPS / scale


def still_at(source: GeneratedSource, x: float, y: float) -> GeneratedSource:
    return replace(
        source, params={**source.params, "pos_x": AnimatedValue(x), "pos_y": AnimatedValue(y)}
    )


# --- 置く所 -------------------------------------------------------------------


class TestSpot:
    def test_whole_pixels_move_the_offset_and_keep_the_cell(self) -> None:
        # 整数の画素ぶんの動きを描き直しに回すと、1 画素ずつ流れるテロップで毎コマ字を作り直す
        first = spot_of((0.0, 0.0), (1.0, 1.0), (640, 360), (200, 100))
        later = spot_of((17.0, 5.0), (1.0, 1.0), (640, 360), (200, 100))
        assert first.cell == later.cell == (0, 0)
        assert first.offset == (220, 130)
        # 設定の Y は上が正 上へ 5 動かした字は 5 画素上（小さい側）に置く
        assert later.offset == (237, 125)

    def test_a_fraction_is_drawn_into_the_picture(self) -> None:
        # 端数を置く所で丸めると、0.3 画素ずつ動く字が 3 コマに 1 度だけ 1 画素跳ぶ
        spot = spot_of((10.25, -0.5), (1.0, 1.0), (640, 360), (200, 100))
        assert spot.cell == (1, 2)
        assert spot.shift == (0.25, 0.5)
        # 下へ 0.5 は、1 画素下へ置いて、字の絵の中で 0.5 上げて描く
        assert spot.offset == (230, 131)

    def test_a_negative_fraction_counts_down_from_the_pixel_below(self) -> None:
        # 負の位置で端数の向きを取り違えると、左へ動く字が 1 画素ぶん右へ置かれる
        spot = spot_of((-0.25, 0.0), (1.0, 1.0), (640, 360), (200, 100))
        assert spot.cell == (3, 0)
        assert spot.offset == (219, 130)
        assert spot.shift == (0.75, 0.0)

    def test_halfway_rounds_up_the_same_way_everywhere(self) -> None:
        # ``round`` の偶数への丸めでは、0.125 と 0.375 の丸め方が逆になり、動く字の歩幅が揃わない
        assert spot_of((0.125, 0.0), (1.0, 1.0), (640, 360), (200, 100)).cell == (1, 0)
        assert spot_of((0.375, 0.0), (1.0, 1.0), (640, 360), (200, 100)).cell == (2, 0)

    def test_the_fraction_is_counted_in_canvas_pixels(self) -> None:
        # 画質を落としたプレビューで画面の画素の端数を使うと、格子が粗くなって字が跳ぶ
        spot = spot_of((1.0, 0.0), (0.5, 0.5), (320, 180), (100, 50))
        assert spot.cell == (2, 0)
        assert spot.shift == (1.0, 0.0)
        assert spot.offset == (110, 65)


class TestSpriteCanvas:
    @pytest.mark.parametrize("canvas", [(640, 360), (213, 120), (641, 361)])
    def test_the_parity_follows_the_canvas(self, canvas: tuple[int, int]) -> None:
        # 偶奇が違うと、真ん中どうしを重ねたときに半画素ずれて、置き直した字がにじむ
        size = text_sprite_canvas(text(), *canvas)
        assert size is not None
        assert (size[0] - canvas[0]) % 2 == 0
        assert (size[1] - canvas[1]) % 2 == 0

    def test_an_empty_text_has_no_picture(self) -> None:
        # 字が無いのに大きさを返すと、何も無い絵を覚えて置くことになる
        assert text_sprite_canvas(text(text=""), *SIZE) is None

    def test_a_wider_border_widens_the_picture(self) -> None:
        # 縁取りの太さを見積もりから外すと、太い縁が字の絵の端で切れる
        thin = text_sprite_canvas(text(), *SIZE)
        thick = text_sprite_canvas(text(border_width=AnimatedValue(30.0)), *SIZE)
        assert thin is not None and thick is not None
        assert thick[0] >= thin[0] + 50


class TestPlaced:
    def test_reusing_clears_the_old_place(self) -> None:
        # 前のコマの絵を消さずに使い回すと、動いた字の後ろに前の字が残る
        sprite = Sprite(np.full((4, 4, 4), 255, np.uint8), None, (0, 0, 4, 4))
        first, *_ = sprite.placed((2, 2), (16, 8))
        second, _, box, known = sprite.placed((10, 3), (16, 8), (first, (2, 2, 4, 4)))
        assert second is first
        assert int(second[..., 3].sum()) == 16 * 255
        assert known and box == (10, 3, 14, 7)

    def test_a_cut_sprite_leaves_the_box_to_be_counted(self) -> None:
        # 端で切れた字に字の絵の範囲を渡すと、画面の外まで入れ物にしてエフェクトがずれる
        sprite = Sprite(np.full((4, 4, 4), 255, np.uint8), None, (0, 0, 4, 4))
        image, _, box, known = sprite.placed((14, -2), (16, 8))
        assert not known and box is None
        assert int(image[..., 3].sum()) == 2 * 2 * 255

    def test_the_frame_moves_with_the_sprite(self) -> None:
        # 文字の枠を置く所へずらさないと、AviUtl2 の組み方の字の入れ物が元の所に残る
        sprite = Sprite(np.zeros((4, 4, 4), np.uint8), (0.5, 1.0, 3.5, 3.0), None)
        _, framed, _, known = sprite.placed((5, 6), (16, 16))
        assert framed == (5.5, 7.0, 8.5, 9.0)
        assert known


class TestShelf:
    def _sprite(self, side: int) -> Sprite:
        return Sprite(np.zeros((side, side, 4), np.uint8), None, None)

    def test_a_new_key_drops_the_old_sprites(self) -> None:
        # 字の色を変えた後に前の鍵の絵を返すと、色を変えても前の色の字が出る
        shelf = SpriteShelf()
        clip = Clip(0, 10).id
        shelf.put(clip, "黄", (0, 0), self._sprite(2), budget=10**6)
        assert shelf.get(clip, "赤", (0, 0)) is None
        shelf.put(clip, "赤", (1, 0), self._sprite(2), budget=10**6)
        assert shelf.get(clip, "黄", (0, 0)) is None
        assert len(shelf) == 1

    def test_the_oldest_clip_goes_first_when_full(self) -> None:
        # 量を数えないと、長いタイムラインを流すほど字の絵が溜まってメモリを食い潰す
        shelf = SpriteShelf()
        first, second = Clip(0, 10).id, Clip(0, 10).id
        # 4 x 4 の RGBA は 64 バイト 2 本で 128 バイトは上限の 100 を超える
        shelf.put(first, "a", (0, 0), self._sprite(4), budget=100)
        shelf.put(second, "b", (0, 0), self._sprite(4), budget=100)
        assert shelf.get(first, "a", (0, 0)) is None
        assert shelf.get(second, "b", (0, 0)) is not None

    def test_one_clip_over_the_limit_keeps_the_newest(self) -> None:
        # 1 本で上限を超えたときに今描いた絵まで捨てると、毎コマ描き直しになる
        shelf = SpriteShelf()
        clip = Clip(0, 10).id
        for cell in range(4):
            shelf.put(clip, "a", (cell, 0), self._sprite(4), budget=130)
        assert shelf.get(clip, "a", (3, 0)) is not None
        assert len(shelf) == 2


# --- 作る回数 -----------------------------------------------------------------


def test_whole_pixel_motion_draws_the_text_once(
    gl_context: OffscreenGLContext, made: list[tuple[int, int]]
) -> None:
    # 位置が変わるたびに描き直すと、動く字幕 20 本で 1 コマ 0.4 秒かかり再生が止まる
    source = text(pos_x=moving(-100.0, 3.0), pos_y=moving(20.0, -2.0))
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for frame in range(LENGTH):
            renderer.render(frame)
    finally:
        renderer.close()
    assert len(made) == 1


@pytest.mark.parametrize(
    ("step_x", "step_y", "most"),
    [(0.37, 0.0, SUBPIXEL_STEPS), (0.37, 0.61, SUBPIXEL_STEPS**2)],
    ids=["横だけ", "斜め"],
)
def test_a_fraction_draws_once_per_cell(
    gl_context: OffscreenGLContext,
    made: list[tuple[int, int]],
    step_x: float,
    step_y: float,
    most: int,
) -> None:
    # 端数の位置ごとに描き直し続けると、ゆっくり流れる字幕でいつまでも軽くならない
    source = text(pos_x=moving(-50.0, step_x), pos_y=moving(10.0, step_y))
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for _ in range(2):
            for frame in range(LENGTH):
                renderer.render(frame)
    finally:
        renderer.close()
    assert 1 < len(made) <= most


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("color", RED),
        ("size", AnimatedValue(80.0)),
        ("text", "別の字"),
        ("border_width", AnimatedValue(9.0)),
    ],
    ids=["色", "大きさ", "文字", "縁の太さ"],
)
def test_changing_the_look_draws_again(
    gl_context: OffscreenGLContext, made: list[tuple[int, int]], name: str, value: ParamValue
) -> None:
    # 置き直す絵の鍵から字の設定が漏れると、色や文字を変えても前の字が動き続ける
    source = text(pos_x=moving(-100.0, 3.0))
    project, clip = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        before = renderer.render(1)
        changed = SetParam(ParamPath.of_source(clip.id, name), value).apply(project)
        renderer.set_project(changed)
        after = renderer.render(2)
    finally:
        renderer.close()
    assert len(made) == 2
    assert not np.array_equal(before, after)


def test_a_changing_size_still_draws_every_frame(
    gl_context: OffscreenGLContext, made: list[tuple[int, int]]
) -> None:
    # 大きさが動く字を使い回すと、拡大していく字幕が最初の大きさのまま動く
    source = text(pos_x=moving(-100.0, 3.0), size=moving(40.0, 2.0))
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for frame in range(6):
            renderer.render(frame)
    finally:
        renderer.close()
    assert len(made) == 6


def test_stroke_layers_are_drawn_once_too(
    gl_context: OffscreenGLContext, made: list[tuple[int, int]]
) -> None:
    # 縁取りの層を持つ字だけ使い回さないと、層を足した字幕ほど重くなる（層 3 つで 1 コマ 1.3 秒）
    blur = registry.require("blur").create(radius=3.0)
    source = text(
        layer(4.0, RED),
        replace(layer(10.0, WHITE), effects=(blur,)),
        pos_x=moving(-100.0, 2.0),
    )
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for frame in range(LENGTH):
            renderer.render(frame)
    finally:
        renderer.close()
    assert len(made) == 1


@pytest.mark.parametrize(
    ("strokes", "params"),
    [
        (
            (
                Stroke(
                    params={
                        "width": AnimatedValue(6.0),
                        "color": RED,
                        "opacity": moving(20.0, 15.0),
                    }
                ),
            ),
            {},
        ),
        ((), {"reveal": moving(10.0, 20.0)}),
    ],
    ids=["層の不透明度", "文字送り"],
)
def test_a_look_that_moves_without_growing_draws_every_frame(
    gl_context: OffscreenGLContext,
    made: list[tuple[int, int]],
    strokes: tuple[Stroke, ...],
    params: dict[str, ParamValue],
) -> None:
    # 字の絵の大きさが変わらないまま見た目だけが動く字を使い回すと、縁が濃くなる演出や
    # 文字送りが、流れる字幕でだけ最初のコマのまま止まる
    source = text(*strokes, pos_x=moving(-100.0, 2.0), **params)
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for frame in range(5):
            renderer.render(frame)
    finally:
        renderer.close()
    assert len(made) == 5


# --- 絵 -----------------------------------------------------------------------


_LOOKS: dict[str, GeneratedSource] = {
    "前からの縁": text(),
    "縁取りの層 3 つ": text(
        layer(3.0, RED),
        layer(8.0, WHITE),
        layer(2.0, RED, position="inside"),
        border_width=AnimatedValue(0.0),
    ),
    "ぼかした影": text(
        shadow_x=AnimatedValue(6.0),
        shadow_y=AnimatedValue(-5.0),
        shadow_blur=AnimatedValue(3.0),
    ),
    "AviUtl2 の組み方": text(layout="aviutl", text="AviUtl 字"),
    "縦書き": text(vertical=True),
    # 塗り・縁取りの層・影のグラデーション（#312） 字の枠を基準に塗るので、位置 0 で描いて
    # 置き直しても、その位置で描き直した字と同じ色の並びになる 画面を基準に塗ると色がずれる
    # 縁取りの層は入れない 丸い角の揺れ（単色の層でも塊 13）に、グラデーションの丸めの
    # 1〜4 の差が隣り合うと塊が上限を超え、塗り方と関係の無い所で落ちる 層だけのグラデーションは
    # 下の「グラデーションの層」で見る
    "グラデーション": text(
        border_width=AnimatedValue(0.0),
        fill_type="radial",
        fill_gradient=(0.0, *YELLOW, 1.0, *RED),
        shadow_x=AnimatedValue(6.0),
        shadow_y=AnimatedValue(-5.0),
        shadow_blur=AnimatedValue(3.0),
        shadow_fill_type="linear",
        shadow_fill_gradient=(0.0, *RED, 1.0, *YELLOW),
    ),
    "グラデーションの層": text(
        layer(3.0, RED, fill_type="linear", fill_gradient=(0.0, *RED, 1.0, *WHITE)),
        border_width=AnimatedValue(0.0),
    ),
}


#: 違いとして数える画素の差 これ以下は塗りの端の丸めの揺れ
_VISIBLE = 4

#: 違う画素が繋がった塊の大きさの上限（画素） 縁の丸い角を太らせる Qt の計算は座標の桁で
#: 揺れ、今までの描き方でも字を整数の画素だけずらすと、丸い角の弧の数画素が変わる
#: （等倍・1/2・1/3 で測ると塊は多くて 10 画素） 端数の扱いや置く所を間違えると、違いが
#: 字の輪郭に沿って一続きになり、塊は何百画素にもなる
_PATCH = 16


def _drawn_still(source: GeneratedSource, context: OffscreenGLContext, divisor: int) -> np.ndarray:
    """止めた字を今までの道（位置の止まった字は、置き直さずに描く）で描いた絵"""
    project, _ = project_of(source)
    renderer = renderer_of(project, context, divisor)
    try:
        return renderer.render(0)
    finally:
        renderer.close()


def _largest_patch(image: np.ndarray, other: np.ndarray) -> int:
    """違う画素が、斜めも含めて繋がった塊のうち一番大きい物の画素の数"""
    difference = np.abs(image.astype(np.int32) - other.astype(np.int32)).max(axis=2)
    rows, columns = np.nonzero(difference > _VISIBLE)
    left = {(int(row), int(column)) for row, column in zip(rows, columns, strict=True)}
    largest = 0
    while left:
        stack = [left.pop()]
        size = 0
        while stack:
            row, column = stack.pop()
            size += 1
            for near in ((row + dy, column + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                if near in left:
                    left.remove(near)
                    stack.append(near)
        largest = max(largest, size)
    return largest


@pytest.mark.parametrize("divisor", [1, 2, 3], ids=["等倍", "1/2", "1/3"])
@pytest.mark.parametrize("look", list(_LOOKS), ids=list(_LOOKS))
def test_each_frame_matches_the_text_drawn_there(
    gl_context: OffscreenGLContext, look: str, divisor: int
) -> None:
    # 置き直した字が、その位置で描き直した字と違うと、動く字幕だけ縁がにじむか半画素ずれる
    # 1/3 は合成の幅が奇数（213）になり、字の絵と偶奇が食い違うと半画素ずれる
    # 同じレンダラで続けて描くので、前のコマの字を消し忘れると後ろに字が残って落ちる
    source = replace(
        _LOOKS[look],
        params={**_LOOKS[look].params, "pos_x": moving(-80.0, 1.37), "pos_y": moving(15.0, -0.61)},
    )
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context, divisor)
    scale = renderer.compositor.width / SIZE[0], renderer.compositor.height / SIZE[1]
    try:
        for frame in (0, 1, 2, 3, 9, 17):
            moved = renderer.render(frame)
            x = snapped(-80.0 + 1.37 * frame, scale[0])
            y = snapped(15.0 - 0.61 * frame, scale[1])
            expected = _drawn_still(still_at(source, x, y), gl_context, divisor)
            assert _largest_patch(moved, expected) <= _PATCH, frame
    finally:
        renderer.close()


def test_a_layer_effect_matches_the_text_drawn_there(gl_context: OffscreenGLContext) -> None:
    # 層のエフェクト（GPU）を掛けた字も、置き直した絵と描き直した絵が同じであること
    # 層の面を字の絵の端で切ると、ぼかした縁が動く字幕でだけ四角く切れる
    blur = registry.require("blur").create(radius=6.0)
    source = text(
        replace(layer(8.0, RED), effects=(blur,)),
        border_width=AnimatedValue(0.0),
        pos_x=moving(-60.0, 2.25),
    )
    project, _ = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for frame in (0, 5):
            moved = renderer.render(frame)
            expected = _drawn_still(still_at(source, -60.0 + 2.25 * frame, 0.0), gl_context, 1)
            # 丸い角の揺れ（:data:`_PATCH`）は層のぼかしで薄く広がる 塊ではなく差の大きさで見る
            # （測ると 8 まで 層の面を切ったり置く所を間違えたりすると、縁の色そのものの差になる）
            difference = np.abs(moved.astype(np.int32) - expected.astype(np.int32))
            assert int(difference.max()) <= 12, frame
    finally:
        renderer.close()


def _centre_x(image: np.ndarray) -> float:
    weight = image[..., 3].astype(np.float64).sum(axis=0)
    return float((weight * np.arange(weight.size)).sum() / weight.sum())


def test_the_snapped_position_stays_within_an_eighth_of_a_pixel() -> None:
    # 格子へ丸めるずれが 1/8 画素を超えると、ゆっくり流れる字が揺れて見える
    for x in (0.1, 0.13, 0.37, 0.5, 0.88, 3.9):
        source = still_at(text(), x, 0.0)
        exact, _ = render_source_framed(source, *SIZE)
        rounded, _ = render_source_framed(still_at(text(), snapped(x, 1.0), 0.0), *SIZE)
        assert exact is not None and rounded is not None
        assert abs(_centre_x(exact) - _centre_x(rounded)) <= 1 / 8 + 0.02, x


def test_the_outline_only_pass_does_not_keep_an_unbaked_layer(
    gl_context: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 外枠を出す所（:meth:`object_extent`）は GL の外から呼ばれ、層のエフェクト（GPU）を
    # 掛けられない 掛けに行くと、画面のスレッドから GL を触って落ちる 掛けずに描いた絵を
    # 覚えると、その端数の位置へ来るたびに縁のぼかしが抜けた字が出る
    blur = registry.require("blur").create(radius=6.0)
    source = text(
        replace(layer(8.0, RED), effects=(blur,)),
        border_width=AnimatedValue(0.0),
        pos_x=moving(-60.0, 2.0),
    )
    project, clip = project_of(source)
    renderer = renderer_of(project, gl_context)
    fresh = renderer_of(project, gl_context)
    baking = renderer._stroke_baker

    def refuse(*_arguments: object) -> None:
        pytest.fail("外枠を出すだけの所で、層のエフェクトを GPU で掛けようとした")

    try:
        monkeypatch.setattr(renderer, "_stroke_baker", refuse)
        assert renderer.object_extent(clip, 3) is not None
        monkeypatch.setattr(renderer, "_stroke_baker", baking)
        drawn = renderer.render(3).astype(np.int32)
        expected = fresh.render(3).astype(np.int32)
    finally:
        renderer.close()
        fresh.close()
    assert np.abs(drawn - expected).max() <= 1


def test_the_known_box_matches_a_fresh_scan(gl_context: OffscreenGLContext) -> None:
    # 字の絵から求めた範囲が、置いた絵を数え直した範囲と違うと、エフェクトの入れ物がずれる
    source = text(pos_x=moving(-100.0, 3.0))
    project, clip = project_of(source)
    renderer = renderer_of(project, gl_context)
    try:
        for frame in (0, 7):
            renderer.render(frame)
            held = renderer._generated[clip.id]
            assert held.scanned
            assert held.box == renderer_module._content_box(held.image)
    finally:
        renderer.close()


def test_a_silent_waveform_does_not_drop_the_text_pictures(
    gl_context: OffscreenGLContext, tmp_path: Path
) -> None:
    # 読めない音の波形があると、編集のたびに音の波形の絵を捨てる 文字の絵の鍵の読む音の所に
    # 別の項目（縁取りの層・置いた所）が入ると、文字の絵まで捨てて、編集のたびに全部の字を
    # 描き直す
    project = Project.create(
        ProjectSettings(width=SIZE[0], height=SIZE[1], frame_rate=FrameRate(30))
    )
    silent = GeneratedSource(
        kind="shape",
        params={"shape": "waveform", "audio_path": str(tmp_path / "無い音.wav")},
    )
    clips = {
        "止まった字": Clip(0, LENGTH, source=text(layer(4.0, RED))),
        "動く字": Clip(0, LENGTH, source=text(pos_x=moving(-100.0, 3.0))),
        "音の無い波形": Clip(0, LENGTH, source=silent),
    }
    for name, clip in clips.items():
        track = Track(kind=TrackKind.VIDEO, name=name)
        for command in (AddTrack(track), AddClip(track.id, clip)):
            project = command.apply(project)
    renderer = renderer_of(project, gl_context)
    try:
        renderer.render(1)
        assert set(renderer._generated) == {clip.id for clip in clips.values()}
        renderer.set_project(project)
        kept = set(renderer._generated)
    finally:
        renderer.close()
    assert kept == {clips["止まった字"].id, clips["動く字"].id}
