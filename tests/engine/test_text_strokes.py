"""テキストの縁取りの層（#272）と、層に掛けるエフェクト（#273）が絵になるか

字の形は書体で変わるので、画素の位置を決め打ちにせず、前からの縁取り（1 組の項目）で
描いた絵から帯を切り出して比べる 太さ 4 の縁の帯は「字の外で、太さ 4 の縁が塗る所」、
太さ 12 の帯は「太さ 12 の縁が塗り、太さ 4 の縁が塗らない所」
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddTrack, adopted_source
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Project,
    ProjectSettings,
    Stroke,
    StrokeId,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.sources import render_source_framed, source_canvas

SIZE = (360, 200)
WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0, 1.0)


def text(*strokes: Stroke, **params: object) -> GeneratedSource:
    base: dict[str, object] = {
        "text": "字あL",
        "size": AnimatedValue(110.0),
        "color": YELLOW,
    }
    base.update(params)
    return GeneratedSource(kind="text", params=base, strokes=strokes)  # type: ignore[arg-type]


def layer(width: float, colour: tuple[float, ...], **extra: object) -> Stroke:
    params: dict[str, object] = {"width": AnimatedValue(width), "color": colour, **extra}
    return Stroke(params=params)  # type: ignore[arg-type]


def draw(
    source: GeneratedSource,
    *,
    scale: float = 1.0,
    baker: object = None,
) -> np.ndarray:
    width, height = round(SIZE[0] * scale), round(SIZE[1] * scale)
    image, _ = render_source_framed(
        source,
        width,
        height,
        scale=(scale, scale),
        stroke_effects=baker,  # type: ignore[arg-type]
    )
    assert image is not None
    return image


def alpha(image: np.ndarray) -> np.ndarray:
    return image[..., 3].astype(np.int32)


def bands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """（字の塗り, 太さ 4 の帯, 太さ 4 から 12 の帯） どれも縁の滑らかな所を外した芯だけ"""
    plain = alpha(draw(text()))
    thin = alpha(draw(text(border_width=AnimatedValue(4.0))))
    thick = alpha(draw(text(border_width=AnimatedValue(12.0))))
    fill = plain == 255
    inner = (thin == 255) & (plain == 0)
    outer = (thick == 255) & (thin == 0)
    assert fill.sum() > 500 and inner.sum() > 200 and outer.sum() > 500
    return fill, inner, outer


def mostly(image: np.ndarray, where: np.ndarray, colour: tuple[float, ...]) -> float:
    """``where`` の画素のうち、色が ``colour`` に近い物の割合"""
    target = np.array([round(c * 255) for c in colour[:3]])
    near = np.abs(image[..., :3].astype(np.int32) - target).max(axis=2) <= 8
    return float((near & where).sum() / max(1, where.sum()))


class TestALegacyBorderLooksTheSame:
    """縁取りの層を持たない前からのデータは、前と同じ絵（層へ移しても同じ絵）"""

    @pytest.mark.parametrize(
        "params",
        [
            {"border_width": AnimatedValue(6.0), "border_color": RED},
            {
                "border_width": AnimatedValue(9.5),
                "border_color": (0.0, 0.2, 1.0, 0.7),
                "shadow_x": AnimatedValue(8.0),
                "shadow_y": AnimatedValue(-6.0),
                "shadow_blur": AnimatedValue(4.0),
            },
            {"border_width": AnimatedValue(5.0), "vertical": True},
            {"border_width": AnimatedValue(7.0), "layout": "aviutl", "bold": True},
            {"border_width": AnimatedValue(6.0), "color": (1.0, 1.0, 1.0, 0.5)},
        ],
        ids=["縁", "縁と影", "縦書き", "AviUtl2 の組み方", "半透明の字"],
    )
    @pytest.mark.parametrize("scale", [1.0, 0.5])
    def test_moving_the_border_into_a_layer_changes_no_pixel(
        self, params: dict[str, object], scale: float
    ) -> None:
        # 前からの縁を層へ移すだけで絵が変わると、開いて縁を触っただけの作品の見た目が変わる
        # 前からの縁を 1 つ目の層へ移す（設定パネルで縁を触る・層を足す）だけで絵が 1 画素でも
        # 変わると、開いて触っただけの作品の見た目が変わる
        legacy = text(**params)
        adopted = adopted_source(legacy, StrokeId("移した縁"))
        assert adopted.strokes and adopted.params["border_width"] == AnimatedValue(0.0)
        assert np.array_equal(draw(legacy, scale=scale), draw(adopted, scale=scale))

    def test_the_old_border_is_round_and_under_the_letter(self) -> None:
        # 前からの縁は輪郭をまたいで太さの 2 倍で引き、内側の半分を字の塗りで隠す 角は丸
        # 塗りの上に描くように変えると、半透明の字の中に縁の色がそのまま出て、前の作品の
        # 半透明の字幕の見た目が変わる
        fill, inner, _ = bands()
        half = (1.0, 1.0, 1.0, 0.5)
        image = draw(text(border_width=AnimatedValue(4.0), border_color=RED, color=half))
        # 字の芯（縁の届かない所）は半透明の白、縁の内側半分は赤の上に白を重ねた色
        reds = image[..., 0].astype(np.int32)
        greens = image[..., 1].astype(np.int32)
        under = fill & (alpha(image) == 255) & (reds > 200)
        assert under.sum() > 50
        assert int(np.median(greens[under])) > 100
        rounded = alpha(draw(text(border_width=AnimatedValue(14.0))))
        assert np.array_equal(rounded > 0, alpha(draw(text(layer(14.0, RED)))) > 0)
        assert mostly(image, inner, RED) > 0.95

    def test_a_text_without_layers_still_draws_the_old_border(self) -> None:
        # 層を持たない字が前からの項目を読まなくなると、前の作品の縁が全部消える
        bordered = draw(text(border_width=AnimatedValue(6.0), border_color=RED))
        assert alpha(bordered).sum() > alpha(draw(text())).sum()

    def test_a_layered_text_ignores_the_old_fields(self) -> None:
        # 層を持つ字で前からの縁まで描くと、層を全部隠しても縁が残る
        layered = text(replace(layer(6.0, RED), enabled=False), border_width=AnimatedValue(10.0))
        assert np.array_equal(draw(layered), draw(text()))


class TestTwoOutsideLayers:
    def test_the_first_layer_is_on_top_and_widths_count_from_the_glyph(self) -> None:
        # 内側の白（太さ 4）と外側の黒（太さ 12） 太さを字の輪郭から数えずに前の層の外から
        # 数えると黒が 16 まで出て、並びを無視すると白が黒の下に隠れる
        fill, inner, outer = bands()
        image = draw(text(layer(4.0, WHITE), layer(12.0, BLACK)))
        assert mostly(image, inner, WHITE) > 0.95
        assert mostly(image, outer, BLACK) > 0.95
        assert mostly(image, fill, YELLOW) > 0.95
        # 一番外の縁は太さ 12 の縁と同じ所で終わる
        thick = alpha(draw(text(border_width=AnimatedValue(12.0))))
        assert np.array_equal(alpha(image) > 0, thick > 0)

    def test_swapping_the_order_hides_the_thin_layer(self) -> None:
        # 並びの頭が一番上 太い黒を上にすると細い白はその下に隠れる
        _, inner, outer = bands()
        image = draw(text(layer(12.0, BLACK), layer(4.0, WHITE)))
        assert mostly(image, inner, BLACK) > 0.95
        assert mostly(image, inner, WHITE) < 0.01
        assert mostly(image, outer, BLACK) > 0.95

    def test_a_hidden_layer_draws_nothing(self) -> None:
        # 隠した層が描かれると、隠したつもりの縁がプレビューにも書き出しにも残る
        shown = draw(text(layer(4.0, WHITE)))
        hidden = draw(text(layer(4.0, WHITE), replace(layer(12.0, BLACK), enabled=False)))
        assert np.array_equal(shown, hidden)

    def test_the_opacity_of_a_layer_thins_only_that_layer(self) -> None:
        # 層の不透明度は層の色だけに掛かる 字の塗りは不透明のまま
        fill, inner, _ = bands()
        image = draw(text(layer(4.0, BLACK, opacity=AnimatedValue(50.0))))
        values = alpha(image)[inner]
        assert abs(float(np.median(values)) - 128.0) <= 2.0
        assert (alpha(image)[fill] == 255).all()


class TestPositions:
    def test_an_inside_layer_stays_within_the_letter(self) -> None:
        # 内側の線は字の外へ出ない 字の塗りの上に描くので見える（下に描くと塗りに隠れる）
        fill, _, _ = bands()
        plain = alpha(draw(text()))
        # 字の線は 10 画素ほどしかないので、細い線で真ん中に塗りを残す
        image = draw(text(layer(2.0, RED, position="inside")))
        assert np.array_equal(alpha(image) > 0, plain > 0)
        assert mostly(image, fill, RED) > 0.05
        assert mostly(image, fill, YELLOW) > 0.05

    def test_a_centre_layer_straddles_the_outline(self) -> None:
        # 中央の線は輪郭をまたいで半分ずつ 外へは太さの半分だけ出て、字の上に重なる
        fill, _, _ = bands()
        half = alpha(draw(text(border_width=AnimatedValue(5.0))))
        image = draw(text(layer(10.0, RED, position="center")))
        assert np.array_equal(alpha(image) > 0, half > 0)
        assert mostly(image, fill, RED) > 0.05
        # 外側の線は字の下 字の塗りの所に縁の色は出ない
        outside = draw(text(layer(10.0, RED)))
        assert mostly(outside, fill, RED) == 0.0

    def test_the_join_changes_the_corners(self) -> None:
        # 角の形を読まずに丸で描くと、角・面取りを選んでも絵が変わらない
        rounded = alpha(draw(text(layer(14.0, RED))))
        mitred = alpha(draw(text(layer(14.0, RED, join="miter"))))
        bevelled = alpha(draw(text(layer(14.0, RED, join="bevel"))))
        assert (mitred > 0).sum() > (rounded > 0).sum() > (bevelled > 0).sum()


class _Painter:
    """層のエフェクトの掛け手の代わり 受け取った絵の色の付いた所を緑で塗る"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

    def __call__(self, image: np.ndarray, effects: tuple[Effect, ...]) -> np.ndarray:
        self.calls.append((image.shape, tuple(effect.kind for effect in effects)))
        painted = image.copy()
        painted[image[..., 3] > 0, :3] = (0, 255, 0)
        return painted


def blur(radius: float = 6.0) -> Effect:
    return registry.require("blur").create(radius=radius)


class TestLayerEffects:
    def test_the_effect_touches_only_its_layer(self) -> None:
        # 層のエフェクトは層の絵だけに掛かる 字の塗りや他の層まで緑になれば、字全体に
        # 掛けているのと同じ
        fill, inner, outer = bands()
        painter = _Painter()
        source = text(layer(4.0, WHITE), replace(layer(12.0, BLACK), effects=(blur(),)))
        image = draw(source, baker=painter)
        assert [kinds for _, kinds in painter.calls] == [("blur",)]
        green = (0.0, 1.0, 0.0, 1.0)
        assert mostly(image, outer, green) > 0.95
        assert mostly(image, inner, WHITE) > 0.95
        assert mostly(image, fill, YELLOW) > 0.95

    def test_without_a_baker_the_layer_is_drawn_plainly(self) -> None:
        # GPU を持たない所（素材の見本）では掛けずに描く 層ごと消えると縁が無くなる
        plain = draw(text(layer(8.0, BLACK)))
        assert np.array_equal(draw(text(replace(layer(8.0, BLACK), effects=(blur(),)))), plain)

    @pytest.mark.parametrize(
        "effect",
        [
            replace(blur(), enabled=False),
            registry.require("noise").create(),
            Effect("無い種類"),
        ],
        ids=["切ってある", "時間で動く", "定義が無い"],
    )
    def test_effects_that_cannot_apply_are_skipped(self, effect: Effect) -> None:
        # 層に掛けられない物を渡すと、止まった字幕でも毎フレーム絵が変わったり落ちたりする
        painter = _Painter()
        source = text(replace(layer(8.0, BLACK), effects=(effect,)))
        assert np.array_equal(draw(source, baker=painter), draw(text(layer(8.0, BLACK))))
        assert painter.calls == []

    def test_a_blur_widens_the_picture(self) -> None:
        # ぼかしの広がりを絵の大きさの見積もりに入れないと、画面の端に寄せた字の縁が切れる
        near_edge = {"pos_x": AnimatedValue(150.0)}
        plain = source_canvas(text(layer(8.0, BLACK), **near_edge), *SIZE)
        blurred = source_canvas(
            text(replace(layer(8.0, BLACK), effects=(blur(60.0),)), **near_edge), *SIZE
        )
        assert blurred[0] >= plain[0] + 2 * 50

    def test_a_zoomed_shadow_widens_the_picture(self) -> None:
        # 影の拡大は画素の項目に出ない 見積もりに入れないと、画面の端に寄せた字の大きくした
        # 影が絵の端で切れる
        near_edge = {"pos_x": AnimatedValue(120.0)}

        def shadowed(zoom: float) -> tuple[int, int]:
            shadow = registry.require("shadow").create(zoom=zoom, blur=0)
            return source_canvas(
                text(replace(layer(8.0, BLACK), effects=(shadow,)), **near_edge), *SIZE
            )

        assert shadowed(300.0)[0] > shadowed(100.0)[0] + 200

    def test_the_layer_is_handed_over_around_the_border_only(self) -> None:
        # 層の絵は縁のある所の周りだけ 字の絵の全体を渡すと、字幕 1 本ごとに画面 1 枚ぶんを
        # GPU へ送って読み戻すことになる 周りにはエフェクトが広げる分を空けておく
        painter = _Painter()
        draw(text(replace(layer(8.0, BLACK), effects=(blur(20.0),))), baker=painter)
        ((shape, _),) = painter.calls
        thick = alpha(draw(text(border_width=AnimatedValue(8.0)))) > 0
        rows = np.flatnonzero(thick.any(axis=1))
        columns = np.flatnonzero(thick.any(axis=0))
        inked = (int(rows[-1] - rows[0] + 1), int(columns[-1] - columns[0] + 1))
        assert shape[0] < SIZE[1] or shape[1] < SIZE[0]
        assert shape[0] >= min(SIZE[1], inked[0] + 2 * 20)
        assert shape[1] >= min(SIZE[0], inked[1] + 2 * 20)

    def test_a_preview_at_half_quality_hands_a_half_sized_layer(self) -> None:
        # 画質を落としたプレビューは層の絵も縮めて描く（ぼかしの強さはレンダラが縮める）
        # 縮めずに渡すと、プレビューだけ層の縁が字からずれた大きさで重なる
        full, half = _Painter(), _Painter()
        source = text(replace(layer(8.0, BLACK), effects=(blur(),)))
        draw(source, baker=full)
        draw(source, scale=0.5, baker=half)
        ((full_shape, _),) = full.calls
        ((half_shape, _),) = half.calls
        assert abs(half_shape[0] - full_shape[0] / 2) <= 3
        assert abs(half_shape[1] - full_shape[1] / 2) <= 3


def test_a_preview_at_half_quality_is_the_export_shrunk() -> None:
    # 層の太さは画面の画素 縮めずに描くと、プレビューだけ縁が 2 倍の太さに出る
    source = text(layer(4.0, WHITE), layer(12.0, BLACK), layer(5.0, RED, position="inside"))
    full = draw(source).astype(np.float32)
    half = draw(source, scale=0.5).astype(np.float32)
    shrunk = full.reshape(SIZE[1] // 2, 2, SIZE[0] // 2, 2, 4).mean(axis=(1, 3))
    assert float(np.abs(half[..., 3] - shrunk[..., 3]).mean()) < 6.0


# --- GPU で掛ける（レンダラ） ------------------------------------------------


@pytest.fixture(scope="module")
def gl_context() -> Iterator[object]:
    from sashimono.engine.gpu import GLContextError, OffscreenGLContext

    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _rendered(
    source: GeneratedSource,
    context: object,
    divisor: int = 1,
    *,
    below: GeneratedSource | None = None,
) -> np.ndarray:
    from sashimono.engine.render import FrameRenderer, RenderQuality

    project = Project.create(
        ProjectSettings(width=SIZE[0], height=SIZE[1], frame_rate=FrameRate(30))
    )
    for name, placed in (("V1", below), ("V2", source)):
        if placed is None:
            continue
        track = Track(kind=TrackKind.VIDEO, name=name)
        for command in (AddTrack(track), AddClip(track.id, Clip(0, 10, source=placed))):
            project = command.apply(project)
    renderer = FrameRenderer(
        project,
        context=context,  # type: ignore[arg-type]
        quality=RenderQuality(divisor),
    )
    try:
        return renderer.render(0)[..., :3].astype(np.int32)
    finally:
        renderer.close()


def test_a_blurred_layer_softens_only_the_border(gl_context: object) -> None:
    # 書き出しと同じ道（レンダラ）で、層のぼかしが縁だけをぼかし、字の塗りに掛からないこと
    # 壊れると、縁だけをぼかしたつもりの字幕が書き出しで字までぼけるか、縁がぼけないまま出る
    fill, _, outer = bands()
    sharp = _rendered(text(layer(12.0, RED)), gl_context)
    soft = _rendered(text(replace(layer(12.0, RED), effects=(blur(8.0),))), gl_context)
    assert np.abs(sharp[fill] - soft[fill]).max() <= 2
    assert np.abs(sharp[outer] - soft[outer]).mean() > 1.0
    # ぼけて、太さ 12 の縁の外まで縁の色がにじむ
    thick = alpha(draw(text(border_width=AnimatedValue(12.0))))
    outside = thick == 0
    assert soft[outside][:, 0].max() > sharp[outside][:, 0].max() + 20


def test_baking_a_layer_keeps_what_is_already_drawn(gl_context: object) -> None:
    # 層のエフェクトは字の絵を作る途中で GPU に掛ける 描画の作業場を使うと、先に重ねた
    # 下のクリップの絵を壊す
    ground = GeneratedSource(
        kind="shape",
        params={"shape": "background", "color": (0.0, 0.0, 1.0, 1.0)},
    )
    image = _rendered(
        text(replace(layer(12.0, RED), effects=(blur(8.0),))), gl_context, below=ground
    )
    corner = image[:8, :8].reshape(-1, 3)
    assert (corner == np.array([0, 0, 255])).all()


def _extent(mask: np.ndarray) -> tuple[int, int]:
    """色の付いた所の（幅, 高さ）"""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return 0, 0
    return int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1)


@pytest.mark.parametrize(
    ("words", "zoom", "angle", "axis", "factor"),
    [("字", 300.0, 0.0, 0, 2.5), ("字字字字", 100.0, 90.0, 1, 1.5)],
    ids=["拡大した影", "回した影"],
)
def test_a_zoomed_or_turned_shadow_is_not_cut(
    gl_context: object, words: str, zoom: float, angle: float, axis: int, factor: float
) -> None:
    # 層の影は層の絵の真ん中を支点に拡大・回転する 画素の項目だけで作業面を切ると、大きくした
    # 影や回した影の外側が四角の端で欠け、書き出しで影が途中で切れて見える
    shadow = registry.require("shadow").create(
        offset_x=0,
        offset_y=0,
        blur=0,
        opacity=100,
        color=(0.0, 0.0, 1.0, 1.0),
        zoom=zoom,
        angle=angle,
    )
    size = AnimatedValue(36.0)
    image = _rendered(
        text(replace(layer(4.0, RED), effects=(shadow,)), text=words, size=size), gl_context
    )
    plain = _rendered(text(layer(4.0, RED), text=words, size=size), gl_context)
    drawn = _extent(plain.max(axis=2) > 0)
    shadowed = _extent(image[..., 2] > 40)
    assert shadowed[axis] > drawn[axis] * factor


def test_a_layer_effect_without_change_matches_the_plain_layer(gl_context: object) -> None:
    # 何もしない値のエフェクト（ぼかし 0）を掛けた層は、掛けない層と同じ絵
    # 層を別の面に描いて GPU を通す道で、色や位置がずれると、ここが食い違う
    plain = _rendered(text(layer(12.0, RED), layer(4.0, WHITE)), gl_context)
    passed = _rendered(
        text(replace(layer(12.0, RED), effects=(blur(0.0),)), layer(4.0, WHITE)), gl_context
    )
    assert np.abs(plain - passed).max() <= 2
