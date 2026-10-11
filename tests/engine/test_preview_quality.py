"""プレビューの画質を落としたとき（1/2・1/4）に、書き出しと同じ絵を縮めた物が出ること（Issue #151）

画質を落とすと合成の大きさが縦横 1/2・1/4 になる 画素で決める値（位置・大きさ・ぼかしの強さ・
粒の大きさ・文字の大きさ）をそのまま当てると、小さい合成の画素で数えてしまい、絵が 2 倍・4 倍の
位置と大きさに出る 書き出し（等倍）は正しいので、プレビューだけがずれて見える

等倍で描いた絵を面積の平均で縮めた物と、画質を落として描いた絵を比べる 縮め方の違い
（輪郭の丸め）だけが残るよう、見る物は画質を落とした合成でも数画素以上の大きさにしてある
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
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    MediaItem,
    ParamValue,
    Project,
    ProjectSettings,
    Stroke,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import ParamInput, registry
from sashimono.effects.region import REGION_BLUR
from sashimono.effects.sources import source_registry
from sashimono.effects.spec import TrackSpec
from sashimono.effects.strokes import STROKE
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, RenderQuality

#: 1/2 でも 1/4 でも割り切れる大きさ 割り切れないと、縮めた等倍の絵と升目がずれる
SETTINGS = ProjectSettings(width=320, height=176, frame_rate=FrameRate(30))

#: 見比べる画質の分母
DIVISORS = (2, 4)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _write_png(path: Path, image: np.ndarray) -> Path:
    from PySide6.QtGui import QImage

    height, width = image.shape[:2]
    data = np.ascontiguousarray(image, dtype=np.uint8)
    qimage = QImage(data.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    assert qimage.save(str(path))
    return path


@pytest.fixture
def checker(tmp_path: Path) -> MediaItem:
    """160 × 96 の白黒の市松 升目は 8 画素

    1/4 に縮めても升目が 2 画素残る ぼかしやモザイクが掛かった所は灰色に均され、
    掛からない所は市松のまま残るので、範囲の位置と大きさが絵の差に出る
    """
    ys, xs = np.mgrid[0:96, 0:160]
    white = ((xs // 8 + ys // 8) % 2) == 0
    image = np.zeros((96, 160, 4), dtype=np.uint8)
    image[white, :3] = 255
    image[..., 3] = 255
    return probe_media(_write_png(tmp_path / "市松.png", image))


def _project(clip: Clip, media: MediaItem | None = None) -> Project:
    project = Project.create(SETTINGS)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    commands = [AddTrack(track), AddClip(track.id, clip)]
    if media is not None:
        commands.insert(0, AddMedia(media))
    for command in commands:
        project = command.apply(project)
    return project


def _param(value: float | str | bool | tuple[float, ...]) -> ParamValue:
    # 数は動かせる値として持つ（定義の TrackSpec と同じ形） 真偽・選択肢・色はそのまま
    if isinstance(value, bool | str | tuple):
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


def _picture(media: MediaItem, *effects: Effect, **placement: float | str | bool) -> Clip:
    clip = Clip(timeline_start=0, duration=10, media_id=media.id, native_size=True)
    return _placed(replace(clip, effects=effects), **placement)


def _source(kind: str, *effects: Effect, **params: float | str | bool | tuple[float, ...]) -> Clip:
    """生成オブジェクトのクリップ ``effects`` は固定の欄の前に積む（足したエフェクトと同じ）"""
    source = GeneratedSource(kind=kind, params={k: _param(v) for k, v in params.items()})
    clip = _placed(Clip(timeline_start=0, duration=10, source=source))
    return replace(clip, effects=(*effects, *clip.effects))


def _effect(kind: str, **params: ParamInput) -> Effect:
    return registry.require(kind).create(**params)


def _render(project: Project, context: OffscreenGLContext, divisor: int = 1) -> np.ndarray:
    renderer = FrameRenderer(project, context=context, quality=RenderQuality(divisor))
    try:
        return renderer.render(0)[..., :3].astype(np.float32)
    finally:
        renderer.close()


def _shrunk(image: np.ndarray, divisor: int) -> np.ndarray:
    """等倍の絵を ``divisor`` 画素四方の平均で縮める"""
    height, width = image.shape[0] // divisor, image.shape[1] // divisor
    blocks = image[: height * divisor, : width * divisor].reshape(
        height, divisor, width, divisor, -1
    )
    shrunk: np.ndarray = blocks.mean(axis=(1, 3))
    return shrunk


def _mismatch(project: Project, context: OffscreenGLContext, divisor: int) -> float:
    """画質を落とした絵と、等倍の絵を縮めた物の、色の差の平均（0..255）"""
    full = _shrunk(_render(project, context), divisor)
    light = _render(project, context, divisor)
    assert light.shape == full.shape
    # 何も描かれていない絵どうしを比べて通ってしまわないように
    assert full.max() > 100
    return float(np.abs(light - full).mean())


def test_every_pixel_setting_is_marked() -> None:
    # 画素の値だと示す単位（px・px/秒 など）を持つのに縮める印が付いていないと、その値だけ
    # 画質を落としたプレビューで 2 倍・4 倍に出る 新しいエフェクトで書き忘れたら落ちる
    # 画面の画素ではない物は pixels=False を明に書く（書いたなら見落としではない）
    parameters = [
        (definition.kind, spec) for definition in registry.all() for spec in definition.parameters
    ] + [
        (definition.kind, spec)
        for definition in (*source_registry.all(), STROKE)
        for spec in definition.parameters
    ]
    forgotten = [
        f"{kind}.{spec.name}"
        for kind, spec in parameters
        if isinstance(spec, TrackSpec)
        and spec.unit.startswith("px")
        and spec.pixels is None
        and not spec.in_pixels
    ]
    assert forgotten == []


#: 色の差の平均の上限（0..255） 輪郭の丸めとぼかしの畳み方の違いはこの内に収まる
#: （直した後はどれも 3.5 より小さい） 直す前は位置や大きさが 2 倍・4 倍に出て、
#: どれもこの数倍になっていた
TOLERANCE = 5.0


@pytest.mark.parametrize("divisor", DIVISORS)
class TestAPreviewAtLowerQualityIsTheExportShrunk:
    def test_a_moved_picture(
        self, gl_context: OffscreenGLContext, checker: MediaItem, divisor: int
    ) -> None:
        # 固定の配置の X・Y を小さい合成の画素で当てると、絵が 2 倍・4 倍の所まで動く（#162）
        # 回した市松は 1/4 だと升目が 2 画素しかなく、読み方の違いで縁が崩れる（直した後で 7 ほど）
        # 直す前は 40 を超えるので、倍の幅でも見分けられる
        clip = _picture(checker, pos_x=48, pos_y=-24, anchor_x=16, rotation=20)
        assert _mismatch(_project(clip, checker), gl_context, divisor) < TOLERANCE * 2

    def test_a_picture_at_its_own_pixels(
        self, gl_context: OffscreenGLContext, checker: MediaItem, divisor: int
    ) -> None:
        # 素材の画素で置く大きさ（native_size #158）は、合成の大きさに合わせて縮む
        clip = _picture(checker)
        assert _mismatch(_project(clip, checker), gl_context, divisor) < TOLERANCE

    def test_a_region_blur(
        self, gl_context: OffscreenGLContext, checker: MediaItem, divisor: int
    ) -> None:
        # 部分ぼかしの範囲（#150） 画素で持つ中心と幅がそのままだと、範囲が 2 倍の所に出る
        blur = _effect(
            REGION_BLUR,
            mode="blur",
            center_x=24,
            center_y=16,
            region_width=96,
            region_height=64,
            blur_radius=12,
        )
        clip = _picture(checker, blur)
        assert _mismatch(_project(clip, checker), gl_context, divisor) < TOLERANCE

    def test_a_region_mosaic(
        self, gl_context: OffscreenGLContext, checker: MediaItem, divisor: int
    ) -> None:
        mosaic = _effect(
            REGION_BLUR,
            mode="mosaic",
            center_x=-24,
            center_y=8,
            region_width=80,
            region_height=40,
            mosaic_size=24,
        )
        clip = _picture(checker, mosaic)
        assert _mismatch(_project(clip, checker), gl_context, divisor) < TOLERANCE

    def test_a_mask(self, gl_context: OffscreenGLContext, checker: MediaItem, divisor: int) -> None:
        # 前からあるマスクも同じ 幅と中心を画素で持つ
        mask = _effect("mask", center_x=24, center_y=-8, mask_width=96, mask_height=56, feather=8)
        clip = _picture(checker, mask)
        assert _mismatch(_project(clip, checker), gl_context, divisor) < TOLERANCE

    def test_a_blur(self, gl_context: OffscreenGLContext, divisor: int) -> None:
        # ぼかしの強さ（画素） 小さい合成の画素で畳むと、2 倍・4 倍にぼける
        clip = _source("shape", _effect("blur", radius=16), shape="rect", width=96, height=64)
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 2

    def test_an_emboss(self, gl_context: OffscreenGLContext, divisor: int) -> None:
        # エンボスの取り込み幅（画素） 小さい合成の画素で離して読むと、なだらかな坂の高さの差が
        # 2 倍・4 倍に出て、面全体の明暗が書き出しより強くなる
        # 縮めないと差の平均が 1/2 で 33・1/4 で 81 ほど、縮めると 0.3 より小さい
        # グラデーションの 90 度は横へ変わる坂 光も横（0 度）から当てて、面全体に明暗を出す
        ramp = _effect("gradient", angle=90, span=320)
        emboss = _effect("emboss", angle=0, height=300, reach=12)
        clip = _source(
            "shape", ramp, emboss, shape="rect", width=320, height=176, color=(1.0, 1.0, 1.0, 1.0)
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 5

    def test_a_shadow_and_a_border(self, gl_context: OffscreenGLContext, divisor: int) -> None:
        clip = _source(
            "shape",
            _effect("border", width=8, color=(1.0, 0.0, 0.0, 1.0)),
            _effect("shadow", offset_x=24, offset_y=-16, blur=4, opacity=100),
            shape="ellipse",
            width=96,
            height=64,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE

    def test_a_shape_placed_in_its_own_settings(
        self, gl_context: OffscreenGLContext, divisor: int
    ) -> None:
        # 図形の幅・高さ・位置・線の太さは画素 そのまま描くと小さい合成で 2 倍・4 倍になる
        clip = _source(
            "shape",
            shape="rounded",
            width=120,
            height=56,
            corner_radius=16,
            line_width=8,
            outline_only=True,
            pos_x=-40,
            pos_y=24,
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE

    def test_text(self, gl_context: OffscreenGLContext, divisor: int) -> None:
        # 文字の大きさ・縁取りの太さ・影の距離とぼかし・位置は画素
        clip = _source(
            "text",
            text="字あA",
            size=56,
            border_width=6,
            border_color=(1.0, 0.0, 0.0, 1.0),
            shadow_x=12,
            shadow_y=-12,
            shadow_blur=4,
            pos_x=32,
            pos_y=16,
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE

    def test_text_moved_by_the_placement(
        self, gl_context: OffscreenGLContext, divisor: int
    ) -> None:
        clip = _placed(_source("text", text="字", size=64), pos_x=-64, pos_y=32, scale=150)
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE

    def test_text_with_stroke_layers(self, gl_context: OffscreenGLContext, divisor: int) -> None:
        # 縁取りの層の太さは画面の画素（#272） 縮めずに描くと、層の縁だけ 2 倍・4 倍の太さに出る
        clip = _stroked(
            Stroke(params={"width": AnimatedValue(4.0), "color": (1.0, 1.0, 1.0, 1.0)}),
            Stroke(params={"width": AnimatedValue(12.0), "color": (1.0, 0.0, 0.0, 1.0)}),
            Stroke(
                params={
                    "width": AnimatedValue(5.0),
                    "color": (0.0, 0.0, 1.0, 1.0),
                    "position": "inside",
                }
            ),
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE

    def test_a_blurred_stroke_layer(self, gl_context: OffscreenGLContext, divisor: int) -> None:
        # 層に掛けたぼかしの強さも画面の画素（#273） 層の絵は縮めて描くので、ぼかしも同じだけ
        # 縮めて掛ける 縮めないと、プレビューだけ縁が 2 倍・4 倍にぼける
        clip = _stroked(
            Stroke(
                params={"width": AnimatedValue(16.0), "color": (1.0, 0.0, 0.0, 1.0)},
                effects=(_effect("blur", radius=24),),
            )
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 2


def _stroked(*strokes: Stroke) -> Clip:
    """縁取りの層を持つテキストのクリップ"""
    clip = _source("text", text="字あA", size=56, pos_x=24)
    assert clip.source is not None
    return replace(clip, source=clip.source.with_strokes(strokes))


@pytest.mark.parametrize("divisor", DIVISORS)
class TestFineDetailAtLowerQuality:
    """1 画素の粒や 1 画素より細い縁は、合成の画素へそのまま当てると太く・粗く見える（#173）

    等倍の絵を縮めると、1 画素の粒は周りと平均されて薄まり、細い縁は外の画素と混ざって
    淡い線になる 画質を落とした合成で 1 合成画素ずつ塗ると、粒は 2 倍・4 倍の大きさで
    濃いまま、縁は 1 合成画素（画面の 2〜4 画素）の太さで濃く出る
    """

    def test_noise_grains_match_the_export_shrunk(
        self, gl_context: OffscreenGLContext, divisor: int
    ) -> None:
        # 粒が合成の 1 画素のままだと、縮めた等倍の絵より粒が大きく濃く見える
        # 画面いっぱいの図形に掛けて、背景で差が薄まらないようにする 暗い色は 0 で止まる
        # 粒が多く、止め方の違いも差に出る 直す前は差の平均が 30 ほど、直した後は 0.3 ほど
        noise = _effect("noise", strength=100, monochrome=False, animate=False)
        clip = _source(
            "shape", noise, shape="rect", width=320, height=176, color=(0.2, 0.2, 0.2, 1.0)
        )
        assert _mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 5

    @pytest.mark.parametrize("width", [1, 3])
    def test_a_border_with_a_fraction_of_a_canvas_pixel(
        self, gl_context: OffscreenGLContext, divisor: int, width: int
    ) -> None:
        # 太さ 1 の縁は 1/2・1/4 で合成の 1 画素に満たない 3 の縁は 1/2 で 1.5 画素
        # 端数を切り捨てると縁が消えるか細り、切り上げると太く濃く出る
        # 直す前は差の平均が 20〜40 ほど、直した後はどれも 1.5 より小さい
        clip = _source(
            "shape",
            _effect("border", width=width, color=(1.0, 0.0, 0.0, 1.0)),
            shape="rect",
            width=96,
            height=64,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        assert _rim_mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 2

    def test_a_bevel_thinner_than_a_canvas_pixel(
        self, gl_context: OffscreenGLContext, divisor: int
    ) -> None:
        # 縁の反射の太さも同じ 1 画素に切り上げると、光る帯が 2 倍・4 倍の太さになる
        # 直す前は 1/2 で 4.8・1/4 で 11.8、直した後はどちらも 0.4 より小さい
        bevel = _effect("bevel_light", thickness=1, constant=100)
        clip = _source(
            "shape", bevel, shape="rect", width=96, height=64, color=(0.2, 0.2, 0.2, 1.0)
        )
        assert _rim_mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 5

    @pytest.mark.parametrize("thickness", [3, 5, 6, 10])
    def test_a_bevel_with_a_fraction_of_a_canvas_pixel(
        self, gl_context: OffscreenGLContext, divisor: int, thickness: int
    ) -> None:
        # 太さ 6 の縁の反射は 1/4 で 1.5 画素 3 は 1/2 で 1.5 画素、5 は 1/4 で 1.25 画素（#182）
        # 輪を太さの端数で刻むと、急な面が緩く帯の外まで広がる 書き出しの段は 1 画素ずつで
        # いちばん外の段の傾きが半分なので、合成の 1 画素に急な段・半分の段・平らな面が
        # 端数に応じて入り交じる 光は傾きに比例しないので、段ごとに光を当てて色で平均する
        # 直す前は 1.35〜9.6（太さ 5 の 1/4 がいちばん大きい）、直した後はどれも 1 より小さい
        bevel = _effect("bevel_light", thickness=thickness, constant=100)
        clip = _source(
            "shape", bevel, shape="rect", width=96, height=64, color=(0.2, 0.2, 0.2, 1.0)
        )
        assert _rim_mismatch(_project(clip), gl_context, divisor) < TOLERANCE / 4

    @pytest.mark.parametrize("thickness", [3, 6, 10])
    def test_a_bevel_on_a_curved_edge(
        self, gl_context: OffscreenGLContext, divisor: int, thickness: int
    ) -> None:
        # 楕円の縁では、書き出しの段の境が画素の升目に揃わない 書き出しは 24 方向の輪で
        # 縁を探すので、斜めの所では縁までの本当の距離より少し遠くに段が来る 合成の画素で
        # 輪を刻むと、この癖と 1 合成画素より細かい縁の位置が消えて段がずれる（#194）
        # 楕円の輪郭そのものも縮め方で違う（エフェクトが無くても 1/2 で 7 ほど）ので、
        # 四角より緩い上限にする 直す前は 4.4〜5.9、直した後は 0.7〜2.8
        bevel = _effect("bevel_light", thickness=thickness, constant=100)
        clip = _source(
            "shape", bevel, shape="ellipse", width=96, height=64, color=(0.2, 0.2, 0.2, 1.0)
        )
        assert _rim_mismatch(_project(clip), gl_context, divisor) < TOLERANCE * 0.7

    @pytest.mark.parametrize(
        ("shape", "thickness", "blur"),
        [("rect", 3, 4), ("rect", 5, 3), ("ellipse", 3, 4)],
    )
    def test_a_blurred_bevel(
        self,
        gl_context: OffscreenGLContext,
        divisor: int,
        shape: str,
        thickness: int,
        blur: int,
    ) -> None:
        # 書き出しは書き出しの画素で段を刻んでからぼかす 合成の画素で刻んでぼかすと、
        # 太さ 3 は 1/4 で 1 画素に切り上がって段が 1 つになり、急な坂が縁の内へ広がる
        # 覆う書き出しの画素で刻んでから縮めてぼかしても、補うと縁の近くの坂が
        # 1 合成画素の幅へ均される（#194） 1/4 ほど崩れやすいので上限を分母で分ける
        # 直す前は 1/2 で 1.48〜1.94・1/4 で 5.0〜7.1、直した後は 1/2 で 1.2・1/4 で 1.9 より小さい
        bevel = _effect("bevel_light", thickness=thickness, blur=blur, constant=100)
        clip = _source("shape", bevel, shape=shape, width=96, height=64, color=(0.2, 0.2, 0.2, 1.0))
        limit = {2: 1.4, 4: 2.5}[divisor]
        assert _rim_mismatch(_project(clip), gl_context, divisor) < limit


@pytest.mark.parametrize(
    ("shape", "thickness", "blur", "profile", "divisor", "before"),
    [
        ("rect", 60, 0, "straight", 2, 0.56),
        ("rect", 60, 0, "round", 2, 0.64),
        ("rect", 40, 1, "straight", 4, 0.82),
        ("rect", 40, 4, "straight", 4, 0.74),
        ("rect", 5, 30, "straight", 4, 0.32),
        ("ellipse", 3, 12, "straight", 2, 1.24),
    ],
)
def test_a_bevel_is_no_further_from_the_export_than_before(
    gl_context: OffscreenGLContext,
    shape: str,
    thickness: int,
    blur: int,
    profile: str,
    divisor: int,
    before: float,
) -> None:
    # #194 の思い描き方は、四角と楕円・太さ 1〜128・ぼかし 0〜96 の 592 通りの多くで縮めた
    # 書き出しに近づいたが、一部は #182 までの描き方の方が近かった（ぼかしの無い 48 を
    # 超える太さ、ぼかしの坂が広い所、太さ 3 に強いぼかし） そこでは前の描き方を使う
    # before は #182 までの差（小数 2 桁へ切り上げ） 今の思い描き方をそのまま使うと、
    # 四角の太さ 60 で 2.5・丸で 3.6 など、どれもこれの 1 割増しを超える GPU やドライバの
    # 丸めの違いで同じ描き方でも差が少し揺れるので、1 割の余裕を見る 前と今の差が 1 割に
    # 満たない組み合わせ（楕円の太さ 3 ぼかし 8 の 1/2 は 1.42 と 1.47）は、この余裕では
    # 見分けられないので並べない
    bevel = _effect("bevel_light", thickness=thickness, blur=blur, profile=profile, constant=100)
    clip = _source("shape", bevel, shape=shape, width=96, height=64, color=(0.2, 0.2, 0.2, 1.0))
    assert _rim_mismatch(_project(clip), gl_context, divisor) <= before * 1.1


def _rim_mismatch(project: Project, context: OffscreenGLContext, divisor: int) -> float:
    """縁の周りだけで見た差の平均（0..255）

    細い縁は絵の中で占める画素が少なく、絵全体で平均すると差が埋もれる 等倍と
    画質を落とした絵のどちらかで色が変わっている画素（縁と、その外の 1 画素）だけを見る
    """
    full = _shrunk(_render(project, context), divisor)
    light = _render(project, context, divisor)
    assert light.shape == full.shape
    plain = _shrunk(_render(_without_effects(project), context), divisor)
    rim = (np.abs(full - plain).max(axis=-1) > 4) | (np.abs(light - plain).max(axis=-1) > 4)
    assert rim.any()
    return float(np.abs(light - full)[rim].mean())


def _without_effects(project: Project) -> Project:
    """足したエフェクトを外した同じプロジェクト 固定の欄（配置）は残す"""
    tracks = tuple(
        replace(
            track,
            clips=tuple(
                replace(clip, effects=tuple(e for e in clip.effects if e.fixed))
                for clip in track.clips
            ),
        )
        for track in project.timeline.tracks
    )
    return replace(project, timeline=replace(project.timeline, tracks=tracks))
