"""最初から持つ欄（既定のままの配置と反転）と、素材の画素の大きさで置く描き方

置いたクリップすべてに配置と反転が付く 既定のままでシェーダを通すと、全クリップが
中間のバッファを通って遅くなり、絵も前の版と変わる 飛ばしていることを画素で押さえる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddMedia, AddTrack, insert_media
from sashimono.core.commands.fixed import (
    FLIP_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    fixed_effect,
    with_fixed_items,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import EffectProcessor, GLContextError, OffscreenGLContext
from sashimono.engine.gpu.glutil import ScreenQuad
from sashimono.engine.render import FrameRenderer, RenderQuality

SETTINGS = ProjectSettings(width=320, height=180, frame_rate=FrameRate(30))


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
def picture(tmp_path: Path) -> MediaItem:
    """64 × 32 の画像 左半分が赤、右半分が緑 画面（320 × 180）とは縦横比も違う"""
    image = np.zeros((32, 64, 4), dtype=np.uint8)
    image[:, :32] = (255, 0, 0, 255)
    image[:, 32:] = (0, 255, 0, 255)
    return probe_media(_write_png(tmp_path / "絵.png", image))


def _with_clip(clip: Clip, media: MediaItem | None = None) -> Project:
    project = Project.create(SETTINGS)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    commands = [AddTrack(track), AddClip(track.id, clip)]
    if media is not None:
        commands.insert(0, AddMedia(media))
    for command in commands:
        project = command.apply(project)
    return project


def _render(
    project: Project, context: OffscreenGLContext, quality: RenderQuality | None = None
) -> np.ndarray:
    renderer = (
        FrameRenderer(project, context=context)
        if quality is None
        else FrameRenderer(project, context=context, quality=quality)
    )
    try:
        return renderer.render(0)
    finally:
        renderer.close()


def _painted(image: np.ndarray) -> tuple[int, int, int, int]:
    """色の付いた範囲 左・上・右・下（右と下は含まない）"""
    ys, xs = np.nonzero(image[..., :3].max(axis=2) > 16)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _picture_clip(media: MediaItem, *effects: Effect, native: bool = False) -> Clip:
    return Clip(
        timeline_start=0, duration=10, media_id=media.id, effects=effects, native_size=native
    )


class TestIdleItemsChangeNothing:
    def test_a_media_clip_draws_the_same_with_the_items(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 前の main（欄を持たないクリップ）と 1 画素も変わらないこと 既定のままの配置を
        # シェーダへ通すと中間のバッファを通り、縁の画素が変わる
        bare = _picture_clip(picture)
        before = _render(_with_clip(bare, picture), gl_context)
        after = _render(_with_clip(with_fixed_items(bare, picture=True), picture), gl_context)
        assert np.array_equal(before, after)

    def test_text_draws_the_same_with_the_items(self, gl_context: OffscreenGLContext) -> None:
        # 既定の欄を通すと文字が中間のバッファで描き直され、前の版で作った作品のテロップの縁が変わる
        text = Clip(
            timeline_start=0,
            duration=10,
            source=GeneratedSource(kind="text", params={"text": "字", "size": AnimatedValue(60)}),
        )
        before = _render(_with_clip(text), gl_context)
        after = _render(_with_clip(with_fixed_items(text, picture=True)), gl_context)
        assert before[..., :3].max() > 0
        assert np.array_equal(before, after)

    def test_an_added_effect_draws_the_same_before_the_items(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 足したエフェクトは欄の前に入る 欄を通っても何もしないので、欄が無かったころと同じ絵
        blur = registry.require("blur").create(radius=3)
        before = _render(_with_clip(_picture_clip(picture, blur), picture), gl_context)
        placed = with_fixed_items(_picture_clip(picture, blur), picture=True)
        after = _render(_with_clip(placed, picture), gl_context)
        assert np.array_equal(before, after)

    def test_the_idle_items_are_not_work(self, gl_context: OffscreenGLContext) -> None:
        # 仕事があると見なすと、全クリップが画面 1 枚ぶんの中間バッファを通る
        with gl_context:
            quad = ScreenQuad()
            processor = EffectProcessor(16, 16, quad)
            try:
                idle = (fixed_effect(FLIP_EFFECT_KIND), fixed_effect(TRANSFORM_EFFECT_KIND))
                assert not processor.has_work(idle)
                moved = fixed_effect(TRANSFORM_EFFECT_KIND).with_param("pos_x", AnimatedValue(5.0))
                assert processor.has_work((moved,))
            finally:
                processor.release()
                quad.release()

    def test_a_moved_item_still_moves(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 飛ばし過ぎていないこと 動かした配置まで飛ばすと、X を変えても絵が動かない
        clip = with_fixed_items(_picture_clip(picture, native=True), picture=True)
        moved = replace(
            clip,
            effects=(
                clip.effects[0],
                clip.effects[1].with_param("pos_x", AnimatedValue(40.0)),
            ),
        )
        left, _, _, _ = _painted(_render(_with_clip(moved, picture), gl_context))
        assert left == (320 - 64) // 2 + 40

    def test_a_flipped_item_still_flips(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 反転の欄まで何もしない物として飛ばすと、パネルで左右反転を入れても絵が裏返らない
        clip = with_fixed_items(_picture_clip(picture, native=True), picture=True)
        flipped = replace(
            clip, effects=(clip.effects[0].with_param("horizontal", True), clip.effects[1])
        )
        image = _render(_with_clip(flipped, picture), gl_context)
        left, top, _, _ = _painted(image)
        # 裏返すと左側が緑になる
        assert image[top + 4, left + 4, 1] > 200 and image[top + 4, left + 4, 0] < 50


class TestNativeSize:
    def test_a_newly_placed_picture_is_its_own_pixels(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 利用者の決定 拡大率 100% は素材の画素 画面に収めると 64 × 32 が 320 × 160 に伸びる
        project = Project.create(SETTINGS)
        for command in insert_media(project, picture):
            project = command.apply(project)
        box = _painted(_render(project, gl_context))
        assert box == (128, 74, 192, 106)

    def test_an_old_clip_still_fits_the_screen(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 前の版のファイル（項目の無いクリップ）は収めて描いた見た目のまま
        box = _painted(_render(_with_clip(_picture_clip(picture), picture), gl_context))
        assert box == (0, 10, 320, 170)

    def test_the_size_follows_the_preview_quality(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # プレビューの解像度を半分にしたら絵も半分 届いた絵の画素のまま置くと、画面に対して
        # 倍の大きさで見え、書き出しと構図が変わる
        clip = _picture_clip(picture, native=True)
        image = _render(_with_clip(clip, picture), gl_context, RenderQuality(2))
        assert image.shape[:2] == (90, 160)
        assert _painted(image) == (64, 37, 96, 53)

    def test_the_size_comes_from_the_source_not_the_proxy(
        self, gl_context: OffscreenGLContext, picture: MediaItem
    ) -> None:
        # 控え（プロキシ）は縮めて作る 届いた絵の大きさで置くと、控えの有無で大きさが変わる
        # 素材の記録の解像度を 2 倍にすると、同じ画像でも 2 倍の大きさで置かれる
        (stream,) = picture.video_streams
        doubled = replace(picture, video_streams=(replace(stream, width=128, height=64),))
        box = _painted(
            _render(_with_clip(_picture_clip(doubled, native=True), doubled), gl_context)
        )
        assert box == (96, 58, 224, 122)
