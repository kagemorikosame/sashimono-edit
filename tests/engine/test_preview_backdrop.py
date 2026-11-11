"""プレビューで何も映らない所に敷く地（#456）

地はプレビューにだけ敷く 書き出しとスナップショットの画素が変わると、YMM4 と見え方を
合わせた作りが崩れ、黒のつもりで書き出した作品に市松が焼き込まれる
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from OpenGL import GL
from PySide6.QtCore import QCoreApplication

from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.gpu import Framebuffer, OffscreenGLContext
from sashimono.engine.render import FULL_QUALITY, FrameRenderer
from sashimono.engine.render.backdrop import (
    BLACK,
    CHECKER_COLORS,
    FILL_BLACK,
    FILL_CHECKER,
    FILL_DARK,
    FILL_LIGHT,
    LIGHT_COLOR,
    fill_backdrop,
    hex_rgb,
)
from sashimono.engine.render.background import BackgroundPrefetch
from sashimono.engine.render.prefetch import PreviewCache

pytestmark = pytest.mark.usefixtures("gpu")

SIZE = 72
SETTINGS = ProjectSettings(width=SIZE, height=SIZE, frame_rate=FrameRate(30))
VIEWPORT = (0, 0, SIZE, SIZE)
FRAME_BYTES = SIZE * SIZE * 4
#: 黒い映像が映っているコマと、何も映っていないコマ
#: 何も無い所を先に置く 先読みはタイムラインの長さまでしか描かないので、後ろに置くと貯まらない
NOTHING = 0
BLACK_VIDEO = 20


def _byte(value: str) -> tuple[int, int, int]:
    red, green, blue = hex_rgb(value)
    return (round(red * 255), round(green * 255), round(blue * 255))


def _project() -> Project:
    """20 コマ目から真っ黒な映像 その前は何も置いていない"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create(SETTINGS))
    source = GeneratedSource(
        kind="shape", params={"shape": "background", "color": (0.0, 0.0, 0.0, 1.0)}
    )
    return AddClip(track.id, Clip(timeline_start=BLACK_VIDEO, duration=10, source=source)).apply(
        project
    )


@pytest.fixture
def context() -> Iterator[OffscreenGLContext]:
    made = OffscreenGLContext()
    yield made
    made.release()


def _read(surface: Framebuffer) -> np.ndarray:
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, surface.handle)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    raw = GL.glReadPixels(0, 0, surface.width, surface.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(raw, dtype=np.uint8).reshape(surface.height, surface.width, 4)
    return np.ascontiguousarray(image)


def _shades(image: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in pixel) for pixel in image[:, :, :3].reshape(-1, 3)}  # type: ignore[misc]


def _wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout
    while not condition():
        if time.perf_counter() > deadline:
            raise AssertionError("待っても終わらなかった")
        QCoreApplication.processEvents()
        time.sleep(0.005)


class TestTheOutputStaysBlack:
    def test_render_is_unchanged_after_the_preview_used_a_checker(
        self, context: OffscreenGLContext
    ) -> None:
        """プレビューで市松を敷いた後も、書き出しとスナップショットの道（render）は同じ画素

        塗りのシェーダの市松の値が残ると、同じレンダラで撮ったスナップショットの何も無い所が
        市松になる 書き出しは黒のつもりで、気付かずに焼き込まれる
        """
        fresh = FrameRenderer(_project(), context=context)
        used = FrameRenderer(_project(), context=context)
        try:
            before = fresh.render(NOTHING)
            with context:
                used.compose(NOTHING, backdrop=fill_backdrop(FILL_CHECKER))
            after = used.render(NOTHING)
        finally:
            fresh.close()
            used.close()
        assert np.array_equal(before, after)
        assert _shades(after) == {(0, 0, 0)}
        assert (after[:, :, 3] == 255).all()

    def test_the_default_backdrop_is_black(self, context: OffscreenGLContext) -> None:
        """地を渡さない合成（書き出しの道）は黒 既定が変わると書き出しの絵が変わる"""
        renderer = FrameRenderer(_project(), context=context)
        try:
            with context:
                target = Framebuffer(SIZE, SIZE, internal_format=GL.GL_RGBA8)
                try:
                    renderer.compose(NOTHING)
                    renderer.compositor.present(target.handle, VIEWPORT)
                    image = _read(target)
                finally:
                    target.release()
        finally:
            renderer.close()
        assert _shades(image) == {(0, 0, 0)}
        assert BLACK.black and fill_backdrop(FILL_BLACK) == BLACK


class TestThePreviewTellsThemApart:
    def _shown(self, context: OffscreenGLContext, mode: str, frame: int) -> np.ndarray:
        renderer = FrameRenderer(_project(), context=context)
        try:
            with context:
                target = Framebuffer(SIZE, SIZE, internal_format=GL.GL_RGBA8)
                try:
                    renderer.compose(frame, backdrop=fill_backdrop(mode))
                    renderer.compositor.present(target.handle, VIEWPORT)
                    return _read(target)
                finally:
                    target.release()
        finally:
            renderer.close()

    def test_a_checker_shows_where_nothing_is_drawn(self, context: OffscreenGLContext) -> None:
        """何も描いていない所は市松、黒い映像の所は黒のまま

        地をクリップの下にまで敷くと、黒い映像も市松に見えて、今度は逆の見分けがつかない
        """
        empty = self._shown(context, FILL_CHECKER, NOTHING)
        video = self._shown(context, FILL_CHECKER, BLACK_VIDEO)
        assert _shades(empty) == {_byte(CHECKER_COLORS[0]), _byte(CHECKER_COLORS[1])}
        assert _shades(video) == {(0, 0, 0)}

    def test_the_checker_cells_follow_the_picture_height(self, context: OffscreenGLContext) -> None:
        """市松の目は画面の高さで決まる 1 画素ごとの細かい模様だと灰色の地にしか見えない"""
        empty = self._shown(context, FILL_CHECKER, NOTHING)
        row = empty[0, :, 0]
        changes = int(np.count_nonzero(row[1:] != row[:-1]))
        # 72 画素を 18 目に割ると 1 目 4 画素 1 行に 17 回色が変わる
        assert changes == 17

    @pytest.mark.parametrize(("mode", "color"), [(FILL_LIGHT, LIGHT_COLOR)])
    def test_a_plain_backdrop_fills_the_empty_part(
        self, context: OffscreenGLContext, mode: str, color: str
    ) -> None:
        # 単色の地が黒のままだと、選んでも何も変わらない
        assert _shades(self._shown(context, mode, NOTHING)) == {_byte(color)}

    def test_the_margin_has_its_own_color(self, context: OffscreenGLContext) -> None:
        """画面の外の余白は余白の色 中の何も無い所の地とは別に決める

        余白を塗らずに前の色を残すと、プレビューの枠を広げたときに前のコマの絵が残る
        """
        renderer = FrameRenderer(_project(), context=context)
        try:
            with context:
                target = Framebuffer(SIZE * 2, SIZE, internal_format=GL.GL_RGBA8)
                try:
                    renderer.compositor.margin = (0.25, 0.25, 0.25)
                    renderer.compose(BLACK_VIDEO)
                    renderer.compositor.present(target.handle, (0, 0, SIZE * 2, SIZE))
                    image = _read(target)
                finally:
                    target.release()
        finally:
            renderer.close()
        assert _shades(image[:, : SIZE // 4]) == {(64, 64, 64)}
        assert _shades(image[:, SIZE // 2 + 2 : SIZE + SIZE // 2 - 2]) == {(0, 0, 0)}


class TestTheCacheFollowsTheChoice:
    def test_changing_the_backdrop_drops_the_frames_drawn_with_the_old_one(
        self, context: OffscreenGLContext
    ) -> None:
        """地を変えたら貯めた絵を捨てる 残すと、貯めた所だけ前の地のまま出る"""
        renderer = FrameRenderer(_project(), context=context)
        try:
            with context:
                cache = PreviewCache(renderer, budget_bytes=FRAME_BYTES * 4)
                target = Framebuffer(SIZE, SIZE, internal_format=GL.GL_RGBA8)
                try:
                    cache.draw(NOTHING, target.handle, VIEWPORT)
                    assert NOTHING in cache.cache.cached
                    cache.set_backdrop(fill_backdrop(FILL_DARK))
                    assert cache.cache.cached == frozenset()
                    hit = cache.draw(NOTHING, target.handle, VIEWPORT)
                    image = _read(target)
                    # 同じ地を渡し直しただけでは捨てない 設定の窓で OK を押すたびに
                    # 貯め直すと、何も変えていないのに再生が重くなる
                    cache.set_backdrop(fill_backdrop(FILL_DARK))
                    assert NOTHING in cache.cache.cached
                finally:
                    target.release()
                    cache.release()
        finally:
            renderer.close()
        assert not hit
        assert _shades(image) == {(32, 32, 32)}

    def test_the_worker_redraws_with_the_new_backdrop(self, context: OffscreenGLContext) -> None:
        """別のスレッドで貯めた絵も、地を変えたその場で出さなくなり、新しい地で描き直す"""
        screen = FrameRenderer(_project(), context=context)
        background = BackgroundPrefetch(
            _project(),
            share=context.context,
            quality=FULL_QUALITY,
            proxies=None,
            decode_threads=1,
            budget_bytes=FRAME_BYTES * 2,
            playhead=NOTHING,
        )
        try:
            _wait_until(lambda: NOTHING in background.cached)
            background.set_backdrop(fill_backdrop(FILL_LIGHT))
            assert background.cached == frozenset(), "地を変えた後も前の地の絵を出せるまま"
            _wait_until(lambda: NOTHING in background.cached)
            with context:
                target = Framebuffer(SIZE, SIZE, internal_format=GL.GL_RGBA8)
                try:
                    assert background.show(NOTHING, screen.compositor, target.handle, VIEWPORT)
                    image = _read(target)
                finally:
                    target.release()
        finally:
            background.close()
            screen.close()
        assert _shades(image) == {_byte(LIGHT_COLOR)}
