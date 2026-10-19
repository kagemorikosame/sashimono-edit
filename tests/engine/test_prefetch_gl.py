"""先読みした絵を、実際に GPU で作って出す所

取っておく絵は **sRGB へ符号化済み** 出すときにもう 1 度符号化すると白っぽくなるが、
絵としては「なんとなく明るい」だけで、当たったコマでしか起きないので気づけない
その場で描いた絵と画素で突き合わせる
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from OpenGL import GL

from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.gpu import Framebuffer, GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, Invalidation
from sashimono.engine.render.prefetch import PreviewCache

SETTINGS = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30))
VIEWPORT = (0, 0, 64, 64)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


#: 画面を塗る色（リニア） **白でも黒でもない値にする**
#: 白は符号化を何度掛けても 255 のまま、黒は 0 のまま 2 重に掛けた誤りが
#: 絵に出ないので、突き合わせても通ってしまう
GRAY = 0.5


def _gray_project(frames: int = 40) -> Project:
    """画面いっぱいの中間色 どのコマでも同じ絵で、貯まったかどうかだけを見る"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create(SETTINGS))
    source = GeneratedSource(
        kind="shape", params={"shape": "background", "color": (GRAY, GRAY, GRAY, 1.0)}
    )
    # コマごとに薄さが変わる **どのコマの絵かを画素で見分けるため**
    # どのコマも同じ絵だと、取っておいた絵と今の絵を取り違えても気づけない
    fading = AnimatedValue(
        keyframes=(Keyframe(frame=0, value=1.0), Keyframe(frame=frames - 1, value=0.1))
    )
    clip = Clip(timeline_start=0, duration=frames, source=source, opacity=fading)
    return AddClip(track.id, clip).apply(project)


def _read(surface: Framebuffer) -> np.ndarray:
    """描画先の中身を読み出す 画素で突き合わせるため"""
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, surface.handle)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    raw = GL.glReadPixels(0, 0, surface.width, surface.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(raw, dtype=np.uint8).reshape(surface.height, surface.width, 4)
    return np.ascontiguousarray(image)


class TestTheCachedPictureMatches:
    def test_it_is_not_encoded_twice(self, gl_context: OffscreenGLContext) -> None:
        """取っておいた絵と、その場で描いた絵が同じであること

        取ってあるのは符号化済みなので、出すときに :meth:`present` を通すと
        2 度掛かって白っぽくなる 当たったコマだけ明るくなるので目では気づけない
        """
        project = _gray_project()
        renderer = FrameRenderer(project, context=gl_context)
        try:
            with gl_context:
                screen = Framebuffer(64, 64, internal_format=GL.GL_RGBA8)
                try:
                    renderer.compose(5)
                    renderer.compositor.present(screen.handle, VIEWPORT)
                    direct = _read(screen)

                    cache = PreviewCache(renderer, budget_bytes=64 * 64 * 4 * 4)
                    assert not cache.draw(5, screen.handle, VIEWPORT)
                    first = _read(screen)
                    # 合成用の画面を別のコマで上書きしてから当てる 上書きしないと、
                    # 取っておいた絵を使わずに合成用の画面をそのまま出していても通る
                    renderer.compose(35)
                    assert cache.draw(5, screen.handle, VIEWPORT)
                    second = _read(screen)
                    cache.release()
                finally:
                    screen.release()
        finally:
            renderer.close()

        assert np.array_equal(direct, first)
        assert np.array_equal(direct, second)


class TestFillingAhead:
    def _cache(self, renderer: FrameRenderer, frames: int) -> PreviewCache:
        return PreviewCache(renderer, budget_bytes=64 * 64 * 4 * frames)

    def test_it_fills_forward_from_the_playhead(self, gl_context: OffscreenGLContext) -> None:
        """再生ヘッドから前へ順に貯める 後ろから貯めると、いま要る所が最後になる"""
        renderer = FrameRenderer(_gray_project(), context=gl_context)
        try:
            with gl_context:
                cache = self._cache(renderer, 3)
                for _ in range(3):
                    assert cache.step(10)
                assert cache.cache.cached == {10, 11, 12}
                cache.release()
        finally:
            renderer.close()

    def test_it_stops_when_it_is_full(self, gl_context: OffscreenGLContext) -> None:
        """貯まりきったら止まる 止まらないと、空き時間のたびに数え続ける"""
        renderer = FrameRenderer(_gray_project(), context=gl_context)
        try:
            with gl_context:
                cache = self._cache(renderer, 2)
                assert cache.step(0)
                assert cache.step(0)
                assert not cache.step(0)
                cache.release()
        finally:
            renderer.close()

    def test_it_stops_at_the_end_of_the_timeline(self, gl_context: OffscreenGLContext) -> None:
        """タイムラインの終わりより先は作らない 作っても黒いだけで、その枚数だけ
        貯められる長さが減る
        """
        renderer = FrameRenderer(_gray_project(frames=3), context=gl_context)
        try:
            with gl_context:
                cache = self._cache(renderer, 8)
                while cache.step(0):
                    pass
                assert cache.cache.cached == {0, 1, 2}
                cache.release()
        finally:
            renderer.close()

    def test_an_edit_takes_back_only_its_range(self, gl_context: OffscreenGLContext) -> None:
        # 直した所を捨て損ねると古い絵のまま再生され、外まで捨てると編集のたびに先読みが空になる
        renderer = FrameRenderer(_gray_project(), context=gl_context)
        try:
            with gl_context:
                cache = self._cache(renderer, 8)
                while cache.step(0):
                    pass
                stored = cache.cache.cached
                assert cache.invalidate(Invalidation.over([(2, 4)])) == 2
                assert cache.cache.cached == stored - {2, 3}
                cache.release()
        finally:
            renderer.close()


class TestWithoutABudget:
    def test_it_draws_straight_to_the_screen(self, gl_context: OffscreenGLContext) -> None:
        """先読みを切っても絵は出る 切ったのに真っ黒では使い物にならない"""
        renderer = FrameRenderer(_gray_project(), context=gl_context)
        try:
            with gl_context:
                screen = Framebuffer(64, 64, internal_format=GL.GL_RGBA8)
                try:
                    cache = PreviewCache(renderer, budget_bytes=0)
                    assert not cache.enabled
                    assert not cache.draw(3, screen.handle, VIEWPORT)
                    image = _read(screen)
                    cache.release()
                finally:
                    screen.release()
        finally:
            renderer.close()
        # 何か映っていること 真っ黒でも真っ白でもない
        assert 100 < int(image[32, 32, 0]) < 160
