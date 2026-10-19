"""写しを撒くエフェクトの効き方（AviUtl の ランダム配置）

値の意味は AviUtl2 に見本を描かせて測った ここに並ぶ数はその実測から取ったもので、
落ちたときに疑うのはこちらの実装の側
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from sashimono.core.model import (
    Clip,
    Effect,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

WIDTH, HEIGHT = 240, 240
#: 下地の四角の一辺 真ん中に置く
SQUARE = 120


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture
def draw(gl: OffscreenGLContext) -> Callable[..., np.ndarray]:
    """真ん中に白い四角を置いた絵へエフェクトを掛ける"""

    def run(*effects: Effect, width: int = SQUARE, height: int = SQUARE) -> np.ndarray:
        source: GeneratedSource = SHAPE.create(
            shape="rect", width=width, height=height, color=(1.0, 1.0, 1.0, 1.0)
        )
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(Clip(timeline_start=0, duration=30, source=source, effects=effects),),
        )
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            return renderer.render(0)
        finally:
            renderer.close()

    return run


def _lit(image: np.ndarray) -> np.ndarray:
    return image[:, :, :3].max(axis=2) > 24


def _extent(image: np.ndarray) -> tuple[int, int, int, int]:
    """光っている所の 左・右・上・下（画面の座標 下が大きい）"""
    rows, columns = np.where(_lit(image))
    if not len(rows):
        return (0, 0, 0, 0)
    return (int(columns.min()), int(columns.max()), int(rows.min()), int(rows.max()))


class TestScatter:
    """ランダム配置 数のぶん写しを撒く 位置は乱数なので広がりと量で確かめる"""

    def _scatter(self, **params: float | bool) -> Effect:
        return registry.require("scatter").create(**params)

    def test_more_copies_cover_more(self, draw: Callable[..., np.ndarray]) -> None:
        # 数を読まずにいると、ランダム配置の数を 12 にしても（AviUtl の作品から写した物も）
        # 2 のときと同じ量にしか撒かれない
        few = int(_lit(draw(self._scatter(count=2, span=200.0))).sum())
        many = int(_lit(draw(self._scatter(count=12, span=200.0))).sum())
        assert many > few

    def test_a_wider_span_spreads_further(self, draw: Callable[..., np.ndarray]) -> None:
        # 範囲を無視していると、どちらも同じ広がりになる
        narrow = _extent(draw(self._scatter(count=8, span=40.0)))
        wide = _extent(draw(self._scatter(count=8, span=200.0)))
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0]) + 20
        assert (wide[3] - wide[2]) > (narrow[3] - narrow[2]) + 20

    def test_the_copies_spread_in_both_directions(self, draw: Callable[..., np.ndarray]) -> None:
        """縦と横のどちらにも散らばる

        乱数の 2 つの数が揃っていると斜めの線の上にしか並ばず、
        縦の広がりが足りなくなる（実測で 4 分の 3 しか出なかった）
        """
        image = draw(self._scatter(count=16, span=160.0), width=20, height=20)
        left, right, top, bottom = _extent(image)
        assert (right - left) > 60
        assert (bottom - top) > 60

    def test_one_copy_with_no_span_is_the_original(self, draw: Callable[..., np.ndarray]) -> None:
        # 写しが 1 枚で範囲も 0 なら、元の絵がそのまま出る
        # ここが変わるなら、撒く前に絵を動かしてしまっている
        plain = draw()
        single = draw(self._scatter(count=1, span=0.0))
        assert int(np.abs(plain.astype(np.int16) - single.astype(np.int16)).max()) <= 2

    def test_turning_does_not_move_the_copies(self, draw: Callable[..., np.ndarray]) -> None:
        """回転しても写しは撒いた場所に留まる

        引く側の計算なので、写しの中心（絵の真ん中を引く所）は回しても動かない
        forward の式と取り違えると、写しが全体の中心のまわりを公転する
        """

        def middle(image: np.ndarray) -> tuple[float, float]:
            rows, columns = np.where(_lit(image))
            return (float(columns.mean()), float(rows.mean()))

        flat = draw(self._scatter(count=1, span=200.0, angle=0.0), width=40, height=12)
        turned = draw(self._scatter(count=1, span=200.0, angle=90.0), width=40, height=12)
        assert middle(turned) == pytest.approx(middle(flat), abs=2)
        # 回ってはいる 横長の帯が縦長になる
        assert _extent(turned)[3] - _extent(turned)[2] > _extent(flat)[3] - _extent(flat)[2]
