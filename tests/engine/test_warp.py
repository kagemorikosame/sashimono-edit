"""絵を歪めるエフェクトの効き方（AviUtl の変形まわり）

値の意味は AviUtl2 に 1 つずつ動かした見本を描かせて測った ここに並ぶ数は
その実測から取ったもので、落ちたときに疑うのはこちらの実装の側
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

WIDTH, HEIGHT = 200, 200
#: 下地の四角の一辺 真ん中に置く
SQUARE = 60


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


def _extent(image: np.ndarray) -> tuple[int, int, int, int]:
    """光っている所の 左・右・上・下（画面の座標 下が大きい）"""
    lit = image[:, :, :3].max(axis=2) > 24
    rows, columns = np.where(lit)
    if not len(rows):
        return (0, 0, 0, 0)
    return (int(columns.min()), int(columns.max()), int(rows.min()), int(rows.max()))


class TestExpandArea:
    """領域拡張 入れ物を広げると、絵は真ん中に残るので**半分ずれる**"""

    def test_expanding_the_top_pushes_the_picture_down(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # AviUtl2 に 上=200 を描かせると、絵は下へ 100（半分）動いた
        # ずれの向きを逆にすると、配布物の飾り枠が上下逆の位置に出る
        before = _extent(draw())
        after = _extent(draw(registry.require("expand_area").create(top=40.0)))
        assert after[2] - before[2] == pytest.approx(20, abs=2)
        assert after[0] == pytest.approx(before[0], abs=2)

    def test_expanding_the_right_pushes_the_picture_left(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 右を広げて絵が左へ半分ずれないと、領域拡張を使う配布の飾り枠が
        # AviUtl2 と左右で違う所に出る
        before = _extent(draw())
        after = _extent(draw(registry.require("expand_area").create(right=60.0)))
        assert before[0] - after[0] == pytest.approx(30, abs=2)

    def test_expanding_evenly_does_not_move_the_picture(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 四方を同じだけ広げれば真ん中は動かない ここがずれるなら符号が片側だけ違う
        before = _extent(draw())
        after = _extent(
            draw(
                registry.require("expand_area").create(top=40.0, bottom=40.0, left=40.0, right=40.0)
            )
        )
        assert after == before

    def test_fill_paints_the_new_area_with_the_edge(self, draw: Callable[..., np.ndarray]) -> None:
        # 塗りつぶし は広げた所を縁の色で埋める（#190） 埋めないと .exa の 領域拡張 は
        # 塗りつぶしを付けても入れ物が透明に広がるだけで、スクリプトから広げたときと食い違う
        effect = registry.require("expand_area").create(top=40.0, left=20.0, fill=True)
        lit = _extent(draw(effect))
        plain = _extent(draw())
        # 絵は右下へ半分ずれ（10, 20）、埋めた所は左へ 20・上へ 40 広がる
        assert lit[0] == pytest.approx(plain[0] - 10, abs=2)
        assert lit[1] == pytest.approx(plain[1] + 10, abs=2)
        assert lit[2] == pytest.approx(plain[2] - 20, abs=2)
        assert lit[3] == pytest.approx(plain[3] + 20, abs=2)

    def test_fill_stays_inside_the_new_area(self, draw: Callable[..., np.ndarray]) -> None:
        # 広げた量の外まで縁の色を伸ばすと、画面の端まで帯が伸びる
        effect = registry.require("expand_area").create(right=30.0, fill=True)
        image = draw(effect)
        lit = image[:, :, :3].max(axis=2) > 24
        assert int(lit.sum()) == pytest.approx((SQUARE + 30) * SQUARE, rel=0.05)


class TestMirror:
    def _mirror(self, **params: float | str) -> Effect:
        return registry.require("mirror").create(**params)

    def test_the_reflection_appears_below(self, draw: Callable[..., np.ndarray]) -> None:
        # 鏡像が下に出ないと、床に映る反射を付けたテロップの下に何も映らない
        # 上へはみ出すと、文字の上に逆さの字が重なる
        before = _extent(draw())
        after = _extent(draw(self._mirror(side="bottom")))
        assert after[3] > before[3] + 20, "下に鏡像が出ていない"
        assert after[2] == before[2], "上へはみ出している"

    def test_the_reflection_can_go_to_the_right(self, draw: Callable[..., np.ndarray]) -> None:
        # 向きを読まずにいつも下へ写すと、右側に映す設定にしても反射が下に出る
        before = _extent(draw())
        after = _extent(draw(self._mirror(side="right")))
        assert after[1] > before[1] + 20
        assert after[3] == before[3]

    def test_the_gap_pushes_the_reflection_twice_as_far(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """境目調整 を足すと鏡像は**その倍**離れる

        折り返す線が動くので、鏡像はその倍動く AviUtl2 で 40 を入れると
        80px 離れた 1 倍で写すと、境目の空きが半分になる
        """
        # 小さめの四角で試す 大きいと鏡像が絵の下端で切れて、
        # 離れた量が測れない（境目調整 0 でも既に端まで届いてしまう）
        close = _extent(draw(self._mirror(side="bottom", gap=0.0), width=30, height=30))
        far = _extent(draw(self._mirror(side="bottom", gap=10.0), width=30, height=30))
        assert far[3] - close[3] == pytest.approx(20, abs=3)

    def test_full_transparency_hides_the_reflection(self, draw: Callable[..., np.ndarray]) -> None:
        # 透明度 100 でも鏡像が残ると、反射を消すつもりで透明にしても薄い逆さの絵が下に残る
        before = _extent(draw())
        after = _extent(draw(self._mirror(side="bottom", opacity=100.0)))
        assert after == before

    def test_the_falloff_thins_the_far_end(self, draw: Callable[..., np.ndarray]) -> None:
        # 減衰 を上げると、線から遠い方が先に消える
        hard = draw(self._mirror(side="bottom", falloff=0.0))
        soft = draw(self._mirror(side="bottom", falloff=100.0))
        assert _extent(soft)[3] < _extent(hard)[3]


class TestDisplacementMap:
    def _displace(self, **params: float | str) -> Effect:
        base: dict[str, float | str] = {"size": 120.0, "blur": 5.0}
        base.update(params)
        return registry.require("displacement_map").create(**base)

    def test_only_what_is_inside_the_map_moves(self, draw: Callable[..., np.ndarray]) -> None:
        """マップの中だけが動き、外は**1 画素も**変わらない

        絵ぜんたいを動かす実装でも「絵が変わった」だけなら通ってしまうので、
        中と外を別々に確かめる

        下地は横に長い帯（160x20） 帯の上下の縁はマップ（直径 60）の中に、
        左右の端は外にある 帯を**縦へ**ずらせば、中では縁が動いて差が出る
        （横へずらしても、一様な帯は動いたことが絵に出ない）
        """
        before = draw(width=160, height=20)
        after = draw(self._displace(size=60.0, blur=0.0, move_y=20.0), width=160, height=20)

        rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
        # マップは絵の中身の真ん中（ここでは画面の真ん中）に置かれる
        radius = np.hypot(columns - WIDTH / 2.0, rows - HEIGHT / 2.0)
        outside = radius > 40.0
        inside = radius < 25.0

        difference = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
        assert int(difference[outside].max()) <= 2, "マップの外まで動いている"
        assert int(difference[inside].max()) > 40, "マップの中が動いていない"

    def test_a_map_with_no_movement_changes_nothing(self, draw: Callable[..., np.ndarray]) -> None:
        # ずらす量が 0 なら絵は変わらない ここが変わるならマップの中心がずれている
        plain = draw()
        still = draw(self._displace(move_x=0.0, move_y=0.0))
        assert int(np.abs(plain.astype(np.int16) - still.astype(np.int16)).max()) <= 2

    def test_the_map_size_decides_how_much_moves(self, draw: Callable[..., np.ndarray]) -> None:
        """マップを広げるほど、動く所が増える

        大きさを無視して全体を動かす実装だと、どちらも同じだけ動いてしまう
        """
        before = draw(width=160, height=20)

        def moved(size: float) -> int:
            after = draw(self._displace(size=size, blur=0.0, move_y=20.0), width=160, height=20)
            difference = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
            return int((difference > 8).sum())

        assert moved(100.0) > moved(50.0) * 1.5


class TestTheExpandedBoxReachesLaterEffects:
    """領域拡張のあとに積んだ効果は、**広げた後**の範囲で動く

    AviUtl の 領域拡張 → ミラー は、広げたぶんだけ鏡像を離すための並べ方
    広げる前の範囲のまま後ろが動くと、鏡像が絵に食い込む
    """

    def test_the_mirror_folds_at_the_expanded_edge(self, draw: Callable[..., np.ndarray]) -> None:
        # 四方を同じだけ広げる 片側だけだと絵そのものもずれて、
        # 折り返す線が動いたのか絵が動いたのか分からなくなる
        mirror = registry.require("mirror").create(side="bottom")
        expand = registry.require("expand_area").create(
            top=20.0, bottom=20.0, left=20.0, right=20.0
        )
        plain = _extent(draw(mirror, width=30, height=30))
        expanded = _extent(draw(expand, mirror, width=30, height=30))
        assert expanded[3] > plain[3] + 10, "広げた後の縁で折り返していない"

    def test_without_the_mark_nothing_moves(self, draw: Callable[..., np.ndarray]) -> None:
        # 印を付けていない効果は範囲を触らない ここが動くなら、
        # 全部のエフェクトが範囲を広げてしまっている
        mirror = registry.require("mirror").create(side="bottom")
        blur = registry.require("blur").create(radius=1.0)
        assert _extent(draw(blur, mirror, width=30, height=30))[3] == pytest.approx(
            _extent(draw(mirror, width=30, height=30))[3], abs=3
        )
