"""ランダムに登場する 3 種の効き方（AviUtl の登場もの）

値の意味は AviUtl2 に見本を描かせ、**時間を追って**測った 登場の効果は途中の絵が
全部なので、進み具合ごとの位置と量を並べて読む
ここに並ぶ数はその実測から取ったもので、落ちたときに疑うのはこちらの実装の側
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
from sashimono.effects.spec import ParamInput
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

WIDTH, HEIGHT = 200, 200
#: クリップの長さ（フレーム） 30fps なので 2 秒
DURATION = 60


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
    """真ん中の白い四角へ登場の効果を掛け、``frame`` の絵を返す"""

    def run(effect: Effect, frame: int, grey: float = 1.0) -> np.ndarray:
        # ``grey`` は下地の明るさ 明るさの足し算を見るときは 255 で
        # 頭打ちにならないよう、白ではなく灰色を使う
        source: GeneratedSource = SHAPE.create(
            shape="rect", width=60, height=20, color=(grey, grey, grey, 1.0)
        )
        project = Project.create(
            ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
        )
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(Clip(timeline_start=0, duration=DURATION, source=source, effects=(effect,)),),
        )
        project = project.with_timeline(
            project.timeline.__class__(rate=project.rate, tracks=(track,))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            return renderer.render(frame)
        finally:
            renderer.close()

    return run


def _lit(image: np.ndarray) -> np.ndarray:
    return image[:, :, :3].max(axis=2) > 24


def _middle(image: np.ndarray) -> float | None:
    """光っている所の縦の重心 何も無ければ ``None``"""
    rows, _ = np.where(_lit(image))
    return float(rows.mean()) if len(rows) else None


class TestFall:
    """ランダム間隔で落ちながら登場 上から降りてきて濃くなる"""

    def _fall(self, **params: ParamInput) -> Effect:
        base: dict[str, ParamInput] = {
            "effect_in": True,
            "effect_out": False,
            "effect_time": 1.0,
            "easing": "linear",
            "easing_mode": "in",
            "distance_": 60.0,
            "interval": 0.0,
        }
        base.update(params)
        return registry.require("inout_fall").create(**base)

    def test_it_comes_down_from_above(self, draw: Callable[..., np.ndarray]) -> None:
        """**上から**降りてくる

        AviUtl2 に 距離 200 を描かせると 200px 上から降りてきた
        符号を逆にすると、下から浮き上がってくる別の動きになる
        """
        # 始まりのフレームは不透明度 0 なので、少し進んだ所で見る
        start = _middle(draw(self._fall(), 5))
        home = _middle(draw(self._fall(), DURATION - 1))
        assert start is not None and home is not None
        assert start < home - 20, "上から降りてきていない"

    def test_it_lands_on_time(self, draw: Callable[..., np.ndarray]) -> None:
        # 時間 1 秒 ＝ 30 フレームで着く 真ん中（15 フレーム）では途中にいる
        half = _middle(draw(self._fall(), 15))
        landed = _middle(draw(self._fall(), 30))
        home = _middle(draw(self._fall(), DURATION - 1))
        assert half is not None and landed is not None and home is not None
        assert landed == pytest.approx(home, abs=2)
        assert half < landed - 5

    def test_it_fades_in_while_falling(self, draw: Callable[..., np.ndarray]) -> None:
        # 落ちながら濃くなる 実測でも 118 → 205 → 239 と上がった
        early = draw(self._fall(), 5)
        late = draw(self._fall(), 25)
        # 見るのは**黒へ重ねた後**の色 シェーダが触るのは不透明度だけなので、
        # 重ねる前の色を見ても差が出ない（どちらも白のまま）
        assert int(early[..., :3][_lit(early)].max()) < int(late[..., :3][_lit(late)].max())

    def test_it_rises_away_on_the_way_out(self, draw: Callable[..., np.ndarray]) -> None:
        """退場でも動いて消える

        登場だけを見て退場の進み具合を無視すると、クリップの終わりで
        絵が元の位置に残り続ける（見送るはずの所でずっと出たまま）
        """
        leaving = self._fall(effect_in=False, effect_out=True)
        settled = _middle(draw(leaving, DURATION // 2))
        assert settled is not None

        # 退場の半ば まだ見えていて、かつ上へ戻っていること
        # 「消えていれば通す」形にすると、薄くなるだけで動かない実装でも通る
        halfway = draw(leaving, DURATION - DURATION // 4)
        assert _lit(halfway).any(), "退場の半ばで消えている"
        assert (_middle(halfway) or 0) < settled - 5, "上へ戻っていない"

    def test_a_delay_holds_it_back(self, draw: Callable[..., np.ndarray]) -> None:
        # 間隔 を入れると落ち始めが遅れる 遅れが効かないと同じ絵になる
        prompt = _middle(draw(self._fall(interval=0.0), 20))
        waited = _middle(draw(self._fall(interval=1.0), 20))
        assert prompt is not None
        assert waited is None or waited < prompt


class TestBlink:
    """点滅して登場 半端な濃さは通らず、点くか消えるかのどちらか"""

    def _blink(self, **params: ParamInput) -> Effect:
        base: dict[str, ParamInput] = {
            "effect_in": True,
            "effect_out": False,
            "effect_time": 1.0,
            "easing": "linear",
            "easing_mode": "in",
            "interval": 5.0,
            "even": True,
        }
        base.update(params)
        return registry.require("inout_blink").create(**base)

    def test_it_turns_on_and_off(self, draw: Callable[..., np.ndarray]) -> None:
        # 点滅しないまま出ると、ただのカットインになる
        seen = {bool(_lit(draw(self._blink(), frame)).any()) for frame in range(0, 30)}
        assert seen == {True, False}, "点滅していない"

    def test_it_is_solid_once_it_has_arrived(self, draw: Callable[..., np.ndarray]) -> None:
        # 時間を過ぎたら点滅しない ここが残ると、ずっとちらつく
        assert all(bool(_lit(draw(self._blink(), frame)).any()) for frame in range(35, 60, 5))

    def test_it_is_never_half_lit(self, draw: Callable[..., np.ndarray]) -> None:
        """点いているときは元の明るさのまま

        薄く出す作りにすると、点滅ではなく溶け込み（フェード）になる
        AviUtl2 の実物も、見えるときは 239 のままだった
        """
        full = draw(self._blink(), DURATION - 1)
        top = int(full[_lit(full)].max())
        for frame in range(0, 30):
            image = draw(self._blink(), frame)
            if _lit(image).any():
                assert int(image[_lit(image)].max()) == pytest.approx(top, abs=3)

    def test_an_even_interval_repeats(self, draw: Callable[..., np.ndarray]) -> None:
        """一定にする を付けると、間隔ごとにきっちり入れ替わる

        AviUtl2 の実物（点滅間隔 5）も ``.....#####.....#####`` と、
        進み具合に関係なく同じ長さで切り替わっていた
        """
        lit = [bool(_lit(draw(self._blink(interval=5.0), frame)).any()) for frame in range(0, 20)]
        assert lit[0:5] == lit[10:15]
        assert lit[0:5] != lit[5:10]

    def test_without_the_even_flag_the_lit_time_grows(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        """一定にする を外すと、進むほど点いている時間が増える

        AviUtl2 の実物（点滅間隔 1）は
        ``..........#....#..#..#..#.#.#.#.##.##.####.######`` と、
        **乱数ではなく決まった並び**で点く時間が伸びていく
        一定の点滅のままにすると、最後まで同じ割合でちらつく
        """
        # AviUtl2 に 時間 1.0・点滅間隔 1 を描かせた 30 フレームぶんの並び
        # （60fps の実物を読んだものと同じ式から出る 30fps ぶん）
        # 回数だけを数える形にすると、別の並びや乱数でも通ってしまう
        blink = self._blink(interval=1.0, even=False)
        lit = "".join("#" if _lit(draw(blink, frame)).any() else "." for frame in range(0, 30))
        assert lit == ".......#..#..#.#.##.##.#######", lit

    def test_the_blink_leaves_on_the_way_out(self, draw: Callable[..., np.ndarray]) -> None:
        """退場でも点滅しながら消える

        クリップ先頭からのフレーム数で測ると、退場の頃には毎フレーム点いて
        絵が消えない 測るのは「終わりまでの残り」の側
        """
        # 退場のあいだの並びをそのまま固定する 真ん中と最後だけを見る形にすると、
        # 最後に一度消えるだけの実装（点滅しない）でも通ってしまう
        #
        # 入りの並びをひっくり返した形（進むほど点いている時間が減る）
        leaving = self._blink(interval=1.0, even=False, effect_in=False, effect_out=True)
        lit = "".join(
            "#" if _lit(draw(leaving, frame)).any() else "." for frame in range(30, DURATION)
        )
        assert lit == "########.##.##.#.#..#..#......", lit
        assert lit.count("#") > 5, "退場のあいだに点いていない"
        assert lit.endswith("."), "終わりで消えていない"


class TestRandomDirection:
    """ランダム方向から登場 画面の外から飛んでくる"""

    def _fly(self, **params: ParamInput) -> Effect:
        base: dict[str, ParamInput] = {
            "effect_in": True,
            "effect_out": False,
            "effect_time": 1.0,
            "easing": "linear",
            "easing_mode": "in",
            "spin": 0.0,
            "light": 0.0,
        }
        base.update(params)
        return registry.require("inout_random_direction").create(**base)

    def test_it_is_away_at_the_start_and_home_at_the_end(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 登場の頭で既に画面に出ていると飛んでくる動きにならず、着かないと最後まで画面から消えたまま
        assert not _lit(draw(self._fly(), 0)).any(), "始めから画面に居座っている"
        assert _lit(draw(self._fly(), DURATION - 1)).any(), "着いていない"

    def test_the_seed_changes_the_direction(self, draw: Callable[..., np.ndarray]) -> None:
        # 向きが種で変わらないと、並べたときに全部同じ方向から飛んでくる
        middles = {
            tuple(np.round(np.mean(np.where(_lit(draw(self._fly(seed=seed), 20))), axis=1), 1))
            if _lit(draw(self._fly(seed=seed), 20)).any()
            else None
            for seed in (0, 1, 2, 3)
        }
        assert len(middles) > 1

    def test_the_light_brightens_the_flight(self, draw: Callable[..., np.ndarray]) -> None:
        """ライト は**飛んでいるあいだ**の明るさの足し算 着いたら効かない

        着いた後だけを見ると、ライトを外した実装でも通ってしまう
        （着いた後は効かないのが正しい姿なので、差が出ないのが当たり前）
        """
        # 下地は灰色 白だと 255 で頭打ちになり、足しても増えないので
        # ライトを外した実装でも通ってしまう
        flying_plain = int(draw(self._fly(light=0.0), 20, grey=0.3)[..., :3].max())
        flying_lit = int(draw(self._fly(light=80.0), 20, grey=0.3)[..., :3].max())
        assert flying_lit > flying_plain + 10, "飛んでいるあいだに効いていない"

        # 着いた後は差が出ない
        landed_plain = draw(self._fly(light=0.0), DURATION - 1)
        landed_lit = draw(self._fly(light=80.0), DURATION - 1)
        assert int(np.abs(landed_plain.astype(np.int16) - landed_lit.astype(np.int16)).max()) <= 2
