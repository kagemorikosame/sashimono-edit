"""AviUtl スクリプトが実際に描画へ届くこと

スクリプトが書き換えた位置・回転・拡大が、GPU の合成結果にそのまま出るかを
画素で見る ここが通れば、配布されているアニメーション効果がそのまま動く土台が
できている
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from sashimono.compat.aviutl.catalog import ScriptCatalog, set_script_catalog
from sashimono.core.commands import AddClip, AddEffect, AddTrack
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.effects.sources import FILTER
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer, RenderQuality

#: 位置と回転をスライダーで動かすだけのスクリプト
MOVE = """--track0:X,-500,500,0,1
--track1:回転,-360,360,0,1
--track2:拡大,10,400,100,1
obj.ox = obj.track0
obj.rz = obj.track1
obj.zoom = obj.track2 / 100
"""

#: 残像 1 回の実行で何枚も描くスクリプトの代表
TRAIL = """--track0:枚数,1,8,4,1
for i = 0, obj.track0 - 1 do
  obj.draw(i * 30, 0, 0, 1, 1 - i * 0.2)
end
"""

#: 自分の大きさで動く AviUtl の obj.w は「オブジェクト自身の幅」
OWN_SIZE = """obj.ox = obj.w
"""

#: 板を傾ける X 軸・Y 軸の回転と奥行き
TILT = """--track0:X回転,-360,360,0,1
--track1:Y回転,-360,360,0,1
--track2:奥行き,-2000,2000,0,1
obj.rx = obj.track0
obj.ry = obj.track1
obj.oz = obj.track2
"""

#: 四隅を決めて貼る 上が狭い台形
#: 図形の絵は画面と同じ大きさで、真ん中に四角がある その四角だけを切り出して貼る
POLY = """local l = (obj.w - 40) / 2
local t = (obj.h - 40) / 2
obj.drawpoly(-10,-20,0, 10,-20,0, 40,20,0, -40,20,0, l,t, l+40,t, l+40,t+40, l,t+40)
"""

#: 絵に何もしないスクリプト 積んだ前と後で絵が変わらないことを見る
NOTHING = """obj.ox = 0
"""

SCREEN = (320, 240)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def catalog() -> ScriptCatalog:
    created = ScriptCatalog(roots=())
    created.add_text("aviutl:試験.anm:移動", MOVE)
    created.add_text("aviutl:試験.anm:残像", TRAIL)
    created.add_text("aviutl:試験.anm:傾き", TILT)
    created.add_text("aviutl:試験.anm:自分の幅", OWN_SIZE)
    created.add_text("aviutl:試験.anm:四隅", POLY)
    created.add_text("aviutl:試験.anm:何もしない", NOTHING)
    set_script_catalog(created)
    return created


def build(identifier: str, size: float = 40.0, **params: float) -> Project:
    """``size`` 四方（既定 40）の白い四角に、スクリプトを 1 本積んだプロジェクト"""
    width, height = SCREEN
    project = Project.create(ProjectSettings(width=width, height=height, frame_rate=FrameRate(30)))
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = AddTrack(track).apply(project)

    shape = GeneratedSource(kind="shape", params={"shape": "rect"})
    clip = Clip(timeline_start=0, duration=30, source=shape)
    # 図形の既定は 400x400 画面より小さくして位置を見やすくする
    clip = Clip(
        timeline_start=0,
        duration=30,
        source=GeneratedSource(
            kind="shape",
            params={"shape": "rect", "width": size, "height": size},  # type: ignore[dict-item]
        ),
    )
    project = AddClip(track.id, clip).apply(project)

    definition = registry.get(identifier)
    assert definition is not None
    return AddEffect(clip.id, definition.create(**params)).apply(project)


def bounds(image: np.ndarray) -> tuple[int, int, int, int]:
    """明るい画素の外接矩形 ``(左, 右, 上, 下)``"""
    bright = image[..., :3].max(axis=2) > 100
    ys, xs = np.nonzero(bright)
    assert len(xs), "何も描かれていない"
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def render(project: Project, context: OffscreenGLContext) -> np.ndarray:
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(0)
    finally:
        renderer.close()


class TestTransform:
    def test_without_a_script_the_shape_is_centred(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 値を 0 のまま積んだだけで位置がずれると、配布の効果を当てた途端に置いた物が中央から外れる
        del catalog
        left, right, top, bottom = bounds(render(build("aviutl:試験.anm:移動"), gl_context))
        assert (left + right) // 2 == pytest.approx(SCREEN[0] // 2, abs=2)
        assert (top + bottom) // 2 == pytest.approx(SCREEN[1] // 2, abs=2)

    def test_ox_moves_the_object(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # obj.ox が描画に届かないと、配布の効果で横へ動くはずの物がその場に止まったまま書き出される
        del catalog
        left, right, _, _ = bounds(render(build("aviutl:試験.anm:移動", track0=80), gl_context))
        assert (left + right) // 2 == pytest.approx(SCREEN[0] // 2 + 80, abs=2)

    def test_rz_rotates_around_the_centre(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 回転の中心が物の中央からずれると、回すスクリプトを当てた物が回りながら横へずれていく
        del catalog
        left, right, top, bottom = bounds(
            render(build("aviutl:試験.anm:移動", track1=45), gl_context)
        )
        # 40px の正方形を 45 度回すと、外接矩形は約 56px になる
        assert right - left == pytest.approx(56, abs=3)
        assert bottom - top == pytest.approx(56, abs=3)
        assert (left + right) // 2 == pytest.approx(SCREEN[0] // 2, abs=2)

    def test_zoom_scales_the_object(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # obj.zoom が届かないと、拡大・縮小で出入りする配布の効果が大きさを変えずに出る
        del catalog
        left, right, _, _ = bounds(render(build("aviutl:試験.anm:移動", track2=200), gl_context))
        assert right - left == pytest.approx(80, abs=3)

    def test_rotation_does_not_leak_into_the_next_draw(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 回転の状態が描画係に残ると、回すスクリプトを使った後に描いた別のクリップまで傾いて出る
        del catalog
        # 回転を掛けた直後に、掛けていないものを描いても回らないこと
        render(build("aviutl:試験.anm:移動", track1=30), gl_context)
        left, right, top, bottom = bounds(render(build("aviutl:試験.anm:移動"), gl_context))
        assert right - left == pytest.approx(40, abs=2)
        assert bottom - top == pytest.approx(40, abs=2)


class TestObjectSize:
    def test_obj_w_is_the_object_not_the_screen(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 画面と同じ大きさの絵を渡すと obj.w が画面の幅になり、配布スクリプトの
        # 位置の計算が画面の幅ぶん飛ぶ
        del catalog
        left, right, _, _ = bounds(render(build("aviutl:試験.anm:自分の幅", size=40), gl_context))
        assert (left + right) // 2 == pytest.approx(SCREEN[0] // 2 + 40, abs=2)
        assert right - left == pytest.approx(40, abs=2)


class TestDepth:
    def test_turning_about_y_narrows_the_board(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 以前は rx / ry を読み捨てていて、板を回すスクリプトが平らなまま動かなかった
        del catalog
        left, right, top, bottom = bounds(
            render(build("aviutl:試験.anm:傾き", track1=60), gl_context)
        )
        assert right - left == pytest.approx(20, abs=4)
        assert bottom - top >= 38

    def test_a_tilted_board_is_a_trapezoid(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 遠近が無いと、X 軸で倒しても上下の幅が同じ長方形のまま縮むだけになる
        del catalog
        image = render(build("aviutl:試験.anm:傾き", size=160.0, track0=60), gl_context)
        _, _, top, bottom = bounds(image)
        bright = image[..., :3].max(axis=2) > 100
        top_width = int(bright[top + 1].sum())
        bottom_width = int(bright[bottom - 1].sum())
        assert top_width < bottom_width

    def test_depth_shrinks_the_object(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 奥へ置いても大きさが変わらないと、奥行きの演出が消える
        del catalog
        left, right, _, _ = bounds(render(build("aviutl:試験.anm:傾き", track2=1024), gl_context))
        assert right - left == pytest.approx(20, abs=3)

    def test_drawpoly_reaches_the_screen(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # 四隅どおりの台形が描かれること（上 20px、下 80px）
        del catalog
        image = render(build("aviutl:試験.anm:四隅"), gl_context)
        left, right, top, bottom = bounds(image)
        assert right - left == pytest.approx(80, abs=3)
        assert bottom - top == pytest.approx(40, abs=3)
        bright = image[..., :3].max(axis=2) > 100
        assert int(bright[top + 1].sum()) < int(bright[bottom - 1].sum())


class TestMultipleDraws:
    def test_a_script_can_draw_several_times(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # obj.draw の 2 回目以降を捨てると、残像のように何枚も描く配布スクリプトが 1 枚しか出さない
        del catalog
        image = render(build("aviutl:試験.anm:残像", track0=3), gl_context)
        left, right, _, _ = bounds(image)
        # 0, 30, 60 の 3 枚 左端は中央 -20、右端は中央 +60+20
        assert left == pytest.approx(SCREEN[0] // 2 - 20, abs=2)
        assert right == pytest.approx(SCREEN[0] // 2 + 80, abs=3)

    def test_the_alpha_of_each_draw_is_honoured(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog
    ) -> None:
        # obj.draw ごとの不透明度を無視すると、残像が後ろまで同じ濃さで出て、ただの複製に見える
        del catalog
        image = render(build("aviutl:試験.anm:残像", track0=3), gl_context)
        middle = SCREEN[1] // 2
        first = int(image[middle, SCREEN[0] // 2, 0])
        third = int(image[middle, SCREEN[0] // 2 + 60, 0])
        # 後ろの枚ほど薄くなる
        assert first > third > 0


def render_at(project: Project, context: OffscreenGLContext, divisor: int) -> np.ndarray:
    renderer = FrameRenderer(project, context=context, quality=RenderQuality(divisor))
    try:
        return renderer.render(0)
    finally:
        renderer.close()


@pytest.mark.parametrize("divisor", [2, 4])
class TestALighterPreview:
    """画質を落としたプレビューでも、スクリプトの絵が書き出しを縮めた所に出る（Issue #151）

    スクリプトの値（obj.ox や obj.draw の位置）はどれが画素かを定義から読めない
    スクリプトは画面の画素で走らせ、返った位置と大きさを描く直前に縮める 合成の画素で
    走らせると、1/2 画質で位置も残像の間隔も 2 倍に出る
    """

    @pytest.mark.parametrize(
        ("identifier", "params"),
        [
            ("aviutl:試験.anm:移動", {"track0": 80.0, "track1": 30.0, "track2": 150.0}),
            ("aviutl:試験.anm:残像", {"track0": 3.0}),
            ("aviutl:試験.anm:傾き", {"track1": 40.0, "track2": 200.0}),
            ("aviutl:試験.anm:自分の幅", {}),
            ("aviutl:試験.anm:四隅", {}),
        ],
    )
    def test_the_drawing_is_the_export_shrunk(
        self,
        gl_context: OffscreenGLContext,
        catalog: ScriptCatalog,
        divisor: int,
        identifier: str,
        params: dict[str, float],
    ) -> None:
        # 縮めた合成の画素でスクリプトを走らせると、軽くしたプレビューで位置や残像の間隔が
        # 2 倍や 4 倍に出て、書き出しと見た目が食い違う
        del catalog
        project = build(identifier, **params)
        full = bounds(render(project, gl_context))
        light = bounds(render_at(project, gl_context, divisor))
        # 外接矩形の端を縮めた所と、縮めた合成の丸めの幅で重なる
        for got, want in zip(light, full, strict=True):
            assert got == pytest.approx(want / divisor, abs=1.5), (light, full)

    def test_a_script_on_a_filter_keeps_the_screen_sharp(
        self, gl_context: OffscreenGLContext, catalog: ScriptCatalog, divisor: int
    ) -> None:
        # フィルタのクリップは描き終えた合成の画素の絵をスクリプトへ渡す 画面の画素へ
        # 滑らかに引き伸ばしてから縮め戻すと、何もしないスクリプトでも下の絵がぼける
        # 升目を並べるだけに伸ばせば、同じ所へ縮め戻したときに元の画素へ戻る
        del catalog
        project = build("aviutl:試験.anm:移動")
        track = Track(kind=TrackKind.VIDEO, name="V2")
        project = AddTrack(track).apply(project)
        filter_clip = Clip(timeline_start=0, duration=30, source=FILTER.create())
        project = AddClip(track.id, filter_clip).apply(project)
        without = render_at(project, gl_context, divisor)
        definition = registry.get("aviutl:試験.anm:何もしない")
        assert definition is not None
        project = AddEffect(filter_clip.id, definition.create()).apply(project)
        with_script = render_at(project, gl_context, divisor)
        difference = np.abs(with_script.astype(int) - without.astype(int))[..., :3]
        assert without[..., :3].max() > 200
        # 読み戻しと載せ直しの 8 ビットの丸めだけが残る
        assert difference.max() <= 2
