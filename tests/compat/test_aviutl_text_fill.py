"""AviUtl2 のテキストのすぐ後ろの「グラデーション」を、テキストの塗りとして読む（#312）

決まりは AviUtl2 v2.1.6a に描かせて確かめた（``sashimono.compat.aviutl.text_fill`` の説明）
ここでは同じ形のエイリアスをコードで作って見る（配布物はリポジトリに入れない）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.aviutl.exo import load_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.text_fill import text_gradient_fill
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import AnimatedValue, Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

#: 手元のエイリアス（Premiere自作字幕）と同じ形 テキストのすぐ後ろにグラデーション、その後ろに縁取り
ALIAS = """[Object]
frame=0,59
[Object.0]
effect.name=テキスト
サイズ=100.00
字間=0.00
行間=0.00
表示速度=0.00
フォント=Yu Gothic UI
文字色=ffffff
影・縁色=000000
文字装飾={decoration}
文字揃え={align}
B=0
I=0
テキスト=■■■■
文字毎に個別オブジェクト=0
自動スクロール=0
移動座標上に表示=0
オブジェクトの長さを自動調節=0
[Object.1]
effect.name={first}
強さ=100.0
中心X=0.0
中心Y=0.0
角度={angle}
幅=100
合成モード={blend}
形状=線形
開始色=ff0000
終了色=0000ff
[Object.2]
effect.name=縁取り
サイズ=6
ぼかし=0
縁色=000000
パターン画像=
[Object.3]
effect.name=標準描画
X=0.00
Y=0.00
Z=0.00
Group=1
中心X=0.00
中心Y=0.00
中心Z=0.00
Group3=1
X軸回転=0.00
Y軸回転=0.00
Z軸回転=0.00
Group2=1
拡大率=100.000
縦横比=0.000
透明度=0.00
合成モード=通常
"""


def mapped(tmp_path: Path, **values: str) -> Clip:
    filled = {
        "decoration": "標準文字",
        "align": "中央揃え[下]",
        "first": "グラデーション",
        "angle": "0.00",
        "blend": "通常",
        **values,
    }
    path = tmp_path / "見本.object"
    path.write_text(ALIAS.format(**filled), encoding="utf-8")
    (obj,) = load_exo(path).objects
    found = map_object(obj, FrameRate(60))
    assert found is not None
    return found.clip


class TestWhatIsFolded:
    def test_the_gradient_right_after_the_text_becomes_its_fill(self, tmp_path: Path) -> None:
        # 塗りへ写さないと、読み込んだ字幕のグラデーションを設定パネルのバーで直せない
        # 縁取りより前にあるので、縁は塗らない（縁取りは並びに残る）
        clip = mapped(tmp_path)
        assert clip.source is not None
        params = clip.source.params
        assert params["fill_type"] == "linear"
        assert params["fill_span"] == "pixels"
        assert params["fill_length"] == AnimatedValue(100.0)
        assert [effect.kind for effect in clip.effects if not effect.fixed] == ["border"]

    def test_the_s_curve_is_kept(self) -> None:
        # 直線で混ぜると、AviUtl2 の帯より端の色が早く崩れ、
        # 幅の広いグラデーションの中ほどが違う色になる
        effect = registry.require("gradient").create(
            start_color=(1.0, 0.0, 0.0, 1.0), end_color=(0.0, 0.0, 1.0, 1.0), span=800.0
        )
        params = text_gradient_fill(effect, TEXT.create(layout="aviutl"))
        assert params is not None
        stops = params["fill_gradient"]
        assert isinstance(stops, tuple)
        # 4 つ目の止まり（位置 0.2）の赤は、S 字なら 1 - 0.104 直線なら 0.8
        position, red = stops[15], stops[16]
        assert position == pytest.approx(0.2)
        assert red == pytest.approx(1.0 - (0.2 * 0.2 * (3.0 - 0.4)))

    def test_the_centre_shifts_along_the_band(self) -> None:
        # 中心のずらしを捨てると、帯をずらした字幕の色の切り替わりが真ん中に戻る
        effect = registry.require("gradient").create(angle=90.0, center_x=150.0)
        params = text_gradient_fill(effect, TEXT.create(layout="aviutl"))
        assert params is not None
        # 角度 90 の帯は右から左へ進むので、右へ 150 ずらすと向きの上では -150
        shift = params["fill_shift"]
        assert isinstance(shift, AnimatedValue)
        assert shift.static == pytest.approx(-150.0)

    @pytest.mark.parametrize(
        "values",
        [
            # 加算などは字の色との混ぜ方が違う
            {"blend": "加算"},
            # 文字装飾の縁と影まで AviUtl2 は塗る 字の塗りへ移すと縁が塗られなくなる
            {"decoration": "縁取り文字"},
        ],
        ids=["加算", "縁取り文字"],
    )
    def test_what_cannot_be_the_same_picture_stays_an_effect(
        self, tmp_path: Path, values: dict[str, str]
    ) -> None:
        # 違う絵になる物まで塗りにすると、読み込んだ字幕の色が AviUtl2 と変わる
        clip = mapped(tmp_path, **values)
        assert clip.source is not None
        assert clip.source.params.get("fill_type") != "linear"
        assert any(effect.kind == "gradient" for effect in clip.effects)

    def test_a_gradient_after_another_filter_stays_an_effect(self, tmp_path: Path) -> None:
        # 前のフィルタが変えた絵（縁も）を AviUtl2 は塗る 字の塗りへ移すと縁が塗られない
        alias = ALIAS.replace("[Object.1]", "[Object.9]").replace("[Object.2]", "[Object.1]")
        alias = alias.replace("[Object.9]", "[Object.2]")
        path = tmp_path / "後ろ.object"
        path.write_text(
            alias.format(
                decoration="標準文字",
                align="中央揃え[下]",
                first="グラデーション",
                angle="0.00",
                blend="通常",
            ),
            encoding="utf-8",
        )
        (obj,) = load_exo(path).objects
        found = map_object(obj, FrameRate(60))
        assert found is not None and found.clip.source is not None
        assert found.clip.source.params.get("fill_type") != "linear"


#: テキストの後ろに積むフィルタ 1 つ目の後ろに、2 つ目のフィルタを挟んでグラデーションを置く
_GRADIENT_BLOCK = """[Object.{index}]
effect.name=グラデーション
強さ=100.0
中心X=0.0
中心Y=0.0
角度=0.00
幅=100
合成モード=通常
形状=線形
開始色={start}
終了色=0000ff
"""

_TEXT_HEAD = ALIAS.split("[Object.1]")[0].format(
    decoration="標準文字", align="中央揃え[下]", first="", angle="0.00", blend="通常"
)


def _stacked(tmp_path: Path, *blocks: str) -> Clip:
    """テキストの後ろに ``blocks`` を順に積んだエイリアスを読む"""
    body = _TEXT_HEAD + "".join(
        block.replace("[Object.?]", f"[Object.{index}]") for index, block in enumerate(blocks, 1)
    )
    path = tmp_path / "積んだ.object"
    path.write_text(body, encoding="utf-8")
    (obj,) = load_exo(path).objects
    found = map_object(obj, FrameRate(60))
    assert found is not None
    return found.clip


def _gradient(start: str) -> str:
    return _GRADIENT_BLOCK.format(index="?", start=start)


class TestOnlyTheFirstFilter:
    def test_a_second_gradient_stays_an_effect(self, tmp_path: Path) -> None:
        # 2 つ目も塗りへ写すと 1 つ目の塗りを上書きし、1 つ目のグラデーションが黙って消える
        clip = _stacked(tmp_path, _gradient("ff0000"), _gradient("00ff00"))
        assert clip.source is not None
        stops = clip.source.params["fill_gradient"]
        assert isinstance(stops, tuple)
        assert stops[1:4] == (1.0, 0.0, 0.0)
        (kept,) = [effect for effect in clip.effects if effect.kind == "gradient"]
        assert kept.params["start_color"] == (0.0, 1.0, 0.0, 1.0)

    @pytest.mark.parametrize(
        "between",
        [
            # 読めないフィルタ 何をしているか分からない絵をグラデーションは塗る
            "[Object.?]\neffect.name=知らないフィルタ\n",
            # オブジェクト分割 分けた後の絵に掛かる
            "[Object.?]\neffect.name=オブジェクト分割\n横分割数=2\n縦分割数=2\n",
        ],
        ids=["読めないフィルタ", "オブジェクト分割"],
    )
    def test_a_gradient_after_another_entry_stays_an_effect(
        self, tmp_path: Path, between: str
    ) -> None:
        # 間に何かを挟んだグラデーションまで塗りへ写すと、AviUtl2 で前の物が変えた絵に掛かる
        # はずの色が字の塗りだけに掛かる
        clip = _stacked(tmp_path, between, _gradient("ff0000"))
        assert clip.source is not None
        assert clip.source.params.get("fill_type") != "linear"
        assert any(effect.kind == "gradient" for effect in clip.effects)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _render(clip: Clip, context: OffscreenGLContext) -> np.ndarray:
    base = Project.create(ProjectSettings(width=1920, height=1080, frame_rate=FrameRate(60)))
    track = Track(
        TrackKind.VIDEO, "V1", (with_fixed_items(replace(clip, timeline_start=0), picture=True),)
    )
    project = base.with_timeline(replace(base.timeline, tracks=(track,)))
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(0).astype(np.int16)
    finally:
        renderer.close()


def test_the_bottom_aligned_text_switches_at_its_middle(
    tmp_path: Path, gl_context: OffscreenGLContext
) -> None:
    # AviUtl2 は文字の枠の真ん中を中心に塗る（中央揃え[下] の字を縦に塗ると枠の真ん中で色が
    # 切り替わった） 絵の真ん中を中心にすると、枠が中心より上にあるので字が全部開始色になる
    clip = mapped(tmp_path, angle="0.00")
    image = _render(clip, gl_context)
    lit = image[..., :3].sum(axis=2) > 120
    ys, xs = np.nonzero(lit)
    x = int(xs.min()) + (int(xs.max()) - int(xs.min())) // 8
    rows = [y for y in range(int(ys.min()), int(ys.max()) + 1) if lit[y, x]]
    top, bottom = image[rows[8], x], image[rows[-9], x]
    assert top[0] > 180 and top[2] < 80, top
    assert bottom[2] > 180 and bottom[0] < 80, bottom


def test_a_centred_text_draws_as_the_effect_did(
    tmp_path: Path, gl_context: OffscreenGLContext
) -> None:
    # 中央揃え[中] では、絵の真ん中と枠の真ん中が重なる エフェクトのまま描いた絵と変わると、
    # 前の版で合っていた字幕の色が写しただけで変わる
    clip = mapped(tmp_path, align="中央揃え[中]", angle="90.00")
    assert clip.source is not None
    effect = registry.require("gradient").create(
        start_color=(1.0, 0.0, 0.0, 1.0), end_color=(0.0, 0.0, 1.0, 1.0), angle=90.0, span=100.0
    )
    plain = {
        name: value for name, value in clip.source.params.items() if not name.startswith("fill_")
    }
    as_effect = replace(
        clip, source=replace(clip.source, params=plain), effects=(effect, *clip.effects)
    )
    difference = np.abs(_render(clip, gl_context) - _render(as_effect, gl_context))[..., :3]
    assert difference.mean() < 0.05, difference.mean()
    assert np.percentile(difference, 99.9) <= 8
