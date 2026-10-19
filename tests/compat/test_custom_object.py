"""カスタムオブジェクト（中身を作るスクリプト ``.obj`` ``.obj2``）の置き方と見え方（#147）

専用の種類を持たず、空のテキストの最初のエフェクトにスクリプトを置く 形式の版を
上げずに済むため（種類を足すと、前の版の本体で開いたときに黙って何も描かない）
その代わり、見分け方と表示名をここで 1 か所に決める 右クリックの〔追加〕と
``.exa`` の読み込みが別々の形で置くと、同じカスタムオブジェクトが片方では
「テキスト」、片方では何も無いクリップになる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.aviutl import catalog as catalog_module
from sashimono.compat.aviutl import raster
from sashimono.compat.aviutl.catalog import ScriptCatalog, ScriptEntry, set_script_catalog
from sashimono.compat.aviutl.custom_object import (
    custom_object_clip,
    custom_object_script,
    script_label,
)
from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.objapi import MAX_FIGURE_SIZE, ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime
from sashimono.core.model import AnimatedValue, Effect, GeneratedSource, ParamValue
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.sources import render_source

RATE = FrameRate(30)

#: sigma の ``単純図形σ - 矩形.exa`` の骨組み（dialog の値は短くした）
CUSTOM_ALIAS = """[vo.0]
_name=カスタムオブジェクト
type=0
filter=0
name=円@図形集
track0=80.00
track1=0.00
track2=0.00
track3=0.00
check0=1
param=_1=0xff0000;
[vo.1]
_name=標準描画
X=0.0
Y=0.0
Z=0.0
拡大率=100.00
透明度=0.0
回転=0.00
blend=0
"""

CIRCLE = """@円
--track0:大きさ,0,500,100,1
--check0:塗る,0
--dialog:色/col,_1=0xffffff;
obj.load("figure", "円", _1, obj.track0)
"""


@pytest.fixture
def circle() -> Iterator[ScriptEntry]:
    """``@図形集.obj`` に ``@円`` を 1 本だけ持つ一覧 終わったら元の一覧へ戻す"""
    saved = catalog_module._catalog
    catalog = ScriptCatalog(roots=())
    entry = catalog.add_text("aviutl:試験/@図形集.obj:円", CIRCLE, kind="obj")
    catalog._entries[entry.identifier] = replace(entry, path=Path("@図形集.obj"))
    set_script_catalog(catalog)
    yield catalog._entries[entry.identifier]
    catalog_module._catalog = saved
    if saved is not None:
        saved.register_all()


@pytest.fixture
def empty() -> Iterator[ScriptCatalog]:
    """スクリプトの無い一覧 既定の置き場（本人の機械の AviUtl2 など）を読みに行かない"""
    saved = catalog_module._catalog
    catalog = ScriptCatalog(roots=())
    set_script_catalog(catalog)
    yield catalog
    catalog_module._catalog = saved
    if saved is not None:
        saved.register_all()


class TestRecognition:
    def test_the_placed_clip_is_recognised(self, circle: ScriptEntry) -> None:
        # 見分けられないと、置いた直後のクリップがタイムラインでも見出しでも「テキスト」と出る
        clip = custom_object_clip(circle.definition().create(), duration=60)
        assert clip.source is not None and clip.source.kind == "text"
        script = custom_object_script(clip)
        assert script is not None and script.kind == circle.identifier

    def test_the_fixed_items_behind_it_do_not_hide_it(self, circle: ScriptEntry) -> None:
        # 描画の欄（反転・配置）は列の末尾に付く 欄の有無で見分けが変わると、
        # 置いた直後だけ「テキスト」に戻る
        clip = custom_object_clip(circle.definition().create(), duration=60)
        fixed = Effect(kind="transform", params={}, fixed=True)
        clip = replace(clip, effects=(*clip.effects, fixed))
        assert custom_object_script(clip) is not None

    def test_a_text_with_an_animation_script_is_still_a_text(self) -> None:
        # アニメーション効果（.anm）を掛けたテキストは、中身を作るスクリプトではない
        # カスタムオブジェクトと呼ぶと、字幕に揺れを掛けただけのクリップの名前が変わる
        effect = Effect(kind="aviutl:試験/揺れ.anm:0", params={})
        clip = custom_object_clip(effect, duration=60)
        assert custom_object_script(clip) is None

    def test_a_text_with_words_is_a_text(self, circle: ScriptEntry) -> None:
        # 文字を打ったテキストにスクリプトを積んだ物は、中身がテキスト
        clip = custom_object_clip(circle.definition().create(), duration=60)
        assert clip.source is not None
        worded = replace(clip, source=clip.source.with_param("text", "見出し"))
        assert custom_object_script(worded) is None

    def test_the_label_is_the_script_name(self, circle: ScriptEntry) -> None:
        # 名前を取れないと、タイムラインに ``aviutl:…`` の識別子がそのまま出る
        assert script_label(circle.identifier) == "円"

    def test_a_missing_script_still_has_a_name(self) -> None:
        # 手元にスクリプトが無い機械で開いても、何を置いたかは識別子から分かる
        # 分からないまま「テキスト」と出すと、足りないスクリプトを探せない
        assert script_label("aviutl:どこか/@図形集.obj:星") == "星"
        assert script_label("aviutl:どこか/集中線.obj2:0") == "集中線"


class TestExaImport:
    """``.exa`` のカスタムオブジェクトも右クリックと同じ形で置く"""

    def test_it_becomes_the_same_form_as_the_add_menu(self, circle: ScriptEntry) -> None:
        # 以前は「カスタムオブジェクト: 円@図形集」と数えるだけで、中身の無いクリップだった
        report = CompatibilityReport()
        mapped = map_object(parse_exo(CUSTOM_ALIAS).objects[0], RATE, report=report)
        assert mapped is not None
        script = custom_object_script(mapped.clip)
        assert script is not None and script.kind == circle.identifier
        assert not [line for line in report.missing if line.startswith("カスタムオブジェクト")]

    def test_the_values_are_carried(self, circle: ScriptEntry) -> None:
        # track・check・dialog の値を落とすと、既定の白い 100 の円になる
        del circle
        mapped = map_object(parse_exo(CUSTOM_ALIAS).objects[0], RATE, report=CompatibilityReport())
        assert mapped is not None
        script = custom_object_script(mapped.clip)
        assert script is not None
        size = script.params["track0"]
        assert isinstance(size, AnimatedValue) and size.static == 80.0
        assert script.params["check0"] is True
        colour = script.params["_1"]
        assert isinstance(colour, tuple) and tuple(round(c * 255) for c in colour[:3]) == (
            255,
            0,
            0,
        )

    def test_a_missing_script_is_still_counted(self, empty: ScriptCatalog) -> None:
        # 手元に無いスクリプトは、何が足りないかを名前で数える（埋める順を決めるため）
        del empty
        report = CompatibilityReport()
        map_object(parse_exo(CUSTOM_ALIAS).objects[0], RATE, report=report)
        assert report.missing["カスタムオブジェクト: 円@図形集"] == 1


def _source(kind: str, params: dict[str, object], width: int, height: int) -> np.ndarray:
    wrapped: dict[str, ParamValue] = {}
    for name, value in params.items():
        if isinstance(value, bool | str):
            wrapped[name] = value
        elif isinstance(value, int | float):
            wrapped[name] = AnimatedValue(float(value))
        elif isinstance(value, tuple):
            wrapped[name] = tuple(float(part) for part in value)
    image = render_source(GeneratedSource(kind=kind, params=wrapped), width, height)
    assert image is not None
    return image


def _state() -> ObjectState:
    return ObjectState(image=np.zeros((1, 1, 4), np.uint8), screen_w=320, screen_h=180)


class TestShapesTheScriptsUse:
    """sigma の 単純図形σ が図形を作る手順 どれか 1 つが崩れると何も描かれない"""

    def test_resize_in_dots_sets_the_size(self, qt_application: object) -> None:
        # ``矩形`` は 1 画素の四角を読み、リサイズで幅と高さにする リサイズの X を
        # 位置のずれとして読むと、1 画素のまま 100 画素横へずれて何も見えない
        del qt_application
        state = _state()
        runtime = LuaScriptRuntime(render_source=_source, instruction_limit=500_000)
        result = runtime.run(
            'obj.load("figure", "四角形", 0xffffff, 1)'
            ' obj.effect("リサイズ", "X", 120, "Y", 40, "ドット数でサイズ指定", 1, "補間なし", 1)',
            state,
        )
        assert not result.failed, result.message
        assert state.image.shape == (40, 120, 4)
        assert int((state.image[..., 3] == 255).sum()) == 120 * 40
        assert state.effects == []

    def test_resize_in_percent_scales(self, qt_application: object) -> None:
        # リサイズの拡大率と X を両方掛けないと、百分率で図形を大きくする
        # カスタムオブジェクトの絵の大きさが AviUtl と食い違う
        del qt_application
        state = _state()
        runtime = LuaScriptRuntime(render_source=_source, instruction_limit=500_000)
        runtime.run(
            'obj.load("figure", "四角形", 0xffffff, 40)'
            ' obj.effect("リサイズ", "拡大率", 50, "X", 200)',
            state,
        )
        # 拡大率 50 に X 200 を掛けて横は 40 x 0.5 x 2、縦は 40 x 0.5
        # 片方しか掛けないと、百分率で大きさを決めるスクリプトの図形が倍や半分の大きさで出る
        assert state.image.shape == (20, 40, 4)

    def test_a_resize_past_the_limit_is_recorded(self, qt_application: object) -> None:
        # 上限で黙って切ると、要求より小さく描かれた理由が互換性レポートに出ない
        del qt_application
        state = _state()
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(render_source=_source, report=report)
        runtime.run(
            'obj.effect("リサイズ", "X", 5000, "Y", 10, "ドット数でサイズ指定", 1, "補間なし", 1)',
            state,
        )
        assert state.image.shape[:2] == (10, MAX_FIGURE_SIZE)
        assert any("リサイズ" in line and "上限" in line for line in report.missing)

    def test_effects_stacked_before_a_resize_are_recorded(self, qt_application: object) -> None:
        # 先に積んだぼかしは描くときに掛かるので、リサイズの後の絵へ掛かり順が入れ替わる
        # 焼き込めないまま黙ると、ぼかしの幅が違う理由が分からない
        del qt_application
        state = _state()
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(render_source=_source, report=report)
        runtime.run(
            'obj.load("figure", "四角形", 0xffffff, 10) obj.effect("ぼかし", "範囲", 4)'
            ' obj.effect("リサイズ", "拡大率", 200)',
            state,
        )
        assert state.image.shape[:2] == (20, 20)
        assert any("リサイズ" in line and "先に積んだ" in line for line in report.missing)

    def test_a_large_smooth_resize_matches_row_by_row(self) -> None:
        # 大きな絵は行の束ごとに補間する 束の継ぎ目で値が変わると横縞が出る
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
        whole = raster.resize(image, 301, 523)
        banded = raster.resize(image, 301, 523, band=17)
        assert np.array_equal(whole, banded)

    def test_a_line_as_wide_as_the_figure_fills_it(self, qt_application: object) -> None:
        # ``楕円`` は線の幅に 8000 を渡して塗りつぶしを頼む 図形オブジェクトの読み込みと
        # 同じく、図形より太い線は塗りつぶし 輪郭のままだと円の中も外も透明になる
        del qt_application
        state = _state()
        runtime = LuaScriptRuntime(render_source=_source, instruction_limit=500_000)
        runtime.run('obj.load("figure", "円", 0xffffff, 100, 8000)', state)
        assert state.image.shape == (100, 100, 4)
        assert state.image[50, 50, 3] == 255


def test_the_add_menu_and_text_share_the_empty_text() -> None:
    # 空のテキストの持ち方が 2 か所で違うと、片方だけ見分けられなくなる
    clip = custom_object_clip(Effect(kind="aviutl:a/@b.obj:c", params={}), duration=30)
    assert clip.source == TEXT.create(text="")
    assert clip.duration == 30
