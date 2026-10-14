"""テキストの塗りのグラデーションの項目と保存の形（#312）

止まりの並びの整え方・設定パネルに出す欄と並び・保存の形の版・プリセットの版を見る
描いた絵は ``tests/engine/test_text_gradient.py``、画面の操作は ``tests/ui/test_gradient_editor.py``
"""

from __future__ import annotations

import json
import math

import pytest

from sashimono.core.io import ProjectFileError
from sashimono.core.io.presets import FORMAT_VERSION as PRESET_VERSION
from sashimono.core.io.presets import Preset
from sashimono.core.io.serialize import (
    FORMAT_VERSION,
    project_from_dict,
    project_to_dict,
    source_from_json,
    source_to_json,
)
from sashimono.core.model import AnimatedValue, Project, Stroke, uses_gradient
from sashimono.core.model.stroke import SOURCE_FILL_KINDS, STROKE_FILL_KIND
from sashimono.effects import GradientSpec
from sashimono.effects.fills import GRADIENT_SUFFIXES, fill_names
from sashimono.effects.paint import MAX_STOPS
from sashimono.effects.sources import SHADOW_FILL, TEXT, TEXT_FILL
from sashimono.effects.spec import MAX_GRADIENT_STOPS, ParamInput
from sashimono.effects.strokes import STROKE, STROKE_FILL

SPEC = GradientSpec("fill_gradient", "グラデーション")
RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def _flat(*stops: tuple[float, tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(number for position, colour in stops for number in (position, *colour))


class TestTheStops:
    def test_they_are_kept_in_position_order(self) -> None:
        # 並びの順のまま描くと、画面で後ろへ動かした止まりが前の止まりとの間に逆向きの段を作る
        value = SPEC.coerce(_flat((1.0, BLUE), (0.0, RED), (0.5, GREEN)))
        assert [position for position, _ in GradientSpec.stops(value)] == [0.0, 0.5, 1.0]
        assert [colour for _, colour in GradientSpec.stops(value)] == [RED, GREEN, BLUE]

    def test_stops_at_the_same_place_keep_their_order(self) -> None:
        # 同じ位置の 2 つは色の段 入れ替えると、段の上下の色が逆になる
        value = SPEC.coerce(_flat((0.0, RED), (0.5, GREEN), (0.5, BLUE), (1.0, RED)))
        assert [colour for _, colour in GradientSpec.stops(value)] == [RED, GREEN, BLUE, RED]

    @pytest.mark.parametrize(
        "broken",
        [
            # 数の合わない並び 5 つずつ読むと最後の止まりが欠ける
            (0.0, 1.0, 1.0, 1.0),
            # 止まり 1 つ 単色と同じで、バーで動かせない
            (0.0, 1.0, 1.0, 1.0, 1.0),
            # 上限を超える
            _flat(*((index / 20, RED) for index in range(MAX_GRADIENT_STOPS + 1))),
            # 数でない値と NaN
            (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, math.nan),
            "赤から青",
            None,
        ],
    )
    def test_a_broken_value_falls_back_to_the_default(self, broken: ParamInput) -> None:
        # 手で書いたファイルの誤りで、開く所や描く所が落ちる
        assert SPEC.coerce(broken) == SPEC.default

    def test_numbers_outside_the_range_are_clamped(self) -> None:
        # 1 を超える色を Qt へ渡すと、警告を出して黒で塗る
        value = SPEC.coerce(_flat((-0.5, (2.0, -1.0, 0.5, 1.5)), (1.5, BLUE)))
        assert GradientSpec.stops(value) == [(0.0, (1.0, 0.0, 0.5, 1.0)), (1.0, BLUE)]

    def test_the_upper_limit_matches_the_paint_effect(self) -> None:
        # 違うと、模様の塗りから写した色の並びが片方の上限で切れる
        assert MAX_STOPS == MAX_GRADIENT_STOPS == 16


class TestTheNames:
    def test_the_core_knows_the_same_names(self) -> None:
        # 食い違うと、グラデーションのプリセットを前の版の形で書き、前の版が単色で当てる
        assert set(SOURCE_FILL_KINDS) == {f"{TEXT_FILL}_type", f"{SHADOW_FILL}_type"}
        assert f"{STROKE_FILL}_type" == STROKE_FILL_KIND
        names = {spec.name for spec in TEXT.parameters}
        assert set(fill_names(TEXT_FILL)) <= names
        assert set(fill_names(SHADOW_FILL)) <= names
        assert set(fill_names(STROKE_FILL)) <= {spec.name for spec in STROKE.parameters}

    def test_old_items_keep_their_place(self) -> None:
        # 前の項目の並びが動くと、項目の位置で読む物（プリセットの比べ方など）が食い違う
        assert [spec.name for spec in STROKE.parameters[:5]] == [
            "width",
            "color",
            "position",
            "opacity",
            "join",
        ]
        assert [spec.name for spec in TEXT.parameters][-2 * len(GRADIENT_SUFFIXES) :] == [
            *fill_names(TEXT_FILL),
            *fill_names(SHADOW_FILL),
        ]

    def test_new_text_starts_solid(self) -> None:
        # 既定がグラデーションだと、置いただけの字が白黒の帯になる
        params = TEXT.create().params
        assert params["fill_type"] == "solid"
        assert params["shadow_fill_type"] == "solid"
        assert not uses_gradient(params)


class TestTheShownFields:
    def _shown(self, **values: ParamInput) -> list[str]:
        params = TEXT.create(**values).params
        unused = TEXT.unused_names(params)
        kept = [s for s in TEXT.parameters if s.name not in unused]
        return [s.name for s in TEXT.display_order(kept)]

    def test_a_solid_fill_shows_the_colour_only(self) -> None:
        # 単色でグラデーションの欄が出ると、動かしても絵が変わらず壊れたように見える
        shown = self._shown()
        assert "color" in shown
        assert "fill_type" in shown
        assert "fill_gradient" not in shown and "fill_angle" not in shown

    def test_a_linear_fill_shows_the_bar_and_hides_the_colour(self) -> None:
        # 色の欄が残ると、グラデーションの字で色を変えても何も変わらない
        shown = self._shown(fill_type="linear")
        assert "color" not in shown
        assert {"fill_gradient", "fill_angle", "fill_length", "fill_shift", "fill_span"} <= set(
            shown
        )
        assert "fill_radius" not in shown and "fill_center_x" not in shown

    def test_a_radial_fill_shows_the_centre_and_radius(self) -> None:
        shown = self._shown(fill_type="radial")
        assert {"fill_center_x", "fill_center_y", "fill_radius"} <= set(shown)
        assert "fill_angle" not in shown

    def test_the_fill_sits_by_the_colour(self) -> None:
        # 末尾に出ると、色を探す所から 20 行も離れた底にあって見つけられない
        shown = self._shown(fill_type="linear", shadow_x=4)
        assert shown[shown.index("size") + 1] == "fill_type"
        assert shown[shown.index("fill_type") + 1] == "fill_gradient"
        assert shown[shown.index("shadow_blur") + 1] == "shadow_fill_type"

    def test_the_shadow_fill_hides_without_a_shadow(self) -> None:
        # 影が無いのに影の塗りの欄が並ぶと、動かしても何も変わらない
        assert "shadow_fill_type" not in self._shown()
        assert "shadow_fill_type" in self._shown(shadow_y=-4)

    def test_a_layer_shows_the_bar_when_it_is_a_gradient(self) -> None:
        params = {**STROKE.default_params(), "fill_type": "linear"}
        unused = STROKE.unused_names(params)
        assert "color" in unused and "fill_gradient" not in unused
        assert "fill_gradient" in STROKE.unused_names(STROKE.default_params())


class TestSaving:
    def _gradient_text(self) -> dict[str, ParamInput]:
        return {"fill_type": "linear", "fill_gradient": _flat((0.0, RED), (1.0, BLUE))}

    def test_the_format_version_went_up(self) -> None:
        # 9 の本体はグラデーションを知らずに単色で描く 「更新してください」で止めるため上げた
        assert FORMAT_VERSION == 10
        written = project_to_dict(Project.create())
        assert written["version"] == 10
        with pytest.raises(ProjectFileError, match="更新"):
            project_from_dict({**written, "version": FORMAT_VERSION + 1})

    def test_an_old_text_opens_solid(self) -> None:
        # 9 までのファイルは項目を持たない そのまま単色で読めないと、前の作品が開けない
        old = {"kind": "text", "params": {"text": "字", "color": [1.0, 0.0, 0.0, 1.0]}}
        source = source_from_json(old)
        assert "fill_type" not in source.params
        assert not uses_gradient(source.params)

    def test_the_stops_survive_a_round_trip(self) -> None:
        # 保存し直すたびに止まりが欠けたり順が変わったりすると、作ったグラデーションが崩れる
        source = TEXT.create(**self._gradient_text())
        back = source_from_json(json.loads(json.dumps(source_to_json(source))))
        assert back.params["fill_gradient"] == source.params["fill_gradient"]
        assert back.params["fill_type"] == "linear"

    def test_a_gradient_preset_is_not_read_by_the_old_version(self) -> None:
        # 版 2 のまま書くと、前の版の本体がグラデーションを捨てて単色の見た目を当てる
        gradient = Preset(name="グラデ", source=TEXT.create(**self._gradient_text()))
        plain = Preset(name="単色", source=TEXT.create())
        layered = Preset(
            name="層",
            source=TEXT.create().with_strokes(
                (Stroke(params={"width": AnimatedValue(4.0), "fill_type": "solid"}),)
            ),
        )
        layer_gradient = Preset(
            name="層のグラデ",
            source=TEXT.create().with_strokes(
                (Stroke(params={"width": AnimatedValue(4.0), "fill_type": "radial"}),)
            ),
        )
        assert gradient.to_dict()["version"] == 3
        assert layer_gradient.to_dict()["version"] == 3
        # グラデーションを使わない物は前と同じ版 前の版の本体でもこれまでどおり読める
        assert plain.to_dict()["version"] == 1
        assert layered.to_dict()["version"] == 2
        assert PRESET_VERSION == 3
        back = Preset.from_dict(json.loads(json.dumps(gradient.to_dict())))
        assert back.source == gradient.source
        with pytest.raises(ProjectFileError, match="新しい形式"):
            Preset.from_dict({**gradient.to_dict(), "version": PRESET_VERSION + 1})
