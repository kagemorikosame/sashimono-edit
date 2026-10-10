"""テキストの縁取りの層の命令と保存（#272 #273）"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddStroke,
    AddTrack,
    AdoptLegacyStroke,
    MoveEffect,
    MoveStroke,
    ParamPath,
    RemoveEffect,
    RemoveStroke,
    SetEffectEnabled,
    SetKeyframe,
    SetParam,
    SetStrokeEnabled,
    SplitClip,
    resolve_param,
)
from sashimono.core.io.serialize import (
    FORMAT_VERSION,
    ProjectFileError,
    clip_from_json,
    clip_to_json,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
    source_from_json,
    source_to_json,
)
from sashimono.core.model import (
    MAX_STROKES,
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    GeneratedSource,
    Keyframe,
    ParamValue,
    Project,
    Stroke,
    StrokeId,
    Track,
    TrackKind,
    legacy_in_use,
)

RED = (1.0, 0.0, 0.0, 1.0)


def _project(source: GeneratedSource) -> tuple[Project, ClipId]:
    clip = Clip(timeline_start=0, duration=60, source=source)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = _apply(Project.create(), AddTrack(track), AddClip(track.id, clip))
    return project, clip.id


def _text(**params: ParamValue) -> GeneratedSource:
    return GeneratedSource(kind="text", params={"text": "字", **params})


def _source(project: Project, clip_id: ClipId) -> GeneratedSource:
    located = project.timeline.locate_clip(clip_id)
    assert located is not None and located[1].source is not None
    return located[1].source


def _layer(width: float, stroke_id: str) -> Stroke:
    return Stroke(params={"width": AnimatedValue(width)}, id=StrokeId(stroke_id))


@pytest.fixture
def project() -> tuple[Project, ClipId]:
    return _project(_text())


def _apply(project: Project, *commands: object) -> Project:
    for command in commands:
        project = command.apply(project)  # type: ignore[attr-defined]
    return project


class TestAddingLayers:
    def test_the_old_border_becomes_the_first_layer(self) -> None:
        # 前からの縁がある字に層を足すと、前の縁は 1 つ目の層へ移る 移さないと、前の縁が
        # 層の一覧の外で描かれ続けるか、層を足しただけで消える
        width = AnimatedValue(3.0, (Keyframe(0, 3.0), Keyframe(30, 9.0)))
        project, clip_id = _project(_text(border_width=width, border_color=RED))
        project = AddStroke(clip_id, _layer(12.0, "外")).apply(project)
        source = _source(project, clip_id)
        assert [stroke.params.get("width") for stroke in source.strokes] == [
            width,
            AnimatedValue(12.0),
        ]
        assert source.strokes[0].params["color"] == RED
        assert source.strokes[1].id == "外"
        # 前の太さは 0 にする 残すと、層を全部消したときに前の縁がまた出てくる
        assert not legacy_in_use(source.params)
        project = RemoveStroke(clip_id, source.strokes[0].id).apply(project)
        project = RemoveStroke(clip_id, StrokeId("外")).apply(project)
        assert not legacy_in_use(_source(project, clip_id).params)

    def test_a_text_without_a_border_gets_only_the_new_layer(
        self, project: tuple[Project, ClipId]
    ) -> None:
        state, clip_id = project
        state = AddStroke(clip_id, _layer(4.0, "一")).apply(state)
        assert [stroke.id for stroke in _source(state, clip_id).strokes] == ["一"]

    def test_the_index_places_the_layer(self, project: tuple[Project, ClipId]) -> None:
        state, clip_id = project
        state = _apply(
            state,
            AddStroke(clip_id, _layer(4.0, "a")),
            AddStroke(clip_id, _layer(8.0, "b")),
            AddStroke(clip_id, _layer(2.0, "上"), index=0),
        )
        assert [stroke.id for stroke in _source(state, clip_id).strokes] == ["上", "a", "b"]

    def test_there_is_a_limit(self, project: tuple[Project, ClipId]) -> None:
        # 層の数に上限が無いと、字幕を作り直す時間が層の数だけ延び続ける
        state, clip_id = project
        for number in range(MAX_STROKES):
            state = AddStroke(clip_id, _layer(1.0 + number, f"層{number}")).apply(state)
        with pytest.raises(ValueError, match="つまで"):
            AddStroke(clip_id, _layer(20.0, "多すぎ")).apply(state)

    def test_only_text_takes_layers(self) -> None:
        state, clip_id = _project(GeneratedSource(kind="shape"))
        with pytest.raises(ValueError, match="テキスト"):
            AddStroke(clip_id, _layer(4.0, "a")).apply(state)

    def test_adopting_twice_is_refused(self) -> None:
        state, clip_id = _project(_text(border_width=AnimatedValue(5.0)))
        state = AdoptLegacyStroke(clip_id, StrokeId("移した")).apply(state)
        assert _source(state, clip_id).strokes[0].id == "移した"
        with pytest.raises(ValueError):
            AdoptLegacyStroke(clip_id, StrokeId("もう一度")).apply(state)


class TestEditingLayers:
    @pytest.fixture
    def layered(self, project: tuple[Project, ClipId]) -> tuple[Project, ClipId]:
        state, clip_id = project
        state = _apply(
            state, AddStroke(clip_id, _layer(4.0, "a")), AddStroke(clip_id, _layer(8.0, "b"))
        )
        return state, clip_id

    def test_moving_a_layer(self, layered: tuple[Project, ClipId]) -> None:
        state, clip_id = layered
        state = MoveStroke(clip_id, StrokeId("b"), 0).apply(state)
        assert [stroke.id for stroke in _source(state, clip_id).strokes] == ["b", "a"]

    def test_hiding_a_layer(self, layered: tuple[Project, ClipId]) -> None:
        state, clip_id = layered
        state = SetStrokeEnabled(clip_id, StrokeId("a"), False).apply(state)
        assert [stroke.enabled for stroke in _source(state, clip_id).strokes] == [False, True]

    def test_values_and_keyframes_go_through_the_usual_commands(
        self, layered: tuple[Project, ClipId]
    ) -> None:
        # 層の値も中身の値と同じ命令で変えられる 別の命令にすると、キーフレームや初期値へ
        # 戻す操作を層のために書き直すことになる
        state, clip_id = layered
        path = ParamPath.of_stroke(clip_id, StrokeId("b"), "width")
        state = _apply(state, SetParam(path, AnimatedValue(10.0)), SetKeyframe(path, 20, 16.0))
        value = resolve_param(state, path)
        assert isinstance(value, AnimatedValue) and value.at(20) == 16.0
        # ほかの層と中身は変わらない
        assert resolve_param(state, ParamPath.of_stroke(clip_id, StrokeId("a"), "width")) == (
            AnimatedValue(4.0)
        )
        assert resolve_param(state, ParamPath.of_source(clip_id, "width")) is None

    def test_layer_effects_go_through_the_usual_commands(
        self, layered: tuple[Project, ClipId]
    ) -> None:
        state, clip_id = layered
        stroke = StrokeId("b")
        blur = Effect("blur", {"radius": AnimatedValue(4.0)})
        glow = Effect("glow")
        state = _apply(
            state,
            AddEffect(clip_id, blur, stroke_id=stroke),
            AddEffect(clip_id, glow, stroke_id=stroke),
            MoveEffect(clip_id, glow.id, 0, stroke_id=stroke),
            SetEffectEnabled(clip_id, blur.id, False, stroke_id=stroke),
            SetParam(
                ParamPath.of_stroke_effect(clip_id, stroke, blur.id, "radius"), AnimatedValue(9.0)
            ),
        )
        held = _source(state, clip_id).strokes[1]
        assert [effect.kind for effect in held.effects] == ["glow", "blur"]
        assert held.effects[1].enabled is False
        assert held.effects[1].params["radius"] == AnimatedValue(9.0)
        # クリップのエフェクトにも、ほかの層にも入らない
        located = state.timeline.locate_clip(clip_id)
        assert located is not None and located[1].effects == ()
        assert _source(state, clip_id).strokes[0].effects == ()
        state = RemoveEffect(clip_id, glow.id, stroke_id=stroke).apply(state)
        assert [effect.kind for effect in _source(state, clip_id).strokes[1].effects] == ["blur"]

    def test_removing_a_layer_takes_its_effects(self, layered: tuple[Project, ClipId]) -> None:
        state, clip_id = layered
        state = AddEffect(clip_id, Effect("blur"), stroke_id=StrokeId("a")).apply(state)
        state = RemoveStroke(clip_id, StrokeId("a")).apply(state)
        assert [stroke.id for stroke in _source(state, clip_id).strokes] == ["b"]

    def test_a_fixed_effect_is_refused(self, layered: tuple[Project, ClipId]) -> None:
        state, clip_id = layered
        with pytest.raises(ValueError, match="固定"):
            AddEffect(clip_id, Effect("transform", fixed=True), stroke_id=StrokeId("a")).apply(
                state
            )

    def test_an_unknown_layer_is_an_error(self, layered: tuple[Project, ClipId]) -> None:
        state, clip_id = layered
        with pytest.raises(KeyError):
            SetParam(
                ParamPath.of_stroke(clip_id, StrokeId("無い"), "width"), AnimatedValue(1.0)
            ).apply(state)
        assert resolve_param(state, ParamPath.of_stroke(clip_id, StrokeId("無い"), "width")) is None

    def test_splitting_the_clip_splits_layer_keyframes(
        self, layered: tuple[Project, ClipId]
    ) -> None:
        # 割ったときに層のキーをずらさないと、後ろのクリップの縁の太さが割った所で跳ぶ
        state, clip_id = layered
        path = ParamPath.of_stroke(clip_id, StrokeId("a"), "width")
        blur = Effect("blur")
        state = _apply(
            state,
            SetKeyframe(path, 0, 0.0),
            SetKeyframe(path, 40, 40.0),
            AddEffect(clip_id, blur, stroke_id=StrokeId("a")),
            SetKeyframe(
                ParamPath.of_stroke_effect(clip_id, StrokeId("a"), blur.id, "radius"), 0, 0.0
            ),
            SetKeyframe(
                ParamPath.of_stroke_effect(clip_id, StrokeId("a"), blur.id, "radius"), 40, 20.0
            ),
            SplitClip(clip_id, 20),
        )
        clips = sorted(state.timeline.tracks[0].clips, key=lambda clip: clip.timeline_start)
        after = clips[1].source
        assert after is not None
        width = after.strokes[0].params["width"]
        radius = after.strokes[0].effects[0].params["radius"]
        assert isinstance(width, AnimatedValue) and isinstance(radius, AnimatedValue)
        assert width.at(0) == pytest.approx(20.0)
        assert radius.at(0) == pytest.approx(10.0)


class TestSaving:
    def test_layers_come_back(self, tmp_path: Path) -> None:
        source = _text(border_width=AnimatedValue(0.0)).with_strokes(
            (
                Stroke(
                    params={"width": AnimatedValue(6.0), "position": "center"},
                    effects=(Effect("blur", {"radius": AnimatedValue(3.0)}),),
                    enabled=False,
                    id=StrokeId("縁"),
                ),
            )
        )
        project, _ = _project(source)
        path = tmp_path / "層.sme"
        save_project(project, path)
        assert load_project(path).timeline == project.timeline

    def test_a_text_without_layers_writes_what_it_wrote_before(self) -> None:
        # 層の無い字（ほとんどの作品）の書き物に空の一覧が増えると、差分が読みにくくなる
        assert "strokes" not in source_to_json(_text(border_width=AnimatedValue(4.0)))

    def test_an_old_file_reads_without_layers(self) -> None:
        # 版 8 までのファイルは層を持たない 前からの縁の項目のまま読めば同じ絵になる
        old = {"kind": "text", "params": {"text": "字", "border_width": {"static": 5.0}}}
        source = source_from_json(old)
        assert source.strokes == ()
        assert source.params["border_width"] == AnimatedValue(5.0)

    def test_a_layer_without_an_id_gets_one(self) -> None:
        # 手で書いたファイルで ID が無い層が 2 つあると、片方を消したつもりで両方消える
        raw = {"kind": "text", "params": {}, "strokes": [{"params": {}}, {"params": {}}]}
        first, second = source_from_json(raw).strokes
        assert first.id and second.id and first.id != second.id

    def test_the_format_version_went_up(self) -> None:
        # 前の版の本体は層を捨てて開き、層へ移した字は縁が 1 本も無くなる
        # 「更新してください」で止めるため、版を上げた
        project, _ = _project(_text())
        assert FORMAT_VERSION == 9
        assert project_to_dict(project)["version"] == 9
        too_new = {**project_to_dict(project), "version": FORMAT_VERSION + 1}
        with pytest.raises(ProjectFileError, match="更新"):
            project_from_dict(too_new)

    def test_aliases_and_presets_carry_layers(self) -> None:
        # エイリアスはクリップの書き物、プリセットは中身の書き物を使う どちらも層を持ち運ぶ
        source = _text().with_strokes((_layer(7.0, "持ち運ぶ"),))
        clip = Clip(timeline_start=0, duration=10, source=source)
        assert clip_from_json(json.loads(json.dumps(clip_to_json(clip)))).source == source
        assert source_from_json(json.loads(json.dumps(source_to_json(source)))) == source
