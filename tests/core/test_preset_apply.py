"""プリセットの保存の形と、当てるコマンドの組み方（#275）

画面からの往復は tests/ui/test_preset_roundtrip.py で見る ここは、画面を通すと
作りにくい組み合わせ（欄の無い前の版のクリップ・種類の違う中身・音のクリップなど）を見る
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sashimono.core.commands import AddEffect, SetEffectEnabled, SetParam, SetSource
from sashimono.core.commands.fixed import (
    FLIP_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
    fixed_effect,
    with_fixed_items,
)
from sashimono.core.commands.preset import (
    PICTURE_SIDE,
    SOUND_SIDE,
    PresetOptions,
    origin_sides,
    preset_commands,
)
from sashimono.core.io import Preset, PresetStore
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    Track,
    TrackKind,
)
from sashimono.core.model.fitting import fitted_value
from sashimono.effects.sources import SHAPE, TEXT, TRANSITION


def _text(duration: int = 60, **params: object) -> Clip:
    source = TEXT.create(**params)  # type: ignore[arg-type]
    return with_fixed_items(Clip(timeline_start=0, duration=duration, source=source), picture=True)


def _applied(clip: Clip, preset: Preset, **kwargs: object) -> Clip:
    project = Project.create()
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    project = project.with_timeline(replace(project.timeline, tracks=(track,)))
    for command in preset_commands(preset, clip, **kwargs):  # type: ignore[arg-type]
        project = command.apply(project)
    located = project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[1]


class TestFormat:
    def test_everything_survives_the_file(self, tmp_path: Path) -> None:
        moving = AnimatedValue(1.0, (Keyframe(0, 1.0), Keyframe(30, 0.2)))
        clip = replace(
            _text(color=(1.0, 0.0, 0.0, 1.0)),
            opacity=moving,
            blend_mode="screen",
            clip_to_below=True,
            after_effects=(Effect(kind="glow"),),
        )
        store = PresetStore(tmp_path)
        store.save(Preset.capture("見出し", clip))
        (loaded,) = store.all()
        assert loaded.source == clip.source
        assert loaded.opacity == moving
        assert loaded.blend_mode == "screen"
        assert loaded.clip_to_below is True
        assert [e.kind for e in loaded.after_effects] == ["glow"]
        assert {e.kind for e in loaded.fixed} == {FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND}
        assert all(e.fixed for e in loaded.fixed)
        assert loaded.span == 60

    def test_an_old_file_reads_as_add_only(self) -> None:
        # 前の版が書いた形 長さを持たないので、当てるときは足すだけ・長さ合わせ無し
        old = Preset.from_dict(
            {"format": "sashimono-preset", "version": 1, "name": "昔", "effects": []}
        )
        assert old.span is None
        assert old.opacity is None
        assert old.fixed == ()

    def test_the_old_keys_still_hold_the_effects_and_the_source(self) -> None:
        # 前の版の本体は effects と source だけを読む そこへ足したエフェクトと中身を書いておけば、
        # 前の版でも少なくとも今までどおりに当てられる
        clip = replace(_text(), effects=(Effect(kind="blur"), *_text().effects))
        data = Preset.capture("見出し", clip).to_dict()
        effects = data["effects"]
        assert isinstance(effects, list)
        assert [e["kind"] for e in effects] == ["blur"]
        assert data["source"] is not None

    def test_a_sound_only_clip_keeps_no_picture_values(self) -> None:
        # 持つと、絵のクリップへ当てたときに向こうの不透明度を 100% へ戻してしまう
        preset = Preset.capture("音", Clip(timeline_start=0, duration=30), picture=False)
        assert preset.opacity is None
        assert preset.blend_mode is None
        assert not preset.has_look

    def test_a_broken_clip_value_is_dropped_not_the_whole_preset(self) -> None:
        data = Preset.capture("見出し", _text()).to_dict()
        data["clip"] = {"kind": "clip", "params": {"blend_mode": 3}}
        loaded = Preset.from_dict(data)
        assert loaded.blend_mode is None
        assert loaded.source is not None


class TestApply:
    def test_a_different_kind_of_source_is_left_alone(self) -> None:
        # テキストを図形に変えるのは置き直し（エイリアス）で、見た目を当てる操作ではない
        shape = Preset.capture("四角", Clip(timeline_start=0, duration=30, source=SHAPE.create()))
        target = _text(text="残る")
        commands = preset_commands(shape, target)
        assert not any(isinstance(c, SetSource) for c in commands)

    def test_a_timer_does_not_turn_plain_text_into_a_timer(self) -> None:
        timer = Preset.capture("時計", _text(timer_format="mm\\:ss", size=90))
        after = _applied(_text(text="ふつう"), timer)
        assert after.source is not None
        assert after.source.params["timer_format"] == ""
        assert after.source.params["size"] == AnimatedValue(90.0)

    def test_an_old_clip_without_the_fixed_items_gets_them(self) -> None:
        # 前の版のファイルのクリップは欄をまだ持たない 値を捨てずに欄ごと足す
        saved = _text()
        transform = next(e for e in saved.effects if e.kind == TRANSFORM_EFFECT_KIND)
        saved = replace(
            saved,
            effects=tuple(
                replace(e, params={**e.params, "rotation": AnimatedValue(45.0)})
                if e is transform
                else e
                for e in saved.effects
            ),
        )
        bare = Clip(timeline_start=0, duration=60, source=TEXT.create())
        after = _applied(bare, Preset.capture("回す", saved))
        (added,) = [e for e in after.effects if e.kind == TRANSFORM_EFFECT_KIND]
        assert added.fixed
        assert added.params["rotation"] == AnimatedValue(45.0)
        # 位置は当てないが、足した欄の既定の位置は持つ
        assert added.params["pos_x"] == fixed_effect(TRANSFORM_EFFECT_KIND).params["pos_x"]

    def test_switching_a_fixed_item_off_travels(self) -> None:
        saved = _text()
        saved = replace(
            saved,
            effects=tuple(
                replace(e, enabled=False) if e.kind == FLIP_EFFECT_KIND else e
                for e in saved.effects
            ),
        )
        commands = preset_commands(Preset.capture("切る", saved), _text())
        assert any(isinstance(c, SetEffectEnabled) and not c.enabled for c in commands)

    def test_sound_items_are_not_given_to_a_text(self) -> None:
        # 音の欄はテキストに出ない 足すと、設定パネルに出ない欄が増える
        sound = with_fixed_items(Clip(timeline_start=0, duration=30), sound=True)
        preset = Preset.capture("音", sound, picture=False)
        commands = preset_commands(preset, _text(), picture=True, sound=False)
        assert not any(
            isinstance(c, AddEffect) and c.effect.kind == VOLUME_EFFECT_KIND for c in commands
        )

    def test_effects_the_clip_cannot_take_are_skipped(self) -> None:
        clip = replace(_text(), effects=(Effect(kind="blur"), *_text().effects))
        commands = preset_commands(
            Preset.capture("ぼかし", clip), _text(), accepts=lambda kind: kind != "blur"
        )
        assert not any(isinstance(c, AddEffect) and c.effect.kind == "blur" for c in commands)

    def test_nothing_to_add_leaves_the_clips_own_effects(self) -> None:
        # 1 つも当たらないのに先に消すと、当てる先のエフェクトだけが消える
        saved = replace(_text(), effects=(Effect(kind="glow"), *_text().effects))
        target = replace(_text(), effects=(Effect(kind="blur"), *_text().effects))
        after = _applied(target, Preset.capture("光る", saved), accepts=lambda kind: False)
        assert [e.kind for e in after.effects if not e.fixed] == ["blur"]

    def test_a_preset_without_effects_still_clears_them(self) -> None:
        # エフェクトの無い見た目を保存した物 当てると同じくエフェクトの無い見た目になる
        target = replace(_text(), effects=(Effect(kind="blur"), *_text().effects))
        after = _applied(target, Preset.capture("素", _text()), accepts=lambda kind: False)
        assert [e for e in after.effects if not e.fixed] == []

    def test_a_transition_ignores_the_picture_values(self) -> None:
        # 場面切り替えは不透明度・合成モードを読まない 当てても何も変わらない段が積まれる
        saved = replace(_text(), opacity=AnimatedValue(0.3), blend_mode="add")
        target = Clip(timeline_start=0, duration=30, source=TRANSITION.create())
        commands = preset_commands(Preset.capture("薄い", saved), target)
        assert not any(isinstance(c, SetParam) and c.path.name == "opacity" for c in commands)

    def test_nothing_to_do_gives_no_commands(self) -> None:
        clip = _text(color=(0.0, 1.0, 0.0, 1.0))
        assert preset_commands(Preset.capture("同じ", clip), clip) == []

    def test_text_comes_along_only_when_asked(self) -> None:
        preset = Preset.capture("文字", _text(text="新しい"))
        kept = _applied(_text(text="古い"), preset)
        taken = _applied(_text(text="古い"), preset, options=PresetOptions(with_text=True))
        assert kept.source is not None
        assert taken.source is not None
        assert kept.source.params["text"] == "古い"
        assert taken.source.params["text"] == "新しい"

    def test_keyframes_are_fitted_like_the_template_shelf(self) -> None:
        # 棚の着せ替えと同じ伸ばし方 片方だけ直すと同じ動きの伸び方が食い違う
        moving = AnimatedValue(0.0, (Keyframe(0, 0.0), Keyframe(29, 1.0)))
        saved = replace(_text(duration=30), opacity=moving)
        after = _applied(_text(duration=90), Preset.capture("出る", saved))
        assert after.opacity == fitted_value(moving, 30, 89)
        assert [k.frame for k in after.opacity.keyframes] == [0, 89]


def _side_of(kind: str) -> str:
    return SOUND_SIDE if kind.startswith("audio_") else PICTURE_SIDE


class TestSides:
    """入れ替えは、保存したクリップと当てる先の両方が持つ側（絵・音）だけ"""

    def test_a_picture_look_does_not_touch_the_sound_side(self) -> None:
        # 映像と音を両方持つクリップへ、映像だけの見た目（エフェクト無し）を当てる
        target = replace(
            _text(), effects=(Effect(kind="blur"), Effect(kind="audio_reverb"), *_text().effects)
        )
        after = _applied(
            target, Preset.capture("素", _text()), picture=True, sound=True, side_of=_side_of
        )
        assert _loose_kinds(after) == ["audio_reverb"]

    def test_a_sound_look_replaces_only_the_sound_side(self) -> None:
        saved = replace(_text(), effects=(Effect(kind="audio_delay"),))
        target = replace(
            _text(), effects=(Effect(kind="blur"), Effect(kind="audio_reverb"), *_text().effects)
        )
        preset = Preset.capture("響き", saved, picture=False, sound=True)
        after = _applied(target, preset, picture=True, sound=True, side_of=_side_of)
        assert _loose_kinds(after) == ["blur", "audio_delay"]

    def test_a_side_the_target_does_not_have_is_left_alone(self) -> None:
        # 映像と音の両方を持つクリップの見た目でも、音を鳴らさないクリップの音の側には触らない
        # （鳴らさないクリップに残った音のエフェクトは、本人が後で音を戻すときのための物）
        both = Preset.capture("両方", _text(), picture=True, sound=True)
        target = replace(_text(), effects=(Effect(kind="audio_reverb"), *_text().effects))
        after = _applied(target, both, picture=True, sound=False, side_of=_side_of)
        assert _loose_kinds(after) == ["audio_reverb"]

    def test_the_origin_of_a_file_without_the_flags_is_inferred(self) -> None:
        data = Preset.capture("音", with_fixed_items(Clip(0, 30), sound=True), picture=False)
        data_dict = data.to_dict()
        del data_dict["picture"], data_dict["sound"]
        assert origin_sides(Preset.from_dict(data_dict)) == {SOUND_SIDE}
        text = Preset.capture("字", _text()).to_dict()
        del text["picture"], text["sound"]
        assert origin_sides(Preset.from_dict(text)) == {PICTURE_SIDE}
        assert origin_sides(Preset(name="昔", effects=(Effect(kind="blur"),))) == frozenset()


class TestSparseSource:
    def test_the_defaults_are_the_base_not_the_target(self) -> None:
        # 読み込んだテキストの持つ項目は少ない 当てる先を土台にすると、無い項目が残る
        sparse = Clip(
            timeline_start=0,
            duration=60,
            source=GeneratedSource(kind="text", params={"text": "字", "size": AnimatedValue(90)}),
        )
        target = _text(text="残る", wrap_width=300, font_style="Light", pos_x=80)
        after = _applied(
            target,
            Preset.capture("読んだ", sparse),
            defaults_of=lambda kind: TEXT.default_params() if kind == "text" else None,
        )
        assert after.source is not None
        params = after.source.params
        assert params["size"] == AnimatedValue(90.0)
        assert params["wrap_width"] == TEXT.default_params()["wrap_width"]
        assert params["font_style"] == TEXT.default_params()["font_style"]
        assert params["text"] == "残る"
        assert params["pos_x"] == AnimatedValue(80.0)

    def test_an_unknown_kind_keeps_the_target_as_the_base(self) -> None:
        saved = Clip(0, 30, source=GeneratedSource(kind="謎", params={"a": 1}))
        target = Clip(0, 30, source=GeneratedSource(kind="謎", params={"a": 0, "b": 2}))
        after = _applied(target, Preset.capture("謎", saved), defaults_of=lambda _kind: None)
        assert after.source is not None
        assert after.source.params == {"a": 1, "b": 2}


def _body_of(clip: Clip) -> Effect | None:
    """カスタムオブジェクトの見分けを写した物 空のテキストの最初のふつうのエフェクトが本体"""
    if clip.source is None or clip.source.params.get("text"):
        return None
    first = next((e for e in clip.effects if not e.fixed), None)
    return first if first is not None and first.kind == "本体" else None


def _custom(body: str = "円", *extra: Effect) -> Clip:
    base = _text(text="")
    return replace(base, effects=(Effect(kind="本体", params={"形": body}), *extra, *base.effects))


def _loose_kinds(clip: Clip) -> list[str]:
    return [e.kind for e in clip.effects if not e.fixed]


class TestBody:
    """カスタムオブジェクトの本体（中身を作るスクリプト）は見た目ではなく中身として扱う"""

    def test_replacing_keeps_the_body(self) -> None:
        # 前は入れ替えで本体まで消え、何も描かないクリップが残った
        target = _custom("円", Effect(kind="blur"))
        after = _applied(target, Preset.capture("素の字", _text()), body_of=_body_of)
        assert _loose_kinds(after) == ["本体"]
        assert _body_of(after) is not None

    def test_the_text_of_a_plain_preset_is_not_put_into_the_body(self) -> None:
        # 文字を入れると空のテキストでなくなり、本体として読まれなくなる
        after = _applied(
            _custom(),
            Preset.capture("見出し", _text(text="見出し")),
            options=PresetOptions(with_text=True),
            body_of=_body_of,
        )
        assert _body_of(after) is not None

    def test_the_saved_body_stays_out_without_the_text_option(self) -> None:
        saved = _custom("星", Effect(kind="glow"))
        after = _applied(_custom("円"), Preset.capture("星", saved), body_of=_body_of)
        body = _body_of(after)
        assert body is not None
        assert body.params["形"] == "円"
        assert _loose_kinds(after) == ["本体", "glow"]

    def test_the_text_option_swaps_the_body(self) -> None:
        # 同じカスタムオブジェクトを作り直したいときの当て方 本体は 1 つのまま入れ替わる
        saved = _custom("星", Effect(kind="glow"))
        after = _applied(
            _custom("円", Effect(kind="blur")),
            Preset.capture("星", saved),
            options=PresetOptions(with_text=True),
            body_of=_body_of,
        )
        body = _body_of(after)
        assert body is not None
        assert body.params["形"] == "星"
        assert _loose_kinds(after) == ["本体", "glow"]

    def test_a_plain_text_becomes_the_custom_object_only_with_the_text_option(self) -> None:
        saved = Preset.capture("星", _custom("星"))
        plain = _applied(_text(text="字"), saved, body_of=_body_of)
        assert _loose_kinds(plain) == []
        made = _applied(
            _text(text="字"), saved, options=PresetOptions(with_text=True), body_of=_body_of
        )
        body = _body_of(made)
        assert body is not None
        assert body.params["形"] == "星"
