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
from sashimono.core.commands.preset import PresetOptions, preset_commands
from sashimono.core.io import Preset, PresetStore
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
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
