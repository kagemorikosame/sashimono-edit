"""プリセットを、保存したクリップの種類 x 当てる先の種類の全部の組み合わせで当てる（#275）

レビューで「保存元に無い物まで当て先から消える」「保存元が読まない既定の値で当て先を
上書きする」という指摘が種類の組み合わせを変えて何度も来た 1 つずつ直すのではなく、
画面と同じ入口（:func:`apply_preset_commands`）で :data:`KINDS` の全部（10 x 10）の
組み合わせを当て、どの組でも次の 2 つが成り立つことを見る

* 保存元に無い物は、当てる先から消えない・変わらない
* 当たるのは保存元にある物だけ（保存元が読まない値はそもそも保存しない）

種類は テキスト・図形・映像・音・カスタムオブジェクト・場面切り替え・フィルタ・グループ制御・
音付き映像（レイヤーの動画 1 本で絵と音の両方を持つ）・混合のシーン（レイヤーに置いたシーン
音のエフェクトは効くが、素材の音ではないので音声の欄は出ない）
どれも足したエフェクト・最初から持つ欄・不透明度などを既定と違う値にして作り込む 保存元と
当てる先で値を変えておくと、当たったかどうかが値で分かる
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from sashimono.compat.aviutl.custom_object import custom_object_clip, custom_object_script
from sashimono.core.commands.fixed import (
    PICTURE_FIXED,
    SOUND_FIXED,
    TRANSFORM_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
    takes_picture_items,
    with_fixed_items,
)
from sashimono.core.commands.preset import (
    CONTENT_PARAMS,
    PLACE_PARAMS,
    POSITION_PARAMS,
    PresetOptions,
    origin_sides,
)
from sashimono.core.io import Preset
from sashimono.core.io.presets import clip_values_for
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.sources import FILTER, GROUP, SHAPE, TEXT, TRANSITION, source_registry
from sashimono.ui.inspector.panel import (
    _effect_side,
    apply_preset_commands,
    capture_preset,
    picture_and_sound,
)
from tests.conftest import make_clip

KINDS = (
    "テキスト",
    "図形",
    "映像",
    "音",
    "カスタムオブジェクト",
    "場面切り替え",
    "フィルタ",
    "グループ制御",
    "音付き映像",
    "混合のシーン",
)

#: 混合のシーンが置く中身 空のタイムラインでよい（音を鳴らすかは中身ではなく置き方で決まる）
_SCENE = Scene(name="入れ子", timeline=Project.create().timeline)

#: カスタムオブジェクトの本体 手元に無いスクリプトでも識別子で見分ける
_SCRIPT = "aviutl:試験/@図形集.obj:円"


@dataclass(frozen=True)
class _Look:
    """保存元（A）と当てる先（B）で変える値"""

    color: tuple[float, float, float, float]
    scale: float
    volume: float
    opacity: float
    blend: str
    clipped: bool
    picture_effect: str
    sound_effect: str
    words: str


A = _Look((1.0, 0.0, 0.0, 1.0), 150.0, 50.0, 0.7, "screen", False, "blur", "audio_delay", "保存")
B = _Look((0.0, 0.0, 1.0, 1.0), 80.0, 120.0, 0.4, "add", True, "glow", "audio_reverb", "当て先")


def _effect(kind: str) -> Effect:
    return registry.require(kind).create()


def _set_fixed(clip: Clip, kind: str, name: str, value: float) -> Clip:
    effects = tuple(
        replace(e, params={**e.params, name: AnimatedValue(value)})
        if e.fixed and e.kind == kind
        else e
        for e in clip.effects
    )
    return replace(clip, effects=effects)


def _picture_values(clip: Clip, look: _Look) -> Clip:
    return replace(
        clip, opacity=AnimatedValue(look.opacity), blend_mode=look.blend, clip_to_below=look.clipped
    )


def _with_loose(clip: Clip, *loose: Effect) -> Clip:
    return replace(clip, effects=(*loose, *clip.effects))


def _make(
    kind: str, look: _Look, start: int, video: MediaItem, audio: MediaItem
) -> tuple[TrackKind, Clip]:
    """作り込んだ ``kind`` のクリップと、置くトラックの種類"""
    if kind == "音":
        # 鳴らす音の番号も持たせる 音声トラックでは読まず、混合のトラックでは鳴らすのに要る
        sounding = replace(make_clip(start, 60, audio), audio_stream=audio.audio_streams[0].index)
        clip = with_fixed_items(sounding, sound=True)
        clip = _set_fixed(clip, VOLUME_EFFECT_KIND, "volume", look.volume)
        return TrackKind.AUDIO, _with_loose(clip, _effect(look.sound_effect))
    if kind == "音付き映像":
        # レイヤー（混合）のトラックの動画は 1 本で絵と音の両方を持つ 絵の側と音の側の
        # エフェクトを両方積み、側ごとに入れ替わるかを見る
        # 混合のトラックでは鳴らす音の番号を持つクリップだけが音を鳴らす
        sounding = replace(make_clip(start, 60, video), audio_stream=video.audio_streams[0].index)
        clip = with_fixed_items(_picture_values(sounding, look), picture=True, sound=True)
        clip = _set_fixed(clip, TRANSFORM_EFFECT_KIND, "scale", look.scale)
        clip = _set_fixed(clip, VOLUME_EFFECT_KIND, "volume", look.volume)
        return TrackKind.MIXED, _with_loose(
            clip, _effect(look.picture_effect), _effect(look.sound_effect)
        )
    if kind == "混合のシーン":
        # 混合のトラックに置いたシーンは音を鳴らし、音のエフェクトも掛けられるが、素材の音では
        # ないので設定パネルに音声の欄は出ない 欄を出すかで側を決めると、音の側が入れ替わらない
        clip = with_fixed_items(
            _picture_values(Clip(timeline_start=start, duration=60, scene_id=_SCENE.id), look),
            picture=True,
        )
        clip = _set_fixed(clip, TRANSFORM_EFFECT_KIND, "scale", look.scale)
        return TrackKind.MIXED, _with_loose(
            clip, _effect(look.picture_effect), _effect(look.sound_effect)
        )
    if kind == "映像":
        base = make_clip(start, 60, video)
    elif kind == "テキスト":
        # 保存元は AviUtl や YMM4 から読んだテキストと同じく持つ項目が少ない形にする
        # 当て先の値を土台にすると、無い項目（折り返しの幅など）が当て先のまま残る
        source = GeneratedSource(kind="text", params={"text": look.words, "color": look.color})
        if look is B:
            source = TEXT.create(
                text=look.words, color=look.color, wrap_width=300, font_style="Light", pos_x=40
            )
        base = Clip(timeline_start=start, duration=60, source=source)
    elif kind == "図形":
        base = Clip(timeline_start=start, duration=60, source=SHAPE.create(color=look.color))
    elif kind == "カスタムオブジェクト":
        base = custom_object_clip(Effect(kind=_SCRIPT), duration=60, timeline_start=start)
    elif kind == "場面切り替え":
        base = Clip(timeline_start=start, duration=60, source=TRANSITION.create())
        base = replace(base, after_effects=(_effect(look.picture_effect),))
    elif kind == "フィルタ":
        base = Clip(timeline_start=start, duration=60, source=FILTER.create())
    else:
        # 対象レイヤー数は場所の事情 保存元と当て先で変えて、当たらないことを見る
        layers = 3 if look is A else 1
        base = Clip(timeline_start=start, duration=60, source=GROUP.create(layers=layers))
    clip = _picture_values(base, look)
    if kind not in ("場面切り替え", "フィルタ"):
        clip = with_fixed_items(clip, picture=True)
        clip = _set_fixed(clip, TRANSFORM_EFFECT_KIND, "scale", look.scale)
        clip = _set_fixed(clip, TRANSFORM_EFFECT_KIND, "pos_x", look.scale)
    loose = _effect(look.picture_effect)
    if kind == "カスタムオブジェクト":
        # 本体の後ろに足す（本体は最初のふつうのエフェクト）
        return TrackKind.VIDEO, replace(clip, effects=(clip.effects[0], loose, *clip.effects[1:]))
    return TrackKind.VIDEO, _with_loose(clip, loose)


def _project(
    saved: tuple[TrackKind, Clip], target: tuple[TrackKind, Clip], media: list[MediaItem]
) -> Project:
    mixed = TrackKind.MIXED in (saved[0], target[0])
    settings = ProjectSettings(layer_mode=LayerMode.MIXED) if mixed else ProjectSettings()
    base = Project.create(settings, media=tuple(media))
    kinds = (TrackKind.MIXED, TrackKind.MIXED) if mixed else (saved[0], target[0])
    tracks = (
        Track(kinds[0], "保存元", (saved[1],)),
        Track(kinds[1], "当て先", (target[1],)),
    )
    with_scene = replace(base, scenes=(_SCENE,))
    return with_scene.with_timeline(replace(base.timeline, tracks=tracks))


def _loose(effects: tuple[Effect, ...], body: Effect | None) -> list[Effect]:
    return [e for e in effects if not e.fixed and (body is None or e.id != body.id)]


def _check(saved: Clip, before: Clip, after: Clip, preset: Preset, project: Project) -> None:
    """どの組み合わせでも成り立つこと"""
    target_track = project.timeline.tracks[1]
    picture, _ = picture_and_sound(project, target_track, before)

    # 本体（カスタムオブジェクトのスクリプト）は中身 見た目を当てても消えない
    body = custom_object_script(before)
    if body is not None:
        kept = custom_object_script(after)
        assert kept is not None and kept.id == body.id

    # 中身 種類が違えば触らない 同じなら、残す物は当て先のまま、保存元の物は保存元の値、
    # どちらにも無い物は種類の既定の値
    if preset.source is None or before.source is None or preset.source.kind != before.source.kind:
        assert after.source == before.source
    else:
        assert after.source is not None
        definition = source_registry.get(before.source.kind)
        defaults = definition.default_params() if definition is not None else {}
        for name in {*after.source.params, *before.source.params}:
            got = after.source.params.get(name)
            if name in CONTENT_PARAMS or name in PLACE_PARAMS:
                assert got == before.source.params.get(name, defaults.get(name)), name
            elif name in preset.source.params:
                assert got == preset.source.params[name], name
            else:
                assert got == defaults.get(name, before.source.params.get(name)), name

    # 足したエフェクト 増えたのは保存元の物だけ 消えたのは保存元が持つ側の物だけ
    sides = origin_sides(preset)
    saved_body = custom_object_script(saved)
    preset_kinds = {e.kind for e in _loose(preset.effects, saved_body)}
    before_loose = _loose(before.effects, body)
    after_loose = _loose(after.effects, custom_object_script(after))
    before_ids = {e.id for e in before_loose}
    after_ids = {e.id for e in after_loose}
    assert {e.kind for e in after_loose if e.id not in before_ids} <= preset_kinds
    for gone in (e for e in before_loose if e.id not in after_ids):
        assert _effect_side(gone.kind) in sides, gone.kind

    # 後の場面の列 保存元も場面切り替えのときだけ触る
    if preset.source is None or preset.source.kind != "transition":
        assert after.after_effects == before.after_effects

    # 最初から持つ欄 保存元に無い欄は変わらない 位置は保存元に有っても変わらない
    preset_fixed = {e.kind: e for e in preset.fixed}
    _, sound = picture_and_sound(project, target_track, before)
    for item in (e for e in after.effects if e.fixed):
        own = next((e for e in before.effects if e.fixed and e.kind == item.kind), None)
        if own is None:
            # 新しく足した欄は、当て先が出す組の物だけ（設定パネルに出ない欄を増やさない）
            if item.kind in PICTURE_FIXED:
                assert picture and takes_picture_items(before), item.kind
            if item.kind in SOUND_FIXED:
                assert sound, item.kind
        shows = sound if item.kind in SOUND_FIXED else picture and takes_picture_items(before)
        if item.kind not in preset_fixed or not shows:
            # 保存元に無い欄と、当て先が出さない組の欄は変わらない
            assert own is not None and item == own, item.kind
            continue
        for name, value in item.params.items():
            if item.kind == TRANSFORM_EFFECT_KIND and name in POSITION_PARAMS:
                expected = own.params.get(name) if own is not None else value
                assert value == expected, name
            else:
                assert value == preset_fixed[item.kind].params.get(name, value), name

    # 不透明度・合成モード・切り抜き 保存元が読む物だけを持ち、当て先が読む物だけを当てる
    saved_reads = clip_values_for(
        saved, picture=picture_and_sound(project, project.timeline.tracks[0], saved)[0]
    )
    target_reads = clip_values_for(before, picture=picture)
    for name in ("opacity", "blend_mode", "clip_to_below"):
        held = getattr(preset, name)
        if name not in saved_reads:
            assert held is None, name
        if getattr(after, name) != getattr(before, name):
            assert held is not None and name in target_reads, name
            assert getattr(after, name) == held, name


@pytest.mark.parametrize("target_kind", KINDS)
@pytest.mark.parametrize("saved_kind", KINDS)
def test_every_pair_keeps_what_the_preset_does_not_have(
    saved_kind: str, target_kind: str, video_media: MediaItem, audio_media: MediaItem
) -> None:
    saved = _make(saved_kind, A, 0, video_media, audio_media)
    target = _make(target_kind, B, 100, video_media, audio_media)
    project = _project(saved, target, [video_media, audio_media])
    saved_track, target_track = project.timeline.tracks
    preset = capture_preset("組み合わせ", project, saved_track, saved[1])

    commands = apply_preset_commands(
        project, target_track, target[1], preset, options=PresetOptions()
    )
    applied = project
    for command in commands:
        applied = command.apply(applied)
    located = applied.timeline.locate_clip(target[1].id)
    assert located is not None
    _check(saved[1], target[1], located[1], preset, project)


@pytest.mark.parametrize("kind", KINDS)
def test_the_same_kind_takes_the_saved_look(
    kind: str, video_media: MediaItem, audio_media: MediaItem
) -> None:
    # 同じ種類どうしでは、保存元の見た目（エフェクト・欄・不透明度など）が実際に当たる
    # 何も当てない作りでも上の試験は通るので、当たる側もここで押さえる
    saved = _make(kind, A, 0, video_media, audio_media)
    target = _make(kind, B, 100, video_media, audio_media)
    project = _project(saved, target, [video_media, audio_media])
    saved_track, target_track = project.timeline.tracks
    preset = capture_preset("同じ種類", project, saved_track, saved[1])
    applied = project
    for command in apply_preset_commands(
        project, target_track, target[1], preset, options=PresetOptions()
    ):
        applied = command.apply(applied)
    located = applied.timeline.locate_clip(target[1].id)
    assert located is not None
    after = located[1]
    loose = [e.kind for e in _loose(after.effects, custom_object_script(after))]
    expected = {
        "音": [A.sound_effect],
        "音付き映像": [A.picture_effect, A.sound_effect],
        # 前は音の側が入れ替わらず、当て先の残響と保存元の遅延が重なった
        "混合のシーン": [A.picture_effect, A.sound_effect],
    }.get(kind, [A.picture_effect])
    assert loose == expected
    if kind in ("音", "音付き映像"):
        volume = next(e for e in after.effects if e.fixed and e.kind == VOLUME_EFFECT_KIND)
        assert volume.params["volume"] == AnimatedValue(A.volume)
    if kind == "場面切り替え":
        assert [e.kind for e in after.after_effects] == [A.picture_effect]
    reads = clip_values_for(after, picture=picture_and_sound(applied, target_track, after)[0])
    if "opacity" in reads:
        assert after.opacity == AnimatedValue(A.opacity)
    if "blend_mode" in reads:
        assert after.blend_mode == A.blend
