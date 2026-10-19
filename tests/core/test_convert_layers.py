"""置き方の方式を途中で切り替えたときの変換（:class:`ConvertLayers` Issue #27）

分ける方式（映像トラックと音声トラック）と混合の方式（レイヤー）を行き来する
ここが崩れると、変換しただけで音付きの動画が 2 本のまま残る・BGM が消える・
音声トラックで下げていた音量が戻る・行き帰りで作品が別物になる
絵と音が変換の前後で同じことは :mod:`tests.engine.test_convert_layers_output` で見る
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import (
    Command,
    ConvertLayers,
    Document,
    SetTrackState,
    convert_layers,
    insert_filter,
    insert_media,
    place_media,
    switch_layer_mode,
)
from sashimono.core.commands.fixed import (
    FADE_EFFECT_KIND,
    FLIP_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
)
from sashimono.core.commands.insert import is_effect_track, new_track
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    EffectId,
    GroupId,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    Timeline,
    Track,
    TrackId,
    TrackKind,
)
from sashimono.effects import registry
from tests.conftest import RATE_30

SOUND_KINDS = registry.sound_kinds()


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _separated(*media: MediaItem) -> Project:
    """分ける方式で ``media`` を順に置いたプロジェクト（素材を置く所と同じ道を通す）"""
    project = Project.create(ProjectSettings(frame_rate=RATE_30), media=())
    for item in media:
        project = _apply(project, insert_media(project, item))
    return project


def _stacked(mode: str, *media: MediaItem) -> Project:
    """``mode`` の方式で ``media`` を 0 フレーム目に重ねて置いたプロジェクト"""
    project = Project.create(ProjectSettings(frame_rate=RATE_30, layer_mode=mode))
    for item in media:
        project = _apply(project, place_media(project, [item], at_frame=0))
    return project


def _to(project: Project, mode: str) -> Project:
    return ConvertLayers(mode, SOUND_KINDS).apply(project)


def _shape(timeline: Timeline) -> list[Track]:
    """ID（トラック・クリップ・エフェクト・リンクの値）を並びの番号へ置き換えた形

    行き帰りで ID は作り直されるので、そこ以外が同じかを比べるために使う
    リンクの値は、同じ組かどうかだけが意味を持つので、出てきた順の番号にする
    """
    links: dict[GroupId, GroupId] = {}

    def link(group: GroupId | None) -> GroupId | None:
        if group is None:
            return None
        return links.setdefault(group, GroupId(f"link{len(links)}"))

    shaped = []
    for number, track in enumerate(timeline.tracks):
        clips = tuple(
            replace(
                clip,
                id=ClipId(f"clip{number}.{index}"),
                link_group=link(clip.link_group),
                effects=tuple(replace(e, id=EffectId("effect")) for e in clip.effects),
            )
            for index, clip in enumerate(track.clips)
        )
        shaped.append(replace(track, id=TrackId(f"track{number}"), clips=clips))
    return shaped


def _kinds(project: Project) -> list[TrackKind]:
    return [t.kind for t in project.timeline.tracks]


# --- 分ける → 混合 ---


def test_a_linked_pair_becomes_one_clip_on_a_layer(video_media: MediaItem) -> None:
    # まとめ損ねると、混合にしたのに映像と音声の 2 本がレイヤーに並び、音が 2 重に鳴るか
    # 絵の無いレイヤーが残る
    project = _to(_separated(video_media), LayerMode.MIXED)
    assert _kinds(project) == [TrackKind.MIXED]
    (layer,) = project.timeline.tracks
    (clip,) = layer.clips
    assert layer.name == "レイヤー 1"
    assert clip.audio_stream == video_media.audio_streams[0].index
    assert clip.stream_index == video_media.video_streams[0].index
    assert clip.show_picture
    # 1 本しか無いのにリンクを残すと、あとで別の物と誤って連動する
    assert clip.link_group is None


def test_the_fixed_items_line_up_once_in_the_ymm4_order(video_media: MediaItem) -> None:
    # 絵の側（反転・配置）と音の側（音量・フェード）を並べるとき、同じ種類が 2 つ並ぶと
    # どちらがパネルの欄か分からなくなる 並びが崩れると音量の後ろに反転が来る
    project = _to(_separated(video_media), LayerMode.MIXED)
    (clip,) = project.timeline.tracks[0].clips
    fixed = [e.kind for e in clip.effects if e.fixed]
    assert fixed == [FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND, VOLUME_EFFECT_KIND, FADE_EFFECT_KIND]


def test_loose_sound_effects_go_before_the_fixed_items(video_media: MediaItem) -> None:
    # 音の側に足したモノラル化が固定の項目の後ろへ回ると、パネルの並びの決まりから外れる
    project = _separated(video_media)
    audio = project.timeline.tracks[1]
    (sound,) = audio.clips
    mono = Effect(kind="audio_monaural", params={"ratio": AnimatedValue(static=50.0)})
    audio = audio.with_clips((replace(sound, effects=(mono, *sound.effects)),))
    project = project.with_timeline(project.timeline.replace_track(audio))
    (clip,) = _to(project, LayerMode.MIXED).timeline.tracks[0].clips
    kinds = [e.kind for e in clip.effects]
    assert kinds.index("audio_monaural") < kinds.index(FLIP_EFFECT_KIND)


def test_background_music_moves_to_a_layer_without_picture(
    video_media: MediaItem, audio_media: MediaItem
) -> None:
    # 相手の無い音（BGM）を捨てると、変換しただけで BGM が消える
    project = _separated(video_media, audio_media)
    bgm = next(
        c for t in project.timeline.audio_tracks() for c in t.clips if c.media_id == audio_media.id
    )
    converted = _to(project, LayerMode.MIXED)
    assert set(_kinds(converted)) == {TrackKind.MIXED}
    located = converted.timeline.locate_clip(bgm.id)
    assert located is not None
    _, moved = located
    assert not moved.show_picture
    assert moved.audio_stream == audio_media.audio_streams[0].index


def test_a_pair_trimmed_on_one_side_stays_as_two_clips(video_media: MediaItem) -> None:
    # 長さが違うのにまとめると、絵か音のどちらかの長さが変わる
    # リンクした 2 本は一緒にトリムされるので、片方だけ短い物（リンクを外して詰めた物・
    # 古いファイル）を直に作る
    project = _separated(video_media)
    audio = project.timeline.tracks[1]
    (sound,) = audio.clips
    audio = audio.with_clips((replace(sound, duration=sound.duration - 30),))
    project = project.with_timeline(project.timeline.replace_track(audio))
    converted = _to(project, LayerMode.MIXED)
    picture_layer, sound_layer = converted.timeline.tracks
    (picture,) = picture_layer.clips
    (moved,) = sound_layer.clips
    assert picture.audio_stream is None, "映像の側が音を鳴らすと 2 重に鳴る"
    assert not moved.show_picture
    assert moved.duration == sound.duration - 30
    # 組の相手が別のレイヤーに残るので、一緒に動くようにリンクは保つ
    assert picture.link_group is not None and picture.link_group == moved.link_group


def test_the_audio_track_volume_and_pan_move_onto_the_clip(video_media: MediaItem) -> None:
    # まとめた先のレイヤーは映像トラックから来るので、音声トラックの音量を持たない
    # 写さないと、音声トラックで下げていた音量が変換で元に戻る
    project = _separated(video_media)
    audio = replace(project.timeline.tracks[1], volume_db=-6.0, pan=0.5)
    project = project.with_timeline(project.timeline.replace_track(audio))
    converted = _to(project, LayerMode.MIXED)
    (layer,) = converted.timeline.tracks
    assert layer.volume_db == 0.0 and layer.pan == 0.0
    (clip,) = layer.clips
    gains = [e for e in clip.effects if e.kind == VOLUME_EFFECT_KIND and not e.fixed]
    assert len(gains) == 1, "固定の音量を書き換えると、パネルの欄の値が変わって見える"
    volume = gains[0].params["volume"]
    pan = gains[0].params["pan"]
    assert isinstance(volume, AnimatedValue) and isinstance(pan, AnimatedValue)
    # 定位 0.5 は左右の二乗和を保つ振り方 右が大きい側で、左を絞る
    assert volume.static == pytest.approx(10 ** (-6 / 20) * 2**0.5 * 0.9238795 * 100, rel=1e-4)
    assert pan.static == pytest.approx((1 - 0.3826834 / 0.9238795) * 100, rel=1e-4)
    # 固定の項目の前（ふつうのエフェクトの位置）に入る
    assert clip.effects.index(gains[0]) < next(i for i, e in enumerate(clip.effects) if e.fixed)


def test_background_music_keeps_its_track_volume_on_the_layer(
    video_media: MediaItem, audio_media: MediaItem
) -> None:
    # 音だけのレイヤーは音声トラックからそのまま作るので、トラックの音量をそのまま持てる
    # クリップへ写すと、行き帰りで音量調整が 1 つずつ増えていく
    project = _separated(audio_media)
    audio = replace(project.timeline.tracks[0], volume_db=-12.0, pan=-0.25)
    project = project.with_timeline(project.timeline.replace_track(audio))
    (layer,) = _to(project, LayerMode.MIXED).timeline.tracks
    assert (layer.volume_db, layer.pan) == (-12.0, -0.25)
    assert not any(e.kind == VOLUME_EFFECT_KIND and not e.fixed for e in layer.clips[0].effects)


def test_the_stacking_order_of_picture_tracks_is_kept(
    video_media: MediaItem,
) -> None:
    # 並びを入れ替えると、奥にあった動画が手前の動画を隠す
    project = _stacked(LayerMode.SEPARATED, video_media, video_media)
    before = [t.id for t in project.timeline.video_tracks()]
    converted = _to(project, LayerMode.MIXED)
    assert [t.id for t in converted.timeline.tracks] == before
    assert [t.name for t in converted.timeline.tracks] == ["レイヤー 1", "レイヤー 2"]


def test_mute_and_solo_that_cannot_carry_over_are_listed(video_media: MediaItem) -> None:
    # 映像トラックのソロはレイヤーでは音にも効く 知らせずに変えると、変換しただけで
    # ほかのレイヤーの音が止まる
    project = _separated(video_media, video_media)
    first_video = next(project.timeline.video_tracks())
    project = SetTrackState(first_video.id, solo=True).apply(project)
    result = convert_layers(project, LayerMode.MIXED, SOUND_KINDS)
    assert any("音が止まる" in notice for notice in result.notices), result.notices


def test_nothing_is_listed_when_the_output_stays_the_same(
    video_media: MediaItem, audio_media: MediaItem
) -> None:
    # 何も変わらないのに一覧が出ると、利用者が毎回読むだけの一覧になる
    result = convert_layers(_separated(video_media, audio_media), LayerMode.MIXED, SOUND_KINDS)
    assert result.notices == ()


# --- 混合 → 分ける ---


def test_a_clip_with_picture_and_sound_splits_into_a_linked_pair(video_media: MediaItem) -> None:
    # 分けた絵と音が結ばれないと、変換のあとで動画をずらしても音が付いてこず口の動きと声がずれる
    mixed = _to(_separated(video_media), LayerMode.MIXED)
    separated = _to(mixed, LayerMode.SEPARATED)
    assert _kinds(separated) == [TrackKind.VIDEO, TrackKind.AUDIO]
    (picture,) = separated.timeline.tracks[0].clips
    (sound,) = separated.timeline.tracks[1].clips
    assert picture.link_group is not None and picture.link_group == sound.link_group
    # 音のエフェクトは音の側へ、絵のエフェクトは絵の側へ 残ると効かない欄が並ぶ
    assert [e.kind for e in picture.effects] == [FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND]
    assert [e.kind for e in sound.effects] == [VOLUME_EFFECT_KIND, FADE_EFFECT_KIND]
    assert sound.stream_index == video_media.audio_streams[0].index


def test_there_and_back_gives_the_same_project_apart_from_ids(video_media: MediaItem) -> None:
    # 行き帰りで何か 1 つでも変わると、方式を試しに切り替えて戻しただけで作品が変わる
    project = _separated(video_media)
    back = _to(_to(project, LayerMode.MIXED), LayerMode.SEPARATED)
    assert _shape(back.timeline) == _shape(project.timeline)


@pytest.mark.parametrize("mode", [LayerMode.SEPARATED, LayerMode.MIXED], ids=["分ける", "混合"])
def test_stacked_videos_and_music_come_back_the_same(
    video_media: MediaItem, audio_media: MediaItem, mode: str
) -> None:
    # 2 本の動画と BGM を同じ所に重ねる（映像 2 本と音声 3 本、またはレイヤー 3 本）
    # 重なり順や BGM のトラックが入れ替わると、戻したときに別の作品になる
    project = _stacked(mode, video_media, video_media, audio_media)
    other = LayerMode.MIXED if mode == LayerMode.SEPARATED else LayerMode.SEPARATED
    there = _to(project, other)
    assert set(_kinds(there)).isdisjoint(_kinds(project)), "変換されていない"
    back = _to(there, mode)
    assert _shape(back.timeline) == _shape(project.timeline)


def test_the_layer_volume_goes_to_the_audio_track(video_media: MediaItem) -> None:
    # 映像トラックは音量を使わないので、レイヤーの音量は音声トラックへ移さないと消える
    mixed = _to(_separated(video_media), LayerMode.MIXED)
    layer = replace(mixed.timeline.tracks[0], volume_db=-3.0, pan=0.2)
    mixed = mixed.with_timeline(mixed.timeline.replace_track(layer))
    video, audio = _to(mixed, LayerMode.SEPARATED).timeline.tracks
    assert (audio.volume_db, audio.pan) == (-3.0, 0.2)
    assert (video.volume_db, video.pan) == (0.0, 0.0)


def test_a_clip_showing_nothing_is_kept_disabled(video_media: MediaItem) -> None:
    # 絵を隠して音も選んでいない動画は、映像トラックへそのまま移すと絵が出てしまう
    clip = Clip(timeline_start=0, duration=30, media_id=video_media.id, show_picture=False)
    project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media,))
    layer = Track(kind=TrackKind.MIXED, name="レイヤー 1", clips=(clip,))
    project = project.with_timeline(replace(project.timeline, tracks=(layer,)))
    result = convert_layers(project, LayerMode.SEPARATED, SOUND_KINDS)
    (video,) = result.project.timeline.tracks
    assert not video.clips[0].enabled
    assert result.notices


def test_scenes_are_converted_too(video_media: MediaItem) -> None:
    # シーンの中だけ分ける方式のまま残ると、シーンを開いたときに置き方が混ざる
    inner = _separated(video_media).timeline
    project = replace(_separated(video_media), scenes=(Scene(name="OP", timeline=inner),))
    converted = _to(project, LayerMode.MIXED)
    assert {t.kind for t in converted.scenes[0].timeline.tracks} == {TrackKind.MIXED}


def test_a_scene_too_loud_to_copy_is_listed(video_media: MediaItem) -> None:
    # 映像トラックのシーンはトラックの音量を使わないので、レイヤーの音量をクリップへ写す
    # 上限（400%）で縮めたのに一覧が空だと、変わらないと言われたのにシーンの音が小さくなる
    inner = _separated(video_media).timeline
    scene = Scene(name="OP", timeline=inner)
    placed = Clip(timeline_start=0, duration=30, scene_id=scene.id)
    layer = Track(kind=TrackKind.MIXED, name="レイヤー 1", clips=(placed,), volume_db=10.0, pan=1.0)
    base = _separated(video_media)
    project = replace(base, scenes=(scene,)).with_timeline(replace(base.timeline, tracks=(layer,)))
    result = convert_layers(project, LayerMode.SEPARATED, SOUND_KINDS)
    assert any("上限" in notice for notice in result.notices), result.notices


# --- 命令と取り消し ---


def test_the_switch_and_the_conversion_undo_in_one_step(video_media: MediaItem) -> None:
    # 方式と変換を別々の段にすると、1 回戻しただけでは方式だけ戻った中途半端な作品になる
    project = _separated(video_media)
    document = Document(project)
    commands = switch_layer_mode(project, LayerMode.MIXED, convert=True, sound_kinds=SOUND_KINDS)
    with document.checkpoint("方式を切り替える"):
        for command in commands:
            document.execute(command)
    assert document.project.settings.layer_mode == LayerMode.MIXED
    assert _kinds(document.project) == [TrackKind.MIXED]
    document.undo()
    assert document.project is project


def test_switching_only_leaves_the_tracks_alone(video_media: MediaItem) -> None:
    # 「これから置く物だけ」を選んだのにトラックまで変わると、答えを聞いた意味が無い
    project = _separated(video_media)
    commands = switch_layer_mode(project, LayerMode.MIXED, convert=False, sound_kinds=SOUND_KINDS)
    converted = _apply(project, commands)
    assert converted.settings.layer_mode == LayerMode.MIXED
    assert converted.timeline == project.timeline


def test_converting_nothing_changes_nothing(video_media: MediaItem) -> None:
    # 変える物が無いのに新しいプロジェクトを返すと、何も起きない段が履歴に積まれる
    project = _separated(video_media)
    assert _to(project, LayerMode.SEPARATED) is project


def test_an_unknown_mode_is_refused(video_media: MediaItem) -> None:
    # 知らない方式を受けると「混合でない側」として扱われ、打ち間違えただけでレイヤーが
    # 黙って映像と音声に分かれる
    with pytest.raises(ValueError, match="方式"):
        _to(_separated(video_media), "layered")


def test_the_speed_and_source_position_must_match_to_pair(video_media: MediaItem) -> None:
    # 素材の中の位置が違う 2 本をまとめると、絵と音がずれる
    project = _separated(video_media)
    audio = project.timeline.tracks[1]
    (sound,) = audio.clips
    audio = audio.with_clips((replace(sound, source_in=Fraction(1)),))
    project = project.with_timeline(project.timeline.replace_track(audio))
    converted = _to(project, LayerMode.MIXED)
    assert len(converted.timeline.tracks) == 2


# --- エフェクトトラック（FX と番号の映像トラック） ---


def test_an_effect_track_becomes_an_ordinary_layer_and_comes_back(video_media: MediaItem) -> None:
    # 利用者の要望 分けない方式ではエフェクトのレイヤーを持たない 変換で名前を変えたり
    # 中身を動かしたりすると、戻したときにエフェクトトラックとフィルタの位置が失われる
    project = _separated(video_media)
    project = _apply(project, [new_track(project, TrackKind.VIDEO, effect=True)])
    fx = project.timeline.tracks[-1]
    project = _apply(project, insert_filter(project, at_frame=0, duration=30, track_id=fx.id))
    assert is_effect_track(project.timeline.tracks[-1])

    mixed = _to(project, LayerMode.MIXED)
    layer = mixed.timeline.tracks[-1]
    assert (layer.kind, layer.name) == (TrackKind.MIXED, "FX1")
    # 混合の方式では名前で振る舞いを変えない 普通のレイヤーとして扱う
    assert not is_effect_track(layer)
    (placed,) = layer.clips
    assert placed.is_filter

    back = _to(mixed, LayerMode.SEPARATED)
    restored = back.timeline.tracks[-1]
    assert (restored.kind, restored.name) == (TrackKind.VIDEO, "FX1")
    assert is_effect_track(restored)
    assert restored.clips[0].is_filter
