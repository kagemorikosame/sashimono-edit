"""混合トラック（YMM4 型のレイヤー Issue #27）のモデル・命令・保存

混合トラックは映像・音声・テキストを何でも置ける 1 本のレイヤー 音付きの動画は
絵と音を 1 本のクリップで持つ ここが崩れると、置いた動画の絵か音のどちらかが
黙って消える
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from sashimono.core.commands import (
    AddClip,
    AddScene,
    AddTrack,
    GroupClips,
    MoveClip,
    MoveClips,
    MoveTrack,
    RippleCut,
    SetFrameRate,
    SetLayerMode,
    SplitClip,
    TrimClip,
    new_scene,
    reorder_group,
)
from sashimono.core.io import ProjectFileError, project_from_dict, project_to_dict
from sashimono.core.io.serialize import FORMAT_VERSION
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    LayerMode,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    SceneId,
    Timeline,
    Track,
    TrackKind,
    default_track_name,
    draws_picture,
    plays_sound,
)
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)


def _names(tracks: tuple[Track, ...]) -> list[str]:
    return [t.name for t in tracks]


def _line(*tracks: Track) -> Timeline:
    return Timeline(rate=RATE, tracks=tracks)


def _mixed_clip(media: MediaItem, **changes: Any) -> Clip:
    """音付きの動画を混合トラックに置いたクリップ 絵は映像の、音は音声のストリーム"""
    clip = Clip(
        timeline_start=0,
        duration=90,
        media_id=media.id,
        stream_index=media.video_streams[0].index,
        audio_stream=media.audio_streams[0].index,
    )
    return replace(clip, **changes)


@pytest.fixture
def mixed_project(video_media: MediaItem, audio_media: MediaItem) -> Project:
    """混合トラック 2 本と映像トラック 1 本 レイヤー 1 に音付きの動画を置いてある"""
    base = Project.create(ProjectSettings(frame_rate=RATE), media=(video_media, audio_media))
    tracks = (
        Track(TrackKind.MIXED, "レイヤー 1", (_mixed_clip(video_media),)),
        Track(TrackKind.MIXED, "レイヤー 2"),
        Track(TrackKind.VIDEO, "V1"),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


class TestRoles:
    def test_mixed_tracks_both_draw_and_sound(self) -> None:
        # 混合トラックが片方の役割にしか入らないと、置いた動画の絵か音が消える
        line = _line(
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.AUDIO, "A1"),
            Track(TrackKind.MIXED, "レイヤー 2"),
        )
        assert _names(line.picture_tracks()) == ["V1", "レイヤー 1", "レイヤー 2"]
        assert _names(line.sound_tracks()) == ["レイヤー 1", "A1", "レイヤー 2"]

    def test_the_draw_order_is_the_track_order(self) -> None:
        # 並びの先頭が一番奥 レイヤー 1 が奥で番号が大きいほど手前（YMM4・AviUtl と同じ）
        # 種類ごとに並べ替えると、映像トラックと混合トラックが混ざったときに重なりが裏返る
        line = _line(
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.MIXED, "レイヤー 2"),
        )
        assert _names(line.active_picture_tracks()) == ["レイヤー 1", "V1", "レイヤー 2"]

    def test_the_old_kind_query_still_answers_for_separated_projects(self) -> None:
        # 分ける方式だけのタイムラインでは、今までの active_tracks と同じ答え
        # 答えが変わると、分ける方式の作品で今までの呼び手（画面の行の表示など）のソロの結果が変わる
        line = _line(
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.VIDEO, "V2", solo=True),
            Track(TrackKind.AUDIO, "A1"),
        )
        assert _names(line.active_tracks(TrackKind.VIDEO)) == ["V2"]
        assert _names(line.active_tracks(TrackKind.AUDIO)) == ["A1"]


class TestSolo:
    """ソロは役割（絵か音か）の中で決まる 混合トラックは両方に入る"""

    def test_a_soloed_layer_is_the_only_thing_seen_and_heard(self) -> None:
        # 種類の中で決めると、混合をソロにしても映像トラックが映り、音声トラックも鳴る
        line = _line(
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.MIXED, "レイヤー 1", solo=True),
            Track(TrackKind.MIXED, "レイヤー 2"),
            Track(TrackKind.AUDIO, "A1"),
        )
        assert _names(line.active_picture_tracks()) == ["レイヤー 1"]
        assert _names(line.active_sound_tracks()) == ["レイヤー 1"]

    def test_an_audio_solo_keeps_the_pictures(self) -> None:
        # 分ける方式で音声のソロが絵を消さないのと同じ 消すと、音を聞き比べる間に画面が消える
        line = _line(
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.AUDIO, "A1", solo=True),
        )
        assert _names(line.active_picture_tracks()) == ["V1", "レイヤー 1"]
        assert _names(line.active_sound_tracks()) == ["A1"]

    def test_a_video_solo_keeps_the_sound_of_layers(self) -> None:
        # 映像トラックのソロが音の側にまで効くと、絵を確かめる間にレイヤーの動画の音が止まる
        line = _line(
            Track(TrackKind.VIDEO, "V1", solo=True),
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.AUDIO, "A1"),
        )
        assert _names(line.active_picture_tracks()) == ["V1"]
        assert _names(line.active_sound_tracks()) == ["レイヤー 1", "A1"]

    def test_a_muted_layer_is_neither_seen_nor_heard(self) -> None:
        # 片方の役割にだけ効くと、ミュートしたレイヤーの絵か音のどちらかが残る
        line = _line(Track(TrackKind.MIXED, "レイヤー 1", muted=True))
        assert line.active_picture_tracks() == ()
        assert line.active_sound_tracks() == ()


class TestClipRoles:
    def test_a_video_with_sound_draws_and_plays_on_a_layer(self, video_media: MediaItem) -> None:
        # どちらかが偽になると、レイヤーに置いた音付きの動画の絵か音が黙って消える
        track = Track(TrackKind.MIXED)
        clip = _mixed_clip(video_media)
        assert draws_picture(track, clip, video_media)
        assert plays_sound(track, clip, video_media)

    def test_the_picture_can_be_hidden(self, video_media: MediaItem) -> None:
        # 隠しても描くと音だけ使いたい動画の絵が出て、隠すと音まで消えるならクリップを分けるしかない
        track = Track(TrackKind.MIXED)
        clip = _mixed_clip(video_media, show_picture=False)
        assert not draws_picture(track, clip, video_media)
        assert plays_sound(track, clip, video_media)

    def test_no_stream_means_no_sound(self, video_media: MediaItem) -> None:
        # 番号が無いのに鳴らすと、音を切ったつもりの動画が鳴り続ける
        track = Track(TrackKind.MIXED)
        clip = _mixed_clip(video_media, audio_stream=None)
        assert not plays_sound(track, clip, video_media)

    def test_sound_only_media_draws_nothing_on_a_layer(self, audio_media: MediaItem) -> None:
        # 重ねに残すと、上のクリップがそれで切り抜いて何も映らなくなる
        track = Track(TrackKind.MIXED)
        clip = Clip(0, 30, media_id=audio_media.id, audio_stream=0)
        assert not draws_picture(track, clip, audio_media)
        assert plays_sound(track, clip, audio_media)

    def test_generated_clips_draw_but_stay_silent(self) -> None:
        # テキストを鳴らす側に数えると、テキストしか無いレイヤーの書き出しに黙った音声が付く
        track = Track(TrackKind.MIXED)
        clip = Clip(0, 30, source=GeneratedSource(kind="text"))
        assert draws_picture(track, clip, None)
        assert not plays_sound(track, clip, None)

    def test_scenes_draw_and_play_on_a_layer(self) -> None:
        # シーンの音を鳴らさないと、レイヤーに置いたシーンの中の BGM やナレーションが消える
        clip = Clip(0, 30, scene_id=SceneId("scene"))
        assert draws_picture(Track(TrackKind.MIXED), clip, None)
        assert plays_sound(Track(TrackKind.MIXED), clip, None)

    def test_separated_tracks_keep_their_rules(self, video_media: MediaItem) -> None:
        # 映像トラックの動画は鳴らさず（音は組の音声クリップが鳴らす）、音声トラックは描かない
        # 映像トラックが鳴ると組の音声クリップと 2 重に鳴り、音声トラックが描くと絵が 2 枚重なる
        clip = _mixed_clip(video_media, audio_stream=None, show_picture=False)
        assert draws_picture(Track(TrackKind.VIDEO), clip, video_media)
        assert not plays_sound(Track(TrackKind.VIDEO), clip, video_media)
        assert not draws_picture(Track(TrackKind.AUDIO), clip, video_media)
        assert plays_sound(Track(TrackKind.AUDIO), clip, video_media)

    def test_a_stream_the_media_lacks_stays_silent(self, video_media: MediaItem) -> None:
        # 手で直したファイルや差し替えた素材で番号が外れても、先頭の音を鳴らさない
        clip = _mixed_clip(video_media, audio_stream=7)
        assert not plays_sound(Track(TrackKind.MIXED), clip, video_media)

    def test_a_negative_stream_is_refused(self) -> None:
        # 負の番号を通すと、鳴らすときにどのストリームを開くかがデコーダの作り次第になる
        with pytest.raises(ValueError, match="音声ストリーム"):
            Clip(0, 30, audio_stream=-1)

    def test_layers_are_named_like_ymm4(self) -> None:
        # 頭の文字を映像か音声の 2 択で決めると、レイヤーに「A3」のような名前が付く
        assert default_track_name(TrackKind.MIXED, 1) == "レイヤー 1"
        assert default_track_name(TrackKind.VIDEO, 2) == "V2"
        assert default_track_name(TrackKind.AUDIO, 3) == "A3"

    def test_a_name_left_behind_is_skipped(self) -> None:
        # レイヤー 1 を消してレイヤー 2 が残ると、本数で数えた名前がもう 1 本のレイヤー 2 になる
        assert default_track_name(TrackKind.MIXED, 2, {"レイヤー 2"}) == "レイヤー 3"


class TestCommands:
    def test_any_media_goes_on_a_layer(
        self, mixed_project: Project, audio_media: MediaItem
    ) -> None:
        # 混合トラックで素材の種類を断ると、BGM をレイヤーに置けない
        layer = mixed_project.timeline.tracks[1]
        clip = Clip(0, 30, media_id=audio_media.id, audio_stream=0)
        placed = AddClip(layer.id, clip).apply(mixed_project)
        assert placed.timeline.tracks[1].clips == (clip,)

    def test_a_sounding_clip_cannot_move_to_a_video_track(self, mixed_project: Project) -> None:
        # 映像トラックはクリップの音を鳴らさない 移せると音が黙って消える
        clip = mixed_project.timeline.tracks[0].clips[0]
        video = mixed_project.timeline.tracks[2]
        with pytest.raises(ValueError, match="音が鳴らなくなる"):
            MoveClip(clip.id, 0, video.id).apply(mixed_project)

    def test_a_hidden_picture_cannot_move_to_a_video_track(self, mixed_project: Project) -> None:
        # 映像トラックは show_picture を読まずに描く 移せると、隠していた絵が黙って映り出す
        clip = mixed_project.timeline.tracks[0].clips[0]
        project = _set_clip(mixed_project, replace(clip, audio_stream=None, show_picture=False))
        video = project.timeline.tracks[2]
        with pytest.raises(ValueError, match="絵が出てしまう"):
            MoveClip(clip.id, 0, video.id).apply(project)
        text = Clip(40, 30, source=GeneratedSource(kind="text"), show_picture=False)
        project = AddClip(project.timeline.tracks[1].id, text).apply(project)
        with pytest.raises(ValueError, match="絵が出てしまう"):
            MoveClip(text.id, 40, video.id).apply(project)

    def test_a_silent_clip_can_move_to_a_video_track(self, mixed_project: Project) -> None:
        # 音を鳴らさないクリップまで断ると、絵だけの動画をレイヤーから映像トラックへ戻せない
        clip = mixed_project.timeline.tracks[0].clips[0]
        project = _set_clip(mixed_project, replace(clip, audio_stream=None))
        video = project.timeline.tracks[2]
        moved = MoveClip(clip.id, 0, video.id).apply(project)
        assert moved.timeline.tracks[2].clips[0].id == clip.id

    def test_a_stream_the_media_lacks_is_refused(self, mixed_project: Project) -> None:
        # デコーダは無い番号を頼まれると先頭の音へ逃げる 置けると、選んでいない言語が鳴る
        layer = mixed_project.timeline.tracks[1]
        clip = _mixed_clip(mixed_project.media[0], audio_stream=5)
        with pytest.raises(ValueError, match="音声ストリーム 5"):
            AddClip(layer.id, clip).apply(mixed_project)

    def test_a_clip_of_missing_media_is_refused(self, mixed_project: Project) -> None:
        # 音を鳴らさないクリップで確かめを飛ばすと、無い素材を指すクリップが残る
        layer = mixed_project.timeline.tracks[1]
        clip = Clip(0, 30, media_id=MediaId("無い素材"))
        with pytest.raises(KeyError, match="素材が見つからない"):
            AddClip(layer.id, clip).apply(mixed_project)

    def test_an_audio_clip_keeps_its_sound_on_a_layer_and_back(
        self, mixed_project: Project, video_media: MediaItem
    ) -> None:
        # 移し替えないと、音声クリップをレイヤーへ移しただけで音が消え、戻すと
        # 絵のストリームの番号で音を開く
        audio = Track(TrackKind.AUDIO, "A1")
        project = AddTrack(audio).apply(mixed_project)
        clip = Clip(0, 30, media_id=video_media.id, stream_index=1)
        project = AddClip(audio.id, clip).apply(project)
        layer_2 = project.timeline.tracks[1]

        on_layer = MoveClip(clip.id, 0, layer_2.id).apply(project)
        (moved,) = on_layer.timeline.tracks[1].clips
        assert (moved.audio_stream, moved.stream_index) == (1, 0)
        assert on_layer.plays_sound(on_layer.timeline.tracks[1], moved)
        # 絵を出し始めると、リンクした映像クリップの絵がもう 1 枚重なる
        assert not on_layer.draws_picture(on_layer.timeline.tracks[1], moved)

        back = MoveClip(clip.id, 0, audio.id).apply(on_layer)
        (returned,) = back.timeline.tracks[-1].clips
        assert (returned.stream_index, returned.audio_stream) == (1, None)

    def test_a_drawn_clip_cannot_move_to_an_audio_track(self, mixed_project: Project) -> None:
        # 映像トラックへ音を鳴らすクリップを移せないのと同じく、絵が黙って消える
        audio = Track(TrackKind.AUDIO, "A1")
        project = AddTrack(audio).apply(mixed_project)
        clip = project.timeline.tracks[0].clips[0]
        with pytest.raises(ValueError, match="絵が消える"):
            MoveClip(clip.id, 0, audio.id).apply(project)

    def test_a_text_or_scene_on_a_layer_cannot_move_to_an_audio_track(
        self, mixed_project: Project
    ) -> None:
        # 素材が無いからと確かめを飛ばすと、レイヤーで描いていたテキストやシーンの絵が消える
        audio = Track(TrackKind.AUDIO, "A1")
        project, scene_id = _with_scene(AddTrack(audio).apply(mixed_project))
        layer_2 = project.timeline.tracks[1]
        text = Clip(0, 30, source=GeneratedSource(kind="text"))
        scene = Clip(40, 30, scene_id=scene_id)
        project = AddClip(layer_2.id, text).apply(project)
        project = AddClip(layer_2.id, scene).apply(project)
        for clip in (text, scene):
            with pytest.raises(ValueError, match="絵が消える"):
                MoveClip(clip.id, clip.timeline_start, audio.id).apply(project)

    def test_a_scene_from_an_audio_track_stays_sound_only_on_a_layer(
        self, mixed_project: Project
    ) -> None:
        # 音声トラックでは音だけを出していたシーンが、レイヤーへ移しただけで絵を描き始める
        audio = Track(TrackKind.AUDIO, "A1")
        project, scene_id = _with_scene(AddTrack(audio).apply(mixed_project))
        scene = Clip(0, 30, scene_id=scene_id)
        project = AddClip(audio.id, scene).apply(project)
        layer_2 = project.timeline.tracks[1]
        moved = MoveClip(scene.id, 0, layer_2.id).apply(project)
        (arrived,) = moved.timeline.tracks[1].clips
        assert not moved.draws_picture(moved.timeline.tracks[1], arrived)
        assert moved.plays_sound(moved.timeline.tracks[1], arrived)

    def test_a_silent_layer_clip_cannot_move_to_an_audio_track(
        self, mixed_project: Project
    ) -> None:
        # 音声トラックはクリップの音を鳴らすので、黙らせていた音が鳴り出す
        audio = Track(TrackKind.AUDIO, "A1")
        project = AddTrack(audio).apply(mixed_project)
        clip = project.timeline.tracks[0].clips[0]
        project = _set_clip(project, replace(clip, show_picture=False, audio_stream=None))
        with pytest.raises(ValueError, match="鳴り出して"):
            MoveClip(clip.id, 0, audio.id).apply(project)

    def test_moving_between_layers_keeps_picture_and_sound(self, mixed_project: Project) -> None:
        # 移し替えがレイヤー同士にまで効くと、別のレイヤーへ動かしただけで音か絵が消える
        clip = mixed_project.timeline.tracks[0].clips[0]
        layer_2 = mixed_project.timeline.tracks[1]
        moved = MoveClip(clip.id, 15, layer_2.id).apply(mixed_project)
        (arrived,) = moved.timeline.tracks[1].clips
        assert (arrived.timeline_start, arrived.audio_stream, arrived.show_picture) == (15, 1, True)

    def test_shifting_by_rows_stays_among_layers(self, mixed_project: Project) -> None:
        # 行ずらしが映像トラックへ入ると、音付きの動画が映像トラックへ落ちて音が消える
        clip = mixed_project.timeline.tracks[0].clips[0]
        moved = MoveClips((clip.id,), 0, track_delta=1).apply(mixed_project)
        assert moved.timeline.tracks[1].clips[0].id == clip.id
        with pytest.raises(ValueError):
            MoveClips((clip.id,), 0, track_delta=2).apply(mixed_project)

    def test_split_halves_keep_the_stream_and_the_picture_switch(
        self, mixed_project: Project
    ) -> None:
        # 落とすと、分けた後半だけ音が消える・絵が出る
        clip = mixed_project.timeline.tracks[0].clips[0]
        project = _set_clip(mixed_project, replace(clip, show_picture=False))
        split = SplitClip(clip.id, 30).apply(project)
        halves = split.timeline.tracks[0].clips
        assert len(halves) == 2
        assert all(h.audio_stream == 1 and not h.show_picture for h in halves)
        assert halves[1].source_in == Fraction(1)

    def test_trim_keeps_the_stream(self, mixed_project: Project) -> None:
        # 作り直すときに項目を落とすと、端を削っただけで動画の音が消える
        clip = mixed_project.timeline.tracks[0].clips[0]
        trimmed = TrimClip(clip.id, head_delta=10).apply(mixed_project)
        (kept,) = trimmed.timeline.tracks[0].clips
        assert kept.audio_stream == 1
        assert kept.timeline_start == 10

    def test_ripple_cut_closes_the_gap_on_layers(self, mixed_project: Project) -> None:
        # 詰めがレイヤーに効かないと、ジェットカットの後にレイヤーの動画だけが元の位置に残ってずれる
        cut = RippleCut(((30, 60),)).apply(mixed_project)
        clips = cut.timeline.tracks[0].clips
        assert [(c.timeline_start, c.duration) for c in clips] == [(0, 30), (30, 30)]
        assert all(c.audio_stream == 1 for c in clips)

    def test_grouping_works_on_layers(self, mixed_project: Project, audio_media: MediaItem) -> None:
        # レイヤーのクリップを束ねられないと、字幕と動画をまとめて動かす手間が混合でだけ増える
        layer_2 = mixed_project.timeline.tracks[1]
        other = Clip(0, 30, media_id=audio_media.id, audio_stream=0)
        project = AddClip(layer_2.id, other).apply(mixed_project)
        first = project.timeline.tracks[0].clips[0]
        grouped = GroupClips((first.id, other.id)).apply(project)
        groups = {c.group_id for t in grouped.timeline.tracks[:2] for c in t.clips}
        assert len(groups) == 1 and None not in groups


class TestLayerMode:
    def test_the_model_default_is_separated(self) -> None:
        # 既定を混合にすると、古いファイルと今の試験の置き方が黙って変わる
        assert ProjectSettings().layer_mode == LayerMode.SEPARATED

    def test_the_command_changes_only_the_setting(self, mixed_project: Project) -> None:
        # 設定を変えるだけでトラックまで変えると、方式を戻したいだけでも置いた物が作り直される
        changed = SetLayerMode(LayerMode.MIXED).apply(mixed_project)
        assert changed.settings.layer_mode == LayerMode.MIXED
        assert changed.timeline is mixed_project.timeline

    def test_an_unknown_mode_is_refused(self, mixed_project: Project) -> None:
        # 知らない値を通すと、置き方がどちらになるかが置く側の作り次第になり、保存にもそのまま残る
        with pytest.raises(ValueError, match="方式"):
            SetLayerMode("both").apply(mixed_project)
        with pytest.raises(ValueError, match="方式"):
            ProjectSettings(layer_mode="both")


class TestSaving:
    def test_the_format_is_at_least_7(self) -> None:
        # 上げ忘れると形式 6 の本体がこのファイルを開き、混合トラックで止まるか、
        # audio_stream と show_picture を黙って落として保存し直す
        # 8 で字幕を音声ごとにした（test_stream_transcripts.py）
        assert FORMAT_VERSION >= 7

    def test_layers_and_their_clips_round_trip(self, mixed_project: Project) -> None:
        # 落とすと、保存して開き直しただけで動画の音が消える・隠した絵が出る
        clip = mixed_project.timeline.tracks[0].clips[0]
        project = _set_clip(mixed_project, replace(clip, show_picture=False))
        project = SetLayerMode(LayerMode.MIXED).apply(project)
        loaded = project_from_dict(project_to_dict(project))
        assert loaded.timeline.tracks[0].kind is TrackKind.MIXED
        assert loaded.timeline.tracks[0].clips[0].audio_stream == 1
        assert loaded.timeline.tracks[0].clips[0].show_picture is False
        assert loaded.settings.layer_mode == LayerMode.MIXED
        assert loaded == project

    def test_version_6_opens_as_separated(self, project: Project, video_media: MediaItem) -> None:
        # 6 までのファイルに項目は無い 変換なしで、分ける方式のまま開けること
        # 既定値で補わないと、形式 6 までに保存した作品がすべて開けなくなる
        placed = AddClip(project.timeline.tracks[0].id, Clip(0, 30, media_id=video_media.id)).apply(
            project
        )
        data = project_to_dict(placed)
        data["version"] = 6
        del data["settings"]["layer_mode"]
        for track in data["timeline"]["tracks"]:
            for clip in track["clips"]:
                del clip["audio_stream"]
                del clip["show_picture"]
        loaded = project_from_dict(data)
        assert loaded.settings.layer_mode == LayerMode.SEPARATED
        (clip,) = loaded.timeline.tracks[0].clips
        assert (clip.audio_stream, clip.show_picture) == (None, True)
        assert loaded == placed

    def test_a_newer_file_asks_for_an_update(self, mixed_project: Project) -> None:
        # 開けてしまうと、形式 8 の知らない項目を黙って落とし、保存し直すと中身が消える
        data = project_to_dict(mixed_project)
        data["version"] = FORMAT_VERSION + 1
        with pytest.raises(ProjectFileError, match="更新"):
            project_from_dict(data)

    def test_a_broken_stream_number_is_a_file_error(self, mixed_project: Project) -> None:
        # 文字のまま通すと、鳴らすときに型の違いで落ちる
        data = project_to_dict(mixed_project)
        data["timeline"]["tracks"][0]["clips"][0]["audio_stream"] = "1"
        with pytest.raises(ProjectFileError, match="audio_stream"):
            project_from_dict(data)


def test_a_layer_can_be_added(mixed_project: Project) -> None:
    # トラックの追加が種類を映像か音声へ寄せると、足したレイヤーに動画を置いても音が鳴らない
    track = Track(TrackKind.MIXED, "レイヤー 3")
    added = AddTrack(track).apply(mixed_project)
    assert added.timeline.tracks[-1].kind is TrackKind.MIXED


class TestTrackOrderAndFormat:
    """並べ替え（MoveTrack）とフレームレートの変更（SetFrameRate）が混合トラックでも効くこと"""

    def test_layers_reorder_among_layers(self) -> None:
        # 映像トラックとまたがせると、画面で落とした所と違う場所に出る
        l1, v1, l2, a1 = (
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.MIXED, "レイヤー 2"),
            Track(TrackKind.AUDIO, "A1"),
        )
        project = Project.create().with_timeline(Timeline(rate=RATE, tracks=(l1, v1, l2, a1)))
        assert [t.name for t in reorder_group(project.timeline, l1)] == ["レイヤー 1", "レイヤー 2"]
        moved = MoveTrack(l1.id, 1).apply(project)
        # 映像トラックと音声トラックの席は動かない レイヤーの席だけが入れ替わる
        assert _names(moved.timeline.tracks) == ["レイヤー 2", "V1", "レイヤー 1", "A1"]
        # 並びが重なり順なので、動かしたレイヤーが手前へ出る
        assert _names(moved.timeline.active_picture_tracks()) == ["レイヤー 2", "V1", "レイヤー 1"]

    def test_in_mixed_mode_every_track_is_a_partner(self, mixed_project: Project) -> None:
        # レイヤーだけの作品では全部を並べ替えられる 仲間が欠けると、動かせない行が出る
        layers = tuple(t for t in mixed_project.timeline.tracks if t.kind is TrackKind.MIXED)
        only = mixed_project.with_timeline(replace(mixed_project.timeline, tracks=layers))
        assert reorder_group(only.timeline, layers[1]) == layers

    def test_the_rate_changes_with_layers_kept(self) -> None:
        # 空のプロジェクトでレートを変えたときに、レイヤーや方式が落ちると置き方が変わる
        base = SetLayerMode(LayerMode.MIXED).apply(Project.create())
        layer = Track(TrackKind.MIXED, "レイヤー 1")
        project = AddTrack(layer).apply(base)
        changed = SetFrameRate(FrameRate(60)).apply(project)
        assert changed.rate == FrameRate(60)
        assert [t.kind for t in changed.timeline.tracks] == [TrackKind.MIXED]
        assert changed.settings.layer_mode == LayerMode.MIXED

    def test_the_rate_is_locked_once_a_layer_has_a_clip(self, mixed_project: Project) -> None:
        # 混合トラックのクリップを数え忘れると、置いた動画の長さと位置が黙って換算される
        with pytest.raises(ValueError, match="フレームレート"):
            SetFrameRate(FrameRate(60)).apply(mixed_project)


def _with_scene(project: Project) -> tuple[Project, SceneId]:
    """空のシーンを 1 つ足す 無いシーンを指すクリップはプロジェクトが断る"""
    scene = new_scene(project, "中")
    return AddScene(scene).apply(project), scene.id


def _set_clip(project: Project, clip: Clip) -> Project:
    located = project.timeline.locate_clip(clip.id)
    assert located is not None
    track, _ = located
    clips = tuple(clip if c.id == clip.id else c for c in track.clips)
    return project.with_timeline(project.timeline.replace_track(track.with_clips(clips)))
