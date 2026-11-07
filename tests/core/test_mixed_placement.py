"""混合の方式（YMM4 型のレイヤー Issue #27）での置き方

プロジェクトの方式が混合なら、素材・テキスト・フィルタ・シーン・貼り付け・字幕の
焼き込み・ドロップが、どれもレイヤー（混合トラック）へ入る 音付きの動画は絵と音の
1 本のクリップで、リンクした 2 本には分けない ここが崩れると、混合にしたのに
映像トラックと音声トラックが増えていき、レイヤーの重なり順も効かなくなる
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.clipboard import copy_clips, paste_commands
from sashimono.core.commands import (
    AddClip,
    AddScene,
    AddTrack,
    Command,
    GroupClips,
    MoveClip,
    MoveTrack,
    RippleCut,
    SetFrameRate,
    SetLayerMode,
    SetTranscript,
    SplitClips,
    TrimClip,
    burn_subtitles,
    format_to_match,
    insert_filter,
    insert_generated,
    insert_media,
    insert_scene,
    match_commands,
    new_scene,
    place_media,
)
from sashimono.core.commands.fixed import (
    FIXED_ORDER,
    PICTURE_FIXED,
    SOUND_FIXED,
    with_fixed_items,
)
from sashimono.core.commands.insert import VOLUME_EFFECT_KIND, is_effect_track, new_track
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from tests.conftest import RATE_30

TEXT = GeneratedSource(kind="text", params={"text": "テロップ"})


def _mixed(*tracks: Track, media: tuple[MediaItem, ...] = ()) -> Project:
    base = Project.create(
        ProjectSettings(frame_rate=RATE_30, layer_mode=LayerMode.MIXED), media=media
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _added(commands: list[Command]) -> list[AddClip]:
    return [c for c in commands if isinstance(c, AddClip)]


def _kinds(project: Project) -> list[TrackKind]:
    return [t.kind for t in project.timeline.tracks]


@pytest.fixture
def still_media() -> MediaItem:
    """1 枚の画像"""
    return MediaItem(
        path=Path("C:/素材/背景.png"),
        duration=Fraction(0),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=RATE_30,
                time_base=Fraction(1, 30),
                codec="png",
            ),
        ),
    )


class TestMedia:
    def test_a_video_with_sound_is_one_clip_on_a_layer(self, video_media: MediaItem) -> None:
        # 分けて置くと、混合にしたのに映像トラックと音声トラックができ、リンクの 2 本になる
        placed = _apply(_mixed(), insert_media(_mixed(), video_media))
        (layer,) = placed.timeline.tracks
        assert layer.kind is TrackKind.MIXED
        assert layer.name == "レイヤー 1"
        (clip,) = layer.clips
        assert clip.stream_index == video_media.video_streams[0].index
        assert clip.audio_stream == video_media.audio_streams[0].index
        assert clip.show_picture
        assert clip.link_group is None
        assert placed.draws_picture(layer, clip)
        assert placed.plays_sound(layer, clip)

    def test_the_fixed_volume_comes_along(self, video_media: MediaItem) -> None:
        # 音を持つクリップに固定の音量が無いと、置いた直後に音量をいじれない（#145）
        (added,) = _added(insert_media(_mixed(), video_media))
        volumes = [e for e in added.clip.effects if e.kind == VOLUME_EFFECT_KIND]
        assert len(volumes) == 1
        assert volumes[0].fixed

    def test_an_image_draws_and_stays_silent(self, still_media: MediaItem) -> None:
        # 音の無い素材に音のストリームや音量を付けると、鳴らない音の項目がパネルに出る
        commands = insert_media(_mixed(), still_media)
        (added,) = _added(commands)
        assert _kinds(_apply(_mixed(), commands)) == [TrackKind.MIXED]
        assert added.clip.show_picture
        assert added.clip.audio_stream is None
        assert not any(e.kind == VOLUME_EFFECT_KIND for e in added.clip.effects)

    def test_sound_only_media_hides_its_picture(self, audio_media: MediaItem) -> None:
        # 絵を出す印のままだと、描く物の無いクリップが重ねに残り、上の切り抜きの相手になる
        placed = _apply(_mixed(), insert_media(_mixed(), audio_media))
        (layer,) = placed.timeline.tracks
        (clip,) = layer.clips
        assert not clip.show_picture
        assert clip.audio_stream == audio_media.audio_streams[0].index
        assert placed.plays_sound(layer, clip)
        assert any(e.kind == VOLUME_EFFECT_KIND and e.fixed for e in clip.effects)

    def test_imports_follow_each_other_on_layer_1(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 末尾へ続けて置く物が毎回新しいレイヤーへ入ると、読み込むほどレイヤーが増える
        project = _mixed()
        for media in (video_media, audio_media):
            project = _apply(project, insert_media(project, media))
        (layer,) = project.timeline.tracks
        assert [c.timeline_start for c in layer.clips] == [0, 300]

    def test_a_busy_range_goes_to_a_new_layer_in_front(self, video_media: MediaItem) -> None:
        # 空きを見ずにレイヤー 1 へ入れると、重なりで断られて何も置かれない
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        project = _apply(project, insert_media(project, video_media, at_frame=30))
        assert [t.name for t in project.timeline.tracks] == ["レイヤー 1", "レイヤー 2"]
        assert project.timeline.tracks[1].clips[0].timeline_start == 30

    def test_the_separated_default_still_links_two_clips(self, video_media: MediaItem) -> None:
        # 既定の方式の置き方まで変わると、今までの作品の編集の仕方が変わる
        base = Project.create(ProjectSettings(frame_rate=RATE_30))
        commands = insert_media(base, video_media)
        video, audio = _added(commands)
        assert video.clip.link_group is not None
        assert video.clip.link_group == audio.clip.link_group
        assert _kinds(_apply(base, commands)) == [TrackKind.VIDEO, TrackKind.AUDIO]


def _fixed_kinds(clip: Clip) -> list[str]:
    return [e.kind for e in clip.effects if e.fixed]


class TestFixedItems:
    """置いた 1 本のクリップが、絵と音の両方の固定の項目を持つこと（P2 #158）"""

    def test_a_video_with_sound_has_picture_and_sound_items(self, video_media: MediaItem) -> None:
        # 絵の側だけ・音の側だけを付けると、パネルに位置か音量の欄が欠ける
        # 両方から同じ種類を 2 つ持つと、どちらがパネルの欄なのか分からなくなる
        (added,) = _added(insert_media(_mixed(), video_media))
        bare = replace(added.clip, effects=())
        expected = with_fixed_items(bare, picture=True, sound=True)

        # ID は作るたびに違うので、種類・値・印で比べる
        def shape(clip: Clip) -> list[tuple[str, object, bool]]:
            return [(e.kind, e.params, e.fixed) for e in clip.effects]

        assert shape(added.clip) == shape(expected)
        assert _fixed_kinds(added.clip) == list(FIXED_ORDER)

    def test_the_picture_is_placed_at_its_own_size(self, video_media: MediaItem) -> None:
        # 画面に収めて置くと、分ける方式で置いた絵と大きさが変わる
        (added,) = _added(insert_media(_mixed(), video_media))
        assert added.clip.native_size

    def test_an_image_has_only_picture_items(self, still_media: MediaItem) -> None:
        # 音の無い素材に音量やフェードの欄があると、効かない欄がパネルに並ぶ
        (added,) = _added(insert_media(_mixed(), still_media))
        assert _fixed_kinds(added.clip) == list(PICTURE_FIXED)
        assert added.clip.native_size

    def test_sound_only_media_has_only_sound_items(self, audio_media: MediaItem) -> None:
        # 絵を出さないクリップに反転や配置の欄があると、効かない欄がパネルに並ぶ
        (added,) = _added(insert_media(_mixed(), audio_media))
        assert _fixed_kinds(added.clip) == list(SOUND_FIXED)

    def test_the_items_match_the_separated_pair(self, video_media: MediaItem) -> None:
        # 分ける方式の 2 本と中身が違うと、方式を変えただけで絵や音が変わる
        base = Project.create(ProjectSettings(frame_rate=RATE_30))
        video, audio = _added(insert_media(base, video_media))
        (mixed,) = _added(insert_media(_mixed(), video_media))
        assert [e.kind for e in mixed.clip.effects] == [
            *(e.kind for e in video.clip.effects),
            *(e.kind for e in audio.clip.effects),
        ]
        assert [e.params for e in mixed.clip.effects] == [
            *(e.params for e in video.clip.effects),
            *(e.params for e in audio.clip.effects),
        ]

    def test_text_on_a_layer_has_picture_items(self) -> None:
        # テキストだけ描画の欄が無いと、位置を直すのに変形を探して足すことになる
        (added,) = _added(insert_generated(_mixed(), TEXT, at_frame=0))
        assert _fixed_kinds(added.clip) == list(PICTURE_FIXED)


class TestPictures:
    def test_text_goes_in_front_of_the_picture(self, video_media: MediaItem) -> None:
        # 奥から空きを探すと、空いているレイヤー 1 へ入り、レイヤー 2 の動画の後ろに隠れる
        clip = Clip(0, 300, media_id=video_media.id, audio_stream=1)
        project = _mixed(
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.MIXED, "レイヤー 2", (clip,)),
            media=(video_media,),
        )
        placed = _apply(project, insert_generated(project, TEXT, at_frame=30))
        assert [t.name for t in placed.timeline.tracks] == [
            "レイヤー 1",
            "レイヤー 2",
            "レイヤー 3",
        ]
        assert placed.timeline.tracks[2].clips[0].source == TEXT

    def test_a_video_track_in_front_counts_as_a_picture(self, video_media: MediaItem) -> None:
        # 方式を切り替えた作品では、映像トラックがレイヤーより手前にあることがある
        # 映像トラックを数えないと、テキストが空いたレイヤー 1 へ入り、動画の後ろに隠れる
        clip = Clip(0, 300, media_id=video_media.id)
        project = _mixed(
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.VIDEO, "V1", (clip,)),
            media=(video_media,),
        )
        placed = _apply(project, insert_generated(project, TEXT, at_frame=30))
        assert not placed.timeline.tracks[0].clips
        front = placed.timeline.tracks[-1]
        assert front.kind is TrackKind.MIXED
        assert front.clips[0].source is not None

    def test_text_uses_a_free_layer_above_the_picture(self, video_media: MediaItem) -> None:
        # 手前に空いたレイヤーがあるのに足すと、置くたびにレイヤーが増える
        clip = Clip(0, 300, media_id=video_media.id, audio_stream=1)
        project = _mixed(
            Track(TrackKind.MIXED, "レイヤー 1", (clip,)),
            Track(TrackKind.MIXED, "レイヤー 2"),
            media=(video_media,),
        )
        commands = insert_generated(project, TEXT, at_frame=30)
        assert not [c for c in commands if isinstance(c, AddTrack)]
        (added,) = _added(commands)
        assert added.track_id == project.timeline.tracks[1].id

    def test_a_sound_only_clip_does_not_push_text_forward(self, audio_media: MediaItem) -> None:
        # 音だけのクリップは絵を描かない それより手前へ回すと、レイヤーが無駄に増える
        bgm = Clip(0, 300, media_id=audio_media.id, audio_stream=0, show_picture=False)
        project = _mixed(
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.MIXED, "レイヤー 2", (bgm,)),
            media=(audio_media,),
        )
        (added,) = _added(insert_generated(project, TEXT, at_frame=0))
        assert added.track_id == project.timeline.tracks[0].id

    def test_a_filter_covers_the_pictures_below(self, video_media: MediaItem) -> None:
        # フィルタは下の絵にしか効かない 動画より奥へ入ると、置いたのに何も変わらない
        clip = Clip(0, 300, media_id=video_media.id, audio_stream=1)
        project = _mixed(
            Track(TrackKind.MIXED, "レイヤー 1"),
            Track(TrackKind.MIXED, "レイヤー 2", (clip,)),
            media=(video_media,),
        )
        placed = _apply(project, insert_filter(project, at_frame=0))
        front = placed.timeline.tracks[-1]
        assert front.kind is TrackKind.MIXED
        assert front.clips[0].is_filter

    def test_a_scene_goes_on_a_layer(self) -> None:
        # シーンを映像トラックへ置くと、混合の作品でシーンの音だけトラックの音量が効かない
        scene = new_scene(_mixed(), "場面")
        project = AddScene(scene).apply(_mixed())
        placed = _apply(project, insert_scene(project, scene.id, at_frame=0))
        (layer,) = placed.timeline.tracks
        assert layer.kind is TrackKind.MIXED
        assert layer.clips[0].scene_id == scene.id

    def test_a_right_clicked_layer_is_used(self) -> None:
        # 右クリックした所へ置かないと、どのレイヤーに入ったのかを探すことになる
        back, front = Track(TrackKind.MIXED, "レイヤー 1"), Track(TrackKind.MIXED, "レイヤー 2")
        project = _mixed(back, front)
        (added,) = _added(insert_generated(project, TEXT, at_frame=0, track_id=back.id))
        assert added.track_id == back.id

    def test_a_right_clicked_muted_layer_is_not_used(self) -> None:
        # 右クリックした所を鍵と空きだけで受けると、ミュートしたレイヤーへ置けて映らない
        muted = Track(TrackKind.MIXED, "レイヤー 1", muted=True)
        visible = Track(TrackKind.MIXED, "レイヤー 2")
        project = _mixed(muted, visible)
        (added,) = _added(insert_generated(project, TEXT, at_frame=0, track_id=muted.id))
        assert added.track_id == visible.id

    def test_a_muted_layer_is_skipped(self) -> None:
        # ミュートしたレイヤーへ置くと、置いたのにプレビューにも書き出しにも出ない
        muted = Track(TrackKind.MIXED, "レイヤー 1", muted=True)
        project = _mixed(muted)
        placed = _apply(project, insert_generated(project, TEXT, at_frame=0))
        assert not placed.timeline.tracks[0].clips
        assert _kinds(placed) == [TrackKind.MIXED, TrackKind.MIXED]
        assert len(placed.timeline.tracks[1].clips) == 1


class TestSolo:
    def test_a_new_layer_joins_the_solo(self, video_media: MediaItem) -> None:
        # ソロの間に作ったレイヤーにソロが無いと、置いた動画が映らず鳴らない
        clip = Clip(0, 300, media_id=video_media.id, audio_stream=1)
        project = _mixed(
            Track(TrackKind.MIXED, "レイヤー 1", (clip,), solo=True), media=(video_media,)
        )
        placed = _apply(project, insert_media(project, video_media, at_frame=0))
        new = placed.timeline.tracks[-1]
        assert new.solo
        assert new in placed.timeline.active_picture_tracks()

    def test_an_audio_solo_does_not_solo_a_picture_layer(self, video_media: MediaItem) -> None:
        # 音声トラックのソロで新しいレイヤーにソロが付くと、ほかの絵がすべて消える
        project = _mixed(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1", solo=True))
        placed = _apply(project, insert_generated(project, TEXT, at_frame=0))
        assert not placed.timeline.tracks[-1].solo

    def test_a_sound_only_layer_joins_an_audio_solo(self, audio_media: MediaItem) -> None:
        # 音だけの物は音の側のソロを見る 見ないと、置いた BGM が鳴らない
        project = _mixed(Track(TrackKind.AUDIO, "A1", solo=True))
        placed = _apply(project, insert_media(project, audio_media))
        layer = placed.timeline.tracks[-1]
        assert layer.kind is TrackKind.MIXED
        assert layer.solo
        assert layer in placed.timeline.active_sound_tracks()


class TestDrop:
    def test_it_lands_on_the_dropped_layer(self, video_media: MediaItem) -> None:
        # 落とした所ではなくレイヤー 1 へ入ると、落とした意味が無い
        back, front = Track(TrackKind.MIXED, "レイヤー 1"), Track(TrackKind.MIXED, "レイヤー 2")
        project = _mixed(back, front)
        (added,) = _added(place_media(project, [video_media], at_frame=45, track_id=back.id))
        assert added.track_id == back.id
        assert added.clip.timeline_start == 45
        assert added.clip.audio_stream == 1

    def test_a_muted_layer_is_not_used_even_when_dropped_on(self, video_media: MediaItem) -> None:
        # 落とした所を空きと鍵だけで受けると、ミュートしたレイヤーへ入り、映らず鳴らない
        muted = Track(TrackKind.MIXED, "レイヤー 1", muted=True)
        project = _mixed(muted)
        commands = place_media(project, [video_media], at_frame=0, track_id=muted.id)
        (added,) = _added(commands)
        assert added.track_id != muted.id
        placed = _apply(project, commands)
        layer = placed.timeline.find_track(added.track_id)
        assert layer is not None
        assert layer in placed.timeline.active_picture_tracks()

    def test_a_layer_outside_the_solo_is_not_used(self, video_media: MediaItem) -> None:
        # ほかのレイヤーがソロの間に、その外のレイヤーへ入れると映らない
        soloed = Track(TrackKind.MIXED, "レイヤー 1", solo=True)
        outside = Track(TrackKind.MIXED, "レイヤー 2")
        project = _mixed(soloed, outside)
        (added,) = _added(place_media(project, [video_media], at_frame=0, track_id=outside.id))
        assert added.track_id == soloed.id

    def test_a_busy_layer_moves_it_to_a_free_one(self, video_media: MediaItem) -> None:
        # 埋まった所へ無理に置くと、重なりで断られて何も置かれない
        clip = Clip(0, 300, media_id=video_media.id, audio_stream=1)
        busy = Track(TrackKind.MIXED, "レイヤー 1", (clip,))
        project = _mixed(busy, media=(video_media,))
        commands = place_media(project, [video_media], at_frame=30, track_id=busy.id)
        placed = _apply(project, commands)
        assert [len(t.clips) for t in placed.timeline.tracks] == [1, 1]

    def test_several_files_line_up(self, video_media: MediaItem, audio_media: MediaItem) -> None:
        # 何本かを落とすと、落とした所から隙間なく後ろへ並ぶ
        project = _mixed()
        placed = _apply(project, place_media(project, [video_media, audio_media], at_frame=0))
        (layer,) = placed.timeline.tracks
        assert [(c.timeline_start, c.show_picture) for c in layer.clips] == [
            (0, True),
            (300, False),
        ]


class TestTracks:
    def test_a_new_track_is_a_numbered_layer(self) -> None:
        # 「V1」の名前で足すと、混合の作品でレイヤーの番号と名前が食い違う
        project = _mixed(Track(TrackKind.MIXED, "レイヤー 1"))
        command = new_track(project, TrackKind.MIXED)
        assert command.track.kind is TrackKind.MIXED
        assert command.track.name == "レイヤー 2"

    def test_no_effect_layer_is_made(self) -> None:
        # 利用者の要望 分けない方式ではエフェクトのレイヤーという種類を持たず、フィルタも
        # 普通のレイヤーへ置く 作れてしまうと、FX の名前のレイヤーだけ振る舞いが変わる
        with pytest.raises(ValueError):
            new_track(_mixed(), TrackKind.MIXED, effect=True)
        with pytest.raises(ValueError):
            new_track(_mixed(), TrackKind.AUDIO, effect=True)

    def test_an_old_fx_layer_is_an_ordinary_layer(self) -> None:
        # 前の版で作った FX のレイヤーが残った作品で、名前だけで特別に扱うと、
        # 同じレイヤーなのに右クリックの中身や置き方が変わる 名前は変えない
        old = Track(TrackKind.MIXED, "FX1")
        assert not is_effect_track(old)
        project = _mixed(Track(TrackKind.MIXED, "レイヤー 1"), old)
        assert new_track(project, TrackKind.MIXED).track.name == "レイヤー 3"
        assert [t.name for t in project.timeline.tracks] == ["レイヤー 1", "FX1"]

    def test_a_filter_goes_to_an_ordinary_layer_above_the_pictures(
        self, video_media: MediaItem
    ) -> None:
        # フィルタの置き場の決まり（範囲に絵のある一番上より上の空いたレイヤー 無ければ
        # 一番手前に足す）は保つ 足すのは普通の名前のレイヤー
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        commands = insert_filter(project, at_frame=0, duration=30)
        (added,) = [c for c in commands if isinstance(c, AddTrack)]
        assert added.track.name == "レイヤー 2"
        placed = _apply(project, commands)
        assert placed.timeline.tracks[-1].clips[0].is_filter

    def test_layers_made_by_placing_can_be_reordered(self, video_media: MediaItem) -> None:
        # 置いて作ったレイヤーが並べ替えの仲間に入らないと、重なり順を変えられない
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        project = _apply(project, insert_generated(project, TEXT, at_frame=0))
        back, front = project.timeline.tracks
        moved = MoveTrack(front.id, 0).apply(project)
        assert [t.id for t in moved.timeline.tracks] == [front.id, back.id]


class TestPaste:
    def test_a_pasted_layer_clip_keeps_picture_and_sound(self, video_media: MediaItem) -> None:
        # 貼った物が映像トラックへ入ると、絵と音の 1 本から音が消える
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        original = project.timeline.tracks[0].clips[0]
        content = copy_clips(project, [original.id])
        pasted = _apply(project, paste_commands(project, content, 30))
        assert _kinds(pasted) == [TrackKind.MIXED, TrackKind.MIXED]
        clip = pasted.timeline.tracks[1].clips[0]
        assert clip.audio_stream == original.audio_stream
        assert pasted.timeline.tracks[1].name == "レイヤー 2"

    def test_a_pasted_layer_joins_the_solo(self, video_media: MediaItem) -> None:
        # ソロの間に貼って作ったレイヤーにソロが無いと、貼った物が出ない
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        layer = project.timeline.tracks[0]
        project = project.with_timeline(project.timeline.replace_track(replace(layer, solo=True)))
        content = copy_clips(project, [layer.clips[0].id])
        pasted = _apply(project, paste_commands(project, content, 30))
        assert pasted.timeline.tracks[1].solo


class TestSubtitles:
    def test_burned_subtitles_go_on_a_layer_in_front(self, video_media: MediaItem) -> None:
        # 映像トラックへ焼き込むと、混合の作品に 1 本だけ別の種類のトラックができる
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        segment = TranscriptSegment(start=Fraction(1), end=Fraction(3), text="こんにちは")
        project = SetTranscript(video_media.id, Transcript((segment,))).apply(project)
        burned = _apply(project, burn_subtitles(project, GeneratedSource(kind="text")))
        layer = burned.timeline.tracks[-1]
        assert layer.kind is TrackKind.MIXED
        assert layer.name == "字幕"
        assert [(c.timeline_start, c.duration) for c in layer.clips] == [(30, 60)]


class TestEditing:
    """置いた 1 本のクリップ（絵と音）が、編集の命令でそのまま扱えること"""

    @pytest.fixture
    def placed(self, video_media: MediaItem) -> Project:
        project = _apply(_mixed(), insert_media(_mixed(), video_media))
        # 分けて置かれていたら、以下はリンクした 2 本の編集を見ていることになる
        assert _kinds(project) == [TrackKind.MIXED]
        return project

    def _clips(self, project: Project) -> tuple[Clip, ...]:
        return project.timeline.tracks[0].clips

    def test_split_keeps_picture_and_sound(self, placed: Project) -> None:
        # 分けた後半が音を失うと、切った所から先が無音になる
        (clip,) = self._clips(placed)
        halves = self._clips(SplitClips((clip.id,), 90).apply(placed))
        assert [(c.timeline_start, c.duration) for c in halves] == [(0, 90), (90, 210)]
        assert all(c.audio_stream == 1 and c.show_picture for c in halves)
        assert halves[1].source_in == Fraction(3)

    def test_move_and_trim(self, placed: Project) -> None:
        # リンクの相手を探す作りのままだと、相手の無い 1 本を動かせない
        (clip,) = self._clips(placed)
        moved = MoveClip(clip.id, 60).apply(placed)
        trimmed = TrimClip(clip.id, head_delta=30).apply(moved)
        (result,) = self._clips(trimmed)
        assert (result.timeline_start, result.duration) == (90, 270)
        assert result.audio_stream == 1

    def test_ripple_cut(self, placed: Project) -> None:
        # 詰めたときに絵だけ動いて音が残る、が起きないこと（1 本なので同時に動く）
        (clip,) = self._clips(placed)
        cut = RippleCut(((0, 60),)).apply(placed)
        (result,) = self._clips(cut)
        assert (result.timeline_start, result.duration) == (0, 240)
        assert result.source_in == Fraction(2)
        assert result.id == clip.id

    def test_group(self, placed: Project) -> None:
        # 束ねの相手を探すときにリンクを要ると、1 本のクリップを束ねられない
        placed = _apply(placed, insert_generated(placed, TEXT, at_frame=0))
        ids = [t.clips[0].id for t in placed.timeline.tracks]
        grouped = GroupClips(tuple(ids)).apply(placed)
        groups = {t.clips[0].group_id for t in grouped.timeline.tracks}
        assert len(groups) == 1
        assert None not in groups


class TestProjectFormat:
    def test_matching_the_first_video_keeps_the_mode(self, video_media: MediaItem) -> None:
        # フレームレートを合わせたときに方式が分ける側へ戻ると、置いた物が 2 本に分かれる
        sixty = replace(
            video_media,
            video_streams=(replace(video_media.video_streams[0], frame_rate=FrameRate(60)),),
        )
        project = _mixed(Track(TrackKind.MIXED, "レイヤー 1"))
        wanted = format_to_match(project, [sixty])
        assert wanted is not None
        matched = _apply(project, match_commands(project, wanted))
        assert matched.settings.frame_rate == FrameRate(60)
        assert matched.settings.layer_mode == LayerMode.MIXED
        placed = _apply(matched, insert_media(matched, sixty))
        (layer,) = placed.timeline.tracks
        assert layer.kind is TrackKind.MIXED
        assert layer.clips[0].duration == 600
        assert SetFrameRate(FrameRate(24)).apply(_mixed()).settings.layer_mode == LayerMode.MIXED

    def test_switching_the_mode_changes_only_what_comes_next(self, video_media: MediaItem) -> None:
        # 方式を変えたときに置いてあるトラックまで変わると、設定を戻しても元に戻らない
        base = Project.create(ProjectSettings(frame_rate=RATE_30))
        project = _apply(base, insert_media(base, video_media))
        project = SetLayerMode(LayerMode.MIXED).apply(project)
        project = _apply(project, insert_media(project, video_media))
        assert _kinds(project) == [TrackKind.VIDEO, TrackKind.AUDIO, TrackKind.MIXED]
