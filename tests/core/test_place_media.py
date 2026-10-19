"""タイムラインへ落とした素材の置き方（:func:`place_media`）

落とした所のトラックが埋まっている・種類が違う・ロックしている、のどれでも
「何も置かれない」にならないことと、落とした所から外れすぎないことを見る
"""

from __future__ import annotations

from dataclasses import replace

from sashimono.core.commands import AddClip, AddTrack, Command, place_media
from sashimono.core.model import (
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    new_media_id,
)
from tests.conftest import RATE_30, make_clip


def _project(*tracks: Track) -> Project:
    base = Project.create(ProjectSettings(frame_rate=RATE_30))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _clips(commands: list[Command]) -> list[AddClip]:
    return [command for command in commands if isinstance(command, AddClip)]


class TestWhereItLands:
    def test_it_lands_on_the_dropped_track_at_the_dropped_frame(
        self, audio_media: MediaItem
    ) -> None:
        # 壊れると、落とした所ではなく末尾や 1 本目のトラックに出る
        a1, a2 = Track(TrackKind.AUDIO, "A1"), Track(TrackKind.AUDIO, "A2")
        commands = place_media(_project(a1, a2), [audio_media], at_frame=45, track_id=a2.id)
        (clip,) = _clips(commands)
        assert clip.track_id == a2.id
        assert clip.clip.timeline_start == 45

    def test_a_video_file_dropped_on_an_audio_track_splits_by_kind(
        self, video_media: MediaItem
    ) -> None:
        # 種類の合わないトラックへ入れようとすると、コマンドが断られて何も置かれない
        v1, a1 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1")
        project = _project(v1, a1)
        commands = place_media(project, [video_media], at_frame=10, track_id=a1.id)
        placed = {clip.track_id: clip.clip.timeline_start for clip in _clips(commands)}
        assert placed == {v1.id: 10, a1.id: 10}
        # 映像と音声は同じ組に入る 別々だと、動かしたときに口と声がずれる
        video, audio = _clips(commands)
        assert video.clip.link_group is not None
        assert video.clip.link_group == audio.clip.link_group
        _apply(project, commands)

    def test_an_empty_timeline_gets_tracks(self, video_media: MediaItem) -> None:
        # 起動した直後のプロジェクトはトラックを持たない そこへ落としても置ける
        project = _project()
        commands = place_media(project, [video_media], at_frame=0)
        kinds = [c.track.kind for c in commands if isinstance(c, AddTrack)]
        assert kinds == [TrackKind.VIDEO, TrackKind.AUDIO]
        placed = _apply(project, commands)
        assert all(len(track.clips) == 1 for track in placed.timeline.tracks)

    def test_an_occupied_track_moves_it_to_a_free_one(self, audio_media: MediaItem) -> None:
        # 落とした所に別のクリップがいると、重なりで断られて何も置かれない
        taken = Track(TrackKind.AUDIO, "A1", (make_clip(0, 300, audio_media),))
        free = Track(TrackKind.AUDIO, "A2")
        project = _project(taken, free)
        commands = place_media(project, [audio_media], at_frame=30, track_id=taken.id)
        (clip,) = _clips(commands)
        assert clip.track_id == free.id
        _apply(project, commands)

    def test_no_free_track_makes_a_new_one(self, audio_media: MediaItem) -> None:
        # 空きトラックが無いと重なりで断られ、落とした音声が何も置かれずに消える
        taken = Track(TrackKind.AUDIO, "A1", (make_clip(0, 300, audio_media),))
        project = _project(taken)
        commands = place_media(project, [audio_media], at_frame=30, track_id=taken.id)
        (new_track,) = [c.track for c in commands if isinstance(c, AddTrack)]
        assert new_track.name == "A2"
        _apply(project, commands)

    def test_a_locked_track_is_skipped(self, audio_media: MediaItem) -> None:
        # ロックしたトラックへ置くコマンドは断られる 先に外しておく
        locked = replace(Track(TrackKind.AUDIO, "A1"), locked=True)
        open_track = Track(TrackKind.AUDIO, "A2")
        project = _project(locked, open_track)
        commands = place_media(project, [audio_media], at_frame=0, track_id=locked.id)
        (clip,) = _clips(commands)
        assert clip.track_id == open_track.id

    def test_several_files_line_up_from_the_drop(self, audio_media: MediaItem) -> None:
        # 何本かを落としたら、落とした所から順に並ぶ 同じ所に重ねると、
        # 2 本目から別のトラックへ散らばる
        a1 = Track(TrackKind.AUDIO, "A1")
        second = replace(audio_media, id=new_media_id())
        commands = place_media(_project(a1), [audio_media, second], at_frame=60, track_id=a1.id)
        starts = [(clip.track_id, clip.clip.timeline_start) for clip in _clips(commands)]
        length = _clips(commands)[0].clip.duration
        assert starts == [(a1.id, 60), (a1.id, 60 + length)]
