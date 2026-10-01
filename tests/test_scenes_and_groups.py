"""シーン（入れ子のタイムライン）とグループ（束ね）

シーンは「中を編集したら、置いた先すべてに効く」ことと「自分自身を入れ子に
できない」ことが肝 グループは、束ねても解いても絵が変わらず、保存と貼り付けで
崩れないこと
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.clipboard import copy_clips, paste_commands
from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddScene,
    AddTrack,
    Command,
    Document,
    GroupClips,
    InScene,
    RemoveMedia,
    RemoveScene,
    RenameProject,
    RenameScene,
    UngroupClips,
    insert_media,
    insert_scene,
    new_scene,
)
from sashimono.core.io import load_project, save_project
from sashimono.core.io.serialize import ProjectFileError, project_from_dict, project_to_dict
from sashimono.core.model import Clip, MediaItem, Project, SceneId, Track, TrackKind, Transcript
from sashimono.core.projection import project_timeline
from sashimono.effects.sources import TEXT
from tests.conftest import make_clip


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _text(start: int = 0, duration: int = 30) -> Clip:
    return Clip(timeline_start=start, duration=duration, source=TEXT.create())


@pytest.fixture
def with_scene() -> tuple[Project, SceneId]:
    project = Project.create()
    scene = new_scene(project, "オープニング")
    project = AddScene(scene).apply(project)
    return project, scene.id


class TestScenes:
    def test_editing_inside_changes_only_the_scene(
        self, with_scene: tuple[Project, SceneId]
    ) -> None:
        # 中の編集がメインへ漏れると、シーンを開いて編集しただけで本編が変わる
        project, scene_id = with_scene
        track = Track(TrackKind.VIDEO, "V1")
        project = InScene(scene_id, AddTrack(track)).apply(project)
        project = InScene(scene_id, AddClip(track.id, _text())).apply(project)
        scene = project.require_scene(scene_id)
        assert len(scene.timeline.tracks[0].clips) == 1
        assert project.timeline.tracks == ()

    def test_project_wide_changes_inside_still_apply(
        self, with_scene: tuple[Project, SceneId]
    ) -> None:
        # シーンを開いたまま名前や素材を変えても、プロジェクト全体に効くこと
        project, scene_id = with_scene
        project = InScene(scene_id, RenameProject("本編")).apply(project)
        assert project.name == "本編"

    def test_placing_uses_the_scene_length(self, with_scene: tuple[Project, SceneId]) -> None:
        project, scene_id = with_scene
        track = Track(TrackKind.VIDEO, "V1")
        project = InScene(scene_id, AddTrack(track)).apply(project)
        project = InScene(scene_id, AddClip(track.id, _text(0, 90))).apply(project)
        project = _apply(project, insert_scene(project, scene_id, at_frame=10))
        (placed,) = project.timeline.tracks[0].clips
        assert (placed.timeline_start, placed.duration, placed.scene_id) == (10, 90, scene_id)

    def test_a_scene_cannot_contain_itself(self, with_scene: tuple[Project, SceneId]) -> None:
        # 自分を入れ子にすると、描くときに無限に潜り続けて固まる
        project, scene_id = with_scene
        commands = insert_scene(project, scene_id, at_frame=0)
        with pytest.raises(ValueError, match="入れ子"):
            for command in commands:
                project = InScene(scene_id, command).apply(project)

    def test_two_scenes_cannot_contain_each_other(
        self, with_scene: tuple[Project, SceneId]
    ) -> None:
        project, first = with_scene
        second = new_scene(project, "エンディング")
        project = AddScene(second).apply(project)
        for command in insert_scene(project, second.id, at_frame=0):
            project = InScene(first, command).apply(project)
        with pytest.raises(ValueError, match="入れ子"):
            for command in insert_scene(project, first, at_frame=0):
                project = InScene(second.id, command).apply(project)

    def test_a_placed_scene_cannot_be_removed(self, with_scene: tuple[Project, SceneId]) -> None:
        # 置いたまま消すと、置いた場所が黙って何も映らなくなる
        project, scene_id = with_scene
        project = _apply(project, insert_scene(project, scene_id, at_frame=0))
        with pytest.raises(ValueError, match="メイン"):
            RemoveScene(scene_id).apply(project)

    def test_media_used_elsewhere_cannot_be_removed_inside(
        self, with_scene: tuple[Project, SceneId], video_media: MediaItem
    ) -> None:
        # シーンの中からはメインのクリップが見えない 見ずに消すと、メインの
        # クリップが無い素材を指したまま残る
        project, scene_id = with_scene
        project = AddMedia(video_media).apply(project)
        project = _apply(project, insert_media(project, video_media, at_frame=0))
        with pytest.raises(ValueError, match="使われている"):
            InScene(scene_id, RemoveMedia(video_media.id)).apply(project)

    def test_media_used_in_another_scene_cannot_be_removed(
        self, with_scene: tuple[Project, SceneId], video_media: MediaItem
    ) -> None:
        # ほかのシーンの参照を見落とすと、そのシーンに素材の無いクリップが残り、
        # 開いたときに原因の分からない再生エラーになる
        project, scene_id = with_scene
        project = AddMedia(video_media).apply(project)
        project = InScene(scene_id, AddTrack(Track(TrackKind.VIDEO, "V1"))).apply(project)
        inner = replace(project, timeline=project.require_scene(scene_id).timeline)
        for command in insert_media(inner, video_media, at_frame=0):
            project = InScene(scene_id, command).apply(project)
        with pytest.raises(ValueError, match="使われている"):
            RemoveMedia(video_media.id).apply(project)

    def test_unused_media_can_still_be_removed_inside(
        self, with_scene: tuple[Project, SceneId], video_media: MediaItem
    ) -> None:
        # 確かめを広げすぎて使っていない素材まで消せなくなると、メディアプールを片付けられない
        project, scene_id = with_scene
        project = AddMedia(video_media).apply(project)
        removed = InScene(scene_id, RemoveMedia(video_media.id)).apply(project)
        assert removed.media == ()

    def test_rename_trims_surrounding_spaces(self, with_scene: tuple[Project, SceneId]) -> None:
        # 空白が残ると、シーンバーの表示とファイルの中身に見えない空白が入る
        project, scene_id = with_scene
        project = RenameScene(scene_id, "  導入  ").apply(project)
        assert project.require_scene(scene_id).name == "導入"

    def test_one_undo_step_inside(self, with_scene: tuple[Project, SceneId]) -> None:
        # 中の編集も 1 段で戻ること 戻らないと、シーンの中だけ取り消せない
        project, scene_id = with_scene
        document = Document(project)
        with document.checkpoint("シーンの編集"):
            document.execute(InScene(scene_id, AddTrack(Track(TrackKind.VIDEO, "V1"))))
        document.undo()
        assert document.project.require_scene(scene_id).timeline.tracks == ()


class TestGroups:
    def _two(self) -> tuple[Project, Clip, Clip]:
        a, b = _text(0), _text(40)
        base = Project.create()
        track = Track(TrackKind.VIDEO, "V1", (a, b))
        return base.with_timeline(replace(base.timeline, tracks=(track,))), a, b

    def test_group_and_ungroup(self) -> None:
        project, a, b = self._two()
        grouped = GroupClips((a.id, b.id)).apply(project)
        clips = grouped.timeline.tracks[0].clips
        assert clips[0].group_id is not None and clips[0].group_id == clips[1].group_id
        freed = UngroupClips((a.id,)).apply(grouped)
        assert all(clip.group_id is None for clip in freed.timeline.tracks[0].clips)

    def test_the_group_id_is_fixed_when_the_command_is_made(self) -> None:
        # 当て直すたびに ID が変わると、ID から決まるグループの色まで変わる
        project, a, b = self._two()
        command = GroupClips((a.id, b.id))
        first = command.apply(project).timeline.tracks[0].clips[0].group_id
        again = command.apply(project).timeline.tracks[0].clips[0].group_id
        assert first == again == command.group_id

    def test_a_single_clip_is_not_a_group(self) -> None:
        # 1 本だけの束ねは選択を広げないのに、解除の手間だけが残る
        project, a, _ = self._two()
        with pytest.raises(ValueError, match="2 本"):
            GroupClips((a.id,)).apply(project)

    def test_pasting_gets_a_new_group(self) -> None:
        # 元と同じグループのままだと、貼ったものを選ぶと元のクリップまで動く
        project, a, b = self._two()
        project = GroupClips((a.id, b.id)).apply(project)
        original = project.timeline.tracks[0].clips[0].group_id
        pasted = [
            command.clip
            for command in paste_commands(project, copy_clips(project, [a.id, b.id]), 200)
            if isinstance(command, AddClip)
        ]
        assert pasted[0].group_id is not None
        assert pasted[0].group_id == pasted[1].group_id != original


class TestSaving:
    def test_scenes_and_groups_round_trip(
        self, with_scene: tuple[Project, SceneId], tmp_path: Path
    ) -> None:
        # 保存し直してシーンやグループが消えると、開き直すたびに作り直すことになる
        project, scene_id = with_scene
        track = Track(TrackKind.VIDEO, "V1")
        project = InScene(scene_id, AddTrack(track)).apply(project)
        project = InScene(scene_id, AddClip(track.id, _text())).apply(project)
        project = _apply(project, insert_scene(project, scene_id, at_frame=0))
        placed = project.timeline.tracks[0].clips[0]
        project = _apply(project, [AddClip(project.timeline.tracks[0].id, _text(200))])
        other = project.timeline.tracks[0].clips[1]
        project = GroupClips((placed.id, other.id)).apply(project)

        path = tmp_path / "本編.sme"
        save_project(project, path)
        loaded = load_project(path)
        assert loaded.scenes[0].name == "オープニング"
        assert len(loaded.scenes[0].timeline.tracks[0].clips) == 1
        clips = loaded.timeline.tracks[0].clips
        assert clips[0].scene_id == scene_id
        assert clips[0].group_id is not None and clips[0].group_id == clips[1].group_id

    def test_a_looping_file_is_refused(self, with_scene: tuple[Project, SceneId]) -> None:
        # 手で書き換えたファイルなどで入れ子が自分へ戻っていたら、開く時点で止める
        project, scene_id = with_scene
        data = project_to_dict(project)
        data["scenes"][0]["timeline"]["tracks"] = [
            {
                "id": "t1",
                "kind": "video",
                "clips": [{"id": "c1", "timeline_start": 0, "duration": 10, "scene_id": scene_id}],
            }
        ]
        with pytest.raises(ProjectFileError, match="入れ子"):
            project_from_dict(data)

    def test_a_clip_pointing_to_a_missing_scene_is_refused(
        self, with_scene: tuple[Project, SceneId]
    ) -> None:
        # 開けてしまうと、そのクリップは絵も音も出さずに黙って残り、原因が追えない
        project, _ = with_scene
        data = project_to_dict(project)
        data["timeline"]["tracks"] = [
            {
                "id": "t1",
                "kind": "video",
                "clips": [{"id": "c1", "timeline_start": 0, "duration": 10, "scene_id": "無い"}],
            }
        ]
        with pytest.raises(ProjectFileError, match="無いシーン"):
            project_from_dict(data)

    def test_a_broken_clip_inside_a_scene_is_a_file_error(
        self, with_scene: tuple[Project, SceneId]
    ) -> None:
        # 開く側は ProjectFileError しか受けない 素の ValueError が漏れると起動ごと落ちる
        project, _ = with_scene
        data = project_to_dict(project)
        data["scenes"][0]["timeline"]["tracks"] = [
            {
                "id": "t1",
                "kind": "video",
                "clips": [{"id": "c1", "timeline_start": 0, "duration": 0}],
            }
        ]
        with pytest.raises(ProjectFileError):
            project_from_dict(data)


class TestSceneSubtitles:
    def test_subtitles_inside_a_scene_appear_where_it_is_placed(
        self, with_scene: tuple[Project, SceneId], video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 見なければ、シーンにまとめた途端に字幕パネル・焼き込み・字幕ファイルから消える
        project, scene_id = with_scene
        media = video_media.with_transcript(transcript)
        project = AddMedia(media).apply(project)
        track = Track(TrackKind.VIDEO, "V1")
        project = InScene(scene_id, AddTrack(track)).apply(project)
        project = InScene(scene_id, AddClip(track.id, make_clip(0, 300, media))).apply(project)
        # シーンの頭 2 秒（60 フレーム）を飛ばして、メインの 100 フレーム目から 90 フレームだけ置く
        project = _apply(project, insert_scene(project, scene_id, at_frame=100, duration=90))
        placed = project.timeline.tracks[0].clips[0]
        project = project.with_timeline(
            project.timeline.replace_track(
                replace(
                    project.timeline.tracks[0],
                    clips=(replace(placed, source_in=Fraction(2)),),
                )
            )
        )

        subtitles = list(project_timeline(project))
        # 「今日は」は 1〜3 秒 頭を 2 秒飛ばしたので 100〜130 フレームに切り詰めて出る
        assert [(s.segment.text, s.start_frame, s.end_frame) for s in subtitles] == [
            ("今日は", 100, 130),
            ("編集ソフトを", 160, 190),
        ]
        assert subtitles[0].clipped_head and subtitles[1].clipped_tail
        assert all(s.clip_id == placed.id for s in subtitles)

    def test_a_fractional_start_keeps_the_frame(
        self, with_scene: tuple[Project, SceneId], video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 端数の source_in を先にフレームへ落とすと、速度を掛けたときに 1 フレームずれる
        # 描画は秒のまま足してから 1 回だけフレームへ直している（そちらに合わせる）
        project, scene_id = with_scene
        media = video_media.with_transcript(transcript)
        project = AddMedia(media).apply(project)
        track = Track(TrackKind.VIDEO, "V1")
        project = InScene(scene_id, AddTrack(track)).apply(project)
        project = InScene(scene_id, AddClip(track.id, make_clip(0, 300, media))).apply(project)
        project = _apply(project, insert_scene(project, scene_id, at_frame=0, duration=300))
        placed = project.timeline.tracks[0].clips[0]
        project = project.with_timeline(
            project.timeline.replace_track(
                replace(
                    project.timeline.tracks[0],
                    clips=(replace(placed, source_in=Fraction(1, 60), speed=Fraction(1, 2)),),
                )
            )
        )
        subtitles = list(project_timeline(project))
        # 1 秒の字幕は、半分の速さでは 2 秒目へ 端数の 1/60 秒（0.5 フレーム）は切り捨てる
        assert subtitles[0].start_frame == 59
