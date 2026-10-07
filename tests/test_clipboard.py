"""クリップのコピー・切り取り・貼り付け

貼り付けは「新しいクリップを置く」ことなので、元のクリップとの縁が切れていることが
いちばん大事 ID やリンクが元と同じままだと、貼ったものを動かしたときに元のものまで
動いたり、片方を消すともう片方まで消えたりする
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.clipboard import (
    ClipboardContent,
    copy_clips,
    cut_commands,
    insert_paste_commands,
    paste_commands,
)
from sashimono.core.commands import AddClip, AddTrack, Command, Document, insert_media
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    Marker,
    MediaItem,
    Project,
    ProjectSettings,
    SegmentId,
    SubtitleOrigin,
    Track,
    TrackKind,
    new_group_id,
)
from sashimono.core.timebase import FrameRate


def apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


@pytest.fixture
def linked(video_media: MediaItem) -> Project:
    """映像と音声がリンクした素材を 1 本、先頭に置いたもの（300 フレーム）"""
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
    return apply(base, insert_media(base, video_media, at_frame=0))


def added(commands: list[Command]) -> list[AddClip]:
    return [command for command in commands if isinstance(command, AddClip)]


class TestCopy:
    def test_the_linked_partner_comes_along(self, linked: Project) -> None:
        # 映像だけ運ぶと、貼った映像に音が付いてこない
        video = linked.timeline.tracks[0].clips[0]
        content = copy_clips(linked, [video.id])
        assert len(content.clips) == 2

    def test_an_unknown_clip_is_ignored(self, linked: Project) -> None:
        # 消えたクリップを選んだまま Ctrl+C を押しても、例外で止まらないこと
        assert copy_clips(linked, [Clip(timeline_start=0, duration=1).id]).clips == ()


class TestPaste:
    def test_it_lands_at_the_playhead(self, linked: Project) -> None:
        # 壊れると、貼ったクリップが再生ヘッドとは違う時刻に置かれる
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        starts = {
            command.clip.timeline_start for command in added(paste_commands(linked, content, 300))
        }
        assert starts == {300}

    def test_the_copy_is_cut_loose_from_the_original(self, linked: Project) -> None:
        # ID もリンクも元と同じだと、貼ったものを動かすと元まで動く
        original = linked.timeline.tracks[0].clips[0]
        content = copy_clips(linked, [original.id])
        pasted = [command.clip for command in added(paste_commands(linked, content, 300))]
        assert all(clip.id != original.id for clip in pasted)
        groups = {clip.link_group for clip in pasted}
        assert len(groups) == 1
        assert original.link_group not in groups

    def test_two_pastes_are_two_separate_pairs(self, linked: Project) -> None:
        # 壊れると、2 回貼ったもの同士が同じ組になり、片方を動かすともう片方も動く
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        first = apply(linked, paste_commands(linked, content, 300))
        second = paste_commands(first, content, 600)
        groups_first = {c.link_group for t in first.timeline.tracks for c in t.clips}
        assert {command.clip.link_group for command in added(second)}.isdisjoint(groups_first)

    def test_a_busy_track_sends_it_to_a_new_one(self, linked: Project) -> None:
        # 元の場所に貼ると重なる クリップの重なりは許されないので、別のトラックへ
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        commands = paste_commands(linked, content, 0)
        assert sum(isinstance(command, AddTrack) for command in commands) == 2
        pasted = apply(linked, commands)
        assert len(pasted.timeline.tracks) == 4

    def test_a_locked_track_is_skipped(self, linked: Project) -> None:
        # ロックしたトラックへ貼ると、AddClip が「ロックされている」で失敗する
        video_track = linked.timeline.tracks[0]
        locked = linked.with_timeline(
            linked.timeline.replace_track(replace(video_track, locked=True))
        )
        content = copy_clips(locked, [video_track.clips[0].id])
        commands = paste_commands(locked, content, 300)
        video_target = next(c for c in added(commands) if c.clip.stream_index == 0)
        assert video_target.track_id != video_track.id
        apply(locked, commands)

    def test_a_removed_source_is_refused(self, linked: Project) -> None:
        # 素材の無いクリップを置くと、再生したときに理由の分からない穴になる
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        with pytest.raises(ValueError, match="素材"):
            paste_commands(linked.with_media(()), content, 300)

    def test_a_generated_clip_needs_no_source(self) -> None:
        # 壊れると、テロップ（素材を持たないクリップ）だけコピーできない
        base = Project.create()
        text = Clip(timeline_start=0, duration=30, source=GeneratedSource(kind="text"))
        track = Track(TrackKind.VIDEO, "V1")
        project = apply(base, [AddTrack(track), AddClip(track.id, text)])
        content = copy_clips(project, [text.id])
        pasted = apply(project, paste_commands(project, content, 30))
        assert len(pasted.timeline.tracks[0].clips) == 2

    def test_a_paste_is_one_undo_step(self, linked: Project) -> None:
        # 壊れると、映像と音声を貼っただけで取り消しを何度も押すことになる
        document = Document(linked)
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        with document.checkpoint("貼り付け"):
            for command in paste_commands(linked, content, 300):
                document.execute(command)
        document.undo()
        assert document.project is linked


def _texts(*tracks: tuple[str, tuple[tuple[int, int], ...]]) -> Project:
    """テキストのクリップだけを並べたプロジェクト 中身は (トラック名, ((頭, 長さ), ...))"""
    base = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
    built = tuple(
        Track(
            TrackKind.VIDEO,
            name,
            tuple(
                Clip(timeline_start=start, duration=length, source=GeneratedSource(kind="text"))
                for start, length in spans
            ),
        )
        for name, spans in tracks
    )
    return base.with_timeline(replace(base.timeline, tracks=built))


def _spans(project: Project, index: int) -> list[tuple[int, int]]:
    track = project.timeline.tracks[index]
    return sorted((clip.timeline_start, clip.timeline_end) for clip in track.clips)


class TestInsertPaste:
    """挿入貼り付け（Ctrl+Shift+V） 再生ヘッドの後ろを貼る長さぶん押し出してから貼る"""

    def test_clips_behind_the_playhead_move_by_the_pasted_length(self) -> None:
        # 壊れると、後ろのクリップが動かず、貼った物が別のトラックへ逃げる（普通の貼り付けと同じ）
        project = _texts(("V1", ((0, 30), (60, 30))))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        commands = insert_paste_commands(project, content, 60)
        assert not any(isinstance(command, AddTrack) for command in commands)
        pasted = apply(project, commands)
        assert _spans(pasted, 0) == [(0, 30), (60, 90), (90, 120)]

    def test_clips_in_front_of_the_playhead_stay(self) -> None:
        # 壊れると、再生ヘッドより前の編集まで崩れる
        project = _texts(("V1", ((0, 30), (40, 10))))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 50))
        assert _spans(pasted, 0)[:2] == [(0, 30), (40, 50)]

    def test_a_clip_under_the_playhead_is_split_and_its_tail_pushed(self) -> None:
        # Premiere と同じく割ってから押す 丸ごと押すと、再生ヘッドより前に見えていた絵が消える
        project = _texts(("V1", ((0, 20), (40, 60))))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 50))
        assert _spans(pasted, 0) == [(0, 20), (40, 50), (50, 70), (70, 120)]

    def test_the_split_tail_keeps_its_keyframes_in_place(self) -> None:
        # 割った後ろのキーを頭から数え直さないと、押しただけで動きの時刻がずれる
        project = _texts(("V1", ((0, 20), (40, 60))))
        long = project.timeline.tracks[0].clips[1]
        keyed = replace(
            long,
            opacity=AnimatedValue(static=1.0, keyframes=(Keyframe(0, 0.0), Keyframe(40, 1.0))),
        )
        project = project.with_timeline(
            project.timeline.replace_track(
                project.timeline.tracks[0].with_clips((project.timeline.tracks[0].clips[0], keyed))
            )
        )
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 50))
        tail = next(c for c in pasted.timeline.tracks[0].clips if c.timeline_start == 70)
        # 元のキー 40（タイムラインの 80）は、割った所（50）から 30 後 押した後は 70 + 30
        assert tail.opacity.keyframes[-1].frame == 30

    def test_a_linked_pair_moves_together(self, linked: Project) -> None:
        # 映像と音声で押す量が違うと、貼った後ろの口と声がずれる
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        pasted = apply(linked, insert_paste_commands(linked, content, 0, all_tracks=False))
        assert _spans(pasted, 0) == [(0, 300), (300, 600)]
        assert _spans(pasted, 1) == [(0, 300), (300, 600)]

    def test_a_split_pair_keeps_its_tails_linked_apart_from_the_heads(
        self, linked: Project
    ) -> None:
        # 前後が同じ組のままだと、後ろを動かすと間を越えて前まで動く
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        pasted = apply(linked, insert_paste_commands(linked, content, 100))
        clips = [c for t in pasted.timeline.tracks for c in t.clips]
        heads = {c.link_group for c in clips if c.timeline_start == 0}
        tails = {c.link_group for c in clips if c.timeline_start == 400}
        assert len(heads) == 1
        assert len(tails) == 1
        assert heads != tails
        assert {(c.timeline_start, c.timeline_end) for c in clips} == {
            (0, 100),
            (100, 400),
            (400, 600),
        }

    def test_all_tracks_mode_pushes_every_track_and_the_markers(self) -> None:
        # Premiere の既定（全トラックの同期ロック） ほかのトラックの字幕や BGM が置いていかれない
        project = _texts(("V1", ((0, 30),)), ("V2", ((50, 10),)))
        project = project.with_timeline(replace(project.timeline, markers=(Marker(10), Marker(70))))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 40))
        assert _spans(pasted, 1) == [(80, 90)]
        assert [marker.frame for marker in pasted.timeline.markers] == [10, 100]

    def test_target_mode_leaves_unrelated_tracks_alone(self) -> None:
        # 貼り先だけを押す設定で全部動くと、設定が効いていない
        project = _texts(("V1", ((0, 30), (40, 10))), ("V2", ((50, 10),)))
        project = project.with_timeline(replace(project.timeline, markers=(Marker(70),)))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 40, all_tracks=False))
        assert _spans(pasted, 0) == [(0, 30), (40, 70), (70, 80)]
        assert _spans(pasted, 1) == [(50, 60)]
        assert [marker.frame for marker in pasted.timeline.markers] == [70]

    def test_target_mode_still_pushes_the_group(self) -> None:
        # グループの仲間を置いていくと、束ねたテロップと絵がずれる
        project = _texts(("V1", ((0, 30), (40, 10))), ("V2", ((45, 10),)))
        bundle = new_group_id()
        timeline = project.timeline
        v1, v2 = timeline.tracks
        timeline = timeline.replace_track(
            v1.with_clips((v1.clips[0], replace(v1.clips[1], group_id=bundle)))
        )
        timeline = timeline.replace_track(v2.with_clips((replace(v2.clips[0], group_id=bundle),)))
        project = project.with_timeline(timeline)
        content = copy_clips(project, [v1.clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 40, all_tracks=False))
        assert _spans(pasted, 1) == [(75, 85)]

    def test_target_mode_still_pushes_burned_subtitles(self, linked: Project) -> None:
        # 焼き込んだ字幕を置いていくと、話している所と字幕が貼った長さぶんずれる
        media_id = linked.timeline.tracks[0].clips[0].media_id
        assert media_id is not None
        line = Clip(
            timeline_start=120,
            duration=30,
            source=GeneratedSource(kind="text"),
            subtitle_origin=SubtitleOrigin(media_id=media_id, stream=1, segment_id=SegmentId("s")),
        )
        subtitles = Track(TrackKind.VIDEO, "字幕", (line,))
        project = linked.with_timeline(
            replace(linked.timeline, tracks=(*linked.timeline.tracks, subtitles))
        )
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 100, all_tracks=False))
        assert _spans(pasted, 2) == [(420, 450)]

    def test_a_locked_track_with_clips_behind_refuses(self, linked: Project) -> None:
        # ロックしたトラックだけ残して押すと、そこから後ろの同期がすべて崩れる
        audio_track = linked.timeline.tracks[1]
        locked = linked.with_timeline(
            linked.timeline.replace_track(replace(audio_track, locked=True))
        )
        content = copy_clips(locked, [locked.timeline.tracks[0].clips[0].id])
        with pytest.raises(ValueError, match="ロック"):
            insert_paste_commands(locked, content, 0)

    def test_a_locked_track_with_nothing_behind_is_no_obstacle(self) -> None:
        # 押す物の無いロックしたトラックで止めると、BGM を固めておくだけで挿入が使えない
        project = _texts(("V1", ((0, 30), (60, 30))), ("V2", ((0, 20),)))
        v2 = project.timeline.tracks[1]
        project = project.with_timeline(project.timeline.replace_track(replace(v2, locked=True)))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 60))
        assert _spans(pasted, 0) == [(0, 30), (60, 90), (90, 120)]

    def test_the_pasted_clips_land_where_the_gap_was_opened(self, linked: Project) -> None:
        # 押す前の姿で行き先を決めると、空けた所ではなく新しいトラックへ置かれる
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        commands = insert_paste_commands(linked, content, 0)
        targets = {command.track_id for command in added(commands)}
        assert targets == {track.id for track in linked.timeline.tracks}

    def test_an_insert_paste_is_one_undo_step(self, linked: Project) -> None:
        # 押し出しと貼り付けが別の段だと、取り消すと間だけ空いたまま残る
        document = Document(linked)
        content = copy_clips(linked, [linked.timeline.tracks[0].clips[0].id])
        with document.checkpoint("貼り付け（挿入）"):
            for command in insert_paste_commands(linked, content, 100):
                document.execute(command)
        document.undo()
        assert document.project is linked

    @pytest.mark.parametrize(
        ("at", "expected"),
        [(10, (70, 90)), (40, (70, 90)), (50, (40, 90)), (60, (40, 60)), (80, (40, 60))],
    )
    def test_the_export_range_follows_the_push(self, at: int, expected: tuple[int, int]) -> None:
        # 範囲だけ古いフレームに残すと、書き出しの頭に意図しない部分が入り末尾が欠ける
        # 範囲の前（頭ちょうども）なら両端を押し、途中なら終わりだけ延ばす 後ろなら動かさない
        project = _texts(("V1", ((0, 30), (100, 10))))
        project = project.with_timeline(replace(project.timeline, work_area=(40, 60)))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, at))
        assert pasted.timeline.work_area == expected

    def test_target_mode_leaves_the_export_range_alone(self) -> None:
        # ほかのトラックの中身は動かないので、範囲を押すと書き出す中身がずれる
        project = _texts(("V1", ((0, 30),)), ("V2", ((50, 10),)))
        project = project.with_timeline(replace(project.timeline, work_area=(40, 60)))
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        pasted = apply(project, insert_paste_commands(project, content, 10, all_tracks=False))
        assert pasted.timeline.work_area == (40, 60)

    def test_the_export_range_comes_back_with_one_undo(self) -> None:
        # 範囲だけ押したまま残ると、取り消した後の書き出しがずれる
        project = _texts(("V1", ((0, 30),)))
        project = project.with_timeline(replace(project.timeline, work_area=(40, 60)))
        document = Document(project)
        content = copy_clips(project, [project.timeline.tracks[0].clips[0].id])
        with document.checkpoint("貼り付け（挿入）"):
            for command in insert_paste_commands(project, content, 10):
                document.execute(command)
        assert document.project.timeline.work_area == (70, 90)
        document.undo()
        assert document.project.timeline.work_area == (40, 60)

    def test_target_mode_pushes_where_a_copy_from_another_scene_lands(self) -> None:
        # 別のシーンでコピーした物はコピー元のトラックが今のタイムラインに無い その ID のまま
        # 押すと何も押さず、普通の貼り付けと同じく新しいトラックへ逃げた
        elsewhere = _texts(("V1", ((0, 30),)))
        content = copy_clips(elsewhere, [elsewhere.timeline.tracks[0].clips[0].id])
        project = _texts(("V1", ((0, 30), (60, 30))))
        commands = insert_paste_commands(project, content, 60, all_tracks=False)
        assert not any(isinstance(command, AddTrack) for command in commands)
        pasted = apply(project, commands)
        assert _spans(pasted, 0) == [(0, 30), (60, 90), (90, 120)]

    def test_nothing_copied_does_nothing(self, linked: Project) -> None:
        assert insert_paste_commands(linked, ClipboardContent(()), 0) == []


class TestLockedRemoval:
    def test_a_locked_track_is_not_cut(self, linked: Project) -> None:
        # 移動とトリムはロックを見ていたのに、削除と切り取りだけ素通しだった
        # 右クリックに削除を並べたので、ロックしたつもりのクリップが消えやすくなっていた
        audio_track = linked.timeline.tracks[1]
        locked = linked.with_timeline(
            linked.timeline.replace_track(replace(audio_track, locked=True))
        )
        content = copy_clips(locked, [locked.timeline.tracks[0].clips[0].id])
        with pytest.raises(ValueError, match="ロック"):
            apply(locked, cut_commands(locked, content))


class TestCut:
    def test_one_removal_takes_the_pair(self, linked: Project) -> None:
        # 組の両方に RemoveClip を出すと、2 つ目が「見つからない」で失敗する
        content = copy_clips(linked, [linked.timeline.tracks[1].clips[0].id])
        commands = cut_commands(linked, content)
        assert len(commands) == 1
        emptied = apply(linked, commands)
        assert all(not track.clips for track in emptied.timeline.tracks)
