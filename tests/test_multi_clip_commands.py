"""何本かのクリップをまとめて扱うコマンドと、履歴のまとめ方

まとめて扱う操作は「全部できるか、何もしないか」でなければならない 途中まで
動いた状態が 1 段として残ると、どれが動いたのかを人も AI も追えなくなる
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import (
    Document,
    MoveClips,
    RemoveClips,
    RenameProject,
    SetTrackHeights,
    TrimClips,
    insert_media,
)
from sashimono.core.model import Clip, MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT


def _text(start: int, duration: int = 30) -> Clip:
    return Clip(timeline_start=start, duration=duration, source=TEXT.create())


def _project(*clips: Clip, locked: bool = False) -> Project:
    base = Project.create()
    track = Track(TrackKind.VIDEO, "V1", clips, locked=locked)
    return base.with_timeline(replace(base.timeline, tracks=(track,)))


def _starts(project: Project) -> list[int]:
    return [clip.timeline_start for clip in project.timeline.tracks[0].clips]


class TestMoveClips:
    def test_neighbours_can_shift_into_each_others_place(self) -> None:
        # 1 本ずつ動かすと、前のクリップが後ろのクリップの元の場所に入った瞬間に
        # 重なって失敗する 全体としては重ならないので、まとめてなら動かせる
        first, second = _text(0), _text(30)
        project = _project(first, second)
        moved = MoveClips((first.id, second.id), 30).apply(project)
        assert _starts(moved) == [30, 60]

    def test_the_linked_partner_moves_too(self, video_media: MediaItem) -> None:
        # 映像だけ動くと、音がずれる
        base = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
        for command in insert_media(base, video_media, at_frame=0):
            base = command.apply(base)
        video = base.timeline.tracks[0].clips[0]
        moved = MoveClips((video.id,), 15).apply(base)
        assert [t.clips[0].timeline_start for t in moved.timeline.tracks] == [15, 15]

    def test_a_locked_track_stops_everything(self) -> None:
        # ロックしたトラックのクリップが選択に混ざっていたら、ほかも動かさない
        clip = _text(0)
        project = _project(clip, locked=True)
        with pytest.raises(ValueError, match="ロック"):
            MoveClips((clip.id,), 10).apply(project)

    def test_nothing_goes_before_the_start(self) -> None:
        # 先頭より前へ出たクリップは、どこにも表示されずに消えたように見える
        clip = _text(5)
        with pytest.raises(ValueError, match="先頭"):
            MoveClips((clip.id,), -10).apply(_project(clip))

    def test_a_locked_partner_stops_everything(self, video_media: MediaItem) -> None:
        # 相手を残して動くと、選んだクリップの映像と音声が黙ってずれる
        base = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
        for command in insert_media(base, video_media, at_frame=0):
            base = command.apply(base)
        audio = base.timeline.tracks[1]
        base = base.with_timeline(base.timeline.replace_track(replace(audio, locked=True)))
        video = base.timeline.tracks[0].clips[0]
        with pytest.raises(ValueError, match="相手"):
            MoveClips((video.id,), 15).apply(base)

    def test_a_collision_with_an_unselected_clip_fails(self) -> None:
        # 選んでいないクリップに重なるなら、黙って上書きせずに断る
        moving, staying = _text(0), _text(40)
        with pytest.raises(ValueError):
            MoveClips((moving.id,), 20).apply(_project(moving, staying))


class TestMoveClipsAcrossTracks:
    def _two_tracks(self) -> Project:
        base = Project.create()
        upper = Track(TrackKind.VIDEO, "V1", (_text(0), _text(60)))
        lower = Track(TrackKind.VIDEO, "V2", ())
        return base.with_timeline(replace(base.timeline, tracks=(upper, lower)))

    def test_the_whole_selection_changes_track(self) -> None:
        # まとめて段を移したクリップの一部が元のトラックに残り、選んだ並びが上下に割れる
        project = self._two_tracks()
        clips = project.timeline.tracks[0].clips
        moved = MoveClips(tuple(c.id for c in clips), 0, track_delta=1).apply(project)
        assert [c.timeline_start for c in moved.timeline.tracks[1].clips] == [0, 60]
        assert moved.timeline.tracks[0].clips == ()

    def test_moving_past_the_last_track_fails(self) -> None:
        # 並びの外を ValueError で断らないと、まとめて段を跨ぐドラッグで時間だけ動かす
        # 逃げ道に入らず、別の例外で操作が止まる
        project = self._two_tracks()
        clips = project.timeline.tracks[0].clips
        with pytest.raises(ValueError, match="並びの外"):
            MoveClips(tuple(c.id for c in clips), 0, track_delta=5).apply(project)

    def test_a_clip_already_there_stops_the_move(self) -> None:
        # 移した先のクリップと重なっても断らないと、段を移したクリップが下のクリップに重なる
        base = Project.create()
        upper = Track(TrackKind.VIDEO, "V1", (_text(0),))
        lower = Track(TrackKind.VIDEO, "V2", (_text(10),))
        project = base.with_timeline(replace(base.timeline, tracks=(upper, lower)))
        moving = upper.clips[0].id
        # 断った理由が重なりであることまで見る 段の移し方そのものが壊れて別の
        # ValueError を出しても、型だけを見ていると通ってしまう
        with pytest.raises(ValueError, match="重なっている"):
            MoveClips((moving,), 0, track_delta=1).apply(project)
        # 重ならない所へなら同じ段の移し方で通る 断ったのが重なりのせいだと確かめる
        moved = MoveClips((moving,), 40, track_delta=1).apply(project)
        assert [c.id for c in moved.timeline.tracks[1].clips][-1] == moving


class TestTrimClips:
    def test_every_selected_clip_is_trimmed(self) -> None:
        # 掴んだ 1 本だけが縮むと、まとめて端を詰めたつもりのほかのクリップが元の長さのまま残る
        project = _project(_text(0), _text(60))
        clips = project.timeline.tracks[0].clips
        trimmed = TrimClips(tuple(c.id for c in clips), tail_delta=-10).apply(project)
        assert [c.duration for c in trimmed.timeline.tracks[0].clips] == [20, 20]

    def test_a_failure_leaves_everything_alone(self) -> None:
        # 2 本目が隣へ食い込む長さ 途中まで伸びた状態が残ってはいけない
        project = _project(_text(0), _text(30), _text(60))
        clips = project.timeline.tracks[0].clips
        with pytest.raises(ValueError):
            TrimClips(tuple(c.id for c in clips), tail_delta=20).apply(project)
        assert [c.duration for c in project.timeline.tracks[0].clips] == [30, 30, 30]

    def test_a_locked_track_is_refused(self) -> None:
        # ロックしたトラックのクリップまで端が動くと、触らないつもりでロックしたクリップの
        # 長さがまとめての調整で変わる
        project = _project(_text(0), locked=True)
        clip = project.timeline.tracks[0].clips[0]
        with pytest.raises(ValueError, match="ロック"):
            TrimClips((clip.id,), tail_delta=-5).apply(project)


class TestRemoveClips:
    def test_ripple_closes_every_gap(self) -> None:
        # 前から消して詰めると、後ろのクリップの位置がずれて詰める量が狂う
        a, b, c = _text(0), _text(30), _text(60)
        removed = RemoveClips((a.id, b.id), ripple=True).apply(_project(a, b, c))
        assert _starts(removed) == [0]

    def test_a_missing_clip_removes_nothing(self) -> None:
        # 一部だけ消すと、どれが消えたのか分からない
        a = _text(0)
        project = _project(a)
        other = _text(100)
        with pytest.raises(KeyError):
            RemoveClips((a.id, other.id)).apply(project)


class TestCheckpointRollback:
    def test_a_failure_leaves_the_project_as_it_was(self) -> None:
        # 3 本のうち 2 本だけ置かれた状態が 1 段として残ると、戻す段も分からない
        document = Document(_project(_text(0)))
        before = document.project
        with pytest.raises(ValueError), document.checkpoint("まとめて"):
            document.execute(RenameProject("途中"))
            document.execute(MoveClips((before.timeline.tracks[0].clips[0].id,), -5))
        assert document.project is before
        assert not document.can_undo


class TestMerge:
    def _heights(self, document: Document, height: int) -> None:
        track = document.project.timeline.tracks[0]
        with document.checkpoint("高さ", merge=True):
            document.execute(SetTrackHeights(((track.id, height),)))

    def test_continued_changes_share_one_step(self) -> None:
        # 壊れると、ホイールを回した回数だけ取り消すことになる
        document = Document(_project())
        with document.checkpoint("高さ"):
            document.execute(SetTrackHeights(((document.project.timeline.tracks[0].id, 80),)))
        self._heights(document, 100)
        self._heights(document, 120)
        assert document.history_labels == ("高さ",)
        document.undo()
        assert document.project.timeline.tracks[0].height == 60

    def test_an_unchanged_operation_still_breaks_the_run(self) -> None:
        # 変わらなかった操作のあとで続きを頼まれても、前の段へまとめない まとめると、
        # 1 回の取り消しで、しばらく前に変えた高さまで戻る
        document = Document(_project())
        self._heights(document, 80)
        with document.checkpoint("高さ"):
            document.execute(SetTrackHeights(((document.project.timeline.tracks[0].id, 80),)))
        self._heights(document, 100)
        assert document.history_labels == ("高さ", "高さ")

    def test_an_unchanged_continuation_also_breaks_the_run(self) -> None:
        # 続きとして頼まれても、何も変わらなければ区切りになる 区切らないと、
        # その次の変更が、しばらく前の段へまとまる
        document = Document(_project())
        self._heights(document, 80)
        self._heights(document, 80)
        self._heights(document, 100)
        assert document.history_labels == ("高さ", "高さ")

    def test_nothing_merges_into_a_step_uncovered_by_undo(self) -> None:
        # 取り消したあとの一番上は古い操作 そこへまとめると、関係ない操作と一緒に戻る
        document = Document(_project())
        self._heights(document, 80)
        document.execute(RenameProject("別名"))
        self._heights(document, 100)
        document.undo()
        document.undo()
        self._heights(document, 120)
        assert document.history_labels == ("高さ", "高さ")

    def test_a_different_operation_is_its_own_step(self) -> None:
        # 壊れると、名前の変更まで高さの段にまとまり、戻すと両方いっぺんに消える
        document = Document(_project())
        self._heights(document, 80)
        with document.checkpoint("名前", merge=True):
            document.execute(RenameProject("別名"))
        assert document.history_labels == ("高さ", "名前")
