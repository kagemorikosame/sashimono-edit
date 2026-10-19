"""ミュート・ソロ・ロックと解像度の変更

どちらも「UI が無かっただけで中身はあった」ものに、触る入口を付けた
入口がコマンドであることと、プレビュー・ミキサ・書き出しが同じ判断をすることを押さえる
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import Document, SetResolution, SetTrackHeights, SetTrackState
from sashimono.core.io import project_from_dict, project_to_dict
from sashimono.core.model import Project, Timeline, Track, TrackKind
from sashimono.core.timebase import FrameRate


def timeline(*tracks: Track) -> Timeline:
    return Timeline(rate=FrameRate(30), tracks=tracks)


def names(tracks: tuple[Track, ...]) -> list[str]:
    return [track.name for track in tracks]


class TestActiveTracks:
    def test_everything_plays_by_default(self) -> None:
        # 何も付けていないトラックが外れると、そのトラックの音がミキサで鳴らなくなる
        line = timeline(Track(TrackKind.AUDIO, "A1"), Track(TrackKind.AUDIO, "A2"))
        assert names(line.active_tracks(TrackKind.AUDIO)) == ["A1", "A2"]

    def test_a_muted_track_is_left_out(self) -> None:
        # ミュートしたトラックがミキサから外れず、消したはずの音が鳴る
        line = timeline(Track(TrackKind.AUDIO, "A1", muted=True), Track(TrackKind.AUDIO, "A2"))
        assert names(line.active_tracks(TrackKind.AUDIO)) == ["A2"]

    def test_solo_leaves_only_the_soloed(self) -> None:
        # ソロにしてもほかのトラックの音が混ざり、1 本だけを聞き比べられない
        line = timeline(Track(TrackKind.AUDIO, "A1"), Track(TrackKind.AUDIO, "A2", solo=True))
        assert names(line.active_tracks(TrackKind.AUDIO)) == ["A2"]

    def test_solo_does_not_cross_kinds(self) -> None:
        # 音声をソロにして映像まで消えると、音を聞き比べるたびに画面が真っ黒になる
        line = timeline(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1", solo=True))
        assert names(line.active_tracks(TrackKind.VIDEO)) == ["V1"]

    def test_solo_applies_to_video_too(self) -> None:
        # 以前は映像のソロが無視されていた（ミキサだけが見ていた）
        line = timeline(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2", solo=True))
        assert names(line.active_tracks(TrackKind.VIDEO)) == ["V2"]

    def test_a_muted_solo_does_not_silence_the_rest(self) -> None:
        # ソロを外し忘れたトラックをミュートしただけで、全部が無音になっては困る
        line = timeline(
            Track(TrackKind.AUDIO, "A1", muted=True, solo=True), Track(TrackKind.AUDIO, "A2")
        )
        assert names(line.active_tracks(TrackKind.AUDIO)) == ["A2"]

    def test_the_order_is_kept(self) -> None:
        # 映像は並び順が重なり順 入れ替わると手前と奥が逆になる
        line = timeline(
            Track(TrackKind.VIDEO, "V1", solo=True),
            Track(TrackKind.VIDEO, "V2"),
            Track(TrackKind.VIDEO, "V3", solo=True),
        )
        assert names(line.active_tracks(TrackKind.VIDEO)) == ["V1", "V3"]


@pytest.fixture
def two_tracks() -> Project:
    base = Project.create()
    tracks = (Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1", locked=True))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


class TestSetTrackState:
    def test_only_the_given_flag_changes(self, two_tracks: Project) -> None:
        # ソロを押しただけでミュートやロックまで書き換わると、音が消えるかクリップに触れなくなる
        track = two_tracks.timeline.tracks[0]
        changed = SetTrackState(track.id, solo=True).apply(two_tracks)
        updated = changed.timeline.tracks[0]
        assert (updated.solo, updated.muted, updated.locked) == (True, False, False)

    def test_a_locked_track_can_still_be_muted(self, two_tracks: Project) -> None:
        # ロックはクリップを守るもの 聞こえ方まで固めると消音できなくなる
        track = two_tracks.timeline.tracks[1]
        changed = SetTrackState(track.id, muted=True).apply(two_tracks)
        assert changed.timeline.tracks[1].muted

    def test_it_is_one_undo_step(self, two_tracks: Project) -> None:
        # ミュートを押したあと 1 回の取り消しで押す前へ戻らない
        document = Document(two_tracks)
        document.execute(SetTrackState(two_tracks.timeline.tracks[0].id, muted=True))
        document.undo()
        assert document.project is two_tracks

    def test_an_unknown_track_fails(self, two_tracks: Project) -> None:
        # 呼び元の画面と AI の道具は先に有無を見ている 断らないと、消えたトラックを指す
        # コマンドが黙って通り、呼び手は変えられなかったことに気付けない
        with pytest.raises(KeyError):
            SetTrackState(Track(TrackKind.VIDEO).id, muted=True).apply(two_tracks)

    def test_nothing_given_changes_nothing(self, two_tracks: Project) -> None:
        # 何も指定しないのに新しいプロジェクトを返すと、取り消しても何も起きない段が履歴に積まれる
        assert SetTrackState(two_tracks.timeline.tracks[0].id).apply(two_tracks) is two_tracks

    def test_the_label_says_what_happened(self, two_tracks: Project) -> None:
        # 履歴には操作名だけが出る 「状態を変更」では、どれを押したのか分からない
        track_id = two_tracks.timeline.tracks[0].id
        assert SetTrackState(track_id, muted=True).label == "ミュート"
        assert SetTrackState(track_id, solo=False).label == "ソロを解除"


class TestSetResolution:
    def test_the_size_changes(self) -> None:
        # 壊れると、設定画面で選んだ解像度と書き出した動画の解像度が食い違う
        changed = SetResolution(1080, 1920).apply(Project.create())
        assert changed.settings.resolution == (1080, 1920)

    def test_the_frame_rate_is_untouched(self) -> None:
        # 一緒に変わると、フレーム番号で持っている全クリップの時刻がずれる
        project = Project.create()
        assert SetResolution(1280, 720).apply(project).rate == project.rate

    @pytest.mark.parametrize("size", [(1919, 1080), (1920, 1081)])
    def test_odd_sizes_are_refused(self, size: tuple[int, int]) -> None:
        # yuv420p は 2x2 画素ごとに色を持つ 奇数だと書き出しの最後で断られる
        with pytest.raises(ValueError, match="偶数"):
            SetResolution(*size).apply(Project.create())

    @pytest.mark.parametrize("size", [(0, 1080), (1920, 10000)])
    def test_out_of_range_is_refused(self, size: tuple[int, int]) -> None:
        # 受け付けると、GPU のテクスチャが作れずプレビューごと落ちる
        with pytest.raises(ValueError):
            SetResolution(*size).apply(Project.create())

    def test_it_survives_saving(self) -> None:
        # 壊れると、開き直すたびに解像度が 1920x1080 へ戻る
        changed = SetResolution(1080, 1080).apply(Project.create())
        assert project_from_dict(project_to_dict(changed)).settings.resolution == (1080, 1080)


class TestSetTrackHeights:
    def test_the_height_changes(self, two_tracks: Project) -> None:
        # 壊れると、境目をドラッグしてもメニューで変えても、高さが画面に残らない
        track = two_tracks.timeline.tracks[0]
        changed = SetTrackHeights(((track.id, 120),)).apply(two_tracks)
        assert changed.timeline.tracks[0].height == 120

    @pytest.mark.parametrize(("asked", "expected"), [(1, 28), (9999, 240)])
    def test_it_stays_within_range(self, two_tracks: Project, asked: int, expected: int) -> None:
        # 0 にすると押せない行が、巨大にすると画面を占領する 1 本ができる
        track = two_tracks.timeline.tracks[0]
        changed = SetTrackHeights(((track.id, asked),)).apply(two_tracks)
        assert changed.timeline.tracks[0].height == expected

    def test_many_tracks_are_one_undo_step(self, two_tracks: Project) -> None:
        # 全トラックをまとめて変えて、トラックの数だけ取り消すのは手間
        document = Document(two_tracks)
        heights = tuple((track.id, 100) for track in two_tracks.timeline.tracks)
        document.execute(SetTrackHeights(heights))
        document.undo()
        assert document.project is two_tracks

    def test_it_survives_saving(self, two_tracks: Project) -> None:
        # 壊れると、開き直すたびに高さが既定へ戻る
        track = two_tracks.timeline.tracks[0]
        changed = SetTrackHeights(((track.id, 150),)).apply(two_tracks)
        assert project_from_dict(project_to_dict(changed)).timeline.tracks[0].height == 150
