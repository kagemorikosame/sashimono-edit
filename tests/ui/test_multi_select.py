"""タイムラインで何本かのクリップを選んで、まとめて扱う

選び方（Ctrl・Shift・囲む）と、選んだものへの操作（動かす・消す・コピー）が
正しいコマンド 1 つになって出てくるかを見る 中身の正しさは core 側
（test_multi_clip_commands.py）で見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import (
    AddEffect,
    Command,
    GroupClips,
    MoveClips,
    ParamPath,
    RemoveClips,
    SetClipProperty,
    SetParam,
    SetTrackHeights,
    SplitClip,
    TrimClips,
    insert_media,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    MediaItem,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.main_window import MainWindow
from sashimono.ui.timeline import TimelineView

_CTRL = Qt.KeyboardModifier.ControlModifier
_SHIFT = Qt.KeyboardModifier.ShiftModifier
_LEFT = Qt.MouseButton.LeftButton


def _project() -> Project:
    """V1 に 2 本（0〜30、40〜70）、V2 に 1 本（0〜30）"""
    base = Project.create()

    def text(start: int) -> Clip:
        return Clip(timeline_start=start, duration=30, source=TEXT.create())

    tracks = (
        Track(TrackKind.VIDEO, "V1", (text(0), text(40))),
        Track(TrackKind.VIDEO, "V2", (text(0),)),
        Track(TrackKind.AUDIO, "A1"),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def view(qt_application: QApplication) -> Iterator[TimelineView]:
    del qt_application
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    created = TimelineView(_project(), analyzer)
    created.resize(900, 400)
    yield created
    analyzer.close()


def _ids(view: TimelineView) -> tuple[ClipId, ClipId, ClipId]:
    v1, v2 = view.project.timeline.tracks[0], view.project.timeline.tracks[1]
    return v1.clips[0].id, v1.clips[1].id, v2.clips[0].id


def _point(view: TimelineView, track: int, frame: int) -> QPoint:
    """``tracks[track]`` の上の点 画面の並びは映像が下から積まれるので、ID で引く"""
    wanted = view.project.timeline.tracks[track].id
    band = next(b for b in view._layout.bands(view.project.timeline) if b.track.id == wanted)
    return QPoint(int(view._layout.frame_to_x(frame)), band.top + band.height // 2)


def _received(view: TimelineView) -> list[list[Command]]:
    received: list[list[Command]] = []
    view.commands_requested.connect(lambda commands, _label: received.append(commands))
    return received


class TestChoosing:
    def test_ctrl_click_adds_and_removes(self, view: TimelineView) -> None:
        # 壊れると、2 本目を押したときに 1 本目の選択が消えてまとめて扱えない
        a, b, _ = _ids(view)
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 0, 50))
        assert list(view.selected_clips) == [a, b]
        QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 0, 10))
        assert list(view.selected_clips) == [b]

    def test_shift_click_takes_everything_in_between(self, view: TimelineView) -> None:
        # 起点と今のクリップを両隅にした範囲 間にあるクリップが漏れると、1 本ずつ
        # Ctrl で足し直すことになる
        a, b, c = _ids(view)
        QTest.mouseClick(view, _LEFT, pos=_point(view, 1, 10))
        QTest.mouseClick(view, _LEFT, _SHIFT, _point(view, 0, 50))
        assert set(view.selected_clips) == {a, b, c}
        assert view.selected_clip == b

    def test_shift_range_takes_whole_groups(self, view: TimelineView) -> None:
        # 範囲に仲間の一部だけが入ったまま動かすと、束が裂ける
        a, b, c = _ids(view)
        view.set_project(GroupClips((b, c)).apply(view.project))
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseClick(view, _LEFT, _SHIFT, _point(view, 0, 50))
        assert set(view.selected_clips) == {a, b, c}

    def test_ctrl_adds_a_group_in_timeline_order(self, view: TimelineView) -> None:
        # 集合から並べると、選んだ順（selected_clips）が実行のたびに変わる
        a, b, c = _ids(view)
        view.set_project(GroupClips((a, c)).apply(view.project))
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 50))
        QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 0, 10))
        assert list(view.selected_clips) == [b, c, a]

    def test_shift_range_follows_the_order_on_screen(self, view: TimelineView) -> None:
        # 映像は下から積むので、画面では V2・V1・A1 の順 モデルの並びで範囲を取ると、
        # 画面で間に見えている V1 が抜ける
        project = view.project
        audio = Clip(timeline_start=0, duration=30, source=TEXT.create())
        a1 = project.timeline.tracks[2]
        view.set_project(
            project.with_timeline(project.timeline.replace_track(a1.with_clips((audio,))))
        )
        a, _, c = _ids(view)
        view.select(c)
        view._select_range(c, audio.id)
        assert a in view.selected_clips

    def test_dragging_on_empty_space_draws_a_box(self, view: TimelineView) -> None:
        # 壊れると、たくさんのクリップを 1 本ずつクリックして選ぶことになる
        start, end = _point(view, 0, 80), _point(view, 1, 5)
        QTest.mousePress(view, _LEFT, pos=start)
        QTest.mouseMove(view, end)
        QTest.mouseRelease(view, _LEFT, pos=end)
        assert set(view.selected_clips) == set(_ids(view))

    def test_a_click_on_empty_space_clears_and_moves_the_playhead(self, view: TimelineView) -> None:
        # 囲む操作を足しても、空いた所のクリックの意味（選択を解いて移動）は変えない
        view.select_all()
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 80))
        assert view.selected_clips == ()
        assert view.playhead == 80

    def test_escape_clears_the_selection(self, view: TimelineView) -> None:
        # 壊れると選択を解く手段がクリックしか無く、次の操作が前の選択へ掛かる
        view.select_all()
        QTest.keyClick(view, Qt.Key.Key_Escape)
        assert view.selected_clips == ()

    def test_removed_clips_leave_the_selection(self, view: TimelineView) -> None:
        # 消えたクリップの ID が残ると、次の操作が「見つからない」で失敗する
        a, b, _ = _ids(view)
        view.set_selection((a, b))
        view.set_project(RemoveClips((b,)).apply(view.project))
        assert view.selected_clips == (a,)


class TestActingOnMany:
    def test_dragging_one_of_them_moves_them_all(self, view: TimelineView) -> None:
        # 壊れると、掴んだ 1 本だけが動いて並びが崩れる
        a, _, c = _ids(view)
        view.set_selection((a, c))
        received = _received(view)
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseMove(view, _point(view, 0, 100))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 100))
        ((command,),) = received
        assert isinstance(command, MoveClips)
        assert set(command.clip_ids) == {a, c}
        assert command.delta > 0

    def test_a_plain_click_on_a_selected_clip_keeps_the_rest(self, view: TimelineView) -> None:
        # 掴んだ瞬間に選び直すと、まとめて動かすことができない
        a, b, _ = _ids(view)
        view.set_selection((a, b))
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 10))
        assert set(view.selected_clips) == {a, b}
        assert view.selected_clip == a

    def test_ctrl_click_can_grab_what_it_just_added(self, view: TimelineView) -> None:
        # Ctrl を押したまま足したクリップを掴めないと、足すたびに手を離して掴み直す
        a, _, c = _ids(view)
        view.select(a)
        received = _received(view)
        QTest.mousePress(view, _LEFT, _CTRL, _point(view, 1, 10))
        QTest.mouseMove(view, _point(view, 1, 100))
        QTest.mouseRelease(view, _LEFT, _CTRL, _point(view, 1, 100))
        ((command,),) = received
        assert isinstance(command, MoveClips)
        assert set(command.clip_ids) == {a, c}

    def test_a_group_stops_at_the_start_together(self, view: TimelineView) -> None:
        # 掴んだ 1 本だけで 0 に止めると、前にいる別のクリップが先頭より前へ出て、
        # 離したときに断られる（枠では動けたように見えたのに）
        _, b, c = _ids(view)
        view.set_selection((c, b))
        received = _received(view)
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 50))
        QTest.mouseMove(view, _point(view, 0, 0))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 0))
        # c は 0 から始まっているので、どこへ引いても前へは動けない
        assert received == []

    def test_locked_clips_stay_out_of_a_group_move(self, view: TimelineView) -> None:
        # Ctrl+A はロックしたトラックも選ぶ そのまま渡すと、ほかも一緒に動かせなくなる
        project = view.project
        v2 = project.timeline.tracks[1]
        view.set_project(
            project.with_timeline(project.timeline.replace_track(replace(v2, locked=True)))
        )
        a, b, c = _ids(view)
        view.select_all()
        received = _received(view)
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseMove(view, _point(view, 0, 100))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 100))
        ((command,),) = received
        assert isinstance(command, MoveClips)
        assert set(command.clip_ids) == {a, b}
        assert c not in command.clip_ids

    def test_grabbing_a_locked_clip_moves_nothing(self, view: TimelineView) -> None:
        # 動かすと、掴んだクリップはその場に残り、選んだほかのクリップだけが動く
        project = view.project
        v2 = project.timeline.tracks[1]
        view.set_project(
            project.with_timeline(project.timeline.replace_track(replace(v2, locked=True)))
        )
        view.select_all()
        received = _received(view)
        QTest.mousePress(view, _LEFT, pos=_point(view, 1, 10))
        QTest.mouseMove(view, _point(view, 1, 100))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 1, 100))
        assert received == []

    def test_a_clip_with_a_locked_partner_stays_out(
        self, view: TimelineView, video_media: MediaItem
    ) -> None:
        # 相手がロックした組を残すと、枠では動いて見えたのに、離すと全体が断られる
        base = view.project.with_media((video_media,))
        for command in insert_media(base, video_media, at_frame=100):
            base = command.apply(base)
        audio = base.timeline.tracks[2]
        base = base.with_timeline(base.timeline.replace_track(replace(audio, locked=True)))
        view.set_project(base)
        a, b, _ = _ids(view)
        linked = next(
            c.id
            for t in base.timeline.tracks
            if t.kind is TrackKind.VIDEO
            for c in t.clips
            if c.link_group is not None
        )
        view.set_selection((linked, b, a))
        assert set(view._movable_selection()) == {a, b}

    def test_removing_with_ctrl_leaves_the_anchor_on_what_remains(self, view: TimelineView) -> None:
        # 外したクリップが起点に残ると、次の Shift+クリックが選んでいないクリップから
        # 範囲を取る
        a, b, c = _ids(view)
        view.set_selection((a, c))
        QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 1, 10))
        QTest.mouseClick(view, _LEFT, _SHIFT, _point(view, 0, 50))
        assert set(view.selected_clips) == {a, b}

    def test_the_anchor_follows_any_selection(self, view: TimelineView) -> None:
        # AI が選んだあとの Shift+クリックが古い起点から範囲を取ると、見ていない
        # クリップまで選ばれる
        a, b, c = _ids(view)
        view.select(c)
        view.set_selection((a,))
        QTest.mouseClick(view, _LEFT, _SHIFT, _point(view, 0, 50))
        assert set(view.selected_clips) == {a, b}

    def test_delete_is_one_command(self, view: TimelineView) -> None:
        # 1 本ずつのコマンドになると、取り消しを本数ぶん押すことになる
        view.select_all()
        received = _received(view)
        view.delete_selected(ripple=True)
        ((command,),) = received
        assert isinstance(command, RemoveClips)
        assert command.ripple
        assert len(command.clip_ids) == 3

    def test_pasting_selects_everything_pasted(self, window: MainWindow) -> None:
        # 貼ったうち 1 本しか選ばれないと、続けてまとめて動かせない
        timeline = window._timeline
        a, b, _ = _ids(timeline)
        timeline.set_selection((a, b))
        assert timeline.copy_selected()
        window.seek(200)
        assert timeline.paste_at_playhead()
        assert len(timeline.selected_clips) == 2
        assert {a, b}.isdisjoint(timeline.selected_clips)

    def test_the_menu_speaks_for_all_of_them(self, view: TimelineView) -> None:
        # 右クリックで選択が 1 本に戻ると、まとめて消すつもりが 1 本だけ消える
        view.select_all()
        menu = view.build_context_menu(_point(view, 0, 10))
        assert "削除（3 本）" in [action.text() for action in menu.actions()]
        assert len(view.selected_clips) == 3


class TestHeightMerging:
    def test_quick_repeats_ask_to_continue(self, view: TimelineView) -> None:
        # 壊れると、ホイールを回した回数だけ取り消しの段が積まれる
        continued: list[list[Command]] = []
        view.commands_continued.connect(lambda commands, _label: continued.append(commands))
        received = _received(view)
        view.adjust_track_heights(12)
        view.adjust_track_heights(12)
        assert len(received) == 1
        assert len(continued) == 1
        assert isinstance(continued[0][0], SetTrackHeights)

    def test_hitting_the_limit_ends_the_run(self, view: TimelineView) -> None:
        # 上限に張り付いたあとすぐ反対へ回したぶんが前の段へまとまると、1 回の
        # 取り消しで張り付く前まで戻る
        def at(height: int) -> Project:
            timeline = view.project.timeline
            for track in timeline.tracks:
                timeline = timeline.replace_track(replace(track, height=height))
            return view.project.with_timeline(timeline)

        view.set_project(at(228))
        received = _received(view)
        view.adjust_track_heights(12)
        # ビューは自分では書き換えない 窓が実行したあとの形を渡し直す
        view.set_project(at(240))
        view.adjust_track_heights(12)
        view.adjust_track_heights(-12)
        assert len(received) == 2

    def test_the_window_undoes_them_at_once(self, window: MainWindow) -> None:
        # 壊れると、高さを変えた回数だけ取り消しを押すことになる
        timeline = window._timeline
        for _ in range(3):
            timeline.adjust_track_heights(12)
        assert window.document.project.timeline.tracks[0].height == 96
        window.undo()
        assert window.document.project.timeline.tracks[0].height == 60


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    yield created
    created.close()


class TestTogether:
    """何本か選んだときのトリム・トラック跨ぎ・設定パネル"""

    def test_trimming_the_edge_trims_them_all(self, view: TimelineView) -> None:
        a, b, _ = _ids(view)
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 0, 50))
        received = _received(view)
        # 2 本目の末尾（内側 1 フレーム）を掴んで 10 フレーム縮める
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 69))
        QTest.mouseMove(view, _point(view, 0, 60))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 60))
        (commands,) = received
        (command,) = commands
        assert isinstance(command, TrimClips)
        assert set(command.clip_ids) == {a, b}
        assert command.tail_delta == -10

    def test_a_group_can_change_track(self, view: TimelineView) -> None:
        a, b, _ = _ids(view)
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseClick(view, _LEFT, _CTRL, _point(view, 0, 50))
        received = _received(view)
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseMove(view, _point(view, 1, 10))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 1, 10))
        (commands,) = received
        (command,) = commands
        assert isinstance(command, MoveClips)
        assert set(command.clip_ids) == {a, b}
        assert command.track_delta == 1

    def test_the_inspector_sets_every_selected_clip(self, window: MainWindow) -> None:
        timeline = window._timeline
        a, b, _ = (
            timeline.project.timeline.tracks[0].clips[0].id,
            timeline.project.timeline.tracks[0].clips[1].id,
            timeline.project.timeline.tracks[1].clips[0].id,
        )
        timeline.set_selection((a, b))
        received: list[list[Command]] = []
        window._inspector.commands_requested.connect(lambda cs, _l: received.append(cs))
        # 設定パネルは主のクリップ（最後に選んだ b）の設定を出す 触ると a にも当たる
        window._inspector._emit(SetClipProperty(b, "blend_mode", "add"))
        (commands,) = received
        assert [c.clip_id for c in commands if isinstance(c, SetClipProperty)] == [b, a]

    def test_the_second_effect_of_a_kind_maps_to_the_second(self, window: MainWindow) -> None:
        # 同じ種類を 2 つ積んだクリップで、2 つ目を触ったのに 1 つ目へ当たってはいけない
        timeline = window._timeline
        clips = timeline.project.timeline.tracks[0].clips
        a, b = clips[0].id, clips[1].id
        blur = registry.get("blur")
        assert blur is not None
        first, second = blur.create(radius=4.0), blur.create(radius=8.0)
        window.execute_all(
            [
                AddEffect(a, first),
                AddEffect(a, second),
                AddEffect(b, blur.create(radius=1.0)),
                AddEffect(b, blur.create(radius=2.0)),
            ],
            "準備",
        )
        timeline.set_selection((b, a))
        received: list[list[Command]] = []
        window._inspector.commands_requested.connect(lambda cs, _l: received.append(cs))
        path = ParamPath.of_effect(a, second.id, "radius")
        window._inspector._emit(SetParam(path, AnimatedValue(16.0)))
        (commands,) = received
        targets = [c.path.effect_id for c in commands if isinstance(c, SetParam)]
        other = window.view_project.timeline.locate_clip(b)
        assert other is not None
        assert targets == [second.id, other[1].effects[1].id]

    def test_dropping_a_clip_from_the_selection_reaches_the_inspector(
        self, window: MainWindow
    ) -> None:
        # 主のクリップが変わらない増減で知らせないと、外したクリップへ設定が当たる
        timeline = window._timeline
        clips = timeline.project.timeline.tracks[0].clips
        a, b = clips[0].id, clips[1].id
        timeline.set_selection((a, b))
        assert set(window._inspector._selection) == {a, b}
        timeline.set_selection((b,))
        assert window._inspector._selection == (b,)


class TestGroupedValues:
    """グループの仲間として引き込まれたクリップへ、設定パネルの値を当てない

    AviUtl のグループ化は動かす・選ぶ所だけを束ね、設定は押した 1 本だけが変わる
    前は 1 本の拡大率を変えると束ねた全部の拡大率が一緒に変わり、束ねた物ごとに
    大きさを合わせられなかった
    """

    def _grouped(self, window: MainWindow) -> tuple[ClipId, ClipId, ClipId]:
        timeline = window._timeline
        a, b, c = _ids(timeline)
        window.execute_all([GroupClips((a, c))], "グループ化")
        timeline.resize(900, 400)
        return a, b, c

    def test_clicking_a_member_sets_only_that_clip(self, window: MainWindow) -> None:
        # 壊れると、1 本の拡大率を変えただけで仲間の拡大率まで変わる
        timeline = window._timeline
        a, _b, c = self._grouped(window)
        QTest.mouseClick(timeline, _LEFT, pos=_point(timeline, 0, 10))
        assert set(timeline.selected_clips) == {a, c}
        received: list[list[Command]] = []
        window._inspector.commands_requested.connect(lambda cs, _l: received.append(cs))
        window._inspector._emit(SetClipProperty(a, "blend_mode", "add"))
        (commands,) = received
        assert [cmd.clip_id for cmd in commands if isinstance(cmd, SetClipProperty)] == [a]

    def test_clips_chosen_by_hand_still_share_the_value(self, window: MainWindow) -> None:
        # 仲間を除くのは引き込んだ物だけ Ctrl で足した別のクリップにはこれまでどおり当てる
        timeline = window._timeline
        a, b, c = self._grouped(window)
        QTest.mouseClick(timeline, _LEFT, pos=_point(timeline, 0, 50))
        QTest.mouseClick(timeline, _LEFT, _CTRL, _point(timeline, 0, 10))
        assert set(timeline.selected_clips) == {a, b, c}
        assert set(window._inspector._selection) == {a, b}

    def test_a_marquee_counts_what_it_touched(self, view: TimelineView) -> None:
        # 囲んで掛かった物は自分で選んだ物 仲間だけが引き込んだ物になる
        a, _b, c = _ids(view)
        view.set_project(GroupClips((a, c)).apply(view.project))
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 35))
        QTest.mouseMove(view, _point(view, 0, 20))
        QTest.mouseMove(view, _point(view, 0, 5))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 5))
        assert {a, c} <= set(view.selected_clips)
        assert a in view.edit_targets
        assert c not in view.edit_targets


class TestLockedGroup:
    """ロックしたレイヤーのクリップを含むグループには、まとめて当てる操作をしない

    ロックした仲間だけが残り、ほかだけが割れたり動いたりすると、束ねた物の頭や長さが
    食い違う 黙って何もしないと効いていないのか分からないので理由を出す
    """

    def _locked_group(self, view: TimelineView) -> tuple[ClipId, ClipId, ClipId, list[str]]:
        a, b, c = _ids(view)
        project = GroupClips((a, c)).apply(view.project)
        v2 = project.timeline.tracks[1]
        view.set_project(
            project.with_timeline(project.timeline.replace_track(replace(v2, locked=True)))
        )
        messages: list[str] = []
        view.status_message.connect(messages.append)
        return a, b, c, messages

    def test_splitting_is_refused(self, view: TimelineView) -> None:
        # 壊れると、ロックしていない a だけが割れ、c とグループの頭がずれる
        a, _b, _c, messages = self._locked_group(view)
        view.select(a)
        view.set_playhead(10)
        received = _received(view)
        view.split_at_playhead()
        assert received == []
        assert any("ロック" in m and "グループ" in m for m in messages)

    def test_splitting_everything_skips_only_the_group(self, view: TimelineView) -> None:
        # 何も選ばずに切るときは、グループの外のクリップは今までどおり切る
        _a, _b, _c, messages = self._locked_group(view)
        project = view.project
        extra = Clip(timeline_start=0, duration=30, source=TEXT.create())
        tracks = (*project.timeline.tracks, Track(TrackKind.VIDEO, "V3", (extra,)))
        view.set_project(project.with_timeline(replace(project.timeline, tracks=tracks)))
        view.select(None)
        view.set_playhead(10)
        received = _received(view)
        view.split_at_playhead()
        (commands,) = received
        assert [c.clip_id for c in commands if isinstance(c, SplitClip)] == [extra.id]
        assert len(commands) == 1
        assert any("グループ" in m for m in messages)

    def test_deleting_and_cutting_are_refused(self, view: TimelineView) -> None:
        a, _b, _c, messages = self._locked_group(view)
        view.select(a)
        received = _received(view)
        view.delete_selected()
        assert not view.cut_selected()
        assert received == []
        assert len([m for m in messages if "ロック" in m]) == 2

    def test_moving_is_refused(self, view: TimelineView) -> None:
        # 前はロックした c を外して a だけを動かし、グループが裂けた
        _a, _b, _c, messages = self._locked_group(view)
        received = _received(view)
        QTest.mousePress(view, _LEFT, pos=_point(view, 0, 10))
        QTest.mouseMove(view, _point(view, 0, 60))
        QTest.mouseRelease(view, _LEFT, pos=_point(view, 0, 60))
        assert received == []
        assert any("ロック" in m for m in messages)

    def test_selecting_alone_says_nothing(self, view: TimelineView) -> None:
        # 押しただけで断りを出すと、選ぶたびにステータスバーが埋まる
        _a, _b, _c, messages = self._locked_group(view)
        QTest.mouseClick(view, _LEFT, pos=_point(view, 0, 10))
        assert messages == []
