"""タイムラインの操作（Issue #27 の 2 回目の要望）

- 分割は選んだクリップだけを切る 選んでいなければ再生ヘッドの下の全部
- ヘッダを掴んでトラックの順を入れ替える
- スクロールバーのつまみの端で、表示の大きさを変える
- ドラッグ中に端へ寄ったら表示を送る
- キーフレームの位置をクリップの上にひし形で出す
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from itertools import pairwise

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.core.commands import (
    Command,
    MoveTrack,
    ParamPath,
    SetKeyframe,
    SetTrackHeights,
    SplitClip,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    Keyframe,
    Project,
    Track,
    TrackKind,
    new_group_id,
)
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline import TimelineArea, TimelineView
from sashimono.ui.timeline.auto_scroll import EDGE_ZONE, edge_speed
from sashimono.ui.timeline.keyframes import (
    MIN_KEYFRAME_GAP,
    keyframe_frames,
)
from sashimono.ui.timeline.zoom_scrollbar import ZoomScrollBar

_LEFT = Qt.MouseButton.LeftButton
_NONE = Qt.KeyboardModifier.NoModifier


def _text(start: int, duration: int, **changes: object) -> Clip:
    clip = Clip(timeline_start=start, duration=duration, source=TEXT.create())
    return replace(clip, **changes)  # type: ignore[arg-type] # 試験で差し替える項目は様々


def _project(*tracks: Track) -> Project:
    base = Project.create()
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def analyzer() -> Iterator[MediaAnalyzer]:
    created = MediaAnalyzer(sample_rate=48000, channels=2)
    yield created
    created.close()


class _Harness:
    """ビューを窓の代わりに受け持つ 出たコマンドを当てて、プロジェクトを差し戻す"""

    def __init__(self, view: TimelineView) -> None:
        self.view = view
        self.received: list[list[Command]] = []
        view.commands_requested.connect(self._apply)

    def _apply(self, commands: list[Command], _label: str) -> None:
        self.received.append(list(commands))
        project = self.view.project
        for command in commands:
            project = command.apply(project)
        self.view.set_project(project)


def _area(project: Project, analyzer: MediaAnalyzer, size: tuple[int, int]) -> TimelineArea:
    area = TimelineArea(TimelineView(project, analyzer))
    area.resize(*size)
    area.show()
    QApplication.processEvents()
    return area


@pytest.fixture
def make_area(
    qt_application: QApplication, analyzer: MediaAnalyzer
) -> Iterator[list[TimelineArea]]:
    del qt_application
    made: list[TimelineArea] = []
    yield made
    for area in made:
        area.close()
        # 閉じただけで残すと、いつかのごみ集めで壊され、そのとき走っている別の試験で落ちる
        shiboken6.delete(area)


def _open(
    made: list[TimelineArea],
    analyzer: MediaAnalyzer,
    project: Project,
    size: tuple[int, int] = (900, 300),
) -> tuple[TimelineView, _Harness]:
    area = _area(project, analyzer, size)
    made.append(area)
    harness = _Harness(area.view)
    # 受け持ちを試験の側で捨てても（``view, _ = ...`` のあと ``_`` を使い回すなど）、
    # 信号の繋がりが切れてコマンドが当たらなくならないよう、ビューに持たせておく
    area.view.setProperty("harness", harness)
    return area.view, harness


def _band(view: TimelineView, name: str) -> tuple[int, int]:
    """名前のトラックの帯の ``(上, 下)``"""
    for band in view.view_layout.bands(view.project.timeline):
        if band.track.name == name:
            return band.top, band.bottom
    raise AssertionError(f"トラックが無い: {name}")


def _clip_point(view: TimelineView, name: str, frame: int) -> QPoint:
    top, bottom = _band(view, name)
    return QPoint(int(view.view_layout.frame_to_x(frame)), (top + bottom) // 2)


# --- 分割 ---


class TestSplit:
    def _two_layers(self) -> Project:
        return _project(
            Track(TrackKind.VIDEO, "V1", (_text(0, 100),)),
            Track(TrackKind.VIDEO, "V2", (_text(0, 100),)),
        )

    def test_only_the_selected_clip_is_cut(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 選択を見ずに全部を切っていたので、1 本だけ切るつもりでほかのレイヤーまで切れた
        view, harness = _open(make_area, analyzer, self._two_layers())
        first = view.project.timeline.tracks[0].clips[0]
        view.select(first.id)
        view.set_playhead(50)
        view.split_at_playhead()
        ((command,),) = harness.received
        assert isinstance(command, SplitClip)
        assert command.clip_id == first.id
        assert [len(t.clips) for t in view.project.timeline.tracks] == [2, 1]

    def test_everything_under_the_playhead_without_a_selection(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 選んでいないときまで何も切らなくなると、S キーで再生ヘッドの下をまとめて割る
        # いつもの使い方（Premiere の Ctrl+K や AviUtl の分割と同じ）ができなくなる
        view, _ = _open(make_area, analyzer, self._two_layers())
        view.set_playhead(50)
        view.split_at_playhead()
        assert [len(t.clips) for t in view.project.timeline.tracks] == [2, 2]

    def test_a_locked_track_is_left_alone(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # ロックしたトラックまで割ると、ロックで守っているはずのクリップが変わる
        project = _project(
            Track(TrackKind.VIDEO, "V1", (_text(0, 100),)),
            Track(TrackKind.VIDEO, "V2", (_text(0, 100),), locked=True),
        )
        view, _ = _open(make_area, analyzer, project)
        view.set_playhead(50)
        view.split_at_playhead()
        assert [len(t.clips) for t in view.project.timeline.tracks] == [2, 1]

    def test_the_linked_partner_is_cut_with_the_selection(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 映像だけ選んでも、リンクした音声は一緒に割れる 片方だけ割ると同期が崩れる
        link = new_group_id()
        project = _project(
            Track(TrackKind.VIDEO, "V1", (_text(0, 100, link_group=link),)),
            Track(TrackKind.VIDEO, "V2", (_text(0, 100),)),
            Track(TrackKind.AUDIO, "A1", (_text(0, 100, link_group=link),)),
        )
        view, _ = _open(make_area, analyzer, project)
        view.select(project.timeline.tracks[0].clips[0].id)
        view.set_playhead(40)
        view.split_at_playhead()
        assert [len(t.clips) for t in view.project.timeline.tracks] == [2, 1, 2]

    def test_nothing_is_cut_when_the_selection_misses_the_playhead(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 選んだものが無いからと全部を切ると、選び方の意味が消える
        project = _project(
            Track(TrackKind.VIDEO, "V1", (_text(0, 30), _text(60, 100))),
            Track(TrackKind.VIDEO, "V2", (_text(0, 200),)),
        )
        view, harness = _open(make_area, analyzer, project)
        messages: list[str] = []
        view.status_message.connect(messages.append)
        view.select(project.timeline.tracks[0].clips[0].id)
        view.set_playhead(100)
        view.split_at_playhead()
        assert harness.received == []
        assert messages and "位置" in messages[-1]

    def test_a_locked_selection_is_told_as_locked(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 理由がロックなのに「位置にない」と出すと、再生ヘッドを動かし直すだけで終わる
        # （PR #155 の指摘）
        project = _project(
            Track(TrackKind.VIDEO, "V1", (_text(0, 100),), locked=True),
            Track(TrackKind.VIDEO, "V2", (_text(0, 100),)),
        )
        view, harness = _open(make_area, analyzer, project)
        messages: list[str] = []
        view.status_message.connect(messages.append)
        view.select(project.timeline.tracks[0].clips[0].id)
        view.set_playhead(50)
        view.split_at_playhead()
        assert harness.received == []
        assert messages and "ロック" in messages[-1]

    def test_the_context_menu_follows_the_same_rule(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 右クリックの分割だけが選択を見ないと、S キーと右クリックで切れる物が変わり、
        # 右クリックしたクリップ以外のレイヤーまで切れる
        view, _ = _open(make_area, analyzer, self._two_layers())
        view.set_playhead(50)
        menu = view.build_context_menu(_clip_point(view, "V2", 20))
        (split,) = [a for a in menu.actions() if a.text() == "再生ヘッドで分割"]
        split.trigger()
        assert [len(t.clips) for t in view.project.timeline.tracks] == [1, 2]


# --- トラックの並べ替え ---


def _four_tracks() -> Project:
    return _project(
        Track(TrackKind.VIDEO, "V1"),
        Track(TrackKind.VIDEO, "V2"),
        Track(TrackKind.VIDEO, "V3"),
        Track(TrackKind.AUDIO, "A1"),
    )


def _drag_header(view: TimelineView, name: str, y: int) -> None:
    top, bottom = _band(view, name)
    start = QPoint(20, (top + bottom) // 2)
    QTest.mousePress(view, _LEFT, _NONE, start)
    QTest.mouseMove(view, QPoint(20, (start.y() + y) // 2))
    QTest.mouseMove(view, QPoint(20, y))
    QTest.mouseRelease(view, _LEFT, _NONE, QPoint(20, y))


def _names(view: TimelineView) -> list[str]:
    return [t.name for t in view.project.timeline.tracks]


class TestTrackReorder:
    def test_dragging_the_bottom_video_to_the_top(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 画面は上から V3 V2 V1 A1 V1 をいちばん上へ運ぶと、いちばん手前に重なる
        view, harness = _open(make_area, analyzer, _four_tracks())
        top, _ = _band(view, "V3")
        _drag_header(view, "V1", top + 2)
        ((command,),) = harness.received
        assert isinstance(command, MoveTrack)
        assert _names(view) == ["V2", "V3", "V1", "A1"]

    def test_a_video_track_does_not_cross_into_the_audio(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 音声の欄で離しても、映像のいちばん下に入る 映像を音声の間に置く並びは無い
        view, _harness = _open(make_area, analyzer, _four_tracks())
        _, bottom = _band(view, "A1")
        _drag_header(view, "V3", bottom - 2)
        assert _names(view) == ["V3", "V1", "V2", "A1"]
        assert [b.track.name for b in view.view_layout.bands(view.project.timeline)][-1] == "A1"

    def test_a_locked_track_cannot_be_grabbed(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # ロックしたトラックを名前で掴んで動かせると、触らないと決めた重ね順が手ぶれで崩れる
        project = _four_tracks()
        locked = replace(project.timeline.tracks[0], locked=True)
        view, harness = _open(
            make_area, analyzer, project.with_timeline(project.timeline.replace_track(locked))
        )
        top, _ = _band(view, "V3")
        _drag_header(view, "V1", top + 2)
        assert harness.received == []

    def test_a_click_on_the_name_does_not_reorder(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 名前を押しただけの手ぶれで並びが変わると、何もしていないのに重ね順が変わる
        view, harness = _open(make_area, analyzer, _four_tracks())
        top, bottom = _band(view, "V2")
        point = QPoint(20, (top + bottom) // 2)
        QTest.mousePress(view, _LEFT, _NONE, point)
        QTest.mouseRelease(view, _LEFT, _NONE, point + QPoint(0, 1))
        assert harness.received == []

    def test_the_drop_line_is_drawn_while_dragging(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 引いている間に線が出ないと、離すまでどのトラックの間へ入るのか分からない
        view, _ = _open(make_area, analyzer, _four_tracks())
        top, bottom = _band(view, "V1")
        v3_top, _ = _band(view, "V3")
        QTest.mousePress(view, _LEFT, _NONE, QPoint(20, (top + bottom) // 2))
        QTest.mouseMove(view, QPoint(20, v3_top + 20))
        QTest.mouseMove(view, QPoint(20, v3_top + 2))
        image = view.grab().toImage()
        QTest.mouseRelease(view, _LEFT, _NONE, QPoint(20, v3_top + 2))
        # 線は入る位置（V3 の上端）に全幅で引く ヘッダの外（タイムラインの上）でも見える
        assert QColor(image.pixel(400, v3_top)) == Colors.SELECTION


# --- スクロールバーのつまみの端 ---


def _long(frames: int = 3000, tracks: int = 2) -> Project:
    made = [Track(TrackKind.VIDEO, "V1", (_text(0, frames),))]
    made += [Track(TrackKind.VIDEO, f"V{i + 2}") for i in range(tracks - 1)]
    return _project(*made)


def _hover(widget: QWidget, point: QPoint) -> None:
    """ボタンを押さずに乗せる

    ``QTest.mouseMove`` は画面のある環境では本物のマウスを動かし、届くのが遅れる
    乗せただけの形の変化を見るので、出来事を直に渡す
    """
    local = QPointF(point)
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        local,
        QPointF(widget.mapToGlobal(point)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        _NONE,
    )
    QApplication.sendEvent(widget, event)


def _drag_bar(bar: ZoomScrollBar, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(bar, _LEFT, _NONE, start)
    QTest.mouseMove(bar, end)
    QTest.mouseRelease(bar, _LEFT, _NONE, end)


class TestZoomScrollBar:
    def test_pulling_the_right_edge_in_zooms_in(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 右の端を引いても拡大しないと、見たい所を広げるのにつまみと別の拡大の操作を往復する
        view, _ = _open(make_area, analyzer, _long())
        bar = view.horizontal_scroll_bar
        assert isinstance(bar, ZoomScrollBar)
        before = view.view_layout
        handle = bar.handle_rect()
        edge = QPoint(handle.right() - 1, handle.center().y())
        _drag_bar(bar, edge, edge - QPoint(handle.width() // 2, 0))
        after = view.view_layout
        assert after.pixels_per_frame > before.pixels_per_frame * 1.5
        # 動かしていない左の端は、見ていた場所のまま 逃げると、拡大するたびに探し直す
        assert after.scroll_frame == pytest.approx(before.scroll_frame, abs=1.0)

    def test_pushing_the_left_edge_out_zooms_out(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 左の端を外へ押しても縮小しないと、長い作品の全体を見渡すのに別の操作が要る
        view, _ = _open(make_area, analyzer, _long())
        bar = view.horizontal_scroll_bar
        assert isinstance(bar, ZoomScrollBar)
        bar.setValue(bar.maximum() // 2)
        before = view.view_layout
        right_frame = before.scroll_frame + before.frames_in(view.width())
        handle = bar.handle_rect()
        edge = QPoint(handle.left() + 1, handle.center().y())
        _drag_bar(bar, edge, edge - QPoint(60, 0))
        after = view.view_layout
        assert after.pixels_per_frame < before.pixels_per_frame
        # 右の端（動かしていない側）は、見ていたフレームのまま
        assert after.scroll_frame + after.frames_in(view.width()) == pytest.approx(
            right_frame, rel=0.02
        )

    def test_the_middle_still_scrolls(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 中を掴んで伸び縮みになると、今までのスクロールができなくなる
        view, _ = _open(make_area, analyzer, _long())
        bar = view.horizontal_scroll_bar
        before = view.view_layout
        handle = bar.handle_rect()
        _drag_bar(bar, handle.center(), handle.center() + QPoint(80, 0))
        after = view.view_layout
        assert after.pixels_per_frame == before.pixels_per_frame
        assert after.scroll_frame > before.scroll_frame

    def test_the_cursor_shows_the_edge(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 形が変わらないと、つまみの端を掴めば伸び縮みすると気付けず、中を掴んでスクロールしてしまう
        view, _ = _open(make_area, analyzer, _long())
        bar = view.horizontal_scroll_bar
        handle = bar.handle_rect()
        _hover(bar, QPoint(handle.right() - 1, handle.center().y()))
        assert bar.cursor().shape() == Qt.CursorShape.SizeHorCursor
        _hover(bar, handle.center())
        assert bar.cursor().shape() != Qt.CursorShape.SizeHorCursor

    def test_the_vertical_edge_makes_every_track_taller_in_one_step(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 途中をコマンドにすると、取り消しの履歴が伸び縮みの途中で埋まる
        view, harness = _open(make_area, analyzer, _long(tracks=8))
        bar = view.vertical_scroll_bar
        assert bar.isVisible()
        assert isinstance(bar, ZoomScrollBar)
        before = [t.height for t in view.project.timeline.tracks]
        handle = bar.handle_rect()
        edge = QPoint(handle.center().x(), handle.bottom() - 1)
        QTest.mousePress(bar, _LEFT, _NONE, edge)
        QTest.mouseMove(bar, edge - QPoint(0, handle.height() // 3))
        QTest.mouseMove(bar, edge - QPoint(0, handle.height() // 2))
        assert harness.received == []
        QTest.mouseRelease(bar, _LEFT, _NONE, edge - QPoint(0, handle.height() // 2))
        ((command,),) = harness.received
        assert isinstance(command, SetTrackHeights)
        after = [t.height for t in view.project.timeline.tracks]
        assert all(new > old for new, old in zip(after, before, strict=True))

    def test_tracks_that_fit_can_be_made_taller(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 収まっているとバーを隠していたので、ふだんの状態から端で高くできなかった
        view, harness = _open(make_area, analyzer, _long(tracks=2))
        bar = view.vertical_scroll_bar
        assert bar.isVisible()
        before = [t.height for t in view.project.timeline.tracks]
        handle = bar.handle_rect()
        edge = QPoint(handle.center().x(), handle.bottom() - 1)
        _drag_bar(bar, edge, edge - QPoint(0, handle.height() // 2))
        ((command,),) = harness.received
        assert isinstance(command, SetTrackHeights)
        after = [t.height for t in view.project.timeline.tracks]
        assert all(new > old for new, old in zip(after, before, strict=True))

    def test_tracks_that_fit_can_be_made_lower(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # つまみが全体を占めているので、バーの外まで引けないと低くする手が無い
        view, harness = _open(make_area, analyzer, _long(tracks=2))
        bar = view.vertical_scroll_bar
        before = [t.height for t in view.project.timeline.tracks]
        handle = bar.handle_rect()
        edge = QPoint(handle.center().x(), handle.bottom() - 1)
        _drag_bar(bar, edge, edge + QPoint(0, handle.height()))
        ((command,),) = harness.received
        assert isinstance(command, SetTrackHeights)
        after = [t.height for t in view.project.timeline.tracks]
        assert all(new < old for new, old in zip(after, before, strict=True))


# --- 端での自動スクロール ---


class TestEdgeScroll:
    def test_speed_grows_toward_and_past_the_edge(self) -> None:
        # 端の近くで速さが付かないか逆を向くと、引いたまま画面の外へ運べないか、逆へ流れる
        assert edge_speed(500, 100, 900) == 0.0
        near = edge_speed(900 - EDGE_ZONE // 2, 100, 900)
        far = edge_speed(950, 100, 900)
        assert 0 < near < far
        assert edge_speed(100, 100, 900) < 0

    def test_the_playhead_is_followed_off_the_right_edge(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 目盛りで再生ヘッドを右へ運んでも、表示が付いてこなかった（Issue #27）
        view, _ = _open(make_area, analyzer, _long())
        QTest.mousePress(view, _LEFT, _NONE, QPoint(400, 10))
        QTest.mouseMove(view, QPoint(view.width() - 4, 10))
        assert view._edge_scroll.running
        before = view.view_layout.scroll_frame
        start = view.playhead
        for _ in range(10):
            view._edge_scroll.tick()
        assert view.view_layout.scroll_frame > before
        # 押したまま止まっていても、再生ヘッドは表示と一緒に進む
        assert view.playhead > start
        assert view.view_layout.frame_to_x(view.playhead) < view.width()
        QTest.mouseRelease(view, _LEFT, _NONE, QPoint(view.width() - 4, 10))
        assert not view._edge_scroll.running

    def test_the_left_edge_scrolls_back(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 左の端で戻らないと、後ろから前の方へクリップを運ぶのに一度離してスクロールし直す
        view, _ = _open(make_area, analyzer, _long())
        view.horizontal_scroll_bar.setValue(2000)
        before = view.view_layout.scroll_frame
        QTest.mousePress(view, _LEFT, _NONE, QPoint(500, 10))
        QTest.mouseMove(view, QPoint(Metrics.TRACK_HEADER_WIDTH + 2, 10))
        view._edge_scroll.tick()
        assert view.view_layout.scroll_frame < before
        QTest.mouseRelease(view, _LEFT, _NONE, QPoint(Metrics.TRACK_HEADER_WIDTH + 2, 10))

    def test_moving_a_clip_to_the_edge_carries_it_along(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 端で表示が付いてこないと、画面の外の遠い所へクリップを運ぶのに何回も掴み直す
        project = _project(Track(TrackKind.VIDEO, "V1", (_text(0, 50), _text(3000, 10))))
        view, harness = _open(make_area, analyzer, project)
        grab = _clip_point(view, "V1", 25)
        QTest.mousePress(view, _LEFT, _NONE, grab)
        edge = QPoint(view.width() - 3, grab.y())
        QTest.mouseMove(view, edge)
        for _ in range(10):
            view._edge_scroll.tick()
        QTest.mouseRelease(view, _LEFT, _NONE, edge)
        (commands,) = harness.received
        moved = view.project.timeline.tracks[0].clips[0]
        # 端で止まっていた頃は、画面の右端のフレームまでしか運べなかった
        assert moved.timeline_start > view.view_layout.frames_in(view.width())
        assert commands

    def test_it_stops_when_the_mouse_comes_back(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # マウスを戻しても止まらないと、表示が流れ続けて狙った所で離せない
        view, _ = _open(make_area, analyzer, _long())
        QTest.mousePress(view, _LEFT, _NONE, QPoint(400, 10))
        QTest.mouseMove(view, QPoint(view.width() - 4, 10))
        QTest.mouseMove(view, QPoint(500, 10))
        assert not view._edge_scroll.running
        QTest.mouseRelease(view, _LEFT, _NONE, QPoint(500, 10))


# --- キーフレームの印 ---


def _animated(*frames: int) -> AnimatedValue:
    return AnimatedValue(1.0, tuple(Keyframe(frame, 1.0) for frame in frames))


class TestKeyframeMarks:
    def test_every_animated_value_is_collected(self) -> None:
        # 1 か所でも数え漏らすと、その値だけキーフレームを打っても印が出ない
        blur = Effect("blur", {"radius": _animated(20)})
        after = Effect("blur", {"radius": _animated(40)})
        source = TEXT.create()
        source = source.with_param("size", _animated(60))
        clip = _text(
            0,
            100,
            opacity=_animated(0, 10),
            effects=(blur,),
            after_effects=(after,),
            source=source,
        )
        assert keyframe_frames(clip) == (0, 10, 20, 40, 60)

    def test_keyframes_outside_the_clip_are_left_out(self) -> None:
        # トリムで外れた所には描く場所が無い
        clip = _text(0, 30, opacity=_animated(10, 50))
        assert keyframe_frames(clip) == (10,)

    def test_a_diamond_is_drawn_and_brighter_when_selected(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 印が描かれないと、どのクリップのどこに動きがあるのか開いて調べるまで分からない
        project = _project(Track(TrackKind.VIDEO, "V1", (_text(0, 200, opacity=_animated(50)),)))
        view, _ = _open(make_area, analyzer, project)
        point = _mark_point(view, "V1", 50)
        plain = QColor(view.grab().toImage().pixel(point))
        view.select(project.timeline.tracks[0].clips[0].id)
        chosen = QColor(view.grab().toImage().pixel(point))
        assert plain.red() > plain.blue() + 60, "印が描かれていない"
        assert chosen.lightness() > plain.lightness()

    def test_crowded_marks_are_thinned(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 詰まった印を間引かないと、細かく打ったクリップが塗りつぶした帯になり
        # 印を 1 つずつ見分けられない
        from sashimono.ui.timeline.keyframes import keyframe_marks
        from sashimono.ui.timeline.painter import clip_rect_for

        clip = _text(0, 400, opacity=_animated(*range(0, 400, 2)))
        view, _ = _open(make_area, analyzer, _project(Track(TrackKind.VIDEO, "V1", (clip,))))
        band = view.view_layout.bands(view.project.timeline)[0]
        rect = clip_rect_for(clip, band, view.view_layout, view.width())
        assert rect is not None
        marks = keyframe_marks(clip, view.view_layout, rect)
        xs = [centre.x() for _, centre in marks]
        assert len(marks) < 200
        assert all(b - a >= MIN_KEYFRAME_GAP for a, b in pairwise(xs))

    def test_clicking_a_diamond_moves_the_playhead_there(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 印を押しても再生ヘッドが来ないと、キーの値を直すのにタイムラインで位置を合わせ直す
        project = _project(Track(TrackKind.VIDEO, "V1", (_text(0, 200, opacity=_animated(70)),)))
        view, harness = _open(make_area, analyzer, project)
        point = _mark_point(view, "V1", 70) + QPoint(2, 0)
        QTest.mousePress(view, _LEFT, _NONE, point)
        QTest.mouseMove(view, point + QPoint(40, 0))
        QTest.mouseRelease(view, _LEFT, _NONE, point + QPoint(40, 0))
        assert view.playhead == 70
        assert view.selected_clip == project.timeline.tracks[0].clips[0].id
        # 印を押したのはクリップを動かすためではない 動かすと、確かめに押しただけで位置がずれる
        assert harness.received == []

    def test_a_keyframe_set_later_shows_up(
        self, make_area: list[TimelineArea], analyzer: MediaAnalyzer
    ) -> None:
        # 打った直後に印が出ないと、打てたのかどうかが分からない
        project = _project(Track(TrackKind.VIDEO, "V1", (_text(0, 200),)))
        view, _ = _open(make_area, analyzer, project)
        clip = project.timeline.tracks[0].clips[0]
        point = _mark_point(view, "V1", 30)
        empty = QColor(view.grab().toImage().pixel(point))
        command = SetKeyframe(ParamPath.of_clip(clip.id, "opacity"), 30, 0.5)
        view.set_project(command.apply(view.project))
        marked = QColor(view.grab().toImage().pixel(point))
        assert marked != empty


def _mark_point(view: TimelineView, name: str, frame: int) -> QPoint:
    """ひし形の中心 クリップの下端の少し上"""
    top, bottom = _band(view, name)
    clip_bottom = top + 1 + (bottom - top) - 3 - 1
    return QPoint(round(view.view_layout.frame_to_x(frame)), clip_bottom - 4 - 2)
