"""編集画面でテキストの端をドラッグして伸ばす（#309）

実際の編集画面（MainWindow）を組み、〔テキストを追加〕で置いたテキストの頭の端を掴んで
前へ引く 伸ばした量は TrimClip と同じ決まりで止まり（タイムラインの先頭・前のクリップの
終わり・素材の先頭）、止まった所はドラッグの最中の枠でも見える 終わりの端を後ろへ引いた
ときも、後ろのクリップの頭で止まる 窓は表示しない

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.core.commands import MoveClip
from sashimono.core.io import load_project
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackId,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui.main_window import MainWindow
from sashimono.ui.timeline import TimelineView
from tests.conftest import make_clip

_LEFT = Qt.MouseButton.LeftButton
_NONE = Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """尋ねる窓・知らせる窓・ファイルを選ぶ窓が開いたらその場で落とす

    開いたまま待つと、試験は誰も押さないボタンを待って止まる
    """

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, popped)


def _window(project: Project | None = None, path: Path | None = None) -> MainWindow:
    created = MainWindow(
        project or Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30))),
        path=path,
        confirm_unsaved=False,
    )
    created.resize(1400, 800)
    return created


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _window()
    yield created
    created.close()


def _place_text(window: MainWindow, frame: int, beside: Clip | None = None) -> Clip:
    """メニューと同じ入口でテキストを置き、``frame`` へ動かす ``beside`` があればその隣へ

    再生ヘッドはタイムラインの終わりより先へ行けないので、空のタイムラインでは頭（0）に
    しか置けない 置いてから動かすのは、本人が置いたテキストを引いて動かすのと同じ
    """
    before = {clip.id for clip in _texts(window)}
    window._seek(0)
    window.add_text()
    (placed,) = [clip for clip in _texts(window) if clip.id not in before]
    track = _track_id(window, beside) if beside is not None else None
    assert window.execute(MoveClip(placed.id, frame, track))
    return _now(window, placed)


def _texts(window: MainWindow) -> list[Clip]:
    return [
        clip
        for track in window.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == "text"
    ]


def _now(window: MainWindow, clip: Clip) -> Clip:
    located = window.document.project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[1]


def _track_id(window: MainWindow, clip: Clip) -> TrackId:
    located = window.document.project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[0].id


def _point(view: TimelineView, clip: Clip, frame: int) -> QPoint:
    """``clip`` のトラックの帯の真ん中で、``frame`` の所"""
    located = view.project.timeline.locate_clip(clip.id)
    assert located is not None
    band = next(
        b for b in view.view_layout.bands(view.project.timeline) if b.track.id == located[0].id
    )
    return QPoint(int(view.view_layout.frame_to_x(frame)), band.top + band.height // 2)


def _drag_head(window: MainWindow, clip: Clip, to_frame: int) -> None:
    """頭の端を掴んで ``to_frame`` まで引いて離す 途中を 1 度通すのは本人の手と同じ"""
    view = window._timeline
    start = _point(view, clip, clip.timeline_start)
    end = _point(view, clip, to_frame)
    QTest.mousePress(view, _LEFT, _NONE, start)
    QTest.mouseMove(view, QPoint((start.x() + end.x()) // 2, start.y()))
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, _LEFT, _NONE, end)


class TestDraggingTheHead:
    def test_a_new_text_extends_forward(self, window: MainWindow) -> None:
        # 置いたテキストの頭を前へ引くと、離したときに失敗して何も変わらない
        text = _place_text(window, 100)
        _drag_head(window, text, 70)
        extended = _now(window, text)
        assert (extended.timeline_start, extended.timeline_end) == (70, text.timeline_end)
        assert extended.source_in == 0

    def test_one_undo_brings_it_back(self, window: MainWindow) -> None:
        # 1 回で戻らないと、伸ばす前に戻したつもりでテキストを置いた手まで消す
        text = _place_text(window, 100)
        before = window.document.project
        _drag_head(window, text, 70)
        assert _now(window, text).timeline_start == 70
        window.undo()
        assert window.document.project == before

    def test_it_stops_at_the_start_of_the_timeline(self, window: MainWindow) -> None:
        # 0 の手前まで引くと、離したときに失敗して何も伸びない（0 で止まってほしい）
        text = _place_text(window, 30)
        told: list[str] = []
        window._timeline.status_message.connect(told.append)
        _drag_head(window, text, -40)
        assert _now(window, text).timeline_start == 0
        # ヘッダの上まで引いて 0 で止まったときも、止まった理由を出す
        assert any("タイムラインの先頭" in message for message in told)

    def test_the_tail_stops_at_the_text_after(self, window: MainWindow) -> None:
        # 後ろのテキストに重なる所まで引くと、離したときに重なりで失敗して何も伸びない
        first = _place_text(window, 0)
        second = _place_text(window, first.timeline_end + 30, beside=first)
        view = window._timeline
        told: list[str] = []
        view.status_message.connect(told.append)
        # 終わりの位置はクリップの外（含まない） 1 フレーム内側の端を掴む
        start = _point(view, first, first.timeline_end - 1)
        end = _point(view, first, second.timeline_start + 40)
        QTest.mousePress(view, _LEFT, _NONE, start)
        QTest.mouseMove(view, _point(view, first, first.timeline_end + 10))
        QTest.mouseMove(view, end)
        assert view._drag.preview_tail_delta == 30
        QTest.mouseRelease(view, _LEFT, _NONE, end)
        assert _now(window, first).timeline_end == second.timeline_start
        assert any("後ろのクリップの頭" in message for message in told)

    def test_it_stops_at_the_end_of_the_text_before(self, window: MainWindow) -> None:
        # 前のテキストに重なる所まで引くと、離したときに重なりで失敗して何も伸びない
        first = _place_text(window, 0)
        second = _place_text(window, first.timeline_end + 60, beside=first)
        # 別のトラックへ置かれると、前のクリップで止まる所を確かめられない
        assert _track_id(window, first) == _track_id(window, second)
        _drag_head(window, second, first.timeline_end - 30)
        assert _now(window, second).timeline_start == first.timeline_end

    def test_the_stop_shows_while_dragging(self, window: MainWindow) -> None:
        # 離すまで止まる所と理由が見えないと、どこまで伸びるか分からず何度も引き直す
        first = _place_text(window, 0)
        text = _place_text(window, first.timeline_end + 30, beside=first)
        view = window._timeline
        start = _point(view, text, text.timeline_start)
        far = _point(view, text, first.timeline_end - 40)
        told: list[str] = []
        view.status_message.connect(told.append)
        QTest.mousePress(view, _LEFT, _NONE, start)
        QTest.mouseMove(view, _point(view, text, text.timeline_start - 10))
        QTest.mouseMove(view, far)
        try:
            # 枠は前のテキストの終わりで止まる（30 フレーム前まで）
            assert view._drag.preview_head_delta == -30
            assert any("前のクリップの終わり" in message for message in told)
        finally:
            QTest.mouseRelease(view, _LEFT, _NONE, far)

    def test_a_video_stops_at_the_start_of_its_media(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 素材の前まで引けて見えると、離したときに失敗して何も伸びない
        del qt_application
        base = Project.create(
            ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)), media=(video_media,)
        )
        clip = make_clip(100, 60, video_media, source_in=1)
        track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
        project = base.with_timeline(replace(base.timeline, tracks=(track,)))
        created = _window(project)
        try:
            _drag_head(created, clip, 20)
            assert _now(created, clip).timeline_start == 70
        finally:
            created.close()


class TestSavingAndOpening:
    def test_it_is_the_same_after_opening_again(
        self, window: MainWindow, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 開き直すと位置が戻る・source_in が変わると、保存した作品のテロップが出る時刻がずれる
        del qt_application
        text = _place_text(window, 100)
        _drag_head(window, text, 70)
        extended = _now(window, text)
        path = tmp_path / "伸ばした.sme"
        window._path = path
        assert window.save_project()

        reopened = _window(load_project(path), path)
        try:
            assert _now(reopened, text) == extended
            # 開き直した後も同じ決まりで前へ伸び、取り消し 1 回で戻る
            before = reopened.document.project
            _drag_head(reopened, extended, 40)
            assert _now(reopened, text).timeline_start == 40
            reopened.undo()
            assert reopened.document.project == before
        finally:
            reopened.close()
