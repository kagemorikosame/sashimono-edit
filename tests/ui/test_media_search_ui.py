"""素材の意味検索（#326）の人の入口と、本物の窓の口

素材一覧の〔探す…〕で開く窓が、AI の search_media と同じ引き方で置いていない素材の字幕と
場面の説明を引き、選ぶとその所へ行くこと 窓（MainWindow）が場面の説明を作品の外に覚え、
素材 1 本の絵を描けることを見る 確かめ・警告・ファイルの窓が開いたら、その場で落とす
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QTabBar,
)

from sashimono.ai.operations import find_operation
from sashimono.ai.sheet import sheet_layout
from sashimono.core.commands import AddClip, SetTranscript
from sashimono.core.commands.insert import media_alone
from sashimono.core.model import (
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.core.model.transcript import TranscriptSegment
from sashimono.core.search import SceneNote
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import forget_probe, probe_media
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_pool import MediaPoolWidget
from sashimono.ui.media_search import MediaSearchDialog
from tests.conftest import make_clip
from tests.scene_video import write_scene_video


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 探すのも行くのも、どの確かめの窓も出さないはず"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _transcript(*lines: tuple[int, str]) -> Transcript:
    return Transcript(
        segments=tuple(
            TranscriptSegment(start=Fraction(start), end=Fraction(start + 2), text=text)
            for start, text in lines
        )
    )


@pytest.fixture
def placed(video_media: MediaItem) -> MediaItem:
    return video_media.with_transcript(_transcript((4, "編集ソフトを作ります")))


@pytest.fixture
def unplaced(video_media: MediaItem) -> MediaItem:
    """タイムラインに置いていない素材"""
    return replace(
        video_media,
        path=Path("C:/素材/旅行.mp4"),
        id=MediaId(f"{video_media.id}x"),
        transcripts=(),
    ).with_transcript(_transcript((6, "夜はラーメンを食べました")))


@pytest.fixture
def window(
    qt_application: QApplication, placed: MediaItem, unplaced: MediaItem
) -> Iterator[MainWindow]:
    del qt_application
    project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(placed, unplaced))
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = project.with_timeline(replace(project.timeline, tracks=(track,)))
    # 頭の 2 秒を切って 30 フレーム目から置く 素材の 4 秒はタイムラインの 90 フレーム目
    project = AddClip(track.id, make_clip(30, 240, placed, source_in=2)).apply(project)
    created = MainWindow(project, confirm_unsaved=False)
    yield created
    created.close()
    created.deleteLater()


def _search(dialog: MediaSearchDialog, text: str) -> list[str]:
    dialog.field.setText(text)
    dialog.search()
    return [dialog.results.item(row).text() for row in range(dialog.results.count())]


class TestTheDialog:
    def test_the_button_opens_one_window(self, window: MainWindow) -> None:
        # 壊れると、〔探す…〕を押しても何も出ないか、押すたびに窓が増えて編集画面の子に残る
        pool = window.findChild(MediaPoolWidget)
        assert pool is not None
        button = next(b for b in pool.findChildren(QPushButton) if b.text() == "探す…")
        button.click()
        button.click()
        dialogs = window.findChildren(MediaSearchDialog)
        assert len(dialogs) == 1
        assert dialogs[0].isVisible()

    def test_media_off_the_timeline_are_found_with_folded_words(
        self, window: MainWindow, unplaced: MediaItem
    ) -> None:
        # 壊れると、置いていない素材の字幕を人が探せず、AI に頼むしかなくなる（入口で答えが変わる）
        rows = _search(window.show_media_search(), "らーめん")
        assert len(rows) == 1
        assert "旅行.mp4" in rows[0] and "夜はラーメンを食べました" in rows[0]

    def test_scene_notes_are_found_too(self, window: MainWindow, unplaced: MediaItem) -> None:
        # 壊れると、AI が覚えた場面の説明を人の欄では引けず、話していない場面を人が探せない
        window.remember_scene_notes(unplaced, (SceneNote(1.0, 3.0, "屋台の湯気"),))
        rows = _search(window.show_media_search(), "屋台")
        assert len(rows) == 1 and "[場面]" in rows[0]

    def test_the_project_is_read_again_for_each_search(
        self, window: MainWindow, placed: MediaItem
    ) -> None:
        # 壊れると、窓を開いたまま字幕を直しても、直す前の字幕で探し続ける
        dialog = window.show_media_search()
        assert _search(dialog, "カレー") == []
        window.execute_all([SetTranscript(placed.id, _transcript((4, "カレーを作ります")))], "字幕")
        assert len(_search(dialog, "カレー")) == 1

    def test_nothing_found_says_how_to_widen(self, window: MainWindow) -> None:
        # 壊れると、見つからないときに何も出ず、言い換えを並べられることが分からない
        dialog = window.show_media_search()
        assert _search(dialog, "存在しない言葉") == []
        assert "空白で区切って" in dialog.summary


class TestGoingThere:
    def test_a_subtitle_goes_to_the_clip_that_plays_its_language(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 壊れると、2 か国語の素材で別の言語の音のクリップへ再生位置が飛び、探した言葉が
        # 聞こえない所で止まる（窓が当たりの音声の番号を渡さない）
        del qt_application
        second = replace(video_media.audio_streams[0], index=2)
        media = replace(
            video_media, audio_streams=(video_media.audio_streams[0], second)
        ).with_transcript(_transcript((4, "英語で話します")), 2)
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
        tracks = tuple(
            Track(
                kind=TrackKind.AUDIO,
                name=f"A{stream}",
                clips=(replace(make_clip(start, 300, media), stream_index=stream),),
            )
            for start, stream in ((0, 1), (600, 2))
        )
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        window = MainWindow(project, confirm_unsaved=False)
        try:
            dialog = window.show_media_search()
            assert _search(dialog, "英語")
            dialog.results.itemActivated.emit(dialog.results.item(0))
            assert window.playhead == 600 + 120
        finally:
            window.close()
            window.deleteLater()

    def test_a_placed_media_moves_the_playhead(self, window: MainWindow) -> None:
        # 壊れると、見つけた行を選んでも再生位置が動かず、人がタイムラインで探し直す
        dialog = window.show_media_search()
        _search(dialog, "編集")
        before = window.document.project
        dialog.results.itemActivated.emit(dialog.results.item(0))
        assert window.playhead == 90
        assert window.document.project is before

    def test_an_unplaced_media_is_shown_in_the_list_without_placing_it(
        self, window: MainWindow, unplaced: MediaItem
    ) -> None:
        # 壊れると、探しただけで素材がタイムラインに置かれて作品が変わるか、どの素材か分からない
        dialog = window.show_media_search()
        _search(dialog, "ラーメン")
        before = window.document.project
        window.seek(5)
        dialog.results.itemActivated.emit(dialog.results.item(0))
        assert window.document.project is before
        assert window.playhead == 5
        pool = window.findChild(MediaPoolWidget)
        assert pool is not None and pool.selected_media_id() == unplaced.id
        assert "置いていません" in window.statusBar().currentMessage()

    def test_the_media_list_comes_to_the_front_for_an_unplaced_media(
        self, window: MainWindow
    ) -> None:
        # 壊れると、字幕パネルを前に出していたり素材一覧を閉じていたりしたとき、見えない一覧で
        # 行を選ぶだけになり、案内の〔タイムラインへ〕でどの素材を置けばよいか分からない
        pool = window.findChild(MediaPoolWidget)
        assert pool is not None
        dock = pool.parentWidget()
        assert isinstance(dock, QDockWidget)
        dialog = window.show_media_search()
        _search(dialog, "ラーメン")
        window.show_subtitles()
        assert _front_tab(window, dock) == "字幕"
        dialog.results.itemActivated.emit(dialog.results.item(0))
        assert _front_tab(window, dock) == "メディア"
        dock.close()
        dialog.results.itemActivated.emit(dialog.results.item(0))
        assert not dock.isHidden()


def _front_tab(window: MainWindow, dock: QDockWidget) -> str:
    """``dock`` を含むタブの並びで、前に出ているタブの名前"""
    for bar in window.findChildren(QTabBar):
        names = [bar.tabText(index) for index in range(bar.count())]
        if dock.windowTitle() in names:
            return bar.tabText(bar.currentIndex())
    raise AssertionError("素材一覧がタブで重なっていない")


class TestTheHost:
    def test_scene_notes_outlive_the_window(self, window: MainWindow, unplaced: MediaItem) -> None:
        # 壊れると、アプリを開き直すたびに説明が消え、絵を送って作り直す費用が毎回かかる
        notes = (SceneNote(0.0, 4.0, "海辺"),)
        window.remember_scene_notes(unplaced, notes)
        again = MainWindow(Project.create(media=(unplaced,)), confirm_unsaved=False)
        try:
            assert again.scene_notes(unplaced) == notes
        finally:
            again.close()
            again.deleteLater()

    def test_the_sheet_holds_the_last_picture_of_an_old_media(
        self, window: MainWindow, unplaced: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、映像の終わりを持たない古い作品の素材（映像 6 秒・音 10 秒）で、映像より
        # 後ろの場面の絵が黒くなり、AI が黒い絵に説明を付ける（#402 の黒いコマと同じ）
        assert unplaced.video_streams[0].end_time is None
        stream = replace(unplaced.video_streams[0], end_time=Fraction(6))
        fresh = replace(unplaced, video_streams=(stream,))
        monkeypatch.setattr(main_window_module, "_probe_or_none", lambda _path: fresh)
        drawn: list[Project] = []
        original = media_alone

        def recorded(settings: ProjectSettings, media: MediaItem) -> Project:
            project = original(settings, media)
            drawn.append(project)
            return project

        monkeypatch.setattr(main_window_module, "media_alone", recorded)
        window.start_media_sheet(unplaced, sheet_layout(2, cell_width=240, frame_size=(320, 180)))
        (project,) = drawn
        pictures = [
            clip
            for track in project.timeline.tracks
            if track.kind is TrackKind.VIDEO
            for clip in track.clips
        ]
        assert pictures
        assert all(clip.hold_at is not None and clip.hold_at < 6 for clip in pictures)

    def test_remembering_through_the_tool_leaves_the_history_alone(
        self, window: MainWindow, unplaced: MediaItem
    ) -> None:
        # 壊れると、説明を覚えるたびに取り消しの段が積まれ、人の Ctrl+Z が説明で埋まる
        labels = window.document.history_labels
        operation = find_operation("remember_scene_descriptions")
        assert operation is not None
        operation(
            window,
            {
                "media_id": str(unplaced.id),
                "scenes": [{"from_seconds": 0, "to_seconds": 4, "text": "海辺"}],
            },
        )
        assert window.document.history_labels == labels
        assert window.scene_notes(unplaced) == (SceneNote(0.0, 4.0, "海辺"),)


@pytest.mark.usefixtures("gpu")
def test_the_sheet_draws_the_media_off_the_timeline(
    qt_application: QApplication, tmp_path: Path
) -> None:
    """壊れると、置いていない素材の場面の絵を AI が見られないか、頼んだ秒と違う絵で説明を付ける"""
    del qt_application
    path = write_scene_video(tmp_path / "場面.mp4")
    forget_probe(path)
    media = probe_media(path)
    # 作品は素材と違う大きさ 素材の絵が作品の枠に合わせて縮むと、升の真ん中の色が地の色になる
    settings = ProjectSettings(width=1920, height=1080, frame_rate=FrameRate(30))
    window = MainWindow(Project.create(settings, media=(media,)), confirm_unsaved=False)
    try:
        layout = sheet_layout(2, cell_width=240, frame_size=(320, 180))
        canvas = window.start_media_sheet(media, layout)
        # 3 秒は暗い青の地、5 秒は暗い赤の地（tests/scene_video.py の表）
        canvas.draw(0, 90, "1) 2〜4 秒")
        canvas.draw(1, 150, "2) 4〜6 秒")
        image = QImage.fromData(canvas.png())
        corners = []
        for index in range(2):
            x, y = layout.cell(index)
            corners.append(image.pixelColor(x + 8, y + 8))
        assert _near(corners[0], QColor(40, 60, 140)), corners[0].name()
        assert _near(corners[1], QColor(140, 60, 40)), corners[1].name()
        # 描いただけで作品に素材が置かれていない
        assert window.document.project.duration == 0
    finally:
        window.close()
        window.deleteLater()


def _near(color: QColor, want: QColor) -> bool:
    return all(
        abs(a - b) < 30
        for a, b in zip(
            (color.red(), color.green(), color.blue()),
            (want.red(), want.green(), want.blue()),
            strict=True,
        )
    )


def test_the_pool_button_asks_for_the_window(qt_application: QApplication) -> None:
    """壊れると、〔探す…〕を押しても窓を開く頼みが編集画面へ届かない"""
    del qt_application
    pool = MediaPoolWidget(Project.create())
    asked: list[bool] = []
    pool.search_requested.connect(lambda: asked.append(True))
    button = next(b for b in pool.findChildren(QPushButton) if b.text() == "探す…")
    assert button.toolTip() == "素材の字幕と場面の説明から、言葉を含む所を探す"
    button.click()
    assert asked == [True]
    pool.deleteLater()
