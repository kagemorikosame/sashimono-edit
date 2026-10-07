"""テキストの設定のスタイルの欄（#248）

フォントの欄の下に、ファミリの中のスタイル（Yu Gothic UI の Light など）を選ぶ欄を出す
スタイルを選んでいる間は太字と斜体を灰色にし、AviUtl2 の組み方ではスタイルの欄を
灰色にする どちらも隠さずに理由を添える（隠すと「太字が消えた」と見える）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from sashimono.core.commands import AddClip, SetTranscript, burn_subtitles
from sashimono.core.model import Clip, MediaItem, Project, Track, TrackKind, Transcript
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.widgets import FONT_STYLE_DEFAULT_LABEL, FontStyleEditor
from sashimono.ui.main_window import MainWindow
from tests.conftest import make_clip

FAMILY = "Yu Gothic UI"
LIGHT = "Light"


def _project(**params: object) -> Project:
    base = Project.create()
    text = Clip(timeline_start=0, duration=30, source=TEXT.create(**params))  # type: ignore[arg-type]
    tracks = (Track(TrackKind.VIDEO, "V1", (text,)), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _flush() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


@pytest.fixture
def open_window(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    """作った窓を最後に閉じる 窓は試験ごとに値を変えて作る"""
    del qt_application
    made: list[MainWindow] = []
    yield made
    for window in made:
        window.close()


def _window(made: list[MainWindow], **params: object) -> MainWindow:
    window = MainWindow(_project(font=FAMILY, **params), confirm_unsaved=False)
    made.append(window)
    window._timeline.select(window.document.project.timeline.tracks[0].clips[0].id)
    _flush()
    return window


def _param(window: MainWindow, name: str) -> object:
    clip = window.document.project.timeline.tracks[0].clips[0]
    assert clip.source is not None
    return clip.source.params.get(name)


def _style_box(window: MainWindow) -> QComboBox:
    editor = window._inspector._editors[("source", "font_style")]
    assert isinstance(editor, FontStyleEditor)
    box = editor.findChild(QComboBox)
    assert box is not None
    return box


def _notes(window: MainWindow) -> list[str]:
    return [label.text() for label in window._inspector.findChildren(QLabel)]


@pytest.fixture(autouse=True)
def _needs_family(qt_application: QApplication) -> None:
    del qt_application
    if LIGHT not in QFontDatabase.styles(FAMILY):
        pytest.skip(f"{FAMILY} の {LIGHT} が入っていない")


class TestStyleField:
    def test_it_sits_right_under_the_font(self, open_window: list[MainWindow]) -> None:
        # 定義では末尾にある 並びのまま出すと、フォントから 20 行ほど離れた底に出る
        window = _window(open_window)
        names = [name for owner, name in window._inspector._editors if owner == "source"]
        assert names[names.index("font") + 1] == "font_style"

    def test_it_lists_the_family_styles_after_the_default(
        self, open_window: list[MainWindow]
    ) -> None:
        box = _style_box(_window(open_window))
        items = [box.itemText(index) for index in range(box.count())]
        assert items[0] == FONT_STYLE_DEFAULT_LABEL
        assert set(QFontDatabase.styles(FAMILY)) <= set(items[1:])
        assert box.currentIndex() == 0

    def test_choosing_a_style_saves_its_name(self, open_window: list[MainWindow]) -> None:
        window = _window(open_window)
        box = _style_box(window)
        box.setCurrentIndex(box.findData(LIGHT))
        _flush()
        assert _param(window, "font_style") == LIGHT
        # 戻すと空（今までの描き方）になる
        box = _style_box(window)
        box.setCurrentIndex(0)
        _flush()
        assert _param(window, "font_style") == ""

    def test_changing_the_font_relists_the_styles(self, open_window: list[MainWindow]) -> None:
        # フォントを替えても欄は作り直さない 一覧だけが前のファミリのまま残っていた
        other = "Segoe UI"
        if "Semibold" not in QFontDatabase.styles(other):
            pytest.skip(f"{other} の Semibold が入っていない")
        window = _window(open_window)
        window._inspector._editors[("source", "font")].value_changed.emit(other)
        _flush()
        assert _style_box(window).findData("Semibold") > 0

    def test_a_style_missing_from_the_family_stays_visible(
        self, open_window: list[MainWindow]
    ) -> None:
        # 消すと、ほかの機械で作ったプロジェクトを開いただけで選んでいたスタイルが分からない
        box = _style_box(_window(open_window, font_style="無いスタイル"))
        assert "無いスタイル" in box.currentText()
        assert box.currentData() == "無いスタイル"


class TestGreyedFields:
    def test_bold_and_italic_grey_out_while_a_style_is_chosen(
        self, open_window: list[MainWindow]
    ) -> None:
        window = _window(open_window, font_style=LIGHT, bold=True)
        editors = window._inspector._editors
        assert not editors[("source", "bold")].isEnabled()
        assert not editors[("source", "italic")].isEnabled()
        assert editors[("source", "font_style")].isEnabled()
        # 理由は 2 つの欄に 1 つだけ添える
        reason = editors[("source", "bold")].toolTip()
        assert reason and _notes(window).count(reason) == 1
        # 値は消さない スタイルを既定に戻すと前の太字で描く
        assert _param(window, "bold") is True

    def test_choosing_a_style_greys_them_at_once(self, open_window: list[MainWindow]) -> None:
        # 欄の構成に灰色を入れていないと、作り直さずに押せるまま残る
        window = _window(open_window)
        assert window._inspector._editors[("source", "bold")].isEnabled()
        box = _style_box(window)
        box.setCurrentIndex(box.findData(LIGHT))
        _flush()
        assert not window._inspector._editors[("source", "bold")].isEnabled()

    def test_the_aviutl_layout_greys_the_style(self, open_window: list[MainWindow]) -> None:
        window = _window(open_window, layout="aviutl", font_style=LIGHT)
        editors = window._inspector._editors
        assert not editors[("source", "font_style")].isEnabled()
        assert editors[("source", "font_style")].toolTip() in _notes(window)
        # AviUtl2 の組み方は太字を自前で太らせるので、太字は効く
        assert editors[("source", "bold")].isEnabled()

    def test_vertical_text_keeps_the_style_in_the_aviutl_layout(self) -> None:
        # 縦書きは組み方に関わらず標準の組み方で描くので、スタイルが効く
        params = TEXT.create(layout="aviutl", vertical=True, font_style=LIGHT).params
        assert "font_style" not in TEXT.locked_reasons(params)
        assert "bold" in TEXT.locked_reasons(params)

    def test_no_style_greys_nothing(self) -> None:
        assert TEXT.locked_reasons(TEXT.create().params) == {}


class TestSubtitles:
    def test_burned_subtitles_keep_the_template_style(
        self, project: Project, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 字幕の焼き込みは、選んでいるテキストを写して本文だけ差し替える スタイルも写る
        with_transcript = SetTranscript(video_media.id, transcript).apply(project)
        track = with_transcript.timeline.tracks[0]
        placed = AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)
        template = Clip(
            timeline_start=0, duration=30, source=TEXT.create(font=FAMILY, font_style=LIGHT)
        )
        burned = placed
        for command in burn_subtitles(placed, template):
            burned = command.apply(burned)
        sources = [c.source for c in burned.timeline.tracks[-1].clips]
        assert sources and all(s is not None for s in sources)
        assert {s.params["font_style"] for s in sources if s is not None} == {LIGHT}
