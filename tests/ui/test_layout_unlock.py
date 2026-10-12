"""AviUtl2 の組み方で灰色になった欄に、組み方を標準へ切り替えるボタンを添える（#301）

スタイルの欄が灰色で「既定」とだけ出ていて、スタイルに対応していないか、壊れていると
受け取られた 理由の添え書きはあったが、原因の組み方の欄が 20 行ほど下にあって気付かれなかった
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QMessageBox, QPushButton

from sashimono.core.commands import GroupClips
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    GeneratedSource,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects.sources import AVIUTL_NO_STYLE, AVIUTL_NO_WRAP, TEXT, TO_NATIVE_LAYOUT
from sashimono.engine.sources import render_source_framed
from sashimono.ui.inspector.widgets import FontStyleEditor
from sashimono.ui.main_window import MainWindow

UNAVAILABLE = "（AviUtl2 の組み方では選べません）"


@pytest.fixture(autouse=True)
def no_modal_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """小窓を開かせない 開くと試験が止まったまま返らない 出たらその場で落とす"""

    def refuse(*args: object, **_kwargs: object) -> object:
        pytest.fail(f"思わぬ窓が出た: {args}")

    for name in ("information", "warning", "question", "critical", "exec"):
        monkeypatch.setattr(QMessageBox, name, refuse)
    monkeypatch.setattr(QDialog, "exec", refuse)


def _project(*sources: GeneratedSource) -> Project:
    base = Project.create()
    clips = tuple(
        Clip(timeline_start=40 * index, duration=30, source=source)
        for index, source in enumerate(sources)
    )
    tracks = (Track(TrackKind.VIDEO, "V1", clips), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _flush() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


@pytest.fixture
def open_window(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    """作った窓を最後に閉じる 窓は試験ごとに中身を変えて作る"""
    del qt_application
    made: list[MainWindow] = []
    yield made
    for window in made:
        window.close()


def _window(made: list[MainWindow], *sources: GeneratedSource) -> MainWindow:
    window = MainWindow(_project(*sources), confirm_unsaved=False)
    made.append(window)
    window._timeline.select(_ids(window)[0])
    _flush()
    return window


def _ids(window: MainWindow) -> list[ClipId]:
    return [clip.id for clip in window.document.project.timeline.tracks[0].clips]


def _param(window: MainWindow, name: str, index: int = 0) -> object:
    clip = window.document.project.timeline.tracks[0].clips[index]
    assert clip.source is not None
    return clip.source.params.get(name)


def _buttons(window: MainWindow) -> list[QPushButton]:
    return [
        button
        for button in window._inspector.findChildren(QPushButton)
        if button.objectName() == "unlock_button"
    ]


def _notes(window: MainWindow) -> list[str]:
    return [label.text() for label in window._inspector.findChildren(QLabel)]


def _style_box(window: MainWindow) -> QComboBox:
    editor = window._inspector._editors[("source", "font_style")]
    assert isinstance(editor, FontStyleEditor)
    box = editor.findChild(QComboBox)
    assert box is not None
    return box


def _press(window: MainWindow) -> None:
    (button,) = _buttons(window)
    button.click()
    _flush()


class TestButton:
    def test_the_aviutl_layout_shows_one_switch_under_the_style(
        self, open_window: list[MainWindow]
    ) -> None:
        # ボタンが無いと、理由を読んでも 20 行下の組み方の欄を探しに行くことになる
        # 折り返しの幅も同じ理由で灰色だが、ボタンを 2 つ出すと別々の操作に見える
        window = _window(open_window, TEXT.create(layout="aviutl"))
        buttons = _buttons(window)
        assert [b.text() for b in buttons] == [TO_NATIVE_LAYOUT.button]
        assert AVIUTL_NO_STYLE in _notes(window)
        # 折り返しの幅の添え書きは、ボタンのある欄を指す
        assert f"{AVIUTL_NO_WRAP} 切り替えのボタンは「スタイル」の欄の下にあります" in _notes(
            window
        )

    def test_the_button_tells_how_far_the_letters_move(self, open_window: list[MainWindow]) -> None:
        # 押す前に、字が動くことと、どのくらい動くかを読めないと、押してから驚く
        window = _window(open_window, TEXT.create(layout="aviutl"))
        (button,) = _buttons(window)
        assert button.toolTip() == TO_NATIVE_LAYOUT.detail
        assert "画素" in button.toolTip() and "取り消し" in button.toolTip()

    def test_pressing_it_frees_the_style_and_one_undo_brings_it_back(
        self, open_window: list[MainWindow]
    ) -> None:
        # 押しても組み方が変わらなければ、ボタンを出した意味が無い
        # 取り消しが 2 段に分かれると、1 回戻しただけで横の基準だけが変わったまま残る
        window = _window(open_window, TEXT.create(layout="aviutl", align="left"))
        _press(window)
        assert _param(window, "layout") == "native"
        assert window._inspector._editors[("source", "font_style")].isEnabled()
        assert window._inspector._editors[("source", "wrap_width")].isEnabled()
        assert _buttons(window) == []
        assert window.document.undo_label == TO_NATIVE_LAYOUT.button
        window.undo()
        _flush()
        assert _param(window, "layout") == "aviutl"
        assert _param(window, "anchor") == "center"
        assert not window._inspector._editors[("source", "font_style")].isEnabled()

    def test_the_anchor_follows_the_alignment(self, open_window: list[MainWindow]) -> None:
        # 組み方だけを変えると、左寄せの字が塊の幅の半分だけ右へ跳ぶ
        window = _window(open_window, TEXT.create(layout="aviutl", align="right"))
        _press(window)
        assert _param(window, "anchor") == "right"

    def test_the_native_layout_has_no_button(self, open_window: list[MainWindow]) -> None:
        # 使える欄の下にボタンが出ると、押す必要の無い切り替えを勧め、押すと横の基準まで変わる
        window = _window(open_window, TEXT.create())
        assert _buttons(window) == []

    def test_vertical_text_is_not_greyed_and_has_no_button(
        self, open_window: list[MainWindow]
    ) -> None:
        # 縦書きは AviUtl2 の組み方でも標準で描くので、スタイルも折り返しも効く
        # ボタンを出すと、効いている欄を使えないと思わせる
        window = _window(open_window, TEXT.create(layout="aviutl", vertical=True))
        assert window._inspector._editors[("source", "font_style")].isEnabled()
        assert _buttons(window) == []
        assert UNAVAILABLE not in _style_box(window).currentText()


class TestStyleField:
    def test_the_greyed_field_says_why_instead_of_default(
        self, open_window: list[MainWindow]
    ) -> None:
        # 「既定」とだけ出すと、スタイルに対応していないか壊れていると読まれる
        window = _window(open_window, TEXT.create(layout="aviutl"))
        box = _style_box(window)
        assert [box.itemText(i) for i in range(box.count())] == [UNAVAILABLE]

    def test_a_chosen_style_stays_readable(self, open_window: list[MainWindow]) -> None:
        # 選んでいたスタイルを消すと、標準へ戻したときに何を選んでいたかが分からない
        window = _window(open_window, TEXT.create(layout="aviutl", font_style="Light"))
        box = _style_box(window)
        assert box.currentText() == f"Light{UNAVAILABLE}"
        assert box.currentData() == "Light"
        assert _param(window, "font_style") == "Light"

    def test_after_the_switch_the_list_comes_back(self, open_window: list[MainWindow]) -> None:
        # 切り替えた後も理由の文が残ると、使えるのに選べないように見える
        window = _window(open_window, TEXT.create(layout="aviutl"))
        _press(window)
        box = _style_box(window)
        assert box.itemText(0) != UNAVAILABLE


class TestManyClips:
    def test_every_selected_aviutl_text_switches_in_one_step(
        self, open_window: list[MainWindow]
    ) -> None:
        # 主のクリップだけが切り替わると、まとめて選んだ字幕の半分だけ見た目が変わる
        # 横の基準を主のクリップから写すと、右寄せのクリップが左寄せの位置へ跳ぶ
        # 縦書きは灰色になっていないので触らない（選んでいただけで設定が変わる）
        window = _window(
            open_window,
            TEXT.create(layout="aviutl", align="left"),
            TEXT.create(layout="aviutl", align="right"),
            TEXT.create(layout="aviutl", vertical=True),
            TEXT.create(),
        )
        # 主のクリップは最後に選んだ物 灰色の 1 本目を主にする（主が標準ならボタンが出ない）
        ids = _ids(window)
        window._timeline.set_selection([*ids[1:], ids[0]])
        _flush()
        assert window._inspector._clip_id == ids[0]
        _press(window)
        assert [_param(window, "layout", i) for i in range(4)] == [
            "native",
            "native",
            "aviutl",
            "native",
        ]
        assert [_param(window, "anchor", i) for i in range(2)] == ["left", "right"]
        assert window.document.undo_label == f"{TO_NATIVE_LAYOUT.button}（2 本）"
        window.undo()
        _flush()
        assert [_param(window, "layout", i) for i in range(3)] == ["aviutl"] * 3
        assert [_param(window, "anchor", i) for i in range(2)] == ["center", "center"]

    def test_grouped_texts_chosen_together_all_switch(self, open_window: list[MainWindow]) -> None:
        # 2 本を選んでからグループ化した字幕で押すと、主の 1 本だけが標準になり、もう 1 本は
        # 灰色のまま残っていた 押したのに半分しか切り替わらず、字の組み方が食い違う
        project = _project(
            TEXT.create(layout="aviutl", align="left"),
            TEXT.create(layout="aviutl", align="right"),
        )
        ids = [clip.id for clip in project.timeline.tracks[0].clips]
        window = MainWindow(GroupClips(tuple(ids)).apply(project), confirm_unsaved=False)
        open_window.append(window)
        window._timeline.set_selection([ids[1], ids[0]])
        _flush()
        assert window._inspector._clip_id == ids[0]
        _press(window)
        assert [_param(window, "layout", i) for i in range(2)] == ["native", "native"]
        assert [_param(window, "anchor", i) for i in range(2)] == ["left", "right"]
        window.undo()
        _flush()
        assert [_param(window, "layout", i) for i in range(2)] == ["aviutl", "aviutl"]


class TestPosition:
    @pytest.mark.parametrize("align", ["left", "center", "right"])
    def test_a_plain_line_stays_within_two_pixels(
        self, qt_application: QApplication, align: str
    ) -> None:
        # ボタンの説明に書いた幅（太字でない 1 行は 2 画素以内）を守る 外れると説明が嘘になる
        del qt_application
        if not QFont("Yu Gothic UI").exactMatch():
            pytest.skip("Yu Gothic UI が入っていない")
        before = TEXT.create(
            text="Sashimono 編集",
            font="Yu Gothic UI",
            size=AnimatedValue(100.0),
            layout="aviutl",
            align=align,
        )
        after = replace(before, params={**before.params, **TO_NATIVE_LAYOUT.switch(before.params)})
        assert after.params["layout"] == "native"
        boxes = [_ink(source) for source in (before, after)]
        assert max(abs(a - b) for a, b in zip(*boxes, strict=True)) <= 2


def _ink(source: GeneratedSource) -> tuple[int, int, int, int]:
    image, _ = render_source_framed(source, 1920, 1080)
    assert image is not None
    ys, xs = np.nonzero(image[..., 3] > 8)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
