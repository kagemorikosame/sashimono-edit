"""設定パネルの欄を続けて操作できるか（Issue #251）

欄に 1 文字打つたびに値が確定し、コマンドを通って設定パネルへ新しいプロジェクトが戻る
戻るたびに中身を全部作り直していたので、打っていた欄が消えてフォーカスが外れ、
2 文字目からはウィンドウ本体（S の分割・スペースの再生）へ届いていた

ここでは実際にキーを打ち、``deleteLater`` を片付けたあと（実機の次の打鍵の前と同じ）も
同じ欄にフォーカスと本文が残るかを見る ``setPlainText`` で値を入れるだけの試験では、
1 文字ずつ確定する道を通らないので気付けなかった
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QFocusEvent, QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QToolButton,
    QWidget,
)

from sashimono.core.commands import AddEffect, ParamPath, SetParam
from sashimono.core.model import AnimatedValue, Clip, ClipId, Project, Track, TrackKind
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector import widgets as widgets_module
from sashimono.ui.inspector.panel import InspectorPanel, KeyframeControls
from sashimono.ui.inspector.widgets import TextEditor, TrackEditor
from sashimono.ui.main_window import MainWindow


def _project(clips: int = 1) -> Project:
    base = Project.create()
    texts = tuple(
        Clip(timeline_start=40 * index, duration=30, source=TEXT.create()) for index in range(clips)
    )
    tracks = (Track(TrackKind.VIDEO, "V1", texts), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _clip_ids(window: MainWindow) -> list[ClipId]:
    return [clip.id for clip in window.document.project.timeline.tracks[0].clips]


def _source(window: MainWindow, clip_id: ClipId, name: str) -> object:
    located = window.document.project.timeline.locate_clip(clip_id)
    assert located is not None and located[1].source is not None
    return located[1].source.params.get(name)


def _flush() -> None:
    """``deleteLater`` を片付ける 実機では次のキーより前にイベントループが片付ける"""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


def _shown(window: MainWindow) -> MainWindow:
    """窓を出す 前面には出さない（活性にしない）

    CI の Windows の実行機では窓が前面に出られず、活性になる時とならない時がある
    活性でない窓では ``QApplication.focusWidget()`` が ``None`` のままなので、
    試験はどちらでも同じに見える窓の中のフォーカス（``window.focusWidget()``、
    活性になったときにフォーカスを受ける部品）で見る 手元でも CI と同じ条件にするため、
    わざと活性にしない
    """
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.show()
    _flush()
    return window


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _shown(MainWindow(_project(), confirm_unsaved=False))
    created._timeline.select(_clip_ids(created)[0])
    yield created
    created.close()


def _editor(window: MainWindow, name: str) -> TextEditor:
    editor = window._inspector._editors[("source", name)]
    assert isinstance(editor, TextEditor)
    return editor


def _focused(window: MainWindow) -> QWidget | None:
    """窓の中でフォーカスを持つ部品 窓が活性でなくても、活性になったときに受ける物が分かる

    フォーカスのあった部品が消えると ``None`` になる（Qt が窓の中のフォーカスを外す）
    """
    return window.focusWidget()


def _focus_end(window: MainWindow, widget: QWidget) -> None:
    widget.setFocus(Qt.FocusReason.OtherFocusReason)
    if isinstance(widget, QPlainTextEdit):
        widget.moveCursor(QTextCursor.MoveOperation.End)
    elif isinstance(widget, QLineEdit):
        widget.end(False)
    _flush()
    assert _focused(window) is widget


def _leave(window: MainWindow, widget: QWidget) -> None:
    """欄を離れる 活性でない窓ではフォーカスを移しても離れた知らせが来ないので、自分で送る
    （活性の窓では 2 度目の知らせになるが、欄は区切るだけなので同じ結果）"""
    window._timeline.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.sendEvent(
        widget, QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason)
    )
    _flush()


def _type(window: MainWindow, text: str) -> None:
    """フォーカスのある欄へ 1 文字ずつ打つ 毎回、今フォーカスのある物へ送る

    消えた欄へ送り続けると、実機で起きる「次のキーが欄へ入らない」が見えない
    """
    for character in text:
        target = _focused(window)
        assert isinstance(target, QPlainTextEdit | QLineEdit), (
            f"{character!r} を打つ前にフォーカスが欄から外れた（{type(target).__name__}）"
        )
        QTest.keyClicks(target, character)
        _flush()


class TestTyping:
    def test_the_multiline_text_keeps_focus_while_typing(self, window: MainWindow) -> None:
        # 作り直すと 1 文字目で欄が消え、2 文字目からは S の分割やスペースの再生に届く
        area = _editor(window, "text").findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "abc S d")
        assert _source(window, _clip_ids(window)[0], "text") == "テキストabc S d"
        focused = _focused(window)
        assert isinstance(focused, QPlainTextEdit)
        assert focused.toPlainText() == "テキストabc S d"
        assert focused.textCursor().position() == len("テキストabc S d")

    def test_the_single_line_text_keeps_focus_while_typing(self, window: MainWindow) -> None:
        # タイマーの書式は 1 文字目で欄の構成が変わる（文字の欄が消え、タイマーの欄が出る）
        # 作り直す道でも、打っていた欄とカーソルの位置へ戻す
        line = _editor(window, "timer_format").findChild(QLineEdit)
        assert line is not None
        _focus_end(window, line)
        _type(window, "hh:mm")
        assert _source(window, _clip_ids(window)[0], "timer_format") == "hh:mm"
        focused = _focused(window)
        assert isinstance(focused, QLineEdit)
        assert focused is _editor(window, "timer_format").findChild(QLineEdit)
        assert focused.text() == "hh:mm"
        assert focused.cursorPosition() == len("hh:mm")

    def test_typing_without_layout_change_keeps_the_same_widget(self, window: MainWindow) -> None:
        # 構成の変わらない更新で作り直すと、日本語入力の変換中の文字や欄の中の取り消しが消える
        editor = _editor(window, "text")
        area = editor.findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "xy")
        assert _editor(window, "text") is editor

    def test_an_undo_from_outside_keeps_the_cursor_near(self, window: MainWindow) -> None:
        # 取り消しで本文が変わったら入れ直す 頭へ飛ぶと、続きを打つ所を探し直すことになる
        area = _editor(window, "text").findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "ab")
        window.undo()
        _flush()
        focused = _focused(window)
        assert isinstance(focused, QPlainTextEdit)
        assert focused.toPlainText() == "テキスト"
        assert focused.textCursor().position() == len("テキスト")


class TestUndoSteps:
    def test_a_burst_of_typing_is_one_undo_step(self, window: MainWindow) -> None:
        # 1 文字ごとに段が積まれると、打った言葉を戻すのに文字の数だけ取り消すことになる
        before = len(window.document.history_labels)
        area = _editor(window, "text").findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "hello")
        assert len(window.document.history_labels) == before + 1
        window.undo()
        assert _source(window, _clip_ids(window)[0], "text") == "テキスト"

    def test_leaving_the_field_starts_a_new_step(self, window: MainWindow) -> None:
        # 欄を離れて戻ってきた打鍵まで前の段へまとめると、別々に直した物が一緒に戻る
        before = len(window.document.history_labels)
        area = _editor(window, "text").findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "ab")
        _leave(window, area)
        area = _editor(window, "text").findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "cd")
        assert len(window.document.history_labels) == before + 2

    def test_a_long_burst_is_split_after_a_while(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 打ち始めから時間が経った分は別の段にする 長い文を打ったあとの 1 回の取り消しで、
        # 全部が消えないようにする
        now = [0.0]
        monkeypatch.setattr(widgets_module, "_clock", lambda: now[0])
        before = len(window.document.history_labels)
        area = _editor(window, "text").findChild(QPlainTextEdit)
        assert area is not None
        _focus_end(window, area)
        _type(window, "ab")
        now[0] += widgets_module.TYPING_MERGE_SECONDS + 0.1
        _type(window, "cd")
        assert len(window.document.history_labels) == before + 2


class TestOtherFields:
    def test_the_slider_keys_keep_focus(self, window: MainWindow) -> None:
        # 縁取りの太さは 0 から動かすと縁取りの色の欄が出る（構成が変わる）
        # 作り直しても、矢印キーで続けて動かせる
        editor = window._inspector._editors[("source", "border_width")]
        slider = editor.findChild(QSlider)
        assert slider is not None
        slider.setFocus(Qt.FocusReason.OtherFocusReason)
        _flush()
        for _ in range(3):
            target = _focused(window)
            assert isinstance(target, QSlider)
            QTest.keyClick(target, Qt.Key.Key_Right)
            _flush()
        # 矢印キー 1 回で仕様の刻み（1 px）だけ動く 1000 倍した目盛りの 1 では表示の桁で
        # 0 へ丸められ、何度押しても動かなかった
        value = _source(window, _clip_ids(window)[0], "border_width")
        assert value == AnimatedValue(3.0)
        # 構成が変わった（縁取りの色の欄が出た）うえで、同じ欄のスライダーへ戻っている
        assert ("source", "border_color") in window._inspector._editors
        focused = _focused(window)
        assert isinstance(focused, QSlider)
        assert focused is window._inspector._editors[("source", "border_width")].findChild(QSlider)

    def test_the_number_box_keeps_focus_after_enter(self, window: MainWindow) -> None:
        # 数値欄に打って Enter で確定したあとも、続けて打ち直せる
        editor = window._inspector._editors[("source", "size")]
        box = editor.findChild(QDoubleSpinBox)
        assert box is not None
        box.setFocus(Qt.FocusReason.OtherFocusReason)
        box.selectAll()
        _flush()
        QTest.keyClicks(box, "100")
        QTest.keyClick(box, Qt.Key.Key_Return)
        _flush()
        assert _source(window, _clip_ids(window)[0], "size") == AnimatedValue(100.0)
        assert window._inspector._editors[("source", "size")] is editor
        assert _focused(window) is box

    def test_a_slider_drag_is_not_cut_by_an_update(self, window: MainWindow) -> None:
        # ドラッグの最中に別の所から更新が来ても（素材の解析が終わったなど）、掴んだ
        # スライダーを作り直さない 作り直すと、掴んでいた物が消えてドラッグが切れる
        editor = window._inspector._editors[("source", "size")]
        assert isinstance(editor, TrackEditor)
        slider = editor.findChild(QSlider)
        assert slider is not None
        handle = _handle_center(slider)
        QTest.mousePress(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, handle)
        QTest.mouseMove(slider, handle + QPoint(30, 0))
        assert slider.isSliderDown()
        # 構成の変わる更新（縁取りの色が出る）を外から入れる
        clip_id = _clip_ids(window)[0]
        window.execute_all(
            [SetParam(ParamPath.of_source(clip_id, "border_width"), AnimatedValue(4.0))], "外から"
        )
        _flush()
        assert window._inspector._editors[("source", "size")] is editor
        assert slider.isSliderDown()
        QTest.mouseRelease(
            slider,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            handle + QPoint(30, 0),
        )
        _flush()
        size = _source(window, clip_id, "size")
        assert isinstance(size, AnimatedValue) and size.static != 64
        # 離したら、待っていた作り直しを済ませる（縁取りの色の欄が出る）
        assert ("source", "border_color") in window._inspector._editors

    def test_the_keyframe_button_keeps_its_widget(self, window: MainWindow) -> None:
        # ◆ を押すたびに作り直すと、続けて ◀ ▶ を押す前にボタンが入れ替わる
        controls = window._inspector._key_controls[("source", "size")]
        assert isinstance(controls, KeyframeControls)
        controls.toggle.click()
        _flush()
        assert window._inspector._key_controls[("source", "size")] is controls
        assert controls.toggle.text() == "◆"

    @pytest.mark.parametrize(("name", "expected"), [("bold", True), ("align", "right")])
    def test_check_and_choice_keep_focus(
        self, window: MainWindow, name: str, expected: object
    ) -> None:
        # 入り切りや選択肢をキーで変えたあとも、同じ欄で続けて変えられる
        editor = window._inspector._editors[("source", name)]
        box = editor.findChild(QCheckBox) or editor.findChild(QComboBox)
        assert box is not None
        box.setFocus(Qt.FocusReason.OtherFocusReason)
        _flush()
        if isinstance(box, QCheckBox):
            box.click()
        else:
            QTest.keyClick(box, Qt.Key.Key_Down)
        _flush()
        assert _source(window, _clip_ids(window)[0], name) == expected
        assert window._inspector._editors[("source", name)] is editor
        assert _focused(window) is box


class TestValuesFromOutside:
    def test_an_undo_puts_the_toggle_back_quietly(self, window: MainWindow) -> None:
        # 作り直さずに入れ直すとき、切り替えの知らせを出すと、取り消しで戻すたびに
        # 切り替えのコマンドが出て、戻した段の上に新しい段が積まれる
        clip_id = _clip_ids(window)[0]
        window.execute_all([AddEffect(clip_id, registry.require("blur").create())], "足す")
        _flush()
        (section,) = [
            s for s in window._inspector._sections() if s.heading == registry.require("blur").label
        ]
        toggle = next(b for b in section.findChildren(QToolButton) if b.text() == "有効")
        toggle.click()
        _flush()
        assert toggle.text() == "無効"
        steps = len(window.document.history_labels)
        window.undo()
        _flush()
        assert toggle.isChecked() and toggle.text() == "有効"
        assert len(window.document.history_labels) == steps - 1
        assert window.document.can_redo


class TestAnotherClip:
    def test_switching_clips_starts_a_new_step(self, qt_application: QApplication) -> None:
        # フォーカスを変えずに選ぶクリップが変わった（AI の select_clip など）ら、
        # 打ち続けを引き継がない 引き継ぐと、次の 1 文字が前のクリップの段へまとめられ、
        # 1 回の取り消しで別のクリップの編集まで戻る
        del qt_application
        window = _shown(MainWindow(_project(clips=2), confirm_unsaved=False))
        try:
            first, second = _clip_ids(window)
            window._timeline.select(first)
            before = len(window.document.history_labels)
            area = _editor(window, "text").findChild(QPlainTextEdit)
            assert area is not None
            _focus_end(window, area)
            _type(window, "ab")
            window._timeline.select(second)
            _flush()
            # 欄は次のクリップの同じ欄へ移っている（構成が同じなので作り直してフォーカスを戻す）
            focused = _focused(window)
            assert isinstance(focused, QPlainTextEdit)
            focused.moveCursor(QTextCursor.MoveOperation.End)
            _type(window, "c")
            assert _source(window, second, "text") == "テキストc"
            assert len(window.document.history_labels) == before + 2
            window.undo()
            assert _source(window, second, "text") == "テキスト"
            assert _source(window, first, "text") == "テキストab"
        finally:
            window.close()


class TestSeveralClips:
    def test_typing_reaches_every_selected_clip(self, qt_application: QApplication) -> None:
        # 何本も選んでいれば、打った本文は選んだ全部へ入り、1 回の取り消しで全部戻る
        del qt_application
        window = _shown(MainWindow(_project(clips=2), confirm_unsaved=False))
        try:
            first, second = _clip_ids(window)
            window._inspector.set_selection((first, second))
            before = len(window.document.history_labels)
            area = _editor(window, "text").findChild(QPlainTextEdit)
            assert area is not None
            _focus_end(window, area)
            _type(window, "xyz")
            assert _source(window, first, "text") == "テキストxyz"
            assert _source(window, second, "text") == "テキストxyz"
            assert len(window.document.history_labels) == before + 1
            window.undo()
            assert _source(window, second, "text") == "テキスト"
        finally:
            window.close()


def _handle_center(slider: QSlider) -> QPoint:
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider

    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    rect = slider.style().subControlRect(
        QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, slider
    )
    return rect.center()


def test_the_panel_alone_does_not_rebuild_for_values(qt_application: QApplication) -> None:
    # 窓を通さず設定パネルだけでも、値だけの更新では入力欄を作り直さない
    del qt_application
    panel = InspectorPanel()
    try:
        project = _project()
        (clip,) = project.timeline.tracks[0].clips
        panel.set_project(project)
        panel.set_clip(clip.id)
        editor = panel._editors[("source", "text")]
        changed = SetParam(ParamPath.of_source(clip.id, "text"), "別の文").apply(project)
        panel.set_project(changed)
        assert panel._editors[("source", "text")] is editor
        area = editor.findChild(QPlainTextEdit)
        assert area is not None and area.toPlainText() == "別の文"
    finally:
        panel.close()
