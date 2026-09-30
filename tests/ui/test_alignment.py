"""配置のテンプレート（設定パネルの「揃える」 左上・上の中央・中央など 9 か所）

X・Y の数を打たずに、見えている範囲ごと画面の端や中央へ寄せる（利用者の要望）
大きさは描く側の枠から取る 数を自分で出すと、画素で置いた素材や回した絵で端がずれる
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QToolButton

from sashimono.core.model import Project
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preview_handles import ALIGNMENTS
from tests.ui.test_preview_handles import (
    MakeWidget,
    _apply,
    _clip,
    _media,
    _project,
    _Recorder,
    _value,
    make_widget,
)
from tests.ui.test_preview_snap import _placed_at

__all__ = ["make_widget"]


class TestThePreview:
    @pytest.mark.parametrize(
        ("anchor", "x", "y"),
        [
            # 160 × 90 の絵を 320 × 180 の画面へ 左上なら中心は (80, 45) で、X -80・Y +45
            ("top_left", -80.0, 45.0),
            ("top", 0.0, 45.0),
            ("bottom_right", 80.0, -45.0),
            ("left", -80.0, 0.0),
        ],
    )
    def test_it_puts_the_edges_on_the_screen(
        self, make_widget: MakeWidget, anchor: str, x: float, y: float
    ) -> None:
        # Y は上が正 上へ寄せたら Y が増える 逆だと上の中央を選んで下へ行く
        clip = _clip(_media())
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        assert widget.align_selected(anchor)
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(x)
        assert _value(moved, clip.id, "pos_y") == pytest.approx(y)

    def test_the_middle_brings_it_back(self, make_widget: MakeWidget) -> None:
        clip = _clip(_media())
        project, _ = _project(clip)
        project = _placed_at(project, clip, 30.0)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        assert widget.align_selected("center")
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(0.0)

    def test_nothing_is_done_without_a_picture(self, make_widget: MakeWidget) -> None:
        # 今のコマに映っていない物は大きさが分からない 寄せずに断る
        clip = _clip(_media(), start=100)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        assert not widget.align_selected("center")
        assert seen.committed == []


class TestTheInspector:
    def test_the_nine_buttons_reach_the_clip(self, qt_application: QApplication) -> None:
        # 設定パネルの 9 つのボタンから窓を通ってプレビューまで届き、1 段で積まれる
        del qt_application
        clip = _clip(_media())
        project, _ = _project(clip)
        window = MainWindow(project, confirm_unsaved=False)
        try:
            window._timeline.select(clip.id)
            names = {
                button.objectName()
                for button in window._inspector.findChildren(QToolButton)
                if button.objectName().startswith("align_")
            }
            assert names == {f"align_{name}" for name, *_ in ALIGNMENTS}
            button = window._inspector.findChild(QToolButton, "align_top_left")
            assert button is not None
            before = window.document.project
            button.click()
            assert _value(window.document.project, clip.id, "pos_x") == pytest.approx(-80.0)
            window.undo()
            assert window.document.project == before
        finally:
            window.close()

    def test_a_text_without_placement_shows_none(self, qt_application: QApplication) -> None:
        # 置く位置を持たない物（空のプロジェクト）では出さない
        del qt_application
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert window._inspector.findChild(QToolButton, "align_center") is None
        finally:
            window.close()
