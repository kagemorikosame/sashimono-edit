"""プレビューの磁石（位置を動かすときに画面の中央やほかの物の端へ吸い付く）

タイムラインの磁石とは別に切れる（利用者の要望） 既定は入 Shift を押している間は吸い付かない
壊れても絵は動くので、「中央に揃えたつもりが数画素ずれていた」に書き出してから気付く
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import ParamPath, SetParam
from sashimono.core.model import AnimatedValue, Clip, Project
from sashimono.ui.preview_handles import bounding_box, fixed_transform, snap_offset, snap_targets
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ui.test_preview_handles import (
    MakeWidget,
    _apply,
    _clip,
    _drag,
    _media,
    _project,
    _Recorder,
    _value,
    make_widget,
)

__all__ = ["make_widget"]


def _placed_at(project: Project, clip: Clip, pos_x: float) -> Project:
    effect = fixed_transform(clip)
    assert effect is not None
    path = ParamPath.of_effect(clip.id, effect.id, "pos_x")
    return SetParam(path, AnimatedValue(pos_x)).apply(project)


class TestSnapping:
    def test_it_snaps_to_the_middle_of_the_screen(self, make_widget: MakeWidget) -> None:
        # X 30 の絵（110〜270）を 26 だけ左へ 中央が 164 になり、画面の中央 160 へ吸い付く
        clip = _clip(_media())
        project, _ = _project(clip)
        project = _placed_at(project, clip, 30.0)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (190, 90), (170, 90), (164, 90))
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(0.0)

    def test_it_snaps_to_the_edge_of_another(self, make_widget: MakeWidget) -> None:
        # 奥の絵は X -50（30〜190） 手前の絵の左端（80）を 187 まで動かすと、奥の右端 190 へ
        back = _clip(_media("奥.png"))
        front = _clip(_media("手前.png"))
        project, _ = _project(back, front)
        project = _placed_at(project, back, -50.0)
        widget = make_widget(project, front.id)
        seen = _Recorder(widget)
        _drag(widget, (200, 90), (250, 90), (307, 90))
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, front.id, "pos_x") == pytest.approx(110.0)

    def test_shift_lets_it_go_free(self, make_widget: MakeWidget) -> None:
        # タイムラインと同じく Shift を押している間は吸い付かない
        clip = _clip(_media())
        project, _ = _project(clip)
        project = _placed_at(project, clip, 30.0)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (190, 90), (164, 90), modifiers=Qt.KeyboardModifier.ShiftModifier)
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(4.0)

    def test_turning_it_off_stops_it(self, make_widget: MakeWidget) -> None:
        # 切ったのに吸い付くと、設定がある方が質が悪い
        clip = _clip(_media())
        project, _ = _project(clip)
        project = _placed_at(project, clip, 30.0)
        widget = make_widget(project, clip.id)
        widget.set_snap(False, 8)
        seen = _Recorder(widget)
        _drag(widget, (190, 90), (170, 90), (164, 90))
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(4.0)

    def test_far_away_it_moves_freely(self) -> None:
        # 近くに何も無ければずらさない 吸い付く距離の外まで引き寄せると、細かく置けない
        dx, dy, guides = snap_offset((10, 10, 50, 50), snap_targets((320, 180), []), 3.0)
        assert (dx, dy, guides) == (0.0, 0.0, [])

    def test_a_turned_picture_uses_what_is_seen(self) -> None:
        # 回した絵は見えている範囲の端で揃える
        assert bounding_box(((0, 5), (10, 0), (15, 10), (5, 15))) == (0, 0, 15, 15)


class TestSetting:
    def test_it_is_on_by_default_and_kept(self, tmp_path: Path) -> None:
        assert Preferences().preview_snap is True
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(preview_snap=False))
        assert store.load().preview_snap is False

    def test_the_dialog_returns_it(self, qt_application: QApplication) -> None:
        del qt_application
        from sashimono.ui.preferences_dialog import PreferencesDialog

        assert (
            PreferencesDialog(Preferences(preview_snap=False)).preferences().preview_snap is False
        )

    def test_it_is_apart_from_the_timeline(self, qt_application: QApplication) -> None:
        # タイムラインの磁石を切っても、プレビューの磁石は切れない（別の設定）
        del qt_application
        from sashimono.ui.main_window import MainWindow

        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            window._apply_preferences(Preferences(timeline_snap=False, preview_snap=True))
            assert window._preview._snap is True
            window._apply_preferences(Preferences(timeline_snap=True, preview_snap=False))
            assert window._preview._snap is False
        finally:
            window.close()
