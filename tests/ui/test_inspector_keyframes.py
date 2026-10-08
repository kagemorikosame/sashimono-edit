"""オブジェクト設定のキーフレームの操作と、初期値へ戻す操作（利用者の要望）

キーフレーム 各値の横の ◆ で再生位置にキーを打つ・消す ◀ ▶ で前後のキーへ再生位置を
動かす キーの値は隣の入力欄で直す キーはクリップの頭から数えたフレームで持つ
前は ◆ が再生位置（タイムラインのフレーム）のまま打っていて、頭が 0 より後ろの
クリップでは打った所と違う時刻（多くはクリップの外）に点が入っていた

初期値へ戻す 行の名前のダブルクリック（数のスライダーはスライダーのダブルクリックでも）
キーフレームのある値は、再生位置のキーの値だけを初期値にする（無ければ初期値のキーを打つ）
ほかのキーは残す（利用者の決定） どれも取り消せる

窓は表示しない（オフスクリーン）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractSlider,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QWidget,
)

from sashimono.core.commands import Command, Document, ParamPath
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    LayerMode,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects import ColorSpec
from sashimono.effects.sources import TEXT
from sashimono.engine.gpu import BlendMode
from sashimono.ui.inspector.panel import InspectorPanel, KeyframeControls, _RowLabel
from sashimono.ui.inspector.widgets import TrackEditor
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore

#: クリップの頭 0 より後ろに置く 0 に置くと、再生位置とクリップの中の時刻が同じになり、
#: 取り違えても試験が通る
START = 60


def _project(opacity: AnimatedValue | None = None) -> tuple[Project, Clip]:
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    clip = Clip(
        timeline_start=START,
        duration=120,
        source=TEXT.create(),
        opacity=opacity if opacity is not None else AnimatedValue(static=1.0),
    )
    track = Track(TrackKind.MIXED, "レイヤー 1", (clip,))
    return base.with_timeline(replace(base.timeline, tracks=(track,))), clip


class _Harness:
    """窓の代わり 出たコマンドを文書に積み、設定パネルへ戻す 再生位置の移動も受ける"""

    def __init__(self, panel: InspectorPanel, project: Project) -> None:
        self.panel = panel
        self.document = Document(project)
        self.labels: list[str] = []
        self.seeks: list[int] = []
        self.focused: list[ParamPath] = []
        panel.commands_requested.connect(self._run)
        panel.seek_requested.connect(self._seek)
        panel.param_focused.connect(self.focused.append)
        panel.set_project(project)

    def _run(self, commands: list[Command], label: str) -> None:
        with self.document.checkpoint(label):
            for command in commands:
                self.document.execute(command)
        self.labels.append(label)
        self._show(self.document.project)

    def _show(self, project: Project) -> None:
        # ここでは前の部品を消さない 押された部品の知らせの中から呼ばれるので、消すと
        # 押された部品がその場で無くなってプロセスごと落ちる 探す前に消す（:func:`_settle`）
        self.panel.set_project(project)

    def _seek(self, frame: int) -> None:
        self.seeks.append(frame)
        self.panel.set_frame(frame)

    def undo(self) -> None:
        self.document.undo()
        self._show(self.document.project)

    def opacity(self, clip: Clip) -> AnimatedValue:
        located = self.document.project.timeline.locate_clip(clip.id)
        assert located is not None
        return located[1].opacity


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def _open(panel: InspectorPanel, opacity: AnimatedValue | None = None) -> tuple[_Harness, Clip]:
    project, clip = _project(opacity)
    harness = _Harness(panel, project)
    panel.set_selection((clip.id,))
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return harness, clip


def _settle() -> None:
    """作り直した前の部品を消す 消えるまで待たないと、探したときに前の部品が当たる"""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _opacity_editor(panel: InspectorPanel) -> TrackEditor:
    _settle()
    return next(e for e in panel.findChildren(TrackEditor) if e.spec.name == "opacity")


def _controls(panel: InspectorPanel, clip: Clip) -> KeyframeControls:
    _settle()
    path = ParamPath.of_clip(clip.id, "opacity")
    found = [c for c in panel.findChildren(KeyframeControls) if c.path == path]
    assert len(found) == 1, "不透明度の ◀ ◆ ▶ が無い"
    return found[0]


def _label(panel: InspectorPanel, text: str) -> _RowLabel:
    _settle()
    found = [label for label in panel.findChildren(_RowLabel) if label.text() == text]
    assert found, f"{text} の行が無い"
    return found[0]


def _mouse(
    widget: QWidget,
    kind: QEvent.Type,
    point: QPoint,
    buttons: Qt.MouseButton = Qt.MouseButton.LeftButton,
) -> None:
    position = QPointF(point)
    QApplication.sendEvent(
        widget,
        QMouseEvent(
            kind,
            position,
            widget.mapToGlobal(position),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        ),
    )


def _handle(slider: QSlider) -> QPoint:
    """つまみの真ん中 溝を押すと値が飛ぶので、掴む試験はここを押す"""
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    return (
        slider.style()
        .subControlRect(
            QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, slider
        )
        .center()
    )


def _double_click(widget: _RowLabel | QSlider) -> None:
    """2 回目の押下（ダブルクリック）を送って、動かさずに離す"""
    point = _handle(widget) if isinstance(widget, QSlider) else QPoint(4, 4)
    _mouse(widget, QEvent.Type.MouseButtonDblClick, point)
    _mouse(widget, QEvent.Type.MouseButtonRelease, point, Qt.MouseButton.NoButton)


def _drag(slider: QSlider, first: QEvent.Type, distance: int) -> None:
    """つまみを押したまま左へ ``distance`` 画素動かして離す"""
    start = _handle(slider)
    _mouse(slider, first, start)
    for step in range(0, distance + 1, 5):
        _mouse(slider, QEvent.Type.MouseMove, start - QPoint(step, 0))
    _mouse(
        slider, QEvent.Type.MouseButtonRelease, start - QPoint(distance, 0), Qt.MouseButton.NoButton
    )


def _animated(*keys: tuple[int, float]) -> AnimatedValue:
    return AnimatedValue(static=1.0, keyframes=tuple(Keyframe(frame=f, value=v) for f, v in keys))


class TestKeyButton:
    def test_the_key_goes_where_the_playhead_is_inside_the_clip(
        self, panel: InspectorPanel
    ) -> None:
        # 再生位置（タイムラインの 70）のまま打つと、クリップの頭から 70 の所に点が入る
        harness, clip = _open(panel)
        panel.set_frame(START + 10)
        _controls(panel, clip).toggle.click()
        assert [k.frame for k in harness.opacity(clip).keyframes] == [10]
        assert harness.labels == ["キーフレームを打つ"]
        assert harness.focused == [ParamPath.of_clip(clip.id, "opacity")]

    def test_pressing_again_removes_the_key_and_undo_brings_it_back(
        self, panel: InspectorPanel
    ) -> None:
        harness, clip = _open(panel, _animated((10, 0.3), (50, 0.9)))
        panel.set_frame(START + 10)
        controls = _controls(panel, clip)
        assert controls.toggle.text() == "◆", (
            "再生位置にキーがあるのに白抜きだと、消せると分からない"
        )
        controls.toggle.click()
        assert [k.frame for k in harness.opacity(clip).keyframes] == [50]
        harness.undo()
        assert [k.frame for k in harness.opacity(clip).keyframes] == [10, 50]

    def test_the_look_follows_the_playhead(self, panel: InspectorPanel) -> None:
        # 見た目が再生位置に付いてこないと、キーの上にいるのかどうかが分からない
        _, clip = _open(panel, _animated((10, 0.3), (50, 0.9)))
        controls = _controls(panel, clip)
        panel.set_frame(START + 50)
        assert controls.toggle.text() == "◆"
        panel.set_frame(START + 30)
        assert controls.toggle.text() == "◇"
        assert controls.toggle.isEnabled()
        # クリップの外に打ったキーは描く所が無い 押せないようにする
        panel.set_frame(START - 5)
        assert not controls.toggle.isEnabled()

    def test_the_key_button_starts_hollow_without_keys(self, panel: InspectorPanel) -> None:
        _, clip = _open(panel)
        panel.set_frame(START)
        controls = _controls(panel, clip)
        assert controls.toggle.text() == "◇"
        assert not controls.previous.isEnabled()
        assert not controls.next.isEnabled()


class TestJumping:
    def test_the_arrows_move_the_playhead_to_the_neighbouring_keys(
        self, panel: InspectorPanel
    ) -> None:
        # 前後のキーへ飛べないと、キーの値を直すたびにタイムラインで点を探して再生位置を合わせる
        harness, clip = _open(panel, _animated((10, 0.3), (50, 0.9), (90, 0.5)))
        panel.set_frame(START + 30)
        controls = _controls(panel, clip)
        assert controls.previous.isEnabled() and controls.next.isEnabled()
        controls.next.click()
        assert harness.seeks == [START + 50]
        controls.next.click()
        assert harness.seeks[-1] == START + 90
        assert not controls.next.isEnabled(), "最後のキーの先には飛べない"
        controls.previous.click()
        assert harness.seeks[-1] == START + 50

    def test_editing_the_value_changes_the_key_under_the_playhead(
        self, panel: InspectorPanel
    ) -> None:
        # 飛んだ先で値を直すと、そのキーの値が変わる（ほかのキーは動かない）
        harness, clip = _open(panel, _animated((10, 0.3), (50, 0.9)))
        panel.set_frame(START + 30)
        _controls(panel, clip).next.click()
        editor = _opacity_editor(panel)
        editor.value_changed.emit(AnimatedValue(static=0.25))
        keys = harness.opacity(clip).keyframes
        assert [(k.frame, k.value) for k in keys] == [(10, 0.3), (50, 0.25)]


class TestReset:
    def test_double_clicking_the_name_resets_a_plain_value(self, panel: InspectorPanel) -> None:
        # 初期値を覚えていないと、戻すのに数を調べて打ち直すことになる
        harness, clip = _open(panel, AnimatedValue(static=0.4))
        _double_click(_label(panel, "不透明度"))
        assert harness.opacity(clip) == AnimatedValue(static=1.0)
        assert harness.labels == ["不透明度を初期値に戻す"]
        harness.undo()
        assert harness.opacity(clip) == AnimatedValue(static=0.4)

    def test_an_animated_value_resets_only_the_key_under_the_playhead(
        self, panel: InspectorPanel
    ) -> None:
        # アニメーションごと消すと、1 か所だけ戻したいのに打ったキーが全部消える（利用者の決定）
        harness, clip = _open(panel, _animated((10, 0.3), (50, 0.2)))
        panel.set_frame(START + 50)
        _double_click(_label(panel, "不透明度"))
        keys = harness.opacity(clip).keyframes
        assert [(k.frame, k.value) for k in keys] == [(10, 0.3), (50, 1.0)]

    def test_between_keys_a_default_key_is_added(self, panel: InspectorPanel) -> None:
        harness, clip = _open(panel, _animated((10, 0.3), (50, 0.2)))
        panel.set_frame(START + 30)
        _double_click(_label(panel, "不透明度"))
        keys = harness.opacity(clip).keyframes
        assert [(k.frame, k.value) for k in keys] == [(10, 0.3), (30, 1.0), (50, 0.2)]

    def test_double_clicking_the_slider_resets_too(self, panel: InspectorPanel) -> None:
        # 数の欄は数字を打ち直すのにダブルクリックを使う スライダーのダブルクリックで戻す
        harness, clip = _open(panel, AnimatedValue(static=0.4))
        editor = _opacity_editor(panel)
        slider = editor.findChild(QSlider)
        assert slider is not None
        _double_click(slider)
        assert harness.opacity(clip) == AnimatedValue(static=1.0)

    def test_an_unchanged_value_adds_no_step(self, panel: InspectorPanel) -> None:
        # もう初期値なのに段を積むと、戻しても何も変わらない取り消しが増える
        harness, _ = _open(panel)
        _double_click(_label(panel, "不透明度"))
        assert harness.labels == []

    def test_a_colour_goes_back_to_its_default(self, panel: InspectorPanel) -> None:
        harness, clip = _open(panel)
        spec = next(s for s in TEXT.parameters if isinstance(s, ColorSpec))
        assert clip.source is not None
        changed = replace(clip, source=clip.source.with_param(spec.name, (0.1, 0.2, 0.3, 1.0)))
        project = harness.document.project
        track = project.timeline.tracks[0]
        project = project.with_timeline(
            project.timeline.replace_track(replace(track, clips=(changed,)))
        )
        harness.document = Document(project)
        harness._show(project)
        _double_click(_label(panel, spec.label))
        located = harness.document.project.timeline.locate_clip(clip.id)
        assert located is not None and located[1].source is not None
        assert located[1].source.params[spec.name] == spec.default_value()

    def test_the_blend_mode_goes_back_to_normal(self, panel: InspectorPanel) -> None:
        harness, clip = _open(panel)
        blend = next(
            box
            for box in panel.findChildren(QComboBox)
            if box.findData(BlendMode.ADD) >= 0 and box.findData(BlendMode.NORMAL) >= 0
        )
        blend.setCurrentIndex(blend.findData(BlendMode.ADD))
        _double_click(_label(panel, "合成モード"))
        located = harness.document.project.timeline.locate_clip(clip.id)
        assert located is not None
        assert located[1].blend_mode == BlendMode.NORMAL

    def test_text_is_not_reset_by_a_stray_double_click(self, panel: InspectorPanel) -> None:
        # 打った文字は戻すと消える うっかり起きやすいダブルクリックでは戻さない
        _open(panel)
        assert not _label(panel, "文字").resettable


class TestTheResetSetting:
    """ダブルクリックで戻すのは好みが分かれる（うっかり戻るのが嫌な人がいる） 設定で切れる"""

    def test_turning_it_off_really_stops_it(self, panel: InspectorPanel) -> None:
        # 切っても効いたままなら、設定がある方が質が悪い
        harness, clip = _open(panel, AnimatedValue(static=0.4))
        panel.set_double_click_reset(False)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        label = _label(panel, "不透明度")
        assert not label.resettable
        _double_click(label)
        editor = _opacity_editor(panel)
        slider = editor.findChild(QSlider)
        assert slider is not None
        _double_click(slider)
        _double_click(_label(panel, "合成モード"))
        assert harness.labels == []
        assert harness.opacity(clip) == AnimatedValue(static=0.4)

    def test_it_is_on_by_default_and_kept(self, tmp_path: Path) -> None:
        assert Preferences().double_click_reset
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(double_click_reset=False))
        assert not store.load().double_click_reset

    def test_the_dialog_carries_it(self, qt_application: QApplication) -> None:
        # 画面が値を返さないと、設定を開いて OK を押しただけで既定へ戻る
        del qt_application
        dialog = PreferencesDialog(Preferences(double_click_reset=False))
        try:
            assert not dialog.preferences().double_click_reset
        finally:
            dialog.deleteLater()


#: 試験の間、本物の機器から来る入力 試験が送る物（``sendEvent``）は自然に起きた物ではない
_REAL_INPUT = frozenset(
    {
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick,
        QEvent.Type.MouseMove,
        QEvent.Type.Wheel,
        QEvent.Type.KeyPress,
        QEvent.Type.KeyRelease,
        QEvent.Type.TabletPress,
        QEvent.Type.TabletMove,
        QEvent.Type.TabletRelease,
        QEvent.Type.TouchBegin,
        QEvent.Type.TouchUpdate,
        QEvent.Type.TouchEnd,
    }
)


class _RealInputShield(QObject):
    """本物のマウス・キーボードの入力を止める 止めた数を数える

    窓を画面に出して待つ（``qWait``）間に、手元のマウスが窓の上でホイールを回したり
    押したりすると、試験と関係の無い値（スタイルや拡大率）が変わって段が積まれ、
    手元でだけ落ちた（#269） 試験は ``sendEvent`` で送るので、自然に起きた入力
    （``spontaneous``）だけを止めれば試験の操作は通る
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocked = 0

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        if event.spontaneous() and event.type() in _REAL_INPUT:
            self.blocked += 1
            return True
        return super().eventFilter(watched, event)


@pytest.fixture
def shielded(qt_application: QApplication) -> Iterator[_RealInputShield]:
    shield = _RealInputShield()
    qt_application.installEventFilter(shield)
    yield shield
    qt_application.removeEventFilter(shield)


@pytest.mark.usefixtures("shielded")
class TestHoldingTheSlider:
    """押したまま動かす調整（利用者の言う長押し）と、ダブルクリックで戻すのを両立させる"""

    def _slider(self, panel: InspectorPanel) -> QSlider:
        editor = _opacity_editor(panel)
        slider = editor.findChild(QSlider)
        assert slider is not None
        return slider

    def test_pressing_and_dragging_adjusts_the_value(self, panel: InspectorPanel) -> None:
        harness, clip = _open(panel, AnimatedValue(static=0.8))
        _drag(self._slider(panel), QEvent.Type.MouseButtonPress, 40)
        assert harness.labels == ["opacity を変更"]
        assert harness.opacity(clip).static < 0.8

    def test_a_press_right_after_a_click_still_drags(self, panel: InspectorPanel) -> None:
        # クリックの直後の押下はダブルクリックとして届く 前はそれを握りつぶして初期値へ
        # 戻していたので、クリックしてから押したまま動かすと、つまみが動かず値が 1.0 へ飛んだ
        harness, clip = _open(panel, AnimatedValue(static=0.8))
        slider = self._slider(panel)
        _mouse(slider, QEvent.Type.MouseButtonPress, _handle(slider))
        _mouse(slider, QEvent.Type.MouseButtonRelease, _handle(slider), Qt.MouseButton.NoButton)
        assert harness.labels == [], "掴んで離しただけでは何も積まない"
        _drag(slider, QEvent.Type.MouseButtonDblClick, 40)
        assert harness.labels == ["opacity を変更"]
        assert harness.opacity(clip).static < 0.8

    def test_a_still_double_click_resets(self, panel: InspectorPanel) -> None:
        # 動かさない 2 回目の押下は、今までどおり初期値へ戻す（1 回の取り消しで戻る）
        harness, clip = _open(panel, AnimatedValue(static=0.4))
        slider = self._slider(panel)
        _mouse(slider, QEvent.Type.MouseButtonPress, _handle(slider))
        _mouse(slider, QEvent.Type.MouseButtonRelease, _handle(slider), Qt.MouseButton.NoButton)
        _double_click(slider)
        assert harness.labels == ["不透明度を初期値に戻す"]
        assert harness.opacity(clip) == AnimatedValue(static=1.0)

    @pytest.mark.parametrize("start", [1.0, 0.5])
    def test_a_small_drag_near_the_default_is_kept(
        self, panel: InspectorPanel, start: float
    ) -> None:
        # つまみを掴んで数画素だけ動かす細かい合わせ（初期値の 1.0 の近くでよくやる）を、
        # マウスの動いた距離がドラッグの距離に足りないからと「動かしていない」扱いにして
        # 捨てていた 離すと値が元へ戻り、何度やっても変わらなかった（利用者の報告）
        harness, clip = _open(panel, AnimatedValue(static=start))
        slider = self._slider(panel)
        handle = _handle(slider)
        _mouse(slider, QEvent.Type.MouseButtonPress, handle)
        for step in (1, 2, 3):
            _mouse(slider, QEvent.Type.MouseMove, handle - QPoint(step, 0))
        _mouse(
            slider, QEvent.Type.MouseButtonRelease, handle - QPoint(3, 0), Qt.MouseButton.NoButton
        )
        assert harness.labels == ["opacity を変更"]
        assert harness.opacity(clip).static < start

    def test_holding_on_the_groove_is_one_step(self, panel: InspectorPanel) -> None:
        # 溝を押したままにすると、見た目によっては押した所へ飛び、ほかの見た目では 1 目盛りずつ
        # 進み続ける どちらでも離したときに 1 段だけ積む 進むたびに積むと、長押し 1 回で
        # 取り消しが何段も要る
        harness, clip = _open(panel, AnimatedValue(static=0.2))
        slider = self._slider(panel)
        slider.window().show()
        point = QPoint(slider.width() - 3, slider.height() // 2)
        _mouse(slider, QEvent.Type.MouseButtonPress, point)
        QTest.qWait(700)
        _mouse(slider, QEvent.Type.MouseButtonRelease, point, Qt.MouseButton.NoButton)
        assert harness.labels == ["opacity を変更"]
        assert harness.opacity(clip).static > 0.2

    def test_holding_a_spin_arrow_is_one_step(self, panel: InspectorPanel) -> None:
        # 数値欄の増減のボタンの長押しも同じ 1 段ずつ積むと、戻すのに何回も押すことになる
        harness, clip = _open(panel, AnimatedValue(static=0.2))
        box = _opacity_editor(panel).findChild(QDoubleSpinBox)
        assert box is not None
        box.window().show()
        up = QPoint(box.width() - 4, 3)
        _mouse(box, QEvent.Type.MouseButtonPress, up)
        QTest.qWait(900)
        _mouse(box, QEvent.Type.MouseButtonRelease, up, Qt.MouseButton.NoButton)
        assert harness.labels == ["opacity を変更"]
        # 増えるか減るかはボタンの並べ方（見た目）次第 動いたことだけを見る
        assert abs(harness.opacity(clip).static - 0.2) > 0.02


class TestSlider:
    def test_clicking_the_groove_is_kept(self, panel: InspectorPanel) -> None:
        # 溝を押した分はプレビューにしか出ず、履歴にも保存にも残らなかった
        harness, clip = _open(panel, AnimatedValue(static=0.4))
        editor = _opacity_editor(panel)
        slider = editor.findChild(QSlider)
        assert slider is not None
        slider.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
        assert harness.opacity(clip).static != 0.4
        assert len(harness.labels) == 1

    def test_grabbing_without_moving_adds_no_step(self, panel: InspectorPanel) -> None:
        # 掴んで離しただけ（ダブルクリックの 1 回目も）で同じ値の段が積まれていた
        harness, _ = _open(panel, AnimatedValue(static=0.4))
        editor = _opacity_editor(panel)
        slider = editor.findChild(QSlider)
        assert slider is not None
        slider.setSliderDown(True)
        slider.setSliderDown(False)
        assert harness.labels == []
