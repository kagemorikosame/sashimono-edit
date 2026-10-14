"""グラデーションのバー（#312）の操作

クリックで足す・ドラッグで動かす・下へ引き出して消す・Delete で消す・キーボードで動かす・
色と不透明度と位置の欄 見本の色が字を塗る色と同じこと 明るいテーマでも暗いテーマでも見えること
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QColorDialog, QDialog, QMessageBox

from sashimono.effects import GradientSpec
from sashimono.effects.sources import TEXT
from sashimono.engine.sources import render_source
from sashimono.ui.inspector.gradient import GradientBar, GradientEditor, colour_at
from sashimono.ui.inspector.widgets import create_editor
from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, Colors, current_theme, use_palette

SPEC = GradientSpec("fill_gradient", "グラデーション")
RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
RED_TO_BLUE = (0.0, *RED, 1.0, *BLUE)


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """試験の途中で窓（色の選択・尋ねる・知らせる）が開いたら、その場で落とす

    開くと、閉じる人がいないまま試験が止まる 色の窓を使う試験は自分で差し替える
    """

    def opened(*_args: object, **_kwargs: object) -> None:
        pytest.fail("試験の途中で窓が開いた")

    monkeypatch.setattr(QDialog, "exec", opened)
    monkeypatch.setattr(QMessageBox, "exec", opened)
    monkeypatch.setattr(QColorDialog, "getColor", opened)


@pytest.fixture
def editor(qt_application: QApplication) -> Iterator[GradientEditor]:
    del qt_application
    made = create_editor(SPEC)
    assert isinstance(made, GradientEditor)
    made.resize(320, 120)
    made.set_value(RED_TO_BLUE)
    made.show()
    QApplication.processEvents()
    yield made
    made.close()
    shiboken6.delete(made)


class Recorder:
    """欄が出した確定と途中経過を順に覚える"""

    def __init__(self, editor: GradientEditor) -> None:
        self.changed: list[tuple[float, ...]] = []
        self.previewed: list[tuple[float, ...]] = []
        editor.value_changed.connect(self.changed.append)
        editor.value_previewed.connect(self.previewed.append)


def _mouse(bar: GradientBar, kind: QEvent.Type, x: float, y: float, *, held: bool) -> None:
    buttons = Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton
    point = QPointF(x, y)
    event = QMouseEvent(
        kind,
        point,
        bar.mapToGlobal(point),
        Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(bar, event)


def _key(
    bar: GradientBar, key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
) -> None:
    QApplication.sendEvent(bar, QKeyEvent(QEvent.Type.KeyPress, key, modifiers))


def _handle_y(bar: GradientBar) -> float:
    return bar._track().bottom() + 8.0


def _stops(value: tuple[float, ...]) -> list[tuple[float, tuple[float, ...]]]:
    return [
        (round(p, 3), tuple(round(c, 3) for c in colour)) for p, colour in GradientSpec.stops(value)
    ]


class TestTheMouse:
    def test_a_click_on_the_bar_adds_a_stop_of_that_colour(self, editor: GradientEditor) -> None:
        # 足した所の色が変わると、足しただけで字の色が変わって、どこに足したか分からない
        record = Recorder(editor)
        bar = editor.bar
        x = bar.x_of(0.5)
        _mouse(bar, QEvent.Type.MouseButtonPress, x, 10, held=True)
        _mouse(bar, QEvent.Type.MouseButtonRelease, x, 10, held=False)
        (value,) = record.changed
        stops = GradientSpec.stops(value)
        assert len(stops) == 3
        position, colour = stops[1]
        assert abs(position - 0.5) < 0.02
        assert all(abs(a - b) < 0.03 for a, b in zip(colour, (0.5, 0.0, 0.5, 1.0), strict=True))
        # 足した止まりを選んだ状態になる（すぐ色を選べる）
        assert bar.current == 1

    def test_dragging_moves_a_stop_and_lands_once(self, editor: GradientEditor) -> None:
        # ドラッグの途中で毎回確定すると、取り消しを何十回も押さないと戻らない
        record = Recorder(editor)
        bar = editor.bar
        y = _handle_y(bar)
        _mouse(bar, QEvent.Type.MouseButtonPress, bar.x_of(0.0), y, held=True)
        for share in (0.1, 0.2, 0.3):
            _mouse(bar, QEvent.Type.MouseMove, bar.x_of(share), y, held=True)
        assert record.changed == []
        assert len(record.previewed) >= 2
        _mouse(bar, QEvent.Type.MouseButtonRelease, bar.x_of(0.3), y, held=False)
        (value,) = record.changed
        assert abs(GradientSpec.stops(value)[0][0] - 0.3) < 0.02

    def test_dragging_past_another_stop_keeps_hold_of_it(self, editor: GradientEditor) -> None:
        # 追い越したときに掴んだ止まりが入れ替わると、ドラッグの途中で別の色が動き出す
        record = Recorder(editor)
        bar = editor.bar
        y = _handle_y(bar)
        _mouse(bar, QEvent.Type.MouseButtonPress, bar.x_of(0.0), y, held=True)
        for share in (0.5, 0.9, 0.95):
            _mouse(bar, QEvent.Type.MouseMove, bar.x_of(share), y, held=True)
        _mouse(bar, QEvent.Type.MouseButtonRelease, bar.x_of(0.95), y, held=False)
        (value,) = record.changed
        assert _stops(value) == [(0.95, RED), (1.0, BLUE)]

    def test_pulling_a_stop_off_the_bar_removes_it(self, editor: GradientEditor) -> None:
        # 消す手が Delete しか無いと、マウスだけで作っている人が止まりを消せない
        editor.set_value((0.0, *RED, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, *BLUE))
        record = Recorder(editor)
        bar = editor.bar
        y = _handle_y(bar)
        x = bar.x_of(0.5)
        _mouse(bar, QEvent.Type.MouseButtonPress, x, y, held=True)
        _mouse(bar, QEvent.Type.MouseMove, x, bar.height() + 80, held=True)
        _mouse(bar, QEvent.Type.MouseButtonRelease, x, bar.height() + 80, held=False)
        (value,) = record.changed
        assert _stops(value) == [(0.0, RED), (1.0, BLUE)]

    def test_two_stops_cannot_be_pulled_off(self, editor: GradientEditor) -> None:
        # 1 つにするとバーで動かせる物が無くなり、グラデーションに戻せない
        record = Recorder(editor)
        bar = editor.bar
        y = _handle_y(bar)
        _mouse(bar, QEvent.Type.MouseButtonPress, bar.x_of(1.0), y, held=True)
        _mouse(bar, QEvent.Type.MouseMove, bar.x_of(1.0), bar.height() + 80, held=True)
        _mouse(bar, QEvent.Type.MouseButtonRelease, bar.x_of(1.0), bar.height() + 80, held=False)
        assert all(len(GradientSpec.stops(value)) == 2 for value in record.changed)

    def test_a_press_without_moving_only_selects(self, editor: GradientEditor) -> None:
        # 選ぶつもりで押しただけで履歴が積まれると、取り消しが空振りする
        record = Recorder(editor)
        bar = editor.bar
        y = _handle_y(bar)
        _mouse(bar, QEvent.Type.MouseButtonPress, bar.x_of(1.0), y, held=True)
        _mouse(bar, QEvent.Type.MouseButtonRelease, bar.x_of(1.0), y, held=False)
        assert record.changed == []
        assert bar.current == 1


class TestTheKeyboard:
    def test_arrows_move_and_select(self, editor: GradientEditor) -> None:
        # マウスが使えない人がバーを直せない
        record = Recorder(editor)
        bar = editor.bar
        bar.setFocus()
        bar.select(0)
        _key(bar, Qt.Key.Key_Right)
        assert abs(GradientSpec.stops(record.changed[-1])[0][0] - 0.01) < 1e-6
        _key(bar, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert abs(GradientSpec.stops(record.changed[-1])[0][0] - 0.11) < 1e-6
        _key(bar, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
        assert bar.current == 1
        _key(bar, Qt.Key.Key_Left)
        assert abs(GradientSpec.stops(record.changed[-1])[1][0] - 0.99) < 1e-6

    def test_insert_and_delete(self, editor: GradientEditor) -> None:
        record = Recorder(editor)
        bar = editor.bar
        bar.select(0)
        _key(bar, Qt.Key.Key_Insert)
        assert len(GradientSpec.stops(record.changed[-1])) == 3
        assert bar.current == 1
        _key(bar, Qt.Key.Key_Delete)
        assert _stops(record.changed[-1]) == [(0.0, RED), (1.0, BLUE)]
        # 2 つより減らない
        _key(bar, Qt.Key.Key_Delete)
        assert len(record.changed) == 2

    def test_enter_picks_a_colour(
        self, editor: GradientEditor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 色を選べないと、足した止まりが前後の色の中間のまま変えられない
        asked: list[QColor] = []

        def pick(initial: QColor, *_args: object, **_kwargs: object) -> QColor:
            asked.append(initial)
            return QColor(0, 255, 0, 128)

        monkeypatch.setattr(QColorDialog, "getColor", pick)
        record = Recorder(editor)
        bar = editor.bar
        bar.select(1)
        _key(bar, Qt.Key.Key_Return)
        assert asked[0].blue() == 255
        (value,) = record.changed
        _, colour = GradientSpec.stops(value)[1]
        assert colour == (0.0, 1.0, 0.0, 128 / 255)


class TestTheFields:
    def test_opacity_and_position(self, editor: GradientEditor) -> None:
        # 数で直せないと、50% ちょうどに置く・半透明にする、がドラッグの勘になる
        record = Recorder(editor)
        editor.bar.select(1)
        editor._opacity.setValue(40)
        assert abs(GradientSpec.stops(record.changed[-1])[1][1][3] - 0.4) < 1e-6
        editor._position.setValue(25.0)
        stops = GradientSpec.stops(record.changed[-1])
        assert abs(stops[1][0] - 0.25) < 1e-6 and stops[1][1][:3] == BLUE[:3]
        # 赤を青より後ろへ動かすと並びが入れ替わる 選んだままの止まりが赤でないと、
        # 続けて打った数が別の止まりへ入る
        editor.bar.select(0)
        editor._position.setValue(50.0)
        stops = GradientSpec.stops(record.changed[-1])
        assert [colour[:3] for _, colour in stops] == [BLUE[:3], RED[:3]]
        assert editor.bar.current == 1
        assert editor._position.value() == 50.0

    def test_the_fields_follow_the_selected_stop(self, editor: GradientEditor) -> None:
        editor.set_value((0.0, *RED, 0.75, 0.0, 0.0, 1.0, 0.5))
        editor.bar.select(1)
        assert editor._position.value() == 75.0
        assert editor._opacity.value() == 50
        assert editor._colour.text() == "#0000FF"

    def test_the_buttons_add_and_remove(self, editor: GradientEditor) -> None:
        record = Recorder(editor)
        assert not editor._remove.isEnabled()
        editor._add.click()
        assert len(GradientSpec.stops(record.changed[-1])) == 3
        assert editor._remove.isEnabled()
        editor._remove.click()
        assert len(GradientSpec.stops(record.changed[-1])) == 2


class TestWhatItShows:
    def test_the_bar_shows_the_colours_the_text_gets(self, editor: GradientEditor) -> None:
        # 見本と字で混ぜ方が違うと、見本を見て決めた色が字では違って出る
        fading = (0.0, *RED, 1.0, 0.0, 0.0, 1.0, 0.0)
        editor.set_value(fading)
        bar = editor.bar
        grabbed = bar.grab().toImage()
        ratio = grabbed.devicePixelRatio()
        track = bar._track()
        text = render_source(
            TEXT.create(
                text="■",
                size=200,
                fill_type="linear",
                fill_gradient=fading,
                fill_angle=-90.0,
            ),
            640,
            360,
        )
        assert text is not None
        ys, xs = np.nonzero(text[..., 3] > 0)
        row = (int(ys.min()) + int(ys.max())) // 2
        left, right = int(xs.min()), int(xs.max())
        for share in (0.25, 0.5):
            x = round(left + (right - left) * share)
            red, alpha = int(text[row, x, 0]), int(text[row, x, 3])
            sample = grabbed.pixelColor(
                round((track.left() + track.width() * share) * ratio),
                round((track.top() + 2) * ratio),
            )
            # バーは市松の上に重ねてあるので、字の色と不透明度から同じ重ね方で見本の色を出す
            # 赤と青だけの止まりなので、見本の緑は透けた市松の分だけ（赤にも同じだけ足される）
            expected = round(red * alpha / 255 + sample.green())
            assert abs(sample.red() - expected) <= 6, (share, sample.getRgb(), red, alpha)

    @pytest.mark.parametrize("theme", [THEME_DARK, THEME_LIGHT])
    def test_the_handles_are_seen_in_both_themes(self, editor: GradientEditor, theme: str) -> None:
        # つまみの枠が地と同じ色だと、テーマを替えた途端につまみが見えなくなる
        before = current_theme()
        use_palette(theme)
        try:
            editor.bar.select(0)
            grabbed = editor.bar.grab().toImage()
            ratio = grabbed.devicePixelRatio()
            bar = editor.bar
            # 選んでいないつまみ（2 つ目）の三角の左の辺は、その時のテーマの文字の色
            rect = bar._handle_rect(1)
            colours = [
                grabbed.pixelColor(math.floor(rect.left() * ratio), math.floor(y * ratio))
                for y in range(int(rect.top() + 7), int(rect.bottom() - 1))
            ]
            assert Colors.TEXT.rgb() in {colour.rgb() for colour in colours}
            assert max(_contrast(colour, Colors.PANEL) for colour in colours) >= 3.0
            assert _contrast(Colors.TEXT, Colors.PANEL) >= 3.0
            assert _contrast(Colors.ACCENT, Colors.PANEL) >= 3.0
        finally:
            use_palette(before)


def _contrast(first: QColor, second: QColor) -> float:
    """WCAG の明るさの比"""

    def luminance(colour: QColor) -> float:
        def channel(value: float) -> float:
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

        return (
            0.2126 * channel(colour.redF())
            + 0.7152 * channel(colour.greenF())
            + 0.0722 * channel(colour.blueF())
        )

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_colour_at_matches_the_text_mixing() -> None:
    # 足した止まりの色が字の色と違うと、足しただけで字の色が段になる
    stops = [(0.0, RED), (1.0, (0.0, 0.0, 1.0, 0.0))]
    assert colour_at(stops, 0.5) == (0.5, 0.0, 0.5, 0.5)
    assert colour_at(stops, -1.0) == RED
