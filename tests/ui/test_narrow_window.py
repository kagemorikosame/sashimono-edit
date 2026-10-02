"""1366x768 と 1280x720 のノート PC の画面で、編集画面が崩れずに使えるか

前は部品の最小の幅の和が 1485 画素（Windows の書体）あり、1366 の画面に窓が収まらなかった
その幅まで狭めると、設定パネルが中身より狭くなって横の巻物に入り、数値欄の「100.00 %」と
エフェクトの見出しの ✕ が画面の外へ隠れた

窓の大きさは、画面いっぱいに広げたときの中身の大きさで頼む 画面の高さからタスクバー
（Windows 11 で 48 画素）と題名の帯（32 画素）を引いた高さ Windows の見た目でも
オフスクリーン（書体の幅が違う）でも同じ試験が通ること
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDockWidget,
    QLabel,
    QLayout,
    QScrollArea,
    QToolButton,
    QWidget,
)

from sashimono.core.commands import AddEffect, insert_generated
from sashimono.core.model import ClipId, Project
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.effects.sources import TEXT
from sashimono.ui.flow_layout import ElidedLabel
from sashimono.ui.inspector import InspectorPanel
from sashimono.ui.inspector.panel import _Section
from sashimono.ui.main_window import DEFAULT_SIZE, MainWindow, initial_size
from sashimono.ui.theme import style_sheet
from sashimono.ui.transport import TransportBar

#: タスクバーと、画面いっぱいに広げた窓の題名の帯（画素）
TASKBAR, TITLE = 48, 32

#: 画面いっぱいに広げた窓の中身の大きさ
SCREENS = {
    "1366x768": QSize(1366, 768 - TASKBAR - TITLE),
    "1280x720": QSize(1280, 720 - TASKBAR - TITLE),
}

#: 1 行の文字の左右に Qt の入力欄が空ける余白（QLineEdit の決まり 1 画素の枠と 1 画素の余白）
LINE_EDIT_MARGIN = 4


def _settle() -> None:
    for _ in range(6):
        QApplication.processEvents()


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    """テキストにグローとぼかしを積んで選んだ編集画面 本番と同じスタイルシートを当てる

    スタイルシートはボタンの余白を変えるので、当てずに測ると本番より狭く出る
    当てる前に前の試験が捨てた窓を片付け、外す前にこの窓を壊す 捨てた編集画面に
    スタイルシートが当たったまま後のごみ集めで壊れると、GPU の無い CI でプロセスごと落ちる
    （#149 tests/ui/test_issue27_polish.py と同じ作法）
    """
    gc.collect()
    saved = qt_application.styleSheet()
    qt_application.setStyleSheet(style_sheet())
    created = MainWindow(Project.create(), confirm_unsaved=False)
    created.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    created.reset_layout()
    commands = insert_generated(
        created.project, TEXT.create(text="見本のテロップ"), at_frame=0, duration=60
    )
    created.apply_commands(commands, "テキストを追加")
    clip = next(c for t in created.project.timeline.tracks for c in t.clips)
    created.apply_commands(
        [
            AddEffect(clip.id, registry.require("glow").create()),
            AddEffect(clip.id, registry.require("blur").create()),
        ],
        "エフェクトを追加",
    )
    created.select_clip(clip.id)
    created.show()
    _settle()
    try:
        yield created
    finally:
        created.close()
        created.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        if shiboken6.isValid(created):
            shiboken6.delete(created)
        gc.collect()
        qt_application.setStyleSheet(saved)


def _dock(window: MainWindow, name: str) -> QDockWidget:
    dock = window.findChild(QDockWidget, name)
    assert dock is not None
    return dock


def _laid_out(widget: QWidget) -> list[QWidget]:
    """``widget`` の置き方（入れ子の置き方も）が並べている、見えている部品"""
    found: list[QWidget] = []

    def walk(layout: QLayout) -> None:
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is None:
                continue
            child = item.widget()
            if child is not None and child.isVisible() and not child.geometry().isEmpty():
                found.append(child)
            inner = item.layout()
            if inner is not None:
                walk(inner)

    layout = widget.layout()
    if layout is not None:
        walk(layout)
    return found


def _overlaps(root: QWidget) -> list[str]:
    """置き方が並べた部品どうしで重なっている組 縮めすぎると Qt は部品を重ねて置く"""
    found: list[str] = []
    for parent in [root, *root.findChildren(QWidget)]:
        if not parent.isVisible():
            continue
        children = _laid_out(parent)
        for index, first in enumerate(children):
            for second in children[index + 1 :]:
                if first.geometry().intersects(second.geometry()):
                    found.append(
                        f"{type(first).__name__}{first.geometry().getRect()} と "
                        f"{type(second).__name__}{second.geometry().getRect()}"
                    )
    return found


def _squeezed(root: QWidget) -> list[str]:
    """最小の幅より狭く置かれた操作の部品 文字や印が欠けて見える

    幅を決め打ちにした部品（再生の印のボタン・◀ ◆ ▶）は、置き方に縮められたのではないので
    数えない 数値欄の中身は :func:`_cut_numbers` が文字の幅で見る
    """
    kinds = (QAbstractButton, QAbstractSpinBox, QComboBox)
    return [
        f"{type(widget).__name__} {getattr(widget, 'text', lambda: '')()!r} "
        f"{widget.width()} < {widget.minimumSizeHint().width()}"
        for widget in root.findChildren(QWidget)
        if isinstance(widget, kinds)
        and widget.isVisible()
        and widget.minimumWidth() != widget.maximumWidth()
        and widget.width() < widget.minimumSizeHint().width()
    ]


def _cut_numbers(panel: InspectorPanel) -> list[str]:
    """数値欄のうち、出ている数字と単位が欄に入り切らない物"""
    cut = []
    for spin in panel.findChildren(QAbstractSpinBox):
        edit = spin.lineEdit()
        if not spin.isVisible() or edit is None:
            continue
        need = edit.fontMetrics().horizontalAdvance(edit.text())
        room = edit.contentsRect().width() - LINE_EDIT_MARGIN
        if need > room:
            cut.append(f"{edit.text()!r} {need} > {room}")
    return cut


def _hidden_removes(panel: InspectorPanel) -> list[str]:
    """見出しの ✕ のうち、設定パネルの見える範囲から横にはみ出す物"""
    scroll = panel.findChild(QScrollArea)
    assert scroll is not None
    viewport = scroll.viewport()
    hidden = []
    removes = [b for b in panel.findChildren(QToolButton) if b.text() == "✕" and b.isVisible()]
    assert removes, "エフェクトの見出しに ✕ が無い"
    for button in removes:
        rect = QRect(button.mapTo(viewport, button.rect().topLeft()), button.size())
        if rect.left() < 0 or rect.right() >= viewport.width():
            hidden.append(f"✕ {rect.getRect()} 見える幅 {viewport.width()}")
    return hidden


def _check_panels(window: MainWindow) -> None:
    """見えているパネルが中身より狭くなく、部品が重ならず、値と ✕ が欠けない"""
    panels = [window.centralWidget()] + [
        dock.widget()
        for dock in window.findChildren(QDockWidget)
        if dock.isVisible() and not dock.visibleRegion().isEmpty()
    ]
    for panel in panels:
        assert panel is not None
        hint = panel.minimumSizeHint()
        assert panel.width() >= hint.width(), (type(panel).__name__, panel.width(), hint)
        assert _overlaps(panel) == [], type(panel).__name__
        assert _squeezed(panel) == [], type(panel).__name__
    inspector = window.findChild(InspectorPanel)
    assert inspector is not None
    scroll = inspector.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.horizontalScrollBar().maximum() == 0, "設定パネルが横の巻物に入った"
    assert _cut_numbers(inspector) == []
    assert _hidden_removes(inspector) == []


def _shown_texts(widget: QWidget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel) if label.isVisible()]


def _regions_overlap(window: MainWindow) -> list[str]:
    """パネルどうし（とプレビューの列）が重なっている組"""
    central = window.centralWidget()
    assert central is not None
    areas = [("プレビュー", central.geometry())] + [
        (dock.objectName(), dock.geometry())
        for dock in window.findChildren(QDockWidget)
        if dock.isVisible() and not dock.visibleRegion().isEmpty()
    ]
    return [
        f"{first} と {second}"
        for index, (first, a) in enumerate(areas)
        for second, b in areas[index + 1 :]
        if a.intersects(b)
    ]


@pytest.mark.parametrize("screen", sorted(SCREENS))
def test_the_editor_fits_a_small_laptop_screen(window: MainWindow, screen: str) -> None:
    """窓の最小の幅が画面に収まり、頼んだ大きさのまま崩れない

    最小の幅が画面より広いと、窓は画面からはみ出し、右の設定パネルが画面の外へ出る
    """
    size = SCREENS[screen]
    assert window.minimumSizeHint().width() <= size.width()
    assert window.minimumSizeHint().height() <= size.height()
    window.resize(size)
    _settle()
    assert window.size() == size
    assert _regions_overlap(window) == []
    _check_panels(window)


#: 書体の違いで窓の最小の高さが伸びる分の余裕（画素） GitHub Actions の Windows では手元より
#: 84 画素高く出て（562 → 646）、1280x720 の画面の中身（640）を超えた
FONT_HEADROOM = 100


def test_the_minimum_height_leaves_room_for_taller_fonts(window: MainWindow) -> None:
    """窓の最小の高さは、1280x720 の画面の中身より、書体の違いの分だけ余裕を持って低い

    手元でぎりぎり収まるだけだと、行の高さが高い書体の機械（CI・別の既定の書体）で窓が
    画面から下へはみ出し、ステータスバーとタイムラインの下が見えなくなる 高さを決めて
    いたのは、タイムラインの最小（160）と AI のパネルの会話・入力の欄の最小（Qt の既定）
    """
    assert window.minimumSizeHint().height() <= SCREENS["1280x720"].height() - FONT_HEADROOM


@pytest.mark.parametrize("name", ["subtitles", "chat", "media", "inspector"])
def test_every_panel_works_at_the_narrowest_window(window: MainWindow, name: str) -> None:
    """窓をいちばん狭くしても、どのパネルを前に出しても部品が重ならず欠けない

    字幕と AI のパネルは、メディアと設定パネルに重ねたタブの裏にある 前に出したときに
    ボタンの列が折り返さずにはみ出すと、パネルの幅が足りていても押せない所ができる
    """
    narrowest = QSize(window.minimumSizeHint().width(), SCREENS["1280x720"].height())
    window.resize(narrowest)
    _dock(window, name).raise_()
    _settle()
    assert window.width() == narrowest.width()
    assert _regions_overlap(window) == []
    _check_panels(window)


def test_the_transport_hides_labels_before_cutting_them(qt_application: QApplication) -> None:
    """プレビューの下の帯を狭めると、全体の長さと「再生品質」の名前を丸ごと隠す

    縮めて残すと時刻の数字が途中で欠ける 再生の操作・今の位置・画質の欄は隠さない
    帯は置き方の無い入れ物に入れて、幅を直に決める（窓の置き方に戻されないように）
    """
    del qt_application
    holder = QWidget()
    holder.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    bar = TransportBar(FrameRate(30), holder)
    bar.set_duration(30 * 60 * 61)
    holder.resize(2000, 100)
    holder.show()
    everything = sorted(["00:00:00:00", "/", "01:01:00:00", "再生品質"])
    try:
        bar.resize(1500, bar.sizeHint().height())
        _settle()
        assert sorted(_shown_texts(bar)) == everything
        full = bar.sizeHint().width()

        bar.resize(bar.minimumSizeHint().width(), bar.height())
        _settle()
        assert _shown_texts(bar) == ["00:00:00:00"]
        assert _squeezed(bar) == []
        assert _overlaps(bar) == []

        # ちょうど全部が入る幅まで戻せば全部出す 隠したまま戻らないと、狭めた後は
        # 窓を広げても全体の長さが出ない
        bar.resize(full, bar.height())
        _settle()
        assert sorted(_shown_texts(bar)) == everything
        assert _overlaps(bar) == []
    finally:
        holder.close()
        shiboken6.delete(holder)


def test_a_long_effect_name_does_not_widen_the_panel(qt_application: QApplication) -> None:
    """配布スクリプトの長い名前でも、設定パネルの最小の幅は変わらず、名前の方を「…」で省く

    名前をそのまま出すと、その幅がパネルの最小の幅になり、長い名前のスクリプトを選んだ
    だけで窓が画面からはみ出す 省いた名前は指せば全部を読める
    """
    del qt_application
    effect = registry.require("blur").create()
    long_name = "とても長い名前の配布スクリプトのアニメーション効果 改訂版" * 3
    longer = _Section(long_name * 2, effect=effect, clip_id=ClipId("clip"), index=0)
    long = _Section(long_name, effect=effect, clip_id=ClipId("clip"), index=0)
    holder = QWidget()
    holder.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    try:
        long.setParent(holder)
        holder.resize(2000, 200)
        holder.show()
        # 名前の長さが倍になっても、最小の幅は変わらない
        assert long.minimumSizeHint().width() == longer.minimumSizeHint().width()
        long.resize(long.minimumSizeHint().width(), long.sizeHint().height())
        _settle()
        title = next(label for label in long.findChildren(ElidedLabel))
        assert title.text().endswith("…")
        assert title.toolTip() == long_name
        remove = next(b for b in long.findChildren(QToolButton) if b.text() == "✕")
        assert long.rect().contains(
            QRect(remove.mapTo(long, remove.rect().topLeft()), remove.size())
        )
        assert _overlaps(long) == []
    finally:
        holder.close()
        shiboken6.delete(holder)
        shiboken6.delete(longer)


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (QSize(1920, 1032), DEFAULT_SIZE),
        (QSize(1366, 720), QSize(1350, 680)),
        (QSize(1280, 672), QSize(1264, 632)),
    ],
)
def test_the_first_window_fits_the_screen(available: QSize, expected: QSize) -> None:
    """初めて開いた窓が画面からはみ出さない 前は画面の広さによらず 1440x900 で開いた"""
    assert initial_size(available) == expected
