"""AI のコンタクトシートと範囲のコマ（#320 A2）を本物の窓で描く

MainWindow を組み、AI の道具から 1 枚を取り、絵の大きさ・コマの並び・時刻の書き込みを
画素で確かめる 大きい作品（4K）で描く間も編集画面が手を返すことも見る

GPU が要るので ``gpu`` を付けてある GPU の無い所（CI）では飛ぶ 並べ方そのものは
``tests/ai/test_contact_sheet.py`` が GPU なしで見る

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai import operations
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import ImageResult, find_operation
from sashimono.ai.sheet import LABEL_HEIGHT, MAX_SHEET_EDGE, sheet_layout
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.ui.ai_sheet import SHEET_BACKGROUND
from sashimono.ui.main_window import MainWindow
from tests.ai.test_bridge import call_in_thread

pytestmark = pytest.mark.usefixtures("gpu")

#: 3 つの区間に敷く色 区間ごとに違う色にして、升の並びを色で読む
COLORS = ("#ff0000", "#00ff00", "#0000ff")


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 道具は読むだけなので、どの窓も開かないはず"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _window(width: int, height: int) -> MainWindow:
    settings = ProjectSettings(width=width, height=height, frame_rate=FrameRate(30))
    window = MainWindow(Project.create(settings), confirm_unsaved=False)
    # 0〜99 は赤、100〜199 は緑、200〜299 は青の画面全体の図形 人と同じ道具で置く
    for index, color in enumerate(COLORS):
        _run(
            window, "add_shape", shape="background", color=color, at_frame=index * 100, duration=100
        )
    return window


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _window(640, 360)
    yield created
    created.close()
    created.deleteLater()


@pytest.fixture
def big_window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = _window(3840, 2160)
    yield created
    created.close()
    created.deleteLater()


def _run(window: MainWindow, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(window, arguments)


def _picture(result: object) -> QImage:
    assert isinstance(result, ImageResult)
    image = QImage.fromData(result.png)
    assert not image.isNull(), "PNG として読めない"
    return image


def _near(color: QColor, expected: str) -> bool:
    want = QColor(expected)
    return (
        abs(color.red() - want.red()) < 24
        and abs(color.green() - want.green()) < 24
        and abs(color.blue() - want.blue()) < 24
    )


def _cell_color(image: QImage, layout_cell: tuple[int, int], size: tuple[int, int]) -> QColor:
    x, y = layout_cell
    width, height = size
    return image.pixelColor(x + width // 2, y + height // 2)


def _label_has_text(image: QImage, x: int, y: int, width: int) -> bool:
    """時刻の帯に字が書いてあるか 地の色と違う明るい画素を数える"""
    lit = 0
    for row in range(y, y + LABEL_HEIGHT):
        for column in range(x, x + width):
            if image.pixelColor(column, row).lightness() > 150:
                lit += 1
    return lit > 20


class TestContactSheet:
    def test_the_frames_appear_in_order_with_their_times(self, window: MainWindow) -> None:
        # 壊れると、AI が見る 1 枚の升の並びや時刻が実際と食い違い、違う所を切る
        result = _run(window, "contact_sheet", frames=[150, 50, 250], cell_width=200)
        image = _picture(result)
        layout = sheet_layout(3, cell_width=200, frame_size=(640, 360))
        assert (image.width(), image.height()) == (layout.width, layout.height)
        size = (layout.cell_width, layout.cell_height)
        for index, color in enumerate((COLORS[1], COLORS[0], COLORS[2])):
            seen = _cell_color(image, layout.cell(index), size)
            assert _near(seen, color), f"{index + 1} 番目の升が {seen.name()}"
            x, y = layout.cell(index)
            assert _label_has_text(image, x, y + layout.cell_height, layout.cell_width), (
                f"{index + 1} 番目の升の下に時刻が書いていない"
            )
        # 升の間の隙間は地の色 升が隙間なく続くと 1 枚の絵に見える
        assert image.pixelColor(1, 1) == SHEET_BACKGROUND
        assert "00:00:05:00 150" in result.caption

    def test_the_default_sheet_overviews_the_whole_timeline(self, window: MainWindow) -> None:
        # 壊れると、全体を見渡したつもりで一部の区間しか並ばず、AI が後半の場面を見落とす
        image = _picture(_run(window, "contact_sheet"))
        layout = sheet_layout(12, cell_width=240, frame_size=(640, 360))
        assert (image.width(), image.height()) == (layout.width, layout.height)
        size = (layout.cell_width, layout.cell_height)
        seen = [_cell_color(image, layout.cell(index), size) for index in range(12)]
        # 300 フレームを 12 等分した真ん中 4 枚ずつ赤・緑・青
        expected = [COLORS[0]] * 4 + [COLORS[1]] * 4 + [COLORS[2]] * 4
        assert [_near(color, want) for color, want in zip(seen, expected, strict=True)] == [
            True
        ] * 12

    def test_drawing_leaves_the_project_and_the_playhead_alone(self, window: MainWindow) -> None:
        # 壊れると、AI が見ただけで履歴が積まれるか再生ヘッドが飛び、人の作業位置が失われる
        window.seek(42)
        before = window.document.project
        labels = window.document.history_labels
        _run(window, "contact_sheet", count=6)
        _run(window, "preview_range", start_frame=95, count=4)
        assert window.document.project == before
        assert window.document.history_labels == labels
        assert window.playhead == 42


class TestPreviewRange:
    def test_consecutive_frames_show_the_cut(self, window: MainWindow) -> None:
        # 壊れると、切り替わりの前後を並べても動きが読めず、AI が場面切り替えの出来を確かめられない
        result = _run(window, "preview_range", start_frame=97, count=6)
        image = _picture(result)
        layout = sheet_layout(6, cell_width=320, frame_size=(640, 360), columns=4)
        assert (image.width(), image.height()) == (layout.width, layout.height)
        assert (layout.columns, layout.rows) == (4, 2)
        size = (layout.cell_width, layout.cell_height)
        seen = [_cell_color(image, layout.cell(index), size) for index in range(6)]
        expected = [COLORS[0]] * 3 + [COLORS[1]] * 3
        assert [_near(color, want) for color, want in zip(seen, expected, strict=True)] == [
            True
        ] * 6
        assert "00:00:03:07 97" in result.caption


class TestBigProject:
    def test_a_4k_sheet_stays_small(self, big_window: MainWindow) -> None:
        # 壊れると、4K の作品では 1 枚が作品の大きさで描かれ、費用が膨らむか上限で縮んで字が潰れる
        image = _picture(_run(big_window, "contact_sheet", count=36))
        assert max(image.width(), image.height()) <= MAX_SHEET_EDGE
        layout = sheet_layout(36, cell_width=240, frame_size=(3840, 2160))
        assert (image.width(), image.height()) == (layout.width, layout.height)
        size = (layout.cell_width, layout.cell_height)
        assert _near(_cell_color(image, layout.cell(0), size), COLORS[0])
        assert _near(_cell_color(image, layout.cell(35), size), COLORS[2])

    def test_the_editor_keeps_handing_back_while_drawing(
        self, big_window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、大きい作品で何十コマを描く間、画面が止まり、押した釦も中断も効かない
        bridge = EditorBridge(big_window)
        # 初めの 1 枚は描き手（GL の文脈）を作るので重い 区切りの見張りから外す
        _run(big_window, "preview_frame", frame=0)
        # 1 コマを重くして（4K の動画のシークのつもり）、1 回で描き切れば必ず長くなるようにする
        slow = 0.03
        real = big_window._render_for_ai

        def heavy(project: Project, frame: int, divisor: int) -> Any:
            time.sleep(slow)
            return real(project, frame, divisor)

        monkeypatch.setattr(big_window, "_render_for_ai", heavy)
        operation = find_operation("contact_sheet")
        assert operation is not None
        done, box = call_in_thread(bridge, operation, {"count": 36})
        longest = 0.0
        turns = 0
        deadline = time.monotonic() + 60.0
        while not done.is_set() and time.monotonic() < deadline:
            started = time.monotonic()
            bridge.pump()
            QApplication.processEvents()
            longest = max(longest, time.monotonic() - started)
            turns += 1
            time.sleep(0.002)
        assert done.is_set(), "描き終わらない"
        assert isinstance(box.get("result"), ImageResult), box.get("error")
        # 36 コマを 1 回で描くと 1 秒を超える 区切っていれば 0.1 秒と数コマぶんで手を返す
        assert longest < operations.SHEET_SLICE_SECONDS + 0.4
        assert turns > 5
