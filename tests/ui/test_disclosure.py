"""押すと欄が開き、もう一度押すと閉じるボタン（AI パネルの〔AI の部品〕）

ふつうのボタンと見分けが付かず、開け閉めのボタンだと分からなかった（利用者の確認）
押下の見た目・向きの印・ツールチップ・欄の見出しと閉じる印が、開け閉めに合わせて
変わるのを見る
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.ai import AI_PACK
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.runtime import PackageStatus, PackStatus
from sashimono.ui.chat import ChatPanel
from sashimono.ui.disclosure import CLOSED_MARK, OPEN_MARK, DisclosureButton
from sashimono.ui.setup import SetupSection
from sashimono.ui.theme import PALETTES, THEME_DARK, THEME_LIGHT, use_palette
from tests.ai.conftest import FakeHost, make_loaded


def _ready(monkeypatch: pytest.MonkeyPatch, version: str | None = "0.2.164") -> None:
    packages = (PackageStatus("claude-agent-sdk>=0.2.158,<0.3", version, "0.2.158"),)
    status = PackStatus(pack=AI_PACK, packages=packages)
    monkeypatch.setattr(SetupSection, "status", property(lambda _self: status))


@pytest.fixture
def loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    return make_loaded(video_media, transcript)


@pytest.fixture
def panel(
    qt_application: QApplication, loaded: Project, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ChatPanel]:
    del qt_application
    _ready(monkeypatch)
    created = ChatPanel(FakeHost(loaded))
    yield created
    created.close_session()
    created.deleteLater()


def _has_color(widget: QWidget, color: QColor) -> bool:
    widget.ensurePolished()
    widget.resize(widget.sizeHint())
    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    widget.render(image)
    wanted = color.rgb()
    return any(
        image.pixel(x, y) == wanted for x in range(image.width()) for y in range(image.height())
    )


class TestDisclosureButton:
    def test_the_mark_and_the_pressed_state_follow_opening(
        self, qt_application: QApplication
    ) -> None:
        # 文字だけだと、押した後に開いているのか閉じているのか分からない
        del qt_application
        button = DisclosureButton("AI の部品", "開く / 閉じる")
        try:
            assert button.isCheckable() is True
            assert button.text() == f"{CLOSED_MARK} AI の部品"
            button.click()
            assert button.isChecked() is True
            assert button.text() == f"{OPEN_MARK} AI の部品"
            button.click()
            assert button.text() == f"{CLOSED_MARK} AI の部品"
        finally:
            button.deleteLater()

    @pytest.mark.parametrize("theme", [THEME_DARK, THEME_LIGHT])
    def test_the_open_state_looks_pressed_in_both_themes(
        self, qt_application: QApplication, theme: str
    ) -> None:
        # 押下の見た目がテーマの地に溶けると、開いているのか分からない
        del qt_application
        try:
            use_palette(theme)
            button = DisclosureButton("AI の部品", "開く / 閉じる")
            accent = PALETTES[theme]["ACCENT"]
            assert _has_color(button, accent) is False
            button.set_open(True)
            assert _has_color(button, accent) is True
            button.deleteLater()
        finally:
            use_palette(THEME_DARK)

    def test_set_open_does_not_announce(self, qt_application: QApplication) -> None:
        # 欄の側の都合で合わせただけなのに知らせると、受け手が欄を開け閉めし直す
        del qt_application
        button = DisclosureButton("AI の部品", "開く / 閉じる")
        heard: list[bool] = []
        button.toggled.connect(heard.append)
        button.set_open(True)
        assert heard == []
        assert button.text().startswith(OPEN_MARK)
        button.deleteLater()


class TestPartsSection:
    def test_it_starts_closed(self, panel: ChatPanel) -> None:
        # 開け閉めは覚えない 使える間は閉じた状態から始める
        assert panel._parts_box.isHidden() is True
        assert panel._parts_button.isChecked() is False
        assert panel._parts_button.toolTip() == "AI の部品の版と更新を開く / 閉じる"

    def test_the_button_opens_and_closes_the_section(self, panel: ChatPanel) -> None:
        panel._parts_button.click()
        assert panel._parts_box.isHidden() is False
        assert panel._parts_button.text().startswith(OPEN_MARK)
        panel._parts_button.click()
        assert panel._parts_box.isHidden() is True
        assert panel._parts_button.text().startswith(CLOSED_MARK)

    def test_the_close_mark_in_the_section_closes_it(self, panel: ChatPanel) -> None:
        # 欄の中にも閉じる手がかりが無いと、上のボタンへ戻らないと閉じられない
        panel._parts_button.click()
        assert panel._parts_close.isHidden() is False
        panel._parts_close.click()
        assert panel._parts_box.isHidden() is True
        assert panel._parts_button.isChecked() is False
        assert panel._parts_button.text().startswith(CLOSED_MARK)

    def test_it_stays_open_while_the_parts_are_missing(
        self, qt_application: QApplication, loaded: Project, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 入っていない間は導入の欄が入口 閉じられると入れる所が無くなる
        del qt_application
        _ready(monkeypatch, version=None)
        widget = ChatPanel(FakeHost(loaded))
        try:
            assert widget._parts_box.isHidden() is False
            assert widget._parts_button.isChecked() is True
            assert widget._parts_close.isHidden() is True
            widget._on_parts_toggled(False)
            assert widget._parts_box.isHidden() is False
        finally:
            widget.close_session()
            widget.deleteLater()
