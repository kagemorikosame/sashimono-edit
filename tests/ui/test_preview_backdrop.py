"""プレビューの何も映らない所の設定（#456）と、画面へのつなぎ

絵そのもの（地が敷かれること・書き出しが黒のままなこと）は tests/engine/test_preview_backdrop.py
ここで見るのは、設定が残ること・窓に出ること・プレビューの貯めまで届くこと
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication

from sashimono.engine.render.backdrop import (
    BLACK,
    FILL_BLACK,
    FILL_CHECKER,
    FILL_MODES,
    MARGIN_BLACK,
    MARGIN_GRAY,
    MARGIN_GRAY_COLOR,
    MARGIN_MODES,
    MARGIN_THEME,
    Backdrop,
    fill_backdrop,
    hex_rgb,
)
from sashimono.ui.library_view import BACKDROP_CHECKER, BACKDROP_DARK, BACKDROP_LIGHT
from sashimono.ui.preferences_dialog import (
    PREVIEW_FILL_CHOICES,
    PREVIEW_MARGIN_CHOICES,
    PreferencesDialog,
)
from sashimono.ui.preview import PreviewWidget
from sashimono.ui.theme import Colors
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ui.test_preview_prefetch import _project


class _Recorder:
    """地を受け取った回数を覚える 貯めの置き場と走り係の代わり"""

    def __init__(self) -> None:
        self.backdrops: list[Backdrop] = []

    def set_backdrop(self, backdrop: Backdrop) -> None:
        self.backdrops.append(backdrop)


@pytest.fixture
def preview(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[PreviewWidget, _Recorder, _Recorder]]:
    del qt_application
    project, _, _ = _project()
    widget = PreviewWidget(project, prefetch_bytes=0, prefetch_thread=False)
    cache, background = _Recorder(), _Recorder()
    monkeypatch.setattr(widget, "_cache", cache)
    monkeypatch.setattr(widget, "_background", background)
    monkeypatch.setattr(widget, "_restart_prefetch", lambda: None)
    yield widget, cache, background
    shiboken6.delete(widget)


class TestThePreferences:
    def test_the_defaults_keep_the_output_look(self) -> None:
        """既定の地は黒、余白は暗い灰色

        地の既定を変えると、何も設定していない人のプレビューが書き出しと食い違う
        余白が黒のままだと、黒い映像の端がどこか分からないという困りごとが残る
        """
        preferences = Preferences()
        assert preferences.preview_fill == FILL_BLACK
        assert preferences.preview_margin == MARGIN_GRAY
        assert fill_backdrop(preferences.preview_fill) == BLACK

    def test_they_come_back(self, tmp_path: Path) -> None:
        # 残らないと、起動するたびに黒へ戻って選び直しになる
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(preview_fill=FILL_CHECKER, preview_margin=MARGIN_THEME))
        loaded = store.load()
        assert loaded.preview_fill == FILL_CHECKER
        assert loaded.preview_margin == MARGIN_THEME

    def test_an_unknown_value_falls_back(self, tmp_path: Path) -> None:
        # 知らない値で起動を止めたり、知らない値のまま描いたりしない 項目ごとに既定へ戻す
        path = tmp_path / "preferences.json"
        path.write_text('{"preview_fill": "purple", "preview_margin": 3}', encoding="utf-8")
        loaded = PreferenceStore(path).load()
        assert loaded.preview_fill == FILL_BLACK
        assert loaded.preview_margin == MARGIN_GRAY

    def test_the_dialog_offers_every_choice(self, qt_application: QApplication) -> None:
        """窓の選択肢は値の一覧と同じで、選んだ物が OK で返る

        一覧に足した値を窓に出し忘れると、選べない値が残る 返し忘れると OK で既定へ戻る
        """
        del qt_application
        assert {value for value, _ in PREVIEW_FILL_CHOICES} == set(FILL_MODES)
        assert {value for value, _ in PREVIEW_MARGIN_CHOICES} == set(MARGIN_MODES)
        dialog = PreferencesDialog(Preferences(preview_fill=FILL_CHECKER))
        try:
            assert dialog.preferences().preview_fill == FILL_CHECKER
            dialog._preview_margin.setCurrentIndex(dialog._preview_margin.findData(MARGIN_BLACK))
            assert dialog.preferences().preview_margin == MARGIN_BLACK
        finally:
            shiboken6.delete(dialog)

    def test_the_fills_match_the_library_backdrops(self) -> None:
        # 見本の絵の地と値の名前がずれると、同じ「市松」を選んだつもりで別の値が残る
        assert {BACKDROP_CHECKER, BACKDROP_DARK, BACKDROP_LIGHT} <= set(FILL_MODES)


class TestThePreviewWidget:
    def test_a_new_backdrop_reaches_both_caches(
        self, preview: tuple[PreviewWidget, _Recorder, _Recorder]
    ) -> None:
        """地を変えたら、画面の側の貯めにも走り係にも渡す

        片方へ渡し忘れると、そちらで貯めた絵だけ前の地のまま出て、再生と停止で見え方が変わる
        """
        widget, cache, background = preview
        checker = fill_backdrop(FILL_CHECKER)
        widget.set_backdrop(checker, MARGIN_GRAY)
        assert cache.backdrops == [checker]
        assert background.backdrops == [checker]
        assert widget.backdrop == checker

    def test_the_same_backdrop_is_not_sent_again(
        self, preview: tuple[PreviewWidget, _Recorder, _Recorder]
    ) -> None:
        # 設定の窓で OK を押すたびに貯めを捨てると、何も変えていないのに描き直しで重くなる
        widget, cache, background = preview
        widget.set_backdrop(BLACK, MARGIN_BLACK)
        assert cache.backdrops == [] and background.backdrops == []
        assert widget.margin == MARGIN_BLACK

    def test_the_margin_colors(self, preview: tuple[PreviewWidget, _Recorder, _Recorder]) -> None:
        """余白の色 テーマに合わせるときはプレビューの周りの色を、描くたびに読む

        先に読んで持っておくと、テーマを切り替えても余白だけ前のテーマの色のまま残る
        """
        widget, _, _ = preview
        widget.set_backdrop(BLACK, MARGIN_GRAY)
        assert widget.margin_color() == hex_rgb(MARGIN_GRAY_COLOR)
        widget.set_backdrop(BLACK, MARGIN_BLACK)
        assert widget.margin_color() == (0.0, 0.0, 0.0)
        widget.set_backdrop(BLACK, MARGIN_THEME)
        theme = Colors.VIEWER_BACKGROUND
        assert widget.margin_color() == (theme.redF(), theme.greenF(), theme.blueF())
