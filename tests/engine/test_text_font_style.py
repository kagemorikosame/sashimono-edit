"""テキストのフォントのスタイル（ファミリの中の太さの段階など #248）が描いた絵に効くか

前はファミリ名に太字と斜体を掛けるだけで、ファミリの中のスタイル（Yu Gothic UI の
Light、ルイカの ０９ など）を選べなかった 描いた絵の字の量（不透明な画素の和）で、
細いスタイルほど少なく、太いスタイルほど多くなることを見る
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from sashimono.core.model import GeneratedSource
from sashimono.effects.sources import TEXT
from sashimono.engine.render.scripts import text_font
from sashimono.engine.sources import render_source

WIDTH, HEIGHT = 480, 200
#: Windows 10 と 11 なら必ず入っているファミリ CI の実行機でも同じに試せる
FAMILY = "Yu Gothic UI"
#: ファミリの中にスタイルとして入っている物（OS が別のファミリ名でも出す物とは別）
LIGHT, BOLD = "Light", "Bold"
TEXT_BODY = "あア亜Ag"


@pytest.fixture(autouse=True)
def _needs_family(qt_application: QApplication) -> None:
    del qt_application
    styles = QFontDatabase.styles(FAMILY)
    if LIGHT not in styles or BOLD not in styles:
        pytest.skip(f"{FAMILY} の {LIGHT} と {BOLD} が入っていない")


def _ink(**params: object) -> float:
    """字の量 不透明度の和（画素）"""
    image = render_source(
        TEXT.create(text=TEXT_BODY, font=FAMILY, size=96, **params),  # type: ignore[arg-type]
        WIDTH,
        HEIGHT,
    )
    assert image is not None
    return float(image[..., 3].astype(np.float64).sum() / 255.0)


def _image(**params: object) -> np.ndarray:
    image = render_source(
        TEXT.create(text=TEXT_BODY, font=FAMILY, size=96, **params),  # type: ignore[arg-type]
        WIDTH,
        HEIGHT,
    )
    assert image is not None
    return image


class TestStyleChangesTheGlyphs:
    def test_light_is_thinner_and_bold_is_thicker_than_the_default(self) -> None:
        # 壊れると、スタイルを選んでも既定と同じ太さで出る（選んだ意味が無い）
        default = _ink()
        assert _ink(font_style=LIGHT) < default * 0.9
        assert _ink(font_style=BOLD) > default * 1.1

    def test_vertical_text_uses_the_same_style(self) -> None:
        # 縦書きは別の関数で 1 文字ずつ置く そちらでスタイルを落とすと縦書きだけ既定で出る
        default = _ink(vertical=True)
        assert _ink(vertical=True, font_style=LIGHT) < default * 0.9

    def test_a_bold_style_is_placed_like_the_bold_check(self) -> None:
        # 太字は細字で置いたときの字の外形の中心へ戻す（``_bold_drift``） スタイルで太くしたとき
        # だけ戻さないと、同じ Bold の字が太字の欄で選んだときと横にずれて出る
        for align in ("left", "center", "right"):
            assert np.array_equal(
                _image(font_style=BOLD, align=align), _image(bold=True, align=align)
            ), align

    def test_the_style_wins_over_the_bold_check(self) -> None:
        # 太さを持つスタイルへ太字を重ねると二重に太くなる 選んだスタイルのまま描く
        assert np.array_equal(
            _image(font_style=LIGHT, bold=True, italic=True), _image(font_style=LIGHT)
        )


class TestOldLookIsKept:
    def test_a_project_without_the_item_draws_as_before(self) -> None:
        # 前の版のプロジェクトは項目そのものを持たない 太字の欄で今までどおり太くなる
        params = dict(TEXT.create(text=TEXT_BODY, font=FAMILY, size=96, bold=True).params)
        del params["font_style"]
        old = render_source(GeneratedSource(kind="text", params=params), WIDTH, HEIGHT)
        assert old is not None
        assert np.array_equal(old, _image(bold=True))
        assert _ink(bold=True) > _ink() * 1.1

    def test_a_missing_style_falls_back_to_the_default_style(self) -> None:
        # フォントを入れていない機械で開いた・ファミリを替えたとき 落ちずに既定の字で出る
        # 太字の欄は灰色のまま（スタイルを選んでいる）なので、太字も掛けない
        assert np.array_equal(_image(font_style="無いスタイル", bold=True), _image())

    def test_the_aviutl_layout_ignores_the_style(self) -> None:
        # AviUtl2 の組み方はファミリ名と太字・斜体だけで字を選ぶ（設定パネルも欄を灰色にする）
        assert np.array_equal(_image(layout="aviutl", font_style=LIGHT), _image(layout="aviutl"))


class TestEmbeddedLuaSeesTheDrawnFont:
    """埋め込み Lua の ``obj.getfont`` へ渡す太字と斜体を、描いた字に合わせる

    スタイルを選んでいると太字と斜体の欄は灰色で効かない その古い値を渡すと、
    描いた字は変わらないのに展開した結果だけが変わる
    """

    @staticmethod
    def _font(**params: object) -> tuple[bool, bool, bool, bool]:
        font = text_font(TEXT.create(font=FAMILY, **params).params, 0)  # type: ignore[arg-type]
        given = font["given"]
        return font["bold"], font["italic"], given[5], given[6]

    def test_greyed_bold_and_italic_are_not_passed(self) -> None:
        assert self._font(font_style=LIGHT, bold=True, italic=True) == (False,) * 4

    def test_a_bold_style_is_passed_as_bold(self) -> None:
        assert self._font(font_style=BOLD) == (True, False, True, False)

    def test_without_a_style_the_checks_are_passed_as_before(self) -> None:
        assert self._font(bold=True, italic=True) == (True,) * 4
        assert self._font() == (False,) * 4

    def test_the_aviutl_layout_passes_the_checks(self) -> None:
        # AviUtl2 の組み方の横書きはスタイルを読まず、欄の太字と斜体で描く
        assert self._font(layout="aviutl", font_style=LIGHT, bold=True) == (
            True,
            False,
            True,
            False,
        )


class TestRealFamilyWithNumberedStyles:
    """スタイルが数字の名前で入っているフォント（Issue の画像の「ルイカ」）

    フォントの入っていない機械では飛ばす 配布物なのでリポジトリには入れない
    """

    @pytest.mark.parametrize(
        ("family", "thin", "thick"),
        [("Ruika", "05", "09"), ("ルイカ", "０５", "０９")],
    )
    def test_numbered_styles_change_the_weight(self, family: str, thin: str, thick: str) -> None:
        styles = QFontDatabase.styles(family)
        if thin not in styles or thick not in styles:
            pytest.skip(f"{family} の {thin} と {thick} が入っていない")

        def ink(style: str) -> float:
            source = TEXT.create(text=TEXT_BODY, font=family, size=96, font_style=style)
            image = render_source(source, WIDTH, HEIGHT)
            assert image is not None
            return float(image[..., 3].astype(np.float64).sum())

        assert ink(thick) > ink(thin) * 1.1
