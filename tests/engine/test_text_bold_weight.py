"""可変フォントの太さの系列へ太字を掛けても、元の太さより細くならないこと（#419）

Windows は可変フォントの名前付きの太さを「Noto Serif JP Medium」のような別のファミリとして
並べる そのファミリへ太字を掛けると、Qt は太字の面を探しても見つからず、可変フォントの
既定の太さ（Noto Serif JP なら ExtraLight）へ戻してから合成の太字で太らせていた
Medium に太字を掛けた字が、太字を外した Medium より細く出る

Noto の書体は CI に入っている保証が無いので、Windows に必ず入っている可変フォントの
Bahnschrift でも同じ崩れ（既定の太さへ戻る）を確かめる 入っていない書体は飛ばす
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from sashimono.effects.sources import TEXT
from sashimono.engine.sources import render_source

WIDTH, HEIGHT = 640, 200
LATIN = "HAMBxe"
JAPANESE = "永田あア"

#: (太さの系列のファミリ, 同じ可変フォントの太さを持たない元のファミリ, 描く字)
#: 元のファミリへ太字を掛けた字は前から正しい Bold の面で出ていたので、それを物差しにする
SERIES = [
    ("Noto Serif JP Medium", "Noto Serif JP", JAPANESE),
    ("Noto Serif JP Light", "Noto Serif JP", JAPANESE),
    ("Noto Serif JP ExtraLight", "Noto Serif JP", JAPANESE),
    ("Noto Serif JP SemiBold", "Noto Serif JP", JAPANESE),
    ("Noto Sans JP Medium", "Noto Sans JP", JAPANESE),
    ("Noto Sans JP Light", "Noto Sans JP", JAPANESE),
    ("Bahnschrift Light", "Bahnschrift", LATIN),
    ("Bahnschrift SemiLight", "Bahnschrift", LATIN),
    ("Bahnschrift SemiBold", "Bahnschrift", LATIN),
    ("Segoe UI Variable Display Light", "Segoe UI Variable Display", LATIN),
]


@pytest.fixture(autouse=True)
def _qt(qt_application: QApplication) -> None:
    del qt_application


def _needs(*families: str) -> None:
    installed = set(QFontDatabase.families())
    missing = [family for family in families if family not in installed]
    if missing:
        pytest.skip(f"{'・'.join(missing)} が入っていない")


def _ink(family: str, body: str, **params: object) -> float:
    """字の量 不透明度の和（画素）"""
    image = render_source(
        TEXT.create(text=body, font=family, size=96, **params),  # type: ignore[arg-type]
        WIDTH,
        HEIGHT,
    )
    assert image is not None
    return float(image[..., 3].astype(np.float64).sum() / 255.0)


@pytest.mark.parametrize(("series", "family", "body"), SERIES)
def test_bold_is_never_thinner_than_the_plain_weight(series: str, family: str, body: str) -> None:
    """壊れると、Medium に太字を掛けた字が太字を外した Medium より細く出る

    SemiBold は前は太字を掛けても何も変わらなかった（Qt が 600 を太字に数える）
    太字の欄を入れて字が変わらないと、効いていないように見えるので、太くなることまで見る
    """
    _needs(series)
    assert _ink(series, body, bold=True) > _ink(series, body) * 1.05


@pytest.mark.parametrize(("series", "family", "body"), SERIES)
def test_bold_of_a_weight_series_is_the_family_bold(series: str, family: str, body: str) -> None:
    """壊れると、系列ごとの太字が既定の太さを合成で太らせた字になり、同じ書体の Bold と違って出る

    太字の太さは系列に依らず Bold の面（700）になる 元の系列が 700 より太い（Black）
    ときだけ、そのままの太さを保つ
    """
    _needs(series, family)
    expected = _ink(family, body, bold=True)
    assert _ink(series, body, bold=True) == pytest.approx(expected, rel=0.02)


def test_a_heavier_series_keeps_its_weight() -> None:
    """壊れると、Black に太字を掛けた字が Bold（700）まで細くなる"""
    _needs("Noto Serif JP Black")
    plain = _ink("Noto Serif JP Black", JAPANESE)
    assert _ink("Noto Serif JP Black", JAPANESE, bold=True) >= plain * 0.99


def test_vertical_text_uses_the_same_bold() -> None:
    """縦書きは別の関数で 1 字ずつ置く そちらで太字の直しを落とすと縦書きだけ細く出る"""
    _needs("Bahnschrift SemiBold", "Bahnschrift")
    expected = _ink("Bahnschrift", LATIN, bold=True, vertical=True)
    drawn = _ink("Bahnschrift SemiBold", LATIN, bold=True, vertical=True)
    assert drawn == pytest.approx(expected, rel=0.02)
