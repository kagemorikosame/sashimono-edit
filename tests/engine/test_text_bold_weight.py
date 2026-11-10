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
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QRawFont
from PySide6.QtWidgets import QApplication

from sashimono.effects.sources import TEXT
from sashimono.engine import sources
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


def test_glyphs_from_a_fallback_family_are_bold_too() -> None:
    """壊れると、Bahnschrift の日本語のように書体に無い字だけが、太字の欄を入れても細字で出る

    可変フォントの太字は太さの軸で掛ける Qt の太字の印を立てないので、代わりの書体へ
    太字が渡らないと、同じ行の中で英字だけ太く仮名と漢字が細い字になる
    代わりに選ばれる書体は機械で違い、太る割合も違う（手元で 2.7 倍・CI で 1.26 倍）
    太字が渡らなければ 1 倍のままなので、ほかの試験と同じ 1.1 倍で見分ける
    """
    _needs("Bahnschrift Light")
    assert (
        _ink("Bahnschrift Light", JAPANESE, bold=True) > _ink("Bahnschrift Light", JAPANESE) * 1.1
    )


@pytest.mark.parametrize(
    "family", ["Bahnschrift Light", "Bahnschrift SemiBold", "Segoe UI Variable Display Light"]
)
@pytest.mark.parametrize("body", ["永田あア", "한국어", "ภาษาไทย"])
def test_fallback_glyphs_match_the_qt_bold_face(family: str, body: str) -> None:
    """壊れると、書体に無い字だけが、前の版の太字（Qt の太字の印）と違う太さの代わりの書体で出る

    代わりの書体（游ゴシック・Malgun Gothic・Leelawadee など）は可変フォントでないことが多く、
    太さの軸では太らない Qt が代わりの書体の面を、軸の太さから前と同じ太字の面へ選ぶことを、
    前の版と同じ ``setBold`` の字と比べて押さえる（字の量が同じなら同じ面）
    書体に含まれる字は比べない（そこは前の版が崩れていた所）
    """
    _needs(family)
    plain = sources._text_font({"font": family}, 96)
    if sources._weight_axis_maximum(QRawFont.fromFont(plain)) is None:
        pytest.skip(f"{family} は可変フォントでない")
    if QRawFont.fromFont(plain).supportsCharacter(body[0]):
        pytest.skip(f"{family} は {body} を持っている（代わりの書体を使わない）")
    reference = QFont(plain)
    reference.setBold(True)
    expected = _path_ink(reference, body)
    if expected == 0.0:
        pytest.skip(f"{body} を描ける代わりの書体が入っていない")
    drawn = _path_ink(sources._text_font({"font": family, "bold": True}, 96), body)
    assert drawn == pytest.approx(expected, rel=0.02)


def _path_ink(font: QFont, body: str) -> float:
    """``font`` で ``body`` を描いた字の量（画素） 代わりの書体も Qt が選ぶまま描く"""
    path = QPainterPath()
    path.addText(QPointF(0.0, 120.0), font, body)
    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.fillPath(path, QColor(255, 255, 255))
    painter.end()
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    return float(pixels.astype(np.float64).sum() / 255.0)


def test_a_series_at_the_axis_maximum_draws_as_before() -> None:
    """壊れると、軸の上限（Black の 900）の書体で太字が前の版と違う字で出る

    上限で軸を回しても字は変わらないので、前と同じく Qt の太字に任せる
    """
    _needs("Noto Serif JP Black")
    font = QFont("Noto Serif JP Black")
    font.setPixelSize(96)
    assert sources._weight_axis_maximum(QRawFont.fromFont(font)) == pytest.approx(900.0)
    drawn = sources._text_font({"font": "Noto Serif JP Black", "bold": True}, 96)
    assert drawn.bold()


def test_the_weight_axis_is_read_from_the_font_table() -> None:
    """壊れると、可変フォントを固定の書体と見なして前の崩れ（既定の太さへ戻る）に戻る"""
    _needs("Bahnschrift")
    font = QFont("Bahnschrift")
    font.setPixelSize(96)
    assert sources._weight_axis_maximum(QRawFont.fromFont(font)) == pytest.approx(700.0)
    fixed = QFont("Yu Gothic UI")
    fixed.setPixelSize(96)
    assert sources._weight_axis_maximum(QRawFont.fromFont(fixed)) is None
