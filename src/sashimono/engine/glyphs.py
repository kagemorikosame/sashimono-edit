"""書体に無い字を見つけ、代わりの書体で出る字の太さをそろえる（#494）

Qt は書体に無い字を、何も知らせずに代わりの書体の標準の太さで描く 太い見出し用の書体では
その字だけが細く浮き、気付かずに書き出す 見つける所（検査・設定パネル）と太らせる所（描く所）が
同じ答えを使うよう、ここにまとめる
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from PySide6.QtCore import QPointF
from PySide6.QtGui import (
    QFont,
    QFontMetricsF,
    QPainterPath,
    QPainterPathStroker,
    QRawFont,
    QTextLayout,
)

from sashimono.engine.text_wrap import graphemes

__all__ = ["absent_characters", "thickened"]

#: 太さを測る字の大きさ（画素） 小さいと画素への丸めで太さの差が読めない
_SAMPLE_SIZE = 120
#: 太さを測る字 書体ごとに持っている物だけで測る 漢字・かな・英数の順に試す
_SAMPLES = ("田永書", "あのま", "AHO8")
#: これより差が小さければ太らせない（字の大きさに対する割合） 揃っている字まで滲ませない
_SAME_WEIGHT = 0.01
#: ここから先は絵文字 代わりの書体で出すのがふつうで、知らせも太らせもしない
_EMOJI_FROM = 0x1F000


def _expected(character: str) -> bool:
    """書体が持っているはずの字か 空白・制御・異体字の選び・絵文字は数えない"""
    if character.isspace() or ord(character) >= _EMOJI_FROM:
        return False
    category = unicodedata.category(character)
    return category not in ("Cc", "Cf", "Cs", "Mn", "So")


def absent_characters(font: QFont, text: str) -> str:
    """``text`` のうち ``font`` の書体に無い字を、出た順に 1 つずつ"""
    raw = QRawFont.fromFont(font)
    if not raw.isValid():
        return ""
    found: dict[str, None] = {}
    for character in text:
        if _expected(character) and not raw.supportsCharacter(ord(character)):
            found.setdefault(character)
    return "".join(found)


def _thickness(path: QPainterPath) -> float:
    """線の太さの見積もり（面積の 2 倍を輪郭の長さで割る）"""
    length = path.length()
    if length <= 0:
        return 0.0
    area = 0.0
    # 穴（口の中）は外の輪郭と逆に回る 符号のまま足すと引かれる
    for contour in path.toSubpathPolygons():
        points = [(point.x(), point.y()) for point in contour.toList()]
        following = [*points[1:], *points[:1]]
        area += sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, following, strict=True))
    return abs(area) / length


def _drawn_family(font: QFont, text: str) -> str:
    """``text`` を実際に描く書体（代わりに選ばれた物）"""
    layout = QTextLayout(text, font)
    layout.beginLayout()
    layout.createLine()
    layout.endLayout()
    runs = layout.glyphRuns()
    return runs[0].rawFont().familyName() if runs else ""


@lru_cache(maxsize=256)
def _family_thickness(key: str, family: str) -> float:
    font = QFont()
    if not font.fromString(key):
        return 0.0
    if family:
        font.setFamilies([family])
    font.setPixelSize(_SAMPLE_SIZE)
    for sample in _SAMPLES:
        if absent_characters(font, sample):
            continue
        path = QPainterPath()
        path.addText(QPointF(0.0, 0.0), font, sample)
        return _thickness(path) / _SAMPLE_SIZE
    return 0.0


def thickened(font: QFont, text: str, origin: QPointF) -> QPainterPath | None:
    """``origin`` に組んだ ``text`` へ足す、書体に無い字を太らせた輪郭 要らなければ ``None``

    足りない太さの分だけ縁を足す 送り幅は変えない 細いままの代わりの字より太くするだけで、
    字の位置と折り返しが動くと、欄を入れただけで組んだ見た目が崩れる
    """
    absent = absent_characters(font, text)
    if not absent:
        return None
    size = max(1, font.pixelSize())
    sample = QFont(font)
    sample.setPixelSize(_SAMPLE_SIZE)
    # 大きさを鍵に入れない 大きさの動くテキストで毎コマ測り直しになる
    key = sample.toString()
    own = _family_thickness(key, "")
    added = QPainterPath()
    consumed = ""
    for part in graphemes(text):
        before = consumed
        consumed += part
        if not any(character in absent for character in part):
            continue
        family = _drawn_family(font, part)
        theirs = _family_thickness(key, family) if family else 0.0
        short = own - theirs
        if theirs <= 0 or short <= _SAME_WEIGHT:
            continue
        glyph = QPainterPath()
        glyph.addText(QPointF(origin.x() + _advance(font, before), origin.y()), font, part)
        stroker = QPainterPathStroker()
        stroker.setWidth(short * size)
        added.addPath(glyph.united(stroker.createStroke(glyph)))
    return None if added.isEmpty() else added


def _advance(font: QFont, text: str) -> float:
    return QFontMetricsF(font).horizontalAdvance(text) if text else 0.0
