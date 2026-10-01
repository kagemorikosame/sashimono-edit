"""PyAV の有理数（時刻の単位・フレームレート）を :class:`~fractions.Fraction` にする

PyAV は 19 から ``time_base`` や ``average_rate`` の型を ``AVRational`` にした（18 までの型の
書き方は ``Fraction``） 中身は分子と分母を持つ有理数で、実行では ``Fraction(...)`` に
そのまま渡せるが、型の上では ``Fraction`` の受け取れる形ではないので mypy（strict）が
落ちた（CI が 19 を入れて分かった） どちらの版でも分子と分母から作る 1 か所にまとめる
"""

from __future__ import annotations

from fractions import Fraction
from typing import Protocol

__all__ = ["Ratio", "as_fraction"]


class Ratio(Protocol):
    """分子と分母を持つ有理数（``Fraction`` と PyAV の ``AVRational`` のどちらも満たす）"""

    @property
    def numerator(self) -> int: ...

    @property
    def denominator(self) -> int: ...


def as_fraction(value: Ratio) -> Fraction:
    """有理数を ``Fraction`` へ 分子と分母から作るので、値は丸めずにそのまま"""
    return Fraction(int(value.numerator), int(value.denominator))
