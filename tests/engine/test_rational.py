"""PyAV の有理数を Fraction にする助け

PyAV 19 の ``AVRational`` は ``Fraction`` ではないので、型の上で ``Fraction(...)`` に
渡せず CI の mypy が落ちた 分子と分母だけを見て作れば、どちらの版でも同じ値になる
"""

from __future__ import annotations

from fractions import Fraction

from sashimono.engine.decode.rational import as_fraction


class _Rational:
    """``AVRational`` と同じく分子と分母だけを持つ（``Fraction`` の仲間ではない）"""

    def __init__(self, numerator: int, denominator: int) -> None:
        self._numerator = numerator
        self._denominator = denominator

    @property
    def numerator(self) -> int:
        return self._numerator

    @property
    def denominator(self) -> int:
        return self._denominator


def test_a_fraction_stays_the_same() -> None:
    assert as_fraction(Fraction(1, 15360)) == Fraction(1, 15360)


def test_a_bare_rational_keeps_its_value() -> None:
    # 壊れると、時刻の単位が丸まりシークの位置やクリップの長さがずれる
    value = as_fraction(_Rational(30000, 1001))
    assert isinstance(value, Fraction)
    assert value == Fraction(30000, 1001)
