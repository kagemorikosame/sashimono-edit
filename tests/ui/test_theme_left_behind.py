"""試験が変えたまま終えたテーマを、次の試験へ持ち越さない（tests/conftest.py の theme_left_behind）

前の試験が変えた見た目のまま次の試験が走ると、どの試験が同じワーカーに並ぶかで落ちたり
通ったりする（CI の run 38041203074 明るいテーマのまま帯の色を比べて落ちた）
2 本ずつ組にし、前の 1 本が変えたまま終え、後の 1 本で戻っていることを見る 並列でも
同じワーカーで続けて走るよう、組を xdist の同じ群にする
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, current_theme, use_palette

#: 前の 1 本がスタイルシートの末尾に足す印
_MARK = "/* theme-left-behind */"

pytestmark = pytest.mark.xdist_group("theme_left_behind")


def test_a_test_that_only_changes_the_style_sheet(qt_application: QApplication) -> None:
    # テーマの選び方も色も変えず、スタイルシートだけを書き換えたまま終える
    qt_application.setStyleSheet(qt_application.styleSheet() + _MARK)
    assert _MARK in qt_application.styleSheet()


def test_the_next_test_gets_the_style_sheet_back(qt_application: QApplication) -> None:
    # 残ると、書き換えた縁や余白のまま後の試験が部品の大きさや画素を見て落ちる
    assert _MARK not in qt_application.styleSheet()


def test_a_test_that_leaves_the_light_colours(qt_application: QApplication) -> None:
    del qt_application
    use_palette(THEME_LIGHT)
    assert current_theme() == THEME_LIGHT


def test_the_next_test_gets_the_dark_colours_back(qt_application: QApplication) -> None:
    # 残ると、帯の色を明るいテーマの目盛りの色と比べて落ちた（test_work_area_ui）
    del qt_application
    assert current_theme() == THEME_DARK
