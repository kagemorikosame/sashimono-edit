"""PySide6 による編集画面

``MainWindow`` は引かれたときに初めて読む（#526） ここで読むと ``sashimono.ui`` の下の
どの部品を読んでも編集画面の部品（numpy・PyAV・OpenGL）まで一緒に読み、起動画面
（:mod:`sashimono.ui.splash`）とテーマを、重い部品を読み終えるまで出せない
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sashimono.ui.main_window import MainWindow

__all__ = ["MainWindow"]


def __getattr__(name: str) -> object:
    if name == "MainWindow":
        from sashimono.ui.main_window import MainWindow

        return MainWindow
    # 読み込み済みの下の部品は名前で返す 試験が部品を読み直したとき（``sys.modules`` から
    # ``sashimono.ui`` だけを外して読み直すなど）、下の部品は読み込み済みのまま新しい頭に
    # 結び付かない 前は頭が main_window を読むときに結び直していたので、引けなくなると
    # ``monkeypatch.setattr("sashimono.ui.preview....")`` が落ちる
    loaded = sys.modules.get(f"{__name__}.{name}")
    if loaded is not None:
        return loaded
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
