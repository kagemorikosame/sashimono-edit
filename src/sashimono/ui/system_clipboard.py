"""OS のクリップボードの入口

アプリからクリップボードへ書く所は、どれもここの :func:`clipboard` を通す
``QApplication.clipboard()`` を各所で直に呼ぶと、試験で差し替える口が無く、
試験を走らせるたびに本人がコピーしていた物が消える 同時に走るほかの作業と
取り合うと読み戻しが空になり、試験が機械の様子で落ちる（#154）

クリップの切り取り・貼り付けはアプリの中だけの話で、:mod:`sashimono.core.clipboard`
が持つ こちらは OS と受け渡す物だけ
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtGui import QGuiApplication, QImage

__all__ = ["Clipboard", "clipboard", "replace"]


class Clipboard(Protocol):
    """アプリが使う分だけの ``QClipboard`` の形 試験の偽物もこれに合わせる"""

    def setText(self, text: str, /) -> None: ...  # noqa: N802 Qt の名前に合わせる

    def setImage(self, image: QImage, /) -> None: ...  # noqa: N802 Qt の名前に合わせる

    def text(self) -> str: ...

    def image(self) -> QImage: ...


#: 差し替えたクリップボード ``None`` なら OS のもの
_replacement: Clipboard | None = None


def clipboard() -> Clipboard:
    """いま使うクリップボード"""
    if _replacement is not None:
        return _replacement
    return QGuiApplication.clipboard()


def copy_text(text: str) -> None:
    """文字を置く 字幕パネルの〔文字をコピー〕と AI の copy_subtitle_text（#349）が通る所"""
    clipboard().setText(text)


def replace(replacement: Clipboard | None) -> None:
    """クリップボードを差し替える ``None`` で OS のものへ戻す 試験のための口"""
    global _replacement
    _replacement = replacement
