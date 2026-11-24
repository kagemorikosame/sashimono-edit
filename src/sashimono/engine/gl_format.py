"""アプリ全体で使う OpenGL のサーフェス形式

:mod:`sashimono.engine.gpu` から分けて置く 形式は ``QApplication`` を作る前に決める必要があり、
起動画面（#526）は ``QApplication`` の直後に出す ``sashimono.engine.gpu`` の頭は合成と
エフェクトの部品（numpy・OpenGL）まで読み、冷えた機械では 6 秒ほど掛かる 形式を決めるためだけに
そこを読むと、その間は起動画面を出せず、何も映らないまま待たせる
"""

from __future__ import annotations

from PySide6.QtGui import QSurfaceFormat

__all__ = ["REQUIRED_GL_VERSION", "preferred_surface_format"]

#: 要求する OpenGL のバージョン フレームバッファの浮動小数点フォーマットと
#: コンピュートシェーダ（将来のエフェクト用）が使える最低ラインとして 4.3 を選ぶ
REQUIRED_GL_VERSION = (4, 3)


def preferred_surface_format() -> QSurfaceFormat:
    """アプリ全体で使うサーフェス形式

    ``QApplication`` を作る前に :meth:`QSurfaceFormat.setDefaultFormat` へ渡すこと
    後から設定してもウィジェットのコンテキストには反映されない
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(*REQUIRED_GL_VERSION)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt
