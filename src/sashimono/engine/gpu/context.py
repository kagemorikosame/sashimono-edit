"""OpenGL コンテキストの管理

コンテキストは Qt に持たせる ``QOpenGLWidget`` と共有できるので、プレビュー用と
書き出し用でテクスチャやシェーダを作り直さずに済む GLFW などを別に持ち込むと、
Qt のコンテキストと二重管理になり、共有もできなくなる

ヘッドレス（GUI を出さない書き出しやテスト）でも同じ経路を通る ``QOffscreenSurface``
に描くだけで、実機の GPU がそのまま使える
"""

from __future__ import annotations

import functools
from types import TracebackType
from typing import Protocol

from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtGui import (
    QGuiApplication,
    QOffscreenSurface,
    QOpenGLContext,
    QSurface,
    QSurfaceFormat,
)

__all__ = [
    "CurrentGLContext",
    "GLContextError",
    "GLScope",
    "OffscreenGLContext",
    "ensure_qt_application",
    "opengl_usable",
    "preferred_surface_format",
]

#: 要求する OpenGL のバージョン フレームバッファの浮動小数点フォーマットと
#: コンピュートシェーダ（将来のエフェクト用）が使える最低ラインとして 4.3 を選ぶ
REQUIRED_GL_VERSION = (4, 3)


class GLContextError(RuntimeError):
    """OpenGL コンテキストを用意できない"""


class GLScope(Protocol):
    """GL を触る間だけコンテキストを current にする、という約束

    実装は 2 つある 自前でコンテキストを持つ :class:`OffscreenGLContext` と、
    Qt がすでに current にしている状況で使う :class:`CurrentGLContext`
    描画側はどちらを渡されても同じ書き方で済む
    """

    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def release(self) -> None: ...


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


def ensure_qt_application() -> QCoreApplication:
    """Qt のアプリケーションオブジェクトを用意する

    GL コンテキストは Qt のイベントループが無くても作れるが、``QGuiApplication``
    の存在は要る 書き出しやテストから呼ばれたときのために、無ければここで作る
    """
    existing = QCoreApplication.instance()
    if existing is not None:
        return existing
    QSurfaceFormat.setDefaultFormat(preferred_surface_format())
    return QGuiApplication([])


def _reported_gl_version() -> str:
    """ドライバが名乗っている版 取れなければ「不明」

    案内を親切にするためだけの問い合わせなので、**失敗しても止めない**
    壊れたコンテキストでは ``glGetString`` 自体が ``invalid operation`` を
    返すことがあり、そこで例外を出すと本来伝えたい内容が伝わらなくなる
    """
    try:
        from OpenGL.GL import GL_VERSION, glGetString

        raw = glGetString(GL_VERSION)
    except Exception:
        # 版を取れないこと自体は異常ではない 案内の文面が「不明」になるだけ
        return "不明"
    return raw.decode("ascii", "replace") if raw else "不明"


class OffscreenGLContext:
    """画面を持たない GL コンテキスト

    ``share`` に既存のコンテキストを渡すと、テクスチャやバッファを共有できる
    プレビューウィジェットが作ったテクスチャを書き出し側から読む、といった用途向け
    """

    def __init__(self, share: QOpenGLContext | None = None, *, deferred: bool = False) -> None:
        """``deferred`` なら、コンテキストはまだ作らない（サーフェスだけ作る）

        コンテキストを作るのは重い（RTX 5060 Ti で 1 つ 280ms 最初の 1 つは 700ms）
        使うスレッドへ移してから、そこで :meth:`complete` を呼ぶと、画面のスレッドを
        その間止めずに済む サーフェスは画面のスレッドで作る決まりなので、ここで作る
        """
        ensure_qt_application()
        fmt = preferred_surface_format()

        self._depth = 0
        self._created = False
        self._surface = QOffscreenSurface()
        self._context = QOpenGLContext()
        try:
            self._surface.setFormat(fmt)
            self._surface.create()
            if not self._surface.isValid():
                raise GLContextError("オフスクリーンサーフェスを作れない")

            self._context.setFormat(fmt)
            if share is not None:
                self._context.setShareContext(share)
            if not deferred:
                self.complete()
        except BaseException:
            self._discard()
            raise

    def complete(self) -> None:
        """コンテキストを作って使えるかを確かめる ``deferred`` で作った物は、使うスレッドで呼ぶ

        作れなければ :class:`GLContextError` 後片付けは呼んだ側が :meth:`release` で行う
        """
        if self._created:
            return
        if not self._context.create():
            raise GLContextError(
                f"OpenGL {REQUIRED_GL_VERSION[0]}.{REQUIRED_GL_VERSION[1]} "
                "のコンテキストを作れない GPU ドライバを確認すること"
            )
        self._require_usable_gl()
        self._created = True

    def _discard(self) -> None:
        """作りかけのコンテキストとサーフェスを、その場で壊す（#149）

        GPU の無い機械では、起動のたびに :func:`opengl_usable` がここを通る 投げた
        例外は作りかけの物を掴んだまま残り、いつ壊れるかがごみ集めの気分次第になる
        共有元（プレビューのコンテキスト）が生きているうちに、ここで壊しておく
        """
        # 使う所で読む 頭で読むと、Qt の欠けた配布版で自己診断がこのモジュールを
        # 読んだ所でプロセスごと止まり、Qt が欠けていると報告できない
        import shiboken6

        if QOpenGLContext.currentContext() is self._context:
            self._context.doneCurrent()
        self._surface.destroy()
        shiboken6.delete(self._context)
        shiboken6.delete(self._surface)

    def _require_usable_gl(self) -> None:
        """要求した版の関数が本当に呼べるかを確かめる

        **作れたことと使えることは別** ドライバが無い環境（仮想機械や CI）でも
        Qt は software / GDI の経路でコンテキストを作ってしまう そこには
        ``glCreateShader`` のようなシェーダの関数すら無く、呼んだ瞬間に PyOpenGL が
        ``NullFunctionError`` を投げる

        作った直後に確かめておけば、呼び出し側は :class:`GLContextError` 1 つを
        見ればよくなる（テストは飛ばし、アプリは案内を出す） 描画の奥まで進んで
        から中身の分からない例外で落ちるより、ここで止めたほうが原因に近い
        """
        from OpenGL.GL import glCreateShader, glGenVertexArrays

        # 実際に取れた版を先に見る 下の関数は 2.0 / 3.0 から在るので、3.x の
        # コンテキストでも素通りしてしまう 4.3 で入った機能（計算シェーダなど）を
        # 使う所まで進んでから落ちることになる
        granted = self._context.format()
        version = (granted.majorVersion(), granted.minorVersion())
        if version < REQUIRED_GL_VERSION:
            raise GLContextError(
                f"OpenGL {REQUIRED_GL_VERSION[0]}.{REQUIRED_GL_VERSION[1]} が要るが、"
                f"取れたのは {version[0]}.{version[1]} GPU ドライバを確認すること"
            )

        with self:
            missing = [
                name
                for name, function in (
                    ("glCreateShader", glCreateShader),
                    ("glGenVertexArrays", glGenVertexArrays),
                )
                if not bool(function)
            ]
            if missing:
                raise GLContextError(
                    f"OpenGL {REQUIRED_GL_VERSION[0]}.{REQUIRED_GL_VERSION[1]} "
                    f"の関数が見つからない（{'、'.join(missing)}）"
                    f"ドライバが返した版は {_reported_gl_version()}"
                    "GPU ドライバを確認すること"
                )

    @property
    def context(self) -> QOpenGLContext:
        return self._context

    @property
    def surface(self) -> QSurface:
        return self._surface

    def move_to_thread(self, thread: QThread) -> None:
        """コンテキストを ``thread`` で使えるようにする **いま持っているスレッドから呼ぶこと**

        ``QOpenGLContext`` はスレッドに付く 別のスレッドで ``makeCurrent`` すると
        Qt が断って current にならない サーフェスは GUI のスレッドで作る決まりなので、
        作るのはこちら、使うのは移した先、という分け方になる
        """
        self._context.moveToThread(thread)

    def make_current(self) -> None:
        if not self._context.makeCurrent(self._surface):
            raise GLContextError("GL コンテキストを current にできない")

    def done_current(self) -> None:
        self._context.doneCurrent()

    def __enter__(self) -> OffscreenGLContext:
        # 入れ子にできるようにしておく 合成の途中でテクスチャを作るような経路で、
        # 内側が抜けた拍子にコンテキストが外れると診断の難しい失敗になる
        if self._depth == 0:
            self.make_current()
        self._depth += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._depth -= 1
        if self._depth == 0:
            self.done_current()

    def release(self) -> None:
        """コンテキストとサーフェスを破棄する"""
        if self._depth:
            self.done_current()
            self._depth = 0
        self._surface.destroy()


class CurrentGLContext:
    """すでに current になっているコンテキストを表す、何もしないスコープ

    ``QOpenGLWidget`` の ``initializeGL`` / ``paintGL`` の中では Qt がすでに
    コンテキストを current にしている そこで :class:`OffscreenGLContext` を
    使うと、別のコンテキストに切り替わって描画先を見失う

    :class:`FrameRenderer` のような「スコープに入ってから GL を触る」書き方を
    変えずに済ませるために、入口だけ用意して何もしない実装を置く
    """

    def __enter__(self) -> CurrentGLContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def release(self) -> None:
        """所有していないので何もしない"""


@functools.cache
def opengl_usable() -> bool:
    """この機械で OpenGL 4.3 を使えるか 1 つのプロセスで 1 度だけ確かめる

    使えない機械（GPU の無い仮想機械や CI）で GL のプレビューを窓に出すと、
    窓ごと GL で描くようになり、閉じた後の片付け（ごみ集めか Python の終わり）で
    プロセスごと落ちる（#149） 片付けの順を変えても直らなかったので、使えない
    機械ではプレビューを 1 度も出さない その判断に使う
    """
    try:
        context = OffscreenGLContext()
    except GLContextError:
        return False
    context.release()
    return True
