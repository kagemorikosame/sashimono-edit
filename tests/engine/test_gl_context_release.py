"""オフスクリーンの GL コンテキストを手放すとき、コンテキストとサーフェスの両方をその場で壊す

前の ``release()`` はサーフェスだけを壊し、``QOpenGLContext`` はごみ集めに任せていた
見本の走り係（``LookWorker``）で、コンテキストを作れたのに使える版でなかった（``complete()`` の
中の確かめで断られた）とき、簡易の描き方へ切り替えた後もコンテキストが残った いつ壊れるかが
ごみ集めの気分次第になり、作りかけの物を掴み続ける（``_discard`` と同じ理由 #149）
"""

from __future__ import annotations

import pytest
import shiboken6
from PySide6.QtGui import QOpenGLContext

from sashimono.engine.gpu import GLContextError, OffscreenGLContext

pytestmark = pytest.mark.usefixtures("gpu")


def test_release_destroys_the_context_too() -> None:
    # 書き出しや先読みを終えてもコンテキストがごみ集めまで残り、いつ壊れるかが分からなくなる
    # （作りかけを壊す _discard と同じ理由 #149）
    context = OffscreenGLContext()
    gl, surface = context.context, context.surface
    with context:
        assert QOpenGLContext.currentContext() is gl
    context.release()
    assert not shiboken6.isValid(gl)
    assert not shiboken6.isValid(surface)
    # 2 度呼んでも落ちない（書き出しや先読みは片付けの道が重なることがある）
    context.release()


def test_a_context_refused_after_creation_is_destroyed_on_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # create() は通ったのに、使える版かの確かめで断られた形
    def refuse(_self: OffscreenGLContext) -> None:
        raise GLContextError("試験で版の確かめを断る")

    monkeypatch.setattr(OffscreenGLContext, "_require_usable_gl", refuse)
    context = OffscreenGLContext(deferred=True)
    gl = context.context
    with pytest.raises(GLContextError):
        context.complete()
    context.release()
    assert not shiboken6.isValid(gl)


def test_the_look_worker_lets_go_of_a_refused_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GL の使えない機械で見本を簡易の描き方へ切り替えた後も、断られたコンテキストを掴み続け
    # いつ壊れるかがごみ集め次第になる
    from sashimono.engine.render.look_preview import LookWorker

    def refuse(_self: OffscreenGLContext) -> None:
        raise GLContextError("試験で版の確かめを断る")

    monkeypatch.setattr(OffscreenGLContext, "_require_usable_gl", refuse)
    worker = LookWorker(gpu=True)
    assert worker._context is not None
    gl = worker._context.context
    failed: list[str] = []
    worker.gpu_failed.connect(failed.append)
    worker.start()
    worker.stop()
    assert not shiboken6.isValid(gl)
    shiboken6.delete(worker)
