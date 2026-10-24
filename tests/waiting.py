"""裏の仕事の終わりを待つ試験の、共通の待ち方（#405 #406）

CI の機械は回ごとに速さが違う 「10 秒で終わるはず」「60 秒で終わるはず」と区切ると、
遅い機械に当たったときに中身が正しくても落ちる ここは終わりの知らせ（条件が真になる・
待つ関数が真を返す）を待ち、上限は止まったまま終わらない作りの崩れを見つけるためだけに置く
速い機械では終わればすぐ戻るので、上限を長くしても待ちは増えない
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections.abc import Callable

import pytest

__all__ = ["GUARD_SECONDS", "dump_threads", "wait_until"]

#: 止まったまま終わらない崩れを見分ける上限（秒） 終わりを待つ試験はみなこれを使う
#: CI で pytest を並べると、画面のスレッドで GL のコンテキストを作る所が、ほかの試験の
#: 描画と GPU を取り合って数十秒止まることがある（#386） 試験が見るのは結果で速さではない
GUARD_SECONDS = 600.0


def wait_until(
    done: Callable[[], bool],
    what: str,
    *,
    step: Callable[[], None] | None = None,
    pause: float = 0.005,
) -> None:
    """``done`` が真になるまで待つ 間に ``step``（画面の仕事を拾うなど）を回す

    :data:`GUARD_SECONDS` を越えたら、全部のスレッドの居場所を書き残して落とす
    書き残さないと、どこで止まったかが記録から分からない（#405）
    """
    deadline = time.monotonic() + GUARD_SECONDS
    while not done():
        if step is not None:
            step()
        if time.monotonic() > deadline:
            dump_threads()
            pytest.fail(f"{what}が {GUARD_SECONDS:g} 秒で終わらなかった")
        time.sleep(pause)


def dump_threads() -> None:
    """全部のスレッドの今の呼び出しの並びを標準エラーへ書く（pytest が落ちた試験に添える）

    ``faulthandler`` は使わない pytest の取り込み方によっては標準エラーがファイルの
    番号を持たず、書けずに例外で落ちる Qt のスレッド（検査の走り係）も、Python を
    動かしている間はここに出る
    """
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        print(f"--- スレッド {names.get(ident, ident)}", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
