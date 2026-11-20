"""エンコーダの確かめの子で走らせる作り物（#464）

子のプロセスは関数を名前で import して走らせる 試験の中で定義した関数や差し替えは
子に届かないので、名前で import できるこのモジュールに置く 試験のモジュールに置くと、
子が pytest ごと読み込んで起きるのが遅くなる

名前の頭で振る舞いを変える 壊れた駆動を本当に作らずに、子が落ちる・固まるを真似る

- ``crash_`` 試し始めを知らせてから、例外を通さずにプロセスごと終わる（access violation の真似）
- ``vanish_`` 試し始めを知らせる前に終わる（どれが落としたか親に分からない形）
- ``hang_`` 試し始めを知らせてから、返らない（駆動の中で固まった形）
- ほかは本物の確かめ
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from sashimono.engine import encoders

#: 固まった子が待つ長さ（秒） 親が止めるまで返らなければよい 試験が漏らしても残り続けない長さ
HANG_S = 120.0


def broken_driver_job(
    *,
    progress: Callable[[float], None],
    should_cancel: Callable[[], bool],
    publish: Callable[[object], None],
    names: list[str],
) -> None:
    """:func:`sashimono.engine.encoders._probe_job` と同じ形で、名前の頭に応じて落ちる・固まる"""
    del progress
    for name in names:
        if should_cancel():
            return
        if name.startswith("vanish_"):
            os._exit(5)
        publish(("begin", name))
        if name.startswith("crash_"):
            # 0xC0000005（access violation）で落ちたときと同じく、後片付けをせずに終わる
            os._exit(3)
        if name.startswith("hang_"):
            # やめてと言われても返らない 駆動の中で固まると、Python の側は何もできない
            time.sleep(HANG_S)
        publish(("verdict", name, encoders._probe_here(name)))
