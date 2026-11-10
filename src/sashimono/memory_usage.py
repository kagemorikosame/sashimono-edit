"""このプロセスのメモリの使い方を測る（#454 #455）

止まりの記録（:mod:`sashimono.stall_watch`）と、メモリの内訳の測り（``tools/bench_memory.py``）が
使う 外の部品（psutil）は足さない 配る zip が大きくなり、読むのは 2 つの数だけで済む

Windows はタスクマネージャーの「メモリ」と同じ物（専用の作業セット）を出さず、作業セットと
コミット（Private Bytes）を出す 作業セットは見張りのソフトや OS の都合で上下し、増え続けて
いるかを見るにはコミットの方が当てになる
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

__all__ = ["MemoryUsage", "process_memory"]

_MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """ある時点のメモリ（MB）"""

    #: 実際に物理メモリに載っている分（作業セット） ほかの OS では起動からの最大の RSS で、
    #: 減らない（増え続けているかを見るのには使えるが、減ったかは読めない）
    resident_mb: float
    #: このプロセスだけが確保した分（Windows のコミット ほかの OS では ``None``）
    private_mb: float | None


class _Counters(ctypes.Structure):
    """``PROCESS_MEMORY_COUNTERS_EX``（psapi.h）"""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory() -> MemoryUsage | None:
    """今のメモリ 測れない環境では ``None``（診断のための値なので、測れなくても止めない）"""
    if sys.platform == "win32":
        return _windows()
    return _posix()


def _windows() -> MemoryUsage | None:
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        # 自分を指す仮のハンドルは -1 既定の戻りの型（int）のままだと 64 ビットで桁があふれ、
        # 次の呼び出しへ渡すときに落ちる
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        process = ctypes.c_void_p(kernel.GetCurrentProcess())
        # K32 の名前は kernel32 に入っている psapi.dll を読まずに済む
        ok = kernel.K32GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError):
        return None
    if not ok:
        return None
    return MemoryUsage(
        resident_mb=counters.WorkingSetSize / _MB, private_mb=counters.PrivateUsage / _MB
    )


def _posix() -> MemoryUsage | None:
    if sys.platform == "win32":
        # 型の検査を Windows で走らせても、下の resource の呼び方が見えるように分ける
        return None
    import resource

    # 今の値は取れないので、ここまでの最大を出す（Linux は KB macOS はバイト）
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return MemoryUsage(resident_mb=peak * scale / _MB, private_mb=None)
