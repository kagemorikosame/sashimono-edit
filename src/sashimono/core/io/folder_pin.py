"""確かめたフォルダを、書き終えるまで差し替えさせない（#327）

書ける場所かを確かめてから実際に書くまでの間に、フォルダをリンク（ジャンクション）へ
差し替えられると、確かめた場所とは別の所へ書く 確かめをもう 1 度挟んでも、その確かめと
書く間にまた隙間が残る そこで、確かめたフォルダを開いたまま書く

- Windows では、フォルダを「消す・名前を変えるのは断る」共有の形で開いたまま持つ
  持っている間は、そのフォルダも、その上のフォルダも、名前を変えたり消したりできない
  （2026-10-15 に手元の NTFS で確かめた 中のファイルは今までどおり作れて置き換えられる）
  開いた物の本当の場所（``GetFinalPathNameByHandleW``）も照らす
- ほかの OS では差し替えそのものは止められない 開いたフォルダの身元（装置と番号）を
  持っておき、書く前と書いた後に、その場所の今の身元と照らして、違えば断る

Qt を使わない AI の道具（``ai/safety.py``）と書き出しの片付け（``ui/ai_export.py``）が使う
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

__all__ = ["FolderSwappedError", "PinnedFolder", "pin_folder"]


class FolderSwappedError(OSError):
    """持っているフォルダと、その場所の今のフォルダが違う（差し替えられた）"""


class PinnedFolder:
    """開いたまま持っているフォルダ :func:`pin_folder` が渡す"""

    def __init__(self, path: Path, identity: tuple[int, int], real: Path | None) -> None:
        #: 開いたときの場所（リンクを辿った後の物）
        self.path = path
        self._identity = identity
        self._real = real

    def verify(self) -> None:
        """その場所に今あるのが、開いたフォルダのままかを照らす

        違えば :class:`FolderSwappedError`
        """
        try:
            now = self.path.stat()
        except OSError as exc:
            raise FolderSwappedError(f"フォルダが見つかりません: {self.path}") from exc
        if (now.st_dev, now.st_ino) != self._identity:
            raise FolderSwappedError(f"フォルダが差し替えられました: {self.path}")
        if self._real is not None and _key(self._real) != _key(self.path):
            # 開いた時点でもうリンクの先だった（解決した後に差し替えられた）
            raise FolderSwappedError(f"フォルダの本当の場所が違います: {self._real}")


@contextlib.contextmanager
def pin_folder(path: Path) -> Iterator[PinnedFolder]:
    """``path`` のフォルダを開いたまま持つ 抜けるときに放す

    ``path`` はリンクを辿った後の場所を渡す（:meth:`Path.resolve`） 開けなければ ``OSError``
    """
    path = Path(path)
    if sys.platform == "win32":
        with _windows_pin(path) as real:
            pinned = PinnedFolder(path, _identity(path), real)
            pinned.verify()
            yield pinned
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        status = os.fstat(descriptor)
        pinned = PinnedFolder(path, (status.st_dev, status.st_ino), None)
        pinned.verify()
        yield pinned
    finally:
        os.close(descriptor)


def _identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino


def _key(path: Path) -> str:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[len("\\\\?\\UNC\\") :]
    elif text.startswith("\\\\?\\"):
        text = text[len("\\\\?\\") :]
    return os.path.normcase(os.path.normpath(text))


@contextlib.contextmanager
def _windows_pin(path: Path) -> Iterator[Path]:
    """フォルダを、消す・名前を変える共有を許さずに開いて持つ 本当の場所を渡す"""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.restype = wintypes.HANDLE
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    final = kernel32.GetFinalPathNameByHandleW
    final.restype = wintypes.DWORD
    final.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    close = kernel32.CloseHandle
    close.argtypes = [wintypes.HANDLE]

    # 中を読む権利で開く 属性を読むだけの権利で開くと、共有の決まりが効かず、
    # 持っていても名前を変えられる（手元で確かめた）
    list_directory = 0x0001
    share_read_write = 0x0001 | 0x0002
    open_existing = 3
    backup_semantics = 0x02000000
    invalid = wintypes.HANDLE(-1).value
    handle = create(
        str(path), list_directory, share_read_write, None, open_existing, backup_semantics, None
    )
    if handle is None or handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        length = final(handle, buffer, len(buffer), 0)
        if length == 0 or length >= len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        yield Path(buffer.value)
    finally:
        close(handle)
