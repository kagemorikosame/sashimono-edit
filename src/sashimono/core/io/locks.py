"""「いま使っている」を錠のファイルで示す

自動退避（:mod:`sashimono.core.io.recovery`）と、プロジェクトを 2 つの窓で開いたことの
検出が同じ仕組みを使う

生きているかどうかは、持ち主が開いたままにしているファイルで見分ける 中身
（プロセス番号）は読まない 番号は使い回されるうえ、書き終わる前の空の中身を読むと
「持ち主は終わっている」と取り違えて、作ったばかりの錠を消してしまう（PR #13）

- Windows: 開いているファイルは消せない 「消せたら持ち主はもういない」
- それ以外: カーネルのファイルロック（``flock``）を掛けたまま持つ 掛けられたら
  持ち主はもういない プロセスが終わるとロックは自然に外れる
"""

from __future__ import annotations

import contextlib
import os
import sys
import uuid
from pathlib import Path
from typing import IO

__all__ = ["HeldLock", "hold_new", "is_held", "others_holding", "try_hold"]


class HeldLock:
    """開いたままにしている錠 :meth:`release` を呼ぶか、プロセスが終わるまで効く"""

    def __init__(self, path: Path, handle: IO[str]) -> None:
        self.path = path
        self._handle: IO[str] | None = handle

    def release(self) -> None:
        """手放して錠のファイルを片付ける 片付けるのは自分が持っている錠だけ

        手放してから名前で消すと、その間に待っていた相手が同じ名前で取り直した錠を消す
        POSIX では 3 人目も入れてしまい、Windows では相手が開いているので消せずに上がり、
        錠の中で済ませた仕事（書き終えた保存）まで失敗に見える 手放した後（:meth:`abandon`）
        や 2 度目に呼んだときは、もう自分の錠ではないので何もしない
        """
        handle, self._handle = self._handle, None
        if handle is not None:
            _let_go(self.path, handle)

    def abandon(self) -> None:
        """錠のファイルを片付けずに手放す プロセスが落ちたときと同じ状態になる

        落ちたあとに残った錠を正しく扱えるかを確かめるための入口 テストが
        中身の持ち方（``_handle``）に触らずに済むように置いてある
        """
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def is_held(path: Path) -> bool:
    """誰かが持っているか 持ち主が終わっていれば錠を片付けて偽を返す"""
    if not path.exists():
        return False
    return _held(path)


def try_hold(path: Path) -> HeldLock | None:
    """空いていれば押さえる 誰かが持っていれば ``None``

    1 回の操作で押さえる 「空いているか見てから作る」の 2 段だと、2 つの窓を同時に
    開いたときに両方が錠を取れてしまう 既にあった錠の持ち主が終わっていれば、
    :func:`is_held` が片付けるので 1 度だけ取り直す
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        handle = _create(path)
        if handle is not None:
            return HeldLock(path, handle)
        if is_held(path):
            return None
    return None


def hold_new(folder: Path) -> HeldLock:
    """``folder`` の中に、自分だけの錠を 1 つ作る 名前が重ならないので必ず取れる

    同じものを使っている相手を全員数えたいときに使う（:func:`others_holding`）
    """
    while True:
        lock = try_hold(folder / f"{uuid.uuid4().hex}.lock")
        if lock is not None:
            return lock


def others_holding(folder: Path, mine: Path | None = None) -> bool:
    """``folder`` の中に、自分（``mine``）以外の生きた錠があるか

    終わった持ち主の錠は :func:`is_held` がついでに片付ける
    """
    if not folder.is_dir():
        return False
    return any(path != mine and is_held(path) for path in folder.glob("*.lock"))


if sys.platform == "win32":

    def _create(path: Path) -> IO[str] | None:
        try:
            return path.open("x", encoding="utf-8")
        except FileExistsError:
            return None

    def _let_go(path: Path, handle: IO[str]) -> None:
        """閉じてから消す 開いたままでは自分でも消せない

        閉じた後に消せないのは、待っていた相手がもう同じ名前で取り直して開いているとき
        （``PermissionError``） その錠は相手の物なので残す 無いのは、相手が持ち主のいない
        錠として先に片付けたとき
        """
        handle.close()
        with contextlib.suppress(FileNotFoundError, PermissionError):
            path.unlink()

    def _held(path: Path) -> bool:
        """消せたら持ち主はもういない（開いているファイルは消せない）"""
        try:
            path.unlink()
        except PermissionError:
            return True
        except FileNotFoundError:
            return False
        return False

else:
    import fcntl

    def _create(path: Path) -> IO[str] | None:
        """ロックを掛けてから所定の名前に置く

        作ってからロックを掛けると、その間に見に来た相手が「ロックが無い = 持ち主が
        いない」と判断して消してしまう 仮の名前で作ってロックを掛け、``link`` で
        置く ``link`` は既にあれば失敗するので、置くのも 1 回の操作で済む
        """
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        handle = temporary.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.link(temporary, path)
        except FileExistsError:
            handle.close()
            return None
        finally:
            temporary.unlink(missing_ok=True)
        return handle

    def _let_go(path: Path, handle: IO[str]) -> None:
        """ロックを持ったまま消してから閉じる 消すのは名前がまだ自分のファイルを指すときだけ

        閉じてから消すと、その間に相手がロックを掛けて取り直した錠を消す 持ったまま消せば、
        相手は消えた名前へ新しい錠を置くので取り違えない
        """
        try:
            with contextlib.suppress(FileNotFoundError):
                mine, named = os.fstat(handle.fileno()), path.stat()
                if (mine.st_dev, mine.st_ino) == (named.st_dev, named.st_ino):
                    path.unlink()
        finally:
            handle.close()

    def _held(path: Path) -> bool:
        """ロックを掛けられたら持ち主はもういない（プロセスが終わると外れる）"""
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError:
            return False
        with handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        path.unlink(missing_ok=True)
        return False
