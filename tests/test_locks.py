"""錠のファイルと、プロジェクトを 2 つの窓で開いたことの検出

錠を取り違えると、2 つの窓で同じファイルを開いても何も言われず、あとから
保存した方の内容だけが黙って残る 逆に、持ち主が終わった錠を「使用中」と見ると、
落ちたあとに開き直すたびに「別の窓で開いています」が出る
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from sashimono.core.io import (
    HeldLock,
    hold_new,
    is_held,
    others_holding,
    project_presence_dir,
    try_hold,
)
from sashimono.core.io.atomic import held_for_update


class TestLock:
    def test_a_second_holder_is_refused(self, tmp_path: Path) -> None:
        first = try_hold(tmp_path / "a.lock")
        assert first is not None
        try:
            assert try_hold(tmp_path / "a.lock") is None
        finally:
            first.release()

    def test_a_released_lock_is_free_again(self, tmp_path: Path) -> None:
        lock = try_hold(tmp_path / "a.lock")
        assert lock is not None
        lock.release()
        assert not is_held(tmp_path / "a.lock")
        again = try_hold(tmp_path / "a.lock")
        assert again is not None
        again.release()

    def test_only_one_of_simultaneous_takers_wins(self, tmp_path: Path) -> None:
        # 「空いているか見てから作る」だと、2 つの窓を同時に開いたとき両方が取れる
        # そうなると、どちらにも「別の窓で開いています」が出ない
        path = tmp_path / "a.lock"
        barrier = threading.Barrier(8)
        results: list[HeldLock | None] = []

        def take() -> None:
            barrier.wait()
            results.append(try_hold(path))

        threads = [threading.Thread(target=take) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        held = [lock for lock in results if lock is not None]
        try:
            assert len(held) == 1
        finally:
            for lock in held:
                lock.release()

    def test_a_lock_with_nothing_written_yet_is_still_held(self, tmp_path: Path) -> None:
        # 作った直後の空の錠を「中身が読めない = 持ち主がいない」と取ると、作った
        # ばかりの錠を消して、2 つ目の窓も錠を取れてしまう（中身は見ないこと）
        path = tmp_path / "a.lock"
        lock = try_hold(path)
        assert lock is not None
        try:
            assert path.read_text(encoding="utf-8") == ""
            assert is_held(path)
            assert try_hold(path) is None
            assert path.exists()
        finally:
            lock.release()

    def test_a_dead_owner_can_be_replaced(self, tmp_path: Path) -> None:
        # 落ちた窓の錠が残っていても、次に開いた窓は錠を取れる
        (tmp_path / "a.lock").write_text("終了済み", encoding="utf-8")
        lock = try_hold(tmp_path / "a.lock")
        assert lock is not None
        lock.release()

    def test_a_release_leaves_no_file_behind(self, tmp_path: Path) -> None:
        # 壊れると、窓を閉じるたびに錠のファイルが置き場に積もる
        lock = try_hold(tmp_path / "a.lock")
        assert lock is not None
        lock.release()
        assert list(tmp_path.iterdir()) == []

    def test_a_release_does_not_remove_the_lock_a_waiter_just_took(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、手放す途中で取り直した相手の錠を消し、3 人目も入れてしまう
        # Windows では相手の錠を消せずに上がり、錠の中で書き終えた保存まで失敗に見える
        path = tmp_path / "a.lock"
        first = try_hold(path)
        assert first is not None
        plain_unlink = Path.unlink
        waiter: list[HeldLock | None] = []

        def unlink(self: Path, missing_ok: bool = False) -> None:
            if self == path and not waiter:
                # 手放す側が名前を消しに来た瞬間に、待っていた相手が取りに来る
                monkeypatch.setattr(Path, "unlink", plain_unlink)
                waiter.append(try_hold(path))
            plain_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)
        first.release()
        monkeypatch.setattr(Path, "unlink", plain_unlink)
        (second,) = waiter
        if second is None:
            # まだ持っている間に来た相手は断られる 手放した後は取れる
            second = try_hold(path)
        assert second is not None
        try:
            assert is_held(path)
            assert try_hold(path) is None
        finally:
            second.release()

    def test_a_lock_given_up_is_not_removed_later(self, tmp_path: Path) -> None:
        # 壊れると、手放した錠を後から片付けようとして、その名前で取り直した窓の錠を消す
        path = tmp_path / "a.lock"
        first = try_hold(path)
        assert first is not None
        first.abandon()
        second = try_hold(path)
        assert second is not None
        try:
            first.release()
            assert is_held(path)
        finally:
            second.release()

    def test_a_leftover_from_a_dead_owner_is_not_held(self, tmp_path: Path) -> None:
        # 落ちた窓の錠が残っていても、開き直すたびに警告が出ないこと
        (tmp_path / "a.lock").write_text("終了済み", encoding="utf-8")
        assert not is_held(tmp_path / "a.lock")


def _delete_while_scanned(path: Path) -> Callable[[], None]:
    """ほかのプロセス（ウイルス対策など）が開いている間に、旧来の消し方で ``path`` を消す

    名前は開いている物が閉じられるまで、消す途中（delete pending）で残る 返す関数で閉じる
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.restype = wintypes.HANDLE
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    share_all, open_existing = 0x7, 3
    scanner = create(str(path), 0x80000000, share_all, None, open_existing, 0, None)
    deleter = create(str(path), 0x00010000, share_all, None, open_existing, 0, None)
    flag = ctypes.c_ubyte(1)
    # 4 は FileDispositionInfo 名前を残す旧来の消し方
    assert kernel32.SetFileInformationByHandle(deleter, 4, ctypes.byref(flag), 1)
    kernel32.CloseHandle(deleter)
    return lambda: kernel32.CloseHandle(scanner)


@pytest.mark.skipif(sys.platform != "win32", reason="消す途中の名前が残るのは Windows")
class TestDeletePending:
    """前の持ち主が消した錠を、ほかのプロセスがまだ開いている間（#354 で見つけた揺れ）"""

    def test_a_name_being_deleted_is_busy_not_an_error(self, tmp_path: Path) -> None:
        # 壊れると、2 つの窓が続けて同じ物（設定・置き場の一覧）を書いたとき、ウイルス対策が
        # 前の錠を開いていただけで、後の方の保存や覚え直しが「アクセスが拒否」で落ちる
        path = tmp_path / "places.lock"
        path.write_text("", encoding="utf-8")
        close = _delete_while_scanned(path)
        try:
            assert try_hold(path) is None
        finally:
            close()
        lock = try_hold(path)
        assert lock is not None
        lock.release()

    def test_waiting_for_the_update_lock_rides_over_it(self, tmp_path: Path) -> None:
        # 壊れると、読んでから書き戻す所の錠が、待てば空く名前で待たずに落ちる
        path = tmp_path / "places.lock"
        path.write_text("", encoding="utf-8")
        close = _delete_while_scanned(path)
        timer = threading.Timer(0.1, close)
        timer.start()
        try:
            with held_for_update(path, 2.0):
                pass
        finally:
            timer.join()

    def test_a_folder_that_cannot_be_written_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると（作れないのを全部「持たれている」と読むと）、書けない置き場で、別の窓が
        # 開いていると取り違えるか、錠を取れるまで回り続けて止まる
        def refuse(self: Path, *_arguments: object, **_options: object) -> object:
            raise PermissionError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "open", refuse)
        with pytest.raises(PermissionError):
            try_hold(tmp_path / "a.lock")


class TestPresence:
    def test_each_window_is_counted(self, tmp_path: Path) -> None:
        # 1 つの錠を取り合う形だと、「それでも開く」の窓が数に入らず、先の窓が
        # 閉じたあとに 3 つ目の窓が警告なしで開ける（PR #13） 窓ごとに数える
        folder = tmp_path / "open"
        first, second = hold_new(folder), hold_new(folder)
        try:
            assert others_holding(folder, second.path)
            first.release()
            assert not others_holding(folder, second.path)
            assert others_holding(folder)
        finally:
            second.release()
        assert not others_holding(folder)

    def test_a_crashed_window_is_not_counted(self, tmp_path: Path) -> None:
        # 落ちた窓の錠を数えると、開き直すたびに「別の窓で開いています」が出る
        # 本当に 2 つ開いたときの警告まで読み飛ばされるようになる
        folder = tmp_path / "open"
        folder.mkdir()
        (folder / "dead.lock").write_text("", encoding="utf-8")
        assert not others_holding(folder)


class TestProjectPresenceDir:
    def test_the_same_file_gets_the_same_place(self, tmp_path: Path) -> None:
        # 書き方の違い（.. を挟むなど）で別の場所になると、2 つの窓に気付けない
        direct = project_presence_dir(tmp_path / "本編.sme", tmp_path / "state")
        roundabout = project_presence_dir(tmp_path / "x" / ".." / "本編.sme", tmp_path / "state")
        assert direct == roundabout

    def test_other_files_get_other_places(self, tmp_path: Path) -> None:
        # 名前だけで分けると、別のフォルダの「本編.sme」を開いただけで警告が出る
        state = tmp_path / "state"
        assert project_presence_dir(tmp_path / "a" / "本編.sme", state) != project_presence_dir(
            tmp_path / "b" / "本編.sme", state
        )
