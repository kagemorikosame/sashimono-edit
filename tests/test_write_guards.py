"""確かめた場所へだけ書くための守り（B1 #327）

- 仮のファイルは必ず新しく作る（``core.io.atomic``） 先に置かれた同じ名前の物の中身へは書かない
- 確かめたフォルダを、書き終えるまで差し替えさせない（``core.io.folder_pin``）

AI の保存と書き出しはこの 2 つに乗っている ここが落ちると、確かめた後にフォルダや仮の
ファイルの名前を差し替えられたとき、確かめていない所や元の素材の中身へ書く
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, tzinfo
from pathlib import Path

import pytest

from sashimono.core.io import atomic
from sashimono.core.io.atomic import TargetChangedError, copy_atomic, expect_now, write_atomic
from sashimono.core.io.folder_pin import FolderSwappedError, pin_folder
from sashimono.core.io.subtitles import save_subtitles
from sashimono.core.model import Project


def _hard_link(link: Path, target: Path) -> None:
    try:
        os.link(target, link)
    except OSError as exc:
        pytest.skip(f"ハードリンクを作れない: {exc}")


class _FixedMark:
    """仮のファイルの名前の印を決め打ちにする 先回りして同じ名前を置く試験のため"""

    hex = "0a1b2c3d" + "0" * 24


class TestScratchIsAlwaysNew:
    def test_a_planted_scratch_name_is_not_written_into(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、仮のファイルの名前に元の素材へのハードリンクを先に置かれたとき、
        # 保存の中身でその素材を上書きする
        original = tmp_path / "素材.mp4"
        original.write_bytes(b"original")
        target = tmp_path / "作品.sme"
        monkeypatch.setattr(uuid, "uuid4", lambda: _FixedMark())
        planted = tmp_path / f"作品.sme.{_FixedMark.hex[:8]}{atomic.WRITING_SUFFIX}"
        _hard_link(planted, original)

        with pytest.raises(FileExistsError):
            write_atomic(target, "保存の中身")
        assert original.read_bytes() == b"original"
        # 先に置かれていた物は自分の物ではないので片付けない（消すと人の物を消す）
        assert planted.exists()
        assert not target.exists()

    def test_copying_writes_a_new_file_and_says_which(self, tmp_path: Path) -> None:
        # 壊れると、書き出した動画を書き出し先へ写した後に、名前の先が自分の書いた物か照らせない
        source = tmp_path / "仮.mp4"
        source.write_bytes(b"x" * (3 << 20))
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"previous export")
        written = copy_atomic(source, target)
        now = target.stat()
        assert (now.st_dev, now.st_ino) == (written.st_dev, written.st_ino)
        assert target.read_bytes() == source.read_bytes()
        assert sorted(path.name for path in tmp_path.iterdir()) == ["仮.mp4", "完成.mp4"]

    def test_copying_over_a_hard_link_leaves_the_linked_file_alone(self, tmp_path: Path) -> None:
        # 壊れると（開いて上書きすると）、書き出し先が元の素材へのハードリンクだったとき素材が壊れる
        original = tmp_path / "素材.mp4"
        original.write_bytes(b"original")
        link = tmp_path / "完成.mp4"
        _hard_link(link, original)
        source = tmp_path / "仮.mp4"
        source.write_bytes(b"new")
        copy_atomic(source, link)
        assert link.read_bytes() == b"new"
        assert original.read_bytes() == b"original"


#: 残した前のファイルを片付けで消すのは、開いている人を確かめられる Windows だけ
#: （_delete_if_unheld） ほかの OS では消さないので、本数を数える試験は Windows で走らせる
_HOLDERS_SEEN_ONLY_ON_WINDOWS = "開いている人を確かめて片付けるのは Windows だけ"


class TestTheTargetIsAsExpected:
    """許したときの書く先の身元（:class:`Expected`）と、名前を付ける所で照らす（#327）"""

    def test_a_name_that_appears_after_approval_is_not_replaced(self, tmp_path: Path) -> None:
        # 壊れると、許したときは無かった名前のファイル（別のアプリが作った物）を置き換えて消す
        target = tmp_path / "完成.mp4"
        expect = expect_now(target)
        target.write_bytes(b"other app")
        with pytest.raises(TargetChangedError, match="前のファイルを残しました"):
            write_atomic(target, "新しい中身", expect=expect)
        assert target.read_bytes() == b"other app"
        assert [path.name for path in tmp_path.iterdir()] == ["完成.mp4"]

    def test_a_file_changed_after_approval_is_not_replaced(self, tmp_path: Path) -> None:
        # 壊れると、人が置き換えを許したときの中身ではなく、その後に書き換わった中身を消す
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"previous")
        expect = expect_now(target)
        target.write_bytes(b"changed afterwards")
        source = tmp_path / "仮.bin"
        source.write_bytes(b"new")
        with pytest.raises(TargetChangedError):
            copy_atomic(source, target, expect=expect)
        assert target.read_bytes() == b"changed afterwards"

    def test_a_file_put_between_the_check_and_the_placing_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、身元を照らしてから名前を付けるまでの間に別のアプリが置いたファイルを、
        # 許していないまま上書きして消す（Codex の P1 照らしと置き方が別の操作だった）
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"approved")
        expect = expect_now(target)
        interjected: list[str] = []

        def interject(real: Callable[[Path, Path], None]) -> Callable[[Path, Path], None]:
            def wrapped(temporary: Path, path: Path) -> None:
                if not interjected:
                    # 照らした後、名前を付ける直前に別のアプリが書く先へ新しいファイルを置いた
                    interjected.append("x")
                    other = tmp_path / "別のアプリ.tmp"
                    other.write_bytes(b"other app")
                    other.replace(target)
                real(temporary, path)

            return wrapped

        monkeypatch.setattr(atomic, "_replace", interject(atomic._replace))
        monkeypatch.setattr(atomic, "_place_new", interject(atomic._place_new))
        with pytest.raises(TargetChangedError) as caught:
            write_atomic(target, "新しい中身", expect=expect)
        assert interjected
        assert target.read_bytes() == b"other app"
        # 許した前のファイルは黙って消さない 退けた名前を知らせて残す
        kept = [path for path in tmp_path.iterdir() if path.read_bytes() == b"approved"]
        assert len(kept) == 1
        assert kept[0].name in str(caught.value)

    def test_a_file_changed_before_it_was_moved_aside_is_put_back(self, tmp_path: Path) -> None:
        # 壊れると、照らしが違ったときに退けたファイルが仮の名前のまま残り、元の名前から消える
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"approved")
        expect = expect_now(target)
        target.write_bytes(b"changed!")
        # 同じ大きさで続けて書くと、時計の刻みしだいで時刻が同じになる 時刻を確かに進める
        later = expect.identity[3] + 1_000_000 if expect.identity else 0
        os.utime(target, ns=(later, later))
        with pytest.raises(TargetChangedError):
            write_atomic(target, "新しい中身", expect=expect)
        assert target.read_bytes() == b"changed!"
        assert [path.name for path in tmp_path.iterdir()] == ["完成.mp4"]

    def test_the_replaced_file_is_kept_beside_under_a_findable_name(self, tmp_path: Path) -> None:
        # 壊れると、退けた前のファイルを消し、別のアプリが消す共有を許したまま書き続けていた
        # 中身（照らした後の更新）まで一緒に失う（Codex の P1 安全を言い切れないので残す）
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"approved")
        expect = expect_now(target)
        write_atomic(target, "新しい中身", expect=expect)
        kept = [path for path in tmp_path.iterdir() if path != target]
        assert len(kept) == 1
        assert kept[0].name.startswith("完成.before-") and kept[0].suffix == ".mp4"
        assert kept[0].read_bytes() == b"approved"
        assert expect.kept == [kept[0]]
        assert target.read_text(encoding="utf-8") == "新しい中身"

    @pytest.mark.skipif(sys.platform != "win32", reason=_HOLDERS_SEEN_ONLY_ON_WINDOWS)
    def test_kept_files_are_tidied_to_the_generation_count(self, tmp_path: Path) -> None:
        # 壊れると、AI が置き換えるたびに前のファイルが溜まり続け、作品のフォルダが埋まる
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"0")
        for number in range(1, 5):
            expect = replace(expect_now(target), keep=2)
            write_atomic(target, str(number), expect=expect)
        kept = sorted(path for path in tmp_path.iterdir() if path != target)
        assert [path.read_bytes() for path in kept] == [b"2", b"3"]

    @pytest.mark.skipif(sys.platform != "win32", reason=_HOLDERS_SEEN_ONLY_ON_WINDOWS)
    def test_kept_files_with_the_same_stamp_are_tidied_oldest_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、時計の刻みで残す名前の時刻が重なったとき、名前の順と作った順が食い違い、
        # 片付けが新しい方を消す（1 本目と 2 本目、-2 と -10 の並び）
        class _SameMoment(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> _SameMoment:
                return cls(2026, 10, 16, 10, 15, 0, 123456)

        monkeypatch.setattr(atomic, "datetime", _SameMoment)
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"0")
        for number in range(1, 13):
            write_atomic(target, str(number), expect=replace(expect_now(target), keep=2))
        kept = sorted(path for path in tmp_path.iterdir() if path != target)
        assert [path.read_bytes() for path in kept] == [b"10", b"11"]

    def test_as_expected_it_is_written(self, tmp_path: Path) -> None:
        # 壊れると（照らし方が厳しすぎると）、何も変わっていないのに書けない
        new = tmp_path / "新しい.txt"
        write_atomic(new, "新", expect=expect_now(new))
        assert new.read_text(encoding="utf-8") == "新"
        old = tmp_path / "前からある.txt"
        old.write_text("前", encoding="utf-8")
        write_atomic(old, "後", expect=expect_now(old))
        assert old.read_text(encoding="utf-8") == "後"


def _approved(tmp_path: Path) -> tuple[Path, atomic.Expected]:
    target = tmp_path / "完成.mp4"
    target.write_bytes(b"approved")
    return target, expect_now(target)


def _asides(tmp_path: Path) -> list[Path]:
    return [path for path in tmp_path.iterdir() if path.name.endswith(atomic.ASIDE_SUFFIX)]


class TestEveryFailureAfterMovingAside:
    """前のファイルを退けてから付け終えるまでのどこで失敗しても、元の名前に前のファイルが
    戻るか、残した名前が人に伝わる（Codex の P1） 失敗を 1 か所ずつ差し込んで確かめる
    """

    def test_a_failure_while_checking_puts_it_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、退けた物の身元を読めなかったとき、元の名前が消えたまま退けた名前だけが残る
        target, expect = _approved(tmp_path)
        real = atomic.expect_now

        def refuse_aside(path: Path) -> atomic.Expected:
            if path.name.endswith(atomic.ASIDE_SUFFIX):
                raise PermissionError("試験: 読めない")
            return real(path)

        monkeypatch.setattr(atomic, "expect_now", refuse_aside)
        with pytest.raises(OSError, match="読めない"):
            write_atomic(target, "新しい中身", expect=expect)
        assert target.read_bytes() == b"approved"
        assert _asides(tmp_path) == []

    def test_a_failure_while_placing_puts_it_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、ウイルス対策や権限で付けられなかったとき、元の名前が消えたまま残る
        target, expect = _approved(tmp_path)

        def refuse(_temporary: Path, _path: Path) -> None:
            raise PermissionError("試験: 付けられない")

        monkeypatch.setattr(atomic, "_place_new", refuse)
        with pytest.raises(atomic.PlacingError, match="付けられない"):
            write_atomic(target, "新しい中身", expect=expect)
        assert target.read_bytes() == b"approved"
        assert _asides(tmp_path) == []

    def test_when_it_cannot_be_put_back_the_kept_name_is_told(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、戻せなかった前のファイルがどこにあるか、断りの文から分からない
        target, expect = _approved(tmp_path)

        def refuse(_temporary: Path, _path: Path) -> None:
            raise PermissionError("試験: 付けられない")

        real_rename = atomic._rename_new

        def no_way_back(source: Path, destination: Path) -> None:
            if source.name.endswith(atomic.ASIDE_SUFFIX) and destination == target:
                raise PermissionError("試験: 戻せない")
            real_rename(source, destination)

        monkeypatch.setattr(atomic, "_place_new", refuse)
        monkeypatch.setattr(atomic, "_rename_new", no_way_back)
        with pytest.raises(atomic.PlacingError) as caught:
            write_atomic(target, "新しい中身", expect=expect)
        asides = _asides(tmp_path)
        assert len(asides) == 1
        assert asides[0].read_bytes() == b"approved"
        assert str(asides[0]) in str(caught.value)

    def test_an_interruption_while_placing_puts_it_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、中断（Ctrl+C など）で止まったとき、元の名前が消えたまま残る
        target, expect = _approved(tmp_path)

        def interrupt(_temporary: Path, _path: Path) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(atomic, "_place_new", interrupt)
        with pytest.raises(KeyboardInterrupt):
            write_atomic(target, "新しい中身", expect=expect)
        assert target.read_bytes() == b"approved"
        assert _asides(tmp_path) == []

    def test_a_failure_while_keeping_still_tells_where_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、新しいファイルは付けたのに、前のファイルの残し場所で失敗して保存ごと失敗に
        # 見えるか、退けた名前のまま残った前のファイルの場所が伝わらない
        target, expect = _approved(tmp_path)

        def fail(_aside: Path, _path: Path, _keep: int) -> Path:
            raise RuntimeError("試験: 残せない")

        monkeypatch.setattr(atomic, "_keep_beside", fail)
        write_atomic(target, "新しい中身", expect=expect)
        assert target.read_text(encoding="utf-8") == "新しい中身"
        asides = _asides(tmp_path)
        assert len(asides) == 1 and asides[0].read_bytes() == b"approved"
        assert expect.kept == asides


def _hold_sharing_delete(path: Path) -> Callable[[], None]:
    """別のアプリのつもりで、消す共有を許した書き込みの開き方で ``path`` を開いたままにする"""
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
    close = kernel32.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    generic_write = 0x40000000
    share_all = 0x1 | 0x2 | 0x4
    handle = create(str(path), generic_write, share_all, None, 3, 0, None)
    assert handle not in (None, wintypes.HANDLE(-1).value)
    return lambda: close(handle)


class TestTidyingKeepsWhatOthersHold:
    """残した前のファイルの片付けは、誰かが開いている物を消さない（Codex の P2）"""

    @pytest.mark.skipif(sys.platform != "win32", reason=_HOLDERS_SEEN_ONLY_ON_WINDOWS)
    def test_an_oldest_kept_file_held_by_another_app_is_not_deleted(self, tmp_path: Path) -> None:
        # 壊れると、別のアプリが消す共有を許して書き込み中の前のファイルを片付けで消し、
        # その更新ごと失う
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"0")
        write_atomic(target, "1", expect=replace(expect_now(target), keep=1))
        oldest = next(path for path in tmp_path.iterdir() if path != target)
        release = _hold_sharing_delete(oldest)
        try:
            write_atomic(target, "2", expect=replace(expect_now(target), keep=1))
            assert oldest.exists(), "誰かが開いている前のファイルを消した"
        finally:
            release()
        # 放した後の片付けでは消える（溜まり続けない）
        write_atomic(target, "3", expect=replace(expect_now(target), keep=1))
        assert not oldest.exists()

    def test_where_holders_cannot_be_seen_nothing_is_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、開いている人を確かめられない OS で、書き込み中かもしれない物を消す
        kept = tmp_path / "完成.before-20261015-101500-123456-000.mp4"
        kept.write_bytes(b"before")
        monkeypatch.setattr(sys, "platform", "linux")
        assert atomic._delete_if_unheld(kept) is False
        assert kept.exists()


class TestSubtitleFiles:
    def test_writing_subtitles_over_a_hard_link_leaves_the_linked_file_alone(
        self, tmp_path: Path
    ) -> None:
        # 壊れると（字幕のファイルを開いて上書きすると）、書き出し先が元の素材への
        # ハードリンクだったとき、素材の中身が字幕の文字に変わる
        original = tmp_path / "素材.srt"
        original.write_bytes(b"original")
        link = tmp_path / "字幕.srt"
        _hard_link(link, original)
        save_subtitles(Project.create(), link)
        assert original.read_bytes() == b"original"


class TestPinnedFolder:
    @pytest.mark.skipif(sys.platform != "win32", reason="持っている間に止められるのは Windows")
    def test_a_pinned_folder_cannot_be_renamed_or_removed(self, tmp_path: Path) -> None:
        # 壊れると、確かめたフォルダを書いている間にリンクへ差し替えられ、外へ書く
        parent = tmp_path / "親"
        folder = parent / "作品"
        folder.mkdir(parents=True)
        with pin_folder(folder.resolve()) as pinned:
            with pytest.raises(OSError):
                folder.rename(parent / "別")
            with pytest.raises(OSError):
                folder.rmdir()
            # 上のフォルダの名前を変えても、持っているフォルダの場所が変わる
            with pytest.raises(OSError):
                parent.rename(tmp_path / "親2")
            # 中のファイルは今までどおり作れて置き換えられる（保存や書き出しが書ける）
            (folder / "a.txt").write_text("x", encoding="utf-8")
            (folder / "a.txt").replace(folder / "b.txt")
            pinned.verify()
        # 放した後は普通に動かせる 放し忘れると、人がフォルダを消せなくなる
        parent.rename(tmp_path / "親2")

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows では差し替えそのものを断る")
    def test_a_swapped_folder_is_noticed(self, tmp_path: Path) -> None:
        # 壊れると、差し替えを止められない OS で、差し替えられたまま書いたと伝える
        folder = tmp_path / "作品"
        folder.mkdir()
        with pin_folder(folder) as pinned:
            folder.rename(tmp_path / "別")
            folder.mkdir()
            with pytest.raises(FolderSwappedError):
                pinned.verify()

    def test_a_missing_folder_cannot_be_pinned(self, tmp_path: Path) -> None:
        # 壊れると、無いフォルダを持ったつもりで書き、作られたフォルダへ書く
        with pytest.raises(OSError), pin_folder(tmp_path / "無い"):
            pass
