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

    def test_as_expected_it_is_written(self, tmp_path: Path) -> None:
        # 壊れると（照らし方が厳しすぎると）、何も変わっていないのに書けない
        new = tmp_path / "新しい.txt"
        write_atomic(new, "新", expect=expect_now(new))
        assert new.read_text(encoding="utf-8") == "新"
        old = tmp_path / "前からある.txt"
        old.write_text("前", encoding="utf-8")
        write_atomic(old, "後", expect=expect_now(old))
        assert old.read_text(encoding="utf-8") == "後"


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
