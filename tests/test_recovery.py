"""保存していない作業の退避と、上書き前のバックアップ

退避は「落ちたときにだけ」拾えなければならない 生きている別の起動の退避を
拾うと、同じ作業が 2 つの窓で別々に進み、どちらかの変更が消える
"""

from __future__ import annotations

from datetime import datetime, tzinfo
from pathlib import Path

import pytest

from sashimono.core.io import (
    RecoverySession,
    backup_before_save,
    backup_folder,
    backups_over,
    discard,
    find_orphans,
    find_orphans_in,
    folder_problem,
    load_project,
    recovery,
)
from sashimono.core.model import Project


def crash(session: RecoverySession) -> None:
    """落ちたことにする 退避は消さずに錠だけ手放す（プロセスが消えたときと同じ）

    錠のファイルは残したまま、開いていた手元（Windows では開いたファイル、それ以外では
    ``flock``）だけを閉じる 生死は中身ではなくこの手元で見るので、錠のファイルが
    残っていても「持ち主はもういない」と判定される 落ちたあとの状態そのものになる
    """
    lock = session._lock
    assert lock is not None
    lock.abandon()
    session._lock = None


class TestRecovery:
    def test_a_live_session_is_not_an_orphan(self, tmp_path: Path) -> None:
        session = RecoverySession(tmp_path)
        session.save(Project.create(name="作業中"), None)
        try:
            assert find_orphans(tmp_path) == []
        finally:
            session.close()

    def test_a_crashed_session_is_found(self, tmp_path: Path) -> None:
        session = RecoverySession(tmp_path)
        source = tmp_path / "本編.sme"
        session.save(Project.create(name="本編"), source)
        crash(session)

        entries = find_orphans(tmp_path)
        assert [(entry.name, entry.source) for entry in entries] == [("本編", source)]
        assert load_project(entries[0].path).name == "本編"

    def test_a_clean_exit_leaves_nothing(self, tmp_path: Path) -> None:
        session = RecoverySession(tmp_path)
        session.save(Project.create(), None)
        session.close()
        assert list((tmp_path / "recovery").iterdir()) == []

    def test_clearing_forgets_the_saved_state(self, tmp_path: Path) -> None:
        # 保存した直後に退避が残っていると、次に落ちたとき保存前の古い状態を勧めてしまう
        session = RecoverySession(tmp_path)
        session.save(Project.create(), None)
        session.clear()
        crash(session)
        assert find_orphans(tmp_path) == []

    def test_discard_removes_it(self, tmp_path: Path) -> None:
        session = RecoverySession(tmp_path)
        session.save(Project.create(), None)
        crash(session)
        discard(find_orphans(tmp_path)[0])
        assert find_orphans(tmp_path) == []
        assert list((tmp_path / "recovery").iterdir()) == []

    def test_newest_comes_first(self, tmp_path: Path) -> None:
        older = RecoverySession(tmp_path)
        older.save(Project.create(name="古い"), None)
        crash(older)
        meta = tmp_path / "recovery" / f"{older.session}.json"
        meta.write_text(meta.read_text("utf-8").replace("20", "19", 1), "utf-8")

        newer = RecoverySession(tmp_path)
        newer.save(Project.create(name="新しい"), None)
        crash(newer)
        assert [entry.name for entry in find_orphans(tmp_path)] == ["新しい", "古い"]

    def test_a_broken_note_is_skipped_not_deleted(self, tmp_path: Path) -> None:
        # 読めないからといって消すと、中身の退避まで失う
        session = RecoverySession(tmp_path)
        session.save(Project.create(), None)
        crash(session)
        (tmp_path / "recovery" / f"{session.session}.json").write_text("{", "utf-8")
        assert find_orphans(tmp_path) == []
        assert session.path.exists()

    def test_a_crash_during_the_first_save_is_still_found(self, tmp_path: Path) -> None:
        # 中身を書いた直後、メモを書く前に落ちた形 メモだけを数えていると、
        # 最初の 30 秒ぶんの作業が復元の候補に出ずに消える
        session = RecoverySession(tmp_path)
        session.save(Project.create(name="最初の退避"), None)
        (tmp_path / "recovery" / f"{session.session}.json").unlink()
        crash(session)

        (entry,) = find_orphans(tmp_path)
        assert entry.path == session.path
        assert load_project(entry.path).name == "最初の退避"

    def test_two_sessions_do_not_share_a_file(self, tmp_path: Path) -> None:
        first, second = RecoverySession(tmp_path), RecoverySession(tmp_path)
        try:
            assert first.path != second.path
        finally:
            first.close()
            second.close()


class TestBackup:
    def test_nothing_to_back_up_before_the_first_save(self, tmp_path: Path) -> None:
        # 壊れると、初めての保存が「控えるものが無い」例外で失敗する
        assert backup_before_save(tmp_path / "無い.sme", tmp_path / "state") is None

    def test_the_previous_contents_are_kept(self, tmp_path: Path) -> None:
        # 壊れると、上書きで壊した保存を戻せない（バックアップの意味が無くなる）
        target = tmp_path / "本編.sme"
        target.write_text("前の中身", "utf-8")
        copied = backup_before_save(target, tmp_path / "state")
        assert copied is not None
        assert copied.read_text("utf-8") == "前の中身"

    def test_old_generations_are_pruned(self, tmp_path: Path) -> None:
        # 壊れると、保存するたびに控えが増え続けてディスクを埋める
        target = tmp_path / "本編.sme"
        for index in range(4):
            target.write_text(str(index), "utf-8")
            backup_before_save(target, tmp_path / "state", keep=2)
        kept = sorted(backup_folder(target, tmp_path / "state").iterdir())
        assert [path.read_text("utf-8") for path in kept] == ["2", "3"]

    def test_saves_within_the_same_instant_keep_every_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Windows の Python 3.12 は時刻の刻みが約 15ms 続けて保存すると同じ時刻になり、
        # 名前が重なって前の控えを上書きしていた（CI で 3.12 だけ、たまに落ちた）
        frozen = datetime(2026, 9, 12, 12, 0, 0)

        class Stopped(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> Stopped:
                del tz
                return cls.fromtimestamp(frozen.timestamp())

        monkeypatch.setattr(recovery, "datetime", Stopped)
        target = tmp_path / "本編.sme"
        for index in range(3):
            target.write_text(str(index), "utf-8")
            backup_before_save(target, tmp_path / "state")
        kept = sorted(backup_folder(target, tmp_path / "state").iterdir())
        assert [path.read_text("utf-8") for path in kept] == ["0", "1", "2"]

    def test_a_name_already_taken_is_never_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 別の窓が同じ瞬間に同じ名前を押さえた形 上書きすると、その窓の控えが消える
        class Stopped(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> Stopped:
                del tz
                return cls(2026, 9, 12, 12, 0, 0)

        monkeypatch.setattr(recovery, "datetime", Stopped)
        target = tmp_path / "本編.sme"
        target.write_text("こちら", "utf-8")
        folder = backup_folder(target, tmp_path / "state")
        folder.mkdir(parents=True)
        taken = folder / "20260912-120000-000000-000.sme"
        taken.write_text("別の窓", "utf-8")

        copied = backup_before_save(target, tmp_path / "state")
        assert copied is not None and copied != taken
        assert taken.read_text("utf-8") == "別の窓"
        assert copied.read_text("utf-8") == "こちら"

    def test_same_name_in_another_folder_is_kept_apart(self, tmp_path: Path) -> None:
        # 「本編.sme」はどこにでもある 名前だけで分けると別の作品の控えが混ざる
        state = tmp_path / "state"
        assert backup_folder(tmp_path / "a" / "本編.sme", state) != backup_folder(
            tmp_path / "b" / "本編.sme", state
        )

    @pytest.mark.parametrize("name", ["a:b*c?.sme", "con.sme"])
    def test_awkward_names_still_get_a_folder(self, tmp_path: Path, name: str) -> None:
        # Windows で使えない文字や予約名がそのまま残ると、控えのフォルダを作れず控えが取れない
        folder = backup_folder(tmp_path / name, tmp_path / "state")
        folder.mkdir(parents=True)
        assert folder.is_dir()


class TestChosenFolders:
    """退避とバックアップの置き場を設定で変えた人のための道具（#271）"""

    def test_orphans_are_found_in_every_folder(self, tmp_path: Path) -> None:
        # 選んだ置き場へ書けずに既定へ戻した起動が落ちると、退避は既定の側に残る
        chosen, default = tmp_path / "選んだ", tmp_path / "既定"
        for root, name in ((chosen, "こちら"), (default, "あちら")):
            session = RecoverySession(root)
            session.save(Project.create(name=name), None)
            crash(session)
        found = find_orphans_in([chosen, default])
        assert sorted(entry.name for entry in found) == ["あちら", "こちら"]

    def test_the_same_folder_is_not_counted_twice(self, tmp_path: Path) -> None:
        # 既定を選んだ人のところでは 2 つの置き場が同じ 2 度数えると同じ退避を 2 回勧める
        session = RecoverySession(tmp_path)
        session.save(Project.create(), None)
        crash(session)
        assert len(find_orphans_in([tmp_path, tmp_path / "."])) == 1

    def test_a_writable_folder_has_no_problem(self, tmp_path: Path) -> None:
        assert folder_problem(tmp_path / "新しい") is None
        # 確かめに書いた物を残さない
        assert list((tmp_path / "新しい").iterdir()) == []

    def test_an_unwritable_folder_says_why(self, tmp_path: Path) -> None:
        blocker = tmp_path / "ファイル"
        blocker.write_text("x", "utf-8")
        assert folder_problem(blocker / "置き場") is not None

    def test_the_backups_over_a_smaller_count_are_counted(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        for name, count in (("一.sme", 5), ("二.sme", 2)):
            target = tmp_path / name
            for number in range(count):
                target.write_text(str(number), "utf-8")
                backup_before_save(target, state)
        assert backups_over(3, state) == 2
        assert backups_over(20, state) == 0
        # 数えるだけで消さない
        assert len(list(backup_folder(tmp_path / "一.sme", state).iterdir())) == 5

    def test_no_backups_yet_counts_nothing(self, tmp_path: Path) -> None:
        assert backups_over(1, tmp_path / "無い") == 0

    def test_close_lets_go_of_the_lock_even_when_clearing_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 置き場のドライブを抜いたあとに閉じても、錠のファイルを開いたままにしない
        session = RecoverySession(tmp_path)

        def unplugged() -> None:
            raise OSError("抜いた")

        monkeypatch.setattr(session, "clear", unplugged)
        with pytest.raises(OSError):
            session.close()
        assert session._lock is None
