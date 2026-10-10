"""保存していない作業の退避と、上書き前のバックアップ

退避は「落ちたときにだけ」拾えなければならない 生きている別の起動の退避を
拾うと、同じ作業が 2 つの窓で別々に進み、どちらかの変更が消える
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, tzinfo
from pathlib import Path

import pytest

from sashimono.core.io import (
    RecoveryEntry,
    RecoverySession,
    backup_before_save,
    backup_folder,
    backups_over,
    discard,
    discard_items,
    find_orphans,
    find_orphans_in,
    folder_problem,
    forget_empty_roots,
    load_project,
    mark_offered,
    plan_prune,
    plan_trim,
    recovery,
    remember_root,
    remembered_roots,
    state_usage,
    tidy_orphans,
    trim_state,
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


def _offered_orphan(root: Path, name: str) -> RecoveryEntry:
    """落ちた作業を 1 つ作り、復元を勧めた印を付ける"""
    session = RecoverySession(root)
    session.save(Project.create(name=name), None)
    crash(session)
    (entry,) = [found for found in find_orphans(root) if found.name == name]
    mark_offered(entry)
    return entry


class TestTidyingOldOrphans:
    """残った退避を日数で片付ける（#271） 消してはいけない物が残ることを中心に見る"""

    later = datetime.now() + timedelta(days=10)

    def test_marking_is_remembered(self, tmp_path: Path) -> None:
        _offered_orphan(tmp_path, "見せた")
        assert [entry.offered for entry in find_orphans(tmp_path)] == [True]

    def test_an_old_offered_orphan_is_tidied(self, tmp_path: Path) -> None:
        _offered_orphan(tmp_path, "見せた")
        tidied = tidy_orphans([tmp_path], 7, now=self.later)
        assert [entry.name for entry in tidied] == ["見せた"]
        assert find_orphans(tmp_path) == []
        # 印も一緒に消す 残ると、同じ名前の起動は無いので印だけが溜まる
        assert list((tmp_path / "recovery").iterdir()) == []

    def test_zero_days_tidies_nothing(self, tmp_path: Path) -> None:
        # 既定は片付けない（前からの動き）
        _offered_orphan(tmp_path, "見せた")
        assert tidy_orphans([tmp_path], 0, now=self.later) == []
        assert len(find_orphans(tmp_path)) == 1

    def test_a_never_offered_orphan_is_kept_however_old(self, tmp_path: Path) -> None:
        # 長く起動しなかった人のところで、一度も見ていない落ちた作業を消さない
        session = RecoverySession(tmp_path)
        session.save(Project.create(name="見せていない"), None)
        crash(session)
        assert tidy_orphans([tmp_path], 1, now=self.later) == []
        assert [entry.name for entry in find_orphans(tmp_path)] == ["見せていない"]

    def test_a_recent_offered_orphan_is_kept(self, tmp_path: Path) -> None:
        _offered_orphan(tmp_path, "見せた")
        assert tidy_orphans([tmp_path], 30, now=self.later) == []
        assert len(find_orphans(tmp_path)) == 1

    def test_a_live_session_is_kept(self, tmp_path: Path) -> None:
        session = RecoverySession(tmp_path)
        session.save(Project.create(name="作業中"), None)
        try:
            assert tidy_orphans([tmp_path], 1, now=self.later) == []
            assert session.path.is_file()
        finally:
            session.close()


class TestTheSizeLimit:
    """置き場の容量の上限（#271） 消してはいけない物が残ることを中心に見る"""

    def _backups(self, state: Path, name: str, count: int, size: int = 1000) -> list[Path]:
        target = state.parent / name
        made = []
        for number in range(count):
            target.write_bytes(bytes([number]) * size)
            copied = backup_before_save(target, state)
            assert copied is not None
            made.append(copied)
        return made

    def test_under_the_limit_nothing_goes(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        self._backups(state, "本編.sme", 3)
        assert plan_trim(10_000, state) == []

    def test_the_oldest_goes_first(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        made = self._backups(state, "本編.sme", 4)
        # 時刻はファイルの更新時刻ではなく名前で見る 控えは元の更新時刻を写すので、
        # 更新時刻で並べると作った順と食い違う
        os.utime(made[0], (made[-1].stat().st_mtime + 100,) * 2)
        items = plan_trim(state_usage(state) - 500, state)
        assert [item.paths for item in items] == [(made[0],)]
        # 数えるだけでは消さない
        assert made[0].is_file()

    def test_the_newest_backup_of_each_project_is_kept(self, tmp_path: Path) -> None:
        # 最後の保存の前の中身 これが消えると「さっきの保存で壊した」を戻せない
        state = tmp_path / "state"
        first = self._backups(state, "一.sme", 3)
        second = self._backups(state, "二.sme", 1)
        trimmed = trim_state(1, state)
        assert sorted(path for item in trimmed for path in item.paths) == sorted(first[:-1])
        assert first[-1].is_file()
        assert second[-1].is_file()

    def test_live_and_unoffered_recovery_are_kept(self, tmp_path: Path) -> None:
        # 開いている作業の今の退避と、まだ勧めていない落ちた作業は、超えていても消さない
        state = tmp_path / "state"
        live = RecoverySession(state)
        live.save(Project.create(name="作業中"), None)
        unseen = RecoverySession(state)
        unseen.save(Project.create(name="見せていない"), None)
        crash(unseen)
        offered = _offered_orphan(state, "見せた")
        try:
            trimmed = trim_state(1, state)
            assert [item.label for item in trimmed] == ["落ちた作業 見せた"]
            assert live.path.is_file()
            assert unseen.path.is_file()
            assert not offered.path.exists()
        finally:
            live.close()

    def test_the_items_say_what_they_are(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        self._backups(state, "本編.sme", 2)
        (item,) = plan_trim(1, state)
        assert item.label == "バックアップ 本編"
        assert item.size == 1000

    def test_only_the_confirmed_items_go(self, tmp_path: Path) -> None:
        # 確かめた後に置き場の中身が変わっても、確かめに出していない物は消さない
        state = tmp_path / "state"
        made = self._backups(state, "本編.sme", 4)
        confirmed = plan_trim(state_usage(state) - 500, state)
        assert [item.paths for item in confirmed] == [(made[0],)]
        trimmed = trim_state(1, state, allowed=[p for item in confirmed for p in item.paths])
        assert [item.paths for item in trimmed] == [(made[0],)]
        assert made[1].is_file() and made[2].is_file()

    def test_nothing_confirmed_means_nothing_goes(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        made = self._backups(state, "本編.sme", 4)
        assert trim_state(1, state, allowed=[]) == []
        assert all(path.is_file() for path in made)


class TestRememberedFolders:
    """退避を書いた置き場を覚える（置き場を変えた後に落ちた作業を見失わない）"""

    def test_they_come_back_newest_first(self, tmp_path: Path) -> None:
        registry = tmp_path / "既定"
        remember_root(tmp_path / "A", registry)
        remember_root(tmp_path / "B", registry)
        remember_root(tmp_path / "A", registry)
        assert remembered_roots(registry) == [tmp_path / "A", tmp_path / "B"]

    def test_nothing_remembered_is_empty(self, tmp_path: Path) -> None:
        assert remembered_roots(tmp_path) == []

    def test_a_broken_file_is_empty(self, tmp_path: Path) -> None:
        # 読めないからと復元の検索を止めない
        (tmp_path / "places.json").write_text("{壊れている", "utf-8")
        assert remembered_roots(tmp_path) == []

    def test_a_folder_with_work_left_is_not_forgotten(self, tmp_path: Path) -> None:
        registry = tmp_path / "既定"
        crashed = RecoverySession(tmp_path / "A")
        crashed.save(Project.create(name="落ちた"), None)
        crash(crashed)
        RecoverySession(tmp_path / "B").close()
        remember_root(tmp_path / "A", registry)
        remember_root(tmp_path / "B", registry)
        remember_root(tmp_path / "C", registry)
        forget_empty_roots([tmp_path / "C"], registry)
        # B は空なので忘れる A は落ちた作業が残っている C は今使っている
        assert remembered_roots(registry) == [tmp_path / "C", tmp_path / "A"]


class TestPruningGenerations:
    """世代数で消す控え 見せた数と消える数を一致させる"""

    def _made(self, state: Path, count: int) -> list[Path]:
        target = state.parent / "本編.sme"
        made = []
        for number in range(count):
            target.write_bytes(bytes([number]) * 100)
            copied = backup_before_save(target, state, keep=200)
            assert copied is not None
            made.append(copied)
        return made

    def test_a_save_drops_at_most_what_it_is_told(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        made = self._made(state, 6)
        target = tmp_path / "本編.sme"
        backup_before_save(target, state, keep=2, most=1)
        assert not made[0].exists()
        assert all(path.is_file() for path in made[1:])

    def test_the_plan_matches_the_count(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        made = self._made(state, 6)
        plan = plan_prune(2, state)
        assert [item.paths for item in plan] == [(path,) for path in made[:4]]
        assert backups_over(2, state) == len(plan)
        assert all(path.is_file() for path in made)

    def test_discarding_the_plan_removes_exactly_it(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        made = self._made(state, 6)
        plan = plan_prune(2, state)
        # 同じ物が 2 つの一覧（世代数と容量）に入っていても 1 度だけ数える
        done = discard_items([*plan, plan[0]])
        assert len(done) == 4
        assert [path.exists() for path in made] == [False] * 4 + [True] * 2

    def test_the_size_plan_leaves_out_what_pruning_frees(self, tmp_path: Path) -> None:
        # 世代数で空く分を知らずに容量の計画を立てると、余計な物まで挙げる
        state = tmp_path / "state"
        made = self._made(state, 6)
        pruned = [path for item in plan_prune(2, state) for path in item.paths]
        assert plan_trim(state_usage(state) - 300, state, already=pruned) == []
        extra = plan_trim(state_usage(state) - 450, state, already=pruned)
        assert [item.paths for item in extra] == [(made[4],)]
