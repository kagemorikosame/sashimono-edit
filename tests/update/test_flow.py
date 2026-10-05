"""起動の頭で入れる・前の入れ替えの結果を知らせる"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.core.io.locks import try_hold
from sashimono.update.flow import (
    UpdateBusyError,
    UpdateChoices,
    apply_on_start,
    prepare,
    reconcile,
    settle,
)
from sashimono.update.package import APP_EXE, Layout, write_build_info
from sashimono.update.state import (
    STAGE_LOCK,
    SWAP_LOCK,
    UpdateState,
    UpdateStateStore,
    lock_path,
)
from sashimono.update.swap import SwapPlan, take_result
from tests.update.helpers import release


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    install = tmp_path / "Sashimono"
    install.mkdir()
    (install / APP_EXE).write_bytes(b"MZ")
    return Layout(install)


@pytest.fixture
def store(tmp_path: Path) -> UpdateStateStore:
    return UpdateStateStore(tmp_path / "state.json")


def _stage(layout: Layout, version: str) -> None:
    layout.staged.mkdir()
    (layout.staged / APP_EXE).write_bytes(b"MZ new")
    write_build_info(layout.staged, version, "cp314")


class TestApplyingOnStart:
    def test_a_chosen_version_is_handed_to_the_swapper(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        plans: list[SwapPlan] = []

        def swap(plan: SwapPlan) -> bool:
            plans.append(plan)
            return True

        assert apply_on_start(
            ["Sashimono.exe", "作品.sme"], layout=layout, store=store, swap=swap, current="1.1.0"
        )
        assert len(plans) == 1
        plan = plans[0]
        assert (plan.mode, plan.pid, tuple(plan.arguments)) == ("apply", os.getpid(), ("作品.sme",))
        # 印は入れ替え係を起こす前に下ろす 失敗し続ける機械で、起動のたびに試さない
        assert not store.load().apply_on_start

    def test_files_that_cannot_be_carried_stop_the_swap(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """exe の隣の本人の物を新しい版へ写せなければ入れ替えない（PR #245 の CodeRabbit の指摘）

        写せないまま入れ替えると、本人の物は次の更新で消える .previous にだけ残る
        """
        _stage(layout, "1.2.0")
        mine = layout.install / "scripts" / "自分の" / "効果.anm2"
        mine.parent.mkdir(parents=True)
        mine.write_text("--track", encoding="utf-8")
        # 写す先にフォルダを作れない（同じ名前のファイルが塞いでいる）
        (layout.staged / "scripts").mkdir()
        (layout.staged / "scripts" / "自分の").write_text("塞ぐ", encoding="utf-8")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        plans: list[SwapPlan] = []

        def swap(plan: SwapPlan) -> bool:
            plans.append(plan)
            return True

        assert not apply_on_start(["x"], layout=layout, store=store, swap=swap, current="1.1.0")
        assert plans == []
        state = store.load()
        assert state.auto_blocked == "1.2.0" and not state.apply_on_start
        assert "自分の/効果.anm2" in state.pending_notice and "止めました" in state.pending_notice
        # 落として確かめた新しい版は残す 今の版はそのまま動く
        assert layout.staged_version() == "1.2.0"
        assert mine.is_file() and (layout.install / APP_EXE).is_file()

    def test_parked_originals_return_before_the_swap(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """移している途中で閉じて元をよけたまま残っても、起動の頭の入れ替えの前に元の場所へ
        戻して新しい版へ写す（PR #245 の Codex の指摘 戻さずに入れ替えると .previous へ
        回って消える）
        """
        _stage(layout, "1.2.0")
        parked = layout.install / ".scripts-removing-4242" / "自分の" / "効果.anm2"
        parked.parent.mkdir(parents=True)
        parked.write_text("--track", encoding="utf-8")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        plans: list[SwapPlan] = []

        def swap(plan: SwapPlan) -> bool:
            plans.append(plan)
            return True

        assert apply_on_start(["x"], layout=layout, store=store, swap=swap, current="1.1.0")
        assert plans
        assert (layout.install / "scripts" / "自分の" / "効果.anm2").is_file()
        assert (layout.staged / "scripts" / "自分の" / "効果.anm2").is_file()

    def test_a_marker_that_cannot_be_written_starts_no_swapper(
        self, layout: Layout, store: UpdateStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """印を書けなければ入れ替え係を起こさない（PR #245 の CodeRabbit の指摘）"""
        from sashimono.update import package as package_module

        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))

        def unwritable(*_args: object) -> None:
            raise PermissionError("書けない")

        monkeypatch.setattr(package_module, "mark_swap_pending", unwritable)
        assert not apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.1.0")
        assert "印を書けない" in store.load().pending_notice

    def test_the_swap_pending_marker_follows_the_swapper(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """引き継ぎは錠を放す前に「入れ替え係を待っている」印を置く（移しがその隙に始まらない）
        入れ替え係を起こせなければ外す
        """
        from sashimono.core import userdirs
        from sashimono.update.portable import swap_pending

        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        seen: list[bool] = []

        def swap(plan: SwapPlan) -> bool:
            seen.append(swap_pending(userdirs.config_root()))
            return True

        assert apply_on_start(["x"], layout=layout, store=store, swap=swap, current="1.1.0")
        assert seen == [True] and swap_pending(userdirs.config_root())

        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))

        def cannot_start(plan: SwapPlan) -> bool:
            return False  # 台本の実行が止められている など

        assert not apply_on_start(
            ["x"], layout=layout, store=store, swap=cannot_start, current="1.1.0"
        )
        # 起こせなかった 2 回目は自分の印だけを外す 1 回目（起こせた）の印は残る
        markers = list(userdirs.config_root().glob("scripts-move.swap-pending.*"))
        assert len(markers) == 1
        markers[0].unlink()
        assert not swap_pending(userdirs.config_root())

    def test_a_move_in_another_window_stops_the_swap(
        self, layout: Layout, store: UpdateStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """別の窓が移している（錠を持っている）間は、起動の頭でも入れ替えない 次の起動で試す"""
        from sashimono.core import userdirs
        from sashimono.update import package as package_module
        from sashimono.update.portable import MOVE_LOCK

        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        monkeypatch.setattr(package_module, "CARRY_LOCK_WAIT", 0.2)
        held = try_hold(userdirs.config_root() / MOVE_LOCK)
        assert held is not None
        try:
            assert not apply_on_start(["x"], layout=layout, store=store, swap=_never)
        finally:
            held.release()
        assert "移している" in store.load().pending_notice

    def test_turning_checks_off_cancels_the_reservation(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """予約の後に〔起動したときに新しい版を確かめる〕を切った人は、今の版に留まりたい
        設定を読む前に入れ替えると、切ったのに次の起動で黙って入れ替わる
        """
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
        assert not apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=_never,
            current="1.1.0",
            choices=lambda: UpdateChoices(check=False),
        )
        assert not store.load().apply_on_start
        # 落としてある版は残す 手で確かめれば入れられる
        assert layout.staged_version() == "1.2.0"

    def test_turning_confirm_on_drops_an_automatic_reservation(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """尋ねない設定が自動で付けた予約は、後で〔入れる前に尋ねる〕を入れたら入れない"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
        assert not apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=_never,
            current="1.1.0",
            choices=lambda: UpdateChoices(confirm=True),
        )
        assert not store.load().apply_on_start

    def test_an_automatic_reservation_applies_while_not_asking(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
        assert apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=lambda _plan: True,
            current="1.1.0",
            choices=lambda: UpdateChoices(confirm=False),
        )

    def test_a_chosen_reservation_applies_while_asking(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """本人が〔次の起動で入れる〕を選んだ予約は、尋ねる設定でも入れる（もう答えてある）"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        assert apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=lambda _plan: True,
            current="1.1.0",
            choices=lambda: UpdateChoices(confirm=True),
        )

    def test_a_running_swapper_ends_this_start(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """入れ替え係が走っている間の起動は、画面を出さずに終わる（入れ替え係を待たせない）
        2 つ目の入れ替え係も起こさない
        """
        held = try_hold(lock_path(SWAP_LOCK, store.path.parent))
        assert held is not None
        try:
            assert apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.1.0")
        finally:
            held.release()

    def test_staging_elsewhere_is_left_alone(self, layout: Layout, store: UpdateStateStore) -> None:
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        held = try_hold(lock_path(STAGE_LOCK, store.path.parent))
        assert held is not None
        try:
            assert not apply_on_start(
                ["x"], layout=layout, store=store, swap=_never, current="1.1.0"
            )
        finally:
            held.release()
        # 予約は残す 落とし終えた後の起動で入れる
        assert store.load().apply_on_start


#: 予約の 3 通り 無い・本人が選んだ・尋ねない設定が自動で付けた
NONE = UpdateState(ready_version="1.2.0")
CHOSEN = UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True)
AUTOMATIC = UpdateState(ready_version="1.2.0", apply_on_start=True)
BETA_CHOSEN = UpdateState(ready_version="1.3.0b1", apply_on_start=True, apply_chosen=True)
BETA_AUTOMATIC = UpdateState(ready_version="1.3.0b1", apply_on_start=True)


def _reserved(state: UpdateState) -> tuple[str, bool, bool]:
    return state.ready_version, state.apply_on_start, state.apply_chosen


class TestTheSwitchTable:
    """3 つの設定の、どちらの向きの切り替えでも、予約が今の好みにそろう（docs の表と同じ）

    切り替えた後の好み（``after``）だけで決まる 切り替える前が何だったかで答えが変わると、
    画面で切り替えたときと起動の頭とで別の答えになる
    """

    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            # 起動したときに確かめる 入 → 切 予約は全部外す 待っている版は残す
            (CHOSEN, UpdateChoices(check=False), ("1.2.0", False, False)),
            (AUTOMATIC, UpdateChoices(check=False, confirm=False), ("1.2.0", False, False)),
            # 切 → 入 尋ねない設定なら自動で付ける 尋ねる設定なら付けない
            (NONE, UpdateChoices(check=True, confirm=False), ("1.2.0", True, False)),
            (NONE, UpdateChoices(check=True, confirm=True), ("1.2.0", False, False)),
            # 入れる前に尋ねる 入 → 切 待っている版を自動で予約する（Codex P2）
            (NONE, UpdateChoices(confirm=False), ("1.2.0", True, False)),
            (CHOSEN, UpdateChoices(confirm=False), ("1.2.0", True, True)),
            # 切 → 入 自動の予約だけ外す 選んだ予約は残す
            (AUTOMATIC, UpdateChoices(confirm=True), ("1.2.0", False, False)),
            (CHOSEN, UpdateChoices(confirm=True), ("1.2.0", True, True)),
            # ベータ版も受け取る 入 → 切 先行版は予約ごと外す（Codex P2） 正式版は残す
            (BETA_CHOSEN, UpdateChoices(beta=False), ("", False, False)),
            (BETA_AUTOMATIC, UpdateChoices(beta=False, confirm=False), ("", False, False)),
            (CHOSEN, UpdateChoices(beta=False), ("1.2.0", True, True)),
            # 切 → 入 変えない
            (BETA_CHOSEN, UpdateChoices(beta=True), ("1.3.0b1", True, True)),
            (BETA_AUTOMATIC, UpdateChoices(beta=True, confirm=False), ("1.3.0b1", True, False)),
        ],
    )
    def test_every_switch_lands_on_the_same_rule(
        self, before: UpdateState, after: UpdateChoices, expected: tuple[str, bool, bool]
    ) -> None:
        assert _reserved(reconcile(before, after)) == expected

    def test_a_failed_version_is_not_reserved_again(self) -> None:
        """入れ替えに失敗した版を自動で予約し直すと、起動のたびに失敗して待たされる"""
        blocked = replace(NONE, auto_blocked="1.2.0")
        assert _reserved(reconcile(blocked, UpdateChoices(confirm=False))) == (
            "1.2.0",
            False,
            False,
        )
        # 本人が選んだ予約は入れる
        chosen = replace(CHOSEN, auto_blocked="1.2.0")
        assert reconcile(chosen, UpdateChoices(confirm=False)).apply_on_start

    def test_it_is_settled(self) -> None:
        """2 度通しても変わらない（起動の頭と画面の両方が通しても食い違わない）"""
        for state in (NONE, CHOSEN, AUTOMATIC, BETA_CHOSEN, BETA_AUTOMATIC):
            for choices in (
                UpdateChoices(check, confirm, beta)
                for check in (True, False)
                for confirm in (True, False)
                for beta in (True, False)
            ):
                once = reconcile(state, choices)
                assert reconcile(once, choices) == once


class TestStartFollowsTheTable:
    def test_turning_confirm_off_applies_on_the_next_start(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """確認ありで落とした後に〔入れる前に尋ねる〕を切った 説明どおり次の起動の頭で入れる"""
        _stage(layout, "1.2.0")
        store.save(NONE)
        plans: list[SwapPlan] = []

        def swap(plan: SwapPlan) -> bool:
            plans.append(plan)
            return True

        assert apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=swap,
            current="1.1.0",
            choices=lambda: UpdateChoices(confirm=False),
        )
        assert len(plans) == 1

    def test_a_beta_is_not_applied_after_turning_beta_off(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """ベータを予約した後にベータを切った人に、次の起動でベータを入れない"""
        _stage(layout, "1.3.0b1")
        store.save(BETA_CHOSEN)
        assert not apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=_never,
            current="1.2.0",
            choices=lambda: UpdateChoices(beta=False),
        )
        assert store.load().ready_version == ""

    def test_a_failed_swap_is_not_retried_on_every_start(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        _stage(layout, "1.2.0")
        store.save(NONE)
        not_asking = lambda: UpdateChoices(confirm=False)  # noqa: E731
        assert not apply_on_start(
            ["x"],
            layout=layout,
            store=store,
            swap=lambda _plan: False,
            current="1.1.0",
            choices=not_asking,
        )
        assert store.load().auto_blocked == "1.2.0"
        assert not apply_on_start(
            ["x"], layout=layout, store=store, swap=_never, current="1.1.0", choices=not_asking
        )

    def test_a_failure_in_the_results_blocks_the_automatic_retry(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        _stage(layout, "1.2.0")
        store.save(NONE)
        settle(layout=layout, store=store, current="1.1.0", results=["busy"])
        assert store.load().auto_blocked == "1.2.0"


class TestOneAtATime:
    """窓を 2 つ開いていても、落とす・展開するのは 1 つだけ"""

    def test_a_second_window_does_not_stage(self, layout: Layout, store: UpdateStateStore) -> None:
        """同時に落とすと、片方が展開している .new をもう片方が消す"""
        transport, manifest = release(Ed25519PrivateKey.generate(), "1.2.0")
        held = try_hold(lock_path(STAGE_LOCK, store.path.parent))
        assert held is not None
        try:
            with pytest.raises(UpdateBusyError):
                prepare(manifest, transport=transport, layout=layout, store=store)
        finally:
            held.release()
        assert transport.requested == []
        prepare(manifest, transport=transport, layout=layout, store=store)
        assert layout.staged_version() == "1.2.0"

    def test_nothing_is_staged_while_swapping(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        transport, manifest = release(Ed25519PrivateKey.generate(), "1.2.0")
        held = try_hold(lock_path(SWAP_LOCK, store.path.parent))
        assert held is not None
        try:
            with pytest.raises(UpdateBusyError):
                prepare(manifest, transport=transport, layout=layout, store=store)
        finally:
            held.release()

    def test_settling_spares_a_stage_in_progress(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """ほかの窓が展開し終えて覚え書きを書く前の .new を、起動の片付けで消さない"""
        _stage(layout, "1.2.0")
        held = try_hold(lock_path(STAGE_LOCK, store.path.parent))
        assert held is not None
        try:
            settle(layout=layout, store=store, current="1.1.0", results=[])
        finally:
            held.release()
        assert layout.staged.exists()

    def test_results_are_not_read_while_swapping(self, tmp_path: Path) -> None:
        """走っている入れ替え係の結果を消すと、その後の結果が次の起動まで残る"""
        folder = tmp_path / "update"
        folder.mkdir()
        (folder / "result-1.txt").write_text("started\nswapped\n", encoding="utf-8")
        held = try_hold(lock_path(SWAP_LOCK, folder))
        assert held is not None
        try:
            assert take_result(folder) == []
        finally:
            held.release()
        assert take_result(folder) == ["started", "swapped"]
        assert list(folder.glob("result-*.txt")) == []

    def test_nothing_happens_without_the_mark(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """尋ねる設定で本人が選んでいなければ、起動の頭では入れない"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0"))
        assert not apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.1.0")

    def test_a_vanished_stage_is_not_applied(self, layout: Layout, store: UpdateStateStore) -> None:
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        assert not apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.1.0")
        assert not store.load().apply_on_start

    def test_an_older_stage_is_not_applied(self, layout: Layout, store: UpdateStateStore) -> None:
        """入れ替え待ちの版が今より古い（手で新しい版を入れた） 戻してはいけない"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True, apply_chosen=True))
        assert not apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.3.0")

    def test_the_development_tree_is_left_alone(self, store: UpdateStateStore) -> None:
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
        assert not apply_on_start(["x"], layout=None, store=store, swap=_never)


def _never(_plan: SwapPlan) -> bool:
    raise AssertionError("入れ替え係を起こしてはいけない")


class TestSettling:
    def test_a_finished_update_is_told(self, layout: Layout, store: UpdateStateStore) -> None:
        store.save(UpdateState(ready_version="1.2.0", ready_notes_url="https://x"))
        notice = settle(layout=layout, store=store, current="1.2.0", results=["swapped"])
        assert notice is not None and "1.2.0 に更新しました" in notice
        assert store.load().ready_version == ""

    def test_a_rolled_back_version_is_skipped(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """起動できなかった版を、次の確認でまた落として入れない"""
        store.save(UpdateState(ready_version="1.2.0"))
        notice = settle(
            layout=layout, store=store, current="1.1.0", results=["swapped", "rolled-back"]
        )
        assert notice is not None and "戻しました" in notice
        state = store.load()
        assert state.skipped == ("1.2.0",)
        assert state.ready_version == ""

    def test_a_failed_restore_says_where_it_runs(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """戻しまで失敗したら、前の版を隣のフォルダから起こしたことを言う（名前を戻す案内）"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0"))
        notice = settle(
            layout=layout,
            store=store,
            current="1.1.0",
            results=["started", "staged-locked", "restore-failed", "started-from-aside"],
        )
        assert notice is not None and ".previous" in notice

    def test_an_unfinished_rollback_still_skips_the_version(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        store.save(UpdateState(ready_version="1.2.0"))
        notice = settle(
            layout=layout, store=store, current="1.1.0", results=["swapped", "rollback-failed"]
        )
        assert notice is not None and "戻し切れなかった" in notice
        assert "1.2.0" in store.load().skipped

    def test_a_failed_swap_is_told(self, layout: Layout, store: UpdateStateStore) -> None:
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0"))
        notice = settle(layout=layout, store=store, current="1.1.0", results=["busy"])
        assert notice is not None and "ほかの Sashimono の窓" in notice
        # 待っている版はそのまま 手で入れ直せる
        assert store.load().ready_version == "1.2.0"

    def test_nothing_to_tell(self, layout: Layout, store: UpdateStateStore) -> None:
        assert settle(layout=layout, store=store, current="1.1.0", results=[]) is None

    def test_an_orphan_stage_is_removed(self, layout: Layout, store: UpdateStateStore) -> None:
        """待つ版の無い展開済みのフォルダ（100 MB 超）を残さない"""
        _stage(layout, "1.2.0")
        settle(layout=layout, store=store, current="1.1.0", results=[])
        assert not layout.staged.exists()

    def test_a_broken_state_does_not_stop_the_start(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        store.path.write_text("{壊れた", encoding="utf-8")
        assert settle(layout=layout, store=store, current="1.1.0", results=[]) is None


class TestTheState:
    def test_it_round_trips(self, store: UpdateStateStore) -> None:
        state = UpdateState(
            last_checked=12.5,
            skipped=("1.0.0",),
            ready_version="1.2.0",
            ready_notes_url="https://x",
            ready_python_abi="cp314",
            apply_on_start=True,
        )
        store.save(state)
        assert store.load() == state

    @pytest.mark.parametrize(
        "text",
        ['{"last_checked": "昨日", "skipped": "1.0", "apply_on_start": 1}', "[]", "壊れた"],
    )
    def test_broken_values_fall_back(self, store: UpdateStateStore, text: str) -> None:
        store.path.write_text(text, encoding="utf-8")
        assert store.load() == UpdateState()
