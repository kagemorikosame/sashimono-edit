"""起動の頭で入れる・前の入れ替えの結果を知らせる"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sashimono.update.flow import apply_on_start, settle
from sashimono.update.package import APP_EXE, Layout, write_build_info
from sashimono.update.state import UpdateState, UpdateStateStore
from sashimono.update.swap import SwapPlan


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
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
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

    def test_turning_checks_off_cancels_the_reservation(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """予約の後に〔起動したときに新しい版を確かめる〕を切った人は、今の版に留まりたい
        設定を読む前に入れ替えると、切ったのに次の起動で黙って入れ替わる
        """
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
        assert not apply_on_start(
            ["x"], layout=layout, store=store, swap=_never, current="1.1.0", allowed=lambda: False
        )
        assert not store.load().apply_on_start
        # 落としてある版は残す 手で確かめれば入れられる
        assert layout.staged_version() == "1.2.0"

    def test_nothing_happens_without_the_mark(
        self, layout: Layout, store: UpdateStateStore
    ) -> None:
        """尋ねる設定で本人が選んでいなければ、起動の頭では入れない"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0"))
        assert not apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.1.0")

    def test_a_vanished_stage_is_not_applied(self, layout: Layout, store: UpdateStateStore) -> None:
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
        assert not apply_on_start(["x"], layout=layout, store=store, swap=_never, current="1.1.0")
        assert not store.load().apply_on_start

    def test_an_older_stage_is_not_applied(self, layout: Layout, store: UpdateStateStore) -> None:
        """入れ替え待ちの版が今より古い（手で新しい版を入れた） 戻してはいけない"""
        _stage(layout, "1.2.0")
        store.save(UpdateState(ready_version="1.2.0", apply_on_start=True))
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
