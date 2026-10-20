"""AI の開く・新規（#349 2026-10-20 利用者の決定）

- 取り消せない操作として毎回確かめる（「変更を自動で承認」でも）
- 保存していない変更があれば断り、save_project か人の窓を案内する
- 開けるのは作品のフォルダと書き出しの窓で前に選んだフォルダの中の .sme だけ
- 開いた・作った指示の中では、AI の編集の道具を断る 人の編集は止めない

本物の窓とつないだ所は ``tests/ui/test_ai_screen_parity.py``
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import UNSAVED_REFUSAL, ToolError
from sashimono.ai.operations import Operation, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddMarkers, Author
from sashimono.core.commands.history import REPLACED_REFUSAL
from sashimono.core.io.atomic import expect_now
from sashimono.core.io.serialize import save_project
from sashimono.core.model import Marker, Project
from sashimono.core.timebase import FrameRate
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_safety import answer_approvals


def _operation(name: str) -> Operation:
    operation = find_operation(name)
    assert operation is not None, name
    return operation


OPEN = _operation("open_project")
NEW = _operation("new_project")


@pytest.fixture
def saved(host: FakeHost, tmp_path: Path) -> FakeHost:
    """作品を tmp_path/作品/作品.sme に保存し、同じフォルダに別の作品 第二稿.sme を置いてある"""
    folder = tmp_path / "作品"
    folder.mkdir()
    host.saved_at = folder / "作品.sme"
    host.write_project(host.saved_at, adopt=False, expect=expect_now(host.saved_at))
    other = replace(Project.create(), name="第二稿")
    save_project(other, folder / "第二稿.sme")
    return host


def _second(host: FakeHost) -> Path:
    assert host.saved_at is not None
    return host.saved_at.parent / "第二稿.sme"


class TestTheKindOfOperation:
    @pytest.mark.parametrize("operation", [OPEN, NEW])
    def test_it_is_asked_every_time(self, operation: Operation) -> None:
        # 壊れると、「変更を自動で承認」の人の所で、AI が黙って作品を閉じて別の作品にする
        assert operation.safety is Safety.IRREVERSIBLE

    def test_the_bridge_asks_even_with_auto_approve(self, saved: FakeHost) -> None:
        # 壊れると、確かめの欄が出ないまま作品が差し替わり、人は何が起きたか分からない
        bridge = allow_sending(EditorBridge(saved, auto_approve=True))
        done, box = call_in_thread(bridge, OPEN, {"path": str(_second(saved))})
        seen = answer_approvals(bridge, done, lambda _approval: False)
        assert len(seen) == 1 and seen[0].irreversible
        assert "閉じます" in seen[0].summary
        assert isinstance(box.get("error"), ToolError)
        assert saved.document.project.name != "第二稿"


class TestOpening:
    def test_a_saved_project_is_swapped_for_the_one_in_the_folder(self, saved: FakeHost) -> None:
        # 壊れると、AI に「第二稿を開いて」と頼んでも開けない
        result = OPEN(saved, {"path": "第二稿.sme"})
        assert saved.document.project.name == "第二稿"
        assert saved.saved_at == _second(saved).resolve()
        assert result["name"] == "第二稿"
        assert not saved.document.can_undo

    def test_unsaved_changes_are_refused_with_the_way_forward(self, saved: FakeHost) -> None:
        # 壊れると、保存していない編集が、AI の〔開く〕で黙って消える
        saved.apply_commands([AddMarkers((Marker(10, "目印"),))], "目印")
        before = saved.document.project
        with pytest.raises(ToolError) as caught:
            OPEN(saved, {"path": "第二稿.sme"})
        assert str(caught.value) == UNSAVED_REFUSAL
        assert "save_project" in UNSAVED_REFUSAL
        assert saved.document.project is before

    @pytest.mark.parametrize("name", ["外.sme", "第二稿.txt"])
    def test_only_sme_files_in_the_allowed_folders_open(
        self, saved: FakeHost, tmp_path: Path, name: str
    ) -> None:
        # 壊れると、AI が本人の見せるつもりの無い作品や、作品でないファイルを読み込む
        outside = tmp_path / name
        save_project(Project.create(), outside)
        with pytest.raises(ToolError):
            OPEN(saved, {"path": str(outside)})

    def test_nothing_opens_before_the_project_has_a_place(self, host: FakeHost) -> None:
        # 壊れると、まだ保存していない作品から、どこの作品でも開けてしまう
        with pytest.raises(ToolError, match="開ける場所がありません"):
            OPEN(host, {"path": "C:/作品/第二稿.sme"})


class TestInsideOneInstruction:
    def test_editing_after_opening_is_refused_until_the_instruction_ends(
        self, saved: FakeHost
    ) -> None:
        # 壊れると、開き直した作品へ同じ指示のまま編集が入り、指示の取り消しが前の作品を指す
        saved.document.begin_checkpoint("AI: 開いて直す", by=Author.AI)
        OPEN(saved, {"path": "第二稿.sme"})
        opened = saved.document.project
        with pytest.raises(ToolError, match=REPLACED_REFUSAL):
            _operation("add_text")(saved, {"text": "字"})
        assert saved.document.project is opened
        saved.document.end_checkpoint()
        # 次の指示では編集できる 開いた所までは取り消しで戻らない（履歴は差し替えで消えた）
        saved.document.begin_checkpoint("AI: 次", by=Author.AI)
        _operation("add_text")(saved, {"text": "字"})
        saved.document.end_checkpoint()
        saved.document.undo()
        assert saved.document.project is opened
        assert not saved.document.can_undo

    def test_the_person_can_still_edit_the_opened_project(self, saved: FakeHost) -> None:
        # 壊れると、AI が開いた作品を、指示が終わるまで人も触れない
        saved.document.begin_checkpoint("AI: 開く", by=Author.AI)
        OPEN(saved, {"path": "第二稿.sme"})
        saved.document.execute(AddMarkers((Marker(5, "人"),)), by=Author.PERSON)
        saved.document.end_checkpoint()
        assert [marker.label for marker in saved.document.project.timeline.markers] == ["人"]


class TestNew:
    def test_a_new_project_takes_the_size_and_rate_asked(self, host: FakeHost) -> None:
        # 壊れると、「縦の 60fps の作品を新しく作って」と頼んでも既定の作品になる
        result = NEW(host, {"width": 1080, "height": 1920, "frame_rate": 59.94})
        settings = host.document.project.settings
        assert settings.resolution == (1080, 1920)
        assert settings.frame_rate == FrameRate(60000, 1001)
        assert result["resolution"] == "1080x1920"
        assert host.document.project.timeline.tracks == ()

    def test_unsaved_changes_are_refused(self, host: FakeHost) -> None:
        # 壊れると、保存していない編集が、AI の〔新規〕で黙って消える
        host.apply_commands([AddMarkers((Marker(10, "目印"),))], "目印")
        with pytest.raises(ToolError, match="save_project"):
            NEW(host, {})

    @pytest.mark.parametrize(
        "arguments", [{"width": 1081, "height": 1920}, {"width": 1920}, {"frame_rate": 0}]
    )
    def test_settings_that_cannot_be_exported_are_refused(
        self, host: FakeHost, arguments: dict[str, object]
    ) -> None:
        # 壊れると、書き出しの最後でエンコーダに断られる作品を作ってしまう
        before = host.document.project
        with pytest.raises(ToolError):
            NEW(host, arguments)
        assert host.document.project is before
