"""AI の指示の段を開いたまま作品を差し替える（#349 AI の開く・新規）"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import AddMarkers, Author, Document
from sashimono.core.commands.history import REPLACED_REFUSAL
from sashimono.core.model import Marker, Project


def _project(name: str) -> Project:
    return replace(Project.create(), name=name)


def _mark(label: str) -> AddMarkers:
    return AddMarkers((Marker(10, label),))


class TestResetWithinTheInstruction:
    def test_the_instruction_stays_open_and_leaves_no_step(self) -> None:
        # 壊れると、指示の段が閉じて AI のパネルの閉じる数と食い違うか、開いた作品の上に
        # 前の作品へ戻る段が積まれ、Ctrl+Z 1 回で別の作品の中身へ跳ぶ
        document = Document(_project("前"))
        document.begin_checkpoint("AI: 開く", by=Author.AI)
        document.execute(_mark("前の作品の目印"))
        opened = _project("開いた")
        document.reset(opened, within_checkpoint=True)
        assert document.in_checkpoint and document.replaced_in_checkpoint
        document.end_checkpoint()
        assert document.project is opened
        assert not document.can_undo
        assert not document.replaced_in_checkpoint

    def test_the_assistant_s_edits_are_refused_but_the_person_s_are_not(self) -> None:
        # 壊れると、同じ指示の中で AI が開き直した作品を編集するか、人まで編集できなくなる
        document = Document(_project("前"))
        document.begin_checkpoint("AI: 開く", by=Author.AI)
        document.reset(_project("開いた"), within_checkpoint=True)
        with (
            pytest.raises(ValueError, match=REPLACED_REFUSAL),
            document.checkpoint("道具", by=Author.AI),
        ):
            document.execute(_mark("AI"))
        with pytest.raises(ValueError, match=REPLACED_REFUSAL):
            document.execute(_mark("AI"))
        document.execute(_mark("人"), by=Author.PERSON)
        document.end_checkpoint()
        assert [m.label for m in document.project.timeline.markers] == ["人"]
        # 次の指示では断らない
        document.begin_checkpoint("AI: 次", by=Author.AI)
        document.execute(AddMarkers((Marker(20, "次"),)))
        document.end_checkpoint()
        assert len(document.project.timeline.markers) == 2

    def test_outside_or_inside_a_nested_step_it_is_still_refused(self) -> None:
        # 壊れると、まとめ操作の途中で作品が替わり、まとめの残りが別の作品へ当たる
        document = Document(_project("前"))
        with pytest.raises(RuntimeError):
            document.begin_checkpoint("AI", by=Author.AI)
            document.begin_checkpoint("道具", by=Author.AI)
            document.reset(_project("開いた"), within_checkpoint=True)
        plain = Document(_project("前"))
        plain.begin_checkpoint("人", by=Author.PERSON)
        with pytest.raises(RuntimeError):
            plain.reset(_project("開いた"))
