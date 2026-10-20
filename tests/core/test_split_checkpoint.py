"""開いた段を途中で区切る（#339 検査の直しを 1 回ごとの段にする）

AI が検査で見つけた問題を直すたびに、直した回だけを人が Ctrl+Z 1 回で戻せるようにする
区切りの前の変更は前の名前の段、後の変更は新しい名前の段として履歴に並ぶことを押さえる
"""

from __future__ import annotations

from sashimono.core.commands import Author, Document, RenameProject
from sashimono.core.model import Project


def _named(document: Document, *names: str) -> None:
    for name in names:
        document.execute(RenameProject(name))


class TestSplit:
    def test_each_fix_round_is_undone_on_its_own(self, project: Project) -> None:
        """壊れると、直した回を戻すつもりの Ctrl+Z 1 回で、指示の編集まで一緒に消える"""
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _named(document, "編集")
        assert document.split_checkpoint("AI: 検査の直し（1 回目）")
        _named(document, "直し 1")
        assert document.split_checkpoint("AI: 検査の直し（2 回目）")
        _named(document, "直し 2")
        document.end_checkpoint()

        assert document.history_labels == (
            "AI: 仕上げて",
            "AI: 検査の直し（1 回目）",
            "AI: 検査の直し（2 回目）",
        )
        assert document.undo_by is Author.AI
        document.undo()
        assert document.project.name == "直し 1"
        document.undo()
        assert document.project.name == "編集"
        document.undo()
        assert document.project is project

    def test_a_split_without_changes_leaves_no_empty_step(self, project: Project) -> None:
        """壊れると、何も変えずに調べ直しただけで中身の無い段が挟まり、Ctrl+Z が空振りする"""
        document = Document(project)
        document.begin_checkpoint("AI: 調べて", by=Author.AI)
        assert document.split_checkpoint("AI: 検査の直し（1 回目）")
        assert document.split_checkpoint("AI: 検査の直し（1 回目）")
        _named(document, "直し")
        document.end_checkpoint()

        assert document.history_labels == ("AI: 検査の直し（1 回目）",)

    def test_the_whole_instruction_still_counts_as_this_instruction(self, project: Project) -> None:
        """壊れると、AI の「この指示を全部戻す」が区切った前の編集を戻さずに止まる"""
        document = Document(project)
        _named(document, "前の指示")
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _named(document, "編集")
        document.split_checkpoint("AI: 検査の直し（1 回目）")
        _named(document, "直し")

        undone = 0
        while document.undo_in_checkpoint:
            document.undo()
            undone += 1
        assert undone == 2
        assert document.project.name == "前の指示"
        document.end_checkpoint()

    def test_it_does_not_split_outside_a_step_or_inside_a_group(self, project: Project) -> None:
        """壊れると、まとめて当てている途中で区切り、閉じる所で履歴の形が合わずに段が消える"""
        document = Document(project)
        assert document.split_checkpoint("AI: 検査の直し（1 回目）") is False
        assert document.checkpoint_label is None
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        with document.checkpoint("道具"):
            _named(document, "途中")
            assert document.split_checkpoint("AI: 検査の直し（1 回目）") is False
        assert document.checkpoint_label == "AI: 仕上げて"
        document.end_checkpoint()
        assert document.history_labels == ("AI: 仕上げて",)

    def test_each_new_instruction_gets_a_new_serial(self, project: Project) -> None:
        """壊れると、同じ文の指示を続けたときに直しの回数が数え直されず、前の指示の分で止まる"""
        document = Document(project)
        first = document.checkpoint_serial
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        with document.checkpoint("道具"):
            _named(document, "途中")
        inside = document.checkpoint_serial
        document.end_checkpoint()
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        assert inside == first + 1
        assert document.checkpoint_serial == first + 2
        document.end_checkpoint()
