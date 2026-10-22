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


#: 名前を変えるコマンドの段の名前
RENAMED = RenameProject("x").label


class TestStepsSplitOff:
    """前半として先に積んだ段の中の手の名前（#346 引き継ぎの「変えた手」に使う）"""

    def test_names_split_off_by_a_person_are_kept(self, project: Project) -> None:
        """壊れると、指示の途中で人が変えたとき、それまでに AI が変えた手の名前が消え、
        AI がその後何もせずに止まると引き継ぎに「変えた手 0 件」と出る
        """
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _named(document, "1", "2")
        document.execute(RenameProject("人"), by=Author.PERSON)
        assert document.steps_split_off == (RENAMED, RENAMED)
        assert document.steps_in_checkpoint == ()
        assert (document.parts_split_by_others, document.parts_split_for_rounds) == (1, 0)

    def test_names_split_off_for_a_fix_round_are_kept(self, project: Project) -> None:
        """壊れると、直しの回で区切った前の回の手が、引き継ぎから消える"""
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _named(document, "1")
        assert document.split_checkpoint("AI: 検査の直し（1 回目）")
        _named(document, "2")
        assert document.steps_split_off == (RENAMED,)
        assert (document.parts_split_by_others, document.parts_split_for_rounds) == (0, 1)

    def test_both_kinds_of_split_keep_their_order(self, project: Project) -> None:
        """壊れると、直しの回と人の割り込みが混ざった指示で、どちらかの手の名前が抜ける"""
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _named(document, "1")
        document.split_checkpoint("AI: 検査の直し（1 回目）")
        _named(document, "2", "3")
        document.execute(RenameProject("人"), by=Author.PERSON)
        _named(document, "4")
        assert document.steps_split_off == (RENAMED, RENAMED, RENAMED)
        assert document.steps_in_checkpoint == (RENAMED,)
        assert (document.parts_split_by_others, document.parts_split_for_rounds) == (1, 1)

    def test_undoing_a_split_off_part_drops_its_names(self, project: Project) -> None:
        """壊れると、指示の中で戻した前半の手も、引き継ぎに変えた手として残る"""
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _named(document, "1", "2")
        document.split_checkpoint("AI: 検査の直し（1 回目）")
        document.undo()
        assert document.project is project
        # == () と書くと、mypy が型を空のタプルに絞り、やり直した後の比べを通さない
        assert not document.steps_split_off
        assert document.parts_split_for_rounds == 0
        document.redo()
        assert document.steps_split_off == (RENAMED, RENAMED)

    def test_names_do_not_carry_over_to_the_next_instruction(self, project: Project) -> None:
        """壊れると、前の指示で割った前半の手が、次の指示の引き継ぎに混ざる"""
        document = Document(project)
        document.begin_checkpoint("AI: 前の指示", by=Author.AI)
        _named(document, "1")
        document.execute(RenameProject("人"), by=Author.PERSON)
        document.end_checkpoint()
        assert document.steps_split_off == ()
        document.begin_checkpoint("AI: 次の指示", by=Author.AI)
        assert document.steps_split_off == ()
        assert document.parts_split_by_others == 0


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


class TestChangesBy:
    def test_changes_are_counted_per_author(self, project: Project) -> None:
        """壊れると、AI の指示の途中で人が手で変えた分まで AI の変更に数え、AI が直した回を
        数え違えて、直す前に止まる"""
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        with document.checkpoint("道具", by=Author.AI):
            _named(document, "一", "二")
        document.execute(RenameProject("人"), by=Author.PERSON)
        _named(document, "三")
        document.undo()
        document.end_checkpoint()
        # まとめは 1 回 段の中の AI のコマンドも 1 回 取り消しは数えない
        assert document.changes_by(Author.AI) == 2
        assert document.changes_by(Author.PERSON) == 1

    def test_a_refused_group_is_not_counted(self, project: Project) -> None:
        """壊れると、断られて何も変わらなかった道具を AI が直した回に数える"""
        document = Document(project)
        document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        try:
            with document.checkpoint("断られる道具", by=Author.AI):
                _named(document, "途中")
                raise KeyError("断る")
        except KeyError:
            pass
        document.end_checkpoint()
        assert document.changes_by(Author.AI) == 0
