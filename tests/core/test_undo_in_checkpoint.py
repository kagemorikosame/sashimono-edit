"""開いた段（チェックポイント）の中の取り消しとやり直し（#333）

AI の 1 回の指示は、始めから終わりまで 1 つの段を開いたまま進む（ui/chat/panel.py）
その間に AI の道具や人の Ctrl+Z が取り消しを頼むと、段の外の履歴を引き抜いてしまい、
閉じた後の履歴から 1 段が消えていた ここでは段の中の取り消しが、段の中の小さな段
（道具 1 回分）から順に戻し、段の外の履歴を壊さないことを押さえる
"""

from __future__ import annotations

import pytest

from sashimono.core.commands import AddTrack, Author, Document, RenameProject
from sashimono.core.model import Project, Track, TrackKind


def _named(document: Document, *names: str) -> None:
    for name in names:
        document.execute(RenameProject(name))


class TestUndoInsideAnOpenStep:
    def test_undo_in_a_turn_reverts_only_the_last_change_of_the_turn(
        self, project: Project
    ) -> None:
        """壊れると、AI が直前の 1 手だけ戻すつもりで前の指示まで戻し、閉じた後の履歴が飛ぶ

        閉じた後に人が 1 段ずつ戻すと、段が 1 つ消えていて途中の形へ戻れない
        """
        document = Document(project)
        _named(document, "一", "二")
        document.begin_checkpoint("AI: 名前を変えて")
        _named(document, "三", "四")

        document.undo()

        assert document.project.name == "三"
        document.end_checkpoint()
        assert document.history_labels == (
            "プロジェクト名を変更",
            "プロジェクト名を変更",
            "AI: 名前を変えて",
        )
        # 人が 1 段ずつ戻すと、指示の前・二・一・最初の順に辿れる（どの段も飛ばない）
        document.undo()
        assert document.project.name == "二"
        document.undo()
        assert document.project.name == "一"
        document.undo()
        assert document.project.name == project.name

    def test_a_tool_that_grouped_several_commands_is_undone_as_one(self, project: Project) -> None:
        """壊れると、道具 1 回でまとめて当てた変更のうち、最後の 1 つだけが戻って半端な形が残る"""
        document = Document(project)
        document.begin_checkpoint("AI: まとめて")
        _named(document, "前")
        with document.checkpoint("道具"):
            _named(document, "途中", "後")

        assert document.undo_label == "道具"
        document.undo()

        assert document.project.name == "前"
        document.end_checkpoint()

    def test_a_failed_tool_leaves_nothing_to_undo(self, project: Project) -> None:
        """壊れると、断られて何も変わらなかった道具の分まで取り消しの手数に数えられる"""
        document = Document(project)
        document.begin_checkpoint("AI: 失敗する")
        _named(document, "前")
        with pytest.raises(KeyError), document.checkpoint("断られる道具"):
            _named(document, "途中")
            raise KeyError("断る")

        assert document.undo_label == "プロジェクト名を変更"
        document.undo()
        assert document.project is project
        document.end_checkpoint()
        assert document.can_undo is False

    def test_with_nothing_done_in_the_turn_undo_reverts_the_previous_step(
        self, project: Project
    ) -> None:
        """壊れると、「前の指示を取り消して」と頼んでも段の外を戻せず、戻せても閉じた所で消える"""
        document = Document(project)
        document.begin_checkpoint("AI: 前の指示")
        _named(document, "前の指示の結果")
        document.end_checkpoint()

        document.begin_checkpoint("AI: さっきのを取り消して")
        assert document.undo_label == "AI: 前の指示"
        document.undo()
        assert document.project is project
        document.end_checkpoint()

        # 取り消しだけの指示は段を残さず、人は Ctrl+Y で前の指示をやり直せる
        assert document.can_undo is False
        assert document.redo_label == "AI: 前の指示"
        document.redo()
        assert document.project.name == "前の指示の結果"

    def test_what_the_turn_undid_at_the_end_can_be_redone_after_it_closes(
        self, project: Project
    ) -> None:
        """壊れると、AI が最後に取り消した 1 手を、指示が終わった後に人がやり直せない"""
        document = Document(project)
        document.begin_checkpoint("AI: 試して戻す")
        _named(document, "残す", "戻す")
        document.undo()
        document.end_checkpoint()

        assert document.project.name == "残す"
        assert document.history_labels == ("AI: 試して戻す",)
        assert document.redo_label == "プロジェクト名を変更"
        document.redo()
        assert document.project.name == "戻す"
        # やり直した後も 2 段で、指示の前まで戻れる
        document.undo()
        document.undo()
        assert document.project is project

    def test_redo_in_a_turn_brings_back_what_the_turn_undid(self, project: Project) -> None:
        """壊れると、AI が取り消しすぎた 1 手を同じ指示の中で戻せない"""
        document = Document(project)
        document.begin_checkpoint("AI: 行って戻って")
        _named(document, "一", "二")
        document.undo()
        assert document.can_redo is True
        document.redo()
        assert document.project.name == "二"
        document.end_checkpoint()
        assert document.history_labels == ("AI: 行って戻って",)
        assert document.can_redo is False

    def test_a_new_change_in_the_turn_drops_the_older_redo(self, project: Project) -> None:
        """壊れると、取り消した後に別の編集をしたのに、古いやり直しで新しい編集が消える"""
        document = Document(project)
        _named(document, "一")
        document.undo()
        assert document.can_redo is True

        document.begin_checkpoint("AI: 別の名前に")
        _named(document, "別")
        assert document.can_redo is False
        document.undo()
        document.end_checkpoint()

        # 戻せるのは指示の中で取り消した「別」だけ 先に取り消した「一」は戻らない
        assert document.redo_label == "プロジェクト名を変更"
        document.redo()
        assert document.project.name == "別"
        assert document.can_redo is False

    def test_a_failed_group_keeps_the_redo_from_before(self, project: Project) -> None:
        """壊れると、断られて何も変わらなかったまとめ操作のせいで、前に取り消した物をやり直せない"""
        document = Document(project)
        _named(document, "一")
        document.undo()
        with pytest.raises(KeyError), document.checkpoint("途中で断られる"):
            _named(document, "途中")
            raise KeyError("断る")
        assert document.redo_label == "プロジェクト名を変更"

    def test_undo_while_a_group_is_being_applied_does_nothing(self, project: Project) -> None:
        """壊れると、まとめて当てている途中に届いた Ctrl+Z が、閉じる所で戻した物を変えたと数える

        窓の中で待つ間（名前を尋ねる窓など）にキーが届くことがある 途中の形を戻すと、
        まとめの始まりと行き先が食い違い、履歴の段が今の形と合わなくなる
        """
        document = Document(project)
        document.begin_checkpoint("AI: 指示")
        _named(document, "前")
        with document.checkpoint("まとめ"):
            _named(document, "途中")
            assert document.can_undo is False
            assert document.undo_label is None
            document.undo()
            document.redo()
            assert document.project.name == "途中"
        assert document.history_labels == ("プロジェクト名を変更", "まとめ")
        document.end_checkpoint()
        # 読み直す（mypy は前の比べた形のまま見て、閉じた後の形と食い違うと言う）
        closed: tuple[str, ...] = tuple(document.history_labels)
        assert closed == ("AI: 指示",)

    def test_projects_only_in_the_turn_are_kept_for_hiding_names(self, project: Project) -> None:
        """壊れると、指示の中で取り消した作品の名前が、AI へ送る結果の伏せ字の表から漏れる"""
        document = Document(project)
        document.begin_checkpoint("AI: 名前")
        _named(document, "ひとつめ", "ふたつめ")
        undone = document.project
        document.undo()

        assert undone in document.past_projects
        document.end_checkpoint()
        assert undone in document.past_projects


def _by_ai(document: Document, name: str) -> None:
    """AI の道具 1 回分（ホストの apply_commands と同じく AI の印を付けてまとめる）"""
    with document.checkpoint("AI の道具", by=Author.AI):
        document.execute(RenameProject(name))


def _person_adds_a_track(document: Document) -> None:
    """人が画面で 1 つ変えた（編集画面と同じく人の印を付ける）"""
    document.execute(AddTrack(Track(TrackKind.AUDIO, "人の足したトラック")), by=Author.PERSON)


def _undo_by(document: Document) -> Author | None:
    # 読むたびに今の値を見る（mypy は前に比べた値のまま見て、変わった後の比べ方を咎める）
    return document.undo_by


def _redo_by(document: Document) -> Author | None:
    return document.redo_by


def _tracks(document: Document) -> int:
    return len(document.project.timeline.tracks)


class TestWhoMadeTheStep:
    """段ごとの「誰が変えたか」の印（2026-10-18 利用者の決定 AI は人の段を戻さない）"""

    def test_each_step_remembers_who_made_it(self, project: Project) -> None:
        """壊れると、AI が人の手の段と自分の段を見分けられず、人の変更を巻き戻す"""
        document = Document(project)
        _person_adds_a_track(document)
        assert _undo_by(document) is Author.PERSON
        _by_ai(document, "AI")
        assert _undo_by(document) is Author.AI
        document.undo()
        # 取り消した後のやり直しも、誰の段かを覚えている
        assert _redo_by(document) is Author.AI
        assert _undo_by(document) is Author.PERSON

    def test_a_persons_edit_during_an_ai_instruction_is_its_own_step(
        self, project: Project
    ) -> None:
        """壊れると、AI の指示の途中に人が変えた物が「AI: 〜」の段に混ざり、指示と一緒に消える"""
        document = Document(project)
        tracks = _tracks(document)
        document.begin_checkpoint("AI: 名前を変えて", by=Author.AI)
        _by_ai(document, "AI の一つ目")
        _person_adds_a_track(document)
        _by_ai(document, "AI の二つ目")
        document.end_checkpoint()

        assert document.history_labels == (
            "AI: 名前を変えて",
            "トラックを追加: 人の足したトラック",
            "AI: 名前を変えて",
        )
        assert _undo_by(document) is Author.AI
        # 指示の後半を取り消しても、人の足したトラックは残る
        document.undo()
        assert document.project.name == "AI の一つ目"
        assert _tracks(document) == tracks + 1
        assert _undo_by(document) is Author.PERSON

    def test_a_persons_grouped_edit_during_an_instruction_is_its_own_step(
        self, project: Project
    ) -> None:
        """壊れると、人がまとめて当てた操作（貼り付けなど）が AI の指示の段に飲み込まれる"""
        document = Document(project)
        document.begin_checkpoint("AI: 指示", by=Author.AI)
        _by_ai(document, "AI")
        with document.checkpoint("人のまとめ", by=Author.PERSON):
            _person_adds_a_track(document)
            _person_adds_a_track(document)
        document.end_checkpoint()

        assert document.history_labels == ("AI: 指示", "人のまとめ")
        assert _undo_by(document) is Author.PERSON
        document.undo()
        assert _tracks(document) == len(project.timeline.tracks)
        assert document.project.name == "AI"

    def test_a_persons_edit_right_at_the_start_leaves_no_empty_ai_step(
        self, project: Project
    ) -> None:
        """壊れると、AI がまだ何もしていない所で人が変えると、何も起きない「AI: 〜」の段が挟まる"""
        document = Document(project)
        document.begin_checkpoint("AI: 指示", by=Author.AI)
        _person_adds_a_track(document)
        _by_ai(document, "AI")
        document.end_checkpoint()
        assert document.history_labels == ("トラックを追加: 人の足したトラック", "AI: 指示")

    def test_a_failed_persons_edit_does_not_split_the_instruction(self, project: Project) -> None:
        """壊れると、断られて何も変わらなかった人の操作のせいで、AI の指示が 2 段に割れる"""
        document = Document(project)
        document.begin_checkpoint("AI: 指示", by=Author.AI)
        _by_ai(document, "一")
        with pytest.raises(KeyError), document.checkpoint("人の断られる操作", by=Author.PERSON):
            _person_adds_a_track(document)
            raise KeyError("断る")
        _by_ai(document, "二")
        document.end_checkpoint()
        assert document.history_labels == ("AI: 指示",)

    def test_the_persons_ctrl_z_still_undoes_everything(self, project: Project) -> None:
        """壊れると、人が Ctrl+Z で AI の段も人の段も戻せず、印のせいで履歴の途中で止まる"""
        document = Document(project)
        document.begin_checkpoint("AI: 指示", by=Author.AI)
        _by_ai(document, "一")
        _person_adds_a_track(document)
        _by_ai(document, "二")
        document.end_checkpoint()
        for _ in range(3):
            document.undo()
        assert document.project is project
        for _ in range(3):
            document.redo()
        assert document.project.name == "二"

    def test_split_instruction_projects_stay_in_the_table_for_hiding_names(
        self, project: Project
    ) -> None:
        """壊れると、人の手で割れた指示の前半の形が伏せ字の表から落ち、そこにだけある名前が漏れる"""
        document = Document(project)
        document.begin_checkpoint("AI: 指示", by=Author.AI)
        _by_ai(document, "前半だけの名前")
        first_half = document.project
        _person_adds_a_track(document)
        _by_ai(document, "後半")
        document.end_checkpoint()
        assert first_half in document.past_projects
        for _ in range(3):
            document.undo()
        assert first_half in document.past_projects
