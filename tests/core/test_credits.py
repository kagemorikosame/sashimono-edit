"""作品が覚える表記（ナレーションの声のクレジット #335）"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sashimono.core.commands import CREDIT_LIMIT, AddCredits, Author, Document
from sashimono.core.io.serialize import (
    ProjectFileError,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from sashimono.core.model import Project


class TestAddingCredits:
    def test_a_new_credit_is_kept_and_undone_in_one_step(self) -> None:
        # 壊れると、声を作った表記が作品に残らず、書き出しのときに知らせ漏らす（規約に反する
        # のは本人） 取り消しで戻らないと、消した声の表記がいつまでも知らされる
        document = Document(Project.create())
        document.execute(AddCredits(("VOICEVOX:ずんだもん",)), by=Author.AI)
        assert document.project.credits == ("VOICEVOX:ずんだもん",)
        document.undo()
        assert not document.project.credits

    def test_the_same_credit_is_not_added_twice(self) -> None:
        # 壊れると、同じ声で何度も作るたびに同じ表記が並び、概要欄に貼る文が長くなる
        project = AddCredits(("VOICEVOX:ずんだもん",)).apply(Project.create())
        again = AddCredits(("VOICEVOX:ずんだもん", "VOICEVOX:四国めたん")).apply(project)
        assert again.credits == ("VOICEVOX:ずんだもん", "VOICEVOX:四国めたん")

    def test_nothing_new_leaves_the_project_as_it_is(self) -> None:
        # 壊れると、何も足さなかった表記の命令が履歴に段を積み、AI の取り消しを人の段で止める
        project = AddCredits(("VOICEVOX:ずんだもん",)).apply(Project.create())
        assert AddCredits(("VOICEVOX:ずんだもん", " ")).apply(project) is project

    def test_a_too_long_credit_is_refused(self) -> None:
        # 壊れると、壊れたエンジンの返す長い名前で、書き出しの窓の知らせが読めなくなる
        with pytest.raises(ValueError, match="長すぎ"):
            AddCredits(("あ" * (CREDIT_LIMIT + 1),)).apply(Project.create())


class TestSavingCredits:
    def test_credits_survive_saving_and_opening(self, tmp_path: Path) -> None:
        # 壊れると、保存して開き直した作品で、表記を知らせなくなる
        project = AddCredits(("VOICEVOX:ずんだもん",)).apply(Project.create())
        path = tmp_path / "作品.sme"
        save_project(project, path)
        assert load_project(path).credits == ("VOICEVOX:ずんだもん",)

    def test_a_project_without_credits_writes_nothing_new(self) -> None:
        # 壊れると、声を作っていない作品のファイルにも空の項目が増え、前の版と中身が変わる
        assert "credits" not in project_to_dict(Project.create())

    def test_an_older_file_opens_without_credits(self) -> None:
        # 壊れると、表記の項目を持たない前の作品が開けない
        data = project_to_dict(Project.create())
        data.pop("credits", None)
        assert project_from_dict(json.loads(json.dumps(data))).credits == ()

    def test_a_broken_credit_is_refused(self) -> None:
        # 壊れると、手で書き換えた壊れた表記を黙って捨て、知らせるべき表記を落とす
        data = project_to_dict(Project.create())
        data["credits"] = ["VOICEVOX:ずんだもん", 3]
        with pytest.raises(ProjectFileError, match="credits"):
            project_from_dict(data)
