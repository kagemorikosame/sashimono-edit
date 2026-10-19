"""編集の方針（スタイルガイド #343）のモデル・コマンド・保存の形

方針は作品に付き、保存した作品を開き直しても同じ方針で AI が編集する 前の版のファイル
（方針の項目が無い）は「決めていない」で開く
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sashimono.core.commands import Author, Document, SetStyleGuide
from sashimono.core.io.serialize import (
    FORMAT_VERSION,
    ProjectFileError,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from sashimono.core.model import STYLE_ITEMS_LIMIT, STYLE_TEXT_LIMIT, Project, StyleGuide
from sashimono.core.model.style import STYLE_EFFECT_NAME_LIMIT, STYLE_NAME_LIMIT

GUIDE = StyleGuide(
    fonts=("源ノ角ゴシック", "M PLUS 1p"),
    colors=("#ffffff", "#000000"),
    subtitles="下の中央に白の字・黒の縁",
    tempo="言葉の間は 0.3 秒まで詰める",
    effects=("blur", "shadow"),
    notes="絵文字は使わない",
)


class TestModel:
    def test_a_new_project_has_no_guide(self) -> None:
        """壊れると、方針を決めていない作品でも AI が空の決まりに縛られて読みに行く"""
        assert Project.create().style_guide.empty

    def test_cleaning_drops_blank_lines_and_repeats(self) -> None:
        """壊れると、窓の空の行が「決めた書体」に数えられ、照らしの結果に同じ物が 2 度並ぶ"""
        written = StyleGuide(
            fonts=(" 源ノ角ゴシック ", "", "源ノ角ゴシック"),
            colors=("#FFFFFF", "  ", "#ffffff"),
            tempo="  速め \n",
        )
        assert written.cleaned() == StyleGuide(
            fonts=("源ノ角ゴシック",), colors=("#ffffff",), tempo="速め"
        )

    @pytest.mark.parametrize("color", ["red", "#fff", "#gggggg", "ffffff"])
    def test_a_colour_not_in_rrggbb_is_a_problem(self, color: str) -> None:
        """通すと、照らしのときに数に直せず、方針の色がどれとも一致しない"""
        assert StyleGuide(colors=(color,)).cleaned().problems()

    def test_long_text_and_long_lists_are_problems(self) -> None:
        """通すと、AI へ毎回送る方針が膨らみ、読むたびに費用がかさむ"""
        assert StyleGuide(notes="あ" * (STYLE_TEXT_LIMIT + 1)).problems()
        too_many = tuple(f"書体{n}" for n in range(STYLE_ITEMS_LIMIT + 1))
        assert StyleGuide(fonts=too_many).problems()
        assert not GUIDE.problems()

    def test_a_long_name_is_a_problem(self) -> None:
        """名前の長さを見ないと、数の上限の中でも方針が膨らみ、AI へ渡す返事の上限を越えて切れる"""
        assert StyleGuide(fonts=("あ" * (STYLE_NAME_LIMIT + 1),)).problems()
        assert not StyleGuide(fonts=("あ" * STYLE_NAME_LIMIT,)).problems()
        assert StyleGuide(effects=("a" * (STYLE_EFFECT_NAME_LIMIT + 1),)).problems()

    def test_a_real_aviutl_effect_name_fits(self) -> None:
        """手元の AviUtl のスクリプトの種類は最長 79 字

        書体と同じ 64 字では、その効果に印を付けた方針を持てない
        """
        kind = (
            "aviutl:YamatanaMarkerCaptionMini/@YamatanaMarkerCaptionMini.obj2:"
            "マーカー字幕ミニ（進捗制御）"
        )
        assert len(kind) == 79
        assert not StyleGuide(effects=(kind,)).problems()

    def test_fitting_keeps_what_can_be_kept(self) -> None:
        """丸ごと捨てると、手で足しすぎた 1 つのために書いた方針が全部消える 長すぎる名前は頭だけ
        残すとどの書体とも合わないので、切らずに除く"""
        written = StyleGuide(
            fonts=("源ノ角ゴシック", "あ" * (STYLE_NAME_LIMIT + 1)),
            colors=("#ffffff", "白"),
            notes="長い" * STYLE_TEXT_LIMIT,
        )
        fitted = written.fitted()
        assert fitted.fonts == ("源ノ角ゴシック",)
        assert fitted.colors == ("#ffffff",)
        assert len(fitted.notes) == STYLE_TEXT_LIMIT
        assert not fitted.problems()

    def test_a_name_with_a_line_break_is_a_problem(self) -> None:
        """通すと、1 行に 1 つの欄へ戻したときに 2 つの書体に割れる"""
        assert StyleGuide(fonts=("源ノ\n角ゴシック",)).problems()


class TestCommand:
    def test_setting_is_undone_in_one_step(self) -> None:
        """壊れると、方針を書いた後に Ctrl+Z 1 回で前の方針へ戻れない"""
        document = Document(Project.create())
        document.execute(SetStyleGuide(GUIDE), by=Author.PERSON)
        assert document.project.style_guide == GUIDE
        assert document.undo_label == "編集の方針を変更"
        assert document.undo_by is Author.PERSON
        document.undo()
        assert document.project.style_guide.empty

    def test_the_same_guide_leaves_no_step(self) -> None:
        """壊れると、何も変えずに OK を押しただけで履歴に人の段が積まれ、AI の取り消しが止まる"""
        project = SetStyleGuide(GUIDE).apply(Project.create())
        assert SetStyleGuide(GUIDE).apply(project) is project

    def test_a_bad_guide_is_refused(self) -> None:
        """通すと、読めない色の方針が保存され、照らしがその色で落ちる"""
        with pytest.raises(ValueError, match="#RRGGBB"):
            SetStyleGuide(StyleGuide(colors=("white",))).apply(Project.create())

    def test_the_command_stores_the_cleaned_guide(self) -> None:
        """壊れると、大文字の色と小文字の色が別の色として残り、照らしで同じ色が外れる"""
        project = SetStyleGuide(StyleGuide(colors=("#FFFFFF",))).apply(Project.create())
        assert project.style_guide.colors == ("#ffffff",)


def _written(project: Project) -> dict[str, Any]:
    data = project_to_dict(project)
    return json.loads(json.dumps(data, ensure_ascii=False))  # type: ignore[no-any-return]


class TestFile:
    def test_the_guide_comes_back_after_saving(self, tmp_path: Path) -> None:
        """壊れると、保存して開き直した作品で方針が消え、AI が決まりを知らずに編集する"""
        project = SetStyleGuide(GUIDE).apply(Project.create())
        path = tmp_path / "作品.sme"
        save_project(project, path)
        assert load_project(path).style_guide == GUIDE

    def test_no_guide_writes_nothing(self) -> None:
        """方針を使わない作品のファイルは前と同じ中身のまま（項目を足すと差分が増える）"""
        assert "style_guide" not in _written(Project.create())

    def test_a_version_9_file_opens_with_no_guide(self) -> None:
        """0.2.2 までの作品（版 9 項目が無い）を開けないか、方針の欄が壊れて開く"""
        data = _written(Project.create())
        data["version"] = 9
        assert project_from_dict(data).style_guide.empty

    def test_the_version_is_not_raised(self) -> None:
        """10 はまだどの公開版も書いていない 上げると、0.2.2 の人へ 2 段の版上げを強いる

        9 の本体は 10 のファイルを「更新してください」で止めるので、方針を黙って捨てて
        保存し直すことは無い（:data:`FORMAT_VERSION` の注記）
        """
        data = _written(SetStyleGuide(GUIDE).apply(Project.create()))
        assert data["version"] == FORMAT_VERSION == 10

    def test_a_newer_file_is_still_refused(self) -> None:
        """方針を足しても、知らない版を開かない決まりは変わらない"""
        data = _written(SetStyleGuide(GUIDE).apply(Project.create()))
        data["version"] = FORMAT_VERSION + 1
        with pytest.raises(ProjectFileError, match="更新"):
            project_from_dict(data)

    @pytest.mark.parametrize(
        "broken",
        [
            "方針",
            {"fonts": "源ノ角ゴシック"},
            {"fonts": [1]},
            {"notes": 3},
        ],
    )
    def test_a_broken_guide_is_a_broken_file(self, broken: object) -> None:
        """素のまま漏らすと、壊れたファイル 1 つで起動ごと落ちる

        開く側は ProjectFileError しか受けない
        """
        data = _written(Project.create())
        data["style_guide"] = broken
        with pytest.raises(ProjectFileError):
            project_from_dict(data)

    def test_an_unchecked_colour_still_opens(self) -> None:
        """手で書き換えた 1 色のために作品ごと開けなくなる（目印の色と同じ扱い）"""
        data = _written(Project.create())
        data["style_guide"] = {"colors": ["white"]}
        assert project_from_dict(data).style_guide.colors == ("white",)
