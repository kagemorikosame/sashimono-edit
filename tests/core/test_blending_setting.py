"""重ね合わせの方法（sRGB / リニア）をプロジェクトの設定として持つこと（Issue #65）

新しく作るプロジェクトは AviUtl・YMM4 と同じ sRGB で混ぜる この項目ができる前に
保存したプロジェクトはリニアで描いていたので、開くときもリニアのまま 取り違えると、
保存した作品の半透明の文字やフェードが、開き直しただけで暗くなる
"""

from __future__ import annotations

import pytest

from sashimono.core.commands import AddScene, Document, InScene, SetBlending, new_scene
from sashimono.core.io import (
    FORMAT_VERSION,
    ProjectFileError,
    project_from_dict,
    project_to_dict,
)
from sashimono.core.model import Blending, Project, ProjectSettings
from sashimono.engine.render import changed_spans


class TestDefault:
    def test_a_new_project_blends_like_aviutl(self) -> None:
        # 既定がリニアだと、読み込んだ AviUtl・YMM4 の素材の半透明が 60 近く明るく浮く
        assert Project.create().settings.blending == Blending.SRGB

    def test_an_unknown_blending_is_refused(self) -> None:
        # 通すと、どちらで混ぜるかがレンダラの作り次第になる
        with pytest.raises(ValueError, match="重ね合わせ"):
            ProjectSettings(blending="multiply")


def _saved(blending: str) -> dict[str, object]:
    project = Project.create(ProjectSettings(blending=blending))
    return project_to_dict(project)


class TestFile:
    def test_an_old_file_opens_with_linear_blending(self) -> None:
        """項目の無いファイル（版 2 まで）はリニアで開く

        新規作成の既定（sRGB）で開くと、保存したときより半透明の所が暗くなる
        """
        data = _saved(Blending.SRGB)
        settings = data["settings"]
        assert isinstance(settings, dict)
        del settings["blending"]
        data["version"] = 2
        assert project_from_dict(data).settings.blending == Blending.LINEAR

    def test_a_new_file_without_blending_is_refused(self) -> None:
        """版 3 のファイルで項目が無ければ壊れたファイルとして断る

        版 3 は必ず書く 無いのに古いファイルと同じくリニアで補うと、sRGB で作った
        作品が黙って明るく描かれる
        """
        data = _saved(Blending.SRGB)
        settings = data["settings"]
        assert isinstance(settings, dict)
        del settings["blending"]
        with pytest.raises(ProjectFileError, match="重ね合わせ"):
            project_from_dict(data)

    @pytest.mark.parametrize("blending", Blending.ALL)
    def test_the_blending_survives_saving(self, blending: str) -> None:
        # 書き忘れると、sRGB で作った作品が開き直すたびにリニアへ戻る
        assert project_from_dict(_saved(blending)).settings.blending == blending

    def test_the_format_version_is_raised(self) -> None:
        """書くファイルは版 3 項目を知らない古い本体に、黙って開かせない

        古い本体で開くと sRGB の作品をリニアで描き、保存し直すと項目ごと消える
        """
        data = _saved(Blending.SRGB)
        assert FORMAT_VERSION >= 3
        assert data["version"] == FORMAT_VERSION

    def test_a_broken_blending_is_a_file_error(self) -> None:
        # 素の ValueError で漏れると、壊れたファイル 1 つで起動ごと落ちる
        data = _saved(Blending.SRGB)
        settings = data["settings"]
        assert isinstance(settings, dict)
        settings["blending"] = "rgb"
        with pytest.raises(ProjectFileError):
            project_from_dict(data)


class TestCommand:
    def test_the_change_can_be_undone(self) -> None:
        # モデルを直に書き換えると、取り消しで戻らない
        document = Document(Project.create())
        document.execute(SetBlending(Blending.LINEAR))
        assert document.project.settings.blending == Blending.LINEAR
        document.undo()
        assert document.project.settings.blending == Blending.SRGB

    def test_an_unknown_blending_is_refused(self) -> None:
        # AI や台本が知らない方式を渡して通ると、保存した作品が次に開くとき
        # 読めないファイルとして断られる
        with pytest.raises(ValueError, match="重ね合わせ"):
            SetBlending("add").apply(Project.create())

    def test_it_works_from_inside_a_scene(self) -> None:
        # シーンを開いたまま設定画面で変えても、プロジェクト全体に効く
        project = Project.create()
        scene = new_scene(project, "中")
        project = AddScene(scene).apply(project)
        changed = InScene(scene.id, SetBlending(Blending.LINEAR)).apply(project)
        assert changed.settings.blending == Blending.LINEAR


def test_switching_throws_away_every_prefetched_frame() -> None:
    """切り替えたら先読みした絵を全部捨てる

    半透明の所があるフレームはすべて明るさが変わる 捨て損ねると、先読みが
    当たった所だけ前の混ぜ方の絵が出る
    """
    project = Project.create()
    invalidation = changed_spans(project, SetBlending(Blending.LINEAR).apply(project))
    assert invalidation.everything
