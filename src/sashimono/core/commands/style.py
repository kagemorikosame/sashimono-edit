"""編集の方針（:class:`~sashimono.core.model.StyleGuide` #343）を変えるコマンド

方針は作品に付く 変えるのはプロジェクト設定の窓だけで、AI の道具には変える物を置かない
（本人が AI に渡す決まりなので、AI が自分で緩められないようにする）
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sashimono.core.commands.base import Command
from sashimono.core.model import Project, StyleGuide

__all__ = ["SetStyleGuide"]


@dataclass(frozen=True, slots=True)
class SetStyleGuide(Command):
    """方針を丸ごと差し替える 欄ごとのコマンドにしない

    窓の OK 1 回で書いた物は 1 回の取り消しで戻す 欄ごとに積むと、取り消しの途中で
    書体だけ戻った、本人の書いたことの無い方針を通る
    """

    guide: StyleGuide

    @property
    def label(self) -> str:
        return "編集の方針を変更"

    def apply(self, project: Project) -> Project:
        guide = self.guide.cleaned()
        problems = guide.problems()
        if problems:
            raise ValueError(" ".join(problems))
        if guide == project.style_guide:
            # 同じ物を返すと履歴に段を積まない（何も変えなかった OK で、人の段が 1 つ増えない）
            return project
        return replace(project, style_guide=guide)
