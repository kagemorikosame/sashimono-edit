"""作品の表記（:attr:`~sashimono.core.model.Project.credits` #335）を足すコマンド

ナレーションの声を作ったときに、声の規約が求める表記（「VOICEVOX:ずんだもん」）を覚える
声を置くのと同じ取り消し 1 回で戻るよう、置く命令と同じまとめに積む
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sashimono.core.commands.base import Command
from sashimono.core.model import Project

__all__ = ["CREDIT_LIMIT", "AddCredits"]

#: 1 つの表記の長さの上限（文字） 声の名前とエンジンの名前を足した長さより十分長い
#: 壊れたエンジンの返す長い名前で、書き出しの窓の知らせが読めなくならないように
CREDIT_LIMIT = 200


@dataclass(frozen=True, slots=True)
class AddCredits(Command):
    """表記を足す すでにある物は足さない（同じ声を何度作っても 1 行）"""

    credits: tuple[str, ...]

    @property
    def label(self) -> str:
        return "表記を追加"

    def apply(self, project: Project) -> Project:
        added = [
            line
            for line in dict.fromkeys(credit.strip() for credit in self.credits)
            if line and line not in project.credits
        ]
        for line in added:
            if len(line) > CREDIT_LIMIT:
                raise ValueError(f"表記が長すぎます（{CREDIT_LIMIT} 文字まで）")
        if not added:
            # 同じ物を返すと履歴に段を積まない（同じ声を作り直しても段が増えない）
            return project
        return replace(project, credits=(*project.credits, *added))
