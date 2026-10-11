"""AI エージェント連携

編集操作を MCP のツールとして公開し、Claude Agent SDK 経由で会話しながら
実際の編集をさせる

層の分け方:

- :mod:`~sashimono.ai.operations` — 何ができるか Qt も MCP も知らない素の関数
- :mod:`~sashimono.ai.host` — 編集ソフト側に求めること
- :mod:`~sashimono.ai.bridge` — AI スレッドから UI スレッドへの橋渡しと確認
- :mod:`~sashimono.ai.server` — 操作を MCP ツールへ包む
- :mod:`~sashimono.ai.session` — SDK との会話

SDK を import するのは server と session だけ 未導入でも操作の定義は読めるので、
チャットパネルは開けるし、導入ボタンも出せる
"""

from sashimono.ai.bridge import Approval, EditorBridge, Question
from sashimono.ai.environment import AI_PACK
from sashimono.ai.host import EditorHost, ToolError
from sashimono.ai.operations import OPERATIONS, ImageResult, Operation, find_operation

__all__ = [
    "AI_PACK",
    "OPERATIONS",
    "Approval",
    "EditorBridge",
    "EditorHost",
    "ImageResult",
    "Operation",
    "Question",
    "ToolError",
    "find_operation",
]
