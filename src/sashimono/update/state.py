"""自動更新が覚えておくこと（最後に確かめた時刻・用意できた版・飛ばす版）

本人の好み（確かめるか・ベータを受け取るか・尋ねるか）は ``Preferences`` に置く
ここは好みではなく、更新の仕組みが次の起動へ渡す覚え書き 置き場は退避の側
（``%LOCALAPPDATA%\\Sashimono\\update``） 設定と一緒に別の機械へ持っていく物ではない

壊れていても起動は止めない 既定から始めれば、次に確かめたときに作り直せる
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sashimono.core import userdirs

__all__ = ["UpdateState", "UpdateStateStore", "update_dir"]


def update_dir() -> Path:
    """覚え書き・入れ替えの道具・その結果を置く場所"""
    return userdirs.state_root() / "update"


@dataclass(frozen=True, slots=True)
class UpdateState:
    """次の起動へ渡すこと"""

    #: 最後に確かめに行った時刻（UNIX 秒） 失敗した回も数える 繋がらない間に
    #: 起動のたびに取りに行かないため
    last_checked: float = 0.0
    #: 「この版を飛ばす」と言われた版・入れたら起動できずに戻した版
    skipped: tuple[str, ...] = field(default_factory=tuple)
    #: 落として確かめ、隣のフォルダへ展開し終えた版 空なら無い
    ready_version: str = ""
    #: その版の案内のページ
    ready_notes_url: str = ""
    #: その版の Python の ABI 導入済みの実行環境と違うかを、入れる前に言うため
    ready_python_abi: str = ""
    #: 次の起動の頭で入れる（本人が「次の起動で」を選んだ・尋ねない設定）
    apply_on_start: bool = False


class UpdateStateStore:
    """:class:`UpdateState` の読み書き"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else update_dir() / "state.json"

    def load(self) -> UpdateState:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return UpdateState()
        if not isinstance(data, dict):
            return UpdateState()
        plain = UpdateState()
        checked = data.get("last_checked")
        skipped = data.get("skipped")
        return UpdateState(
            last_checked=(
                float(checked)
                if isinstance(checked, int | float) and not isinstance(checked, bool)
                else plain.last_checked
            ),
            skipped=(
                tuple(v for v in skipped if isinstance(v, str))
                if isinstance(skipped, list)
                else plain.skipped
            ),
            ready_version=_text(data.get("ready_version")),
            ready_notes_url=_text(data.get("ready_notes_url")),
            ready_python_abi=_text(data.get("ready_python_abi")),
            apply_on_start=data.get("apply_on_start") is True,
        )

    def save(self, state: UpdateState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".writing")
        data = asdict(state)
        data["skipped"] = list(state.skipped)
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
