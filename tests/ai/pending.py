"""裏で待つ道具の答え（:class:`~sashimono.ai.operations.Pending`）を辿る試験の部品"""

from __future__ import annotations

from typing import Any

from sashimono.ai.operations import Pending, finish


def settle(result: object) -> Any:
    """ブリッジと同じく待ってから辿る

    辿らないと、待った後の ``then`` で積むコマンド（ナレーション #335 の声を置く・表記を
    足す、音量の見積もりの後に掛ける #497 など）を数えず、何も変えていないように見える
    """
    while isinstance(result, Pending):
        result.wait(result.seconds)
        result = finish(result.then())
    return result


def run_now(name: str, result: object, arguments: dict[str, Any]) -> Any:
    """測るのを待たない音量の道具（``wait_seconds=0``）の、裏の見積もり（#497）だけを辿る

    ほかの道具と測るのを待つ呼び方は、``Pending`` のまま返す（待つ形そのものを確かめる試験がある）
    """
    if name == "normalize_loudness" and arguments.get("wait_seconds") == 0:
        return settle(result)
    return result
