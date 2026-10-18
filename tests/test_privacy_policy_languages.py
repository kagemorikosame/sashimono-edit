"""個人情報の扱いの AI アシスタントの行が、日本語と英語でそろっているか

表は docs/code-signing-policy.md にある

AI の道具が外へ送る物を変えたら、同じ PR でこの表を直す決まり（docs/development.md） 日本語だけ
直すと、英語で読む人には送る物の一部が伝わらない
"""

from __future__ import annotations

from pathlib import Path

POLICY = Path(__file__).resolve().parents[1] / "docs" / "code-signing-policy.md"

#: 日本語の行に書いた送る物と、英語の行で同じ物を指す言葉
SAME_THINGS = {
    "テンプレートの棚の名前": "template-shelf entries",
    "テンプレートの棚のフォルダの名前": "folder names of the template shelf",
    "やり直し": "redo",
    "編集の方針": "editing style guide",
}


def _row(starts: str) -> str:
    rows = [
        line for line in POLICY.read_text(encoding="utf-8").splitlines() if line.startswith(starts)
    ]
    assert len(rows) == 1, starts
    return rows[0]


def test_the_english_row_names_everything_the_japanese_row_sends() -> None:
    """壊れると、英語で読む人には、テンプレートの棚の名前やフォルダの名前を送ることが伝わらない"""
    japanese = _row("| AI アシスタント |")
    english = _row("| AI assistant |")
    missing = [
        f"{ja} → {en}" for ja, en in SAME_THINGS.items() if ja in japanese and en not in english
    ]
    assert missing == []
