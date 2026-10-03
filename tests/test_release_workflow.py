"""リリースの workflow（.github/workflows/release.yml）の権限

組み立ては PyPI から依存を入れて走らせる 依存のどれかが悪さをしても、リリースを書き換え
られるトークンに届かないようにする（書ける権限は、依存を入れないジョブだけに渡す）
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def _jobs() -> dict[str, str]:
    """ジョブの名前 → そのジョブの本文（字面で切る 字下げ 2 つのキーがジョブの頭）"""
    body = WORKFLOW.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_nothing_is_granted_by_default() -> None:
    """既定で書く権限を持たせると、権限を書き忘れたジョブがそれを受け継ぐ"""
    assert re.search(r"^permissions: \{\}$", WORKFLOW, flags=re.M)


def test_installing_jobs_cannot_write() -> None:
    jobs = _jobs()
    installing = [name for name, text in jobs.items() if "pip install" in text]
    assert installing
    for name in installing:
        assert "contents: write" not in jobs[name], name


def test_writing_jobs_install_nothing() -> None:
    jobs = _jobs()
    writing = [name for name, text in jobs.items() if "contents: write" in text]
    assert writing
    for name in writing:
        text = jobs[name]
        assert "pip install" not in text and "actions/checkout" not in text, name


def test_checkout_leaves_no_token_behind() -> None:
    """checkout は既定でトークンを .git/config に残し、後で走る依存のコードが読める"""
    for name, text in _jobs().items():
        for step in text.split("- uses: ")[1:]:
            if step.startswith("actions/checkout"):
                assert "persist-credentials: false" in step.split("\n      - ")[0], name
