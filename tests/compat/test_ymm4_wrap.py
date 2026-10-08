"""YMM4 のテキストの MaxWidth と WordWrap を折り返しの幅として読む（#249）

手元の実物（配布テンプレートと探りの .ymmt 114 本・テキスト 310 個 2026-10-07 に数えた）は
どれも ``WordWrap`` が ``NoWrap``・``MaxWidth`` が 1920 だった 折り返す見本はまだ無いので、
折り返す値を読んだときは互換性レポートに残して数える
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import TemplateCatalog
from sashimono.compat.ymm4.template import map_template
from sashimono.core.model import AnimatedValue
from tests.compat.test_ymm4 import still, text_item

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ymm4"


def mapped(**fields: Any) -> tuple[dict[str, Any], CompatibilityReport]:
    report = CompatibilityReport()
    source = map_template([text_item(**fields)], report=report)[0].clip.source
    assert source is not None
    return dict(source.params), report


def test_no_wrap_keeps_one_line() -> None:
    # 実物はどれも NoWrap MaxWidth 1920 を幅として入れると、1920 を超える行が勝手に折れる
    params, report = mapped(WordWrap="NoWrap", MaxWidth=still(1920.0))
    width = params.get("wrap_width")
    assert width is None or (isinstance(width, AnimatedValue) and width.static == 0)
    assert not any("折り返し" in name for name in report.missing)


def test_a_wrap_reads_the_max_width() -> None:
    # 折り返す指定は MaxWidth の幅で折り返す 読まないと 1 行のまま画面からはみ出す
    params, report = mapped(WordWrap="Wrap", MaxWidth=still(800.0))
    width = params["wrap_width"]
    assert isinstance(width, AnimatedValue) and width.static == 800
    # 位置の決まりは実物と描き比べていない 数えられるよう残す
    assert any("折り返し" in name for name in report.missing)


@pytest.mark.skipif(not ROOT.is_dir(), reason="tests/fixtures/ymm4 に配布物が置かれていない")
def test_the_real_templates_do_not_wrap() -> None:
    # 実物を通して、折り返しの幅が入るテキストが無いこと（どれも NoWrap）
    shelf = TemplateCatalog()
    shelf.scan((ROOT,))
    report = CompatibilityReport()
    texts = [
        item.clip.source
        for entry in shelf.all()
        for item in entry.load(report=report)
        if item.clip.source is not None and item.clip.source.kind == "text"
    ]
    if not texts:
        pytest.skip("テキストを持つテンプレートが無い")
    for source in texts:
        width = source.params.get("wrap_width")
        assert width is None or (isinstance(width, AnimatedValue) and width.static == 0)
