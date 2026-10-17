"""手元の実物（配布されているエイリアスとテンプレート）を、AI の棚の道具に通す（#328）

対象は 3 つ どれもリポジトリには入れていない（配布物） 無ければ黙って飛ばす

* ``%PROGRAMDATA%\\aviutl2\\Alias`` の AviUtl2 世代のエイリアス（``.object``）
* ``tests/fixtures/aviutl/aviutl1`` の AviUtl1 世代のエイリアス（``.exa``）
* ``tests/fixtures/ymm4`` の YMM4 のアイテムテンプレート（``.ymmt``）

2026-10-17 に数えた結果（PR の本文と #328 にも書いた）

* AviUtl2 の 36 本 置く 36・着せる 36
* AviUtl1 の 67 本 置く 24・効果だけ 43（うち 41 本は ``.anm`` のスクリプトが置き場に無く効果を
  1 つも読めない 着せると理由を返して断る）・着せる 22・文字を持たない 4
* YMM4 の 230 本 置く 186・効果だけ 44・着せる 158・文字を持たない 72
* ``import_objects`` で ``.object`` と ``.exa`` の 103 本を全部読めた

ここで見るのは、AI の道具が実物で落ちない（Python の例外で止まらない）ことと、断るときに
画面と同じ理由を返すこと 何本読めるかは対応表の仕事で、``tests/compat/test_real_*.py`` が見る
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.compat.catalog import TemplateCatalog, TemplateEntry
from sashimono.compat.subtitle_looks import template_choice
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from tests.ai.conftest import FakeHost

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 断ってよい理由 これ以外の ToolError と、ToolError でない例外は落とす
_KNOWN_REFUSALS = (
    "に置けるオブジェクトがありません",
    "読めた効果がありません",
    "文字を持たないテンプレート",
)


def _aviutl2_root() -> Path | None:
    program_data = os.environ.get("PROGRAMDATA")
    root = Path(program_data) / "aviutl2" / "Alias" if program_data else None
    return root if root is not None and root.is_dir() else None


def _shelf(root: Path | None) -> list[TemplateEntry]:
    if root is None or not root.is_dir():
        return []
    return list(TemplateCatalog().scan((root,)))


AVIUTL2 = _shelf(_aviutl2_root())
AVIUTL1 = _shelf(_FIXTURES / "aviutl" / "aviutl1")
YMM4 = _shelf(_FIXTURES / "ymm4")


def _run(host: FakeHost, name: str, **arguments: object) -> object:
    operation = find_operation(name)
    assert operation is not None
    return operation(host, dict(arguments))


def _empty() -> FakeHost:
    return FakeHost(Project.create(ProjectSettings(frame_rate=FrameRate(30))))


def _try(host: FakeHost, name: str, **arguments: object) -> str:
    """道具を動かして結果を 1 語で返す 知らない理由の断りと例外はそのまま投げる"""
    try:
        _run(host, name, **arguments)
    except ToolError as exc:
        reason = next((known for known in _KNOWN_REFUSALS if known in str(exc)), None)
        if reason is None:
            raise
        return reason
    return "できた"


def _place_and_wear(entry: TemplateEntry) -> tuple[str, str]:
    """置いて取り消し、テキストに着せる 置いた物が取り消しで残れば落とす"""
    template_id = template_choice(entry).key
    host = _empty()
    host.shelf = [entry]
    placed = _try(host, "place_template", template_id=template_id, at_frame=0)
    if placed == "できた":
        _run(host, "undo")
        assert not any(track.clips for track in host.document.project.timeline.tracks), entry.name
    host = _empty()
    host.shelf = [entry]
    _run(host, "add_text", text="本人の字幕", at_frame=0, duration=90)
    clip = host.document.project.timeline.tracks[0].clips[0]
    worn = _try(host, "apply_template", template_id=template_id, clip_id=str(clip.id))
    return placed, worn


@pytest.mark.skipif(not AVIUTL2, reason="このマシンに AviUtl2 の配布エイリアスが無い")
def test_every_aviutl2_alias_is_placed_and_worn() -> None:
    # 壊れると、手元の字幕エイリアスを AI に「これで字幕を付けて」と頼んでも置けない・着せられない
    results = {entry.name: _place_and_wear(entry) for entry in AVIUTL2}
    assert all(result == ("できた", "できた") for result in results.values()), results


@pytest.mark.skipif(not AVIUTL1, reason="このマシンに AviUtl1 の配布エイリアスが無い")
def test_aviutl1_aliases_never_stop_the_tool() -> None:
    # 壊れると、効果だけのエイリアスを読んだところで道具ごと落ち、AI は何が悪いのか分からない
    results = [_place_and_wear(entry) for entry in AVIUTL1]
    assert any(placed == "できた" for placed, _ in results)
    assert any(worn == "できた" for _, worn in results)


@pytest.mark.skipif(not YMM4, reason="このマシンに YMM4 のアイテムテンプレートが無い")
def test_ymm4_templates_never_stop_the_tool() -> None:
    # 壊れると、YMM4 のテンプレートのどれかで道具ごと落ち、棚の残りも AI から使えなくなる
    results = [_place_and_wear(entry) for entry in YMM4]
    assert any(placed == "できた" for placed, _ in results)
    # 効果だけのテンプレート（アニメーション効果）はテキストに着せられる
    assert any(
        placed.endswith("置けるオブジェクトがありません") and worn == "できた"
        for placed, worn in results
    )


@pytest.mark.skipif(not (AVIUTL2 or AVIUTL1), reason="このマシンに AviUtl の配布エイリアスが無い")
def test_every_aviutl_file_is_read_by_import_objects() -> None:
    # 壊れると、人は〔オブジェクトを読み込む…〕で読めるファイルを、AI からは読めない
    files = sorted({entry.path for entry in (*AVIUTL2, *AVIUTL1)})
    unread = []
    for path in files:
        host = _empty()
        try:
            _run(host, "import_objects", path=str(path))
        except ToolError as exc:
            unread.append(f"{path.name}: {exc}")
    assert not unread
