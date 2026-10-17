"""試験の書き添えを見張る道具（tools/test_notes.py #363）

この道具が抜けを見逃すと、壊れたときに何が起きるかを書かない試験がまた増え、
落ちた試験を直すのか消すのかを次の人が決められなくなる 逆に書いてある試験を
抜けと言うと、verify が通らず誰も試験を足せなくなる その境目をここで固定する
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_notes", ROOT / "tools" / "test_notes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass は定義の途中で自分のモジュールを sys.modules から引く
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def names(tool: ModuleType, tmp_path: Path, text: str) -> list[str]:
    path = tmp_path / "test_sample.py"
    path.write_text(text, encoding="utf-8")
    return [item.name for item in tool.scan(path)]


@pytest.mark.parametrize(
    "text",
    [
        'def test_a():\n    """壊れると保存した作品が開けない"""\n    assert True\n',
        "# 壊れると保存した作品が開けない\ndef test_a():\n    assert True\n",
        "# 壊れると保存した作品が開けない\n@pytest.mark.slow\ndef test_a():\n    assert True\n",
        "def test_a():\n    # 壊れると保存した作品が開けない\n    assert True\n",
        "def test_a(\n    x: int,\n) -> None:\n    # 壊れると作品が開けない\n    assert x\n",
        "class TestA:\n    def test_a(self):\n        # 壊れると作品が開けない\n        pass\n",
    ],
    ids=["docstring", "above", "above-decorator", "body-head", "folded-arguments", "in-class"],
)
def test_the_three_places_count_as_written(tool: ModuleType, tmp_path: Path, text: str) -> None:
    # 決めた 3 か所のどれかを抜けと言うと、決まりどおりに書いた人の verify が落ちる
    assert names(tool, tmp_path, text) == []


@pytest.mark.parametrize(
    ("text", "name"),
    [
        (
            "def test_the_saved_project_opens_again():\n    assert True\n",
            "test_the_saved_project_opens_again",
        ),
        ("# 前の段落のコメント\n\ndef test_a():\n    assert True\n", "test_a"),
        ("x = 1  # 行の後ろのコメント\ndef test_a():\n    assert True\n", "test_a"),
        ("def test_a():\n    x = 1\n    # 途中のコメント\n    assert x\n", "test_a"),
        (
            'class TestA:\n    """クラスの説明"""\n\n    def test_a(self):\n        assert True\n',
            "test_a",
        ),
        ("async def test_a():\n    assert True\n", "test_a"),
    ],
    ids=["name-only", "blank-line", "trailing", "middle", "class-docstring", "async"],
)
def test_a_missing_note_is_reported(tool: ModuleType, tmp_path: Path, text: str, name: str) -> None:
    # 見逃すと、書き添えの無い試験が verify を通って main に入る 名前だけで分かる試験を
    # 認めないのは、名前が利用者の困りごとを言っているかを機械では見分けられないため
    assert names(tool, tmp_path, text) == [name]


def test_helpers_and_other_classes_are_not_tests(tool: ModuleType, tmp_path: Path) -> None:
    # pytest が集めない関数まで求めると、試験の部品にまで書き添えを強いて verify が落ちる
    text = (
        "def helper():\n    pass\n"
        "class Helper:\n    def test_like(self):\n        pass\n"
        "def test_a():\n    # 壊れると作品が開けない\n    def test_inner():\n        pass\n"
    )
    assert names(tool, tmp_path, text) == []


def test_only_the_watched_places_are_checked(tool: ModuleType) -> None:
    # 書き添えを埋めた置き場から順に見張る まだ埋めていない置き場まで見ると verify が
    # 落ち続け、--all を付けなければ数えられないと誰も抜けの数を追えない
    watched = tool.targets()
    everything = tool.targets(everything=True)
    assert watched
    assert set(watched) < set(everything)
    for path in watched:
        assert path.relative_to(ROOT).as_posix().startswith(tool.WATCHED_PREFIXES)
    assert not any("fixtures" in path.parts for path in everything)


def test_verify_runs_the_check() -> None:
    # verify の段から外れると、CI でも手元でも抜けが見張られなくなる
    text = (ROOT / "tools" / "verify.py").read_text(encoding="utf-8")
    assert '["tools/test_notes.py"]' in text
