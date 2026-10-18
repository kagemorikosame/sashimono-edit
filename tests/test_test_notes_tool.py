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


class TestInheritedTests:
    """pytest は Test で始まるクラスが基底から受け継いだ test_* も集める（#377 の Codex）"""

    def test_a_mixin_in_the_same_file_is_followed(self, tool: ModuleType, tmp_path: Path) -> None:
        # 具象クラスの本体だけを見ると、名前が Test で始まらない基底に書き添えの無い試験を
        # 置くだけで見張りをすり抜け、落ちても直すのか消すのか決められない試験が増える
        text = (
            "class Checks:\n    def test_a(self):\n        assert True\n"
            "class TestReal(Checks):\n    pass\n"
        )
        assert names(tool, tmp_path, text) == ["test_a"]

    def test_a_mixin_from_another_module_is_followed(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # 共通の試験を別のファイルの mixin に置く形も pytest は集める ここを見逃すと
        # 置き場を分けただけで書き添えの決まりが効かなくなる
        package = tmp_path / "tests"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "shared.py").write_text(
            "class Checks:\n    def test_a(self):\n        assert True\n", encoding="utf-8"
        )
        absolute = package / "test_absolute.py"
        absolute.write_text(
            "from tests.shared import Checks\nclass TestReal(Checks):\n    pass\n",
            encoding="utf-8",
        )
        relative = package / "test_relative.py"
        relative.write_text(
            "from .shared import Checks as Base\nclass TestReal(Base):\n    pass\n",
            encoding="utf-8",
        )
        reader = tool.Reader(tmp_path)
        for path in (absolute, relative):
            found = tool.scan(path, reader)
            assert [(item.path.name, item.name) for item in found] == [("shared.py", "test_a")]

    def test_a_base_published_by_a_package_is_followed(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # パッケージの __init__.py で読み込み直して公開した基底を辿らないと、置き場を
        # 1 段深くしただけで書き添えの無い試験が見張りをすり抜ける（#377 の CodeRabbit）
        package = tmp_path / "tests" / "shared"
        package.mkdir(parents=True)
        (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("from .checks import Checks\n", encoding="utf-8")
        (package / "checks.py").write_text(
            "class Checks:\n    def test_a(self):\n        assert True\n", encoding="utf-8"
        )
        # 互いを読み込む輪でも止まらずに終わること
        (package / "loop.py").write_text("from .loop import Loop\n", encoding="utf-8")
        path = tmp_path / "tests" / "test_published.py"
        path.write_text(
            "from tests.shared import Checks\nfrom tests.shared.loop import Loop\n"
            "class TestReal(Checks):\n    pass\n"
            "class TestLoop(Loop):\n    pass\n",
            encoding="utf-8",
        )
        found = tool.scan(path, tool.Reader(tmp_path))
        assert [(item.path.name, item.name) for item in found] == [("checks.py", "test_a")]

    def test_an_overridden_test_counts_only_once(self, tool: ModuleType, tmp_path: Path) -> None:
        # 子が同じ名前で書き直した試験は pytest が子の物だけを集める 基底の物まで求めると、
        # 走らない試験に書き添えを強いて verify が落ちる
        text = (
            "class Checks:\n    def test_a(self):\n        assert True\n"
            "class TestReal(Checks):\n    def test_a(self):\n        # 壊れると作品が開けない\n"
            "        assert True\n"
        )
        assert names(tool, tmp_path, text) == []

    def test_a_shared_base_is_listed_once(self, tool: ModuleType, tmp_path: Path) -> None:
        # 何本ものクラスが受け継ぐ基底の試験を数だけ並べると、抜けの本数が水増しされる
        text = (
            "class TestBase:\n    def test_a(self):\n        assert True\n"
            "class TestOne(TestBase):\n    pass\n"
            "class TestTwo(TestBase):\n    pass\n"
        )
        assert names(tool, tmp_path, text) == ["test_a"]

    def test_a_base_outside_the_repository_is_skipped(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # unittest.TestCase などリポジトリの外の基底は書き添えを求める先が無い 辿ろうとして
        # 落ちると、外の基底を使う試験を足しただけで verify が止まる
        text = (
            "import unittest\nfrom unittest import TestCase\n"
            "class TestA(TestCase):\n    pass\n"
            "class TestB(unittest.TestCase):\n    pass\n"
        )
        assert names(tool, tmp_path, text) == []


#: 別のファイルに置いた試験の材料 書き添えはどれにも無い
SHARED_CHECKS = (
    "class Checks:\n    def test_a(self):\n        assert True\n"
    "def test_b():\n    assert True\n"
    "class TestImported:\n    def test_c(self):\n        assert True\n"
    "def helper(self):\n    assert True\n"
)


class TestEveryWayPytestCollects:
    """pytest が試験のファイルの名前の表から集める道は、どれを通っても見張る

    #377 で Codex と CodeRabbit が基底の辿り方の穴を 3 度続けて指摘した 1 つずつ塞ぐのを
    やめ、名前の解き方をまとめて作り直したので、知っている道を全部ここに並べる どれかが
    抜けると、その書き方に変えるだけで書き添えの無い試験が verify を通る
    """

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                "from tests.shared import Checks\nclass TestReal(Checks):\n    pass\n",
                [("checks.py", "test_a")],
            ),
            (
                "from tests import shared\nclass TestReal(shared.Checks):\n    pass\n",
                [("checks.py", "test_a")],
            ),
            (
                "import tests.shared\nclass TestReal(tests.shared.Checks):\n    pass\n",
                [("checks.py", "test_a")],
            ),
            (
                "import tests.shared.checks as c\nclass TestReal(c.Checks):\n    pass\n",
                [("checks.py", "test_a")],
            ),
            (
                "from tests.shared.checks import Checks\nBase = Checks\n"
                "class TestReal(Base):\n    pass\n",
                [("checks.py", "test_a")],
            ),
            (
                "if True:\n    from tests.shared.checks import Checks\n"
                "class TestReal(Checks):\n    pass\n",
                [("checks.py", "test_a")],
            ),
            ("from tests.shared.checks import test_b\n", [("checks.py", "test_b")]),
            ("from tests.shared.checks import TestImported\n", [("checks.py", "test_c")]),
            (
                "from tests.shared.checks import *\n",
                [("checks.py", "test_b"), ("checks.py", "test_c")],
            ),
            (
                "from tests.shared.checks import helper\nclass TestReal:\n    test_d = helper\n",
                [("checks.py", "helper")],
            ),
            (
                "class Outer:\n    class TestInner:\n        def test_e(self):\n"
                "            assert True\n"
                "class TestReal(Outer):\n    pass\n",
                [("test_real.py", "test_e")],
            ),
            # ここから下はクラスの本体 モジュールの直下と同じ形をすべて同じ手順で読む
            (
                "class TestReal:\n    if True:\n        def test_f(self):\n"
                "            assert True\n",
                [("test_real.py", "test_f")],
            ),
            (
                "class TestReal:\n    def _check(self):\n        assert True\n"
                "    test_g = _check\n",
                [("test_real.py", "_check")],
            ),
            (
                "class TestReal:\n    try:\n        def test_h(self):\n            assert True\n"
                "    except Exception:\n        pass\n",
                [("test_real.py", "test_h")],
            ),
            (
                "import contextlib\nclass TestReal:\n    with contextlib.suppress(Exception):\n"
                "        def test_w(self):\n            assert True\n",
                [("test_real.py", "test_w")],
            ),
            (
                "class TestReal:\n    from tests.shared.checks import test_b\n",
                [("checks.py", "test_b")],
            ),
            (
                "class TestReal:\n    if True:\n        class TestInner:\n"
                "            def test_i(self):\n                assert True\n",
                [("test_real.py", "test_i")],
            ),
            (
                "class TestReal:\n    class Helpers:\n        def check(self):\n"
                "            assert True\n    test_j = Helpers.check\n",
                [("test_real.py", "check")],
            ),
            (
                "from tests.shared.checks import Checks\nclass TestReal(Checks):\n"
                "    if True:\n        def test_a(self):\n"
                "            # 壊れると作品が開けない\n            assert True\n",
                # if が走らなければ基底の試験が集まるので、基底の物も数える
                [("checks.py", "test_a")],
            ),
            (
                "class TestReal:\n    class Base:\n        def test_k(self):\n"
                "            assert True\n    class TestInner(Base):\n        pass\n",
                [("test_real.py", "test_k")],
            ),
            (
                "def test_l():\n    assert True\nclass TestReal:\n    test_l = test_l\n",
                [("test_real.py", "test_l")],
            ),
            (
                "class TestReal:\n    def _m(self):\n        assert True\n"
                "    test_m: object = _m\n",
                [("test_real.py", "_m")],
            ),
        ],
        ids=[
            "star-reexport",
            "from-package-attribute",
            "import-attribute",
            "import-as-attribute",
            "renamed-base",
            "base-in-block",
            "imported-function",
            "imported-class",
            "star-into-test-file",
            "class-attribute-function",
            "inherited-nested-class",
            "class-def-in-if",
            "class-local-alias",
            "class-def-in-try",
            "class-def-in-with",
            "class-import",
            "class-nested-in-if",
            "class-attribute-of-local-class",
            "class-override-in-if",
            "nested-base-in-enclosing-class",
            "class-self-named-alias",
            "class-annotated-alias",
        ],
    )
    def test_each_way_is_followed(
        self, tool: ModuleType, tmp_path: Path, source: str, expected: list[tuple[str, str]]
    ) -> None:
        # どの道でも、試験が書いてあるファイルと名前で抜けを出す 出なければ、その書き方で
        # 書き添えの無い試験が見張りをすり抜ける
        package = tmp_path / "tests" / "shared"
        package.mkdir(parents=True)
        (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("from .checks import *\n", encoding="utf-8")
        (package / "checks.py").write_text(SHARED_CHECKS, encoding="utf-8")
        path = tmp_path / "tests" / "test_real.py"
        path.write_text(source, encoding="utf-8")
        found = tool.scan(path, tool.Reader(tmp_path))
        assert sorted((item.path.name, item.name) for item in found) == expected

    @pytest.mark.parametrize(
        "source",
        [
            "from tests.shared.checks import Checks\nclass Other:\n    pass\n"
            "if flag:\n    Base = Checks\nelse:\n    Base = Other\n"
            "class TestReal(Base):\n    pass\n",
            "try:\n    from tests.shared.checks import Checks as Base\n"
            "except ImportError:\n    Base = object\n"
            "class TestReal(Base):\n    pass\n",
        ],
        ids=["if-else-base", "try-except-base"],
    )
    def test_a_base_chosen_in_a_block_counts_every_candidate(
        self, tool: ModuleType, tmp_path: Path, source: str
    ) -> None:
        # どちらの枝が走るかは読むだけでは決まらない 1 つに決めて読むと、決めなかった
        # 枝の基底の試験を見逃す（#377 の CodeRabbit が読み解く側に倒す所を指摘した）
        package = tmp_path / "tests" / "shared"
        package.mkdir(parents=True)
        (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "checks.py").write_text(SHARED_CHECKS, encoding="utf-8")
        path = tmp_path / "tests" / "test_real.py"
        path.write_text(source, encoding="utf-8")
        found = tool.scan(path, tool.Reader(tmp_path))
        assert [(item.path.name, item.name) for item in found] == [("checks.py", "test_a")]

    def test_a_test_defined_in_either_branch_is_counted(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # 後ろの枝だけを読むと、前の枝の書き添えの無い試験が走る場合を見逃す
        source = (
            "if flag:\n    def test_a():\n        assert True\n"
            "else:\n    def test_a():\n        # 壊れると作品が開けない\n        assert True\n"
        )
        assert names(tool, tmp_path, source) == ["test_a"]

    @pytest.mark.parametrize(
        "source",
        [
            "import unittest\nclass Checks(unittest.TestCase):\n    def test_a(self):\n"
            "        assert True\n",
            "from unittest import TestCase\nclass Mixin(TestCase):\n    pass\n"
            "class Checks(Mixin):\n    def test_a(self):\n        assert True\n",
        ],
        ids=["module-attribute", "through-a-mixin"],
    )
    def test_a_unittest_class_is_collected_by_any_name(
        self, tool: ModuleType, tmp_path: Path, source: str
    ) -> None:
        # pytest は unittest.TestCase を受け継ぐクラスを名前に関係なく集める 名前だけで
        # 選ぶと、Test で始まらない名前を付けるだけで見張りをすり抜ける
        assert names(tool, tmp_path, source) == ["test_a"]

    def test_files_named_with_a_trailing_test_are_checked(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # pytest は *_test.py も集める test_*.py だけを見ると、名前の付け方を変えた
        # ファイルの試験が見張りの外に出る
        (tmp_path / "tests").mkdir()
        for name in ("test_a.py", "b_test.py", "helper.py"):
            (tmp_path / "tests" / name).write_text("", encoding="utf-8")
        found = [path.name for path in tool.targets(tmp_path, everything=True)]
        assert found == ["b_test.py", "test_a.py"]


class TestWhatPytestSkips:
    """pytest が集めない物には書き添えを求めない 求めると、試験の部品を外した人の verify が落ちる"""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                "class TestHelper:\n    __test__ = False\n    def test_a(self):\n"
                "        assert True\n",
                [],
            ),
            (
                "class Base:\n    __test__ = False\n"
                "class TestOff(Base):\n    def test_a(self):\n        assert True\n"
                "class TestOn(Base):\n    __test__ = True\n    def test_b(self):\n"
                "        assert True\n",
                ["test_b"],
            ),
            ("def test_a():\n    assert True\ntest_a.__test__ = False\n", []),
            (
                "class TestX:\n    def test_a(self):\n        assert True\n"
                "TestX.__test__ = False\n",
                [],
            ),
            ("__test__ = False\ndef test_a():\n    assert True\n", []),
            # ここから下は書かれ方が 1 通りに決まらない __test__ どれも数える側に倒す
            (
                "class TestX:\n    if TYPE_CHECKING:\n        __test__ = False\n"
                "    def test_a(self):\n        assert True\n",
                ["test_a"],
            ),
            ("if flag:\n    __test__ = False\ndef test_a():\n    assert True\n", ["test_a"]),
            (
                "def test_a():\n    assert True\nif flag:\n    test_a.__test__ = False\n",
                ["test_a"],
            ),
            (
                "class TestX:\n    __test__ = False\n    __test__ = True\n"
                "    def test_a(self):\n        assert True\n",
                ["test_a"],
            ),
            (
                "class TestX:\n    def test_a(self):\n        assert True\n"
                "setattr(TestX, '__test__', False)\n",
                ["test_a"],
            ),
            (
                "class TestX:\n    __test__ = flag()\n    def test_a(self):\n        assert True\n",
                ["test_a"],
            ),
        ],
        ids=[
            "class-off",
            "inherited-off-and-back-on",
            "function-off",
            "class-off-from-outside",
            "module-off",
            "class-flag-in-if",
            "module-flag-in-if",
            "function-flag-in-if",
            "class-flag-twice",
            "setattr",
            "unknown-value-counts",
        ],
    )
    def test_switched_off_tests_are_not_asked_for(
        self, tool: ModuleType, tmp_path: Path, source: str, expected: list[str]
    ) -> None:
        # 値が分からない物（呼び出しで決める __test__）は数える側に倒す 外す側に倒すと、
        # 書き添えの無い試験を見逃す
        assert names(tool, tmp_path, source) == expected

    @pytest.mark.parametrize(
        ("exports", "expected"),
        [
            ('__all__ = ["test_b"]\n', ["test_b"]),
            ('__all__ = ("test_b", "test_z")\n', ["test_b", "test_z"]),
            ("__all__ = names()\n", ["test_b", "test_z"]),
            ("", ["test_b", "test_z"]),
            # ここから下は書き換えた __all__ どれも読み解かず全部数える
            ('__all__ = ["test_b"]\n__all__ += ["test_z"]\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b"]\n__all__.extend(["test_z"])\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b"]\n__all__.append("test_z")\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b"]\n__all__.insert(0, "test_z")\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b", "x"]\n__all__[1] = "test_z"\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b"]\ngrow(__all__)\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b"]\nglobals()["__all__"] = ["test_z"]\n', ["test_b", "test_z"]),
            ('__all__ = ["test_b"]\n__all__ = ["test_b", "test_z"]\n', ["test_b", "test_z"]),
            (
                'def grow():\n    global __all__\n    __all__ = ["test_z"]\n__all__ = ["test_b"]\n',
                ["test_b", "test_z"],
            ),
            # ブロックの中の __all__ は走るかどうかが決まらない どれも読み解かず全部数える
            ('if TYPE_CHECKING:\n    __all__ = ["test_b"]\n', ["test_b", "test_z"]),
            ('for _ in []:\n    __all__ = ["test_b"]\n', ["test_b", "test_z"]),
            ('while False:\n    __all__ = ["test_b"]\n', ["test_b", "test_z"]),
            (
                'try:\n    __all__ = ["test_b"]\nexcept Exception:\n    pass\n',
                ["test_b", "test_z"],
            ),
            ('with suppress():\n    __all__ = ["test_b"]\n', ["test_b", "test_z"]),
        ],
        ids=[
            "list",
            "tuple",
            "expression-counts-all",
            "no-all",
            "augmented",
            "extend",
            "append",
            "insert",
            "item-assignment",
            "passed-to-a-call",
            "through-globals",
            "assigned-twice",
            "inside-a-function",
            "in-if",
            "in-for",
            "in-while",
            "in-try",
            "in-with",
        ],
    )
    def test_a_star_import_follows_all(
        self, tool: ModuleType, tmp_path: Path, exports: str, expected: list[str]
    ) -> None:
        # __all__ で外した名前まで数えると、pytest が集めない関数に書き添えを強いる
        # 式で作った __all__ は読み解かず、全部数える側に倒す
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "tests" / "shared.py").write_text(
            exports + "def test_b():\n    assert True\ndef test_z():\n    assert True\n",
            encoding="utf-8",
        )
        path = tmp_path / "tests" / "test_real.py"
        path.write_text("from tests.shared import *\n", encoding="utf-8")
        found = tool.scan(path, tool.Reader(tmp_path))
        assert sorted(item.name for item in found) == expected


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
