"""PyInstaller の起動部が既成の物と違うことを確かめる道具（tools/check_bootloader.py）

既成の起動部のまま配ると、Windows Defender の機械学習の推測に exe ごと消される（0.1.1）
比べ方が甘いと、組み直し損ねたまま「違う」と通ってしまう
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_bootloader", ROOT / "tools" / "check_bootloader.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rebuilt_bootloaders_pass(tool: ModuleType) -> None:
    """全部のハッシュが既成の物と違えば通る"""
    assert tool.compare({"run.exe": "a", "runw.exe": "b"}, {"run.exe": "x", "runw.exe": "y"}) == []


def test_a_stock_bootloader_is_caught(tool: ModuleType) -> None:
    """1 つでも既成の物と同じなら止める 配る exe がどれを使うかに頼らない"""
    problems = tool.compare({"run.exe": "a", "runw.exe": "y"}, {"run.exe": "x", "runw.exe": "y"})
    assert len(problems) == 1 and "runw.exe" in problems[0]


def test_missing_or_unmatched_files_are_not_counted_as_different(tool: ModuleType) -> None:
    """片方にしか無い物を「違う」と数えると、置き場や名前が変わった版で比べ漏れて通る"""
    assert tool.compare({}, {"run.exe": "x"})
    assert tool.compare({"run.exe": "a"}, {"run.exe": "x", "runw.exe": "y"})
    assert tool.compare({"run.exe": "a", "new.exe": "b"}, {"run.exe": "x"})


def test_only_the_windows_bootloaders_are_read_from_the_wheel(
    tool: ModuleType, tmp_path: Path
) -> None:
    """wheel の中のほかの OS の起動部を拾うと、配る物と違う物を比べてしまう"""
    wheel = tmp_path / "pyinstaller-6.0-py3-none-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("PyInstaller/bootloader/Windows-64bit-intel/run.exe", b"stock")
        archive.writestr("PyInstaller/bootloader/Windows-64bit-intel/README", b"x")
        archive.writestr("PyInstaller/bootloader/Windows-32bit-intel/run.exe", b"other")
    built = tmp_path / "Windows-64bit-intel"
    built.mkdir()
    (built / "run.exe").write_bytes(b"stock")
    stock = tool.digests_in_wheel(wheel)
    assert list(stock) == ["run.exe"]
    assert tool.compare(tool.digests_in_folder(built), stock)
