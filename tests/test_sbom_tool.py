"""配る zip の部品の一覧（SBOM）を作る道具（tools/sbom.py #517）

zip は試験の中で作る 使用許諾の一覧はリポジトリの THIRD_PARTY_NOTICES.md をそのまま入れ、
DLL と写しのフォルダは一覧の表の版で置く 確かめるのは、足し忘れ・版違い・決まっていない DLL で
一覧を書かずに止まること・前の一覧を残さないこと・同じ zip から同じ一覧ができること・
作り直した物と違う一覧を見分けること・2 つの一覧の違いを出すこと
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from tests.fake_python_dll import minimum_python, python_dll_bytes

ROOT = Path(__file__).resolve().parent.parent
NOTICES = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sbom", ROOT / "tools" / "sbom.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _listed(tool: ModuleType) -> dict[str, str]:
    from package_notices import listed_versions

    found: dict[str, str] = listed_versions(NOTICES)
    return found


def _zip(
    tool: ModuleType,
    *,
    extra: dict[str, bytes] | None = None,
    drop: tuple[str, ...] = (),
    numpy: str | None = None,
) -> bytes:
    """配る zip の形 Python の包みは numpy と av、DLL は FFmpeg の 2 つと x264 を積む"""
    listed = _listed(tool)
    numpy_version = numpy or listed["numpy"]
    files: dict[str, bytes] = {
        "Sashimono/Sashimono.exe": b"MZ",
        "Sashimono/build-info.json": json.dumps(
            {"version": "1.2.3", "python_abi": "cp314"}
        ).encode(),
        "Sashimono/THIRD_PARTY_NOTICES.txt": NOTICES.encode("utf-8"),
        "Sashimono/_internal/python314.dll": python_dll_bytes(minimum_python()),
        f"Sashimono/licenses/numpy-{numpy_version}/LICENSE.txt": b"numpy license",
        f"Sashimono/licenses/av-{listed['av']}/LICENSE.txt": b"av license",
        "Sashimono/licenses/ffmpeg-8.1.2/LICENSE.md": b"ffmpeg license",
        "Sashimono/licenses/x264-b35605ac/COPYING": b"x264 license",
        "Sashimono/licenses/GPL-2.0.txt": b"gpl",
        "Sashimono/_internal/av.libs/avcodec-62-" + "0" * 32 + ".dll": b"avcodec",
        "Sashimono/_internal/av.libs/avutil-60-" + "1" * 32 + ".dll": b"avutil",
        "Sashimono/_internal/av.libs/libx264-165-" + "2" * 32 + ".dll": b"x264",
        "Sashimono/_internal/OpenGL/__init__.pyc": b"pyc",
        **(extra or {}),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            if name not in drop:
                archive.writestr(name, data)
    return buffer.getvalue()


def _write(tmp_path: Path, data: bytes, name: str = "SashimonoEdit-1.2.3-windows-x64.zip") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _components(bom: dict[str, object]) -> dict[str, dict[str, object]]:
    items = bom["components"]
    assert isinstance(items, list)
    return {str(item["name"]): item for item in items}


class TestGenerate:
    def test_the_bundled_parts_are_listed_with_hashes(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """版や SHA-256 が欠けると、一覧で脆弱性も差し替えも見分けられない"""
        archive = _write(tmp_path, _zip(tool))
        bom = tool.generate(archive)
        data = bom.data
        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.6"
        assert tool.zip_digest(data) == hashlib.sha256(archive.read_bytes()).hexdigest()
        found = _components(data)
        listed = _listed(tool)
        assert found["numpy"]["version"] == listed["numpy"]
        assert found["numpy"]["purl"] == f"pkg:pypi/numpy@{listed['numpy']}"
        # 写しを持たない包みは、zip の中の置き場で積んだと見る
        assert found["PyOpenGL"]["purl"] == f"pkg:pypi/pyopengl@{listed['pyopengl']}"
        ffmpeg = found["FFmpeg"]
        assert ffmpeg["version"] == "8.1.2"
        assert ffmpeg["cpe"] == "cpe:2.3:a:ffmpeg:ffmpeg:8.1.2:*:*:*:*:*:*:*"
        inner = ffmpeg["components"]
        assert isinstance(inner, list)
        assert {item["hashes"][0]["content"] for item in inner} == {
            hashlib.sha256(b"avcodec").hexdigest(),
            hashlib.sha256(b"avutil").hexdigest(),
        }
        assert found["x264"]["version"] == "b35605ace3ddf7c1a5d67a2eb553f034aef41d55"
        # 一覧の表は 3.14 までしか書かない 脆弱性は修正版で決まるので DLL から読む
        python = next(name for name in found if name.startswith("Python"))
        assert found[python]["version"] == ".".join(str(part) for part in minimum_python())

    def test_listed_but_not_bundled_parts_are_left_out(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """一覧の表は積む候補を広めに書いてある 積んでいない物を一覧に載せない"""
        bom = tool.generate(_write(tmp_path, _zip(tool)))
        found = _components(bom.data)
        assert "PySide6_Essentials" not in found
        assert "x265" not in found
        assert "PySide6_Essentials" in bom.left_out
        assert "x265" in bom.left_out

    def test_the_same_zip_gives_the_same_list(self, tool: ModuleType, tmp_path: Path) -> None:
        """下書きの一覧を作り直して照らすので、時刻や乱数を入れない"""
        archive = _write(tmp_path, _zip(tool))
        assert tool.generate(archive).text() == tool.generate(archive).text()

    def test_a_dll_without_a_part_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """どの部品の物か決まっていない DLL を黙って外すと、一覧への足し忘れになる"""
        extra = {"Sashimono/_internal/av.libs/libnew-3-" + "3" * 32 + ".dll": b"new"}
        with pytest.raises(tool.BomError, match="libnew"):
            tool.generate(_write(tmp_path, _zip(tool, extra=extra)))

    def test_a_package_of_another_version_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """版違いを通すと、配った物と違う版の一覧を配る"""
        with pytest.raises(tool.BomError, match=r"numpy-0.0.1"):
            tool.generate(_write(tmp_path, _zip(tool, numpy="0.0.1")))

    def test_an_unlisted_package_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """表に無い包みを黙って外すと、一覧への足し忘れになる"""
        extra = {"Sashimono/licenses/leftpad-1.0/LICENSE": b"leftpad"}
        with pytest.raises(tool.BomError, match=r"leftpad-1.0"):
            tool.generate(_write(tmp_path, _zip(tool, extra=extra)))

    def test_a_dll_without_its_copy_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """写しの無い DLL を通すと、使用許諾をそろえずに配る"""
        drop = ("Sashimono/licenses/x264-b35605ac/COPYING",)
        with pytest.raises(tool.BomError, match="x264-b35605ac"):
            tool.generate(_write(tmp_path, _zip(tool, drop=drop)))

    def test_a_zip_without_the_notices_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """表が読めないまま書くと、空の一覧を配る"""
        drop = ("Sashimono/THIRD_PARTY_NOTICES.txt",)
        with pytest.raises(tool.BomError, match="THIRD_PARTY_NOTICES"):
            tool.generate(_write(tmp_path, _zip(tool, drop=drop)))

    def test_a_license_text_that_is_not_an_spdx_expression_is_a_name(
        self, tool: ModuleType
    ) -> None:
        """文章を SPDX の式として書くと、読み手が使用許諾ごと捨てる"""
        assert tool.license_entry("GPL-2.0-or-later") == [{"expression": "GPL-2.0-or-later"}]
        assert tool.license_entry("MIT（Lua の使用許諾を含む）") == [{"expression": "MIT"}]
        # 版の無い略し方は SPDX の識別子ではない
        assert tool.license_entry("LGPL（COPYING は GNU Library GPL 2）") == [
            {"license": {"name": "LGPL（COPYING は GNU Library GPL 2）"}}
        ]


class TestWrite:
    def test_a_failure_leaves_no_list_behind(self, tool: ModuleType, tmp_path: Path) -> None:
        """前の版の一覧が今の zip の隣に残ると、一緒に上げられて誤った一覧を配る"""
        extra = {"Sashimono/_internal/av.libs/libnew-3-" + "3" * 32 + ".dll": b"new"}
        archive = _write(tmp_path, _zip(tool, extra=extra))
        out = tmp_path / tool.BOM_NAME
        out.write_text("{}", encoding="utf-8")
        assert tool.main(["write", "--archive", str(archive), "--out", str(out)]) == 1
        assert not out.exists()

    def test_it_writes_beside_the_zip(self, tool: ModuleType, tmp_path: Path) -> None:
        """書いた一覧が作り直した物と違うと、release.py の照らしで必ず止まる"""
        archive = _write(tmp_path, _zip(tool))
        assert tool.main(["write", "--archive", str(archive)]) == 0
        out = tmp_path / tool.BOM_NAME
        assert json.loads(out.read_text(encoding="utf-8")) == tool.generate(archive).data
        assert tool.main(["check", str(archive), str(out)]) == 0


class TestCompare:
    def test_a_list_for_another_zip_is_caught(self, tool: ModuleType, tmp_path: Path) -> None:
        """組み直した zip の隣に前の zip の一覧が上がっていたら、照らして見つける"""
        old = _write(tmp_path, _zip(tool), "old.zip")
        bom = tool.generate(old).data
        extra = {"Sashimono/_internal/av.libs/avutil-60-" + "1" * 32 + ".dll": b"rebuilt"}
        new = _write(tmp_path, _zip(tool, extra=extra), "new.zip")
        found = tool.mismatches(new, bom)
        assert any("SHA-256" in line for line in found)
        assert any("版が同じで中身が違う: FFmpeg" in line for line in found)
        assert tool.mismatches(old, bom) == []

    def test_differences_name_what_changed(self, tool: ModuleType, tmp_path: Path) -> None:
        """違いを出せないと、前の版から部品が出入りしても公開の前に気付けない"""
        before = tool.generate(_write(tmp_path, _zip(tool), "before.zip")).data
        after = json.loads(json.dumps(before))
        after["components"] = [c for c in after["components"] if c["name"] != "av"]
        for component in after["components"]:
            if component["name"] == "numpy":
                component["version"] = "9.9.9"
        after["components"].append({"type": "library", "name": "新しい部品", "version": "1.0"})
        lines = tool.differences(before, after)
        assert "足した: 新しい部品 1.0" in lines
        assert any(line.startswith("外した: av ") for line in lines)
        assert any(line.startswith("版が変わった: numpy ") for line in lines)
        assert tool.differences(before, before) == []
