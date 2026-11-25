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


#: 試験の zip のファイルと、組み立ての記録に書く出どころ
OWNERS = {
    "Sashimono.exe": "build",
    "_internal/python314.dll": "python",
    "_internal/av.libs/avcodec-62-" + "0" * 32 + ".dll": "av",
    "_internal/av.libs/avutil-60-" + "1" * 32 + ".dll": "av",
    "_internal/av.libs/libx264-165-" + "2" * 32 + ".dll": "av",
    "_internal/OpenGL/__init__.pyc": "pyopengl",
}


def _zip(
    tool: ModuleType,
    *,
    extra: dict[str, bytes] | None = None,
    owners: dict[str, str] | None = None,
    packages: dict[str, tuple[str, str]] | None = None,
    drop: tuple[str, ...] = (),
    numpy: str | None = None,
    record: bool = True,
) -> bytes:
    """配る zip の形 Python の包みは numpy と av と PyOpenGL、DLL は FFmpeg の 2 つと x264 を積む

    ``owners`` は足したファイルの出どころ 書かなければ記録に載らない（記録に無いファイルを試す）
    """
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
    bundled = {
        "numpy": {"name": "numpy", "version": numpy_version},
        "av": {"name": "av", "version": listed["av"]},
        "pyopengl": {"name": "PyOpenGL", "version": listed["pyopengl"]},
        **{
            key: {"name": name, "version": version}
            for key, (name, version) in (packages or {}).items()
        },
    }
    if record:
        files["Sashimono/bundled-files.json"] = json.dumps(
            {"packages": bundled, "files": {**OWNERS, **(owners or {})}}
        ).encode()
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
        name = "_internal/av.libs/libnew-3-" + "3" * 32 + ".dll"
        archive = _zip(tool, extra={f"Sashimono/{name}": b"new"}, owners={name: "av"})
        with pytest.raises(tool.BomError, match="libnew"):
            tool.generate(_write(tmp_path, archive))

    def test_a_package_of_another_version_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """版違いを通すと、配った物と違う版の一覧を配る"""
        with pytest.raises(tool.BomError, match=r"numpy は 0\.0\.1 を積んでいる"):
            tool.generate(_write(tmp_path, _zip(tool, numpy="0.0.1")))

    def test_an_unlisted_package_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """表に無い包みを黙って外すと、一覧への足し忘れになる"""
        archive = _zip(tool, packages={"leftpad": ("leftpad", "1.0")})
        with pytest.raises(tool.BomError, match=r"leftpad 1\.0 を積んでいるが"):
            tool.generate(_write(tmp_path, archive))

    def test_a_stale_license_folder_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """積んでいない包みの写しを配ると、一覧と写しが食い違う"""
        extra = {"Sashimono/licenses/leftpad-1.0/LICENSE": b"leftpad"}
        with pytest.raises(tool.BomError, match=r"leftpad-1\.0 が積んだ包みにも"):
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

    def test_a_zip_without_the_record_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """出どころの記録の無い zip から中身を推して作ると、推し方の漏れが一覧の欠けになる"""
        with pytest.raises(tool.BomError, match="出どころの記録"):
            tool.generate(_write(tmp_path, _zip(tool, record=False)))

    @pytest.mark.parametrize(
        "name",
        [
            "Sashimono/_internal/unlisted.dll",
            "Sashimono/_internal/stranger/helper.dll",
            # 純 Python の包みの中身も、記録に無ければ止める（.pyc でも同じ）
            "Sashimono/_internal/packaging/__init__.pyc",
            "Sashimono/tool.exe",
            "外.txt",
        ],
    )
    def test_a_file_outside_the_record_stops(
        self, tool: ModuleType, tmp_path: Path, name: str
    ) -> None:
        """組み立ての記録に無いファイルを通すと、出どころの分からない物を一覧に載せずに配る"""
        with pytest.raises(tool.BomError, match="組み立ての記録に無い"):
            tool.generate(_write(tmp_path, _zip(tool, extra={name: b"binary"})))

    def test_a_recorded_file_missing_from_the_zip_stops(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """記録と zip が食い違うのは、別の組み立ての記録を入れた印"""
        drop = ("Sashimono/_internal/OpenGL/__init__.pyc",)
        with pytest.raises(tool.BomError, match="記録にあるが zip に無い"):
            tool.generate(_write(tmp_path, _zip(tool, drop=drop)))

    def test_an_unknown_owner_stops(self, tool: ModuleType, tmp_path: Path) -> None:
        """出どころの分からないファイルは、どの使用許諾で配るのか決められない"""
        archive = _zip(
            tool,
            extra={"Sashimono/_internal/odd.dll": b"odd"},
            owners={"_internal/odd.dll": "unknown"},
        )
        with pytest.raises(tool.BomError, match="出どころが分からない"):
            tool.generate(_write(tmp_path, archive))

    @pytest.mark.parametrize(
        ("name", "owner"),
        [
            # 部品へ当てる走査は av.libs と lupa の直下だけを当てる
            # 下のフォルダの DLL や、lupa の直下の Lua でない DLL を包みの名前で許さない
            ("_internal/av.libs/nested/libnew.dll", "av"),
            ("_internal/lupa/nested/lua54.cp314-win_amd64.pyd", "lupa"),
            ("_internal/lupa/libnew.dll", "lupa"),
        ],
    )
    def test_an_unassigned_native_file_stops(
        self, tool: ModuleType, tmp_path: Path, name: str, owner: str
    ) -> None:
        """記録にあっても、部品へ当たらない DLL はハッシュも部品も無いまま配られる"""
        archive = _zip(tool, extra={f"Sashimono/{name}": b"binary"}, owners={name: owner})
        with pytest.raises(tool.BomError, match="どの部品の物か決まっていない"):
            tool.generate(_write(tmp_path, archive))

    def test_recorded_files_are_listed_with_their_package(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        """記録のそろった包みは載せ、Python と一緒に入った DLL は Python の部品にハッシュを付ける"""
        listed = _listed(tool)
        lua = "_internal/lupa/lua54.cp314-win_amd64.pyd"
        extra = {
            "Sashimono/_internal/libssl-3.dll": b"ssl",
            "Sashimono/_internal/packaging/__init__.pyc": b"pyc",
            f"Sashimono/{lua}": b"lua",
            f"Sashimono/licenses/lupa-{listed['lupa']}/LICENSE.txt": b"lupa license",
        }
        owners = {
            "_internal/libssl-3.dll": "python",
            "_internal/packaging/__init__.pyc": "packaging",
            lua: "lupa",
        }
        packages = {
            "packaging": ("packaging", listed["packaging"]),
            "lupa": ("lupa", listed["lupa"]),
        }
        archive = _zip(tool, extra=extra, owners=owners, packages=packages)
        found = _components(tool.generate(_write(tmp_path, archive)).data)
        assert found["packaging"]["purl"] == f"pkg:pypi/packaging@{listed['packaging']}"
        assert "lupa" in found
        python = next(item for name, item in found.items() if name.startswith("Python"))
        inner = python["components"]
        assert isinstance(inner, list)
        assert "_internal/libssl-3.dll" in {item["name"] for item in inner}

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

    def test_the_zip_is_never_the_output(self, tool: ModuleType, tmp_path: Path) -> None:
        """前の一覧を消す所で、同じ場所を渡された配る zip を消さない"""
        archive = _write(tmp_path, _zip(tool))
        with pytest.raises(SystemExit):
            tool.main(["write", "--archive", str(archive), "--out", str(archive)])
        assert archive.is_file()

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

    def test_swapped_dlls_are_a_change(self, tool: ModuleType, tmp_path: Path) -> None:
        """2 つの DLL の中身が入れ替わっても、値だけを並べて比べると同じに見える"""
        before = tool.generate(_write(tmp_path, _zip(tool), "before.zip")).data
        after = json.loads(json.dumps(before))
        ffmpeg = next(c for c in after["components"] if c["name"] == "FFmpeg")
        first, second = ffmpeg["components"]
        first["hashes"], second["hashes"] = second["hashes"], first["hashes"]
        assert any(
            "版が同じで中身が違う: FFmpeg" in line for line in tool.differences(before, after)
        )
