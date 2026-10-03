"""新しい版を落として確かめ、隣のフォルダへ置く"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.compat.aviutl.catalog import PORTABLE_SCRIPTS_DIR
from sashimono.update import package as package_module
from sashimono.update.fetch import FetchError, MemoryTransport
from sashimono.update.package import (
    APP_EXE,
    Layout,
    PackageError,
    carry_user_files,
    read_build_info,
    stage,
)
from tests.update.helpers import manifest_for, package_bytes, publish, release


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    install = tmp_path / "Programs" / "Sashimono"
    install.mkdir(parents=True)
    (install / APP_EXE).write_bytes(b"MZ current")
    return Layout(install)


def _leftovers(layout: Layout) -> list[str]:
    return sorted(p.name for p in layout.install.parent.iterdir() if p != layout.install)


class TestStaging:
    def test_a_good_package_waits_beside(self, layout: Layout) -> None:
        transport, manifest = release(Ed25519PrivateKey.generate(), "1.2.0")
        staged = stage(manifest, transport, layout)
        assert staged == layout.staged
        assert layout.staged_version() == "1.2.0"
        # 落とした zip と、展開の途中の物は残さない
        assert _leftovers(layout) == [layout.staged.name]

    def test_a_wrong_hash_stages_nothing(self, layout: Layout) -> None:
        transport, manifest = release(Ed25519PrivateKey.generate(), "1.2.0")
        bad = replace(manifest, package=replace(manifest.package, sha256="0" * 64))
        with pytest.raises(FetchError):
            stage(bad, transport, layout)
        assert _leftovers(layout) == []

    def test_a_package_reaching_outside_is_refused(self, layout: Layout) -> None:
        """``..`` を含む zip をそのまま展開すると、本人の設定やスタートアップへ書かれる"""
        transport = MemoryTransport()
        body = package_bytes("1.2.0", extra={"Sashimono/../../escaped.txt": b"x"})
        manifest = manifest_for("1.2.0", body)
        publish(transport, Ed25519PrivateKey.generate(), manifest, body)
        with pytest.raises(PackageError):
            stage(manifest, transport, layout)
        assert not (layout.install.parent.parent / "escaped.txt").exists()
        assert _leftovers(layout) == []

    @pytest.mark.parametrize("field", ["version", "python_abi"])
    def test_a_package_unlike_its_manifest_is_refused(self, layout: Layout, field: str) -> None:
        """目録の言う版・Python と、zip の中の書き付けが違えば入れない

        違う版を入れると「最新です」と言い続けるか、入れた実行環境が読めなくなる
        """
        transport = MemoryTransport()
        body = package_bytes("1.2.0", abi="cp399" if field == "python_abi" else None)
        if field == "version":
            body = package_bytes("1.1.9")
        manifest = manifest_for("1.2.0", body)
        publish(transport, Ed25519PrivateKey.generate(), manifest, body)
        with pytest.raises(PackageError):
            stage(manifest, transport, layout)
        assert layout.staged_version() is None
        assert _leftovers(layout) == []

    def test_an_older_stage_is_replaced(self, layout: Layout) -> None:
        key = Ed25519PrivateKey.generate()
        transport, first = release(key, "1.2.0")
        stage(first, transport, layout)
        _, second = release(key, "1.3.0", transport=transport)
        stage(second, transport, layout)
        assert layout.staged_version() == "1.3.0"

    def test_the_hash_is_checked_on_the_bytes(self, layout: Layout) -> None:
        transport, manifest = release(Ed25519PrivateKey.generate(), "1.2.0")
        stage(manifest, transport, layout)
        body = transport.pages[next(u for u in transport.pages if u.endswith(".zip"))]
        assert hashlib.sha256(body).hexdigest() == manifest.package.sha256


class TestTheLayout:
    def test_siblings_share_the_parent(self, layout: Layout) -> None:
        """入れ替えは改名だけで行う 同じ親の下でないと、改名がコピーになる"""
        for path in (layout.staged, layout.previous, layout.failed, layout.download):
            assert path.parent == layout.install.parent

    def test_an_unwritable_parent_is_reported(self, tmp_path: Path) -> None:
        """Program Files のように書けない所では、自動では入れ替えない"""
        assert not Layout(tmp_path / "missing" / "Sashimono").writable()

    def test_a_writable_parent_is_left_clean(self, layout: Layout) -> None:
        assert layout.writable()
        assert _leftovers(layout) == []

    def test_build_info_is_read(self, layout: Layout) -> None:
        assert read_build_info(layout.install) is None
        package_module.write_build_info(layout.install, "1.2.3", "cp314")
        info = read_build_info(layout.install)
        assert info is not None and (info.version, info.python_abi) == ("1.2.3", "cp314")


class TestUserScripts:
    def test_they_follow_the_new_version(self, layout: Layout) -> None:
        """exe の隣に置いたスクリプトを写さないと、2 回目の入れ替えで消える"""
        mine = layout.install / PORTABLE_SCRIPTS_DIR / "自分の" / "効果.anm2"
        mine.parent.mkdir(parents=True)
        mine.write_text("--track", encoding="utf-8")
        (layout.install / PORTABLE_SCRIPTS_DIR / "README.txt").write_text(
            "古い説明", encoding="utf-8"
        )
        new_readme = layout.staged / PORTABLE_SCRIPTS_DIR / "README.txt"
        new_readme.parent.mkdir(parents=True)
        new_readme.write_text("新しい説明", encoding="utf-8")

        assert carry_user_files(layout.install, layout.staged) == 1
        copied = layout.staged / PORTABLE_SCRIPTS_DIR / "自分の" / "効果.anm2"
        assert copied.read_text(encoding="utf-8") == "--track"
        # 新しい版が持っている物は新しい方を残す
        assert new_readme.read_text(encoding="utf-8") == "新しい説明"

    def test_the_folder_name_matches_the_catalog(self) -> None:
        """名前が食い違うと、写す先が読まれない場所になる"""
        assert package_module._PORTABLE_SCRIPTS_DIR == PORTABLE_SCRIPTS_DIR
