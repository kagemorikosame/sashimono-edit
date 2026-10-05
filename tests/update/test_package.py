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

    @staticmethod
    def _write(folder: Path, relative: str, text: str) -> Path:
        path = folder / PORTABLE_SCRIPTS_DIR / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_rolling_back_takes_what_was_edited(self, layout: Layout) -> None:
        """戻す先の previous には更新する前の古い中身が残る 今の側で直した物を飛ばすと、
        直した中身が戻した版に入らず、次の自動更新で .previous ごと消える（PR #245 の Codex の指摘）
        """
        self._write(layout.install, "自分の/効果.anm2", "直した")
        old = self._write(layout.previous, "自分の/効果.anm2", "直す前")
        old_readme = self._write(layout.previous, "README.txt", "前の版の説明")
        self._write(layout.install, "README.txt", "今の版の説明")

        assert carry_user_files(layout.install, layout.previous, overwrite=True) == 1
        assert old.read_text(encoding="utf-8") == "直した"
        # 同梱の物は写さない（戻した版の説明はその版の物）
        assert old_readme.read_text(encoding="utf-8") == "前の版の説明"
        assert not list(layout.previous.rglob("*.moving"))

    def test_the_same_content_is_left_alone(self, layout: Layout) -> None:
        self._write(layout.install, "効果.anm2", "同じ")
        self._write(layout.previous, "効果.anm2", "同じ")
        assert carry_user_files(layout.install, layout.previous, overwrite=True) == 0

    def test_a_name_the_new_version_bundles_goes_aside(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """新しい版が同じ名前の物を同梱していれば、同梱物は新しい版の物を残し、本人の物は
        %APPDATA% の scripts へ写す %APPDATA% が後に読まれて勝つので、使われる物は変わらない
        """
        aside = tmp_path / "roaming" / "Sashimono" / "scripts"
        self._write(layout.install, "見本/揺れ.anm2", "本人が直した")
        bundled = self._write(layout.staged, "見本/揺れ.anm2", "新しい版の見本")

        assert carry_user_files(layout.install, layout.staged, aside=aside) == 1
        assert bundled.read_text(encoding="utf-8") == "新しい版の見本"
        assert (aside / "見本" / "揺れ.anm2").read_text(encoding="utf-8") == "本人が直した"

    def test_aside_is_not_overwritten(self, layout: Layout, tmp_path: Path) -> None:
        """新しい版の同梱物と %APPDATA% の両方に同じ名前の別の物があれば、どこへ写しても束が
        割れるか本人の物が落ちる %APPDATA% は上書きせず、入れ替えを止める（手で片付けてもらう）
        前は黙って飛ばし、exe の隣の本人の物は次の更新で消える .previous にだけ残っていた
        """
        aside = tmp_path / "roaming" / "Sashimono" / "scripts"
        (aside / "見本").mkdir(parents=True)
        (aside / "見本" / "揺れ.anm2").write_text("前から置いた", encoding="utf-8")
        self._write(layout.install, "見本/揺れ.anm2", "本人が直した")
        self._write(layout.staged, "見本/揺れ.anm2", "新しい版の見本")
        with pytest.raises(package_module.CarryError) as raised:
            carry_user_files(layout.install, layout.staged, aside=aside)
        assert "%APPDATA% の両方" in raised.value.failed[0][1]
        assert (aside / "見本" / "揺れ.anm2").read_text(encoding="utf-8") == "前から置いた"

    def test_a_bundle_goes_aside_as_a_whole(self, layout: Layout, tmp_path: Path) -> None:
        """束の 1 つが新しい版の同梱物とぶつかれば、束ごと %APPDATA% へ写す

        PR #245 の Codex の指摘 効果だけ新しい版へ写すと、効果のフォルダの新しい版の
        同梱モジュールを読んで描画が変わる
        """
        aside = tmp_path / "roaming" / "Sashimono" / "scripts"
        self._write(layout.install, "配布物/効果.anm2", 'local m = require("common")\n')
        self._write(layout.install, "配布物/common.mod2", "本人の common")
        bundled = self._write(layout.staged, "配布物/common.mod2", "新しい版の common")

        assert carry_user_files(layout.install, layout.staged, aside=aside) == 2
        assert (aside / "配布物" / "効果.anm2").is_file()
        assert (aside / "配布物" / "common.mod2").read_text(encoding="utf-8") == "本人の common"
        assert not (layout.staged / PORTABLE_SCRIPTS_DIR / "配布物" / "効果.anm2").exists()
        assert bundled.read_text(encoding="utf-8") == "新しい版の common"

    def test_a_module_name_the_new_version_bundles_elsewhere_moves_the_bundle(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """モジュールは名前で探す 新しい版が別の場所に同じ名前のモジュールを同梱していれば、
        束ごと %APPDATA% へ写す
        """
        aside = tmp_path / "roaming" / "Sashimono" / "scripts"
        self._write(layout.install, "配布物/common.lua", "本人の common")
        self._write(layout.staged, "common.lua", "新しい版の common")
        carry_user_files(layout.install, layout.staged, aside=aside)
        assert (aside / "配布物" / "common.lua").is_file()
        assert not (layout.staged / PORTABLE_SCRIPTS_DIR / "配布物").exists()

    def test_a_stopped_carry_takes_back_what_went_aside(
        self, layout: Layout, tmp_path: Path
    ) -> None:
        """入れ替えを止めるときは %APPDATA% へ写した物を外す %APPDATA% は今の版でも読まれるので、
        束の一部だけが残ると今の版の描画が変わる
        """
        aside = tmp_path / "roaming" / "Sashimono" / "scripts"
        self._write(layout.install, "配布物/common.mod2", "本人の common")
        self._write(layout.staged, "配布物/common.mod2", "新しい版の common")
        self._write(layout.install, "塞がれた/効果.anm2", "写せない")
        (layout.staged / PORTABLE_SCRIPTS_DIR / "塞がれた").write_text("塞ぐ", encoding="utf-8")
        with pytest.raises(package_module.CarryError):
            carry_user_files(layout.install, layout.staged, aside=aside)
        assert not (aside / "配布物" / "common.mod2").exists()

    def test_a_failure_is_raised_after_trying_everything(self, layout: Layout) -> None:
        """写せなかった物を黙って飛ばすと、呼んだ側が気付かずに入れ替え、本人の物は次の更新で
        消える版にだけ残る（PR #245 の CodeRabbit の指摘） 全部を試してから並べて上げる
        """
        self._write(layout.install, "塞がれた/効果.anm2", "写せない")
        self._write(layout.install, "通る/効果.anm2", "写せる")
        (layout.staged / PORTABLE_SCRIPTS_DIR).mkdir(parents=True)
        (layout.staged / PORTABLE_SCRIPTS_DIR / "塞がれた").write_text("塞ぐ", encoding="utf-8")

        with pytest.raises(package_module.CarryError) as raised:
            carry_user_files(layout.install, layout.staged)
        assert [path.as_posix() for path, _ in raised.value.failed] == ["塞がれた/効果.anm2"]
        assert "塞がれた/効果.anm2" in raised.value.explain()
        assert (layout.staged / PORTABLE_SCRIPTS_DIR / "通る" / "効果.anm2").is_file()
        assert not list(layout.staged.rglob("*.moving"))

    def test_a_failure_to_put_it_aside_is_raised_too(self, layout: Layout, tmp_path: Path) -> None:
        aside = tmp_path / "roaming" / "Sashimono" / "scripts"
        aside.mkdir(parents=True)
        (aside / "見本").write_text("塞ぐ", encoding="utf-8")
        self._write(layout.install, "見本/揺れ.anm2", "本人が直した")
        self._write(layout.staged, "見本/揺れ.anm2", "新しい版の見本")
        with pytest.raises(package_module.CarryError):
            carry_user_files(layout.install, layout.staged, aside=aside)

    def test_the_folder_name_matches_the_catalog(self) -> None:
        """名前が食い違うと、写す先が読まれない場所になる"""
        assert package_module._PORTABLE_SCRIPTS_DIR == PORTABLE_SCRIPTS_DIR
