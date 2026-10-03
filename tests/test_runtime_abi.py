"""入れた実行環境と、本体の Python の食い違い（自動更新 F-12-16）

本体の更新で Python が上がる（3.14 → 3.15）と、入れてある字幕起こしと AI 連携の
拡張モジュールは読めなくなる import して落ちる前に気付いて「入れ直してください」と言う
2 GB を黙って捨てて落とし直すことはしない
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono import runtime as runtime_module
from sashimono.ai.environment import AI_PACK
from sashimono.asr.environment import ASR_PACK
from sashimono.runtime import (
    ABI_MARKER,
    FeaturePack,
    install_runtime,
    python_abi,
    runtime_abi,
    stale_runtime,
)

#: 前の Python の印 走らせている Python から 1 つ前を作る 決め打ちすると、その版の Python
#: （対応している 3.12〜）で走らせたときに今の印と同じになり、古さを見分ける試験が通らない
OLD = f"cp{sys.version_info.major}{sys.version_info.minor - 1}"


def test_the_old_tag_is_really_old() -> None:
    """試験の前提 OLD が今の Python の印と同じだと、どの試験も古さを見分けたことにならない"""
    assert python_abi() != OLD


@pytest.fixture
def target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    folder = tmp_path / "runtime"
    folder.mkdir()
    monkeypatch.setattr(runtime_module, "runtime_target_dir", lambda: folder)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return folder


def _extension(target: Path, package: str, abi: str) -> None:
    folder = target / package
    folder.mkdir(exist_ok=True)
    (folder / f"_ext.{abi}-win_amd64.pyd").write_bytes(b"MZ")


class TestFindingOut:
    def test_the_tag_is_major_and_minor(self) -> None:
        assert python_abi() == f"cp{sys.version_info.major}{sys.version_info.minor}"

    def test_extension_names_tell_the_old_python(self, target: Path) -> None:
        """印を書くようになる前に入れた導入先も、拡張モジュールの名前で分かる"""
        _extension(target, "ctranslate2", OLD)
        _extension(target, "pydantic_core", OLD)
        assert runtime_abi(target) == OLD
        assert stale_runtime() == OLD

    def test_the_guess_is_written_down(self, target: Path) -> None:
        """数えた結果を書き残す 片方だけ入れ直した後は、数えても決められない"""
        _extension(target, "ctranslate2", OLD)
        runtime_abi(target)
        assert json.loads((target / ABI_MARKER).read_text(encoding="utf-8")) == {"*": OLD}

    def test_a_pure_python_runtime_is_not_stale(self, target: Path) -> None:
        """拡張モジュールの無い物は分からない 分からない物を古いと決めると、動く物を読まなくなる"""
        (target / "pure").mkdir()
        assert runtime_abi(target) is None
        assert stale_runtime() is None

    def test_the_current_python_is_not_stale(self, target: Path) -> None:
        _extension(target, "ctranslate2", python_abi())
        assert stale_runtime() is None


class TestWhatHappens:
    def test_a_stale_runtime_is_not_imported(self, target: Path) -> None:
        """道へ足すと、字幕起こしを始めた瞬間に拡張モジュールが読めずに落ちる"""
        _extension(target, "ctranslate2", OLD)
        assert runtime_module.activate_runtime() is None
        assert str(target) not in sys.path

    def test_it_is_not_deleted(self, target: Path) -> None:
        _extension(target, "ctranslate2", OLD)
        runtime_module.activate_runtime()
        assert (target / "ctranslate2" / f"_ext.{OLD}-win_amd64.pyd").exists()

    def test_the_pack_asks_for_a_reinstall(self, target: Path) -> None:
        """「未導入」と出すと 2 GB が消えたように見える 入れ直す理由を言う"""
        _extension(target, "ctranslate2", OLD)
        status = ASR_PACK.status()
        assert status.stale_abi == OLD
        assert "入れ直して" in status.summary()
        # 入れ直すときは上書きさせる 付けないと pip は名前が在るだけで飛ばす
        assert status.needs_upgrade

    def test_reinstalling_one_pack_leaves_the_other_stale(self, target: Path) -> None:
        """導入先は 2 つの機能で分け合う 片方を入れ直しても、もう片方の古さは残る"""
        _extension(target, "ctranslate2", OLD)
        runtime_abi(target)
        runtime_module._mark_installed(target, ASR_PACK.key)
        assert stale_runtime(ASR_PACK.key) is None
        assert stale_runtime(AI_PACK.key) == OLD
        # 読める機能が 1 つでもあれば導入先は読む
        assert runtime_module.activate_runtime() == target


class TestOnlyOnePackReinstalled:
    """片方だけを入れ直すと導入先は読まれ、もう片方も名前の上では「入っている」に見える

    そのまま動かせることにすると、古い拡張モジュールの import で落ちる
    """

    PACK = FeaturePack(key="ai", label="見本の機能", required=("sashimono-abi-probe",))

    @pytest.fixture
    def half(self, target: Path) -> Path:
        info = target / "sashimono_abi_probe-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: sashimono-abi-probe\nVersion: 1.0\n", encoding="utf-8"
        )
        _extension(target, "pydantic_core", OLD)
        runtime_abi(target)
        runtime_module._mark_installed(target, ASR_PACK.key)
        assert runtime_module.activate_runtime() == target
        importlib.invalidate_caches()
        return target

    def test_the_other_pack_is_not_ready(self, half: Path) -> None:
        status = self.PACK.status()
        assert status.installed
        assert status.stale_abi == OLD
        assert not status.ready

    def test_the_setup_asks_for_a_reinstall(self, half: Path, qt_application: QApplication) -> None:
        """導入の欄が「入れ直して」と出し、動かせない（AI の送信・起こしを止める）と伝える"""
        del qt_application
        from sashimono.ui.setup import SetupSection

        section = SetupSection(self.PACK)
        seen: list[bool] = []
        section.changed.connect(seen.append)
        try:
            section.refresh()
            assert "入れ直して" in section._status.text()
            assert seen == [False]
            assert "--upgrade" in section.command_text()
        finally:
            section.deleteLater()


class TestInstalling:
    def test_a_finished_install_is_marked(self, target: Path) -> None:
        code = install_runtime(ASR_PACK, command=[sys.executable, "-c", "pass"])
        assert code == 0
        marks = json.loads((target / ABI_MARKER).read_text(encoding="utf-8"))
        assert marks[ASR_PACK.key] == python_abi()

    def test_a_failed_install_is_not(self, target: Path) -> None:
        """途中で止めた導入先に今の印を書くと、古い拡張モジュールが残ったまま読まれる"""
        _extension(target, "ctranslate2", OLD)
        install_runtime(ASR_PACK, command=[sys.executable, "-c", "raise SystemExit(1)"])
        assert stale_runtime(ASR_PACK.key) == OLD
