"""形式を上げる前の写し（自動更新 F-12-22）

自動更新の後に前の版へ戻した人は、新しい形式で保存し直した作品を開けない
（前の版は「更新してください」で止まる） 上げる前の中身を横に残しておく
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sashimono.core.io.serialize import (
    FORMAT_VERSION,
    PRE_UPGRADE_SUFFIX,
    keep_pre_upgrade_copy,
    save_project,
)
from sashimono.core.model import Project


def _written(path: Path, version: int) -> str:
    text = json.dumps({"format": "sashimono-project", "version": version, "marker": "前の中身"})
    path.write_text(text, encoding="utf-8")
    return text


class TestThePreUpgradeCopy:
    def test_an_older_file_is_copied_beside(self, tmp_path: Path) -> None:
        # 写しが残らないと、更新の後で前の版へ戻した人は、新しい形で上書き保存した作品を
        # 前の版で開けず、更新前の中身へも戻せない
        path = tmp_path / "作品.sme"
        text = _written(path, FORMAT_VERSION - 1)
        copy = keep_pre_upgrade_copy(path)
        assert copy == tmp_path / f"作品.sme{PRE_UPGRADE_SUFFIX}"
        assert copy.read_text(encoding="utf-8") == text

    def test_the_copy_is_the_last_one_before_the_upgrade(self, tmp_path: Path) -> None:
        """前に上げたときの写しは古すぎる 上げる直前の中身で書き直す"""
        path = tmp_path / "作品.sme"
        (tmp_path / f"作品.sme{PRE_UPGRADE_SUFFIX}").write_text("もっと前", encoding="utf-8")
        text = _written(path, FORMAT_VERSION - 1)
        copy = keep_pre_upgrade_copy(path)
        assert copy is not None and copy.read_text(encoding="utf-8") == text

    def test_a_current_file_is_left_alone(self, tmp_path: Path) -> None:
        """上げないなら写さない 保存のたびに写しを作ると、上げる前の中身が上書きされる"""
        path = tmp_path / "作品.sme"
        save_project(Project.create(), path)
        assert keep_pre_upgrade_copy(path) is None
        assert sorted(p.name for p in tmp_path.iterdir()) == ["作品.sme"]

    def test_missing_or_broken_files_are_left_alone(self, tmp_path: Path) -> None:
        # 読めない中身で例外が漏れると、壊れた作品の名前へ上書き保存するときに保存ごと止まり
        # いまの作業を失う
        assert keep_pre_upgrade_copy(tmp_path / "無い.sme") is None
        broken = tmp_path / "壊れた.sme"
        broken.write_text("{", encoding="utf-8")
        assert keep_pre_upgrade_copy(broken) is None


class TestTheCopyDoesNotWriteThroughLinks:
    """写しの名前に置かれたリンクの先へは書かない（#327 CodeRabbit の指摘）

    AI の上書き保存も画面の保存もここを通る 写しを開いて上書きすると、写しの名前が
    元の素材へのハードリンクや外へのリンクだったとき、素材や外のファイルが作品の中身に変わる
    """

    def test_a_hard_link_at_the_copy_name_keeps_the_linked_file(self, tmp_path: Path) -> None:
        # 壊れると、作品のフォルダに置かれた「作品.sme.bak」という名前の素材へのハードリンクを
        # 通って、保存のたびに素材が作品の中身で上書きされる
        original = tmp_path / "素材.mp4"
        original.write_bytes(b"original")
        path = tmp_path / "作品.sme"
        text = _written(path, FORMAT_VERSION - 1)
        copy_name = tmp_path / f"作品.sme{PRE_UPGRADE_SUFFIX}"
        try:
            os.link(original, copy_name)
        except OSError as exc:
            pytest.skip(f"ハードリンクを作れない: {exc}")
        copy = keep_pre_upgrade_copy(path)
        assert copy is not None and copy.read_text(encoding="utf-8") == text
        assert original.read_bytes() == b"original"

    def test_a_symbolic_link_at_the_copy_name_keeps_the_outside_file(self, tmp_path: Path) -> None:
        # 壊れると、写しの名前に置かれたリンクを辿って、作品のフォルダの外のファイルを上書きする
        outside = tmp_path / "外" / "大事.txt"
        outside.parent.mkdir()
        outside.write_bytes(b"outside")
        folder = tmp_path / "作品"
        folder.mkdir()
        path = folder / "作品.sme"
        _written(path, FORMAT_VERSION - 1)
        copy_name = folder / f"作品.sme{PRE_UPGRADE_SUFFIX}"
        try:
            copy_name.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"リンクを作れない: {exc}")
        keep_pre_upgrade_copy(path)
        assert outside.read_bytes() == b"outside"
        assert not copy_name.is_symlink()
