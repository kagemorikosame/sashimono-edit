"""形式を上げる前の写し（自動更新 F-12-22）

自動更新の後に前の版へ戻した人は、新しい形式で保存し直した作品を開けない
（前の版は「更新してください」で止まる） 上げる前の中身を横に残しておく
"""

from __future__ import annotations

import json
from pathlib import Path

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
        assert keep_pre_upgrade_copy(tmp_path / "無い.sme") is None
        broken = tmp_path / "壊れた.sme"
        broken.write_text("{", encoding="utf-8")
        assert keep_pre_upgrade_copy(broken) is None
