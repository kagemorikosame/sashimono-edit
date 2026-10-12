"""改名前の名前で保存されたファイルを読めること

旧名を残す: このファイル全体（名前の一括置換でも書き換えない 旧名そのものを試す）

改名は 2 回あった（NovaEdit → Kumiki → Sashimono Edit） どの名前で保存した作品も
プリセットも、新しい版でそのまま開けなければならない 書くときは常に新しい名前で書く

ここが落ちるのは、旧名を読む手当て（``LEGACY_FORMAT_NAMES`` と ``LEGACY_SUFFIXES``）を
消したか、名前の一括置換がそこまで書き換えたとき 改名前に作った作品が
「プロジェクトファイルではない」で開けなくなり、プリセットは一覧から黙って消える
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sashimono.core.io import (
    RecoverySession,
    backup_before_save,
    backup_folder,
    find_orphans,
    load_project,
    save_project,
)
from sashimono.core.io.presets import Preset, PresetStore
from sashimono.core.model import Effect, Project


def _write_with_format(project: Project, path: Path, format_name: str) -> None:
    """新しい版で書いてから、形式の名前だけを旧名へ差し替える

    旧版が書いたファイルと違うのは形式の名前だけ（中身の形は改名で変えていない）
    """
    scratch = path.with_name("scratch.json")
    save_project(project, scratch)
    data = json.loads(scratch.read_text("utf-8"))
    data["format"] = format_name
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    scratch.unlink()


class TestOldProjects:
    @pytest.mark.parametrize(
        ("format_name", "suffix"),
        [("kumiki-project", ".kmk"), ("novaedit-project", ".nvep")],
    )
    def test_a_project_saved_under_an_old_name_opens(
        self, tmp_path: Path, format_name: str, suffix: str
    ) -> None:
        path = tmp_path / f"改名前の作品{suffix}"
        _write_with_format(Project.create(name="改名前の作品"), path, format_name)
        assert load_project(path).name == "改名前の作品"

    def test_saving_an_old_project_writes_the_new_format(self, tmp_path: Path) -> None:
        # 旧名のまま書き戻すと、旧名を読む手当てをいつまでも外せなくなる
        path = tmp_path / "改名前の作品.kmk"
        _write_with_format(Project.create(name="改名前の作品"), path, "kumiki-project")
        save_project(load_project(path), path)
        assert json.loads(path.read_text("utf-8"))["format"] == "sashimono-project"

    def test_the_open_dialog_offers_the_old_suffixes(self) -> None:
        # 開く窓の絞り込みはこの一覧から作る 外すと、改名前の作品がフォルダにあっても
        # 開く窓に出てこない（「すべてのファイル」に切り替えないと選べない）
        from sashimono.core.io import LEGACY_SUFFIXES

        assert {".kmk", ".nvep"} <= set(LEGACY_SUFFIXES)


def _write_preset(path: Path, format_name: str, name: str) -> None:
    data = Preset(name=name, effects=(Effect(kind="blur"),)).to_dict()
    data["format"] = format_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")


class TestOldPresets:
    @pytest.mark.parametrize(
        ("format_name", "suffix"),
        [("kumiki-preset", ".kmkp"), ("novaedit-preset", ".nvpreset")],
    )
    def test_an_old_preset_is_listed(self, tmp_path: Path, format_name: str, suffix: str) -> None:
        # 一覧に出ないと、本人には「プリセットが全部消えた」に見える
        _write_preset(tmp_path / "ユーザー" / f"ぼかし{suffix}", format_name, "ぼかし")
        assert [preset.name for preset in PresetStore(tmp_path).all()] == ["ぼかし"]

    def test_resaving_an_old_preset_does_not_list_it_twice(self, tmp_path: Path) -> None:
        # 上書き保存は新しい拡張子で書くので、旧いファイルが隣に残る 両方出すと
        # 同じ名前が 2 つ並び、選んだ方によって中身が違う
        store = PresetStore(tmp_path)
        _write_preset(tmp_path / "ユーザー" / "ぼかし.kmkp", "kumiki-preset", "ぼかし")
        saved = store.save(Preset(name="ぼかし", effects=(Effect(kind="glow"),)))
        assert saved.suffix == ".smep"
        assert json.loads(saved.read_text("utf-8"))["format"] == "sashimono-preset"
        (only,) = store.all()
        assert [effect.kind for effect in only.effects] == ["glow"]

    def test_deleting_removes_the_old_file_too(self, tmp_path: Path) -> None:
        # 旧いファイルを残すと、消したはずのプリセットが一覧に戻ってくる
        store = PresetStore(tmp_path)
        _write_preset(tmp_path / "ユーザー" / "ぼかし.kmkp", "kumiki-preset", "ぼかし")
        (preset,) = store.all()
        store.delete(preset)
        assert store.all() == ()


class TestOldRecoveryAndBackups:
    def test_a_crash_left_by_the_old_version_is_offered(self, tmp_path: Path) -> None:
        # 旧版で落ちた退避は、置き場を引き継いだあとも旧い拡張子のまま残る
        # 拾わないと、その作業は復元を勧められずに埋もれる
        folder = tmp_path / "recovery"
        folder.mkdir()
        _write_with_format(
            Project.create(name="落ちた作業"),
            folder / "0123456789abcdef0123456789abcdef.kmk",
            "kumiki-project",
        )

        (entry,) = find_orphans(tmp_path)
        assert entry.path == folder / "0123456789abcdef0123456789abcdef.kmk"
        assert load_project(entry.path).name == "落ちた作業"

    def test_a_live_session_still_wins(self, tmp_path: Path) -> None:
        # 旧い拡張子を拾うようにしても、生きている起動の退避は拾わない
        session = RecoverySession(tmp_path)
        try:
            session.save(Project.create(), None)
            assert find_orphans(tmp_path) == []
        finally:
            session.close()

    def test_old_backups_count_as_generations(self, tmp_path: Path) -> None:
        # 数えないと、改名前の控えはいつまでも消えずに残る
        target = tmp_path / "本編.sme"
        state = tmp_path / "state"
        folder = backup_folder(target, state)
        folder.mkdir(parents=True)
        (folder / "20200101-000000-000000-000.kmk").write_text("改名前", "utf-8")
        for index in range(2):
            target.write_text(str(index), "utf-8")
            backup_before_save(target, state, keep=2)
        assert [path.read_text("utf-8") for path in sorted(folder.iterdir())] == ["0", "1"]
