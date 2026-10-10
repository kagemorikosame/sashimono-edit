"""保存したプリセットとエイリアスの整理（#276）

前は作る・使うだけで、消す・名前を変える・分類を変える道が画面に無かった
（``PresetStore.delete`` と ``AliasStore.delete`` はどこからも呼ばれていなかった）
ここではファイルの扱い（コアの :class:`Library`）を見る 窓は tests/ui/test_library_dialog.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.io import Preset, PresetStore
from sashimono.core.io.aliases import DEFAULT_CATEGORY, Alias, AliasStore
from sashimono.core.io.library import (
    ALIAS,
    PRESET,
    Library,
    LibraryChangedError,
    LibraryConflictError,
    LibraryEntry,
    LibraryError,
    LibraryKind,
    look_fingerprint,
)
from sashimono.core.io.presets import TRASH_FOLDER
from sashimono.core.model import Clip, Effect
from sashimono.effects.sources import TEXT


def _clip(text: str = "見出し") -> Clip:
    return with_fixed_items(
        Clip(timeline_start=0, duration=60, source=TEXT.create(text=text)), picture=True
    )


@pytest.fixture
def library(tmp_path: Path) -> Library:
    return Library(
        presets=PresetStore(tmp_path / "presets"), aliases=AliasStore(tmp_path / "aliases")
    )


def _names(library: Library, kind: LibraryKind = PRESET) -> list[tuple[str, str]]:
    return [(e.category, e.name) for e in library.entries(kind)]


def _entry(library: Library, name: str, kind: LibraryKind = PRESET) -> LibraryEntry:
    return next(e for e in library.entries(kind) if e.name == name)


class TestRename:
    def test_the_file_follows_the_new_name(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        old = _entry(library, "赤")
        # 名前を変えても「保存した日」の並びが動かないように、保存した時刻を残す
        os.utime(old.path, (1_000_000_000, 1_000_000_000))
        old = _entry(library, "赤")

        renamed = library.rename(old, "青")

        assert not old.path.exists()
        assert renamed.path.name == "青.smep"
        assert _names(library) == [("ユーザー", "青")]
        assert renamed.modified == pytest.approx(1_000_000_000)
        assert library.presets.load(renamed.path).name == "青"

    def test_a_taken_name_is_asked_and_the_old_one_goes_to_the_trash(
        self, library: Library
    ) -> None:
        library.presets.save(Preset.capture("赤", _clip("赤の文字")))
        library.presets.save(Preset.capture("青", _clip("青の文字")))
        with pytest.raises(LibraryConflictError):
            library.rename(_entry(library, "赤"), "青")
        # 断られたら何も変えない
        assert _names(library) == [("ユーザー", "赤"), ("ユーザー", "青")]

        library.rename(_entry(library, "赤"), "青", overwrite=True)
        assert _names(library) == [("ユーザー", "青")]
        survivor = _entry(library, "青").item
        assert isinstance(survivor, Preset) and survivor.source is not None
        assert survivor.source.params["text"] == "赤の文字"
        # 上書きで消えた方は戻せる
        (trashed,) = library.trashed(PRESET)
        assert trashed.name == "青"

    def test_unsafe_characters_stay_inside_the_folder(
        self, library: Library, tmp_path: Path
    ) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        renamed = library.rename(_entry(library, "赤"), "a/b:c")
        moved = library.recategorize(renamed, "../外")
        # 分類も名前もファイル名に使えない文字を外す 外さないと置き場の外へ書く
        assert moved.path.is_relative_to(tmp_path / "presets")
        assert TRASH_FOLDER not in moved.path.parts
        assert _names(library) == [("../外", "a/b:c")]

    def test_an_empty_name_is_refused(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        with pytest.raises(LibraryError):
            library.rename(_entry(library, "赤"), "   ")

    def test_alias_names_differing_only_in_case_do_not_collide(self, library: Library) -> None:
        library.aliases.save(Alias.of("abc", _clip()))
        renamed = library.rename(_entry(library, "abc", ALIAS), "ABC")
        assert _names(library, ALIAS) == [(DEFAULT_CATEGORY, "ABC")]
        assert renamed.path.exists()


class TestCategory:
    def test_a_preset_moves_to_the_new_folder(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        moved = library.recategorize(_entry(library, "赤"), "見出し")
        assert moved.path.parent.name == "見出し"
        assert _names(library) == [("見出し", "赤")]
        assert library.categories(PRESET) == ("見出し",)

    def test_an_alias_keeps_its_file_and_carries_the_category(self, library: Library) -> None:
        saved = library.aliases.save(Alias.of("テロップ", _clip()))
        moved = library.recategorize(_entry(library, "テロップ", ALIAS), "字幕")
        assert moved.path == saved
        assert library.aliases.load(saved).category == "字幕"

    def test_an_old_alias_without_a_category_reads_as_the_default(self, tmp_path: Path) -> None:
        store = AliasStore(tmp_path)
        path = store.save(Alias.of("前の版", _clip()))
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["category"]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert store.load(path).category == DEFAULT_CATEGORY


class TestDuplicate:
    def test_the_original_stays(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        copy = library.duplicate(_entry(library, "赤"), "赤 のコピー")
        assert _names(library) == [("ユーザー", "赤"), ("ユーザー", "赤 のコピー")]
        assert copy.item.name == "赤 のコピー"

    def test_a_taken_name_is_a_conflict(self, library: Library) -> None:
        library.aliases.save(Alias.of("a", _clip()))
        library.aliases.save(Alias.of("b", _clip()))
        with pytest.raises(LibraryConflictError):
            library.duplicate(_entry(library, "a", ALIAS), "b")


class TestTrash:
    def test_deleted_items_leave_the_list_and_come_back(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip(), category="見出し"))
        entry = _entry(library, "赤")
        trashed = library.delete(entry)

        assert not entry.path.exists()
        # 置き場の一覧（メニュー）にも出ない ごみ箱は置き場の中にあるが数えない
        assert library.presets.all() == ()
        assert library.trashed(PRESET) == (trashed,)

        restored = library.restore(trashed)
        assert restored.path == entry.path
        assert _names(library) == [("見出し", "赤")]
        assert library.trashed(PRESET) == ()

    def test_restoring_over_a_newer_one_asks_first(self, library: Library) -> None:
        library.aliases.save(Alias.of("a", _clip("前")))
        trashed = library.delete(_entry(library, "a", ALIAS))
        library.aliases.save(Alias.of("a", _clip("後")))
        with pytest.raises(LibraryConflictError):
            library.restore(trashed)
        library.restore(trashed, overwrite=True)
        item = _entry(library, "a", ALIAS).item
        assert isinstance(item, Alias) and item.clip.source is not None
        assert item.clip.source.params["text"] == "前"
        # 上書きされた「後」もごみ箱に残る
        assert [t.name for t in library.trashed(ALIAS)] == ["a"]

    def test_emptying_removes_for_good(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        library.presets.save(Preset.capture("青", _clip()))
        library.delete(_entry(library, "赤"))
        library.delete(_entry(library, "青"))
        assert library.empty_trash(PRESET) == 2
        assert library.trashed(PRESET) == ()
        assert not any((library.presets.root / TRASH_FOLDER).iterdir())

    def test_a_trash_entry_gone_elsewhere_is_reported(self, library: Library) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        trashed = library.delete(_entry(library, "赤"))
        library.empty_trash(PRESET)
        with pytest.raises(LibraryChangedError):
            library.restore(trashed)


class TestAnotherWindow:
    def test_a_file_removed_elsewhere_is_not_recreated(self, library: Library) -> None:
        # 2 つの窓で同じ一覧を開き、片方で消した後にもう片方で名前を変える
        library.presets.save(Preset.capture("赤", _clip()))
        seen = _entry(library, "赤")
        seen.path.unlink()
        for action in (
            lambda: library.rename(seen, "青"),
            lambda: library.recategorize(seen, "別"),
            lambda: library.delete(seen),
            lambda: library.duplicate(seen, "写し"),
        ):
            with pytest.raises(LibraryChangedError):
                action()
        # 黙って作り直すと、ほかの窓で消した物が戻ってくる
        assert library.entries(PRESET) == ()


class TestLegacyFiles:
    def _legacy(self, library: Library, name: str, suffix: str = ".kmkp") -> Path:
        folder = library.presets.root / "ユーザー"
        folder.mkdir(parents=True, exist_ok=True)
        data = Preset(name=name, effects=(Effect(kind="glow"),)).to_dict()
        data["format"] = "kumiki-preset"
        path = folder / f"{name}{suffix}"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_old_files_are_listed_and_rewritten_in_the_new_form(self, library: Library) -> None:
        old = self._legacy(library, "昔")
        (entry,) = library.entries(PRESET)
        assert entry.legacy
        renamed = library.rename(entry, "今")
        assert not old.exists()
        assert renamed.path.suffix == ".smep"
        assert not renamed.legacy

    def test_a_hidden_old_copy_does_not_come_back_after_a_rename(self, library: Library) -> None:
        # 改名前に保存した物を新しい版で上書きすると、旧い拡張子のファイルが隠れて残る
        hidden = self._legacy(library, "見出し")
        library.presets.save(Preset.capture("見出し", _clip()))
        (entry,) = library.entries(PRESET)
        assert entry.shadows == (hidden,)
        library.rename(entry, "新しい名前")
        assert _names(library) == [("ユーザー", "新しい名前")]
        # 消さずにごみ箱へ（黙って消さない）
        assert [t.name for t in library.trashed(PRESET)] == ["見出し"]


class TestExchange:
    def test_export_then_import_on_another_machine(self, library: Library, tmp_path: Path) -> None:
        library.presets.save(Preset.capture("赤", _clip(), category="見出し"))
        library.aliases.save(Alias.of("テロップ", _clip(), category="字幕"))
        out = tmp_path / "out"
        files = [library.export(entry, out) for entry in library.entries(PRESET)]
        files += [library.export(entry, out) for entry in library.entries(ALIAS)]
        assert sorted(path.suffix for path in files) == [".smea", ".smep"]

        other = Library(
            presets=PresetStore(tmp_path / "other" / "presets"),
            aliases=AliasStore(tmp_path / "other" / "aliases"),
        )
        for path in files:
            kind, item = other.read_file(path)
            other.add(kind, item)
        assert _names(other) == [("見出し", "赤")]
        assert _names(other, ALIAS) == [("字幕", "テロップ")]

    def test_export_does_not_overwrite_without_asking(
        self, library: Library, tmp_path: Path
    ) -> None:
        library.presets.save(Preset.capture("赤", _clip()))
        entry = _entry(library, "赤")
        library.export(entry, tmp_path / "out")
        with pytest.raises(LibraryConflictError):
            library.export(entry, tmp_path / "out")
        library.export(entry, tmp_path / "out", overwrite=True)

    def test_keeping_both_numbers_the_new_one(self, library: Library) -> None:
        item = Preset.capture("赤", _clip())
        library.add(PRESET, item)
        assert library.occupied(PRESET, item)
        assert library.free_name(PRESET, item) == "赤 (2)"
        with pytest.raises(LibraryConflictError):
            library.add(PRESET, item)

    @pytest.mark.parametrize(
        ("name", "content"),
        [
            ("壊れ.smep", "{"),
            ("違う.smep", json.dumps({"format": "sashimono-project"})),
            ("新しい.smep", json.dumps({"format": "sashimono-preset", "version": 99})),
            ("文字化け.smea", b"\xff\xfe\x00".decode("latin-1")),
            ("知らない.txt", "{}"),
        ],
    )
    def test_unreadable_files_are_refused(
        self, library: Library, tmp_path: Path, name: str, content: str
    ) -> None:
        path = tmp_path / name
        path.write_text(content, encoding="latin-1")
        with pytest.raises(LibraryError):
            library.read_file(path)


class TestFingerprint:
    def test_names_and_categories_do_not_change_the_picture(self) -> None:
        first = Preset.capture("赤", _clip())
        renamed = Preset.capture("青", _clip(), category="別")
        assert look_fingerprint(first) == look_fingerprint(renamed)

    def test_the_look_does(self) -> None:
        assert look_fingerprint(Preset.capture("a", _clip("x"))) != look_fingerprint(
            Preset.capture("a", _clip("y"))
        )


class TestMany:
    def test_three_hundred_items_list_quickly(self, library: Library) -> None:
        for number in range(300):
            library.presets.save(Preset.capture(f"見出し {number:03}", _clip()))
            library.aliases.save(Alias.of(f"テロップ {number:03}", _clip()))
        started = time.perf_counter()
        assert len(library.entries(PRESET)) == 300
        assert len(library.entries(ALIAS)) == 300
        # 手元で 0.4 秒ほど CI の遅い機械でも窓を開く操作がもたつかない目安
        assert time.perf_counter() - started < 5.0
