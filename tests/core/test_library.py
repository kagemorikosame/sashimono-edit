"""保存したプリセットとエイリアスの整理（#276）

前は作る・使うだけで、消す・名前を変える・分類を変える道が画面に無かった
（``PresetStore.delete`` と ``AliasStore.delete`` はどこからも呼ばれていなかった）
ここではファイルの扱い（コアの :class:`Library`）を見る 窓は tests/ui/test_library_dialog.py
"""

from __future__ import annotations

import json
import os
import re
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
        # ファイル名が古いままだと、一覧では「青」なのに「赤」で保存すると上書きの確かめが
        # 出る 保存した日が変わると改名しただけで並びが入れ替わる
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
        # 同じ名前への改名を黙って通すと、向こうのプリセットが確かめも無く消え、
        # ごみ箱にも残らない
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
        # 名前や分類の / や .. をそのまま道に使うと、プリセットが置き場の外へ書かれ、
        # 一覧から見えなくなる
        library.presets.save(Preset.capture("赤", _clip()))
        renamed = library.rename(_entry(library, "赤"), "a/b:c")
        moved = library.recategorize(renamed, "../外")
        # 分類も名前もファイル名に使えない文字を外す 外さないと置き場の外へ書く
        assert moved.path.is_relative_to(tmp_path / "presets")
        assert TRASH_FOLDER not in moved.path.parts
        assert _names(library) == [("../外", "a/b:c")]

    def test_an_empty_name_is_refused(self, library: Library) -> None:
        # 空白だけの名前を通すと、一覧で名前の見えない項目ができ、どれか見分けられない
        library.presets.save(Preset.capture("赤", _clip()))
        with pytest.raises(LibraryError):
            library.rename(_entry(library, "赤"), "   ")

    def test_alias_names_differing_only_in_case_do_not_collide(self, library: Library) -> None:
        # Windows では大文字小文字だけ違う名前は同じファイル 別の名前と見ると、上書きの
        # 確かめが出るか、古い名前として消してエイリアスそのものを失う
        library.aliases.save(Alias.of("abc", _clip()))
        renamed = library.rename(_entry(library, "abc", ALIAS), "ABC")
        assert _names(library, ALIAS) == [(DEFAULT_CATEGORY, "ABC")]
        assert renamed.path.exists()


class TestCategory:
    def test_a_preset_moves_to_the_new_folder(self, library: Library) -> None:
        # ファイルが元の分類のフォルダに残ると、移した先で同じ名前を保存しても重なりに
        # 気付かず、同じ名前のプリセットが 2 つできる
        library.presets.save(Preset.capture("赤", _clip()))
        moved = library.recategorize(_entry(library, "赤"), "見出し")
        assert moved.path.parent.name == "見出し"
        assert _names(library) == [("見出し", "赤")]
        assert library.categories(PRESET) == ("見出し",)

    def test_the_same_name_in_the_target_category_is_not_crushed(self, library: Library) -> None:
        # 移す先の分類に同じ名前があると、黙って書くと向こうのプリセットが消える
        library.presets.save(Preset.capture("赤", _clip("こちら")))
        library.presets.save(Preset.capture("赤", _clip("向こう"), category="見出し"))
        mine = next(e for e in library.entries(PRESET) if e.category == "ユーザー")
        with pytest.raises(LibraryConflictError):
            library.recategorize(mine, "見出し")
        assert _names(library) == [("ユーザー", "赤"), ("見出し", "赤")]

        library.recategorize(mine, "見出し", overwrite=True)
        assert _names(library) == [("見出し", "赤")]
        assert [t.category for t in library.trashed(PRESET)] == ["見出し"]

    def test_an_alias_keeps_its_file_and_carries_the_category(self, library: Library) -> None:
        # エイリアスの一覧は置き場の直下しか見ない 分類でフォルダへ移すと一覧から消え、
        # 分類をファイルへ書かないと開き直すと元の分類に戻る
        saved = library.aliases.save(Alias.of("テロップ", _clip()))
        moved = library.recategorize(_entry(library, "テロップ", ALIAS), "字幕")
        assert moved.path == saved
        assert library.aliases.load(saved).category == "字幕"

    def test_an_old_alias_without_a_category_reads_as_the_default(self, tmp_path: Path) -> None:
        # 前の版で保存したエイリアスは分類を持たない 既定へまとめないと、管理の窓の
        # フォルダ分けと〔追加〕→〔エイリアス〕の並びに置き所が無くなる
        store = AliasStore(tmp_path)
        path = store.save(Alias.of("前の版", _clip()))
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["category"]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert store.load(path).category == DEFAULT_CATEGORY


class TestDuplicate:
    def test_the_original_stays(self, library: Library) -> None:
        # 複製で元が消えたり名前が変わったりすると、写しを作って手直しするつもりが元の見た目を失う
        library.presets.save(Preset.capture("赤", _clip()))
        copy = library.duplicate(_entry(library, "赤"), "赤 のコピー")
        assert _names(library) == [("ユーザー", "赤"), ("ユーザー", "赤 のコピー")]
        assert copy.item.name == "赤 のコピー"

    def test_a_taken_name_is_a_conflict(self, library: Library) -> None:
        # 写しの名前が既にあるのを黙って通すと、そのエイリアスが確かめも無く写しで上書きされる
        library.aliases.save(Alias.of("a", _clip()))
        library.aliases.save(Alias.of("b", _clip()))
        with pytest.raises(LibraryConflictError):
            library.duplicate(_entry(library, "a", ALIAS), "b")


class TestTrash:
    def test_deleted_items_leave_the_list_and_come_back(self, library: Library) -> None:
        # 消した物がメニューに残ったり、〔元に戻す〕で元の分類へ戻らなかったりすると、
        # 押し間違えて消したプリセットを取り戻せない
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
        # 戻す先に後から作った同じ名前があるのに黙って戻すと、新しく作った方を失う
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
        # 空にしてもファイルが残ると、ごみ箱を空にしたつもりで容量が減らず、空にした物が
        # 管理の窓に出続ける
        library.presets.save(Preset.capture("赤", _clip()))
        library.presets.save(Preset.capture("青", _clip()))
        library.delete(_entry(library, "赤"))
        library.delete(_entry(library, "青"))
        assert library.empty_trash(PRESET) == 2
        assert library.trashed(PRESET) == ()
        assert not any((library.presets.root / TRASH_FOLDER).iterdir())

    def test_a_trash_entry_gone_elsewhere_is_reported(self, library: Library) -> None:
        # ほかの窓でごみ箱を空にした後に〔元に戻す〕を押すと、「見つからない」と知らせて
        # 一覧を読み直す代わりに、無いファイルを読む例外で止まる
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

    def test_a_file_saved_again_elsewhere_is_not_overwritten_with_the_old_look(
        self, library: Library, tmp_path: Path
    ) -> None:
        # ほかの窓が同じ名前で保存し直した後に、古い一覧の項目から書き換えると、古い中身を
        # 書いて新しい中身を消していた（ファイルがあるだけで確かめを通っていた）
        library.presets.save(Preset.capture("赤", _clip("前の文字")))
        seen = _entry(library, "赤")
        library.presets.save(Preset.capture("赤", _clip("後の文字")))
        for action in (
            lambda: library.rename(seen, "青"),
            lambda: library.recategorize(seen, "別"),
            lambda: library.delete(seen),
            lambda: library.duplicate(seen, "写し"),
            lambda: library.export(seen, tmp_path / "out"),
        ):
            with pytest.raises(LibraryChangedError):
                action()
        (now,) = library.entries(PRESET)
        assert isinstance(now.item, Preset) and now.item.source is not None
        assert now.item.source.params["text"] == "後の文字"
        assert library.trashed(PRESET) == ()
        # 読み直した項目からなら変えられる
        assert library.rename(now, "青").name == "青"

    def test_a_hidden_old_copy_appearing_elsewhere_counts_as_a_change(
        self, library: Library
    ) -> None:
        # ほかの窓が置いた旧い拡張子（.kmkp）の写しに気付かずに消すと、写しが残り、
        # 消したはずのプリセットが一覧に戻ってくる
        library.presets.save(Preset.capture("赤", _clip()))
        seen = _entry(library, "赤")
        seen.path.with_suffix(".kmkp").write_text(
            seen.path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        with pytest.raises(LibraryChangedError):
            library.delete(seen)


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
        # Kumiki のころの .kmkp が一覧に出ないと、更新した人が前に保存したプリセットを
        # 整理できない 改名で旧い形のまま書くと新しい形へいつまでも移らない
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


class TestSave:
    """保存の窓（#276 の追記）の保存 上書きは明かに頼んだときだけ、前の内容はごみ箱へ"""

    def test_a_taken_name_is_refused_unless_replacing_is_given(self, library: Library) -> None:
        # 保存の窓で同じ名前を黙って上書きすると、前のプリセットが確かめも無く消え、
        # ごみ箱からも戻せない
        library.presets.save(Preset.capture("赤", _clip("前")))
        item = Preset.capture("赤", _clip("後"))
        with pytest.raises(LibraryConflictError):
            library.save(PRESET, item)
        shown = library.occupant(PRESET, item)
        assert shown is not None
        library.save(PRESET, item, replacing=shown)
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["text"] == "後"
        assert [t.name for t in library.trashed(PRESET)] == ["赤"]

    def test_an_alias_in_another_category_is_the_same_file(self, library: Library) -> None:
        # エイリアスは分類が違っても同じ名前なら同じファイル 上書きの印を出さないと、
        # 保存で「字幕」のテロップが黙って置き換わる
        library.aliases.save(Alias.of("テロップ", _clip(), category="字幕"))
        shown = library.occupant(ALIAS, Alias.of("テロップ", _clip(), category="別"))
        assert shown is not None and shown.category == "字幕"

    def test_a_save_elsewhere_after_showing_is_not_overwritten(self, library: Library) -> None:
        # 保存の窓を開いている間にほかの窓が同じ名前で保存し直すと、窓で見せた物と違う
        # 中身を、確かめた気で上書きしてしまう
        library.presets.save(Preset.capture("赤", _clip("前")))
        item = Preset.capture("赤", _clip("いま"))
        shown = library.occupant(PRESET, item)
        library.presets.save(Preset.capture("赤", _clip("ほかの窓")))
        with pytest.raises(LibraryChangedError):
            library.save(PRESET, item, replacing=shown)
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["text"] == "ほかの窓"


class TestUndoOverwrite:
    def test_the_old_one_comes_back_and_the_new_one_goes_to_the_trash(
        self, library: Library
    ) -> None:
        # 保存した知らせの〔元に戻す〕で前の中身が戻らないと、上書きの押し間違いを
        # 取り消せない 戻した後の新しい方もごみ箱から取り出せないと失う
        library.presets.save(Preset.capture("赤", _clip("前")))
        item = Preset.capture("赤", _clip("後"))
        saved, trashed = library.save_keeping(
            PRESET, item, replacing=library.occupant(PRESET, item)
        )
        assert trashed is not None
        library.undo_overwrite(saved, trashed)
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["text"] == "前"
        (moved,) = library.trashed(PRESET)
        library.restore(moved, overwrite=True)
        (again,) = library.presets.all()
        assert again.source is not None and again.source.params["text"] == "後"

    def test_a_change_elsewhere_is_kept(self, library: Library) -> None:
        # 上書きを戻すまでの間にほかの窓が保存し直していると、〔元に戻す〕がその新しい
        # 保存を消してしまう
        library.presets.save(Preset.capture("赤", _clip("前")))
        item = Preset.capture("赤", _clip("後"))
        saved, trashed = library.save_keeping(
            PRESET, item, replacing=library.occupant(PRESET, item)
        )
        assert trashed is not None
        library.presets.save(Preset.capture("赤", _clip("ほかの窓")))
        with pytest.raises(LibraryChangedError):
            library.undo_overwrite(saved, trashed)
        (now,) = library.presets.all()
        assert now.source is not None and now.source.params["text"] == "ほかの窓"


class TestExchange:
    def test_export_then_import_on_another_machine(self, library: Library, tmp_path: Path) -> None:
        # 書き出した形を別の機械で読めないと、プリセットとエイリアスを人へ渡したり新しい PC
        # へ移したりできず、分類も向こうで失う
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
        # 書き出し先に前のファイルがあるのに黙って上書きすると、前に書き出して手を入れた
        # ファイルが確かめも無く消える
        library.presets.save(Preset.capture("赤", _clip()))
        entry = _entry(library, "赤")
        library.export(entry, tmp_path / "out")
        with pytest.raises(LibraryConflictError):
            library.export(entry, tmp_path / "out")
        library.export(entry, tmp_path / "out", overwrite=True)

    def test_keeping_both_numbers_the_new_one(self, library: Library) -> None:
        # 読み込みで「両方残す」を選んだときに番号を付けないと、同じ名前に重なって
        # 読み込んだ物か手持ちの物のどちらかが消える
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
        # 読み込みで断らないと壊れたファイルが置き場へ足され、一覧では読めずに飛ばされるので、
        # 読み込めたと思った物がどこにも出ない
        path = tmp_path / name
        path.write_text(content, encoding="latin-1")
        with pytest.raises(LibraryError):
            library.read_file(path)


class TestFingerprint:
    def test_names_and_categories_do_not_change_the_picture(self) -> None:
        # 名前や分類で見本の絵の鍵が変わると、名前を変えるたびに見本の絵を描き直し、
        # 管理の窓の一覧が重くなる
        first = Preset.capture("赤", _clip())
        renamed = Preset.capture("青", _clip(), category="別")
        assert look_fingerprint(first) == look_fingerprint(renamed)

    def test_the_look_does(self) -> None:
        # 中身を変えても鍵が同じだと、見本の絵が直す前の古い見た目のまま出続ける
        assert look_fingerprint(Preset.capture("a", _clip("x"))) != look_fingerprint(
            Preset.capture("a", _clip("y"))
        )


def _filled(root: Path, count: int) -> Library:
    """プリセットとエイリアスを ``count`` 件ずつ入れた置き場"""
    library = Library(presets=PresetStore(root / "presets"), aliases=AliasStore(root / "aliases"))
    for number in range(count):
        library.presets.save(Preset.capture(f"見出し {number:03}", _clip()))
        library.aliases.save(Alias.of(f"テロップ {number:03}", _clip()))
    return library


def _listing_seconds(library: Library, count: int) -> float:
    started = time.perf_counter()
    assert len(library.entries(PRESET)) == count
    assert len(library.entries(ALIAS)) == count
    return time.perf_counter() - started


def _reading_seconds(files: list[Path]) -> float:
    """同じファイルを開いて JSON として読み、日時を見るだけの時間 一覧が必ずする仕事の下限"""
    started = time.perf_counter()
    for path in files:
        json.loads(path.read_text(encoding="utf-8"))
        path.stat()
    return time.perf_counter() - started


class TestMany:
    #: 一覧に掛かる時間が、同じファイルを読むだけの時間の何倍まで許すか
    #: 手元の実測は 2.5 倍前後 1 件ごとに置き場を探し直す作りにすると 38 倍になった
    #: 機械（計算と記憶装置の速さの釣り合い）や Python の版で比が動く分を見て 10 倍
    SLOWDOWN_LIMIT = 10.0

    #: 測る回数 それぞれの一番速い回を使う ほかの処理に割り込まれた回を外すため
    ROUNDS = 5

    def test_three_hundred_items_list_without_slowing_down(self, tmp_path: Path) -> None:
        """300 件ずつの一覧が、ファイルを読むだけの時間に比べて遅くならない

        壊れると、プリセットやエイリアスを貯めた人ほど管理の窓や右クリックのメニューを
        開くのがもたつき、300 件で何秒も待たされる
        前は「5 秒未満」の決まった秒で見ていて、ほかの作業で混んだ機械では、遅くする変更が
        無くても落ちた 同じ機械で同じ時に測った「読むだけ」の時間との比で見る 機械が
        混めば両方が同じように遅れるので、比は揺れにくい 件数を増やしたときの伸び方で
        見る形は、1 件ごとの決まった手間が増える遅れ（件数に比例する）を見逃したので使わない
        """
        count = 300
        library = _filled(tmp_path, count)
        files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
        assert len(files) == 2 * count
        # 交互に測る 片方を測る間だけ機械が混んでも、もう片方の一番速い回で比が崩れない
        reading = listing = float("inf")
        for _ in range(self.ROUNDS):
            reading = min(reading, _reading_seconds(files))
            listing = min(listing, _listing_seconds(library, count))
        slowdown = listing / reading
        assert slowdown < self.SLOWDOWN_LIMIT, (
            f"読むだけで {reading:.3f} 秒 一覧で {listing:.3f} 秒（{slowdown:.1f} 倍）"
        )


class TestAtomicWrites:
    """書き出しとごみ箱の覚え書きは、書き手ごとの仮の名前で書いてから置き換える（#310 の決まり）

    決まった名前（``<名前>.writing``）だと、2 つの窓が同じ所へ同時に書いたときに相手の
    書きかけを上書きする 失敗しても元のファイルは無傷で、仮のファイルも残らない
    """

    @pytest.fixture
    def seen(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        from sashimono.core.io import atomic

        names: list[str] = []
        real = atomic._replace

        def spy(temporary: Path, path: Path) -> None:
            names.append(temporary.name)
            real(temporary, path)

        monkeypatch.setattr(atomic, "_replace", spy)
        return names

    def test_export_uses_a_name_per_writer(
        self, library: Library, tmp_path: Path, seen: list[str]
    ) -> None:
        # 決まった仮の名前で書くと、2 つの窓が同じ所へ同時に書き出したときに相手の書きかけを
        # 上書きし、壊れたファイルが残る
        library.presets.save(Preset.capture("赤", _clip()))
        seen.clear()
        library.export(_entry(library, "赤"), tmp_path / "out")
        assert [name for name in seen if name.startswith("赤.smep.")]
        assert all(re.fullmatch(r".+\.[0-9a-f]{8}\.writing", name) for name in seen)

    def test_the_trash_note_uses_a_name_per_writer(self, library: Library, seen: list[str]) -> None:
        # 覚え書きを仮の名前を通さずに書くと、書く途中で落ちたときに読めない覚え書きが残り、
        # 移した物がごみ箱の一覧から消えて戻せない
        library.presets.save(Preset.capture("赤", _clip()))
        seen.clear()
        library.delete(_entry(library, "赤"))
        assert any(re.fullmatch(r"entry\.json\.[0-9a-f]{8}\.writing", name) for name in seen)

    def test_a_failed_export_leaves_the_old_file_whole(
        self, library: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 書けなかった書き出しが前の書き出しを先に壊すと、容量不足などの後に手元の前の
        # ファイルも失い、書きかけの .writing も残る
        from sashimono.core.io import atomic

        library.presets.save(Preset.capture("赤", _clip()))
        out = tmp_path / "out"
        (out / "ユーザー").mkdir(parents=True)
        (out / "ユーザー" / "赤.smep").write_text("前の書き出し", encoding="utf-8")

        def refuse(_temporary: Path, _path: Path) -> None:
            raise OSError("試験で置き換えを断る")

        monkeypatch.setattr(atomic, "_replace", refuse)
        with pytest.raises(OSError):
            library.export(_entry(library, "赤"), out, overwrite=True)
        assert (out / "ユーザー" / "赤.smep").read_text(encoding="utf-8") == "前の書き出し"
        assert not list(out.rglob("*.writing"))


class TestFailedWritesKeepTheOldOne:
    """上書きの保存で書けなかったとき（容量不足など）、使っていた物を一覧から消さない

    前は書く前に古いファイルをごみ箱へ移していて、書くのに失敗すると、失敗の知らせが出たのに
    使っていたプリセットやエイリアスが一覧とメニューから消えた
    """

    @pytest.fixture
    def refusing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(_self: Library, _item: object) -> Path:
            raise OSError("試験で書き込みを断る（容量不足のつもり）")

        monkeypatch.setattr(Library, "_save", refuse)

    def _text(self, library: Library, kind: LibraryKind = PRESET) -> str:
        (entry,) = library.entries(kind)
        item = entry.item
        source = item.source if isinstance(item, Preset) else item.clip.source
        assert source is not None
        return str(source.params["text"])

    def test_saving_over(self, library: Library, refusing: None) -> None:
        # 書けなかった保存の前に古い物をごみ箱へ移したままだと、失敗の知らせが出たのに
        # 使っていたプリセットが一覧とメニューから消える
        library.presets.save(Preset.capture("赤", _clip("前")))
        item = Preset.capture("赤", _clip("後"))
        with pytest.raises(OSError):
            library.save(PRESET, item, replacing=library.occupant(PRESET, item))
        assert self._text(library) == "前"
        assert library.trashed(PRESET) == ()

    def test_importing_over(self, library: Library, refusing: None) -> None:
        # 読み込みの上書きに失敗したときに戻さないと、手持ちのエイリアスが
        # 〔追加〕→〔エイリアス〕から消え、読み込んだ物も入らない
        library.aliases.save(Alias.of("テロップ", _clip("前")))
        with pytest.raises(OSError):
            library.add(ALIAS, Alias.of("テロップ", _clip("後")), overwrite=True)
        assert self._text(library, ALIAS) == "前"
        assert library.trashed(ALIAS) == ()

    def test_renaming_over(self, library: Library, refusing: None) -> None:
        # 改名の上書きに失敗したときに戻さないと、改名先にあった「青」が一覧から消え、
        # ごみ箱を探さないと見つからない
        library.presets.save(Preset.capture("赤", _clip("赤の中身")))
        library.presets.save(Preset.capture("青", _clip("青の中身")))
        with pytest.raises(OSError):
            library.rename(_entry(library, "赤"), "青", overwrite=True)
        assert {e.name for e in library.entries(PRESET)} == {"赤", "青"}
        blue = _entry(library, "青").item
        assert isinstance(blue, Preset) and blue.source is not None
        assert blue.source.params["text"] == "青の中身"
        assert library.trashed(PRESET) == ()


class TestExportAcrossCategories:
    def test_the_same_name_in_two_categories_keeps_both(
        self, library: Library, tmp_path: Path
    ) -> None:
        # 前は同じファイル名に重なり、上書きを選ぶと「2 件書き出した」と出て 1 件しか残らなかった
        library.presets.save(Preset.capture("赤", _clip("見出しの赤"), category="見出し"))
        library.presets.save(Preset.capture("赤", _clip("字幕の赤"), category="字幕"))
        out = tmp_path / "out"
        written = [library.export(entry, out) for entry in library.entries(PRESET)]
        assert len({path.resolve() for path in written}) == 2
        other = Library(
            presets=PresetStore(tmp_path / "other" / "presets"),
            aliases=AliasStore(tmp_path / "other" / "aliases"),
        )
        for path in written:
            other.add(*other.read_file(path))
        # 読み込むと分類も戻る（分類はファイルの中にある）
        found = set()
        for entry in other.entries(PRESET):
            assert isinstance(entry.item, Preset) and entry.item.source is not None
            found.add((entry.category, str(entry.item.source.params["text"])))
        assert found == {("見出し", "見出しの赤"), ("字幕", "字幕の赤")}
