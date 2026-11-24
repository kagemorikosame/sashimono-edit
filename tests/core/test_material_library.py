"""登録したフォルダの素材（#520 #521） 登録・外し・見て回る・名前で絞る・測った物を覚える

エクスプローラーの画面は tests/ui/test_explorer.py、AI の道具は tests/ai/test_library_tools.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sashimono.core.material_library import (
    CATEGORY_IMAGE,
    CATEGORY_MUSIC,
    CATEGORY_OTHER,
    CATEGORY_SOUND_EFFECT,
    CATEGORY_VIDEO,
    DEFAULT_AI_CATEGORIES,
    LibraryFolder,
    MaterialLibrary,
    Measure,
    MeasureBook,
    add_folder,
    file_id,
    match_files,
    parse_categories,
    parse_folders,
    remove_folder,
    scan_folder,
    set_category,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


class TestRegistration:
    def test_the_same_folder_is_registered_once(self, tmp_path: Path) -> None:
        # 2 つの分類で登録できると、AI に使わせないつもりで分類を変えても、もう片方から選ばれる
        folders = add_folder((), str(tmp_path), CATEGORY_SOUND_EFFECT)
        folders = add_folder(folders, str(tmp_path) + os.sep, CATEGORY_VIDEO)
        assert folders == (LibraryFolder(str(tmp_path) + os.sep, CATEGORY_VIDEO),)

    def test_removing_keeps_the_others_and_the_files(self, tmp_path: Path) -> None:
        # 壊れると、登録を外しただけで本人の素材のファイルやほかの登録が消える
        sound = _touch(tmp_path / "効果音" / "ドン.wav")
        folders = add_folder((), str(sound.parent), CATEGORY_SOUND_EFFECT)
        folders = add_folder(folders, str(tmp_path / "曲"), CATEGORY_MUSIC)
        left = remove_folder(folders, str(sound.parent))
        assert left == (LibraryFolder(str(tmp_path / "曲"), CATEGORY_MUSIC),)
        # 外すのは登録だけ ファイルを消すと本人の素材が無くなる
        assert sound.exists()

    def test_changing_the_category_keeps_the_order(self, tmp_path: Path) -> None:
        # 壊れると、分類を変えるたびに登録の並びが動き、画面の選択肢が入れ替わる
        folders = (
            LibraryFolder(str(tmp_path / "a"), CATEGORY_SOUND_EFFECT),
            LibraryFolder(str(tmp_path / "b"), CATEGORY_MUSIC),
        )
        changed = set_category(folders, str(tmp_path / "a"), CATEGORY_OTHER)
        assert [f.category for f in changed] == [CATEGORY_OTHER, CATEGORY_MUSIC]

    def test_unknown_categories_are_refused(self, tmp_path: Path) -> None:
        # 壊れると、知らない分類の登録が設定に残り、AI に使わせるかを選べない
        with pytest.raises(ValueError, match="分類"):
            add_folder((), str(tmp_path), "mood")
        with pytest.raises(ValueError, match="フォルダ"):
            add_folder((), " ", CATEGORY_MUSIC)

    def test_broken_settings_rows_are_skipped(self) -> None:
        # 壊れた行で起動を止めない 読めない行だけを落とす
        rows = [
            {"path": "D:/SE", "category": CATEGORY_SOUND_EFFECT},
            {"path": "D:/SE", "category": CATEGORY_MUSIC},
            {"path": "", "category": CATEGORY_MUSIC},
            {"path": "D:/x", "category": "mood"},
            "D:/y",
        ]
        assert parse_folders(rows) == (LibraryFolder("D:/SE", CATEGORY_SOUND_EFFECT),)
        assert parse_folders("D:/SE") == ()

    def test_ai_categories_fall_back_and_keep_the_table_order(self) -> None:
        # 壊れると、手で書き換えた設定で起動が止まるか、何も使わせないと決めた人の選びが既定へ戻る
        assert parse_categories(None, DEFAULT_AI_CATEGORIES) == DEFAULT_AI_CATEGORIES
        assert parse_categories(["image", "mood", "sound_effect"], ()) == (
            CATEGORY_SOUND_EFFECT,
            CATEGORY_IMAGE,
        )
        # 全部を外した人の選びを既定へ戻さない（AI に何も使わせないと決めた人）
        assert parse_categories([], DEFAULT_AI_CATEGORIES) == ()

    def test_the_default_lets_the_ai_use_only_sounds(self) -> None:
        # 2026-11-24 本人の決定 動画と画像は選んだ人だけ
        assert DEFAULT_AI_CATEGORIES == (CATEGORY_SOUND_EFFECT, CATEGORY_MUSIC)


class TestScan:
    def test_finds_files_of_the_category_in_subfolders(self, tmp_path: Path) -> None:
        # 説明の画像や規約の文を効果音として出すと、AI が音でない物を選ぶ
        _touch(tmp_path / "ドン.WAV")
        _touch(tmp_path / "爆発" / "大.mp3")
        _touch(tmp_path / "説明.png")
        _touch(tmp_path / "規約.txt")
        scan = scan_folder(LibraryFolder(str(tmp_path), CATEGORY_SOUND_EFFECT))
        assert not scan.missing
        assert sorted(file.relative for file in scan.files) == sorted(
            ["ドン.WAV", str(Path("爆発") / "大.mp3")]
        )
        images = scan_folder(LibraryFolder(str(tmp_path), CATEGORY_IMAGE))
        assert [file.name for file in images.files] == ["説明.png"]

    def test_a_vanished_folder_is_reported_not_raised(self, tmp_path: Path) -> None:
        # 壊れると、外付けのドライブを外しただけでエクスプローラーも AI の道具も落ちる
        scan = scan_folder(LibraryFolder(str(tmp_path / "無い"), CATEGORY_MUSIC))
        assert scan.missing and scan.files == ()

    def test_thousands_of_files_are_quick(self, tmp_path: Path) -> None:
        # 市販の効果音集は数千のファイルを持つ 一覧を開くたびに止まると使えない
        for number in range(4000):
            (tmp_path / f"音{number:04}.wav").write_bytes(b"")
        started = time.perf_counter()
        scan = scan_folder(LibraryFolder(str(tmp_path), CATEGORY_SOUND_EFFECT))
        elapsed = time.perf_counter() - started
        assert len(scan.files) == 4000
        # 手元では 0.1 秒ほど CI の遅い機械と、並んで走る試験の分の余裕を見る
        assert elapsed < 5

    def test_a_huge_folder_is_cut_at_the_limit(self, tmp_path: Path) -> None:
        # 壊れると、ドライブの頭を登録しただけで一覧を作るたびに画面が止まる
        for number in range(12):
            (tmp_path / f"{number:02}.wav").write_bytes(b"")
        scan = scan_folder(LibraryFolder(str(tmp_path), CATEGORY_SOUND_EFFECT), limit=10)
        assert len(scan.files) == 10 and scan.truncated

    def test_names_and_folder_names_narrow_the_list(self, tmp_path: Path) -> None:
        # 壊れると、下のフォルダの名前や全角の英字の名前で絞れず、効果音を探せない
        _touch(tmp_path / "爆発" / "ドカン.wav")
        _touch(tmp_path / "拍手" / "ＡＰＰＬＡＵＳＥ.wav")
        _touch(tmp_path / "ドン.wav")
        files = scan_folder(LibraryFolder(str(tmp_path), CATEGORY_SOUND_EFFECT)).files
        assert [f.name for f in match_files(files, "爆発")] == ["ドカン.wav"]
        # 全角の英字の名前も半角の語で引ける 大文字と小文字も区別しない
        assert [f.name for f in match_files(files, "applause")] == ["ＡＰＰＬＡＵＳＥ.wav"]
        assert [f.name for f in match_files(files, "ドン ＷＡＶ")] == ["ドン.wav"]
        assert len(match_files(files, "  ")) == 3

    def test_the_id_does_not_depend_on_the_spelling(self, tmp_path: Path) -> None:
        # 壊れると、同じファイルが書き方で別の印になり、AI が選んだ物を指せない
        path = tmp_path / "ドン.wav"
        assert file_id(path) == file_id(tmp_path / "." / "ドン.wav")
        assert file_id(path).startswith("lib_")
        assert file_id(path) != file_id(tmp_path / "ドン2.wav")


class TestMeasureBook:
    def test_remembers_across_runs_and_forgets_changed_files(self, tmp_path: Path) -> None:
        # 覚えないと、数百の効果音から選ぶたびに測り直して待たせる 中身を差し替えた素材の
        # 古い長さを使い続けると、効果音のつもりで長い音を置く
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        folder = LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT)
        file = scan_folder(folder).files[0]
        book = MeasureBook(tmp_path / "cache" / "measures.json")
        measure = Measure(seconds=0.8, has_audio=True, has_video=False, loudness=-18.5, peak=-1.0)
        book.put(file, measure)
        book.save()
        again = MeasureBook(tmp_path / "cache" / "measures.json")
        assert again.get(file) == measure
        sound.write_bytes(b"longer")
        changed = scan_folder(folder).files[0]
        assert again.get(changed) is None

    def test_a_broken_cache_is_ignored(self, tmp_path: Path) -> None:
        # 壊れると、控えのファイルが壊れただけで AI の道具とエクスプローラーが落ちる
        path = tmp_path / "measures.json"
        path.write_text("{壊れた", encoding="utf-8")
        file = scan_folder(
            LibraryFolder(str(_touch(tmp_path / "a" / "x.wav").parent), CATEGORY_MUSIC)
        ).files[0]
        assert MeasureBook(path).get(file) is None


class TestMaterialLibrary:
    def test_the_ai_sees_only_the_chosen_categories(self, tmp_path: Path) -> None:
        # 壊れると、本人が AI に使わせないと決めた分類のフォルダから AI が素材を選ぶ
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        movie = _touch(tmp_path / "動画" / "空.mp4")
        library = MaterialLibrary(
            folders=(
                LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),
                LibraryFolder(str(movie.parent), CATEGORY_VIDEO),
            )
        )
        assert [f.category for f in library.ai_folders] == [CATEGORY_SOUND_EFFECT]
        assert library.find_for_ai(file_id(sound)) is not None
        assert library.find_for_ai(file_id(movie)) is None
        opened = MaterialLibrary(
            folders=library.folders, ai_categories=(CATEGORY_SOUND_EFFECT, CATEGORY_VIDEO)
        )
        assert opened.find_for_ai(file_id(movie)) is not None

    def test_measures_only_when_allowed(self, tmp_path: Path) -> None:
        # 壊れると、一覧を作るだけで全部を測って画面が止まるか、同じ素材を何度も測る
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        calls: list[Path] = []

        def measure(path: Path) -> Measure:
            calls.append(path)
            return Measure(seconds=1.0, has_audio=True, has_video=False)

        library = MaterialLibrary(
            folders=(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),), measure=measure
        )
        file = library.scan_for_ai()[0].files[0]
        assert library.measured(file, allow_measure=False) is None
        assert library.measured(file, allow_measure=True) is not None
        assert library.measured(file, allow_measure=True) is not None
        assert calls == [sound]
