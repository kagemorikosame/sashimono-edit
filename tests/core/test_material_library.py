"""登録したフォルダの素材（#520 #521） 登録・外し・見て回る・名前で絞る・測った物を覚える

エクスプローラーの画面は tests/ui/test_explorer.py、AI の道具は tests/ai/test_library_tools.py
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from sashimono.core import material_library
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
    is_favorite,
    list_folder,
    match_files,
    parse_categories,
    parse_folders,
    parse_paths,
    remove_folder,
    same_place,
    scan_folder,
    set_category,
    toggle_favorite,
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

    def test_scan_returns_all_4000_files_without_truncation(self, tmp_path: Path) -> None:
        # 市販の効果音集は数千のファイルを持つ 一覧を開くたびに止まると使えない
        for number in range(4000):
            (tmp_path / f"音{number:04}.wav").write_bytes(b"")
        # 秒は測らない（CI の書き込みの混み具合で落ちる） 速さは 1 段ずつの scandir と、
        # 画面では裏で読む作りで受け持つ ここは数千でも全部を拾うことだけを見る
        scan = scan_folder(LibraryFolder(str(tmp_path), CATEGORY_SOUND_EFFECT))
        assert len(scan.files) == 4000 and not scan.truncated

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
        # 書き方の違う場所（.. を含む）でも同じ印になる 同じ書き方を比べても正規化は確かめられない
        assert file_id(path) == file_id(tmp_path / "sub" / ".." / "ドン.wav")
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


class TestFavorites:
    def test_toggle_keeps_order_and_ignores_spelling(self, tmp_path: Path) -> None:
        # 壊れると、同じ物が書き方の違いで 2 つ並ぶか、外したつもりで残る
        a, b = str(tmp_path / "a.wav"), str(tmp_path / "b")
        favorites = toggle_favorite((), a, True)
        favorites = toggle_favorite(favorites, b, True)
        favorites = toggle_favorite(favorites, a.upper() if os.name == "nt" else a, True)
        assert favorites == (b, a.upper() if os.name == "nt" else a)
        assert toggle_favorite(favorites, a, False) == (b,)

    def test_files_inside_a_favorite_folder_count(self, tmp_path: Path) -> None:
        # 壊れると、お気に入りのフォルダに入れた効果音を AI が「いつもの」として選べない
        favorites = (str(tmp_path / "いつもの"),)
        assert is_favorite(favorites, tmp_path / "いつもの" / "ドン.wav")
        assert not is_favorite(favorites, tmp_path / "いつもの2" / "ドン.wav")
        assert same_place(tmp_path / "x", str(tmp_path / "." / "x"))

    def test_broken_settings_are_dropped(self) -> None:
        # 壊れた値で起動を止めない
        assert parse_paths("D:/x") == ()
        assert parse_paths(["D:/x", "", 3, "D:/x"]) == ("D:/x",)


class TestReviewFixes:
    def test_broken_rows_do_not_stop_the_start(self) -> None:
        # 壊れると、手で書き換えた設定の 1 行（分類が配列など）で型の例外になり起動が止まる
        assert parse_folders([{"path": "D:/SE", "category": []}]) == ()
        assert parse_categories(["music", {}], ()) == (CATEGORY_MUSIC,)
        assert parse_categories([], DEFAULT_AI_CATEGORIES) == ()

    def test_a_nested_registration_is_its_own_boundary(self, tmp_path: Path) -> None:
        # 壊れると、親を効果音・子を動画で登録して効果音だけを AI に許したのに、子の音を
        # 親の効果音として AI が選ぶ
        _touch(tmp_path / "SE" / "ドン.wav")
        _touch(tmp_path / "SE" / "動画" / "隠し.wav")
        _touch(tmp_path / "SE" / "下" / "見える.wav")
        library = MaterialLibrary(
            folders=(
                LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT),
                LibraryFolder(str(tmp_path / "SE" / "動画"), CATEGORY_VIDEO),
            )
        )
        names = sorted(file.name for scan in library.scan_for_ai() for file in scan.files)
        assert names == ["ドン.wav", "見える.wav"]
        assert library.find_for_ai(file_id(tmp_path / "SE" / "動画" / "隠し.wav")) is None

    def test_a_broken_cache_row_is_measured_again(self, tmp_path: Path) -> None:
        # 壊れると、控えの 1 行（1e999 など）で一覧を作る途中に例外が出て一覧が止まる
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        file = scan_folder(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT)).files[0]
        path = tmp_path / "measures.json"
        row = {"size": file.size, "mtime": file.modified_ns, "seconds": 1.0, "width": 1e999}
        path.write_text(json.dumps({"version": 2, "files": {file.id: row}}), encoding="utf-8")
        assert MeasureBook(path).get(file) is None

    def test_a_file_that_cannot_be_measured_is_not_tried_again(self, tmp_path: Path) -> None:
        # 壊れると、読めないファイルが頭に並んだときに、呼び直すたびに同じ物で測る数を使い切る
        sound = _touch(tmp_path / "SE" / "壊れた.wav")
        calls: list[Path] = []

        def measure(path: Path) -> Measure | None:
            calls.append(path)
            return None

        library = MaterialLibrary(
            folders=(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),), measure=measure
        )
        file = library.scan_for_ai()[0].files[0]
        assert library.measured(file, allow_measure=True) is None
        assert library.tried(file)
        assert library.measured(file, allow_measure=True) is None
        assert calls == [sound]

    def test_huge_integers_in_the_cache_do_not_raise(self, tmp_path: Path) -> None:
        # 壊れると、控えの桁違いの整数（10**400）を float にする所で例外になり一覧が止まる
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        file = scan_folder(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT)).files[0]
        path = tmp_path / "measures.json"
        huge = "1" + "0" * 400
        text = (
            f'{{"version": 2, "files": {{"{file.id}": {{"size": {file.size},'
            f' "mtime": {file.modified_ns}, "seconds": 1.0, "width": {huge}}}}}}}'
        )
        path.write_text(text, encoding="utf-8")
        assert MeasureBook(path).get(file) is None

    def test_a_failure_is_forgotten_after_a_while(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、一時だけ応えなかったドライブの素材を、挿し直しても測り直さない
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        file = scan_folder(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT)).files[0]
        book = MeasureBook(tmp_path / "measures.json")
        book.put_failure(file)
        assert book.failed(file)
        book.save()
        # 控えのファイルには書かない（起動し直せば測り直す）
        assert not MeasureBook(tmp_path / "measures.json").failed(file)
        monkeypatch.setattr(material_library, "FAILURE_SECONDS", 0)
        assert not book.failed(file)

    def test_a_link_into_another_registration_follows_that_registration(
        self, tmp_path: Path
    ) -> None:
        # 壊れると、別の登録（AI に使わせない分類）を指すリンクを親の分類の素材として選ぶ
        hidden = _touch(tmp_path / "SE" / "動画" / "隠し.wav")
        link = tmp_path / "SE" / "リンク.wav"
        try:
            link.symlink_to(hidden)
        except OSError:
            pytest.skip("この機械ではリンクを作れない（Windows の開発者モードが要る）")
        library = MaterialLibrary(
            folders=(
                LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT),
                LibraryFolder(str(tmp_path / "SE" / "動画"), CATEGORY_VIDEO),
            )
        )
        names = [file.name for scan in library.scan_for_ai() for file in scan.files]
        assert "リンク.wav" not in names

    def test_a_junction_is_not_followed(self, tmp_path: Path) -> None:
        # 壊れると、登録の下のジャンクションを辿って、登録の外の普通のファイルを AI が選ぶ
        if os.name != "nt":
            pytest.skip("ジャンクションは Windows だけ")
        import _winapi

        outside = _touch(tmp_path / "外" / "外の音.wav")
        _touch(tmp_path / "SE" / "ドン.wav")
        _winapi.CreateJunction(str(outside.parent), str(tmp_path / "SE" / "抜け道"))
        scan = scan_folder(LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT))
        assert [file.name for file in scan.files] == ["ドン.wav"]
        listing = list_folder(
            LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT), tmp_path / "SE"
        )
        assert listing.folders == ()

    def test_names_are_matched_before_the_limit(self, tmp_path: Path) -> None:
        # 壊れると、上限より後ろにしか無い名前は、名前を詳しくしても見つからない
        for number in range(12):
            (tmp_path / f"{number:02}.wav").write_bytes(b"")
        (tmp_path / "zz目当て.wav").write_bytes(b"")
        scan = scan_folder(
            LibraryFolder(str(tmp_path), CATEGORY_SOUND_EFFECT),
            limit=5,
            accept=lambda file: "目当て" in file.name,
        )
        assert [file.name for file in scan.files] == ["zz目当て.wav"]

    def test_a_link_owned_by_an_allowed_child_is_kept(self, tmp_path: Path) -> None:
        # 壊れると、効果音として許した子の登録の中のリンクを、親の登録の中だからと捨てる
        real = _touch(tmp_path / "動画" / "SE" / "ドン.wav")
        link = tmp_path / "動画" / "SE" / "リンク.wav"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("この機械ではリンクを作れない（Windows の開発者モードが要る）")
        library = MaterialLibrary(
            folders=(
                LibraryFolder(str(tmp_path / "動画"), CATEGORY_VIDEO),
                LibraryFolder(str(tmp_path / "動画" / "SE"), CATEGORY_SOUND_EFFECT),
            )
        )
        names = sorted(file.name for scan in library.scan_for_ai() for file in scan.files)
        assert names == ["ドン.wav", "リンク.wav"]

    def test_a_sound_that_could_not_be_read_is_not_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、一時だけ読めなかった音の「大きさ無し」を控えに書き、直っても測り直さない
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        file = scan_folder(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT)).files[0]
        book = MeasureBook(tmp_path / "measures.json")
        partial = Measure(seconds=1.0, has_audio=True, has_video=False, audio_failed=True)
        book.put(file, partial)
        assert book.get(file) == partial
        # 別の素材の完全な物と一緒に書いても、一時の物は書かれない（書く道を通して確かめる）
        other = _touch(tmp_path / "SE" / "ピコン.wav")
        whole_file = next(
            f
            for f in scan_folder(LibraryFolder(str(other.parent), CATEGORY_SOUND_EFFECT)).files
            if f.path == other
        )
        whole = Measure(seconds=0.4, has_audio=True, has_video=False, loudness=-12.0)
        book.put(whole_file, whole)
        book.save()
        again = MeasureBook(tmp_path / "measures.json")
        assert again.get(file) is None
        assert again.get(whole_file) == whole
        monkeypatch.setattr(material_library, "FAILURE_SECONDS", 0)
        assert book.get(file) is None

    def test_a_whole_measure_wins_over_a_partial_one(self, tmp_path: Path) -> None:
        # 壊れると、片方だけ音を読めなかったときに、完全に測れた物があっても大きさ無しを返し続ける
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        file = scan_folder(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT)).files[0]
        book = MeasureBook()
        whole = Measure(seconds=1.0, has_audio=True, has_video=False, loudness=-14.0)
        book.put(file, whole)
        book.put(file, Measure(seconds=1.0, has_audio=True, has_video=False, audio_failed=True))
        assert book.get(file) == whole

    def test_a_looping_link_in_another_registration_does_not_stop_the_rest(
        self, tmp_path: Path
    ) -> None:
        # 壊れると、別の登録の壊れた循環リンク 1 つで、ほかの登録まで AI が見て回れない
        _touch(tmp_path / "SE" / "ドン.wav")
        loop = tmp_path / "SE" / "輪"
        try:
            loop.symlink_to(loop, target_is_directory=True)
        except OSError:
            pytest.skip("この機械ではリンクを作れない（Windows の開発者モードが要る）")
        library = MaterialLibrary(
            folders=(
                LibraryFolder(str(tmp_path / "SE"), CATEGORY_SOUND_EFFECT),
                LibraryFolder(str(loop), CATEGORY_VIDEO),
            )
        )
        names = [file.name for scan in library.scan_for_ai() for file in scan.files]
        assert names == ["ドン.wav"]

    def test_a_remembered_file_is_found_without_scanning_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、置く道具のたびに登録したフォルダを全部見て回り直し、大きな登録先で画面が止まる
        # 本人がその分類を使わせない設定にした後は、覚えていても返さない
        sound = _touch(tmp_path / "SE" / "ドン.wav")
        library = MaterialLibrary(
            folders=(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),)
        )
        library.scan_for_ai()

        def no_scan(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("見て回り直した")

        monkeypatch.setattr(material_library, "scan_folder", no_scan)
        found = library.find_for_ai(file_id(sound))
        assert found is not None and found.path == sound
        closed = MaterialLibrary(folders=library.folders, ai_categories=(), book=library.book)
        assert closed.find_for_ai(file_id(sound)) is None

    def test_a_file_replaced_while_measuring_is_not_kept(self, tmp_path: Path) -> None:
        # 壊れると、測っている間に差し替えた長いファイルの長さを、前の短いファイルの物として覚える
        sound = _touch(tmp_path / "SE" / "ドン.wav")

        def measure(path: Path) -> Measure:
            path.write_bytes(b"longer now")
            return Measure(seconds=95.0, has_audio=True, has_video=False)

        library = MaterialLibrary(
            folders=(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),), measure=measure
        )
        file = library.scan_for_ai()[0].files[0]
        assert library.measured(file, allow_measure=True) is None
        assert library.book.get(file) is None
        assert not library.tried(file)

    def test_a_kind_filter_is_applied_before_the_limit(self, tmp_path: Path) -> None:
        # 壊れると、音が上限まで並んだ「その他」のフォルダで動画を選ぶと 0 個になる
        for number in range(6):
            (tmp_path / f"{number:02}.wav").write_bytes(b"")
        (tmp_path / "zz.mp4").write_bytes(b"")
        listing = list_folder(
            LibraryFolder(str(tmp_path), CATEGORY_OTHER),
            tmp_path,
            limit=3,
            accept=lambda file: file.path.suffix == ".mp4",
        )
        assert [file.name for file in listing.files] == ["zz.mp4"]


def test_an_unknown_id_is_not_scanned_for_when_asked_not_to(tmp_path: Path) -> None:
    # 壊れると、画面のスレッドの道具が知らない印を渡されるたびに登録先を全部見て回る
    sound = _touch(tmp_path / "SE" / "ドン.wav")
    library = MaterialLibrary(folders=(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),))
    assert library.find_for_ai(file_id(sound), scan=False) is None
    assert library.find_for_ai(file_id(sound)) is not None
    # 一度見つけた印は見て回らずに引ける
    assert library.find_for_ai(file_id(sound), scan=False) is not None


def test_reading_the_book_ahead_does_not_hold_up_saving_or_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 壊れると、初めての先読みの間に窓を閉じると終わりの保存が待たされ、先読みが終わったときに
    # その間に測った値を古い控えで上書きする
    sound = _touch(tmp_path / "SE" / "ドン.wav")
    store = tmp_path / "measures.json"
    store.write_text(json.dumps({"version": 1, "files": {}}), encoding="utf-8")
    book = MeasureBook(path=store)
    gate = threading.Event()
    reading = threading.Event()
    original = material_library._read_book
    calls: list[int] = []

    def slow(path: Path | None) -> dict[str, dict[str, object]]:
        calls.append(1)
        if len(calls) == 1:
            reading.set()
            gate.wait()
        return original(path)

    monkeypatch.setattr(material_library, "_read_book", slow)
    worker = threading.Thread(target=book.preload)
    worker.start()
    try:
        assert reading.wait(5)
        book.save()  # 何も測っていないので、先読みを待たずに戻る
        file = (
            MaterialLibrary(folders=(LibraryFolder(str(sound.parent), CATEGORY_SOUND_EFFECT),))
            .scan_for_ai()[0]
            .files[0]
        )
        measure = Measure(seconds=0.8, has_audio=True, has_video=False)
        book.put(file, measure)
    finally:
        gate.set()
        worker.join(5)
    assert book.get(file) == measure
