"""場面の説明（#326）を素材の解析と同じ置き場（engine/cache）に覚える

説明は AI が絵を見て作った文で、作品の外に置く 消しても作り直せる 素材の場所・大きさ・
更新時刻の鍵で引くので、別の作品に同じ素材を読み込んでも同じ説明を引ける
"""

from __future__ import annotations

import json
import threading
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.model import MediaItem
from sashimono.core.search import SceneNote
from sashimono.engine.cache import CacheStore
from sashimono.engine.cache.scene_notes import NAMESPACE, SUFFIX, SceneNoteStore, notes_key


@pytest.fixture
def store(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache")


@pytest.fixture
def media(tmp_path: Path) -> MediaItem:
    path = tmp_path / "素材.mp4"
    path.write_bytes(b"x" * 100)
    return MediaItem(path=path, duration=Fraction(60))


NOTES = (
    SceneNote(start=0.0, end=10.0, text="海辺を歩く"),
    SceneNote(start=10.0, end=25.5, text="夜の屋台でラーメンをすする"),
)


class TestSceneNoteStore:
    def test_notes_survive_a_restart(self, store: CacheStore, media: MediaItem) -> None:
        # 壊れると、アプリを開き直すたびに説明が消え、絵を送って作り直す費用が毎回かかる
        SceneNoteStore(store).remember(media, NOTES)
        assert SceneNoteStore(store).notes(media) == NOTES

    def test_the_same_file_in_another_project_shares_the_notes(
        self, store: CacheStore, media: MediaItem
    ) -> None:
        # 壊れると、同じ素材を別の作品に読み込んだとき（素材の ID が違う）に説明が引けない
        SceneNoteStore(store).remember(media, NOTES)
        again = MediaItem(path=media.path, duration=media.duration)
        assert again.id != media.id
        assert SceneNoteStore(store).notes(again) == NOTES

    def test_a_replaced_file_forgets_the_notes(self, store: CacheStore, media: MediaItem) -> None:
        # 壊れると、素材を撮り直して差し替えた後も前の中身の説明が当たり、無い場面へ案内する
        notes = SceneNoteStore(store)
        notes.remember(media, NOTES)
        media.path.write_bytes(b"y" * 300)
        assert notes.notes(media) == ()

    def test_a_broken_file_reads_as_no_notes(self, store: CacheStore, media: MediaItem) -> None:
        # 壊れると、書きかけで落ちた控えを読んだときに例外になり、検索の道具ごと動かなくなる
        path = store.prepare(NAMESPACE, notes_key(media.path), SUFFIX)
        path.write_text("{壊れた", encoding="utf-8")
        assert SceneNoteStore(store).notes(media) == ()

    def test_notes_written_while_reading_are_kept(
        self, store: CacheStore, media: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、探す窓の裏のスレッドが控えを読んでいる間に AI が説明を書くと、読み終えた
        # 古い中身（空）が書いたばかりの説明を上書きし、同じ起動の中で引けなくなる
        from sashimono.engine.cache import scene_notes as module

        notes = SceneNoteStore(store)
        reading = threading.Event()
        written = threading.Event()
        original = module._load

        def slow(path: Path) -> tuple[SceneNote, ...]:
            loaded = original(path)
            reading.set()
            assert written.wait(5)
            return loaded

        monkeypatch.setattr(module, "_load", slow)
        seen: list[tuple[SceneNote, ...]] = []
        reader = threading.Thread(target=lambda: seen.append(notes.notes(media)))
        reader.start()
        assert reading.wait(5)
        notes.remember(media, NOTES)
        written.set()
        reader.join(5)
        assert seen == [NOTES]
        assert notes.notes(media) == NOTES

    def test_an_unreadable_place_reads_as_no_notes(
        self, store: CacheStore, media: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、控えの置き場を読めない（権限が無い）ときに有るかを見る所で OSError が出て、
        # search_media が控えの場所入りの例外で落ち、字幕も引けなくなる
        written = store.path_for(NAMESPACE, notes_key(media.path), SUFFIX)
        original = Path.exists

        def denied(path: Path, *args: object, **kwargs: object) -> bool:
            if path == written:
                raise PermissionError(13, "Access is denied", str(path))
            return original(path)

        monkeypatch.setattr(Path, "exists", denied)
        assert SceneNoteStore(store).notes(media) == ()

    def test_notes_written_over_a_broken_file_while_reading_survive_a_restart(
        self, store: CacheStore, media: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、壊れた控えを読んでいる間に AI が正しい説明へ置き換えたとき、読み終えた側が
        # 壊れていたとして新しい控えを消し、立ち上げ直すと書いたはずの説明が無くなる
        path = store.prepare(NAMESPACE, notes_key(media.path), SUFFIX)
        path.write_text("{壊れた", encoding="utf-8")
        notes = SceneNoteStore(store)
        reading = threading.Event()
        written = threading.Event()
        original = json.loads

        def slow(text: str) -> object:
            if text.startswith("{壊れた"):
                reading.set()
                assert written.wait(5)
            return original(text)

        monkeypatch.setattr(json, "loads", slow)
        seen: list[tuple[SceneNote, ...]] = []
        reader = threading.Thread(target=lambda: seen.append(notes.notes(media)))
        reader.start()
        assert reading.wait(5)
        notes.remember(media, NOTES)
        written.set()
        reader.join(5)
        assert seen == [NOTES]
        assert SceneNoteStore(store).notes(media) == NOTES

    def test_a_broken_file_that_cannot_be_removed_reads_as_no_notes(
        self, store: CacheStore, media: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、壊れた控えを消せない（別のプログラムが掴んでいる）ときに消す所で OSError が
        # 出て、壊れた控え 1 つで検索の道具ごと動かなくなる
        path = store.prepare(NAMESPACE, notes_key(media.path), SUFFIX)
        path.write_text("{壊れた", encoding="utf-8")

        def held(_path: Path, missing_ok: bool = False) -> None:
            raise PermissionError(13, "in use", str(_path))

        monkeypatch.setattr(Path, "unlink", held)
        assert SceneNoteStore(store).notes(media) == ()

    def test_forgetting_all_notes_removes_the_file(
        self, store: CacheStore, media: MediaItem
    ) -> None:
        # 壊れると、説明を全部消すよう頼んでも空の控えが残り続ける（消したつもりの文が残る）
        notes = SceneNoteStore(store)
        notes.remember(media, NOTES)
        notes.remember(media, ())
        assert not store.exists(NAMESPACE, notes_key(media.path), SUFFIX)
        assert notes.notes(media) == ()

    def test_no_name_of_the_media_is_written(self, store: CacheStore, media: MediaItem) -> None:
        # 壊れると、作品の外の控えに素材の名前が残り、名前を送らないと決めた人の伏せ字の表
        # （EditorHost.remembered_names）から漏れる 鍵は名前を読めない形にしてある
        SceneNoteStore(store).remember(media, NOTES)
        written = store.path_for(NAMESPACE, notes_key(media.path), SUFFIX)
        assert media.path.stem not in written.read_text(encoding="utf-8")
        assert media.path.stem not in str(written)
