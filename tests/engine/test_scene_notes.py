"""場面の説明（#326）を素材の解析と同じ置き場（engine/cache）に覚える

説明は AI が絵を見て作った文で、作品の外に置く 消しても作り直せる 素材の場所・大きさ・
更新時刻の鍵で引くので、別の作品に同じ素材を読み込んでも同じ説明を引ける
"""

from __future__ import annotations

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
