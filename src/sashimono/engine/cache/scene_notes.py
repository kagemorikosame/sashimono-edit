"""場面の説明（#326）を素材の解析と同じ置き場に覚える

説明は AI が場面の絵を見て作った文（:class:`~sashimono.core.search.SceneNote`） 作品の外
（``engine/cache``）に置くのは 2026-10-20 に利用者が決めた既定の案 消しても作り直せ、
同じ素材を別の作品に読み込んでも同じ説明を引ける 鍵は素材の場所・大きさ・更新時刻
（:func:`~sashimono.engine.cache.store.media_key`）なので、素材を差し替えると前の中身の
説明は引かれない

素材の名前は書かない 鍵は名前を読めない形（ハッシュ）で、中身は時刻と説明だけ 作品の外に
名前を残すと、名前を送らないと決めた人の伏せ字の表から漏れる（#342）
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

from sashimono.core.model import MediaItem
from sashimono.core.search import SceneNote

from .store import CacheStore, media_key

__all__ = ["NAMESPACE", "SUFFIX", "SceneNoteStore", "notes_key"]

NAMESPACE = "scene_notes"
SUFFIX = ".notes.json"

#: 保存形式の版 中身の形を変えたら上げる 上げると前の説明は引かれなくなる（作り直す）
FORMAT_VERSION = 1


def notes_key(path: Path) -> str:
    """控えの鍵 素材の場所・大きさ・更新時刻と形式の版から作る"""
    return media_key(path, extra=f"sn{FORMAT_VERSION}")


class SceneNoteStore:
    """素材ごとの場面の説明 読んだ物は覚えておき、検索のたびにディスクを読まない"""

    def __init__(self, store: CacheStore | None = None) -> None:
        self._store = store if store is not None else CacheStore()
        self._lock = threading.Lock()
        #: 鍵 → 説明 鍵に更新時刻が入るので、素材が変われば別の鍵になって読み直す
        self._held: dict[str, tuple[SceneNote, ...]] = {}

    def notes(self, media: MediaItem) -> tuple[SceneNote, ...]:
        """覚えた説明 無い・壊れている・素材が変わったなら空"""
        key = notes_key(media.path)
        with self._lock:
            held = self._held.get(key)
        if held is not None:
            return held
        # 読むのは錠の外（探す窓の裏のスレッドから呼ばれ、読む間に AI の書き込みを待たせない）
        # 読む間に remember が書いた物があれば、そちらを残す 読んだ物で上書きすると、書く前の
        # 控えを読んだ古い中身で書いたばかりの説明を消し、同じ起動の中で引けなくなる
        loaded = _load(self._store.path_for(NAMESPACE, key, SUFFIX))
        with self._lock:
            return self._held.setdefault(key, loaded)

    def remember(self, media: MediaItem, notes: Sequence[SceneNote]) -> None:
        """素材の説明を丸ごと置き換える 空なら控えを消す

        空の控えを残さないのは、説明を全部消すよう頼まれた後に、消したはずの物が
        ディスクに残っているように見えないため
        """
        key = notes_key(media.path)
        kept = tuple(notes)
        path = self._store.path_for(NAMESPACE, key, SUFFIX)
        if kept:
            _save(self._store.prepare(NAMESPACE, key, SUFFIX), kept)
        else:
            path.unlink(missing_ok=True)
        with self._lock:
            self._held[key] = kept


def _save(path: Path, notes: tuple[SceneNote, ...]) -> None:
    """書いてから置き換える 書く途中で落ちても、半端な控えを残さない"""
    body = json.dumps(
        {
            "version": FORMAT_VERSION,
            "notes": [{"start": n.start, "end": n.end, "text": n.text} for n in notes],
        },
        ensure_ascii=False,
    )
    # 一時ファイルは毎回別の名前にする 同じ素材を 2 つの窓で開いて同時に書くと、
    # 決まった名前ではもう片方の書きかけを奪う（波形の控えと同じ）
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(body)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: Path) -> tuple[SceneNote, ...]:
    """書いた説明を読む 壊れていれば消して空を返す（作り直せる物なので知らせない）"""
    # 有るかを見る所も try の中に置く 置き場を読めない（権限が無い）と exists が投げ、
    # 検索の道具が控えの場所入りの OSError で落ちる 読めないなら覚えていないのと同じに扱う
    try:
        if not path.exists():
            return ()
    except OSError:
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != FORMAT_VERSION:
            return ()
        return tuple(
            SceneNote(start=float(row["start"]), end=float(row["end"]), text=str(row["text"]))
            for row in data["notes"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        # 消せなくても空を返す 消す所で投げると、壊れた控え 1 つで検索が落ちる
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return ()
