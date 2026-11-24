"""素材の置き方の既定（#524）を、作品をまたいで本人の置き場に覚える

作品の中の既定（:attr:`~sashimono.core.model.MediaItem.defaults`）は、その作品でしか効かない
よく使う効果音や立ち絵は別の作品でも同じ置き方をしたいので、本人の設定の置き場にも覚え、
素材を読み込んだときに写す

素材は中身で見分ける（:func:`fingerprint`） 場所や名前で引くと、素材のフォルダを整理した
だけで覚えた既定を見失う 中身の全体を読むと長い動画で読み込みが待たされるので、
大きさと頭と尻の一部だけを読む 同じ大きさで頭と尻が同じ別のファイルは、まず無い
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from sashimono.core.io.atomic import held_for_update, write_atomic
from sashimono.core.io.serialize import (
    ProjectFileError,
    media_defaults_from_json,
    media_defaults_to_json,
)
from sashimono.core.model import MediaDefaults, MediaItem
from sashimono.core.userdirs import config_root

__all__ = [
    "MEMORY_LIMIT",
    "SAMPLE_BYTES",
    "MediaMemory",
    "fingerprint",
    "with_remembered",
]

#: 中身の見分けに読む頭と尻の大きさ（バイト） 1 MiB ずつなら、読み込みのたびに読んでも
#: 1 本あたり数ミリ秒で済む 小さすぎると、頭に同じ見出しを持つ録画どうしを取り違える
SAMPLE_BYTES = 1 << 20

#: 覚えておく素材の数 越えたら古く覚えた物から忘れる 置き場のファイルが際限なく育つと、
#: 読み込みのたびに読む時間が延びる
MEMORY_LIMIT = 2000

#: 置き場のファイルの形の版
_FORMAT_VERSION = 1


def fingerprint(path: Path) -> str | None:
    """素材の中身の見分け 読めなければ ``None``

    大きさと、頭と尻の :data:`SAMPLE_BYTES` を混ぜた SHA-256 場所と名前と更新時刻は
    混ぜない（移しただけ・写しただけで別の素材になる）
    """
    try:
        with path.open("rb") as file:
            size = path.stat().st_size
            digest = hashlib.sha256(b"sashimono-media-1")
            digest.update(size.to_bytes(8, "little"))
            digest.update(file.read(SAMPLE_BYTES))
            if size > SAMPLE_BYTES * 2:
                file.seek(size - SAMPLE_BYTES)
                digest.update(file.read(SAMPLE_BYTES))
            elif size > SAMPLE_BYTES:
                digest.update(file.read())
    except OSError:
        return None
    return digest.hexdigest()


class MediaMemory:
    """作品をまたいで覚える素材の置き方の既定

    壊れていても読み込みは止めない 覚えた既定が効かないだけで、素材は読める
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_root() / "media_defaults.json"

    def recall(self, key: str) -> MediaDefaults | None:
        """``key``（:func:`fingerprint`）の素材に覚えた既定 無い・壊れていれば ``None``"""
        entry = self._entries(self._read()).get(key)
        if not isinstance(entry, dict):
            return None
        try:
            defaults = media_defaults_from_json(entry.get("defaults"))
        except ProjectFileError:
            return None
        return None if defaults.is_empty else defaults

    def remember(self, key: str, name: str, defaults: MediaDefaults | None) -> None:
        """``key`` の素材の既定を覚える ``None`` か空なら忘れる 書けなければ ``OSError``

        読んでから書き戻すまでを錠で守る 2 つの窓が別々の素材を同時に覚えたとき、
        後から書いた方が先の素材を消さないため
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with held_for_update(self.path.with_name(self.path.name + ".lock")):
            entries = self._entries(self._read())
            if defaults is None or defaults.is_empty:
                if entries.pop(key, None) is None:
                    return
            else:
                entries[key] = {
                    # 名前は人が置き場のファイルを開いたときの手掛かり 引くのには使わない
                    "name": name,
                    "remembered": _now(),
                    "defaults": media_defaults_to_json(defaults),
                }
                entries = _trimmed(entries)
            write_atomic(
                self.path,
                json.dumps(
                    {"version": _FORMAT_VERSION, "entries": entries}, ensure_ascii=False, indent=1
                ),
            )

    def _read(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _entries(data: dict[str, Any] | None) -> dict[str, Any]:
        if data is None or not isinstance(data.get("version"), int):
            return {}
        # 新しい版の本体が書いた形は読まずに空として扱う 知らない形を書き戻すと壊す
        # ので、覚えるときも空から始まり、新しい版の記憶を消す しかし版を戻して使う人は
        # まれで、作品の中の既定は残る
        if data["version"] > _FORMAT_VERSION:
            return {}
        entries = data.get("entries")
        return dict(entries) if isinstance(entries, dict) else {}


def _now() -> float:
    """覚えた時刻 試験が差し替える（錠と書き込みも時計を読むので、時計そのものは差し替えない）"""
    return time.time()


def _when(entry: object) -> float:
    if isinstance(entry, dict):
        value = entry.get("remembered")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _trimmed(entries: dict[str, Any]) -> dict[str, Any]:
    """古く覚えた物から忘れて :data:`MEMORY_LIMIT` に収める"""
    if len(entries) <= MEMORY_LIMIT:
        return entries
    newest = sorted(entries.items(), key=lambda pair: _when(pair[1]), reverse=True)
    return dict(newest[:MEMORY_LIMIT])


def with_remembered(media: MediaItem, memory: MediaMemory) -> MediaItem:
    """読み込んだ素材へ、本人の置き場に覚えた既定を写す 覚えていなければそのまま

    作品の中で既に既定を持つ素材は変えない 作品ごとに決めた置き方の方が、その作品では正しい
    """
    if media.defaults is not None:
        return media
    key = fingerprint(media.path)
    if key is None:
        return media
    remembered = memory.recall(key)
    return media if remembered is None else replace(media, defaults=remembered)
