"""画面の中身を調べた結果（文字と顔の位置 #324）の永続化

書くのは終わりまで調べた物だけ 途中で止めた物を書くと、次に開いたときに途中までの結果を
「全部」と取り違え、後ろの場面の文字や顔が無いことにされる
"""

from __future__ import annotations

import json
from pathlib import Path

from sashimono.core.io.atomic import write_atomic
from sashimono.engine.vision.report import ScreenKind, ScreenReport

from .store import CacheStore, media_key

__all__ = ["FORMAT_VERSION", "NAMESPACE", "SUFFIX", "load_screen", "save_screen", "screen_key"]

NAMESPACE = "screen"
SUFFIX = ".screen.json"

#: 保存形式の版 調べる時刻の選び方（:mod:`sashimono.engine.vision.sampling`）や結果の形を
#: 変えたら上げる 上げないと、前の選び方で調べた結果を今の物として返す
#: 2 で長い場面を 10 秒ごとに調べるようにした（:data:`~sashimono.engine.vision.sampling.MAX_GAP`）
FORMAT_VERSION = 2


def screen_key(
    path: Path, kind: ScreenKind, engine: str, per_scene: int, stream: int | None
) -> str:
    """控えの鍵 素材の場所・大きさ・更新時刻（:func:`media_key`）と、調べ方から作る

    調べた部品（OCR の言語・顔のモデル）と場面ごとの枚数が違えば別の結果なので、鍵に入れる
    """
    return media_key(path, extra=f"sv{FORMAT_VERSION}:{kind.value}:{engine}:{per_scene}:{stream}")


def save_screen(store: CacheStore, key: str, report: ScreenReport) -> Path | None:
    """終わりまで調べた結果を書く 途中の物は書かずに ``None``"""
    if not report.complete:
        return None
    path = store.prepare(NAMESPACE, key, SUFFIX)
    payload = {"version": FORMAT_VERSION, "report": report.to_json()}
    write_atomic(path, json.dumps(payload, ensure_ascii=False))
    return path


def load_screen(store: CacheStore, key: str) -> ScreenReport | None:
    """書いた結果を読む 無い・壊れている・版が違うなら ``None``（調べ直す）"""
    path = store.path_for(NAMESPACE, key, SUFFIX)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
        return None
    try:
        report = ScreenReport.from_json(data["report"])
    except (KeyError, ValueError):
        return None
    return report if report.complete else None
