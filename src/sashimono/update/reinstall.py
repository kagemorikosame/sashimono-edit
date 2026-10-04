"""Python が上がる更新で、入れ直しが要る機能と、その大きさと時間の目安

``python_abi`` が変わる版では、画面のボタンで入れた字幕起こしと AI 連携の環境が読めなくなる
（:func:`sashimono.runtime.stale_runtime`） 消しはしないが、入れ直すまでその機能は使えない
入れる前の確認で「入れ直しが要る」とだけ言うと、2 GB を超える落とし直しを知らずに選ぶ
どれを入れ直すのか・どれだけ落とすのか・どれだけ掛かるのかを添える

大きさは今の導入先を実際に測る 入れ直すと同じ物をもう 1 度落とすので、今の大きさが一番
近い 測るのに時間が掛かりすぎたら（遅いディスク・数万のファイル）、機能ごとの目安
（:attr:`~sashimono.runtime.FeaturePack.size_mb`）を使い、目安だと書く
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sashimono.runtime import FeaturePack

__all__ = [
    "MEASURE_SECONDS",
    "SPEEDS_MBPS",
    "Reinstall",
    "describe",
    "estimate",
    "folder_size",
    "installed_packs",
]

#: 測るのに使ってよい長さ 入れる前の確認を開くときに測るので、窓が固まって見えない長さにする
MEASURE_SECONDS = 1.5

#: 時間の目安に使う回線の速さ（Mbps） 速い側と遅い側の 2 つを出す 1 つだけだと、
#: 遅い回線の人が「数分で済む」と読んで、締め切り前に選んでしまう
SPEEDS_MBPS = (100, 20)


@dataclass(frozen=True, slots=True)
class Reinstall:
    """入れ直しの見積もり"""

    labels: tuple[str, ...]
    size_bytes: int
    #: 実際に測った大きさか 偽なら機能ごとの目安を足した物
    measured: bool


def _distribution(requirement: str) -> str:
    """``faster-whisper>=1.1`` → ``faster_whisper``（dist-info の名前の書き方にそろえる）"""
    name = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "_", name).lower()


def _installed_names(target: Path) -> set[str]:
    try:
        folders = list(target.glob("*.dist-info"))
    except OSError:
        return set()
    return {_distribution(folder.name.split("-", 1)[0]) for folder in folders}


def installed_packs(target: Path, packs: Sequence[FeaturePack]) -> list[tuple[FeaturePack, bool]]:
    """導入先に入っている機能と、追加（CUDA ランタイムなど）も入っているか

    import して確かめない Python が上がった後の導入先は、読むと拡張モジュールで落ちる
    pip が置いた dist-info の名前だけを見る
    """
    names = _installed_names(target)
    found = []
    for pack in packs:
        if not pack.required or not all(_distribution(r) in names for r in pack.required):
            continue
        extra = bool(pack.extra) and all(_distribution(r) in names for r in pack.extra)
        found.append((pack, extra))
    return found


def folder_size(
    target: Path,
    *,
    seconds: float = MEASURE_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> int | None:
    """フォルダの中のファイルの大きさの合計 ``seconds`` の内に測り終えなければ ``None``"""
    deadline = clock() + seconds
    total = 0
    pending = [target]
    while pending:
        if clock() > deadline:
            return None
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue  # 読めないフォルダは数えない 少なめに出るだけで、止めるほどではない
    return total


def estimate(
    target: Path,
    packs: Sequence[FeaturePack],
    *,
    seconds: float = MEASURE_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Reinstall | None:
    """入れ直しの見積もり 何も入っていなければ ``None``"""
    found = installed_packs(target, packs)
    measured = folder_size(target, seconds=seconds, clock=clock)
    if not found and not measured:
        return None
    labels = tuple(pack.label for pack, _extra in found)
    if measured is not None and measured > 0:
        return Reinstall(labels, measured, measured=True)
    guessed = sum(pack.size_mb + (pack.extra_size_mb if extra else 0) for pack, extra in found)
    return Reinstall(labels, guessed * 1024 * 1024, measured=False)


def _size_text(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB"
    return f"{max(1, round(size / 1024**2))} MB"


def _minutes(size: int, mbps: int) -> int:
    return max(1, math.ceil(size * 8 / (mbps * 1_000_000) / 60))


def describe(reinstall: Reinstall) -> str:
    """確認の文面に添える 1 行"""
    what = "・".join(reinstall.labels) if reinstall.labels else "入れてある機能"
    size = _size_text(reinstall.size_bytes)
    amount = f"{size}（今入れてある分を測った値）" if reinstall.measured else f"約 {size}（目安）"
    fast, slow = SPEEDS_MBPS
    quick = _minutes(reinstall.size_bytes, fast)
    long = _minutes(reinstall.size_bytes, slow)
    return (
        f"入れ直しが要るのは {what} 落とし直す量は {amount}"
        f" 落とすのに回線が {fast} Mbps なら約 {quick} 分、{slow} Mbps なら約 {long} 分掛かり、"
        "ほかに入れるのに数分掛かります"
    )
