"""見つからない素材を探し直して付け替える（#447）

作品（``.sme``）と素材のフォルダを別の PC へ移すと、素材の場所が変わって見つからない
1 本を選び直したら、同じフォルダか同じ並びのフォルダにある残りも名前で見つけて、まとめて
付け替える（数十本を 1 本ずつ選ばせない） 付け替えは :class:`RelinkMedia` で取り消せる
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from sashimono.core.commands.base import Command
from sashimono.core.model import MediaId, MediaItem, Project

__all__ = ["RelinkMedia", "missing_media", "siblings"]


def _exists(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def missing_media(
    project: Project, exists: Callable[[Path], bool] = _exists
) -> tuple[MediaItem, ...]:
    """ファイルが見つからない素材（素材一覧の並び順）"""
    return tuple(item for item in project.media if not exists(Path(item.path)))


def _moved_root(old: Path, new: Path) -> tuple[Path, Path] | None:
    """``old`` が ``new`` へ移ったとき、移ったフォルダの組（元・先）

    後ろから同じ名前が続く所までを同じ並びと見る（``D:/素材/本編/a.mp4`` が
    ``E:/移した/素材/本編/a.mp4`` なら ``D:/`` → ``E:/移した/``） 名前の大きさは見比べない
    （Windows は大きさを区別しない）
    """
    old_parts, new_parts = old.parts, new.parts
    same = 0
    while (
        same < min(len(old_parts), len(new_parts)) - 1
        and old_parts[-1 - same].casefold() == new_parts[-1 - same].casefold()
    ):
        same += 1
    if same == 0:
        return None
    return Path(*old_parts[: len(old_parts) - same]), Path(*new_parts[: len(new_parts) - same])


def siblings(
    project: Project,
    media_id: MediaId,
    chosen: Path,
    exists: Callable[[Path], bool] = _exists,
) -> dict[MediaId, Path]:
    """``media_id`` を ``chosen`` へ付け替えるとき、一緒に見つかる残りの見つからない素材

    探す所は 2 つ 選んだファイルと同じフォルダの同じ名前と、移ったフォルダの並びを保った場所
    （元の ``素材/本編/a.mp4`` と ``素材/効果音/b.wav`` を、移した先の ``素材/効果音/b.wav`` で
    見つける） 見つかった物だけを返す（ファイルの中身が同じかは呼ぶ側が確かめる）
    """
    item = project.require_media(media_id)
    root = _moved_root(Path(item.path), chosen)
    found: dict[MediaId, Path] = {}
    for other in missing_media(project, exists):
        if other.id == media_id:
            continue
        old = Path(other.path)
        candidates = [chosen.parent / old.name]
        if root is not None:
            # 移ったフォルダの外にあった素材は、並びを保った場所を作れない（同じフォルダだけ探す）
            with contextlib.suppress(ValueError):
                candidates.insert(0, root[1] / old.relative_to(root[0]))
        for candidate in candidates:
            if exists(candidate):
                found[other.id] = candidate
                break
    return found


@dataclass(frozen=True, slots=True)
class RelinkMedia(Command):
    """素材のファイルの場所を付け替える（何本でも 1 段で戻る）

    持ち出し用パッケージの入っている範囲（:attr:`MediaItem.packed`）は外す 探し直して
    付け替えるのは元の素材の全体なので、範囲の外を断り続けると探し直した意味が無い（#445）
    素材の情報（長さ・道・字幕）はそのまま 違うファイルを選んでいないかは呼ぶ側が確かめる
    """

    changes: tuple[tuple[MediaId, Path], ...]

    @property
    def label(self) -> str:
        return (
            "素材を付け替える"
            if len(self.changes) == 1
            else f"素材を {len(self.changes)} 本付け替える"
        )

    def apply(self, project: Project) -> Project:
        if not self.changes:
            return project
        for media_id, path in self.changes:
            item = project.require_media(media_id)
            if Path(item.path) == Path(path) and not item.packed:
                continue
            project = project.replace_media(replace(item, path=Path(path), packed=()))
        return project
