r"""exe の隣のスクリプト置き場（``scripts``）に本人が置いた物を見つけ、``%APPDATA%`` 側へ移す

exe の隣の ``scripts`` は、前の版の案内どおりそこへ置いた人のために今も読む
（:func:`sashimono.compat.aviutl.catalog.default_script_roots`） 自動更新では新しい版へ写す
（:func:`.package.carry_user_files`）が、zip を手で展開し直してフォルダごと入れ替えると
中身が消える そこで、置いた物があれば ``%APPDATA%\Sashimono\scripts`` へ移すことを勧める

移すときの決まり

- **上書きしない** 移し先に同じ名前があれば、その物は移さずに残す 移し先の物は本人が
  後から置いた新しい物かもしれない 残した物は exe の隣から今までどおり読まれる
- **写し終えて中身を照らしてから元を消す** 写す途中で止まっても元は残る 写しは作業用の
  名前で書き、元と同じ中身かを確かめてから本来の名前を付ける 途中の物が本来の名前で
  残ると、次から「もう在る」と見て移さず、しかも欠けた中身の方が読まれる（後の置き場が勝つ）
- 元を消せなかった物は数えて知らせる 両方に在っても、読まれるのは ``%APPDATA%`` の側
"""

from __future__ import annotations

import contextlib
import filecmp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BUNDLED_SCRIPT_FILES",
    "PORTABLE_SCRIPTS_DIR",
    "ScriptMove",
    "move_user_scripts",
    "unoffered",
    "user_script_files",
]

#: exe の隣のスクリプト置き場の名前（compat.aviutl.catalog の PORTABLE_SCRIPTS_DIR）
#: catalog は読み込みが重い（効果の一覧を作る）ので名前だけ持つ 食い違えば試験が落とす
PORTABLE_SCRIPTS_DIR = "scripts"

#: 配る zip が ``scripts`` の直下に入れている物（tools/build_package.py の assemble）
#: 本人の物として数えない 数えると、何も置いていない人にまで移すよう勧める
#: Windows は大文字と小文字を区別しないので、比べるときは揃える
BUNDLED_SCRIPT_FILES = frozenset({"README.txt"})

#: 写している途中の名前に付ける印 スクリプトの拡張子ではないので、途中の物は読まれない
_MOVING_SUFFIX = ".moving"


@dataclass(frozen=True, slots=True)
class ScriptMove:
    """移した結果 どれも ``scripts`` からの相対の場所"""

    moved: tuple[Path, ...] = ()
    #: 移し先に同じ名前があったので移さなかった物 exe の隣に残っている
    kept: tuple[Path, ...] = ()
    #: 写せたが元を消せなかった物 両方に在る
    left: tuple[Path, ...] = ()
    #: 写せなかった物と理由 元はそのまま
    failed: tuple[tuple[Path, str], ...] = ()


def _bundled(relative: Path) -> bool:
    if len(relative.parts) != 1:
        return False
    folded = relative.name.casefold()
    return any(folded == name.casefold() for name in BUNDLED_SCRIPT_FILES)


def user_script_files(install: Path) -> list[Path]:
    """exe の隣の ``scripts`` に本人が置いた物（相対の場所） 同梱の物と写しの途中の物は除く

    スクリプトに限らず全部数える 共通処理（``.mod2`` ``.lua``）や画像もスクリプトが読む
    片方だけ移すと、移した側で見つからなくなる
    """
    source = install / PORTABLE_SCRIPTS_DIR
    try:
        found = sorted(path for path in source.rglob("*") if path.is_file())
    except OSError:
        return []
    files = []
    for path in found:
        relative = path.relative_to(source)
        if _bundled(relative) or path.name.endswith(_MOVING_SUFFIX):
            continue
        files.append(relative)
    return files


def unoffered(files: list[Path], offered: tuple[str, ...]) -> list[Path]:
    """まだ勧めていない物 1 度勧めた物だけなら、もう勧めない（1 度だけにする）"""
    seen = {name.casefold() for name in offered}
    return [path for path in files if path.as_posix().casefold() not in seen]


def move_user_scripts(install: Path, target: Path) -> ScriptMove:
    """exe の隣の ``scripts`` に本人が置いた物を ``target`` へ移す 決まりは上の説明のとおり"""
    source = install / PORTABLE_SCRIPTS_DIR
    moved: list[Path] = []
    kept: list[Path] = []
    left: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for relative in user_script_files(install):
        origin = source / relative
        destination = target / relative
        if destination.exists():
            kept.append(relative)
            continue
        # 作業用の名前は起動ごとに分ける 窓を 2 つ開くと、どちらも起動のときに移しに来る
        writing = destination.with_name(f"{destination.name}.{os.getpid()}{_MOVING_SUFFIX}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, writing)
            # 大きさだけでなく中身を照らす 元を消した後では、欠けた写しに気付いても戻せない
            if not filecmp.cmp(origin, writing, shallow=False):
                raise OSError("写した中身が元と違う")
            # 置き換えはしない Windows の rename は在る名前へは付けられないので、写している
            # 間に本人かほかの窓が同じ名前を置いたら、そちらを残してここで止まる
            writing.rename(destination)
        except OSError as exc:
            with contextlib.suppress(OSError):
                writing.unlink(missing_ok=True)
            failed.append((relative, str(exc)))
            continue
        try:
            origin.unlink()
        except OSError:
            left.append(relative)
            continue
        moved.append(relative)
    _remove_empty_folders(source)
    return ScriptMove(tuple(moved), tuple(kept), tuple(left), tuple(failed))


def _remove_empty_folders(source: Path) -> None:
    """移して空になったフォルダを片付ける ``scripts`` そのものは残す（同梱の説明が入る所）"""
    folders = []
    with contextlib.suppress(OSError):
        for path, _names, _files in os.walk(source):
            folders.append(Path(path))
    for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        if folder == source:
            continue
        with contextlib.suppress(OSError):
            folder.rmdir()  # 中身が残っていれば断られる それでよい
