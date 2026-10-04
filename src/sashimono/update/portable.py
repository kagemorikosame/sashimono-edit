r"""exe の隣のスクリプト置き場（``scripts``）に本人が置いた物を見つけ、``%APPDATA%`` 側へ移す

exe の隣の ``scripts`` は、前の版の案内どおりそこへ置いた人のために今も読む
（:func:`sashimono.compat.aviutl.catalog.default_script_roots`） 自動更新では新しい版へ写す
（:func:`.package.carry_user_files`）が、zip を手で展開し直してフォルダごと入れ替えると
中身が消える そこで、置いた物があれば ``%APPDATA%\Sashimono\scripts`` へ移す

移すときの決まり

- **一式（束）で移すか、一式で残す** スクリプトは同じフォルダのモジュール（``.mod2`` ``.lua``
  ``.mod`` と DLL）を先に探す（``compat.aviutl.runtime`` の ``_find_module`` スクリプト自身の
  フォルダ → 置き場の直下 → 置き場の 1 段下 → 深い所） 束の一部だけを移すと、移した先の同じ
  フォルダにある別のモジュールが先に見つかり、更新しただけで描画が変わる 束の決め方は
  :func:`bundles` を見る
- **上書きしない** 移し先に同じ名前で中身の違う物があれば、その束は一式残す 移し先の物は
  本人が後から置いた新しい物かもしれない 中身が同じ（バイト列が同じ）なら衝突ではないので、
  移す側を消すだけにする
- **モジュールの名前が移し先とぶつかる束は残す** モジュールは名前で探し、置き場の直下・
  1 段下・深い所のどこにあっても見つかる 移し先に同じ名前の別のモジュールがあると、束を
  丸ごと移しても、移した先でどちらが先に見つかるかが変わりうる 残した束と同じ名前の
  モジュールを持つ束も残す（今まで exe の隣の中で決まっていた勝ち負けを変えない）
- **束を一式写し終えて中身を照らしてから確定し、確定し終えてから元を消す**（:func:`_place_bundle`）
  束の途中で写せなくなっても（ディスクの空き・権限）、束が 2 つの置き場に割れない 写しは
  移し先と同じドライブの作業用のフォルダ（置き場の外）で作るので、途中の物は読まれない
  途中の物が本来の名前で残ると、次から「もう在る」と見て移さず、しかも欠けた中身の方が読まれる
- 元を消せなかった物は数えて知らせる 両方に在っても、読まれるのは ``%APPDATA%`` の側
"""

from __future__ import annotations

import contextlib
import filecmp
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BUNDLED_SCRIPT_FILES",
    "MODULE_FILE_SUFFIXES",
    "PORTABLE_SCRIPTS_DIR",
    "ScriptMove",
    "bundles",
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

#: 名前で探されるモジュールの拡張子（compat.aviutl.runtime の MODULE_SUFFIXES と
#: C_MODULE_SUFFIX） runtime は Lua と numpy を読むので名前だけ持つ 食い違えば試験が落とす
MODULE_FILE_SUFFIXES = (".lua", ".mod", ".mod2", ".dll")

#: 写している途中の名前に付ける印 スクリプトの拡張子ではないので、途中の物は読まれない
_MOVING_SUFFIX = ".moving"

#: 置き場の直下に置かれた物の束の名前 フォルダの名前と取り違えないよう、パスに使えない文字にする
ROOT_BUNDLE = "*"


@dataclass(frozen=True, slots=True)
class ScriptMove:
    """移した結果 どれも ``scripts`` からの相対の場所"""

    moved: tuple[Path, ...] = ()
    #: 移し先に同じ名前で中身の違う物があったので移さなかった物 exe の隣に残っている
    kept: tuple[Path, ...] = ()
    #: 一緒に残した物 同じ束の中に移せない物があった、またはモジュールの名前が移し先とぶつかる
    held: tuple[Path, ...] = ()
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


def bundles(files: Iterable[Path]) -> dict[str, list[Path]]:
    """一緒に動かす束 置き場の直下のフォルダ 1 つが 1 束、直下に置かれたファイルは全部で 1 束

    フォルダで分けるのは、スクリプトがモジュールを自分のフォルダから先に探すため
    （``_find_module``） 配布物は 1 つのフォルダにまとめて置かれ、その中の階層ごと
    束にする（1 段下へ分けて置く配布物もある） 直下のファイルは同じフォルダ（置き場の直下）を
    分け合うので、ばらばらにすると同じ問題が起きる 名前は大文字小文字を揃えて比べる
    """
    grouped: dict[str, list[Path]] = {}
    for relative in files:
        key = relative.parts[0].casefold() if len(relative.parts) > 1 else ROOT_BUNDLE
        grouped.setdefault(key, []).append(relative)
    return grouped


def _module_stem(path: Path) -> str | None:
    return path.stem.casefold() if path.suffix.casefold() in MODULE_FILE_SUFFIXES else None


def _modules_in(folder: Path) -> dict[str, list[Path]]:
    """置き場にあるモジュール 名前 → 相対の場所 どの深さにあっても名前で見つかる"""
    found: dict[str, list[Path]] = {}
    try:
        paths = [path for path in folder.rglob("*") if path.is_file()]
    except OSError:
        return found
    for path in paths:
        stem = _module_stem(path)
        if stem is not None:
            found.setdefault(stem, []).append(path.relative_to(folder))
    return found


def _same(first: Path, second: Path) -> bool:
    try:
        return filecmp.cmp(first, second, shallow=False)
    except OSError:
        return False


def _plan(source: Path, target: Path, files: list[Path]) -> tuple[set[str], set[Path]]:
    """残す束と、移し先と中身の違う同じ名前の物（衝突） 束はモジュールの名前をたどって広げる"""
    grouped = bundles(files)
    clashes = {
        relative
        for relative in files
        if (target / relative).exists() and not _same(source / relative, target / relative)
    }
    stay = {key for key, members in grouped.items() if any(p in clashes for p in members)}
    # 移し先に同じ名前のモジュールが、同じ場所で同じ中身でない形であれば、その束は残す
    # 移した後に、どちらが先に見つかるかが変わりうる
    there = _modules_in(target)
    for key, members in grouped.items():
        for relative in members:
            stem = _module_stem(relative)
            if stem is None:
                continue
            for other in there.get(stem, []):
                elsewhere = other.as_posix().casefold() != relative.as_posix().casefold()
                if elsewhere or not _same(source / relative, target / other):
                    stay.add(key)
    # 残した束と同じ名前のモジュールを持つ束も残す 片方だけ移すと、今まで exe の隣の中で
    # 決まっていた勝ち負けが、置き場の順（exe の隣が先）で決まるように変わる
    names = {
        key: {stem for p in members if (stem := _module_stem(p)) is not None}
        for key, members in grouped.items()
    }
    changed = True
    while changed:
        changed = False
        held_names = set().union(*(names[key] for key in stay)) if stay else set()
        for key in grouped:
            if key not in stay and names[key] & held_names:
                stay.add(key)
                changed = True
    return stay, clashes


def move_user_scripts(install: Path, target: Path) -> ScriptMove:
    """exe の隣の ``scripts`` に本人が置いた物を ``target`` へ移す 決まりは上の説明のとおり"""
    source = install / PORTABLE_SCRIPTS_DIR
    files = user_script_files(install)
    stay, clashes = _plan(source, target, files)
    moved: list[Path] = []
    kept: list[Path] = []
    held: list[Path] = []
    left: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for key, members in bundles(files).items():
        if key in stay:
            for relative in members:
                (kept if relative in clashes else held).append(relative)
            continue
        problem = _place_bundle(source, target, members)
        if problem is not None:
            # 束は一式 exe の隣に残る（元には触っていない） 束の全部を写せなかった物として数える
            failed.extend((relative, problem) for relative in members)
            continue
        # 移し先に一式そろってから元を消す 一部を消せなくても、後に読まれる %APPDATA% の一式が勝つ
        for relative in members:
            try:
                (source / relative).unlink()
            except OSError:
                left.append(relative)
            else:
                moved.append(relative)
    _remove_empty_folders(source)
    return ScriptMove(tuple(moved), tuple(kept), tuple(held), tuple(left), tuple(failed))


#: 束を写している作業用のフォルダの名前の頭（移し先の親 ``%APPDATA%\Sashimono`` の下）
#: 置き場（``scripts``）の外なので、途中の物がスクリプトとして読まれない
_STAGING_PREFIX = ".scripts-moving-"


def _place_bundle(source: Path, target: Path, members: list[Path]) -> str | None:
    """束を一式、移し先へ置く 置けたら ``None``、置けなければ理由（元にも移し先にも何も残さない）

    1. 作業用のフォルダ（移し先と同じドライブ）へ全部写し、元と中身を照らす
    2. 全部そろってから、移し先へ 1 つずつ名前を付け替えて確定する（同じドライブなので一瞬）
    3. 途中で失敗したら、確定した分を移し先から外し、作業用を片付ける

    確定の途中で落ちても（電源が切れた など）元は消していないので exe の隣に一式残る 移し先に
    入った分は元と同じ中身なので、次の起動では「同じ中身」と見て束ごと移し直す（:func:`_plan`）
    移し先に同じ中身で在る物は写さない（元を消すだけ）
    """
    staging = target.parent / f"{_STAGING_PREFIX}{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    pending: list[Path] = []
    try:
        for relative in members:
            destination = target / relative
            if destination.exists():
                if not _same(source / relative, destination):
                    return f"移し先に中身の違う物がある（{relative.as_posix()}）"
                continue
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, staged)
            # 大きさだけでなく中身を照らす 元を消した後では、欠けた写しに気付いても戻せない
            if not filecmp.cmp(source / relative, staged, shallow=False):
                return f"写した中身が元と違う（{relative.as_posix()}）"
            pending.append(relative)
        placed: list[Path] = []
        try:
            for relative in pending:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 置き換えはしない Windows の rename は在る名前へは付けられないので、写している
                # 間に本人かほかの窓が同じ名前を置いたら、そちらを残してここで止まる
                (staging / relative).rename(destination)
                placed.append(destination)
        except OSError:
            for destination in placed:
                # 外すのは今置いた写しだけ 元は exe の隣に残っている
                with contextlib.suppress(OSError):
                    destination.unlink()
                # 置くために作って空になったフォルダも外す 本人が前から持っていた空の
                # フォルダは消さないよう、置いた物の親だけを移し先の手前まで見る
                for folder in destination.parents:
                    if folder == target or not folder.is_relative_to(target):
                        break
                    with contextlib.suppress(OSError):
                        folder.rmdir()
            raise
    except OSError as exc:
        return str(exc) or type(exc).__name__
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return None


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
