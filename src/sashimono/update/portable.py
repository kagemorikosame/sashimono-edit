r"""exe の隣のスクリプト置き場（``scripts``）に本人が置いた物を見つけ、``%APPDATA%`` 側へ移す

exe の隣の ``scripts`` は、前の版の案内どおりそこへ置いた人のために今も読む
（:func:`sashimono.compat.aviutl.catalog.default_script_roots`） 自動更新では新しい版へ写す
（:func:`.package.carry_user_files`）が、zip を手で展開し直してフォルダごと入れ替えると
中身が消える そこで、置いた物があれば ``%APPDATA%\Sashimono\scripts`` へ移す

移すときの決まり（詳しくは docs/development.md の「exe の隣の scripts」）

- **一式（束）で移すか、一式で残す** スクリプトは同じフォルダのモジュール（``.mod2`` ``.lua``
  ``.mod`` と DLL）を先に探す（``compat.aviutl.runtime`` の ``_find_module`` スクリプト自身の
  フォルダ → 置き場の直下 → 置き場の 1 段下 → 深い所） 束の一部だけを移すと、移した先の同じ
  フォルダにある別のモジュールが先に見つかり、更新しただけで描画が変わる 束の決め方は
  :func:`bundles` を見る
- **上書きしない** 移し先に同じ名前で中身の違う物があれば、その束は一式残す 中身が同じ
  （バイト列が同じ）なら衝突ではないので、移す側を消すだけにする
- **モジュールの名前が移し先とぶつかる束は残す** 残した束と同じ名前のモジュールを持つ束も残す
- **リンクは辿らず、消さず、その束は残す** シンボリックリンクとジャンクションは、置き場の外を
  指していることがある 辿って写すと外の物を写し、消すとリンクの先まで消したように見える
- **束を一式写し終えて中身を照らしてから確定し、確定し終えてから元を消す**（:func:`_place_bundle`）
  消す直前にもう 1 度、元と移し先を照らす 移している間に外のエディタや同期ソフトが元を
  書き換えていれば、移し先の写しを外して元の側に戻す（移し先の方が後に読まれて勝つので、
  古い写しを残すと書き換えた中身が使われなくなる）
- **1 つずつ** 移し先の親の錠（``MOVE_LOCK``）を持って行う
- **同期フォルダの中では自動で移さない**（:func:`in_synced_folder`） exe の隣が OneDrive などの
  中にあると、元を消すとほかの機械からも消える ほかの機械の ``%APPDATA%`` には写っていない
- 読み取り専用の印は、消す直前に外す（付いたままだと Windows は消させず、毎回残る）
"""

from __future__ import annotations

import contextlib
import filecmp
import os
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from sashimono.core.io.locks import HeldLock, try_hold

__all__ = [
    "BUNDLED_SCRIPT_FILES",
    "MODULE_FILE_SUFFIXES",
    "MOVE_LOCK",
    "PORTABLE_SCRIPTS_DIR",
    "RECOVERED_DIR",
    "SWAP_PENDING",
    "ScriptMove",
    "bundles",
    "clean_leftovers",
    "clear_finished_swaps",
    "clear_swap_pending",
    "hold_move_lock",
    "in_synced_folder",
    "is_link",
    "mark_swap_pending",
    "module_stem",
    "modules_in",
    "move_user_scripts",
    "new_swap_token",
    "remove_file",
    "script_links",
    "swap_pending",
    "unoffered",
    "user_script_files",
    "walk_all",
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

#: 移している間に持つ錠の名前（移し先の親 ``%APPDATA%\Sashimono`` の下）
#: 自動更新の ``stage.lock`` ``swap.lock`` と同じく :mod:`sashimono.core.io.locks` の錠
MOVE_LOCK = "scripts-move.lock"

#: 写している途中の名前に付ける印 スクリプトの拡張子ではないので、途中の物は読まれない
_MOVING_SUFFIX = ".moving"

#: 置き場の直下に置かれた物の束の名前 フォルダの名前と取り違えないよう、パスに使えない文字にする
ROOT_BUNDLE = "*"

#: 束を写している作業用のフォルダの名前の頭（移し先の親 ``%APPDATA%\Sashimono`` の下）
#: 置き場（``scripts``）の外なので、途中の物がスクリプトとして読まれない
_STAGING_PREFIX = ".scripts-moving-"

#: 写す前に残しておく空き 写し終えた直後にディスクが一杯になると、ほかの保存（作品・退避）が落ちる
_FREE_MARGIN = 64 * 1024 * 1024

#: 同期ソフトの置き場を示す環境変数（OneDrive は個人用と会社用で別の変数を置く）
_SYNC_VARIABLES = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")

#: 同期ソフトの置き場によく付く名前 環境変数を置かない同期ソフト（Dropbox・Google ドライブ・
#: iCloud）は名前で見分ける 外れても移さないだけで、物は失われない
_SYNC_FOLDER_NAMES = (
    "dropbox",
    "google drive",
    "googledrive",
    "icloud drive",
    "iclouddrive",
    "box",
)


@dataclass(frozen=True, slots=True)
class ScriptMove:
    """移した結果 どれも ``scripts`` からの相対の場所"""

    moved: tuple[Path, ...] = ()
    #: 移し先に同じ名前で中身の違う物があったので移さなかった物 exe の隣に残っている
    kept: tuple[Path, ...] = ()
    #: 一緒に残した物 同じ束の中に移せない物（リンクも）があった、またはモジュールの名前が
    #: 移し先とぶつかる
    held: tuple[Path, ...] = ()
    #: 写せたが元を消せなかった物 両方に在る
    left: tuple[Path, ...] = ()
    #: 写せなかった物と理由 元はそのまま
    failed: tuple[tuple[Path, str], ...] = ()
    #: ほかの Sashimono が移している最中だったので、何もしなかった
    busy: bool = False
    #: exe の隣の scripts そのものがリンクなので、何もしなかった（リンクの先の物を消さない）
    linked: bool = False


def _bundled(relative: Path) -> bool:
    if len(relative.parts) != 1:
        return False
    folded = relative.name.casefold()
    return any(folded == name.casefold() for name in BUNDLED_SCRIPT_FILES)


def is_link(path: Path) -> bool:
    """シンボリックリンクかジャンクション（Windows のフォルダの付け替え）か"""
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return False


def _walk(source: Path) -> tuple[list[Path], list[Path]]:
    """置き場の中のファイルとリンク（相対の場所） リンクの先へは入らない"""
    files, links, _unreadable = walk_all(source)
    return files, links


def walk_all(source: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """置き場の中のファイル・リンク・読めなかった物（相対の場所） リンクの先へは入らない

    読めなかったフォルダや物は、黙って飛ばさずに返す 飛ばすと、その中の物を数えないまま束を
    移して束が割れる（移した側で見つからない）、自動更新の引き継ぎで写し漏れる、が起きる
    呼び手は、読めなかった物を含む束を移さず、引き継ぎなら入れ替えを止める
    置き場そのものが無いのは「何も無い」で、読めなかった物には数えない
    """
    files: list[Path] = []
    links: list[Path] = []
    unreadable: list[Path] = []
    if not source.is_dir():
        return files, links, unreadable
    pending = [source]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                listed = list(entries)
        except OSError:
            unreadable.append(folder.relative_to(source))
            continue
        for entry in listed:
            path = Path(entry.path)
            relative = path.relative_to(source)
            try:
                if entry.is_symlink() or entry.is_junction():
                    links.append(relative)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(relative)
            except OSError:
                unreadable.append(relative)
    return sorted(files), sorted(links), sorted(unreadable)


def user_script_files(install: Path) -> list[Path]:
    """exe の隣の ``scripts`` に本人が置いた物（相対の場所） 同梱の物・写しの途中の物・リンクは除く

    スクリプトに限らず全部数える 共通処理（``.mod2`` ``.lua``）や画像もスクリプトが読む
    片方だけ移すと、移した側で見つからなくなる
    """
    files, _links = _walk(install / PORTABLE_SCRIPTS_DIR)
    return [p for p in files if not _bundled(p) and not p.name.endswith(_MOVING_SUFFIX)]


def script_links(install: Path) -> list[Path]:
    """exe の隣の ``scripts`` の中のリンク（相対の場所） 先へは入らない"""
    return _walk(install / PORTABLE_SCRIPTS_DIR)[1]


def in_synced_folder(path: Path) -> bool:
    """OneDrive などの同期フォルダの中か

    中なら自動では移さない 元を消すと、同期しているほかの機械からも消える
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for variable in _SYNC_VARIABLES:
        value = os.environ.get(variable)
        if value:
            with contextlib.suppress(OSError, ValueError):
                if resolved.is_relative_to(Path(value).resolve()):
                    return True
    for part in resolved.parts:
        folded = part.casefold()
        if folded.startswith("onedrive") or folded in _SYNC_FOLDER_NAMES:
            return True
    return False


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
    （Windows では大文字小文字だけ違う名前は同じフォルダ）
    """
    grouped: dict[str, list[Path]] = {}
    for relative in files:
        key = relative.parts[0].casefold() if len(relative.parts) > 1 else ROOT_BUNDLE
        grouped.setdefault(key, []).append(relative)
    return grouped


def module_stem(path: Path) -> str | None:
    """モジュールとして名前で探される物なら、その名前（大文字小文字を揃える） 違えば ``None``"""
    return path.stem.casefold() if path.suffix.casefold() in MODULE_FILE_SUFFIXES else None


def modules_in(folder: Path) -> dict[str, list[Path]]:
    """置き場にあるモジュール 名前 → 相対の場所 どの深さにあっても名前で見つかる"""
    found: dict[str, list[Path]] = {}
    files, _links = _walk(folder)
    for relative in files:
        stem = module_stem(relative)
        if stem is not None:
            found.setdefault(stem, []).append(relative)
    return found


def _same(first: Path, second: Path) -> bool:
    """中身が同じか 読めなければ（ほかのプログラムが掴んでいる など）違うと見る"""
    try:
        return filecmp.cmp(first, second, shallow=False)
    except OSError:
        return False


def remove_file(path: Path) -> None:
    """ファイルを消す 読み取り専用の印が付いていれば外してから消す（消せなければ OSError）"""
    try:
        path.unlink()
    except PermissionError:
        mode = path.stat().st_mode
        if mode & stat.S_IWRITE:
            raise  # 印のせいではない（ほかのプログラムが開いている） 印を外しても変わらない
        path.chmod(mode | stat.S_IWRITE)
        path.unlink()


def _remove_tree(folder: Path) -> None:
    """作業用のフォルダを片付ける 読み取り専用の印が付いた写しも消す"""

    def clear_and_retry(function: Callable[..., object], name: str, _exc: BaseException) -> None:
        with contextlib.suppress(OSError):
            Path(name).chmod(stat.S_IWRITE)
            function(name)

    if folder.exists():
        shutil.rmtree(folder, onexc=clear_and_retry)


def _plan(
    source: Path, target: Path, files: list[Path], links: list[Path]
) -> tuple[set[str], set[Path]]:
    """残す束と、移し先と中身の違う同じ名前の物（衝突） 束はモジュールの名前をたどって広げる"""
    grouped = bundles(files)
    clashes = {
        relative
        for relative in files
        if (target / relative).exists() and not _same(source / relative, target / relative)
    }
    stay = {key for key, members in grouped.items() if any(p in clashes for p in members)}
    # リンクを含む束は残す 辿って写すと置き場の外の物を写し、元を消すとリンクの先が
    # 消えたように見える リンクだけ残して束の残りを移すと、相対で読む物が別れる
    stay.update(bundles(links))
    # 移し先に同じ名前のモジュールが、同じ場所で同じ中身でない形であれば、その束は残す
    # 移した後に、どちらが先に見つかるかが変わりうる
    there = modules_in(target)
    for key, members in grouped.items():
        for relative in members:
            stem = module_stem(relative)
            if stem is None:
                continue
            for other in there.get(stem, []):
                elsewhere = other.as_posix().casefold() != relative.as_posix().casefold()
                if elsewhere or not _same(source / relative, target / other):
                    stay.add(key)
    # 残した束と同じ名前のモジュールを持つ束も残す 片方だけ移すと、今まで exe の隣の中で
    # 決まっていた勝ち負けが、置き場の順（exe の隣が先）で決まるように変わる
    names = {
        key: {stem for p in members if (stem := module_stem(p)) is not None}
        for key, members in grouped.items()
    }
    changed = True
    while changed:
        changed = False
        held_names = set().union(*(names[key] for key in stay if key in names)) if stay else set()
        for key in grouped:
            if key not in stay and names[key] & held_names:
                stay.add(key)
                changed = True
    return stay, clashes


def hold_move_lock(
    folder: Path,
    *,
    wait: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> HeldLock | None:
    """exe の隣の ``scripts`` と ``%APPDATA%`` の ``scripts`` を触る間の錠を取る

    ``folder`` は ``%APPDATA%\\Sashimono`` 移す（:func:`move_user_scripts`）も、自動更新の
    引き継ぎ（``package.carry_user_files``）も同じ錠を持つ 重なると、片方が置いた写しを
    もう片方が当てにして元を消し、巻き戻しで写しが外れて両方から失われる（PR #245 の Codex の
    指摘） ``wait`` 秒まで取り直す 取れなければ ``None``
    """
    deadline = clock() + wait
    while True:
        lock = try_hold(folder / MOVE_LOCK)
        if lock is not None or clock() >= deadline:
            return lock
        sleep(0.1)


def clean_leftovers(install: Path, target: Path) -> list[Path]:
    """落ちた起動（電源が切れた・移している途中で閉じた）が残した物を片付ける 錠を持って呼ぶ

    作業用のフォルダ・写しの途中の物（``.moving``）を捨て、よけたまま残った元を元の場所へ戻す
    （:func:`_recover_removed`） 自動更新は入れ替える前に呼ぶ 呼ばずに今の版のフォルダを
    ``.previous`` へ回すと、よけた元は戻らずに次の更新で消える（PR #245 の Codex の指摘）
    よけたまま扱えなかった物を返す（作業用の写しは作り直せるので、捨て損ねても返さない）
    """
    for leftover in target.parent.glob(f"{_STAGING_PREFIX}*"):
        _remove_tree(leftover)
    # 写しの途中の物を探すのはリンクを辿らない走査で行い、移し先そのものがリンクなら探さない
    # リンクの先（AviUtl と分け合うフォルダなど）の物を、名前だけで消さない
    if not is_link(target):
        for relative in walk_all(target)[0]:
            if relative.name.endswith(_MOVING_SUFFIX):
                with contextlib.suppress(OSError):
                    remove_file(target / relative)  # 捨て損ねても読まれない
    return _recover_removed(install, target)


def move_user_scripts(
    install: Path, target: Path, *, should_stop: Callable[[], bool] | None = None
) -> ScriptMove:
    """exe の隣の ``scripts`` に本人が置いた物を ``target`` へ移す 決まりは上の説明のとおり

    **移すのは 1 つずつ** 移し先の親（``%APPDATA%\\Sashimono``）の錠（``MOVE_LOCK``）を持って
    行う（:func:`hold_move_lock`） 取れなければ何もせずに ``busy`` を返す（次の起動で移す）
    錠の持ち主が落ちていれば、次に取るときに片付く（:mod:`sashimono.core.io.locks`）

    ``should_stop`` が真を返したら、次の束へは入らずに終える（束の途中では止めない）
    アプリを閉じるときに、元をよけたまま止まらないようにするため
    """
    lock = hold_move_lock(target.parent)
    if lock is None:
        return ScriptMove(busy=True)
    try:
        if swap_pending(target.parent) or _swapping():
            # 引き継ぎを終えて入れ替え係を待っている・入れ替え係が走っている 今の版のフォルダが
            # .previous へ回る途中なので触らない（よけた元が .previous へ行って失われる）
            return ScriptMove(busy=True)
        if is_link(install / PORTABLE_SCRIPTS_DIR):
            # exe の隣の scripts そのものがリンク（ジャンクション・シンボリックリンク）
            # AviUtl と分け合う Script フォルダや同期先を指していることがある 辿って移すと、
            # 元を消す段でリンクの先の物を消し、起動しただけで共有のフォルダが空になる
            # （PR #245 の Codex の指摘） 何も移さずに残す 本人の置き場はリンクの先にある
            return ScriptMove(linked=True)
        # 錠を持てた ほかに移している人はいない 落ちた起動が残した物を片付ける
        clean_leftovers(install, target)
        return _move_all(install, target, should_stop or (lambda: False))
    finally:
        lock.release()


#: 引き継ぎを終えて入れ替え係を起こすまでの印（``%APPDATA%\\Sashimono`` の下）
#: 引き継ぎは錠（``MOVE_LOCK``）を放してから入れ替え係を起こす 入れ替え係は別のプロセス
#: （PowerShell）で、この錠を受け取れない 放してから入れ替え係が ``swap.lock`` を取るまでの
#: 隙に別の窓が移し始めると、よけた元が今の版のフォルダごと .previous へ回って失われる
#: （PR #245 の CodeRabbit の指摘） そこで錠を放す前にこの印を置き、移しは印が新しい間は始めない
#: 印は引き継ぎごとに別のファイル（``scripts-move.swap-pending.<識別子>``）にする 1 つのファイルを
#: 分け合うと、2 つの窓が続けて引き継いだとき、先の窓が入れ替え係を起こせずに外した所で後の窓の
#: 印まで消える（PR #245 の CodeRabbit の指摘） 移しは、どれか 1 つでも新しければ始めない
SWAP_PENDING = "scripts-move.swap-pending"

#: 印が効く長さ（秒） 入れ替え係は窓が全部閉じるのを待ち（120 秒まで）、新しい版が起動できたかを
#: 確かめる 入れ替えに失敗して印が残っても、この長さを過ぎれば移しは再び始まる
SWAP_PENDING_SECONDS = 10 * 60


def new_swap_token() -> str:
    """引き継ぎごとの識別子 プロセス番号と乱数（同じプロセスで 2 度引き継いでも重ならない）"""
    return f"{os.getpid()}-{uuid.uuid4().hex[:12]}"


def _marker(folder: Path, token: str) -> Path:
    return folder / f"{SWAP_PENDING}.{token}"


def _markers(folder: Path) -> list[Path]:
    try:
        return list(folder.glob(f"{SWAP_PENDING}.*"))
    except OSError:
        return []


def _read_marker(marker: Path) -> tuple[str, float] | None:
    """印の中身（入れ替える前の版・置いた時刻） 読めなければ ``None``"""
    try:
        version, written = marker.read_text(encoding="utf-8").split("\n")[:2]
        return version, float(written)
    except (OSError, ValueError):
        return None


def mark_swap_pending(folder: Path, version: str, token: str) -> None:
    """引き継ぎを終えて入れ替え係を起こす前に置く ``version`` は入れ替える前の版

    ``token`` は :func:`new_swap_token` で作った、この引き継ぎの識別子 外すときに同じ物を渡す

    **書けなければ OSError** 書けたかは読み戻して確かめる 黙って先へ進むと、印の無いまま
    錠を放して入れ替え係を起こし、その隙に別の窓の移しと重なる（PR #245 の CodeRabbit の指摘）
    """
    marker = _marker(folder, token)
    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{version}\n{time.time()}", encoding="utf-8")
    read = _read_marker(marker)
    if read is None or read[0] != version:
        raise OSError(f"入れ替え係を待っている印を書けない: {marker.name}")


def clear_swap_pending(folder: Path, token: str) -> None:
    """自分の引き継ぎの印だけを外す 入れ替え係を起こせなかったときに呼ぶ ほかの窓の印は残す"""
    with contextlib.suppress(OSError):
        _marker(folder, token).unlink(missing_ok=True)


def clear_finished_swaps(
    folder: Path, current_version: str, *, clock: Callable[[], float] = time.time
) -> None:
    """入れ替えが済んだ印と、古くなった印を外す 起動の後に呼ぶ

    印の版（入れ替える前の版）が今の版と違えば、その入れ替えは済んだ 同じ版の印は、まだ入れ替えて
    いない（入れ替え係がほかの窓の閉じるのを待っている）か、入れ替えに失敗して戻った物 前者を外すと
    受け渡しの隙が開くので残す 後者は :data:`SWAP_PENDING_SECONDS` を過ぎたら外す（効かなくなった
    印を溜めない）
    """
    for marker in _markers(folder):
        read = _read_marker(marker)
        stale = read is None or not 0 <= clock() - read[1] < SWAP_PENDING_SECONDS
        if stale or (read is not None and read[0] != current_version):
            with contextlib.suppress(OSError):
                marker.unlink(missing_ok=True)


def swap_pending(folder: Path, *, clock: Callable[[], float] = time.time) -> bool:
    """入れ替え係を待っている印が 1 つでも新しいか"""
    for marker in _markers(folder):
        read = _read_marker(marker)
        if read is not None and 0 <= clock() - read[1] < SWAP_PENDING_SECONDS:
            return True
    return False


def _swapping() -> bool:
    """入れ替え係が走っているか（``swap.lock``） 移しの錠を取った後にもう 1 度見る"""
    # update.state は userdirs と錠しか読まないが、ここから上で読むと package との読み込みの
    # 順に縛られるので、要る所で読む
    from sashimono.update.state import busy_with

    return busy_with() == "swap"


def _move_all(install: Path, target: Path, should_stop: Callable[[], bool]) -> ScriptMove:
    source = install / PORTABLE_SCRIPTS_DIR
    walked, links, unreadable = walk_all(source)
    files = [p for p in walked if not _bundled(p) and not p.name.endswith(_MOVING_SUFFIX)]
    stay, clashes = _plan(source, target, files, links)
    # 読めなかった物を含む束は移さない 中の物を数えないまま移すと束が割れる
    # 置き場そのもの（``.``）が読めなければ、どの束も移さない
    if any(p == Path() for p in unreadable):
        return ScriptMove(failed=((Path(), "exe の隣の scripts を読めない"),))
    stay.update(bundles(unreadable))
    files = [*files, *unreadable]
    moved: list[Path] = []
    kept: list[Path] = []
    held: list[Path] = []
    left: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for key, members in bundles([*files, *links]).items():
        if key in stay:
            for relative in members:
                (kept if relative in clashes else held).append(relative)
            continue
        if should_stop():
            break  # 閉じる 新しい束には入らない 残りは次の起動で移す
        if not source.is_dir():
            # 今の版のフォルダが付け替えられた（入れ替え係が先に動いた） 次の束には入らない
            break
        problem, placed = _place_bundle(source, target, members)
        if problem is None:
            problem, gone, stuck = _retire(install, target, members, placed)
            moved.extend(gone)
            left.extend(stuck)
        if problem is not None:
            # 束は一式 exe の隣に残る（元には触っていない） 束の全部を写せなかった物として数える
            failed.extend((relative, problem) for relative in members)
    _remove_empty_folders(source)
    return ScriptMove(tuple(moved), tuple(kept), tuple(held), tuple(left), tuple(failed))


#: 消す元をいったんよける作業用のフォルダの名前の頭（exe の隣のフォルダ ``install`` の直下）
#: 元と同じドライブなので名前の付け替えだけで済む 置き場（``scripts``）の外なので読まれない
_REMOVING_PREFIX = ".scripts-removing-"


def _retire(
    install: Path, target: Path, members: list[Path], placed: list[Path]
) -> tuple[str | None, list[Path], list[Path]]:
    """移し先に一式そろった束の元を消す ``(書き換わっていればその理由, 消した物, 消せなかった物)``

    照らしてから消すまでの間に書き換わると、新しい中身を消してしまう（PR #245 の Codex の指摘）
    そこで消す代わりに、元を作業用のフォルダへ名前を付け替えてよける よけた物は、外のエディタや
    同期ソフトが元の場所へ書いても変わらない よけた物を移し先と照らし、全部一致したら消す
    1 つでも違えば（よける前に書き換わった）、または元の場所に新しい物が置かれていれば、
    よけた物を元の場所へ戻し、移し先へ置いた写しを外して、束を元の側に戻す（次の起動で移し直す）

    よけられなかった物（ほかのプログラムが開いている）は元の場所で照らす 一致すれば「消せなかった」
    として数える（移し先の一式が勝つ 次の起動で同じ中身と見て消し直す） 違えば束を戻す
    途中で落ちても、よけた物は次に錠を取ったときに :func:`_recover_removed` が元へ戻すか片付ける
    """
    source = install / PORTABLE_SCRIPTS_DIR
    parking = install / f"{_REMOVING_PREFIX}{os.getpid()}"
    parked: list[Path] = []
    stuck: list[Path] = []
    for relative in members:
        try:
            (parking / relative).parent.mkdir(parents=True, exist_ok=True)
            (source / relative).rename(parking / relative)
        except OSError:
            stuck.append(relative)
        else:
            parked.append(relative)
    changed = [p for p in parked if not _same(parking / p, target / p)]
    changed += [p for p in stuck if not _same(source / p, target / p)]
    changed += [p for p in parked if (source / p).exists() or is_link(source / p)]
    if not changed:
        gone: list[Path] = []
        for relative in parked:
            try:
                remove_file(parking / relative)
            except OSError:
                # よけた所で消せない 元の場所へ戻せば、次の起動で同じ中身と見て消し直す
                with contextlib.suppress(OSError):
                    (parking / relative).rename(source / relative)
                stuck.append(relative)
            else:
                gone.append(relative)
        _remove_tree_if_empty(parking)
        return None, gone, stuck
    for relative in parked:
        back = source / relative
        if back.exists() or is_link(back):
            # 元の場所に新しい物が置かれた（本人が書き直した） そちらが正 よけた古い物は、
            # 移し先の写しと同じ中身なら捨て、違えば消さずによけたまま残す（次の片付けで扱う）
            if _same(parking / relative, target / relative):
                with contextlib.suppress(OSError):
                    remove_file(parking / relative)
            continue
        with contextlib.suppress(OSError):
            back.parent.mkdir(parents=True, exist_ok=True)
            (parking / relative).rename(back)
    _take_back(source, target, placed)
    _remove_tree_if_empty(parking)
    return f"移している間に書き換わった（{changed[0].as_posix()}）", [], []


def _remove_tree_if_empty(folder: Path) -> None:
    """中にファイルが残っていなければ片付ける 残っていれば触らない（本人の物かもしれない）"""
    if not folder.exists():
        return
    if any(path.is_file() for path in folder.rglob("*")):
        return
    _remove_tree(folder)


#: よけた元が、元の場所の新しい物とも移し先とも違ったときに取っておく所（``%APPDATA%\\Sashimono``
#: の下 置き場の外なので読まれない） exe の隣のフォルダの中に残すと、自動更新で .previous へ回り、
#: 次の更新で消える
RECOVERED_DIR = "scripts-recovered"


def _recover_removed(install: Path, target: Path) -> list[Path]:
    """落ちた起動がよけたまま残した元を扱う 錠を持っているときだけ呼ぶ 扱えずに残った物を返す

    元の場所が空いていれば戻す（移し先に同じ中身があれば、次の移しで同じ中身と見て片付く）
    元の場所に物があり、よけた物がそれか移し先と同じ中身なら捨てる 違えば（本人が後から書き直した
    物より前の中身） ``%APPDATA%\\Sashimono\\scripts-recovered`` へ取っておく
    どれもできなかった物は返す 自動更新の引き継ぎは、これが空でなければ入れ替えを止める
    （exe の隣のフォルダに残ったまま入れ替えると、.previous へ回って次の更新で消える）
    """
    source = install / PORTABLE_SCRIPTS_DIR
    stuck: list[Path] = []
    for parking in install.glob(f"{_REMOVING_PREFIX}*"):
        files, links, unreadable = walk_all(parking)
        stuck.extend(parking / p for p in (*links, *unreadable))
        for relative in files:
            parked = parking / relative
            back = source / relative
            try:
                if not back.exists() and not is_link(back):
                    back.parent.mkdir(parents=True, exist_ok=True)
                    parked.rename(back)
                elif _same(parked, back) or _same(parked, target / relative):
                    remove_file(parked)
                else:
                    kept = _free_name(target.parent / RECOVERED_DIR / relative)
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(parked, kept)
                    if not _same(parked, kept):
                        raise OSError("取っておく写しが元と違う")
                    remove_file(parked)
            except OSError:
                stuck.append(parked)
        _remove_tree_if_empty(parking)
    return stuck


def _free_name(path: Path) -> Path:
    """在る名前なら「名前 (2).拡張子」のように空いている名前にする"""
    candidate = path
    number = 2
    while candidate.exists() or is_link(candidate):
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        number += 1
    return candidate


def _take_back(source: Path, target: Path, placed: list[Path]) -> None:
    """自分が移し先へ置いた写しを外す 元が exe の隣に残っているときだけ

    元が無ければ、ほかの誰か（錠を持たない古い版の移し・本人）がこの写しを当てにして元を消した
    外すと両方の置き場から失われる（PR #245 の Codex の指摘）
    """
    for relative in placed:
        destination = target / relative
        if not (source / relative).is_file():
            continue
        with contextlib.suppress(OSError):
            remove_file(destination)
        # 置くために作って空になったフォルダも外す 本人が前から持っていた空の
        # フォルダは消さないよう、置いた物の親だけを移し先の手前まで見る
        for folder in destination.parents:
            if folder == target or not folder.is_relative_to(target):
                break
            with contextlib.suppress(OSError):
                folder.rmdir()


def _place_bundle(source: Path, target: Path, members: list[Path]) -> tuple[str | None, list[Path]]:
    """束を一式、移し先へ置く 置けたら ``(None, 置いた物)``、置けなければ ``(理由, [])``
    （置けなかったときは元にも移し先にも何も残さない）

    1. 空きを確かめる（足りなければ写し始めない 途中で一杯にして、ほかの保存を落とさない）
    2. 作業用のフォルダ（移し先と同じドライブ）へ全部写し、元と中身を照らす
    3. 全部そろってから、移し先へ 1 つずつ名前を付け替えて確定する（同じドライブなので一瞬）
    4. 途中で失敗したら、確定した分を移し先から外し、作業用を片付ける

    確定の途中で落ちても（電源が切れた など）元は消していないので exe の隣に一式残る 移し先に
    入った分は元と同じ中身なので、次の起動では「同じ中身」と見て束ごと移し直す（:func:`_plan`）
    移し先に同じ中身で在る物は写さない（元を消すだけ）
    """
    staging = target.parent / f"{_STAGING_PREFIX}{os.getpid()}"
    _remove_tree(staging)
    pending: list[Path] = []
    placed: list[Path] = []
    try:
        for relative in members:
            destination = target / relative
            if destination.exists():
                if not _same(source / relative, destination):
                    return f"移し先に中身の違う物がある（{relative.as_posix()}）", []
                continue
            pending.append(relative)
        need = sum((source / relative).stat().st_size for relative in pending)
        target.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target.parent).free
        if need + _FREE_MARGIN > free:
            return (
                f"移し先の空きが足りない（{need // 1024**2} MB 要る 空きは {free // 1024**2} MB）",
                [],
            )
        for relative in pending:
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, staged)
            # 大きさだけでなく中身を照らす 元を消した後では、欠けた写しに気付いても戻せない
            if not filecmp.cmp(source / relative, staged, shallow=False):
                return f"写した中身が元と違う（{relative.as_posix()}）", []
        try:
            for relative in pending:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 置き換えはしない Windows の rename は在る名前へは付けられないので、写している
                # 間に本人かほかの窓が同じ名前を置いたら、そちらを残してここで止まる
                (staging / relative).rename(destination)
                placed.append(relative)
        except OSError:
            _take_back(source, target, placed)
            raise
    except OSError as exc:
        return str(exc) or type(exc).__name__, []
    finally:
        _remove_tree(staging)
    return None, placed


def _remove_empty_folders(source: Path) -> None:
    """移して空になったフォルダを片付ける ``scripts`` そのものは残す（同梱の説明が入る所）

    リンクの先へは入らない（``os.walk`` はシンボリックリンクを辿らず、ジャンクションは中身が
    あれば rmdir が断る 空のフォルダを指すジャンクションの rmdir はリンクだけを外す）
    """
    folders = []
    with contextlib.suppress(OSError):
        for path, names, _files in os.walk(source):
            names[:] = [name for name in names if not is_link(Path(path) / name)]
            folders.append(Path(path))
    for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        if folder == source:
            continue
        with contextlib.suppress(OSError):
            folder.rmdir()  # 中身が残っていれば断られる それでよい
