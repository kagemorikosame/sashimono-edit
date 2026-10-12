r"""本人の置き場（設定・退避・キャッシュ・導入した実行環境）と、旧名の置き場からの移行

置き場は 4 つあり、消えたときの痛さが違う

- **設定**（``%APPDATA%\Sashimono``） 画面配置・ショートカット・設定・プリセット・
  自分で置いたスクリプトとテンプレート 作り直せない
- **退避**（``%LOCALAPPDATA%\Sashimono``） 落ちたときの退避と、保存前のバックアップ
  作り直せない
- **キャッシュ**（``%LOCALAPPDATA%\Sashimono\cache``） 波形・サムネイル・控え
  消しても作り直せるが、大きい（控えは数 GB になる）
- **実行環境**（``%LOCALAPPDATA%\Sashimono\runtime``） 画面のボタンで入れた
  字幕起こしなど 入れ直せるが 2 GB を超え、落とし直しに時間がかかる

名前をここ 1 か所に集めるのは、置き場の名前を変えたときに移行の側と食い違わない
ようにするため（前の名前から改名したときに、7 か所に同じ名前が散っていた）

Windows 以外（開発と CI の Linux）では XDG の決まりに寄せる 環境変数が無ければ
ホームの下の小文字の名前にする
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "APP_FOLDER",
    "LEGACY_APP_FOLDERS",
    "MigrationNote",
    "cache_root",
    "config_root",
    "data_root",
    "migrate_legacy_folders",
    "state_root",
]

#: 置き場の名前 表示名（Sashimono Edit）ではなく短い名前にする 空白を含む
#: フォルダ名はコマンドで打つときに括りが要り、問い合わせで場所を伝えにくい
APP_FOLDER = "Sashimono"

# 旧名を残す: ここから（名前の一括置換でも書き換えない 旧名の置き場を探すのに要る）
#: 前の名前の置き場 新しい順に並べる 商標の都合で Kumiki から改名した
#: ここから消すと、改名前から使っている人の設定・プリセット・スクリプト・退避が
#: 新しい版から見えなくなる（消えはしないが、無いものとして既定で始まる）
LEGACY_APP_FOLDERS = ("Kumiki",)
# 旧名を残す: ここまで

#: 退避の置き場の中で、キャッシュが入るフォルダの名前
_CACHE_DIR = "cache"

#: 退避の置き場を丸ごと付け替えられなかったときに、1 つずつ拾い上げるフォルダ
#: 作り直せない物だけ キャッシュ（``cache``）は作り直せ、実行環境（``runtime``）は
#: 画面のボタンで入れ直せる 窓ごとの錠（``open``）は動いている起動の印で、移す物ではない
_SALVAGED_DIRS = ("recovery", "backups")

#: 拾い上げを続ける印（新しい退避の置き場に置く） 名前に旧名を入れて、旧名ごとに分ける
#: 旧版が動いている間は付け替えられず、旧版はその後も旧い置き場へ退避と控えを書く
#: 印が無いと、最初の起動で拾った後に旧版が書いた物は旧い置き場に取り残される
_PARTIAL_PREFIX = ".partial-migration-"


def config_root(folder: str = APP_FOLDER) -> Path:
    """設定を置く場所 消えると作り直せないので ``%APPDATA%`` に置く"""
    return _root(folder, ("APPDATA", "XDG_CONFIG_HOME"), (".config",))


def state_root(folder: str = APP_FOLDER) -> Path:
    """退避とバックアップを置く場所 大きくなりうるので移動プロファイルに載せない"""
    return _root(folder, ("LOCALAPPDATA", "XDG_STATE_HOME"), (".local", "state"))


def cache_root(folder: str = APP_FOLDER) -> Path:
    """キャッシュを置く場所

    Windows では退避の置き場の中の ``cache`` 退避と分けておくのは、キャッシュは
    消してよいものとして案内するので、同じフォルダにあると一緒に消されるため
    """
    base = _first_env(("LOCALAPPDATA", "XDG_CACHE_HOME"))
    if base is not None:
        return base / folder / _CACHE_DIR
    return Path.home() / ".cache" / folder.lower()


def data_root(folder: str = APP_FOLDER) -> Path:
    """配布版で、画面のボタンから入れた実行環境を置く場所の親"""
    return _root(folder, ("LOCALAPPDATA", "XDG_DATA_HOME"), (".local", "share"))


def _root(folder: str, variables: tuple[str, ...], fallback: tuple[str, ...]) -> Path:
    base = _first_env(variables)
    if base is not None:
        return base / folder
    return Path.home().joinpath(*fallback, folder.lower())


def _first_env(variables: tuple[str, ...]) -> Path | None:
    for name in variables:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


# --- 旧名の置き場からの移行 -------------------------------------------------


@dataclass(frozen=True, slots=True)
class MigrationNote:
    """移行で行ったこと 1 件 起動の記録と試験に使う"""

    source: Path
    target: Path
    #: ``copied``（写した 元は残す）・``moved``（移した）・
    #: ``moved-partly``（丸ごとは移せず、作り直せない物を 1 つずつ移した）・``failed``
    action: str
    detail: str = ""


def migrate_legacy_folders() -> list[MigrationNote]:
    """旧名の置き場を、新しい名前の置き場へ引き継ぐ 起動のたびに呼んでよい

    **新しい置き場がまだ無く、旧い置き場が在るときだけ**動く 1 度引き継げば
    新しい置き場が在るので、2 回目からは何もしない（旧版と行き来しても、
    新しい版で作った物を旧い中身で上書きしない）

    置き場ごとに扱いを変える

    - **設定は写す**（元は残す） 小さく、作り直せない 移すと、写している途中で
      落ちたときや旧版へ戻したときに設定が無くなる 写しなら元が必ず残る
      自分で置いたスクリプトとテンプレートもここに入っているので、写せば
      新しい版の探す先（新しい置き場の ``scripts``）で見つかる
    - **退避の置き場は移す**（名前の付け替え） キャッシュと実行環境が中に入っていて、
      合わせて数 GB になる 写すと起動が何分も止まり、ディスクも倍使う 同じ親の下の
      名前の付け替えなので一瞬で済み、キャッシュの鍵（素材の場所）も変わらないので
      そのまま使える
    - 付け替えられなかったとき（旧版が動いていてファイルを掴んでいる、など）は、
      **作り直せない物（退避とバックアップ）だけを 1 つずつ移す** キャッシュは
      作り直せ、実行環境は画面のボタンで入れ直せる 大きな物を写して起動を止める
      よりよい 旧い置き場が残っている間は、次の起動からも拾い上げを続ける
      （旧版がその後に書いた退避と控えを取り残さないため）
    - Windows 以外でキャッシュや実行環境が退避と別の場所にあるときは、それぞれ
      移すだけにする 移せなければ作り直させる

    失敗しても起動は止めない 引き継げなかった物は既定から始まるだけで、旧い
    置き場の中身は残っている
    """
    notes: list[MigrationNote] = []
    for legacy in LEGACY_APP_FOLDERS:
        state = state_root()
        found = [
            _copy_once(config_root(legacy), config_root()),
            _move_state(state_root(legacy), state, legacy),
        ]
        for old, new in ((cache_root(legacy), cache_root()), (data_root(legacy), data_root())):
            # Windows ではどちらも退避の置き場の中にあり、上で一緒に移っている
            if not new.is_relative_to(state):
                found.append(_move_once(old, new))
        notes.extend(note for note in found if note is not None)
    return notes


def _needs_migration(old: Path, new: Path) -> bool:
    # 新しい側が在れば、中身が空でも引き継ぎは済んだものとする 空だからと写すと、
    # 本人が新しい版で片付けた物が旧い置き場から戻ってくる
    return old.is_dir() and not new.exists() and old != new


def _copy_once(old: Path, new: Path) -> MigrationNote | None:
    """写す 書き終えてから新しい名前を付ける

    直に新しい名前へ写すと、途中で落ちたときに半分だけの置き場ができ、
    次の起動は「もう在る」と見て残りを引き継がない
    """
    if not _needs_migration(old, new):
        return None
    temporary = _temporary_beside(new)
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(old, temporary)
        # 付け替えは同時に起動したもう 1 つと競う 先に付けた方が勝ち、負けた方は
        # 自分の写しを捨てる（中身は同じ）
        temporary.rename(new)
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        return MigrationNote(old, new, "failed", str(exc))
    return MigrationNote(old, new, "copied")


def _move_once(old: Path, new: Path) -> MigrationNote | None:
    """移す 移せなければ何もしない（作り直せる物にだけ使う）"""
    if not _needs_migration(old, new):
        return None
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
    except OSError as exc:
        return MigrationNote(old, new, "failed", str(exc))
    return MigrationNote(old, new, "moved")


def _move_state(old: Path, new: Path, legacy: str) -> MigrationNote | None:
    """退避の置き場を移す 丸ごと付け替えられなければ、作り直せない物を拾い上げる"""
    marker = new / f"{_PARTIAL_PREFIX}{legacy}"
    started = _needs_migration(old, new)
    if started:
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        except OSError:
            pass
        else:
            return MigrationNote(old, new, "moved")
        # 印を入れた空の置き場を作ってから名前を付ける 名前だけ先に付けると、
        # 印を置く前に落ちたとき「引き継ぎ済み」に見えて拾い上げが止まる
        temporary = _temporary_beside(new)
        try:
            temporary.mkdir()
            (temporary / marker.name).write_text(str(old), encoding="utf-8")
            temporary.rename(new)
        except OSError as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            return MigrationNote(old, new, "failed", str(exc))
    if not marker.is_file():
        return None
    if not old.is_dir():
        # 旧い置き場が片付けられた もう拾う物は無い
        marker.unlink(missing_ok=True)
        return None
    moved = _salvage(old, new)
    # 2 回目からは、拾った物があったときだけ知らせる 毎回出すと起動のたびに記録が増える
    if moved or started:
        return MigrationNote(old, new, "moved-partly", f"{moved} 件")
    return None


def _salvage(old: Path, new: Path) -> int:
    """退避とバックアップを 1 つずつ新しい置き場へ移す 移した数を返す

    写さずに移すのは、新しい版で捨てた退避や消えた控えが、次の起動で旧い置き場から
    戻ってこないようにするため 移し先に同じ名前があれば触らない（名前は時刻や
    起動ごとの番号を含むので、同じ名前なら同じ物）
    """
    moved = 0
    for folder in _SALVAGED_DIRS:
        source = old / folder
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file() or not _can_take(path, folder):
                continue
            target = new / path.relative_to(old)
            if target.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
            except OSError:
                # 旧版が書いている最中などで掴まれている 次の起動でまた試す
                continue
            moved += 1
    return moved


def _can_take(path: Path, folder: str) -> bool:
    """拾い上げてよいファイルか

    - 錠（``.lock``）は移さない 開いているプロセスが握っていて初めて意味があり、
      移すと持ち主の居ない錠になる
    - 書きかけ（``<名前>.<印>.writing`` と前の版の ``<名前>.writing``）は移さない 途中の中身を
      完成品として扱うことになる 名前の形は :mod:`sashimono.core.io.atomic` が決める
    - 旧版で**まだ動いている起動**の退避は移さない 移すと、その窓がまだ開いているのに
      新しい版が「落ちた起動の退避」として復元を勧める 落ちたあとの起動で拾う
    """
    if path.suffix in (".lock", ".writing"):
        return False
    if folder != "recovery":
        # 空の控えは移さない 旧版の控えは、名前を空のファイルで押さえてから中身を写す
        # その間に移すと、新しい置き場には空の控えが残り、旧版は元の場所へ中身を書く
        # 写している最中は旧版がファイルを開いているので、Windows では移せずに飛ばされる
        try:
            return path.stat().st_size > 0
        except OSError:
            return False
    # core.io は置き場を決めるためにここを読むので、上で読むと読み込みが輪になる
    from sashimono.core.io.locks import is_held

    session = path.name.split(".", 1)[0]
    return not is_held(path.parent / f"{session}.lock")


def _temporary_beside(new: Path) -> Path:
    """新しい置き場の隣の作業用の名前 前の起動が残した物があれば片付けてから返す"""
    temporary = new.with_name(f"{new.name}.migrating-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    return temporary
