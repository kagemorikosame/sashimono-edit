"""追加機能の実行環境を、ソフト内から導入する

字幕起こしも AI 連携も、依存が重い（前者は 2 GB 超、後者は Claude Code 本体を
要求する） これを最初から同梱すると、その機能を使わない人にまで負担させることに
なるので、**初期状態では未導入**とし、必要になった時点で画面のボタンから入れる

機能ごとに :class:`FeaturePack` を 1 つ定義する 導入の手順・状態の見せ方・ログの
流し方は全部の機能で同じなので、ここに 1 つだけ置く

パッケージ版（PyInstaller）では ``sys.executable`` がアプリ本体になり、そこへは
書き込めない その場合は ``--target`` で専用フォルダへ入れ、起動時にそのフォルダを
``sys.path`` へ足す :func:`activate_runtime` がその役目を負う
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from packaging.version import InvalidVersion, Version

from sashimono.core import userdirs

__all__ = [
    "ABI_MARKER",
    "FeaturePack",
    "PackStatus",
    "PackageStatus",
    "activate_runtime",
    "app_dir",
    "install_arguments",
    "install_command",
    "install_runtime",
    "is_frozen",
    "pip_arguments",
    "python_abi",
    "refresh_runtime",
    "remove_stale_metadata",
    "restart_note",
    "run_pip",
    "runtime_abi",
    "runtime_target_dir",
    "snapshot_runtime_modules",
    "stale_runtime",
]

#: 導入したものを置くフォルダの名前（パッケージ版のみ）
_RUNTIME_DIR = "runtime"

#: 導入したときの Python の ABI を書いておくファイル（導入先の中）
#: 本体の更新で Python が上がると（3.14 → 3.15）、導入先の拡張モジュール（CTranslate2 など）は
#: 読めなくなる 何向けに入れたかを覚えておかないと、import して落ちるまで分からない
ABI_MARKER = ".python-abi"

#: 拡張モジュールの名前に入る ABI の印（``_ext.cp314-win_amd64.pyd`` の ``cp314``）
_ABI_TAG = re.compile(r"\.(cp3\d+)-")


@dataclass(frozen=True, slots=True)
class PackageStatus:
    """1 つのパッケージの導入状況"""

    name: str
    version: str | None
    #: ``>=`` で求めた最低の版 無ければ入っているだけでよい
    minimum: str | None = None

    @property
    def installed(self) -> bool:
        """使える版が入っているか 古い版は入っていないのと同じに扱う

        名前だけを見ると、前の条件で入れた古い版でも「導入済み」になる
        新しい版にしか無い引数を渡した所で、会話を始めた瞬間に落ちる
        """
        return self.version is not None and not self.outdated

    @property
    def outdated(self) -> bool:
        """入ってはいるが、求める版より古い"""
        if self.version is None or self.minimum is None:
            return False
        return _is_older(self.version, self.minimum)


@dataclass(frozen=True, slots=True)
class FeaturePack:
    """1 つの機能を動かすのに要るもの

    ``extra`` は「あると良いが無くても動く」もの 字幕起こしの CUDA ランタイムが
    これにあたり、外せば導入量を大きく減らせる
    """

    key: str
    label: str
    #: pip の指定 名前だけでも、バージョン条件付きでもよい
    required: tuple[str, ...]
    extra: tuple[str, ...] = ()
    #: ``extra`` を入れると何ができるようになるか
    extra_label: str = ""
    #: おおよその導入量（MB） 何が起きるかを先に見せるために使う
    size_mb: int = 0
    extra_size_mb: int = 0
    #: PATH 上に必要な外部コマンド pip では入らないものを表す
    commands: tuple[str, ...] = ()
    #: 外部コマンドが無いときの案内
    command_hint: str = ""
    #: コマンドの探し方 既定は PATH だけ PATH に載らない場所へ入る
    #: ものがあるので、機能ごとに差し替えられるようにしてある
    locate: Callable[[str], object | None] = shutil.which

    def status(self) -> PackStatus:
        return PackStatus(
            pack=self,
            packages=tuple(_package_status(n) for n in self.required),
            extras=tuple(_package_status(n) for n in self.extra),
            missing_commands=tuple(c for c in self.commands if self.locate(c) is None),
            stale_abi=stale_runtime(self.key),
        )

    def requirements(self, *, extra: bool) -> tuple[str, ...]:
        return self.required + (self.extra if extra else ())


@dataclass(frozen=True, slots=True)
class PackStatus:
    """機能が動く状態にあるか"""

    pack: FeaturePack
    packages: tuple[PackageStatus, ...]
    extras: tuple[PackageStatus, ...] = ()
    #: PATH に見つからなかった外部コマンド
    missing_commands: tuple[str, ...] = field(default_factory=tuple)
    #: 導入先が別の Python 向けに入っているときの、その ABI（``cp314``）
    #: 本体の更新で Python が上がった後に立つ 立っている間は導入先を読まない
    #: （:func:`activate_runtime`）
    stale_abi: str | None = None

    @property
    def installed(self) -> bool:
        """pip で入るものが揃っているか"""
        return all(p.installed for p in self.packages)

    @property
    def extra_installed(self) -> bool:
        return self.installed and all(p.installed for p in self.extras)

    @property
    def ready(self) -> bool:
        """実際に動かせるか 外部コマンドも含めて見る

        別の Python 向けに入った物も動かせない 片方の機能だけを入れ直すと導入先は読まれる
        ようになり、もう片方も名前の上では「入っている」に見えるが、古い拡張モジュールの
        import で落ちる
        """
        return self.installed and not self.missing_commands and self.stale_abi is None

    @property
    def needs_upgrade(self) -> bool:
        """古い版を入れ替える必要があるか

        pip は ``--target`` に同じ名前が在ると、``--upgrade`` 無しでは入れ替えない
        （配布版の導入先） 付けないと、入れ直しても古い版のまま残る 別の Python 向けに
        入っている物も同じで、付けないと名前が在るだけで飛ばされ、読めない拡張モジュールが残る
        """
        return self.stale_abi is not None or any(p.outdated for p in self.packages + self.extras)

    def missing(self, *, extra: bool) -> tuple[str, ...]:
        """まだ入っていないものの pip 指定"""
        pending = [p.name for p in self.packages if not p.installed]
        if extra:
            pending.extend(p.name for p in self.extras if not p.installed)
        return tuple(pending)

    def download_mb(self, *, extra: bool) -> int:
        total = self.pack.size_mb if not self.installed else 0
        if extra and not self.extra_installed:
            total += self.pack.extra_size_mb
        return total

    def summary(self) -> str:
        """画面に 1 行で出す説明"""
        if self.stale_abi is not None:
            # 「未導入」と出すと、2 GB が消えたように見える 消してはいない 入れ直す理由を言う
            return (
                f"入っている環境は前の Python（{self.stale_abi}）向けで、この版では読み込めません"
                " ここから入れ直してください"
            )
        if any(p.outdated for p in self.packages):
            old = "、".join(f"{_name_of(p.name)} {p.version}" for p in self.packages if p.outdated)
            return f"古い版が入っています（{old}） ここから入れ直せます"
        if not self.installed:
            return "未導入 ここから環境を用意できます"
        if self.missing_commands:
            missing = "、".join(self.missing_commands)
            hint = f" {self.pack.command_hint}" if self.pack.command_hint else ""
            return f"導入済み ただし {missing} が見つかりません {hint}"
        if self.extras and not self.extra_installed:
            return f"導入済み {self.pack.extra_label}は入っていません"
        return "導入済み"


def _name_of(requirement: str) -> str:
    """``faster-whisper>=1.1`` のような指定から配布名だけを取り出す"""
    for separator in (">=", "<=", "==", "~=", ">", "<", "[", "!"):
        index = requirement.find(separator)
        if index > 0:
            return requirement[:index].strip()
    return requirement.strip()


def _package_status(requirement: str) -> PackageStatus:
    minimum = None
    if ">=" in requirement:
        minimum = requirement.split(">=", 1)[1].split(",", 1)[0].strip() or None
    return PackageStatus(requirement, _version(_name_of(requirement)), minimum)


def _is_older(version: str, minimum: str) -> bool:
    """``version`` が ``minimum`` より古いか 版の決まり（PEP 440）どおりに比べる

    数字だけを拾って比べると ``0.2.152rc1`` を ``0.2.152`` と同じに読み、
    まだ出ていない版の前触れを「条件を満たす」と見てしまう
    読めない版は古いと見ない 入っている物を使えないと決めつけて止めるより、
    使わせてみて失敗の文面を見せる方が、次にすることが分かる
    """
    try:
        return Version(version) < Version(minimum)
    except InvalidVersion:
        return False


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def is_frozen() -> bool:
    """PyInstaller などで固めた実行ファイルとして動いているか"""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path | None:
    """配った zip を展開したフォルダ（``Sashimono.exe`` の置き場） 通常の実行では ``None``

    利用者が手で触る物（スクリプト置き場）はここに置く ``_internal`` の中は
    PyInstaller の持ち物で、更新のたびに丸ごと置き換わる
    """
    if not is_frozen():
        return None
    return Path(sys.executable).resolve().parent


def pip_arguments(argv: Sequence[str]) -> list[str] | None:
    """``Sashimono.exe -m pip ...`` と呼ばれたときの pip への引数 それ以外は ``None``

    パッケージ版には Python の本体が無い 導入ボタンは ``sys.executable -m pip`` を
    呼ぶが、パッケージ版の ``sys.executable`` は ``Sashimono.exe`` 自身なので、
    ここで受けて pip を動かさないと、**導入するつもりで Sashimono がもう 1 つ起動する**

    通常の実行では受けない ``sys.executable`` が本物の Python なので、そちらが
    pip を動かす
    """
    if not is_frozen():
        return None
    if list(argv[1:3]) != ["-m", "pip"]:
        return None
    return list(argv[3:])


def runtime_target_dir() -> Path | None:
    """導入先の専用フォルダ 通常の実行では ``None``（動いている環境へ直接入れる）"""
    if not is_frozen():
        return None
    return userdirs.data_root() / _RUNTIME_DIR


def activate_runtime() -> Path | None:
    """専用フォルダへ入れたものを import できるようにする

    起動時に 1 度呼ぶ 通常の実行では何もしない 導入の直後は
    :func:`refresh_runtime` を呼ぶ（こちらも中で呼ばれる）
    """
    target = runtime_target_dir()
    if target is None or not target.exists():
        return None
    if stale_runtime() is not None:
        # 別の Python 向けに入れた物は読まない 道へ足すと、import した所で拡張モジュールが
        # 読めずに落ちる（字幕起こしを始めた瞬間・AI に送った瞬間） 消しもしない 2 GB を
        # 黙って捨てて落とし直させないため 入れ直しの案内は導入の欄が出す（PackStatus.summary）
        return None
    path = str(target)
    if path not in sys.path:
        # 先頭へ入れる 同名の古いものが同梱されていた場合に、あとから入れた方を
        # 使わせるため
        sys.path.insert(0, path)
    read_path_files(path)
    return target


def python_abi() -> str:
    """動いている Python の ABI の印（``cp314``） 更新の目録の ``python_abi`` と同じ書き方

    版の上 2 つだけで決まる 3.14.1 と 3.14.6 は同じ拡張モジュールを読める
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


#: 印を書くようになる前に入れた物（どの機能の物か分からない）をまとめて表す名前
_EVERY_PACK = "*"


def _read_marks(target: Path) -> dict[str, str]:
    """導入先の印 機能（:attr:`FeaturePack.key`）→ ABI 読めなければ空"""
    try:
        data = json.loads((target / ABI_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, str) and v}


def _write_marks(target: Path, marks: Mapping[str, str]) -> None:
    # 書けなくても、名前の印から読み直せる
    with contextlib.suppress(OSError):
        (target / ABI_MARKER).write_text(json.dumps(dict(sorted(marks.items()))), encoding="utf-8")


def _scan_abi(target: Path) -> str | None:
    """拡張モジュールの名前の印を数え、いちばん多い物 CTranslate2 も pydantic-core も
    名前に ``cp314`` を持つ 無ければ分からない（``None``）
    """
    counts: dict[str, int] = {}
    try:
        # 包みの直下までで足りる 深く辿ると、数千のファイルを起動のたびに見ることになる
        candidates = [*target.glob("*.pyd"), *target.glob("*/*.pyd")]
    except OSError:
        return None
    for path in candidates:
        found = _ABI_TAG.search(path.name)
        if found is not None:
            counts[found.group(1)] = counts.get(found.group(1), 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda tag: counts[tag])


def runtime_abi(target: Path, key: str | None = None) -> str | None:
    """導入先（``key`` を渡せばその機能）が何の Python 向けか 分からなければ ``None``

    導入先は字幕起こしと AI 連携で 1 つを分け合う 機能ごとに印を持つのは、片方だけを
    入れ直した後に、もう片方の古い拡張モジュールを「合っている」と見ないため

    印を書くようになる前に入れた導入先は、名前の印を数えて決め、全部の機能の物として
    書き残す 書き残さないと、片方を入れ直した後は新旧の拡張モジュールが混ざり、数えても
    もう片方が古いことが分からない 分からない物を「別の Python 向け」と決めると、
    動いている導入先を読まなくなるので ``None`` にする
    """
    marks = _read_marks(target)
    if not marks:
        scanned = _scan_abi(target)
        if scanned is None:
            return None
        marks = {_EVERY_PACK: scanned}
        _write_marks(target, marks)
    if key is not None:
        return marks.get(key, marks.get(_EVERY_PACK))
    # 導入先そのものは、今の Python 向けの機能が 1 つでもあれば読める
    current = python_abi()
    return current if current in marks.values() else min(marks.values())


def stale_runtime(key: str | None = None) -> str | None:
    """導入先（``key`` を渡せばその機能）が今の Python と違う向けなら、その ABI

    合っている・入っていない・分からないなら ``None``
    """
    target = runtime_target_dir()
    if target is None or not target.is_dir():
        return None
    found = runtime_abi(target, key)
    if found is None or found == python_abi():
        return None
    return found


def _mark_installed(target: Path, key: str) -> None:
    """入れ終えた機能に今の Python の印を付ける"""
    marks = _read_marks(target)
    marks[key] = python_abi()
    _write_marks(target, marks)


def read_path_files(place: str) -> None:
    """置き場の ``.pth`` を読む（``site.addsitedir`` と同じ） 置き場そのものは足し直さない

    pip の ``--target`` で入れた置き場は、Python が起動のときに読む場所ではないので、
    ``.pth`` がそのままでは読まれない pywin32（mcp が Windows で頼む）は ``.pth`` で
    ``win32`` ``win32/lib`` を探す道へ足し、DLL の置き場を登録する 読まないと
    ``No module named 'pywintypes'`` で AI 連携が動かない（組み立ての確かめで分かった）
    """
    import site

    try:
        site.addsitedir(place)
    except OSError:
        return
    if not is_frozen():
        return
    # pywin32 の pywintypes は、固めた exe の中では DLL（``pywintypes314.dll``）を
    # **探す道の上でだけ**探す 入れた置き場では DLL が ``pywin32_system32`` にあり、``.pth`` は
    # そこを DLL の置き場として登録するだけで探す道へは足さない 足さないと mcp の import で
    # ``Module 'pywintypes' isn't in frozen sys.path`` と言って AI 連携が動かない（0.1.0 の zip）
    # PyInstaller が pywin32 を積むときに足す差し込み（pyi_rth_pywintypes）と同じことをする
    # 配布版は pywin32 を積まない（tools/build_package.py の EXCLUDED_MODULES）ので、ここで足す
    dlls = Path(place) / "pywin32_system32"
    if dlls.is_dir() and str(dlls) not in sys.path:
        sys.path.append(str(dlls))


def refresh_runtime(before: Mapping[str, int] | None = None) -> tuple[str, ...]:
    """導入を終えた直後に呼び、入れたものを再起動なしで使えるようにする

    戻り値は「入れ直したのに、古い方がもう読み込まれていて入れ替えられなかった」
    モジュールの名前 空でなければ、再起動を勧める

    ``before`` は導入を始める前の :func:`snapshot_runtime_modules` 渡すと、専用
    フォルダから読み込み済みの物が同じ場所で上書きされたことも見分けられる

    これが無いと配布版では導入が済んだことに気付けなかった 初めて導入する人は
    起動時に専用フォルダがまだ無いので :func:`activate_runtime` が何もせず、
    導入のあとも import の道に載らないまま、状態を見直しても「未導入」と出て
    ボタンが押せなかった（Issue #27）

    通常の実行でも import の控えは捨てる 導入先（site-packages）の中身を
    覚えている探し手が、入れたばかりのパッケージを見落とすことがあるため
    """
    target = activate_runtime()
    if target is not None:
        remove_stale_metadata(target)
    # パッケージの探し手と、配布メタデータ（導入状況の判定に使う）の探し手の
    # 両方の控えがここで捨てられる
    importlib.invalidate_caches()
    if target is None:
        return ()
    loaded = list(_already_loaded_elsewhere(target))
    for name in _replaced_in_place(before or {}):
        if name not in loaded:
            loaded.append(name)
    return tuple(loaded)


def snapshot_runtime_modules() -> dict[str, int]:
    """専用フォルダから読み込み済みのモジュールと、そのファイルの更新時刻（ns）

    導入を始める前に呼び、結果をその導入の :func:`refresh_runtime` へ渡す
    導入ごとに持たせるのは、字幕起こしとアシスタントの導入が重なっても、
    片方の控えがもう片方に上書きされないため
    """
    target = runtime_target_dir()
    if target is None or not target.exists():
        return {}
    root = target.resolve()
    found: dict[str, int] = {}
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        path = Path(location)
        try:
            if path.resolve().is_relative_to(root):
                found[name] = path.stat().st_mtime_ns
        except OSError:
            continue
    return found


def _replaced_in_place(before: Mapping[str, int]) -> list[str]:
    """専用フォルダから読み込み済みだったのに、導入で同じ場所のファイルが入れ替わった物

    ``invalidate_caches`` は ``sys.modules`` の読み込み済みの物を入れ替えない
    同じ場所へ新しい版を上書きすると、場所の比べ方では見分けられず、古い版の
    まま動いているのに「再起動しなくても使えます」と言ってしまう
    """
    replaced: list[str] = []
    for name, stamp in before.items():
        module = sys.modules.get(name)
        location = getattr(module, "__file__", None) if module is not None else None
        if location is None:
            continue
        try:
            changed = Path(location).stat().st_mtime_ns != stamp
        except OSError:
            changed = True  # 消えた 入れ替えで無くなった
        top = name.split(".", 1)[0]
        if changed and top not in replaced:
            replaced.append(top)
    return replaced


def remove_stale_metadata(target: Path) -> tuple[Path, ...]:
    """専用フォルダに残った古い版の ``*.dist-info`` を消す 戻り値は消した物

    pip の ``--target --upgrade`` は、同じ名前の項目しか入れ替えない 版が違えば
    ``*.dist-info`` のフォルダ名も違うので、古い版のメタデータが残る
    ``importlib.metadata`` は最初に見つけた方を返すので、古い方を拾うと、
    入れ直したのに「古い版が入っています」のまま使えない

    同じ配布名の物が 2 つ以上あるときだけ、一番新しい版を残して消す
    版を読めない物が混ざるときは、どれが新しいか決められないので触らない
    """
    groups: dict[str, list[tuple[Version, Path]]] = {}
    unreadable: set[str] = set()
    try:
        entries = list(target.glob("*.dist-info"))
    except OSError:
        return ()
    for entry in entries:
        name, _, version = entry.name[: -len(".dist-info")].partition("-")
        key = name.lower().replace("-", "_").replace(".", "_")
        try:
            groups.setdefault(key, []).append((Version(version), entry))
        except InvalidVersion:
            unreadable.add(key)
    removed: list[Path] = []
    for key, found in groups.items():
        if len(found) < 2 or key in unreadable:
            continue
        found.sort(key=lambda item: item[0])
        for _, stale in found[:-1]:
            try:
                shutil.rmtree(stale)
            except OSError:
                # 使用中などで消せなくても導入は済んでいる 次の起動で消える
                # 機会があるので、ここでは止めない
                continue
            removed.append(stale)
    return tuple(removed)


def restart_note(loaded: Sequence[str], *, visible: bool = True) -> str:
    """導入のあとに出す 1 行 再起動が要るかどうかがそのまま分かるようにする

    ``visible`` が偽なら、pip は通ったのに入れたものが見つからなかった
    黙って押せないボタンを残すより、次にできることを書く
    """
    if not visible:
        return (
            "導入は終わりましたが、入れたものを読み込めませんでした"
            " ソフトを再起動してからもう一度開いてください"
        )
    if not loaded:
        return "導入が終わりました 再起動しなくてもそのまま使えます"
    names = "、".join(loaded[:5]) + (" ほか" if len(loaded) > 5 else "")
    return (
        "導入が終わりました そのまま使えますが、同梱の部品（"
        f"{names}）を入れ直したので、うまく動かないときはソフトを再起動してください"
    )


def _already_loaded_elsewhere(target: Path) -> tuple[str, ...]:
    """専用フォルダに入ったのに、別の場所から読み込み済みのモジュール

    配布版に同梱したもの（numpy など）を、導入した機能の依存としてもう 1 つ
    入れることがある 起動し直すと専用フォルダの方が先に見つかるが、今の実行では
    同梱の方が ``sys.modules`` に残っていて入れ替わらない
    """
    loaded: list[str] = []
    try:
        entries = sorted(target.iterdir())
    except OSError:
        return ()
    root = target.resolve()
    for entry in entries:
        name = _module_name(entry)
        if name is None:
            continue
        # 下のモジュールまで見る 名前空間パッケージ（``nvidia`` など）は親に
        # ``__file__`` が無く、親だけを見ると、同梱の方から読み込み済みの
        # 子を見落として「再起動しなくても使えます」と言ってしまう
        prefix = f"{name}."
        for module_name, module in list(sys.modules.items()):
            if module_name != name and not module_name.startswith(prefix):
                continue
            location = getattr(module, "__file__", None)
            if location is None:
                continue
            if not Path(location).resolve().is_relative_to(root):
                loaded.append(name)
                break
    return tuple(loaded)


def _module_name(entry: Path) -> str | None:
    """専用フォルダの 1 項目が、何という名前で import されるか"""
    if entry.is_dir():
        if entry.suffix in {".dist-info", ".egg-info", ".data"} or entry.name in {
            "bin",
            "__pycache__",
        }:
            return None
        return entry.name
    if entry.suffix in {".py", ".pyd"}:
        return entry.name.split(".", 1)[0]
    return None


def run_pip(arguments: Sequence[str]) -> int:
    """配布版の中で pip を走らせる :func:`pip_arguments` が受けたときに使う

    pip が中で使う distlib は、同梱の部品（``t64.exe`` など）を
    **読み込み方式ごとの探し方**で見つける PyInstaller の読み込み方式は
    distlib の一覧に無いので、``pip install`` は部品を探す所で落ちる
    （``pip --version`` は部品を探さないので通ってしまう 実物で確かめた）

    配布版では部品が ``_internal`` の下にファイルとして置いてあるので、
    ファイルとして探す方式を割り当てれば見つかる
    """
    from pip._vendor import distlib
    from pip._vendor.distlib import resources

    loader = getattr(distlib, "__loader__", None)
    if loader is not None:
        # distlib は型を配っていない 呼び方は distlib 0.3 系の resources.py で確かめた
        resources.register_finder(loader, resources.ResourceFinder)  # type: ignore[no-untyped-call]

    from pip._internal.cli.main import main as pip_main

    return int(pip_main(list(arguments)))


def install_command(
    pack: FeaturePack, *, extra: bool = True, upgrade: bool = False, python: str | None = None
) -> list[str]:
    """導入に使う ``pip`` のコマンド列を組み立てる

    実行せずに文字列として得られるようにしてあるのは、画面に「これを実行します」と
    出すため 何が入るのか分からないままダウンロードが始まるのは不安が大きい
    """
    return [
        python or sys.executable,
        "-m",
        "pip",
        *install_arguments(pack, runtime_target_dir(), extra=extra, upgrade=upgrade),
    ]


def install_arguments(
    pack: FeaturePack, target: Path | None, *, extra: bool = True, upgrade: bool = False
) -> list[str]:
    """``pip`` へ渡す引数（``install`` から） ``target`` は配布版の導入先 開発の環境では ``None``

    配る zip の確かめ（:mod:`sashimono.addon_check` 組み立ての道具と CI が呼ぶ）も、ここで
    組んだ引数で exe の pip に入れる 導入ボタンと違う入れ方で確かめると、使う人の手元で
    落ちる物を見落とす

    配布版では、ソースの形（sdist）しか無い版を選ばせない（``--only-binary :all:``）
    pip はソースから組むとき ``sys.executable`` に自分の起動部を渡して子を立てるが、
    配布版の ``sys.executable`` は Sashimono.exe で、その起動を pip と見分けられず編集画面が
    裏で立ち、導入が終わらなくなる（claude-agent-sdk 0.2.163 が sdist だけで出ていた日に
    0.1.0 の zip で起きた） 組めたとしても、配布版の中には組むための道具が無い
    wheel だけに絞れば、pip は wheel のある一番新しい版を選ぶ
    """
    arguments = ["install"]
    if upgrade:
        arguments.append("--upgrade")
    if target is not None:
        arguments.extend(["--only-binary", ":all:", "--target", str(target)])
    arguments.extend(pack.requirements(extra=extra))
    return arguments


def install_runtime(
    pack: FeaturePack | None = None,
    *,
    extra: bool = True,
    on_output: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    command: Sequence[str] | None = None,
) -> int:
    """``pip`` を子プロセスで走らせる 戻り値は終了コード（0 が成功）

    出力は 1 行ずつ ``on_output`` へ渡す まとめて最後に渡すと、数分間なにも
    起きていないように見える
    """
    if command is None:
        if pack is None:
            raise ValueError("pack か command のどちらかが要る")
        command = install_command(pack, extra=extra)
    argv = list(command)
    if on_output is not None:
        on_output("> " + " ".join(argv))

    target = runtime_target_dir()
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        # 入れる前の中身の印を書き残す（印が無い導入先だけ） 入れた後に数えると、
        # 今入れた物の印と、入れ直していない機能の古い印が混ざって決められない
        runtime_abi(target)

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # 子プロセスの出力を UTF-8 に揃える Windows の既定は cp932 で、素材やユーザー名に
    # 日本語が入っているとログが文字化けし、失敗の原因が読めなくなる
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        # 引数はここで組み立てたものだけで、shell も通さない
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
            env=child_env,
        )
    except OSError as exc:
        if on_output is not None:
            on_output(f"pip を起動できない: {exc}")
        return 1

    assert process.stdout is not None
    finished = threading.Event()
    cancelled = threading.Event()

    def watch(ask: Callable[[], bool]) -> None:
        # 出力を読む所とは別に見張る 読む所は次の 1 行が来るまで止まるので、
        # そこで中断を見ると、pip が黙って落としている間（数分ある）は止まらない
        while not finished.wait(_CANCEL_POLL_SECONDS):
            if ask():
                cancelled.set()
                try:
                    process.terminate()
                except OSError:
                    return  # もう終わっていた
                return

    if should_cancel is not None:
        threading.Thread(target=watch, args=(should_cancel,), daemon=True).start()
    with process:
        # 止めた後も最後まで読む 途中で読むのをやめると、子の書き込みが詰まって
        # 終わらず、終了コードも取れない
        for line in process.stdout:
            if on_output is not None:
                on_output(line.rstrip())
    finished.set()
    if cancelled.is_set() and on_output is not None:
        on_output("中断した")
    code = process.returncode if process.returncode is not None else 1
    if code == 0 and target is not None and pack is not None:
        # 入れ終えたときにだけ書く 途中で止めた導入先に今の印を書くと、前の Python 向けの
        # 拡張モジュールが残ったまま「合っている」として読まれる
        _mark_installed(target, pack.key)
    return code


#: 導入の中断の頼みを見る間隔（秒） 長いと、閉じるボタンを押してから止まるまでが延びる
_CANCEL_POLL_SECONDS = 0.1
