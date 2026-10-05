r"""新しい版の zip を落とし、確かめ、入れ替える前のフォルダへ展開する

置き場は今の版のフォルダ（``Sashimono.exe`` のある所）の隣に並べる

    <親>\Sashimono            今の版
    <親>\Sashimono.new        展開し終えた新しい版（入れ替え待ち）
    <親>\Sashimono.previous   1 つ前の版（戻すため 1 世代だけ残す）
    <親>\Sashimono.failed     入れたら起動できずに外した版（次の入れ替えで消える）

**同じ親の下に置く** 入れ替えは改名だけで行う（:mod:`.swap`） 同じドライブの中の改名なら
一瞬で終わり、途中で止まっても混ざった状態が残らない 別の場所へ展開すると、入れ替えが
コピーになり、途中で止まったときに半分だけ新しいフォルダが残る

既定の置き場は ``%LOCALAPPDATA%\Programs\Sashimono`` 本人の権限で書ける場所なので、
入れ替えのたびに管理者の確認が出ない ``Program Files`` のように親へ書けない場所に
置かれていたら、自動では入れ替えず、配布のページを案内する
"""

from __future__ import annotations

import contextlib
import filecmp
import json
import os
import shutil
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sashimono.core import userdirs
from sashimono.runtime import app_dir
from sashimono.update.fetch import Transport, download
from sashimono.update.manifest import MAX_PACKAGE_BYTES, Manifest
from sashimono.update.portable import (
    PORTABLE_SCRIPTS_DIR,
    bundles,
    clean_leftovers,
    hold_move_lock,
    is_link,
    module_stem,
    modules_in,
    remove_file,
    script_links,
    user_script_files,
)

__all__ = [
    "APP_EXE",
    "BUILD_INFO_NAME",
    "BuildInfo",
    "CarryError",
    "Layout",
    "PackageError",
    "carry_user_files",
    "current_layout",
    "extract_package",
    "read_build_info",
    "stage",
    "write_build_info",
]

#: 本体の実行ファイルの名前（tools/build_package.py の APP_NAME と同じ）
APP_EXE = "Sashimono.exe"

#: zip に入れる、版と Python の ABI の書き付け 展開した物が目録の言う版かを確かめる
#: zip の名前や README では確かめない 名前は付け替えられ、README は人が読む文
BUILD_INFO_NAME = "build-info.json"

#: 展開した中身の大きさの上限 zip 爆弾で本人のディスクを埋めない
_MAX_EXTRACTED_BYTES = 2 * MAX_PACKAGE_BYTES

#: exe の隣にある、本人がスクリプトを置いてよいフォルダ 名前は移す側（.portable）と 1 つにする
_PORTABLE_SCRIPTS_DIR = PORTABLE_SCRIPTS_DIR


class PackageError(Exception):
    """展開できない・中身が目録と合わない"""


@dataclass(frozen=True, slots=True)
class BuildInfo:
    version: str
    python_abi: str


def write_build_info(folder: Path, version: str, python_abi: str) -> Path:
    """書き付けを置く 配る zip を組み立てる道具が使う"""
    path = folder / BUILD_INFO_NAME
    path.write_text(
        json.dumps({"version": version, "python_abi": python_abi}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_build_info(folder: Path) -> BuildInfo | None:
    try:
        data = json.loads((folder / BUILD_INFO_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    version, abi = data.get("version"), data.get("python_abi")
    if not isinstance(version, str) or not isinstance(abi, str):
        return None
    return BuildInfo(version, abi)


@dataclass(frozen=True, slots=True)
class Layout:
    """今の版のフォルダと、その隣に並べるフォルダ"""

    install: Path

    def _beside(self, suffix: str) -> Path:
        return self.install.with_name(self.install.name + suffix)

    @property
    def staged(self) -> Path:
        return self._beside(".new")

    @property
    def previous(self) -> Path:
        return self._beside(".previous")

    @property
    def failed(self) -> Path:
        return self._beside(".failed")

    @property
    def download(self) -> Path:
        return self._beside(".download.zip")

    @property
    def partial(self) -> Path:
        """展開している途中 書き終えてから :attr:`staged` へ名前を付ける"""
        return self._beside(".new-partial")

    @property
    def exe(self) -> Path:
        return self.install / APP_EXE

    def writable(self) -> bool:
        """隣にフォルダを作れるか 作れなければ自動では入れ替えられない"""
        probe = self._beside(".write-test")
        try:
            probe.mkdir()
            probe.rmdir()
        except OSError:
            return False
        return True

    def has_previous(self) -> bool:
        return (self.previous / APP_EXE).is_file()

    def staged_version(self) -> str | None:
        """展開し終えて入れ替えを待っている版 無ければ ``None``"""
        if not (self.staged / APP_EXE).is_file():
            return None
        info = read_build_info(self.staged)
        return info.version if info is not None else None


def current_layout() -> Layout | None:
    """配布版で動いているときの置き場 開発の環境（``python -m sashimono``）では ``None``

    開発の環境は git で新しくするので、自動では入れ替えない
    """
    folder = app_dir()
    return Layout(folder) if folder is not None else None


def _remove(path: Path) -> None:
    """フォルダかファイルを消す 読み取り専用の印が付いた物も消す"""

    def clear_and_retry(function: Callable[..., object], name: str, _exc: BaseException) -> None:
        Path(name).chmod(stat.S_IWRITE)
        function(name)

    if path.is_dir():
        shutil.rmtree(path, onexc=clear_and_retry)
    else:
        path.unlink(missing_ok=True)


def extract_package(archive: Path, target: Path) -> Path:
    """zip を ``target`` へ展開し、``Sashimono.exe`` のあるフォルダを返す

    中の名前を 1 つずつ確かめる ``..`` や絶対の名前を含む zip をそのまま展開すると、
    フォルダの外（本人の設定やスタートアップ）へ書かれる 署名と SHA-256 で守っていても、
    展開の側でも守る（鍵が漏れたときの最後の砦）
    """
    target.mkdir(parents=True)
    root = target.resolve()
    total = 0
    try:
        with zipfile.ZipFile(archive) as opened:
            for member in opened.infolist():
                name = PurePosixPath(member.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or ":" in member.filename:
                    raise PackageError(f"zip の中に外を指す名前がある: {member.filename}")
                total += member.file_size
                if total > _MAX_EXTRACTED_BYTES:
                    raise PackageError("展開すると大きすぎる")
                destination = (root / name).resolve()
                if not destination.is_relative_to(root):
                    raise PackageError(f"zip の中に外を指す名前がある: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with opened.open(member) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackageError(f"展開できない: {exc}") from exc
    # 配る zip は ``Sashimono\`` を 1 つ持つ形（tools/build_package.py の make_zip）
    if (root / APP_EXE).is_file():
        return root
    inner = [entry for entry in root.iterdir() if (entry / APP_EXE).is_file()]
    if len(inner) != 1:
        raise PackageError(f"zip の中に {APP_EXE} が無い")
    return inner[0]


def stage(
    manifest: Manifest,
    transport: Transport,
    layout: Layout,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """新しい版を落とし、確かめ、:attr:`Layout.staged` へ置く 置いた場所を返す

    確かめるのは 3 つ 大きさと SHA-256（目録と照らす 目録は署名で守られている）、
    展開した中の書き付けの版と Python の ABI（目録と照らす） どれかが合わなければ
    何も残さずに止まる（落とした物も消す） 次の確認でまた最初からやり直す
    """
    for leftover in (layout.download, layout.partial):
        _remove(leftover)
    try:
        download(
            manifest.package.url,
            transport,
            layout.download,
            size=manifest.package.size,
            sha256=manifest.package.sha256,
            should_cancel=should_cancel,
        )
        folder = extract_package(layout.download, layout.partial)
        info = read_build_info(folder)
        if info is None:
            raise PackageError(f"zip の中に {BUILD_INFO_NAME} が無い")
        if info.version != manifest.version:
            raise PackageError(f"zip の中の版（{info.version}）が目録（{manifest.version}）と違う")
        if info.python_abi != manifest.python_abi:
            raise PackageError(
                f"zip の中の Python（{info.python_abi}）が目録（{manifest.python_abi}）と違う"
            )
        _remove(layout.staged)
        folder.rename(layout.staged)
    except BaseException:
        _remove(layout.partial)
        raise
    finally:
        layout.download.unlink(missing_ok=True)
    _remove(layout.partial)
    return layout.staged


def carry_user_files(
    install: Path,
    destination: Path,
    *,
    overwrite: bool = False,
    aside: Path | None = None,
    config: Path | None = None,
) -> int:
    """今の版の exe の隣のスクリプト置き場に本人が置いた物を、入れ替え先の版へ写す 写した数を返す

    入れ替えはフォルダを丸ごと替えるので、写さないと前の版の案内どおりそこへ置いた
    スクリプトが 2 回目の入れ替えで消える（1 回目は ``previous`` に残るが、次で消える）
    **本人の物は今の版の側が正** 同梱の物（``portable.BUNDLED_SCRIPT_FILES``）は写さない

    - 前の版へ戻す（``overwrite``） 戻る先の ``previous`` には、更新する前に写した古い中身が
      残っている 今の側で直した物を飛ばすと、直した中身が戻した版に入らず、次の自動更新で
      ``.previous`` ごと消える 中身が違えば今の側で上書きする（古い中身は本人が直す前の物で、
      直した側が正なので取っておかない）
    - 新しい版を入れる（既定） 入れ替え先は展開したばかりの新しい版で、そこに在る物はすべて
      新しい版の同梱物 同梱物は新しい版の物を残す

    **束（:func:`.portable.bundles` 移す側と同じ単位）で振り分ける** スクリプトはモジュールを
    自分のフォルダから先に探すので、束の一部だけを別の置き場へ写すと、残った側が同じフォルダの
    別のモジュールを読み、更新しただけで描画が変わる（PR #245 の Codex の指摘）
    新しい版を入れるとき、束の中に新しい版の同梱物とぶつかる物（同じ名前で中身が違う、または
    モジュールの名前が同梱物と重なる）が 1 つでもあれば、束ごと ``aside``（``%APPDATA%`` の
    ``scripts``）へ写す 読む順は ``%APPDATA%`` が後で勝つので、今まで使われていた本人の物が
    使われ続ける ``aside`` にも同じ名前で中身の違う物があれば、どこへ写しても束が割れるか
    本人の物が落ちるので、写さずに :class:`CarryError` にする（入れ替えを止め、手で片付けてもらう）

    **1 つでも写せなければ :class:`CarryError`** 全部を試してから、写せなかった物を並べて上げる
    呼んだ側は入れ替えを止める 写せないまま入れ替えると、本人の物は次の更新で消える版
    （``previous`` か ``.new`` へよけた側）にだけ残る 止めるときは ``aside`` へ写した物を外す
    （``%APPDATA%`` は今の版でも読まれるので、束の一部だけが残ると今の版の描画が変わる）
    入れ替え先の版（``.new`` ``previous``）へ写した物はそのまま置く（今は読まれない）

    **移す側と同じ錠（``portable.MOVE_LOCK``）を、計画から止めるときの巻き戻しまで持つ**
    （``config`` は ``%APPDATA%\\Sashimono`` 既定は本人の置き場） 別の窓が移している最中に
    写すと、巻き戻しで写しが外れて両方から失われる（PR #245 の Codex の指摘） 移しは束ごとに
    短いので ``CARRY_LOCK_WAIT`` 秒まで待ち、取れなければ止める 錠を持ったら、落ちた起動が
    よけたまま残した元を元の場所へ戻してから写す（戻さずに入れ替えると、今の版のフォルダごと
    ``.previous`` へ回り、次の更新で消える）
    """
    folder = config if config is not None else userdirs.config_root()
    lock = hold_move_lock(folder, wait=CARRY_LOCK_WAIT)
    if lock is None:
        raise CarryError(
            (
                (
                    Path(PORTABLE_SCRIPTS_DIR),
                    "ほかの Sashimono の窓が exe の隣のスクリプトを移している",
                ),
            )
        )
    try:
        clean_leftovers(install, folder / PORTABLE_SCRIPTS_DIR)
        return _carry(install, destination, overwrite=overwrite, aside=aside)
    finally:
        lock.release()


#: 引き継ぐ前に、移している別の窓が錠を放すのを待つ長さ（秒） 移しは束 1 つずつ錠を
#: 持ち直さないので、全部を移し終えるまで待つことになる 長く待たせるより止めて知らせる
CARRY_LOCK_WAIT = 10.0


def _carry(install: Path, destination: Path, *, overwrite: bool, aside: Path | None) -> int:
    source = install / _PORTABLE_SCRIPTS_DIR
    root = destination / _PORTABLE_SCRIPTS_DIR
    files = user_script_files(install)
    links = script_links(install)
    linked = set(links)
    # 新しい版が同梱しているモジュール（名前で探されるので、場所が違っても重なる）
    bundled = {} if overwrite else modules_in(root)
    copied = 0
    failed: list[tuple[Path, str]] = []
    placed_aside: list[Path] = []
    for members in bundles([*files, *links]).values():
        where = root
        if not overwrite and _clashes(source, root, members, linked, bundled):
            if aside is None:
                failed.extend((p, "新しい版の同梱物と同じ名前の物がある") for p in members)
                continue
            if any(_differs(source / p, aside / p, p in linked) for p in members):
                failed.extend(
                    (p, "新しい版の同梱物と %APPDATA% の両方に、同じ名前の別の物がある")
                    for p in members
                )
                continue
            where = aside
        for relative in members:
            target = where / relative
            present = target.exists() or is_link(target)
            if present and not _differs(source / relative, target, relative in linked):
                continue  # 同じ物が在る
            if relative in linked:
                problem = _carry_link(source / relative, target, overwrite=overwrite)
            else:
                # %APPDATA% へは置き換えない 確かめた後に本人やほかの窓が同じ名前を置いていれば、
                # そちらを残して止める（%APPDATA% は今の版でも読まれる）
                problem = _copy_over(source / relative, target, replace=where is not aside)
            if problem is None:
                copied += 1
                if where is aside:
                    placed_aside.append(relative)
            else:
                failed.append((relative, problem))
    if failed:
        assert aside is not None or not placed_aside
        for relative in placed_aside:
            # 外すのは自分が置いた写しで、元が exe の隣に同じ中身で残っているときだけ（移す側の
            # 巻き戻しと同じ） 元が無ければ誰かがこの写しを当てにした 外すと両方から失われる
            target = aside / relative if aside is not None else relative
            origin = source / relative
            with contextlib.suppress(OSError):
                if relative in linked:
                    if _same_link(origin, target):
                        _remove_link(target)
                elif origin.is_file() and _same_file(origin, target):
                    remove_file(target)
        raise CarryError(tuple(failed), links=tuple(p for p, _r in failed if p in linked))
    return copied


def _differs(origin: Path, target: Path, link: bool) -> bool:
    """入れ替え先に、同じ名前の別の物が在るか 無ければ偽（写せばよい）"""
    if not target.exists() and not is_link(target):
        return False
    if link:
        return not _same_link(origin, target)
    return is_link(target) or not _same_file(origin, target)


def _clashes(
    source: Path,
    root: Path,
    members: list[Path],
    linked: set[Path],
    bundled: dict[str, list[Path]],
) -> bool:
    """束が新しい版の同梱物とぶつかるか 同じ名前で中身が違う物か、名前の重なるモジュールがある"""
    for relative in members:
        if _differs(source / relative, root / relative, relative in linked):
            return True
        stem = module_stem(relative)
        for other in bundled.get(stem, []) if stem is not None else []:
            if other.as_posix().casefold() != relative.as_posix().casefold():
                return True
    return False


def _same_link(origin: Path, target: Path) -> bool:
    """両方がリンクで、同じ先を指しているか"""
    if not is_link(target):
        return False
    try:
        return origin.readlink() == target.readlink()
    except OSError:
        return False


def _remove_link(path: Path) -> None:
    """リンクそのものだけを外す 先は辿らない（先の中身は消えない）

    外す前に本当にリンクかを確かめる 普通のフォルダを rmdir やファイルの unlink で消すと、
    本人の物を消すことになる ジャンクションとフォルダを指すシンボリックリンクは rmdir、
    ファイルを指すシンボリックリンクは unlink で外す どちらもリンクの先には触らない
    """
    if not is_link(path):
        raise OSError(f"リンクではないので外さない: {path.name}")
    if path.is_junction() or path.is_dir():
        path.rmdir()
    else:
        path.unlink()


def _carry_link(origin: Path, target: Path, *, overwrite: bool = False) -> str | None:
    """リンクを入れ替え先に作り直す 作ったら ``None``、作れなければ理由

    ``overwrite``（前の版へ戻す）では、入れ替え先に別の先を指すリンクがあれば、リンクそのもの
    だけを外して今の版の先へ付け直す（今の版の側が正 本人がジャンクションを付け替えた）
    リンクではない物（本当のフォルダやファイル）は外さない 本人の物かもしれない
    """
    try:
        pointed = origin.readlink()
    except OSError as exc:
        return str(exc) or type(exc).__name__
    if is_link(target):
        if not overwrite:
            return "入れ替え先に同じ名前の別のリンクがある"
        try:
            _remove_link(target)
        except OSError as exc:
            return str(exc) or type(exc).__name__
    elif target.exists():
        return "入れ替え先に同じ名前の、リンクではない物がある"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_junction():
            # ジャンクションは本人の権限で作れる（シンボリックリンクは開発者モードか管理者が要る）
            import _winapi  # type: ignore[import-not-found,unused-ignore]

            text = str(pointed)
            _winapi.CreateJunction(text.removeprefix("\\\\?\\"), str(target))
        else:
            target.symlink_to(pointed, target_is_directory=origin.is_dir())
    except (OSError, ImportError) as exc:
        return str(exc) or type(exc).__name__
    return None


class CarryError(Exception):
    """exe の隣の本人の物を、入れ替え先の版（か ``%APPDATA%``）へ写せなかった 入れ替えは止める"""

    def __init__(
        self, failed: tuple[tuple[Path, str], ...], *, links: tuple[Path, ...] = ()
    ) -> None:
        self.failed = failed
        #: 写せなかった物のうちリンク（シンボリックリンク・ジャンクション） 直し方が違う
        self.links = links
        first, reason = failed[0]
        super().__init__(
            f"exe の隣の scripts の {len(failed)} 個を写せなかった（{first.as_posix()}: {reason}）"
        )

    def explain(self) -> str:
        """本人に見せる文 止めたこと・今の版のまま動くこと・どうすれば入れられるか"""
        lines = [
            f"Sashimono.exe の隣の scripts に置いた物のうち {len(self.failed)} 個を"
            "写せなかったので、入れ替えを止めました 今の版のまま動きます",
        ]
        lines.extend(f"  {path.as_posix()}（{reason}）" for path, reason in self.failed[:10])
        if len(self.failed) > 10:
            lines.append(f"  ほか {len(self.failed) - 10} 個")
        if len(self.links) < len(self.failed):
            lines.append(
                "ディスクの空きと書き込みの権限を確かめるか、"
                "〔互換〕→〔exe の隣のスクリプトを移す…〕で %APPDATA% へ移してから、"
                "〔ヘルプ〕→〔更新を確かめる…〕で入れ直してください"
            )
        if self.links:
            # リンクを含む束は自動では移さないので、「移してから」では片付かない 本人が外せる
            # 手順を出す リンクを消しても先の中身は消えない
            lines.append(
                "リンク（ジャンクション・シンボリックリンク）は入れ替え先で作り直せませんでした"
                " exe の隣の scripts でそのリンクを消し（エクスプローラで消しても、リンクの先の"
                "中身は消えません）、入れ替えた後に作り直してください"
                " または、リンクの先の物を〔互換〕→〔スクリプトフォルダを開く〕で開く所へ写して、"
                "リンクを消してください"
            )
        return "\n".join(lines)


def _same_file(first: Path, second: Path) -> bool:
    try:
        return filecmp.cmp(first, second, shallow=False)
    except OSError:
        return False


def _copy_over(origin: Path, target: Path, *, replace: bool = True) -> str | None:
    """作業用の名前へ写してから置き換える 写せなければ理由を返す

    途中で止まっても、半分の中身が本来の名前で残らない 作業用の写しは片付ける
    ``replace`` が偽なら置き換えない（在る名前へは付けない Windows の rename は断る）
    """
    # 印は移す側（.portable）と同じ ``.moving`` 残っても本人の物と数えず、スクリプトとしても読まない
    writing = target.with_name(f"{target.name}.{os.getpid()}.moving")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, writing)
        # 写した物を照らしてから本来の名前を付ける 写している間に書き換わった・欠けた写しを
        # 入れ替え先へ置かない
        if not _same_file(origin, writing):
            raise OSError("写した中身が元と違う（写している間に書き換わった）")
        if not replace:
            if target.exists() or is_link(target):
                raise FileExistsError("写す先に同じ名前の物が置かれた")
            writing.rename(target)
            return None
        try:
            writing.replace(target)
        except PermissionError:
            # 置き換える先に読み取り専用の印が付いていると、Windows は置き換えさせない
            if not target.exists() or target.stat().st_mode & stat.S_IWRITE:
                raise
            target.chmod(target.stat().st_mode | stat.S_IWRITE)
            writing.replace(target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            remove_file(writing)
        return str(exc) or type(exc).__name__
    return None
