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

import json
import shutil
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sashimono.runtime import app_dir
from sashimono.update.fetch import Transport, download
from sashimono.update.manifest import MAX_PACKAGE_BYTES, Manifest
from sashimono.update.portable import PORTABLE_SCRIPTS_DIR

__all__ = [
    "APP_EXE",
    "BUILD_INFO_NAME",
    "BuildInfo",
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


def carry_user_files(install: Path, staged: Path) -> int:
    """今の版の exe の隣のスクリプト置き場に本人が置いた物を、新しい版へ写す 写した数を返す

    入れ替えはフォルダを丸ごと替えるので、写さないと前の版の案内どおりそこへ置いた
    スクリプトが 2 回目の入れ替えで消える（1 回目は ``previous`` に残るが、次で消える）
    新しい版が同じ名前の物を持っていれば新しい方を残す（同梱の README など）
    """
    source = install / _PORTABLE_SCRIPTS_DIR
    if not source.is_dir():
        return 0
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = staged / _PORTABLE_SCRIPTS_DIR / path.relative_to(source)
        if destination.exists():
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        except OSError:
            continue  # 1 つ写せなくても入れ替えは止めない 元は previous に残る
        copied += 1
    return copied
