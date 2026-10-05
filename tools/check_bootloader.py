r"""入っている PyInstaller の起動部（bootloader）が、配布の既成の物と違うことを確かめる

    .venv\Scripts\python.exe tools\check_bootloader.py

PyPI の wheel に入った既成の起動部は、同じ物を多くのマルウェアが使っているので、
Windows Defender の機械学習の推測（``Trojan:Win32/Bearfoos.A!ml`` など）に引っ掛かりやすい
0.1.1 の zip は、AI 連携を入れる途中で ``Sashimono.exe`` ごと消された
そこで組み立ての前に、PyInstaller を sdist から ``PYINSTALLER_COMPILE_BOOTLOADER=1`` で
入れて起動部を組み直す（.github/workflows/package.yml 手元の手順は docs/development.md）

組み直したつもりで既成の wheel が入っていると、気付かずに同じ物を配ってしまう
ここで同じ版の既成の wheel を落とし、起動部のハッシュを 1 つずつ比べ、1 つでも同じなら止める
比べた結果は ``GITHUB_STEP_SUMMARY`` があればそこへも書き、後から見返せるようにする
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 配る exe が使う起動部の置き場（PyInstaller の包みの中）
BOOTLOADER_DIR = PurePosixPath("PyInstaller/bootloader/Windows-64bit-intel")

#: 必ず比べる起動部 窓あり・窓なしと、それぞれの調べる用（PyInstaller 6 の置き場の中身）
#: どれを配る exe が使うかは組み立ての引数で変わるので、全部を見る
REQUIRED = ("run.exe", "run_d.exe", "runw.exe", "runw_d.exe")


def digests_in_wheel(wheel: Path) -> dict[str, str]:
    """既成の wheel の中の起動部 名前 → sha256"""
    found: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.parent == BOOTLOADER_DIR and path.suffix == ".exe":
                found[path.name] = hashlib.sha256(archive.read(name)).hexdigest()
    return found


def digests_in_folder(folder: Path) -> dict[str, str]:
    """入っている起動部 名前 → sha256"""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.glob("*.exe"))
    }


def compare(installed: Mapping[str, str], stock: Mapping[str, str]) -> list[str]:
    """既成の物と同じ・比べられない起動部を挙げる 空なら全部組み直されている

    片方にしか無い物も問題に数える 名前が変わった版で比べ漏れて「違う」と通るのを防ぐ
    要る名前（``REQUIRED``）は両側に無くても数える 両側から同じ物が欠けると、
    比べる相手が無いまま残りだけで通ってしまう
    """
    problems: list[str] = []
    if not installed:
        problems.append("入っている起動部が見つからない")
    for name in sorted(set(installed) | set(stock) | set(REQUIRED)):
        if name not in stock:
            problems.append(f"{name} が既成の wheel に無く比べられない")
        elif name not in installed:
            problems.append(f"{name} が入っていない")
        elif installed[name] == stock[name]:
            problems.append(f"{name} が既成の物と同じ（組み直されていない）")
    return problems


def report(version: str, installed: Mapping[str, str], stock: Mapping[str, str]) -> str:
    """比べた結果を Markdown の表にする（CI の要約と画面の両方へ出す）"""
    lines = [
        f"### PyInstaller {version} の起動部",
        "",
        "| ファイル | 組み直した物 | 既成の wheel |",
        "| --- | --- | --- |",
    ]
    for name in sorted(set(installed) | set(stock) | set(REQUIRED)):
        lines.append(f"| {name} | {installed.get(name, '-')} | {stock.get(name, '-')} |")
    return "\n".join(lines) + "\n"


def download_stock_wheel(version: str, folder: Path) -> Path:
    """同じ版の既成の wheel を落とす 依存は要らない（起動部だけを見る）"""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            f"pyinstaller=={version}",
            "--no-deps",
            "--only-binary=:all:",
            "--platform",
            "win_amd64",
            "--dest",
            str(folder),
            "--quiet",
        ],
        check=True,
    )
    wheels = sorted(folder.glob("pyinstaller-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"既成の wheel が 1 つに決まらない: {wheels}")
    return wheels[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2] if __doc__ else None)
    parser.parse_args(argv)

    import PyInstaller

    version = PyInstaller.__version__
    folder = Path(PyInstaller.__file__).resolve().parent.parent / BOOTLOADER_DIR
    installed = digests_in_folder(folder)
    with tempfile.TemporaryDirectory() as temp:
        stock = digests_in_wheel(download_stock_wheel(version, Path(temp)))

    text = report(version, installed, stock)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(text)

    problems = compare(installed, stock)
    for problem in problems:
        print(f"止める: {problem}", file=sys.stderr)
    if problems:
        print(
            "PyInstaller を sdist から組み直して入れる"
            "（PYINSTALLER_COMPILE_BOOTLOADER=1 と --no-binary pyinstaller）",
            file=sys.stderr,
        )
        return 1
    print("起動部はすべて既成の物と違う")
    return 0


if __name__ == "__main__":
    sys.exit(main())
