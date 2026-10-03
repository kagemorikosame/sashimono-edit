r"""リリースの目録（update.json）を作り、手元の鍵で署名する

    .venv\Scripts\python.exe tools\update_sign.py manifest <zip> [--minimum 版] [--out update.json]
    .venv\Scripts\python.exe tools\update_sign.py sign update.json --key <鍵のファイル>
    .venv\Scripts\python.exe tools\update_sign.py verify update.json
    .venv\Scripts\python.exe tools\update_sign.py check-tag v1.2.3

- ``manifest`` Actions が作った zip（リリースの下書きから落とした物）から目録を作る
  版と Python の ABI は zip の中の書き付け（build-info.json）から読む 手で打つと食い違う
  大きさと SHA-256 はこの zip から測る
- ``sign`` 合言葉を尋ねて鍵のファイルを開き、目録に署名して ``update.json.sig`` を書く
  書く前に、ソフトと同じ読み方で目録を読めるかを確かめる（読めない目録に署名しない）
  署名した鍵がソースに埋め込まれていなければ知らせる（配った版はその署名を信じない）
- ``verify`` 埋め込んだ公開鍵で、目録と署名が通るかを確かめる 上げる前の最後の確認
- ``check-tag`` タグと ``__version__`` が同じか（リリースの workflow が使う F-12-8）

秘密鍵はリリースのときだけ手元で開く ビルド機・リポジトリ・CI には置かない
手順は docs/development.md の「自動更新のリリース」
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sashimono import __version__  # noqa: E402
from sashimono.links import RELEASES_URL  # noqa: E402
from sashimono.update.manifest import (  # noqa: E402
    SCHEMA,
    Manifest,
    ManifestError,
    PackageInfo,
    manifest_to_json,
    parse_manifest,
    parse_version,
)
from sashimono.update.package import APP_EXE, BUILD_INFO_NAME  # noqa: E402
from sashimono.update.signing import (  # noqa: E402
    SIGNATURE_SUFFIX,
    TRUSTED_PUBLIC_KEYS,
    KeyFileError,
    decrypt_private_key,
    public_key_text,
    sign_manifest,
    trusted_keys,
    verify_manifest,
)


class ReleaseError(Exception):
    """目録を作れない・署名できない"""


def read_zip_info(archive: Path) -> tuple[str, str]:
    """zip の中の書き付けから、版と Python の ABI を読む"""
    try:
        with zipfile.ZipFile(archive) as opened:
            names = [PurePosixPath(name) for name in opened.namelist()]
            found = [name for name in names if name.name == BUILD_INFO_NAME]
            exes = [name for name in names if name.name == APP_EXE]
            if len(found) != 1 or not exes or found[0].parent != exes[0].parent:
                raise ReleaseError(f"zip の中に {APP_EXE} と並んだ {BUILD_INFO_NAME} が無い")
            data = json.loads(opened.read(str(found[0])).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise ReleaseError(f"zip を読めない: {exc}") from exc
    version, abi = data.get("version"), data.get("python_abi")
    if not isinstance(version, str) or not isinstance(abi, str):
        raise ReleaseError(f"{BUILD_INFO_NAME} が読めない")
    return version, abi


def build_manifest(
    archive: Path,
    *,
    minimum: str = "0.0.0",
    url: str | None = None,
    notes_url: str | None = None,
) -> bytes:
    """zip から目録を作る 作った物をソフトと同じ読み方で読み直してから返す"""
    version, abi = read_zip_info(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    tag = f"v{version}"
    manifest = Manifest(
        schema=SCHEMA,
        version=version,
        minimum=minimum,
        python_abi=abi,
        notes_url=notes_url or f"{RELEASES_URL}/tag/{tag}",
        package=PackageInfo(
            url or f"{RELEASES_URL}/download/{tag}/{archive.name}",
            archive.stat().st_size,
            digest.hexdigest(),
        ),
    )
    data = manifest_to_json(manifest)
    try:
        parsed = parse_manifest(data)
        if parse_version(minimum) > parse_version(version):
            raise ReleaseError(f"minimum（{minimum}）が版（{version}）より新しい")
    except ManifestError as exc:
        raise ReleaseError(f"作った目録をソフトが読めない: {exc}") from exc
    assert parsed == manifest
    return data


def sign_file(manifest: Path, key_file: Path, ask: Callable[[str], str]) -> tuple[Path, str]:
    """目録に署名して .sig を書く 書いた場所と、使った鍵の公開鍵を返す"""
    data = manifest.read_bytes()
    try:
        parse_manifest(data)
    except ManifestError as exc:
        raise ReleaseError(f"ソフトが読めない目録には署名しない: {exc}") from exc
    try:
        key = decrypt_private_key(key_file.read_bytes(), ask("鍵の合言葉: "))
    except (OSError, KeyFileError) as exc:
        raise ReleaseError(str(exc)) from exc
    signature = sign_manifest(data, key)
    public = key.public_key()
    if not verify_manifest(data, signature, (public,)):
        raise ReleaseError("書いた署名が通らない")
    target = manifest.with_name(manifest.name + SIGNATURE_SUFFIX)
    target.write_bytes(signature)
    return target, public_key_text(public)


def verify_file(manifest: Path, signature: Path | None = None) -> str:
    """埋め込んだ公開鍵で通るか 通れば版を返す"""
    keys = trusted_keys()
    if not keys:
        raise ReleaseError("ソースに公開鍵が入っていない（TRUSTED_PUBLIC_KEYS が空）")
    signature = signature or manifest.with_name(manifest.name + SIGNATURE_SUFFIX)
    data = manifest.read_bytes()
    if not verify_manifest(data, signature.read_bytes(), keys):
        raise ReleaseError("埋め込んだ公開鍵のどれでも署名が通らない")
    try:
        return parse_manifest(data).version
    except ManifestError as exc:
        raise ReleaseError(str(exc)) from exc


def check_tag(tag: str, version: str = __version__) -> None:
    """タグが ``v`` + ``__version__`` か 食い違ったまま配ると「最新です」と言い続ける版ができる"""
    if tag != f"v{version}":
        raise ReleaseError(f"タグ {tag} と __version__ {version} が違う（v{version} のはず）")


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = getpass.getpass) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("manifest", help="zip から update.json を作る")
    made.add_argument("zip", type=Path)
    made.add_argument("--minimum", default="0.0.0", help="これより古い版は自動では入れられない")
    made.add_argument("--url", help="zip の URL（既定はタグのリリースの資産）")
    made.add_argument("--notes-url", help="変わった所のページ（既定はタグのリリース）")
    made.add_argument("--out", type=Path, default=Path("update.json"))
    signed = commands.add_parser("sign", help="update.json に署名する")
    signed.add_argument("manifest", type=Path)
    signed.add_argument("--key", type=Path, required=True)
    checked = commands.add_parser("verify", help="埋め込んだ公開鍵で確かめる")
    checked.add_argument("manifest", type=Path)
    checked.add_argument("--sig", type=Path)
    tagged = commands.add_parser("check-tag", help="タグと __version__ を照らす")
    tagged.add_argument("tag")
    args = parser.parse_args(argv)

    try:
        if args.command == "manifest":
            data = build_manifest(
                args.zip, minimum=args.minimum, url=args.url, notes_url=args.notes_url
            )
            args.out.write_bytes(data)
            print(f"書いた: {args.out}")
            print(data.decode("utf-8"))
        elif args.command == "sign":
            target, public = sign_file(args.manifest, args.key, ask)
            print(f"書いた: {target}")
            print(f"使った鍵: {public}")
            if public not in TRUSTED_PUBLIC_KEYS:
                print(
                    "[注意] この鍵は今のソースに埋め込まれていない 配った版はこの署名を信じない"
                    "（鍵を差し替えるときの手順は docs/development.md）"
                )
                return 2
        elif args.command == "verify":
            print(f"通った: {verify_file(args.manifest, args.sig)}")
        else:
            check_tag(args.tag)
            print(f"タグと版が同じ: {args.tag}")
    except (OSError, ReleaseError) as exc:
        print(f"[NG] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
