r"""自動更新の署名に使う鍵を作る（利用者が手元で 1 度だけ走らせる）

    .venv\Scripts\python.exe tools\update_keys.py generate --out <鍵を置くフォルダ>
    .venv\Scripts\python.exe tools\update_keys.py show <鍵のファイル>

``generate`` は 2 本の鍵（今使う鍵と予備の鍵）を作る

- 秘密鍵は、尋ねた合言葉で包んだファイルとして ``--out`` のフォルダへ書く
  （``sashimono-update-current.key`` と ``sashimono-update-spare.key``）
  同じ名前のファイルがあれば書かずに止まる（前の鍵を上書きで失くさないため）
- 画面に出すのは公開鍵だけ ``src/sashimono/update/signing.py`` の ``TRUSTED_PUBLIC_KEYS`` へ
  貼る 秘密鍵の中身・合言葉は画面にも記録にも出さない

書いた鍵のファイルは、パスワード管理アプリ（原本）とオフラインの控え（USB や紙）へ移し、
ここで書いたファイルは消す ビルド機・リポジトリ・CI には置かない
手順は docs/development.md の「自動更新の署名鍵」

``show`` は鍵のファイルに添えた公開鍵を出す（合言葉は要らない） 貼った公開鍵と
手元の鍵が対になっているかを見るのに使う
"""

from __future__ import annotations

import argparse
import getpass
import io
import sys
from collections.abc import Callable
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from sashimono.update.signing import (  # noqa: E402
    SCRYPT_N,
    KeyFileError,
    encrypt_private_key,
    key_file_public_key,
    public_key_text,
)

#: 書く鍵のファイルの名前 今使う鍵と予備の鍵
KEY_FILES = (
    ("今使う鍵", "sashimono-update-current.key"),
    ("予備の鍵", "sashimono-update-spare.key"),
)

#: 合言葉の短さの下限 鍵のファイルが漏れたとき、総当たりで開けられないように
MIN_PASSPHRASE = 12


def ask_passphrase(label: str, ask: Callable[[str], str]) -> str:
    """合言葉を 2 回尋ねる 打ち間違えたまま包むと、その鍵は二度と開けない"""
    while True:
        first = ask(f"{label}の合言葉（{MIN_PASSPHRASE} 文字以上 画面には出ない）: ")
        if len(first) < MIN_PASSPHRASE:
            print(f"{MIN_PASSPHRASE} 文字以上にしてください")
            continue
        if ask("もう一度: ") != first:
            print("1 回目と違います もう一度")
            continue
        return first


def generate(out: Path, ask: Callable[[str], str], *, n: int = SCRYPT_N) -> list[str]:
    """2 本の鍵を作って書く 公開鍵を返す 1 本でも書けなければ何も残さない"""
    targets = [(label, out / name) for label, name in KEY_FILES]
    existing = [str(path) for _, path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "同じ名前の鍵のファイルがある（上書きしない）: " + "、".join(existing)
        )
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    publics: list[str] = []
    try:
        for label, path in targets:
            key = Ed25519PrivateKey.generate()
            blob = encrypt_private_key(key, ask_passphrase(label, ask), n=n)
            # 排他で作る 確かめてから書くまでの間に別の物が置かれても、上書きしない
            with path.open("xb") as handle:
                handle.write(blob)
            written.append(path)
            publics.append(public_key_text(key.public_key()))
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return publics


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = getpass.getpass) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("generate", help="今使う鍵と予備の鍵を作る")
    made.add_argument("--out", type=Path, required=True, help="鍵のファイルを書くフォルダ")
    shown = commands.add_parser("show", help="鍵のファイルの公開鍵を出す")
    shown.add_argument("key", type=Path)
    args = parser.parse_args(argv)

    if args.command == "show":
        try:
            print(key_file_public_key(args.key.read_bytes()))
        except (OSError, KeyFileError, ValueError) as exc:
            print(f"読めない: {exc}")
            return 1
        return 0

    try:
        publics = generate(args.out, ask)
    except (OSError, KeyFileError) as exc:
        print(f"作れなかった: {exc}")
        return 1
    print()
    print("鍵のファイルを書いた（中身は合言葉で包んである）")
    for (label, name), _ in zip(KEY_FILES, publics, strict=True):
        print(f"  {label}: {args.out / name}")
    print()
    print("src/sashimono/update/signing.py の TRUSTED_PUBLIC_KEYS を、次のとおりにする")
    print()
    print("TRUSTED_PUBLIC_KEYS: tuple[str, ...] = (")
    for (label, _), public in zip(KEY_FILES, publics, strict=True):
        print(f'    "{public}",  # {label}')
    print(")")
    print()
    print("鍵のファイルは、パスワード管理アプリとオフラインの控えへ移してから、ここから消す")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
