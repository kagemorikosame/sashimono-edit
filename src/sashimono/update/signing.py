"""目録の署名（Ed25519）と、署名に使う鍵のファイル

考え方は minisign と同じ 目録（``update.json``）に署名し、署名は別の資産
（``update.json.sig``）として置く 目録には zip の SHA-256 が入っているので、
目録に署名すれば zip も守られる

- ソフトに埋め込むのは公開鍵だけ（:data:`TRUSTED_PUBLIC_KEYS`） 今使う鍵と予備の鍵の 2 本で、
  どちらかで通れば信じる 今使う鍵を失くしても、予備の鍵で署名した版を出せば、
  配った版はそのまま新しい版を受け取れる（鍵の差し替えの手順は docs/development.md）
- **公開鍵が 1 本も無い版は、更新を確かめない**（黙って何もしない）
- 秘密鍵は本人がパスワード管理アプリとオフラインの控えに置く ビルド機・リポジトリ・
  CI には置かない 鍵のファイルは合言葉から scrypt で作った鍵で AES-256-GCM に包む

署名は目録のバイト列そのものに付ける 読み直して書き直すと、空白 1 つで通らなくなる
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

__all__ = [
    "CONTEXT",
    "KEY_FILE_FORMAT",
    "KEY_PREFIX",
    "MAX_SIGNATURE_BYTES",
    "SCRYPT_N",
    "SIGNATURE_SUFFIX",
    "TRUSTED_PUBLIC_KEYS",
    "KeyFileError",
    "decrypt_private_key",
    "encrypt_private_key",
    "key_file_public_key",
    "parse_public_key",
    "public_key_text",
    "sign_manifest",
    "trusted_keys",
    "verify_manifest",
]

# --- 埋め込む公開鍵 -------------------------------------------------------------
#
# ここへ ``tools/update_keys.py generate`` が出した公開鍵（``sashimono-ed25519:`` で始まる行）を
# 2 本貼る 1 本目が今使う鍵、2 本目が予備の鍵 秘密鍵（鍵のファイルの中身）は決して貼らない
# 空のままの版は更新を確かめない（試験は試験の中で作る使い捨ての鍵を渡す）
#
# 例（形だけ この値は使わない）
#     TRUSTED_PUBLIC_KEYS = (
#         "sashimono-ed25519:AAAA…（今使う鍵）",
#         "sashimono-ed25519:BBBB…（予備の鍵）",
#     )

#: 信じる公開鍵 今使う鍵と予備の鍵
TRUSTED_PUBLIC_KEYS: tuple[str, ...] = ()

# ------------------------------------------------------------------------------

#: 公開鍵の書き方の頭 素の base64 にすると、別の物（秘密鍵の中身など）を貼り間違えても気付けない
KEY_PREFIX = "sashimono-ed25519:"

#: 署名の資産の名前の尻 目録の URL にこれを足した所に置く
SIGNATURE_SUFFIX = ".sig"

#: 署名する中身の頭 同じ鍵で別の物に付けた署名を、目録の署名として通さない
CONTEXT = b"sashimono-update-manifest-v1\n"

#: 署名の資産の大きさの上限 中身は 90 バイトほど
MAX_SIGNATURE_BYTES = 1024

#: 鍵のファイルの形の名前
KEY_FILE_FORMAT = "sashimono-update-key"

#: scrypt の重さ 2**17 で 128 MB と 1 秒ほど掛かる 合言葉を総当たりで試す手間を上げる
#: 署名はリリースのときに 1 回だけなので、待つのは困らない
SCRYPT_N = 2**17
_SCRYPT_R = 8
_SCRYPT_P = 1


class KeyFileError(ValueError):
    """鍵のファイルが読めない・合言葉が違う"""


def public_key_text(key: Ed25519PublicKey) -> str:
    """埋め込む形の公開鍵"""
    raw = key.public_bytes_raw()
    return KEY_PREFIX + base64.b64encode(raw).decode("ascii")


def parse_public_key(text: str) -> Ed25519PublicKey:
    """埋め込んだ形の公開鍵を読む 形が違えば ``ValueError``"""
    text = text.strip()
    if not text.startswith(KEY_PREFIX):
        raise ValueError(f"公開鍵は {KEY_PREFIX} で始まる")
    try:
        raw = base64.b64decode(text[len(KEY_PREFIX) :], validate=True)
    except binascii.Error as exc:
        raise ValueError("公開鍵の base64 が読めない") from exc
    if len(raw) != 32:
        raise ValueError(f"公開鍵が 32 バイトでない（{len(raw)}）")
    return Ed25519PublicKey.from_public_bytes(raw)


def trusted_keys(texts: Iterable[str] = TRUSTED_PUBLIC_KEYS) -> tuple[Ed25519PublicKey, ...]:
    """埋め込んだ公開鍵 読めない物は飛ばす（貼り間違いで起動を止めない 試験が別に落とす）"""
    found = []
    for text in texts:
        try:
            found.append(parse_public_key(text))
        except ValueError:
            continue
    return tuple(found)


def sign_manifest(data: bytes, key: Ed25519PrivateKey) -> bytes:
    """目録のバイト列への署名 ``update.json.sig`` の中身"""
    signature = key.sign(CONTEXT + data)
    return base64.b64encode(signature) + b"\n"


def verify_manifest(data: bytes, signature: bytes, keys: Iterable[Ed25519PublicKey]) -> bool:
    """どれかの鍵で署名が通るか 鍵が 1 本も無ければ通さない"""
    if len(signature) > MAX_SIGNATURE_BYTES:
        return False
    try:
        raw = base64.b64decode(signature.strip(), validate=True)
    except (binascii.Error, ValueError):
        return False
    if len(raw) != 64:
        return False
    for key in keys:
        try:
            key.verify(raw, CONTEXT + data)
        except InvalidSignature:
            continue
        return True
    return False


# --- 鍵のファイル ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Header:
    public_key: str
    salt: bytes
    nonce: bytes
    n: int


def _kdf(passphrase: str, salt: bytes, n: int) -> bytes:
    return Scrypt(salt=salt, length=32, n=n, r=_SCRYPT_R, p=_SCRYPT_P).derive(
        passphrase.encode("utf-8")
    )


def _associated(public_key: str, n: int) -> bytes:
    """包むときに一緒に守る値 公開鍵や重さを書き換えたファイルは開けない"""
    return f"{KEY_FILE_FORMAT}\n{public_key}\n{n}".encode()


def encrypt_private_key(key: Ed25519PrivateKey, passphrase: str, *, n: int = SCRYPT_N) -> bytes:
    """秘密鍵を合言葉で包んだファイルの中身 公開鍵は包まずに添える（合言葉なしで見せられる）"""
    if not passphrase:
        raise KeyFileError("合言葉が空")
    public = public_key_text(key.public_key())
    salt = os.urandom(16)
    nonce = os.urandom(12)
    sealed = AESGCM(_kdf(passphrase, salt, n)).encrypt(
        nonce, key.private_bytes_raw(), _associated(public, n)
    )
    document = {
        "format": KEY_FILE_FORMAT,
        "version": 1,
        "public_key": public,
        "kdf": {
            "name": "scrypt",
            "n": n,
            "r": _SCRYPT_R,
            "p": _SCRYPT_P,
            "salt": base64.b64encode(salt).decode("ascii"),
        },
        "cipher": {"name": "aes-256-gcm", "nonce": base64.b64encode(nonce).decode("ascii")},
        "ciphertext": base64.b64encode(sealed).decode("ascii"),
    }
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def _read_key_file(blob: bytes) -> tuple[_Header, bytes]:
    try:
        document = json.loads(blob.decode("utf-8"))
        if not isinstance(document, dict) or document.get("format") != KEY_FILE_FORMAT:
            raise KeyFileError("Sashimono の更新の鍵のファイルでない")
        kdf = document["kdf"]
        if kdf["name"] != "scrypt" or kdf["r"] != _SCRYPT_R or kdf["p"] != _SCRYPT_P:
            raise KeyFileError("知らない鍵の作り方")
        if document["cipher"]["name"] != "aes-256-gcm":
            raise KeyFileError("知らない包み方")
        header = _Header(
            public_key=str(document["public_key"]),
            salt=base64.b64decode(kdf["salt"], validate=True),
            nonce=base64.b64decode(document["cipher"]["nonce"], validate=True),
            n=int(kdf["n"]),
        )
        sealed = base64.b64decode(document["ciphertext"], validate=True)
    except KeyFileError:
        raise
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise KeyFileError(f"鍵のファイルが読めない: {exc}") from exc
    return header, sealed


def key_file_public_key(blob: bytes) -> str:
    """鍵のファイルに添えた公開鍵 合言葉は要らない"""
    header, _ = _read_key_file(blob)
    parse_public_key(header.public_key)
    return header.public_key


def decrypt_private_key(blob: bytes, passphrase: str) -> Ed25519PrivateKey:
    """鍵のファイルを合言葉で開く 違えば :class:`KeyFileError`"""
    header, sealed = _read_key_file(blob)
    try:
        raw = AESGCM(_kdf(passphrase, header.salt, header.n)).decrypt(
            header.nonce, sealed, _associated(header.public_key, header.n)
        )
    except (InvalidTag, ValueError) as exc:
        raise KeyFileError("合言葉が違うか、鍵のファイルが壊れている") from exc
    key = Ed25519PrivateKey.from_private_bytes(raw)
    if public_key_text(key.public_key()) != header.public_key:
        raise KeyFileError("添えた公開鍵と中身の鍵が合わない")
    return key
