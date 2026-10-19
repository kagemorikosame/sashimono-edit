"""AviUtl 由来のファイルの文字コードを判別する

AviUtl1 世代のファイル（``.exo`` ``.exa`` ``.anm``）は Shift_JIS、AviUtl2 世代は
UTF-8 で書かれている どちらも拡張子からは区別が付かないので、中身を見て決める

判別を間違えると、読めないのではなく**文字化けしたまま読めてしまう** あとから
「なぜかスクリプト名が化ける」という形で出てくるので、ここで確実に決める
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["decode_bytes", "decode_utf16_hex", "encode_utf16_hex", "read_text"]

#: AviUtl1 世代の文字コード Shift_JIS ではなく cp932 を使う
#: Windows の実装は cp932 で、丸数字や罫線など Shift_JIS に無い文字を含む
LEGACY_ENCODING = "cp932"

MODERN_ENCODING = "utf-8"


def decode_bytes(data: bytes) -> tuple[str, str]:
    """バイト列を文字列へ ``(本文, 使った文字コード)`` を返す

    順番に意味がある UTF-8 として厳密に読めるバイト列が cp932 でも読めることは
    あるが、その逆はほとんど起きない UTF-8 を先に試すのが安全側
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode(MODERN_ENCODING, errors="replace"), "utf-8-sig"

    try:
        return data.decode(MODERN_ENCODING), MODERN_ENCODING
    except UnicodeDecodeError:
        pass

    try:
        return data.decode(LEGACY_ENCODING), LEGACY_ENCODING
    except UnicodeDecodeError:
        # どちらでも読めない 読めた分だけ返す 1 文字のために
        # ファイル全体を捨てる方が損が大きい
        return data.decode(LEGACY_ENCODING, errors="replace"), LEGACY_ENCODING


def read_text(path: Path) -> tuple[str, str]:
    """ファイルを読んで ``(本文, 文字コード)`` を返す"""
    return decode_bytes(Path(path).read_bytes())


def decode_utf16_hex(value: str) -> str:
    """``.exo`` のテキスト欄を復号する

    AviUtl はテキストを UTF-16LE の 16 進文字列で書く 末尾は 0 で埋められて
    いるので、最初の終端文字までを本文とする

    崩れた欄でも例外にしない（テキスト欄が 1 つ壊れただけで .exo の読み込み全体を
    止めない） 16 進の字だけを残し、4 桁（1 文字分）の倍数へ切り詰めるので、
    ``bytes.fromhex`` が断る字（16 進でない字・半端な桁）は渡らない 前は念のための
    ``except ValueError`` を置いていたが、ここを通る入力は無く、試験で確かめられない
    分かれ道だった（#384） 約束は試験が任意の文字列で押さえている
    """
    cleaned = "".join(c for c in value if c in "0123456789abcdefABCDEF")
    if len(cleaned) % 4:
        cleaned = cleaned[: len(cleaned) // 4 * 4]
    raw = bytes.fromhex(cleaned)
    text = raw.decode("utf-16-le", errors="replace")
    return text.split("\x00", 1)[0]


def encode_utf16_hex(text: str, *, length: int = 1024) -> str:
    """テキストを ``.exo`` のテキスト欄の形へ

    ``length`` は AviUtl が確保している文字数 足りない分は 0 で埋める
    """
    raw = text.encode("utf-16-le")
    padding = max(0, length * 2 - len(raw))
    return (raw + b"\x00" * padding).hex()
