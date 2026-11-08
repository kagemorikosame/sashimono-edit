"""静止画から、撮った場所などの付け足しの情報を取り除く（#423）

スマートフォンの写真は EXIF に撮った場所（GPS）と機器の名前を持つ 持ち出し用の
パッケージは人へ渡す物なので、既定で取り除く 絵の符号は触らない（画質はそのまま）

向き（EXIF の Orientation）だけは残す 縦に撮った写真は向きの印で立てて見せており、
消すと横倒しの絵になる 残すのは向き 1 つだけの小さな EXIF に作り直した物

扱うのは JPEG と PNG だけ ほかの形（WebP・TIFF など）は中を書き換えずにそのまま入れ、
呼ぶ側が「取り除けなかった」として知らせる（推して書き換えて壊すよりよい）
"""

from __future__ import annotations

import struct

__all__ = ["scrub_image", "scrubbable"]

#: JPEG の印 SOI と、画素の符号が始まる SOS
_SOI = b"\xff\xd8"
_SOS = 0xDA
#: 残す APP の印 APP0（JFIF）と APP2（ICC の色の情報）と APP14（Adobe の色の並び）
#: 色の情報を消すと、色の広い写真の色が変わる
_KEPT_APPS = frozenset({0xE0, 0xE2, 0xEE})
#: 落とす印 APP1（EXIF と XMP）・APP13（IPTC）とほかの APP、COM（注記）
_COM = 0xFE

#: PNG の頭と、落とす塊 eXIf（EXIF）と文字の塊（tEXt・zTXt・iTXt XMP もここに入る）と時刻
_PNG_HEAD = b"\x89PNG\r\n\x1a\n"
_PNG_DROPPED = frozenset({b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"})

#: EXIF の Orientation の札
_ORIENTATION = 0x0112


def scrubbable(suffix: str) -> bool:
    """この拡張子の静止画から付け足しの情報を取り除けるか"""
    return suffix.lower() in {".jpg", ".jpeg", ".png"}


def scrub_image(data: bytes, suffix: str) -> bytes | None:
    """付け足しの情報を取り除いた中身 形が読めなければ ``None``（呼ぶ側がそのまま入れる）"""
    try:
        if suffix.lower() in {".jpg", ".jpeg"}:
            return _scrub_jpeg(data)
        if suffix.lower() == ".png":
            return _scrub_png(data)
    except (ValueError, struct.error, IndexError):
        return None
    return None


def _segments(data: bytes) -> list[tuple[int, bytes]] | None:
    """SOS の前までの印と塊の並び 最後は SOS から終わりまで 読めなければ ``None``"""
    found: list[tuple[int, bytes]] = []
    position = 2
    while position < len(data):
        if data[position] != 0xFF:
            return None
        marker = data[position + 1]
        if marker == 0xFF:
            # 詰め物の 0xFF は飛ばす
            position += 1
            continue
        if marker == _SOS:
            found.append((marker, data[position:]))
            return found
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            # 長さを持たない印
            found.append((marker, data[position : position + 2]))
            position += 2
            continue
        (length,) = struct.unpack(">H", data[position + 2 : position + 4])
        segment = data[position : position + 2 + length]
        if len(segment) < 2 + length:
            return None
        found.append((marker, segment))
        position += 2 + length
    return None


def _scrub_jpeg(data: bytes) -> bytes | None:
    if not data.startswith(_SOI):
        return None
    segments = _segments(data)
    if segments is None:
        return None
    # 向きは SOS までのどの EXIF にあってもよい 先に全部を読んでから置く 画素の表（DQT）の
    # 後ろに EXIF がある写真で、置いた後に見つけた向きを落とすと寝た絵になる
    # （#449 の CodeRabbit の指摘）
    orientation: int | None = None
    for marker, segment in segments:
        if marker == 0xE1 and segment[4:].startswith(b"Exif\x00\x00"):
            orientation = _read_orientation(segment[10:]) or orientation
    out = bytearray(_SOI)
    inserted = False
    for marker, segment in segments:
        is_app = 0xE0 <= marker <= 0xEF
        if (is_app and marker not in _KEPT_APPS) or marker == _COM:
            continue
        if not inserted and marker not in _KEPT_APPS:
            # 向きは APP0 の後ろ、画素の表（DQT など）の前に置く
            out += _orientation_segment(orientation)
            inserted = True
        out += segment
    return bytes(out)


def _read_orientation(tiff: bytes) -> int | None:
    """EXIF（TIFF の形）の最初の表から向きの値を読む 無ければ ``None``"""
    order = tiff[:2]
    if order == b"II":
        prefix = "<"
    elif order == b"MM":
        prefix = ">"
    else:
        return None
    (offset,) = struct.unpack(prefix + "I", tiff[4:8])
    (count,) = struct.unpack(prefix + "H", tiff[offset : offset + 2])
    for index in range(count):
        entry = tiff[offset + 2 + index * 12 : offset + 14 + index * 12]
        tag, kind = struct.unpack(prefix + "HH", entry[:4])
        if tag == _ORIENTATION and kind == 3:
            (value,) = struct.unpack(prefix + "H", entry[8:10])
            return int(value) if 1 <= value <= 8 else None
    return None


def _orientation_segment(orientation: int | None) -> bytes:
    """向き 1 つだけを持つ APP1（EXIF） 向きが無い・既定（1）なら何も足さない"""
    if orientation is None or orientation == 1:
        return b""
    # 大きい方から並べる TIFF の頭（MM 42 最初の表は 8 バイト目）と、札 1 つの表
    tiff = b"MM\x00\x2a" + struct.pack(">I", 8)
    tiff += struct.pack(">H", 1)
    tiff += struct.pack(">HHI", _ORIENTATION, 3, 1) + struct.pack(">HH", orientation, 0)
    tiff += struct.pack(">I", 0)
    body = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body


def _scrub_png(data: bytes) -> bytes | None:
    if not data.startswith(_PNG_HEAD):
        return None
    out = bytearray(_PNG_HEAD)
    position = len(_PNG_HEAD)
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(data):
            return None
        if kind not in _PNG_DROPPED:
            out += data[position:end]
        position = end
        if kind == b"IEND":
            return bytes(out)
    return None
