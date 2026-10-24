"""版の書き付けだけを持つ python3XX.dll の偽物（#410）

組み立てとリリースの道具は、配る物に入った Python の版を DLL の版情報（VS_VERSIONINFO の
固定の欄）から読む 本物の DLL は 7 MB ほどあり、版ごとに用意できないので、読む所に要る
形だけを作る 並びは Windows の VS_VERSIONINFO と同じ（長さ・値の長さ・種類・名前・詰め物・
VS_FIXEDFILEINFO）
"""

from __future__ import annotations

import re
from pathlib import Path

#: 正式版の段階（PY_RELEASE_LEVEL_FINAL = 0xF）と、版情報の 4 つ目の欄（PYTHON_API_VERSION）
#: 読む側は 4 つ目を使わないが、本物と同じ値にしておく
_FINAL = 0xF
_API = 1013

_FIXED_LENGTH = 52

#: 版情報の 3 つ目の欄（16 ビット）に入る micro の上限 ``65 * 1000 + 150 = 65150`` まで入る
#: 本物の DLL も同じ欄に書くので、これを超える micro の Python は版情報を持てない
MAX_MICRO = (0xFFFF - _FINAL * 10) // 1000


def python_dll_bytes(version: tuple[int, int, int]) -> bytes:
    """``version`` の Python の DLL に見える中身 頭に本物と同じく PE の印と詰め物を置く

    欄に入らない版は ValueError 黙って桁があふれた値を作ると、読む側の試験が版を比べる前に
    別の理由で落ち、何を確かめたのかが分からなくなる
    """
    major, minor, micro = version
    if not (0 <= major <= 0xFFFF and 0 <= minor <= 0xFFFF and 0 <= micro <= MAX_MICRO):
        raise ValueError(f"{version} は版情報の欄に入らない（micro は 0 から {MAX_MICRO}）")
    field3 = micro * 1000 + _FINAL * 10
    fixed = b"".join(
        value.to_bytes(4, "little")
        for value in (
            0xFEEF04BD,
            0x00010000,
            (major << 16) | minor,
            (field3 << 16) | _API,
            (major << 16) | minor,
            (field3 << 16) | _API,
            0x3F,
            0,
            0x40004,
            2,
            0,
            0,
            0,
        )
    )
    assert len(fixed) == _FIXED_LENGTH
    key = "VS_VERSION_INFO\0".encode("utf-16-le")
    header = (6 + len(key) + 2 + len(fixed)).to_bytes(2, "little")
    header += _FIXED_LENGTH.to_bytes(2, "little") + (0).to_bytes(2, "little")
    # 6 + 32 = 38 バイトの後、4 バイト境界まで 2 バイト詰める
    return b"MZ" + bytes(126) + header + key + bytes(2) + fixed + bytes(64)


def just_below(version: tuple[int, int, int]) -> tuple[int, int, int]:
    """``version`` のすぐ下の版 下限を 3.15.0 のような系列の頭へ上げても作れる版を返す

    系列の頭なら 1 つ前の系列の、版情報に入るいちばん新しい micro（:data:`MAX_MICRO`）にする
    負の数や欄に入らない micro を返すと、偽の DLL を作る所で落ちる
    """
    major, minor, micro = version
    if micro:
        return (major, minor, micro - 1)
    if minor:
        return (major, minor - 1, MAX_MICRO)
    raise ValueError(f"{version} のすぐ下の版は決められない（系列の数が 0）")


def minimum_python() -> tuple[int, int, int]:
    """tools/build_package.py の下限 道具を読み込まずに字面で読む

    リリースの試験は偽の zip を組む所で使う 下限を上げても試験の zip が下限を割らない
    """
    text = (Path(__file__).resolve().parent.parent / "tools" / "build_package.py").read_text(
        encoding="utf-8"
    )
    found = re.search(r"^MINIMUM_PYTHON = \((\d+), (\d+), (\d+)\)$", text, flags=re.M)
    assert found is not None, "tools/build_package.py に MINIMUM_PYTHON が無い"
    return (int(found[1]), int(found[2]), int(found[3]))
