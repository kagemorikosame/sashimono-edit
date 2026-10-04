"""後から入れる部品を、使う人と同じ道で入れて読めるかを確かめる（``Sashimono.exe --add-on-check``）

配る zip の確かめ（``tools/build_package.py`` と CI の ``tools/check_clean_machine.ps1``）が使う
CI の確かめる側の機械には Python が無いので、入れ方と読み方を exe 自身に持たせる
PowerShell 側へ書き写すと、導入ボタンの入れ方が変わったときに確かめだけが古いまま残る

1. 導入ボタンと同じ引数（:func:`~sashimono.runtime.install_arguments`）で、exe の pip に
   渡された導入先へ入れる 機能は使う人が押す順（AI 連携 → 字幕起こし）で 1 つずつ
2. 起動のときに導入先を読むのと同じ読み方（前へ足して ``.pth`` も読む ``--import-check``）で、
   別の exe の中で import する 入れた所と同じプロセスで読むと、pip が読み込んだ物が
   混ざり、起動したばかりの exe とは違う状態で確かめることになる

**ネットにつなぐ** 字幕起こしの CUDA ランタイム（1.7 GB）は入れない 入るのは DLL で、
import する Python の部品は変わらない
"""

from __future__ import annotations

import locale
import subprocess
import sys
from pathlib import Path

from sashimono.ai.environment import AI_PACK
from sashimono.asr.environment import ASR_PACK
from sashimono.runtime import install_arguments, is_frozen

__all__ = ["ADD_ON_MODULES", "ADD_ON_PACKS", "INSTALL_TIMEOUT", "main", "self_command"]

#: 確かめる機能 使う人が導入の欄で押す物と同じ
ADD_ON_PACKS = (AI_PACK, ASR_PACK)
#: 入れた後に import する名前 送った途端・起こし始めた途端に読まれる物
ADD_ON_MODULES = ("claude_agent_sdk", "faster_whisper", "ctranslate2")
#: 1 つの機能を入れるのに待つ秒数 初めては合わせて 200 MB ほど落とす 遅い回線でも待てるよう長めに
INSTALL_TIMEOUT = 1800


def self_command() -> list[str]:
    """自分自身を起こすコマンドの頭 配布版は exe 開発の環境では ``python -m sashimono``

    pip はどちらも ``sys.executable -m pip`` で起こす（配布版の exe は ``-m pip`` を受ける）
    """
    if is_frozen():
        return [sys.executable]
    return [sys.executable, "-m", "sashimono"]


def _call(arguments: list[str], lines: list[str]) -> int:
    """子を走らせ、子の出力を受けて ``lines`` へ足す

    受けずに引き継がせると、手元で組んだ配布版では子の書いた行（``[ok] claude_agent_sdk``
    など）が 1 つも呼んだ道具へ届かなかった
    """
    try:
        # 引数は、この部品の中で決めた物と、呼んだ人が渡した導入先だけ shell も通さない
        done = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=INSTALL_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        lines.append(f"[NG] {INSTALL_TIMEOUT} 秒で終わらない: {' '.join(arguments[:4])} …")
        return 1
    # 子は同じ exe か Python なので、書く文字コードはこの機械の既定
    text = done.stdout.decode(locale.getencoding(), errors="replace").rstrip()
    if text:
        lines.extend(text.splitlines())
    return done.returncode


def _emit(lines: list[str]) -> None:
    """まとめて書く 英語の Windows（CI）でも日本語の行で落ちないよう、自己診断と同じ手当てをする"""
    if sys.stdout is None:
        # 窓の無い配布版を、出力を受けずに起こした 書いても誰にも届かない
        return
    from sashimono.selfcheck import _make_writable

    text = "\n".join(lines)
    _make_writable(sys.stdout, text)
    print(text, flush=True)


def main(target: str) -> int:
    """``target`` へ入れて読む すべて通れば 0"""
    from sashimono.app import IMPORT_CHECK_FLAG

    place = Path(target)
    lines: list[str] = []
    try:
        for pack in ADD_ON_PACKS:
            code = _call(
                [sys.executable, "-m", "pip", *install_arguments(pack, place, extra=False)], lines
            )
            if code != 0:
                lines.append(
                    f"[NG] {pack.label}を exe の pip で入れられない（終了コード {code}"
                    " 導入ボタンも同じ所で止まる ネットにつながっているかも確かめる）"
                )
                return 1
            lines.append(f"[ok] {pack.label}を exe の pip で入れた: {place}")
        return _call([*self_command(), IMPORT_CHECK_FLAG, str(place), *ADD_ON_MODULES], lines)
    finally:
        _emit(lines)
