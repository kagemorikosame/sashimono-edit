"""後から入れる部品を、使う人と同じ道で入れて読めるかを確かめる（``Sashimono.exe --add-on-check``）

配る zip の確かめ（``tools/build_package.py`` と CI の ``tools/check_clean_machine.ps1``）が使う
CI の確かめる側の機械には Python が無いので、入れ方と読み方を exe 自身に持たせる
PowerShell 側へ書き写すと、導入ボタンの入れ方が変わったときに確かめだけが古いまま残る

1. 導入ボタンと同じ引数（:func:`~sashimono.runtime.install_arguments`）と同じ走らせ方で、
   渡された導入先へ入れる 機能は使う人が押す順（AI 連携 → 字幕起こし）で 1 つずつ
   配布版は導入ボタンと同じく、子を起こさずにこのプロセスの中で pip を走らせる
   （:func:`~sashimono.runtime.run_pip_here`）
2. 起動のときに導入先を読むのと同じ読み方（前へ足して ``.pth`` も読む ``--import-check``）で
   import する 配布版はこのプロセスの中で読む 前は別の exe を起こして読んでいたが、
   exe が自分自身を子として起こし、落とした物を書き込んでから別の exe で読む流れの途中で、
   利用者の機械の Windows Defender が 0.1.3 の exe を ``Trojan:Win32/Bearfoos.A!ml`` と見て
   消した 使う人の導入ボタンももう子を起こさないので、確かめも同じにそろえる 読んだ部品が
   このプロセスに残るのは、確かめるためだけの起動なので構わない pip の部品は走り終えた所で
   捨てる（:func:`~sashimono.runtime.run_pip_here`）ので、読む所へは混ざらない
   開発の環境は今までどおり ``python -m pip`` と ``python -m sashimono --import-check`` を
   子で走らせる

**ネットにつなぐ** 字幕起こしの CUDA ランタイム（1.7 GB）は入れない 入るのは DLL で、
import する Python の部品は変わらない
"""

from __future__ import annotations

import locale
import subprocess
import sys
import threading
import time
from pathlib import Path

from sashimono.ai.environment import AI_PACK
from sashimono.app import IMPORT_CHECK_FLAG, import_check
from sashimono.asr.environment import ASR_PACK
from sashimono.runtime import FeaturePack, install_arguments, is_frozen, run_pip_in_worker

__all__ = ["ADD_ON_MODULES", "ADD_ON_PACKS", "INSTALL_TIMEOUT", "main", "self_command"]

#: 確かめる機能 使う人が導入の欄で押す物と同じ
ADD_ON_PACKS = (AI_PACK, ASR_PACK)
#: 入れた後に import する名前 送った途端・起こし始めた途端に読まれる物
ADD_ON_MODULES = ("claude_agent_sdk", "faster_whisper", "ctranslate2")
#: 段（機能ごとの pip と読む所）を 1 つ待つ秒数の既定（手元の組み立ての道具） 初めては
#: 合わせて 200 MB ほど落とす 遅い回線でも待てるよう長めに CI は短い値を渡す
#: （``--add-on-check <置き場> <秒数>``）
INSTALL_TIMEOUT = 1800


def self_command() -> list[str]:
    """自分自身を起こすコマンドの頭 開発の環境は ``python -m sashimono`` 配布版は exe

    配布版の確かめはもう自分を起こさない（このプロセスの中で読む） 開発の環境で
    ``--import-check`` を子で走らせるときに使う
    """
    if is_frozen():
        return [sys.executable]
    return [sys.executable, "-m", "sashimono"]


def _call(arguments: list[str], lines: list[str], timeout: float) -> int:
    """子を走らせ、子の出力を受けて ``lines`` へ足す 開発の環境だけが使う

    受けずに引き継がせると、子の書いた行（``[ok] claude_agent_sdk`` など）が 1 つも
    呼んだ道具へ届かなかった
    """
    try:
        # 引数は、この部品の中で決めた物と、呼んだ人が渡した導入先だけ shell も通さない
        done = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        lines.append(f"[NG] {timeout:g} 秒で終わらない: {' '.join(arguments[:4])} …")
        return 1
    # 子は Python なので、書く文字コードはこの機械の既定
    text = done.stdout.decode(locale.getencoding(), errors="replace").rstrip()
    if text:
        lines.extend(text.splitlines())
    return done.returncode


def _install_here(arguments: list[str], lines: list[str], timeout: float) -> int:
    """このプロセスの中で pip を走らせる ``timeout`` 秒を過ぎたら中断して [NG] を書く

    子なら時間切れで止めれば済んだ 同じプロセスでは導入ボタンの中断と同じ口で止める
    止めずに待ち続けると、呼んだ側が先に待ちきれずに止め、何で止まったかが残らない
    """
    deadline = time.monotonic() + timeout
    late = threading.Event()

    def too_late() -> bool:
        if time.monotonic() >= deadline:
            late.set()
        return late.is_set()

    # 作業スレッドで走らせる 同じスレッドで走らせると、pip が同期の読み書きの中で戻らない間は
    # 期限を見られず、[NG] を書く前に外の待ちの制限で exe ごと止められる（PR #254 のレビュー）
    code = run_pip_in_worker(arguments, on_output=lines.append, should_cancel=too_late)
    if late.is_set():
        lines.append(f"[NG] {timeout:g} 秒で終わらない: pip {' '.join(arguments[:3])} …")
        return code or 1
    return code


def _install(pack: FeaturePack, place: Path, lines: list[str], timeout: float) -> int:
    arguments = install_arguments(pack, place, extra=False)
    if is_frozen():
        return _install_here(arguments, lines, timeout)
    return _call([sys.executable, "-m", "pip", *arguments], lines, timeout)


def _read(place: Path, lines: list[str], timeout: float) -> int:
    if is_frozen():
        return import_check(str(place), list(ADD_ON_MODULES), write=lines.append)
    return _call([*self_command(), IMPORT_CHECK_FLAG, str(place), *ADD_ON_MODULES], lines, timeout)


def _emit(lines: list[str]) -> None:
    """まとめて書く 英語の Windows（CI）でも日本語の行で落ちないよう、自己診断と同じ手当てをする"""
    # 時間切れで戻らない pip を残して返したときは、標準出力がまだ pip の受け口のまま
    # 元の出口へ書く 受け口へ書くと文字コードの手当てが元の出口に届かない
    stream = getattr(sys.stdout, "fallback", sys.stdout)
    if stream is None:
        # 窓の無い配布版を、出力を受けずに起こした 書いても誰にも届かない
        return
    from sashimono.selfcheck import _make_writable

    # pip の作業スレッドがまだ足しているかもしれないので、写してから書く
    text = "\n".join(list(lines))
    _make_writable(stream, text)
    print(text, file=stream, flush=True)


def main(target: str, timeout: float = INSTALL_TIMEOUT) -> int:
    """``target`` へ入れて読む すべて通れば 0

    ``timeout`` は段（機能ごとの pip と読む所）を 1 つずつ待つ秒数 段は
    ``len(ADD_ON_PACKS) + 1`` 個 時間切れなら ``[NG]`` を書いて 1 で終わる 呼んだ側が
    先に待ちきれずに止めると、何で止まったかが残らないので、呼ぶ側の待ちに収まる値を渡す
    （CI の tools/check_clean_machine.ps1） 配布版の読む所はこのプロセスの中で読むので
    待ちを切らない（import は数秒で終わる）
    """
    place = Path(target)
    lines: list[str] = []
    try:
        for pack in ADD_ON_PACKS:
            code = _install(pack, place, lines, timeout)
            if code != 0:
                lines.append(
                    f"[NG] {pack.label}を exe の pip で入れられない（終了コード {code}"
                    " 導入ボタンも同じ所で止まる ネットにつながっているかも確かめる）"
                )
                return 1
            lines.append(f"[ok] {pack.label}を exe の pip で入れた: {place}")
        return _read(place, lines, timeout)
    finally:
        _emit(lines)
