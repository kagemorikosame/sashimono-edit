"""字幕起こしの実行環境

導入の仕組みそのものは :mod:`sashimono.runtime` にある ここは「字幕起こしに何が
要るか」だけを定義する AI 連携も同じ仕組みに載っているので、導入の画面と手順は
どちらも共通になる

CTranslate2 は cuDNN と cuBLAS の DLL を実行時に探す これが無いと CUDA を指定した
瞬間に落ちるが、無くても CPU では動く だから追加扱い（:attr:`FeaturePack.extra`）に
してある 合計で 1.7 GB あるので、要らない人が落とさずに済むことには意味がある
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sashimono.runtime import FeaturePack, PackStatus, activate_runtime, install_runtime
from sashimono.runtime import install_command as _install_command

__all__ = [
    "ASR_PACK",
    "CUDA_PACKAGES",
    "REQUIRED_PACKAGES",
    "PackStatus",
    "activate_runtime",
    "cuda_library_dirs",
    "install_command",
    "install_runtime",
    "model_cache_dir",
    "register_cuda_libraries",
    "runtime_status",
]

#: 起こしそのものに要るもの
REQUIRED_PACKAGES: tuple[str, ...] = ("faster-whisper>=1.1",)

#: GPU で動かすために要るもの
CUDA_PACKAGES: tuple[str, ...] = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9.1")

ASR_PACK = FeaturePack(
    key="asr",
    label="字幕起こし",
    required=REQUIRED_PACKAGES,
    extra=CUDA_PACKAGES,
    extra_label="CUDA ランタイム",
    size_mb=300,
    extra_size_mb=1700,
)


def runtime_status() -> PackStatus:
    """いま何が入っているかを調べる

    パッケージを import せずに配布メタデータだけを見る faster-whisper の import は
    数秒かかるうえ、CUDA の DLL 探索まで走るので、状態確認のためにやってよい重さでは
    ない
    """
    return ASR_PACK.status()


#: 足した DLL の置き場の控え ``os.add_dll_directory`` の戻り値は捨てると置き場が外れる
_REGISTERED: dict[str, object] = {}


def cuda_library_dirs(roots: list[str] | None = None) -> list[Path]:
    """pip で入れた CUDA ランタイム（``nvidia-cublas-cu12`` など）の DLL の置き場

    pip の CUDA ランタイムは ``nvidia/cublas/bin`` のようにパッケージの中へ DLL を置く
    CTranslate2 は PATH と既定の置き場しか探さないので、入れただけでは cuBLAS が読めず
    「cublas64_12.dll is not found」で起こしが落ちる（利用者の画面）
    """
    found: list[Path] = []
    for root in roots if roots is not None else sys.path:
        base = Path(root or ".") / "nvidia"
        if not base.is_dir():
            continue
        for package in sorted(base.iterdir()):
            folder = package / "bin"
            if folder.is_dir() and any(folder.glob("*.dll")) and folder not in found:
                found.append(folder)
    return found


def register_cuda_libraries(roots: list[str] | None = None) -> list[Path]:
    """CUDA ランタイムの DLL の置き場を、DLL を探す道へ足す 足した置き場を返す

    PATH の頭にも足す ``os.add_dll_directory`` は ``LoadLibraryEx`` に置き場を探させる
    指定のときしか効かず、読み込み方によっては見てもらえない
    """
    folders = cuda_library_dirs(roots)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    for folder in folders:
        text = str(folder)
        if text not in parts:
            parts.insert(0, text)
        add = getattr(os, "add_dll_directory", None)
        if add is not None and text not in _REGISTERED:
            try:
                _REGISTERED[text] = add(text)
            except OSError:
                continue
    if folders:
        os.environ["PATH"] = os.pathsep.join(parts)
    return folders


def install_command(*, cuda: bool = True, upgrade: bool = False) -> list[str]:
    """導入に使う ``pip`` のコマンド列"""
    return _install_command(ASR_PACK, extra=cuda, upgrade=upgrade)


def model_cache_dir() -> Path:
    """モデルの取得先

    Hugging Face の既定に合わせる ここを独自の場所にすると、他のツールで
    落とし済みのモデルを二重に持つことになる
    """
    override = os.environ.get("HF_HOME")
    if override:
        return Path(override) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"
