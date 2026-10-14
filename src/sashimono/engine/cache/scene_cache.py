"""場面の解析（フレーム間の差の並び）の永続化（#321）

覚えるのは感度を掛ける前の差の並び（:class:`~sashimono.engine.scenes.SceneSignal`）
感度を変えても読み直さずに済む 読み終えた物だけを書く 途中で止めた物を書くと、
次に開いたときに途中までの並びを「読み終えた」と取り違える
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sashimono.engine.scenes import SAMPLE_WIDTH, SceneSignal

from .store import CacheStore, load_arrays, media_key, save_arrays

__all__ = ["NAMESPACE", "SUFFIX", "load_scenes", "save_scenes", "scenes_key"]

NAMESPACE = "scenes"
SUFFIX = ".scenes.npz"

#: 保存形式の版 差の測り方（縮め方・ヒストグラムの段）を変えたら上げる
#: 上げないと、前の測り方の並びに今の感度を掛けることになり、拾う所がずれる
FORMAT_VERSION = 1


def scenes_key(path: Path) -> str:
    """控えの鍵 素材の場所・大きさ・更新時刻（:func:`media_key`）と測り方から作る

    素材を差し替えると大きさか更新時刻が変わり、別の鍵になって読み直す
    """
    return media_key(path, extra=f"sc{FORMAT_VERSION}:{SAMPLE_WIDTH}")


def save_scenes(store: CacheStore, key: str, signal: SceneSignal) -> Path | None:
    """読み終えた並びを書く 途中の物は書かずに ``None``"""
    if not signal.complete:
        return None
    return save_arrays(
        store.prepare(NAMESPACE, key, SUFFIX),
        {
            "version": np.array([FORMAT_VERSION]),
            "times": signal.times,
            "pixel": signal.pixel,
            "histogram": signal.histogram,
            "analyzed": np.array([signal.analyzed], np.float64),
        },
        compressed=True,
    )


def load_scenes(store: CacheStore, key: str) -> SceneSignal | None:
    """書いた並びを読む 無い・壊れている・版が違うなら ``None``（読み直す）"""
    data = load_arrays(store.path_for(NAMESPACE, key, SUFFIX))
    if data is None:
        return None
    try:
        if int(data["version"][0]) != FORMAT_VERSION:
            return None
        times = np.asarray(data["times"], np.float64)
        pixel = np.asarray(data["pixel"], np.float32)
        histogram = np.asarray(data["histogram"], np.float32)
        if not times.shape == pixel.shape == histogram.shape or times.ndim != 1:
            return None
        return SceneSignal(
            times=times,
            pixel=pixel,
            histogram=histogram,
            analyzed=float(data["analyzed"][0]),
            complete=True,
        )
    except (KeyError, IndexError, ValueError):
        return None
