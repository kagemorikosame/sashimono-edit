"""場面の解析（フレーム間の差の並び）の永続化（#321）

覚えるのは感度を掛ける前の差の並び（:class:`~sashimono.engine.scenes.SceneSignal`）
感度を変えても読み直さずに済む 書くのは、読み終えた物と、素材が途中で壊れていて
読めた所までで終えた物（``broken``）だけ 途中で止めた物を書くと、次に開いたときに
途中までの並びを「読み終えた」と取り違える 壊れた物は読み直しても同じ所で止まるので、
未完了の印を付けたまま覚え、素材が変わる（大きさか更新時刻）までは読み直さない
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from sashimono.engine.scenes import SAMPLE_WIDTH, SceneSignal
from sashimono.engine.subject import SUBJECT_BINS, SubjectTrack

from .store import CacheStore, load_arrays, media_key, save_arrays

__all__ = ["NAMESPACE", "SUFFIX", "load_scenes", "save_scenes", "scenes_key"]

NAMESPACE = "scenes"
SUFFIX = ".scenes.npz"

#: 保存形式の版 差の測り方（縮め方・ヒストグラムの段）を変えたら上げる
#: 上げないと、前の測り方の並びに今の感度を掛けることになり、拾う所がずれる
#: 2 で調べた映像ストリームの番号を持たせた（PR #358） 1 の物は番号が無く、映像が何本もある
#: 素材でどのクリップに当てはまるかが分からないので読み直す
#: 3 で途中で壊れた素材の印を持たせた（PR #358） 2 までは壊れた素材も読み終えた物として
#: 書いていたので、後ろの切り替わりの無い一覧が残っている 読み直す
#: 4 で被写体のいそうな所（#334 縦動画への再構図）を持たせた 3 までの物には無く、縦の作品へ
#: 作り替えるときに枠を真ん中から動かせないので読み直す
FORMAT_VERSION = 4

#: 番号が分からない（``None``）ときに書く値 ストリームの番号は 0 以上
_NO_STREAM = -1


def scenes_key(path: Path) -> str:
    """控えの鍵 素材の場所・大きさ・更新時刻（:func:`media_key`）と測り方から作る

    素材を差し替えると大きさか更新時刻が変わり、別の鍵になって読み直す
    """
    return media_key(path, extra=f"sc{FORMAT_VERSION}:{SAMPLE_WIDTH}")


def save_scenes(store: CacheStore, key: str, signal: SceneSignal) -> Path | None:
    """読み終えた並びか、壊れていて読めた所までの並びを書く 途中で止めた物は書かずに ``None``"""
    if not signal.complete and not signal.broken:
        return None
    return save_arrays(
        store.prepare(NAMESPACE, key, SUFFIX),
        {
            "version": np.array([FORMAT_VERSION]),
            "times": signal.times,
            "pixel": signal.pixel,
            "histogram": signal.histogram,
            "analyzed": np.array([signal.analyzed], np.float64),
            "stream": np.array([_NO_STREAM if signal.stream is None else signal.stream]),
            "broken": np.array([signal.broken]),
            # 分布は 0〜1 の割合なので半精度で足りる 1 時間の素材で 1MB ほどに収まる
            "subject_times": signal.subject.times,
            "subject_columns": signal.subject.columns.astype(np.float16),
            "subject_rows": signal.subject.rows.astype(np.float16),
            "subject_strength": signal.subject.strength,
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
        stream = int(data["stream"][0])
        broken = bool(data["broken"][0])
        subject = _subject(data)
        if subject is None:
            return None
        return SceneSignal(
            times=times,
            pixel=pixel,
            histogram=histogram,
            analyzed=float(data["analyzed"][0]),
            complete=not broken,
            stream=None if stream == _NO_STREAM else stream,
            broken=broken,
            subject=subject,
        )
    except (KeyError, IndexError, ValueError):
        return None


def _subject(data: Mapping[str, np.ndarray]) -> SubjectTrack | None:
    """被写体の並び 形が合わなければ ``None``（読み直す） 合わないまま使うと、行と時刻が
    ずれた所の分布で枠の位置を決める
    """
    times = np.asarray(data["subject_times"], np.float64)
    columns = np.asarray(data["subject_columns"], np.float32)
    rows = np.asarray(data["subject_rows"], np.float32)
    strength = np.asarray(data["subject_strength"], np.float32)
    count = times.shape[0]
    if (
        times.ndim != 1
        or columns.shape != (count, SUBJECT_BINS)
        or rows.shape != (count, SUBJECT_BINS)
        or strength.shape != (count,)
    ):
        return None
    return SubjectTrack(times=times, columns=columns, rows=rows, strength=strength)
