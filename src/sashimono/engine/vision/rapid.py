"""飾った文字も読む OCR（#324） RapidOCR（Apache-2.0）を onnxruntime で動かす

Windows の OCR は速いが、飾った題字やゲームの画面の文字はほとんど読めない（実素材の題字で字の 5%）
RapidOCR は 1 枚 1 秒ほどと 10 倍遅いが、同じ題字で 56% を読んだ どちらを使うかは本人が設定で選ぶ
（`Preferences.ocr_engine` 2026-10-21 本人の決定）

モデル（PP-OCRv6 の小さい物 Apache-2.0）は RapidOCR の wheel に入っている物を使う 言語を
変える引数を渡すと、wheel に無いモデルを外から落としに行くので、渡さない（通信する所を増やさない）
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np

from .ocr import OcrUnavailableError, load_onnxruntime_first
from .report import Box, TextLine

__all__ = ["RAPIDOCR_VERSION", "RapidReader"]

#: 入れる版 版を固定するのは、モデルが wheel の中にあり、版が変わると読み方も変わるため
#: （控えの鍵に版が入るので、上げたら読み直す）
RAPIDOCR_VERSION = "3.10.0"


class RapidReader:
    """RapidOCR で 1 枚ずつ読む 作ったスレッドとは別のスレッドからも呼べる"""

    def __init__(self, *, threads: int = 2) -> None:
        # onnxruntime を winrt の包みより先に読む（:func:`~.ocr.load_onnxruntime_first`）
        # 先に Windows の OCR を使っていても、そこで先に読んであるので順は崩れない
        load_onnxruntime_first()
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OcrUnavailableError(
                f"飾った文字の読み取り（RapidOCR）の部品が入っていません（{exc.name or exc}）"
            ) from exc
        self._engine: Any = RapidOCR(
            params={
                # 読めない絵ごとに警告を書き出さない 素材 1 本で数百行になる
                "Global.log_level": "error",
                # 裏で調べる間も編集と再生に CPU を残す（顔の検出と同じ 2 本）
                "EngineConfig.onnxruntime.intra_op_num_threads": max(1, threads),
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
        try:
            self._version = metadata.version("rapidocr")
        except metadata.PackageNotFoundError:
            self._version = RAPIDOCR_VERSION

    @property
    def engine(self) -> str:
        return f"rapidocr:{self._version}"

    def read(self, rgba: np.ndarray) -> tuple[TextLine, ...]:
        """絵（高さ x 幅 x 4 の RGBA）の中の文字を、行ごとに上から返す"""
        height, width = rgba.shape[:2]
        # RapidOCR は OpenCV の並び（BGR）で受け取る
        result = self._engine(np.ascontiguousarray(rgba[:, :, 2::-1]))
        return lines_from(result.boxes, result.txts, (width, height))


def lines_from(boxes: Any, texts: Any, frame: tuple[int, int]) -> tuple[TextLine, ...]:
    """RapidOCR の結果（4 隅の点と文）を行にする 枠は 4 隅を囲む軸に沿った四角"""
    if boxes is None or not texts:
        return ()
    lines: list[TextLine] = []
    for corners, text in zip(np.asarray(boxes, dtype=np.float64), texts, strict=False):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        left, top = float(corners[:, 0].min()), float(corners[:, 1].min())
        right, bottom = float(corners[:, 0].max()), float(corners[:, 1].max())
        lines.append(
            TextLine(cleaned, Box.from_corner(left, top, right - left, bottom - top, frame))
        )
    return tuple(sorted(lines, key=lambda line: (-line.box.y, line.box.x)))
