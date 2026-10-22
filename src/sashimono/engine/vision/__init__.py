"""画面の中身を調べる（#324 A6） 画面の中の文字（OCR）と顔の位置

場面の切り替わり（#319）で素材を区切り、場面ごとに数枚だけを調べる 結果は
``engine/cache`` に素材の鍵で覚える（:mod:`sashimono.engine.cache.screen_cache`）
部品（onnxruntime・Windows の OCR の包み・顔の検出のモデル）は後から入れる物で、
入っていなければ理由を返して断る（:class:`~.parts.PartsMissingError`）
"""

from .faces import YUNET
from .parts import (
    OCR_ENGINES,
    OCR_RAPID,
    OCR_WINDOWS,
    RAPIDOCR_PACK,
    VISION_PACK,
    PartsMissingError,
    make_detector,
    ocr_of,
    parts_note,
)
from .report import Box, Face, ScreenKind, ScreenReport, ScreenSample, TextLine
from .sampling import fallback_cuts, sample_times
from .scan import Detector, ScreenScan, scan_screen

__all__ = [
    "OCR_ENGINES",
    "OCR_RAPID",
    "OCR_WINDOWS",
    "RAPIDOCR_PACK",
    "VISION_PACK",
    "YUNET",
    "Box",
    "Detector",
    "Face",
    "PartsMissingError",
    "ScreenKind",
    "ScreenReport",
    "ScreenSample",
    "ScreenScan",
    "TextLine",
    "fallback_cuts",
    "make_detector",
    "ocr_of",
    "parts_note",
    "sample_times",
    "scan_screen",
]
