"""画面の解析（#324）に要る部品と、調べる係の作り方

部品は後から入れる（:class:`~sashimono.runtime.FeaturePack`） 字幕起こしと同じく、使わない人に
負担させない onnxruntime は字幕起こし（faster-whisper の声の区間の検出）と同じ包みを使う
同じ置き場（配布版の ``runtime``）に入るので、字幕起こしを入れた人は増える物が少ない
顔の検出のモデルは PyPI に無いので、導入のボタンで pip の後に落とす（:data:`~.faces.YUNET`）
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sashimono.model_files import models_root
from sashimono.runtime import FeaturePack

from .faces import YUNET, FaceDetector
from .ocr import OcrUnavailableError, WindowsOcr
from .rapid import RAPIDOCR_VERSION, RapidReader
from .report import ScreenKind, ScreenSample, TextLine
from .scan import Detector

__all__ = [
    "MEASURED_DECORATED_RATE",
    "MEASURED_OCR_SECONDS",
    "OCR_ENGINES",
    "OCR_RAPID",
    "OCR_WINDOWS",
    "RAPIDOCR_PACK",
    "RAPIDOCR_PACKAGES",
    "REQUIRED_PACKAGES",
    "VISION_PACK",
    "PartsMissingError",
    "make_detector",
    "ocr_of",
    "parts_note",
    "rapid_note",
]

#: 文字を読む方式（本人の設定 ``Preferences.ocr_engine`` の値と、detect_text の結果の ``ocr``）
#: 既定は Windows の OCR（速い） RapidOCR は飾った文字も読むが 10 倍遅い（2026-10-21 本人の決定）
OCR_WINDOWS = "windows"
OCR_RAPID = "rapidocr"
OCR_ENGINES = (OCR_WINDOWS, OCR_RAPID)

#: 方式ごとの 1 枚の時間（2026-10-21 実素材の 1280x720 の 593 枚の中央値 94 ms と 1001 ms CPU だけ）
#: 設定の説明に出す（速さと読める物の釣り合いを、数字を見て選べるように）
MEASURED_OCR_SECONDS = {OCR_WINDOWS: "720p で 1 枚約 0.1 秒", OCR_RAPID: "720p で 1 枚約 1 秒"}
#: 実素材（ゲームの画面）のいつも出ている飾った題字で、読めた字の割合（60 枚の平均）
MEASURED_DECORATED_RATE = {OCR_WINDOWS: "5%", OCR_RAPID: "56%"}


def ocr_of(engine: str) -> str:
    """結果の部品の名前（``windows-ocr:ja`` ``rapidocr:3.10.0``）から、読んだ方式を引く"""
    return OCR_RAPID if engine.startswith("rapidocr") else OCR_WINDOWS


#: 入れる包み 版の下限は字幕起こしの faster-whisper（``onnxruntime>=1.14,<2``）と両立する所に置く
#: winrt の包みは名前空間ごとに分かれている OCR の結果（行の並び）を読むのに
#: Foundation.Collections が、絵を渡すのに Graphics.Imaging と Storage.Streams が要る
REQUIRED_PACKAGES: tuple[str, ...] = (
    "onnxruntime>=1.20",
    "winrt-runtime>=3.2",
    "winrt-Windows.Foundation>=3.2",
    "winrt-Windows.Foundation.Collections>=3.2",
    "winrt-Windows.Globalization>=3.2",
    "winrt-Windows.Graphics.Imaging>=3.2",
    "winrt-Windows.Media.Ocr>=3.2",
    "winrt-Windows.Storage.Streams>=3.2",
)

VISION_PACK = FeaturePack(
    key="vision",
    label="画面の解析（文字と顔）",
    required=REQUIRED_PACKAGES,
    # onnxruntime（20 MB ほど 字幕起こしを入れていれば既にある）と、winrt の包み
    # （合わせて 1 MB ほど）
    size_mb=25,
    files=(YUNET,),
)

#: 飾った文字の読み取り（RapidOCR）に要る包み 画面の解析の部品とは別に入れる（使わない人に
#: OpenCV など 90 MB ほどを負わせない 2026-10-21 本人の決定） onnxruntime は画面の解析と同じ
#: 包みを使う 版は固定する（モデルが wheel の中にあり、版で読み方が変わる） OpenCV も固定する
#: のは、配布版の中で読めることを確かめた版だけを入れるため（--add-on-check）
RAPIDOCR_PACKAGES: tuple[str, ...] = (
    "onnxruntime>=1.20",
    f"rapidocr=={RAPIDOCR_VERSION}",
    "opencv-python-headless==5.0.0.93",
)

RAPIDOCR_PACK = FeaturePack(
    key="rapidocr",
    label="飾った文字の読み取り（RapidOCR）",
    required=RAPIDOCR_PACKAGES,
    # RapidOCR（wheel に入ったモデル 30 MB ほど）と OpenCV（50 MB ほど）と小さな依存
    size_mb=90,
)


class PartsMissingError(RuntimeError):
    """調べる部品が使えない 文はそのまま AI と画面へ出す断りの理由"""


def parts_note(detail: str) -> str:
    """部品が無いときの断りの文 どこから入れるかを添える"""
    return (
        f"画面の解析の部品が使えません（{detail}） 〔表示〕→〔設定…〕の"
        "〔画面の解析の部品…〕から入れてください"
    )


def rapid_note(detail: str) -> str:
    """飾った文字の読み取り（RapidOCR）を選んだのに使えないときの断りの文

    黙って Windows の OCR に落とさない（2026-10-21 本人の決定） 落とすと、飾った文字を読ませる
    つもりの人が、読めなかった結果を RapidOCR の結果だと思い込む
    """
    return (
        f"OCR の方式に「飾った文字も読む（RapidOCR）」が選ばれていますが、使えません（{detail}）"
        " 〔表示〕→〔設定…〕の〔飾った文字の読み取りの部品…〕から入れるか、OCR の方式を"
        "「Windows の OCR」に戻してください"
    )


class _Reader(Protocol):
    @property
    def engine(self) -> str: ...

    def read(self, rgba: np.ndarray) -> tuple[TextLine, ...]: ...


class _TextReader:
    def __init__(self, ocr: _Reader) -> None:
        self._ocr = ocr

    @property
    def kind(self) -> ScreenKind:
        return ScreenKind.TEXT

    @property
    def engine(self) -> str:
        return self._ocr.engine

    def look(self, rgba: np.ndarray, seconds: float) -> ScreenSample:
        return ScreenSample(seconds=seconds, lines=self._ocr.read(rgba))


class _FaceFinder:
    def __init__(self, detector: FaceDetector) -> None:
        self._detector = detector

    @property
    def kind(self) -> ScreenKind:
        return ScreenKind.FACES

    @property
    def engine(self) -> str:
        return self._detector.engine

    def look(self, rgba: np.ndarray, seconds: float) -> ScreenSample:
        return ScreenSample(seconds=seconds, faces=self._detector.detect(rgba))


def make_detector(kind: ScreenKind, ocr: str = OCR_WINDOWS, *, threads: int = 2) -> Detector:
    """調べる係を作る 使えなければ :class:`PartsMissingError`（理由と入れ方の文）

    ``ocr`` は文字を読む方式（:data:`OCR_ENGINES` 本人の設定 ``Preferences.ocr_engine``）
    重い（onnxruntime を読む・モデルを読む）ので、調べる係のスレッドで呼ぶ
    """
    if kind is ScreenKind.TEXT and ocr == OCR_RAPID:
        try:
            return _TextReader(RapidReader(threads=threads))
        except OcrUnavailableError as exc:
            raise PartsMissingError(rapid_note(str(exc))) from exc
        except Exception as exc:  # onnxruntime と OpenCV の読み込みの失敗は独自の例外で来る
            raise PartsMissingError(rapid_note(f"読み込めません: {exc}")) from exc
    if kind is ScreenKind.TEXT:
        if ocr not in OCR_ENGINES:
            raise PartsMissingError(f"知らない OCR の方式です: {ocr}")
        try:
            return _TextReader(WindowsOcr())
        except OcrUnavailableError as exc:
            raise PartsMissingError(parts_note(str(exc))) from exc
        except OSError as exc:
            # Windows の外で winrt を読んだときなど 呼ぶ所で落ちる物も断りにする
            raise PartsMissingError(parts_note(f"Windows の OCR を呼べません: {exc}")) from exc
    model = YUNET.path(models_root())
    if not YUNET.present(models_root()):
        raise PartsMissingError(parts_note(f"{YUNET.label} がありません"))
    try:
        return _FaceFinder(FaceDetector(model, threads=threads))
    except ImportError as exc:
        raise PartsMissingError(
            parts_note(f"onnxruntime が入っていません（{exc.name or exc}）")
        ) from exc
    except Exception as exc:  # onnxruntime の読み込みの失敗は独自の例外で来る
        raise PartsMissingError(parts_note(f"顔の検出のモデルを読めません: {exc}")) from exc
