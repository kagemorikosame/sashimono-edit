"""画面の解析（#324）に要る部品と、調べる係の作り方

部品は後から入れる（:class:`~sashimono.runtime.FeaturePack`） 字幕起こしと同じく、使わない人に
負担させない onnxruntime は字幕起こし（faster-whisper の声の区間の検出）と同じ包みを使う
同じ置き場（配布版の ``runtime``）に入るので、字幕起こしを入れた人は増える物が少ない
顔の検出のモデルは PyPI に無いので、導入のボタンで pip の後に落とす（:data:`~.faces.YUNET`）
"""

from __future__ import annotations

import numpy as np

from sashimono.model_files import models_root
from sashimono.runtime import FeaturePack

from .faces import YUNET, FaceDetector
from .ocr import OcrUnavailableError, WindowsOcr
from .report import ScreenKind, ScreenSample
from .scan import Detector

__all__ = [
    "REQUIRED_PACKAGES",
    "VISION_PACK",
    "PartsMissingError",
    "make_detector",
    "parts_note",
]

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


class PartsMissingError(RuntimeError):
    """調べる部品が使えない 文はそのまま AI と画面へ出す断りの理由"""


def parts_note(detail: str) -> str:
    """部品が無いときの断りの文 どこから入れるかを添える"""
    return (
        f"画面の解析の部品が使えません（{detail}） 〔表示〕→〔設定…〕の"
        "〔画面の解析の部品…〕から入れてください"
    )


class _TextReader:
    def __init__(self, ocr: WindowsOcr) -> None:
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


def make_detector(kind: ScreenKind, *, threads: int = 2) -> Detector:
    """調べる係を作る 使えなければ :class:`PartsMissingError`（理由と入れ方の文）

    重い（onnxruntime を読む・モデルを読む）ので、調べる係のスレッドで呼ぶ
    """
    if kind is ScreenKind.TEXT:
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
