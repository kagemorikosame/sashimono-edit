"""画面の中の文字を読む（#324） Windows の OCR（``Windows.Media.Ocr``）を winrt の包みで呼ぶ

Windows 11 に付いている物を使う 入れる物は呼ぶための包み（MIT 数百 KB）だけで、外へは送らず、
日本語の Windows なら日本語の読み取りが最初から入っている 精度が足りないと分かったら
RapidOCR（onnxruntime を共有できる）を次の段にする（#324 の比べた案）

日本語は 1 文字ずつを語として返し、行の文は語の間に空白を挟む 「本 日 の お す す め」の
まま AI へ渡すと、字幕や検索の言葉と突き合わなくなるので、英数字どうしの間だけに空白を残す
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from .faces import shrink
from .report import Box, TextLine

__all__ = ["OcrUnavailableError", "WindowsOcr", "join_words", "load_onnxruntime_first"]


class OcrUnavailableError(RuntimeError):
    """この機械では読めない（包みが無い・Windows でない・読み取りの言語が無い） 文は断りの理由"""


def _ascii_word(character: str) -> bool:
    return character.isascii() and character.isalnum()


def join_words(words: Sequence[str]) -> str:
    """語を 1 行の文へつなぐ 英数字どうしの間だけ空白を挟む"""
    text = ""
    for word in words:
        if not word:
            continue
        if text and _ascii_word(text[-1]) and _ascii_word(word[0]):
            text += " "
        text += word
    return text


def load_onnxruntime_first() -> None:
    """winrt の包みより先に onnxruntime を読んでおく（入っていれば）

    winrt の包み（winrt-runtime 3.2）は自分の ``msvcp140.dll`` を持っていて、先に読むと
    その古い DLL がプロセスに残る 後から onnxruntime（1.31）を読むと、新しい DLL を求める所で
    アプリごと落ちる（access violation 手元の Python 3.14 で確かめた） 逆の順なら両方動く
    文字を先に頼まれてから顔を頼まれても落ちないよう、winrt を読む所で必ず先に読む
    """
    with contextlib.suppress(ImportError):
        import onnxruntime  # noqa: F401  読むだけ（DLL を先に入れる）


def _language_note() -> str:
    return (
        "Windows の OCR の言語が入っていません 〔設定〕→〔時刻と言語〕→〔言語と地域〕で"
        "日本語を選び、言語のオプションの「光学式文字認識」を入れてください"
    )


class WindowsOcr:
    """Windows の OCR で 1 枚ずつ読む 作ったスレッドとは別のスレッドからも呼べる"""

    def __init__(self, language: str | None = None) -> None:
        load_onnxruntime_first()
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
            from winrt.windows.media.ocr import OcrEngine
            from winrt.windows.storage.streams import DataWriter
        except ImportError as exc:
            # Windows の外（CI の一部）か、部品を入れていない 入れ方を断りの理由にする
            raise OcrUnavailableError(
                f"Windows の OCR を呼ぶ部品が入っていません（{exc.name or exc}）"
            ) from exc
        self._bitmap: Any = SoftwareBitmap
        self._format: Any = BitmapPixelFormat
        self._writer: Any = DataWriter
        engine: Any = None
        if language is not None:
            engine = OcrEngine.try_create_from_language(Language(language))
        if engine is None:
            # 利用者の言語の設定の順に選ぶ 日本語の Windows なら日本語
            engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise OcrUnavailableError(_language_note())
        self._engine: Any = engine
        self._limit = int(OcrEngine.max_image_dimension)
        self.language = str(engine.recognizer_language.language_tag)

    @property
    def engine(self) -> str:
        """結果と控えに書く部品の名前 言語が変われば読める文字も変わるので含める"""
        return f"windows-ocr:{self.language}"

    def read(self, rgba: np.ndarray) -> tuple[TextLine, ...]:
        """絵（高さ x 幅 x 4 の RGBA）の中の文字を、行ごとに上から返す"""
        height, width = rgba.shape[:2]
        scale = min(1.0, self._limit / max(height, width, 1))
        picture = rgba
        if scale < 1.0:
            # 扱える大きさ（Windows 11 で 10000 画素）を超えると読み取りが断る
            picture = shrink(rgba, max(1, int(width * scale)), max(1, int(height * scale)))
            picture = picture.round().clip(0, 255).astype(np.uint8)
        bgra = np.ascontiguousarray(picture[:, :, [2, 1, 0, 3]])
        writer = self._writer()
        writer.write_bytes(bgra.tobytes())
        bitmap = self._bitmap.create_copy_from_buffer(
            writer.detach_buffer(), self._format.BGRA8, bgra.shape[1], bgra.shape[0]
        )
        result = asyncio.run(self._recognize(bitmap))
        lines: list[TextLine] = []
        for line in result.lines:
            words = list(line.words)
            text = join_words([str(word.text) for word in words]).strip()
            if not text or not words:
                continue
            rects = [word.bounding_rect for word in words]
            left = min(float(rect.x) for rect in rects) / scale
            top = min(float(rect.y) for rect in rects) / scale
            right = max(float(rect.x) + float(rect.width) for rect in rects) / scale
            bottom = max(float(rect.y) + float(rect.height) for rect in rects) / scale
            box = Box.from_corner(left, top, right - left, bottom - top, (width, height))
            lines.append(TextLine(text=text, box=box))
        # 上から順に 作品の座標は上が正なので、y の大きい方が上
        return tuple(sorted(lines, key=lambda line: (-line.box.y, line.box.x)))

    async def _recognize(self, bitmap: Any) -> Any:
        return await self._engine.recognize_async(bitmap)
