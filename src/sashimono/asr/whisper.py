"""faster-whisper（CTranslate2）による起こし

このモジュールは**読み込まれただけでは何も import しない** faster-whisper の
import は数秒かかり、CUDA の DLL 探索まで走る 字幕を使わない起動でその代償を
払わせないため、実際に起こすときまで遅らせている

導入されていない環境でも :meth:`FasterWhisperBackend.is_available` は落ちずに
偽を返す これが「未導入の状態で起動し、必要になったらソフト内から入れる」という
配布方針の前提になる（:mod:`sashimono.asr.environment` を参照）
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sashimono.asr.backend import (
    AsrError,
    Progress,
    ShouldCancel,
    TranscribeOptions,
    to_source_time,
)
from sashimono.asr.environment import register_cuda_libraries, runtime_status
from sashimono.core.model import Transcript, TranscriptSegment, Word
from sashimono.engine.decode import AudioDecoder, ProbeError, probe_media

#: faster-whisper が受け取る音の形（16kHz・モノラル・float32）
WHISPER_SAMPLE_RATE = 16000
#: 音を読むときの 1 回の長さ（秒） 長い素材を 1 回で読むと、途中で止める機会が無い
_READ_CHUNK_SECONDS = 60

__all__ = ["FasterWhisperBackend", "gpu_fallback_notice", "is_cuda_library_error"]

#: GPU の道具（CUDA のライブラリ・ドライバ）が無い・読めないときの失敗の文言
_CUDA_WORDS = re.compile(
    r"cublas|cudnn|cudart|cufft|curand|nvrtc|cuda|\.dll|no cuda-capable device", re.IGNORECASE
)
#: 失敗の文言から読めなかったライブラリの名前を拾う
_LIBRARY = re.compile(r"([A-Za-z0-9_]+\.dll|lib[A-Za-z0-9_]+\.so[.0-9]*)")


def is_cuda_library_error(error: BaseException) -> bool:
    """GPU の道具が無い・読めないことによる失敗か CPU へ落とせば起こせる物"""
    return bool(_CUDA_WORDS.search(str(error)))


def gpu_fallback_notice(error: BaseException) -> str:
    """GPU で起こせず CPU へ落としたときの知らせ 何が足りないかと、入れ方を言う"""
    found = _LIBRARY.search(str(error))
    what = f"{found.group(1)} が読めない" if found else "GPU の道具（CUDA）が使えない"
    status = runtime_status()
    if status.extras and not status.extra_installed:
        size = status.pack.extra_size_mb / 1000
        hint = (
            f"字幕起こしの窓で「GPU を使う」に印を付けて「環境を更新」を押すと、"
            f"{status.pack.extra_label}（約 {size:.1f} GB）が入り GPU で起こせます"
        )
    else:
        hint = (
            f"{status.pack.extra_label}は入っていますが読み込めませんでした"
            " NVIDIA のドライバを新しくするか、「環境を更新」で入れ直してください"
        )
    return f"{what}ので CPU で起こしました（GPU より遅くなります） {hint}"


class _CudaMissingError(Exception):
    """GPU の道具が読めずに止まった 呼ぶ側が CPU で起こし直す"""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


class FasterWhisperBackend:
    """faster-whisper を呼ぶバックエンド

    読み込んだモデルは保持する 1 本目と 2 本目で同じモデルなら、2 回目は
    数秒の読み込みを省ける
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded_with: tuple[str, str, str] | None = None
        #: 起こし終えたときに添える知らせ（GPU から CPU へ落とした など）
        self._notice = ""

    def take_notice(self) -> str:
        """最後の起こしで添える知らせを受け取る 受け取ったら空にする"""
        notice, self._notice = self._notice, ""
        return notice

    @property
    def name(self) -> str:
        return "faster-whisper"

    def is_available(self) -> bool:
        return runtime_status().ready

    def unload(self) -> None:
        """モデルを解放する GPU のメモリを書き出しへ譲りたいときに呼ぶ"""
        self._model = None
        self._loaded_with = None

    def transcribe(
        self,
        path: Path,
        options: TranscribeOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> Transcript | None:
        source = Path(path)
        if not source.exists():
            raise AsrError(f"素材が見つからない: {source}")

        self._notice = ""
        if progress is not None:
            progress(0.0, "モデルを読み込んでいる")
        try:
            model = self._load(options)
        except _CudaMissingError as missing:
            options, model = self._fall_back(options, missing, progress)
        if should_cancel is not None and should_cancel():
            return None

        audio = _media_audio(source, should_cancel, options.audio_stream)
        if audio is None:
            return None

        try:
            return self._run(model, audio, options, progress, should_cancel)
        except _CudaMissingError as missing:
            # 起こし始めてから cuBLAS を読みに行く版がある（利用者の画面はこちら）
            # 音は読み直さずに CPU で頭から起こし直す
            options, model = self._fall_back(options, missing, progress)
            return self._run(model, audio, options, progress, should_cancel)

    def _load(self, options: TranscribeOptions) -> Any:
        """モデルを読む GPU の道具が読めなければ :class:`_CudaMissingError`"""
        try:
            return self._ensure_model(options)
        except AsrError as exc:
            cause = exc.__cause__ or exc
            if options.device != "cpu" and is_cuda_library_error(cause):
                raise _CudaMissingError(cause) from exc
            raise

    def _fall_back(
        self, options: TranscribeOptions, missing: _CudaMissingError, progress: Progress | None
    ) -> tuple[TranscribeOptions, Any]:
        """CPU で読み直したモデル 何が足りないかを知らせに残す"""
        self._notice = gpu_fallback_notice(missing.cause)
        if progress is not None:
            progress(0.0, self._notice)
        cpu = replace(options, device="cpu", compute_type="int8")
        return cpu, self._ensure_model(cpu)

    def _run(
        self,
        model: Any,
        audio: np.ndarray | str,
        options: TranscribeOptions,
        progress: Progress | None,
        should_cancel: ShouldCancel | None,
    ) -> Transcript | None:
        """読んだ音をモデルへ渡して起こす GPU の道具が読めなければ :class:`_CudaMissingError`"""
        if progress is not None:
            progress(0.02, "音声を解析している")
        gpu = options.device != "cpu"
        try:
            segments, info = model.transcribe(
                audio,
                language=options.language,
                beam_size=options.beam_size,
                vad_filter=options.vad_filter,
                word_timestamps=options.word_timestamps,
                initial_prompt=options.initial_prompt or None,
            )
        except Exception as exc:  # faster-whisper は独自の例外型を公開していない
            if gpu and is_cuda_library_error(exc):
                raise _CudaMissingError(exc) from exc
            raise AsrError(f"起こしを開始できない: {exc}") from exc

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        language = str(getattr(info, "language", "") or options.language or "")

        collected: list[TranscriptSegment] = []
        try:
            # segments は生成器で、回した分だけ認識が進む ここで中断を見るので、
            # 「止めたのに GPU が回り続ける」状態にならない
            for raw in segments:
                if should_cancel is not None and should_cancel():
                    return None
                converted = _to_segment(raw)
                if converted is not None:
                    collected.append(converted)
                if progress is not None and duration > 0:
                    ratio = min(1.0, float(getattr(raw, "end", 0.0)) / duration)
                    progress(max(0.02, ratio), f"{len(collected)} 文を起こした")
        except Exception as exc:
            if gpu and is_cuda_library_error(exc):
                raise _CudaMissingError(exc) from exc
            raise AsrError(f"起こしに失敗した: {exc}") from exc

        if progress is not None:
            progress(1.0, f"{len(collected)} 文")
        return Transcript(
            segments=tuple(_ordered(collected)),
            language=language,
            model=options.model,
        )

    def _ensure_model(self, options: TranscribeOptions) -> Any:
        key = (options.model, options.device, options.compute_type)
        if self._model is not None and self._loaded_with == key:
            return self._model

        if options.device != "cpu":
            # pip で入れた CUDA ランタイムの DLL を探す道へ足す 足さないと入れてあっても
            # cuBLAS が見つからず、起こし始めた所で落ちる
            register_cuda_libraries()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrError(
                "起こしの実行環境が入っていません 字幕パネルの「環境を導入」から用意してください"
            ) from exc

        try:
            self._model = WhisperModel(
                options.model,
                device=options.device,
                compute_type=options.compute_type,
            )
        except Exception as exc:
            raise AsrError(f"モデルを読み込めない ({options.model}): {exc}") from exc
        self._loaded_with = key
        return self._model


def _media_audio(
    source: Path, should_cancel: ShouldCancel | None, stream: int | None = None
) -> np.ndarray | str | None:
    """素材の音を faster-whisper の形で読む 止められたら ``None``

    ``stream`` は起こす音声ストリームの番号 ``None`` なら 1 本目 素材に無い番号は断る
    （黙って 1 本目を起こすと、声ではなくゲームの音を起こしたことに気付けない）

    素材の時刻の原点（:func:`~sashimono.engine.decode.probe.media_origin`）から数えて読む
    パスを渡して faster-whisper に読ませると、音の最初のサンプルを 0 秒として数えるので、
    音の頭が原点と違う素材（AAC の前置き・音が映像より早く始まる物）では、起こした字幕が
    その差の分だけずれる（Issue #125） 原点より前の音（前置きなど）は置いたクリップでも
    鳴らない区間なので、起こさなくてよい

    長さの分からない素材（調べても 0 と出た物）は、読む量を決められないので前と同じく
    パスを渡す 音の頭と原点の差の分はずれうるが、断って起こせないよりよい
    """
    try:
        item = probe_media(source)
        chunk = _READ_CHUNK_SECONDS * WHISPER_SAMPLE_RATE
        total = int(item.duration * WHISPER_SAMPLE_RATE)
        # 配列を作る前に断る 音の無い長い動画で先に全長の配列を作ると、音が無いと
        # 分かる前に大きな確保が走る
        if not item.audio_streams:
            raise AsrError(f"音声が無い: {source}")
        if stream is not None and all(s.index != stream for s in item.audio_streams):
            raise AsrError(f"その音声は素材に無い（番号 {stream}）: {source}")
        if total <= 0 and stream is not None and stream != item.audio_streams[0].index:
            # 長さが分からないとパスを渡すしかなく、faster-whisper は 1 本目しか読まない
            raise AsrError(f"長さが分からない素材では 2 本目以降の音声を起こせない: {source}")
        if total <= 0:
            audio: np.ndarray | str = str(source)
        else:
            # 全長の配列を先に 1 つだけ作って書き込む 読んだ分を貯めてから最後につなぐと、
            # つなぐ瞬間に同じ長さの配列が 2 つ並ぶ（1 時間で 230MB が 460MB になる）
            audio = np.empty(total, dtype=np.float32)
            with AudioDecoder(
                source, sample_rate=WHISPER_SAMPLE_RATE, channels=1, stream_index=stream
            ) as decoder:
                for start in range(0, total, chunk):
                    if should_cancel is not None and should_cancel():
                        return None
                    count = min(chunk, total - start)
                    audio[start : start + count] = decoder.read(start, count)[:, 0]
                    # 壊れた所から先は無音で返ってくる そのまま渡すと、そこから先の字幕が
                    # 欠けたのに起こしは成功したように見える
                    if decoder.decode_error is not None:
                        # 失敗した位置が分かればそこを出す 読んだ塊の頭を出すと、
                        # 実際より最大で 1 塊（60 秒）手前から読めないように見える
                        at = decoder.decode_error_at
                        seconds = (start if at is None else at) / WHISPER_SAMPLE_RATE
                        raise AsrError(
                            f"音声の {seconds:.1f} 秒から先に読めない所がある: "
                            f"{decoder.decode_error}"
                        )
    except ProbeError as exc:
        raise AsrError(f"音声を読めない: {exc}") from exc
    except (MemoryError, ValueError) as exc:
        # 全長の配列を作る所だけでなく、読むたびの配列でも足りなくなりうる ValueError は
        # 配列の大きさが NumPy の上限を越えたとき（異常な長さの素材） 起こしの失敗として
        # 出す 素のまま投げると起こしの枠の外（想定外の失敗）になる
        raise AsrError(f"音声を読むメモリが足りない: {exc}") from exc
    # 読み終わりにも見る 見ないと、1 回で読み切る短い素材や最後の読み込みの間、長さの
    # 分からない素材を調べている間に止めても、そのままモデルへ渡して起こしが始まる
    if should_cancel is not None and should_cancel():
        return None
    return audio


def _to_segment(raw: Any) -> TranscriptSegment | None:
    """faster-whisper のセグメントをモデルの形へ"""
    text = str(getattr(raw, "text", "")).strip()
    if not text:
        return None
    start = to_source_time(float(getattr(raw, "start", 0.0)))
    end = to_source_time(float(getattr(raw, "end", 0.0)))
    if end < start:
        end = start

    words: list[Word] = []
    for entry in getattr(raw, "words", None) or ():
        word_text = str(getattr(entry, "word", "")).strip()
        if not word_text:
            continue
        word_start = to_source_time(float(getattr(entry, "start", 0.0)))
        word_end = to_source_time(float(getattr(entry, "end", 0.0)))
        words.append(Word(start=word_start, end=max(word_start, word_end), text=word_text))

    return TranscriptSegment(start=start, end=end, text=text, words=tuple(words))


def _ordered(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """開始時刻の昇順に整える

    :class:`~sashimono.core.model.Transcript` は順序を不変条件にしている 認識器が
    まれに前後した時刻を返すので、モデルへ渡す前にここで揃える 例外にして
    起こし全体を捨てるのは割に合わない
    """
    return sorted(segments, key=lambda s: (s.start, s.end))
