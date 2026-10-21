"""話者の分離（#323） 誰がどの時刻に話しているかを、手元の機械だけで求める

部品は sherpa-onnx（Apache-2.0）で、字幕起こしとは別の包み（:data:`SPEAKER_PACK`）にして
使う人だけが入れる（2026-10-20 本人の決定） 字幕起こしに混ぜると、1 人で話す動画しか
作らない人まで落とすことになる 音は外へ送らない

モデルは 2 つで、どちらも登録の要らない Hugging Face の置き場から字幕起こしと同じ置き場
（:func:`sashimono.asr.environment.model_cache_dir`）へ落とす

- 区間の切り出し: pyannote segmentation-3.0 を ONNX にした物の int8（1.5 MB MIT）
- 声の特徴: 3D-Speaker の CAM++（中国語と英語 26 MB Apache-2.0）

このモジュールは読み込まれただけでは sherpa-onnx を import しない 字幕パネルを開くたびに
DLL を探しに行かないように、使う所で初めて読む
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sashimono.asr.backend import AsrError, Progress, ShouldCancel, to_source_time
from sashimono.core.speakers import SpeakerTurn
from sashimono.runtime import FeaturePack, PackStatus
from sashimono.runtime import install_command as _install_command

__all__ = [
    "EMBEDDING_FILE",
    "EMBEDDING_REPOSITORY",
    "MAX_SPEAKERS",
    "SEGMENTATION_FILE",
    "SEGMENTATION_REPOSITORY",
    "SPEAKER_PACK",
    "SherpaSpeakerBackend",
    "SpeakerBackend",
    "SpeakerOptions",
    "install_command",
    "runtime_status",
    "separate_with_retries",
]

#: 分離に要るもの huggingface-hub はモデルを落とすため 字幕起こしを入れた人には既にある
REQUIRED_PACKAGES: tuple[str, ...] = ("sherpa-onnx>=1.13", "huggingface-hub>=0.21")

SPEAKER_PACK = FeaturePack(
    key="speaker",
    label="話者の分離",
    required=REQUIRED_PACKAGES,
    # 2026-10-20 に Python 3.14 の Windows で入れて測った wheel の合計（sherpa-onnx 2.4 MB と、
    # onnxruntime.dll を持つ sherpa-onnx-core 16 MB と、huggingface-hub とその依存）
    # モデル（28 MB）は初めて分けるときに落とす
    size_mb=25,
)

#: 区間の切り出しのモデル 落とす先は字幕起こしのモデルと同じ置き場
SEGMENTATION_REPOSITORY = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_FILE = "model.int8.onnx"
#: 声の特徴のモデル 中国語と英語で学んだ物 日本語の合成の 3 人の声（2026-10-20）を 1 か所の
#: 取り違えだけで分けた
EMBEDDING_REPOSITORY = "csukuangfj/speaker-embedding-models"
EMBEDDING_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
#: モデルの大きさ（MB） 初めての取得で何が落ちるかを窓に出す
MODEL_SIZE_MB = 28

#: 分ける音の形（16kHz・モノラル・float32） 区間の切り出しのモデルはこれしか受けない
SAMPLE_RATE = 16000
#: 人数を自動で決めるときの、同じ人とみなす近さ sherpa-onnx の既定 日本語の合成の 3 人の声で
#: 3 人に分かれた
CLUSTER_THRESHOLD = 0.5
#: 渡せる人数の上限 区間の切り出しは 10 秒の中に 3 人までしか見ないが、長い対談では
#: 入れ替わりで 10 人ほどまで出る これより多いと聞き分けの当てにならない
MAX_SPEAKERS = 10
#: 人数を渡したのに少なく分かれたときに、まとめ直す回数 sherpa-onnx の速いまとめは小さな
#: 塊を落とすので、渡した数より 1 人少なく出ることがある（合成の 3 人の声で 3 を渡すと
#: 2 人になり、4 を渡すと 3 人になった） 1 つずつ増やしてやり直す
EXTRA_TRIES = 2


@dataclass(frozen=True, slots=True)
class SpeakerOptions:
    """1 回の分離の条件"""

    #: 話す人の数 ``None`` なら自動で決める
    speakers: int | None = None
    #: 分ける音声ストリームの番号（:attr:`AudioStreamInfo.index`） ``None`` なら 1 本目
    audio_stream: int | None = None

    def __post_init__(self) -> None:
        if self.speakers is not None and not 1 <= self.speakers <= MAX_SPEAKERS:
            raise ValueError(f"人数は 1 から {MAX_SPEAKERS} まで: {self.speakers}")


class SpeakerBackend(Protocol):
    """音から話者の区間を求めるもの"""

    def is_available(self) -> bool:
        """いま実行できるか 部品が入っていなければ偽"""
        ...

    def separate(
        self,
        path: Path,
        options: SpeakerOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> tuple[SpeakerTurn, ...] | None:
        """分ける 中断されたら ``None`` 失敗は :class:`AsrError`"""
        ...


def runtime_status() -> PackStatus:
    """入っているか import せずに見る（sherpa-onnx の import は DLL を探しに行く）"""
    return SPEAKER_PACK.status()


def install_command(*, upgrade: bool = False) -> list[str]:
    """導入に使う ``pip`` のコマンド列"""
    return _install_command(SPEAKER_PACK, extra=False, upgrade=upgrade)


#: 1 回のまとめ 人数（``-1`` は自動）を受けて、話者の区間を返す 中断されたら ``None``
type Clustering = Callable[[int], tuple[SpeakerTurn, ...] | None]


def separate_with_retries(run: Clustering, speakers: int | None) -> tuple[SpeakerTurn, ...] | None:
    """人数を渡されたら、その人数に分かれるまで人数を 1 つずつ増やしてまとめ直す

    増やしても届かなければ一番多く分かれた回を返す 渡した人数より多く分かれた回は使わない
    （頼んだより多い人数を出すと、1 人を 2 人に割ったと読める）
    """
    if speakers is None:
        return run(-1)
    best: tuple[SpeakerTurn, ...] | None = None
    for asked in range(speakers, speakers + EXTRA_TRIES + 1):
        turns = run(asked)
        if turns is None:
            return None
        found = len({turn.speaker for turn in turns})
        if found > speakers:
            break
        if best is None or found > len({turn.speaker for turn in best}):
            best = turns
        if found == speakers:
            break
    return best if best is not None else ()


class SherpaSpeakerBackend:
    """sherpa-onnx の話者分離を呼ぶ 読み込んだモデルは持っておく（2 回目から速い）"""

    def __init__(self) -> None:
        self._model_files: tuple[str, str] | None = None

    def is_available(self) -> bool:
        return runtime_status().ready

    def separate(
        self,
        path: Path,
        options: SpeakerOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> tuple[SpeakerTurn, ...] | None:
        source = Path(path)
        if not source.exists():
            raise AsrError(f"素材が見つからない: {source}")
        if progress is not None:
            progress(0.0, "モデルを用意している 初めては 28 MB ほど落とします")
        sherpa = _import_sherpa()
        segmentation, embedding = self._models()
        if should_cancel is not None and should_cancel():
            return None
        if progress is not None:
            progress(0.05, "音声を読んでいる")
        samples = _media_samples(source, should_cancel, options.audio_stream)
        if samples is None:
            return None

        def run(clusters: int) -> tuple[SpeakerTurn, ...] | None:
            return _cluster(
                sherpa, segmentation, embedding, samples, clusters, progress, should_cancel
            )

        return separate_with_retries(run, options.speakers)

    def _models(self) -> tuple[str, str]:
        if self._model_files is None:
            self._model_files = (
                _fetch(SEGMENTATION_REPOSITORY, SEGMENTATION_FILE),
                _fetch(EMBEDDING_REPOSITORY, EMBEDDING_FILE),
            )
        return self._model_files


def _import_sherpa() -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise AsrError(
            "話者の分離の部品が入っていません 字幕パネルの〔話者を分ける…〕から入れてください"
        ) from exc
    return sherpa_onnx


def _fetch(repository: str, filename: str) -> str:
    """モデルを落とす（落としてあれば置き場の物） 字幕起こしと同じ置き場"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AsrError("モデルを落とす部品（huggingface-hub）が入っていません") from exc
    from sashimono.asr.environment import model_cache_dir

    try:
        return str(hf_hub_download(repository, filename, cache_dir=str(model_cache_dir())))
    except Exception as exc:  # 落とせない理由（ネット・置き場）はどれも同じに知らせる
        raise AsrError(f"話者の分離のモデルを落とせない（{filename}）: {exc}") from exc


def _media_samples(
    source: Path, should_cancel: ShouldCancel | None, stream: int | None
) -> np.ndarray | None:
    """素材の音を 16kHz・モノラルで読む 字幕起こしと同じ読み方（素材の時刻の原点から）

    同じ読み方にしないと、音の頭が原点と違う素材で、話者の区間と字幕の時刻がずれる
    """
    from sashimono.asr.whisper import _media_audio

    audio = _media_audio(source, should_cancel, stream)
    if isinstance(audio, str):
        raise AsrError(f"長さの分からない素材は話者を分けられない: {source}")
    return audio


def _cluster(
    sherpa: Any,
    segmentation: str,
    embedding: str,
    samples: np.ndarray,
    clusters: int,
    progress: Progress | None,
    should_cancel: ShouldCancel | None,
) -> tuple[SpeakerTurn, ...] | None:
    """1 回分ける ``clusters`` は人数（``-1`` は自動）"""
    threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    try:
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation),
                num_threads=threads,
            ),
            embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=embedding, num_threads=threads),
            clustering=sherpa.FastClusteringConfig(
                num_clusters=clusters, threshold=CLUSTER_THRESHOLD
            ),
        )
        if not config.validate():
            raise AsrError("話者の分離の設定を部品が受け付けない（モデルが壊れているかもしれない）")
        diarization = sherpa.OfflineSpeakerDiarization(config)
    except AsrError:
        raise
    except Exception as exc:
        raise AsrError(f"話者の分離のモデルを読めない: {exc}") from exc

    stopped = False

    def report(done: int, total: int) -> int:
        nonlocal stopped
        if should_cancel is not None and should_cancel():
            stopped = True
            # 0 でない値を返すと部品が止まる
            return 1
        if progress is not None and total > 0:
            progress(0.1 + 0.9 * done / total, f"話者を分けている {done}/{total}")
        return 0

    try:
        result = diarization.process(samples, callback=report).sort_by_start_time()
    except Exception as exc:
        if stopped:
            return None
        raise AsrError(f"話者を分けられない: {exc}") from exc
    if stopped:
        return None
    return _turns(result)


def _turns(result: Sequence[Any]) -> tuple[SpeakerTurn, ...]:
    turns = []
    for item in result:
        start = to_source_time(float(item.start))
        end = to_source_time(float(item.end))
        if end > start:
            turns.append(SpeakerTurn(start=start, end=end, speaker=int(item.speaker)))
    return tuple(turns)
