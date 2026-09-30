"""起こしをバックグラウンドで走らせる

数分かかる処理なので、UI スレッドで回すわけにはいかない かといってワーカー
スレッドから直接ウィジェットを触るのも危ない（Qt が落ちる）

そこで、ワーカーは出来事をキューへ積むだけにして、UI 側は自分の都合の良い間隔で
:meth:`Job.poll` して取り出す :class:`~sashimono.engine.cache.MediaAnalyzer` が
解析結果を一定間隔で反映しているのと同じ考え方で、Qt に依存しないので試験も書ける
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sashimono.asr.backend import (
    AsrError,
    TranscribeOptions,
    TranscriptionBackend,
)
from sashimono.core.model import MediaId, Transcript

__all__ = ["Job", "JobEvent", "JobKind", "TranscriptionService"]


class JobKind(Enum):
    PROGRESS = "progress"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobEvent:
    """ワーカーからの 1 件の知らせ"""

    kind: JobKind
    ratio: float = 0.0
    message: str = ""
    transcript: Transcript | None = None
    #: 起こせたが知らせておくこと（GPU の道具が読めず CPU で起こした など）
    notice: str = ""

    @property
    def finished(self) -> bool:
        return self.kind is not JobKind.PROGRESS


class Job:
    """走っている（または走り終わった）起こし 1 件"""

    def __init__(
        self, media_id: MediaId, path: Path, options: TranscribeOptions | None = None
    ) -> None:
        self.media_id = media_id
        self.path = path
        #: 起こす条件 音声ストリーム（``options.audio_stream``）で結果の取り込み先が決まる
        self.options = options if options is not None else TranscribeOptions()
        self._events: queue.Queue[JobEvent] = queue.Queue()
        self._cancel = threading.Event()
        self._started = threading.Event()
        self._done = threading.Event()

    @property
    def stream(self) -> int | None:
        """起こす音声ストリームの番号 ``None`` なら 1 本目"""
        return self.options.audio_stream

    @property
    def running(self) -> bool:
        """まだ終わっていない（順番待ちも含む）"""
        return not self._done.is_set()

    @property
    def waiting(self) -> bool:
        """順番待ち 前の起こしが終わるのを待っている"""
        return not self._started.is_set() and not self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        """中断を頼む 実際に止まるのは次の区切りまで"""
        self._cancel.set()

    def poll(self) -> list[JobEvent]:
        """溜まった出来事を取り出す ブロックしない"""
        drained: list[JobEvent] = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                return drained

    def wait(self, timeout: float | None = None) -> bool:
        """終わるまで待つ 試験と終了処理のためにある"""
        return self._done.wait(timeout)

    def _emit(self, event: JobEvent) -> None:
        self._events.put(event)


class TranscriptionService:
    """起こしの実行を受け付ける

    同時に走らせるのは 1 件だけ 音声認識は GPU とメモリを丸ごと使うので、
    2 件並べても速くならず、どちらも落ちる可能性が上がるだけ 走っている間に来た依頼は
    順番待ちに入れ、前が終わったら 1 本ずつ走らせる（窓からの依頼も AI の道具からの依頼も
    同じ列） 前は走っている間の依頼を断っていて、AI が 2 本目を頼むと 1 本目の結果が
    取り込まれずに消えることがあった
    """

    def __init__(self, backend: TranscriptionBackend) -> None:
        self._backend = backend
        self._queue: list[Job] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    @property
    def backend(self) -> TranscriptionBackend:
        return self._backend

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(job.running for job in self._queue)

    def jobs(self) -> list[Job]:
        """走っている物と順番待ちの物 頼まれた順"""
        with self._lock:
            return [job for job in self._queue if job.running]

    def find(self, media_id: MediaId, stream: int | None) -> Job | None:
        """その素材と音声を起こしている（待っている）依頼"""
        return next((j for j in self.jobs() if j.media_id == media_id and j.stream == stream), None)

    def start(self, media_id: MediaId, path: Path, options: TranscribeOptions) -> Job:
        """起こしを頼む 走っている物があれば順番待ちに入れる"""
        job = Job(media_id, Path(path), options)
        with self._lock:
            self._queue = [j for j in self._queue if j.running]
            self._queue.append(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drain, name="sashimono-asr", daemon=True
                )
                self._worker.start()
        return job

    def cancel(self) -> None:
        """走っている物と待っている物をすべて止める"""
        for job in self.jobs():
            job.cancel()

    def _drain(self) -> None:
        """列の頭から 1 本ずつ走らせる 列が空になったら終わる（次の依頼で作り直す）"""
        while True:
            with self._lock:
                job = next((j for j in self._queue if not j._started.is_set()), None)
                if job is None:
                    self._worker = None
                    return
                job._started.set()
            if job.cancelled:
                job._emit(JobEvent(JobKind.CANCELLED, message="中断した"))
                job._done.set()
                continue
            self._run(job, job.options)

    def _run(self, job: Job, options: TranscribeOptions) -> None:
        try:
            transcript = self._backend.transcribe(
                job.path,
                options,
                progress=lambda ratio, message: job._emit(
                    JobEvent(JobKind.PROGRESS, ratio=ratio, message=message)
                ),
                should_cancel=job._cancel.is_set,
            )
        except AsrError as exc:
            job._emit(JobEvent(JobKind.FAILED, message=str(exc)))
        except Exception as exc:  # 予期しない失敗でアプリごと落とさない
            job._emit(JobEvent(JobKind.FAILED, message=f"想定外の失敗: {exc}"))
        else:
            # 知らせを受け取れる実装だけが持つ（試験の代わりの実装などは持たない）
            take = getattr(self._backend, "take_notice", None)
            notice = str(take()) if callable(take) else ""
            if transcript is None:
                job._emit(JobEvent(JobKind.CANCELLED, message="中断した"))
            else:
                done = f"{len(transcript)} 文を起こした"
                job._emit(
                    JobEvent(
                        JobKind.DONE,
                        ratio=1.0,
                        message=f"{done} {notice}" if notice else done,
                        transcript=transcript,
                        notice=notice,
                    )
                )
        finally:
            job._done.set()
