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

    def __init__(self, media_id: MediaId, path: Path) -> None:
        self.media_id = media_id
        self.path = path
        self._events: queue.Queue[JobEvent] = queue.Queue()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return not self._done.is_set()

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
    2 件並べても速くならず、どちらも落ちる可能性が上がるだけ
    """

    def __init__(self, backend: TranscriptionBackend) -> None:
        self._backend = backend
        self._current: Job | None = None
        self._lock = threading.Lock()

    @property
    def backend(self) -> TranscriptionBackend:
        return self._backend

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.running

    def start(self, media_id: MediaId, path: Path, options: TranscribeOptions) -> Job:
        """起こしを始める すでに走っていれば :class:`RuntimeError`"""
        with self._lock:
            if self._current is not None and self._current.running:
                raise RuntimeError("すでに起こしが走っている")
            job = Job(media_id, Path(path))
            self._current = job

        thread = threading.Thread(
            target=self._run, args=(job, options), name="sashimono-asr", daemon=True
        )
        job._thread = thread
        thread.start()
        return job

    def cancel(self) -> None:
        with self._lock:
            current = self._current
        if current is not None:
            current.cancel()

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
