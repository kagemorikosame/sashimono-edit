"""起こしのバックグラウンド実行

実際の音声認識は使わない 差し替え可能なバックエンドにしてあるので、ここは
進捗・完了・中断・失敗の 4 つの経路だけを見る
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from pathlib import Path

from sashimono.asr.backend import AsrError, Progress, ShouldCancel, TranscribeOptions
from sashimono.asr.service import JobEvent, JobKind, TranscriptionService
from sashimono.core.model import MediaId, Transcript, TranscriptSegment

RESULT = Transcript((TranscriptSegment(Fraction(0), Fraction(1), "できた"),))


class FakeBackend:
    """好きな振る舞いをさせられるバックエンド"""

    def __init__(
        self,
        *,
        result: Transcript | None = RESULT,
        error: Exception | None = None,
        block: threading.Event | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.block = block
        self.cancelled = False

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def transcribe(
        self,
        path: Path,
        options: TranscribeOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> Transcript | None:
        if progress is not None:
            progress(0.5, "途中")
        if self.block is not None:
            # 中断を確かめるため、合図があるまで待つ
            while not self.block.wait(0.01):
                if should_cancel is not None and should_cancel():
                    self.cancelled = True
                    return None
        if self.error is not None:
            raise self.error
        return self.result


def _finish(service: TranscriptionService) -> list[JobEvent]:
    job = service.start(MediaId("m"), Path("素材.mp4"), TranscribeOptions())
    assert job.wait(5.0)
    # ワーカーが終わってから拾う UI も同じように、自分の都合で取りに来る
    return list(job.poll())


class TestTranscriptionService:
    def test_progress_then_done(self) -> None:
        service = TranscriptionService(FakeBackend())
        events = _finish(service)
        kinds = [event.kind for event in events]
        assert kinds == [JobKind.PROGRESS, JobKind.DONE]
        assert events[-1].transcript is RESULT

    def test_failure_becomes_an_event_not_an_exception(self) -> None:
        service = TranscriptionService(FakeBackend(error=AsrError("モデルが無い")))
        events = _finish(service)
        assert events[-1].kind is JobKind.FAILED
        assert "モデルが無い" in events[-1].message

    def test_unexpected_failures_are_caught_too(self) -> None:
        # 予期しない例外でアプリごと落ちるのが一番まずい
        service = TranscriptionService(FakeBackend(error=RuntimeError("想定外")))
        events = _finish(service)
        assert events[-1].kind is JobKind.FAILED
        assert "想定外" in events[-1].message

    def test_cancelling_stops_the_worker(self) -> None:
        backend = FakeBackend(block=threading.Event())
        service = TranscriptionService(backend)
        job = service.start(MediaId("m"), Path("素材.mp4"), TranscribeOptions())

        deadline = time.monotonic() + 5.0
        # 走り出してから止める 順番待ちのまま止めた物は走らせない（別の試験）
        while job.waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        job.cancel()
        while job.running and time.monotonic() < deadline:
            time.sleep(0.01)

        assert backend.cancelled is True
        assert [event.kind for event in job.poll()][-1] is JobKind.CANCELLED

    def test_a_second_request_waits_its_turn(self) -> None:
        # 前は走っている間の依頼を断っていた 断ると AI の 2 本目の結果が取り込まれずに消える
        # 同時には走らせない（GPU の取り合い） 1 本目が終わってから 2 本目が走る
        gate = threading.Event()
        backend = FakeBackend(block=gate)
        service = TranscriptionService(backend)
        first = service.start(MediaId("m"), Path("素材.mp4"), TranscribeOptions())
        second = service.start(MediaId("n"), Path("別.mp4"), TranscribeOptions(audio_stream=2))
        deadline = time.monotonic() + 5.0
        while first.waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not first.waiting and second.waiting
        assert [job.media_id for job in service.jobs()] == ["m", "n"]
        assert service.find(MediaId("n"), 2) is second
        gate.set()
        assert first.wait(5.0) and second.wait(5.0)
        assert first.poll()[-1].kind is JobKind.DONE
        assert second.poll()[-1].kind is JobKind.DONE
        assert service.jobs() == []

    def test_a_cancelled_waiting_job_never_runs(self) -> None:
        gate = threading.Event()
        service = TranscriptionService(FakeBackend(block=gate))
        first = service.start(MediaId("m"), Path("素材.mp4"), TranscribeOptions())
        second = service.start(MediaId("n"), Path("別.mp4"), TranscribeOptions())
        second.cancel()
        gate.set()
        assert first.wait(5.0) and second.wait(5.0)
        assert second.poll()[-1].kind is JobKind.CANCELLED

    def test_polling_an_idle_job_returns_nothing(self) -> None:
        service = TranscriptionService(FakeBackend())
        job = service.start(MediaId("m"), Path("素材.mp4"), TranscribeOptions())
        assert job.wait(5.0)
        job.poll()
        assert job.poll() == []
