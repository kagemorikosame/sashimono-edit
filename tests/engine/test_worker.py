"""重い裏の処理を別のプロセスで走らせる仕組み（#440）

子を本当に起こして確かめる 子の起動は 0.2 秒ほどなので、数は絞る
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine import worker as worker_module
from sashimono.engine.cache import proxy as proxy_module
from sashimono.engine.cache.proxy import ProxyBuilder, ProxyStore
from sashimono.engine.cache.store import CacheStore
from sashimono.engine.decode import probe_media
from sashimono.engine.worker import (
    WorkerFailedError,
    WorkerUnavailableError,
    probe_job,
    run_in_process,
)
from tests.media_fixtures import SampleMedia


def _wait(predicate: object, timeout: float = 60.0) -> bool:
    assert callable(predicate)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestRunInProcess:
    def test_the_answer_and_progress_come_back(self) -> None:
        """壊れると、子が作り終えても親は結果も進み具合も受け取れず、控えが使われない"""
        seen: list[float] = []
        assert run_in_process(probe_job, {"value": "子"}, progress=seen.append) == "子"
        assert seen and seen[-1] == 1.0

    def test_a_failure_in_the_child_reaches_the_parent(self) -> None:
        """壊れると、子の中の失敗が黙って消え、控えが無いまま「作れなかった」も出ない"""
        with pytest.raises(WorkerFailedError, match="頼まれた失敗"):
            run_in_process(probe_job, {"fail": True})

    def test_cancelling_stops_the_child(self) -> None:
        """壊れると、素材を外しても窓を閉じても子が控えを作り続ける"""
        began = time.monotonic()
        answer = run_in_process(
            probe_job, {"wait_s": 30.0}, should_cancel=lambda: time.monotonic() - began > 0.5
        )
        assert answer is None
        assert time.monotonic() - began < 10.0
        assert not list(worker_module._LIVE), "止めた子が残っている"

    def test_a_child_that_cannot_start_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """壊れると、子を起こせない機械で控えの係が落ち、スレッドへ戻れない"""

        class Refusing:
            def Process(self, *_args: object, **_kwargs: object) -> object:  # noqa: N802
                class Unstartable:
                    def start(self) -> None:
                        raise OSError("起こせない")

                return Unstartable()

            def Pipe(self, duplex: bool = True) -> tuple[object, object]:  # noqa: N802
                class End:
                    def close(self) -> None:
                        pass

                return End(), End()

        monkeypatch.setattr(worker_module, "_context", Refusing)
        with pytest.raises(WorkerUnavailableError):
            run_in_process(probe_job, {})


def _tall(path: Path) -> MediaItem:
    media = probe_media(path)
    stream = media.video_streams[0]
    return replace(media, video_streams=(replace(stream, width=3840, height=2160),))


class TestProxyInAnotherProcess:
    def test_the_proxy_is_made_by_a_child(self, tmp_path: Path, sample_av: SampleMedia) -> None:
        """壊れると、設定で別のプロセスを選んでも控えができない"""
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path), height=120), separate_process=True)
        media = _tall(sample_av.path)
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            assert _wait(lambda: ready == [media.id])
            found = builder.store.find(media)
            assert found is not None and found.stat().st_size > 0
            assert builder.separate_process
            assert not list(found.parent.glob("*.part")), "書きかけが残っている"
        finally:
            builder.close()

    def test_it_falls_back_to_a_thread(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、子を起こせない機械（見張りのソフトに止められた等）で控えが一切作れない"""

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise WorkerUnavailableError("起こせない")

        monkeypatch.setattr(proxy_module, "run_in_process", refuse)
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path), height=120), separate_process=True)
        media = _tall(sample_av.path)
        done = threading.Event()
        try:
            builder.request(media, on_ready=lambda _id: done.set())
            assert done.wait(60.0)
            assert builder.store.find(media) is not None
            assert not builder.separate_process, "起こせなかったことを覚えていない"
        finally:
            builder.close()

    def test_a_crash_is_a_failure_not_a_retry(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、素材のせいで子が落ちたとき、同じ素材をアプリの中で読み直してアプリごと落ちる"""
        called: list[object] = []

        def crash(*_args: object, **_kwargs: object) -> None:
            raise worker_module.WorkerCrashedError("途中で終わった")

        monkeypatch.setattr(proxy_module, "run_in_process", crash)
        monkeypatch.setattr(proxy_module, "create_proxy", lambda *a, **k: called.append(a))
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path), height=120), separate_process=True)
        media = _tall(sample_av.path)
        try:
            builder.request(media)
            assert _wait(lambda: builder.poll().failed == 1)
            assert called == []
            assert "途中で終わった" in builder.poll().failures[media.id]
        finally:
            builder.close()
