"""重い裏の処理を別のプロセスで走らせる仕組み（#440）

子を本当に起こして確かめる 子の起動は 0.2 秒ほどなので、数は絞る
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine import worker as worker_module
from sashimono.engine.audio.waveform import analyze_waveform
from sashimono.engine.cache import analyzer as analyzer_module
from sashimono.engine.cache import proxy as proxy_module
from sashimono.engine.cache.analyzer import MediaAnalyzer
from sashimono.engine.cache.proxy import ProxyBuilder, ProxyStore
from sashimono.engine.cache.store import CacheStore
from sashimono.engine.decode import probe_media
from sashimono.engine.scenes import SceneSignal, scenes_job
from sashimono.engine.worker import (
    WorkerCrashedError,
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

    def test_partial_values_reach_the_parent_in_order(self) -> None:
        """壊れると、場面を別のプロセスで読む間、AI が途中までの切り替わりを受け取れない"""
        seen: list[object] = []
        answer = run_in_process(probe_job, {"value": "子", "steps": 3}, partial=seen.append)
        assert answer == "子"
        assert seen == ["子 1", "子 2", "子 3"]

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

    def test_running_out_of_pipes_is_also_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、ハンドルが尽きて管を作れない機械で、スレッドへ戻せば作れる控えまで失敗になる"""
        closed: list[object] = []

        class Starved:
            made = 0

            def Pipe(self, duplex: bool = True) -> tuple[object, object]:  # noqa: N802
                Starved.made += 1
                if Starved.made > 1:
                    raise OSError("ハンドルが尽きた")

                class End:
                    def close(self) -> None:
                        closed.append(self)

                return End(), End()

        monkeypatch.setattr(worker_module, "_context", Starved)
        with pytest.raises(WorkerUnavailableError):
            run_in_process(probe_job, {})
        assert len(closed) == 2, "作れた方の管を閉じていない"

    def test_a_child_that_never_wakes_up_is_given_up(self) -> None:
        """壊れると、起動の途中で固まった子を待ち続け、控えの係と自己診断が止まったままになる"""
        began = time.monotonic()
        with pytest.raises(WorkerUnavailableError, match="起きなかった"):
            # 子の起動は 0.2 秒ほど掛かるので、0 秒ではまず「起きた」が届かない
            run_in_process(probe_job, {}, start_timeout_s=0.0)
        assert time.monotonic() - began < 10.0
        assert not list(worker_module._LIVE), "起きなかった子が残っている"

    def test_the_start_limit_does_not_cut_a_long_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """壊れると、起きた後も起動の期限を当て、期限より長い変換（長い素材の控え）を途中で
        打ち切ってスレッドで作り直す
        """
        # 時計は子の知らせ（起きた後にしか来ない）で期限の何倍も先へ飛ばす 本当に待つと、
        # 試験が期限の分だけ遅くなる
        offset = [0.0]

        def clock() -> float:
            return time.monotonic() + offset[0]

        def jump(_fraction: float) -> None:
            offset[0] += 1000.0

        monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=clock))
        result = run_in_process(
            probe_job, {"value": "長い", "steps": 3}, progress=jump, start_timeout_s=30.0
        )
        assert result == "長い"
        assert offset[0] >= 1000.0, "期限を越える前に終わっていて、確かめになっていない"

    def test_a_crash_after_starting_is_a_crash(self) -> None:
        """壊れると、素材のせいで落ちた子を「起こせなかった」と取り違え、同じ素材をアプリの中で
        読み直してアプリごと落ちる
        """
        with pytest.raises(WorkerCrashedError):
            run_in_process(probe_job, {"crash": True})


def _tall(path: Path) -> MediaItem:
    media = probe_media(path)
    stream = media.video_streams[0]
    return replace(media, video_streams=(replace(stream, width=3840, height=2160),))


class TestProxyInAnotherProcess:
    def test_the_proxy_is_made_by_a_child(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、設定で別のプロセスを選んでも控えができない（スレッドで作っていても気付けない）"""
        jobs: list[object] = []

        def recording(job: object, *args: object, **kwargs: object) -> object:
            jobs.append(job)
            return run_in_process(job, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(proxy_module, "run_in_process", recording)
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path), height=120), separate_process=True)
        media = _tall(sample_av.path)
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            assert _wait(lambda: ready == [media.id])
            found = builder.store.find(media)
            assert found is not None and found.stat().st_size > 0
            assert builder.separate_process
            assert jobs == [proxy_module.proxy_job], "子を通さずに作っている"
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


class TestAnalysisInAnotherProcess:
    """素材の解析（波形・サムネイル・場面）を別のプロセスで読む（#453）"""

    def test_waveform_filmstrip_and_scenes_are_read_by_children(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、設定で別のプロセスを選んでも解析がアプリの中で走り、画面と取り合ったまま"""
        jobs: list[object] = []

        def recording(job: object, *args: object, **kwargs: object) -> object:
            jobs.append(job)
            return run_in_process(job, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(analyzer_module, "run_in_process", recording)
        # 試験の素材は 2 秒しかない 長さの下限を外して、寄せる種類が子を通るかを見る
        monkeypatch.setattr(analyzer_module, "PROCESS_MIN_SECONDS", {"scenes": 0, "waveform": 0})
        analyzer = MediaAnalyzer(CacheStore(tmp_path), separate_process=True)
        media = probe_media(sample_av.path)
        ready = threading.Event()
        try:
            analyzer.request(media, scenes=True, on_ready=lambda _id: ready.set())
            assert _wait(
                lambda: (
                    analyzer.waveform(media) is not None
                    and analyzer.filmstrip(media) is not None
                    and analyzer.scenes(media) is not None
                )
            )
            # サムネイルは取り合っても落ちないので寄せない（スレッドで読む）
            assert set(jobs) == {analyze_waveform, scenes_job}
            signal = analyzer.scenes(media)
            assert signal is not None and signal.complete
        finally:
            analyzer.close()

    def test_partial_scenes_from_the_child_are_shown_while_reading(self, tmp_path: Path) -> None:
        """壊れると、長い素材を別のプロセスで読む間、AI が途中までの切り替わりを受け取れない"""
        analyzer = MediaAnalyzer(CacheStore(tmp_path), separate_process=True)
        media_id = MediaId("m")
        signal = SceneSignal.empty(stream=0)
        with analyzer._lock:
            analyzer._running.add(("scenes", media_id, None))
            analyzer._partials[media_id] = signal
        try:
            media = MediaItem(path=tmp_path / "x.mp4", duration=Fraction(1), id=media_id)
            state = analyzer.scene_state(media)
            assert state.running and state.signal is signal
            # 止めた後も、届いていた所までを残す（同じ所で止めたスレッドの読みと同じ扱い）
            analyzer.cancel_scenes()
            assert analyzer._halted[media_id] is signal
        finally:
            analyzer.close()

    def test_it_falls_back_to_a_thread(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、子を起こせない機械で波形もサムネイルも出なくなる"""

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise WorkerUnavailableError("起こせない")

        monkeypatch.setattr(analyzer_module, "run_in_process", refuse)
        monkeypatch.setattr(analyzer_module, "PROCESS_MIN_SECONDS", {"waveform": 0})
        analyzer = MediaAnalyzer(CacheStore(tmp_path), separate_process=True)
        media = probe_media(sample_av.path)
        try:
            analyzer.request(media)
            assert _wait(
                lambda: (
                    analyzer.waveform(media) is not None and analyzer.filmstrip(media) is not None
                )
            )
            assert not analyzer.separate_process, "起こせなかったことを覚えていない"
        finally:
            analyzer.close()

    def test_short_media_stay_in_a_thread(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、短い素材を読むたびに子を起こして部品を読み直し、1 本に 1 秒ずつ余計に待つ"""
        jobs: list[object] = []

        def recording(job: object, *args: object, **kwargs: object) -> object:
            jobs.append(job)
            return run_in_process(job, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(analyzer_module, "run_in_process", recording)
        analyzer = MediaAnalyzer(CacheStore(tmp_path), separate_process=True)
        media = probe_media(sample_av.path)
        try:
            analyzer.request(media, scenes=True)
            assert _wait(
                lambda: analyzer.waveform(media) is not None and analyzer.scenes(media) is not None
            )
            assert jobs == []
        finally:
            analyzer.close()

    def test_a_crash_is_a_failure_not_a_retry(
        self, tmp_path: Path, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、素材のせいで子が落ちたとき、同じ素材をアプリの中で読み直してアプリごと落ちる"""
        here: list[object] = []

        def crash(*_args: object, **_kwargs: object) -> None:
            raise worker_module.WorkerCrashedError("途中で終わった")

        monkeypatch.setattr(analyzer_module, "run_in_process", crash)
        monkeypatch.setattr(analyzer_module, "PROCESS_MIN_SECONDS", {"waveform": 0})
        monkeypatch.setattr(analyzer_module, "analyze_waveform", lambda *a, **k: here.append(a))
        analyzer = MediaAnalyzer(CacheStore(tmp_path), separate_process=True)
        media = probe_media(sample_av.path)
        try:
            analyzer.request(media)
            assert _wait(lambda: analyzer.poll().failed >= 1)
            assert here == []
            assert analyzer.separate_process, "落ちたのを起こせないと取り違えた"
        finally:
            analyzer.close()
