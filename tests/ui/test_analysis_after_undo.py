"""名前は出るのに縮小画像と波形が出ない

素材一覧から外すと、その素材の波形・サムネイル・控えを捨てる 取り消して素材が戻っても
誰も頼み直さず、クリップの名前は出るのに中身がいつまでも空だった また、取り消し印の
付いた解析が走っている間に同じ物を頼むと、依頼は捨てられ、走っていた解析も結果を
載せずに終わるので、やはり空のまま残った
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddMedia
from sashimono.core.model import MediaId, MediaItem, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio import PeakLevel, Waveform
from sashimono.engine.cache import MediaAnalyzer
from sashimono.engine.cache import proxy as proxy_module
from sashimono.engine.cache.proxy import ProxyBuilder, ProxyStore
from sashimono.engine.cache.store import CacheStore
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import Preferences

#: 裏の仕事を待つ上限（秒） 止まったら試験を落とす（いつまでも待たない）
WAIT = 10.0


def _wait_for(done: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(0.01)
    return done()


def _waveform() -> Waveform:
    peaks = np.full((100, 2, 2), 0.5, dtype=np.float32)
    return Waveform(
        sample_rate=48000, channels=2, total_samples=25600, levels=(PeakLevel(256, peaks),)
    )


class TestAnalyzerRequestWhileCancelling:
    def test_a_request_while_a_cancelled_job_runs_is_done_after_it(
        self, tmp_path: Path, audio_media: MediaItem
    ) -> None:
        # 捨てると、外してすぐ取り消した素材の波形がいつまでも出ない
        analyzer = MediaAnalyzer(CacheStore(tmp_path / "cache"))
        gate = threading.Event()
        started = threading.Event()
        runs: list[int] = []

        def work(media: MediaItem, job: tuple[str, MediaId, int | None], report: object) -> bool:
            del report
            runs.append(1)
            if len(runs) == 1:
                started.set()
                gate.wait(WAIT)
            return analyzer._publish("waveform", media.id, _waveform(), job[2])

        analyzer._analyze_waveform = work  # type: ignore[method-assign]  # 素材を開かずに試す
        try:
            analyzer.request(audio_media)
            assert started.wait(WAIT)
            analyzer.forget(audio_media.id)
            analyzer.request(audio_media)
            gate.set()
            assert _wait_for(lambda: analyzer.waveform(audio_media) is not None)
            assert len(runs) == 2
        finally:
            gate.set()
            analyzer.close()

    def test_forgetting_again_drops_the_promised_rerun(
        self, tmp_path: Path, audio_media: MediaItem
    ) -> None:
        # 走らせ直す約束が残ると、外した素材を、外した後にまた解析する
        analyzer = MediaAnalyzer(CacheStore(tmp_path / "cache"))
        gate = threading.Event()
        started = threading.Event()
        finished = threading.Event()
        runs: list[int] = []

        def work(media: MediaItem, job: tuple[str, MediaId, int | None], report: object) -> bool:
            del report
            runs.append(1)
            started.set()
            gate.wait(WAIT)
            published = analyzer._publish("waveform", media.id, _waveform(), job[2])
            finished.set()
            return published

        analyzer._analyze_waveform = work  # type: ignore[method-assign]  # 素材を開かずに試す
        try:
            analyzer.request(audio_media)
            assert started.wait(WAIT)
            analyzer.forget(audio_media.id)
            analyzer.request(audio_media)
            analyzer.forget(audio_media.id)
            gate.set()
            assert finished.wait(WAIT)
            assert _wait_for(lambda: not analyzer._running)
            assert len(runs) == 1
            assert analyzer.waveform(audio_media) is None
        finally:
            gate.set()
            analyzer.close()


class TestProxyRequestWhileCancelling:
    def test_a_request_while_a_cancelled_proxy_runs_is_done_after_it(
        self, tmp_path: Path, video_media: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 解析と同じ 捨てると、外してすぐ取り消した素材の控えがいつまでも作られない
        gate = threading.Event()
        started = threading.Event()
        calls: list[Path] = []

        def fake(source: Path, target: Path, **_kwargs: object) -> Path | None:
            del source
            calls.append(target)
            if len(calls) == 1:
                started.set()
                gate.wait(WAIT)
                return None
            target.write_bytes(b"proxy")
            return target

        monkeypatch.setattr(proxy_module, "create_proxy", fake)
        # 控えを作るのは 1080 を超える素材だけ（is_worth_proxying）
        stream = replace(video_media.video_streams[0], width=3840, height=2160)
        video_media = replace(video_media, video_streams=(stream,))
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path / "cache")))
        try:
            builder.request(video_media)
            assert started.wait(WAIT)
            builder.forget(video_media.id)
            builder.request(video_media)
            gate.set()
            assert _wait_for(lambda: builder.store.find(video_media) is not None)
            assert len(calls) == 2
        finally:
            gate.set()
            builder.close()


@pytest.fixture
def window(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[MainWindow, list[MediaId], list[MediaId]]]:
    del qt_application
    created = MainWindow(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30))),
        confirm_unsaved=False,
    )
    # 素材は本物のファイルではない 頼まれた物だけを書き留め、裏では何も走らせない
    analyzed: list[MediaId] = []
    proxied: list[MediaId] = []
    monkeypatch.setattr(created, "_request_proxy", lambda media: proxied.append(media.id))
    monkeypatch.setattr(
        created._analyzer, "request", lambda media, **_kwargs: analyzed.append(media.id)
    )
    yield created, analyzed, proxied
    created.close()


class TestWindowAsksAgain:
    def test_undoing_a_removal_asks_for_the_analysis_again(
        self,
        window: tuple[MainWindow, list[MediaId], list[MediaId]],
        video_media: MediaItem,
    ) -> None:
        # 頼み直さないと、戻った素材のクリップに名前だけが出て、サムネイルと波形が出ない
        main, analyzed, proxied = window
        assert main.execute(AddMedia(video_media))
        main._remove_media(str(video_media.id))
        assert main.document.project.find_media(video_media.id) is None
        analyzed.clear()
        proxied.clear()
        main.undo()
        assert main.document.project.find_media(video_media.id) is not None
        assert analyzed == [video_media.id]
        assert proxied == [video_media.id]

    def test_other_edits_do_not_ask_again(
        self,
        window: tuple[MainWindow, list[MediaId], list[MediaId]],
        video_media: MediaItem,
        audio_media: MediaItem,
    ) -> None:
        # 編集のたびに全部の素材を頼むと、開けずに失敗した素材の解析を毎回走らせ直す
        main, analyzed, _ = window
        assert main.execute(AddMedia(video_media))
        analyzed.clear()
        assert main.execute(AddMedia(audio_media))
        main.undo()
        main.redo()
        assert video_media.id not in analyzed

    def test_asking_again_happens_once(
        self,
        window: tuple[MainWindow, list[MediaId], list[MediaId]],
        video_media: MediaItem,
    ) -> None:
        # 戻ったあとも覚えたままだと、やり直しと取り消しを繰り返すたびに頼み直す
        main, analyzed, _ = window
        assert main.execute(AddMedia(video_media))
        main._remove_media(str(video_media.id))
        main.undo()
        main.redo()
        main.undo()
        assert analyzed.count(video_media.id) == 1


class TestWindowAppliesTheWidth:
    def test_the_setting_reaches_the_timeline(
        self, window: tuple[MainWindow, list[MediaId], list[MediaId]]
    ) -> None:
        # 設定の窓で変えても、タイムラインに届かなければ見た目は変わらない
        main, _, _ = window
        main._apply_preferences(Preferences(detail_min_width=12))
        assert main._timeline.detail_min_width == 12
