"""場面の一覧を裏で作って覚える（#321）

波形・サムネイルと同じ置き場（:class:`MediaAnalyzer` と ``engine/cache``）に覚え、
2 度目は読み直さないこと・素材を差し替えたら読み直すこと・裏で止められること・
AI の道具が待てることを見る
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.model import MediaItem
from sashimono.engine.cache import (
    CacheStore,
    MediaAnalyzer,
    load_scenes,
    save_scenes,
    scene_cache,
    scenes_key,
)
from sashimono.engine.cache import analyzer as analyzer_module
from sashimono.engine.decode import forget_probe, probe_media
from sashimono.engine.scenes import SceneScan, SceneSignal
from tests.scene_video import CUTS, write_broken_video, write_scene_video


@pytest.fixture(scope="module")
def made_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_scene_video(tmp_path_factory.mktemp("scene-analysis") / "scenes.mp4")


@pytest.fixture
def video(tmp_path: Path, made_video: Path) -> Path:
    """試験ごとに写した動画 書き換える試験が、ほかの試験の素材を変えないように"""
    copied = tmp_path / "video.mp4"
    copied.write_bytes(made_video.read_bytes())
    return copied


@pytest.fixture
def store(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache")


@pytest.fixture
def analyzer(store: CacheStore) -> Iterator[MediaAnalyzer]:
    created = MediaAnalyzer(store)
    yield created
    created.close()


def _media(path: Path) -> MediaItem:
    forget_probe(path)
    return probe_media(path)


def _until(predicate: Callable[[], bool], timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _signal(count: int = 3) -> SceneSignal:
    return SceneSignal(
        times=np.arange(1, count + 1, dtype=np.float64) / 30,
        pixel=np.full(count, 0.1, np.float32),
        histogram=np.full(count, 0.2, np.float32),
        analyzed=count / 30,
        complete=True,
    )


class TestTheStore:
    def test_it_comes_back(self, store: CacheStore, video: Path) -> None:
        # 読み戻せないと、開くたびに長い動画を全フレーム読み直す
        key = scenes_key(video)
        save_scenes(store, key, _signal())
        loaded = load_scenes(store, key)
        assert loaded is not None and loaded.complete
        assert np.array_equal(loaded.times, _signal().times)

    def test_the_stream_comes_back(self, store: CacheStore, video: Path) -> None:
        # 覚えた物から番号が消えると、開き直した後だけ別の映像のクリップに切り替わりを当てる
        key = scenes_key(video)
        save_scenes(store, key, replace(_signal(), stream=3))
        loaded = load_scenes(store, key)
        assert loaded is not None and loaded.stream == 3
        save_scenes(store, key, _signal())
        again = load_scenes(store, key)
        assert again is not None and again.stream is None

    def test_a_partial_signal_is_not_written(self, store: CacheStore, video: Path) -> None:
        # 途中までを書くと、次に開いたときに途中までの一覧を読み終えた物として返す
        partial = SceneScan().snapshot()
        assert save_scenes(store, scenes_key(video), partial) is None
        assert load_scenes(store, scenes_key(video)) is None

    def test_another_version_is_read_again(
        self, store: CacheStore, video: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 測り方を変えた後に前の並びを使うと、感度を掛けたときに拾う所がずれる
        key = scenes_key(video)
        save_scenes(store, key, _signal())
        monkeypatch.setattr(scene_cache, "FORMAT_VERSION", scene_cache.FORMAT_VERSION + 1)
        assert load_scenes(store, key) is None

    def test_changing_the_file_changes_the_key(self, video: Path) -> None:
        # 鍵が変わらないと、差し替えた素材に前の素材の切り替わりを返す
        before = scenes_key(video)
        with video.open("ab") as handle:
            handle.write(b"\0" * 16)
        assert scenes_key(video) != before

    def test_a_broken_file_is_read_again(self, store: CacheStore, video: Path) -> None:
        # 壊れた控えで落ちると、AI の道具がその素材で二度と使えない
        key = scenes_key(video)
        path = store.prepare(scene_cache.NAMESPACE, key, scene_cache.SUFFIX)
        path.write_bytes(b"not a npz")
        assert load_scenes(store, key) is None


class TestTheAnalyzer:
    def test_it_is_not_made_unless_asked(self, analyzer: MediaAnalyzer, video: Path) -> None:
        # 頼まれないのに読むと、設定で切った人の機械でも CPU を使う
        media = _media(video)
        analyzer.request(media)
        assert _until(lambda: analyzer.filmstrip(media) is not None)
        assert analyzer.scene_state(media).signal is None
        assert not analyzer.scene_state(media).running

    def test_it_is_made_in_the_background(self, analyzer: MediaAnalyzer, video: Path) -> None:
        # 読み込みと一緒に頼めないと、AI が頼んだときに全フレーム読むのを待つ
        media = _media(video)
        analyzer.request(media, scenes=True)
        assert _until(lambda: analyzer.scenes(media) is not None), "場面の一覧ができない"
        found = analyzer.scenes(media)
        assert found is not None
        assert len(found.cuts()) == len(CUTS)

    def test_a_second_analyzer_reads_it_from_the_disk(
        self, store: CacheStore, video: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 覚えた物を読まないと、プロジェクトを開くたびに長い動画を全フレーム読み直す
        media = _media(video)
        first = MediaAnalyzer(store)
        try:
            first.request_scenes(media)
            assert _until(lambda: first.scenes(media) is not None)
        finally:
            first.close()

        def refuse(*_arguments: object, **_options: object) -> SceneSignal:
            raise AssertionError("覚えた物があるのに読み直した")

        monkeypatch.setattr(analyzer_module, "analyze_scenes", refuse)
        second = MediaAnalyzer(store)
        try:
            second.request_scenes(media)
            assert _until(lambda: second.scenes(media) is not None)
            assert second.scene_state(media).failure is None
        finally:
            second.close()

    def test_a_changed_file_is_read_again(self, analyzer: MediaAnalyzer, video: Path) -> None:
        # 差し替えた素材に前の切り替わりを返すと、AI が別の動画の位置で切る
        media = _media(video)
        analyzer.request_scenes(media)
        assert _until(lambda: analyzer.scenes(media) is not None)
        stat = video.stat()
        os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        assert analyzer.scenes(media) is None, "変わった素材の一覧を返した"
        analyzer.request_scenes(media)
        assert _until(lambda: analyzer.scenes(media) is not None), "読み直さない"

    def test_forgetting_drops_it(self, analyzer: MediaAnalyzer, video: Path) -> None:
        # 外した素材の一覧が残ると、同じ ID の素材を入れ直したときに古い一覧を返す
        media = _media(video)
        analyzer.request_scenes(media)
        assert _until(lambda: analyzer.scenes(media) is not None)
        analyzer.forget(media.id)
        assert analyzer.scenes(media) is None

    def test_a_sound_only_media_is_not_read(self, analyzer: MediaAnalyzer, video: Path) -> None:
        # 映像の無い素材を読みに行くと、毎回失敗として残る
        media = _media(video)
        silent = MediaItem(path=media.path, duration=media.duration)
        analyzer.request_scenes(silent)
        assert not analyzer.scene_state(silent).running


class _Held:
    """場面の解析を、試験が放すまで途中で止めておく"""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def analyze(
        self,
        path: Path,
        *,
        threads: int = 2,
        scan: SceneScan | None = None,
        progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> SceneSignal | None:
        del path, threads, progress
        if scan is not None:
            scan.add(1.0, 0.5, 0.9)
        self.started.set()
        while not self.release.wait(0.01):
            if should_cancel is not None and should_cancel():
                self.cancelled.set()
                return None
        return _signal()


class TestWhileRunning:
    @pytest.fixture
    def held(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Held]:
        holder = _Held()
        monkeypatch.setattr(analyzer_module, "analyze_scenes", holder.analyze)
        yield holder
        holder.release.set()

    def test_thumbnails_do_not_wait_for_scenes(
        self, store: CacheStore, video: Path, held: _Held
    ) -> None:
        # 同じ置き場で順に走らせると、長い動画の場面を読む間、後の素材のサムネイルが出ない
        # 波形とサムネイルの置き場を 1 本にして、場面が同じ置き場に入れば必ず待たされる形で見る
        analyzer = MediaAnalyzer(store, max_workers=1)
        try:
            media = _media(video)
            analyzer.request_scenes(media)
            assert held.started.wait(10)
            analyzer.request(media)
            assert _until(lambda: analyzer.filmstrip(media) is not None, 20), "待たされた"
        finally:
            held.release.set()
            analyzer.close()

    def test_the_part_read_so_far_is_shown(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 途中を見せないと、長い動画で AI は終わるまで何も受け取れない
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        state = analyzer.scene_state(media)
        assert state.running and not state.complete
        assert state.signal is not None and len(state.signal) == 1

    def test_waiting_gives_up_at_the_limit(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 上限で戻らないと、AI の会話が解析の終わりまで止まる
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        began = time.monotonic()
        assert analyzer.wait_scenes(media.id, 0.2) is False
        assert time.monotonic() - began < 2.0

    def test_waiting_ends_when_it_is_done(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 終わったのに起こさないと、AI は上限まで待ってから結果を受け取る
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        threading.Timer(0.1, held.release.set).start()
        began = time.monotonic()
        assert analyzer.wait_scenes(media.id, 20) is True
        assert time.monotonic() - began < 10
        assert analyzer.scene_state(media).complete

    def test_cancelling_stops_the_reading(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 止まらないと、設定で切っても裏で全フレーム読み続ける
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        analyzer.cancel_scenes()
        assert held.cancelled.wait(10), "止まらない"
        assert analyzer.wait_scenes(media.id, 10)
        state = analyzer.scene_state(media)
        assert state.failure is None and not state.running

    def test_what_was_read_survives_cancelling(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 止めたときに読んだ所まで捨てると、待っていた AI は説明と違って空の一覧を受け取る
        # （PR #358 の Codex の指摘）
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        analyzer.cancel_scenes()
        assert analyzer.wait_scenes(media.id, 10)
        state = analyzer.scene_state(media)
        assert state.signal is not None and not state.complete
        assert len(state.signal) == 1

    def test_forgetting_drops_what_was_read(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 外した素材の途中の一覧が残ると、同じ ID で入れ直したときに古い切り替わりを返す
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        analyzer.cancel_scenes()
        assert analyzer.wait_scenes(media.id, 10)
        analyzer.forget(media.id)
        assert analyzer.scene_state(media).signal is None

    def test_closing_wakes_the_waiter(
        self, analyzer: MediaAnalyzer, video: Path, held: _Held
    ) -> None:
        # 窓を閉じても待ち続けると、AI のスレッドが上限まで残る
        media = _media(video)
        analyzer.request_scenes(media)
        assert held.started.wait(10)
        threading.Timer(0.1, analyzer.close).start()
        began = time.monotonic()
        assert analyzer.wait_scenes(media.id, 20) is True
        assert time.monotonic() - began < 10


class TestABrokenVideo:
    """途中で壊れた素材（PR #358 の CodeRabbit の指摘）"""

    def test_it_is_not_kept_as_complete(self, store: CacheStore, tmp_path: Path) -> None:
        # 読み終えた物として覚えると、後ろ半分の切り替わりの無い一覧を AI が全部だと思って使う
        path = write_broken_video(tmp_path / "broken.mp4")
        media = _media(path)
        first = MediaAnalyzer(store)
        try:
            first.request_scenes(media)
            assert first.wait_scenes(media.id, 30)
            state = first.scene_state(media)
            assert state.signal is not None and state.signal.broken
            assert not state.complete
        finally:
            first.close()
        kept = load_scenes(store, scenes_key(path))
        assert kept is not None and not kept.complete and kept.broken

    def test_it_is_not_read_again_every_time(
        self, store: CacheStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れた素材を頼まれるたびに全フレーム読み直すと、AI が聞くたびに長く待たされる
        # 読めた所までを覚え、素材が変わる（大きさか更新時刻）までは読み直さない
        path = write_broken_video(tmp_path / "broken.mp4")
        media = _media(path)
        first = MediaAnalyzer(store)
        try:
            first.request_scenes(media)
            assert first.wait_scenes(media.id, 30)
        finally:
            first.close()

        def refuse(*_arguments: object, **_options: object) -> SceneSignal:
            raise AssertionError("壊れた素材を読み直した")

        monkeypatch.setattr(analyzer_module, "analyze_scenes", refuse)
        second = MediaAnalyzer(store)
        try:
            second.request_scenes(media)
            assert second.wait_scenes(media.id, 30)
            state = second.scene_state(media)
            assert state.signal is not None and state.signal.broken
            second.request_scenes(media)
            assert not second.scene_state(media).running
        finally:
            second.close()


class TestFailure:
    def test_the_reason_is_kept(
        self, analyzer: MediaAnalyzer, video: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 理由を残さないと、AI は出ない一覧をいつまでも待ち直す
        def broken(*_arguments: object, **_options: object) -> SceneSignal:
            raise RuntimeError("壊れた素材")

        monkeypatch.setattr(analyzer_module, "analyze_scenes", broken)
        media = _media(video)
        analyzer.request_scenes(media)
        assert analyzer.wait_scenes(media.id, 10)
        state = analyzer.scene_state(media)
        assert state.failure is not None and "壊れた素材" in state.failure

    def test_asking_again_clears_the_reason(
        self, analyzer: MediaAnalyzer, video: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 頼み直しても前の失敗が残ると、読み直している間も「出せない」と答える
        def broken(*_arguments: object, **_options: object) -> SceneSignal:
            raise RuntimeError("壊れた素材")

        monkeypatch.setattr(analyzer_module, "analyze_scenes", broken)
        media = _media(video)
        analyzer.request_scenes(media)
        assert analyzer.wait_scenes(media.id, 10)
        monkeypatch.undo()
        analyzer.request_scenes(media)
        assert analyzer.wait_scenes(media.id, 30)
        state = analyzer.scene_state(media)
        assert state.complete and state.failure is None
