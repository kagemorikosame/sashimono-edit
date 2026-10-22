"""画面の中身を裏で調べて覚える係（#324）

調べる部品は偽物（絵の真ん中の色を文字として返す）に替え、取り出す絵は本物の動画から取る
場面の切り替わりは渡した並びを使う（本物の場面の解析は ``tests/ai/test_detect_scenes.py`` が見る）
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine.cache import CacheStore, ScreenAnalyzer
from sashimono.engine.cache.analyzer import MediaAnalyzer
from sashimono.engine.cache.screen_analyzer import scenes_from
from sashimono.engine.decode import probe_media
from sashimono.engine.decode.probe import forget_probe
from sashimono.engine.scenes import SceneSignal
from sashimono.engine.vision import (
    Box,
    PartsMissingError,
    ScreenKind,
    ScreenSample,
    TextLine,
)
from tests.scene_video import CUTS, write_scene_video


class FakeReader:
    """絵の真ん中の画素の赤の値を文字にして返す 何枚調べたかを数える"""

    def __init__(self, engine: str = "偽の読み取り") -> None:
        self.seen: list[float] = []
        self._engine = engine
        #: 調べる前に待つ 止める頼みが途中で効くかを見るため
        self.gate: threading.Event | None = None
        #: この枚数を調べた後から :attr:`gate` で待つ 途中の様子を読むため
        self.free = 0
        #: 待ち始めたら立てる
        self.blocked = threading.Event()

    @property
    def kind(self) -> ScreenKind:
        return ScreenKind.TEXT

    @property
    def engine(self) -> str:
        return self._engine

    def look(self, rgba: np.ndarray, seconds: float) -> ScreenSample:
        if self.gate is not None and len(self.seen) >= self.free:
            self.blocked.set()
            self.gate.wait(5)
        self.seen.append(seconds)
        height, width = rgba.shape[:2]
        red = int(rgba[height // 2, width // 2, 0])
        return ScreenSample(seconds, lines=(TextLine(f"赤{red}", Box(0.0, 0.0, 10.0, 10.0)),))


def _signal(cuts: tuple[float, ...]) -> SceneSignal:
    """``cuts`` の所だけ点数が跳ねる差の並び（30fps 12 秒）"""
    times = np.arange(1, 360, dtype=np.float64) / 30
    pixel = np.zeros_like(times, dtype=np.float32)
    for cut in cuts:
        pixel[int(np.argmin(np.abs(times - cut)))] = 1.0
    return SceneSignal(
        times=times, pixel=pixel, histogram=pixel.copy(), analyzed=12.0, complete=True
    )


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> MediaItem:
    path = write_scene_video(tmp_path_factory.mktemp("screen") / "scenes.mp4")
    forget_probe(path)
    return probe_media(path)


def _make(
    tmp_path: Path,
    reader: FakeReader | None = None,
    scenes: Callable[[MediaItem, Callable[[], bool]], SceneSignal | None] | None = None,
    per_scene: int = 2,
) -> tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]]:
    made: list[ScreenKind] = []
    found = reader or FakeReader()

    def detectors(kind: ScreenKind, ocr: str) -> FakeReader:
        del ocr
        made.append(kind)
        return found

    analyzer = ScreenAnalyzer(
        CacheStore(tmp_path / "cache"),
        scenes=scenes or (lambda _media, _cancelled: _signal(CUTS)),
        detectors=detectors,
        per_scene=per_scene,
    )
    return analyzer, found, made


@pytest.fixture
def made(tmp_path: Path) -> Iterator[tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]]]:
    created = _make(tmp_path)
    yield created
    created[0].close()


def _finish(analyzer: ScreenAnalyzer, media: MediaItem) -> None:
    analyzer.request(media, ScreenKind.TEXT)
    assert analyzer.wait(media.id, ScreenKind.TEXT, 30), "調べ終わらない"


class TestScreenAnalyzer:
    def test_each_scene_is_looked_at(
        self, made: tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]], video: MediaItem
    ) -> None:
        # 壊れると、場面の区切りを使わずに調べ、短い場面の文字や顔を丸ごと見落とす
        analyzer, reader, _ = made
        _finish(analyzer, video)
        state = analyzer.state(video, ScreenKind.TEXT)
        assert state.complete and state.failure is None
        report = state.report
        assert report is not None
        assert report.by_scene
        assert report.planned == len(report.samples) == 2 * (len(CUTS) + 1)
        assert (report.width, report.height) == (320, 180)
        # 場面 2〜4 秒（暗い青）の絵は赤が小さく、4〜6 秒（暗い赤）は大きい 時刻と絵が合っている
        blue = [s for s in report.samples if 2.0 < s.seconds < 4.0]
        red = [s for s in report.samples if 4.0 < s.seconds < 6.0]
        assert all(int(s.lines[0].text[1:]) < 80 for s in blue)
        assert all(int(s.lines[0].text[1:]) > 100 for s in red)
        assert reader.seen == sorted(reader.seen)

    def test_the_result_is_remembered_across_restarts(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、開き直すたびに全部の場面を調べ直し、AI が毎回待たされる
        first, reader, _ = _make(tmp_path)
        _finish(first, video)
        first.close()
        assert reader.seen
        second, again, _ = _make(tmp_path)
        try:
            _finish(second, video)
            assert again.seen == []
            assert second.state(video, ScreenKind.TEXT).complete
        finally:
            second.close()

    def test_missing_parts_are_kept_as_the_reason(self, tmp_path: Path, video: MediaItem) -> None:
        # 壊れると、部品の無い機械で AI が「まだ調べている」と待ち続け、入れ方も知らされない
        def missing(_kind: ScreenKind, _ocr: str) -> FakeReader:
            raise PartsMissingError("画面の解析の部品が使えません（試験）")

        waited: list[MediaItem] = []

        def scenes(media: MediaItem, _cancelled: Callable[[], bool]) -> SceneSignal | None:
            waited.append(media)
            return _signal(CUTS)

        analyzer = ScreenAnalyzer(CacheStore(tmp_path / "c"), scenes=scenes, detectors=missing)
        try:
            _finish(analyzer, video)
            state = analyzer.state(video, ScreenKind.TEXT)
            assert state.failure is not None and "部品" in state.failure
            assert not state.running
            # 部品が無ければ、全フレームを読む場面の解析を待たずに断る
            assert waited == []
        finally:
            analyzer.close()

    def test_asking_again_retries_after_installing(self, tmp_path: Path, video: MediaItem) -> None:
        # 壊れると、部品を入れた後に AI が呼び直しても、前の失敗を返し続ける
        broken = [True]
        reader = FakeReader()

        def detectors(_kind: ScreenKind, _ocr: str) -> FakeReader:
            if broken[0]:
                raise PartsMissingError("無い")
            return reader

        analyzer = ScreenAnalyzer(
            CacheStore(tmp_path / "c"),
            scenes=lambda _m, _c: _signal(CUTS),
            detectors=detectors,
        )
        try:
            _finish(analyzer, video)
            assert analyzer.state(video, ScreenKind.TEXT).failure is not None
            broken[0] = False
            _finish(analyzer, video)
            assert analyzer.state(video, ScreenKind.TEXT).complete
        finally:
            analyzer.close()

    def test_without_scenes_it_looks_evenly_and_does_not_remember(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、場面の解析に失敗した素材で何も調べないか、等間隔の結果を覚えて直した後も返す
        analyzer, _reader, _ = _make(tmp_path, scenes=lambda _m, _c: None)
        try:
            _finish(analyzer, video)
            report = analyzer.state(video, ScreenKind.TEXT).report
            assert report is not None and not report.by_scene
            assert len(report.samples) == 6
        finally:
            analyzer.close()
        second, again, _ = _make(tmp_path)
        try:
            _finish(second, video)
            assert again.seen, "等間隔の結果を控えから読んだ"
        finally:
            second.close()

    def test_a_broken_material_is_only_looked_at_where_it_can_be_read(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、途中で壊れた素材で取り出せない絵まで予定に数え、調べ終えたのに予定より
        # 少ない結果になり、AI が後ろの場面を調べ損ねたと取り違える
        broken = replace(_signal((2.0, 4.0)), complete=False, broken=True, analyzed=6.0)
        analyzer, _reader, _ = _make(tmp_path, scenes=lambda _m, _c: broken)
        try:
            _finish(analyzer, video)
            report = analyzer.state(video, ScreenKind.TEXT).report
            assert report is not None
            assert all(sample.seconds < 6.0 for sample in report.samples)
            assert report.planned == len(report.samples) == 2 * 3
        finally:
            analyzer.close()

    def test_the_chosen_ocr_reads_the_text_and_switching_reads_again(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、設定で「飾った文字も読む」を選んでも Windows の OCR で読み続けるか、
        # 切り替えた後も前の方式で読んだ文字を返す
        asked: list[str] = []
        readers = {"windows": FakeReader("windows-ocr:ja"), "rapidocr": FakeReader("rapidocr:3")}

        def detectors(_kind: ScreenKind, ocr: str) -> FakeReader:
            asked.append(ocr)
            return readers[ocr]

        analyzer = ScreenAnalyzer(
            CacheStore(tmp_path / "c"),
            scenes=lambda _m, _c: _signal(CUTS),
            detectors=detectors,
            ocr="rapidocr",
        )
        try:
            _finish(analyzer, video)
            report = analyzer.state(video, ScreenKind.TEXT).report
            assert report is not None and report.engine == "rapidocr:3"
            analyzer.set_ocr("windows")
            assert analyzer.state(video, ScreenKind.TEXT).report is None
            _finish(analyzer, video)
            again = analyzer.state(video, ScreenKind.TEXT).report
            assert again is not None and again.engine == "windows-ocr:ja"
            assert asked == ["rapidocr", "windows"]
        finally:
            analyzer.close()

    def test_a_missing_rapidocr_is_not_replaced_by_windows(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、RapidOCR を選んだのに黙って Windows の OCR で読み、飾った文字が読めないまま
        # RapidOCR の結果だと思い込む 戻した後も前の断りが残ると、Windows の OCR でも読めない
        def detectors(_kind: ScreenKind, ocr: str) -> FakeReader:
            if ocr == "rapidocr":
                raise PartsMissingError("飾った文字の読み取りの部品が入っていません")
            return FakeReader("windows-ocr:ja")

        analyzer = ScreenAnalyzer(
            CacheStore(tmp_path / "c"),
            scenes=lambda _m, _c: _signal(CUTS),
            detectors=detectors,
            ocr="rapidocr",
        )
        try:
            _finish(analyzer, video)
            state = analyzer.state(video, ScreenKind.TEXT)
            assert state.report is None and state.failure is not None
            assert "飾った文字" in state.failure
            analyzer.set_ocr("windows")
            assert analyzer.state(video, ScreenKind.TEXT).failure is None
        finally:
            analyzer.close()

    def test_an_even_result_is_redone_by_scene_once_the_scenes_are_known(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、場面の解析が 1 度失敗した素材は、同じ起動の中で後から場面が分かっても
        # 等間隔の仮の結果のままで、場面ごとに調べ直せない（PR #403 の Codex の指摘）
        known: list[SceneSignal | None] = [None]
        analyzer, reader, _ = _make(tmp_path, scenes=lambda _m, _c: known[0])
        try:
            _finish(analyzer, video)
            first = analyzer.state(video, ScreenKind.TEXT)
            assert first.report is not None and not first.report.by_scene
            # 場面が分からないままもう 1 度頼まれても、等間隔に調べ直さない（仮の結果を載せ直す）
            count = len(reader.seen)
            _finish(analyzer, video)
            assert len(reader.seen) == count
            known[0] = _signal(CUTS)
            _finish(analyzer, video)
            again = analyzer.state(video, ScreenKind.TEXT).report
            assert again is not None and again.by_scene
            assert again.planned == 2 * (len(CUTS) + 1)
        finally:
            analyzer.close()

    def test_a_job_stopped_while_waiting_asks_for_nothing(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、設定を切った後に順番待ちから出てきた仕事が、部品を作り場面の解析を頼み、
        # 素材ごとに全フレームを読んで CPU を使い続ける（PR #403 の Codex の指摘）
        reader = FakeReader()
        reader.gate = threading.Event()
        asked: list[str] = []

        def scenes(media: MediaItem, _cancelled: Callable[[], bool]) -> SceneSignal | None:
            asked.append(str(media.id))
            return _signal(CUTS)

        made: list[ScreenKind] = []

        def detectors(kind: ScreenKind, _ocr: str) -> FakeReader:
            made.append(kind)
            return reader

        analyzer = ScreenAnalyzer(CacheStore(tmp_path / "c"), scenes=scenes, detectors=detectors)
        other = replace(video, id=MediaId("m_もう一本"))
        try:
            analyzer.request(video, ScreenKind.TEXT)
            assert reader.blocked.wait(30)
            # 1 本目が調べている間に、2 本目（文字と顔）が順番を待っている
            analyzer.request(other, ScreenKind.TEXT)
            analyzer.request(other, ScreenKind.FACES)
            analyzer.cancel()
            reader.gate.set()
            assert analyzer.wait(other.id, ScreenKind.TEXT, 30)
            assert analyzer.wait(other.id, ScreenKind.FACES, 30)
            assert asked == [str(video.id)]
            assert made == [ScreenKind.TEXT]
        finally:
            reader.gate.set()
            analyzer.close()

    def test_changing_the_count_looks_again(
        self, made: tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]], video: MediaItem
    ) -> None:
        # 壊れると、設定で枚数を増やしても前の枚数の結果を返し続け、設定が効かない
        analyzer, _, _ = made
        _finish(analyzer, video)
        analyzer.set_per_scene(4)
        assert analyzer.state(video, ScreenKind.TEXT).report is None
        _finish(analyzer, video)
        report = analyzer.state(video, ScreenKind.TEXT).report
        assert report is not None and report.planned > 2 * (len(CUTS) + 1)

    def test_a_replaced_material_is_looked_at_again(
        self, made: tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]], tmp_path: Path
    ) -> None:
        # 壊れると、素材のファイルを差し替えた後も前の絵の文字や顔を返す
        analyzer, reader, _ = made
        path = write_scene_video(tmp_path / "替える.mp4")
        forget_probe(path)
        media = probe_media(path)
        _finish(analyzer, media)
        count = len(reader.seen)
        with path.open("ab") as handle:
            handle.write(b"\0")
        assert analyzer.state(media, ScreenKind.TEXT).report is None
        _finish(analyzer, replace(media))
        assert len(reader.seen) > count

    def test_forgetting_a_material_drops_its_result(
        self, made: tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]], video: MediaItem
    ) -> None:
        # 壊れると、外した素材の結果をメモリに抱え続ける
        analyzer, _, _ = made
        _finish(analyzer, video)
        analyzer.forget(video.id)
        state = analyzer.state(video, ScreenKind.TEXT)
        assert state.report is None and not state.running

    def test_cancelling_stops_midway_and_keeps_nothing(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、設定で切っても裏で調べ続け、切った意味が無い（途中の結果を全部と取り違えもする）
        reader = FakeReader()
        reader.gate = threading.Event()
        analyzer, _, _ = _make(tmp_path, reader=reader)
        try:
            analyzer.request(video, ScreenKind.TEXT)
            assert not analyzer.wait(video.id, ScreenKind.TEXT, 0.3)
            analyzer.cancel()
            reader.gate.set()
            assert analyzer.wait(video.id, ScreenKind.TEXT, 30)
            state = analyzer.state(video, ScreenKind.TEXT)
            assert state.report is None and state.failure is None and not state.running
            assert len(reader.seen) < 2 * (len(CUTS) + 1)
        finally:
            analyzer.close()

    def test_asking_again_after_stopping_keeps_it_going(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、設定を切った直後に AI が頼んだ素材が止まったままになり、AI が「止まった」を
        # 受け取って呼び直すまで何も調べない
        reader = FakeReader()
        reader.gate = threading.Event()
        analyzer, _, _ = _make(tmp_path, reader=reader)
        try:
            analyzer.request(video, ScreenKind.TEXT)
            assert reader.blocked.wait(30)
            analyzer.cancel()
            analyzer.request(video, ScreenKind.TEXT)
            reader.gate.set()
            assert analyzer.wait(video.id, ScreenKind.TEXT, 30)
            assert analyzer.state(video, ScreenKind.TEXT).complete
        finally:
            reader.gate.set()
            analyzer.close()

    def test_asking_again_after_it_decided_to_stop_is_not_lost(
        self, tmp_path: Path, video: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、設定を切った直後に AI が頼み直したとき、仕事がもう止まると決めていれば
        # 頼み直しが黙って消え、AI は「止まった」を受け取って何も読めない
        # （PR #403 の CodeRabbit の指摘 止まると決めた後・後片付けの前に頼む順を作る）
        reader = FakeReader()
        reader.gate = threading.Event()
        analyzer, _, _ = _make(tmp_path, reader=reader)
        original = analyzer._is_cancelled
        asked_again = threading.Event()

        def deciding(key: tuple[MediaId, ScreenKind]) -> bool:
            stopping = original(key)
            if stopping and not asked_again.is_set():
                # 仕事が止まると決めた所で、後片付けより先に頼み直しが来る
                asked_again.set()
                analyzer.request(video, ScreenKind.TEXT)
            return stopping

        monkeypatch.setattr(analyzer, "_is_cancelled", deciding)
        try:
            analyzer.request(video, ScreenKind.TEXT)
            assert reader.blocked.wait(30)
            analyzer.cancel()
            reader.gate.set()
            assert analyzer.wait(video.id, ScreenKind.TEXT, 30)
            assert asked_again.is_set()
            state = analyzer.state(video, ScreenKind.TEXT)
            assert state.complete, state
        finally:
            reader.gate.set()
            analyzer.close()

    def test_a_removed_material_is_not_asked_again(self, tmp_path: Path, video: MediaItem) -> None:
        # 壊れると、止める仕事の間に頼まれた素材を外しても、止まった後に調べ直して結果が戻る
        reader = FakeReader()
        reader.gate = threading.Event()
        analyzer, _, _ = _make(tmp_path, reader=reader)
        try:
            analyzer.request(video, ScreenKind.TEXT)
            assert reader.blocked.wait(30)
            analyzer.cancel()
            analyzer.request(video, ScreenKind.TEXT)
            analyzer.forget(video.id)
            reader.gate.set()
            assert analyzer.wait(video.id, ScreenKind.TEXT, 30)
            state = analyzer.state(video, ScreenKind.TEXT)
            assert state.report is None and not state.running
        finally:
            reader.gate.set()
            analyzer.close()

    def test_a_partial_result_is_readable_while_looking(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、長い素材で AI が待ちきれなかったとき、調べた所までも受け取れない
        reader = FakeReader()
        reader.gate = threading.Event()
        reader.free = 3
        analyzer, _, _ = _make(tmp_path, reader=reader)
        try:
            analyzer.request(video, ScreenKind.TEXT)
            # 3 枚を調べて 4 枚目の所で止めてある
            assert reader.blocked.wait(30)
            state = analyzer.state(video, ScreenKind.TEXT)
            assert state.running and not state.complete
            assert state.report is not None
            assert len(state.report.samples) == 3
            assert state.report.planned == 2 * (len(CUTS) + 1)
            reader.gate.set()
            assert analyzer.wait(video.id, ScreenKind.TEXT, 30)
            assert analyzer.state(video, ScreenKind.TEXT).complete
        finally:
            reader.gate.set()
            analyzer.close()

    def test_a_closed_analyzer_does_not_keep_the_ai_waiting(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、窓を閉じた後も AI のスレッドが 2 分待ち続ける
        reader = FakeReader()
        reader.gate = threading.Event()
        analyzer, _, _ = _make(tmp_path, reader=reader)
        analyzer.request(video, ScreenKind.TEXT)
        analyzer.close()
        assert analyzer.wait(video.id, ScreenKind.TEXT, 5)
        reader.gate.set()

    def test_a_picture_without_video_is_not_asked(
        self, made: tuple[ScreenAnalyzer, FakeReader, list[ScreenKind]], video: MediaItem
    ) -> None:
        # 壊れると、音だけの素材で取り出せない絵を探して失敗を積む
        analyzer, _, makes = made
        silent = replace(video, video_streams=())
        analyzer.request(silent, ScreenKind.TEXT)
        assert not analyzer.state(silent, ScreenKind.TEXT).running
        assert makes == []


class TestScenesFrom:
    def test_it_waits_for_the_real_scene_analysis(self, tmp_path: Path, video: MediaItem) -> None:
        # 壊れると、場面の解析が終わる前に調べ始め、区切りの無いまま等間隔で調べる
        scenes = MediaAnalyzer(CacheStore(tmp_path / "s"))
        try:
            signal = scenes_from(scenes)(video, lambda: False)
            assert signal is not None and signal.complete
            cuts = [cut.seconds for cut in signal.cuts()]
            assert cuts == pytest.approx(list(CUTS), abs=0.05)
        finally:
            scenes.close()

    def test_a_stopped_job_does_not_ask_for_the_scenes(
        self, tmp_path: Path, video: MediaItem
    ) -> None:
        # 壊れると、止めた後の仕事が場面の解析（全フレームを読む）を予約してから止まる
        scenes = MediaAnalyzer(CacheStore(tmp_path / "s"))
        asked: list[object] = []
        scenes.request_scenes = lambda media, **_options: asked.append(media)  # type: ignore[method-assign]
        try:
            assert scenes_from(scenes)(video, lambda: True) is None
            assert asked == []
        finally:
            scenes.close()

    def test_stopping_while_waiting_gives_up(self, tmp_path: Path, video: MediaItem) -> None:
        # 壊れると、設定で切ったり素材を外したりしても、場面の解析を待ち続けて係が詰まる
        class Slow(MediaAnalyzer):
            """場面の解析がいつまでも終わらない係"""

            def wait_scenes(self, media_id: object, timeout: float) -> bool:
                del media_id, timeout
                return False

        scenes = Slow(CacheStore(tmp_path / "s"))
        asked = [0]

        def cancelled() -> bool:
            asked[0] += 1
            return asked[0] >= 2

        try:
            assert scenes_from(scenes)(video, cancelled) is None
            assert asked[0] == 2
        finally:
            scenes.cancel_scenes()
            scenes.close()
