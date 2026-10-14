"""場面の切り替わりの検出（#319）

切り替わりの位置が分かっている動画をその場で焼き（tests/scene_video.py）、全部を
拾うこと・拾わなくてよい所を拾わないことを見る 感度と最短の場面の効き方は、
差の並びを手で組んで確かめる
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from sashimono.engine.scenes import (
    DEFAULT_THRESHOLD,
    SceneScan,
    SceneSignal,
    analyze_scenes,
)
from tests.scene_video import CUTS, FPS, write_scene_video

#: 見つけた位置と焼いた位置の差の上限（秒） 1 フレームより細かく合っていること
TOLERANCE = 0.5 / FPS


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_scene_video(tmp_path_factory.mktemp("scenes") / "scenes.mp4")


@pytest.fixture(scope="module")
def signal(scene_video: Path) -> SceneSignal:
    found = analyze_scenes(scene_video)
    assert found is not None
    return found


def _signal(times: list[float], pixel: list[float], histogram: list[float]) -> SceneSignal:
    return SceneSignal(
        times=np.asarray(times, np.float64),
        pixel=np.asarray(pixel, np.float32),
        histogram=np.asarray(histogram, np.float32),
        analyzed=times[-1] if times else 0.0,
        complete=True,
    )


def _steady(count: int, *, pixel: float = 0.01, histogram: float = 0.01) -> SceneSignal:
    """30fps で ``count`` 枚 どこも同じ差（切り替わりの無い素材）"""
    times = [(index + 1) / FPS for index in range(count)]
    return _signal(times, [pixel] * count, [histogram] * count)


class TestTheMadeVideo:
    def test_every_cut_is_found_on_its_frame(self, signal: SceneSignal) -> None:
        # 1 つでも抜けると、AI はその場面の頭を見ないまま切る位置を決める
        found = [cut.seconds for cut in signal.cuts()]
        assert len(found) == len(CUTS), found
        for expected, got in zip(CUTS, found, strict=True):
            assert abs(got - expected) <= TOLERANCE, (expected, got)

    def test_a_fast_pan_is_not_a_cut(self, signal: SceneSignal) -> None:
        # 速いパンを切り替わりと見ると、動きの多い素材が細切れの場面の一覧になる
        assert not [cut for cut in signal.cuts() if cut.seconds < CUTS[0] - TOLERANCE]

    def test_a_slow_dark_fade_is_not_a_cut(self, signal: SceneSignal) -> None:
        # 暗い所のゆっくりした明るさの変化は、ヒストグラムの段だけが跳ねる 数えると、
        # 暗転やフェードのある素材で何も無い所を切り替わりとして AI に見せる
        inside = [cut for cut in signal.cuts() if 8.0 + TOLERANCE < cut.seconds < 10.0 - TOLERANCE]
        assert not inside

    def test_the_cut_lies_between_the_two_frames_given(self, signal: SceneSignal) -> None:
        # before と seconds の間に切り替わりがある 逆だと、AI が前の場面だけを見て確かめる
        for cut in signal.cuts():
            assert cut.before < cut.seconds
            assert cut.seconds - cut.before <= 1.5 / FPS

    def test_the_read_stream_is_named(self, signal: SceneSignal, scene_video: Path) -> None:
        # 番号が無いと、映像が何本もある素材で別の映像のクリップにまで切り替わりを当てる
        with av.open(str(scene_video)) as container:
            assert signal.stream == container.streams.video[0].index

    def test_the_whole_video_is_read(self, signal: SceneSignal) -> None:
        # 終わりまで読んだ印が無いと、AI は続きを待ち続ける
        assert signal.complete
        assert signal.analyzed == pytest.approx(len(signal) / FPS + 1 / FPS, abs=0.1)


class TestReading:
    def test_it_stops_when_asked(self, scene_video: Path) -> None:
        # 止まらないと、素材を外したり窓を閉じたりした後も裏で全フレーム読み続ける
        assert analyze_scenes(scene_video, should_cancel=lambda: True) is None

    def test_a_scan_shows_what_was_read_so_far(self, scene_video: Path) -> None:
        # 途中を見せられないと、長い素材で AI は解析が終わるまで何も受け取れない
        scan = SceneScan()
        seen: list[int] = []

        def peek(fraction: float) -> None:
            if fraction >= 0.5 and not seen:
                seen.append(len(scan.snapshot()))

        analyze_scenes(scene_video, scan=scan, progress=peek)
        assert seen and 0 < seen[0] < len(scan.snapshot())
        assert not scan.snapshot().complete

    def test_progress_reaches_the_end(self, scene_video: Path) -> None:
        # 終わりまで届かないと、ステータスバーの割合が 100% にならない
        reported: list[float] = []
        analyze_scenes(scene_video, progress=reported.append)
        assert reported[-1] == 1.0
        assert reported == sorted(reported)

    def test_a_sound_only_file_has_no_scenes(self, tmp_path: Path) -> None:
        # 音だけの素材で並びを返すと、AI は場面が 1 つも無い動画だと受け取る
        path = tmp_path / "tone.wav"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("pcm_s16le", rate=8000)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 800), np.int16), format="s16", layout="mono"
            )
            frame.sample_rate = 8000
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        assert analyze_scenes(path) is None

    def test_a_still_image_is_one_scene(self, tmp_path: Path) -> None:
        # 静止画で失敗にすると、画像を入れたプロジェクトで AI の道具が理由の無い失敗を返す
        path = tmp_path / "still.png"
        image = QImage(64, 48, QImage.Format.Format_RGB32)
        image.fill(QColor(200, 30, 30))
        assert image.save(str(path))
        found = analyze_scenes(path)
        assert found is not None and found.complete and found.cuts() == ()

    def test_a_missing_file_has_no_scenes(self, tmp_path: Path) -> None:
        # 開けない素材で例外を投げると、裏のスレッドの解析が理由を残さずに落ちる
        assert analyze_scenes(tmp_path / "missing.mp4") is None


class TestTheThreshold:
    def test_a_lone_jump_is_a_cut(self) -> None:
        # 跳ねた 1 枚を拾えないと、はっきりしたカットさえ AI に渡せない
        signal = _steady(60)
        signal.pixel[30] = 0.3
        signal.histogram[30] = 0.6
        cuts = signal.cuts()
        assert [round(cut.seconds * FPS) for cut in cuts] == [31]

    def test_a_higher_threshold_drops_weak_cuts(self) -> None:
        # 感度を変えても一覧が変わらないと、AI は細かすぎる一覧を絞れない
        signal = _steady(90)
        signal.pixel[30], signal.histogram[30] = 0.3, 0.6
        signal.pixel[60], signal.histogram[60] = 0.08, 0.2
        assert len(signal.cuts(DEFAULT_THRESHOLD)) == 2
        assert len(signal.cuts(0.3)) == 1

    def test_steady_motion_is_not_a_cut(self) -> None:
        # 差が大きいまま続く所（パン）は、前後の普段の差を引くので点数が上がらない
        signal = _steady(90, pixel=0.2, histogram=0.05)
        assert signal.cuts() == ()

    def test_shape_alone_does_not_make_a_cut(self) -> None:
        # 画素がほとんど動かずヒストグラムだけが跳ねる（暗いフェード）のは切り替わりではない
        signal = _steady(60)
        signal.histogram[30] = 0.9
        signal.pixel[30] = 0.012
        assert signal.cuts() == ()

    def test_shape_helps_a_cut_with_a_small_pixel_change(self) -> None:
        # 明るさの近い 2 つの場面は画素の差が小さい 形の差で押し上げないと見落とす
        signal = _steady(60)
        signal.pixel[30] = 0.06
        signal.histogram[30] = 0.9
        assert len(signal.cuts()) == 1


class TestTheShortestScene:
    def test_a_flash_counts_once(self) -> None:
        # 一瞬のフラッシュは入りと戻りで 2 回跳ねる 2 つとも返すと 1 コマの場面ができる
        signal = _steady(90)
        signal.pixel[40], signal.histogram[40] = 0.4, 0.8
        signal.pixel[41], signal.histogram[41] = 0.3, 0.6
        cuts = signal.cuts(min_scene=0.5)
        assert [round(cut.seconds * FPS) for cut in cuts] == [41]

    def test_zero_keeps_both(self) -> None:
        # 0 でも寄せると、細かいカットを全部見たい AI が最短の場面を外せない
        signal = _steady(90)
        signal.pixel[40], signal.histogram[40] = 0.4, 0.8
        signal.pixel[41], signal.histogram[41] = 0.3, 0.6
        assert len(signal.cuts(min_scene=0.0)) == 2

    def test_the_range_is_applied_after_merging(self) -> None:
        # 範囲で先に絞ると、範囲の外の強い切り替わりに寄せて消えるはずの弱い物が残り、
        # 範囲の取り方で答えが変わる
        signal = _steady(90)
        signal.pixel[40], signal.histogram[40] = 0.4, 0.8
        signal.pixel[45], signal.histogram[45] = 0.2, 0.4
        start = float(signal.times[43])
        assert signal.cuts(start=start) == ()


class TestEdges:
    def test_an_empty_signal_has_no_cuts(self) -> None:
        # 静止画（差の無い素材）で落ちると、画像を入れたプロジェクトで道具が使えない
        assert SceneSignal.empty().cuts() == ()

    def test_one_difference_does_not_break(self) -> None:
        # 2 枚だけの動画（差が 1 つ）で中央値の窓が作れず落ちると、短い素材で道具が使えない
        signal = _signal([1 / FPS], [0.5], [0.9])
        assert len(signal.scores()) == 1
