"""被写体のいそうな所の見積もり（#334）

位置の分かっている動画をその場で焼き（tests/subject_video.py）、場面の解析が同じ読み込みの
ついでに数えた並びで、顔（肌色）と動く物の側を指すこと・手がかりの無い所を手がかりと
言わないことを見る 覚えた物の版と、枠の中心の選び方も見る
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sashimono.engine.cache import CacheStore, load_scenes, save_scenes, scenes_key
from sashimono.engine.cache.scene_cache import NAMESPACE, SUFFIX
from sashimono.engine.cache.store import save_arrays
from sashimono.engine.reframe import MIN_STRENGTH
from sashimono.engine.scenes import SceneSignal, analyze_scenes
from sashimono.engine.subject import SUBJECT_BINS, SubjectTrack, best_center
from tests.subject_video import write_subject_video

#: 縦の枠の幅（1920x1080 から 1080x1920 を切るときの割合）
WINDOW = (607.5 / 1920, 1.0)


@pytest.fixture(scope="module")
def signal(tmp_path_factory: pytest.TempPathFactory) -> SceneSignal:
    video = write_subject_video(tmp_path_factory.mktemp("subject") / "subject.mp4")
    found = analyze_scenes(video)
    assert found is not None
    return found


def _between(signal: SceneSignal, start: float, end: float) -> np.ndarray:
    times = signal.subject.times
    return np.flatnonzero((times > start + 0.3) & (times < end - 0.3))


class TestTheMadeVideo:
    def test_a_face_on_the_right_is_found_on_the_right(self, signal: SceneSignal) -> None:
        # 壊れると、右に座った話し手の縦の作品で、枠が誰もいない左を切り出す
        across, _down = signal.subject.centers(WINDOW)
        rows = _between(signal, 0.0, 2.0)
        assert rows.size
        assert np.all(np.abs(across[rows] - 0.75) < 0.06)

    def test_the_face_is_followed_to_the_left(self, signal: SceneSignal) -> None:
        # 前の位置を覚えたままだと、移った人を追えない
        across, _down = signal.subject.centers(WINDOW)
        rows = _between(signal, 2.0, 4.0)
        assert np.all(np.abs(across[rows] - 0.25) < 0.06)

    def test_a_moving_thing_without_skin_is_followed(self, signal: SceneSignal) -> None:
        # 動きを数えないと、肌の見えない被写体（車・動物・後ろ姿）で枠が真ん中に止まる
        across, _down = signal.subject.centers(WINDOW)
        rows = _between(signal, 4.0, 6.0)
        assert across[rows][-1] - across[rows][0] > 0.2

    def test_a_still_frame_is_no_clue(self, signal: SceneSignal) -> None:
        # 何も無い所を手がかりと言うと、止まった風景で枠が圧縮のざらつきを追って揺れる
        rows = _between(signal, 6.0, 8.0)
        assert rows.size
        assert np.all(signal.subject.strength[rows] < MIN_STRENGTH)

    def test_the_vertical_place_is_the_middle(self, signal: SceneSignal) -> None:
        # 上下を逆に数えると、縦にも切る作品（横長から正方形より横の物）で枠が上下へずれる
        _across, down = signal.subject.centers((0.3, 0.3))
        rows = _between(signal, 0.0, 2.0)
        assert np.all(np.abs(down[rows] - 0.5) < 0.1)


class TestTheCenter:
    def test_a_small_subject_near_the_edge_is_centered(self) -> None:
        # 同じ量の入る所の真ん中だけを取ると、端の近くの人が枠の片側へ寄る
        profile = np.zeros(SUBJECT_BINS, np.float32)
        profile[int(0.8 * SUBJECT_BINS)] = 1.0
        assert best_center(profile, WINDOW[0]) == pytest.approx(0.797, abs=0.01)

    def test_the_bigger_group_wins(self) -> None:
        # 少ない方を選ぶと、大きく動く話し手ではなく端の小さな動きへ枠が寄る
        profile = np.zeros(SUBJECT_BINS, np.float32)
        profile[3] = 0.3
        profile[24] = 0.7
        assert best_center(profile, WINDOW[0]) > 0.6

    def test_nothing_gives_the_middle(self) -> None:
        # 空の分布で端を返すと、手がかりの無い場面が画面の端に寄る
        assert best_center(np.zeros(SUBJECT_BINS, np.float32), WINDOW[0]) == 0.5


class TestTheStore:
    def test_the_subject_comes_back(self, tmp_path: Path, signal: SceneSignal) -> None:
        # 覚えた物から並びが消えると、開き直した後だけ枠が真ん中に止まる
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        store = CacheStore(tmp_path / "cache")
        save_scenes(store, scenes_key(video), signal)
        loaded = load_scenes(store, scenes_key(video))
        assert loaded is not None
        assert np.array_equal(loaded.subject.times, signal.subject.times)
        assert np.allclose(loaded.subject.columns, signal.subject.columns, atol=1e-3)

    def test_an_old_store_without_the_subject_is_read_again(self, tmp_path: Path) -> None:
        # 前の版の物を読み直さないと、被写体の無い並びで作り替えて全部の場面が真ん中になる
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        store = CacheStore(tmp_path / "cache")
        key = scenes_key(video)
        save_arrays(
            store.prepare(NAMESPACE, key, SUFFIX),
            {
                "version": np.array([4]),
                "times": np.zeros(1),
                "pixel": np.zeros(1, np.float32),
                "histogram": np.zeros(1, np.float32),
                "analyzed": np.array([1.0]),
                "stream": np.array([0]),
                "broken": np.array([False]),
            },
        )
        assert load_scenes(store, key) is None

    def test_a_signal_without_a_subject_is_empty(self) -> None:
        # 手で組んだ並び（前からの試験や偽物）に被写体が無くても、作り替えは真ん中で進められる
        empty = SceneSignal(np.zeros(0), np.zeros(0), np.zeros(0), analyzed=0.0, complete=True)
        assert len(empty.subject) == 0
        assert isinstance(empty.subject, SubjectTrack)
