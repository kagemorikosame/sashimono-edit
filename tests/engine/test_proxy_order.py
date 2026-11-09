"""控えを作る順と、待ち時間の見せ方（#441）"""

from __future__ import annotations

import threading
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.model import (
    Clip,
    MediaId,
    MediaItem,
    Project,
    Scene,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import proxy as proxy_module
from sashimono.engine.cache.progress import JobBoard
from sashimono.engine.cache.proxy import ProxyBuilder, ProxyStore, proxy_order, used_media
from sashimono.engine.cache.store import CacheStore
from sashimono.ui.progress_display import describe_background, rough_duration


def _media(name: str, seconds: int = 60) -> MediaItem:
    stream = VideoStreamInfo(
        index=0,
        width=3840,
        height=2160,
        frame_rate=FrameRate(30),
        time_base=Fraction(1, 30),
        codec="h264",
    )
    return MediaItem(path=Path(f"{name}.mp4"), duration=Fraction(seconds), video_streams=(stream,))


def _project(media: list[MediaItem], placed: list[tuple[int, int]]) -> Project:
    """``placed`` は（素材の番号, 置く所） 長さは 30 コマ"""
    base = Project.create()
    clips = tuple(
        Clip(timeline_start=start, duration=30, media_id=media[index].id)
        for index, start in sorted(placed, key=lambda pair: pair[1])
    )
    timeline = replace(base.timeline, tracks=(Track(TrackKind.VIDEO, "V1", clips),))
    return replace(base.with_timeline(timeline), media=tuple(media))


class TestOrder:
    def test_used_media_near_the_playhead_come_first(self) -> None:
        """壊れると、開いた直後に見る所の控えが、使っていない素材の後回しになる"""
        media = [_media(name) for name in "abcd"]
        project = _project(media, [(2, 1000), (1, 100), (3, 0)])
        order = proxy_order(project, 110)
        assert order == [media[1].id, media[3].id, media[2].id, media[0].id]

    def test_unused_media_keep_the_pool_order(self) -> None:
        """壊れると、使っていない素材の作る順が毎回入れ替わり、一覧の上から待つ人の見当が外れる"""
        media = [_media(name) for name in "abc"]
        project = _project(media, [])
        assert proxy_order(project, 0) == [item.id for item in media]

    def test_media_inside_a_placed_scene_follow_the_playhead(self) -> None:
        """壊れると、再生位置がシーンを置いたクリップの上にあるとき、そこで見えている素材の
        控えが、ほかのシーンにだけある素材と同じ後ろへ回り、プレビューが長く重いまま待たされる
        """
        media = [_media(name) for name in "abc"]
        base = Project.create()
        inner = replace(
            base.timeline,
            tracks=(
                Track(
                    TrackKind.VIDEO,
                    "S1",
                    (
                        Clip(timeline_start=0, duration=30, media_id=media[2].id),
                        Clip(timeline_start=500, duration=30, media_id=media[1].id),
                    ),
                ),
            ),
        )
        scene = Scene("場面", inner)
        main = replace(
            base.timeline,
            tracks=(
                Track(
                    TrackKind.VIDEO,
                    "V1",
                    (
                        Clip(timeline_start=0, duration=600, scene_id=scene.id),
                        Clip(timeline_start=2000, duration=30, media_id=media[0].id),
                    ),
                ),
            ),
        )
        project = replace(base, media=tuple(media), scenes=(scene,)).with_timeline(main)
        # 再生位置 510 はシーンの中の 510 コマ目（b の上） c はシーンの頭で 481 コマ離れる
        assert proxy_order(project, 510) == [media[1].id, media[2].id, media[0].id]

    def test_a_fast_scene_counts_in_outer_frames(self) -> None:
        """壊れると、10 倍速で置いたシーンの中の素材を外の 10 倍遠くに数え、外ですぐ先に見える
        素材の控えが、外のもっと先の素材より後へ回る
        """
        media = [_media(name) for name in "ab"]
        base = Project.create()
        inner = replace(
            base.timeline,
            tracks=(
                Track(
                    TrackKind.VIDEO,
                    "S1",
                    (Clip(timeline_start=500, duration=30, media_id=media[0].id),),
                ),
            ),
        )
        scene = Scene("場面", inner)
        main = replace(
            base.timeline,
            tracks=(
                Track(
                    TrackKind.VIDEO,
                    "V1",
                    (
                        Clip(
                            timeline_start=0,
                            duration=100,
                            scene_id=scene.id,
                            speed=Fraction(10),
                        ),
                        Clip(timeline_start=160, duration=30, media_id=media[1].id),
                    ),
                ),
            ),
        )
        project = replace(base, media=tuple(media), scenes=(scene,)).with_timeline(main)
        # a はシーンの中で 500 コマ先 10 倍速なので外では 50 コマ先 b は 160 コマ先
        assert proxy_order(project, 0) == [media[0].id, media[1].id]

    def test_a_scene_placed_twice_at_every_depth_is_walked_once(self) -> None:
        """壊れると、同じシーンを 2 か所に置いて入れ子にした作品で、たどる数が段の数の指数で
        増え、作品を開くたびに画面が長く止まる
        """
        media = [_media("a")]
        base = Project.create()
        leaf = replace(
            base.timeline,
            tracks=(
                Track(
                    TrackKind.VIDEO,
                    "S",
                    (Clip(timeline_start=0, duration=30, media_id=media[0].id),),
                ),
            ),
        )
        scenes = [Scene("0", leaf)]
        for depth in range(1, 40):
            below = scenes[-1].id
            timeline = replace(
                base.timeline,
                tracks=(
                    Track(
                        TrackKind.VIDEO,
                        "S",
                        (
                            Clip(timeline_start=0, duration=30, scene_id=below),
                            Clip(timeline_start=100, duration=30, scene_id=below),
                        ),
                    ),
                ),
            )
            scenes.append(Scene(str(depth), timeline))
        main = replace(
            base.timeline,
            tracks=(
                Track(
                    TrackKind.VIDEO,
                    "V1",
                    (Clip(timeline_start=0, duration=30, scene_id=scenes[-1].id),),
                ),
            ),
        )
        project = replace(base, media=tuple(media), scenes=tuple(scenes)).with_timeline(main)
        # 覚えずにたどると 2 の 40 乗で終わらない
        assert proxy_order(project, 0) == [media[0].id]
        assert used_media(project) == {media[0].id}

    def test_used_media(self) -> None:
        """壊れると、置いたときに作る設定で、使い始めた素材の控えが頼まれない"""
        media = [_media(name) for name in "ab"]
        assert used_media(_project(media, [(1, 0)])) == {media[1].id}


class TestBuilderOrder:
    def test_waiting_proxies_follow_the_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、頼まれた順（素材一覧の並び）に作り、使っている素材が待たされる"""
        gate = threading.Event()
        made: list[str] = []
        first_started = threading.Event()

        def fake(source: Path, target: Path, **_kwargs: object) -> Path:
            made.append(source.stem)
            if len(made) == 1:
                first_started.set()
                gate.wait(30.0)
            target.write_bytes(b"proxy")
            return target

        monkeypatch.setattr(proxy_module, "create_proxy", fake)
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path)))
        media = [_media(name) for name in "abcd"]
        done = threading.Semaphore(0)
        try:
            builder.request(media[0], on_ready=lambda _id: done.release())
            assert first_started.wait(30.0)
            for item in media[1:]:
                builder.request(item, on_ready=lambda _id: done.release())
            assert builder.waiting() == 3
            builder.set_order([media[3].id, media[1].id])
            gate.set()
            for _ in media:
                assert done.acquire(timeout=30.0)
            assert made == ["a", "d", "b", "c"]
        finally:
            gate.set()
            builder.close()

    def test_a_removed_waiting_proxy_is_not_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、待っている間に外した素材でも子を起こして開きに行き、使っている素材の控えが遅れる"""
        gate = threading.Event()
        made: list[str] = []
        started = threading.Event()

        def fake(source: Path, target: Path, **_kwargs: object) -> Path:
            made.append(source.stem)
            started.set()
            gate.wait(30.0)
            target.write_bytes(b"proxy")
            return target

        monkeypatch.setattr(proxy_module, "create_proxy", fake)
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path)))
        first, removed, kept = (_media(name) for name in "abc")
        done = threading.Event()
        try:
            builder.request(first)
            assert started.wait(30.0)
            builder.request(removed)
            builder.request(kept, on_ready=lambda _id: done.set())
            builder.forget(removed.id)
            gate.set()
            assert done.wait(30.0)
            assert made == ["a", "c"]
        finally:
            gate.set()
            builder.close()

    def test_closing_lets_go_of_waiting_proxies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、設定で係を作り直した後も、古い係が待っている素材と知らせ先を抱えたまま残る"""
        gate = threading.Event()
        started = threading.Event()

        def fake(source: Path, target: Path, **_kwargs: object) -> Path:
            started.set()
            gate.wait(30.0)
            return target

        monkeypatch.setattr(proxy_module, "create_proxy", fake)
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path)))
        try:
            builder.request(_media("a"))
            assert started.wait(30.0)
            builder.request(_media("b"))
            assert builder.waiting() == 1
            builder.close()
            assert builder.waiting() == 0
        finally:
            gate.set()
            builder.close()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestRemainingTime:
    def test_it_comes_from_the_speed_so_far(self) -> None:
        """壊れると、何分待つのかが出ず、長い録画の控えをいつまで待つか分からない"""
        clock = Clock()
        board = JobBoard(clock=clock)
        board.start("a", MediaId("a"), weight=60.0)
        board.start("b", MediaId("b"), weight=120.0)
        board.report("a", 0.0)
        clock.now = 10.0
        board.report("a", 0.5)
        snapshot = board.poll()
        # 30 秒ぶんを 10 秒で作った（3 倍速） 残り 30 + 120 秒ぶんは 50 秒
        assert snapshot.remaining_seconds == pytest.approx(50.0)
        assert snapshot.remaining == 2

    def test_it_waits_until_the_speed_settles(self) -> None:
        """壊れると、走り始めの遅い数秒で、何時間も掛かるような見込みが一瞬出る"""
        clock = Clock()
        board = JobBoard(clock=clock)
        board.start("a", MediaId("a"), weight=60.0)
        board.report("a", 0.0)
        clock.now = 1.0
        board.report("a", 0.01)
        assert board.poll().remaining_seconds is None

    def test_finished_work_keeps_the_speed(self) -> None:
        """壊れると、1 本終えるたびに速さが分からなくなり、次の素材の見込みが数秒消える"""
        clock = Clock()
        board = JobBoard(clock=clock)
        board.start("a", MediaId("a"), weight=60.0)
        board.report("a", 0.0)
        clock.now = 20.0
        board.finish("a")
        board.start("b", MediaId("b"), weight=30.0)
        assert board.poll().remaining_seconds == pytest.approx(10.0)

    def test_each_waiting_job_adds_its_start_up(self) -> None:
        """壊れると、子の起動とエンコーダの用意の分が見込みから抜け、長い素材の後に短い素材が
        多く待つとき、残り時間を実際より大幅に短く見積もる
        """
        clock = Clock()
        board = JobBoard(clock=clock)
        board.start("long", MediaId("long"), weight=60.0)
        for index in range(100):
            board.start(index, MediaId(f"short{index}"), weight=1.0)
        board.begin("long")
        clock.now = 10.0
        board.report("long", 0.0)
        clock.now = 20.0
        board.report("long", 0.5)
        # 起動 10 秒 → 10 秒で 30 秒ぶん（3 倍速） 残りは（30 + 100）÷ 3 に、待っている
        # 100 本の起動 10 秒ずつを足す
        assert board.poll().remaining_seconds == pytest.approx(130 / 3 + 1000)

    def test_a_job_starting_up_adds_only_what_is_left(self) -> None:
        """壊れると、起動の途中の 1 本にも起動をまるごと足し、見込みが長く出る"""
        clock = Clock()
        board = JobBoard(clock=clock)
        board.start("a", MediaId("a"), weight=60.0)
        board.start("b", MediaId("b"), weight=60.0)
        board.begin("a")
        clock.now = 10.0
        board.report("a", 0.0)
        clock.now = 20.0
        board.finish("a")
        board.begin("b")
        clock.now = 24.0
        # 10 秒で 60 秒ぶん（6 倍速） 残り 60 秒ぶんは 10 秒 起動は平均 10 秒のうち 4 秒が過ぎた
        assert board.poll().remaining_seconds == pytest.approx(10.0 + 6.0)

    def test_failed_work_is_not_counted_as_speed(self) -> None:
        """壊れると、開けずにすぐ落ちた素材で、見込みが速すぎる側へずれる"""
        clock = Clock()
        board = JobBoard(clock=clock)
        board.start("a", MediaId("a"), weight=600.0)
        board.report("a", 0.0)
        clock.now = 5.0
        board.finish("a", "開けない")
        board.start("b", MediaId("b"), weight=30.0)
        assert board.poll().remaining_seconds is None


class TestStatusText:
    def test_the_count_and_time_left_are_shown(self) -> None:
        """壊れると、何本・何分待つのかがステータスバーに出ない"""
        clock = Clock()
        proxy = JobBoard(clock=clock)
        proxy.start("a", MediaId("a"), weight=60.0)
        proxy.start("b", MediaId("b"), weight=300.0)
        proxy.start("c", MediaId("c"), weight=300.0)
        proxy.report("a", 0.0)
        clock.now = 30.0
        proxy.finish("a")
        text = describe_background(proxy.poll(), JobBoard().poll())
        assert text == "控え 1/3 本 33%（残り 2 本 およそ 5 分）"

    def test_a_single_proxy_does_not_repeat_the_count(self) -> None:
        """壊れると、1 本だけのときに「0/1 本」の隣へ「残り 1 本」と同じ数が並び、
        ステータスバーが煩雑になる
        """
        proxy = JobBoard()
        proxy.start("a", MediaId("a"), weight=60.0)
        assert describe_background(proxy.poll(), JobBoard().poll()) == "控え 0/1 本 0%"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(20, "1 分足らず"), (61, "2 分"), (3600, "1 時間"), (3700, "1 時間 2 分")],
    )
    def test_rough_duration(self, seconds: float, expected: str) -> None:
        """壊れると、見込みが秒まで出て、数割ずれる見込みが正確そうに見える"""
        assert rough_duration(seconds) == expected
