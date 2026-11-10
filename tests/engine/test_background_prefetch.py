"""先読みを別のスレッドで描く（Issue #56 の 3）

壊れ方はどれも絵にしか出ず、しかも当たったコマだけに出るので目では気付きにくい

- 共有したテクスチャを描き終える前に渡す 画面に描きかけの絵や、前のコマが出る
- 編集の後に、古いプロジェクトで描いた絵を渡す 直したのに直っていない絵が残る
- 走り係のデコーダを画面の側からも触る 同じデコーダを 2 スレッドが進めて絵が飛ぶ
- 止めるときに走り係を待たない デコーダと GL の資源を掴んだままスレッドが消える
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from OpenGL import GL
from PySide6.QtCore import QCoreApplication

from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddScene,
    AddTrack,
    InScene,
    insert_scene,
    new_scene,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    MediaId,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import VideoDecoder, probe_media
from sashimono.engine.gpu import Framebuffer, OffscreenGLContext
from sashimono.engine.render import FULL_QUALITY, FrameRenderer, Invalidation
from sashimono.engine.render.background import WAIT_FOR_WORKER_S, BackgroundPrefetch, _Worker
from sashimono.engine.render.prefetch import PreviewCache
from tests.media_fixtures import make_sample

pytestmark = pytest.mark.usefixtures("gpu")

SIZE = 64
SETTINGS = ProjectSettings(width=SIZE, height=SIZE, frame_rate=FrameRate(30))
VIEWPORT = (0, 0, SIZE, SIZE)
FRAME_BYTES = SIZE * SIZE * 4


@pytest.fixture
def screen_context() -> Iterator[OffscreenGLContext]:
    """画面の側のコンテキストの代わり 走り係はこれと共有する"""
    context = OffscreenGLContext()
    yield context
    context.release()


def _gray_project(level: float = 0.5, frames: int = 40) -> Project:
    """コマごとに薄さが変わる中間色 どのコマの絵かを画素で見分けるため"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create(SETTINGS))
    source = GeneratedSource(
        kind="shape", params={"shape": "background", "color": (level, level, level, 1.0)}
    )
    fading = AnimatedValue(
        keyframes=(Keyframe(frame=0, value=1.0), Keyframe(frame=frames - 1, value=0.1))
    )
    clip = Clip(timeline_start=0, duration=frames, source=source, opacity=fading)
    return AddClip(track.id, clip).apply(project)


def _start(
    project: Project, share: OffscreenGLContext, frames: int, *, playhead: int = 0
) -> BackgroundPrefetch:
    return BackgroundPrefetch(
        project,
        share=share.context,
        quality=FULL_QUALITY,
        proxies=None,
        decode_threads=1,
        budget_bytes=FRAME_BYTES * frames,
        playhead=playhead,
    )


def _wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    """走り係からの合図は画面のスレッドのイベントループで届くので、回しながら待つ"""
    deadline = time.perf_counter() + timeout
    while not condition():
        if time.perf_counter() > deadline:
            raise AssertionError("待っても終わらなかった")
        QCoreApplication.processEvents()
        time.sleep(0.005)


def _read(surface: Framebuffer) -> np.ndarray:
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, surface.handle)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    raw = GL.glReadPixels(0, 0, surface.width, surface.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(raw, dtype=np.uint8).reshape(surface.height, surface.width, 4)
    return np.ascontiguousarray(image)


class _Screen:
    """画面の側 自分のレンダラと、出す先の描画先を持つ"""

    def __init__(self, project: Project, context: OffscreenGLContext) -> None:
        self.context = context
        self.renderer = FrameRenderer(project, context=context)
        with context:
            self.target = Framebuffer(SIZE, SIZE, internal_format=GL.GL_RGBA8)

    def direct(self, frame: int, project: Project | None = None) -> np.ndarray:
        """その場で描いた絵"""
        with self.context:
            if project is not None:
                self.renderer.set_project(project)
            self.renderer.compose(frame)
            self.renderer.compositor.present(self.target.handle, VIEWPORT)
            return _read(self.target)

    def shown(self, background: BackgroundPrefetch, frame: int) -> np.ndarray | None:
        """走り係が貯めた絵を出したもの 貯まっていなければ ``None``"""
        with self.context:
            if not background.show(frame, self.renderer.compositor, self.target.handle, VIEWPORT):
                return None
            return _read(self.target)

    def close(self) -> None:
        with self.context:
            self.target.release()
        self.renderer.close()


@pytest.fixture
def screen(screen_context: OffscreenGLContext) -> Iterator[_Screen]:
    made = _Screen(_gray_project(), screen_context)
    yield made
    made.close()


class TestWhatItHandsOver:
    def test_the_frames_match_a_direct_render(
        self, screen_context: OffscreenGLContext, screen: _Screen
    ) -> None:
        """貯めた絵を出すと、その場で描いた絵と画素まで同じ

        共有したテクスチャは、書いた側が描き終える前に読むと中身が保証されない
        走り係が ``glFinish`` を省くと、描きかけの絵や空の絵が出る
        """
        background = _start(_gray_project(), screen_context, 8)
        try:
            _wait_until(lambda: background.cached == frozenset(range(8)))
            first = screen.shown(background, 0)
            later = screen.shown(background, 5)
        finally:
            background.close()
        assert first is not None and later is not None
        assert np.array_equal(first, screen.direct(0))
        assert np.array_equal(later, screen.direct(5))
        # 取り違えたら気付けるように、2 つのコマは別の絵であることも押さえる
        assert not np.array_equal(first, later)

    def test_an_edit_hides_the_old_frames_at_once(
        self, screen_context: OffscreenGLContext, screen: _Screen
    ) -> None:
        """編集したその場で、古い絵は出さなくなる

        走り係が頼みを受け取るのを待ってから外すと、その間の描き直しで
        直す前の絵が出る
        """
        background = _start(_gray_project(0.5), screen_context, 4)
        try:
            _wait_until(lambda: len(background.cached) == 4)
            edited = _gray_project(0.8)
            background.set_project(edited, Invalidation.all())
            assert background.cached == frozenset(), "編集の後も古い絵を出せるまま残っている"
            assert screen.shown(background, 0) is None
            _wait_until(lambda: len(background.cached) == 4)
            shown = screen.shown(background, 2)
        finally:
            background.close()
        assert shown is not None
        assert np.array_equal(shown, screen.direct(2, edited))

    def test_a_frame_drawn_across_an_edit_is_not_handed_over(
        self,
        screen_context: OffscreenGLContext,
        screen: _Screen,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """描いている最中に編集が入ったら、その絵は渡さない

        走り係は古いプロジェクトで描き始めている そのまま載せると、編集で外した
        はずのコマに、直す前の絵が戻ってくる
        """
        started = threading.Event()
        resume = threading.Event()
        original = FrameRenderer.compose

        def held(renderer: FrameRenderer, frame: int, **options: Any) -> None:
            if threading.current_thread() is not threading.main_thread() and not resume.is_set():
                started.set()
                resume.wait(10)
            original(renderer, frame, **options)

        # 描き終えて載せるかどうか決めた直後で走り係を止める 止めないと、次に受け取る
        # 編集の頼みがすぐに古い絵を外すので、古い絵が出ている間を見逃す
        published = threading.Event()
        proceed = threading.Event()
        original_publish = _Worker._publish

        def paused(worker: _Worker, frame: int, cache: PreviewCache) -> None:
            original_publish(worker, frame, cache)
            if not published.is_set():
                published.set()
                proceed.wait(10)

        monkeypatch.setattr(FrameRenderer, "compose", held)
        monkeypatch.setattr(_Worker, "_publish", paused)
        background = _start(_gray_project(0.5), screen_context, 1)
        try:
            assert started.wait(10)
            edited = _gray_project(0.8)
            background.set_project(edited, Invalidation.all())
            resume.set()
            assert published.wait(10)
            stale = screen.shown(background, 0)
            proceed.set()
            _wait_until(lambda: background.cached == frozenset({0}))
            shown = screen.shown(background, 0)
        finally:
            resume.set()
            proceed.set()
            background.close()
        assert stale is None, "編集の前に描いた絵を渡している"
        assert shown is not None
        assert np.array_equal(shown, screen.direct(0, edited))


class TestWaitingForTheWorker:
    def test_a_frame_the_worker_is_about_to_finish_is_waited_for(
        self,
        screen_context: OffscreenGLContext,
        screen: _Screen,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """走り係がすぐ描き終えるコマは待って受け取る

        待たずに画面の側でも描くと、同じコマを 2 つのスレッドで描いて取り合う
        """
        started = threading.Event()
        original = FrameRenderer.compose

        def quick(renderer: FrameRenderer, frame: int, **options: Any) -> None:
            if threading.current_thread() is not threading.main_thread():
                started.set()
                time.sleep(0.05)
            original(renderer, frame, **options)

        monkeypatch.setattr(FrameRenderer, "compose", quick)
        background = _start(_gray_project(), screen_context, 1)
        try:
            assert started.wait(10)
            shown = screen.shown(background, 0)
        finally:
            background.close()
        assert shown is not None, "描き終わる直前のコマを待たずに諦めた"

    def test_a_slow_worker_is_not_waited_for_past_the_frame_budget(
        self,
        screen_context: OffscreenGLContext,
        screen: _Screen,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """走り係がなかなか描き終えないなら、上限で諦めて画面の側で描く

        待つ間は画面のスレッドが止まり、クリックもキーも受け付けない 上限が
        長いと、走り係が固まったときに操作がその間ずっと止まる
        """
        started = threading.Event()
        resume = threading.Event()
        original = FrameRenderer.compose

        def stuck(renderer: FrameRenderer, frame: int, **options: Any) -> None:
            if threading.current_thread() is not threading.main_thread() and not resume.is_set():
                started.set()
                resume.wait(10)
            original(renderer, frame, **options)

        monkeypatch.setattr(FrameRenderer, "compose", stuck)
        background = _start(_gray_project(), screen_context, 1)
        try:
            assert started.wait(10)
            began = time.perf_counter()
            shown = screen.shown(background, 0)
            waited = time.perf_counter() - began
        finally:
            resume.set()
            background.close()
        assert shown is None, "描けていないコマを出したことになっている"
        assert waited < 0.5, f"画面のスレッドを {waited:.1f} 秒止めた"
        assert waited >= WAIT_FOR_WORKER_S * 0.9, "走り係を待たずに諦めた"


class TestWhoTouchesTheDecoders:
    def test_each_decoder_stays_on_one_thread(
        self,
        screen_context: OffscreenGLContext,
        media_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """画面の側と走り係が同じ素材を読んでも、同じデコーダは触らない

        同じデコーダを 2 スレッドから進めると、読み位置が食い違って絵が飛ぶ（PR #98）
        レンダラごとに自分のデコーダを開くので、1 つのデコーダは 1 つのスレッドだけが触る
        """
        project = _video_project(media_dir, "background.mp4")
        guard = threading.Lock()
        threads: dict[int, set[int]] = {}
        busy: set[int] = set()
        overlapped: list[int] = []
        original = VideoDecoder.frame_at

        def watched(decoder: VideoDecoder, when: object) -> object:
            key = id(decoder)
            with guard:
                if key in busy:
                    overlapped.append(key)
                busy.add(key)
                threads.setdefault(key, set()).add(threading.get_ident())
            try:
                return original(decoder, when)  # type: ignore[arg-type]
            finally:
                with guard:
                    busy.discard(key)

        monkeypatch.setattr(VideoDecoder, "frame_at", watched)
        drawn = _Screen(project, screen_context)
        background = _start(project, screen_context, 30)
        try:
            # 画面の側も同じ素材を同時に描く 貯まっていないコマをその場で描くのと同じ
            for frame in range(30):
                drawn.direct(frame)
            _wait_until(lambda: len(background.cached) == 30)
        finally:
            background.close()
            drawn.close()
        assert overlapped == [], "同じデコーダを 2 つのスレッドが同時に進めた"
        assert threads, "デコーダを 1 度も通っていない（見張りが外れている）"
        assert all(len(ids) == 1 for ids in threads.values()), "1 つのデコーダを 2 スレッドが触った"
        assert len({next(iter(ids)) for ids in threads.values()}) == 2, (
            "画面の側と走り係の両方が読んでいない"
        )


def _video_project(media_dir: Path, name: str) -> Project:
    sample = make_sample(media_dir, name, width=64, height=64, audio=False)
    media = probe_media(sample.path)
    track = Track(TrackKind.VIDEO, "V1")
    project = AddMedia(media).apply(Project.create(SETTINGS))
    project = AddTrack(track).apply(project)
    return AddClip(track.id, Clip(timeline_start=0, duration=40, media_id=media.id)).apply(project)


def _nested_video_project(media_dir: Path, name: str, *, at_frame: int) -> Project:
    """素材を入れたシーンを ``at_frame`` から置いたプロジェクト

    外と中の時刻をずらす そろっていると、外のフレームのまま中の素材を頼む
    誤りでも、たまたま同じ時刻になって気付けない
    """
    sample = make_sample(media_dir, name, width=64, height=64, audio=False)
    media = probe_media(sample.path)
    project = AddMedia(media).apply(Project.create(SETTINGS))
    scene = new_scene(project, "中")
    project = AddScene(scene).apply(project)
    track = Track(TrackKind.VIDEO, "S1")
    project = InScene(scene.id, AddTrack(track)).apply(project)
    project = InScene(
        scene.id, AddClip(track.id, Clip(timeline_start=0, duration=40, media_id=media.id))
    ).apply(project)
    for command in insert_scene(project, scene.id, at_frame=at_frame, duration=40):
        project = command.apply(project)
    return project


class TestPriming:
    """再生を始めたときに、貯めた所の端を画面の側のデコーダで裏から読ませる"""

    def test_a_nested_scene_is_primed_at_its_own_time(
        self,
        screen_context: OffscreenGLContext,
        media_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """入れ子のシーンの中の素材も、シーンの中の時刻で裏から読ませる

        一番外のトラックだけを見ると、シーンの中の素材は端を越えた所で画面の
        スレッドが鍵フレームから読み直し、再生がそこで引っかかる 中の時刻を
        外の時刻のまま頼んでも、描くときに時刻が合わずに読み直しになる
        """
        project = _nested_video_project(media_dir, "prime-nested.mp4", at_frame=10)
        calls: list[threading.Thread] = []
        original = VideoDecoder.frame_at

        def recorded(decoder: VideoDecoder, when: object) -> object:
            calls.append(threading.current_thread())
            return original(decoder, when)  # type: ignore[arg-type]

        renderer = FrameRenderer(project, context=screen_context, decode_threads=2)
        try:
            with screen_context:
                renderer.compose(10)
            monkeypatch.setattr(VideoDecoder, "frame_at", recorded)
            renderer.prime(35)
            with screen_context:
                renderer.compose(35)
        finally:
            renderer.close()
        assert calls, "シーンの中の素材を読んでいない"
        assert all(thread is not threading.main_thread() for thread in calls), (
            "シーンの中の素材を、描くときに画面のスレッドで読み直している"
        )

    def test_the_primed_frame_is_decoded_off_the_screen_thread(
        self,
        screen_context: OffscreenGLContext,
        media_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """頼んだコマは裏で読まれ、描くときは受け取るだけ

        画面のスレッドで読み直すなら、頼んだ意味が無い（端で引っかかる所がそのまま残る）
        """
        project = _video_project(media_dir, "prime.mp4")
        calls: list[tuple[threading.Thread, object]] = []
        original = VideoDecoder.frame_at

        def recorded(decoder: VideoDecoder, when: object) -> object:
            calls.append((threading.current_thread(), when))
            return original(decoder, when)  # type: ignore[arg-type]

        renderer = FrameRenderer(project, context=screen_context, decode_threads=2)
        try:
            with screen_context:
                renderer.compose(0)
            monkeypatch.setattr(VideoDecoder, "frame_at", recorded)
            renderer.prime(30)
            with screen_context:
                renderer.compose(30)
        finally:
            renderer.close()
        assert calls, "頼んだコマを読んでいない"
        assert all(thread is not threading.main_thread() for thread, _ in calls), (
            "頼んだコマを、描くときに画面のスレッドで読み直している"
        )

    def test_nothing_runs_when_parallel_decoding_is_off(
        self,
        screen_context: OffscreenGLContext,
        media_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """並列デコードを 1 本にした人には、描く所のほかでデコードを走らせない"""
        project = _video_project(media_dir, "prime-off.mp4")
        calls: list[object] = []
        monkeypatch.setattr(VideoDecoder, "frame_at", lambda decoder, when: calls.append(when))
        renderer = FrameRenderer(project, context=screen_context, decode_threads=1)
        try:
            renderer.prime(30)
        finally:
            renderer.close()
        assert calls == []


class TestLifecycle:
    def test_closing_stops_the_thread_and_frees_the_frames(
        self, screen_context: OffscreenGLContext
    ) -> None:
        """閉じたらスレッドが終わり、貯めた絵の GPU のメモリも返る

        返さないと、プロジェクトを開き直すたびに 1GB ずつ掴んだまま増える
        """
        background = _start(_gray_project(), screen_context, 4)
        _wait_until(lambda: len(background.cached) == 4)
        with background._shelf.lock:
            textures = [surface.color for surface in background._shelf.frames.values()]
        background.close()
        assert not background.running
        assert background.cached == frozenset()
        with screen_context:
            assert not any(GL.glIsTexture(texture) for texture in textures), (
                "閉じた後もテクスチャが残っている"
            )
        # 2 度閉じても落ちない 窓を閉じる道と設定を切る道の両方から呼ばれる
        background.close()

    def test_closing_waits_for_the_frame_in_progress(
        self, screen_context: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """描いている最中に閉じても、描き終えてから走り係のスレッドで片付ける

        待たずに戻ると、画面の側がコンテキストを捨てた後に走り係が GL を触る
        レンダラを画面のスレッドで閉じると、走り係のデコーダを別のスレッドから触る
        """
        started = threading.Event()
        closed_on: list[threading.Thread] = []
        original_compose = FrameRenderer.compose
        original_close = FrameRenderer.close

        def slow(renderer: FrameRenderer, frame: int, **options: Any) -> None:
            started.set()
            time.sleep(0.2)
            original_compose(renderer, frame)

        def recorded(renderer: FrameRenderer) -> None:
            closed_on.append(threading.current_thread())
            original_close(renderer)

        monkeypatch.setattr(FrameRenderer, "compose", slow)
        monkeypatch.setattr(FrameRenderer, "close", recorded)
        background = _start(_gray_project(), screen_context, 4)
        assert started.wait(10)
        began = time.perf_counter()
        background.close()
        assert time.perf_counter() - began > 0.05, "描いている最中の走り係を待たずに戻った"
        assert not background.running
        assert len(closed_on) == 1
        assert closed_on[0] is not threading.main_thread(), "レンダラを画面のスレッドで閉じた"


class TestWhenItRuns:
    def test_pausing_stops_new_frames(self, screen_context: OffscreenGLContext) -> None:
        """再生中は止める 出す側と同じ GPU を奪い合うと、いま出すコマが遅れる"""
        background = _start(_gray_project(), screen_context, 4)
        try:
            _wait_until(lambda: background.cached == frozenset(range(4)))
            background.set_paused(True)
            background.set_playhead(20)
            time.sleep(0.3)
            QCoreApplication.processEvents()
            assert background.cached == frozenset(range(4)), "止めたのに描き続けている"
            background.set_paused(False)
            _wait_until(lambda: background.cached == frozenset(range(20, 24)))
        finally:
            background.close()

    def test_a_smaller_budget_gives_frames_back(self, screen_context: OffscreenGLContext) -> None:
        # 減らしたのに掴んだままだと、ほかの作業のために減らした意味が無い
        background = _start(_gray_project(), screen_context, 6)
        try:
            _wait_until(lambda: len(background.cached) == 6)
            background.set_budget(FRAME_BYTES * 2, 0)
            _wait_until(lambda: background.cached == frozenset({0, 1}))
        finally:
            background.close()

    def test_proxies_dropped_before_a_failure_still_reach_the_screen(
        self, screen_context: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """描けなかったコマの途中で捨てた控えも、画面の側へ渡す（#127 のレビュー）

        渡さないと、壊れた控えを作り直す頼みが出ず、画面の側は同じ控えを使い続ける
        """
        dropped = MediaId("broken-proxy")

        def failing(renderer: FrameRenderer, frame: int, **options: Any) -> None:
            if threading.current_thread() is threading.main_thread():
                return
            renderer._discarded.add(dropped)
            raise RuntimeError("控えが壊れていた")

        monkeypatch.setattr(FrameRenderer, "compose", failing)
        background = _start(_gray_project(), screen_context, 2)
        reasons: list[str] = []
        background.failed.connect(lambda _, reason: reasons.append(reason))
        try:
            _wait_until(lambda: bool(reasons))
            assert background.take_discarded() == {dropped}
        finally:
            background.close()

    def test_a_failure_is_reported_and_tried_again_after_a_change(
        self, screen_context: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """描けなかったら伝えて止まり、次に何か変わったらまた試す

        黙って止まると先読みが効かない理由が分からない 失敗したまま回り続けると、
        同じ所で失敗し続けて CPU を食う
        """
        original = FrameRenderer.compose
        failures = [RuntimeError("デコーダが開けない")]

        def once(renderer: FrameRenderer, frame: int, **options: Any) -> None:
            if failures and threading.current_thread() is not threading.main_thread():
                raise failures.pop()
            original(renderer, frame, **options)

        monkeypatch.setattr(FrameRenderer, "compose", once)
        background = _start(_gray_project(), screen_context, 2)
        reasons: list[str] = []
        background.failed.connect(lambda _, reason: reasons.append(reason))
        try:
            _wait_until(lambda: bool(reasons))
            assert "デコーダが開けない" in reasons[0]
            time.sleep(0.1)
            assert background.cached == frozenset(), "失敗した後も描き続けている"
            background.set_playhead(0)
            _wait_until(lambda: background.cached == frozenset({0, 1}))
        finally:
            background.close()
